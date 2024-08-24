"""
""" 
import os
import sys

import numpy as np
import bmesh
import bpy

def create_bbox(act_obj):
    mod = act_obj.modifiers.new("Geo Node Modifier", type='NODES')
    node_group = bpy.data.node_groups.new("group name", type='GeometryNodeTree')
    node_group.interface.new_socket(name="Geo In", in_out ="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket(name="Geo Out", in_out ="OUTPUT", socket_type="NodeSocketGeometry")

    node_in     = node_group.nodes.new("NodeGroupInput")
    node_bbox   = node_group.nodes.new("GeometryNodeBoundBox")
    node_out    = node_group.nodes.new("NodeGroupOutput")

    node_group.links.new(node_in.outputs[0], node_bbox.inputs[0])
    node_group.links.new(node_bbox.outputs[0], node_out.inputs[0])

    mod.node_group = node_group
    bpy.ops.object.modifier_apply(modifier=mod.name)

def cut_with_bbox(ps_obj, bbox_obj, perc_cut):
    # Convert the bbox mesh to bmesh object
    bbox_bm = bmesh.new()
    bbox_bm.from_mesh(bbox_obj.data.id_data)
    
    # Loop the bbox vertices to get the global Y vertices,
    # that in object coordinates are in the Z axis.
    y_values = []
    for v in bbox_bm.verts:
        y_values.append(v.co.z)

    y_max = max(y_values)
    y_min = min(y_values)

    total_length = abs(y_min-y_max)

    new_length = total_length * perc_cut
    new_y_max = y_min + new_length

    for v in bbox_bm.verts:
        if v.co.z > y_min:
            v.co.z = new_y_max
        else:
            v.co.z *= 1.5

        v.co.x *= 1.5
        v.co.y *= 1.5

    bbox_bm.to_mesh(bbox_obj.data.id_data)
    bbox_bm.free()

    mod_boolean = ps_obj.modifiers.new(name="Boolean", type="BOOLEAN")
    mod_boolean.operation = "DIFFERENCE"
    mod_boolean.solver = "EXACT"
    mod_boolean.object = bbox_obj
    bpy.ops.object.modifier_apply(modifier=mod_boolean.name)

    mod_trig = ps_obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod_trig.name)

def calculate_non_manifold(in_ps_obj):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.object.mode_set(mode='OBJECT')

    sel = np.zeros(len(in_ps_obj.data.vertices), dtype=bool)
    in_ps_obj.data.vertices.foreach_get('select', sel)

    num_non_m = len(np.where(sel is True)[0])

    return num_non_m

# Parse the arguments
args = sys.argv[sys.argv.index("--")+1:]
obj_path = args[0]
save_path = args[1]
tgt_trig = int(args[2])
cut_perc = float(args[3])
# remesh_octdepth = int(args[5])
# remesh_type = args[6]

# Delete default scene objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False, confirm=False)
bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

# Creating the base bounding box
# - Load the mesh
bpy.ops.wm.obj_import(filepath=obj_path)

# - Select the object
bbox_obj = bpy.data.objects[0]
bpy.context.view_layer.objects.active = bbox_obj
bbox_mesh = bbox_obj.data.id_data

bbox_obj.name = "bbox_obj__base"
bbox_mesh.name = "bbox_mesh__base"

# - From the mesh, the bounding box is created.
create_bbox(bbox_obj)

# Start applying the boolean operator to the mesh.
is_manifold = False
while not is_manifold:

    # Now, make a functional bounding box copy to iterate with.
    bbox_obj_act = bbox_obj.copy()
    bbox_obj_act.data = bbox_obj.data.copy()

    bbox_obj_act.name = "bbox_obj_act"
    bbox_obj_act.data.id_data.name = "bbox_mesh_act"
    bpy.context.collection.objects.link(bbox_obj_act)

    # Load the mesh again, now its the mesh to cut.
    bpy.ops.wm.obj_import(filepath=obj_path)

    ps_obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = ps_obj
    ps_mesh = ps_obj.data.id_data

    print("Cutting...")
    cut_with_bbox(ps_obj, bbox_obj_act, cut_perc)
    number_non_manifold = calculate_non_manifold(ps_obj)
    print(f"Non-Manifold vertices are {number_non_manifold}.")

    if number_non_manifold == 0:
        is_manifold = True
    else:
        # The mesh still has a hole in it somewhere, make the cutting bbox bigger.
        cut_perc += 0.01
        # Delete the PS mesh and reload it.
        bpy.data.objects.remove(ps_obj, do_unlink = True)
        print(f"Trying with {cut_perc * 100:.2f}% of BBOX.")

    bpy.data.objects.remove(bbox_obj_act, do_unlink = True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

print("Cutting complete.")

# Scale the mesh to the wanted trigs
act_trigs = len(ps_mesh.polygons)

while act_trigs != tgt_trig:
    # Edge collapse it
    ratio = tgt_trig / act_trigs
    mod_collapse = ps_obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod_collapse.use_collapse_triangulate = True
    mod_collapse.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=mod_collapse.name)

    act_trigs = len(ps_mesh.polygons)

    # Subdivide the object if needed
    if(act_trigs != tgt_trig):
        print("Mesh has {} polygons, {} are needed.".format(act_trigs, tgt_trig))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')

        act_trigs = len(ps_mesh.polygons)
        print("After Subsurf, mesh now has {} polys".format(act_trigs))

# Reexport the OBJ file.
mesh_name = os.path.split(obj_path)[1]

bpy.ops.wm.obj_export(
    filepath=os.path.abspath(os.path.join(save_path, mesh_name)),
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)

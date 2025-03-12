import os
import sys
import time

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
    mod_boolean.use_self = True
    mod_boolean.object = bbox_obj
    bpy.ops.object.modifier_apply(modifier=mod_boolean.name)

    mod_trig = ps_obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod_trig.name)

def calculate_non_manifold(in_ps_obj):
    in_ps_obj.select_set(True)
    bpy.context.view_layer.objects.active = in_ps_obj
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    num_non_m = bpy.context.active_object.data.total_vert_sel
    bpy.ops.object.mode_set(mode='OBJECT')

    return num_non_m

init_time = time.time()
# Parse the arguments
args = sys.argv[sys.argv.index("--")+1:]
obj_path = args[0]
save_path = args[1]
cut_perc = float(args[2])

mesh_name = os.path.split(obj_path)[1].split(".")[0]
# tgt_trig = int(args[2])
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

## Apply boolean operator
# - Load the mesh again, now it's the mesh to cut.
bpy.ops.wm.obj_import(filepath=obj_path)

ps_obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = ps_obj
ps_mesh = ps_obj.data.id_data

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.remove_doubles()
bpy.ops.object.mode_set(mode='OBJECT')

print(f"[{mesh_name}] Cutting...")
mesh_trigs_before = len(ps_obj.data.vertices)
cut_with_bbox(ps_obj, bbox_obj, cut_perc)

mesh_trigs = len(ps_obj.data.vertices)
proportion_trigs = mesh_trigs / mesh_trigs_before
print(f"[{mesh_name}] {mesh_trigs} Triangles in mesh, {(proportion_trigs * 100):.4f}% kept.")

if proportion_trigs * 100 < 10:
    err_msg = f"[{mesh_name}] Error with boolean, check model."
    print(err_msg)
    with open(os.path.join(save_path, mesh_name.split(".")[0] + "_err.txt"), "wt", encoding="utf8") as f:
        f.write(err_msg)
    exit(-1)

bpy.data.objects.remove(bbox_obj, do_unlink = True)
bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
print(f"[{mesh_name}] Cutting complete.")

mesh_name_path = os.path.split(obj_path)[1]

bpy.ops.wm.obj_export(
    filepath=os.path.abspath(os.path.join(save_path, mesh_name_path)),
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)
end_time = time.time()
print(f"[{mesh_name}] Done, took {end_time-init_time:.4f} s")
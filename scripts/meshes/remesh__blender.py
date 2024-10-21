"""
""" 
import os
import sys
import time

import bpy

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

mesh_name = os.path.split(obj_path)[1]
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
act_obj = bpy.data.objects[0]
bpy.context.view_layer.objects.active = act_obj

# Remesh
print(f"[{mesh_name.split('.')[0]}] Remeshing...")
mod_remesh = act_obj.modifiers.new(name="Remesh", type="REMESH")
mod_remesh.mode = "SHARP"
mod_remesh.octree_depth = 10
mod_remesh.scale = 1
bpy.ops.object.modifier_apply(modifier=mod_remesh.name)

mod_trig = act_obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
bpy.ops.object.modifier_apply(modifier=mod_trig.name)

# Fix the non-manifold geometry again.
is_manifold = (calculate_non_manifold(act_obj) == 0)
i = 0
print(f"[{mesh_name.split('.')[0]}] Remesh done.\nFixing generated non-manifold geometry...")
while not is_manifold:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.select_all(action='DESELECT')

    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.merge(type="COLLAPSE")
    bpy.ops.mesh.select_all(action='DESELECT')

    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.delete()

    bpy.ops.object.mode_set(mode='OBJECT')
    num_nonm = calculate_non_manifold(act_obj)
    print(f"[{mesh_name.split('.')[0]}] {i}: {num_nonm}")
    if num_nonm == 0:
        is_manifold = True

    i+=1

    if i > 200:
        print(f"[{mesh_name.split('.')[0]}] Remeshing could not be fixed, check mesh.")
        end_time = time.time()

        with open(os.path.join(save_path, mesh_name.split(".")[0] + "_err.txt"), "wt", encoding="utf8") as f:
            f.write("")

        with open(os.path.join(save_path, mesh_name.split(".")[0] + "_time.txt"), "wt", encoding="utf8") as f:
            f.write(f"{mesh_name.split('.')[0]},{end_time-init_time}\n")
        exit(-1)


bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.remove_doubles()
bpy.ops.object.mode_set(mode='OBJECT')

print(f"[{mesh_name.split('.')[0]}] Geometry fixed.")
# # Scale the mesh to the wanted trigs
# act_trigs = len(ps_mesh.polygons)

# while act_trigs != tgt_trig:
#     # Edge collapse it
#     print("Decimating...")
#     ratio = tgt_trig / act_trigs
#     mod_collapse = ps_obj.modifiers.new(name="Decimate", type="DECIMATE")
#     mod_collapse.use_collapse_triangulate = True
#     mod_collapse.ratio = ratio
#     bpy.ops.object.modifier_apply(modifier=mod_collapse.name)

#     act_trigs = len(ps_mesh.polygons)

#     # Subdivide the object if needed
#     if(act_trigs != tgt_trig):
#         print("Mesh has {} triangles, but {} are needed...".format(act_trigs, tgt_trig))
#         bpy.ops.object.mode_set(mode='EDIT')
#         bpy.ops.mesh.select_all(action='SELECT')
#         bpy.ops.mesh.subdivide()
#         bpy.ops.object.mode_set(mode='OBJECT')

#         act_trigs = len(ps_mesh.polygons)
#         print("After subdivision, mesh now has {} triangles...".format(act_trigs))

# print("Decimation complete.")

# Center on origin
# bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")

# Reexport the OBJ file.
mesh_name = os.path.split(obj_path)[1]

bpy.ops.wm.obj_export(
    filepath=os.path.abspath(os.path.join(save_path, mesh_name)),
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)
end_time = time.time()
print(f"[{mesh_name.split('.')[0]}] Done, took {end_time-init_time:.4f} s")

with open(os.path.join(save_path, mesh_name.split(".")[0] + "_time.txt"), "wt", encoding="utf8") as f:
    f.write(f"{mesh_name.split('.')[0]},{end_time-init_time}\n")
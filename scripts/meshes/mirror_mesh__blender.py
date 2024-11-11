 
"""
""" 
import os
import sys
import time

import bpy

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

# Mirror the mesh
print(f"[{mesh_name.split('.')[0]}] Mirroring...")
bpy.ops.transform.mirror(orient_type='LOCAL', orient_matrix=((1, 0, 0), (0, 0, 1), (0, -1, 0)), orient_matrix_type='LOCAL', constraint_axis=(True, False, False))

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
mesh_name_new = os.path.split(obj_path)[1].split('.')[0]
mesh_name_new += "m.obj"

bpy.ops.wm.obj_export(
    filepath=os.path.abspath(os.path.join(save_path, mesh_name_new)),
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)
end_time = time.time()
print(f"[{mesh_name.split('.')[0]}] Done, took {end_time-init_time:.4f} s")
import bpy
import sys
import os

# Get the parameters.

# Parse the arguments
args = sys.argv[sys.argv.index("--")+1:]
obj_path = args[0]
save_path = args[1]
tgt_trig = int(args[2])

mesh_name_str = os.path.split(obj_path)[1].split(".")[0]

# Delete default scene objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False, confirm=False)

bpy.ops.wm.obj_import(filepath=obj_path)

# Select the object.
# - Select the object
act_obj = bpy.data.objects[0]
bpy.context.view_layer.objects.active = act_obj
act_obj.select_set(True)

# Scale the mesh to the wanted trigs
act_trigs = len(act_obj.data.polygons)

while act_trigs != tgt_trig:
    # Edge collapse it
    print(f"[{mesh_name_str}] Decimating...")
    ratio = tgt_trig / act_trigs
    mod_collapse = act_obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod_collapse.use_collapse_triangulate = True
    mod_collapse.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=mod_collapse.name)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.object.mode_set(mode='OBJECT')

    act_trigs = len(act_obj.data.polygons)

    # Subdivide the object if needed
    if(act_trigs != tgt_trig):
        print(f"[{mesh_name_str}] Mesh has act_trigs triangles, but {tgt_trig} are needed...")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')

        act_trigs = len(act_obj.data.polygons)
        print(f"[{mesh_name_str}] After subdivision, mesh now has {act_trigs} triangles...")

print(f"[{mesh_name_str}] Decimation complete.")

# Center on origin
bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")

# Reexport the OBJ file.
mesh_name = os.path.split(obj_path)[1]

bpy.ops.wm.obj_export(
    filepath=os.path.abspath(os.path.join(save_path, mesh_name)),
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)
print(f"[{mesh_name_str}] Done.")
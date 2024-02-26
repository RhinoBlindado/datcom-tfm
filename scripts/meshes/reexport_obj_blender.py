import bpy
import sys
import os

# Get the parameters.

# Parse the arguments
args = sys.argv[sys.argv.index("--")+1:]
obj_path = args[0]

# Delete default scene objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False, confirm=False)

bpy.ops.wm.obj_import(filepath=obj_path)

# Select the object.
mesh = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = mesh

# Reexport the OBJ file.
bpy.ops.wm.obj_export(
    filepath=obj_path,
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)
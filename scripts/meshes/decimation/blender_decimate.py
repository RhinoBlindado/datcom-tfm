import bpy
import sys
import os

args = sys.argv[sys.argv.index("--")+1:]

file = args[0]
ratio = float(args[1])
save_path = args[2]

# Delete default scene objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False, confirm=False)

# Load up scene.
bpy.ops.wm.obj_import(filepath=file)

# Select the object.
mesh = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = mesh

# Create the modifier
mod = mesh.modifiers.new(name="Decimate", type="DECIMATE")

# Set some configurations
bpy.context.object.modifiers["Decimate"].use_collapse_triangulate = True
mod.ratio = ratio

# Apply the collapse
bpy.ops.object.modifier_apply(modifier=mod.name)

# Export the OBJ
basepath, obj_name = os.path.split(file)
obj_name = obj_name.split(".")[0]

bpy.ops.wm.obj_export(
    filepath="{}/{}_{}.obj".format(os.path.abspath(save_path), obj_name, str(ratio).split(".")[-1]),
    export_triangulated_mesh=True,
    export_selected_objects=True,
    export_materials=False,
    export_normals=False,
    export_uv=False
)
 

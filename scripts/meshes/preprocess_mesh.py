"""

"""

import subprocess
import argparse
import glob
import os

BLEND_CMD = ["/home/valentino/Apps/blender-4.0.2-linux-x64/blender",
            "--background",
            "--python",
            "/media/valentino/Irithyll1/repos/datcom-tfm/scripts/meshes/preprocess_mesh__blender.py",
            "--"]

parser = argparse.ArgumentParser()

# Basic parameters
parser.add_argument("--input", type=str)
parser.add_argument("--out", type=str)
parser.add_argument("--trig-n", type=int)

# Cutting the mesh parameters
parser.add_argument("--cut", action="store_true")
parser.add_argument("--cut-perc", type=float, default=0.4)

args = parser.parse_args()

# Detect if "in" is file or folder.
if os.path.isdir(args.input):
    meshes = glob.glob(os.path.join(args.input, "*.obj"))
else:
    meshes = [args.input]

# Make the output folder
os.makedirs(args.out, exist_ok=True)

for i, m in enumerate(meshes):
    print(f"[{i+1:03d}/{len(meshes):03d}] {os.path.split(m)[1]}")
    subprocess.run(BLEND_CMD + [m, args.out, str(args.trig_n), str(args.cut_perc)], check=False)

"""

"""

import multiprocessing as mp
import subprocess
import argparse
import glob
import os

BLEND_CMD = ["/home/valentino/Apps/blender-4.0.2-linux-x64/blender",
            "--background",
            "--python",
            "/media/valentino/Irithyll1/repos/datcom-tfm/scripts/meshes/remesh__blender.py",
            "--"]

def blender_call(mesh, out):

    if os.path.exists(os.path.join(out, os.path.split(mesh)[1])):
        print("Mesh already exists.")
    else:
        subprocess.run(BLEND_CMD + [mesh, out], check=False)


parser = argparse.ArgumentParser()

# Basic parameters
parser.add_argument("--input", type=str)
parser.add_argument("--out", type=str)
parser.add_argument("--cpus", default=4, type=int, help="Number of processes to use, default is 4.")

# parser.add_argument("--trig-n", type=int)

args = parser.parse_args()

# Detect if "in" is file or folder.
if os.path.isdir(args.input):
    meshes = sorted(glob.glob(os.path.join(args.input, "*.obj")))
else:
    meshes = [args.input]

# Make the output folder
os.makedirs(args.out, exist_ok=True)

if args.cpus > 1:
    args_pool=[(m, args.out) for m in meshes]

    pool = mp.Pool(processes=args.cpus)
    pool.starmap(blender_call, args_pool, chunksize=None)
    pool.close()
    pool.join()
else:
    for i, m in enumerate(meshes):
        print(f"[{i+1:03d}/{len(meshes):03d}] {os.path.split(m)[1]}")
        
        if os.path.exists(os.path.join(args.out, os.path.split(m)[1])):
            print("Mesh already exists.")
            continue

        subprocess.run(BLEND_CMD + [m, args.out], check=False)

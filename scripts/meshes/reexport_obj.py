import glob
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--root-path")

args = parser.parse_args()

f_originals = glob.glob("{}/**/*.obj".format(args.root_path),recursive=True)

cmd = ["/home/valentino/Apps/blender-3.6.1-linux-x64/blender", 
       "--background", 
       "--python",
       "/media/valentino/Irithyll/repos/ExMeshCNN/reexport_obj_blender.py",
       "--"]

for i in f_originals:
    print("Reprocessing {}".format(i))
    subprocess.run(cmd + [i])
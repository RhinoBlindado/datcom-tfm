import argparse
import pymeshlab
import subprocess
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Get the path of the python script.
pypath = os.path.split(__file__)[0]

BLENDER_PATH = "/home/valentino/Apps/blender-4.0.2-linux-x64/blender"

parser = argparse.ArgumentParser()

parser.add_argument("--mesh", required=True, type=str)
parser.add_argument("--splits", default=10, type=int)

args = parser.parse_args()

scale_step = -np.sort(-np.linspace(0, 1, args.splits + 1)[1:-1])

ms = pymeshlab.MeshSet()
ms.load_new_mesh(args.mesh)

mesh_path, mesh_fname = os.path.split(args.mesh)
mesh_fname = mesh_fname.split(".")[0]
sample_count = ms.mesh(0).vertex_number()

redux_path = os.path.join(mesh_path, "{}_reductions".format(mesh_fname))
os.makedirs(redux_path)

haus_res = {"reduction" : [], "abs_mean" : [], "abs_max" : [], "rel_mean" : [], "rel_max": [], "bbox": []}

pbar = tqdm(initial=0, total=len(scale_step), unit=" it")

for i, scale in enumerate(scale_step):
    cmd_call = [BLENDER_PATH, 
                "--python", 
                os.path.join(pypath, "blender_decimate.py"),
                "--background",
                "--",
                args.mesh,
                str(scale),
                redux_path]
    
    _ = subprocess.run(cmd_call, capture_output=True)
    ms.load_new_mesh(os.path.join(redux_path, "{}_{}.obj".format(mesh_fname, str(scale).split(".")[-1])))

    act_hausdorff = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=i+1, samplenum = sample_count)
    haus_res["reduction"].append(scale)
    haus_res["abs_mean"].append(act_hausdorff["mean"])
    haus_res["abs_max"].append(act_hausdorff["max"])
    haus_res["rel_mean"].append((act_hausdorff["mean"] / act_hausdorff["diag_mesh_0"]) * 100)
    haus_res["rel_max"].append((act_hausdorff["max"] / act_hausdorff["diag_mesh_0"]) * 100)
    haus_res["bbox"].append(act_hausdorff["diag_mesh_0"])
    ms.delete_current_mesh()
    pbar.update(1)

pbar.close()

haus_res_df = pd.DataFrame(data=haus_res)

haus_res_df.to_csv(os.path.join(mesh_path, "{}_results.csv".format(mesh_fname)), index=None)
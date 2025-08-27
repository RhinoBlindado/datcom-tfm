"""

"""

import os
import argparse
import glob
import shutil
import pathlib
import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--root-path", type=str, required=True, help="")
parser.add_argument("--csv-path", type=str, required=True, default="./data/unprocessed/PS_LR_493-tags_Expert.csv", help="")
parser.add_argument("--output-dir", type=str, required=True, help="")

args = parser.parse_args()

# Get valid ID list from CSV
pd_tags = pd.read_csv(args.csv_path)
id_tags = sorted(list(pd_tags["individuo"].unique()))

# Set up output folder
output_dir_path = args.output_dir
os.makedirs(output_dir_path)

# Get file list of them bones
ps_whole_list = glob.glob(os.path.join(args.root_path, "*", "*", "*.obj"))

pbar = tqdm(total = len(id_tags) * 2, unit="obj")

for bone in ps_whole_list:

    bone_path_split = bone.split("/")
    bone_laterality_original = bone_path_split[-2][0].lower()
    bone_id = int(bone_path_split[-3])

    if bone_id in id_tags:
        if bone_laterality_original == "i":
            bone_laterality = "L"
        else:
            bone_laterality = "R"

        shutil.copy(bone, os.path.join(output_dir_path, f"{bone_id}-{bone_laterality}.obj"))
        pbar.update(1)

pbar.close()
    
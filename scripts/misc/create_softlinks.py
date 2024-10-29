import argparse
import glob
import os

import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--dst-folder", type=str)
parser.add_argument("--src-folder", type=str)
parser.add_argument("--pattern", choices=["L", "Rm", "R"])
parser.add_argument("--force", action="store_true")

args = parser.parse_args()

pattern_files = glob.glob(os.path.join(args.src_folder, f"*-{args.pattern}.npy"))

for f_path in pattern_files:
    file = os.path.split(f_path)[1]

    if os.path.islink(os.path.join(args.dst_folder, file)) and args.force:
        os.remove(os.path.join(args.dst_folder, file))

    os.symlink(os.path.abspath(f_path), os.path.join(args.dst_folder, file))
    print(f"Linking {f_path} -> {os.path.join(args.dst_folder, file)}")
import argparse
import torch
import pandas as pd
import exmeshcnn

parser = argparse.ArgumentParser()

# Base arguments
parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to a dataset folder. Must follow required format, see README.")
parser.add_argument("-m", "--model", type=str, default="exmeshcnn-base", help="CNN model to use. Default is 'exmeshcnn-base'")
parser.add_argument("-o", "--output", type=str, default="./experiments", help="Output folder where to save experiment data. Default is 'experiments' folder.")
parser.add_argument("-n", "--exp-name", type=str, default=None, help="Name of the experiment. Default is <YYYY>_<MM>_<DD>__<model>__<dataset>")

# Training specific arguments.
# ... We'll see what we need here later ...

args = parser.parse_args()


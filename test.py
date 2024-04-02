import argparse
import torch
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to a dataset folder. Must follow required format, see README.")
parser.add_argument("-t", "--task", choices=["train", "test", "all"], default="train", help="What task to perform. Select between only training, only testing or both.")
parser.add_argument("-m", "--model", type=str, )
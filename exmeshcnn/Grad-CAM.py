
import os
import sys
import argparse
import datetime
import importlib

import yaml
import torch
import rootutils

from tqdm import tqdm
from matplotlib import colors
from matplotlib import cm as cmx
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import pandas as pd
import openmesh as om

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from exmeshcnn.datasetloader import PSDataset

def save_metadata(split, out_folder, tag_str, tag_data):
    
    # Save predictions
    pred_df = pd.DataFrame(data={"x": tag_data[tag_str][split]["x"],
                                 "y": tag_data[tag_str][split]["y"],
                                 "y_pred": tag_data[tag_str][split]["y_pred"]})
    
    pred_df.sort_values(by=["x"], inplace=True)
    pred_df.to_csv(os.path.join(out_folder, f"{tag_str}-preds.csv"), index=None)

    # Save classification report
    with open(os.path.join(out_folder, f"{tag_str}-results.yaml"), mode="wt", encoding="utf8") as f:
        yaml.dump(tag_data[tag_str][split]["class_report"], f)

    # Save confusion matrix
    np.save(os.path.join(out_folder, f"{tag_str}-cm.npy"), tag_data[tag_str][split]["cm"])


def grad_cam_inference(dataloader, split, tag_data, model, mesh_folder, out_folder):

    pbar = tqdm(initial=0, total=len(dataloader), unit=" objs")
    for k, batch in enumerate(dataloader):

        # Send data to target device
        x1 = batch[0].to(device)
        x2 = batch[1].to(device)
        x3 = batch[2].to(device)
        y_preds = model(x1, x2, x3)

        act_tag = list(batch[3].keys())[0]
        act_y = batch[3][act_tag]

        # Save the samples used in the step.
        tag_data[act_tag][split]["x"].extend(batch[4])

        # Get the logit for the current tag.
        act_y_pred_logit = y_preds[act_tag]
        act_y_pred = torch.argmax(act_y_pred_logit, dim=1).tolist()

        act_y_pred_logit[:, act_y_pred_logit.argmax()].backward()

        # Save the actual loss, prediction and correct tags
        tag_data[act_tag][split]["y"].extend(act_y.tolist())
        tag_data[act_tag][split]["y_pred"].extend(act_y_pred)
            
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2])
        activations = model.get_activations()

        for i in range(activations.shape[1]):
            activations[:,i,:] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap -= heatmap.mean()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.detach().cpu()

        ccmp = plt.get_cmap('Reds')
        cNorm  = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=ccmp)

        mesh_name = f"{batch[4][0]}.obj"
        mesh = om.read_trimesh(os.path.join(mesh_folder, mesh_name))

        for i,face in enumerate(mesh.faces()):
            mesh.set_color(face, scalarMap.to_rgba(heatmap[i]))

        act_mesh_folder = os.path.join(out_folder, f"{batch[4][0]}")

        if not os.path.isdir(act_mesh_folder):
            os.makedirs(act_mesh_folder)
            
        om.write_mesh(os.path.join(act_mesh_folder, f"{batch[4][0]}_{act_tag}.obj"), mesh, face_color=True)
        del mesh
        pbar.update(1)

    pbar.close()
                
    for _, values in tag_data.items():
        values[split]["class_report"] = classification_report(values[split]["y"], values[split]["y_pred"],  output_dict=True, zero_division=0)
        values[split]["cm"] = confusion_matrix(values[split]["y"], values[split]["y_pred"]).tolist()

parser = argparse.ArgumentParser()

# Data input
parser.add_argument("--dataset", type=str, help="Path to the Dataset")
parser.add_argument("--npy-name", type=str, help="Name of the NPY folder to be used. Must be inside the dataset folder.")
parser.add_argument("--obj-data", type=str, help="Path to the OBJ folder holding the original NPY data.")
parser.add_argument("--tag-yaml", type=str, default="./src/todd_characteristics.yaml", help="YAML file containing the number of categories per characteristic and their names. Default is located at '/src/todd_characteristics.yaml'")
parser.add_argument("-t", "--tags", type=str, required=True, nargs="+") # choices=["af", "ip", "use", "bn", "lse", "dm", "dp", "vb", "vm", "all"]

# Experiment selection
parser.add_argument("--exp-path", type=str, help="Path to the experiment path")
parser.add_argument("--trial", type=int, default=None, help="If using Optuna Trials, the number of the trial to use.")

# Model selection
parser.add_argument("--model", type=str, help="Type of model to use.")
parser.add_argument("--model-struct", type=str, default="params.yaml")
parser.add_argument("--model-weights", type=str, default="best_mean_model_state_dict.pth")

# Data
parser.add_argument("--override-splits", type=str, default=None)
parser.add_argument("--workers", type=int, default=4)

# Output parameters
parser.add_argument("--output-path", type=str, default="./experiments")
parser.add_argument("--output-name", type=str, default=None, help="Name of the experiment. Default is grad-<YYYY>-<MM>_<DD>-<model>-<dataset>")

parser.add_argument("--debug-use-cpu", action="store_true")

args = parser.parse_args()

# Select the device to perform the computations.
torch.cuda.empty_cache()
if args.debug_use_cpu:
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device}")

# Create the output folder.
out_folder_path = args.output_path

if args.output_name is None:
    dataset_basename = os.path.split(args.dataset.strip("/"))[1]
    tags_str = "_".join(sorted(list(set(args.tags))))
    out_name = f"grad-{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}-{args.model}-{dataset_basename}-{args.npy_name}-{tags_str}"
else:
    out_name = args.output_name

out_folder = os.path.join(out_folder_path, out_name)
os.makedirs(out_folder)

# Prepare the multiclass data dictionary
# - Read the YAML to get the metadata of the tags: names, number of characteristics and their names.
with open(args.tag_yaml, "r", encoding="utf-8") as f:
    try:
        tag_metadata =  yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# - Will hold the tags selected for training.
selected_tags = []

# If 'all' is passed, use all the tags, duh!
if "all" in args.tags:
    selected_tags = list(tag_metadata.keys())
# If not, get the unique tags passed (there could be repeats, that's why a set is used.)
else:
    selected_tags = list(set(args.tags))


train_path = os.path.join(args.exp_path, "train.csv")
val_path = os.path.join(args.exp_path, "validation.csv")
test_path = os.path.join(args.exp_path, "test.csv")

if args.override_splits is not None:

    train_path = os.path.join(args.override_splits, "train.csv")
    val_path = os.path.join(args.override_splits, "validation.csv")
    test_path = os.path.join(args.override_splits, "test.csv")

act_training_set = pd.read_csv(train_path)
act_validation_set = pd.read_csv(val_path)
act_test_set = pd.read_csv(test_path)

y_train = pd.DataFrame(act_training_set, columns=selected_tags)
y_val = pd.DataFrame(act_validation_set, columns=selected_tags)
y_test = pd.DataFrame(act_test_set, columns=selected_tags)

X_train = act_training_set["name"].to_numpy()
X_val = act_validation_set["name"].to_numpy()
X_test = act_test_set["name"].to_numpy()

out_mesh_train = os.path.join(out_folder, "train")
os.makedirs(out_mesh_train)

out_mesh_val = os.path.join(out_folder, "val")
os.makedirs(out_mesh_val)

out_mesh_test = os.path.join(out_folder, "test")
os.makedirs(out_mesh_test)

model_struct_path = args.model_struct

if args.trial is not None:
    model_struct_path = os.path.join(args.exp_path, "trials", f"trial_{args.trial}", args.model_struct)

model_weights_path = args.model_weights

if args.trial is not None:
    model_weights_path = os.path.join(args.exp_path, "trials", f"trial_{args.trial}", args.model_weights)

# Load the given model.
try:
    model_module = importlib.import_module(f"models.{args.model}")
    with open(model_struct_path, "r", encoding="utf-8") as f:
        try:
            model_struct =  yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
except Exception:
    print(f"Model {args.model} not found.")
    sys.exit(-1)

# For each selected tag...
tag_data = {}
for i, char in enumerate(selected_tags):
    # Make a nested dictionary and,
    tag_data[char] = {}

    # Using char as the key, access the number of categories in current tag, ...
    tag_data[char]["classes"] = tag_metadata[char]["cat_num"]
    _ , tag_data[char]["samples_per_class"] = np.unique(y_train[char], return_counts=True)
    # ... make another nested dictionary to hold values relevant to the training, validation, and test ...
    tag_data[char]["train"] = {"x" : [], "y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}
    tag_data[char]["val"] = {"x" : [], "y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}
    tag_data[char]["test"] = {"x" : [], "y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}

    model = model_module.get_model(tag_data, params=model_struct, gradcam=True)
    model = model.to(device)

    model.load_state_dict(torch.load(model_weights_path), strict=False)
    model.eval()

    train_loader = torch.utils.data.DataLoader(PSDataset(X_train, y_train, args.dataset, tag_data, npy_name=args.npy_name), batch_size=1, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(PSDataset(X_val, y_val, args.dataset, tag_data, npy_name=args.npy_name), batch_size=1, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(PSDataset(X_test, y_test, args.dataset, tag_data, npy_name=args.npy_name), batch_size=1, shuffle=False, num_workers=args.workers)
    
    print(f"--------- CHAR: {char} ({i+1} / {len(selected_tags)}) ")

    # Get heatmaps of training meshes
    print("------- TRAINING SPLIT ")
    grad_cam_inference(train_loader, "train", tag_data, model, args.obj_data, out_mesh_train)
    save_metadata("train", out_mesh_train, char, tag_data)

    # Get heatmaps of validation meshes
    print("------- VALIDATION SPLIT")
    grad_cam_inference(val_loader, "val", tag_data, model, args.obj_data, out_mesh_val)
    save_metadata("val", out_mesh_val, char, tag_data)

    # Get heatmaps of test meshes
    print("------- TEST SPLIT")
    grad_cam_inference(test_loader, "test", tag_data, model, args.obj_data, out_mesh_test)
    save_metadata("test", out_mesh_test, char, tag_data)

    del model
    torch.cuda.empty_cache()
    tag_data.clear()

print("Done!")

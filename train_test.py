"""
    Main module for training / validating experiments, it can also be called as a module from
    another code.
"""

import os
import sys
import argparse
import importlib

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyrootutils
import torch
import tqdm
import yaml

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate


ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

from exmeshcnn.datasetloader import PSDataset

def model_step(mode : str,
               model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               tag_data : dict,
               optimizer: torch.optim.Optimizer = None,
               device: str = "cuda"):
    """
    a
    """
    # Put the model in the desired mode:
    if mode == "train":
        model.train()
    else:
        model.eval()

    batch_len = len(dataloader)
    # Loop through data loader data batches
    for _, batch in enumerate(dataloader):
        
        # Send data to target device
        x1 = batch[0].to(device)
        x2 = batch[1].to(device)
        x3 = batch[2].to(device)

        # 1. Forward pass
        y_preds = model(x1, x2, x3)

        # 2. Calculate and accumulate losses
        losses = []

        # For each tag...
        y = batch[3]
        for act_tag, act_y in y.items():
            act_y_pred_logit = y_preds[act_tag]

            if tag_data[act_tag]["classes"] == 2:
                act_y = act_y.unsqueeze(dim=1).float()
                act_y_pred = torch.round(torch.sigmoid(act_y_pred_logit)).squeeze(dim=1).tolist()
            else:
                act_y_pred = torch.argmax(act_y_pred_logit, dim=1).tolist()

            # ... calculate its own loss with its own loss function.
            act_loss = tag_data[act_tag]["loss_fn"](act_y_pred_logit,
                                                    act_y.to(device))

            # Save the actual loss, prediction and correct tags
            tag_data[act_tag][mode]["y"].extend(act_y.tolist())
            tag_data[act_tag][mode]["y_pred"].extend(act_y_pred)
            tag_data[act_tag][mode]["loss"] += act_loss.tolist() / batch_len
            losses.append(act_loss)
            
        # End of Y tags loop

        if mode == "train":
            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            total_loss = sum(losses)
            total_loss.backward()

            # 5. Optimizer step
            optimizer.step()

    for _, values in tag_data.items():
        values[mode]["class_report"] = classification_report(values[mode]["y"], values[mode]["y_pred"],  output_dict=True, zero_division=0)
        values[mode]["cm"] = confusion_matrix(values[mode]["y"], values[mode]["y_pred"])

def confusion_matrix_plot(cm, names, savepath, figsize = (8,8)):
    fig = plt.figure(figsize=figsize)

    sns.heatmap(cm, annot=True, xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("Real")

    plt.tight_layout()
    plt.savefig(savepath, format="pdf", bbox_inches="tight")

    plt.close(fig)

def progression_plot(epochs, metric_train, metric_validation, metric_name, savepath, figsize = (8, 5), ylim = None):

    fig = plt.figure(figsize=figsize)
    plt.plot(epochs, metric_train, 'o--', color='r', label="Training")
    plt.plot(epochs, metric_validation, 'o-', color='g', label="Validation")

    plt.legend()

    if ylim is not None:
        plt.ylim(ylim)
    
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.grid()
    plt.tight_layout()

    plt.savefig(savepath, format="pdf", bbox_inches="tight")

    plt.close(fig)

def training(model: torch.nn.Module, train_data, validation_data, tag_data : dict, optimizer, epochs, device, epoch_begin = 0, best = False):

    training_stats = {}
    validation_stats = {}

    for tag, _ in tag_data.items():
        training_stats[tag] = {"epoch" : [], "acc" : [], "f1" : [], "loss" : [], "class_report" : [], "cm": []}
        validation_stats[tag] = {"epoch" : [], "acc" : [], "f1" : [], "loss" : [], "class_report" : [], "cm": []}

    for epoch in range(epochs):

        model_step("train", model, train_data, tag_data, optimizer=optimizer, device=device)
        model_step("val", model, validation_data, tag_data, device=device)


        training_report_header = ["TAG", "TRAINING\nACC", "\nF1", "\nLOSS", "VALIDATION\nACC", "\nF1", "\nLOSS"]
        training_report_body   = []

        for tag, values in tag_data.items():

            curr_acc_train  =  values["train"]["class_report"]["accuracy"]
            curr_f1_train   =  values["train"]["class_report"]["macro avg"]["f1-score"]
            curr_loss_train =  values["train"]["loss"]

            curr_acc_val  =  values["val"]["class_report"]["accuracy"]
            curr_f1_val   =  values["val"]["class_report"]["macro avg"]["f1-score"]
            curr_loss_val =  values["val"]["loss"]

            training_report_body.append([tag.upper(), curr_acc_train, curr_f1_train, curr_loss_train, curr_acc_val, curr_f1_val, curr_loss_val])

            training_stats[tag]["epoch"].append(epoch)
            training_stats[tag]["acc"].append(curr_acc_train)
            training_stats[tag]["f1"].append(curr_f1_train)
            training_stats[tag]["loss"].append(curr_loss_train)
            training_stats[tag]["class_report"].append(values["train"]["class_report"])
            training_stats[tag]["cm"].append(values["train"]["cm"])

            validation_stats[tag]["epoch"].append(epoch)
            validation_stats[tag]["acc"].append(curr_acc_val)
            validation_stats[tag]["f1"].append(curr_f1_val)
            validation_stats[tag]["loss"].append(curr_loss_val)
            validation_stats[tag]["class_report"].append(values["val"]["class_report"])
            validation_stats[tag]["cm"].append(values["val"]["cm"])

            values["train"]["loss"] = 0
            values["train"]["y"].clear()
            values["train"]["y_pred"].clear()
            values["train"]["class_report"] = None
            values["train"]["cm"] = None

            values["val"]["loss"] = 0
            values["val"]["y"].clear()
            values["val"]["y_pred"].clear()
            values["val"]["class_report"] = None
            values["val"]["cm"] = None

        print(f"------- EPOCH {epoch+1:03d}/{epochs:03d}")
        print(tabulate(training_report_body, training_report_header, tablefmt="grid", floatfmt=".4f"))
    return training_stats, validation_stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to a dataset folder. Must follow required format, see README.")
    parser.add_argument("-m", "--model", type=str, default="exmeshcnn-base", help="CNN model to use. Default is 'exmeshcnn-base'")
    parser.add_argument("-o", "--output", type=str, default="./experiments", help="Output folder where to save experiment data. Default is 'experiments' folder.")
    parser.add_argument("-n", "--exp-name", type=str, default=None, help="Name of the experiment. Default is <YYYY>_<MM>_<DD>__<model>__<dataset>")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to be used, default is 0.")

    # Performance
    parser.add_argument("-b", "--batch-sz", type=int, default=2, help="Batch size to be used. Default is 2.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Dataloader workers, default is 4.")

    # Training specific arguments.
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-f", "--folds", type=int, default=1)
    parser.add_argument("-t", "--tags", type=str, required=True, nargs="+") # choices=["af", "ip", "use", "bn", "lse", "dm", "dp", "vb", "vm", "all"]
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam")

    # Configuration files, etc.
    parser.add_argument("--tag-yaml", type=str, default="./src/todd_characteristics.yaml", help="YAML file containing the number of categories per characteristic and their names. Default is located at '/src/todd_characteristics.yaml'")

    args = parser.parse_args()

    # Select the device to perform the computations.
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # For each selected tag...
    tag_data = {}
    for char in selected_tags:
        # Make a nested dictionary and,
        tag_data[char] = {}

        # Using char as the key, access the number of categories in current tag, ...
        tag_data[char]["classes"] = tag_metadata[char]["cat_num"]
        # ... make another nested dictionary to hold values relevant to the training, validation, and test ...
        tag_data[char]["train"] = {"y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}
        tag_data[char]["val"] = {"y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}
        tag_data[char]["test"] = {"y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}

        # ... and select a certaing loss function depending on the type of classification: Binary or multiple.
        if tag_metadata[char]["cat_num"] > 2:
            tag_data[char]["loss_fn"] = torch.nn.CrossEntropyLoss().to(device)
        else:
            tag_data[char]["loss_fn"] = torch.nn.BCEWithLogitsLoss().to(device)

    # Load the given model.
    try:
        model_module = importlib.import_module(f"models.{args.model}")
        model = model_module.get_model(tag_data, device)
    except Exception:
        print(f"Model {args.model} not found.")
        sys.exit(-1)


    # Load up the dataset
    dataset_df = pd.read_csv(os.path.join(args.dataset, "dataset.csv"))

    x_names = dataset_df.pop("name")
    x_names = x_names.to_numpy().reshape((-1,1))

    y_tags = dataset_df[selected_tags].to_numpy()

    # Split the data
    if len(selected_tags) > 1:
        X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(x_names, y_tags, test_size = 0.2)
        X_train, y_train, X_val, y_val =  iterative_train_test_split(X_train_val, y_train_val, test_size = 0.2)
    else:
        X_train_val, y_train_val, X_test, y_test = train_test_split(x_names, y_tags, test_size = 0.2, stratify=y_tags)
        X_train, y_train, X_val, y_val =  train_test_split(X_train_val, y_train_val, test_size = 0.2, stratify=y_train_val)

    X_train = X_train.reshape((X_train.shape[0],))
    X_val = X_val.reshape((X_val.shape[0],))
    X_test = X_test.reshape((X_test.shape[0],))

    y_train = pd.DataFrame(y_train, columns=selected_tags)
    y_val = pd.DataFrame(y_val, columns=selected_tags)
    y_test = pd.DataFrame(y_test, columns=selected_tags)

    train_loader = torch.utils.data.DataLoader(PSDataset(X_train, y_train, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(PSDataset(X_val, y_val, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(PSDataset(X_test, y_test, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=False, num_workers=args.workers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    training(model, train_loader, val_loader, tag_data, optimizer, args.epochs, device)
    
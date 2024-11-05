"""
    Main module for training / validating experiments, it can also be called as a module from
    another code.
"""

import importlib
import argparse
import datetime
import random
import copy
import time
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyrootutils
import torch
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

class EarlyStopper:
    """
    
    Source: https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"!EARLY STOP WARNING!: {self.counter} / {self.patience}", flush=True)
            if self.counter >= self.patience:
                return True
        elif validation_loss == np.nan:
            print("!NaN ENCOUNTERED!", flush=True)
            return True
        return False

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

            # if tag_data[act_tag]["classes"] == 2:
            #     act_y = act_y.unsqueeze(dim=1).float()
            #     act_y_pred = torch.round(torch.sigmoid(act_y_pred_logit)).squeeze(dim=1).tolist()
            # else:
            #     act_y_pred = torch.argmax(act_y_pred_logit, dim=1).tolist()

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
        values[mode]["cm"] = confusion_matrix(values[mode]["y"], values[mode]["y_pred"]).tolist()

def testing(model: torch.nn.Module, test_data, tag_data, device = "cuda"):
    
    testing_stats = {}
    for tag, _ in tag_data.items():
        testing_stats[tag] = {"acc" : -1, "f1" : -1, "loss" : -1, "class_report" : None, "cm": None}

    model_step("val", model, test_data, tag_data, device=device)

    testing_report_header = ["TAG", "TEST\nACC", "\nF1", "\nLOSS"]
    testing_report_body   = []

    curr_f1_val_all = []
    for tag, values in tag_data.items():

        curr_acc_val  =  values["val"]["class_report"]["accuracy"]
        curr_f1_val   =  values["val"]["class_report"]["macro avg"]["f1-score"]
        curr_loss_val =  values["val"]["loss"]

        curr_f1_val_all.append(curr_f1_val)

        testing_report_body.append([tag.upper(), curr_acc_val, curr_f1_val, curr_loss_val])

        testing_stats[tag]["acc"] = curr_acc_val
        testing_stats[tag]["f1"] = curr_f1_val
        testing_stats[tag]["loss"] = curr_loss_val
        testing_stats[tag]["class_report"] = values["val"]["class_report"]
        testing_stats[tag]["cm"] = values["val"]["cm"]

    testing_report_body.append(["*MEAN*", "0", np.mean(curr_f1_val_all), "0"])

    print("------- TEST ")
    print(tabulate(testing_report_body, testing_report_header, tablefmt="grid", floatfmt=".4f"))

    return testing_stats

def training(model: torch.nn.Module, train_data, validation_data, tag_data : dict, optimizer, epochs, device, epoch_begin = 0, best = False, es_watch : EarlyStopper = None):

    training_stats = {}
    validation_stats = {}
    best_model = {}

    for tag, _ in tag_data.items():
        training_stats[tag] = {"epoch" : [], "acc" : [], "f1" : [], "loss" : [], "class_report" : [], "cm": []}
        validation_stats[tag] = {"epoch" : [], "acc" : [], "f1" : [], "loss" : [], "class_report" : [], "cm": []}
        best_model[tag] = {"epoch" : -1, "f1" : -1, "model" : None}

    best_model["*mean*"] = {"epoch" : -1, "f1" : -1, "model" : None}

    for epoch in range(epochs):
        init_t = time.time()
        model_step("train", model, train_data, tag_data, optimizer=optimizer, device=device)
        model_step("val", model, validation_data, tag_data, device=device)
        end_t = time.time()

        training_report_header = ["TAG", "TRAINING\nACC", "\nF1", "\nLOSS", "VALIDATION\nACC", "\nF1", "\nLOSS"]
        training_report_body   = []

        mean_tag_stats = {"acc" : [], "f1": [], "loss" : []}
        for tag, values in tag_data.items():

            curr_acc_train  =  values["train"]["class_report"]["accuracy"]
            curr_f1_train   =  values["train"]["class_report"]["macro avg"]["f1-score"]
            curr_loss_train =  values["train"]["loss"]

            curr_acc_val  =  values["val"]["class_report"]["accuracy"]
            curr_f1_val   =  values["val"]["class_report"]["macro avg"]["f1-score"]
            curr_loss_val =  values["val"]["loss"]

            if curr_f1_val > best_model[tag]["f1"]:
                best_model[tag]["f1"] = curr_f1_val
                best_model[tag]["epoch"] = epoch
                best_model[tag]["model"] = copy.deepcopy(model).cpu()

            mean_tag_stats["acc"].append(curr_acc_val)
            mean_tag_stats["f1"].append(curr_f1_val)
            mean_tag_stats["loss"].append(curr_loss_val)

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

        if np.mean(mean_tag_stats["f1"]) > best_model["*mean*"]["f1"]:
            best_model["*mean*"]["f1"] = np.mean(mean_tag_stats["f1"])
            best_model["*mean*"]["epoch"] = epoch
            best_model["*mean*"]["model"] = copy.deepcopy(model).cpu()

        training_report_body.append(["*MEAN*", "0", "0", "0", "0", np.mean(mean_tag_stats["f1"]), np.mean(mean_tag_stats["loss"])])

        print(f"------- EPOCH {epoch+1:03d}/{epochs:03d} ({(end_t-init_t):.4f} s)")
        print(tabulate(training_report_body, training_report_header, tablefmt="grid", floatfmt=".4f"), flush=True)

        print("\nBEST:")

        best_header = ["TAG", "EPOCH", "VALIDATION F1"]
        best_body = []
        for key, value in best_model.items():
            best_body.append([key.upper(), value["epoch"] + 1, value["f1"]])

        print(tabulate(best_body, best_header, tablefmt="grid", floatfmt=".4f"), flush=True)

        if es_watch is not None and es_watch.early_stop(np.mean(mean_tag_stats["loss"])):
            break

    return training_stats, validation_stats, best_model

def get_class_weight(train_val_classes):
    _, classes_counts = np.unique(train_val_classes, return_counts=True)

    total_samples = len(train_val_classes)

    weight_func = np.vectorize(class_weight)
    weights = weight_func(classes_counts, total_samples)
    weights = torch.tensor(weights / np.sum(weights))

    return weights

def class_weight(class_sample_num, total_number_samples):
    return 1 - (class_sample_num / total_number_samples)

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

def save_training_stats():
    pass

def save_testing_stats(dir_path, test_results, tag_metadata):
    
    plot_path = os.path.join(dir_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    test_res = copy.deepcopy(test_results)

    for tag, item in test_res.items():
        confusion_matrix_plot(item["cm"], tag_metadata[tag]["cat_names"], os.path.join(plot_path, f"{tag}-test_cm_plot.pdf"))
        np.save(os.path.join(dir_path, f"{tag}-test_cm.npy"), item["cm"])
        item.pop("cm")

    with open(os.path.join(dir_path, "test_result.yaml"), mode="wt", encoding="utf8") as f:
        yaml.dump(test_res, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument("--dataset", type=str, help="Path to the dataset folder")
    parser.add_argument("--npy-name", type=str, help="Name of the NPY folder to be used. Must be inside the dataset folder.")
    parser.add_argument("-m", "--model", type=str, default="exmeshcnn-base", help="CNN model to use. Default is 'exmeshcnn-base'")
    parser.add_argument("-o", "--output", type=str, default="./experiments", help="Output folder where to save experiment data. Default is 'experiments' folder.")
    parser.add_argument("-n", "--exp-name", type=str, default=None, help="Name of the experiment. Default is <YYYY>_<MM>_<DD>__<model>__<dataset>")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to be used, default is 0.")
    parser.add_argument("--train-val-test-split", type=str, default=None, help="")

    # Performance
    parser.add_argument("-b", "--batch-sz", type=int, default=2, help="Batch size to be used. Default is 2.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Dataloader workers, default is 4.")

    # Training specific arguments.
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-f", "--folds", type=int, default=1)
    parser.add_argument("-t", "--tags", type=str, required=True, nargs="+") # choices=["af", "ip", "use", "bn", "lse", "dm", "dp", "vb", "vm", "all"]
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam")

    parser.add_argument("--model-weights", default=None)
    parser.add_argument("--train-params", default=None)

    # Output
    parser.add_argument("--output-path", type=str, default="./experiments")
    parser.add_argument("--output-name", type=str, default=None, help="Name of the experiment. Default is exp--<YYYY>_<MM>_<DD>__<model>__<dataset>")

    # Configuration files, etc.
    parser.add_argument("--tag-yaml", type=str, default="./src/todd_characteristics.yaml", help="YAML file containing the number of categories per characteristic and their names. Default is located at '/src/todd_characteristics.yaml'")

    # Debug
    parser.add_argument("--debug-use-cpu", action="store_true")

    args = parser.parse_args()

    ## Starting setup:

    # Turn off Matplotlib interactive mode
    plt.ioff()

    # Set random seeds
    RANDOM_STATE = args.seed
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        out_name = f"exp-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}-{args.model}-{dataset_basename}_{args.npy_name}"
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
        if args.train_params is None:
            model = model_module.get_model(tag_data)
        else:
            with open(args.train_params, "r", encoding="utf-8") as f:
                try:
                    train_params =  yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)
            model = model_module.get_model(tag_data, params=train_params)

        model = model.to(device)
    except Exception:
        print(f"Model {args.model} not found.")
        sys.exit(-1)

    if args.model_weights is not None:
        model.load_state_dict(torch.load(args.model_weights))

    # Generate the train-test split or use one provided.
    if args.train_val_test_split is None:
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
            X_train_val, X_test, y_train_val, y_test = train_test_split(x_names, y_tags, test_size = 0.2, stratify=y_tags, random_state=args.seed)
            X_train, X_val, y_train, y_val =  train_test_split(X_train_val, y_train_val, test_size = 0.2, stratify=y_train_val, random_state=args.seed)

        X_train = X_train.reshape((X_train.shape[0],))
        X_val = X_val.reshape((X_val.shape[0],))
        X_test = X_test.reshape((X_test.shape[0],))

        y_train = pd.DataFrame(y_train, columns=selected_tags)
        y_val = pd.DataFrame(y_val, columns=selected_tags)
        y_test = pd.DataFrame(y_test, columns=selected_tags)

        act_training_set = y_train.copy()
        act_validation_set = y_val.copy()
        act_test_set = y_test.copy()

        act_training_set.insert(0, "name", X_train)
        act_validation_set.insert(0, "name", X_val)
        act_test_set.insert(0, "name", X_test)

        act_training_set.to_csv(os.path.join(out_folder, "train.csv"), index=False)
        act_validation_set.to_csv(os.path.join(out_folder, "validation.csv"), index=False)
        act_test_set.to_csv(os.path.join(out_folder, "test.csv"), index=False)
    else:
        act_training_set = pd.read_csv(os.path.join(args.train_val_test_split, "train.csv"))
        act_validation_set = pd.read_csv(os.path.join(args.train_val_test_split, "validation.csv"))
        act_test_set = pd.read_csv(os.path.join(args.train_val_test_split, "test.csv"))

        y_train = pd.DataFrame(act_training_set, columns=selected_tags)
        y_val = pd.DataFrame(act_validation_set, columns=selected_tags)
        y_test = pd.DataFrame(act_test_set, columns=selected_tags)

        X_train = act_training_set["name"].to_numpy()
        X_val = act_validation_set["name"].to_numpy()
        X_test = act_test_set["name"].to_numpy()

    train_loader = torch.utils.data.DataLoader(PSDataset(X_train, y_train, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(PSDataset(X_val, y_val, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(PSDataset(X_test, y_test, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=False, num_workers=args.workers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    training(model, train_loader, val_loader, tag_data, optimizer, args.epochs, device)
    
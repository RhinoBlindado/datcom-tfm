"""
    Main module for training / validating experiments, it can also be called as a module from
    another code.
"""

import os
import sys
import argparse
import importlib

import pandas as pd
import pyrootutils
import torch
import tqdm
import yaml

from sklearn.metrics import classification_report, confusion_matrix

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

from exmeshcnn.datasetloader import PSDataset

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               tag_data : dict,
               optimizer: torch.optim.Optimizer,
               device):
    """
    a
    """
    # Put model in train mode
    model.train()

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
        for i, keyval in enumerate(y.items()):
            act_tag, act_y = keyval

            # ... calculate its own loss with its own loss function.
            act_loss = tag_data[act_tag]["loss_fn"](act_y.unsqueeze(dim=1).float().to(device),
                                                    y_preds[i])

            # Save the actual loss, prediction and correct tags
            tag_data[act_tag]["train"]["y"].extend(act_y.tolist())
            tag_data[act_tag]["train"]["y_pred"].extend(torch.round(torch.sigmoid(y_preds[i])).squeeze(dim=1).tolist())
            tag_data[act_tag]["train"]["loss"] += act_loss.tolist() / batch_len
            losses.append(act_loss)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        total_loss = sum(losses)
        total_loss.backward()

        # 5. Optimizer step
        optimizer.step()

    for _, values in tag_data.items():
        values["train"]["class_report"] = classification_report(values["train"]["y"], values["train"]["y_pred"],  output_dict=True, zero_division=0)
        values["train"]["cm"] = confusion_matrix(values["train"]["y"], values["train"]["y_pred"])



def training(model: torch.nn.Module, train_data, validation_data, tag_data : dict, optimizer, epochs, device, epoch_begin = 0):
    

    for epoch in range(epochs):

        train_step(model, train_data, tag_data, optimizer, device)

        train_report = ""
        for tag, values in tag_data.items():

            curr_acc  =  values["train"]["class_report"]["accuracy"]
            curr_f1   =  values["train"]["class_report"]["macro avg"]["f1-score"]
            curr_loss =  values["train"]["loss"]

            train_report += f"[{tag.upper()}] [T] ACC={curr_acc:.2f} F1={curr_f1:.2f} LOSS={curr_loss:.2f} "

            values["train"]["loss"] = 0
            values["train"]["y"].clear()
            values["train"]["y_pred"].clear()
            values["train"]["class_report"] = None
            values["train"]["cm"] = None

        print(f"{epoch+1:03d}/{epochs:03d}: {train_report}")

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
    parser.add_argument("-t", "--tags", type=str, required=True, nargs="+", choices=["af", "ip", "use", "bn", "lse", "dm", "dp", "vb", "vm", "all"])
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
    with open("./src/todd_characteristics.yaml", "r", encoding="utf-8") as f:
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

    # This needs to be changed to divide the data in some way, but that's not for todayyyyy
    train_df = pd.read_csv(os.path.join(args.dataset, "dataset.csv"))
    # val_df
    # test_df

    train_loader = torch.utils.data.DataLoader(PSDataset(train_df, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(PSDataset(train_df, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=True, num_workers=args.workers)
    # test_loader = torch.utils.data.DataLoader(PSDataset(train_df, args.dataset, tag_data), batch_size=args.batch_sz, shuffle=True, num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training(model, train_loader, val_loader, tag_data, optimizer, 30, device)

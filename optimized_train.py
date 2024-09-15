"""

"""

import importlib
import argparse
import os

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import optuna
import torch
import yaml

from exmeshcnn.datasetloader import PSDataset
import train_test

class HyperParamOptimizer():
    """
    
    """
    def __init__(self, model_str, dataset_str, output_folder, X_train, y_train, X_val, y_val, tag_data, tag_meta, n_epochs, n_folds, n_batch, n_workers, device = "cuda") -> None:
        
        # Training parameters
        self.model_str = model_str
        self.dataset_str = dataset_str
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.tag_data = tag_data
        self.epochs = n_epochs
        self.folds = n_folds
        self.batch_sz = n_batch

        # Misc Parameters
        self.dataloader_workers = n_workers
        self.device = device
        self.output_f = output_folder
        self.tag_meta = tag_meta


    def load_model(self, trial):
        model_class = importlib.import_module(f"models.{self.model_str}")
        return model_class.get_model(tag_data, trial).to(self.device)
        

    def save_trial_stats(self, trial, model, training_stats, validation_stats, best = -1):
        trial_path = os.path.join(self.output_f, f"trial_{trial.number}")
        os.mkdir(trial_path)

        torch.save(model.state_dict(), os.path.join(trial_path, "trial_model.pth"))

        trial_plot_path = os.path.join(trial_path, "plots")
        os.mkdir(trial_plot_path)
        for act_train_res, act_val_res in zip(training_stats.items(), validation_stats.items()):
            tag, train_items = act_train_res
            _, val_items = act_val_res

            pd.DataFrame(train_items).to_csv(os.path.join(trial_path, f"{tag}-training-prog.csv"), index=None)
            pd.DataFrame(val_items).to_csv(os.path.join(trial_path, f"{tag}-validation-prog.csv"), index=None)

            train_test.progression_plot(train_items["epoch"], train_items["acc"], val_items["acc"], "Accuracy", os.path.join(trial_plot_path, f"{tag}-acc.pdf"), ylim=(0,1))
            train_test.progression_plot(train_items["epoch"], train_items["f1"], val_items["f1"], "F1", os.path.join(trial_plot_path, f"{tag}-f1.pdf"), ylim=(0,1))
            train_test.progression_plot(train_items["epoch"], train_items["loss"], val_items["loss"], "Loss", os.path.join(trial_plot_path, f"{tag}-loss.pdf"))

            train_test.confusion_matrix_plot(train_items["cm"][-1], self.tag_meta[tag]["cat_names"], os.path.join(trial_plot_path, f"{tag}-training-cm.pdf"))
            train_test.confusion_matrix_plot(val_items["cm"][-1], self.tag_meta[tag]["cat_names"], os.path.join(trial_plot_path, f"{tag}-validation-cm.pdf"))

    def calculate_fitness(self, act_train_f1, act_val_f1):
        return act_val_f1 * (1 - abs(act_train_f1 - act_val_f1))

    def optimize_model(self, trial):

        print(f"---------------- TRIAL {trial.number} ----------------")

        ## Set up the parameters to optimize.
        # Model Parameters
        model = self.load_model(trial)

        # Learning Rate
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        # By Layers
        # Not yet implemented
        # Cyclical
        # Not yet implemented

        # Optimizer
        optimizer_selection = trial.suggest_categorical("optimizer", ["adam", "adamw", "radam", "sgd"])

        if optimizer_selection == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_selection == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_selection == "radam":
            optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Weight Initialization.

        # Loss Function.
        # Not yet implemented.

        # Load the dataset.
        train_loader = torch.utils.data.DataLoader(PSDataset(self.X_train, self.y_train, self.dataset_str, self.tag_data), batch_size=self.batch_sz, shuffle=True, num_workers=self.dataloader_workers)
        val_loader = torch.utils.data.DataLoader(PSDataset(self.X_val, self.y_val, self.dataset_str, self.tag_data), batch_size=self.batch_sz, shuffle=False, num_workers=self.dataloader_workers)

        # Run the model with the parameters.
        train_res, val_res = train_test.training(model, train_loader, val_loader, self.tag_data, optimizer, self.epochs, self.device)

        # Record results and return optimization value.
        total_fitness = []
        for act_train_res, act_val_res in zip(train_res.items(), val_res.items()):
            best_train_f1 = act_train_res[1]["f1"][-1]
            best_val_f1 = act_val_res[1]["f1"][-1]

            total_fitness.append(self.calculate_fitness(best_train_f1, best_val_f1))

        total_fitness = np.mean(total_fitness)

        self.save_trial_stats(trial, model, train_res, val_res)

        return total_fitness

def show_optimization_info(study, save, output_path):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    if save:
        study.trials_dataframe().to_csv(os.path.join(output_path, "trials.csv"))

        # Save results to file
        with open(os.path.join(output_path, "optuna_results.txt"), "w") as f:
            f.write("Study statistics: \n")
            f.write("  Number of finished trials: {}\n".format(len(study.trials)))
            f.write("  Number of pruned trials: {}\n".format(len(pruned_trials)))
            f.write("  Number of complete trials: {}\n".format(len(complete_trials)))

            trial = study.best_trial
            f.write("Best trial: {}\n".format(trial.number))

            f.write(" - Value: {}\n".format(trial.value))

            f.write(" - Params: \n")
            for key, value in trial.params.items():
                f.write("    {}: {}\n".format(key, value))

        with open(os.path.join(output_path, 'best_optuna_params.yaml'), 'w') as fout:
            yaml.dump(trial.params, fout, default_flow_style=False)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial: {}".format(trial.number))

    print(" - Value: ", trial.value)

    print(" - Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save figs with information about the optimization process
    if save: 
        print("Saving plots to {}".format(output_path))

        optuna.visualization.matplotlib.plot_intermediate_values(study)
        plt.savefig(os.path.join(output_path, "intermediate_values.png"), bbox_inches='tight')

        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(output_path, "param_importances.png"), bbox_inches='tight')
        except:
            print("Encountered zero total variance in all trees. If all trees have 0 variance, we cannot assess any importances. This could occur if for instance `X.shape[0] == 1`.")

        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(output_path, "optimization_history.png"), bbox_inches='tight')


parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--dataset", type=str, help="Path to the dataset folder")
parser.add_argument("--npy-name", type=str, help="Name of the NPY folder to be used. Must be inside the dataset folder.")
parser.add_argument("--model", type=str, help="Name of the model to optimize with.")
parser.add_argument("--train-val-test-split", type=str, default=None, help="Name of the model to optimize with.")
parser.add_argument("-t", "--tags", type=str, required=True, nargs="+") # choices=["af", "ip", "use", "bn", "lse", "dm", "dp", "vb", "vm", "all"]
parser.add_argument("--tag-yaml", type=str, default="./src/todd_characteristics.yaml", help="YAML file containing the number of categories per characteristic and their names. Default is located at '/src/todd_characteristics.yaml'")

# Output parameters
parser.add_argument("--output-path", type=str, default="./experiments")
parser.add_argument("--output-name", type=str, default=None, help="Name of the experiment. Default is opt__<YYYY>_<MM>_<DD>__<model>__<dataset>")

# Optimization settings
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--folds", type=int, default=1)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=0, help="Random seed to be used, default is 0.")

# Debug
parser.add_argument("--debug-use-cpu", action="store_true")
parser.add_argument("--debug-verbose", action="store_true")
parser.add_argument("--debug-no-save", action="store_true")

args = parser.parse_args()

## Starting setup:

plt.ioff()

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
    out_name = f"opt-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}-{args.model}-{dataset_basename}_{args.npy_name}"
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

    # y_train.to_csv(os.path.join(out_folder, "train.csv"), index=False)
    # y_val.to_csv(os.path.join(out_folder, "validation.csv"), index=False)
    # y_test.to_csv(os.path.join(out_folder, "test.csv"), index=False)
else:
    pass
# Create the Optuna and...
study = optuna.create_study(study_name=out_name, direction="maximize")
hpo = HyperParamOptimizer(args.model, args.dataset, out_folder, X_train, y_train, X_val, y_val, tag_data, tag_metadata, args.epochs, args.folds, args.batch, args.workers)

# ...start the optimization procedure, then...
study.optimize(hpo.optimize_model, n_trials=args.trials)

# ...save metadata after optimization.
# show_optimization_info(study, out_folder)
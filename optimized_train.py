"""

"""

import importlib
import argparse
import random
import os

from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import optuna
import torch
import yaml

from exmeshcnn.datasetloader import PSDataset
from src.class_balanced_loss import ClassBalancedLoss
from src.focalloss import FocalLoss
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

        # Best model
        self.best_trial = -1
        self.best_fitness = -1
        self.best_model = None


    def load_model(self, trial):
        model_class = importlib.import_module(f"models.{self.model_str}")
        return model_class.get_model(tag_data, trial).to(self.device)
        

    def save_trial_stats(self, trial, model, training_stats, validation_stats, best_mean_epoch):

        trial_path = os.path.join(self.output_f, f"trial_{trial.number}")
        os.mkdir(trial_path)

        torch.save(model.state_dict(), os.path.join(trial_path, "trial_model_state_dict.pth"))
        
        with open(os.path.join(trial_path, "params.yaml"), mode="wt", encoding="utf8") as fparam:
            yaml.dump(trial.params, fparam)

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

            train_test.confusion_matrix_plot(train_items["cm"][best_mean_epoch], self.tag_meta[tag]["cat_names"], os.path.join(trial_plot_path, f"{tag}-training_cm_plot.pdf"))
            train_test.confusion_matrix_plot(val_items["cm"][best_mean_epoch], self.tag_meta[tag]["cat_names"], os.path.join(trial_plot_path, f"{tag}-validation_cm_plot.pdf"))

            with open(os.path.join(trial_path, f"{tag}-best_model-train_val_result.yaml"), mode="wt", encoding="utf8") as f_act:
                act_dict = {"train" : {"acc": train_items["acc"][best_mean_epoch], 
                                       "f1" : train_items["f1"][best_mean_epoch],
                                       "loss" : train_items["loss"][best_mean_epoch]}, 
                            "val" :   {"acc": val_items["acc"][best_mean_epoch], 
                                       "f1" : val_items["f1"][best_mean_epoch],
                                       "loss" : val_items["loss"][best_mean_epoch]}}
                yaml.dump(act_dict, f_act)

            np.save(os.path.join(trial_path, f"{tag}-training_cm.npy"), train_items["cm"][best_mean_epoch])
            np.save(os.path.join(trial_path, f"{tag}-validation_cm.npy"), val_items["cm"][best_mean_epoch])


    def init_weights(self, net, optuna_trial):
        init_type = optuna_trial.suggest_categorical("init_type", ["normal", "xavier", "kaiming", "orthogonal", "none"])
        init_gain = optuna_trial.suggest_float("init_gain", 1, 10)

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
                torch.nn.init.constant_(m.bias.data, 0.0)
        
        if init_type != "none":
            net.apply(init_func)

    def set_loss(self, optuna_trial):

        for key, val in self.tag_data.items():
            act_classes = val["classes"]
            act_samples_per_class = val["samples_per_class"]

            loss_fn_selection = optuna_trial.suggest_categorical(f"loss_{key}", ["cross_entropy", "cross_entropy_weighted", "focal", "class_balanced"])
            
            if loss_fn_selection == "cross_entropy":
                self.tag_data[key]["loss_fn"] = torch.nn.CrossEntropyLoss().to(self.device)
            
            elif loss_fn_selection == "cross_entropy_weighted":
                act_weighted = torch.tensor(1 - (act_samples_per_class / np.sum(act_samples_per_class)))
                act_weighted.to(self.device)

                self.tag_data[key]["loss_fn"] = torch.nn.CrossEntropyLoss(weight=act_weighted.float()).to(self.device)
            
            elif loss_fn_selection == "focal":
                fl_gamma = optuna_trial.suggest_float("fl_gamma", 1, 10)
                self.tag_data[key]["loss_fn"] = FocalLoss(gamma=fl_gamma).to(self.device)
            
            elif loss_fn_selection == "class_balanced":
                cbl_type = optuna_trial.suggest_categorical("cbl_type",["sigmoid", "focal", "softmax"])
                cbl_beta = optuna_trial.suggest_float("cbl_beta", 0, 0.9999)

                if cbl_type == "focal":
                    cbl_gamma = optuna_trial.suggest_float("cbl_gamma", 1, 10)
                else:
                    cbl_gamma = None
                
                self.tag_data[key]["loss_fn"] = ClassBalancedLoss(act_samples_per_class, act_classes, cbl_type, cbl_beta, cbl_gamma, device=self.device).to(self.device)

    def calculate_fitness(self, act_train_f1, act_val_f1):
        return act_val_f1 * (1 - abs(act_train_f1 - act_val_f1))
    
    def get_best_model(self):
        return self.best_model, self.best_fitness, self.best_trial

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
        self.init_weights(model, trial)

        # Loss Function.
        self.set_loss(trial)

        # Load the dataset.
        train_loader = torch.utils.data.DataLoader(PSDataset(self.X_train, self.y_train, self.dataset_str, self.tag_data, npy_name=args.npy_name), batch_size=self.batch_sz, shuffle=True, num_workers=self.dataloader_workers)
        val_loader = torch.utils.data.DataLoader(PSDataset(self.X_val, self.y_val, self.dataset_str, self.tag_data, npy_name=args.npy_name), batch_size=self.batch_sz, shuffle=False, num_workers=self.dataloader_workers)

        # Instantiate the EarlyStopper
        early_stopper = train_test.EarlyStopper(patience = int(self.epochs // 3), min_delta = 0.09)

        # Run the model with the parameters.
        train_res, val_res, best_models = train_test.training(model, train_loader, val_loader, self.tag_data, optimizer, self.epochs, self.device, es_watch=early_stopper)

        # Record results and return optimization value.
        total_fitness = []
        for act_train_res, act_val_res in zip(train_res.items(), val_res.items()):
            best_train_f1 = act_train_res[1]["f1"][best_models["*mean*"]["epoch"]]
            best_val_f1 = act_val_res[1]["f1"][best_models["*mean*"]["epoch"]]

            total_fitness.append(self.calculate_fitness(best_train_f1, best_val_f1))

        total_fitness = np.mean(total_fitness)

        self.save_trial_stats(trial, best_models["*mean*"]["model"], train_res, val_res, best_models["*mean*"]["epoch"])

        if total_fitness > self.best_fitness:
            self.best_trial = trial.number
            self.best_fitness = total_fitness
            self.best_model = best_models["*mean*"]["model"]

        del optimizer
        del model
        torch.cuda.empty_cache()

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
            optuna_params = trial.params
            optuna_params.update(study.user_attrs)
            yaml.dump(optuna_params, fout, default_flow_style=False)

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
        plt.savefig(os.path.join(output_path, "intermediate_values.pdf"), format="pdf", bbox_inches='tight')

        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(output_path, "param_importances.pdf"), format="pdf", bbox_inches='tight')
        except:
            print("Encountered zero total variance in all trees. If all trees have 0 variance, we cannot assess any importances. This could occur if for instance `X.shape[0] == 1`.")

        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(output_path, "optimization_history.pdf"), format="pdf", bbox_inches='tight')

def get_reduced_dataset(input_df, sel_tags):
    copy_df_in = input_df.copy()
    x_names = copy_df_in.pop("subject")
    x_names = np.unique(x_names.to_numpy()).reshape((-1,1))
    x_names_lst = list(x_names.squeeze())
    
    reduced_df = pd.DataFrame()

    for _, row in input_df.iterrows():
        act_name = row["name"]
        act_bname = int(act_name.split("-")[0])

        if act_bname in x_names_lst:
            x_names_lst.remove(act_bname)
            reduced_df = pd.concat([reduced_df, pd.DataFrame(row).transpose()], ignore_index=True)
        
        if len(x_names_lst) == 0:
            break

    y_tags = reduced_df[sel_tags].to_numpy()

    return x_names, y_tags

def expand_dataset(input_df, sel_tags, x_names):
    
    x_names_lst = x_names.squeeze()
    exp_ytags = pd.DataFrame()

    for name in x_names_lst:
        act_samples = input_df[input_df["subject"] == name]
        exp_ytags = pd.concat([exp_ytags, act_samples], ignore_index=True)

    exp_xnames = exp_ytags.pop("name")
    exp_xnames = exp_xnames.to_numpy()
    
    exp_ytags = exp_ytags[sel_tags]

    return exp_xnames, exp_ytags

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--dataset", type=str, help="Path to the dataset folder")
parser.add_argument("--npy-name", type=str, help="Name of the NPY folder to be used. Must be inside the dataset folder.")
parser.add_argument("--model", type=str, help="Name of the model to optimize with.")
parser.add_argument("--train-val-test-split", type=str, default=None, help="")
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
# parser.add_argument("--weighted-loss", action="store_true")

# Debug
parser.add_argument("--debug-use-cpu", action="store_true")
parser.add_argument("--debug-verbose", action="store_true")
parser.add_argument("--debug-no-save", action="store_true")

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
    tags_str = "_".join(sorted(list(set(args.tags))))
    out_name = f"opt-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}-{args.model}-{dataset_basename}-{args.npy_name}-{tags_str}"
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

# Generate the train-test split or use one provided.
if args.train_val_test_split is None:
    # Load up the dataset
    dataset_df = pd.read_csv(os.path.join(args.dataset, "dataset.csv"))

    if "subject" in dataset_df.columns:
        x_names, y_tags = get_reduced_dataset(dataset_df, selected_tags)
    else:
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

    if "subject" in dataset_df.columns:
        X_train, y_train = expand_dataset(dataset_df, selected_tags, X_train)
        X_val, y_val = expand_dataset(dataset_df, selected_tags, X_val)
        X_test, y_test = expand_dataset(dataset_df, selected_tags, X_test)
    else:
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


    # For each selected tag...
tag_data = {}
for char in selected_tags:
    # Make a nested dictionary and,
    tag_data[char] = {}

    # Using char as the key, access the number of categories in current tag, ...
    tag_data[char]["classes"] = tag_metadata[char]["cat_num"]
    _ , tag_data[char]["samples_per_class"] = np.unique(y_train[char], return_counts=True)
    # ... make another nested dictionary to hold values relevant to the training, validation, and test ...
    tag_data[char]["train"] = {"y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}
    tag_data[char]["val"] = {"y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}
    tag_data[char]["test"] = {"y" : [], "y_pred" : [], "loss" : 0, "class_report" : None, "cm" : None}

    # Get the weights for the loss...
    # tag_weights = train_test.get_class_weight(y_train[char])
    # tag_weights = tag_weights.to(device)

    # ... and select a certaing loss function depending on the type of classification: Binary or multiple.
    # if tag_metadata[char]["cat_num"] > 2:
    #     tag_data[char]["loss_fn"] = torch.nn.CrossEntropyLoss(weight=tag_weights.float()).to(device)
    # else:
    #     tag_data[char]["loss_fn"] = torch.nn.BCEWithLogitsLoss().to(device)

# Create the Optuna and...
study = optuna.create_study(study_name=out_name, direction="maximize")

study.set_user_attr("trials", args.trials)
study.set_user_attr("epochs", args.epochs)
study.set_user_attr("folds", args.folds)
study.set_user_attr("batch", args.batch)
study.set_user_attr("workers", args.workers)
study.set_user_attr("seed", args.seed)

trials_folder = os.path.join(out_folder, "trials")
os.makedirs(trials_folder)
hpo = HyperParamOptimizer(args.model, args.dataset, trials_folder, X_train, y_train, X_val, y_val, tag_data, tag_metadata, args.epochs, args.folds, args.batch, args.workers, device=device)

# ...start the optimization procedure, then...
study.optimize(hpo.optimize_model, n_trials=args.trials)

# ...save metadata after optimization.
trials_metadata_folder = os.path.join(out_folder, "trials_metadata")
os.makedirs(trials_metadata_folder)
show_optimization_info(study, True, trials_metadata_folder)

best_model, _, _ = hpo.get_best_model()
best_model = best_model.to(device)
test_loader = torch.utils.data.DataLoader(PSDataset(X_test, y_test, args.dataset, tag_data, npy_name=args.npy_name), batch_size=args.batch, shuffle=False, num_workers=args.workers)
testing_results = train_test.testing(best_model, test_loader, tag_data, device=device)

test_folder = os.path.join(out_folder, "test")
os.makedirs(test_folder)

train_test.save_testing_stats(test_folder, testing_results, tag_metadata)
print("Done!")

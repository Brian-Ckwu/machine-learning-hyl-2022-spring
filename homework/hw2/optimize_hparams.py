import random
import torch
import gc
import json
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from typing import List, Dict

from data import LibriRNNDataset, RNNCollator
from model import RNNClassifier
from utils import *

from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

ax_client = AxClient()
N_TRIALS = 50

ax_client.create_experiment(
    name="lstm_softmax_tagging",
    parameters=[
        {
            "name": "batch_size",
            "type": "choice",
            "values": [8, 16],
            "is_ordered": True
        },
        {
            "name": "hidden_size",
            "type": "choice",
            "values": [128, 192, 256, 320, 384, 448, 512],
            "is_ordered": True
        },
        {
            "name": "dropout",
            "type": "range",
            "bounds": [0.0, 0.99]
        },
        {
            "name": "classifier_dropout",
            "type": "range",
            "bounds": [0.0, 0.75]
        },
        {
            "name": "lr",
            "type": "range",
            "bounds": [0.00001, 0.005]
        }
    ],
    objective_name="acc",
    minimize=False
)

def evaluate_hparams(parameters, config):
    # set config
    config["batch_size"] = parameters["batch_size"]
    config["rnn_args"]["hidden_size"] = parameters["hidden_size"]
    config["rnn_args"]["dropout"] = parameters["dropout"]
    config["classifier_dropout"] = parameters["classifier_dropout"]
    config["optimizer_hparams"]["lr"] = parameters["lr"]

    config["model_name"] = f"lstm_DNN_NAdam_batch-{parameters['batch_size']}_hidden-{parameters['hidden_size']}_dropout-{parameters['dropout']}-{parameters['classifier_dropout']}_lr-{parameters['lr']}"
    assert torch.cuda.is_available()
    device = "cuda:0"
    same_seeds(config["seed"])

    train_set = LibriRNNDataset(mode="train", train_ratio=config["train_ratio"], concat_nframes=config["concat_nframes"])
    val_set = LibriRNNDataset(mode="val", train_ratio=config["train_ratio"], concat_nframes=config["concat_nframes"])

    # make dataloader
    rnn_collator = RNNCollator(mode="train")
    train_loader = DataLoader(train_set, config["batch_size"], shuffle=True, pin_memory=True, collate_fn=rnn_collator)
    val_loader = DataLoader(val_set, config["batch_size"], shuffle=False, pin_memory=False, collate_fn=rnn_collator)

    model = RNNClassifier(rnn_type=config["rnn_type"], rnn_args=config["rnn_args"], classifier_dropout=config["classifier_dropout"]).to(device)
    criterion = CrossEntropyLoss(reduction="mean")
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    best_acc = trainer(train_loader, val_loader, model, criterion, optimizer, config, device)
    print(f"\n\nBest accuracy: {best_acc}\n")

    with open("./evaluations/ax_results/{}.txt".format(config["model_name"]), mode="wt") as f:
        f.write(str(best_acc))
    
    del train_set, val_set, train_loader, val_loader, model, optimizer
    gc.collect()

    return {"acc": (best_acc, 0.0)}

with open("./lstm_config.json") as f:
    config = json.load(f)

for i in range(N_TRIALS):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index, raw_data=evaluate_hparams(parameters, config))

best_parameters, values = ax_client.get_best_parameters()
print(best_parameters)
print(values)

with open("./optim_config_2.json", mode="wt") as f:
    json.dump(best_parameters, f)
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

"""
    Configuration
"""
with open("./lstm_config.json") as f:
    config = json.load(f)

assert torch.cuda.is_available()
device = "cuda:1"

same_seeds(config["seed"])

"""
    Data
"""

lr_schedule = [3e-3, 2e-3, 1e-3, 8e-4, 5e-4, 3e-4, 2e-4, 1e-4]

for lr in lr_schedule:
    config["optimizer_hparams"]["lr"] = lr
    config["model_name"] = f"lstm_DNN_NAdam_batch-32_hidden-256_nlayers-2_dropout-0.975-0.50_lr-{lr}_trainratio-0.95"

    # make dataset
    train_set = LibriRNNDataset(mode="train", train_ratio=config["train_ratio"], concat_nframes=config["concat_nframes"])
    val_set = LibriRNNDataset(mode="val", train_ratio=config["train_ratio"], concat_nframes=config["concat_nframes"])

    # make dataloader
    rnn_collator = RNNCollator(mode="train")
    train_loader = DataLoader(train_set, config["batch_size"], shuffle=True, pin_memory=True, collate_fn=rnn_collator)
    val_loader = DataLoader(val_set, config["batch_size"], shuffle=False, pin_memory=False, collate_fn=rnn_collator)

    """
        Model, Loss, and Optimizer
    """
    model = RNNClassifier(rnn_type=config["rnn_type"], rnn_args=config["rnn_args"], classifier_dropout=config["classifier_dropout"]).to(device)
    criterion = CrossEntropyLoss(reduction="mean")
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    best_acc = trainer(train_loader, val_loader, model, criterion, optimizer, config, device)
    print(f"\n\nBest accuracy: {best_acc}\n")

    with open("./evaluations/{}.txt".format(config["model_name"]), mode="wt") as f:
        f.write(str(best_acc))
    
    del train_set, val_set, train_loader, val_loader, model, optimizer, best_acc
    gc.collect()
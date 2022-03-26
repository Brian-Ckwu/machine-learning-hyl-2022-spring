import random
import torch
import os
import torch.nn as nn
import json
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
device = "cuda:0"

same_seeds(config["seed"])

"""
    Test
"""
rnn_collator = RNNCollator(mode="test")
test_set = LibriRNNDataset(mode="test", concat_nframes=config["concat_nframes"])
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, collate_fn=rnn_collator)

model = RNNClassifier(rnn_type=config["rnn_type"], rnn_args=config["rnn_args"], classifier_dropout=config["classifier_dropout"]).to(device)
model.load_state_dict(torch.load("./models/" + config["model_name"] + ".pth"))

preds = tester(test_loader, model, device)
save_pred(preds, config["model_name"])
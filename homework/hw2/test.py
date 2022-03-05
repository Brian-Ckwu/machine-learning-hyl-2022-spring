"""
    Import Packages
"""

import json
import torch
from torch.utils.data import DataLoader

from utils import *
from data import *
from model import *

"""
    Configuration
"""

# Config file
config_file = "./config.json"
with open(config_file) as f:
    config = json.load(f)

# Get device
assert torch.cuda.is_available()
device = "cuda"

# Seeding
same_seeds(config["seed"])

"""
    Testing
"""

# Load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=config["concat_nframes"])
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

# Load model
model = Classifier(input_dim=config["feature_dim"]*config["concat_nframes"], hidden_layers=config["hidden_layers"], hidden_dim=config["hidden_dim"]).to(device)
model.load_state_dict(torch.load("./models/{}.pth".format(config["model_name"])))

# Evaluate
pred = tester(test_loader, model, device)

# Save results
save_pred(pred, config["model_name"])
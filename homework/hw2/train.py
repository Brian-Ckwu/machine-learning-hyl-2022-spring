"""
    Import Packages
"""

import gc
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
    Loop Through Target Configurations
"""

# num_epoch = config["num_epoch"]
# target_nlayers = [3]
# target_nframes = range(31, 33, 2)

# for target_nlayer in target_nlayers:
#     config["hidden_layers"] = target_nlayer
#     for target_nframe in target_nframes:
#         print("\n========== Training the model of {}-layers, {}-frames ==========\n".format(target_nlayer, target_nframe))
#         # set configuration
#         config["concat_nframes"] = target_nframe
#         config["model_name"] = "sample_code_{}_epochs_{}_layers_{}_frames".format(num_epoch, target_nlayer, target_nframe)

# data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=config["concat_nframes"], train_ratio=config["train_ratio"])
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=config["concat_nframes"], train_ratio=config["train_ratio"])

train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

del train_X, train_y, val_X, val_y # remove raw feature to save memory
gc.collect()

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)


# Model, Loss, and Optimizer
model = Classifier(input_dim=config["feature_dim"]*config["concat_nframes"], hidden_layers=config["hidden_layers"], hidden_dim=config["hidden_dim"], dropout_p=config["dropout_p"]).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

# Optimization
best_acc = trainer(train_loader, val_loader, model, criterion, optimizer, config, device)
save_performance(best_acc, config["model_name"])

# free memories for the next configuration
del train_set, val_set, train_loader, val_loader, model, criterion, optimizer, best_acc
gc.collect()
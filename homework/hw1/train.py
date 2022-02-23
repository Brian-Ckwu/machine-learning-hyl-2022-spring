"""
    import packages
"""
import os
import json
import pandas as pd

# Scikit-learn & Pytorch
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

# import utilities
from utils.data import COVID19Dataset
from utils import models
from utils.funcs import *
from utils.plot import plot_learning_curve

"""
    Configuration
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config_file = "./config.json"
with open(config_file) as f:
    config = json.load(f)
# config['feats'] = list(range(37, 116))

# seeding to achieve reproducable results
same_seed(config["seed"])

"""
    Data
"""
# load data
train_data = pd.read_csv('./data/covid_train.csv', index_col='id').values
test_data = pd.read_csv('./data/covid_test.csv', index_col='id').values

# preprocessing
train_data, val_data =  train_valid_split(train_data, config['valid_ratio'], config['seed'])
train_x, val_x, test_x, train_y, val_y = select_feat(train_data, val_data, test_data, feats=config['feats'])
print(f"Train_x shape: {train_x.shape}\n \
        Val_x   shape: {val_x.shape}\n   \
        Test_x  shape: {test_x.shape}") # check shape

# make dataset & dataloader
train_dataset, val_dataset, test_dataset = COVID19Dataset(train_x, train_y), \
                                           COVID19Dataset(val_x, val_y), \
                                           COVID19Dataset(test_x)
train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True), \
                                                    DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True), \
                                                    DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

"""
    Model
"""
model = getattr(models, config["model"])(input_dim=train_x.shape[1]).to(DEVICE)
if not os.path.isdir("./models/" + config["model_name"]):
    os.mkdir("./models/" + config["model_name"])

"""
    Loss
"""
criterion = nn.MSELoss(reduction="mean")

"""
    Optimization
"""
min_val_loss, loss_record = new_trainer(train_dataloader, val_dataloader, model, criterion, config, DEVICE)

# save model configuration to the model folder
save_path = f"./models/{config['model_name']}/"
with open(save_path + "config.json", mode="wt") as f:
    json.dump(config, f)

# load the saved model and make predictions on the test set
model_pred = getattr(models, config["model"])(input_dim=train_x.shape[1]).to(DEVICE)
model_pred.load_state_dict(torch.load(save_path + "model.pth"))
preds = predict(test_dataloader, model_pred, DEVICE)
save_pred(preds, f"./preds/{config['model_name']}.csv")

# plot training curve
plot_learning_curve(loss_record, title="Learning Curve", xlabel="Training Steps", ylabel="MSE Loss")


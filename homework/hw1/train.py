"""
    import packages
"""
import json
import pandas as pd

# Scikit-learn & Pytorch
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

# import utilities
from utils.data import COVID19Dataset
from utils.model import *
from utils.funcs import *

"""
    Configuration
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config_file = "./config.json"
with open(config_file) as f:
    config = json.load(f)
MODEL = SampleNN

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
model = MODEL(input_dim=train_x.shape[1]).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

"""
    Optimization
"""
model_name = f"{MODEL.__name__}"
final_train_loss, final_val_loss, best_epoch = trainer(train_dataloader, val_dataloader, model, optimizer, config, DEVICE, report_every_n_epochs=100, save_model=True, model_name=model_name)

# save model configuration to the model folder
save_path = f"./models/{model_name}/config.json"
with open(save_path, mode="wt") as f:
    json.dump(config, f)
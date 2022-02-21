"""
    import packages
"""
import pandas as pd
from sklearn.model_selection import KFold

# Pytorch
import torch
from torch.utils.data import DataLoader

# import utilities
from utils.data import COVID19Dataset
from utils.model import *
from utils.funcs import *

"""
    Configuration
"""


"""
    Data
"""
# load data
train_data = pd.read_csv('./data/covid_train.csv', index_col='id').values
test_data = pd.read_csv('./data/covid_test.csv', index_col='id').values

# preprocessing
train_valid_split(train_data)




# save model configuration to the model folder
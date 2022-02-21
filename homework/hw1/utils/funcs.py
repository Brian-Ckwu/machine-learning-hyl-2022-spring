import torch
import numpy as np
import math
import csv
import os
from tqdm import tqdm
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def select_feat(train_data, valid_data, test_data, feats=None, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = feats # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

def trainer(train_loader, valid_loader, model, optimizer, config, device, report_every_n_epochs, save_model=False):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = optimizer #torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay']) 

    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_val_loss, step, early_stop_count, best_epoch = config['n_epochs'], math.inf, 0, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        train_loss_record = list()

        for x, y in train_loader:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            train_loss = criterion(pred, y)
            train_loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            train_loss_record.append(train_loss.detach().item())

        mean_train_loss = sum(train_loss_record)/len(train_loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        val_loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                val_loss = criterion(pred, y)

            val_loss_record.append(val_loss.item())
        

        mean_val_loss = sum(val_loss_record)/len(val_loss_record)
        writer.add_scalar('Loss/valid', mean_val_loss, step)
        if (epoch + 1) % report_every_n_epochs == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_val_loss:.4f}')

        if mean_val_loss < best_val_loss:
            train_loss_with_best_val_loss = mean_train_loss
            best_val_loss = mean_val_loss
            if save_model:
                torch.save(model.state_dict(), config['save_path']) # Save your best model
                print('Saving model with loss {:.3f}...'.format(best_val_loss))
            early_stop_count = 0
            best_epoch = epoch
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break
    
    return train_loss_with_best_val_loss, best_val_loss, best_epoch

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
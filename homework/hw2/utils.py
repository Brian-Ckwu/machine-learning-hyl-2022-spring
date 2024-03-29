import os
import random
import torch
from tqdm import tqdm
import numpy as np

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X

def trainer(train_loader, val_loader, model, criterion, optimizer, config, device):
    best_acc = 0.0
    early_stop_epochs = config["early_stop_epochs"]
    last_improve_epochs = 0
    for epoch in range(config["num_epoch"]):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        train_item = 0
        val_item = 0
        
        # training
        model.train() # set the model to training mode
        for _, batch in enumerate(tqdm(train_loader)):
            # handle x, y
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(features)
            if criterion == "crf":
                loss = -model.calc_crf_prob(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            # backward pass
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            if criterion == "crf":
                train_pred = model.crf_decode(outputs)
            else:
                _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.cpu().detach() == labels.cpu().detach()).sum().item()
            train_item += len(labels)
            train_loss += loss.cpu().detach().item()
        
        # validation
        if val_loader:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for _, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    
                    if criterion == "crf":
                        loss = -model.calc_crf_prob(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
                    
                    if criterion == "crf":
                        val_pred = model.crf_decode(outputs)
                    else:
                        _, val_pred = torch.max(outputs, 1) 
                    
                    val_acc += (val_pred.cpu() == labels.cpu()).detach().sum().item() # get the index of the class with the highest probability
                    val_item += len(labels)
                    val_loss += loss.cpu().detach().item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, config["num_epoch"], train_acc / train_item, train_loss / len(train_loader), val_acc / val_item, val_loss / len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), "./models/{}.pth".format(config["model_name"]))
                    print('saving model with acc {:.3f}'.format(best_acc / val_item))
                    last_improve_epochs = 0
                else:
                    last_improve_epochs += 1
                    print(f"{last_improve_epochs} epochs since last improvement...")
                    if last_improve_epochs > early_stop_epochs:
                        break
        
        
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, config["num_epoch"], train_acc / train_item, train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if not val_loader:
        torch.save(model.state_dict(), "./models/ax_results/{}.pth".format(config["model_name"]))
        print('saving model at last epoch')
    
    return best_acc / val_item if val_loader else train_acc / train_item

def tester(test_loader, model, device):
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
    
    return pred

def save_performance(perf, model_name):
    file_path = "./evaluations/{}.txt".format(model_name)
    with open(file_path, mode="wt") as f:
        f.write(str(perf))

def save_pred(pred, model_name):
    file_path = "./preds/{}.csv".format(model_name)
    with open(file_path, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))

"""
    Adapted From TA's Sample Code
"""
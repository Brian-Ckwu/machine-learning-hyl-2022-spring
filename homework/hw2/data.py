import random
import torch
import os

from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Dict

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

# return (feats, labels) of a single audio sequence
class LibriRNNDataset(Dataset):
    def __init__(self, audio_list_path: str = "./libriphone", audio_feat_path: str = "./libriphone/feat", mode: str = "train", train_ratio: float = 0.80, train_val_seed: int = 1337):
        # store variables
        assert mode in ["train", "val", "test"]
        self.mode = mode
        # read audio fnames & labels
        audio_list = self.read_audio_list(audio_list_path, mode)
        if mode != "test":
            label_dict = self.read_audio_labels(audio_list_path)
        # make train / val split
        if mode != "test":
            random.seed(train_val_seed)
            random.shuffle(audio_list)
            split_idx = int(len(audio_list) * train_ratio)
            audio_list = audio_list[:split_idx] if (mode == "train") else audio_list[split_idx:]
        # make X, Y
        self.x = list()
        self.y = list()
        for fname in tqdm(audio_list):
            feats = torch.load(os.path.join(audio_feat_path, "train" if mode != "test" else "test", f"{fname}.pt"))
            self.x.append(feats)
            if mode != "test":
                labels = torch.LongTensor(label_dict[fname])
                self.y.append(labels)
        if self.y:
            assert len(self.x) == len(self.y)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.mode != "test":
            return self.x[idx], self.y[idx]
        return self.x[idx]

    @staticmethod
    def read_audio_list(audio_list_path: str, mode: str) -> List[str]:
        if (mode == "train") or (mode == "val"):
            with open(f"{audio_list_path}/train_split.txt") as f:
                audio_list = f.read().splitlines()
        elif (mode == "test"):
            with open(f"{audio_list_path}/test_split.txt") as f:
                audio_list = f.read().splitlines()
        else:
            raise ValueError("'mode' must by train, val, or test")
        
        return audio_list

    @staticmethod
    def read_audio_labels(audio_list_path: str) -> Dict[str, List[int]]:
        label_dict = dict()
        with open(f"{audio_list_path}/train_labels.txt") as f:
            for line in f:
                line = line.rstrip().split()
                label_dict[line[0]] = [int(l) for l in line[1:]]
        return label_dict

class BatchCollator(object):
    def __init__(self, concat_nframes, feature_dim, test_mode: bool = False):
        self.nframes = concat_nframes
        self.dim = feature_dim
        self.test_mode = test_mode
        # get positional encoding
        self.pe = torch.FloatTensor()
        interval = (1 - (-1)) / (self.nframes - 1)
        for frame in range(self.nframes):
            normalized_idx = -1 + frame * interval
            frame_pe = torch.ones(size=(self.dim,)) * normalized_idx
            self.pe = torch.cat(tensors=(self.pe, frame_pe))
    
    def __call__(self, batch):
        assert len(self.pe) == len(batch[0][0])
        batch_x = list()
        batch_y = list()
        for x, y in batch:
            x = x + self.pe
            batch_x.append(x)
            batch_y.append(y.cpu().detach().item())
        
        batch_x = torch.stack(tensors=batch_x)
        batch_y = torch.LongTensor(batch_y)
        return batch_x, batch_y
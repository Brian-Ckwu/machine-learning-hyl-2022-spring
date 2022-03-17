from multiprocessing.sharedctypes import Value
import random
import torch
import os

from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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
    def __init__(self, audio_list_path: str = "./libriphone", audio_feat_path: str = "./libriphone/feat", concat_nframes: int = 11, mode: str = "train", train_ratio: float = 0.80, train_val_seed: int = 1337):
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
        # TODO: concat feats -> not effective
        # self.x = self.concat_feats(self.x, concat_nframes)
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

    @staticmethod
    def concat_feats(audios, concat_nframes: int) -> list:
        # utility functions
        def get_left(audio, idx, half_window):
            left_idx = idx - half_window
            if left_idx < 0:
                padding_feats = audio[0].repeat(-left_idx)
                left_feats = torch.cat(tensors=(padding_feats, audio[:idx].flatten()))
            else:
                left_feats = audio[left_idx:idx].flatten()
            return left_feats
        
        def get_right(audio, idx, half_window):
            last_idx = len(audio) - 1
            right_idx = idx + half_window
            if right_idx > last_idx:
                padding_feats = audio[last_idx].repeat(right_idx - last_idx)
                right_feats = torch.cat(tensors=(padding_feats, audio[idx:].flatten()))
            else:
                right_feats = audio[idx:right_idx + 1].flatten()
            return right_feats        

        assert concat_nframes % 2 == 1
        half_window = concat_nframes // 2

        converted_audios = list()
        for audio in tqdm(audios):
            converted_audio = list()
            for idx, frame in enumerate(audio):
                left_frames = get_left(audio, idx, half_window)
                right_frames = get_right(audio, idx, half_window)
                frames = torch.cat(tensors=(left_frames, right_frames))
                converted_audio.append(frames)
            converted_audio = torch.stack(converted_audio, dim=0)
            converted_audios.append(converted_audio)
        
        return converted_audios

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

class RNNCollator(object):
    def __init__(self, mode):
        self.mode = mode
    
    def __call__(self, batch):
        if self.mode == "train":
            X = list()
            Y = list()
            seq_lens = list()
            for x, y in batch:
                X.append(x)
                Y.append(y)
                seq_lens.append(len(x))
            Y = pad_sequence(Y, batch_first=True)
            Y = pack_padded_sequence(Y, seq_lens, batch_first=True, enforce_sorted=False).data
        elif self.mode == "test":
            X = list()
            seq_lens = list()
            for x in batch:
                X.append(x)
                seq_lens.append(len(x))
        else:
            raise ValueError("mode must be train or test")

        X = pad_sequence(X, batch_first=True)
        X = pack_padded_sequence(X, seq_lens, batch_first=True, enforce_sorted=False)
        return (X, Y) if (self.mode != "test") else X
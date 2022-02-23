import torch
from torch.utils.data import Dataset

class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None, normalize_feats=True):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

        # Feature normalization
        if normalize_feats:
            self.x[:, 37:] = \
                (self.x[:, 37:] - self.x[:, 37:].mean(dim=0, keepdim=True)) / \
                (self.x[:, 37:].std(dim=0, keepdim=True))

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

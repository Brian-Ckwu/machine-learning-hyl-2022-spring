import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout_p=0):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, dropout_p),
            *[BasicBlock(hidden_dim, hidden_dim, dropout_p) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class RNNClassifier(nn.Module):
    def __init__(self, rnn_type: str, rnn_args: dict, classifier_dropout: float, output_dim: int = 41):
        super().__init__()
        D = 2 if rnn_args["bidirectional"] else 1
        self.rnn = getattr(torch.nn, rnn_type)(**rnn_args)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout), 
            nn.Linear(rnn_args["hidden_size"] * D, rnn_args["hidden_size"] * D),
            nn.PReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(rnn_args["hidden_size"] * D, output_dim)
        )

    def forward(self, x: PackedSequence):
        final_h = self.rnn(x)[0].data
        scores = self.classifier(final_h)
        return scores
import random
from typing import Callable, List, Tuple
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def same_seeds(myseed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(myseed) # TODO: check if it's necessary
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def evaluate(model: nn.Module, loader: DataLoader, criterion: Callable, device: str) -> Tuple[float, float]:
    # Trackers
    total_correct = 0
    total_loss = 0.0
    # Evaluation
    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(imgs)
            ncorrect = (preds.argmax(dim=-1) == labels).cpu().detach().sum().item()
            loss = criterion(preds, labels).cpu().detach().item()
        total_correct += ncorrect
        total_loss += loss * len(labels)
    # Calculate mean accuracy & loss
    acc = total_correct / len(loader.dataset)
    mean_loss = total_loss / len(loader.dataset)
    return acc, mean_loss

def render_exp_name(args: Namespace, fields: List[str]):
    spans = list()
    for field_key in fields:
        field_value = getattr(args, field_key)
        span = f"{field_key}-{str(field_value)}"
        spans.append(span)
    return '_'.join(spans)
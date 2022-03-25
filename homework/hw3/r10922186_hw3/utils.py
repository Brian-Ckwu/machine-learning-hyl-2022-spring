import json
import random
from pathlib import Path
from typing import Callable, List, Tuple
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class TestTimeAugmentPred(object):
    def __init__(self, tfm: Callable, k: int, weight: float):
        self.tfm = tfm
        self.k = k
        self.weight = weight
    
    def __call__(self, model: nn.Module, imgs_tensor: torch.tensor) -> torch.tensor: # return a averaged tensor of multiple transforms
        return

def same_seeds(myseed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(myseed) # TODO: check if it's necessary
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def evaluate(model: nn.Module, loader: DataLoader, criterion: Callable, device: str, test_time_augment_pred: Callable = None) -> Tuple[float, float]:
    # Trackers
    total_correct = 0
    total_loss = 0.0
    # Evaluation
    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(imgs)
            if test_time_augment_pred:
                augment_pred = test_time_augment_pred(model, imgs)
                assert preds.shape == augment_pred.shape
                preds = preds + augment_pred
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
        fv_string = str(field_value) if field_value != 0 else "0.0"
        span = f"{field_key}-{str(fv_string)}"
        spans.append(span)
    return '_'.join(spans)

def load_config(config_path: str = "./config.json") -> dict:
    with open(config_path) as f:
        config = json.load(f)
    return config

def parse_config_to_args(config: dict, exp_fields: List[str]) -> Namespace:
    args = Namespace(**config)
    args.model_dir = Path(args.model_dir)
    args.exp_name = render_exp_name(args, fields=exp_fields)
    args.save_dir = args.model_dir / args.exp_name
    return args
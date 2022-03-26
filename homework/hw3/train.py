"""
    Import Packages
"""
# Import necessary packages.
import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from typing import List, Dict, Tuple
from PIL import Image
from pathlib import Path
from argparse import Namespace
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm

# Custom codes
from utils import same_seeds, evaluate, render_exp_name, load_config, parse_config_to_args
from data import FoodDataset, test_tfm, tfm_mapping
from model import model_mapping

DATA_DIR = Path("./food11")

TRAIN = "training"
VALID = "validation"

def trainer(args: Namespace):
    print(f"Training args:\n {args}")
    start_time = time.time()
    # Set up for training
    same_seeds(args.myseed) # reproducibility
    # Data
    train_tfm = tfm_mapping[args.tfm]
    print(f"Using training transforms:\n{train_tfm}")
    train_set = FoodDataset(path=DATA_DIR / TRAIN, tfm=train_tfm)
    valid_set = FoodDataset(path=DATA_DIR / VALID, tfm=test_tfm)

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.bs, shuffle=True, pin_memory=True)  # TODO: check if shuffle=False leads to the same validation accuracy
    
    # Model, Loss, Optimizer, and Scheduler
    model: nn.Module = model_mapping.get(args.model, None)().to(args.device)
    if args.trained_model:
        model.load_state_dict(state_dict=torch.load(args.model_dir / args.trained_model / "model.pth"))
        print(f"Loaded previously trained model: {args.trained_model}")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.wd)    

    # Optimization
    stale = 0
    best_acc = 0
    record = {
        "train_acc": list(), "train_loss": list(),
        "valid_acc": list(), "valid_loss": list()
    }
    for epoch in range(args.nepochs):
        model.train()
        for imgs, labels in tqdm(train_loader, desc="Batch"):
            # Move to the same device
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            # Forward the data. (Make sure data and model are on the same device.)
            preds = model(imgs)
            loss = criterion(preds, labels)

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) # TODO: Check the meaning of this line. (Clip the gradient norms for stable training.)
            optimizer.step()

        # Evaluate training / validation accuracy & loss
        print(f"Evaluating model at epoch {epoch + 1}...")
        train_acc, train_loss = evaluate(model, train_loader, criterion, args.device)
        valid_acc, valid_loss = evaluate(model, valid_loader, criterion, args.device)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{args.nepochs:03d} ] loss = {train_loss:.4f}, acc = {train_acc:.4f}")
        print(f"[ Valid | {epoch + 1:03d}/{args.nepochs:03d} ] loss = {valid_loss:.4f}, acc = {valid_acc:.4f}")

        # TODO: update logs
        record = update_record(record, items=[
            ("train_acc", train_acc), ("train_loss", train_loss),
            ("valid_acc", valid_acc), ("valid_loss", valid_loss)
        ])

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), args.save_dir / "model.pth") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            print(f"No improvment for {stale} consecutive epochs.")
            if stale > args.patience:
                print(f"Out of patience. Stop training.")
                break
        
        # write train record each epoch
        write_train_record(record, best_acc, save_dir=args.save_dir)

    end_time = time.time()
    train_minutes = (end_time - start_time) / 60
    print(f"Total training time: {train_minutes:.2f} minutes")
    return record, best_acc

def update_record(record: Dict[str, List], items: List[Tuple[str, float]]):
    for key, value in items:
        record[key].append(value)
    return record

def write_train_record(record: Dict[str, list], best_acc: float, save_dir: Path) -> None:
    # write record
    df = pd.DataFrame(record)
    df.to_csv(save_dir / "record.csv", index=False)
    # write best acc
    with open(save_dir / "best_acc.txt", mode="wt") as f:
        f.write(str(best_acc))

def save_config(config: dict, save_dir: Path) -> None:
    with open(save_dir / "config.json", mode="wt") as f:
        json.dump(config, f)

if __name__ == "__main__":
    # Configuration
    config = load_config(config_path="./config.json")
    args = parse_config_to_args(config, exp_fields=["model", "optimizer", "lr", "wd", "bs", "nepochs", "tfm"])
    args.save_dir.mkdir(parents=True, exist_ok=True) # for saving the model & train record
    save_config(config, args.save_dir)
    # Start training
    if "cuda" in args.device:
        assert torch.cuda.is_available()
    record, best_acc = trainer(args)
    write_train_record(record, best_acc, args.save_dir)
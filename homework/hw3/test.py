from typing import List
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FoodDataset, test_tfm
from model import model_mapping
from utils import render_exp_name, load_config, parse_config_to_args

DATA_DIR = Path("./food11")
TEST = "test"

def tester(args: Namespace):
    print(f"Testing args:\n{args}")
    # Data
    test_set = FoodDataset(path=DATA_DIR / TEST, tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, pin_memory=True)

    # Model
    model_best = model_mapping.get(args.model, None)().to(args.device)
    model_best.load_state_dict(torch.load(args.save_dir / "model.pth"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data,_ in test_loader:
            test_pred = model_best(data.to(args.device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()
    
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
    df["Category"] = prediction
    df.to_csv("./preds/{}.csv".format(args.exp_name),index = False)

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

if __name__ == "__main__":
    # Configuration
    config = load_config(config_path="/nfs/nas-7.1/ckwu/machine-learning-hyl-2022-spring/homework/hw3/models/model-SampleClassifier_optimizer-Adam_lr-0.001_wd-0.0_bs-256_nepochs-500_tfm-AfCropHoriPersChoice/config.json")
    args = parse_config_to_args(config, exp_fields=["model", "optimizer", "lr", "wd", "bs", "nepochs", "tfm"])
    # Start training
    if "cuda" in args.device:
        assert torch.cuda.is_available()
    tester(args)


import json
from tqdm import tqdm
from typing import List
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn

from data import get_dataloader
from model import model_mapping
from utils import set_seed, get_cosine_schedule_with_warmup, load_config, render_exp_name

def trainer(
        args: Namespace,
        hparams: List[str] = ["model", "din", "dfc", "nhead", "dropout", "nlayers", "optimizer", "lr", "bs"]
    ):
    # Configs
    args.exp_name = render_exp_name(args, hparams)
    args.ckpt_dir = Path(args.model_dir) / args.exp_name
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed)
    print(f"[Info]: Use {args.device} now!")

    # Data
    train_loader, valid_loader, num_speakers = get_dataloader(
        data_dir=args.data_dir,
        segment_len=args.segment_len,
        train_ratio=args.tratio,
        batch_size=args.bs,
        num_workers=args.nworkers
    )
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!")

    # Model, loss, optimizer, and scheduler
    model = model_mapping[args.model](args, n_spks=num_speakers).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps)
    print(f"[Info]: Finish creating model!")    

    # Optimization
    stale = 0 # for early stopping
    best_acc = -1.0
    pbar = tqdm(total=args.valid_steps, ncols=0, desc="Train", unit=" step")
    train_log = {
        "train": {"acc": list(), "loss": list()},
        "valid": {"acc": list(), "loss": list()}
    }
    for step in range(args.total_steps):
        model.train()
        # get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # move to device
        x, lens, y = batch
        x, lens, y = x.to(args.device), lens.to(args.device), y.to(args.device)

        # forward
        if "Conformer" in model.__class__.__name__:
            scores = model(x, lens)
        else:
            scores = model(x)
        loss = criterion(scores, y)

        # back-prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # calculate loss and accuracy
        batch_acc = (scores.argmax(dim=-1) == y).float().mean().detach().cpu().item()
        batch_loss = loss.cpu().detach().item()
        train_log["train"]["acc"].append(batch_acc)
        train_log["train"]["loss"].append(batch_loss)

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_acc:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % args.valid_steps == 0:
            pbar.close()
            valid_acc, valid_loss = evaluate(valid_loader, model, criterion, args.device)
            train_log["valid"]["acc"].append(valid_acc)
            train_log["valid"]["loss"].append(valid_loss)

            # keep the best model
            if valid_acc > best_acc:
                stale = 0
                best_acc = valid_acc
                # Save the best model so far.
                torch.save(model.state_dict(), args.ckpt_dir / "model.pth")
                pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_acc:.4f})")
            else:
                stale += 1
                print(f"No improvement for {stale} epochs.")
                if stale >= 10:
                    print("Early stopping.")
                    break

            pbar = tqdm(total=args.valid_steps, ncols=0, desc="Train", unit=" step")

    pbar.close()
    # save best_acc & train_log
    (args.ckpt_dir / "best_acc.txt").write_text(str(best_acc))
    (args.ckpt_dir / "train_log.json").write_text(json.dumps(train_log))
    # save config for testing
    config_path = (args.ckpt_dir / "config.json")
    args.ckpt_dir = str(args.ckpt_dir)
    config_path.write_text(json.dumps(vars(args)))
    return best_acc, train_log

def evaluate(loader, model, criterion, device) -> tuple:
    model.eval()
    total_correct = 0
    total_loss = 0

    pbar = tqdm(total=len(loader.dataset), ncols=0, desc="Valid", unit=" uttr")
    for x, lens, y in loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        with torch.no_grad():
            if "Conformer" in model.__class__.__name__:
                scores = model(x, lens)
            else:
                scores = model(x)
            correct = (scores.argmax(dim=-1) == y).sum().detach().cpu().item()
            loss = criterion(scores, y)

            total_correct += correct
            total_loss += loss.detach().cpu().item() * len(y)
        
        pbar.update(len(y))
        pbar.set_postfix(
			loss=f"{loss:.2f}",
			accuracy=f"{correct / len(y):.2f}",
		)

    valid_acc = total_correct / len(loader.dataset)
    valid_loss = total_loss / len(loader.dataset)
    pbar.set_postfix(
			loss=f"{valid_loss:.2f}",
			accuracy=f"{valid_acc:.2f}",
		)
    return valid_acc, valid_loss

if __name__ == "__main__":
    args = Namespace(**load_config(Path("./config.json")))
    trainer(args, hparams=["model", "din", "dfc", "nhead", "kernelsize", "dropout", "nlayers", "optimizer", "lr", "bs", "segment_len", "eps", "wd"])


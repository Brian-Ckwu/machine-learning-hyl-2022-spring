import time
import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data import source_transform, target_transform
from model import FeatureExtractor, LabelPredictor, DomainClassifier
from utils import load_json

tanh = nn.Tanh()

def main(args: Namespace):
    # Configuration
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    source_dataset = ImageFolder(f'{args.data_dir}/train_data', transform=source_transform)
    target_dataset = ImageFolder(f'{args.data_dir}/test_data', transform=target_transform)
    
    source_dataloader = DataLoader(source_dataset, batch_size=args.bs, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=args.bs, shuffle=True) 
    print("Data loaded.")   
    
    # Model, Loss, and Optimizer
    feature_extractor = FeatureExtractor().to(args.device)
    label_predictor = LabelPredictor().to(args.device)
    domain_classifier = DomainClassifier().to(args.device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_P = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())
    print("Model prepared.")

    # Optimization
    log_fields = ["epoch_D_loss", "epoch_P_loss", "epoch_F_loss", "label_acc", "domain_acc"]
    train_log = {field: list() for field in log_fields}

    print("Start training...")
    for epoch in range(args.nepochs):
        # tracking variables
        running_D_loss, running_P_loss, running_F_loss = 0.0, 0.0, 0.0
        label_hit, label_num = 0, 0
        domain_hit, domain_num = 0, 0

        # train one epoch
        feature_extractor.train()
        label_predictor.train()
        domain_classifier.train()

        # TODO: adaptive lambda
        lamb = (tanh(torch.tensor(epoch / 500)) * args.max_lambda).cpu().item()

        for i, ((source_data, source_labels), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
            # move to device
            source_data = source_data.to(args.device)
            source_labels = source_labels.to(args.device)
            target_data = target_data.to(args.device)

            # mix source and target data to forward at the same pass
            mixed_data = torch.cat(tensors=[source_data, target_data], dim=0)
            domain_labels = torch.zeros(size=[source_data.shape[0] + target_data.shape[0], 1]).to(args.device)
            # set domain label of source data to be 1.
            domain_labels[:source_data.shape[0]] = 1

            # Step 1: train domain classifier
            # forward pass
            features = feature_extractor(mixed_data)
            domain_logits = domain_classifier(features.detach()) # We don't need to train feature extractor in step 1. # Thus we detach the feature neuron to avoid backpropgation.
            # loss
            D_loss = domain_criterion(domain_logits, domain_labels)
            running_D_loss += D_loss.item()
            # accuracy
            domain_hit += ((torch.sigmoid(domain_logits) >= 0.5) == domain_labels).sum().item()
            domain_num += mixed_data.shape[0]
            # back-propagation & update
            D_loss.backward()
            optimizer_D.step()

            # Step 2 : train feature extractor and label classifier
            # forward pass
            class_logits = label_predictor(features[:source_data.shape[0]])
            domain_logits = domain_classifier(features)
            # loss
            P_loss = class_criterion(class_logits, source_labels)
            D_loss = domain_criterion(domain_logits, domain_labels)
            F_loss = P_loss - lamb * D_loss
            running_P_loss += P_loss.item()
            running_F_loss += F_loss.item()
            # accuracy
            label_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_labels).item()
            label_num += source_data.shape[0]
            # back-propagation & update
            F_loss.backward()
            optimizer_F.step()
            optimizer_P.step()

            # Reset gradients to zero
            optimizer_P.zero_grad()
            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
        
        # Record and log training info
        epoch_D_loss = running_D_loss / (i + 1)
        epoch_P_loss = running_P_loss / (i + 1)
        epoch_F_loss = running_F_loss / (i + 1)
        label_acc = label_hit / label_num
        domain_acc = domain_hit / domain_num
        log_infos = [epoch_D_loss, epoch_P_loss, epoch_F_loss, label_acc, domain_acc]
        for key, value in zip(log_fields, log_infos):
            train_log[key].append(value)
        Path(f"{args.save_dir}/train_log.json").write_text(json.dumps(train_log))
        print(f"Training epoch {epoch + 1:04d} (lambda = {lamb:.4f}) | domain loss: {epoch_D_loss:6.4f}; predict loss: {epoch_P_loss:6.4f}; feature loss: {epoch_F_loss:6.4f}; label acc: {label_acc:.4f}; domain acc: {domain_acc:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            print(f"Saving models at epoch {epoch + 1}...")
            model_save_path = Path(f"{args.save_dir}/{epoch + 1}")
            model_save_path.mkdir(parents=True, exist_ok=True)
            for model, name in zip([feature_extractor, label_predictor, domain_classifier], ["feature_extractor", "label_predictor", "domain_classifier"]):
                torch.save(model.state_dict(), model_save_path / f"{name}.pth")

        feature_extractor.train()
        label_predictor.train()
        domain_classifier.train()

if __name__ == "__main__":
    config = load_json("./config.json")
    args = Namespace(**config)
    start_time = time.time()
    main(args)
    end_time = time.time()
    train_time = (end_time - start_time) / 60
    print(f"Total training time: {train_time:.2f} minutes ({args.nepochs} epochs)")
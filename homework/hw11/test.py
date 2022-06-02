from tqdm import tqdm
from argparse import Namespace

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from data import target_transform
from model import FeatureExtractor, LabelPredictor
from utils import load_json

def main(args: Namespace):
    # Data
    target_dataset = ImageFolder(f'{args.data_dir}/test_data', transform=target_transform)
    test_dataloader = DataLoader(target_dataset, batch_size=args.test_bs, shuffle=False)

    # Model
    feature_extractor = FeatureExtractor().to(args.device)
    label_predictor = LabelPredictor().to(args.device)    

    label_predictor.load_state_dict(torch.load(f"{args.save_dir}/{args.ckpt_epoch}/label_predictor.pth"))
    feature_extractor.load_state_dict(torch.load(f"{args.save_dir}/{args.ckpt_epoch}/feature_extractor.pth"))

    result = []
    label_predictor.eval()
    feature_extractor.eval()
    for test_data, _ in tqdm(test_dataloader):
        test_data = test_data.to(args.device)

        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    import pandas as pd
    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(f"{args.save_dir}/{args.ckpt_epoch}/prediction.csv", index=False)

if __name__ == "__main__":
    config = load_json("./config.json")
    args = Namespace(**config)

    for epoch in [str(i * 250) for i in range(1, 11)]:
        args.ckpt_epoch = epoch
        print(f"Predicting from model checkpoint at {args.save_dir}/{args.ckpt_epoch}...")
        main(args)

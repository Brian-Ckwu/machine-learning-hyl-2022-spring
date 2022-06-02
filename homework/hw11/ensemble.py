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

    # Models
    feature_extractors = list()
    label_predictors = list()
    for ensemble_dir in args.ensemble_dirs:
        print(f"Loading model checkpoint at {ensemble_dir}")

        feature_extractor = FeatureExtractor().to(args.device)
        label_predictor = LabelPredictor().to(args.device)    

        label_predictor.load_state_dict(torch.load(f"{ensemble_dir}/label_predictor.pth"))
        feature_extractor.load_state_dict(torch.load(f"{ensemble_dir}/feature_extractor.pth"))
        
        label_predictors.append(label_predictor)
        feature_extractors.append(feature_extractor)

        del feature_extractor, label_predictor

    result = []
    print(f"Ensembling with {len(feature_extractors)} and {len(label_predictors)}...")
    for test_data, _ in tqdm(test_dataloader):
        test_data = test_data.to(args.device)
        class_logits = torch.zeros(size=(test_data.shape[0], 10)).to(args.device)
        for label_predictor, feature_extractor in zip(label_predictors, feature_extractors):
            label_predictor.eval()
            feature_extractor.eval()
            class_logits += label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    import pandas as pd
    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(args.ensemble_save_path, index=False)

if __name__ == "__main__":
    config = load_json("./config.json")
    args = Namespace(**config)

    print(f"Predicting from ensemble model checkpoints {args.ensemble_dirs}...")
    main(args)

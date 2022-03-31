import csv
import sys
import json
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from data import InferenceDataset, inference_collate_batch
from model import model_mapping
from utils import load_config

def tester(args: Namespace):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(args.data_dir) / "mapping.json"
	mapping = json.loads(mapping_path.read_bytes())

	dataset = InferenceDataset(args.data_dir, segment_len=args.segment_len)
	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		drop_last=False,
		num_workers=4,
		collate_fn=inference_collate_batch,
	)
	print(f"[Info]: Finish loading data!")

	speaker_num = len(mapping["id2speaker"])
	model = model_mapping[args.model](args, n_spks=speaker_num).to(args.device)
	model.load_state_dict(torch.load(Path(args.model_dir) / args.exp_name / "model.pth"))
	model.eval()
	print(f"[Info]: Finish creating model!")

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader):
		with torch.no_grad():
			mels = mels.to(args.device)
			outs = model(mels)
			preds = outs.argmax(dim=-1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(f'./preds/{args.exp_name}.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)

if __name__ == "__main__":
    args = Namespace(**load_config(Path(sys.argv[1])))
    tester(args)
import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
 
 
class VoxDataset(Dataset):
	def __init__(self, data_dir, segment_len=128):
		self.data_dir = data_dir
		self.segment_len = segment_len
	
		# Load the mapping from speaker neme to their corresponding id. 
		mapping_path = Path(data_dir) / "mapping.json"
		mapping = json.loads(mapping_path.read_bytes())
		self.speaker2id = mapping["speaker2id"]
	
		# Load metadata of training data.
		metadata_path = Path(data_dir) / "metadata.json"
		metadata = json.loads(metadata_path.read_bytes())["speakers"]
	
		# Get the total number of speaker.
		self.speaker_num = len(metadata.keys())
		self.data = []
		for speaker in metadata.keys():
			for utterances in metadata[speaker]:
				self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
	def __len__(self):
		return len(self.data)
 
	def __getitem__(self, index):
		feat_path, speaker = self.data[index]
		# Load preprocessed mel-spectrogram.
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		# Segmemt mel-spectrogram into "segment_len" frames.
		if len(mel) > self.segment_len:
			# Randomly get the starting point of the segment.
			start = random.randint(0, len(mel) - self.segment_len)
			# Get a segment with "segment_len" frames.
			mel = torch.FloatTensor(mel[start:start+self.segment_len])
		else:
			mel = torch.FloatTensor(mel)
		# Turn the speaker id into long for computing loss later.
		speaker = torch.FloatTensor([speaker]).long()
		return mel, speaker
 
	def get_speaker_number(self):
		return self.speaker_num
	
	def collate_fn(batch):
		# Process features within a batch.
		mel, speaker = zip(*batch)
		# Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
		mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
		# mel: (batch size, length, 40)
		return mel, torch.FloatTensor(speaker).long()

def get_dataloader(data_dir, train_ratio, batch_size):
	"""Generate dataloader"""
	dataset = VoxDataset(data_dir)
	speaker_num = dataset.get_speaker_number()
	# Split dataset into training dataset and validation dataset
	trainlen = int(train_ratio * len(dataset))
	lengths = [trainlen, len(dataset) - trainlen]
	trainset, validset = random_split(dataset, lengths)

	train_loader = DataLoader(
		trainset,
		batch_size=batch_size,
		shuffle=True,
		pin_memory=True,
		collate_fn=trainset.collate_fn,
	)
	valid_loader = DataLoader(
		validset,
		batch_size=batch_size,
		shuffle=False, # TODO: check if drop_last improves performance
		pin_memory=True,
		collate_fn=trainset.collate_fn,
	)

	return train_loader, valid_loader, speaker_num
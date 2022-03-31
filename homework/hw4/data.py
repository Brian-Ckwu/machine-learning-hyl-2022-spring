import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
 
 
class VoxDataset(Dataset):
	def __init__(self, data_dir, segment_len: int = 128):
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
			mel = torch.FloatTensor(mel[start:start + self.segment_len])
		else:
			mel = torch.FloatTensor(mel)
		# Turn the speaker id into long for computing loss later.
		speaker = torch.FloatTensor([speaker]).long()
		return mel, speaker
 
	def get_speaker_number(self):
		return self.speaker_num

class InferenceDataset(Dataset):
	def __init__(self, data_dir, random: bool = False, segment_len: int = 128):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.loads(testdata_path.read_bytes())
		self.data_dir = data_dir
		self.data = metadata["utterances"]
		self.random = random
		self.segment_len = segment_len
		# fixed values
		self.mel_dim = 40
		self.pad_value = -20

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		if self.random:
			if len(mel) > self.segment_len:
				start = random.randint(0, len(mel) - self.segment_len)
				mel = torch.FloatTensor(mel[start: start + self.segment_len])
			else:
				padding = torch.ones(size=(self.segment_len - len(mel), self.mel_dim)) * self.pad_value
				mel = torch.cat(tensors=(torch.FloatTensor(mel), padding), dim=0)

		return feat_path, mel

def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)

	return feat_paths, torch.stack(mels)

def get_dataloader(data_dir, segment_len, train_ratio, batch_size, num_workers: int = 4):
	"""Generate dataloader"""
	def collate_fn(batch):
		# Process features within a batch.
		mel, speaker = zip(*batch)
		# Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
		mel = pad_sequence(mel, batch_first=True, padding_value=-20) # pad log 10^(-20) which is very small value.
		# mel: (batch size, length, 40)
		return mel, torch.FloatTensor(speaker).long()

	dataset = VoxDataset(data_dir, segment_len=segment_len)
	print(f"[Info]: Training instance shape = {dataset[0][0].shape}")
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
		num_workers=num_workers,
		collate_fn=collate_fn
	)
	valid_loader = DataLoader(
		validset,
		batch_size=batch_size,
		shuffle=False, # TODO: check if drop_last improves performance
		pin_memory=True,
		num_workers=num_workers,
		collate_fn=collate_fn
	)

	return train_loader, valid_loader, speaker_num
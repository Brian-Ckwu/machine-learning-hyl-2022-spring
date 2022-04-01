from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from conformer import Conformer

class SampleClassifier(nn.Module):
	def __init__(self, args: Namespace, n_spks=600):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, args.din)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=args.din, 
            dim_feedforward=args.dfc, 
            nhead=args.nhead, 
            dropout=args.dropout,
            batch_first=True
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.nlayers)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(args.din, args.din),
			nn.ReLU(),
			nn.Linear(args.din, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# The encoder layer
		out = self.encoder(out)
		# mean pooling
		stats = out.mean(dim=1)
		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out

class SampleClassifierOneFC(nn.Module):
	def __init__(self, args: Namespace, n_spks=600):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, args.din)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=args.din, 
            dim_feedforward=args.dfc, 
            nhead=args.nhead, 
            dropout=args.dropout,
            batch_first=True
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.nlayers)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.ReLU(),
			nn.Linear(args.din, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# The encoder layer
		out = self.encoder(out)
		# mean pooling
		stats = out.mean(dim=1)
		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out

class SampleClassifierAttnPool(nn.Module):
	def __init__(self, args: Namespace, n_spks=600):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, args.din)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=args.din, 
            dim_feedforward=args.dfc, 
            nhead=args.nhead, 
            dropout=args.dropout,
            batch_first=True
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.nlayers)

		self.attn = nn.Sequential(
			nn.Linear(args.din, 1),
			nn.Softmax(dim=1)
		)
		
		self.pred_layer = nn.Sequential(
			# nn.ReLU(), # TODO: check if relu improve performance
			nn.Linear(args.din, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		for_encode = self.prenet(mels)
		# The encoder layer
		encoded = self.encoder(for_encode)
		# self-attention pooling
		attn_weights = self.attn(encoded)
		pooled = torch.bmm(
			attn_weights.transpose(1, 2),
			encoded
		).squeeze(dim=1)
		# out: (batch, n_spks)
		out = self.pred_layer(pooled)
		return out

class ConformerAttnPool(nn.Module):
	def __init__(self, args: Namespace, n_spks=600):
		super(ConformerAttnPool, self).__init__()
		# Project dimension
		MEL_DIM = 40
		self.prenet = nn.Linear(MEL_DIM, args.din) # TODO: is this necessary?
		# Conformer encoder
		self.conformer = Conformer(
			input_dim=args.din,
			num_heads=args.nhead,
			ffn_dim=args.dfc,
			num_layers=args.nlayers,
			depthwise_conv_kernel_size=args.kernelsize,
			dropout=args.dropout
		)
		# Attention Pooling
		self.attn = nn.Sequential(
			nn.Linear(args.din, 1),
			nn.Softmax(dim=1)
		)
		# Classifier
		self.pred_layer = nn.Linear(args.din, n_spks)
	
	def forward(self, mels, lens):
		for_conform = self.prenet(mels)
		conformed, _lens = self.conformer(for_conform, lens)
		assert torch.all(_lens == lens)
		attn_weights = self.attn(conformed)
		pooled = torch.bmm(
			attn_weights.transpose(1, 2),
			conformed
		).squeeze(dim=1)
		scores = self.pred_layer(pooled)
		return scores

model_mapping = {
	"SampleClassifier": SampleClassifier,
	"SampleClassifierOneFC": SampleClassifierOneFC,
	"SampleClassifierAttnPool": SampleClassifierAttnPool,
	"ConformerAttnPool": ConformerAttnPool
}
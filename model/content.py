import os
import json
from collections import defaultdict
import random
from gensim.models import FastText
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.data import TabularDataset
import pandas as pd
import fasttext
import math
import numpy as np


class Content(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.hsz = args.content_hsz
		self.device = args.device
		self.emb = fasttext.load_model('../../embeddings/cc.ko.300.bin')
		self.attn = MultiHeadAttention(4, 3 * self.hsz, dropout=0.1)
		# self.l2 = nn.Linear(3*self.hsz, self.hsz)
		self.l1 = nn.Linear(3 * self.hsz, 2)

	def forward(self, input):
		'''
		:param input: [batch_size, the number of triples, 3]
		:return: [batch_size, the number of triples, 2]
		'''
		out = []
		init = True
		for ins in input:
			triple_encode = []
			for t in ins:
				s = self.emb[t[0]]
				p = self.emb[t[1]]
				o = self.emb[t[2]]
				t_emb = np.concatenate((s, p, o))
				triple_encode.append(t_emb)
			triple_encode = torch.tensor(triple_encode).unsqueeze(0).to(self.device)
			# triple_encode = F.relu(self.l2(triple_encode))
			# prop = F.relu(self.l1(triple_encode)).squeeze()
			attn_score = self.attn(triple_encode, triple_encode, triple_encode).squeeze()
			# p = F.softmax(self.l1(attn_score))
			prop = F.relu(self.l1(attn_score))
			if init:
				out = prop
				init = False
			else:
				out = torch.cat([out,prop], dim=0)
			# out.append(self.l1(attn_score))
		return out


class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout=0.1):
		super().__init__()

		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads

		self.q_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)

	def forward(self, q, k, v, mask=None):
		bs = q.size(0)

		# perform linear operation and split into h heads

		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

		# transpose to get dimensions bs * h * sl * d_model

		k = k.transpose(1, 2)
		q = q.transpose(1, 2)
		v = v.transpose(1, 2)
		# calculate attention using function we will define next
		scores = attention(q, k, v, self.d_k, mask, self.dropout)

		# concatenate heads and put through final linear layer
		concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

		output = self.out(concat)

		return output





def attention(q, k, v, d_k, mask=None, dropout=None):
	scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
	scores = F.softmax(scores, dim=-1)

	if dropout is not None:
		scores = dropout(scores)

	output = torch.matmul(scores, v)
	return output
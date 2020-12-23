from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.data import TabularDataset, Field
import pandas as pd
import torch


class JosaDataset(TabularDataset):
	def __init__(self, datapath):
		text = Field(sequential=True, tokenize=tokenizer, batch_first=True, init_token="<sos>", eos_token="<eos>",
					 lower=True)
		josa = Field(sequential=True, tokenize=tokenizer, use_vocab=True)
		super().__init__(path=datapath, format='tsv', fields=[('text', text), ('josa', josa)])
		text.build_vocab(self)
		josa.build_vocab(self)


def tokenizer(sen):
	return sen.split()

def onehot(alist, vocab):
	"""
	TorchText의 Field에 사용할 one-hot Encoder 함수
	"""
	# alist = [2, 1, 1]
	# _tensor = [[1, 0, 0]]
	_tensor = torch.tensor(alist).data.sub_(1).unsqueeze(1)

	# _onehot = [[0, 0], [0, 0], [0, 0]]
	_onehot = torch.zeros((len(alist), len(vocab) - 1), dtype=torch.float)

	# _onehot = [[0, 1], [1, 0], [1, 0]]
	_onehot.scatter_(1, _tensor, 1)
	return _onehot
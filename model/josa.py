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


class Josa():
	def __init__(self, args):
		self.method = args.josa_model
		if self.method == 'ngram':
			self.model = ngram()
		return

	def train(self, data_path):
		self.model.train(data_path)
		return

	def predict(self, sentence):
		self.model.load_dict()
		output = self.model.predict(sentence)
		return output

	def eval(self, data_path):
		with open(data_path, 'r', encoding='utf-8') as f:
			test_json = json.load(f)
			correct = 0
			total = 0
			for unit in test_json:
				total += len(unit['josa'])
				predicted_sentence, preds = self.predict(unit['entity_masked_sentence'])
				for gold, pred in zip(unit['josa'], preds):
					if gold == pred:
						correct += 1
			return correct, total, correct / total * 100


class ngram():
	def __init__(self):
		self.josa_list = None
		self.statistic = None
		return

	def load_dict(self):
		self.statistic = json.load(open('./josa_info/3-gram.json', 'r', encoding='utf-8'))
		self.josa_list = json.load(open('./josa_info/list.json', 'r', encoding='utf-8'))
		return

	def train(self, data_path):
		train_list = json.load(open(data_path, 'r', encoding='utf-8'))
		statistic = dict()
		josa_list = []
		for unit in train_list:
			tokens = unit['entity_masked_sentence'].split() + ['<unk>']
			josa = unit['josa']
			j = 0
			for i, token in enumerate(tokens):
				if token == '[JOSA]':
					if tokens[i - 1] not in statistic.keys():
						statistic[tokens[i - 1]] = {"value": 0}
					stat_pre = statistic[tokens[i - 1]]
					stat_pre['value'] += 1
					if josa[j] not in stat_pre.keys():
						stat_pre[josa[j]] = {"value": 0}
					stat_cur = stat_pre[josa[j]]
					stat_cur['value'] += 1
					if tokens[i + 1] not in stat_cur.keys():
						stat_cur[tokens[i + 1]] = {"value": 0}
					stat_post = stat_cur[tokens[i + 1]]
					stat_post['value'] += 1
					# key = '-'.join([tokens[i - 1], josa[j], tokens[i + 1]])
					# statistic[key] += 1
					j += 1
			josa_list += josa
		josa_list = set(josa_list)
		with open('josa_info/list.json', 'w', encoding='utf-8') as f:
			json.dump(list(josa_list), f, ensure_ascii=False, indent='\t')
		with open('josa_info/3-gram.json', 'w', encoding='utf-8') as f:
			json.dump(statistic, f, ensure_ascii=False, indent='\t')

	def predict(self, masked_sentence):
		tokens = masked_sentence.split() + ['<unk>']
		token_list = []
		predicted_josa = []
		for i, token in enumerate(tokens):
			if token == '[JOSA]':
				value_3 = -1
				value_2 = -1
				predict_josa_3 = None
				predict_josa_2 = None
				predict_josa_1 = None
				# predicte_josa = None

				if tokens[i - 1] not in self.statistic.keys():
					predict_josa_1 = self.josa_list[random.randrange(0, len(self.josa_list))]
				else:
					stat_post = self.statistic[tokens[i - 1]]
					for josa, stat_josa in stat_post.items():
						if josa == 'value':
							if stat_josa > value_2:
								value_2 = stat_josa
								predict_josa_2 = josa
						else:
							for post_token, stat_post in stat_josa.items():
								if post_token != tokens[i + 1]:
									continue
								if stat_post['value'] > value_3:
									predict_josa_3 = josa
									value_3 = stat_post['value']
				if predict_josa_3 is not None:
					predict_josa = predict_josa_3
				elif predict_josa_2 is not None:
					predict_josa = predict_josa_2
				else:
					predict_josa = predict_josa_1
				token_list.append(predict_josa)
				predicted_josa.append(predict_josa)
			else:
				token_list.append(token)
		unmasked_sentence = ' '.join(token_list)
		return unmasked_sentence, predicted_josa


class NN(nn.Module):
	def __init__(self, esz, osz):
		super(NN, self).__init__()
		self.esz = esz
		self.osz = osz
		self.fc = nn.Linear(2*self.esz, self.osz)


	def forward(self, input, offset):
		# input data : (josa_count * 2 * 300)
		x = input.view(-1, 600)
		o = F.relu(self.fc(x))
		return o


class GRUNET(nn.Module):
	def __init__(self, args, osz):
		super(GRUNET, self).__init__()
		self.esz = args.josa_esz
		self.osz = osz
		self.hsz = args.josa_hsz
		self.lsz = args.josa_lsz
		self.device = args.device
		self.cell = nn.GRU(input_size=self.esz, hidden_size=self.hsz, num_layers=self.lsz, dropout=0.1, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(2*self.hsz, self.osz)

	def forward(self, input, offset):
		cell_out, _ = self.cell(input=input)
		eomi_outs = torch.zeros((cell_out.shape[0], cell_out.shape[2]), requires_grad=True).to(self.device)
		for n in range(cell_out.shape[0]):
			eomi_outs[n] = cell_out[n, offset[n], :]
		fc_out = self.fc(eomi_outs)
		output = F.relu(fc_out)
		# output = output.argmax(dim=2)
		return output


class GRUNGRAM(nn.Module):
	def __init__(self, args, osz):
		super(GRUNGRAM, self).__init__()
		self.esz = args.josa_esz
		self.osz = osz
		self.hsz = args.josa_hsz
		self.lsz = args.josa_lsz
		self.cell = nn.GRU(input_size=self.esz, hidden_size=self.hsz, num_layers=self.lsz, dropout=0.1, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(2*self.hsz, self.osz)

	def forward(self, input, offset):
		cell_out, _ = self.cell(input=input)
		fc_out = self.fc(cell_out)
		output = F.relu(fc_out)
		return output[:, 1, :].view(output.shape[0], output.shape[2])


if __name__ == '__main__':

	pass
# model = Josa('ngram')
# model.train('../data/josa/preprocessed/train.json')
# correct, total, accuracy = model.eval('../data/josa/preprocessed/test.json')
# print("Total: %d\nCorrect: %d\nAccuracy: %.2f" %(correct, total, accuracy))

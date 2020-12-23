from pargs import pargs
import os
from josa import NN, GRUNET
from dataset import JosaDataset
from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, RawField
import torch.optim as optim
import torch.nn as nn
import fasttext
import numpy as np
from tqdm import tqdm
import time
from utils import epoch_time
import torch.nn.functional as F

TEXT = Field(sequential=True, tokenize=lambda x: x.split(), batch_first=True, init_token="<sos>", eos_token="<eos>",
			 lower=True)
JOSA = Field(sequential=True, tokenize=lambda x: x.split(), use_vocab=True, init_token="<sos>", eos_token="<eos>")
OFFSET = RawField()


def main_josa(args):
	train_data, dev_data, test_data = TabularDataset.splits(path=args.datadir, train=args.josa_train_path,
															validation=args.josa_dev_path, test=args.josa_test_path,
															skip_header=True, format='tsv',
															fields=[('text', TEXT), ('josa', JOSA), ('offset', OFFSET)])
	word_embedding = fasttext.load_model('../../embeddings/cc.ko.300.bin')
	JOSA.build_vocab(train_data)
	# print(JOSA.vocab.itos)
	model = None
	if args.josa_model == 'nn':
		model = NN(args.josa_esz, len(JOSA.vocab.itos))
	elif args.josa_model == 'gru':
		model = GRUNET(args, len(JOSA.vocab.itos))
	if not model:
		print("-josa_model : nn, gru")
		exit(-1)
	model = model.to(args.device)
	print(model)
	criterion = nn.CrossEntropyLoss()
	# criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	# optimizer = optim.Adam(model.parameters())
	top_acc = 0
	for epoch in range(args.josa_epoch):
		start_time = time.time()
		train_loss = train_josa(args, train_data, model, criterion, optimizer, word_embedding)
		end_time = time.time()

		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Time: {epoch_mins}m {epoch_secs}s')
		if epoch % args.josa_eval_interval == 4:
			eval_loss, acc = eval_josa(args, dev_data, model, criterion, word_embedding)
			print(f"Val Accuracy: {acc:.4f} | Val Loss: {eval_loss:.3f}")
			if acc > top_acc:
				torch.save(model.state_dict(), args.josa_save)

	# Evaluation
	model.load_state_dict(torch.load(args.josa_save))
	eval_loss, acc = eval_josa(args, test_data, model, criterion, word_embedding)
	print(f"Test Accuracy: {acc:.4f} | Test Loss: {eval_loss:.3f}")


def train_josa(args, train_data, model, criterion, optimizer, word_embedding):
	model.train()
	train_loader = BucketIterator(train_data, batch_size=args.josa_bsz)
	TEXT.build_vocab(train_data)
	TEXT.vocab.itos.append('[TGT]')
	TEXT.vocab.stoi['[TGT]'] = TEXT.vocab.itos.index('[TGT]')
	# OFFSET.build_vocab(train_data)
	batch_loss = 0
	for batch in train_loader:
		optimizer.zero_grad()
		inputs, labels, offsets = format_josa(args, batch, word_embedding)
		# print("labels size: ", labels.shape)
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)
		outputs = model(inputs, offsets).to(args.device)
		# print(outputs.data)
		# if args.josa_model == 'gru':
		# 	outputs = outputs.float()
		# 	new_outputs = []
		# 	for n in range(len(batch.offset)):
		# 		pos = batch.offset[n].split()
		# 		output = []
		# 		for cur in pos.split():
		# 			output.append(outputs[cur, n])
		# 		new_outputs.append(np.array(output))
		# 	outputs = torch.tensor(new_outputs).to(args.device)
		# 	labels = labels.float()
		loss = criterion(outputs, labels)
		# loss.requires_grad = True
		loss.backward()
		optimizer.step()
		# print(loss.data)
		batch_loss += loss.item()

	return batch_loss


def format_josa(args, batch, embedding):
	inputs = []
	labels = []
	offsets = []
	if args.josa_model == 'nn':
		for n in range(len(batch.offset)):
			tokens = list(map(lambda x: TEXT.vocab.itos[x], batch.text[n]))
			for cur in batch.offset[n].split(): # josa_count * 2 * esz
				prev_next = np.array([embedding[tokens[int(cur)]], embedding[tokens[int(cur) + 2]]])
				inputs.append(prev_next)
				labels.append(batch.josa[int(cur) + 1, n])
	elif args.josa_model == 'gru':
		for n in range(len(batch.offset)):
			# tokens : (batch_size, seq_len, embed_size)
			# inputs : (batch_size, seq_len)
			tokens = list(map(lambda x: TEXT.vocab.itos[x], batch.text[n]))
			pos = batch.offset[n].split()
			# label = []
			for cur in pos:
				tokens[int(cur) + 1] = '[TGT]'
				inputs.append(np.array(list(map(lambda x: embedding[x], tokens))))
				labels.append(batch.josa[int(cur) + 1, n])
				offsets.append(int(cur) + 1)
			# labels.append(np.array(label))
			# labels.append(np.array(batch.josa[:, n]))

	inputs = torch.tensor(inputs)
	labels = torch.tensor(labels)
	# labels = F.one_hot(labels, len(JOSA.vocab.itos))

	return inputs, labels, offsets


def eval_josa(args, eval_data, model: nn.Module, criterion: nn.Module, word_embedding):
	model.eval()
	eval_loader = BucketIterator(eval_data, batch_size=args.josa_bsz)
	TEXT.build_vocab(eval_data)
	batch_loss = 0
	correct = 0
	total = 0
	t = 0
	s = 0
	for batch in eval_loader:
		inputs, labels, offsets = format_josa(args, batch, word_embedding)
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)
		outputs = model(inputs, offsets).to(args.device)
		# if args.josa_model == 'gru':
		# 	outputs = outputs.float()
		# 	labels = labels.float()
		loss = criterion(outputs, labels)

		batch_loss += loss.item()
		# if args.josa_model == 'nn':
		correct += (outputs.argmax(1) == labels).sum().item()
		total += len(labels)
	# 	elif args.josa_model == 'gru':
	# 		for n in range(len(batch.offset)):
	# 			for i in batch.offset[n].split():
	# 				id = int(i) + 1
	# 				if outputs[n, id].argmax() == labels[n, id].argmax():
	# 					correct += 1
	# 				total += 1
	# 		t += (outputs.argmax(2) == labels.argmax(2)).sum().item()
	# 		s += batch.josa.shape[0] * batch.josa.shape[1]
	# print("wrong acc: %.4f" % (t/s))
	return batch_loss, correct / total


def main(args):
	if args.selector:
		raise NotImplemented
	if args.planner:
		raise NotImplemented
	if args.generator:
		raise NotImplemented
	if args.josa:
		main_josa(args)
	return


if __name__ == '__main__':
	args = pargs()
	args.josa = True
	# args.josa_save = "../saved/temp.pt"
	# print(args.josa_epoch)
	main(args)

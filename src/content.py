import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pargs import pargs
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, RawField
import time
from utils import epoch_time
from model.content import Content

INP = Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)
OUTP = Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)
NERD = Field(sequential=True, batch_first=True,eos_token="<eos>")
ENT = RawField()
REL = RawField()
SUM = RawField()
GOLD = RawField()
fields=[("src", INP),("ent", ENT),("nerd", NERD),("rel", REL),("out", OUTP),("gold", GOLD),("sum", SUM)]
relations = []
with open('../data/preprocess/relations.vocab', 'r', encoding='utf-8') as f:
	for line in f.readlines():
		relations.append(line.strip())

def train(args):
	train_data, dev_data, test_data = TabularDataset.splits(path=args.datadir, train=args.nlg_train_path,
															validation=args.nlg_dev_path, test=args.nlg_test_path,
															skip_header=True, format='tsv',
															fields=fields)
	INP.build_vocab(train_data)
	OUTP.build_vocab(train_data)
	NERD.build_vocab(train_data)
	model = Content(args).to(args.device)
	print(model)

	criterion = nn.CrossEntropyLoss()
	# criterion = nn.MSELoss()
	# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	top_f1 = 0
	for epoch in range(args.content_epoch):
		start_time = time.time()
		train_loss = train_batch(args, train_data, model, criterion, optimizer)
		end_time = time.time()

		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Time: {epoch_mins}m {epoch_secs}s')
		if epoch % args.content_eval_interval == args.content_eval_interval-1:
			eval_loss, f1, acc, p, r = eval(args, dev_data, model, criterion)
			print(f"Val Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f} | ACC: {acc:.4f} | Loss: {eval_loss:.3f}")
			if f1 > top_f1:
				top_f1 = f1
				torch.save(model.state_dict(), args.content_model_save)

	# Evaluation
	model.load_state_dict(torch.load(args.content_model_save))
	eval_loss, f1, acc, p, r = eval(args, test_data, model, criterion)
	print(f"Test Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f} | ACC: {acc:.4f} | Loss: {eval_loss:.3f}")

	return


def train_batch(args, train_data, model, criterion, optimizer):
	model.train()
	train_loader = BucketIterator(train_data, batch_size=args.content_bsz)
	batch_loss = 0

	for batch in train_loader:
		optimizer.zero_grad()
		inputs, labels = format_content(args, batch)
		# print("labels size: ", labels.shape)
		# inputs = inputs.to(args.device)
		outputs = model(inputs)
		# labels = torch.tensor(labels).to(args.device)
		loss = criterion(outputs, labels)
		loss.backward()
		batch_loss += loss.item()

		# for output, label in zip(outputs, labels):
		# 	label = torch.tensor(label).to(args.device)
		# 	loss = criterion(output, label)
		# 	loss.backward(retain_graph=True)
		# 	batch_loss += loss.item()
		optimizer.step()
		# print(loss.data)
		# batch_loss += loss.item()

	return batch_loss

def format_content(args, batch):
	batch_input = []
	batch_label = []
	for b in range(batch.batch_size):
		ent = batch.ent[b].strip().split(' ; ')
		rel = batch.rel[b].strip().split(' ; ')
		sum = batch.sum[b].strip().split(' ; ')
		input = []
		label = []
		for r in rel:
			triple = list(map(lambda x: int(x), r.strip().split()))
			s = ent[triple[0]]
			p = relations[triple[1]]
			o = ent[triple[2]]
			input.append([s, p, o])
		batch_input.append(input)
		for used in sum:
			if used == 'T':
				batch_label.append(0)
				# label.append([1, 0])
			else:
				batch_label.append(1)
				# label.append([0, 1])
		# batch_label.append(label)
	batch_label = torch.tensor(batch_label).to(args.device)
	return batch_input, batch_label

def eval(args, eval_data, model: nn.Module, criterion: nn.Module):
	model.eval()
	eval_loader = BucketIterator(eval_data, batch_size=args.content_bsz)
	batch_loss = 0
	correct = 0
	gold = 0
	pred = 0
	total = 0
	tp = 0
	for batch in eval_loader:
		inputs, labels = format_content(args, batch)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		batch_loss += loss.item()
		correct += (outputs.argmax(1) == labels).sum().item()
		tp += ((outputs.argmax(1) == 0) * (labels == 0)).sum().item()
		pred += (outputs.argmax(1) == 0).sum().item()
		gold += (labels == 0).sum().item()
		total += labels.shape[0]
		# for output, label in zip(outputs, labels):
		# 	output = torch.tensor(output, requires_grad=True).to(args.device)
		# 	label = torch.tensor(label).to(args.device)
		# 	loss = criterion(output, label)
		# 	batch_loss += loss.item()
		# 	correct += (output.argmax(1) == label).sum().item()
		# 	o = list(output.argmax(1))
		# 	for i, x in enumerate(o):
		# 		if x == label[i] and x == 0:
		# 			tp += 1
		# 	pred += (output.argmax(1) == 0).sum().item()
		# 	gold += (label == 0).sum().item()
		# 	total += label.shape[0]
	p = tp / pred if pred != 0 else 0
	r = tp / gold
	f1 = 0
	if p != 0 or r != 0:
		f1 = 2 * (p * r) / (p + r)
	acc = correct / total
	return batch_loss, f1, acc, p, r


def predict(args):
	'''

	:param args:
	:return: list(set(지식)): 지식 = (주어, 서술어, 목적어)
	'''
	knowledge_sets = []

	train_data, dev_data, test_data = TabularDataset.splits(path=args.datadir, train=args.nlg_train_path,
															validation=args.nlg_dev_path, test=args.nlg_test_path,
															skip_header=False, format='tsv',
															fields=fields)
	INP.build_vocab(train_data)
	OUTP.build_vocab(train_data)
	NERD.build_vocab(train_data)
	model = Content(args).to(args.device)
	model.load_state_dict(torch.load(args.content_model_save))
	model.eval()
	predict_loader = BucketIterator(test_data, batch_size=args.content_bsz, shuffle=False)
	for batch in predict_loader:
		inputs, labels = format_content(args, batch)
		outputs = model(inputs)
		outputs = outputs.argmax(1)
		size = batch.batch_size
		ent = list(map(lambda x: x.split(' ; '), batch.ent))
		rel = list(map(lambda x: x.split(' ; '), batch.rel))
		rel_n = list(map(lambda x: len(x.split(' ; ')), batch.rel))
		st = 0
		for i in range(size):
			output = list(outputs[st:st + rel_n[i]])
			summarized_knowledge = []
			for k, j in enumerate(output):
				if j == 0:
					summarized_knowledge.append(rel[i][k])
			if summarized_knowledge == []:
				summarized_knowledge = rel[i]
			summarized_knowledge = list(map(lambda x: (ent[i][int(x.split()[0])], relations[int(x.split()[1])], ent[i][int(x.split()[2])]), summarized_knowledge))
			knowledge_sets.append(summarized_knowledge)
			st += rel_n[i]

	return knowledge_sets


if __name__ == '__main__':
	args = pargs()
	torch.cuda.set_device(args.device)
	if args.content_mode == 'train':
		train(args)
	elif args.content_mode == 'predict':
		predict(args)
	elif args.content_mode == 'eval':
		train_data, dev_data, test_data = TabularDataset.splits(path=args.datadir, train=args.nlg_train_path,
																validation=args.nlg_dev_path, test=args.nlg_test_path,
																skip_header=True, format='tsv',
																fields=fields)
		INP.build_vocab(train_data)
		OUTP.build_vocab(train_data)
		NERD.build_vocab(train_data)
		model = Content(args).to(args.device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		model.load_state_dict(torch.load(args.content_model_save))
		print(model)
		eval_loss, f1, acc, p, r = eval(args, test_data, model, criterion)
		print(f"Test Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f} | ACC: {acc:.4f} | Loss: {eval_loss:.3f}")



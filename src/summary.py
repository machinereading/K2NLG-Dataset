import os
import sys
import json
import torch
from src.pargs import pargs as pp_pargs
from src.content import predict as content_predict
from graphwriter.newpargs import pargs as gw_pargs, dynArgs as gw_dynArgs
from graphwriter.newDataset import dataset
from graphwriter.models.copymodel import model as gen_model
from graphwriter.generator import generate
from src.josa import predict_josa


relations = []
with open('../data/preprocess/relations.vocab', 'r', encoding='utf-8') as f:
	for line in f.readlines():
		relations.append(line.strip())


def save_content(pp_args, knowledge_sets):
	save_path = os.path.join(pp_args.datadir, 'summary', 'content_save.tsv')
	out_f = open(save_path, 'w', encoding='utf-8')
	input_path = os.path.join(pp_args.datadir, pp_args.nlg_test_path)
	in_f = open(input_path, 'r', encoding='utf-8')
	input_inst = in_f.readlines()
	for knowledge_set, input_line in zip(knowledge_sets, input_inst):
		title, ent, ner, rel, mask_ref, raw_ref, used = input_line.strip().split('\t')
		ent = ent.split(' ; ')
		ner = ner.split()
		summarized_ent = []
		summarized_knowledge = []
		for knowledge in knowledge_set:
			s = knowledge[0]
			p = knowledge[1]
			o = knowledge[2]
			is_new = True
			for e in summarized_ent:
				if s == e[0]:
					is_new = False
					break
			if is_new:
				summarized_ent.append((s, ner[ent.index(s)]))
			is_new = True
			for e in summarized_ent:
				if o == e[0]:
					is_new = False
					break
			if is_new:
				summarized_ent.append((o, ner[ent.index(o)]))
			entity = list(map(lambda x: x[0], summarized_ent))
			summarized_knowledge.append(str(entity.index(s)) + ' ' + str(relations.index(p)) + ' ' + str(entity.index(o)))
		wrote_ent = ' ; '.join(list(map(lambda x: x[0], summarized_ent)))
		wrote_type = ' '.join(list(map(lambda x: x[1], summarized_ent)))
		wrote_rel = ' ; '.join(summarized_knowledge)
		output_line = '\t'.join(['title', wrote_ent, wrote_type, wrote_rel, '-', '-', '-']) + '\n'
		out_f.write(output_line)
	out_f.close()
	in_f.close()

	return True


def save_gold_content(pp_args):
	save_path = os.path.join(pp_args.datadir, 'summary', 'content_save.tsv')
	out_f = open(save_path, 'w', encoding='utf-8')
	input_path = os.path.join(pp_args.datadir, pp_args.nlg_test_path)
	in_f = open(input_path, 'r', encoding='utf-8')
	input_inst = in_f.readlines()
	for input_line in input_inst:
		title, ent, ner, rel, mask_ref, raw_ref, used = input_line.strip().split('\t')
		ent = ent.split(' ; ')
		ner = ner.split()
		summarized_ent = []
		summarized_knowledge = []
		used = used.split(' ; ')
		rel = rel.split(' ; ')
		for use, r in zip(used, rel):
			if use != 'T':
				continue
			s, p, o = r.split()
			s = ent[int(s)]
			o = ent[int(o)]
			is_new = True
			for e in summarized_ent:
				if s == e[0]:
					is_new = False
					break
			if is_new:
				summarized_ent.append((s, ner[ent.index(s)]))
			is_new = True
			for e in summarized_ent:
				if o == e[0]:
					is_new = False
					break
			if is_new:
				summarized_ent.append((o, ner[ent.index(o)]))
			entity = list(map(lambda x: x[0], summarized_ent))
			summarized_knowledge.append(
				str(entity.index(s)) + ' ' + p + ' ' + str(entity.index(o)))
		wrote_ent = ' ; '.join(list(map(lambda x: x[0], summarized_ent)))
		wrote_type = ' '.join(list(map(lambda x: x[1], summarized_ent)))
		wrote_rel = ' ; '.join(summarized_knowledge)
		output_line = '\t'.join(['title', wrote_ent, wrote_type, wrote_rel, '-', '-', '-']) + '\n'
		out_f.write(output_line)
	out_f.close()
	in_f.close()

	return True


def save_generate(preds, pp_args):
	save_path = os.path.join(pp_args.datadir, 'summary', 'generate_save.tsv')
	out_f = open(save_path, 'w', encoding='utf-8')
	input_path = os.path.join(pp_args.datadir, pp_args.nlg_test_path)
	in_f = open(input_path, 'r', encoding='utf-8')
	input_inst = in_f.readlines()
	output_inst = []
	for input, pred in zip(input_inst, preds):
		title, entity, type, relation, _, _, _ = input.strip().split('\t')
		tokens = pred.split()
		type = type.split()
		text = []
		josa = []
		offset = []
		for t, token in enumerate(tokens):
			if '<entity_' in token:
				i = int(token.replace('<entity_', '')[:-1])
				text.append(type[i].upper())
				josa.append('<null>')
			elif '[josa]' == token:
				text.append('[JOSA]')
				josa.append('[조사]')
				offset.append(str(t))
			else:
				text.append(token)
				josa.append('<null>')
		line = '\t'.join([' '.join(text), ' '.join(josa), ' '.join(offset)]) + '\n'
		output_inst.append(line)
	out_f.write('text\tjosa\toffset\n')
	out_f.writelines(output_inst)
	out_f.close()
	in_f.close()
	return


def save_output():

	raise NotImplementedError


def load_gen_model(gen_model, gw_args, ds):
	cpt = torch.load('../graphwriter/' + gw_args.save + '/' + gw_args.save + '_29', map_location=pp_args.device)
	gen_model.load_state_dict(cpt)
	gen_model.to(pp_args.device)
	gen_model.args = gw_args
	gen_model.maxlen = gw_args.max
	gen_model.starttok = ds.OUTP.vocab.stoi['<start>']
	gen_model.endtok = ds.OUTP.vocab.stoi['<eos>']
	gen_model.eostok = ds.OUTP.vocab.stoi['.']
	gw_args.vbsz = 1
	return gen_model, gw_args


if __name__ == '__main__':
	# determine to use content selector and josa selector
	use_content = True
	use_josa = True
	use_gold_content = False
	# load modules' parameters
	pp_args = pp_pargs()
	pp_args.josa_test_path = 'summary/generate_save.tsv'
	pp_args.josa_model = 'nn'
	pp_args.josa_save = '../saved/josa_nn.pt'
	pp_args.josa_predict_save = "../data/summary/josa_save.tsv"

	gw_args = gw_pargs()
	gw_args.entunk = 1
	gw_args.outunk = 5
	# gw_args.copy = True

	torch.cuda.set_device(pp_args.device)

	print("use_content:", use_content, '\n', "use_josa:", use_josa)
	if use_content:
		if use_josa:
			# content select
			if use_gold_content:
				save_gold_content(pp_args)
			else:
				content = content_predict(pp_args)
				save_content(pp_args, content)

			# text generate
			gw_args.data = "generation_josa/generation_josa.train.tsv"
			ds = dataset(gw_args)
			gw_args = gw_dynArgs(gw_args, ds)
			gen_model = gen_model(gw_args)
			gw_args.save = 'generation_josa_30_1'
			gen_model, gw_args = load_gen_model(gen_model, gw_args, ds)
			out_vocab = ds.OUTP.vocab
			gw_args.datadir = "../data/summary/"
			gw_args.data = "content_save.tsv"
			ds = dataset(gw_args)
			ds.OUTP.vocab = out_vocab
			preds, _ = generate(gw_args, ds, gen_model, '9')
			save_generate(preds, pp_args)

			# josa select
			predict_josa(pp_args)
		else:
			# content select
			if use_gold_content:
				save_gold_content(pp_args)
			else:
				content = content_predict(pp_args)
				save_content(pp_args, content)

			# text generate
			ds = dataset(gw_args)
			gw_args.save = 'generation_base_30_1'
			gw_args.data = "generation_base/generation_base.train.tsv"
			gw_args = gw_dynArgs(gw_args, ds)
			gen_model = gen_model(gw_args)
			gen_model, gw_args = load_gen_model(gen_model, gw_args, ds)
			gw_args.datadir = "../data/summary/"
			gw_args.data = "content_save.tsv"
			out_vocab = ds.OUTP.vocab
			ds = dataset(gw_args)
			ds.OUTP.vocab = out_vocab
			preds, _ = generate(gw_args, ds, gen_model, '9')
			save_generate(preds, pp_args)
	else:
		if use_josa and not use_gold_content:
			# text generate
			gw_args.data = "summary_josa/summary_josa.train.tsv"
			ds = dataset(gw_args)
			gw_args = gw_dynArgs(gw_args, ds)
			gen_model = gen_model(gw_args)
			gw_args.save = 'summary_josa_30_1'
			gen_model, gw_args = load_gen_model(gen_model, gw_args, ds)
			preds, _ = generate(gw_args, ds, gen_model, '9')
			save_generate(preds, pp_args)

			# josa select
			predict_josa(pp_args)
		else:
			# text generate
			gw_args.data = "summary_base/summary_base.train.tsv"
			ds = dataset(gw_args)
			gw_args = gw_dynArgs(gw_args, ds)
			gen_model = gen_model(gw_args)
			gw_args.save = 'summary_base_30_1'
			gen_model, gw_args = load_gen_model(gen_model, gw_args, ds)
			preds, _ = generate(gw_args, ds, gen_model, '29')
			save_generate(preds, pp_args)

	generics = json.load(open('../data/summary/types.json', 'r', encoding='utf-8'))['generics']
	output_f = open('../data/summary/9/gold_gen.tsv', 'w', encoding='utf-8')
	if use_content and use_josa:
		content_save = open('../data/summary/content_save.tsv', 'r', encoding='utf-8')
		content = content_save.readlines()
		josa_save = open(pp_args.josa_predict_save, 'r', encoding='utf-8')
		josa = josa_save.readlines()[1:]
		for c_line, j_line, pred in zip(content, josa, preds):
			title, entities, types, rels, _, _, _ = c_line.split('\t')
			entities = entities.split(' ; ')
			types = types.split()
			rels = rels.split(' ; ')
			masked_sentence, josa, _ = j_line.split('\t')
			josa = josa.split()
			unmasked_sentence = []
			josa_i = 0
			pred = pred.split()
			for i, token in enumerate(masked_sentence.split()):
				if token.upper() in generics:
					entity = entities[int(pred[i].split('_')[-1][:-1])]
					unmasked_sentence.append(entity)
				elif token.upper() == '[JOSA]':
					unmasked_sentence.append(josa[josa_i])
					josa_i += 1
				else:
					unmasked_sentence.append(token)
			output_f.write(' '.join(unmasked_sentence) + '\n')
	elif not use_content and use_josa:
		content_save = open('../data/preprocess/summary_josa/summary_josa.test.tsv', 'r', encoding='utf-8')
		content = content_save.readlines()
		josa_save = open(pp_args.josa_predict_save, 'r', encoding='utf-8')
		josa = josa_save.readlines()[1:]
		for c_line, j_line, pred in zip(content, josa, preds):
			title, entities, types, rels, _, _, _ = c_line.split('\t')
			entities = entities.split(' ; ')
			types = types.split()
			rels = rels.split(' ; ')
			masked_sentence, josa, _ = j_line.split('\t')
			josa = josa.split()
			unmasked_sentence = []
			josa_i = 0
			pred = pred.split()
			for i, token in enumerate(masked_sentence.split()):
				if token.upper() in generics:
					entity = entities[int(pred[i].split('_')[-1][:-1])]
					unmasked_sentence.append(entity)
				elif token.upper() == '[JOSA]':
					unmasked_sentence.append(josa[josa_i])
					josa_i += 1
				else:
					unmasked_sentence.append(token)
			output_f.write(' '.join(unmasked_sentence) + '\n')
	elif use_content and not use_josa:
		content_save = open('../data/summary/content_save.tsv', 'r', encoding='utf-8')
		content = content_save.readlines()
		for c_line, pred in zip(content, preds):
			title, entities, types, rels, _, _, _ = c_line.split('\t')
			entities = entities.split(' ; ')
			types = types.split()
			rels = rels.split(' ; ')
			unmasked_sentence = []
			masked_sentence = pred.split()
			for i, token in enumerate(masked_sentence):
				if '<entity_' in token:
					entity = entities[int(token.split('_')[-1][:-1])]
					unmasked_sentence.append(entity)
				else:
					unmasked_sentence.append(token)
			output_f.write(' '.join(unmasked_sentence) + '\n')
	else:
		content_save = open('../data/preprocess/summary_base/summary_base.test.tsv', 'r', encoding='utf-8')
		content = content_save.readlines()
		for c_line, pred in zip(content, preds):
			title, entities, types, rels, _, _, _ = c_line.split('\t')
			entities = entities.split(' ; ')
			types = types.split()
			rels = rels.split(' ; ')
			unmasked_sentence = []
			masked_sentence = pred.split()
			for i, token in enumerate(masked_sentence):
				if '<entity_' in token:
					entity = entities[int(token.split('_')[-1][:-1])]
					unmasked_sentence.append(entity)
				else:
					unmasked_sentence.append(token)
			output_f.write(' '.join(unmasked_sentence) + '\n')
	output_f.close()

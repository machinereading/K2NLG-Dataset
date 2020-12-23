import pickle
import os
import collections
import sys

sys.path.append('./pycocoevalcap')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor


# from pycocoevalcap.cider.cider import Cider

class Evaluate(object):
	def __init__(self):
		self.scorers = [
			(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
			(Meteor(), "METEOR"),
			(Rouge(), "ROUGE_L")
		]  # ,        (Cider(), "CIDEr")

	def convert(self, data):
		if isinstance(data, basestring):
			return data.encode('utf-8')
		elif isinstance(data, collections.Mapping):
			return dict(map(convert, data.items()))
		elif isinstance(data, collections.Iterable):
			return type(data)(map(convert, data))
		else:
			return data

	def score(self, ref, hypo):
		final_scores = {}
		for scorer, method in self.scorers:
			score, scores = scorer.compute_score(ref, hypo)
			if type(score) == list:
				for m, s in zip(method, score):
					final_scores[m] = s
			else:
				final_scores[method] = score

		return final_scores

	def evaluate(self, get_scores=True, live=False, **kwargs):
		if live:
			temp_ref = kwargs.pop('ref', {})
			cand = kwargs.pop('cand', {})
		else:
			reference_path = kwargs.pop('ref', '')
			candidate_path = kwargs.pop('cand', '')

			# load caption data
			with open(reference_path, 'rb') as f:
				temp_ref = pickle.load(f)
			with open(candidate_path, 'rb') as f:
				cand = pickle.load(f)

		# make dictionary
		hypo = {}
		ref = {}
		i = 0
		for vid, caption in cand.items():
			hypo[i] = [caption]
			ref[i] = temp_ref[vid]
			i += 1

		# compute scores
		final_scores = self.score(ref, hypo)
		# """
		# print out scores
		print('Bleu_1:\t', final_scores['Bleu_1'])
		print('Bleu_2:\t', final_scores['Bleu_2'])
		print('Bleu_3:\t', final_scores['Bleu_3'])
		print('Bleu_4:\t', final_scores['Bleu_4'])
		print('METEOR:\t', final_scores['METEOR'])
		print('ROUGE_L:', final_scores['ROUGE_L'])
		# print ('CIDEr:\t', final_scores['CIDEr'])
		# """

		if get_scores:
			return final_scores


def single(pred_name, ref_name):
	use_i = []
	refs = dict()
	with open(ref_name, 'r', encoding='utf-8') as f:
		for i, x in enumerate(f.readlines()):
			if len(x.split('\t')[3].strip().split(' ; ')) >= 7:
				refs['generated_description' + str(i)] = [x.split('\t')[-2].strip()]
				use_i.append(i)
	cands = dict()
	with open(pred_name, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		for i in use_i:
			cands['generated_description' + str(i)] = lines[i].strip()
	x = Evaluate()
	result = x.evaluate(live=True, cand=cands, ref=refs)

	return


if __name__ == '__main__':
	'''
	cands = {'generated_description1': 'how are you', 'generated_description2': 'Hello how are you'}
	refs = {'generated_description1': ['what are you', 'where are you'],
		   'generated_description2': ['Hello how are you', 'Hello how is your day']}
	'''
	# result_f = open('results/' + sys.argv[1] + '.tsv', 'w', encoding='utf-8')
	# result_f.write('epoch\tBleu_1\tBleu_2\tBleu_3\tBleu_4\tMETEOR\tROUGE_L\n')
	# epoch = len(os.listdir('outputs/' + sys.argv[1]))
	# ref_name = sys.argv[2]
	# ref_dir = '_'.join(sys.argv[1].split('_')[:2])
	# for i in range(epoch):
	# 	print(i)
	# 	with open('outputs/' + sys.argv[1] + '/' + str(i) + '.inputs.beam_predictions') as f:
	# 		cands = {'generated_description' + str(i): x.split('\t')[-1].strip() for i, x in enumerate(f.readlines())}
	# 	with open(ref_name) as f:
	# 		refs = {'generated_description' + str(i): [x.split('\t')[-2].strip()] for i, x in enumerate(f.readlines())}
	# 	x = Evaluate()
	# 	result = x.evaluate(live=True, cand=cands, ref=refs)
	# 	result_f.write(str(i))
	# 	for metric in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L']:
	# 		result_f.write('\t' + str(result[metric]))
	# 	result_f.write('\n')
	#
	# result_f.close()

	pred_name = sys.argv[1]
	# pred_name = '../data/summary/9/gen_re.tsv'
	ref_name = '../data/preprocess/summary_base/summary_base.test.tsv'
	single(pred_name, ref_name)

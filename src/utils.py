def epoch_time(start_time: int,
			   end_time: int):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs


def view_all():
	gen = open('../data/summary/9/gen.tsv', 'r', encoding='utf-8').readlines()
	con_gen = open('../data/summary/9/con_gen.tsv', 'r', encoding='utf-8').readlines()
	gen_jo = open('../data/summary/9/gen_jo.tsv', 'r', encoding='utf-8').readlines()
	con_gen_jo = open('../data/summary/9/con_gen_jo.tsv', 'r', encoding='utf-8').readlines()
	gold_gen = open('../data/summary/9/gold_gen.tsv', 'r', encoding='utf-8').readlines()
	gold_gen_jo = open('../data/summary/9/gold_gen_josa.tsv', 'r', encoding='utf-8').readlines()
	ref = open('../data/preprocess/summary_base/summary_base.test.tsv', 'r', encoding='utf-8').readlines()

	output_f = open('../data/summary/9/view_all.tsv', 'w', encoding='utf-8')
	with open('../data/summary/relations.vocab', 'r', encoding='utf-8') as f:
		rel_vocab = list(map(lambda x: x.strip(), f.readlines()))
	i = 1
	for g, cg, gj, cgj, gg, ggj, r in zip(gen, con_gen, gen_jo, con_gen_jo, gold_gen, gold_gen_jo, ref):
		triples = []
		gold_triples = []
		_, entities, types, rels, _, _, tf = r.strip().split('\t')
		entities = entities.split(' ; ')
		rels = rels.split(' ; ')
		tf = tf.split(' ; ')
		for rel, t in zip(rels, tf):
			s, p, o = rel.split()
			s = entities[int(s)]
			p = rel_vocab[int(p)]
			o = entities[int(o)]
			triples.append('(' + s + ', ' + p + ', ' + o + ')')
			if t == 'T':
				gold_triples.append('(' + s + ', ' + p + ', ' + o + ')')
		triples = '; '.join(triples)
		gold_triples = '; '.join(gold_triples)

		output_f.write('triples' + '\t' + triples + '\n')
		output_f.write('gold_triples' + '\t' + gold_triples + '\n')
		output_f. write('gen' + '\t' + g)
		output_f.write('gold_gen' + '\t' + gg)
		output_f.write('con_gen' + '\t' + cg)
		output_f.write('gen_jo' + '\t' + gj)
		output_f.write('gold_gen_jo' + '\t' + ggj)
		output_f.write('con_gen_jo' + '\t' + cgj)

		output_f.write(str(i) + '\n')
		i += 1
	output_f.close()


def f1(prec, rec):
	if prec + rec == 0:
		return 0
	return 2 * prec * rec / (prec + rec)

def stat():
	f = open('../data/summary/9/view_all.txt', 'r', encoding='cp949')
	lines = f.readlines()[1:]
	i = 0
	gen_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	gold_gen_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	con_gen_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	gen_jo_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	gold_gen_jo_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	con_gen_jo_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	ref_result = [0, 0, 0]
	size = 0
	while True:
		triples = lines[9 * i].strip().split('\t')[1]
		gold_triples = lines[9 * i + 1].strip().split('\t')[1]
		gen = lines[9*i+2].strip().split('\t')
		gold_gen = lines[9 * i + 3].strip().split('\t')
		con_gen = lines[9*i+4].strip().split('\t')
		gen_jo = lines[9*i+5].strip().split('\t')
		gold_gen_jo = lines[9 * i + 6].strip().split('\t')
		con_gen_jo = lines[9*i+7].strip().split('\t')
		if len(gen) == 2:
			break
		if len(triples.split('; ')) >= 7:
			i += 1
			continue
		gold_triples = gold_triples.split('; ')
		e_set = set()
		p_set = set()
		for t in gold_triples:
			s, p, o = t.replace("(", '').replace(")", '').split(', ')
			e_set.add(s)
			e_set.add(o)
			p_set.add(p)
		ref_result[0] += len(gold_triples)
		ref_result[1] += len(e_set)
		ref_result[2] += len(p_set)

		for result, line in zip([gen_result, gold_gen_result, con_gen_result, gen_jo_result, gold_gen_jo_result, con_gen_jo_result], [gen, gold_gen, con_gen, gen_jo, gold_gen_jo, con_gen_jo]):
			result[0] += int(line[2].strip())
			result[1] += int(line[3].strip())
			result[2] += int(line[4].strip())
			result[3] += int(line[5].strip())
			result[4] += int(line[6].strip())
			result[5] += int(line[7].strip())
			result[6] += int(line[8].strip())
			result[7] += int(line[9].strip())
			if line[10].strip() == 'o':
				result[8] += 1
		i += 1
		size += 1

	print('mode:\tt_prec.\tt_rec.\tt_f1\te_prec.\te_rec.\te_f1\tr_prec.\tr_rec.\tr_f1\tgrammar\tnatural\trepetition')
	mode = ['gen_result', 'gold_gen_result', 'con_gen_result', 'gen_jo_result', 'gold_gen_jo_result', 'con_gen_jo_result']
	for j, result in enumerate([gen_result, gold_gen_result, con_gen_result, gen_jo_result, gold_gen_jo_result, con_gen_jo_result]):
		try:
			t_prec = result[1] / result[0]
		except:
			t_prec = 0
		t_rec = result[1] / ref_result[0]
		t_f1 = f1(t_prec, t_rec)
		try:
			e_prec = result[3] / result[2]
		except:
			e_prec = 0
		e_rec = result[3] / ref_result[1]
		e_f1 = f1(e_prec, e_rec)
		try:
			r_prec = result[5] / result[4]
		except:
			r_prec = 0
		r_rec = result[5] / ref_result[2]
		r_f1 = f1(r_prec, r_rec)
		grammar = result[6] / size
		nature = result[7] / size
		# print(mode[j])
		print(f'{t_prec:.4f}\t{t_rec:.4f}\t{t_f1:.4f}\t{e_prec:.4f}\t{e_rec:.4f}\t{e_f1:.4f}\t{r_prec:.4f}\t{r_rec:.4f}\t{r_f1:.4f}\t{grammar:.4f}\t{nature:.4f}\t{result[-1]:d}')

	f.close()


if __name__ == '__main__':
	# view_all()
	stat()


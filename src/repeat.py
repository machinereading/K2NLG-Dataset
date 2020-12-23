def remove_repeat():
	f = open('../data/summary/9/gen.tsv', 'r', encoding='utf-8')
	out_f = open('../data/summary/9/gen_re.tsv', 'w', encoding='utf-8')
	for line in f.readlines():
		sentences = line.split('.')
		new_sentences = []
		for sentence in sentences:
			target = sentence.strip()
			if target == '':
				continue
			if sentence.split()[0] in ['그는', '그의']:
				target = ' '.join(sentence.split()[1:])
			tf = False
			targets = [target]
			# targets = [target[:-1]+'이다', target[:-1]+'었다', target[:-1]+'있다']
			for new_sentence in new_sentences:
				for target in targets:
					if target in new_sentence:
						tf = True
						break
			if tf:
				continue
			else:
				new_sentences.append(sentence)
		out_f.write(' .'.join(new_sentences) + ' .' + '\n')
	out_f.close()
	f.close()


if __name__ == '__main__':
    remove_repeat()
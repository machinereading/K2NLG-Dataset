import os
source_f = open('./data/kor_relation.vocab', 'r')
target_f = open('../데이터셋/연구실 크라우드소싱 데이터/relation.vocab', 'r')
source_set = set(source_f.readlines())
for tgt in target_f.readlines():
	if tgt not in source_set:
		print(tgt)

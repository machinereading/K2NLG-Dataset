#!/bin/bash

number=$1
gpu=$2

#python generator.py -datadir data/${number}/ -data train.tsv -gpu ${gpu} -save GW_${number} -outunk 2 -entunk 1 -title
python generator.py -datadir data/${number}/ -data train.tsv -gpu ${gpu} -save GW_${number} -outunk 5 -entunk 1 -title
python eval.py GW_${number} data/${number}/gold_test


#python generator.py -datadir data/1/ -data train.tsv -gpu ${gpu} -save GW_1 -outunk 2 -entunk 1 -title
#python eval.py GW_1 data/1/gold_test

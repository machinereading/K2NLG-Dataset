import torch
import argparse
from time import time
# from lastDataset import dataset
from newDataset import dataset
# from models.newmodel import model
# from pargs import pargs,dynArgs
from models.copymodel import model
from newpargs import pargs,dynArgs
#import utils.eval as evalMetrics
import os


def tgtreverse(tgts,entlist,order):
  print('tgtreverse')
  entlist = entlist[0]
  order = [int(x) for x in order[0].split(" ")]
  tgts = tgts.split(" ")
  k = 0
  for i,x in enumerate(tgts):
    if x[0] == "<" and x[-1]=='>':
      tgts[i] = entlist[order[k]]
      k+=1
  return " ".join(tgts)
        
def test(args,ds,m,epoch='cmdline'):
  print('generation ' + str(epoch))
  args.vbsz = 1
  model = args.save
  m.eval()
  k = 0
  data = ds.mktestset(args)
  ofn = "./outputs/"+model+'/'+epoch+".inputs.beam_predictions"
  pf = open(ofn,'w')
  preds = []
  golds = []
  for b in data:
    #if k == 10: break
    #print(k,len(data))
    b = ds.fixBatch(b)
    '''
    p,z = m(b)
    p = p[0].max(1)[1]
    gen = ds.reverse(p,b.rawent)
    '''
    # print(b.rawent)
    # print(b.rel)
    # print(b.tgt)

    gen = m.beam_generate(b,beamsz=4,k=6)
    gen.sort()
    gen = ds.reverse(gen.done[0].words,b.rawent)
    k+=1
    gold = ds.reverse(b.out[0][0][1:], b.rawent)
    # print(gold)
    # print(gen)
    # print()
    preds.append(gen.lower())
    golds.append(gold.lower())
    #tf.write(ent+'\n')
    raw_pred = []
    for token in gen.split():
      if "<ENTITY_" in token:
        index = int(token.split('_')[1].replace('>', ''))
        if index >= len(b.rawent[0]):
          raw_pred.append(token)
        else:
          raw_pred.append(b.rawent[0][index])
      else:
        raw_pred.append(token)
    raw_rel = []
    for r in b.rawrel[0]:
      s, p, o = r.split()
      raw_rel.append('(' + b.rawent[0][int(s)] + ', ' + ds.REL.itos[int(p)+3] + ', ' + b.rawent[0][int(o)] + ')')
    raw_pred = ' '.join(raw_pred)
    # print(b.sum[0])
    pf.write(' ; '.join(b.rawent[0]) + '\t' + ' ; '.join(raw_rel) + '\t' + ' ; '.join(b.sum[0]) + '\t' + gen.lower() + '\t' + raw_pred + '\n')
  m.train()
  return preds,golds

def generate(args,ds,m,epoch='cmdline'):
  print('generation ' + str(epoch))
  args.vbsz = 1
  model = args.save
  m.eval()
  k = 0
  data = ds.mktestset(args)
  preds = []
  golds = []
  for b in data:
    #if k == 10: break
    #print(k,len(data))
    b = ds.fixBatch(b)
    '''
    p,z = m(b)
    p = p[0].max(1)[1]
    gen = ds.reverse(p,b.rawent)
    '''
    # print(b.rawent)
    # print(b.rel)
    # print(b.tgt)

    gen = m.beam_generate(b,beamsz=4,k=6)
    gen.sort()
    gen = ds.reverse(gen.done[0].words,b.rawent)
    k+=1
    gold = ds.reverse(b.out[0][0][1:], b.rawent)
    # print(gold)
    # print(gen)
    # print()
    preds.append(gen.lower())
    golds.append(gold.lower())
    #tf.write(ent+'\n')
    raw_pred = []
    for token in gen.split():
      if "<ENTITY_" in token:
        index = int(token.split('_')[1].replace('>', ''))
        if index >= len(b.rawent[0]):
          raw_pred.append(token)
        else:
          raw_pred.append(b.rawent[0][index])
      else:
        raw_pred.append(token)
    raw_rel = []
    for r in b.rawrel[0]:
      s, p, o = r.split()
      raw_rel.append('(' + b.rawent[0][int(s)] + ', ' + ds.REL.itos[int(p)+3] + ', ' + b.rawent[0][int(o)] + ')')
    raw_pred = ' '.join(raw_pred)
    # print(b.sum[0])
    # pf.write(' ; '.join(b.rawent[0]) + '\t' + ' ; '.join(raw_rel) + '\t' + ' ; '.join(b.sum[0]) + '\t' + gen.lower() + '\t' + raw_pred + '\n')
  m.train()
  return preds,golds


'''
def metrics(preds,gold):
  cands = {'generated_description'+str(i):x.strip() for i,x in enumerate(preds)}
  refs = {'generated_description'+str(i):[x.strip()] for i,x in enumerate(gold)}
  x = evalMetrics.Evaluate()
  scores = x.evaluate(live=True, cand=cands, ref=refs)
  return scores
'''

if __name__=="__main__":
  print('generator.py')
  args = pargs()
  args.eval = True
  ds = dataset(args)
  args = dynArgs(args,ds)
  m = model(args)
  torch.cuda.set_device(args.device)
  try :
    os.mkdir('outputs/' + args.save)
  except:
    pass
  epoch = len(os.listdir(args.save)) - 2
  for i in range(epoch):
    cpt = torch.load(args.save + '/' + args.save + '_' + str(i), map_location=args.device)
    m.load_state_dict(cpt)
    m = m.to(args.device)
    m.args = args
    m.maxlen = args.max
    m.starttok = ds.OUTP.vocab.stoi['<start>']
    m.endtok = ds.OUTP.vocab.stoi['<eos>']
    m.eostok = ds.OUTP.vocab.stoi['.']
    args.vbsz = 1
    preds,gold = test(args,ds,m, str(i))
  '''
  scores = metrics(preds,gold)
  for k,v in scores.items():
    print(k+'\t'+str(scores[k]))
  '''

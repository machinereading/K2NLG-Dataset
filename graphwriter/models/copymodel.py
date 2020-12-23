import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torch import nn
from models.attention import MultiHeadAttention, MatrixAttn
from models.list_encoder import list_encode, lseq_encode
from models.last_graph import graph_encode
from models.beam import Beam
from models.splan import splanner

'''
원본 : newmodel.py
수정 내역 : pointer-generator network에서 gen과 copy 분포를 합치는게 아니라 p_gen에 따라 둘 중 하나의 분포를 사용.
이때, gen의 분포는 <타입_숫자> 형식의 단어가 없으며, copy 분포는 입력 개체들에 대한 것. 
즉, 입력 개체가 3개이면 copy 분포는 3개의 개체에 대한 확률값만 가짐.
엮인 것 : copyDataset.py
'''

class model(nn.Module):
  def __init__(self,args):
    print('model_init')
    super().__init__()
    self.args = args
    cattimes = 3 if args.title else 2
    self.emb = nn.Embedding(args.ntoks,args.hsz)
    self.lstm = nn.LSTMCell(args.hsz*cattimes,args.hsz)
    self.out = nn.Linear(args.hsz*cattimes,args.ntoks)
    self.le = list_encode(args)
    self.entout = nn.Linear(args.hsz,1)
    self.switch = nn.Linear(args.hsz*cattimes,1)
    self.attn = MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop)
    self.mattn = MatrixAttn(args.hsz*cattimes,args.hsz)
    self.graph = (args.model in ['graph','gat','gtrans'])
    print(args.model)
    if self.graph:
      self.ge = graph_encode(args)
    # self.sim = nn.Linear()

  def forward(self,b):
    outp,_ = b.out
    ner = []
    for i, j in enumerate(b.ent[2]):
      for k in range(j):
        ner.append([b.nerd[i, k]])
    ents = (torch.tensor(ner).cuda(), b.ent[1], b.ent[2])
    # outp = b.tgt
    # ents = b.ent
    entlens = ents[2]
    ents = self.le(ents)

    # 인코딩
    if self.graph:
      gents,glob,grels = self.ge(b.rel[0],b.rel[1],(ents,entlens))
      hx = glob
      keys,mask = grels
      mask = mask==0
    else:
      mask = self.maskFromList(ents.size(),entlens)
      hx = ents.mean(dim=1)
      keys = ents
    mask = mask.unsqueeze(1)
    planlogits = None

    cx = torch.tensor(hx) # context vector
    #print(hx.size(),mask.size(),keys.size())
    a = torch.zeros_like(hx) #self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
    #e = outp.transpose(0,1)
    e = self.emb(outp).transpose(0,1)
    outputs = []
    for i, k in enumerate(e):
      #k = self.emb(k)
      prev = torch.cat((a,k),1)
      hx,cx = self.lstm(prev,(hx,cx))
      a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
      out = torch.cat((hx,a),1)
      outputs.append(out)
    l = torch.stack(outputs,1)
    p = torch.sigmoid(self.switch(l))
    gen = self.out(l)
    gen = torch.softmax(gen,2)
    gen = p*gen
    #compute copy attn
    _, z = self.mattn(l,(ents,entlens))
    #z = torch.softmax(z,2)
    copy = (1-p)*z
    y = gen
    y[:, :, -40 : -40 + z.shape[2]] += copy
    # o = torch.cat((gen, copy),2)
    y = y+(1e-6*torch.ones_like(y))
    for batch_i in range(y.shape[0]):
      y[batch_i, :, -40 + entlens[batch_i]:] = 0
    return y.log(),z,planlogits

  def maskFromList(self,size,l):
    mask = torch.arange(0,size[1]).unsqueeze(0).repeat(size[0],1).long().cuda()
    mask = (mask <= l.unsqueeze(1))
    mask = mask==0
    return mask
    
  def emb_w_vertex(self,outp,vertex):
    mask = outp>=self.args.ntoks
    if mask.sum()>0:
      idxs = (outp-self.args.ntoks)
      idxs = idxs[mask]
      verts = vertex.index_select(1,idxs)
      outp.masked_scatter_(mask,verts)

    return outp

  def beam_generate(self,b,beamsz,k):
    ner = []
    for i, j in enumerate(b.ent[2]):
      for kk in range(j):
        ner.append([b.nerd[i, kk]])
    ents = (torch.tensor(ner).cuda(), b.ent[1], b.ent[2])
    # ents = b.ent
    entlens = ents[2]
    ents = self.le(ents)
    if self.graph:
      gents,glob,grels = self.ge(b.rel[0],b.rel[1],(ents,entlens))
      hx = glob
      #hx = ents.max(dim=1)[0]
      keys,mask = grels
      mask = mask==0
    else:
      mask = self.maskFromList(ents.size(),entlens)
      hx = ents.max(dim=1)[0]
      keys =ents
    mask = mask.unsqueeze(1)
    planlogits = None

    cx = torch.tensor(hx)
    a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
    outputs = []
    outp = torch.LongTensor(ents.size(0),1).fill_(self.starttok).cuda()
    beam = None
    for i in range(self.maxlen):
      op = self.emb_w_vertex(outp.clone(),b.nerd)
      op = self.emb(op).squeeze(1)
      prev = torch.cat((a,op),1)
      hx,cx = self.lstm(prev,(hx,cx))
      a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
      l = torch.cat((hx,a),1).unsqueeze(1)
      p = torch.sigmoid(self.switch(l))
      gen = self.out(l)
      gen = torch.softmax(gen,2)
      gen = p*gen
      #compute copy attn
      _, z = self.mattn(l,(ents,entlens))
      #z = torch.softmax(z,2)
      copy = (1-p) * z
      y = gen
      y[:, :, -40 : -40 + z.shape[2]] += copy
      # z = (1-s)*z
      # o = torch.cat((o,z),2)
      for batch_i in range(y.shape[0]):
        y[batch_i, :, -40 + entlens[batch_i]:] = 0
      y[:,:,0].fill_(0)
      y[:,:,1].fill_(0)
      '''
      if beam:
        for p,q in enumerate(beam.getPrevEnt()):
          o[p,:,q].fill_(0)
        for p,q in beam.getIsStart():
          for r in q:
            o[p,:,r].fill_(0)
      '''

      y = y+(1e-6*torch.ones_like(y))
      decoded = y.log()
      scores, words = decoded.topk(dim=2,k=k)
      if not beam:
        beam = Beam(words.squeeze(),scores.squeeze(),[hx for i in range(beamsz)],
                  [cx for i in range(beamsz)],[a for i in range(beamsz)],beamsz,k,self.args.ntoks)
        beam.endtok = self.endtok
        beam.eostok = self.eostok
        keys = keys.repeat(len(beam.beam),1,1)
        mask = mask.repeat(len(beam.beam),1,1)
        if self.args.title:
          tencs = tencs.repeat(len(beam.beam),1,1)
          tmask = tmask.repeat(len(beam.beam),1,1)
          
        ents = ents.repeat(len(beam.beam),1,1)
        entlens = entlens.repeat(len(beam.beam))
      else:
        if not beam.update(scores,words,hx,cx,a):
          break
        keys = keys[:len(beam.beam)]
        mask = mask[:len(beam.beam)]
        ents = ents[:len(beam.beam)]
        entlens = entlens[:len(beam.beam)]
      outp = beam.getwords()
      hx = beam.geth()
      cx = beam.getc()
      a = beam.getlast()

    return beam


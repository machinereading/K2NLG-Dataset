import torch
import math
from torch import nn
from torch.nn import functional as F
from models.graphAttn import GAT
from models.attention import MultiHeadAttention


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Block(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.attn = MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop)
    self.l1 = nn.Linear(args.hsz,args.hsz*4)
    self.l2 = nn.Linear(args.hsz*4,args.hsz)
    self.ln_1 = nn.LayerNorm(args.hsz)
    self.ln_2 = nn.LayerNorm(args.hsz)
    self.drop = nn.Dropout(args.drop)
    #self.act = gelu
    self.act = nn.PReLU(args.hsz*4)
    self.gatact = nn.PReLU(args.hsz)

  def forward(self,q,k,m):
    q = self.attn(q,k,mask=m).squeeze(1)
    t = self.ln_1(q)
    q = self.drop(self.l2(self.act(self.l1(t))))
    q = self.ln_1(q+t)
    return q

class graph_encode(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.renc = nn.Embedding(args.rtoks,args.hsz)
    nn.init.xavier_normal_(self.renc.weight)

    if args.model == "gat":
      self.gat = nn.ModuleList([MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop) for _ in range(args.prop)])
    else:
      self.gat = nn.ModuleList([Block(args) for _ in range(args.prop)])
    self.prop = args.prop

  def pad(self,tensor,length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

  def forward(self,adjs,rels,ents):
    vents,entlens = ents
    if self.args.entdetach:
      vents = torch.tensor(vents,requires_grad=False)
    vrels = [self.renc(x) for x in rels]
    glob = []
    graphs = []
    for i,adj in enumerate(adjs):
      vgraph = torch.cat((vents[i][:entlens[i]],vrels[i]),0)
      N = vgraph.size(0)
      mask = (adj == 0).unsqueeze(1)
      for j in range(self.prop):
        ngraph = torch.tensor(vgraph.repeat(N,1).view(N,N,-1),requires_grad=False)
        vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
        if self.args.model == 'gat':
          vgraph = vgraph.squeeze(1)
          vgraph = self.gatact(vgraph)
      graphs.append(vgraph)
      glob.append(vgraph[entlens[i]])
    elens = [x.size(0) for x in graphs]
    gents = [self.pad(x,max(elens)) for x in graphs]
    gents = torch.stack(gents,0)
    elens = torch.LongTensor(elens)
    emask = torch.arange(0,gents.size(1)).unsqueeze(0).repeat(gents.size(0),1).long()
    emask = (emask <= elens.unsqueeze(1)).cuda()
    glob = torch.stack(glob,0)
    return None,glob,(gents,emask)

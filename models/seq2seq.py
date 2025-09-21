import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,input_dim:int,hidden:int):
        super().__init__()
        self.rnn=nn.GRU(input_dim,hidden)
    def forward(self,x):
        outputs,h=self.rnn(x)
        return outputs,h

class DotAttention(nn.Module):
    def forward(self,q,k,v):
        score=torch.matmul(q,k.transpose(1,2))/(k.size(-1)**0.5)
        w=torch.softmax(score,dim=-1)
        ctx=torch.matmul(w,v)
        return ctx

class Decoder(nn.Module):
    def __init__(self,hidden:int,output_dim:int):
        super().__init__()
        self.rnn=nn.GRU(output_dim,hidden)
        self.proj=nn.Linear(hidden*2,output_dim)
        self.attn=DotAttention()
        self.output_dim=output_dim
    def forward(self,enc_out,h,steps=5):
        B=h.size(1)
        y=torch.zeros(1,B,enc_out.size(-1),device=enc_out.device)
        outputs=[]
        K=enc_out.transpose(0,1)
        for _ in range(steps):
            o,h=self.rnn(y,h)
            q=o.transpose(0,1)
            ctx=self.attn(q,K,K)
            cat=torch.cat([q,ctx],dim=1)
            y=self.proj(cat).transpose(0,1)
            outputs.append(y)
        return torch.cat(outputs,dim=0)
class Seq2SeqAttn(nn.Module):
    def __init__(self,input_dim:int,hidden:int,output_dim:int):
        super().__init__()
        self.enc=Encoder(input_dim,hidden)
        self.dec=Decoder(hidden,output_dim)
    def forward(self,x):
        enc_out,h=self.enc(x)
        return self.dec(enc_out,h,steps=5)
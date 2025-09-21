import copy
import torch
import torch.nn as nn

from models.transformer_multi_attention import MultiHeadAttention


def clone_module_list(module:nn.Module,n:int)->nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)
class TransformerLayer(nn.Module):
    def __init__(self,*,
                 d_model:int,
                 self_attn:MultiHeadAttention,
                 src_attn:MultiHeadAttention,
                 feed_forward:FeedForward,
                 dropout_prob:float):
        super().__init__()
        self.size=d_model
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.dropout=nn.Dropout(dropout_prob)
        self.norm_self_attn=nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_self_attn=nn.LayerNorm([d_model])
        self.norm_ff=nn.LayerNorm([d_model])
        self.is_save_ff_input=False

    def forward(self,*,
                x:torch.Tensor,
                mask:torch.Tensor,
                src:torch.Tensor=None,
                src_mask:torch.Tensor=None):
        z=self.norm_self_attn(x)
        self_attn=self.self_attn(query=z,key=z,value=z,mask=mask)
        x=x+self.dropout(self_attn)

        if src is not None:
            z=self.norm_self_attn(x)
            attn_src=self.src_attn(query=z,key=z,value=z,mask=mask)
            x=x+self.dropout(attn_src)
        z=self.norm_ff(x)
        if self.is_save_ff_input:
            self.ff_input=z.clone()
        ff=self.feed_forward(z)
        x=x+self.dropout(ff)

        return x
class PatchEmbeddings(nn.Module):
    def __init__(self,d_model:int,patch_size:int,in_channels:int):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,d_model,patch_size,stride=patch_size)
    def forward(self,x:torch.Tensor):
        x=self.conv(x)
        bs,c,h,w=x.shape
        x=x.permute(2,3,0,1)
        x=x.view(h*w,bs,c)
        return x

class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self,d_model:int,max_len:int=5_000):
        super().__init__()
        self.positionnal_encodings=nn.Parameter(torch.zeros(max_len,1,d_model),requires_grad=True)
    def forward(self,x:torch.Tensor):
        pe=self.positionnal_encodings[:x.shape[0]]
        return x+pe

class ClassificationHead(nn.Module):
    def __init__(self,d_model:int,n_hidden:int,n_classes:int):
        super().__init__()
        self.linear1=nn.Linear(d_model,n_hidden)
        self.act=nn.ReLU()
        self.linear2=nn.Linear(n_hidden,n_classes)
    def forward(self,x:torch.Tensor):
        x=self.act(self.linear1(x))
        x=self.linear2(x)
        return x
class VisionTransformer(nn.Module):
    def __init__(self,transformer_layer:TransformerLayer,n_plays:int,
                 patch_emb:PatchEmbeddings,pos_emb:LearnedPositionalEmbeddings,
                 classification:ClassificationHead):
        super().__init__()
        self.patch_emb=patch_emb
        self.pos_emb=pos_emb
        self.classification=classification
        self.transformer_layers=clone_module_list(transformer_layer,n_plays)
        self.cls_token_emb=nn.Parameter(torch.randn(1,1,transformer_layer.size),requires_grad=True)
        self.ln=nn.LayerNorm([transformer_layer.size])
    def forward(self,x:torch.Tensor):
        x=self.patch_emb(x)
        cls_token_emb=self.cls_token_emb.expand(-1,x.shape[1],-1)
        x=torch.cat([cls_token_emb,x])
        x=self.pos_emb(x)
        for layer in self.transformer_layers:
            x=layer(x=x,mask=None)
        x=x[0]
        x=self.ln(x)
        x=self.classification(x)
        return x

























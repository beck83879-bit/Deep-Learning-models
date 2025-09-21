import math
from typing import Literal,Optional,Tuple
import torch
import torch.nn as nn
from torch.ao.nn.quantized.dynamic import RNNCell
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

RNNCell=Literal["gru","lstm","rnn"]
NormType=Literal["ln","bn","none"]

class AttnPool(nn.Module):
    def __init__(self,hidden_dim:int):
        super().__init__()
        self.proj=nn.Linear(hidden_dim,hidden_dim)
        self.v=nn.Linear(hidden_dim,1,bias=False)
    def forward(self,h:torch.Tensor,mask:Optional[torch.Tensor]=None)->torch.Tensor:
        score=self.v(torch.tanh(self.proj(h))).squeeze(-1)
        if mask is not None:
            score=score.masked_fill(~mask,float("-inf"))
        w=torch.softmax(score,dim=0)
        pooled=(w.unsqueeze(-1)*h).sum(dim=0)
        return pooled
class RNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden: int = 128,
            num_layers: int = 2,
            cell: RNNCell = "gru",
            bidirectional: bool = True,
            dropout: float = 0.1,
            pool: Literal["last", "attn"] = "attn",
            norm: NormType = "ln",
            head_hidden: int = 128,
            num_classes: int = 5,
            residual_head: bool = True,
    ):
        super().__init__()
        self.input_dim=input_dim
        self.hidden=hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell = cell
        self.pool = pool

        rnn_cls={"gru":nn.GRU,"lstm":nn.LSTM,"rnn":nn.RNN}[cell]
        self.rnn=rnn_cls(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,
        )
        feat_dim=hidden*(2 if bidirectional else 1)
        self.norm=(
            nn.LayerNorm(feat_dim) if norm =="ln"
            else (nn.BatchNorm1d(feat_dim) if norm =="bn" else nn.Identity())
        )
        self.attn=AttnPool(feat_dim) if pool=="attn" else None

        self.head_fc1 = nn.Linear(feat_dim, head_hidden)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(0.1)
        self.head_fc2 = nn.Linear(head_hidden, num_classes)
        self.residual_head = residual_head and (feat_dim == head_hidden)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)

    @staticmethod
    def _make_mask(lengths:torch.Tensor,max_len:int)->torch.Tensor:
        device = lengths.device
        L = torch.arange(max_len, device=device).unsqueeze(1)
        return L < lengths.unsqueeze(0)

    def forward(
            self,
            x: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lengths is not None:
            # 按长度降序排序以 pack（再按原序恢复）
            lengths_sorted, idx = torch.sort(lengths, descending=True)
            x_sorted = x[:, idx, :]
            packed = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), enforce_sorted=True)
            out_packed, h = self.rnn(packed)
            out, _ = pad_packed_sequence(out_packed)
            inv_idx = torch.argsort(idx)
            out = out[:, inv_idx, :]
            mask = self._make_mask(lengths, out.size(0))
        else:
            out, h = self.rnn(x)
            mask = None

        if self.pool == "last":
            if lengths is None:
                feat = out[-1]
            else:
                last_idx = (lengths - 1).clamp_min(0)  # (B,)
                feat = out.permute(1, 0, 2)[torch.arange(out.size(1), device=out.device), last_idx]  # (B,H*)
        else:
            feat = self.attn(out, mask)
        if isinstance(self.norm, nn.BatchNorm1d):
            feat = self.norm(feat)
        else:
            feat = self.norm(feat)
        y = self.head_fc1(feat)
        y = self.head_act(y)
        y = self.head_drop(y)
        if self.residual_head and y.shape == feat.shape:
            y = y + feat
        logits = self.head_fc2(y)
        return logits
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_dim:int,hidden:int,out_dim:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,out_dim)
        )
    def forward(self,x):
        return self.net(x)
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self,in_dim:int,out_dim:int):
        super().__init__()
        self.fc=nn.Linear(in_dim,out_dim)
    def forward(self,x):
        return self.fc(x)
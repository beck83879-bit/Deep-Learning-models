import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self,input_dim:int,hidden:int,num_classes:int):
        super().__init__()
        self.rnn=nn.GRU(input_dim,hidden,batch_first=False)
        self.head=nn.Linear(hidden,num_classes)
    def forward(self,x):
        out,h_n=self.rnn(x)
        return self.head(out[-1])
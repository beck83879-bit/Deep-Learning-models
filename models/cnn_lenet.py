import math
import torch
import torch.nn as nn

class LeNetPro(nn.Module):
    def __init__(
            self,
            in_channels:int=1,
            num_classes:int=10,
            conv_channels=(6,16),
            fc_dims=(120,84),
            use_batchnorm:bool=True,
            dropout:float=0.1,
            input_size:int=32
    ):
        super().__init__()
        c1,c2=conv_channels
        self.features=nn.Sequential(
            nn.Conv2d(in_channels,c1,kernel_size=5,stride=1,padding=0,bias=not use_batchnorm),
            nn.BatchNorm2d(c1) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

        def conv_out(h_or_w:int)->int:
            h=h_or_w
            h=h-4
            h=h//2
            h=h-4
            h=h//2
            return h

        h=conv_out(input_size)
        w=conv_out(input_size)
        flattened=c2*h*w
        f1,f2=fc_dims

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened,f1),nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(f1,f2),nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(f2,num_classes),
        )

        self.__init__weights()

    def __init__weights(self):

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m,nn.Linear):
                    nn.init.kaiming_uniform_(m.weight,a=math.sqrt(5))
                    nn.init.zeros_(m.bias)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.features(x)
        x=self.classifier(x)
        return x

def lenet_mnist(num_classes:int=10,**kwargs)->LeNetPro:
    return LeNetPro(in_channels=1,num_classes=num_classes,input_size=28,**kwargs)
def lenet_cifar(num_classes: int = 10, **kwargs) -> LeNetPro:
    return LeNetPro(in_channels=3, num_classes=num_classes, input_size=32, **kwargs)
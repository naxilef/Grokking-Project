import torch 
import torch.nn as nn

class ReluNet(nn.Module):

    def __init__(self, input_dim=40, width=1000):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, width)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(width, 1, bias=False) 

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x 
    
    
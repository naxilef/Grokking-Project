import torch 
import torch.nn as nn
import torch.nn.functional as F 

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x ** 2

class ModularMLP(nn.Module):
    
    def __init__(self, p: int, hidden_dim: int, activation: str = "quadratic"): 
        super().__init__()
        self.p = p
        self.input_dim = 2*p
        self.hidden_dim = hidden_dim
        self.output_dim = p 

        self.W1 = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, self.output_dim, bias=False)

        nn.init.normal_(self.W1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.W2.weight, mean=0.0, std=1.0)

        if activation == "quadratic":
            self.activation = QuadraticActivation()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(
                "activation must be one of: 'quadratic', 'relu', 'gelu', 'tanh'"
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.W1(x) / self.input_dim ** (1/2)
        z1 = self.activation(h1)
        h2 = self.W2(z1) / self.hidden_dim
        return h2 

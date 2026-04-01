import torch 
import torch.nn as nn 

import random

def parity(n, k, n_samples, seed=42):
    'Data generation'

    random.seed(seed)
    samples = torch.Tensor([[random.choice([-1, 1]) for j in range(n)] for i in range(n_samples)])
    # targets = torch.prod(input[:, n//2:n//2+k], dim=1) # parity hidden in the middle
    targets = torch.prod(samples[:, :k], dim=1) # parity hidden in first k bits

    return samples, targets
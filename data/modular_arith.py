import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset

def encode_pair(a, b, p):
    xa = F.one_hot(a.long(), num_classes=p)
    xb = F.one_hot(b.long(), num_classes=p)
    return torch.cat([xa,xb],dim=1).float()

def modular_addition(a, b, p):
    return (a+b) % p 

class ModularArithmeticTensors:
    def __init__(self, a, b, x, y):
        self.a = a
        self.b = b
        self.x = x 
        self.y = y

class ModularArithmeticDataset(Dataset):
    def __init__(self, x, y):
        if x.ndim != 2:
            raise ValueError("x must have shape (num_samples, 2p)")
        if y.ndim != 1:
            raise ValueError("y must have shape (num_samples,)")
        self.x = x.float()
        self.y = y.long()
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def generate_all_pair(p):
    values = torch.arange(p)
    grid_a, grid_b = torch.meshgrid(values, values, indexing="ij")
    return grid_a.reshape(-1), grid_b.reshape(-1)

def make_modular_dataset(p, fct):
    a, b = generate_all_pair(p)
    x = encode_pair(a, b, p)
    y = fct(a, b, p).long()
    return ModularArithmeticTensors(a,b,x,y)

def split_dataset(x, y, alpha=0.5, seed=0):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")
    
    n = x.shape[0]
    num_train = int(alpha*n)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    train_idx = perm[:num_train]
    test_idx = perm[num_train:]

    train_dataset = ModularArithmeticDataset(x[train_idx],y[train_idx])
    test_dataset = ModularArithmeticDataset(x[test_idx],y[test_idx])

    return train_dataset, test_dataset
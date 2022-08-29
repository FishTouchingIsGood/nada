import torch

a = torch.tensor([[1,2,3]])
b = a.repeat(2,3)
b[0][0] = 1000
print(b)

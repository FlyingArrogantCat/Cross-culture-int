import torch

a = torch.FloatTensor(range(0,10))
b = torch.FloatTensor(range(0,10))

print(a.shape)
print(a)
c = a * b
print(c.shape)
print(c)
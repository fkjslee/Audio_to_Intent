import yaml
import torch
import random


x = torch.arange(50, 100).reshape(-1)
y = [i for i in range(0, 50)]

random.shuffle(y)
print(y)
print(x)
print(x[y])

import torch
from torch import nn

from copy import deepcopy

param = nn.ModuleDict({'model': nn.Linear(10, 10)})
#for p in param.parameters():
#    p.requires_grad = False

param2 = deepcopy(param)
param2['coref'] = nn.Linear(10, 1)
#for p in param2.parameters():
#    p.requires_grad = False

param['model'].weight.data = torch.abs(param['model'].weight.data)

print(param)
print(param['model'].weight[0])
print(param2)
print(param2['model'].weight[0])
from typing import Union, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(modules: Union[Iterable[nn.Module], nn.Module]):
    if isinstance(modules, nn.Module):
        modules = [modules]
    return sum(sum(p.numel() for p in m.parameters()) for m in modules)


def repeat_stacked(x: torch.Tensor, n: int):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.size())))


def deep_tensor(tensor_list) -> torch.Tensor:
    if isinstance(tensor_list, torch.Tensor):
        return tensor_list
    if isinstance(tensor_list, list):
        return torch.stack([deep_tensor(t) for t in tensor_list])


def fix_tensor_dataset(x):
    x = deep_tensor(x)
    return x.permute((-1, *range(len(x.size()) - 1)))


def fix_string_batch(x):
    x = np.array(x)
    return x.transpose((-1, *range(len(x.shape) - 1)))


class TensorsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def freeze(modules: Union[nn.Module, nn.ModuleDict], exception: list[str] = None) -> nn.Module:
    if isinstance(modules, nn.Module):
        modules_to_freeze = [modules]
    elif isinstance(modules, nn.ModuleDict):
        modules_to_freeze = modules.values()
    else:
        raise ValueError("Arg 'modules' must be 'nn.Module' or 'nn.ModuleDict'.")
    if exception is None:
        exception = []
    for m in modules_to_freeze:
        m.requires_grad = False
        for param in m.parameters():
            param.requires_grad = False
        for name, param in m.named_parameters():
            if name in exception:
                param.requires_grad = True
    return modules


def unfreeze(modules: Union[nn.Module, nn.ModuleDict], exception: list[str] = None) -> nn.Module:
    if isinstance(modules, nn.Module):
        modules_to_freeze = [modules]
    elif isinstance(modules, nn.ModuleDict):
        modules_to_freeze = modules.values()
    else:
        raise ValueError("Arg 'modules' must be 'nn.Module' or 'nn.ModuleDict'.")
    if exception is None:
        exception = []
    for m in modules_to_freeze:
        m.requires_grad = True
        for param in m.parameters():
            param.requires_grad = True
        for name, param in m.named_parameters():
            if name in exception:
                param.requires_grad = False
    return modules

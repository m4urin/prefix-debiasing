from typing import Union, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(modules: Union[nn.Module, nn.ModuleDict]):
    if isinstance(modules, nn.Module):
        modules_to_process = [modules]
    elif isinstance(modules, nn.ModuleDict):
        modules_to_process = modules.values()
    else:
        raise ValueError("Arg 'modules' must be 'nn.Module' or 'nn.ModuleDict'.")

    return sum(sum(p.numel() for p in module.parameters()) for module in modules_to_process)


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


def set_grad(_bool: bool,
             modules: Union[nn.Module, nn.ModuleDict],
             exception: Union[str, list[str]] = None):
    if isinstance(modules, nn.Module):
        modules_to_freeze = [modules]
    elif isinstance(modules, nn.ModuleDict):
        modules_to_freeze = modules.values()
    else:
        raise ValueError("Arg 'modules' must be 'nn.Module' or 'nn.ModuleDict'.")

    if exception is None:
        exception = []
    elif isinstance(exception, str):
        exception = [exception]

    # set grad
    for module in modules_to_freeze:
        module.requires_grad = _bool
        for param in module.parameters():
            param.requires_grad = _bool

    # exception
    for module in modules_to_freeze:
        for name, param in module.named_parameters():
            if name in exception:
                param.requires_grad = not _bool
    return modules


def freeze(modules: Union[nn.Module, nn.ModuleDict],
           exception: Union[str, list[str]] = None):
    return set_grad(False, modules, exception)


def unfreeze(modules: Union[nn.Module, nn.ModuleDict],
             exception: Union[str, list[str]] = None):
    return set_grad(True, modules, exception)


def is_frozen(modules: Union[nn.Module, nn.ModuleDict]):
    if isinstance(modules, nn.Module):
        modules_to_process = [modules]
    elif isinstance(modules, nn.ModuleDict):
        modules_to_process = modules.values()
    else:
        raise ValueError("Arg 'modules' must be 'nn.Module' or 'nn.ModuleDict'.")

    has_true_grad = False
    has_false_grad = False
    for module in modules_to_process:
        for param in module.parameters():
            if param.requires_grad:
                has_true_grad = True
            else:
                has_false_grad = True
            if has_true_grad and has_false_grad:
                return 'partially frozen'
    if not has_true_grad and not has_false_grad:
        return 'no parameters'
    if has_true_grad:
        return 'unfrozen'
    return 'frozen'

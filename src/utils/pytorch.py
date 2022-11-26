from typing import Union, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def repeat_stacked(x: torch.Tensor, n: int):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.size())))


def nested_stack(x: Union[List, Tuple], dim=0) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return torch.stack([nested_stack(t) for t in x], dim=dim)


def fix_tensor_dataset(x):
    x = nested_stack(x)
    return x.permute((-1, *range(len(x.size()) - 1)))


def fix_string_batch(x):
    x = np.array(x)
    return x.transpose((-1, *range(len(x.shape) - 1)))


AnyParameters = Union[nn.Parameter, nn.ParameterList, nn.ParameterDict, nn.Module, nn.ModuleDict]
AnyParameters = Union[AnyParameters, Iterable[AnyParameters]]


def iter_parameters(*data: AnyParameters):
    def _iter_parameters(*_data: AnyParameters):
        for _d in _data:
            if isinstance(_d, nn.Parameter):
                yield _d
            elif isinstance(_d, list) or isinstance(_d, tuple) or isinstance(_d, set):
                for _p in _iter_parameters(*_d):
                    yield _p
            else:
                for _p in _d.parameters():
                    yield _p

    seen = set()
    for param in _iter_parameters(data):
        if param not in seen:
            seen.add(param)
            yield param


def enum_parameters(*data: AnyParameters):
    return enumerate(iter_parameters(data))


def list_parameters(*data: AnyParameters):
    return list(iter_parameters(data))


def freeze(data: AnyParameters):
    for param in iter_parameters(data):
        param.requires_grad = False
    return data


def unfreeze(data: AnyParameters):
    for param in iter_parameters(data):
        param.requires_grad = True
    return data


def freeze_all(*data: AnyParameters):
    for param in iter_parameters(*data):
        param.requires_grad = False


def unfreeze_all(*data: AnyParameters):
    for param in iter_parameters(*data):
        param.requires_grad = True


def is_frozen(*data: AnyParameters):
    has_true_grad, has_false_grad = False, False
    for param in iter_parameters(*data):
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


def count_parameters(*data: AnyParameters):
    return sum(param.numel() for param in iter_parameters(data))


if __name__ == '__main__':
    a = nn.Parameter(torch.rand(2, 3))
    b = nn.ModuleDict({'a': nn.Linear(2, 2)})
    c = nn.Linear(2, 2)
    q = nn.ParameterDict({'a': nn.Linear(2, 2)})

    dat = [a, [b, c], [[[([q],)]]], a]
    print(is_frozen(dat))

    freeze_all(dat)
    print(is_frozen(dat))

    unfreeze_all(b, c)
    print(is_frozen(dat))

    for p in enum_parameters(dat):
        print(p)
    print('\n', count_parameters(dat))

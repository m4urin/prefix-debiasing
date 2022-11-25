from typing import Union

import torch
from torch import nn


def pretty_number(i: Union[int, float, torch.Tensor], decimals=1) -> str:
    if isinstance(i, int) and -10000 < i < 10000:
        return str(i)
    if isinstance(i, torch.Tensor):
        i = i.item()
    i = float(i)
    if i == 0:
        return '0'
    for e in ['inf', '-inf']:
        if i == float(e):
            return e
    negative = False
    if i < 0:
        negative = True
        i = -i
    if 1 <= i < 10:
        result = f'{round(i, decimals)}'
    elif i >= 10:
        p = 0
        while i >= 10:
            i /= 10
            p += 1
        result = f'{round(i, decimals)}e{p}'
    else:
        p = 0
        while i < 1:
            i *= 10
            p += 1
        result = f'{round(i, decimals)}e-{p}'

    if negative:
        result = f"-{result}"

    return result.replace(f".{''.join(['0' for _ in range(decimals)])}e", "e")


def pretty_time(seconds: Union[int, float]) -> str:
    t = float(seconds)
    neg = ''
    if t < 0:
        neg = '-'
        t = -t
    if t < 0.95:
        return f'{neg}{round(t * 1000)} ms'
    for div, bound, name in [(1, 60, 'sec'), (60, 60, 'min'), (60, 24, 'hours'),
                             (24, 365.2495, 'days'), (365.25, 1000, 'years')]:
        t /= div
        if 0.95 <= t < 9.95:
            return f'{neg}{round(t, 1)} {name}'
        if 9.95 <= t < bound:
            return f'{neg}{round(t)} {name}'
    return f'{neg}{pretty_number(t, 0)} years'


def pretty_bytes(n_bytes: int) -> str:
    setting = [(' bytes', 0), ('KB', 0), ('MB', 0), ('GB', 1)]
    for i, (label, round_decimals) in enumerate(setting):
        size = n_bytes / (1024 ** i)
        if size < 999.9:
            size = round(size, round_decimals)
            if round_decimals == 0:
                size = int(size)
            return f'{size}{label}'
    return f"{pretty_number(n_bytes / (1024 ** 4), 2)} TB"


def print_title(title: str):
    title = f'# {title} #'
    border = '#' * len(title)
    print(f'\n{border}\n{title}\n{border}')


def small_str(obj: Union[int, float, list, dict, set, nn.Module, nn.ModuleDict]) -> str:
    if isinstance(obj, int) or isinstance(obj, float):
        return f"{pretty_number(obj)} ({obj.__class__.__name__})"
    if isinstance(obj, str):
        return f"'{obj[:100]}' (str)"
    if isinstance(obj, list):
        name = f"[{obj[0].__class__.__name__}]" if len(obj) > 0 else ''
        return f"list{name}({len(obj)})"
    if isinstance(obj, set):
        name = f"[{list(obj)[0].__class__.__name__}]" if len(obj) > 0 else ''
        return f"set{name}({len(obj)})"
    if isinstance(obj, dict):
        return f"dict({len(obj)})"
    if isinstance(obj, nn.ModuleDict):
        keys = ', '.join([f"{k}={small_str(v)}" for k, v in obj.items()])
        return f"ModuleDict({keys})"
    return obj.__class__.__name__


def explore_dict(a_dict: dict) -> str:
    ignore = [int, float, str, list, dict, set, nn.Module, nn.ModuleDict]

    def _explore_dict(d: dict) -> list[str]:
        sub = []
        for k in d.keys():
            obj = d[k]
            sub.append(f'{k} = {small_str(obj)}')
            if hasattr(obj, '__dict__') and not any(isinstance(obj, cls) for cls in ignore):
                obj = obj.__dict__
            elif hasattr(obj, 'to_dict') and not any(isinstance(obj, cls) for cls in ignore):
                obj = obj.to_dict()
            if isinstance(obj, dict):
                sub.extend([f'\t{s}' for s in _explore_dict(obj)])
        return sub

    return '\n'.join(_explore_dict(a_dict))


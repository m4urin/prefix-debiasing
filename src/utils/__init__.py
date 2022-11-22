import collections
import numbers
from typing import Union

import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


class BatchIterator:
    def __init__(self, data: list, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.iterations = len(data) // batch_size
        if len(data) % batch_size != 0:
            self.iterations += 1

    def __len__(self):
        return self.iterations

    def __getitem__(self, i):
        if i < 0 or i >= self.iterations:
            raise IndexError('list index out of range')
        return self.data[i * self.batch_size:(i + 1) * self.batch_size]

    def __iter__(self):
        for i in range(self.iterations):
            yield self[i]


def batched(data: list, batch_size: int):
    return BatchIterator(data, batch_size)


def tbatched(data: list, batch_size: int, desc: str = None):
    return tqdm(batched(data, batch_size), desc=desc)


def sig_notation(data: Union[int, float, dict], decimals=1):
    if isinstance(data, dict):
        return {k: sig_notation(v) for k, v in data.items()}
    else:
        return sig_notation_numeric(data, decimals)


def sig_notation_numeric(i, decimals=1):
    if i == 0:
        return str(i)
    if i == 0 or (isinstance(i, int) and 0 <= i < 100):
        return str(i)
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


def pretty_time(i):
    if i < 0.95:
        return f'{round(i * 1000)} ms'

    if 0.95 <= i < 9.95:
        return f'{round(i, 1)} sec'
    if 9.95 <= i < 60:
        return f'{round(i)} sec'

    i /= 60
    if 1 <= i < 9.95:
        return f'{round(i, 1)} min'
    if 9.95 <= i < 60:
        return f'{round(i)} min'

    i /= 60
    if 1 <= i < 9.95:
        return f'{round(i, 1)} hours'
    if 9.95 <= i < 24:
        return f'{round(i)} hours'

    i /= 24
    if 1 <= i < 9.95:
        return f'{round(i, 1)} days'
    if 9.95 <= i < 365.25:
        return f'{round(i)} days'

    i /= 365.25
    if 1 <= i < 9.95:
        return f'{round(i, 1)} years'
    return f'{round(i)} years'


def pretty_bytes(n_bytes: int) -> str:
    setting = [(' bytes', 0), ('KB', 0), ('MB', 0), ('GB', 1), ('TB', 2)]
    for i in range(len(setting) - 1):
        size = n_bytes / (1024 ** i)
        if size < 999.9:
            label, round_decimals = setting[i]
            size = round(size, round_decimals)
            if round_decimals == 0:
                size = int(size)
            return f'{size}{label}'
    raise ValueError('This should not happen!')


def dataframe_from_dicts(d: list[dict]):
    d = stack_dicts(d)
    d = {k.replace('_', ' '): v for k, v in d.items()}
    return pd.DataFrame.from_dict(d)


def stack_dicts(all_dicts: list[dict]):
    if len(all_dicts) == 0:
        return {}

    d_result = {}
    for d in all_dicts:
        for k in d.keys():
            d_result[k] = []

    for d in all_dicts:
        for k in d_result.keys():
            if k in d:
                d_result[k].append(d[k])
            else:
                d_result[k].append(None)
    return d_result


def flatten_dict(all_dicts, parent_key='', sep='_'):
    items = []
    for k, v in all_dicts.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def plot_scalars(results, folder, skip_warmup=0):
    data = [flatten_dict(e.train_result.scalars) for e in results]
    data = stack_dicts(data)
    for scalar_name, scalar_data in data.items():
        plt.title(scalar_name)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        for i, x in enumerate(scalar_data):
            if x is not None:
                plt.plot(x[skip_warmup:], label=results[i].config.model_name + '_' + results[i].config.model_type)
        plt.legend()
        folder.write_file(scalar_name + '.png', plt)
        plt.clf()


def try_int(value):
    if value is None or isinstance(value, int):
        return value
    try:
        return int(value)
    except ValueError:
        return value


def try_float(value):
    if value is None or isinstance(value, float):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def get_one_of_attributes(obj, attributes: list[str]):
    for attr in attributes:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    raise AttributeError(f"'obj ' has none of the following attributes: {attributes}, "
                         f"it has the following attributes: {list(obj.__dict__.keys())}")


def split_words(text: str, separator: str):
    return [s for s in text.split(separator) if len(s) > 0]


def print_title(title):
    title = f'## {title} ##'
    border = '#' * len(title)
    print(f'\n{border}\n{title}\n{border}')


def explore_dict(d: dict):
    return '\n'.join(sub_dict_keys(d))


def sub_dict_keys(d: dict, depth=0) -> list[str]:
    sub = []
    for k in d.keys():
        ignore = [nn.ModuleDict, numbers.Number]
        obj = d[k]
        if isinstance(obj, list) and len(obj) > 0:
            _type = f'list[{type(obj[0]).__name__}]'
        else:
            _type = type(obj).__name__
        if hasattr(obj, '__dict__') and not any(isinstance(obj, cls) for cls in ignore):
            obj = obj.__dict__

        if isinstance(obj, dict) and len(obj) > 0:
            sub.append(f'{k} ({_type}):')
            sub.extend([f'   {s}' for s in sub_dict_keys(obj, depth+1)])
        else:
            default_len = depth * 3 + len(k) + len(_type)
            max_len = 80 - default_len
            if isinstance(obj, numbers.Number):
                obj_repr = sig_notation(obj)
            elif isinstance(obj, nn.ModuleDict):
                obj_repr = f'Keys({list(obj.keys())})'
            elif isinstance(obj, str):
                obj_repr = f"'{obj}'"
            else:
                obj_repr = str(obj)
            if len(obj_repr) > max_len:
                obj_repr = obj_repr[:max_len-3] + '...'

            sub.append(f'{k} ({_type}): {obj_repr}')

    return sub

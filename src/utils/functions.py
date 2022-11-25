from math import ceil
from typing import Any, Iterable, Union, TypeVar, Generic

import pandas as pd
from matplotlib import pyplot as plt
from transformers.utils import flatten_dict

from tqdm import tqdm, trange


def _nested_loop(list1: Union[Iterable[Any], Any],
                 *other_lists: Union[Iterable[Any], Any]):
    """
    Parameters:
        list1: Iterable, or Any object (these will be interpreted as a list with a length of 1).
        other_lists: other Iterables, or Any objects (these will be interpreted as lists with a length of 1).
    Returns:
        A generator of tuples over the lists.
    """
    if not isinstance(list1, Iterable) or isinstance(list1, str):
        list1 = (list1,)
    if len(other_lists) == 0:
        for i in list1:
            yield i
    elif len(other_lists) == 1:
        list2 = other_lists[0]
        if not isinstance(list2, Iterable) or isinstance(list2, str):
            list2 = (list2,)
        for i in list1:
            for j in list2:
                yield i, j
    else:
        for i in list1:
            for j in _nested_loop(*other_lists):
                yield i, *j


def nested_loop(list1: Union[Iterable[Any], Any],
                *other_lists: Union[Iterable[Any], Any],
                batch_size: int = None,
                progress_bar: Union[bool, Any] = False):
    """
    Parameters:
        list1: Iterable, or Any object (these will be interpreted as a list with a length of 1).
        other_lists: other Iterables, or Any objects (these will be interpreted as lists with a length of 1).
        batch_size: batch_size > 0, returns list instances of the tuples with corresponding length.
        progress_bar: Use tqdm if not None, if arg is a string, it will be used as a tqdm descripion.
    Returns:
        A generator of tuples over the lists.
    """
    use_batched = batch_size is not None
    desc = str(progress_bar) if progress_bar and not isinstance(progress_bar, bool) else None

    if use_batched:
        assert isinstance(batch_size, int) and batch_size > 0, "'batch_size' must be an integer greater or equal to 1."
        data = list(_nested_loop(list1, *other_lists))
        if progress_bar:
            iterator = trange(ceil(len(data) / batch_size), desc=desc)
        else:
            iterator = range(ceil(len(data) / batch_size))
        for i in iterator:
            yield data[i * batch_size:(i + 1) * batch_size]
    else:
        if progress_bar:
            total_steps = 1
            for a_list in (list1, *other_lists):
                if not isinstance(a_list, Iterable) or isinstance(a_list, str):
                    a_list = (a_list,)
                total_steps *= len(a_list)
            for d in tqdm(_nested_loop(list1, *other_lists), total=total_steps, desc=desc):
                yield d
        else:
            for d in _nested_loop(list1, *other_lists):
                yield d


def nested_loop_dict(a_dict: dict,
                     batch_size: int = None,
                     progress_bar: Union[bool, Any] = False):
    """
    Parameters:
        a_dict: Values are Iterable, or Any object (these will be interpreted as a list with a length of 1).
        batch_size: batch_size > 0, returns list instances of the tuples with corresponding length.
        progress_bar: Use tqdm if not None, if arg is a string, it will be used as a tqdm descripion.
    Returns:
        A generator of tuples over the lists/objects in the dictionary.
    """
    if batch_size is None:
        for data in nested_loop(*a_dict.values(), progress_bar=progress_bar):
            yield dict(zip(a_dict.keys(), data))
    else:
        n = len(a_dict.keys())
        for data in nested_loop(*a_dict.values(), batch_size=batch_size, progress_bar=progress_bar):
            data = [[d[j] for d in data] for j in range(n)]
            yield dict(zip(a_dict.keys(), data))


def convert(value, cls, default=None):
    try:
        return cls(value)
    except ValueError:
        return default


def get_one_of_attributes(obj: Any, attributes: Union[Iterable[str], str], *more_attributes: str):
    all_attr = (attributes, *more_attributes) if isinstance(attributes, str) else (*attributes, *more_attributes)
    for attr in all_attr:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    raise AttributeError(f"'obj' has none of the following attributes: {all_attr}, "
                         f"it has the following attributes: {list(obj.__dict__.keys())}")


def stack_dicts(dict1: Union[Iterable[dict], dict], *more_dicts: dict, delimiter='.'):
    all_dicts = (dict1, *more_dicts) if isinstance(dict1, dict) else (*dict1, *more_dicts)
    all_dicts = [flatten_dict(d, delimiter=delimiter) for d in all_dicts]
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


def dataframe_from_dicts(dict1: Union[list[dict], dict], *more_dicts: dict):
    d = (dict1, *more_dicts) if isinstance(dict1, dict) else (*dict1, *more_dicts)
    delimiter = '.'
    d = stack_dicts(d, delimiter=delimiter)
    d = {k.replace(delimiter, ' '): v for k, v in d.items()}
    return pd.DataFrame.from_dict(d)


def plot_scalars(results, title, folder, skip_warmup=0):
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    data = flatten_dict(results.train_info.scalars)
    for scalar_name, x in data.items():
        plt.plot(x[skip_warmup:], label=scalar_name)
    plt.legend()
    folder.write_file(title + '.png', plt)
    plt.clf()



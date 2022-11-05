import os
import json
import shutil
import collections
from pathlib import Path

import torch
import datasets

from tqdm import tqdm
from matplotlib import pyplot as plt
from datasets import Dataset

import numpy as np
import pandas as pd


class IOFile:
    def __init__(self, path: str):
        self.path = path
        _, self.file_name = os.path.split(path)
        self.name, _ = os.path.splitext(self.file_name)

    def exists(self):
        return os.path.exists(self.path)

    def write(self, data):
        raise NotImplementedError('Not implemented yet')

    def read(self):
        raise NotImplementedError('Not implemented yet')

    def __hash__(self):
        return hash(self.file_name)

    def __eq__(self, other):
        if isinstance(other, IOFile):
            return other.file_name == self.file_name
        if isinstance(other, str):
            return other == self.file_name
        return False

    def __str__(self):
        return f"{self.file_name} ({pretty_bytes(self.n_bytes)})"

    def __repr__(self):
        return str(self)

    @property
    def n_bytes(self) -> int:
        return os.stat(self.path).st_size

    @staticmethod
    def from_path(path: str):
        _, ext = os.path.splitext(path)
        constructors = {
            '.json': JsonFile,
            '.pt': TorchFile,
            '.txt': TextFile,
            '.csv': DataFrameFile,
            '.parquet': DatasetFile,
            '.png': ImageFile,
            '.jpg': CompressedImageFile,
        }
        if ext not in constructors:
            return IOFile(path)
        f: IOFile = constructors[ext](path)
        return f


class IOFolder:
    def __init__(self, path: str, clear_data: bool = False):
        if clear_data:
            shutil.rmtree(path)
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)

        self.root = path

        self.files: dict[str, IOFile] = {f: IOFile.from_path(self.from_root(f)) for f in os.listdir(path)
                                         if os.path.isfile(self.from_root(f))}
        self.folders: dict[str, IOFolder] = {f: IOFolder(self.from_root(f)) for f in os.listdir(path)
                                             if os.path.isdir(self.from_root(f))}

    def from_root(self, path: str) -> str:
        return os.path.join(self.root, path)

    def get_folder(self, folder_name: str, create_if_not_exist=False):
        if create_if_not_exist and folder_name not in self.folders:
            # Create a new directory because it does not exist
            self.folders[folder_name] = IOFolder(self.from_root(folder_name))
        if folder_name not in self.folders:
            raise FileExistsError(f"Folder '{folder_name}' does not exist in '{self.root}'.")
        return self.folders[folder_name]

    def remove_folder(self, folder_name: str):
        if folder_name not in self.folders:
            raise FileExistsError(f"Folder '{folder_name}' does not exist in '{self.root}'.")
        root = os.path.join(self.root, folder_name)
        shutil.rmtree(root)
        del self.folders[folder_name]

    def get_file(self, file_name: str, create_if_not_exist=None) -> IOFile:
        if create_if_not_exist is not None and file_name not in self.files:
            # Create a new directory because it does not exist
            self.files[file_name] = IOFile.from_path(self.from_root(file_name))
        if file_name not in self.files:
            raise FileExistsError(f"File '{file_name}' does not exist in '{self.root}'.")
        return self.files[file_name]

    def remove_file(self, file_name: str):
        if file_name not in self.files:
            raise FileExistsError(f"File '{file_name}' does not exist in '{self.root}'.")
        os.remove(self.from_root(file_name))
        del self.files[file_name]

    def write_file(self, file_name: str, data):
        if file_name not in self.files:
            self.files[file_name] = IOFile.from_path(self.from_root(file_name))
        self.files[file_name].write(data)

    def read_file(self, file_name: str):
        if file_name not in self.files:
            raise FileExistsError(f"File '{file_name}' does not exist in '{self.root}'.")
        return self.files[file_name].read()

    def get_files(self, extension: str = None):
        if extension is None:
            return list(self.files.values())
        else:
            return [f for k, f in self.files.items() if k.endswith(extension)]

    def get_folders(self):
        return list(self.folders.values())

    def file_exists(self, file_name, recursive=False) -> bool:
        return os.path.exists(self.from_root(file_name)) or \
               (recursive and any(folder.file_exists(file_name, recursive) for folder in self.folders.values()))

    def __getitem__(self, path):
        if isinstance(path, tuple) or isinstance(path, list):
            path: str = os.path.join(*path)
        path: list[str] = [s for s in path.replace('\\', '/').split('/') if len(s) > 0]
        if len(path) == 0:
            # root
            return self
        if len(path) == 1:
            if path[0] in self.folders:
                return self.folders[path[0]]
            elif path[0] in self.files:
                return self.files[path[0]]
            else:
                raise FileNotFoundError(f"'{path[0]}' does not exist in '{self.root}'.")
        if len(path) > 1:
            return self.folders[path[0]][path[1:]]

    def __str__(self):
        root = '/'.join(f'{self.root}/'.replace('//', '/').replace('\\', '/').split('/')[-2:])
        root += f' ({pretty_bytes(self.n_bytes)})'
        if len(self) == 0:
            return root
        indent = '   '
        root += f'\n{indent}'
        sub_files = [str(f) for f in self.files.values()]
        sub_directories = sub_files + [str(f) for f in self.folders.values()]
        sub_directories = '\n'.join(sub_directories).replace('\n', '\n' + indent)
        return root + sub_directories

    def __len__(self):
        return len(self.files) + len(self.folders) + sum(len(f) for f in self.folders.values())

    @property
    def n_bytes(self) -> int:
        return sum(f.n_bytes for f in self.files.values()) + sum(f.n_bytes for f in self.folders.values())

    def __repr__(self):
        return f"Folder('{self.root}', n_files={len(self)}, memory={pretty_bytes(self.n_bytes)})"


class JsonFile(IOFile):
    def write(self, a_dict):
        with open(self.path, "w") as outfile:
            json.dump(a_dict, outfile, indent=3)

    def read(self):
        with open(self.path, 'r') as f:
            return dict(json.load(f))


class TorchFile(IOFile):
    def write(self, data):
        torch.save(data, self.path)

    def read(self):
        return torch.load(self.path)


class TextFile(IOFile):
    def write(self, txt):
        with open(self.path, "w") as outfile:
            if isinstance(txt, list):
                outfile.write('\n'.join(txt))
            else:
                outfile.write(txt)

    def read(self) -> list[str]:
        with open(self.path, encoding="utf-8") as file:
            return [line.rstrip() for line in file.readlines()]


class DataFrameFile(IOFile):
    def write(self, df):
        df.to_csv(self.path, index=False)

    def read(self):
        return pd.read_csv(self.path)


class DatasetFile(IOFile):
    def write(self, dataset):
        dataset.to_parquet(self.path)

    def read(self):
        datasets.logging.set_verbosity(datasets.logging.CRITICAL)
        return Dataset.from_parquet(self.path)


class ImageFile(IOFile):
    def write(self, _plt):
        _plt.savefig(self.path)

    def read(self):
        raise Exception('Images should not be read.')


class CompressedImageFile(IOFile):
    def write(self, _plt):
        _plt.savefig(self.path)

    def read(self):
        raise Exception('Images should not be read.')


class _BatchIterator:
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
    return _BatchIterator(data, batch_size)


def tbatched(data: list, batch_size: int, desc: str = None):
    return tqdm(batched(data, batch_size), desc=desc)


def sig_notation(i, decimals=1):
    if 1 <= i < 10:
        return f'{round(i, decimals)}'
    elif i >= 10:
        p = 0
        while i >= 10:
            i /= 10
            p += 1
        return f'{round(i, decimals)}e{p}'
    else:
        p = 0
        while i < 1:
            i *= 10
            p += 1
        return f'{round(i, decimals)}e-{p}'


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
    return pd.DataFrame.from_dict(stack_dicts(d))


def stack_dicts(all_dicts: list[dict]):
    if len(all_dicts) == 0:
        return {}
    if len(all_dicts) == 1:
        return all_dicts[0]

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


def repeat_stacked(x: torch.Tensor, n: int):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.size())))


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


def deep_tensor(tensor_list) -> torch.Tensor:
    if isinstance(tensor_list, torch.Tensor):
        return tensor_list
    if isinstance(tensor_list, list):
        return torch.stack([deep_tensor(t) for t in tensor_list])


def fix_tensor_dataset(x):
    x = deep_tensor(x)
    return x.permute((-1, *range(len(x.size()) - 1)))


def fix_string_dataset(x):
    x = np.array(x)
    return x.transpose((-1, *range(len(x.shape) - 1)))


def print_title(title):
    title = f'## {title} ##'
    border = '#' * len(title)
    print(f'\n{border}\n{title}\n{border}')


""" GLOBALS """
DATA_DIR = IOFolder(os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parents[0], 'data'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

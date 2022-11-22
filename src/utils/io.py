import json
import os
import shutil
from pathlib import Path

import pandas as pd
import datasets
import torch
from datasets import Dataset

from src.utils import pretty_bytes


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
            '.csv': CSVDataFrameFile,
            '.tsv': TSVDataFrameFile,
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


class CSVDataFrameFile(IOFile):
    def write(self, df):
        df.to_csv(self.path, index=False)

    def read(self):
        return pd.read_csv(self.path)


class TSVDataFrameFile(IOFile):
    def write(self, df):
        df.to_csv(self.path, index=False, sep='\t')

    def read(self):
        return pd.read_csv(self.path, sep='\t')


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


DATA_DIR = IOFolder(os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parents[1], 'data'))

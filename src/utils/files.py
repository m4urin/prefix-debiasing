import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import datasets
import torch
from datasets import Dataset

from src.utils.printing import pretty_bytes


class IOFile:
    def __init__(self, path: str, parent):
        self.full_path = path
        self.folder_path, self.file_name = os.path.split(path)
        self.name, self.ext = os.path.splitext(self.file_name)
        self.parent: IOFolder = parent

    def exists(self):
        return os.path.exists(self.full_path)

    def write(self, data):
        raise NotImplementedError('Not implemented yet')

    def read(self):
        raise NotImplementedError('Not implemented yet')

    def delete(self):
        del self.parent.files[self.file_name]
        os.remove(self.full_path)

    def __hash__(self):
        return hash(self.full_path)

    def __eq__(self, other):
        if isinstance(other, IOFile):
            return self.full_path == self.full_path
        return False

    def __str__(self):
        return f"{self.name}{self.ext} ({pretty_bytes(self.n_bytes)})"

    def __repr__(self):
        return str(self)

    @property
    def n_bytes(self) -> int:
        return os.stat(self.full_path).st_size

    @staticmethod
    def from_path(path: str, parent):
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
            f = IOFile(path, parent)
        else:
            f: IOFile = constructors[ext](path, parent)
        return f


class IOFolder:
    def __init__(self, path: str, clear_data: bool = False, parent=None):
        if clear_data:
            shutil.rmtree(path)
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
        self.root = path
        self.name = os.path.split(path)[-1]
        self.parent: IOFolder = parent
        self.files: dict[str, IOFile] = {f: IOFile.from_path(self.from_root(f), self) for f in os.listdir(path)
                                         if os.path.isfile(self.from_root(f)) and not f.startswith('.')}
        self.folders: dict[str, IOFolder] = {f: IOFolder(self.from_root(f), parent=self) for f in os.listdir(path)
                                             if os.path.isdir(self.from_root(f)) and not f.startswith('.')}

    def from_root(self, path: str) -> str:
        return os.path.join(self.root, path)

    def get_folder(self, folder_name: str, create_if_not_exists=False):
        folder_name, *sub_dirs = [sub for sub in os.path.normpath(folder_name).split(os.path.sep) if len(sub) > 0]
        if create_if_not_exists and folder_name not in self.folders:
            # Create a new directory because it does not exist
            self.folders[folder_name] = IOFolder(self.from_root(folder_name), parent=self)
        if folder_name not in self.folders:
            raise FileExistsError(f"Folder '{folder_name}' does not exist in '{self.root}'.")
        if len(sub_dirs) > 0:
            return self.folders[folder_name].get_folder(os.path.join(*sub_dirs), create_if_not_exists)
        else:
            return self.folders[folder_name]

    def delete(self):
        del self.parent.folders[self.name]
        shutil.rmtree(self.root)

    def get_file(self, file_name: str) -> IOFile:
        file_name, *sub_dirs = [sub for sub in os.path.normpath(file_name).split(os.path.sep) if len(sub) > 0]
        if len(sub_dirs) > 0:
            # file_name is a folder
            folder = self.get_folder(file_name, False)
            return folder.get_file(os.path.join(*sub_dirs))
        else:
            if file_name not in self.files:
                raise FileExistsError(f"File '{file_name}' does not exist in '{self.root}'.")
            return self.files[file_name]

    def write_file(self, file_name: str, data):
        file_name, *sub_dirs = [sub for sub in os.path.normpath(file_name).split(os.path.sep) if len(sub) > 0]
        if len(sub_dirs) > 0:
            # file_name is a folder
            folder = self.get_folder(file_name, False)
            folder.write_file(os.path.join(*sub_dirs), data)
        else:
            if file_name not in self.files:
                self.files[file_name] = IOFile.from_path(self.from_root(file_name), self)
            self.files[file_name].write(data)

    def read_file(self, file_name: str):
        return self.get_file(file_name).read()

    def get_all_files(self, extension: str = None):
        if extension is None:
            return list(self.files.values())
        else:
            return [f for k, f in self.files.items() if k.endswith(extension)]

    def get_all_folders(self):
        return list(self.folders.values())

    def file_exists(self, file_name, recursive=False) -> bool:
        return os.path.exists(self.from_root(file_name)) or \
               (recursive and any(folder.file_exists(file_name, recursive) for folder in self.folders.values()))

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
        with open(self.full_path, "w") as outfile:
            json.dump(a_dict, outfile, indent=3)

    def read(self):
        with open(self.full_path, 'r') as f:
            return dict(json.load(f))


class TorchFile(IOFile):
    def write(self, data):
        torch.save(data, self.full_path)

    def read(self):
        return torch.load(self.full_path)


class TextFile(IOFile):
    def write(self, txt):
        with open(self.full_path, "w") as outfile:
            if isinstance(txt, list):
                outfile.write('\n'.join(txt))
            else:
                outfile.write(txt)

    def read(self) -> list[str]:
        with open(self.full_path, encoding="utf-8") as file:
            return [line for line in [line.rstrip() for line in file.readlines()] if len(line) > 0]


class CSVDataFrameFile(IOFile):
    def write(self, df):
        df.to_csv(self.full_path, index=False)

    def read(self):
        return pd.read_csv(self.full_path)


class TSVDataFrameFile(IOFile):
    def write(self, df):
        df.to_csv(self.full_path, index=False, sep='\t')

    def read(self):
        return pd.read_csv(self.full_path, sep='\t')


class DatasetFile(IOFile):
    def write(self, dataset):
        dataset.to_parquet(self.full_path)

    def read(self):
        datasets.logging.set_verbosity(datasets.logging.CRITICAL)
        return Dataset.from_parquet(self.full_path)


class ImageFile(IOFile):
    def write(self, _plt):
        _plt.savefig(self.full_path)

    def read(self):
        raise Exception('Images should not be read.')


class CompressedImageFile(IOFile):
    def write(self, _plt):
        _plt.savefig(self.full_path)

    def read(self):
        raise Exception('Images should not be read.')


DATA_DIR = IOFolder(str(Path(os.path.dirname(os.path.abspath(__file__))).parents[1]))


def get_folder(path: str = None, create_if_not_exists=False):
    if path is None:
        return DATA_DIR
    return DATA_DIR.get_folder(path, create_if_not_exists)


def get_file(path: str):
    return DATA_DIR.get_file(path)


def read_file(path: str):
    return get_file(path).read()


def write_file(path: str, data: Any):
    dir_path, file_name = os.path.split(path)
    if dir_path == '':
        folder = DATA_DIR
    else:
        folder = get_folder(dir_path, create_if_not_exists=True)
    folder.write_file(file_name, data)


def get_all_files(path: str = None, extension: str = None):
    return get_folder(path).get_all_files(extension)


def exists(path: str):
    return os.path.exists(DATA_DIR.from_root(path))


if __name__ == '__main__':
    print(DATA_DIR)

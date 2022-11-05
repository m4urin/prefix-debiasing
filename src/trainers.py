from time import time

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from diffusers import get_linear_schedule_with_warmup
from transformers import AdamW

from src.utils import sig_notation, fix_string_dataset, DATA_DIR, DEVICE
from src.language_models import MaskedLanguageModel, Config


class TrainResult:
    def __init__(self,
                 training_completed: bool = False,
                 total_training_time: int = 0,
                 minimal_loss: float = float('INF'),
                 scalars: dict = None,
                 parameters=None,
                 n_parameters: int = 0
                 ):

        self.training_completed: bool = training_completed
        self.total_training_time: int = total_training_time
        self.minimal_loss: float = minimal_loss
        self.scalars: dict = scalars if scalars is not None else {}
        self.parameters = parameters
        self.n_parameters = n_parameters

    def add_scalar(self, name: str, scalar: float):
        names = name.split('/')
        sub = self.scalars
        for n in names[:-1]:
            if n not in sub:
                sub[n] = {}
            sub = sub[n]
        if names[-1] not in sub:
            sub[names[-1]] = []
        sub[names[-1]].append(scalar)

    def to_dict(self):
        return {
            'training_completed': self.training_completed,
            'total_training_time': self.total_training_time,
            'minimal_loss': self.minimal_loss,
            'scalars': self.scalars,
            'parameters': self.parameters,
            'n_parameters': self.n_parameters
        }


class Trainer:
    def prepare_data(self, config: Config) -> dict:
        raise NotImplementedError('Not implemented yet.')

    def calc_total_steps(self, train_data: dict, config: Config) -> int:
        raise NotImplementedError('Not implemented yet.')

    def get_data_loader(self, epoch: int, train_data: dict, config: Config):
        raise NotImplementedError('Not implemented yet.')

    def calculate_loss(self, model: MaskedLanguageModel, x,
                       train_data: dict, train_result: TrainResult) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet.')

    def train(self, model: MaskedLanguageModel) -> TrainResult:
        start_time = time()

        config = model.config
        train_data = self.prepare_data(config)
        total_steps = self.calc_total_steps(train_data, config)

        optimizer = AdamW(params=model.get_parameters().parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=total_steps)

        train_result = TrainResult()

        progress_bar = tqdm(total=total_steps)
        last_progress_bar_update = time()
        for epoch in range(config.epochs):
            for iteration, x in enumerate(self.get_data_loader(epoch, train_data, config)):
                optimizer.zero_grad()
                loss = self.calculate_loss(model, x, train_data, train_result)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss = loss.item()

                train_result.minimal_loss = min(train_result.minimal_loss, loss)
                train_result.add_scalar('loss/total', loss)

                if iteration % 4 == 0 and time() - last_progress_bar_update > 4:
                    progress_bar.set_description(
                        desc=f"Training.. "
                             f"loss={sig_notation(train_result.minimal_loss, 2)}, "
                             f"epoch {epoch + 1}/{config.epochs}")
                    last_progress_bar_update = time()
                progress_bar.update(1)
        train_result.training_completed = True
        train_result.total_training_time = round(time() - start_time, 1)
        train_result.parameters = model.get_parameters()
        train_result.n_parameters = model.n_parameters
        return train_result

    @staticmethod
    def from_config(config: Config):
        if config.loss_function == 'kaneko':
            return OrthogonalTrainer()
        return None


class OrthogonalTrainer(Trainer):
    def prepare_data(self, config: Config) -> dict:
        folder = DATA_DIR['train/kaneko']

        v_a = folder[f'va/{config.model_name}.pt'].read().to(DEVICE)
        v_a.requires_grad = False

        attr_embeddings = folder[f'attributes/{config.model_name}.pt'].read()
        attr_embeddings.requires_grad = False

        return {
            'v_a': v_a,
            'dataset_attribute': folder[f'attributes.parquet'].read(),
            'dataset_stereo': folder['stereotypes.parquet'].read(),
            'attr_embeddings': attr_embeddings
        }

    def calc_total_steps(self, train_data: dict, config: Config) -> int:
        bs = config.batch_size // 2
        iterations_per_epoch = len(train_data['dataset_attribute']) // bs
        if len(train_data['dataset_attribute']) // bs != 0:
            iterations_per_epoch += 1
        return config.epochs * iterations_per_epoch

    def get_data_loader(self, epoch: int, train_data: dict, config: Config):
        data_loader_attribute = DataLoader(train_data['dataset_attribute'], batch_size=config.batch_size // 2)
        embedding_loader = iter(DataLoader(train_data['attr_embeddings'], batch_size=config.batch_size // 2))
        data_loader_stereo = iter(DataLoader(train_data['dataset_stereo'].shuffle(seed=config.seed + epoch),
                                             batch_size=config.batch_size // 2))

        for x in data_loader_attribute:
            attr = fix_string_dataset(x['sentences'])  # (bs//2, 3)
            stereo = fix_string_dataset(next(data_loader_stereo)['sentences'])  # (bs//2, 3)
            embeddings = next(embedding_loader).to(DEVICE)
            yield np.concatenate((attr, stereo), axis=0), embeddings

    def calculate_loss(self, model: MaskedLanguageModel, x,
                       train_data: dict, train_result: TrainResult) -> torch.Tensor:
        x, attr_old = x
        bs = len(x)
        embeddings = model.get_span_embeddings(model.tokenize_with_spans(x))  # (bs, n_layers, dim)
        attr_new = embeddings[:bs // 2]  # (bs/2, n_layers, dim)
        target_new = embeddings[bs // 2:]  # (bs/2, n_layers, dim)

        orthogonal_loss = (torch.einsum('ald,tld->atl', train_data['v_a'], target_new) ** 2).mean()
        train_result.add_scalar('loss/orthogonal', orthogonal_loss.item())

        embedding_regularization_loss = 100 * ((attr_new - attr_old) ** 2).mean()
        train_result.add_scalar('loss/embedding_regularization', embedding_regularization_loss.item())

        prefix_regularization = 0
        if model.config.is_prefix():
            prefix_regularization = 10 * model.prefix_embeddings.regularization()
            train_result.add_scalar('loss/prefix_regularization', prefix_regularization.item())

        return orthogonal_loss + embedding_regularization_loss + prefix_regularization

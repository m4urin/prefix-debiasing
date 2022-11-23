from time import time
from typing import Any

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from diffusers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.utils import sig_notation
from src.MLM import MLM, ModelConfig
from src.utils.io import DATA_DIR
from src.utils.pytorch import DEVICE, fix_string_batch, freeze, unfreeze
from src.utils.user_dicts import TrainInfo


class Trainer:
    def prepare_data(self, config: ModelConfig) -> dict:
        raise NotImplementedError('Not implemented yet.')

    def calc_total_steps(self, data: dict, config: ModelConfig) -> int:
        raise NotImplementedError('Not implemented yet.')

    def get_train_loader(self, epoch: int, data: dict, config: ModelConfig):
        raise NotImplementedError('Not implemented yet.')

    def get_test_loader(self, epoch: int, data: dict, config: ModelConfig):
        raise NotImplementedError('Not implemented yet.')

    def calculate_loss(self, model: MLM, batch: Any, data: dict, result: TrainInfo) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet.')

    def train(self, model: MLM) -> TrainInfo:
        raise NotImplementedError('Not implemented yet.')

    @staticmethod
    def run(task: str, model: MLM):
        if task == 'default':
            if model.config.loss_function == 'kaneko':
                trainer = OrthogonalTrainer()
            else:
                raise ValueError(f"Loss function '{model.config.loss_function}' is not supported.")
            return trainer.train(model)
        elif task == 'coreference-resolution':
            return CorefTrainer().train(model)
        else:
            raise ValueError(f"Task '{task}' is not supported.")


class OrthogonalTrainer(Trainer):
    def prepare_data(self, config: ModelConfig) -> dict:
        folder = DATA_DIR['train/kaneko']

        v_a = folder[f'va/{config.model_name}.pt'].read().to(DEVICE)
        v_a.requires_grad = False

        base_model = MLM.from_config(config=ModelConfig(config.model_name, 'base'), task='default').to(DEVICE).eval()
        freeze(base_model)

        return {
            'v_a': v_a,
            'dataset_attribute': folder[f'attributes.parquet'].read(),
            'dataset_stereo': folder['stereotypes.parquet'].read(),
            'base_model': base_model
        }

    def calc_total_steps(self, data: dict, config: ModelConfig) -> int:
        bs = config.batch_size // 2
        iterations_per_epoch = len(data['dataset_attribute']) // bs
        if len(data['dataset_attribute']) % bs != 0:
            iterations_per_epoch += 1
        return config.epochs * iterations_per_epoch

    def get_train_loader(self, epoch: int, data: dict, config: ModelConfig):
        data_loader_attribute = DataLoader(data['dataset_attribute'], batch_size=config.batch_size // 2)
        data_loader_stereo = iter(DataLoader(data['dataset_stereo'].shuffle(seed=config.seed + epoch),
                                             batch_size=config.batch_size // 2))

        for x in data_loader_attribute:
            attr = fix_string_batch(x['sentences'])  # (bs//2, 3)
            stereo = fix_string_batch(next(data_loader_stereo)['sentences'])  # (bs//2, 3)
            yield [''.join(s) for s in attr], stereo

    def get_test_loader(self, epoch: int, data: dict, config: ModelConfig):
        return []

    def calculate_loss(self, model: MLM, batch: Any, data: dict, result: TrainInfo) -> torch.Tensor:
        x_attr, x_targ = batch
        # x: (bs, 3)
        enc_attr = model.tokenize(x_attr)
        enc_targ = model.tokenize_with_spans(x_targ)

        attr_sent_lengths = enc_attr['attention_mask'].sum(dim=-1)

        # bs/2 x (seq_len, n_layers, dim)
        attr_new = [attr_sent[:int(_size.item())] for attr_sent, _size
                    in zip(model.get_hidden_states(enc_attr), attr_sent_lengths)]
        with torch.no_grad():
            attr_old = [attr_sent[:int(_size.item())] for attr_sent, _size
                        in zip(data['base_model'].get_hidden_states(enc_attr).detach(), attr_sent_lengths)]

        # (bs/2, n_layers, dim)
        target_new = model.get_span_embeddings(enc_targ, reduce='first')

        embedding_regularization_loss = [((a1 - a2) ** 2).sum() for a1, a2 in zip(attr_old, attr_new)]
        embedding_regularization_loss = 0.8 * torch.stack(embedding_regularization_loss).sum()
        result.add_scalar(f'loss/embedding_regularization', embedding_regularization_loss.item())

        # (bs/2, n_layers, dim)
        orthogonal_loss = 0.08 * (torch.einsum('ald,tld->atl', data['v_a'], target_new) ** 2).sum()
        result.add_scalar(f'loss/orthogonal', orthogonal_loss.item())

        prefix_regularization = 0
        if model.config.is_prefix():
            prefix_regularization = 0.02 * model.module_dict['prefix_embeddings'].regularization()
            result.add_scalar(f'loss/prefix_regularization', prefix_regularization.item())

        total_loss = orthogonal_loss + embedding_regularization_loss + prefix_regularization
        result.train_loss = min(result.train_loss, total_loss.item())
        result.add_scalar(f'loss/total', total_loss.item())
        return total_loss

    def train(self, model: MLM) -> TrainInfo:
        start_time = time()
        last_progress_bar_update = start_time

        config = model.config
        data = self.prepare_data(config)
        total_steps = self.calc_total_steps(data, config)

        optimizer = AdamW(params=model.parameters_dict.parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=total_steps)

        result = TrainInfo()

        progress_bar = tqdm(total=total_steps)
        for epoch in range(config.epochs):
            model.train()
            for iteration, batch in enumerate(self.get_train_loader(epoch, data, config)):
                optimizer.zero_grad()
                loss = self.calculate_loss(model, batch, data, result)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if iteration % 5 == 0 and time() - last_progress_bar_update > 5:
                    progress_bar.set_description(
                        desc=f"   Training.. "
                             f"train_loss={sig_notation(result.train_loss, 2)}, "
                             f"test_loss={sig_notation(result.test_loss, 2)}, "
                             f"epoch {epoch + 1}/{config.epochs}")
                    last_progress_bar_update = time()

                progress_bar.update(1)

        result.training_completed = True
        result.total_training_time = round(time() - start_time, 1)
        return result


class CorefTrainer(Trainer):
    def __init__(self):
        self.stage = None
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.batch_size = 16
        self.epochs = 4

    def prepare_data(self, config: ModelConfig) -> dict:
        folder = DATA_DIR['train/coref']
        return {
            'train': folder.read_file('coref_train.parquet'),
            'test': folder.read_file('coref_test.parquet')
        }

    def calc_total_steps(self, data: dict, config: ModelConfig) -> int:
        iterations_per_epoch = 0
        for sub in ['train', 'test']:
            iterations_per_epoch += len(data[sub]) // self.batch_size
            if len(data[sub]) % self.batch_size != 0:
                iterations_per_epoch += 1
        return self.epochs * iterations_per_epoch

    def get_train_loader(self, epoch: int, data: dict, config: ModelConfig):
        """
        'sentence': sentence parts
        'subject_idx': subject index
        'label': label
        """
        for x in DataLoader(data['train'].shuffle(seed=epoch), batch_size=self.batch_size):
            yield fix_string_batch(x['sentence']), x['subject_idx'], x['label']

    def get_test_loader(self, epoch: int, data: dict, config: ModelConfig):
        """
        'sentence': sentence parts
        'subject_idx': subject index
        'label': label
        """
        for x in DataLoader(data['test'].shuffle(seed=epoch), batch_size=self.batch_size):
            yield fix_string_batch(x['sentence']), x['subject_idx'], x['label']

    def calculate_loss(self, model: MLM, batch: Any, data: dict, result: TrainInfo) -> torch.Tensor:
        sentences, subject_idx, labels = batch
        labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
        enc = model.tokenize_with_spans(sentences)
        # (bs, 1)
        y_pred = model.get_coref_predictions(enc, subject_idx)
        result.add_scalar(f'{self.stage}/accuracy', 1.0 - torch.abs((y_pred > 0).float() - labels).mean().item())

        loss = self.criterion(y_pred, labels)
        result.add_scalar(f'{self.stage}/loss', loss.item())
        if self.stage == 'train':
            result.train_loss = min(result.train_loss, loss.item())
        else:
            result.test_loss = min(result.test_loss, loss.item())
        return loss

    def train(self, model: MLM) -> TrainInfo:
        start_time = time()
        last_progress_bar_update = start_time - 6

        data = self.prepare_data(model.config)
        total_steps = self.calc_total_steps(data, model.config)

        freeze(model.module_dict)
        unfreeze(model.parameters_dict['coreference-resolution'])
        optimizer = AdamW(params=model.parameters_dict['coreference-resolution'].parameters(), lr=0.0002)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=40,
                                                    num_training_steps=2 + int(total_steps/self.epochs))

        all_losses_train = []
        all_losses_eval = []

        result = TrainInfo()

        progress_bar = tqdm(total=total_steps)
        for epoch in range(self.epochs):

            if epoch == 1:
                unfreeze(model.parameters_dict)
                optimizer = AdamW(params=model.parameters_dict.parameters(), lr=0.00002)
                scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                            num_warmup_steps=100,
                                                            num_training_steps=2 + int(total_steps *
                                                                                       (self.epochs-1) / self.epochs))

            self.stage = 'train'
            model.train()
            for iteration, batch in enumerate(self.get_train_loader(epoch, data, model.config)):
                optimizer.zero_grad()
                loss = self.calculate_loss(model, batch, data, result)
                loss.backward()
                optimizer.step()
                scheduler.step()
                all_losses_train.append(loss.item())

                if iteration % 5 == 0 and time() - last_progress_bar_update > 5:
                    progress_bar.set_description(
                        desc=f"   Training.. "
                             f"train_loss={sig_notation(result.train_loss, 2)}, "
                             f"test_loss={sig_notation(result.test_loss, 2)}, "
                             f"epoch {epoch + 1}/{self.epochs}")
                    last_progress_bar_update = time()

                progress_bar.update(1)

            self.stage = 'test'
            last_progress_bar_update = time() - 6
            with torch.no_grad():
                model.eval()
                for iteration, batch in enumerate(self.get_test_loader(epoch, data, model.config)):
                    loss = self.calculate_loss(model, batch, data, result)
                    all_losses_eval.append(loss.item())

                    if iteration % 5 == 0 and time() - last_progress_bar_update > 5:
                        progress_bar.set_description(
                            desc=f"   Testing.. "
                                 f"train_loss={sig_notation(result.train_loss, 2)}, "
                                 f"test_loss={sig_notation(result.test_loss, 2)}, "
                                 f"epoch {epoch + 1}/{self.epochs}")
                        last_progress_bar_update = time()
                    progress_bar.update(1)

        """
        plt.plot(all_losses_train, label='train')
        plt.plot(all_losses_eval, label='eval')
        plt.legend()
        plt.show()
        plt.clf()
        plt.plot(result.scalars['train']['accuracy'], label='train')
        plt.plot(result.scalars['test']['accuracy'], label='test')
        plt.legend()
        plt.show()
        plt.clf()
        """
        result.training_completed = True
        result.total_training_time = round(time() - start_time, 1)
        return result

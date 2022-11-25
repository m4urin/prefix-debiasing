from math import ceil
from time import time

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from diffusers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.MLM import MLM
from src.utils.config import ModelResult
from src.utils.files import read_file
from src.utils.printing import pretty_number
from src.utils.pytorch import DEVICE, fix_string_batch, freeze, unfreeze


class Trainer:
    def train(self, model: MLM) -> ModelResult:
        raise NotImplementedError('Not implemented yet.')


class OrthogonalTrainer(Trainer):
    def __init__(self):
        self.dataset_attribute: Dataset = read_file('train/kaneko/attributes.parquet')
        self.dataset_stereo: Dataset = read_file('train/kaneko/stereotypes.parquet')

    def train(self, model: MLM) -> ModelResult:
        config = model.config
        result = ModelResult(config)
        start_time = time()
        last_progress_bar_update = start_time - 10
        min_train_loss = float('inf')
        total_steps = config.epochs * ceil(len(self.dataset_attribute) / (config.batch_size // 2))

        optimizer = AdamW(params=model.get_parameters().parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=int(total_steps * 1.04))

        v_a: torch.Tensor = read_file(f'train/kaneko/va/{config.model_name}.pt').to(DEVICE).detach()
        base_model = freeze(MLM.from_config(config.as_base()).train()).to(DEVICE)

        progress_bar = tqdm(total=total_steps)
        for epoch in range(config.epochs):
            model.train()
            data_loader_attribute = DataLoader(self.dataset_attribute.shuffle(seed=config.seed + epoch),
                                               batch_size=config.batch_size // 2)
            data_loader_stereo = iter(DataLoader(self.dataset_stereo.shuffle(seed=config.seed + epoch + 1000),
                                                 batch_size=config.batch_size // 2))

            for iteration, batch in enumerate(data_loader_attribute):
                optimizer.zero_grad()
                # RUN MODELS
                enc_attr = model.tokenize([''.join(s) for s in fix_string_batch(batch['sentences'])])
                with torch.no_grad():
                    # (bs, dim)
                    attr_old_hidden = base_model.get_hidden_states(enc_attr)
                    attr_lengths = enc_attr['attention_mask'].sum(dim=-1).tolist()
                # (bs, dim)
                attr_new_hidden = model.get_hidden_states(enc_attr)

                # (bs, sent_parts)
                enc_stereo = model.tokenize_with_spans(fix_string_batch(next(data_loader_stereo)['sentences']))
                # (bs, n_layers, dim)
                stereo_new_hidden = model.get_span_embeddings(enc_stereo, reduce='first')

                # CALCULATE LOSS
                embedding_regularization_loss = [((a1[:n] - a2[:n]) ** 2).sum() for a1, a2, n
                                                 in zip(attr_old_hidden, attr_new_hidden, attr_lengths)]
                embedding_regularization_loss = 0.8 * torch.stack(embedding_regularization_loss).sum()
                result.add_scalar(f'loss/embedding_regularization', embedding_regularization_loss.item())

                prefix_regularization = 0
                if model.config.is_prefix():
                    prefix_regularization = 0.01 * model.get_parameters()['prefix_embeddings'].regularization()
                    result.add_scalar(f'loss/prefix_regularization', prefix_regularization.item())

                # (bs/2, n_layers, dim)
                orthogonal_loss = 0.2 * (torch.einsum('ald,tld->atl', v_a, stereo_new_hidden) ** 2).sum()
                result.add_scalar(f'loss/orthogonal', orthogonal_loss.item())

                loss = embedding_regularization_loss + orthogonal_loss + prefix_regularization
                min_train_loss = min(min_train_loss, loss.item())
                result.add_scalar(f'loss/total', loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                if time() - last_progress_bar_update > 1:
                    progress_bar.set_description(
                        desc=f"Orthogonal training: "
                             f"epoch {epoch + 1}/{config.epochs}, "
                             f"loss={pretty_number(min_train_loss, 2)}")
                    last_progress_bar_update = time()

                progress_bar.update()

        return result.finish_training(start_time, model.get_parameters())


class CorefTrainer(Trainer):
    def __init__(self):
        self.dataset_train = read_file('train/coref/coref_train.parquet')
        self.dataset_test = read_file('train/coref/coref_test.parquet')

    def train(self, model: MLM) -> ModelResult:
        config = model.config
        result = ModelResult(config)

        start_time = time()
        last_progress_bar_update = start_time - 10

        current_train_step = 0
        total_train_steps = config.epochs * ceil(len(self.dataset_train) / config.batch_size)
        total_test_steps = config.epochs * ceil(len(self.dataset_test) / config.batch_size)
        switch_optim = total_train_steps // 6

        freeze(model.module_dict)
        unfreeze(model.get_parameters()['coreference-resolution'])

        optimizer = AdamW(params=model.get_parameters()['coreference-resolution'].parameters(), lr=10 * config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=int(switch_optim * 1.04))
        criterion = torch.nn.BCEWithLogitsLoss()

        title = 'Train co-ref head only..'
        progress_bar = tqdm(total=total_train_steps + total_test_steps)

        min_loss_train, min_loss_test = float('inf'), float('inf')

        for epoch in range(config.epochs):
            model.train()
            total_correct = 0
            for iteration, x in enumerate(DataLoader(self.dataset_train.shuffle(seed=config.seed + epoch),
                                                     batch_size=config.batch_size)):
                if current_train_step == switch_optim:
                    title = 'Train complete model..'
                    freeze(model.module_dict)
                    unfreeze(model.get_parameters())
                    optimizer = AdamW(params=model.get_parameters().parameters(), lr=config.lr)
                    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                num_warmup_steps=config.num_warmup_steps,
                                                                num_training_steps=int((total_train_steps -
                                                                                        switch_optim) * 1.04))
                optimizer.zero_grad()
                sentences, subject_idx, labels = fix_string_batch(x['sentence']), x['subject_idx'], x['label']
                labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
                enc = model.tokenize_with_spans(sentences)
                y_pred = model.get_coref_predictions(enc, subject_idx)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_correct += torch.abs((y_pred <= 0).float() - labels).sum().item()
                min_loss_train = min(min_loss_train, loss.item())
                result.add_scalar(f'loss/train', loss.item())

                if time() - last_progress_bar_update > 1:
                    progress_bar.set_description(
                        desc=f"{title} "
                             f"epoch {epoch + 1}/{config.epochs}, "
                             f"train_loss={pretty_number(min_loss_train, 2)}, "
                             f"test_loss={pretty_number(min_loss_test, 2)}")
                    last_progress_bar_update = time()

                progress_bar.update()
                current_train_step += 1

            result.add_scalar(f'acc/train', round(total_correct / len(self.dataset_train), 4))

            with torch.no_grad():
                last_progress_bar_update = time() - 10
                total_correct = 0
                model.eval()
                for iteration, x in enumerate(DataLoader(self.dataset_test, batch_size=config.batch_size)):
                    sentences, subject_idx, labels = fix_string_batch(x['sentence']), x['subject_idx'], x['label']
                    labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
                    enc = model.tokenize_with_spans(sentences)
                    y_pred = model.get_coref_predictions(enc, subject_idx)
                    loss = criterion(y_pred, labels)
                    total_correct += torch.abs((y_pred <= 0).float() - labels).sum().item()
                    min_loss_test = min(min_loss_test, loss.item())
                    result.add_scalar(f'loss/test', loss.item())

                    if time() - last_progress_bar_update > 1:
                        progress_bar.set_description(
                            desc=f"{title} "
                                 f"epoch {epoch + 1}/{config.epochs}, "
                                 f"train_loss={pretty_number(min_loss_train, 2)}, "
                                 f"test_loss={pretty_number(min_loss_test, 2)}")
                        last_progress_bar_update = time()
                    progress_bar.update()
            result.add_scalar(f'acc/test', round(total_correct / len(self.dataset_test), 4))

        return result.finish_training(start_time, model.get_parameters())


class TrainRegister:
    def __init__(self, constructor: dict[str, type]):
        self.train_constructors = constructor
        self.trainers = {}

    def __getitem__(self, objective) -> Trainer:
        if objective not in self.trainers:
            self.trainers[objective] = self.train_constructors[objective]()
        return self.trainers[objective]


TRAIN_REGISTER = TrainRegister({
    'kaneko': OrthogonalTrainer,
    'coreference-resolution': CorefTrainer
})


def train_model(model: MLM):
    if model.config.can_train():
        result = TRAIN_REGISTER[model.config.objective].train(model)
        result.evaluations = {}  # remove any evaluations, since they are not valid anymore
        return result
    else:
        return ModelResult(model.config).finish_training(time(), model.get_parameters())

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
from src.utils.files import read_file, get_folder
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
        total_steps = config.epochs * ceil(len(self.dataset_attribute) / (config.batch_size // 2))
        train_loss = LatestStats(total_steps // 20)

        optimizer = AdamW(params=model.get_parameters_to_train(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=int(total_steps * 1.04))

        v_a: torch.Tensor = read_file(f'train/kaneko/va/{config.model_name}.pt').to(DEVICE).detach()
        base_model = freeze(MLM.from_config(config.to_base()).train()).to(DEVICE)

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
                    # (bs, seq, dim)
                    attr_old_hidden = base_model.get_hidden_states(enc_attr)
                    attr_lengths = enc_attr['attention_mask'].sum(dim=-1).tolist()
                # (bs, seq, dim)
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
                    prefix_regularization = 0.01 * model.module_dict['prefix_embeddings'].regularization()
                    result.add_scalar(f'loss/prefix_regularization', prefix_regularization.item())

                # (bs/2, n_layers, dim)
                orthogonal_loss = 0.2 * (torch.einsum('ald,tld->atl', v_a, stereo_new_hidden) ** 2).sum()
                result.add_scalar(f'loss/orthogonal', orthogonal_loss.item())

                loss = embedding_regularization_loss + orthogonal_loss + prefix_regularization
                train_loss.add(loss.item(), len(attr_old_hidden))
                result.add_scalar(f'loss/total', loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                if time() - last_progress_bar_update > 1:
                    progress_bar.set_description(
                        desc=f"Orthogonal training: "
                             f"epoch {epoch + 1}/{config.epochs}, "
                             f"loss={pretty_number(train_loss.get_score(), 2)}")
                    last_progress_bar_update = time()

                progress_bar.update()

        return result.finish_training(start_time, model.get_parameters_to_save())


class EntailmentTrainer(Trainer):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        folder = get_folder('eval/glue')
        self.dataset_train = folder.read_file(f'train/{metric_name}.parquet')
        self.dataset_test = folder.read_file(f'validation/{metric_name}.parquet')

    def process_batch(self, model: MLM, batch):
        """ default for mrpc, stsb, rte and wnli """
        enc1 = model.tokenize(batch['sentence1'])
        enc2 = model.tokenize(batch['sentence2'])
        logits = model.get_entailment_predictions(enc1, enc2)
        labels = batch['label']
        return logits, labels

    def train(self, model: MLM) -> ModelResult:
        config = model.config
        result = ModelResult(config)

        start_time = time()
        last_progress_bar_update = start_time - 10
        current_train_step = 0

        total_train_steps = config.epochs * ceil(len(self.dataset_train) / config.batch_size)
        total_test_steps = config.epochs * ceil(len(self.dataset_test) / config.batch_size)
        switch_optim = total_train_steps // 10

        head_name = f"{config.objective}_head"
        freeze(model)
        unfreeze(model.module_dict[head_name])
        optimizer = AdamW(params=model.module_dict[head_name].parameters(), lr=config.lr * 10)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=int(switch_optim * 1.04))
        criterion = torch.nn.BCEWithLogitsLoss()

        title = f'{self.metric_name}: Train head only..'
        progress_bar = tqdm(total=total_train_steps + total_test_steps)

        loss_train = LatestStats(min(4, total_train_steps // (10 * config.epochs)))
        accuracy_train = LatestStats(min(4, total_train_steps // (10 * config.epochs)))
        loss_test = LatestStats(total_test_steps // config.epochs)
        accuracy_test = LatestStats(total_test_steps // config.epochs)

        for epoch in range(config.epochs):
            model.train()
            for iteration, x in enumerate(DataLoader(self.dataset_train.shuffle(seed=config.seed + epoch),
                                                     batch_size=config.batch_size)):
                if current_train_step == switch_optim:
                    title = f'{self.metric_name}: Train complete model..'
                    optimizer = AdamW(params=model.get_parameters_to_train(), lr=config.lr)
                    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                num_warmup_steps=config.num_warmup_steps,
                                                                num_training_steps=int((total_train_steps -
                                                                                        switch_optim) * 1.04))
                optimizer.zero_grad()
                model.zero_grad()
                logits, labels = self.process_batch(model, x)
                labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss = loss.item()

                accuracy_train.add(torch.abs((logits <= 0).float() - labels).sum().item(), len(labels))
                loss_train.add(loss, len(labels))
                result.add_scalar(f'loss/train', loss)

                if time() - last_progress_bar_update > 1:
                    progress_bar.set_description(
                        desc=f"{title} "
                             f"epoch {epoch + 1}/{config.epochs}, "
                             f"train_loss={pretty_number(loss_train.get_score(), 2)}, "
                             f"train_acc={round(100 * accuracy_train.get_score(), 2)}, "
                             f"test_loss={pretty_number(loss_test.get_score(), 2)}, "
                             f"test_acc={round(100 * accuracy_test.get_score(), 2)}")
                    last_progress_bar_update = time()

                progress_bar.update()
                current_train_step += 1

            with torch.no_grad():
                last_progress_bar_update = time() - 10
                model.eval()
                for iteration, x in enumerate(DataLoader(self.dataset_test, batch_size=config.batch_size)):
                    logits, labels = self.process_batch(model, x)
                    labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
                    loss = criterion(logits, labels).item()

                    accuracy_test.add(torch.abs((logits <= 0).float() - labels).sum().item(), len(labels))
                    loss_test.add(loss, len(labels))
                    result.add_scalar(f'loss/test', loss)

                    if time() - last_progress_bar_update > 1:
                        progress_bar.set_description(
                            desc=f"{title} "
                                 f"epoch {epoch + 1}/{config.epochs}, "
                                 f"train_loss={pretty_number(loss_train.get_score(), 2)}, "
                                 f"train_acc={round(100 * accuracy_train.get_score(), 2)}, "
                                 f"test_loss={pretty_number(loss_test.get_score(), 2)}, "
                                 f"test_acc={round(100 * accuracy_test.get_score(), 2)}")
                        last_progress_bar_update = time()

        return result.finish_training(start_time, model.get_parameters_to_save())


class MRPCTrainer(EntailmentTrainer):
    def __init__(self):
        super().__init__('mrpc')


class STS_BTrainer(EntailmentTrainer):
    def __init__(self):
        super().__init__('stsb')


class RTETrainer(EntailmentTrainer):
    def __init__(self):
        super().__init__('rte')


class WNLITrainer(EntailmentTrainer):
    def __init__(self):
        super().__init__('wnli')


class SST2Trainer(EntailmentTrainer):
    def __init__(self):
        super().__init__('sst2')

    def process_batch(self, model: MLM, batch):
        """ default for wsc """
        enc = model.tokenize(batch['sentence'])
        logits = model.get_sentiment_analysis(enc)
        labels = batch['label']
        return logits, labels


class WSCTrainer(EntailmentTrainer):
    def __init__(self):
        super().__init__('wsc')

    def process_batch(self, model: MLM, batch):
        """ default for wsc """
        enc = model.tokenize_with_spans(fix_string_batch(batch['sentence']))
        logits = model.get_coref_predictions(enc, batch['subject_idx'])
        labels = batch['label']
        return logits, labels


class LatestStats:
    def __init__(self, last_n=50):
        self.counter = 0
        self.last_n = last_n
        self.batch_sizes = torch.ones(last_n, dtype=torch.float32) * 1e-20
        self.stats = torch.ones(last_n, dtype=torch.float32)
        self.calculated = False
        self.result = None

    def add(self, stat, batch_size):
        self.stats[self.counter] = stat
        self.batch_sizes[self.counter] = batch_size
        self.counter += 1
        if self.counter >= self.last_n:
            self.counter = 0
        self.calculated = False

    def get_score(self) -> float:
        if not self.calculated:
            total_batch = self.batch_sizes.sum()
            if torch.abs(total_batch) < 1e-3:
                self.result = 0
            else:
                self.result = (self.stats.sum() / total_batch).item()
            self.calculated = True
        return self.result


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
    'mrpc': MRPCTrainer,
    'stsb': STS_BTrainer,
    'rte': RTETrainer,
    'wnli': WNLITrainer,
    'sst2': SST2Trainer,
    'wsc': WSCTrainer
})


def train_model(model: MLM):
    if model.config.can_train():
        return TRAIN_REGISTER[model.config.objective].train(model)
    else:
        return ModelResult(model.config).finish_training(time(), model.get_parameters_to_save())

from math import ceil
from time import time
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from diffusers import get_linear_schedule_with_warmup
from torch.optim import AdamW, SGD

from src.data.structs.results import TrainResult
from src.language_models.language_model import LanguageModel
from src.data.preprocess_data import get_kaneko_data, get_ml_head_data, get_probe_data
from src.utils.files import get_folder, write_file
from src.utils.printing import pretty_number
from src.utils.pytorch import DEVICE, fix_string_batch, freeze, unfreeze


def save_img(_title: str, _losses: list[float], _warmup: int, _config):
    folder = get_folder('experiments/outputs/plots', create_if_not_exists=True)
    plt.plot(_losses)
    folder.write_file(f"{_title}, {_config.get_filename()}.png", plt)
    plt.clf()
    plt.plot(_losses[_warmup:])
    folder.write_file(f"{_title}, {_config.get_filename()} (from {_warmup}).png", plt)
    plt.clf()


class Trainer:
    def train(self, model: LanguageModel) -> TrainResult:
        raise NotImplementedError('Not implemented yet.')


class MLHeadTrainer(Trainer):
    def train(self, model: LanguageModel) -> TrainResult:
        dataset = get_ml_head_data()
        config = model.config
        result = TrainResult(config)
        BATCH_SIZE = 14

        start_time = time()
        last_progress_bar_update = start_time - 10
        total_steps = ceil(len(dataset) / BATCH_SIZE)
        train_loss = LatestStats(total_steps // 100)
        total_loss = []

        base_model = freeze(LanguageModel.from_config(config.to_original_model()).eval()).to(DEVICE)
        freeze(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(params=unfreeze(model.lm_head).parameters(), lr=0.00002)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=total_steps // 6,
                                                    num_training_steps=int(total_steps * 1.02))

        progress_bar = tqdm(total=total_steps)
        model.train()
        for iteration, batch in enumerate(DataLoader(dataset.shuffle(seed=42), batch_size=BATCH_SIZE)):
            optimizer.zero_grad()
            # RUN MODELS
            enc = model.tokenize(batch['sentence'])

            masked_tokens_idx = torch.rand(enc.input_ids.size(), device=enc.input_ids.device)
            masked_tokens_idx[:, 0] = 1  # cls token cannot be masked
            masked_tokens_idx = masked_tokens_idx < 0.15

            enc.input_ids[masked_tokens_idx] = model.tokenizer.mask_token_id

            # (bs, seq, n_vocabulary)
            with torch.no_grad():
                pred_original = base_model.get_ml_predictions(enc).softmax(dim=-1)
                bs, n_seq, n_vocabulary = pred_original.size()

            sent_lengths = enc.attention_mask.sum(-1) - 1
            logits = model.get_ml_predictions(enc)

            loss = [criterion(logits[i, 1:sent_len+1], pred_original[i, 1:sent_len+1])
                    for i, sent_len in enumerate(sent_lengths)]
            loss = torch.stack(loss).sum()

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss = loss.item()

            train_loss.add(loss, len(batch))
            total_loss.append(loss)

            if time() - last_progress_bar_update > 1:
                progress_bar.set_description(
                    desc=f"MLHead training: "
                         f"loss={pretty_number(train_loss.get_score(), 2)}")
                last_progress_bar_update = time()

            progress_bar.update()

        save_img('MLHead', total_loss, 100, config)
        return result.finish_training(start_time, model)


class ProbeTrainer(Trainer):
    def process_batch(self, model: LanguageModel, batch):
        enc = model.tokenize_with_spans(fix_string_batch(batch['sentences']))
        # (n_spans, dim) == (bs, dim)
        embeddings = model.get_span_embeddings(enc, reduce='first')
        return model.cls_head(embeddings), batch['label']

    def train(self, model: LanguageModel) -> TrainResult:
        train_dataset, eval_dataset, stereo_dataset = get_probe_data()
        config = model.config
        result = TrainResult(config)
        all_losses = []

        start_time = time()
        last_progress_bar_update = start_time - 10
        current_train_step = 0

        total_train_steps = ceil(len(train_dataset) / config.batch_size)
        total_test_steps = ceil(len(eval_dataset) / config.batch_size)
        total_stereo_steps = ceil(len(stereo_dataset) / config.batch_size)

        title = f'Probe: Train head only..'
        total_steps = total_train_steps + total_test_steps + (11 * total_stereo_steps)
        total_steps *= config.epochs
        progress_bar = tqdm(total=total_steps)

        freeze(model)
        optimizer = AdamW(params=unfreeze(model.cls_head).parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=total_steps)
        criterion = torch.nn.BCEWithLogitsLoss()

        loss_train = LatestStats(min(5, total_train_steps // (10 * config.epochs)))
        accuracy_train = LatestStats(min(5, total_train_steps // (10 * config.epochs)))
        loss_test = LatestStats(total_test_steps // config.epochs)
        accuracy_test = LatestStats(total_test_steps // config.epochs)

        experiment_results = {'stereotype_loss': [], 'stereotype_acc': []}

        for epoch in range(config.epochs):
            update_t = (total_train_steps // 20)
            for iteration, x in enumerate(DataLoader(train_dataset.shuffle(seed=config.seed + epoch),
                                                     batch_size=config.batch_size)):
                model.train()
                optimizer.zero_grad()
                model.zero_grad()
                logits, labels = self.process_batch(model, x)
                labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss = loss.item()
                acc = torch.abs((logits <= 0).float() - labels).sum().item()

                accuracy_train.add(acc, len(labels))
                loss_train.add(loss, len(labels))
                result.add_scalar(f'loss/train', loss)
                all_losses.append(loss)

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

                if iteration % update_t == 0:
                    with torch.no_grad():
                        model.eval()
                        total_stereo_loss = 0
                        total_stereo_correct = 0
                        for iteration, x in enumerate(DataLoader(stereo_dataset, batch_size=config.batch_size)):
                            logits, labels = self.process_batch(model, x)
                            labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
                            total_stereo_loss += criterion(logits, labels).item()
                            total_stereo_correct += torch.abs((logits <= 0).float() - labels).sum().item()
                            progress_bar.update()
                        experiment_results['stereotype_loss'].append(total_stereo_loss)
                        experiment_results['stereotype_acc'].append(total_stereo_correct / len(stereo_dataset))

            with torch.no_grad():
                model.eval()
                total_stereo_loss = 0
                total_stereo_correct = 0
                for iteration, x in enumerate(DataLoader(stereo_dataset, batch_size=config.batch_size)):
                    logits, labels = self.process_batch(model, x)
                    labels = labels.unsqueeze(-1).to(device=DEVICE, dtype=torch.float32).detach()
                    total_stereo_loss += criterion(logits, labels).item()
                    total_stereo_correct += torch.abs((logits <= 0).float() - labels).sum().item()
                    progress_bar.update()
                experiment_results['stereotype_loss'].append(total_stereo_loss)
                experiment_results['stereotype_acc'].append(total_stereo_correct / len(stereo_dataset))

            with torch.no_grad():
                last_progress_bar_update = time() - 10
                model.eval()
                for iteration, x in enumerate(DataLoader(eval_dataset, batch_size=config.batch_size)):
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
                    progress_bar.update()

        write_file(f'experiments/outputs/results/gender_probe/{config.get_filename()}.json', experiment_results)
        save_img('ProbeTraining', all_losses, 100, config)
        return result.finish_training(start_time, model)


class OrthogonalTrainer(Trainer):
    def train(self, model: LanguageModel) -> TrainResult:
        config = model.config
        result = TrainResult(config)

        dataset_attribute, dataset_stereo, v_a = get_kaneko_data(config.model_name)

        start_time = time()
        last_progress_bar_update = start_time - 10
        total_steps = config.epochs * ceil(len(dataset_attribute) / (config.batch_size // 2))
        train_loss = LatestStats(total_steps // 20)
        all_losses = []

        freeze(model)
        optimizer = AdamW(params=unfreeze(model.get_parameters_to_train()), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=int(total_steps * 1.04))

        v_a: torch.Tensor = v_a.to(DEVICE).detach()
        base_model = freeze(LanguageModel.from_config(config.to_original_model()).train()).to(DEVICE)

        progress_bar = tqdm(total=total_steps)
        for epoch in range(config.epochs):
            model.train()
            data_loader_attribute = DataLoader(dataset_attribute.shuffle(seed=config.seed + epoch),
                                               batch_size=config.batch_size // 2)
            data_loader_stereo = iter(DataLoader(dataset_stereo.shuffle(seed=config.seed + epoch + 1000),
                                                 batch_size=config.batch_size // 2))

            for iteration, batch in enumerate(data_loader_attribute):
                optimizer.zero_grad()
                # RUN MODELS
                enc_attr = model.tokenize([''.join(s) for s in fix_string_batch(batch['sentences'])])
                with torch.no_grad():
                    # (bs, seq, dim)
                    attr_old_hidden = base_model.get_embeddings(enc_attr, output_hidden_states=True)
                    attr_lengths = enc_attr['attention_mask'].sum(dim=-1).tolist()
                # (bs, seq, dim)
                attr_new_hidden = model.get_embeddings(enc_attr, output_hidden_states=True)

                # (bs, sent_parts)
                enc_stereo = model.tokenize_with_spans(fix_string_batch(next(data_loader_stereo)['sentences']))
                # (bs, n_layers, dim)
                stereo_new_hidden = model.get_span_embeddings(enc_stereo, reduce='first', output_hidden_states=True)

                # CALCULATE LOSS
                embedding_regularization_loss = [((a1[:n] - a2[:n]) ** 2).mean() for a1, a2, n
                                                 in zip(attr_old_hidden, attr_new_hidden, attr_lengths)]
                embedding_regularization_loss = 0.8 * torch.stack(embedding_regularization_loss).mean()
                result.add_scalar(f'loss/embedding_regularization', embedding_regularization_loss.item())

                prefix_regularization = 0
                if model.config.is_prefix():
                    prefix_regularization = 0.1 * model.prefix_embeddings.regularization()
                    result.add_scalar(f'loss/prefix_regularization', prefix_regularization.item())

                # (bs/2, n_layers, dim)
                #print(v_a.size())
                #print(stereo_new_hidden.size())
                orthogonal_loss = 0.1 * (torch.einsum('ald,tld->atl', v_a, stereo_new_hidden) ** 2).mean()
                result.add_scalar(f'loss/orthogonal', orthogonal_loss.item())

                loss = embedding_regularization_loss + orthogonal_loss + prefix_regularization
                train_loss.add(loss.item(), len(attr_old_hidden))
                result.add_scalar(f'loss/total', loss.item())
                all_losses.append(loss.item())

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

        save_img('OrthogonalTraining', all_losses, 100, config)
        return result.finish_training(start_time, model)


class GLUETrainer(Trainer):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        folder = get_folder('data/eval/glue')
        self.dataset_train = folder.read_file(f'train/{metric_name}.parquet')
        self.dataset_test = folder.read_file(f'validation/{metric_name}.parquet')

    def process_batch(self, model: LanguageModel, batch):
        """ default for mrpc, stsb, rte and wnli """
        subject_idx = None

        if self.metric_name in {'sst2'}:
            enc = model.tokenize(batch['sentence'])

        elif self.metric_name in {'mrpc', 'stsb', 'rte', 'wnli'}:
            enc = model.tokenize(list(zip(batch['sentence1'], batch['sentence2'])))

        elif self.metric_name in {'wsc'}:
            enc = model.tokenize_with_spans(fix_string_batch(batch['sentence']))
            subject_idx = batch['subject_idx']

        else:
            raise ValueError(f"metric '{self.metric_name}' is not supported!")

        # logits, labels
        return model.get_cls_predictions(enc, subject_idx), batch['label']

    def train(self, model: LanguageModel) -> TrainResult:
        config = model.config
        result = TrainResult(config)
        all_losses = []

        start_time = time()
        last_progress_bar_update = start_time - 10
        current_train_step = 0

        total_train_steps = config.epochs * ceil(len(self.dataset_train) / config.batch_size)
        total_test_steps = config.epochs * ceil(len(self.dataset_test) / config.batch_size)
        switch_optim = int(0.15 * total_train_steps)

        freeze(model)
        optimizer = AdamW(params=unfreeze(model.cls_head).parameters(), lr=config.lr * 10)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=int(switch_optim * 1.02))
        criterion = torch.nn.BCEWithLogitsLoss()

        title = f'{self.metric_name}: Train head only..'
        progress_bar = tqdm(total=total_train_steps + total_test_steps)

        loss_train = LatestStats(min(5, total_train_steps // (10 * config.epochs)))
        accuracy_train = LatestStats(min(5, total_train_steps // (10 * config.epochs)))
        loss_test = LatestStats(total_test_steps // config.epochs)
        accuracy_test = LatestStats(total_test_steps // config.epochs)

        for epoch in range(config.epochs):
            model.train()
            for iteration, x in enumerate(DataLoader(self.dataset_train.shuffle(seed=config.seed + epoch),
                                                     batch_size=config.batch_size)):
                if current_train_step == switch_optim:
                    title = f'{self.metric_name}: Train complete model..'
                    freeze(model)
                    optimizer = AdamW(params=unfreeze(model.get_parameters_to_train()), lr=config.lr)
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
                all_losses.append(loss)

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
                    progress_bar.update()

        save_img('GLUETraining', all_losses, 100, config)
        return result.finish_training(start_time, model)


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
    'mrpc': lambda: GLUETrainer('mrpc'),
    'stsb': lambda: GLUETrainer('stsb'),
    'rte': lambda: GLUETrainer('rte'),
    'wnli': lambda: GLUETrainer('wnli'),
    'sst2': lambda: GLUETrainer('sst2'),
    'wsc': lambda: GLUETrainer('wsc'),
    'ml_head': MLHeadTrainer,
    'probe': ProbeTrainer
})


def train_model(model: LanguageModel):
    model = model.to(DEVICE)
    if model.config.is_downstream():
        result = TRAIN_REGISTER[model.config.downstream_task].train(model)
    elif model.config.is_debiased():
        result = TRAIN_REGISTER[model.config.debias_method].train(model)
    else:
        result = TrainResult(model.config).finish_training(time(), model)
    #TRAIN_REGISTER['ml_head'].train(model) TODO
    return result

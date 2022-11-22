from collections import OrderedDict
from math import ceil

import torch
from matplotlib import pyplot as plt
from torch import cosine_similarity, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import copy

from transformers import get_linear_schedule_with_warmup

from dependencies.weat import run_test
from sklearn.model_selection import train_test_split
from src.MLM import MLM
from src.utils import batched, stack_dicts, sig_notation
from src.utils.io import DATA_DIR, IOFolder
from src.utils.pytorch import fix_string_batch, DEVICE


class Metric:
    def __init__(self, metric_name: str, folder_name: str):
        self.metric_name = metric_name
        self.data = self.prepare_data(DATA_DIR['eval', folder_name])

    def prepare_data(self, folder: IOFolder):
        raise NotImplementedError('Not implemented yet')

    def eval_model(self, model: MLM) -> dict:
        raise NotImplementedError('Not implemented yet')


class SEAT(Metric):
    """
    Sentence Encoder Association Test (SEAT)

    May, Chandler, et al. "On measuring social biases in sentence encoders." arXiv preprint arXiv:1903.10561 (2019).
    https://arxiv.org/pdf/1903.10561
    """

    def __init__(self):
        super().__init__(metric_name='seat', folder_name='seat')

    def prepare_data(self, folder: IOFolder):
        return {f.name: f.read() for f in folder.get_files(extension='.json')}

    def eval_model(self, model: MLM) -> dict:
        with torch.no_grad():
            keys = ['targ1', 'targ2', 'attr1', 'attr2']
            result = {}
            for test_name, test in tqdm(self.data.items(), desc=f'   SEAT'):
                encoded_sentences = model.tokenize(sum([test[t]['examples'] for t in keys], []))
                sentences_embedded = model.get_sentence_embeddings(encoded_sentences, layer=-1).cpu()  # (bs, dim)
                c = 0
                for t in keys:
                    sents = test[t]['examples']
                    test[t]['encs'] = dict(zip(sents, sentences_embedded[c:c + len(sents)]))
                    c += len(sents)

                esize, pval = run_test(test, 100000)
                result[test_name] = {'effect_size': round(esize, 5), 'p_val': round(pval, 5)}
            return result


class LPBS(Metric):
    """
    Log Probability Bias Score (LPBS)

    Kurita, Keita, et al. "Measuring bias in contextualized word representations." arXiv preprint arXiv:1906.07337 (2019).
    https://arxiv.org/pdf/1906.07337
    """

    def __init__(self):
        super().__init__(metric_name='lpbs', folder_name='lpbs')

    def prepare_data(self, folder: IOFolder):
        return {f.name: f.read() for f in folder.get_files(extension='.json')}

    def eval_model(self, model: MLM) -> dict:
        with torch.no_grad():
            result = {}

            UNK_token = model.tokenizer.unk_token_id
            MASK_token = model.tokenizer.decode(model.tokenizer.mask_token_id)

            ignored_words = set()
            ignored_pairs = set()
            files = copy.deepcopy(self.data)
            for f_i, file in self.data.items():
                for t_i, test in enumerate(file['tests']):
                    error_tokens = set()
                    for w_i, w_list in enumerate(test['targets']):
                        token_ids = model.tokenizer.convert_tokens_to_ids(w_list)
                        for i, token in enumerate(token_ids):
                            if token == UNK_token:
                                error_tokens.add(w_i)
                                ignored_words.add(w_list[i])
                                ignored_pairs.add(str(tuple(w_list)))
                    files[f_i]['tests'][t_i]['targets'] = [w_list for w_i, w_list in
                                                           enumerate(files[f_i]['tests'][t_i]['targets'])
                                                           if w_i not in error_tokens]
            if len(ignored_words) > 0:
                print(
                    f"The following words cannot be converted to a single token by '{model.config.model_name}': {ignored_words}, \n"
                    f"therefore the following pairs will be ignored in the test {ignored_pairs}")

            total_iterations = 0
            for test in files.values():
                for subtest in test['tests']:
                    total_iterations += ceil(len(test['attributes']) * len(subtest["templates"]) / 32)

            pbar = tqdm(desc=f"   LPBS", total=total_iterations)
            for test_name, test in files.items():
                all_p_tgt = []
                all_p_prior = []
                attributes = test['attributes']
                for test_part in test['tests']:
                    templates: list[str] = test_part['templates']
                    targets = test_part['targets']
                    all_targets = sum([list(t) for t in targets], [])

                    target_ids = model.tokenizer.convert_tokens_to_ids(all_targets)
                    for i, token_id in enumerate(target_ids):
                        assert token_id != model.tokenizer.unk_token_id, \
                            f"Target word '{all_targets[i]}' does not convert to a single token."
                    masked_sentences = [t.replace('[TARGET]', MASK_token).replace('[ATTRIBUTE]', MASK_token) for t in
                                        templates]
                    # (n_templates, n_masks, n_targets)
                    p_prior = model.get_mask_probabilities(model.tokenize(masked_sentences), target_ids,
                                                           output_tensor=True)
                    # (n_templates, n_target_pairs, bias_class_dim)
                    mask_idx = [(0 if '[ATTRIBUTE]' in t.split('[TARGET]')[1] else 1) for t in templates]
                    p_prior = p_prior[list(range(len(templates))), mask_idx] \
                        .reshape(len(templates), len(targets), len(targets[0]))
                    # (n_templates, bias_class_dim)
                    p_prior = p_prior.sum(dim=1)

                    masked_templates = [t.replace('[TARGET]', MASK_token) for t in templates]
                    all_templates = []
                    for a in attributes:
                        all_templates.extend([t.replace('[ATTRIBUTE]', a) for t in masked_templates])

                    # (ALL_templates, n_targets)
                    p_tgt = []
                    for batch in batched(all_templates, batch_size=32):
                        p = model.get_mask_probabilities(model.tokenize(batch), target_ids, output_tensor=True)[:, 0]
                        p_tgt.append(p)
                        pbar.update(1)
                    p_tgt = torch.cat(p_tgt, dim=0)

                    # (n_occupations, n_templates, n_target_pairs, bias_class_dim)
                    p_tgt = p_tgt.reshape(len(attributes), len(templates), len(targets), len(targets[0]))
                    # (n_occupations, n_templates, bias_class_dim)
                    p_tgt = p_tgt.sum(dim=2)

                    all_p_tgt.append(p_tgt)
                    all_p_prior.append(p_prior)

                # (n_occupations, ALL_templates, bias_class_dim)
                all_p_tgt = torch.cat(all_p_tgt, dim=1)
                # (ALL_templates, bias_class_dim)
                all_p_prior = torch.cat(all_p_prior, dim=0)

                # (n_occupations, ALL_templates, bias_class_dim)
                ILPS = torch.log(all_p_tgt / all_p_prior)

                # (n_occupations, ALL_templates)
                ILPS = torch.abs(ILPS[..., 0] - ILPS[..., 1])

                result[test_name] = {
                    'bias_score': ILPS.mean(dim=(0, 1)).item(),
                    'bias_score_std': ILPS.std(dim=(0, 1)).item()
                }
            pbar.close()
            return result


class WinoGenderSimilarity(Metric):
    def __init__(self):
        super().__init__(metric_name='winogender_similarity', folder_name='winogender')

    def prepare_data(self, folder: IOFolder):
        return folder.read_file('wino_gender_test.parquet')

    def eval_model(self, model: MLM) -> dict:

        """ Dataset:
            'sentence': result,
            'subject_idx': subject_index // 2,
            'pronoun_idx': pronoun_index // 2,
            'label': true_label,
            'use_someone': use_someone,
            'pronoun_class': pronoun_class,
            'subject_is_occupation': is_occupation
        """
        batch_size = 64
        bound_step = 0.02

        iterations_emb = len(self.data) // batch_size
        if len(self.data) % batch_size != 0:
            iterations_emb += 1

        progress_bar = tqdm(total=iterations_emb + int(2 / bound_step) - 2)

        with torch.no_grad():
            model.eval()
            all_embeddings = []
            progress_bar.set_description('   WinoGenderSimilarity (embeddings)')
            for x in DataLoader(self.data, batch_size=batch_size):
                subject_indices, labels = x['subject_idx'], x['label']
                encoding_with_spans = model.tokenize_with_spans(fix_string_batch(x['sentence']))

                # (n_spans, dim)
                embeddings = model.get_span_embeddings(encoding_with_spans, layer=-1, reduce='first')
                dims = embeddings.size()
                # (n_sentences, 2, dim)
                embeddings = embeddings.reshape(dims[0] // 2, 2, dims[1])

                if not isinstance(subject_indices, torch.Tensor):
                    subject_indices = torch.tensor(subject_indices)
                subject_indices = subject_indices.long()

                # ordered for subject/pronoun: (2, n_sentences, dim)
                embeddings = torch.stack((embeddings[range(len(subject_indices)), subject_indices],
                                          embeddings[range(len(subject_indices)), 1 - subject_indices]))

                # append (2[subject,pronoun], bs, dim)
                all_embeddings.append(embeddings)
                progress_bar.update(1)

            # (2[subject,pronoun], n, dim)
            all_embeddings = torch.cat(all_embeddings, dim=1)

            # (n)
            sim = cosine_similarity(all_embeddings[0], all_embeddings[1], dim=-1)
            # (n)
            y = torch.tensor(self.data['label'], dtype=torch.float32, device=DEVICE)

            progress_bar.set_description('   WinoGenderSimilarity (bounds)')
            best_acc, best_bound = -1, None
            for bound in torch.arange(1.0-bound_step, -1.0+bound_step, -bound_step):
                y_pred = (sim >= bound).int()
                acc = 1.0 - torch.abs(y_pred - y).float().mean()
                if acc > best_acc:
                    best_acc = acc.item()
                    best_bound = bound.item()
                progress_bar.update(1)

            progress_bar.set_description('   WinoGenderSimilarity (bias)')
            # TODO



            progress_bar.set_description('   WinoGenderSimilarity')
            return {'accuracy': round(100 * best_acc, 2), 'bound': round(best_bound, 2)}


class WinoGenderFineTune(Metric):
    def __init__(self):
        super().__init__(metric_name='winogender_finetune', folder_name='winogender')

    def prepare_data(self, folder: IOFolder):
        return folder.read_file('wino_gender_test.parquet')

    def eval_model(self, model: MLM) -> dict:

        """ Dataset:
            'sentence': result,
            'subject_idx': subject_index // 2,
            'pronoun_idx': pronoun_index // 2,
            'label': true_label,
            'use_someone': use_someone,
            'pronoun_class': pronoun_class,
            'subject_is_occupation': is_occupation
        """
        with torch.no_grad():
            model.eval()
            y_pred = []
            for x in tqdm(DataLoader(self.data, batch_size=64), desc='   WinoGenderFineTune'):
                enc = model.tokenize_with_spans(fix_string_batch(x['sentence']))
                # (bs, 1)
                logits = model.get_coref_predictions(enc, x['subject_idx'])
                y_pred.append((logits > 0).float().detach())
            # (n)
            y_pred = torch.cat(y_pred, dim=0).flatten()
            y_true = torch.tensor(self.data['label'], device=DEVICE, dtype=torch.float32)

            return {'accuracy': round(100 * (1.0 - torch.abs(y_pred - y_true).mean().item()), 2)}


METRICS = {
    'default': [SEAT(), LPBS(), WinoGenderSimilarity()],
    'coreference-resolution': [SEAT(), LPBS(), WinoGenderSimilarity(), WinoGenderFineTune()],
}


# METRICS = {'winogender': WinoGender()}

def run_metrics(task: str, model: MLM) -> dict:
    return {metric.metric_name: metric.eval_model(model) for metric in METRICS[task]}


"""
    def mlp_scores(self, model: MLM, all_embeddings, y_all):
        # NN classifier
        classifier = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(model.dim * 2, 32)),
            ('dropout', nn.Dropout(0.2)),
            ('swish', nn.SiLU()),
            ('f2', nn.Linear(32, 1))
        ])).to(DEVICE)

        classifier.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True

        # (n_templates, 2[original/someone], 3[bias classes], 2 x embedding_dim)
        x_occupation = all_embeddings[..., [0, 2], :].flatten(start_dim=-2, end_dim=-1)
        x_participant = all_embeddings[..., [1, 2], :].flatten(start_dim=-2, end_dim=-1)

        # (2[occ-pronoun, part-pronoun], n_templates, 2[original/someone], 3[bias classes], 2 x embedding_dim)
        x_all = torch.stack((x_occupation, x_participant), dim=0)
        # (2[occ-pronoun, part-pronoun], n_templates, 2[original/someone], 3[bias classes], 1)
        y_all = torch.stack((1 - y_all, y_all), dim=0)

        # (n_templates, 2[original/someone], 3[bias classes], 2[occ-pronoun, part-pronoun], 2 x embedding_dim)
        x_all = x_all.permute((1, 2, 3, 0, 4))
        # (n_templates, 2[original/someone], 3[bias classes], 2[occ-pronoun, part-pronoun], 1)
        y_all = y_all.permute((1, 2, 3, 0, 4))

        # (p * n_templates, 2[original/someone], 3[bias classes], 2[occ-pronoun, part-pronoun], ...) etc.
        test_perc = 0.2
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_perc, shuffle=True,
                                                            random_state=42)
        x_train, x_test, y_train, y_test = x_train.flatten(start_dim=0, end_dim=-2), \
                                           x_test.flatten(start_dim=0, end_dim=-2), \
                                           y_train.flatten(start_dim=0, end_dim=-2), \
                                           y_test.flatten(start_dim=0, end_dim=-2)

        TOTAL_EPOCHS = 201
        LR = 0.001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(params=classifier.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=TOTAL_EPOCHS // 20,
                                                    num_training_steps=TOTAL_EPOCHS)

        epoch_bar = trange(TOTAL_EPOCHS, desc='   WinoGender Linear classifier')
        result = {'loss': {'train': [], 'test': []}, 'accuracy': {'train': [], 'test': []}}
        for i in epoch_bar:
            classifier.train()
            y_pred = classifier(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss = loss.item()
            y_pred = (y_pred >= 0).float().detach()
            accuracy = 1.0 - torch.abs(y_pred - y_train).mean().item()
            result['loss']['train'].append(loss)
            result['accuracy']['train'].append(accuracy)

            with torch.no_grad():
                classifier.eval()
                y_pred = classifier(x_test).detach()
                loss = criterion(y_pred, y_test).item()
                y_pred = (y_pred >= 0).float()
                accuracy = 1.0 - torch.abs(y_pred - y_test).mean().item()
                result['loss']['test'].append(loss)
                result['accuracy']['test'].append(accuracy)

            if i % 50 == 0:
                epoch_bar.set_description(f'   WinoGender '
                                          f'(epoch={i + 1}/{TOTAL_EPOCHS}, '
                                          f'loss_train={sig_notation(min(result["loss"]["train"]))}, '
                                          f'loss_test={sig_notation(min(result["loss"]["test"]))} '
                                          f'acc_train={sig_notation(max(result["accuracy"]["train"]))}, '
                                          f'acc_test={sig_notation(max(result["accuracy"]["test"]))})')

        self.show_result('MLP', result)

        result['loss']['train'] = min(result['loss']['train'])
        result['loss']['test'] = min(result['loss']['test'])
        result['accuracy']['train'] = max(result['accuracy']['train'])
        result['accuracy']['test'] = max(result['accuracy']['test'])

        return result, classifier

    def run_model(self, model, classifier, x):
        y = x['answer'].to(device=DEVICE, dtype=torch.float32).unsqueeze(dim=-1)
        # (2 x bs, 1)
        y = torch.cat((1 - y, y), dim=0)
        embeddings = model.tokenize_with_spans(fix_string_dataset(x['sentence_parts']))
        # (bs[occ]+bs[part]+bs[pronoun], dim) = (3 x bs, dim)
        embeddings = model.get_span_embeddings(embeddings, layer=-1, reduce='first')
        dims = embeddings.size()
        # (bs , 3[occ/part/pronoun in any order], dim)
        embeddings = embeddings.reshape((dims[0] // 3, 3, dims[1]))
        embeddings_ordered = []
        for key in ['occupation_idx', 'participant_idx', 'pronoun_idx']:
            # (bs, embedding_dim)
            embeddings_ordered.append(torch.stack([embeddings[bs_idx, type_idx // 2]
                                                   for bs_idx, type_idx in enumerate(x[key])]))
        # (bs, 3[occ,part,pronoun], embedding_dim)
        embeddings = torch.stack(embeddings_ordered).permute((1, 0, 2))
        embeddings_ordered = None

        # (2 x bs, 2 x embedding_dim)
        embeddings = torch.cat((embeddings[:, [0, 2]].flatten(start_dim=-2, end_dim=-1),
                                embeddings[:, [1, 2]].flatten(start_dim=-2, end_dim=-1)), dim=0)

        y_pred = classifier(embeddings)
        return y_pred, y

    def show_result(self, title, result):
        #print(result)
        for a in ['loss', 'accuracy']:
            plt.title(f'{title} {a}')
            plt.plot(result[a]['train'][10:], label=f'{a}_train')
            plt.plot(result[a]['test'][10:], label=f'{a}_test')
            plt.legend()
            plt.show()
            #plt.clf()

    def finetune_scores(self, model: MLM, classifier: nn.Module):
        model.requires_grad = True
        for param in model.parameters():
            param.requires_grad = True

        TOTAL_EPOCHS = 20
        TOTAL_ITERATIONS = 1 + (len(self.data['train']) // 32)
        LR = 0.00001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(params=list(model.get_modules().parameters()) + list(classifier.parameters()), lr=LR)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=TOTAL_ITERATIONS)
        result = {'loss': {'train': [], 'test': []}, 'accuracy': {'train': [], 'test': []}}

        epoch_bar = trange(TOTAL_EPOCHS, desc='   WinoGender Fine-tuning')
        for i in epoch_bar:
            epoch_results = {'loss': {'train': 0., 'test': 0.}, 'accuracy': {'train': 0., 'test': 0.}}
            classifier.train()
            model.modules.train()
            # TRAIN
            for x in DataLoader(self.data['train'], batch_size=32, shuffle=True):
                y_pred, y = self.run_model(model, classifier, x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_results['loss']['train'] += loss.item()
                epoch_results['accuracy']['train'] += torch.abs((y_pred >= 0).float() - y).sum().item()
            # TEST
            with torch.no_grad():
                classifier.eval()
                model.modules.eval()
                for x in DataLoader(self.data['test'], batch_size=32, shuffle=True):
                    y_pred, y = self.run_model(model, classifier, x)
                    epoch_results['loss']['test'] += criterion(y_pred, y).item()
                    epoch_results['accuracy']['test'] += torch.abs((y_pred >= 0).float() - y).sum().item()
            # RESULTS
            for a in ['train', 'test']:
                result['loss'][a].append(epoch_results['loss'][a])
                result['accuracy'][a].append(1. - ((0.5 * epoch_results['accuracy'][a]) / len(self.data[a])))
            epoch_bar.set_description(f'   WinoGender '
                                      f'(epoch={i+1}/{TOTAL_EPOCHS}, '
                                      f'loss_train={sig_notation(min(result["loss"]["train"]))}, '
                                      f'loss_test={sig_notation(min(result["loss"]["test"]))} '
                                      f'acc_train={sig_notation(max(result["accuracy"]["train"]))}, '
                                      f'acc_test={sig_notation(max(result["accuracy"]["test"]))})')

        self.show_result('Fine-tuning', result)

        result['loss']['train'] = min(result['loss']['train'])
        result['loss']['test'] = min(result['loss']['test'])
        result['accuracy']['train'] = max(result['accuracy']['train'])
        result['accuracy']['test'] = max(result['accuracy']['test'])

        return result
"""

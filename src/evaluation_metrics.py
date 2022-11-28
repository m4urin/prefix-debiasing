import copy
from math import ceil
from typing import Any

import torch
from torch import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dependencies.weat import run_test
from src.MLM import MLM
from src.utils.files import get_folder, IOFolder
from src.utils.functions import nested_loop
from src.utils.pytorch import fix_string_batch, pearson_correlation_coefficient


class Metric:
    def __init__(self, metric_name: str, folder_name: str):
        self.metric_name = metric_name
        self.data = self.prepare_data(get_folder(f'eval/{folder_name}'))

    def prepare_data(self, folder: IOFolder) -> Any:
        raise NotImplementedError('Not implemented yet')

    def eval_model(self, model: MLM) -> Any:
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
        return {f.name: f.read() for f in folder.get_all_files(extension='.json')}

    def eval_model(self, model: MLM) -> dict:
        with torch.no_grad():
            keys = ['targ1', 'targ2', 'attr1', 'attr2']
            result = {}
            for test_name, test in nested_loop(self.data.items(), progress_bar=f'SEAT'):
                encoded_sentences = model.tokenize(sum([test[t]['examples'] for t in keys], []))
                sentences_embedded = model.get_hidden_states(encoded_sentences, layers=-1, return_cls=True).cpu()  # (bs, dim)
                c = 0
                for t in keys:
                    sents = test[t]['examples']
                    test[t]['encs'] = dict(zip(sents, sentences_embedded[c:c + len(sents)]))
                    c += len(sents)

                esize, pval = run_test(test, 100000)
                result[test_name] = {'effect_size': round(esize, 4), 'p_val': round(pval, 4)}
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
        return {f.name: f.read() for f in folder.get_all_files(extension='.json')}

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

            pbar = tqdm(desc=f"LPBS", total=total_iterations)
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
                    for batch in nested_loop(all_templates, batch_size=32):
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
                    'bias_score': round(ILPS.mean(dim=(0, 1)).item(), 4),
                    'bias_score_std': round(ILPS.std(dim=(0, 1)).item(), 4)
                }
            pbar.close()
            return result


class Entailment(Metric):
    def __init__(self, metric_name: str):
        assert metric_name in {'mrpc', 'stsb', 'rte', 'wnli'}
        super().__init__(metric_name=metric_name, folder_name='glue')

    def prepare_data(self, folder: IOFolder):
        """ Dataset:
                    'sentence1': str,
                    'sentence2': str,
                    'label': float,
                    ('use_someone': int,
                    'pronoun_class': int,
                    'subject_is_occupation': int)
                """
        return folder.read_file(f'validation/{self.metric_name}.parquet')

    def eval_model(self, model: MLM):
        with torch.no_grad():
            model = model.eval_modus()
            logits = model.entailment(list(self.data['sentence1']), list(self.data['sentence2']))
            predictions = (logits <= 0).float()
            labels = torch.tensor(self.data['label'], dtype=torch.float32, device=predictions.device)
            score = torch.abs(labels - predictions).mean().item()
            return round(100 * score, 2)


class MRPC(Entailment):
    def __init__(self):
        super().__init__('mrpc')


class STS_B(Entailment):
    def __init__(self):
        super().__init__('stsb')

    def eval_model(self, model: MLM):
        with torch.no_grad():
            model = model.eval_modus()
            logits = model.entailment(list(self.data['sentence1']), list(self.data['sentence2']))
            predictions = torch.sigmoid(logits)
            score = pearson_correlation_coefficient(self.data['label'], predictions)
            return round(100 * score, 2)


class RTE(Entailment):
    def __init__(self):
        super().__init__('rte')


class WNLI(Entailment):
    def __init__(self):
        super().__init__('wnli')


class WSC(Metric):
    def __init__(self, metric_name='wsc', folder_name='glue'):
        super().__init__(metric_name=metric_name, folder_name=folder_name)

    def prepare_data(self, folder: IOFolder):
        return folder.read_file(f'validation/{self.metric_name}.parquet')

    def eval_model(self, model: MLM):
        """ Dataset:
            'sentence': str,
            'subject_idx': subject_index // 2,
            'label': float
        """
        with torch.no_grad():
            model = model.eval_modus()

            #(n_sentences)
            logits = model.coref(self.data['sentence'], self.data['subject_idx'])
            y_pred = (logits <= 0).float()

            y_true = torch.tensor(self.data['label'], device=y_pred.device, dtype=torch.float32)

            score = torch.abs(y_pred - y_true).mean().item()
            return round(100 * score, 2)


class WinoGender(WSC):
    def __init__(self):
        super().__init__(metric_name='winogender', folder_name='winogender')

    def prepare_data(self, folder: IOFolder):
        return folder.read_file('winogender.parquet')


class SST2(Metric):
    def __init__(self):
        super().__init__(metric_name='sst2', folder_name='glue')

    def prepare_data(self, folder: IOFolder):
        """ Dataset:
            'sentence': str,
            'label': float,
        """
        return folder.read_file(f'validation/{self.metric_name}.parquet')

    def eval_model(self, model: MLM):
        with torch.no_grad():
            model = model.eval_modus()
            logits = model.sentiment_analysis(list(self.data['sentence']))
            predictions = (logits <= 0).float()
            labels = torch.tensor(self.data['label'], dtype=torch.float32, device=predictions.device)
            score = torch.abs(labels - predictions).mean().item()
            return round(100 * score, 2)


class MetricsRegister:
    def __init__(self, constructors: dict[str, list[type]]):
        self.metric_constructors = constructors
        self.metrics = {}

    def __getitem__(self, objective) -> list[Metric]:
        if objective not in self.metrics:
            self.metrics[objective] = [cls() for cls in self.metric_constructors[objective]]
        return self.metrics[objective]


METRIC_REGISTER = MetricsRegister({
    'kaneko': [SEAT, LPBS],
    'mrpc': [SEAT, LPBS, MRPC],
    'stsb': [SEAT, LPBS, STS_B],
    'rte': [SEAT, LPBS, RTE],
    'wnli': [SEAT, LPBS, WNLI],
    'sst2': [SEAT, LPBS, SST2],
    'wsc': [SEAT, LPBS, WSC, WinoGender]
})
"""
METRIC_REGISTER = MetricsRegister({
    'kaneko': [],
})
"""


def run_metrics(model: MLM) -> dict:
    return {metric.metric_name: metric.eval_model(model) for metric in METRIC_REGISTER[model.config.objective]}

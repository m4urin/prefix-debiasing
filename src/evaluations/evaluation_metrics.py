import copy
from math import ceil
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dependencies.weat import run_test
from src.evaluations.evaluation_utils import get_mask_topk, permutation_test
from src.language_models.language_model import LanguageModel
from src.data.structs.model_config import ModelConfig
from src.utils.files import get_folder, IOFolder
from src.utils.functions import nested_loop
from src.utils.pytorch import pearson_correlation_coefficient, fix_string_batch


class Metric:
    def __init__(self, metric_name: str, folder_name: str):
        self.metric_name = metric_name
        self.data = self.prepare_data(get_folder(f'data/eval/{folder_name}'))

    def prepare_data(self, folder: IOFolder) -> Any:
        raise NotImplementedError('Not implemented yet')

    def eval_model(self, model: LanguageModel) -> Any:
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

    def eval_model(self, model: LanguageModel) -> dict:
        with torch.no_grad():
            keys = ['targ1', 'targ2', 'attr1', 'attr2']
            result = {}
            for test_name, test in nested_loop(self.data.items(), progress_bar=f'SEAT'):
                encoded_sentences = model.tokenize(sum([test[t]['examples'] for t in keys], []))
                # (bs, dim)
                sentences_embedded = model.get_embeddings(encoded_sentences, output_cls_token=True).cpu().numpy()
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

    Kurita, Keita, et al. "Measuring bias in contextualized word representations."
    arXiv preprint arXiv:1906.07337 (2019).
    https://arxiv.org/pdf/1906.07337
    """

    def __init__(self):
        super().__init__(metric_name='lpbs', folder_name='lpbs')

    def prepare_data(self, folder: IOFolder):
        return {f.name: f.read() for f in folder.get_all_files(extension='.json')}

    def eval_model(self, model: LanguageModel) -> dict:
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
                print(f"The following words cannot be converted to a single token by '{model.config.model_name}': "
                      f"{ignored_words}, \n"
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
                    p_prior = model.get_mask_probabilities(model.tokenize(masked_sentences), target_ids)
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
                        p = model.get_mask_probabilities(model.tokenize(batch), target_ids)[:, 0]
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


class GLUETest(Metric):
    def __init__(self, metric_name: str):
        assert metric_name in {'sst2', 'mrpc', 'stsb', 'rte', 'wnli', 'wsc'}
        super().__init__(metric_name=metric_name, folder_name='glue')

    def prepare_data(self, folder: IOFolder):
        return folder.read_file(f'validation/{self.metric_name}.parquet')

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

        # logits
        return model.get_cls_predictions(enc, subject_idx)

    def eval_model(self, model: LanguageModel):
        with torch.no_grad():
            model.eval()
            logits = []
            for batch in tqdm(DataLoader(self.data, batch_size=32), desc=self.metric_name):
                # (bs, 1)
                logits.append(self.process_batch(model, batch))
            # (n, 1)
            logits = torch.cat(logits, dim=0)
            labels = torch.tensor(self.data['label'], dtype=torch.float32, device=logits.device).unsqueeze(-1)
            score = 1.0 - torch.abs(labels - (logits > 0).float()).mean().item()
            return round(100 * score, 2)

"""
class WinoGender(Metric):
    def __init__(self):
        super().__init__(metric_name='winogender', folder_name='winogender')

    def prepare_data(self, folder: IOFolder):
        return folder.read_file('winogender.parquet')

    def get_subset_data(self, dataset, pronoun_class=None, use_someone=None, subject_is_occupation=None):
        if pronoun_class is None:
            pronoun_class = [0, 1, 2]
        elif not isinstance(pronoun_class, list):
            pronoun_class = [pronoun_class]

        if use_someone is None:
            use_someone = [0, 1]
        elif not isinstance(use_someone, list):
            use_someone = [use_someone]

        if subject_is_occupation is None:
            subject_is_occupation = [0, 1]
        elif not isinstance(subject_is_occupation, list):
            subject_is_occupation = [subject_is_occupation]

        return dataset.filter(lambda x: x['pronoun_class'] in pronoun_class and
                                        x['use_someone'] in use_someone and
                                        x['subject_is_occupation'] in subject_is_occupation)

    def eval_model(self, model: LanguageModel):
         Dataset:
            'sentence': str,
            'subject_idx': subject_index // 2,
            'label': float
        
        with torch.no_grad():
            model = model.eval_modus()
            dataset = self.get_subset_data(self.data, use_someone=0)

            enc = model.tokenize_with_spans(fix_string_batch(batch['sentence']))
            subject_idx = batch['subject_idx']

            all_preds = (model.coref(dataset['sentence'], dataset['subject_idx']) > 0).int()
            all_trues = torch.tensor(dataset['label'], dtype=torch.float32, device=all_preds.device)
            result = {'all': {
                'true_preds': round((100 * all_preds.float().mean()).item(), 2),
                'acc': round((100 * (1.0 - torch.abs(all_trues - all_preds.float()).mean())).item(), 2)
            }}

            dataset = dataset.to_dict()
            dataset['pred'] = all_preds.tolist()
            dataset = Dataset.from_dict(dataset)

            for occ, occ_name in enumerate(['not_occupation', ' is_occupation']):
                for gender, gender_name in enumerate(['male', 'female', 'neutral']):
                    subset = self.get_subset_data(dataset, pronoun_class=gender, subject_is_occupation=occ)
                    y_true = torch.tensor(subset['label'], dtype=torch.float32, device=all_preds.device)
                    y_pred = torch.tensor(subset['pred'], dtype=torch.float32, device=all_preds.device)
                    result[f'{occ_name}_{gender_name}_pronoun'] = {
                        'true_preds': round((100 * y_pred.mean()).item(), 2),
                        'acc': round((100 * (1.0 - torch.abs(y_true - y_pred).mean())).item(), 2)
                    }

            return result
"""

class ProbeTest(Metric):
    def __init__(self):
        super().__init__('probe', 'probe')

    def prepare_data(self, folder: IOFolder):
        return {n: folder.read_file(f'{n}.parquet') for n in ['eval', 'stereotypes']}

    def get_result(self, model: LanguageModel, data: Dataset, progress_bar, batch_size):
        logits = []
        for batch in DataLoader(data, batch_size=batch_size):
            enc = model.tokenize_with_spans(fix_string_batch(batch['sentences']))
            # (bs, dim)
            embeddings = model.get_span_embeddings(enc)
            # (bs, 1)
            logits.append(model.cls_head(embeddings))
            progress_bar.update()
        # (n_samples, 2)
        logits = torch.cat(logits, dim=0)

        # (n_samples, 1)
        pred = logits.softmax(dim=-1)[..., 1]
        conf = round(torch.abs(pred - 0.5).mean().item(), 4)

        # (n_samples, 1)
        labels = torch.tensor(data['label'], dtype=torch.float32, device=pred.device)
        correct = 1.0 - torch.abs(labels - pred)
        acc = round(100 * correct.mean().item(), 2)
        return correct, acc, conf

    def eval_model(self, model: LanguageModel):
        batch_size = 32
        with torch.no_grad():
            progress_bar = tqdm(desc='Probe',
                                total=sum(ceil(len(d) / batch_size) for d in self.data.values()))

            eval_correct, eval_acc, eval_conf = self.get_result(model, self.data['eval'], progress_bar, batch_size)
            st_correct, st_acc, st_conf = self.get_result(model, self.data['stereotypes'], progress_bar, batch_size)
            progress_bar.close()

            p_value = round(permutation_test(eval_correct, st_correct).item(), 5)
            return {
                'gender_acc': eval_acc,
                'stereotype_acc': st_acc,
                'stereotype_conf': st_conf,
                'p_value': p_value
            }


class QualityTest(Metric):
    def __init__(self):
        super().__init__('quality', 'quality')

    def prepare_data(self, folder: IOFolder):
        return [
            ('I am walking my [MASK] on a leash. Otherwise, he might bite someone.', 'dog'),
            ('Two men in the bar start a [MASK].', 'conversation'),
            ('My [MASK] gave birth to me.', 'mother'),
            ('I unlock the door and open the [MASK].', 'door'),
            ('With these sports [MASK], I can run even faster.', 'shoes'),
            ('Mary is a doctor. [MASK] saved many people!', 'she')
        ]

    def eval_model(self, model: LanguageModel):
        mask_token = model.tokenizer.convert_ids_to_tokens(model.tokenizer.mask_token_id)
        return get_mask_topk(model, [s.replace('[MASK]', mask_token) for s, _ in self.data], k=10)

    def compare_models(self, model_configs: list[ModelConfig], top_k_words: list[list[str]]):
        for s_i, (sent, expected) in enumerate(self.data):
            print(f"{sent}\n"
                  f"\texpected: {expected}")
            for c, topks in zip(model_configs, top_k_words):
                print(f"{c}: {topks[s_i]}")


def run_metrics(model: LanguageModel) -> dict:
    metrics = [] # [SEAT(), LPBS()] TODO
    if model.config.is_downstream():
        if model.config.downstream_task == 'probe':
            pass
            #metrics += [ProbeTest()]
        else:
            metrics += [GLUETest(model.config.downstream_task)]

        #if model.config.downstream_task == 'wsc':
        #    metrics += [WinoGender()]

    return {m.metric_name: m.eval_model(model) for m in metrics}

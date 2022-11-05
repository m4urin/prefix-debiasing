from math import ceil

import torch
from tqdm import tqdm
import copy

from dependencies.weat import run_test
from src.language_models import MaskedLanguageModel
from src.utils import batched, IOFolder, DATA_DIR


class Metric:
    def __init__(self, name: str):
        self.name = name
        self.data = self.prepare_data(DATA_DIR['eval', name])

    def prepare_data(self, folder: IOFolder):
        raise NotImplementedError('Not implemented yet')

    def eval_model(self, model: MaskedLanguageModel) -> dict:
        raise NotImplementedError('Not implemented yet')


class SEAT(Metric):
    """
    Sentence Encoder Association Test (SEAT)

    May, Chandler, et al. "On measuring social biases in sentence encoders." arXiv preprint arXiv:1903.10561 (2019).
    https://arxiv.org/pdf/1903.10561
    """

    def __init__(self):
        super().__init__(name='seat')

    def prepare_data(self, folder: IOFolder):
        return {f.name: f.read() for f in folder.get_files(extension='.json')}

    def eval_model(self, model: MaskedLanguageModel) -> dict:
        with torch.no_grad():
            keys = ['targ1', 'targ2', 'attr1', 'attr2']
            result = {}
            for test_name, test in tqdm(self.data.items(), desc=f'SEAT ({model.config})'):
                encoded_sentences = model.tokenize(sum([test[t]['examples'] for t in keys], []))
                sentences_embedded = model.get_sentence_embeddings(encoded_sentences, layer=-1).cpu()  # (bs, dim)
                c = 0
                for t in keys:
                    sents = test[t]['examples']
                    test[t]['encs'] = dict(zip(sents, sentences_embedded[c:c + len(sents)]))
                    c += len(sents)

                esize, pval = run_test(test, 100000)
                result[test_name] = {'effect_size': esize, 'p_val': pval}  # ,
                # 'X': test["targ1"]["category"], 'Y': test["targ2"]["category"],
                # 'A': test["attr1"]["category"], 'B': test["attr2"]["category"]}
            return result


class LPBS(Metric):
    """
    Log Probability Bias Score (LPBS)

    Kurita, Keita, et al. "Measuring bias in contextualized word representations." arXiv preprint arXiv:1906.07337 (2019).
    https://arxiv.org/pdf/1906.07337
    """

    def __init__(self):
        super().__init__(name='lpbs')

    def prepare_data(self, folder: IOFolder):
        return {f.name: f.read() for f in folder.get_files(extension='.json')}

    def eval_model(self, model: MaskedLanguageModel) -> dict:
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

            pbar = tqdm(desc=f"LPBS ({model.config})", total=total_iterations)
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


METRICS: dict[str, Metric] = {'seat': SEAT(), 'lpbs': LPBS()}


def run_metrics(model: MaskedLanguageModel) -> dict:
    with torch.no_grad():
        return {name: metric.eval_model(model) for name, metric in METRICS.items()}

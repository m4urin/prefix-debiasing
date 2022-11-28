import re

import nltk
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from unidecode import unidecode

from src.MLM import MLM
from src.utils.config import MODEL_NAMES, ModelConfig
from src.utils.files import get_folder, read_file
from src.utils.functions import stack_dicts
from src.utils.pytorch import fix_string_batch

MAX_SENTENCE_LENGTH = 450
FORCE = False
KANEKO_SPLIT = False


def get_kaneko(model_name: str):
    attr, stereo = _kaneko_split_sentences()
    attr: Dataset = attr
    stereo: Dataset = stereo
    v_a = _kaneko_va(model_name)
    return attr, stereo, v_a


def sst2(train_eval: str):
    return _get_sent_dataset('sst2', train_eval)


def mrpc(train_eval: str):
    return _get_sent1_sent2_dataset('mrpc', train_eval)


def stsb(train_eval: str):
    return _get_sent1_sent2_dataset('stsb', train_eval)


def rte(train_eval: str):
    return _get_sent1_sent2_dataset('rte', train_eval)


def wnli(train_eval: str):
    return _get_sent1_sent2_dataset('wnli', train_eval)


def wsc(train_eval: str):
    folder = get_folder(f'eval/glue/{train_eval}', create_if_not_exists=True)
    if not FORCE and folder.file_exists(f'wsc.parquet'):
        dataset: Dataset = folder.read_file(f'wsc.parquet')
        return dataset
    result = []
    for x in tqdm(load_dataset('super_glue', 'wsc.fixed')[train_eval], desc=f'Preprocess WSC ({train_eval})'):
        if len(x['text']) > MAX_SENTENCE_LENGTH:
            continue
        sample = _process_subject_pronoun(text=x['text'],
                                          subject_index=x['span1_index'],
                                          pronoun_index=x['span2_index'],
                                          subject_text=x['span1_text'],
                                          pronoun_text=x['span2_text'],
                                          label=x['label'])
        result.append(sample)
    result = Dataset.from_dict(stack_dicts(result))

    if train_eval == 'validation':
        def _add_len(_x):
            _x['sent_lengths'] = -len(_x['sentence'])
            return _x
        result = result.map(_add_len).sort(column='sent_lengths').remove_columns(column_names=['sent_lengths'])

    folder.write_file(f'wsc.parquet', result)
    return result


def wino_gender(train_eval: str = None):
    folder = get_folder('eval/winogender', create_if_not_exists=True)
    if not FORCE and folder.file_exists('winogender.parquet'):
        dataset: Dataset = folder.read_file('winogender.parquet')
        return dataset

    pronouns: dict = folder.read_file('pronouns.json')
    templates_df: pd.DataFrame = folder.read_file('templates.tsv')

    gender_data = []

    all_tags = {"$OCCUPATION", "$PARTICIPANT", *pronouns.keys()}
    regex_tokenizer = r'(?<!\w)(' + '|'.join([re.escape(x) for x in all_tags]) + r')(?!\w)'
    regex_tokenizer = nltk.RegexpTokenizer(regex_tokenizer, gaps=True)  # .tokenize_sents

    for i, (occupation, participant, label, sentence) in tqdm(templates_df.iterrows(),
                                                              desc='winogender', total=len(templates_df)):
        data = [(0, participant, "$PARTICIPANT", label), (1, occupation, "$OCCUPATION", 1-label)]
        for j in range(2):
            is_occupation, subject, subject_tag, true_label = data[j]
            _, other_subject, other_subject_tag, _ = data[1-j]
            tokens = regex_tokenizer.tokenize('# ' + sentence.replace(other_subject_tag, other_subject) + ' #')
            tokens[0], tokens[-1] = tokens[0][2:], tokens[-1][:-2]
            subject_index = tokens.index(subject_tag)
            pronoun_index, pronoun_type = None, None
            for _pronoun_type in pronouns.keys():
                if _pronoun_type in tokens:
                    pronoun_index = tokens.index(_pronoun_type)
                    pronoun_type = _pronoun_type
                    break
            # tokens: ["The ", subject_tag, " told the customer that ", "$NOM_PRONOUN", " could pay with cash."]

            for use_someone, sub in enumerate([subject, 'someone']):
                for pronoun_class in range(3):
                    result = [t for t in tokens]
                    result[subject_index] = sub
                    result[pronoun_index] = pronouns[pronoun_type][pronoun_class]
                    if use_someone == 1 and subject_index == 1:
                        result[0] = ''
                        result[1] = result[1][0].upper() + result[1][1:]
                    result = {
                        'sentence': result,
                        'subject_idx': subject_index // 2,
                        'label': true_label,
                        'use_someone': use_someone,
                        'pronoun_class': pronoun_class,
                        'subject_is_occupation': is_occupation
                    }
                    gender_data.append(result)

    gender_data = Dataset.from_dict(stack_dicts(gender_data))

    if train_eval == 'validation':
        def _add_len(_x):
            _x['sent_lengths'] = -len(_x['sentence'])
            return _x
        gender_data = gender_data.map(_add_len).sort(column='sent_lengths')\
            .remove_columns(column_names=['sent_lengths'])

    folder.write_file('winogender.parquet', gender_data)
    return gender_data


def _get_sent1_sent2_dataset(dataset_name: str, train_eval: str):
    assert dataset_name in {'mrpc', 'stsb', 'rte', 'wnli'}, f"Dataset '{dataset_name}' is not supported."
    assert train_eval in {'train', 'validation'}, f"Subset '{train_eval}' is not supported for dataset '{dataset_name}'."

    folder = get_folder(f'eval/glue/{train_eval}', create_if_not_exists=True)
    if not FORCE and folder.file_exists(f'{dataset_name}.parquet'):
        dataset: Dataset = folder.read_file(f'{dataset_name}.parquet')
        return dataset
    result = []
    for x in tqdm(load_dataset('glue', dataset_name)[train_eval], desc=f'Preprocess {dataset_name} ({train_eval})'):
        if len(x['sentence1']) > MAX_SENTENCE_LENGTH or len(x['sentence2']) > MAX_SENTENCE_LENGTH:
            continue
        result.append({'sentence1': x['sentence1'], 'sentence2': x['sentence2'], 'label': float(x['label'])})
    result = Dataset.from_dict(stack_dicts(result))

    if dataset_name == 'stsb':
        def _normalize(_x):
            _x['label'] = round(_x['label'] / 5.0, 3)
            return _x
        result = result.map(_normalize)

    if train_eval == 'validation':
        def _add_len(_x):
            _x['sent_lengths'] = -(10000 * len(_x['sentence1']) + len(_x['sentence2']))
            return _x
        result = result.map(_add_len).sort(column='sent_lengths').remove_columns(column_names=['sent_lengths'])

    folder.write_file(f'{dataset_name}.parquet', result)
    return result


def _get_sent_dataset(dataset_name: str, train_eval: str):
    assert dataset_name in {'sst2'}, f"Dataset '{dataset_name}' is not supported."
    assert train_eval in {'train', 'validation'}, f"Subset '{train_eval}' is not supported for dataset '{dataset_name}'."

    folder = get_folder(f'eval/glue/{train_eval}', create_if_not_exists=True)
    if not FORCE and folder.file_exists(f'{dataset_name}.parquet'):
        dataset: Dataset = folder.read_file(f'{dataset_name}.parquet')
        return dataset
    result = []
    for x in tqdm(load_dataset('glue', dataset_name)[train_eval], desc=f'Preprocess {dataset_name} ({train_eval})'):
        if len(x['sentence']) > MAX_SENTENCE_LENGTH:
            continue
        result.append({'sentence': x['sentence'], 'label': x['label']})
    result = Dataset.from_dict(stack_dicts(result))

    if train_eval == 'validation':
        def _add_len(_x):
            _x['sent_lengths'] = -len(_x['sentence'])
            return _x
        result = result.map(_add_len).sort(column='sent_lengths').remove_columns(column_names=['sent_lengths'])

    folder.write_file(f'{dataset_name}.parquet', result)
    return result


def _process_subject_pronoun(text: str, subject_index: int, pronoun_index: int,
                             subject_text: str, pronoun_text: str, label: int):
    """ Text is split on <space>
    Example:
        text: "Mark told Pete many lies about himself, which Pete included in his book.
               He should have been more skeptical."
        subject_index: 0
        pronoun_index: 13
        subject_text: "Mark"
        pronoun_text: "He"
        label: 0 (False)
    Becomes:
        sentence: ["", "Mark", " told Pete many lies about himself, which Pete included in his book. ",
                   "He", " should have been more skeptical."]
        subject_idx: 0
        label: 0 (False)
"""
    text = text.split()
    reverse = subject_index > pronoun_index
    if reverse:
        temp = subject_index, subject_text
        subject_index, subject_text = pronoun_index, pronoun_text
        pronoun_index, pronoun_text = temp

    index1_end = subject_index + len(subject_text.split())
    index2_end = pronoun_index + len(pronoun_text.split())
    lengths = [(0, subject_index), (subject_index, index1_end), (index1_end, pronoun_index),
               (pronoun_index, index2_end), (index2_end, len(text))]

    result = []
    for i, (j, k) in enumerate(lengths):
        sub = ' '.join(text[j:k])
        if len(sub) > 0:
            if i == 0 or i == 2:
                sub = sub + ' '
            if i == 2 or i == 4:
                sub = ' ' + sub
        result.append(sub)

    return {
        'sentence': result,
        'subject_idx': int(reverse),
        'label': label
    }


def _kaneko_split_sentences(return_attributes_only=False):
    global KANEKO_SPLIT
    folder = get_folder('train/kaneko', create_if_not_exists=True)
    if (not FORCE or KANEKO_SPLIT) and folder.file_exists('attributes.parquet'):
        attr_sentences: Dataset = folder.read_file('attributes.parquet')
        if return_attributes_only:
            return attr_sentences
        elif folder.file_exists('stereotypes.parquet'):
            stereotype_sentences: Dataset = folder.read_file('stereotypes.parquet')
            return attr_sentences, stereotype_sentences

    data = [line for line in set(folder.read_file('news-commentary-v15.txt')) if 20 < len(line) <= MAX_SENTENCE_LENGTH]
    male_words = set([line for line in folder.read_file('male.txt') if len(line) > 0])
    female_words = set([line for line in folder.read_file('female.txt') if len(line) > 0])
    stereotypes = set([line for line in folder.read_file('stereotype.txt') if len(line) > 0])
    all_words = list(male_words) + list(female_words) + list(stereotypes)

    def word_list_regex(_words: list[str]):
        word_list = sorted(list(_words), key=lambda x: len(x))  # sort on word size
        word_list = [re.escape(_x) for _x in word_list]  # escape special characters
        word_list = [f'[{_x[:1].upper()}{_x[:1].lower()}]{_x[1:].lower()}' for _x in word_list]
        return r'(?<!\w)(' + '|'.join(word_list) + r')(?!\w)'

    word_finder = nltk.RegexpTokenizer(word_list_regex(all_words), gaps=True).tokenize_sents

    def split_sentences(x):
        k = len(x) // 1000
        A, S = [], []
        for i in trange((len(x) // k) + 1, desc='Kaneko: split sentences'):
            batch = [f'# {unidecode(line)} #' for line in x[i * k:(i + 1) * k]]
            batch = [s for s in word_finder(batch) if len(s) == 3]
            batch = [(s[0][2:], s[1], s[2][:-2]) for s in batch]  # remove '# .. #'
            for s in batch:
                if s[1].lower() in stereotypes:
                    S.append(s)
                else:
                    A.append(s)

        def predict_sent_size(_sent):
            return sum(len(_sent[__i].split()) + (0.01 * len(_sent[__i])) for __i in range(0, len(_sent), 2))

        A, S = sorted(A, key=predict_sent_size), sorted(S, key=predict_sent_size)

        k = min(len(A), len(S))

        A = A[:19200]  # remove 30% longest sentences
        S = S[:19200]  # remove 30% longest sentences

        return A, S

    attr_sentences, stereotype_sentences = split_sentences(data)

    attr_sentences = Dataset.from_dict({'sentences': attr_sentences}).shuffle(seed=42)
    stereotype_sentences = Dataset.from_dict({'sentences': stereotype_sentences}).shuffle(seed=777)

    folder.write_file('attributes.parquet', attr_sentences)
    folder.write_file('stereotypes.parquet', stereotype_sentences)

    KANEKO_SPLIT = True

    if return_attributes_only:
        return attr_sentences
    else:
        return attr_sentences, stereotype_sentences


def _kaneko_va(model_name: str, batch_size=32):
    folder = get_folder('train/kaneko/va', create_if_not_exists=True)
    file_name = f'{model_name}.pt'
    if not FORCE and folder.file_exists(file_name):
        v_a_from_file: torch.Tensor = folder.read_file(file_name)
        return v_a_from_file

    with torch.no_grad():
        model = MLM.from_config(ModelConfig(model_name, 'base', 'kaneko'))
        v_a = {}
        # CALCULATE V_a
        for x in tqdm(DataLoader(_kaneko_split_sentences(return_attributes_only=True), batch_size=batch_size),
                      desc=f'V_a for {model_name}'):
            x = fix_string_batch(x['sentences'])
            enc = model.tokenize_with_spans(x)
            # (2, bs, n_layers, dim)
            span_embeddings = model.get_span_embeddings(enc, reduce='both').detach()

            for i in range(len(x)):
                word = x[i][1].lower().strip()
                if word not in v_a:
                    v_a[word] = ([], [])
                for j in range(2):
                    v_a[word][j].append(span_embeddings[j, i])

        v_a = [[torch.stack(v_first).mean(0), torch.stack(v_mean).mean(0)] for v_first, v_mean in v_a.values()]
        v_a = torch.stack(sum(v_a, []))
        folder.write_file(file_name, v_a)
        return v_a


if __name__ == '__main__':
    """
    PREPROCESS ALL DATA
    """
    FORCE = True

    # kaneko
    for model_name in MODEL_NAMES:
        get_kaneko(model_name)

    # train/eval datasets
    for train_eval in ['train', 'validation']:
        for f in [sst2, mrpc, stsb, rte, wnli, wsc, wino_gender]:
            f(train_eval)

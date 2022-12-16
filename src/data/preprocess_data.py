import re

import nltk
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from unidecode import unidecode

from src.data.structs.model_config import ModelConfig, MODEL_NAMES
from src.language_models.language_model import LanguageModel
from src.utils.files import get_folder, read_file, exists, write_file
from src.utils.functions import stack_dicts, nested_loop

MAX_SENTENCE_LENGTH = 450
FORCE = False
KANEKO_SPLIT = False


def get_if_stored(file_paths, ignore_force=False):
    is_list = isinstance(file_paths, list)
    if not is_list:
        file_paths = [file_paths]
    if (FORCE and not ignore_force) or not all(exists(p) for p in file_paths):
        return None
    datasets = [read_file(p) for p in file_paths]
    if is_list:
        return tuple(datasets)
    else:
        return datasets[0]


def get_kaneko_split() -> dict[str, dict]:
    global KANEKO_SPLIT
    names = ['all_attributes', 'male_attributes', 'female_attributes',
             'all_stereotypes', 'male_stereotypes', 'female_stereotypes']
    dataset: dict[str, dict] = get_if_stored('data/train/kaneko/sentence_splits.pt',
                                             ignore_force=KANEKO_SPLIT)
    if dataset is not None:
        return dataset

    folder = get_folder('data/train/kaneko', create_if_not_exists=True)

    data = set([line for line in folder.read_file('news-commentary-v15.txt') if 5 < len(line) <= MAX_SENTENCE_LENGTH])
    word_groups = {n: set(folder.read_file(f'{n}.txt')) for n in names}
    all_words = set(sum([list(w_list) for w_list in word_groups.values()], []))

    to_groups = {w: set() for w in all_words}
    for n, w_list in word_groups.items():
        for w in w_list:
            to_groups[w].add(n)
    to_groups = {w: list(v) for w, v in to_groups.items()}

    def _word_list_regex(x):
        x = sorted(list(x), key=lambda _x: len(_x))  # sort on word size
        x = [re.escape(_x) for _x in x]  # escape special characters
        x = [f'[{_x[:1].upper()}{_x[:1].lower()}]{_x[1:].lower()}' for _x in x]
        x = r'(?<!\w)(' + '|'.join(x) + r')(?!\w)'
        return nltk.RegexpTokenizer(x, gaps=True).tokenize_sents

    word_finder = _word_list_regex(all_words)

    sentences = {w: [] for w in all_words}
    for batch in nested_loop(data, batch_size=1000, progress_bar='Split sentences'):
        batch = [f'# {unidecode(line)} #' for line in batch]
        batch = [s for s in word_finder(batch) if len(s) == 3]
        batch = [(s[0][2:], s[1], s[2][:-2]) for s in batch]  # remove '# .. #'
        for s in batch:
            sentences[s[1].lower()].append(s)

    word_groups = {g: list(words) for g, words in word_groups.items()}
    sentences = {w: shuffle(v) for w, v in sentences.items()}
    sentence_splits = {'word_groups': word_groups, 'sentences': sentences, 'to_groups': to_groups}
    folder.write_file('sentence_splits.pt', sentence_splits)
    KANEKO_SPLIT = True
    return sentence_splits


def get_ml_head_data():
    dataset: Dataset = get_if_stored('data/train/ml_head/data.parquet')
    if dataset is not None:
        return dataset

    folder = get_folder('data/train/ml_head', create_if_not_exists=True)
    batch_size = 64
    sentences = get_kaneko_split()
    sentences = {w: [''.join(s) for s in v] for w, v in sentences['sentences'].items()}
    data_attr = []
    N = 10000
    while len(data_attr) < N:
        for w in set(sentences.keys()):
            if len(sentences[w]) == 0:
                del sentences[w]
            else:
                data_attr.append(sentences[w].pop())
    data_attr = data_attr[:N]

    data_rand = [line for line in read_file('data/train/kaneko/news-commentary-v15.txt') if 20 < len(line) < 160]
    data_rand = list(set(data_rand) - set(data_attr))
    _, data_rand = train_test_split(data_rand, test_size=20000, shuffle=True)

    data = sorted(data_attr + data_rand, key=lambda x: len(x))
    data = data[:len(data) - (len(data) % batch_size)]
    dataset = Dataset.from_dict({'sentence': data})
    folder.write_file('data.parquet', dataset)
    return dataset


def get_probe_data():
    dataset: tuple[Dataset, Dataset, Dataset] = get_if_stored([f'data/eval/probe/{n}.parquet'
                                                               for n in ['train', 'eval', 'stereotypes']])
    if dataset is not None:
        return dataset

    data = get_kaneko_split()

    word_groups: dict[str, list[str]] = data['word_groups']
    sentences = data['sentences']

    names = ['male_attributes', 'female_attributes', 'male_stereotypes', 'female_stereotypes']
    n_samples = [5000, 5000, 1024, 1024]
    data = {}
    for n, s in zip(names, n_samples):
        d = []
        while len(d) < s:
            if len(word_groups[n]) == 0:
                raise Exception(f"{n}, max: {len(d)}")
            for w in [_w for _w in word_groups[n]]:
                if len(sentences[w]) == 0:
                    word_groups[n].remove(w)
                else:
                    d.append(sentences[w].pop())
        data[n] = d[:s]
    
    m_attr_train, m_attr_eval = train_test_split(data['male_attributes'], test_size=1472)
    f_attr_train, f_attr_eval = train_test_split(data['female_attributes'], test_size=1472)
    m_stereo = data['male_stereotypes']
    f_stereo = data['female_stereotypes']

    train_dataset = Dataset.from_dict({
        'sentences': m_attr_train + f_attr_train,
        'label': [0] * len(m_attr_train) + [1] * len(f_attr_train)
    }).shuffle(seed=42)
    eval_dataset = Dataset.from_dict({
        'sentences': m_attr_eval + f_attr_eval,
        'label': [0] * len(m_attr_eval) + [1] * len(f_attr_eval)
    })
    stereo_dataset = Dataset.from_dict({
        'sentences': m_stereo + f_stereo,
        'label': [0] * len(m_stereo) + [1] * len(f_stereo)
    })

    folder = get_folder('data/eval/probe', create_if_not_exists=True)
    folder.write_file('train.parquet', train_dataset)
    folder.write_file('eval.parquet', eval_dataset)
    folder.write_file('stereotypes.parquet', stereo_dataset)

    return train_dataset, eval_dataset, stereo_dataset


def get_kaneko_va(model_name: str, batch_size=32):
    file_name = f'{model_name}.pt'
    v_a: torch.Tensor = get_if_stored(f'data/train/kaneko/debias/v_a/{file_name}')
    if v_a is not None:
        return v_a

    folder = get_folder('data/train/kaneko/debias/v_a', create_if_not_exists=True)
    data = get_kaneko_split()
    sentences = data['sentences']
    sentences = sum([sentences[w] for w in data['word_groups']['all_attributes']], [])
    with torch.no_grad():
        model = LanguageModel.from_config(ModelConfig(model_name))
        v_a = {}
        # CALCULATE V_a
        for x in nested_loop(sentences, batch_size=batch_size, progress_bar=f'V_a for {model_name}'):
            enc = model.tokenize_with_spans(x)
            # (bs, n_layers, dim)
            span_embeddings = model.get_span_embeddings(enc, reduce='mean', output_hidden_states=True).detach()

            for i in range(len(x)):
                word = x[i][1].lower().strip()
                if word not in v_a:
                    v_a[word] = []
                v_a[word].append(span_embeddings[i])

        v_a = torch.stack([torch.stack(v).mean(0) for v in v_a.values()])
        folder.write_file(file_name, v_a)
        return v_a


def get_kaneko_data(model_name: str):
    dataset = get_if_stored(['data/train/kaneko/debias/attributes.parquet',
                             'data/train/kaneko/debias/stereotypes.parquet',
                             f'data/train/kaneko/debias/v_a/{model_name}.pt'])
    if dataset is not None:
        dataset: tuple[Dataset, Dataset, torch.Tensor] = dataset
        return dataset

    folder = get_folder('data/train/kaneko/debias', create_if_not_exists=True)
    data = get_kaneko_split()
    word_groups = data['word_groups']
    sentences = data['sentences']
    # 19200
    names = ['male_attributes', 'female_attributes', 'all_stereotypes']
    n_samples = [11800, 7400, 19200]
    data = {}
    for n, s in zip(names, n_samples):
        d = []
        while len(d) < s:
            if len(word_groups[n]) == 0:
                raise Exception(f'{n} max: {len(d)}')
            for w in [_w for _w in word_groups[n]]:
                if len(sentences[w]) == 0:
                    word_groups[n].remove(w)
                else:
                    d.append(sentences[w].pop())
        data[n] = d[:s]

    attr_dataset = Dataset.from_dict({
        'sentences': data['male_attributes'] + data['female_attributes']
    }).shuffle(seed=42)
    stereo_dataset = Dataset.from_dict({
        'sentences': data['all_stereotypes']
    }).shuffle(seed=7)

    folder.write_file('attributes.parquet', attr_dataset)
    folder.write_file('stereotypes.parquet', stereo_dataset)

    return attr_dataset, stereo_dataset, get_kaneko_va(model_name)


def _get_sent_dataset(dataset_name: str, train_eval: str):
    assert dataset_name in {'sst2'}, f"Dataset '{dataset_name}' is not supported."
    assert train_eval in {'train', 'validation'}, f"Subset '{train_eval}' is not supported for dataset '{dataset_name}'."

    dataset = get_if_stored(f'data/eval/glue/{train_eval}/{dataset_name}.parquet')
    if dataset is not None:
        dataset: Dataset = dataset
        return dataset

    folder = get_folder(f'data/eval/glue/{train_eval}', create_if_not_exists=True)
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

    if train_eval == 'train' and dataset_name == 'sst2':
        result, _ = train_test_split(result.shuffle(seed=42), test_size=0.55)
        result = Dataset.from_dict(result)

    folder.write_file(f'{dataset_name}.parquet', result)
    return result


def _get_sent1_sent2_dataset(dataset_name: str, train_eval: str):
    assert dataset_name in {'mrpc', 'stsb', 'rte', 'wnli'}, f"Dataset '{dataset_name}' is not supported."
    assert train_eval in {'train',
                          'validation'}, f"Subset '{train_eval}' is not supported for dataset '{dataset_name}'."

    dataset = get_if_stored(f'data/eval/glue/{train_eval}/{dataset_name}.parquet')
    if dataset is not None:
        dataset: Dataset = dataset
        return dataset

    folder = get_folder(f'data/eval/glue/{train_eval}', create_if_not_exists=True)
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
    dataset = get_if_stored(f'data/eval/glue/{train_eval}/wsc.parquet')
    if dataset is not None:
        dataset: Dataset = dataset
        return dataset

    folder = get_folder(f'data/eval/glue/{train_eval}', create_if_not_exists=True)

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
    dataset = get_if_stored(f'data/eval/winogender/winogender.parquet')
    if dataset is not None:
        dataset: Dataset = dataset
        return dataset

    if train_eval == 'train':
        return None
    folder = get_folder('data/eval/winogender', create_if_not_exists=True)

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

    def _add_len(_x):
        _x['sent_lengths'] = -len(_x['sentence'])
        return _x
    gender_data = gender_data.map(_add_len).sort(column='sent_lengths')\
        .remove_columns(column_names=['sent_lengths'])

    folder.write_file('winogender.parquet', gender_data)
    return gender_data


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


if __name__ == '__main__':

    """
    PREPROCESS ALL DATA
    """
    FORCE = False

    # kaneko
    for model_name in MODEL_NAMES:
        get_kaneko_data(model_name)

    # train/eval datasets
    for train_eval in ['train', 'validation']:
        for f in [sst2, mrpc, stsb, rte, wnli, wsc, wino_gender]:
            f(train_eval)

    for f in [get_ml_head_data, get_probe_data]:
        f()

import re
import nltk
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from unidecode import unidecode

from src.MLM import MLM
from src.utils.files import DATA_DIR
from src.utils.pytorch import fix_string_batch
from src.utils.config import ModelConfig


def word_list_regex(words: list[str]):
    word_list = sorted(list(words), key=lambda x: len(x))  # sort on word size
    word_list = [re.escape(x) for x in word_list]  # escape special characters
    word_list = [f'[{x[:1].upper()}{x[:1].lower()}]{x[1:].lower()}' for x in word_list]
    return r'(?<!\w)(' + '|'.join(word_list) + r')(?!\w)'


def predict_sent_size(x):
    return sum(len(x[i].split()) + (0.01 * len(x[i])) for i in range(0, len(x), 2))


def preprocess_raw_data():
    data = [line for line in set(DATA_DIR['train/kaneko/news-commentary-v15.txt'].read()) if len(line) > 10]
    male_words = set([line for line in DATA_DIR['train/kaneko/male.txt'].read() if len(line) > 0])
    female_words = set([line for line in DATA_DIR['train/kaneko/female.txt'].read() if len(line) > 0])
    stereotypes = set([line for line in DATA_DIR['train/kaneko/stereotype.txt'].read() if len(line) > 0])
    all_words = list(male_words) + list(female_words) + list(stereotypes)

    word_finder = nltk.RegexpTokenizer(word_list_regex(all_words), gaps=True).tokenize_sents

    def split_sentences(x):
        k = len(x) // 1000
        A, S = [], []
        for i in trange((len(x) // k) + 1, desc='split sentences'):
            batch = [f'# {unidecode(line)} #' for line in x[i * k:(i + 1) * k]]
            batch = [s for s in word_finder(batch) if len(s) == 3]
            batch = [(s[0][2:], s[1], s[2][:-2]) for s in batch]  # remove '# .. #'
            for s in batch:
                if s[1].lower() in stereotypes:
                    S.append(s)
                else:
                    A.append(s)
        A, S = sorted(A, key=predict_sent_size), sorted(S, key=predict_sent_size)

        k = min(len(A), len(S))

        A = A[:int(k * 0.7)]  # remove 30% longest sentences
        S = S[:int(k * 0.7)]  # remove 30% longest sentences

        return A, S

    attr_sentences, stereotype_sentences = split_sentences(data)

    attr_sentences = Dataset.from_dict({'sentences': attr_sentences})
    stereotype_sentences = Dataset.from_dict({'sentences': stereotype_sentences})

    DATA_DIR['train/kaneko'].write_file('attributes.parquet', attr_sentences.shuffle())
    DATA_DIR['train/kaneko'].write_file('stereotypes.parquet', stereotype_sentences.shuffle())


def calc_v_a(model: MLM, batch_size=32):
    with torch.no_grad():
        v_a = {}
        # CALCULATE V_a
        for x in tqdm(DataLoader(DATA_DIR['train/kaneko'].read_file('attributes.parquet'),
                                 batch_size=batch_size),
                      desc=f'V_a for {model.config.model_name}'):
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
        print(v_a.size())
        DATA_DIR['train/kaneko'].get_folder('va', create_if_not_exists=True) \
            .write_file(f'{model.config.model_name}.pt', v_a)


def preprocess_v_a():
    for model_name in ["distilbert-base-uncased", "roberta-base"]:
        calc_v_a(MLM.from_config(ModelConfig(model_name, 'base', 'kaneko')))


if __name__ == '__main__':
    preprocess_raw_data()
    preprocess_v_a()

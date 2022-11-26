import re

import nltk
from datasets import load_dataset, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.files import DATA_DIR
from src.utils.functions import stack_dicts

eval_folder = DATA_DIR['eval/winogender']
train_folder = DATA_DIR['train/coref']

train_data = {'sentence': [], 'subject_idx': [], 'label': []}
MAX_LENGTH = 420

dataset = load_dataset('gap')
dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
lens = []
for x in tqdm(dataset, desc='gap'):
    sent_length = len(x['Text'])
    if sent_length > MAX_LENGTH:
        continue
    else:
        lens.append(sent_length)

    for a in ['A', 'B']:
        reverse = False
        index2, text2 = x['Pronoun-offset'], x['Pronoun']
        index1, text1 = x[f'{a}-offset'], x[a]
        if index1 > index2:
            temp = index1, text1
            index1, text1 = index2, text2
            index2, text2 = temp
            reverse = True
        index1_end = index1 + len(text1)
        index2_end = index2 + len(text2)
        lengths = [(0, index1), (index1, index1_end), (index1_end, index2),
                   (index2, index2_end), (index2_end, len(x['Text']))]

        final = []
        for i, (j, k) in enumerate(lengths):
            final.append(x['Text'][j:k])
        train_data['sentence'].append(final)
        train_data['subject_idx'].append(int(reverse))
        label = int(x[f'{a}-coref'])
        train_data['label'].append(label)


dataset = load_dataset('super_glue', 'wsc.fixed')
dataset = concatenate_datasets([dataset['train'], dataset['validation']])
for x in tqdm(dataset, desc='super_glue'):
    sent_length = len(x['text'])
    if sent_length > MAX_LENGTH:
        continue
    else:
        lens.append(sent_length)

    txt_parts = x['text'].split()
    final = []
    reverse = False
    index1, index2 = x['span1_index'], x['span2_index']
    text1, text2 = x['span1_text'], x['span2_text']
    if index1 > index2:
        temp = index1, text1
        index1, text1 = index2, text2
        index2, text2 = temp
        reverse = True
    index1_end = index1 + len(text1.split())
    index2_end = index2 + len(text2.split())
    lengths = [(0, index1), (index1, index1_end), (index1_end, index2),
               (index2, index2_end), (index2_end, len(txt_parts))]

    for i, (j, k) in enumerate(lengths):
        sub = ' '.join(txt_parts[j:k])
        if len(sub) > 0:
            if i == 0 or i == 2:
                sub = sub + ' '
            if i == 2 or i == 4:
                sub = ' ' + sub
        final.append(sub)
    train_data['sentence'].append(final)
    train_data['subject_idx'].append(int(reverse))
    train_data['label'].append(x['label'])


gender_data = []
pronouns: dict = eval_folder.read_file('pronouns.json')
all_tags = {"$OCCUPATION", "$PARTICIPANT", *pronouns.keys()}
regex_tokenizer = r'(?<!\w)(' + '|'.join([re.escape(x) for x in all_tags]) + r')(?!\w)'
regex_tokenizer = nltk.RegexpTokenizer(regex_tokenizer, gaps=True)  # .tokenize_sents

templates_df = eval_folder.read_file('templates.tsv')

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


""" WRITE TO FILES """
train_data = Dataset.from_dict(train_data)
train, test = train_test_split(train_data, test_size=400, shuffle=True, random_state=42)
train, test = Dataset.from_dict(train), Dataset.from_dict(test)

train_folder.write_file('coref_train.parquet', train)
train_folder.write_file('coref_test.parquet', test)
DATA_DIR['eval/coref'].write_file('coref_test.parquet', test)
eval_folder.write_file('wino_gender_test.parquet', Dataset.from_dict(stack_dicts(gender_data)))

for f in ['train/coref/coref_train.parquet',
          'train/coref/coref_test.parquet',
          'eval/winogender/wino_gender_test.parquet']:
    ds = DATA_DIR[f].read()
    print(f)
    print(ds)
    c = 0
    for i, x in enumerate(ds):
        if c > 5:
            break
        if i % 5 == 0:
            print(x)
            c += 1

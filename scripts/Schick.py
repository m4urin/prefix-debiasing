from transformers import AutoModelForMaskedLM, AutoTokenizer


def run_example(sentence, templates, words, skip=False):
    if skip:
        return
    for model_name in ['distilbert-base-uncased']:
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
        mask_id = tokenizer.mask_token_id
        mask_str = tokenizer.decode(mask_id)
        Z = sentence.replace("[MASK]", mask_str)
        word_ids = [tokenizer.convert_tokens_to_ids(w) for w in words]

        sentences = [t.format(Z) for t in templates]
        x = tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors='pt')

        masked_idx = (x['input_ids'] == mask_id).nonzero(as_tuple=True)
        #print(masked_idx)

        model = AutoModelForMaskedLM.from_pretrained(model_name)
        result = model(**x).logits.softmax(dim=-1)
        result = result[masked_idx][..., word_ids]
        #print(result.size(), '<<<')
        result = result[:1]

        print(model_name)
        for i, s in enumerate(sentences):
            print(f"\t{s}")
            total = 0
            for j, w in enumerate(words):
                total += result[i, j].item()
            for j, w in enumerate(words):
                print(f"\t\t{w} = {round(result[i, j].item(), 5)} ({round(100*result[i, j].item()/total, 2)}%)")


run_example(
    sentence='Two guys in the bar are having a [MASK].',
    templates = [
        "{}",
        "{} They like each other."
    ],
    words=['fight', 'conversation'],
    skip=True
)

run_example(
    sentence='He works as a [MASK].',
    templates = [
        "{}",
        "{} You don't see that very often."
    ],
    words=['lawyer', 'nurse'],
    skip=True
)

run_example(
    sentence='The [MASK] gives birth to a child.',
    templates=[
        "{}",
        "{} You don't see that very often."
    ],
    words=['man', 'woman'],
    skip=True
)

# [He/She] has a high paying job as a [nurse].
run_example(
    sentence='The [MASK] works as a nurse.',
    templates=[
        "{}"
    ],
    words=['man', 'woman']
)
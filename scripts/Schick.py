from transformers import AutoModelForMaskedLM, AutoTokenizer

for model_name in ['bert-base-uncased', 'distilbert-base-uncased']:
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
    mask_id = tokenizer.mask_token_id
    mask_str = tokenizer.decode(mask_id)
    Z = 'Two guys in the bar start a [MASK].'.replace("[MASK]", mask_str)
    words = ['fight', 'conversation']
    word_ids = [tokenizer.convert_tokens_to_ids(w) for w in words]
    templates = [
        "{}",
        'Violence: "{}"',
        'Pacifism: "{}"'
    ]
    sentences = [t.format(Z) for t in templates]
    x = tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors='pt')

    #print(x)
    masked_idx = (x['input_ids'] == mask_id).nonzero(as_tuple=True)
    #print(masked_idx)
    #print(x['input_ids'][masked_idx])

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    result = model(**x).logits.softmax(dim=-1)
    result = result[masked_idx][..., word_ids]
    #print(result.size())
    #print(result.sum(dim=(-1), keepdim=True))
    #result = result / result.sum(dim=(-1), keepdim=True)
    #print(result)

    print(model_name)
    for i, s in enumerate(sentences):
        print(f"\t{s}")
        for j, w in enumerate(words):
            print(f"\t\t{w} = {result[i, j]}")



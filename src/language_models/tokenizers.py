from transformers import AutoTokenizer

tokenizers = {}


def get_tokenizer(model_name: str):
    """ Store tokenizers, as loading a tokenizer takes a lot of time. """
    if model_name not in tokenizers:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, fast=True)
    return tokenizers[model_name]

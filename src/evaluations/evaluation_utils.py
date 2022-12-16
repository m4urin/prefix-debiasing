from typing import Union

import numpy as np
import torch
from tqdm import trange

from src.language_models.language_model import LanguageModel
from src.utils.functions import nested_loop


def get_cls(model: LanguageModel, sentences: list[str],
            batch_size: int = 64, progress_bar: Union[bool, str] = False,
            return_tensor='pt') -> Union[torch.Tensor, np.ndarray]:
    """
    Args:
        model: A LanguageModel
        sentences: A list of sentences
        batch_size: Batch size for efficiency
        progress_bar: True to show tqdm progress_bar. When progress_bar is a string,
                      this will be used as the description.
        return_tensor: 'pt' for Tensor or 'np' for numpy array
    Returns:
        An array with the sentence embeddings (CLS tokens) with the size: (n_sentences, embedding_dim)
    """
    sentence_embeddings = []
    for sent_batch in nested_loop(sentences, batch_size=batch_size, progress_bar=progress_bar):
        e = model.get_embeddings(model.tokenize(sent_batch), layers=-1, sequence=True)  # (bs, dim)
        sentence_embeddings.append(e)
    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

    if return_tensor == 'pt':
        return sentence_embeddings
    elif return_tensor == 'np':
        return sentence_embeddings.numpy()
    else:
        raise ValueError("Choose 'pt' for torch.tensor and 'np' for numpy.ndarray")


def get_cls_list(model: LanguageModel, sentences: list[list[str]],
                 batch_size: int = 64, progress_bar: Union[bool, str] = False,
                 return_tensor='pt') -> list[Union[torch.Tensor, np.ndarray]]:
    """
    Args:
        model: A LanguageModel
        sentences: A list of sentences
        batch_size: Batch size for efficiency
        progress_bar: True to show tqdm progress_bar. When progress_bar is a string,
                      this will be used as the description.
        return_tensor: 'pt' for Tensor or 'np' for numpy array
    Returns:
        List of array's with the sentence embeddings (CLS tokens) with the size: (n_sentences, embedding_dim)
    """
    all_embeddings = get_cls(model, sum(sentences, []), batch_size=batch_size,
                             progress_bar=progress_bar, return_tensor=return_tensor)
    result = []
    c = 0
    for i, sent_list in enumerate(sentences):
        result.append(all_embeddings[c:c + len(sent_list)])
        c += len(sent_list)
    return result


def get_cls_dict(model: LanguageModel, sentences,
                 batch_size: int = 64, progress_bar: Union[bool, str] = False,
                 return_tensor='pt') -> dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Args:
        model: A LanguageModel
        sentences: A list of sentences
        batch_size: Batch size for efficiency
        progress_bar: True to show tqdm progress_bar. When progress_bar is a string,
                      this will be used as the description.
        return_tensor: 'pt' for Tensor or 'np' for numpy array
    Returns:
        Dict of array's with the sentence embeddings (CLS tokens) with the size: (n_sentences, embedding_dim)
    """
    all_embeddings = get_cls_list(model, list(sentences.values()), batch_size=batch_size,
                                  progress_bar=progress_bar, return_tensor=return_tensor)
    return dict(zip(sentences.keys(), all_embeddings))


def mask_probabilities(model: LanguageModel, sentences: list[str], words: list[str],
                       mask_token: str = '[MASK]', batch_size: int = 64,
                       progress_bar: Union[bool, str] = False) -> Union[torch.Tensor, list[torch.Tensor]]:
    """
    Get the probabilities of first [MASK] token for the specific words.
    text: sentences: a list of individual sentences containing one or more mask tokens.
    returns: any ndarray/tensor of size list[(n_sentences, n_masks, n_words)] of size n_sentences
    """
    tokenizer = model.tokenizer
    mask_token_model = tokenizer.decode(tokenizer.mask_token_id)

    word_ids = tokenizer.batch_encode_plus(words, add_special_tokens=False)['input_ids']
    for w, wid in zip(words, word_ids):
        if len(wid) > 1:
            raise Exception(f"'{w}' cannot be encoded to a single token")
    word_ids = sum(word_ids, [])

    output_tensor = sentences[0].count(mask_token)
    output_tensor = all(s.count(mask_token) == output_tensor for s in sentences)

    sentences = [s.replace(mask_token, mask_token_model) for s in sentences]
    probabilities = []
    for sent_batch in nested_loop(sentences, batch_size=batch_size, progress_bar=progress_bar):
        # (bs, n_masks, n_word_ids) or bs x (n_masks, n_word_ids)
        p = model.get_mask_probabilities(model.tokenize(sent_batch), word_ids)
        probabilities.append(p)

    if output_tensor:
        # (n_sentences, n_masks, n_word_ids)
        return torch.cat(probabilities, dim=0)
    else:
        # n_sentences x (n_masks, n_word_ids)
        return sum(probabilities, [])


def get_mask_topk(model: LanguageModel, sentences: list[str], k: int = 3,
                  batch_size: int = 64, mask_token: str = '[MASK]',
                  progress_bar: Union[bool, str] = False, return_string=True
                  ) -> Union[torch.Tensor, list[torch.Tensor], list[list[str]]]:
    tokenizer = model.tokenizer
    mask_token_model = tokenizer.decode(tokenizer.mask_token_id)
    output_tensor = sentences[0].count(mask_token)
    output_tensor = all(s.count(mask_token) == output_tensor for s in sentences)

    sentences = [s.replace(mask_token, mask_token_model) for s in sentences]

    all_indices = []
    for sent_batch in nested_loop(sentences, batch_size=batch_size, progress_bar=progress_bar):
        # (bs, n_masks, n_vocabulary) or  bs x (n_masks, n_vocabulary)
        probabilities = model.get_mask_probabilities(model.tokenize(sent_batch))
        if output_tensor:
            # (bs, n_masks, k)
            topk = probabilities.topk(k, dim=-1).indices
        else:
            # bs x (n_masks, k)
            topk = [p.topk(k, dim=-1).indices for p in probabilities]
        all_indices.append(topk)

    if output_tensor:
        # (n_sent, n_masks, k)
        all_indices = torch.cat(all_indices, dim=0)
    else:
        # n_sent x (n_masks, k)
        all_indices = sum(all_indices, [])

    if return_string:
        return [[tokenizer.batch_decode(mask) for mask in sent_indices] for sent_indices in all_indices]
    else:
        return all_indices


def permutation_test(samples1: torch.Tensor, samples2: torch.Tensor, n: int = 20000):
    # samples1: (n1,) , samples2: (n2,)
    mean_diff_original = torch.abs(samples1.mean() - samples2.mean())
    result = torch.zeros(n, dtype=torch.float32, device=samples1.device)
    all_samples = torch.cat((samples1, samples2), dim=0)
    n_all, n_subset = len(all_samples), len(samples1)
    for i in trange(n, desc='Permutation test'):
        idx = torch.randperm(n_all, device=all_samples.device)
        result[i] = all_samples[idx[:n_subset]].mean() - all_samples[idx[n_subset:]].mean()
    return (torch.abs(result) >= mean_diff_original).float().mean()

from collections import OrderedDict

import torch
from torch import nn
import numpy as np
from typing import Optional, Union, Tuple, List

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import BatchEncoding, DistilBertForMaskedLM, RobertaForMaskedLM
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

from src.utils.config import ModelConfig
from src.utils.functions import get_one_of_attributes, nested_loop
from src.utils.pytorch import repeat_stacked, DEVICE, freeze, unfreeze, count_parameters, is_frozen, list_parameters


class Tokenizers:
    """ Class to store tokenizers, as loading a tokenizer takes a lot of time. """

    def __init__(self):
        self.tokenizers = {}

    def __getitem__(self, model_name: str):
        if model_name not in self.tokenizers:
            # utils.start_progress(f"Load '{model_name}' tokenizer..")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            # utils.end_progress()
        return self.tokenizers[model_name]


TOKENIZERS = Tokenizers()


class MLM(nn.Module):
    def __init__(self, config: ModelConfig, modules: nn.ModuleDict = None):
        super().__init__()
        self.config = config
        pretrained_config = AutoConfig.from_pretrained(self.config.model_name)
        self.total_layers = get_one_of_attributes(pretrained_config, ['n_layers', 'num_hidden_layers'])
        self.dim = get_one_of_attributes(pretrained_config, ['dim', 'hidden_size'])
        self.module_dict = nn.ModuleDict() if modules is None else nn.ModuleDict({k: v for k, v in modules.items()})

        if 'model' not in self.module_dict:
            self.module_dict['model'] = AutoModelForMaskedLM.from_pretrained(self.config.model_name)

        if not self.config.is_default():
            # requires a head
            head_name = f"{self.config.objective}_head"
            if head_name not in self.module_dict:
                self.module_dict[head_name] = nn.Sequential(OrderedDict({
                    'f1': nn.Linear(self.dim * self.config.head_size, 32 * self.config.head_size),
                    'swish': nn.SiLU(),
                    'f2': nn.Linear(32 * self.config.head_size, 1)
                }))

    def get_parameters_to_train(self) -> List[nn.Parameter]:
        raise NotImplementedError('Not implemented yet.')

    def get_parameters_to_save(self) -> nn.ModuleDict:
        raise NotImplementedError('Not implemented yet.')

    @property
    def n_parameters(self):
        return count_parameters(list(self.get_parameters_to_train()))

    @property
    def tokenizer(self):
        return TOKENIZERS[self.config.model_name]

    """ STATIC METHODS """
    @staticmethod
    def get_test_model(model_name='distilbert-base-uncased', model_type='base', objective='kaneko'):
        if model_type == 'base':
            config = ModelConfig(model_name, model_type, objective)
        elif model_type == 'finetune':
            config = ModelConfig(model_name, model_type, objective)
        else:
            config = ModelConfig(model_name, model_type, objective, 'linear', 'all', 8)
        return MLM.from_config(config).to(DEVICE)

    @staticmethod
    def from_config(config: ModelConfig, modules: nn.ModuleDict = None):
        if config.model_type == 'base':
            return BaseMLM(config, modules).to(DEVICE)
        elif config.model_type == 'finetune':
            return FineTuneMLM(config, modules).to(DEVICE)
        elif config.model_type == 'prefix':
            return PrefixMLM(config, modules).to(DEVICE)
        else:
            raise ValueError(f"Cannot load MLM with config file: {config}")

    """ MLM functions to use """
    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs):
        return self.module_dict['model'](input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         inputs_embeds=inputs_embeds,
                                         labels=labels,
                                         output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states,
                                         return_dict=return_dict)

    def eval_modus(self):
        return EvalMLM(self.eval())

    def tokenize(self, batch: Union[list[str], np.ndarray]) -> BatchEncoding:
        if isinstance(batch, np.ndarray):
            batch = [list(b) for b in batch]
        encoded = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', truncation=True, padding=True)
        input_ids: torch.Tensor = encoded['input_ids']
        encoded['mask_idx_sent'], encoded['mask_idx_word'] = (input_ids == self.tokenizer.mask_token_id) \
            .nonzero(as_tuple=True)
        return encoded.to(DEVICE)

    def tokenize_with_spans(self, batch: Union[list[list[str]], np.ndarray]) -> BatchEncoding:
        """
        returns encoded batch with first index of middle sentence part
        example:  x = [['This is a ', 'man', '!'], ['', 'She', 'is here.']]
        """
        if isinstance(batch, np.ndarray):
            batch = [list(b) for b in batch]

        # make sure the span cannot be combined with another token
        for s_idx in range(len(batch)):
            for i in range(2, len(batch[s_idx]), 2):
                if not batch[s_idx][i].startswith(' '):
                    batch[s_idx][i] = ' ' + batch[s_idx][i]
            for i in range(0, len(batch[s_idx]) - 1, 2):
                if not batch[s_idx][i].endswith(' '):
                    batch[s_idx][i] = batch[s_idx][i] + ' '

        encoded = self.tokenize([''.join(s) for s in batch])

        all_word_parts = []
        for s_idx in range(len(batch)):
            word_parts, n = [], 0
            span = encoded.word_to_tokens(s_idx, n)
            while span is not None:
                word_parts.append(self.tokenizer.decode(encoded['input_ids'][s_idx, span.start:span.end]).lower())
                n += 1
                span = encoded.word_to_tokens(s_idx, n)
            all_word_parts.append(word_parts)

        sentences_lower = [[s.lower().replace(' ', '') for s in sent_part] for sent_part in batch]
        all_word_parts = [[w.lower().replace(' ', '') for w in word_part] for word_part in all_word_parts]

        all_spans = [[] for _ in range(len(sentences_lower))]
        for s_idx in range(len(sentences_lower)):
            start = 0
            for i in range(0, len(batch[s_idx]) - 1, 2):
                text = ''
                while text != sentences_lower[s_idx][i]:
                    text += all_word_parts[s_idx][start]
                    start += 1
                text, end = '', start
                while text != sentences_lower[s_idx][i + 1]:
                    text += all_word_parts[s_idx][end]
                    end += 1
                all_spans[s_idx].append([start, end])
                start = end

        all_spans = [[[encoded.word_to_tokens(s_idx, start).start, encoded.word_to_tokens(s_idx, end - 1).end]
                      for start, end in span_parts] for s_idx, span_parts in enumerate(all_spans)]

        span_idx_sent, span_idx_words = [], []
        for s_idx, sent_spans in enumerate(all_spans):
            for word_spans in sent_spans:
                span_idx_sent.append(s_idx)
                span_idx_words.append(word_spans)

        encoded['span_idx_sent'] = torch.IntTensor(span_idx_sent)
        encoded['span_idx_words'] = torch.IntTensor(span_idx_words)
        return encoded.to(DEVICE)

    def get_hidden_states(self, encoded: BatchEncoding,
                          return_cls: bool = False,
                          layers: Union[int, list[int]] = None
                          ) -> torch.Tensor:
        """
        Parameters:
            encoded: A BatchEncoding from model.tokenize(..)
            layers: layer/layers to keep
            return_cls: if True, the first token of each sentence will be returned
        Returns:
            depending on 'return_cls' and 'layers' -> Tensor (bs, seq_length?, n_layers?, dim)
        """
        # embeddings: (n_layers, bs, seq_length, dim)
        embeddings = torch.stack(self(**encoded.to(DEVICE), output_hidden_states=True)['hidden_states'])
        # embeddings: (bs, seq_length, n_layers, dim)
        embeddings = embeddings.permute((1, 2, 0, 3))

        if return_cls:
            # embeddings: (bs, n_layers, dim)
            embeddings = embeddings[:, 0]

        if layers is not None:
            embeddings = embeddings[..., layers, :]

        return embeddings

    def get_span_embeddings(self, encoding_with_spans: BatchEncoding, layer: int = None, reduce='mean',
                            return_hidden_states=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(encoding_with_spans, 'span_idx_sent'):
            raise ValueError('Please use tokenize_with_spans() to add spans.')

        span_idx_sent = encoding_with_spans['span_idx_sent']
        span_idx_words = encoding_with_spans['span_idx_words']

        # hidden_states: (bs, seq_length, n_layers, dim)
        hidden_states = self.get_hidden_states(encoding_with_spans)

        # embeddings: n_spans x (**span_length, n_layers, dim), use word spans to get the mean in the sequence axis
        embeddings = [hidden_states[sent_idx, span_idx_words[i, 0]:span_idx_words[i, 1]]
                      for i, sent_idx in enumerate(span_idx_sent)]

        # embeddings: n_spans x (n_layers, dim), use word spans to get the mean in the sequence axis
        if reduce == 'mean':
            # output: (n_spans, n_layers, dim), use word spans to get the mean in the sequence axis
            embeddings = torch.stack([e.mean(dim=0) for e in embeddings])
        elif reduce == 'first':
            # output: (n_spans, n_layers, dim), use word spans to get the mean in the sequence axis
            embeddings = torch.stack([e[0] for e in embeddings])
        elif reduce == 'both':
            # output: (2, n_spans, n_layers, dim), use word spans to get the mean in the sequence axis
            embeddings_first = torch.stack([e[0] for e in embeddings])
            embeddings_mean = torch.stack([e.mean(dim=0) for e in embeddings])
            embeddings = torch.stack((embeddings_first, embeddings_mean))
        else:
            raise ValueError(f"Value '{reduce}' is not a valid setting, use one of ['mean', 'first']")

        if layer is not None:
            embeddings = embeddings[..., layer, :]
            hidden_states = hidden_states[..., layer, :]

        if return_hidden_states:
            return embeddings, hidden_states
        else:
            return embeddings

    def get_word_probabilities(self, encoding: BatchEncoding, word_ids: list[int] = None) -> torch.Tensor:
        # (bs, seq_length, vocabulary or n_word_ids)
        return self(**encoding.to(DEVICE))['logits'].softmax(dim=-1)[..., word_ids]

    def get_mask_probabilities(self, encoding: BatchEncoding, word_ids: list[int] = None, output_tensor=False) \
            -> Union[list[torch.Tensor], torch.Tensor]:
        probabilities = self.get_word_probabilities(encoding, word_ids)  # (bs, seq_length, vocabulary or n_word_ids)
        result = [[] for _ in range(probabilities.size()[0])]
        for bs_idx, seq_idx in zip(encoding['mask_idx_sent'], encoding['mask_idx_word']):
            result[bs_idx].append(probabilities[bs_idx, seq_idx])
        result = [torch.stack(p) for p in result]

        if output_tensor:
            # (bs, n_masks, vocabulary or n_word_ids)
            return torch.stack(result)
        else:
            # list[ (n_masks, vocabulary or n_word_ids) ]
            return result

    def get_coref_predictions(self,
                              encoding_with_spans: BatchEncoding,
                              subject_indices: Union[list[int], torch.Tensor]):
        if 'wsc_head' not in self.module_dict:
            raise Exception(f"Module 'wsc_head' not present in MLM with config: {self.config}")
        # (n_spans, dim)
        embeddings = self.get_span_embeddings(encoding_with_spans, layer=-1, reduce='mean')
        dims = embeddings.size()
        # (n_sentences, 2, dim)
        embeddings = embeddings.reshape(dims[0] // 2, 2, dims[1])

        if not isinstance(subject_indices, torch.Tensor):
            subject_indices = torch.tensor(subject_indices)
        subject_indices = subject_indices.long()

        # ordered for subject/pronoun: (2, n_sentences, dim)
        embeddings = torch.stack((embeddings[range(len(subject_indices)), subject_indices],
                                  embeddings[range(len(subject_indices)), 1 - subject_indices]))
        # (n_sentences, 2 x dim)
        embeddings = embeddings.permute((1, 0, 2)).flatten(start_dim=-2, end_dim=-1)

        return self.module_dict['wsc_head'](embeddings)

    def get_entailment_predictions(self, encoding1: BatchEncoding, encoding2: BatchEncoding):
        head_name = f"{self.config.objective}_head"
        if self.config.objective not in {'mrpc', 'stsb', 'rte', 'wnli'} or head_name not in self.module_dict:
            raise Exception(f"Module '{head_name}' not present in MLM with config: {self.config}")

        # (bs, 2 x dim)
        hidden = torch.cat((
            self.get_hidden_states(encoding1, return_cls=True, layers=-1),
            self.get_hidden_states(encoding2, return_cls=True, layers=-1)), dim=-1)

        return self.module_dict[head_name](hidden)

    def get_sentiment_analysis(self, encoding: BatchEncoding):
        head_name = f"{self.config.objective}_head"
        if self.config.objective != 'sst2' or head_name not in self.module_dict:
            raise Exception(f"Module '{head_name}' not present in MLM with config: {self.config}")

        return self.module_dict[head_name](self.get_hidden_states(encoding, return_cls=True, layers=-1))

    def __str__(self) -> str:
        config_info = str(self.config).replace('\n', '\n\t')
        modules_info = ', '.join([f"'{k}' ({is_frozen(v)})" for k, v in self.module_dict.items()])
        return f"MLM (\n" \
               f"\t{config_info}\n" \
               f"\tmodule_dict=[{modules_info}]\n" \
               f")"

    def __repr__(self) -> str:
        return str(self)


class BaseMLM(MLM):
    def __init__(self, config: ModelConfig, modules: nn.ModuleDict = None):
        super().__init__(config, modules)
        if self.config.is_default():
            self._parameters_to_save = nn.ModuleDict()
        else:
            self._parameters_to_train = list_parameters(self.module_dict)
            self._parameters_to_save = nn.ModuleDict({k: v for k, v in self.module_dict.items()})

    def get_parameters_to_train(self) -> List[nn.Parameter]:
        if self.config.is_default():
            raise Exception('Cannot train a base model with kaneko.')
        else:
            freeze(self)
            return unfreeze(self._parameters_to_train)

    def get_parameters_to_save(self) -> nn.ModuleDict:
        return freeze(self._parameters_to_save)


class FineTuneMLM(MLM):
    def __init__(self, config: ModelConfig, modules: nn.ModuleDict = None):
        super().__init__(config, modules)
        self._parameters_to_train = list_parameters(self.module_dict)
        self._parameters_to_save = nn.ModuleDict({k: v for k, v in self.module_dict.items()})

    def get_parameters_to_train(self) -> List[nn.Parameter]:
        freeze(self)
        return unfreeze(self._parameters_to_train)

    def get_parameters_to_save(self) -> nn.ModuleDict:
        return freeze(self._parameters_to_save)


class PrefixMLM(MLM):
    def __init__(self, config: ModelConfig, modules: nn.ModuleDict = None):
        super().__init__(config, modules)

        if 'prefix_embeddings' not in self.module_dict:
            # initialize new prefix embeddings
            self.module_dict['prefix_embeddings'] = PrefixEmbeddings(total_layers=self.total_layers, dim=self.dim,
                                                                     **self.config.to_dict())
        self.embeddings_bound = False

        if self.config.is_default() or self.config.prefix_finetune == 'prefix':
            self._parameters_to_train = list_parameters([v for k, v in self.module_dict.items()
                                                         if k != 'model'])
        elif self.config.prefix_finetune == 'model':
            self._parameters_to_train = list_parameters([v for k, v in self.module_dict.items()
                                                         if k != 'prefix_embeddings'])
        else:
            self._parameters_to_train = list_parameters(self.module_dict)

    def get_parameters_to_train(self) -> List[nn.Parameter]:
        freeze(self)
        return unfreeze(self._parameters_to_train)

    def get_parameters_to_save(self) -> nn.ModuleDict:
        if self.config.is_default() or self.config.prefix_finetune == 'prefix':
            result = nn.ModuleDict({k: v for k, v in self.module_dict.items() if k != 'model'})
        else:
            if self.embeddings_bound:
                # initialize prefix module to interact with the model
                PrefixModule.remove_from(self.config.model_name, self.module_dict['model'])
                self.embeddings_bound = False
            result = nn.ModuleDict({k: v for k, v in self.module_dict.items()})
        return freeze(result)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs):

        if not self.embeddings_bound:
            # initialize prefix module to interact with the model
            PrefixModule.add_to(self.config.model_name, self.module_dict['model'],
                                self.module_dict['prefix_embeddings'])
            self.embeddings_bound = True

        return super().forward(input_ids, attention_mask, inputs_embeds, labels, output_attentions,
                               output_hidden_states, return_dict, **kwargs)


class PrefixEmbeddings(nn.Module):
    def __init__(self, prefix_mode: str,
                 prefix_layers: Union[str, int],
                 n_prefix_tokens: int,
                 total_layers: int,
                 dim: int,
                 **kwargs):
        super().__init__()
        """
        prefix_mode: one of ['identity', 'linear', 'replace']
        prefix_layers: [n: int, 'all', 'half']
        """
        self.prefix_mode = prefix_mode
        self.n_prefix_tokens = n_prefix_tokens
        self.total_layers = total_layers
        self.dim = dim

        if prefix_layers == 'all':
            self.insert_layer = 0
        elif prefix_layers == 'half':
            self.insert_layer = total_layers // 2
        elif isinstance(prefix_layers, int):
            self.insert_layer = total_layers - prefix_layers
        else:
            raise ValueError(f"Value '{prefix_layers}' for prefix_layers is invalid.")

        if self.insert_layer < 0:
            raise ValueError(f"Value '{prefix_layers}' for prefix_layers is too large. "
                             f"It should be smaller or equal to '{total_layers}'.")
        if self.insert_layer == total_layers - 1:
            raise ValueError(f"Value '{prefix_layers}' for prefix_layers is too large. "
                             f"It should be smaller or equal to than '{total_layers - 2}'.")
        if self.insert_layer >= total_layers:
            raise ValueError(f"Value '{prefix_layers}' for prefix_layers is too small. "
                             f"It should be smaller or equal to than '{total_layers - 2}'.")

        self.prefix_embeddings_insert = torch.nn.Parameter(nn.init.normal_(torch.empty(n_prefix_tokens, dim), std=0.5))
        self.prefix_embeddings_layer = None
        if self.prefix_mode == 'replace':
            params = torch.empty(total_layers - self.insert_layer - 1, n_prefix_tokens, dim)
            self.prefix_embeddings_layer = torch.nn.Parameter(nn.init.normal_(params, std=0.5))
        elif self.prefix_mode == 'linear':
            params = torch.empty(total_layers - self.insert_layer - 1, 2, n_prefix_tokens, dim)
            self.prefix_embeddings_layer = torch.nn.Parameter(nn.init.normal_(params, std=0.5))

    def insert(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # x: (bs, ..., seq_length, dim)
        bs = x.size()[0]
        # x_prefix: (prefix_seq_length, dim)
        x_prefix = self.prefix_embeddings_insert
        # size of "..."
        d = len(x.size()) - 3
        # x_prefix: (..., prefix_seq_length, dim)
        x_prefix = x_prefix[(None,) * d]
        # x_prefix: (bs, ..., prefix_seq_length, dim)
        x_prefix = repeat_stacked(x_prefix, bs)
        # x: (bs, ..., 1, dim)<-[CLS token], (bs, ..., prefix_seq_length, dim), (bs, ..., seq_length-1, dim)
        x = torch.cat((x[..., :1, :], x_prefix, x[..., 1:, :]), -2)

        # attn_mask: (bs, ..., seq_length)
        if attn_mask is not None:
            attn_size = attn_mask.size()
            attn_prefix = torch.ones(*attn_size[:-1], self.n_prefix_tokens, device=attn_mask.device)
            attn_mask = torch.cat((attn_mask[..., :1], attn_prefix, attn_mask[..., 1:]), -1)
        return x, attn_mask

    def remove(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # x: (bs, ..., 1, dim)<-[CLS token], (bs, ..., prefix_seq_length, dim), (bs, ..., seq_length-1, dim)
        x = torch.cat((x[..., :1, :], x[..., 1 + self.n_prefix_tokens:, :]), -2)
        if attn_mask is not None:
            attn_mask = torch.cat((attn_mask[..., :1], attn_mask[..., 1 + self.n_prefix_tokens:]), -1)
        return x, attn_mask

    def prefix_function(self, layer: int, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # x: (bs, ..., seq_length + prefix_length, dim)
        if self.prefix_mode == 'replace':
            x[..., 1:1 + self.n_prefix_tokens, :] = self.prefix_embeddings_layer[layer]
        elif self.prefix_mode == 'linear':
            x[..., 1:1 + self.n_prefix_tokens, :] *= self.prefix_embeddings_layer[layer, 0]
            x[..., 1:1 + self.n_prefix_tokens, :] += self.prefix_embeddings_layer[layer, 1]
        elif self.prefix_mode == 'identity':
            pass
        else:
            raise ValueError(f"prefix_mode is {self.prefix_mode} and not one of ['identity', 'linear', 'replace']")
        return x, attn_mask

    def regularization(self) -> torch.Tensor:
        # x: (bs, ..., seq_length + prefix_length, dim)
        loss = (self.prefix_embeddings_insert.data ** 2).sum()
        if self.prefix_mode == 'replace':
            loss += (self.prefix_embeddings_layer.data ** 2).sum()
        elif self.prefix_mode == 'linear':
            loss += ((self.prefix_embeddings_layer[:, 0].data - 1) ** 2).sum()
            loss += (self.prefix_embeddings_layer[:, 1].data ** 2).sum()
        return loss

    def clean(self, layer: int, x: torch.Tensor):
        layer_offset = layer - self.insert_layer
        if layer_offset < 0:
            return x
        x, _ = self.remove(x)
        return x

    def forward(self, layer: int, x: torch.Tensor, attn_mask: torch.Tensor = None):
        layer_offset = layer - self.insert_layer

        # do nothing
        if layer_offset < 0:
            return x, attn_mask

        # insert tokens
        if layer_offset == 0:
            return self.insert(x, attn_mask)

        # remove tokens
        if layer == self.total_layers:
            return self.remove(x, attn_mask)

        return self.prefix_function(layer_offset - 1, x, attn_mask)


class PrefixModule(nn.Module):
    def __init__(self, prefix_embeddings: PrefixEmbeddings):
        super().__init__()
        self.prefix_embeddings = prefix_embeddings
        self.old_module = None

    @staticmethod
    def remove_from_model(model):
        assert False, 'Not implemented yet'

    def forward(self, **kwargs):
        assert False, 'Not implemented yet'

    @staticmethod
    def add_to(model_name, model, embeddings):
        if model_name == 'distilbert-base-uncased':
            PrefixDistilbert(model, embeddings)
        elif model_name == 'roberta-base':
            PrefixRoberta(model, embeddings)
        else:
            assert False, f'Model "{model_name}" is not supported!'

    @staticmethod
    def remove_from(model_name, model):
        if model_name == 'distilbert-base-uncased':
            PrefixDistilbert.remove_from_model(model)
        elif model_name == 'roberta-base':
            PrefixRoberta.remove_from_model(model)
        else:
            assert False, f'Model "{model_name}" is not supported!'


class PrefixDistilbert(PrefixModule):
    def __init__(self, model: DistilBertForMaskedLM, prefix_embeddings: PrefixEmbeddings):
        super().__init__(prefix_embeddings)
        self.layer = model.distilbert.transformer.layer
        self.old_module = (model.distilbert.transformer,)  # in parenthesis so that Module.parameters() won't pick it up
        model.distilbert.transformer = self

    @staticmethod
    def remove_from_model(model):
        model.distilbert.transformer = model.distilbert.transformer.old_module[0]

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.
        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        ### layer 0 ###
        x, attn_mask = self.prefix_embeddings(0, x, attn_mask)
        ###############

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (self.prefix_embeddings.clean(i, hidden_state),)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
            )

            # layer 1 to n #
            h, attn_mask = self.prefix_embeddings(i + 1, layer_outputs[-1], attn_mask)
            layer_outputs = (*layer_outputs[:-1], h)
            ################

            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


class PrefixRoberta(PrefixModule):
    def __init__(self, model: RobertaForMaskedLM, prefix_embeddings: PrefixEmbeddings):
        super().__init__(prefix_embeddings)
        self.config = model.roberta.encoder.config
        self.layer = model.roberta.encoder.layer
        self.gradient_checkpointing = model.roberta.encoder.gradient_checkpointing

        self.old_module = (model.roberta.encoder,)  # in parenthesis so that Module.parameters() won't pick it up
        model.roberta.encoder = self

    @staticmethod
    def remove_from_model(model):
        model.roberta.encoder = model.roberta.encoder.old_module[0]

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # layer 0 #
        hidden_states, attention_mask = self.prefix_embeddings(0, hidden_states, attention_mask)
        ###########

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (self.prefix_embeddings.clean(i, hidden_states),)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    # logger.warning(
                    #    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # layer 1 to n #
            h, attention_mask = self.prefix_embeddings(i + 1, layer_outputs[0], attention_mask)
            layer_outputs = (h, *layer_outputs[1:])
            ################

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class EvalMLM(nn.Module):
    def __init__(self, model: MLM):
        super().__init__()
        self.model = model

    def embed_sentences(self, sentences: list[str], batch_size=32) -> torch.Tensor:
        """
        Embed a list of sentences to be evaluated
        sentences: a list of individual sentences
        returns: tensor of size (n_sentences, embedding_dim)
        """
        with torch.no_grad():
            sentence_embeddings = []
            info = f'Embed sentences (bs={batch_size}, model={self.model.config.model_name})'
            for sent_batch in nested_loop(sentences, batch_size=batch_size, progress_bar=info):
                encoded = self.model.tokenize(sent_batch)
                # (bs, dim)
                embeddings = self.model.get_hidden_states(encoded, layers=-1, return_cls=True)  # (bs, dim)
                sentence_embeddings.append(embeddings)
            return torch.cat(sentence_embeddings, dim=0)

    def embed_sentences_dict(self, sentences: dict[str, list[str]], batch_size=32) -> dict[torch.Tensor]:
        """
        Embed a list of sentences to be evaluated
        sentences: a list of individual sentences
        returns: tensor of size (n_sentences, embedding_dim)
        """
        with torch.no_grad():
            sentences_flattened = sum([sent_list for _, sent_list in sentences.items()], [])
            embeddings_flattened = self.embed_sentences(sentences_flattened, batch_size=batch_size)
            result = {}
            c = 0
            for i, (k, sent_list) in enumerate(sentences.items()):
                result[k] = embeddings_flattened[c:c + len(sent_list)]
                c += len(sent_list)
            return result

    def mask_probabilities(self, sentences: list[str], words: list[str], batch_size=32) -> torch.Tensor:
        """
        Get the probabilities of first [MASK] token for the specific words.
        text: sentences: a list of individual sentences containing one or more mask tokens.
        returns: any ndarray/tensor of size list[(n_sentences, n_masks, n_words)] of size n_sentences
        """
        with torch.no_grad():
            word_ids = self.model.tokenizer.convert_tokens_to_ids(words)
            for w, wid in zip(words, word_ids):
                if wid == self.model.tokenizer.unk_token_id:
                    raise Exception(f"'{w}' cannot be encoded to a single token")
            sentences = [s.replace('[MASK]', self.model.tokenizer.mask_token_id) for s in sentences]
            probabilities = []
            info = f'Mask probabilities (bs={batch_size}, model={self.model.config.model_name})'
            for sent_batch in nested_loop(sentences, batch_size=batch_size, progress_bar=info):
                encoded = self.model.tokenize(sent_batch)
                # (bs, n_masks, n_word_ids)
                p = self.model.get_mask_probabilities(encoded, word_ids, output_tensor=True)
                probabilities.append(p)
            return torch.cat(probabilities, dim=0)  # (n_sentences, n_masks, n_word_ids)

    def entailment(self, sentence1: list[str], sentence2: list[str], batch_size=16) -> torch.Tensor:
        with torch.no_grad():
            info = f'Entailments (bs={batch_size}, model={self.model.config.model_name})'
            logits = []
            for sent_batch in nested_loop(zip(sentence1, sentence2), batch_size=batch_size, progress_bar=info):
                encoded1 = self.model.tokenize([s1 for s1, _ in sent_batch])
                encoded2 = self.model.tokenize([s2 for _, s2 in sent_batch])
                # (bs, 1)
                p = self.model.get_entailment_predictions(encoded1, encoded2)
                logits.append(p)
            return torch.cat(logits, dim=0).flatten()  # (n_sentences)

    def sentiment_analysis(self, sentences: list[str], batch_size=32):
        with torch.no_grad():
            info = f'Sentiment Analysis (bs={batch_size}, model={self.model.config.model_name})'
            logits = []
            for sent_batch in nested_loop(sentences, batch_size=batch_size, progress_bar=info):
                # (bs, 1)
                p = self.model.get_sentiment_analysis(self.model.tokenize(sent_batch))
                logits.append(p)
            return torch.cat(logits, dim=0).flatten()  # (n_sentences)

    def coref(self, sentences: list[list[str]], subject_idx: list[int], batch_size=32):
        with torch.no_grad():
            info = f'Coreference resolution'
            logits = []
            for batch in nested_loop(zip(sentences, subject_idx), batch_size=batch_size, progress_bar=info):
                # (bs, 1)
                encoded = self.model.tokenize_with_spans([s for s, _ in batch])
                p = self.model.get_coref_predictions(encoded, [i for _, i in batch])
                logits.append(p)
            return torch.cat(logits, dim=0).flatten()  # (n_sentences)

    def train_modus(self):
        return self.model.train()

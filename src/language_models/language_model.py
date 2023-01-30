import torch
from torch import nn
import numpy as np
from typing import Optional, Union, Tuple, List
from collections import OrderedDict
from transformers import AutoModelForMaskedLM, AutoConfig, BatchEncoding

from src.data.structs.model_config import ModelConfig
from src.data.structs.model_output import ModelOutput
from src.language_models.prefix_embeddings import PrefixEmbeddings
from src.language_models.prefix_models import PrefixModule
from src.language_models.tokenizers import get_tokenizer
from src.utils.functions import get_one_of_attributes
from src.utils.pytorch import DEVICE, count_parameters, is_frozen, nested_stack


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        pretrained_config = AutoConfig.from_pretrained(config.model_name)

        self.config = config
        self.total_layers = get_one_of_attributes(pretrained_config, ['n_layers', 'num_hidden_layers'])
        self.dim = get_one_of_attributes(pretrained_config, ['dim', 'hidden_size'])

        original_model = AutoModelForMaskedLM.from_pretrained(config.model_name)
        if config.model_name == 'distilbert-base-uncased':
            self.model = original_model.distilbert
            self.lm_head = nn.Sequential(original_model.vocab_transform, original_model.activation,
                                          original_model.vocab_layer_norm, original_model.vocab_projector)
        elif config.model_name == 'roberta-base':
            self.model = original_model.roberta
            self.lm_head = original_model.lm_head
        elif config.model_name == 'bert-base-uncased':
            self.model = original_model.bert
            self.lm_head = original_model.cls
        else:
            raise ValueError(f"Model '{config.model_name}' is not supported!")

        self.cls_head = None

    def add_cls_head(self, config: ModelConfig):
        self.config = config
        self.cls_head = nn.Sequential(OrderedDict({
            'dense': nn.Linear(self.dim * self.config.head_size, self.dim // 2),
            'activation': nn.ReLU(),
            'decoder': nn.Linear(self.dim // 2, 2)
        }))
        #self.cls_head = nn.Linear(self.dim * self.config.head_size, 2)
        print('added new!')

    def get_parameters_to_train(self) -> List[nn.Parameter]:
        raise NotImplementedError('Not implemented yet.')

    @property
    def n_parameters(self):
        return count_parameters(list(self.get_parameters_to_train()))

    @property
    def tokenizer(self):
        return get_tokenizer(self.config.model_name)

    @staticmethod
    def get_test_model(model_name='distilbert-base-uncased', model_type='base', debias_method=None):
        if model_type == 'prefix':
            config = ModelConfig(model_name, model_type, debias_method, prefix_mode='linear',
                                 prefix_layers='all', n_prefix_tokens=8)
        else:
            config = ModelConfig(model_name, model_type, debias_method)
        return LanguageModel.from_config(config).to(DEVICE)

    @staticmethod
    def from_config(config: ModelConfig):
        if config.model_type == 'base':
            return BaseLanguageModel(config).to(DEVICE)
        elif config.model_type == 'finetune':
            return FineTuneLanguageModel(config).to(DEVICE)
        elif config.model_type == 'prefix':
            return PrefixLanguageModel(config).to(DEVICE)
        else:
            raise ValueError(f"Cannot load MLM with config file: {config}")

    def tokenize(self, batch: Union[list[str], list[Tuple[str, str]], np.ndarray]) -> BatchEncoding:
        if isinstance(batch, np.ndarray):
            batch = [list(b) for b in batch]
        encoded = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', truncation=True, padding=True)
        input_ids: torch.Tensor = encoded['input_ids']
        encoded['mask_idx_sent'], encoded['mask_idx_word'] = (input_ids == self.tokenizer.mask_token_id)\
            .nonzero(as_tuple=True)
        return encoded.to(DEVICE)

    def tokenize_with_spans(self, batch: Union[list[list[str]], np.ndarray]) -> BatchEncoding:
        """
        returns encoded batch with first index of middle sentence part
        example:  x = [['This is a ', 'man', '!'], ['', 'She', 'is here.']]
        """
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

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, output_lm: Optional[bool] = None,
                output_cls: Optional[bool] = None, **kwargs):
        model_output = self.model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=inputs_embeds,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict)
        last_hidden_state = model_output.last_hidden_state
        attentions = model_output.attentions
        hidden_states = model_output.hidden_states
        lm_logits = None
        cls_logits = None

        if output_hidden_states is not None:
            # (bs, seq_length, n_layers, dim)
            hidden_states = torch.stack(hidden_states).permute((1, 2, 0, 3))
        if output_lm is not None:
            lm_logits = self.lm_head(last_hidden_state)
        if output_cls is not None:
            cls_logits = self.cls_head(last_hidden_state)

        return ModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
            lm_logits=lm_logits,
            cls_logits=cls_logits
        )

    def get_embeddings(self,
                       encoded: BatchEncoding,
                       output_cls_token: Optional[bool] = None,
                       output_hidden_states: Optional[bool] = None
                       ) -> torch.Tensor:
        """
        Parameters:
            encoded: A BatchEncoding from model.tokenize(..)
            output_hidden_states:
            output_cls_token: if True, the first token of each sentence will be returned
        Returns:
            depending on 'output_cls_token' and 'output_hidden_states' -> Tensor (bs, seq_length?, n_layers?, dim)
        """
        embeddings = self(**encoded.to(DEVICE), output_hidden_states=output_hidden_states)
        if output_hidden_states is None:
            # (bs, seq, dim)
            embeddings = embeddings.last_hidden_state
        else:
            # (bs, seq_length, n_layers, dim)
            embeddings = embeddings.hidden_states

        if output_cls_token is not None:
            # (bs, n_layers?, dim)
            embeddings = embeddings[:, 0]

        # (bs, seq_length?, layers?, dim)
        return embeddings

    def get_span_embeddings(self,
                            encoding_with_spans: BatchEncoding,
                            reduce='mean',
                            output_hidden_states: Optional[bool] = None,
                            output_cls_token: Optional[bool] = None
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :return: Tensors (n_spans, n_layers?, dim), (bs, n_layers?, dim)?
        """
        if not hasattr(encoding_with_spans, 'span_idx_sent'):
            raise ValueError('Please use tokenize_with_spans() to add spans.')

        span_idx_sent = encoding_with_spans['span_idx_sent']
        span_idx_words = encoding_with_spans['span_idx_words']

        # hidden_states: (bs, seq_length, n_layers?, dim)
        hidden_states = self.get_embeddings(encoding_with_spans, output_hidden_states=output_hidden_states)

        # embeddings: n_spans x (**span_length, n_layers, dim), use word spans to get the mean in the sequence axis
        embeddings = [hidden_states[sent_idx, span_idx_words[i, 0]:span_idx_words[i, 1]]
                      for i, sent_idx in enumerate(span_idx_sent)]

        # embeddings: n_spans x (n_layers?, dim), use word spans to get the mean in the sequence axis
        if reduce == 'mean':
            # output: (n_spans, n_layers?, dim), use word spans to get the mean in the sequence axis
            embeddings = torch.stack([e.mean(dim=0) for e in embeddings])
        elif reduce == 'first':
            # output: (n_spans, n_layers?, dim), use word spans to get the mean in the sequence axis
            embeddings = torch.stack([e[0] for e in embeddings])
        else:
            raise ValueError(f"Value '{reduce}' is not a valid setting, use one of ['mean', 'first']")

        if output_cls_token is None:
            # (n_spans, n_layers?, dim)
            return embeddings
        else:
            # (n_spans, n_layers?, dim), (bs, n_layers?, dim)
            return embeddings, hidden_states[:, 0]

    def get_ml_predictions(self,
                           encoding: BatchEncoding,
                           word_ids: list[int] = None,
                           return_probabilities=False
                           ) -> torch.Tensor:
        """
        Masked Language
        :param encoding: A BatchEncoding
        :param return_probabilities: Apply softmax to get probabilities
        :param word_ids: Return only for a a specific set of words
        :return: logits Tensor (bs, seq, n_vocabulary or word_ids)
        """
        # (bs, seq, dim)
        embeddings = self.get_embeddings(encoding)
        # (bs, seq, n_vocabulary)
        predictions = self.lm_head(embeddings)
        if return_probabilities:
            predictions = predictions.softmax(dim=-1)
        if word_ids is not None:
            predictions = predictions[..., word_ids]
        return predictions

    def get_mask_probabilities(self,
                               encoding: BatchEncoding,
                               word_ids: list[int] = None
                               ) -> Union[list[torch.Tensor], torch.Tensor]:
        """

        :param encoding:
        :param word_ids:
        :return: Tensor (bs, n_masks, vocabulary or n_word_ids)
        """
        # (bs, seq_length, vocabulary or n_word_ids)
        prob = self.get_ml_predictions(encoding, word_ids, return_probabilities=True)
        bs = prob.size()[0]

        result = [[] for _ in range(bs)]
        for bs_idx, seq_idx in zip(encoding['mask_idx_sent'], encoding['mask_idx_word']):
            result[bs_idx].append(prob[bs_idx, seq_idx])

        # (bs, n_masks, vocabulary or n_word_ids)
        return nested_stack(result)

    def get_cls_predictions(self,
                            encoding_with_or_without_spans: BatchEncoding,
                            subject_indices: Union[list[int], torch.Tensor] = None):
        if subject_indices is None:
            # (bs, 1)
            return self.cls_head(self.get_embeddings(encoding_with_or_without_spans, output_cls_token=True))

        # (n_spans, dim), (bs, dim)
        embeddings, cls_tokens = self.get_span_embeddings(encoding_with_or_without_spans, reduce='mean',
                                                          output_cls_token=True)
        n_spans, dim = embeddings.size()
        # (bs, 2, dim)
        embeddings = embeddings.reshape((n_spans // 2, 2, dim))

        if not isinstance(subject_indices, torch.Tensor):
            subject_indices = torch.tensor(subject_indices)
        subject_indices = subject_indices.long()

        # ordered for subject/pronoun: (3, bs, dim)
        embeddings = torch.stack((cls_tokens,
                                  embeddings[range(len(subject_indices)), subject_indices],
                                  embeddings[range(len(subject_indices)), 1 - subject_indices]))
        # (bs, 3, dim)
        embeddings = embeddings.transpose(0, 1)
        # (bs, 3 x dim)
        embeddings = embeddings.flatten(start_dim=-2, end_dim=-1)

        # (bs, 1)
        return self.cls_head(embeddings)

    def __str__(self) -> str:
        modules_info = ', '.join([f"'{k}' ({is_frozen(v)})"
                                  for k, v in self.__dict__.items() if isinstance(v, nn.Module)])
        return f"MLM (\n" \
               f"\t{self.config},\n" \
               f"\tparameters=[{modules_info}]\n" \
               f")"

    def __repr__(self) -> str:
        return str(self).replace('\n', '').replace('\t', '')


class BaseLanguageModel(LanguageModel):
    def get_parameters_to_train(self) -> List[nn.Parameter]:
        return list(self.parameters()) if self.config.is_downstream() else []


class FineTuneLanguageModel(LanguageModel):
    def get_parameters_to_train(self) -> List[nn.Parameter]:
        return list(self.parameters())


class PrefixLanguageModel(LanguageModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.prefix_embeddings = PrefixEmbeddings(**self.config.to_dict(),
                                                  total_layers=self.total_layers,
                                                  dim=self.dim)
        PrefixModule.add_to(self.config.model_name, self.model, self.prefix_embeddings)

    def get_parameters_to_train(self) -> List[nn.Parameter]:
        if self.config.is_downstream():
            return list(set(self.parameters()) - set(self.prefix_embeddings.parameters()))
        else:
            return list(self.prefix_embeddings.parameters())

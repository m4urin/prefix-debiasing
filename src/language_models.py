import torch
from torch import nn
import numpy as np
from typing import Optional, Union, Tuple
import hashlib

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import BatchEncoding, DistilBertForMaskedLM, RobertaForMaskedLM
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

from src.utils import try_int, sig_notation, get_one_of_attributes, tbatched, repeat_stacked, DEVICE

MODEL_NAMES = ['distilbert-base-uncased', 'roberta-base']
MODEL_TYPES = ['base', 'finetune', 'prefix']
PREFIX_MODES = ['identity', 'linear', 'replace']
PREFIX_LAYERS = ['all', 'half']
LOSS_FUNCTIONS = ['kaneko']


class Config:
    def __init__(self,
                 model_name: str,
                 model_type: str,
                 prefix_mode: str = None,
                 prefix_layers: Union[str, int] = None,
                 n_prefix_tokens: int = None,
                 loss_function: str = None,
                 epochs: int = None,
                 batch_size: int = None,
                 lr: float = None,
                 num_warmup_steps: int = None,
                 seed: int = None):
        super().__init__()
        errors = []

        self.model_name = model_name
        self.model_type = model_type
        errors.extend(self.verify_attribute('model_name', text=MODEL_NAMES))
        errors.extend(self.verify_attribute('model_type', text=MODEL_TYPES))

        if model_type == 'prefix':
            self.prefix_mode = prefix_mode
            self.prefix_layers = try_int(prefix_layers)
            self.n_prefix_tokens = n_prefix_tokens
            errors.extend(self.verify_attribute('prefix_mode', text=PREFIX_MODES))
            errors.extend(self.verify_attribute('prefix_layers', numeric=(int, 1, 24), text=PREFIX_LAYERS))
            errors.extend(self.verify_attribute('n_prefix_tokens', numeric=(int, 1, 65)))

        if model_type != 'base':
            self.loss_function = loss_function
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.num_warmup_steps = num_warmup_steps
            self.seed = seed
            errors.extend(self.verify_attribute('loss_function', text=LOSS_FUNCTIONS))
            errors.extend(self.verify_attribute('epochs', numeric=(int, 1, 10)))
            errors.extend(self.verify_attribute('batch_size', numeric=(int, 1, 65)))
            errors.extend(self.verify_attribute('lr', numeric=(float, 1e-7, 0.1)))
            errors.extend(self.verify_attribute('num_warmup_steps', numeric=(int, 10, 10000)))
            errors.extend(self.verify_attribute('seed', numeric=(int, 0, 1000000000)))

        if len(errors) > 0:
            raise ValueError(str(self) + '\n- '.join([''] + errors))

    def verify_attribute(self, attr_name: str, numeric=None, text: list[str] = None) -> list[str]:
        errors = []
        if not hasattr(self, attr_name):
            errors.append(f"Value for attribute '{attr_name}' is missing.")
            return errors
        attr_value = getattr(self, attr_name)
        if text is not None and attr_value in text:
            return errors
        if text is not None and attr_value not in text:
            if isinstance(attr_value, str) or numeric is None:
                txt = f"{type(attr_value).__name__} value '{attr_value}' for attribute '{attr_name}' " \
                      f"is not supported, please choose from: {text}"
                if numeric is not None:
                    numeric_type, lower_bound, upper_bound = numeric
                    txt += f", or provide a {numeric_type.__name__} in the range of [{lower_bound}, {upper_bound}]"
                errors.append(txt + '.')
                return errors
        if numeric is not None:
            numeric_type, lower_bound, upper_bound = numeric
            # is numeric
            if type(attr_value) != numeric_type or attr_value < lower_bound or attr_value >= upper_bound:
                txt = f"{type(attr_value).__name__} value '{attr_value}' for attribute '{attr_name}' " \
                      f"should be a {numeric_type.__name__} in the range of [{lower_bound}, {upper_bound}]"
                if text is not None:
                    txt += f', or one of the following: {text}'
                errors.append(txt + '.')
        return errors

    def is_prefix(self) -> bool:
        return self.model_type == 'prefix'

    def is_trainable(self) -> bool:
        return self.model_type != 'base'

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        txt = f"Config({self.model_name}"
        if self.is_prefix():
            txt += f",{self.prefix_mode},layers={self.prefix_layers},tokens={self.n_prefix_tokens}"
        if self.is_trainable():
            txt += f",loss={self.loss_function}" \
                   f",epochs={self.epochs}" \
                   f",bs={self.batch_size}" \
                   f",lr={sig_notation(self.lr, 3)}" \
                   f",warmup={self.num_warmup_steps}"
        return txt + ')'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self.to_dict()))

    def get_hash(self):
        return hashlib.sha1(str(self.to_dict()).encode()).hexdigest()

    def __eq__(self, other):
        if isinstance(other, Config):
            other_dict = other.to_dict()
            self_dict = self.to_dict()
            for key in set(list(other_dict.keys()) + list(self_dict.keys())):
                if key not in self_dict or key not in other_dict or self_dict[key] != other_dict[key]:
                    return False
            return True
        return False


class TokenizerRegister:
    """ Class to store tokenizers, as loading a tokenizer takes a lot of time. """

    def __init__(self):
        self.tokenizers = {}

    def __getitem__(self, model_name: str):
        if model_name not in self.tokenizers:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizers[model_name]


TOKENIZERS = TokenizerRegister()


class MaskedLanguageModel(nn.Module):
    def __init__(self, config: Union[dict, Config], parameters=None, **kwargs):
        super().__init__()
        self.config = config if isinstance(config, Config) else Config(**config)
        pretrained_config = AutoConfig.from_pretrained(self.config.model_name)
        self.total_layers = get_one_of_attributes(pretrained_config, ['n_layers', 'num_hidden_layers'])
        self.dim = get_one_of_attributes(pretrained_config, ['dim', 'hidden_size'])
        self.model = self.load_model(parameters)

        self.n_parameters = 0
        for param in self.get_parameters().parameters():
            self.n_parameters += param.nelement()

    """ PROPERTIES """
    @property
    def tokenizer(self):
        return TOKENIZERS[self.config.model_name]

    """ METHODS TO OVERRIDE """
    def load_model(self, parameters: nn.Module) -> nn.Module:
        raise NotImplementedError('Not implemented.')

    def get_parameters(self) -> nn.Module:
        raise NotImplementedError('Not implemented.')

    @staticmethod
    def from_config(config: Config, parameters: nn.Module = None, **kwargs):
        if config.model_type == 'base':
            return BaseMLM(config)
        if config.model_type == 'finetune':
            return FineTuneMLM(config, parameters, **kwargs)
        if config.model_type == 'prefix':
            return PrefixMLM(config, parameters, **kwargs)
        raise ValueError(f"Cannot load MLM with config file: {config}")

    """ MLM functions to use """
    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs):
        assert self.model is not None, 'Please call load() once to load the model'
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict)

    def eval_modus(self):
        return EvalMLM(self)

    def tokenize(self, batch: Union[list[str], np.ndarray]) -> BatchEncoding:
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
        if isinstance(batch, np.ndarray):
            batch = [list(b) for b in batch]

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

        all_spans = []
        for s_idx in range(len(sentences_lower)):
            text, start = '', 0
            while text != sentences_lower[s_idx][0]:
                text += all_word_parts[s_idx][start].strip()
                start += 1
            text, end = '', start
            while text != sentences_lower[s_idx][1]:
                text += all_word_parts[s_idx][end]
                end += 1

            all_spans.append([start, end])

        all_spans = [[encoded.word_to_tokens(s_idx, start).start, encoded.word_to_tokens(s_idx, end - 1).end]
                     for s_idx, (start, end) in enumerate(all_spans)]

        encoded['word_spans'] = torch.IntTensor(all_spans)
        return encoded.to(DEVICE)

    def get_hidden_states(self, encoded: BatchEncoding) -> torch.Tensor:
        """
        Embed a list of sentences to be evaluated
        sentences: a list of individual sentences
        returns: tensor of size (bs, embedding_dim)
        """
        # embeddings: (n_layers, bs, seq_length, dim)
        embeddings = torch.stack(self(**encoded.to(DEVICE), output_hidden_states=True)['hidden_states'])
        # embeddings: (bs, seq_length, n_layers, dim)
        embeddings = embeddings.permute((1, 2, 0, 3))

        return embeddings

    def get_sentence_embeddings(self, encoded: BatchEncoding, layer: int = None) -> torch.Tensor:
        """
        Embed a list of sentences to be evaluated
        sentences: a list of individual sentences
        layer: all layers if layer is None
        returns: tensor of size (bs, embedding_dim)
        """
        # embeddings: (bs, seq_length, n_layers, dim)
        embeddings = self.get_hidden_states(encoded)
        # embeddings: (bs, n_layers, dim)
        embeddings = embeddings[:, 0]

        # embeddings: (bs, layer??, dim)
        if layer is not None:
            embeddings = embeddings[:, layer]

        return embeddings

    def get_span_embeddings(self, encoding_with_spans: BatchEncoding, layer: int = None) -> torch.Tensor:
        """
        desc
        """
        if not hasattr(encoding_with_spans, 'word_spans'):
            raise ValueError('Please use tokenize_with_spans() to add spans.')

        span = encoding_with_spans['word_spans']

        # embeddings: (bs, seq_length, n_layers, dim)
        embeddings = self.get_hidden_states(encoding_with_spans)
        bs = embeddings.size()[0]

        # output: (bs, n_layers, dim), use word spans to get the mean in the sequence axis
        embeddings = torch.stack([embeddings[i, span[i, 0]:span[i, 1]].mean(dim=0) for i in range(bs)])
        # get layer is required
        if layer is not None:
            embeddings = embeddings[:, layer]

        return embeddings

    def get_word_probabilities(self, encoding: BatchEncoding, word_ids: list[int] = None) -> torch.Tensor:
        # (bs, seq_length, vocabulary or n_word_ids)
        return self(**encoding.to(DEVICE))['logits'].softmax(dim=-1)[..., word_ids]

    def get_mask_probabilities(self, encoding: BatchEncoding, word_ids: list[int] = None,
                               output_tensor=False) -> Union[list[torch.Tensor], torch.Tensor]:
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

    """ PRINTING """
    def __str__(self) -> str:
        return f"Model({self.config})"

    def __repr__(self) -> str:
        return str(self)


class EvalMLM(nn.Module):
    def __init__(self, model: MaskedLanguageModel):
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
            for sent_batch in tbatched(sentences, batch_size=batch_size, desc=f'Embed sentences (bs={batch_size}, '
                                                                              f'model={self.model.config.model_name})'):
                encoded = self.model.tokenize(sent_batch)
                embeddings = self.model.get_sentence_embeddings(encoded, layer=-1)  # (bs, dim)
                sentence_embeddings.append(embeddings.cpu())
            return torch.cat(sentence_embeddings, dim=0)

    def embed_sentences_dict(self, sentences: dict[list[str]], batch_size=32) -> dict[torch.Tensor]:
        """
        Embed a list of sentences to be evaluated
        sentences: a list of individual sentences
        returns: tensor of size (n_sentences, embedding_dim)
        """
        sentences_flattened = sum([sent_list for k, sent_list in sentences.items()], [])
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
            for sent_batch in tbatched(sentences, batch_size=batch_size, desc=f'Mask probabilities (bs={batch_size}, '
                                                                              f'model={self.model.config.model_name})'):
                encoded = self.model.tokenize(sent_batch)
                # (bs, n_masks, n_word_ids)
                p = self.model.get_mask_probabilities(encoded, word_ids, output_tensor=True)
                probabilities.append(p.cpu())
            return torch.cat(probabilities, dim=0)  # (n_sentences, n_masks, n_word_ids)

    def train_modus(self):
        return self.model


class BaseMLM(MaskedLanguageModel):
    def load_model(self, parameters: dict) -> nn.Module:
        if parameters is None or len(parameters) == 0:
            model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        else:
            raise ValueError(f"Loading custom parameters is not allowed for model_type='base'")
        for param in model.parameters():
            param.requires_grad = False
        return model

    def get_parameters(self) -> nn.Module:
        return nn.Module()


class FineTuneMLM(MaskedLanguageModel):
    def load_model(self, parameters: dict) -> nn.Module:
        if parameters is None:
            model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        else:
            model = parameters

        for param in model.parameters():
            param.requires_grad = True

        return model

    def get_parameters(self) -> nn.Module:
        return self.model


class PrefixMLM(MaskedLanguageModel):
    def __init__(self, config: Union[dict, Config], parameters=None, **kwargs):
        self.prefix_embeddings = None
        super().__init__(config, parameters, **kwargs)

    def load_model(self, parameters: dict) -> nn.Module:
        model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)

        if parameters is None:
            self.prefix_embeddings = PrefixEmbeddings(total_layers=self.total_layers, dim=self.dim,
                                                      **self.config.to_dict())
        else:
            self.prefix_embeddings = parameters

        # initialize prefix module to interact with the model
        PrefixModule.add_to(self.config.model_name, model, self.prefix_embeddings)

        # freeze model
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze prefix
        for param in self.prefix_embeddings.parameters():
            param.requires_grad = True

        return model

    def get_parameters(self) -> nn.Module:
        return self.prefix_embeddings


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
                             f"It should be smaller or equal to than '{total_layers-2}'.")
        if self.insert_layer >= total_layers:
            raise ValueError(f"Value '{prefix_layers}' for prefix_layers is too small. "
                             f"It should be smaller or equal to than '{total_layers-2}'.")

        self.prefix_embeddings_insert = torch.nn.Parameter(nn.init.normal_(torch.empty(n_prefix_tokens, dim), std=0.5))
        self.prefix_embeddings_layer = None
        if self.prefix_mode == 'replace':
            params = torch.empty(total_layers - self.insert_layer - 1, n_prefix_tokens, dim)
            self.prefix_embeddings_layer = torch.nn.Parameter(nn.init.normal_(params, std=0.5))
        elif self.prefix_mode == 'linear':
            params = torch.empty(total_layers - self.insert_layer - 1, 2, n_prefix_tokens, dim)
            self.prefix_embeddings_layer = torch.nn.Parameter(nn.init.normal_(params, std=0.5))

    def get_parameters(self) -> dict:
        params = {'embeddings_insert': self.prefix_embeddings_insert.data}
        if self.prefix_embeddings_layer is not None:
            params['embeddings_layer'] = self.prefix_embeddings_layer.data
        return params

    def set_parameters(self, embeddings_insert, embeddings_layer=None, **kwargs):
        self.prefix_embeddings_insert.data = embeddings_insert
        if embeddings_layer is not None:
            self.prefix_embeddings_layer.data = embeddings_layer

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
        loss = (self.prefix_embeddings_insert.data ** 2).mean()
        layers = self.total_layers - self.insert_layer - 1
        if self.prefix_mode == 'replace':
            loss += (self.prefix_embeddings_layer.data ** 2).mean() * layers
        elif self.prefix_mode == 'linear':
            loss += ((self.prefix_embeddings_layer[:, 0].data - 1) ** 2).mean() * layers
            loss += (self.prefix_embeddings_layer[:, 1].data ** 2).mean() * layers
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
        self.prefix_embeddings: PrefixEmbeddings = prefix_embeddings

    def add_to_model(self, model):
        assert False, 'Not implemented yet'

    def forward(self, **kwargs):
        assert False, 'Not implemented yet'

    @staticmethod
    def from_name(model_name, model, embeddings):
        if model_name == 'distilbert-base-uncased':
            return PrefixDistilbert(model, embeddings)
        if model_name == 'roberta-base':
            return PrefixRoberta(model, embeddings)
        assert False, f'Model "{model_name}" is not supported!'

    @staticmethod
    def add_to(model_name, model, embeddings):
        PrefixModule.from_name(model_name, model, embeddings).add_to_model(model)


class PrefixDistilbert(PrefixModule):
    def __init__(self, model: DistilBertForMaskedLM, prefix_embeddings: PrefixEmbeddings):
        super().__init__(prefix_embeddings)
        self.n_layers = model.distilbert.transformer.total_layers
        self.layer = model.distilbert.transformer.layer

    def add_to_model(self, model):
        model.distilbert.transformer = self

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

    def add_to_model(self, model):
        model.roberta.encoder = self

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

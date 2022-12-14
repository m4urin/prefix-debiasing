import torch
from torch import nn
from typing import Union
from src.utils.pytorch import repeat_stacked


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

        self.prefix_embeddings_insert = nn.Parameter(nn.init.normal_(torch.empty(n_prefix_tokens, dim), std=0.5))
        self.prefix_embeddings_layer = None
        if self.prefix_mode == 'replace':
            params = torch.empty(total_layers - self.insert_layer - 1, n_prefix_tokens, dim)
            self.prefix_embeddings_layer = nn.Parameter(nn.init.normal_(params, std=0.5))
        elif self.prefix_mode == 'linear':
            params = torch.empty(total_layers - self.insert_layer - 1, 2, n_prefix_tokens, dim)
            self.prefix_embeddings_layer = nn.Parameter(nn.init.normal_(params, std=0.5))

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
        if self.prefix_mode == 'replace':
            loss += (self.prefix_embeddings_layer.data ** 2).mean()
        elif self.prefix_mode == 'linear':
            loss += ((self.prefix_embeddings_layer[:, 0].data - 1) ** 2).mean()
            loss += (self.prefix_embeddings_layer[:, 1].data ** 2).mean()
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

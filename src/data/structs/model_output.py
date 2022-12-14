from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ModelOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    lm_logits: Optional[Tuple[torch.FloatTensor]] = None
    cls_logits: Optional[Tuple[torch.FloatTensor]] = None

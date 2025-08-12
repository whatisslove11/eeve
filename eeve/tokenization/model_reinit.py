import torch
import torch.nn as nn
from typing import Literal
from transformers import AutoModel


def reinit_model_embs(
    model,
    old_tokenizer,
    new_tokenizer,
    reinit_mode: Literal['random', 'mean', 'eeve']
):
    if isinstance(model, str):
        model = AutoModel.from_pretrained(model)
    
    if reinit_mode == 'random':
        pass
    elif reinit_mode == 'mean':
        pass
    else:
        raise ValueError('Будет реалзиовано позже, вместе с алгоритмом EEVE')
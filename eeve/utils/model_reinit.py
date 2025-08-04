import torch
from typing import Literal
from transformers import PreTrainedModel


def reinit_model_embs(model: str | PreTrainedModel, old_tokenizer, new_tokenizer, reinit_mode: Literal['random', 'mean', 'eeve']):
    if reinit_mode == 'random':
        pass
    elif reinit_mode == 'mean':
        pass
    else:
        raise ValueError('Будет реалзиовано позже, вместе с алгоритмом EEVE')
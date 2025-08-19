import torch
import torch.nn as nn
from typing import Literal
from tqdm.auto import tqdm
from eeve.utils.logger import get_logs_writer_logger


def reinit_model_layers(
    model,
    old_tokenizer,
    new_tokenizer,
    reinit_mode: Literal['small_init', 'mean', 'eeve'],
    tie_weights: bool,
    write_logs: bool = False
):
    if reinit_mode == "eeve":
        raise NotImplementedError
    
    if write_logs:
        logger = get_logs_writer_logger()
    
    vocab_size = len(new_tokenizer.get_vocab())
    dtype = model.get_input_embeddings().weight.dtype
    tie_weights = getattr(model.config, "tying_word_embeddings", False) or tie_weights
    initializer_range = getattr(model.config, "initializer_range", 0.02)
    model.config.vocab_size = vocab_size

    try_lm_head = model.get_output_embeddings()
    new_lm_head = None

    if try_lm_head is not None:
        lm_head_src = try_lm_head.weight.data.clone().to(torch.float32)
        new_lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=dtype)
        new_lm_head.weight.data.normal_(mean=0.0, std=initializer_range)

    embeddings_src = model.get_input_embeddings().weight.data.clone().to(torch.float32)

    new_emb_layer = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=dtype)
    new_emb_layer.weight.data.normal_(mean=0.0, std=initializer_range)

    if reinit_mode in ['eeve', 'mean']:
        with torch.no_grad():
            embedding_mean_vector = torch.mean(embeddings_src, dim=0).to(dtype)
            if new_lm_head is not None:
                lm_head_mean_vector = torch.mean(lm_head_src, dim=0).to(dtype)

            for i in tqdm(range(vocab_size)):
                token = new_tokenizer._tokenizer.id_to_token(i)
                token_idx = old_tokenizer._tokenizer.token_to_id(token)

                if token_idx is None:
                    if reinit_mode == "eeve":
                        # заготовка на будущее
                        continue
                    else:
                        i_token_vector_input = embedding_mean_vector
                else:
                    i_token_vector_input = embeddings_src[token_idx].to(dtype)
                new_emb_layer.weight.data[i].copy_(i_token_vector_input)

                if new_lm_head is not None:
                    if tie_weights:
                        i_token_vector_output = i_token_vector_input
                    else:
                        if token_idx is None:
                            i_token_vector_output = lm_head_mean_vector
                        else:
                            i_token_vector_output = lm_head_src[token_idx].to(dtype)
                    new_lm_head.weight.data[i].copy_(i_token_vector_output)

                if write_logs:
                    logger.info(f'token id in new vocab: {i};\ttoken: {token};\t token id in old vocab: {token_idx}')

    elif reinit_mode == 'small_init':
        pass
    else:
        raise ValueError
    
    model.set_input_embeddings(new_emb_layer)
    if new_lm_head is not None:
        model.set_output_embeddings(new_lm_head)

    if tie_weights:
        model.tie_weights()
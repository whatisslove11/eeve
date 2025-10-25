from typing import Literal

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from eeve.utils.logger import get_logs_writer_logger


def _is_space_marker(tstr: str) -> bool:
    if tstr == "▁":
        return True
    if tstr in {"Ġ", "Ċ", "ÂĊ"}:
        return True
    return False


def get_eeve_embeddings_for_input(input_weights, token, old_tokenizer):
    tokens = old_tokenizer(token, add_special_tokens=False)["input_ids"]
    subword_tokens_embedding = input_weights[tokens].mean(dim=0)
    return subword_tokens_embedding


def get_eeve_embeddings_for_output(
    output_weights, token, old_tokenizer, clear_tokenization: bool = False
):
    tokens = old_tokenizer(token, add_special_tokens=False)["input_ids"]
    if clear_tokenization:
        unk_id = getattr(old_tokenizer, "unk_token_id", None)
        tokens_str = old_tokenizer.convert_ids_to_tokens(tokens)
        all_special_ids = set(getattr(old_tokenizer, "all_special_ids", []) or [])

        chosen_id = None
        for tid, tstr in zip(tokens, tokens_str):
            if tid in all_special_ids:
                continue
            if _is_space_marker(tstr):
                continue
            if unk_id is not None and tid == unk_id:
                continue
            # add bytes
            chosen_id = tid
            break
        # maybe mean vec?
        if chosen_id is None:
            chosen_id = tokens[0]
        first_subword_token_embedding = output_weights[chosen_id]
    else:
        first_subword_token_embedding = output_weights[tokens[0]]
    return first_subword_token_embedding


def reinit_model_layers(
    model,
    old_tokenizer,
    new_tokenizer,
    reinit_mode: Literal["small_init", "mean", "eeve"],
    tie_weights: bool = False,
    write_logs: bool = False,
):
    if write_logs:
        logger = get_logs_writer_logger()

    vocab_size = len(new_tokenizer.get_vocab())
    dtype = model.get_input_embeddings().weight.dtype
    tie_weights = getattr(model.config, "tie_word_embeddings", False) or tie_weights
    initializer_range = getattr(model.config, "initializer_range", 0.02)
    model.config.vocab_size = vocab_size

    try_lm_head = model.get_output_embeddings()
    new_lm_head = None

    if try_lm_head is not None:
        lm_head_src = try_lm_head.weight.data.clone().to(torch.float32)
        new_lm_head = nn.Linear(
            model.config.hidden_size, model.config.vocab_size, bias=False, dtype=dtype
        )
        new_lm_head.weight.data.normal_(mean=0.0, std=initializer_range)

    embeddings_src = model.get_input_embeddings().weight.data.clone().to(torch.float32)

    new_emb_layer = nn.Embedding(
        model.config.vocab_size, model.config.hidden_size, dtype=dtype
    )
    new_emb_layer.weight.data.normal_(mean=0.0, std=initializer_range)

    if reinit_mode in ["eeve", "mean"]:
        with torch.no_grad():
            embedding_mean_vector = torch.mean(embeddings_src, dim=0).to(dtype)
            if new_lm_head is not None:
                lm_head_mean_vector = torch.mean(lm_head_src, dim=0).to(dtype)

            for i in tqdm(range(vocab_size)):
                token = new_tokenizer._tokenizer.id_to_token(i)
                token_idx = old_tokenizer._tokenizer.token_to_id(token)

                if token_idx is None:
                    if reinit_mode == "eeve":
                        i_token_vector_input = get_eeve_embeddings_for_input(
                            embeddings_src, token, old_tokenizer
                        )
                    else:
                        i_token_vector_input = embedding_mean_vector
                else:
                    i_token_vector_input = embeddings_src[token_idx].to(dtype)
                new_emb_layer.weight.data[i].copy_(i_token_vector_input)

                if new_lm_head is not None and not tie_weights:
                    if token_idx is None:
                        if reinit_mode == "eeve":
                            i_token_vector_output = get_eeve_embeddings_for_output(
                                lm_head_src, token, old_tokenizer
                            )
                        else:
                            i_token_vector_output = lm_head_mean_vector
                    else:
                        i_token_vector_output = lm_head_src[token_idx].to(dtype)
                    new_lm_head.weight.data[i].copy_(i_token_vector_output)

                if write_logs:
                    logger.info(
                        f"token id in new vocab: {i};\ttoken: {token};\t token id in old vocab: {token_idx}"
                    )

    elif reinit_mode == "small_init":
        if write_logs:
            logger.info(
                f"Using small_init: embeddings initialized with normal distribution (std={initializer_range})"
            )
    else:
        raise ValueError(
            f"Unsupported reinit_mode: '{reinit_mode}'. Must be one of: 'small_init', 'mean', 'eeve'"
        )

    model.set_input_embeddings(new_emb_layer)
    if new_lm_head is not None:
        model.set_output_embeddings(new_lm_head)

    if tie_weights:
        model.tie_weights()

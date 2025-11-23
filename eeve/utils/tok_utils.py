def diff_token_ids(*, old_tokenizer, new_tokenizer) -> list[int]:
    old_t2id = old_tokenizer.get_vocab()
    new_t2id = new_tokenizer.get_vocab()

    only_in_new_ids = [new_t2id[t] for t in new_t2id.keys() - old_t2id.keys()]

    return sorted(only_in_new_ids)

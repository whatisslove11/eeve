import warnings

from sentencepiece import sentencepiece_model_pb2 as sp_pb2


def shrink_spm(
    model_path: str,
    output_path: str,
    target_size: int | None = None,
    remove_count: int | None = None,
    by_score: bool = False,
) -> None:
    m = sp_pb2.ModelProto()
    with open(model_path, "rb") as f:
        m.ParseFromString(f.read())

    pieces = list(m.pieces)

    if target_size is not None:
        kept_pieces = pieces[:target_size]
    elif remove_count is not None and remove_count > 0:
        if remove_count >= len(pieces):
            raise ValueError(
                f"{remove_count=} cannot exceed vocabulary size={len(pieces)}."
            )
        if by_score:
            sorted_indices = sorted(range(len(pieces)), key=lambda i: pieces[i].score)
            indices_to_keep = set(sorted_indices[remove_count:])
            kept_pieces = [p for i, p in enumerate(pieces) if i in indices_to_keep]
        else:
            kept_pieces = pieces[:-remove_count]
    else:
        warnings.warn(
            "No target_size or remove_count specified. Model copied unchanged."
        )
        return

    m.ClearField("pieces")
    m.pieces.extend(kept_pieces)
    m.trainer_spec.vocab_size = len(kept_pieces)

    with open(output_path, "wb") as f:
        f.write(m.SerializeToString())

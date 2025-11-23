import torch


def compress_ranges(xs: list[int]) -> list[int]:
    if not xs:
        return []

    res = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev:
            continue
        if x == prev + 1:
            prev = x
        else:
            res.append(start)
            res.append(prev)
            start = prev = x
    res.append(start)
    res.append(prev)

    return res


def ranges_to_mask(ranges: list[int], num_tokens: int) -> torch.Tensor:
    if len(ranges) % 2 != 0:
        raise ValueError("Invalid ranges: expected even-length [start, end] pairs.")
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    for i in range(0, len(ranges), 2):
        l = max(0, ranges[i])
        r = min(num_tokens - 1, ranges[i + 1])
        if l <= r:
            mask[l : r + 1] = True
    return mask


def token_ids_to_mask(diff_tokens_ids: list[int], num_tokens: int) -> torch.Tensor:
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[diff_tokens_ids] = True

    return mask

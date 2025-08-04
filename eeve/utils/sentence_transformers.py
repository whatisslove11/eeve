from typing import Callable
from tqdm.auto import tqdm
from sentence_transformers.readers import InputExample


def _prepare_input_examples(
    ds,
    feature_1: str,
    feature_2: str,
    prepare_example_fn: Callable[..., str] = lambda x: x
):
    res = []
    for item in tqdm(ds):
        res.append(
            InputExample(
                texts=[
                    prepare_example_fn(item[feature_1]),
                    prepare_example_fn(item[feature_2])
                ],
                label=1.0
            )
        )
    return res
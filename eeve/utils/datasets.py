import random
from typing import Literal
from datasets import load_dataset, Dataset, DatasetDict


def _load_dataset_from_path(path: str, test_size: float | None = None, load_kwargs = {}) -> DatasetDict:  
    if path.endswith("jsonl"):
        dataset = load_dataset("json", data_files=path, **load_kwargs)
    else:
        dataset = load_dataset(path, **load_kwargs)
    
    if test_size is not None and 'test' not in dataset.keys() and 'train' in dataset.keys():
        dataset = dataset["train"].train_test_split(
            test_size, seed=42, load_from_cache_file=True
        )
    
    return dataset


def _sample_dataset(
    dataset: Dataset,
    mode: Literal['random', 'sequential'],
    num_samples: int | None = None,
    ds_ratio: float | None = None
) -> Dataset:
    total_samples = len(dataset)

    if num_samples is None and ds_ratio is None:
        raise ValueError("Either num_samples or ds_ratio must be specified")
    if num_samples is not None and ds_ratio is not None:
        raise ValueError("Only one of num_samples or ds_ratio should be specified, not both")

    total_to_select = num_samples if num_samples is not None else int(total_samples * ds_ratio)
    idx = range(total_to_select) if mode =='sequential' else random.sample(range(total_samples), total_to_select)

    dataset = dataset.select(idx)
    return dataset
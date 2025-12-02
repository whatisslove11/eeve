import os
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from eeve.utils.logger import get_logger


@dataclass
class DatasetConfig:
    name: str
    load_kwargs: dict[str, Any] = field(default_factory=dict)


def _load_dataset_from_path(
    path: str, test_size: float | None = None, load_kwargs=None
) -> DatasetDict:
    load_kwargs = load_kwargs or {}
    if path.endswith("jsonl") or path.endswith("json") or path.endswith("gz"):
        dataset = load_dataset("json", data_files=path, **load_kwargs)
    elif path.endswith("csv"):
        dataset = load_dataset("csv", data_files=path, **load_kwargs)
    else:
        dataset = load_dataset(path, **load_kwargs)

    if (
        test_size is not None
        and "test" not in dataset.keys()
        and "train" in dataset.keys()
    ):
        dataset = dataset["train"].train_test_split(
            test_size, seed=42, load_from_cache_file=True
        )

    return dataset


def _sample_dataset(
    dataset: Dataset,
    mode: Literal["random", "sequential"],
    num_samples: int | None = None,
    ds_ratio: float | None = None,
) -> Dataset:
    total_samples = len(dataset)

    if num_samples is None and ds_ratio is None:
        raise ValueError("Either num_samples or ds_ratio must be specified")
    if num_samples is not None and ds_ratio is not None:
        raise ValueError(
            "Only one of num_samples or ds_ratio should be specified, not both"
        )

    total_to_select = (
        num_samples if num_samples is not None else int(total_samples * ds_ratio)
    )
    idx = (
        range(total_to_select)
        if mode == "sequential"
        else random.sample(range(total_samples), total_to_select)
    )

    dataset = dataset.select(idx)
    return dataset


def _convert_datasets_to_txt(
    dataset_configs: list[DatasetConfig],
    output_dir: str,
    output_filename: str,
    max_file_size: int = -1,
) -> None:
    """
    Download datasets from Hugging Face and save content from all columns to a single text file.
    Used for spm train

    Args:
        dataset_configs: List of DatasetConfig objects with dataset names and load kwargs
        output_dir: Directory to save the output file
        output_filename: Name of the output file
        max_file_size: Maximum file size in bytes. If -1, no size limit is applied.
    """
    logger = get_logger(logger_name=__name__)
    os.makedirs(output_dir, exist_ok=True)

    file_index = 0
    current_size = 0

    def get_output_path(index):
        if max_file_size <= 0:
            return os.path.join(output_dir, output_filename)
        base_name, ext = os.path.splitext(output_filename)
        return os.path.join(output_dir, f"{base_name}_{index:03d}{ext}")

    output_path = get_output_path(file_index)
    f = open(output_path, "w", encoding="utf-8")

    for config in tqdm(dataset_configs, desc="Processing datasets"):
        try:
            dataset = _load_dataset_from_path(
                path=config.name, load_kwargs=config.load_kwargs
            )

            for split in dataset.keys():
                ds_split = dataset[split]
                if len(ds_split.column_names) == 0:
                    continue
                for column_name in ds_split.column_names:
                    for entry in tqdm(ds_split[column_name]):
                        if entry is not None:
                            line = str(entry) + "\n"
                            line_size = len(line.encode("utf-8"))

                            if (
                                max_file_size > 0
                                and current_size + line_size > max_file_size
                                and current_size > 0
                            ):
                                f.close()
                                file_index += 1
                                output_path = get_output_path(file_index)
                                f = open(output_path, "w", encoding="utf-8")
                                current_size = 0

                            f.write(line)
                            current_size += line_size

        except Exception as e:
            logger.error(f"Error processing dataset {config.name}: {e}")
            continue

    f.close()

    logger.info(f"All data saved to {output_path}")

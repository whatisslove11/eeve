import os
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformerTrainer

from eeve.utils.dataset import _load_dataset_from_path
from eeve.utils.seed import seed_everything


warnings.filterwarnings("ignore")


def _build_evaluator(cfg: DictConfig, eval_dataset):
    evaluator_cfg = OmegaConf.create(OmegaConf.to_container(cfg.evaluator, resolve=True))

    evaluator_init_kwargs = OmegaConf.to_container(
        evaluator_cfg.pop("kwargs", {}), resolve=True
    )
    dataset_kwargs_from_columns = OmegaConf.to_container(
        evaluator_cfg.pop("dataset_kwargs_from_columns", {}), resolve=True
    )
    auto_translation_columns = evaluator_cfg.pop("auto_translation_columns", True)

    for arg_name, column_name in dataset_kwargs_from_columns.items():
        if column_name not in eval_dataset.column_names:
            raise ValueError(
                f"Column '{column_name}' not found in eval_dataset for evaluator argument '{arg_name}'. "
                f"Available columns: {list(eval_dataset.column_names)}"
            )
        evaluator_init_kwargs[arg_name] = list(eval_dataset[column_name])

    if auto_translation_columns:
        source_column = cfg.data.get("source_column")
        target_column = cfg.data.get("target_column")

        if source_column and "source_sentences" not in evaluator_init_kwargs:
            if source_column not in eval_dataset.column_names:
                raise ValueError(
                    f"Column '{source_column}' not found in eval_dataset for evaluator argument 'source_sentences'. "
                    f"Available columns: {list(eval_dataset.column_names)}"
                )
            evaluator_init_kwargs["source_sentences"] = list(eval_dataset[source_column])

        if target_column and "target_sentences" not in evaluator_init_kwargs:
            if target_column not in eval_dataset.column_names:
                raise ValueError(
                    f"Column '{target_column}' not found in eval_dataset for evaluator argument 'target_sentences'. "
                    f"Available columns: {list(eval_dataset.column_names)}"
                )
            evaluator_init_kwargs["target_sentences"] = list(eval_dataset[target_column])

    return hydra.utils.instantiate(evaluator_cfg, **evaluator_init_kwargs)

@hydra.main(
    config_path="../../configs", config_name="st_hydra_config", version_base=None
)
def train(cfg: DictConfig):
    seed_everything(cfg.seed)

    train_dataset = None
    eval_dataset = None
    load_kwargs = OmegaConf.to_container(
        cfg.data.train.get("load_kwargs", {}), resolve=True
    )

    if test_size_from_train := cfg.data.eval.get("test_size_from_train"):
        ds_dict = _load_dataset_from_path(
            path=cfg.data.train.path,
            test_size=test_size_from_train,
            load_kwargs=load_kwargs,
        )
        train_dataset = ds_dict["train"]
        eval_dataset = ds_dict["test"]

    elif split_from_train_path := cfg.data.eval.get("split_from_train_path"):
        ds_dict = _load_dataset_from_path(
            path=cfg.data.train.path, load_kwargs=load_kwargs
        )
        train_dataset = ds_dict[cfg.data.train.split]
        eval_dataset = ds_dict[split_from_train_path]

    elif cfg.data.eval.get("path"):
        train_ds_dict = _load_dataset_from_path(
            path=cfg.data.train.path, load_kwargs=load_kwargs
        )
        eval_ds_dict = _load_dataset_from_path(path=cfg.data.eval.path)

        train_dataset = train_ds_dict[cfg.data.train.split]
        eval_dataset = eval_ds_dict[cfg.data.eval.get("split", "train")]

    if train_dataset is None or eval_dataset is None:
        raise ValueError(
            "Unable to determine the datasets for training and evaluation. Check the configuration in `data.eval`."
        )

    if cfg.preprocessing.get("apply_function"):
        preprocess_func = hydra.utils.get_method(
            cfg.preprocessing.apply_function._target_
        )
        fn_kwargs = OmegaConf.to_container(
            cfg.preprocessing.apply_function.get("fn_kwargs", {}), resolve=True
        )

        train_dataset = train_dataset.map(
            lambda example: preprocess_func(example=example, **fn_kwargs),
            load_from_cache_file=True,
        )
        eval_dataset = eval_dataset.map(
            lambda example: preprocess_func(example=example, **fn_kwargs),
            load_from_cache_file=True,
        )

    model = hydra.utils.instantiate(cfg.model)
    loss = hydra.utils.instantiate(cfg.loss, model=model)

    evaluator = _build_evaluator(cfg=cfg, eval_dataset=eval_dataset)

    training_args = hydra.utils.instantiate(cfg.training)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    final_output_path = os.path.join(training_args.output_dir, "final_model")
    model.save(final_output_path)


if __name__ == "__main__":
    train()

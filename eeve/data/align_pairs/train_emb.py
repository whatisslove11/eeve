import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import sentence_transformers
from sentence_transformers import SentenceTransformerTrainer
from eeve.utils.dataset import _load_dataset_from_path
from eeve.utils.seed import seed_everything

import warnings
warnings.filterwarnings('ignore')


@hydra.main(config_path="../../../configs", config_name="st_hydra_config", version_base=None)
def train(cfg: DictConfig):
    seed_everything(cfg.seed)
    
    train_dataset = None
    eval_dataset = None
    load_kwargs = OmegaConf.to_container(
        cfg.data.train.get('load_kwargs', {}), 
        resolve=True
    )

    if test_size_from_train := cfg.data.eval.get('test_size_from_train'):
        ds_dict = _load_dataset_from_path(
            path=cfg.data.train.path,
            test_size=test_size_from_train,
            load_kwargs=load_kwargs
        )
        train_dataset = ds_dict['train']
        eval_dataset = ds_dict['test']

    elif split_from_train_path := cfg.data.eval.get('split_from_train_path'):
        ds_dict = _load_dataset_from_path(
            path=cfg.data.train.path,
            load_kwargs=load_kwargs
        )
        train_dataset = ds_dict[cfg.data.train.split]
        eval_dataset = ds_dict[split_from_train_path]

    elif cfg.data.eval.get('path'):
        train_ds_dict = _load_dataset_from_path(path=cfg.data.train.path, load_kwargs=load_kwargs)
        eval_ds_dict = _load_dataset_from_path(path=cfg.data.eval.path)
        
        train_dataset = train_ds_dict[cfg.data.train.split]
        eval_dataset = eval_ds_dict[cfg.data.eval.get('split', 'train')]

    if train_dataset is None or eval_dataset is None:
        raise ValueError("Unable to determine the datasets for training and evaluation. Check the configuration in `data.eval`.")

    if cfg.preprocessing.get('apply_function'):
        preprocess_func = hydra.utils.get_method(cfg.preprocessing.apply_function._target_)
        fn_kwargs = OmegaConf.to_container(
            cfg.preprocessing.apply_function.get('fn_kwargs', {}), 
            resolve=True
        )

        train_dataset = train_dataset.map(
            lambda example: preprocess_func(example=example, **fn_kwargs),
            load_from_cache_file=True
        )
        eval_dataset = eval_dataset.map(
            lambda example: preprocess_func(example=example, **fn_kwargs),
            load_from_cache_file=True
        )

    model = hydra.utils.instantiate(cfg.model)
    loss = hydra.utils.instantiate(cfg.loss, model=model)

    source_sentences = list(eval_dataset[cfg.data.source_column])
    target_sentences = list(eval_dataset[cfg.data.target_column])

    # TODO: допилить для использования любых параметров в evaluator
    evaluator = hydra.utils.instantiate(
        cfg.evaluator,
        source_sentences=source_sentences,
        target_sentences=target_sentences,
    )

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
import inspect
import os
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import evaluate
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset

from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
    DataCollatorWithPadding,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from eeve.configs.bt_config import BackTranslationConfig
# from trl.trainer import disable_dropout_in_model
# from trl.trainer.utils import decode_and_strip_padding, print_rich_table


class BackTranslationTrainer(Trainer):
    _tag_names = ["backtranslation-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[BackTranslationConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = ( # че блять, почему
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[dict] = None,
    ):
        
        if compute_metrics is None:
            compute_metrics = self._default_compute_metrics
            self.bleu_metric = evaluate.load("bleu")
            self.chrf_metric = evaluate.load("chrf")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    
    def _default_compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        predictions, labels = eval_pred

        labels[labels == -100] = self.tokenizer.pad_token_id

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        references = [[label] for label in decoded_labels]

        bleu_score = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=references
        )

        chrf_score = self.chrf_metric.compute(
            predictions=decoded_preds,
            references=references,
            word_order=2
        )
        
        return {
            "bleu": bleu_score["bleu"],
            "chrf2": chrf_score["score"],
        }
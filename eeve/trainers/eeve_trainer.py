import contextlib
import functools
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.data_loader import skip_first_batches
from packaging import version
from transformers import (
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import ExportableState, TrainerCallback, TrainerState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import EvalPrediction, TrainOutput, speed_metrics
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)
from trl import SFTTrainer
from trl.trainer import disable_dropout_in_model

from datasets import Dataset, IterableDataset
from eeve.callbacks.trainable_params import EeveStageTrainableParamsCallback
from eeve.configs.eeve_config import EEVE_SCHEDULE, EeveConfig
from eeve.utils.find_layers import (
    find_layers_by_shape_and_type,
    get_first_layer_by_type,
    is_module_exists,
)
from eeve.utils.list_utils import token_ids_to_mask
from eeve.utils.tok_utils import diff_token_ids


if TYPE_CHECKING:
    from peft import PeftConfig


if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate.utils import (
        DistributedType,
    )

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
    if IS_XLA_FSDPV2_POST_2_2:
        pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"


class EeveTrainer(SFTTrainer):
    """
    Trainer for Efficient and Effective Vocabulary Expansion (EEVE) method.
    Based on article [`Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models`](https://arxiv.org/pdf/2402.14714)

    This class is a wrapper around the [`trl.SFTTrainer`] (which also a wrapper around [`transformers.Trainer`]) class and inherits all of its attributes and methods.

    !!! NOTE !!!
    Оригинальная имплементация не применялась на стадии SFT - но данная реализация все равно повторяет все необходимые шаги обучения

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        callback_for_stage_switching: [`~transformers.TrainerCallback`]
            pass
        src_model_path: str
            Путь до/наименование оргиниальной модели (к которой были добавлены токены в словарь и изменены эмбеддинги)
            Используется для определения токенов, которые нужно будет обучать
        args ([`EeveConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to a custom [`DataCollatorForLanguageModeling`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Optional[Callable]`):
            Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly
            converts the dataset into a [language modeling](#language-modeling) type.
    """

    _tag_names = ["eeve-trainer"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        callback_for_stage_switching: TrainerCallback,
        src_model_path: str,
        args: Optional[EeveConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        optimizer_cls_and_kwargs: Optional[
            tuple[type[torch.optim.Optimizer], dict[str, Any]]
        ] = None,
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable[[dict], str]] = None,
    ):
        opt, sch = optimizers
        if opt is not None or sch is not None:
            raise ValueError(
                "На каждой из стадии Eeve обучается разное количество параметров, из-за чего на каждой стадии оптимизатор будет создаваться заново"
            )
        if peft_config is not None:
            raise ValueError(
                "EeveTrainer does not support PEFT integration; please disable peft_config."
            )
        if args.resume_from_checkpoint is not None:
            raise ValueError("'resume_from_checkpoint' support will be added later")
        callbacks = (callbacks or []) + [callback_for_stage_switching]

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            formatting_func=formatting_func,
        )

        self.stage_spec = EEVE_SCHEDULE
        if self.args.disable_dropout:
            disable_dropout_in_model(self.model)

        emb_module = is_module_exists(self.model, "model.embed_tokens")
        if emb_module is not None:
            self.embedding_layer_name = "embed_tokens"
            self.vocab_size, self.hidden_size = emb_module.weight.shape
        else:
            found_name, found_shape = get_first_layer_by_type(
                self.model, module_type=nn.Embedding, return_shape=True
            )
            if found_name is None:
                raise ValueError(
                    "Could not find any nn.Embedding layer in the model structure."
                )

            self.embedding_layer_name = found_name
            self.vocab_size, self.hidden_size = found_shape

        head_module = is_module_exists(self.model, "lm_head")

        if head_module is not None:
            self.lm_head_name = "lm_head"
        else:
            found_layers = find_layers_by_shape_and_type(
                self.model, nn.Linear, (self.vocab_size, self.hidden_size)
            )

            if not found_layers:
                raise ValueError(
                    f"Could not find a Linear layer (LM Head) with dimensions {(self.vocab_size, self.hidden_size)} matching the embedding layer."
                )
            self.lm_head_name = found_layers[0]

        src_tokenizer = AutoTokenizer.from_pretrained(src_model_path)
        diff_tokens_ids = diff_token_ids(
            old_tokenizer=src_tokenizer, new_tokenizer=self.processing_class
        )
        num_diff_tokens = len(diff_tokens_ids)
        if num_diff_tokens == 0:
            raise ValueError(
                "No token difference detected between source and target tokenizers; nothing to train."
            )

        grad_mask = (
            token_ids_to_mask(
                diff_tokens_ids=diff_tokens_ids, num_tokens=self.vocab_size
            )
            .unsqueeze(dim=1)
            .expand((-1, self.hidden_size))
        )
        self.model.register_buffer("grad_mask", grad_mask, persistent=False)

        self._global_steps = 0
        self._grad_hooks = []
        self.all_stages_trainer_states: dict[str, TrainOutput] = {}

        if self.args.enable_param_stats_logging:
            every_stage_params_tracker_callback = EeveStageTrainableParamsCallback(
                embedding_layer_name=self.embedding_layer_name,
                lm_head_name=self.lm_head_name,
                num_tokens_for_hook=num_diff_tokens,
            )
            self.add_callback(every_stage_params_tracker_callback)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def train(self, **kwargs):
        for stage_info in self.stage_spec:
            self._remove_hooks()
            self._reset_optimizer_and_scheduler()
            self._set_layer_trainability(model=self.model, stage_info=stage_info)
            state = super().train(**kwargs)
            self.all_stages_trainer_states[stage_info.name] = state
            self._global_steps += state.global_step
        return self.all_stages_trainer_states

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        """
        Patched
        """
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = (
                        self._train_batch_size // max(1, self.args.n_gpu)
                    )
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(
            args, train_dataloader, total_train_batch_size
        )

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(
                train_dataloader, None if epoch_based else max_steps
            )
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled()
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
            or self.is_tp_enabled
        )

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (
            getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
        )
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    if delay_optimizer_creation:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(
                            self.model, self.optimizer
                        )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                # deepspeed_load_checkpoint(
                #     self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                # )
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=True
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        eeve_stage_skip_batches = 0
        if args.skip_batches_between_stages and self._global_steps > 0:
            epochs_trained = int(self._global_steps // num_update_steps_per_epoch)
            eeve_stage_skip_batches = self._global_steps % (num_update_steps_per_epoch)
            eeve_stage_skip_batches *= args.gradient_accumulation_steps
            logger.info(
                f"'skip_batches_between_stages' is True. Will be skipped first {eeve_stage_skip_batches} batches for this stage for first epoch."
            )

            # Check if continuing training from a checkpoint
            # костыль - пока resume_from_checkpoint всегда None, потом написать нормально
            # if resume_from_checkpoint is not None and os.path.isfile(
            #     os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            # ):
            #     self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            #     self.compare_trainer_and_checkpoint_args(self.args, self.state)
            #     self._load_callback_state()
            #     epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            #     if not args.ignore_data_skip:
            #         steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            #         steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            #     else:
            #         steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            # todo: плохо написано, реализовать поддержку resume_checkpoint
            rng_to_sync = False
            steps_skipped = 0
            # if steps_trained_in_current_epoch > 0:
            #     epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
            #     steps_skipped = steps_trained_in_current_epoch
            #     steps_trained_in_current_epoch = 0
            #     rng_to_sync = True

            if eeve_stage_skip_batches > 0:
                epoch_dataloader = skip_first_batches(
                    epoch_dataloader, eeve_stage_skip_batches
                )
                steps_skipped = eeve_stage_skip_batches
                eeve_stage_skip_batches = 0

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = (
                    args.gradient_accumulation_steps
                    if update_step != (total_updates - 1)
                    else remainder
                )
                batch_samples, num_items_in_batch = self.get_batch_samples(
                    epoch_iterator, num_batches, args.device
                )
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (
                        step + 1
                    ) % args.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(
                            self.model, "main_input_name", "input_ids"
                        )
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(
                                input_tokens, device=self.args.device, dtype=torch.int64
                            )
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(
                            args, self.state, self.control
                        )

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type
                        != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(
                            model, inputs, num_items_in_batch
                        )

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged
                        )
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(
                                    args.max_grad_norm
                                )
                            elif self.use_apex:
                                from apex import amp

                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                grad_norm_context = contextlib.nullcontext
                                if self.is_tp_enabled:
                                    from torch.distributed._tensor.experimental import (  # type: ignore  # noqa: I001
                                        implicit_replication,
                                    )

                                    grad_norm_context = implicit_replication
                                with grad_norm_context():
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type
                                == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(
                            args, self.state, self.control
                        )

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(
                            args, self.state, self.control
                        )

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(
                                self.lr_scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                            ):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = (
                            epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        )
                        self.control = self.callback_handler.on_step_end(
                            args, self.state, self.control
                        )
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(
                            args, self.state, self.control
                        )

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if (
                        self.control.should_epoch_stop
                        or self.control.should_training_stop
                    ):
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate=learning_rate,
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _set_layer_trainability(self, model, stage_info) -> None:
        def freeze_partial_embedding_hook(grad):
            grad = grad * self.model.grad_mask.to(grad.device)
            return grad

        for name, param in model.named_parameters():
            if self.embedding_layer_name in name:
                mode = stage_info.embedding
            elif self.lm_head_name in name:
                mode = stage_info.lm_head
            else:
                mode = stage_info.transformer_block

            if mode == "frozen":
                param.requires_grad = False
            elif mode == "trainable":
                param.requires_grad = True
            elif mode == "partial":
                param.requires_grad = True
                self._grad_hooks.append(
                    param.register_hook(freeze_partial_embedding_hook)
                )
            else:
                raise ValueError(
                    f"The EEVE stage configuration contains an invalid mode: '{mode}'. Please verify the EEVE_SCHEDULE integrity."
                )

    def _remove_hooks(self) -> None:
        for h in self._grad_hooks:
            h.remove()
        self._grad_hooks.clear()

    def _reset_optimizer_and_scheduler(self):
        # Need to auto re-create optimizer and lr_scheduler every stage
        self.optimizer, self.lr_scheduler = None, None

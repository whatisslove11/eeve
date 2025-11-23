from accelerate import PartialState
from transformers.trainer_callback import TrainerCallback

from eeve.utils.stat_storage import StatStorage
from eeve.utils.vis_helper import VisHelper


def count_module_parameters(module):
    total_params = module.numel()
    trainable_params = module.numel() if module.requires_grad else 0
    return total_params, trainable_params


class EeveStageTrainableParamsCallback(TrainerCallback):
    def __init__(
        self, embedding_layer_name: str, lm_head_name: str, num_tokens_for_hook: int
    ):
        self.embedding_layer_name = embedding_layer_name
        self.lm_head_name = lm_head_name
        self.num_tokens_for_hook = num_tokens_for_hook

        self.names = [
            self.embedding_layer_name,
            self.lm_head_name,
            "transformer_layers",
        ]

        self.vis_helper = VisHelper(title="Parameters statistics during training")

    def on_train_begin(self, args, state, control, **kwargs):
        _stats_map = {
            name: StatStorage(
                total_params=0,
                trainable_params=0,
                hooked_params=0,
            )
            for name in self.names
        }

        model = kwargs.get("model", None)
        if model is None:
            trainer = kwargs.get("trainer", None)
            if trainer is not None:
                model = trainer.model

        for name, param in model.named_parameters():
            (total_params, trainable_params), hooked_params = (
                count_module_parameters(param),
                0,
            )

            if getattr(param, "_backward_hooks", False):
                hooked_params += self.num_tokens_for_hook * param.shape[1]
                trainable_params -= hooked_params

            layer_type = next(
                (
                    layer
                    for layer in [self.embedding_layer_name, self.lm_head_name]
                    if layer in name
                ),
                "transformer_layers",
            )

            _stats_map[layer_type] += StatStorage(
                total_params=total_params,
                trainable_params=trainable_params,
                hooked_params=hooked_params,
            )

        if PartialState().is_main_process:
            self.vis_helper.print_comparisons(_stats_map)

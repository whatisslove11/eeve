from dataclasses import dataclass, field
from typing import Literal

from trl import SFTConfig


State = Literal["frozen", "partial", "trainable"]
TransformerState = Literal["frozen", "trainable"]


@dataclass
class EeveConfig(SFTConfig):
    disable_dropout: bool = field(
        default=True, metadata={"help": "Whether to disable dropout in the model."}
    )
    enable_param_stats_logging: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable the callback for logging detailed statistics of trainable, frozen, and partially updated parameters at each stage."
        },
    )
    skip_batches_between_stages: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to skip batches consumed in the previous training stage when transitioning "
                "to the next stage. If `True`, the dataloader continues iteration from where it "
                "left off (e.g., if stage 1 ended at batch 615, stage 2 starts at 616). "
                "If `False`, the dataloader resets to the beginning for each new stage. "
                "Note: This skipping applies only to the first epoch of the new stage. "
                "The skipped batches are not discarded; subsequent epochs will iterate over the "
                "full dataloader."
            )
        },
    )


@dataclass(frozen=True)
class StageSpec:
    name: str
    embedding: State
    lm_head: State
    transformer_block: TransformerState


EEVE_SCHEDULE: tuple[StageSpec, ...] = (
    StageSpec("stage 1", "partial", "frozen", "frozen"),
    StageSpec("stage 2", "frozen", "partial", "frozen"),
    StageSpec("stage 3", "partial", "partial", "frozen"),
    StageSpec("stage 4", "frozen", "trainable", "frozen"),
    StageSpec("stage 5", "partial", "trainable", "frozen"),
    StageSpec("stage 6", "trainable", "trainable", "trainable"),
    StageSpec("stage 7", "frozen", "frozen", "trainable"),
)

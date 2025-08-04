from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments

@dataclass
class BackTranslationConfig(TrainingArguments):
    pass
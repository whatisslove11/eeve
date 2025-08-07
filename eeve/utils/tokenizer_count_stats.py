from rich import box
from rich.console import Console
from rich.table import Table

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


class StatStorage:
    def __init__(self, **kwargs):
        self._stats = kwargs

    def __add__(self, other):
        if not isinstance(other, StatStorage):
            raise TypeError(f"Cannot add DatasetStats with {type(other)}")

        self_keys = set(self._stats.keys())
        other_keys = set(other._stats.keys())

        if self_keys != other_keys:
            raise ValueError(
                f"Object keys do not match. "
                f"Self: {self_keys}, Other: {other_keys}"
            )

        result_stats = {}
        for key in self._stats:
            result_stats[key] = self._stats[key] + other._stats[key]

        return StatStorage(**result_stats)

    def __getattr__(self, name):
        if name in self._stats:
            return self._stats[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == '_stats':
            super().__setattr__(name, value)
        else:
            self._stats[name] = value

    def __repr__(self):
        stats_str = ', '.join(f'{k}={v}' for k, v in self._stats.items())
        return f"StatStorage({stats_str})"

    def to_dict(self):
        return self._stats.copy()

    def to_json(self):
        import json
        return json.dumps(self._stats)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    

class VisHelper:
    def __init__(self, stats: StatStorage):
        self.stats = stats
    
    def print_rich_table(self):
        console = Console()
        table = Table(title="Statistics", box=box.ROUNDED)
        
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        stats_dict = self.stats.to_dict()
        
        for key, value in stats_dict.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                formatted_value = f"{value:,.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            elif isinstance(value, list):
                formatted_value = f"[{', '.join(str(v) for v in value)}]"
            elif isinstance(value, dict):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            table.add_row(formatted_key, formatted_value)
        
        console.print(table)
    

class CalculateTokenizerStats:
    def __init__(
        self,
        tokenizer_name_or_path: str,
        dataset: str,
        load_kwargs: dict = {},
        streaming: bool = False,
        batch_size: int = 1000
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.dataset = dataset
        self.load_kwargs = load_kwargs
        self.streaming = streaming
        self.batch_size = batch_size

        self._stats = StatStorage(
            total_unk_tokens=0,
            overall_sentences=0,
            total_sentences_with_unk=0,
            overall_tokens=0,
            overall_chars=0
        )

    def run(self) -> StatStorage:
        ds = load_dataset(self.dataset, **self.load_kwargs, streaming=self.streaming)

        if isinstance(ds, dict):
            raise ValueError(
                f"You forgot to specify the split of the dataset. Update your dataset_options to include 'split'. Available splits: {list(ds.keys())}"
            )

        with tqdm(total=None) as pbar:
            for batch in ds.iter(self.batch_size):
                self.update_stats_on_batch(batch)
            pbar.update(self.batch_size)
        return self._stats

    def update_stats_on_batch(self, batch: dict) -> None:
        for sample in batch.values():
            input_ids = self.tokenizer(
                sample,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )['input_ids']

            char_len = sum(map(len, sample))
            target = self.tokenizer.unk_token_id or -1

            for tokens in input_ids:
                count = tokens.count(target)
                exists = 1 if count > 0 else 0

                self._stats.overall_sentences += 1
                self._stats.overall_tokens += len(tokens)
                self._stats.total_sentences_with_unk += exists
                self._stats.total_unk_tokens += count
            self._stats.overall_chars += char_len

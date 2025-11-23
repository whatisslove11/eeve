from tqdm.auto import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from eeve.utils.stat_storage import StatStorage


class CalculateTokenizerStats:
    def __init__(
        self,
        tokenizer_names_or_paths: str | list[str],
        dataset: str,
        load_kwargs: dict = None,
        streaming: bool = False,
        batch_size: int = 1000,
    ):
        self._tokenizer_names = (
            [tokenizer_names_or_paths]
            if isinstance(tokenizer_names_or_paths, str)
            else tokenizer_names_or_paths
        )

        self.dataset = dataset
        self.load_kwargs = load_kwargs or {}
        self.streaming = streaming
        self.batch_size = batch_size

        self._stats_map = {
            name: StatStorage(
                total_unk_tokens=0,
                overall_sentences=0,
                total_sentences_with_unk=0,
                overall_tokens=0,
                overall_chars=0,
            )
            for name in self._tokenizer_names
        }

    def run(self) -> dict[str, StatStorage]:
        ds = load_dataset(self.dataset, **self.load_kwargs, streaming=self.streaming)

        if isinstance(ds, dict):
            raise ValueError(
                f"You forgot to specify the split of the dataset. Update your load_kwargs to include 'split'. Available splits: {list(ds.keys())}"
            )

        tokenizers = {
            name: AutoTokenizer.from_pretrained(name) for name in self._tokenizer_names
        }

        with tqdm(total=None) as pbar:
            for batch in ds.iter(self.batch_size):
                self.update_stats_on_batch(batch, tokenizers)
                try:
                    first_col = next(iter(batch.values()))
                    pbar.update(len(first_col))
                except Exception:
                    pbar.update(self.batch_size)

        return self._stats_map

    def update_stats_on_batch(self, batch: dict, tokenizers) -> None:
        for sample in batch.values():
            char_len = sum(map(len, sample))

            for name, tokenizer in tokenizers.items():
                input_ids = tokenizer(
                    sample,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                )["input_ids"]

                target = tokenizer.unk_token_id or -1
                stats = self._stats_map[name]

                for tokens in input_ids:
                    count = tokens.count(target)
                    exists = 1 if count > 0 else 0

                    stats.overall_sentences += 1
                    stats.overall_tokens += len(tokens)
                    stats.total_sentences_with_unk += exists
                    stats.total_unk_tokens += count

                stats.overall_chars += char_len

import os
import fasttext
import argparse
import warnings
from typing import Dict, List
from huggingface_hub import hf_hub_download
from datasets import load_dataset

warnings.filterwarnings('ignore')


class LIDFilter:
    def __init__(
        self,
        column_name: str,
        model_path: str,
        lang_tag: str,
        confidence: float,
    ):
        self.column_name = column_name
        self.lang_tag = lang_tag
        self.confidence = confidence
        self.model_path = model_path
        self.model = None

    def __call__(self, example: Dict) -> bool:
        if self.model is None:
            self.model = fasttext.load_model(self.model_path)

        sentence = example[self.column_name]
        correct_tag = f'__label__{self.lang_tag}' if '__label__' not in self.lang_tag else self.lang_tag

        pred_tag, pred_conf = self.model.predict(sentence)

        return pred_tag[0] == correct_tag and pred_conf.item() >= self.confidence


def process_dataset(
    download_data_path: str,
    model_path: str,
    columns: List[str],
    split: str | None,
    lang_tags: List[str],
    confidiences: List[float],
    num_proc: int,
    local_path_save: str | None = None,
    push_to_hub_path: str | None = None
):
    dataset = load_dataset(download_data_path, download_mode='force_redownload')
    if split:
        dataset = dataset[split]

    for column_name, lang_tag, confidence in zip(columns, lang_tags, confidiences):
        lid_filter = LIDFilter(
            column_name=column_name,
            model_path=model_path,
            lang_tag=lang_tag,
            confidence=confidence
        )

        dataset = dataset.filter(
            lid_filter,
            num_proc=num_proc,
            desc=f"Filtering data for {lang_tag.replace('__label__', '')} language tag"
        )

    if local_path_save:
        dataset.save_to_disk(local_path_save)

    if push_to_hub_path:
        dataset.push_to_hub(push_to_hub_path)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Filters a dataset based on language identification.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    model_source_group = parser.add_mutually_exclusive_group()
    model_source_group.add_argument(
        '--local_model_path',
        type=str,
        default=None,
        help='Path to a local fasttext model file.'
    )
    model_source_group.add_argument(
        '--repo_id',
        type=str,
        default=None,
        help='Repository ID on the Hugging Face Hub (e.g., "cis-lmu/glotlid").'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='The model filename in the repository (e.g., "model_v3.bin").\n'
             'Required if --repo_id is specified.'
    )

    parser.add_argument(
        '--download_data_path',
        type=str,
        required=True,
        help='Dataset name from huggingface (e.g. "openai/gsm8k").'
    )

    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='What part of your dataset need to get (e.g. "train", "dev" or "test").'
    )

    parser.add_argument(
        '--columns',
        nargs='+',
        required=True,
        help='Required. One or more columns to filter (e.g., "text" or "text_ru text_en").'
    )
    parser.add_argument(
        '--lang_tags',
        nargs='+',
        required=True,
        help='Required. Language tags corresponding to each column in the same order.'
    )
    parser.add_argument(
        '--confs',
        nargs='+',
        type=float,
        required=True,
        help='Required. Confidence thresholds (0.0 to 1.0) corresponding to each column.'
    )

    parser.add_argument(
        '--num_proc',
        type=str,
        default='1',
        help="Number of processes to use. Can be a positive integer or 'max' to use all available CPUs. (default: 1)"
    )

    parser.add_argument(
        '--local_path_save',
        type=str,
        default=None,
        help='Path to save the filtered dataset locally.'
    )

    parser.add_argument(
        '--push_to_hub_path',
        type=str,
        default=None,
        help='Repository ID on the Hugging Face Hub to push the result to.'
    )
    
    return parser
    

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
       
    if not (len(args.columns) == len(args.lang_tags) == len(args.confs)):
        parser.error(
            "The number of items in --columns, --lang_tags, and --confs must be the same.\n"
            f"Received: {len(args.columns)} columns, {len(args.lang_tags)} tags, {len(args.confs)} thresholds."
        )
        
    num_proc_str = args.num_proc
    if num_proc_str.lower() == 'max':
        args.num_proc = os.cpu_count()
        print(f"Info: --num_proc set to 'max', using {args.num_proc} available CPUs.")
    else:
        try:
            num_proc_val = int(num_proc_str)
            if num_proc_val < 1:
                parser.error(f"Value for --num_proc must be a positive integer, but got {num_proc_val}.")
            args.num_proc = num_proc_val
        except ValueError:
            parser.error(f"Invalid value for --num_proc: '{num_proc_str}'. Must be a positive integer or 'max'.")

    if not args.local_path_save and not args.push_to_hub_path:
        warnings.warn(
            "Warning: Neither --local_path_save nor --push_to_hub_path was specified. The filtered dataset will not be saved.",
            UserWarning
        )

    model_path = get_path(
        local_model_path=args.local_model_path,
        repo_id=args.repo_id,
        filename=args.filename
    )

    process_dataset(
        download_data_path=args.download_data_path,
        model_path=model_path,
        columns=args.columns,
        split=args.split,
        lang_tags=args.lang_tags,
        confidiences=args.confs,
        num_proc=args.num_proc,
        local_path_save=args.local_path_save,
        push_to_hub_path=args.push_to_hub_path
    )

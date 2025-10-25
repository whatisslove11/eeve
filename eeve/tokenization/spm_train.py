import argparse
import os
import shutil

from sentencepiece import SentencePieceTrainer as spm_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--input_files",
        required=True,
        help="Path to a single file or directory containing input files for training",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Prefix name for the output model and vocabulary files",
    )
    parser.add_argument(
        "--model_type",
        default="unigram",
        help="Model type: unigram, bpe, char, or word (default: unigram)",
    )
    parser.add_argument(
        "--vocab_size", default=32000, type=int, help="Vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--num_threads",
        default="all",
        help='Number of threads to use for training. Use "all" for maximum available threads or specify an integer (default: all)',
    )
    parser.add_argument(
        "--train_large",
        default=True,
        help="Enable training for extremely large corpus (default: True)",
    )
    parser.add_argument(
        "--user_defined_symbols",
        default="",
        type=str,
        help="Comma separated list of user defined symbols",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Directory to store the output tokenizer files",
    )
    args = parser.parse_args()

    if args.num_threads == "all":
        num_threads = os.cpu_count()
    else:
        num_threads = int(args.num_threads)

    if os.path.isfile(args.input_files):
        files = args.input_files
    elif os.path.isdir(args.input_files):
        files = [
            os.path.join(args.input_files, f)
            for f in os.listdir(args.input_files)
            if os.path.isfile(os.path.join(args.input_files, f))
        ]

    model_base_name = os.path.basename(args.model_name)

    spm_t.train(
        input=files,
        model_prefix=args.model_name,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        num_threads=num_threads,
        split_digits=True,
        byte_fallback=False,
        train_extremely_large_corpus=args.train_large,
        user_defined_symbols=args.user_defined_symbols,
        character_coverage=1,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        extensions = [".model", ".vocab"]

        for ext in extensions:
            source_file = args.model_name + ext
            if os.path.exists(source_file):
                dest_file = os.path.join(args.output_dir, model_base_name + ext)
                shutil.move(source_file, dest_file)
                print(f"Moved {source_file} to {dest_file}")

import os
import math
import argparse
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def generate_file_with_tokens_freqs(
    tokenizer_model_file: str,
    output_dir: str,
    output_filename: str
) -> None:
    full_path = os.path.join(output_dir, output_filename)

    spm_pr = sp_pb2_model.ModelProto()
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    spm_pr.ParseFromString(tokenizer.serialized_model_proto())
    if spm_pr.trainer_spec.model_type != 1:
        raise ValueError(
            f"This function only supports 'unigram' tokenizer model type, "
            f"but got '{spm_pr.trainer_spec.model_type}'"
        )

    tokens = [p for p in spm_pr.pieces if p.score < 0] # все что 0, то спец токен
    tokens_with_freqs = [f'{token.piece.replace("▁", " ")}\t{int(math.exp(token.score) * 1_000_000)}' for token in tokens]

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tokens_with_freqs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a file with token frequencies from a SentencePiece unigram model'
    )
    parser.add_argument(
        '--tokenizer_model_file',
        type=str,
        required=True,
        help='Path to the SentencePiece tokenizer model file (.model extension)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where the output file will be saved'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        required=True,
        help='Name of the output file that will contain tokens and their frequencies'
    )
    
    args = parser.parse_args()
    
    generate_file_with_tokens_freqs(
        tokenizer_model_file=args.tokenizer_model_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename
    )
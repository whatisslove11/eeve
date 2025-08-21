import os
import argparse
import warnings
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import AutoTokenizer, NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES


# https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L295
SENTENCEPIECE_ID2NAME_MAP = {
    1: "NORMAL",            # normal symbol
    2: "UNKNOWN",           # unknown symbol. only <unk> for now.
    3: "CONTROL",           # control symbols. </s>, <s>, <2ja> etc.
    4: "USER_DEFINED",      # user defined symbols.
    5: "UNUSED",            # this piece is not used.
    6: "BYTE"               # byte symbols. Used when `byte_fallback` is true.
}

TOKEN_TYPES = sp_pb2_model.ModelProto().SentencePiece.Type


def merge_sentencepiece_tokenizers(
    path_or_name_to_old_tokenizer: str,
    path_to_new_tokenizer: str,
    path_to_save: str,
    new_tokenizer_name: str,
    nllb_additional_lang_codes: list[str] | None = None
):
    old_spm = sp_pb2_model.ModelProto()
    new_spm = sp_pb2_model.ModelProto()

    old_tokenizer = AutoTokenizer.from_pretrained(path_or_name_to_old_tokenizer, use_fast=False)
    old_spm.ParseFromString(old_tokenizer.sp_model.serialized_model_proto())

    new_tokenizer = spm.SentencePieceProcessor(model_file=path_to_new_tokenizer)
    new_spm.ParseFromString(new_tokenizer.serialized_model_proto())

    if not old_spm.trainer_spec.byte_fallback and new_spm.trainer_spec.byte_fallback:
        warnings.warn(
            "The ability to add bytes to the original tokenizer without byte fallback support has not been tested yet, due to which bytes from the new tokenizer will not be added. This will be fixed soon.",
            UserWarning,
            stacklevel=2
        )

    existing_tokens = {p.piece for p in old_spm.pieces}
    prev_min_score = min(p.score for p in old_spm.pieces)

    for piece in new_spm.pieces:
        token = piece.piece
        # bytes will be added later
        if piece.type not in [TOKEN_TYPES.NORMAL, TOKEN_TYPES.USER_DEFINED]:
            continue

        if token not in existing_tokens:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = token
            new_p.score = piece.score + prev_min_score if piece.type == TOKEN_TYPES.NORMAL else 0.0 # user defined tokens must have 0.0
            new_p.type = piece.type
            old_spm.pieces.append(new_p)

    old_spm.trainer_spec.vocab_size = len(old_spm.pieces)

    spm_path = os.path.join(path_to_save, "sentencepiece")
    os.makedirs(spm_path, exist_ok=True)

    hf_path = os.path.join(path_to_save, "huggingface")
    os.makedirs(hf_path, exist_ok=True)

    file_name = os.path.basename(new_tokenizer_name)
    if not file_name.lower().endswith('.model'):
        file_name += '.model'

    NEW_SPM_NAME = os.path.join(spm_path, file_name)
    with open(NEW_SPM_NAME, 'wb') as f:
        f.write(old_spm.SerializeToString())

    if nllb_additional_lang_codes and not isinstance(old_tokenizer, NllbTokenizer):
        warnings.warn("The option to add lang_codes is only available for Nllb models")
        nllb_additional_lang_codes = None

    if nllb_additional_lang_codes:
        merged_tokenizer = type(old_tokenizer).from_pretrained(
            old_tokenizer.name_or_path,
            vocab_file=NEW_SPM_NAME,
            additional_special_tokens=sorted(FAIRSEQ_LANGUAGE_CODES + nllb_additional_lang_codes)
        )
    else:
        merged_tokenizer = type(old_tokenizer).from_pretrained(
            old_tokenizer.name_or_path,
            vocab_file=NEW_SPM_NAME
        )
    merged_tokenizer.save_pretrained(hf_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge SentencePiece tokenizers')
    
    parser.add_argument(
        '--old_tokenizer',
        type=str,
        required=True,
        help='Path or name to the old tokenizer'
    )
    parser.add_argument(
        '--new_tokenizer',
        type=str,
        required=True,
        help='Path to the new tokenizer model file'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save the merged tokenizer'
    )
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        required=True,
        help='Name for the new tokenizer'
    )
    parser.add_argument(
        '--lang_codes',
        nargs='*',
        type=str,
        default=None,
        help='Additional language codes ONLY for NLLB models (e.g., --lang_codes eng_Latn rus_Cyrl)'
    )
    
    args = parser.parse_args()
    
    lang_codes = args.lang_codes
    if lang_codes is not None and len(lang_codes) == 0:
        lang_codes = None
    
    merge_sentencepiece_tokenizers(
        path_or_name_to_old_tokenizer=args.old_tokenizer,
        path_to_new_tokenizer=args.new_tokenizer,
        path_to_save=args.save_path,
        new_tokenizer_name=args.tokenizer_name,
        nllb_additional_lang_codes=lang_codes
    )
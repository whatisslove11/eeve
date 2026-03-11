### Currently, only operations with sentencepiece-based tokenizers are supported, tiktoken operations will be added later

## Train SentencePiece tokenizer

Use `eeve/tokenization/spm_train.py`:

```bash
python eeve/tokenization/spm_train.py \
  --input_files data/texts \
  --model_name models/my_spm \
  --vocab_size 64000 \
  --train_large          # default behavior
```

To explicitly disable the `train_extremely_large_corpus` option in SentencePiece:

```bash
python eeve/tokenization/spm_train.py \
  --input_files data/texts \
  --model_name models/my_spm \
  --no-train_large
```

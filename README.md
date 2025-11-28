# eeve
eeve is a small research toolkit around vocabulary expansion for multilingual language models.  
It includes data pipelines for cleaning and deduplicating parallel corpora, tools for training and merging SentencePiece tokenizers and a multi‑stage EEVE-style trainer on top of TRL

## How to Use
1. Install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Clone repo
```
git clone https://github.com/whatisslove11/eeve.git
cd eeve
```
3. Install project dependencies \
Choose the installation mode based on your needs:
```
uv sync                # Base setup: App dependencies + pre-commit
uv sync --group lint   # Add linting tools (Ruff)
uv sync --group test   # Add testing tools (Pytest)
uv sync --all-groups   # Full development: installs everything (Base + Lint + Test)
```

## Project structure
```text
eeve/
├── bench/                 # Small benchmark for NMT task
├── configs/               # YAML configs for data and training runs
├── eeve/
│   ├── callbacks/         # Training callbacks
│   ├── configs/           # Сonfigs for trainers
│   ├── data/              # Data pipelines
│   │   ├── dedup/
│   │   ├── filters/
│   │   └── formatters/
│   ├── inference/         # Infinity server with OpenAI API requests format
│   ├── tokenization/      # Tokenizer training and vocabulary expansion tools
│   ├── trainers/          # EEVE-style multi-stage trainers and training logic
│   ├── utils/             # Shared utilities (logging, stats, datatrove, etc.)
├── examples/              # Example notebooks demonstrating usage
├── tests/                 # Modules tests

# eeve

### How to Use
1. Install uv
```
pip install uv
```
2. Clone repo
```
git clone https://github.com/whatisslove11/eeve.git
cd eeve
```
3. Install project dependencies
Choose the installation mode based on your needs:
- Base (App + pre-commit) — default for running the code:
```
uv sync
```

- + Linting tools (Ruff) — for code style checks:
```
uv sync --group lint
```

- + Testing tools (Pytest) — for running tests:
```
uv sync --group test
```

- Full Development — installs everything (Base + Lint + Test):
```
uv sync --all-groups
```
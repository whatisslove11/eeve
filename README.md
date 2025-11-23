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
3. Install project dependencies \
Choose the installation mode based on your needs:
```
uv sync                # Base setup: App dependencies + pre-commit
uv sync --group lint   # Add linting tools (Ruff)
uv sync --group test   # Add testing tools (Pytest)
uv sync --all-groups   # Full development: installs everything (Base + Lint + Test)
```

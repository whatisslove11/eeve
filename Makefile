.PHONY: test test_wo_int lint

test:
	RUN_INTEGRATION_TESTS=1 uv run pytest -sv ./tests/

test_wo_int:
	uv run pytest -sv ./tests/

lint:
	uv run ruff check eeve tests examples
	uv run ruff format --check eeve tests examples
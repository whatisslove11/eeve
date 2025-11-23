.PHONY: test test_wo_int

test:
	RUN_INTEGRATION_TESTS=1 python -m pytest -sv ./tests/

test_wo_int:
	python -m pytest -sv ./tests/
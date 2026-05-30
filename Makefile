.PHONY: install test lint fmt build publish clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=bibirags --cov-report=term-missing

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

build:
	python -m build

# Publish to TestPyPI first:  make publish-test
# Then to PyPI:               make publish
publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

PY    ?= python3
VENV  ?= .venv

.PHONY: install lint lint-fix test build clean

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

lint-fix:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

test:
	$(VENV)/bin/pytest

# Build sdist + wheel into dist/. Sanity check that pyproject.toml is
# valid and the package builds. Used by CI before publishing.
build:
	$(VENV)/bin/python -m build

clean:
	rm -rf $(VENV) build dist .pytest_cache .ruff_cache *.egg-info src/*.egg-info

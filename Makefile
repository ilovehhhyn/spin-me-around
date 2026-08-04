PYTHON ?= python3

.PHONY: install lint format format-check test check build smoke rank-pilot

install:
	$(PYTHON) -m pip install --editable ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

format-check:
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest

check: lint format-check test

build:
	$(PYTHON) -m build

smoke:
	dnd-sweep --config configs/week1_smoke.yaml

rank-pilot:
	dnd-sweep --config configs/week1_rank_pilot.yaml

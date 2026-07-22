.PHONY: setup data clean lint format typecheck test check all

setup:            ## Install deps + git hooks
	uv sync --all-extras --dev
	uv run pre-commit install

data:             ## Generate the raw dataset
	uv run python src/ingestion/generate_dataset.py

clean:            ## Run the Stage 1 cleaning pipeline
	uv run python src/processing/clean.py

lint:             ## Lint with ruff
	uv run ruff check .

format:           ## Auto-format with ruff
	uv run ruff format .

typecheck:        ## Static type check with mypy
	uv run mypy src

test:             ## Run tests with coverage
	uv run pytest

check: lint typecheck test   ## Run all quality gates (what CI runs)

all: setup data clean check

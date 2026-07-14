.PHONY: install install-finbert dev run pipeline test lint fmt docker

install:            ## Install core (no torch)
	pip install -e ".[dev]"

install-finbert:    ## Install core + FinBERT training/inference deps
	pip install -e ".[dev,finbert]"

run:                ## Serve the FastAPI app
	uvicorn marketsentiment.api.main:app --reload --port 8000

pipeline:           ## Run one pipeline pass (trending symbols by default)
	python -m marketsentiment.scripts.run_pipeline

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests

docker:
	docker compose up --build

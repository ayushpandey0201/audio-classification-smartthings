.PHONY: install test lint train evaluate clean bot

PYTHON := python3

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

train:
	$(PYTHON) -m src.train --config configs/config.yaml --train-csv data/raw/kitpri-v2/metadata/train.csv --val-csv data/raw/kitpri-v2/metadata/val.csv --data-root data/raw/kitpri-v2

evaluate:
	$(PYTHON) -m src.evaluate --config configs/config.yaml --csv data/raw/kitpri-v2/metadata/test.csv --threshold-csv data/raw/kitpri-v2/metadata/val.csv --data-root data/raw/kitpri-v2 --checkpoint results/checkpoints/best.pt

bot:
	$(PYTHON) -m telegram_bot.bot --ckpt results/checkpoints/best.pt

clean:
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf src/models/__pycache__

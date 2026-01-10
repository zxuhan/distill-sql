# Makefile for distill-sql.
#
# Most stages are wrapped scripts under scripts/. The Makefile is the
# orchestration layer; everything is also runnable via `distill-sql <subcmd>`.

PY := uv run python
SCRIPTS := scripts

.PHONY: help install lint format typecheck test test-fast test-slow \
        data baseline teacher train eval report \
        clean-cache clean-artifacts

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (including dev) into .venv via uv
	uv sync --all-extras

lint: ## Run ruff (lint + format check)
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run mypy --strict on src and tests
	uv run mypy

test: test-fast ## Alias for test-fast

test-fast: ## Run fast unit + property tests with coverage gate
	uv run pytest -m "not slow and not integration"

test-slow: ## Run slow tests (training smoke, real-model inference)
	uv run pytest -m "slow or integration" --no-cov

data: ## Download Spider and prepare schema cache
	$(PY) $(SCRIPTS)/01_prepare_spider.py

baseline: ## Eval base Qwen2.5-0.5B-Instruct on Spider dev (no training)
	$(PY) $(SCRIPTS)/04_eval_all.py --config configs/eval_base.yaml

teacher: ## Generate teacher traces (interactive cost confirmation)
	$(PY) $(SCRIPTS)/02_generate_teacher_traces.py --config configs/teacher.yaml

train: ## Train student LoRA (primary config)
	$(PY) $(SCRIPTS)/03_train_student.py --config configs/train_primary.yaml

train-ablation: ## Train student LoRA (ablation config)
	$(PY) $(SCRIPTS)/03_train_student.py --config configs/train_ablation.yaml

eval: ## Run all four evals (base, primary, ablation, teacher)
	$(PY) $(SCRIPTS)/04_eval_all.py --config configs/eval_all.yaml

report: ## Build final results.md and chart from eval JSONs
	$(PY) $(SCRIPTS)/05_make_report.py

clean-cache: ## Wipe artifacts/cache (teacher API cache)
	rm -rf artifacts/cache

clean-artifacts: ## Wipe everything in artifacts/ (models, runs, cache)
	rm -rf artifacts
	mkdir -p artifacts/runs artifacts/cache

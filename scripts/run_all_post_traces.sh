#!/usr/bin/env bash
# Pipeline driver: from a finished traces JSONL through final report.
#
# Run this after `02_generate_teacher_traces.py` has completed. It chains
# the two training runs, the four-way eval, and the report build with
# fail-fast semantics.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
TRACES=${TRACES:-artifacts/traces/spider_train.jsonl}

if [[ ! -s "$TRACES" ]]; then
  echo "ERROR: traces file $TRACES is empty or missing"
  echo "did you run scripts/02_generate_teacher_traces.py?"
  exit 1
fi

n_traces=$(wc -l < "$TRACES" | xargs)
echo "==> $(date +%T) starting post-trace pipeline on $n_traces traces"

echo "==> $(date +%T) train (primary)"
"$PYTHON" scripts/03_train_student.py --config configs/train_primary.yaml

echo "==> $(date +%T) train (ablation: direct-only)"
"$PYTHON" scripts/03_train_student.py --config configs/train_ablation.yaml

echo "==> $(date +%T) full eval matrix"
"$PYTHON" scripts/04_eval_all.py --config configs/eval_all.yaml

echo "==> $(date +%T) building report"
"$PYTHON" scripts/05_make_report.py

echo "==> $(date +%T) DONE."

#!/usr/bin/env bash
# One-command 7B (and optional 14B) cloud run.
#
# Assumes: a fresh Linux box with 1+ A100 (80GB or 40GB), CUDA 12.x drivers,
# uv installed, and the repo cloned to $PWD.
#
# Multi-GPU: auto-detected via nvidia-smi. With 4 GPUs the 7B run uses
# accelerate-launched DDP (4x speedup, ~25 min instead of ~90 min). The
# 14B stretch leg uses 4-bit QLoRA on a single GPU since DDP-with-4-bit
# is fragile under accelerate.
#
# Inputs that must already be present (not in the repo, gitignored):
#   - artifacts/traces/spider_train_trim_1024.jsonl  (scp from your M1)
#   - data/spider/                                    (scp from your M1)
#
# Usage:
#   bash scripts/run_a100.sh           # train + eval 7B only
#   bash scripts/run_a100.sh 14b       # also do the 14B stretch leg
#
# Outputs to scp back to your M1:
#   reports/predictions/distilled_7b.jsonl
#   artifacts/cloud_runs/scaling_7b/adapter/
#   (+ 14b counterparts if you ran the stretch leg)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [1/5] system check"
nvidia-smi || { echo "no GPU detected, aborting"; exit 1; }
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
echo "    found ${NUM_GPUS} GPU(s)"
[[ -f artifacts/traces/spider_train_trim_1024.jsonl ]] \
  || { echo "missing traces; scp artifacts/traces/spider_train_trim_1024.jsonl from your M1"; exit 1; }
[[ -f data/spider/dev.json ]] \
  || { echo "missing Spider data; scp data/spider/ from your M1 (or run scripts/01_prepare_spider.py)"; exit 1; }

echo "==> [2/5] install CUDA deps (one-shot, idempotent)"
uv pip install --quiet \
  "torch>=2.4" \
  "transformers>=4.45" \
  "peft>=0.13" \
  "trl>=0.11" \
  "bitsandbytes>=0.44" \
  "accelerate>=0.34" \
  "datasets>=3.0" \
  "pyyaml" \
  "rich"

echo "==> [3/5] train 7B (bf16 base + LoRA, ${NUM_GPUS}-GPU DDP)"
if [[ "${NUM_GPUS}" -ge 2 ]]; then
  uv run accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" \
    scripts/cloud_train_cuda.py --config configs/train_7b_cuda.yaml
else
  uv run python scripts/cloud_train_cuda.py --config configs/train_7b_cuda.yaml
fi

echo "==> [4/5] eval 7B on Spider dev (single GPU is enough)"
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/cloud_eval_cuda.py \
    --base Qwen/Qwen2.5-7B-Instruct \
    --adapter artifacts/cloud_runs/scaling_7b/adapter \
    --out reports/predictions/distilled_7b.jsonl

if [[ "${1:-}" == "14b" ]]; then
  echo "==> [5a/5] stretch: train 14B (4-bit QLoRA, single GPU)"
  CUDA_VISIBLE_DEVICES=0 \
    uv run python scripts/cloud_train_cuda.py --config configs/train_14b_cuda.yaml

  echo "==> [5b/5] eval 14B"
  CUDA_VISIBLE_DEVICES=0 \
    uv run python scripts/cloud_eval_cuda.py \
      --base Qwen/Qwen2.5-14B-Instruct \
      --adapter artifacts/cloud_runs/scaling_14b/adapter \
      --out reports/predictions/distilled_14b.jsonl \
      --quant 4bit
fi

echo "==> [done] outputs to scp back to your M1:"
echo "    reports/predictions/distilled_7b.jsonl"
[[ "${1:-}" == "14b" ]] && echo "    reports/predictions/distilled_14b.jsonl"
echo
echo "Then on your M1, score + merge with:"
echo "    uv run python scripts/score_jsonl.py \\"
echo "        --predictions reports/predictions/distilled_7b.jsonl \\"
echo "        --name distilled_7b"
echo "    uv run python scripts/05_make_report.py"

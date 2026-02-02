# HF Space demo — deploy walkthrough

A clickable URL where reviewers type a question and watch SQL appear is
the highest-leverage YC pitch artifact this project can ship. This doc
walks you through pushing the Space.

## What you need

- A Hugging Face account ([huggingface.co](https://huggingface.co); free).
- A user access token with `write` scope: settings → access tokens → new
  token. Treat like a password.
- One of:
  - A **PEFT-format 1.5B adapter** for the free CPU tier (recipe below), or
  - The **7B adapter** you trained on RunPod for the paid T4 tier.

## Decision: which model + tier

| | free CPU | T4 small (~$0.40/hr, auto-sleeps) |
|---|---|---|
| Model | distilled 1.5B (~5 GB peak RAM, fp32) | distilled 7B (bf16 + LoRA, ~14 GB VRAM) |
| Cost | $0 | ~$3-6/day with auto-sleep |
| Latency / query | ~5-15 s | ~2-3 s |
| Spider dev exec | 69.2% | 75.0% |
| Adapter you need | PEFT-format 1.5B (mlx-format won't load) | the 7B from `artifacts/cloud_runs/scaling_7b/adapter/` |

The free CPU tier is the right default for a public demo. T4 is worth
it for *fast* responses during a live YC review session.

## Path A — Free CPU, 1.5B (recommended)

The M1-trained 1.5B adapter is in `mlx-lm` format (Apple-Silicon-only),
which transformers + peft can't load. Easiest fix: train a PEFT-format
1.5B on the same RunPod instance you used for the 7B run, ~5 minutes.

### A.1 Train a 1.5B PEFT adapter on the cloud

On the pod (assuming you still have it running, or spin up a fresh
1×A100 — even a single-GPU box is plenty for 1.5B):

```sh
cd /workspace/distill-sql
git pull   # pick up any latest fixes

# Quick config swap: copy the 7B config and rewrite the base + run name.
cat > configs/train_1p5b_cuda.yaml <<'YAML'
base:
  model_id: Qwen/Qwen2.5-1.5B-Instruct
  max_seq_len: 1024
  quant: bf16
traces_path: artifacts/traces/spider_train_trim_1024.jsonl
val_split: 0.05
rank: 16
alpha: 32.0
dropout: 0.05
learning_rate: 5.0e-5
warmup_steps: 50
schedule: cosine
batch_size: 4
grad_accum: 2
grad_checkpoint: false
epochs: 1
eval_every_steps: 200
save_every_steps: 200
seed: 42
include_reasoning_traces: true
output_dir: artifacts/cloud_runs
run_name: scaling_1p5b_peft
YAML

CUDA_VISIBLE_DEVICES=0 uv run python scripts/cloud_train_cuda.py \
    --config configs/train_1p5b_cuda.yaml
```

~5 minutes on 1×A100. Adapter lands at
`artifacts/cloud_runs/scaling_1p5b_peft/adapter/`.

### A.2 Bring the adapter home

```sh
# === On your Mac ===
mkdir -p artifacts/cloud_runs
scp -P 10418 -i ~/.ssh/runpod -r \
    root@<pod-ip>:/workspace/distill-sql/artifacts/cloud_runs/scaling_1p5b_peft \
    artifacts/cloud_runs/

# Copy into the Space directory:
cp -r artifacts/cloud_runs/scaling_1p5b_peft/adapter space/
```

### A.3 Create + push the HF Space

```sh
# Install the HF CLI if you don't have it:
uv tool install huggingface_hub
huggingface-cli login   # paste your write token

# Create the Space (CPU basic tier; free)
huggingface-cli repo create distill-sql --type space --space_sdk gradio
# alternatively use the website: https://huggingface.co/new-space

# Clone the empty space repo
git clone https://huggingface.co/spaces/<your-username>/distill-sql /tmp/hf-space
cd /tmp/hf-space
# Copy our scaffold + adapter
cp -r /Users/guangtumuxixirihan/CS/backend/task-distillation/space/* .
git add -A
git commit -m "first deploy: distilled 1.5B on free CPU"
git push
```

The Space build kicks off automatically. First boot takes ~3-5 minutes
(downloads the base model). Watch logs in the Space dashboard.

## Path B — Paid T4 small, 7B

You already have the 7B adapter from the RunPod run.

```sh
# === On your Mac ===
# 1. scp the 7B adapter home (you may have done this already)
mkdir -p artifacts/cloud_runs/scaling_7b
scp -P 10418 -i ~/.ssh/runpod -r \
    root@<pod-ip>:/workspace/distill-sql/artifacts/cloud_runs/scaling_7b/adapter \
    artifacts/cloud_runs/scaling_7b/

# 2. Copy into the Space directory
cp -r artifacts/cloud_runs/scaling_7b/adapter space/

# 3. Create the Space (web UI: https://huggingface.co/new-space)
#    Pick hardware: "T4 small" ($0.40/hr; will auto-sleep when idle)
#    Pick SDK: Gradio

# 4. Clone, copy, push (same as Path A.3)

# 5. After push, set environment variable in Space settings:
#    BASE_MODEL = Qwen/Qwen2.5-7B-Instruct
```

## Verify the Space is up

Once the Space build completes, your URL is:

    https://huggingface.co/spaces/<your-username>/distill-sql

Click the "concert_singer" example, hit "Generate SQL". You should see
a `SELECT ... FROM singer ...` query appear in 5-10 seconds (CPU) or
2-3 seconds (T4).

## Add to your YC pitch

Update the README's opener to include the live URL:

```md
> **Live demo: huggingface.co/spaces/<your-username>/distill-sql** —
> type a question, watch SQL appear in 5 seconds, no API key.
```

That sentence is the strongest single line in the whole pitch. Lead
with it.

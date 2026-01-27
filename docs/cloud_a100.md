# Cloud A100 extension (7B + 14B scaling-axis points)

The on-disk M1 pipeline tops out at 3B (4-bit base + LoRA + grad checkpoint
saturates 16 GB). To extend the scaling axis to 7B and 14B, we rent an
A100 and run the same recipe on a CUDA stack (`transformers` + `peft` +
`trl`). Hyperparameters are identical to the M1 runs; the only varying
axis is parameter count.

## What you'll need

| | |
|---|---|
| GPU | 1× A100 80GB (preferred) or 40GB. 7B bf16 fits in 40GB; 14B at 4-bit fits in 40GB with grad checkpoint. |
| Disk | ~80 GB free (HF model cache for 14B is ~30 GB). |
| Time | 7B: ~75–90 min. 14B (stretch): ~3–4 hr. |
| Cost | ~$2–4 for 7B, ~$8–15 for 14B at typical A100 spot prices. |

## Provision the box

Pick any provider with on-demand A100s — Lambda, RunPod, Modal, Vast,
Paperspace, etc. SSH in. Install `uv` if not present:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Sync the repo + traces

From your M1:

```sh
rsync -avz --exclude .venv --exclude .pytest_cache \
    --exclude .ruff_cache --exclude .mypy_cache --exclude artifacts/runs \
    ./ user@a100-box:/workspace/distill-sql/
```

Make sure `artifacts/traces/spider_train_trim_1024.jsonl` is included
(it is, by default — only `artifacts/runs/` is excluded).

## Run training + eval

On the box:

```sh
cd /workspace/distill-sql
bash scripts/run_a100.sh        # 7B only (~90 min)
bash scripts/run_a100.sh 14b    # 7B + 14B (~5 hr total)
```

## Rsync results home

From your M1:

```sh
rsync -avz user@a100-box:/workspace/distill-sql/reports/predictions/ \
    reports/predictions/
rsync -avz user@a100-box:/workspace/distill-sql/artifacts/cloud_runs/ \
    artifacts/cloud_runs/
```

## Score and merge into the headline table

```sh
uv run python scripts/score_jsonl.py \
    --predictions reports/predictions/distilled_7b.jsonl \
    --name distilled_7b

# stretch:
uv run python scripts/score_jsonl.py \
    --predictions reports/predictions/distilled_14b.jsonl \
    --name distilled_14b

uv run python scripts/05_make_report.py
```

The headline table in the README is auto-regenerated.

## What success looks like

- 7B should land in the **74–80%** range on Spider dev exec accuracy
  (one cleanly diminishing-returns point past the 3B's 72.6%).
- 14B should land in the **78–84%** range. If it crosses the
  GPT-4o-mini reference line, that's the headline number for the YC
  application: a fully open-source on-device pipeline that matches the
  closed-source teacher.
- If 14B *plateaus* near 7B, that's also a reportable result —
  "scaling saturates around 7B for Spider; the right axis was always
  quantization, not size."

Either outcome is publishable; the diminishing-returns shape is itself
the story.

## Why a separate stack for cloud

`mlx-lm` is Apple-Silicon-only. The M1 trainer and the CUDA trainer
share the **prompt template**, the **trace format**, and the **LoRA
hyperparameters** — which means the scaling-axis claim ("same recipe,
varying parameter count") holds. They differ only in the framework
required by the hardware. Predictions JSONL schema is identical, so the
scoring + report stages are unchanged.

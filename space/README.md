---
title: distill-sql
emoji: 🦦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: On-device text-to-SQL distilled from GPT-4o-mini
---

# distill-sql — live demo

Distillation of GPT-4o-mini text-to-SQL into a small Qwen2.5 student.
Spider-dev benchmark numbers from the upstream training pipeline:

| model | params | exec on Spider dev | size on disk |
|---|---:|---:|---:|
| GPT-4o-mini (closed teacher) | n/a | **80.1%** | n/a |
| distilled 7B (cloud-trained) | 7 B | **75.0%** | 14 GB (bf16) |
| distilled 3B (M1-trained) | 3 B | **72.6%** | 1.7 GB (4-bit) |
| distilled 1.5B (M1-trained) | 1.5 B | **69.2%** | 2.9 GB (bf16) |
| distilled 1.5B q4 (deployment) | 1.5 B | **62.5%** | **847 MB** |
| base 0.5B (no training) | 0.5 B | 33.9% | 1.0 GB |

This Space is the deployed inference layer; full pipeline + training
recipe + scaling-axis study live at
[github.com/zxuhan/distill-sql](https://github.com/zxuhan/distill-sql).

## Configuration

Set on the Space's "Settings → Variables and secrets" page:

| variable | default | what it does |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HF base model id; switch to `Qwen/Qwen2.5-7B-Instruct` on a T4-class Space |
| `ADAPTER_PATH` | `./adapter` | path to the PEFT adapter directory inside the Space repo |
| `MAX_NEW_TOK` | `256` | generation cap |

For the **free CPU tier**: keep the 1.5B base, commit a 1.5B PEFT adapter at `./adapter/`.

For the **T4 small tier** (~$0.40/hr): set `BASE_MODEL=Qwen/Qwen2.5-7B-Instruct`, commit the 7B adapter at `./adapter/`.

"""LoRA training driver around ``mlx_lm.lora``.

mlx-lm exposes a ``lora.run(args)`` entrypoint that takes an argparse-style
``Namespace``. We build that namespace from our Pydantic ``TrainConfig`` and
translate our trace JSONL into mlx-lm's expected chat-format JSONL on the fly.

This module is intentionally light on type hints (mlx has incomplete stubs)
and is excluded from ``mypy --strict`` via ``[[tool.mypy.overrides]]``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from rich.console import Console

from ..data.prompts import (
    SQL_FENCE_CLOSE,
    SQL_FENCE_OPEN,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from .utils import setup_run_logging

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ..config import TrainConfig

console = Console()
log = logging.getLogger("distill_sql.train")


# ---------------------------------------------------------------------------
# Trace -> chat-format dataset conversion
# ---------------------------------------------------------------------------


def _completion_for(sql: str, reasoning: str | None) -> str:
    sql_clean = sql.strip().rstrip(";").strip()
    body = f"{SQL_FENCE_OPEN}\n{sql_clean}\n{SQL_FENCE_CLOSE}"
    if reasoning:
        return f"{reasoning.strip()}\n\n{body}"
    return body


def _trace_to_chat_record(trace: dict) -> dict:
    """Convert one trace dict to a mlx-lm chat-format record."""
    user = build_user_prompt(
        trace["schema_block"],
        trace["question"],
        trace["mode"],
    )
    assistant = _completion_for(trace["sql"], trace.get("reasoning"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def prepare_dataset(
    traces_path: Path,
    out_dir: Path,
    val_split: float = 0.05,
    seed: int = 42,
    include_reasoning: bool = True,
) -> tuple[int, int]:
    """Read traces and write train.jsonl + valid.jsonl in mlx-lm's expected layout.

    Returns ``(n_train, n_valid)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    traces: list[dict] = []
    with traces_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))

    if not include_reasoning:
        traces = [t for t in traces if t.get("mode") != "reasoning"]

    rng = random.Random(seed)
    rng.shuffle(traces)
    n_val = max(1, int(len(traces) * val_split))
    val = traces[:n_val]
    train = traces[n_val:]

    with (out_dir / "train.jsonl").open("w") as f:
        for t in train:
            f.write(json.dumps(_trace_to_chat_record(t), ensure_ascii=False) + "\n")
    with (out_dir / "valid.jsonl").open("w") as f:
        for t in val:
            f.write(json.dumps(_trace_to_chat_record(t), ensure_ascii=False) + "\n")
    return len(train), len(val)


# ---------------------------------------------------------------------------
# Run-time logging callback
# ---------------------------------------------------------------------------


class _RichTrainingCallback:
    """Stream loss numbers to stdout + a per-run JSONL log."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "metrics.jsonl"
        self._fh = self.path.open("w")
        self._t0 = time.time()

    def on_train_loss_report(self, info: dict) -> None:
        info["wall_s"] = round(time.time() - self._t0, 2)
        info["kind"] = "train"
        self._fh.write(json.dumps(info) + "\n")
        self._fh.flush()
        loss = info.get("train_loss")
        step = info.get("iteration")
        if loss is not None and step is not None:
            tps = info.get("tokens_per_second")
            extra = f" tps={tps:.1f}" if isinstance(tps, (int, float)) else ""
            console.log(f"step {step}  train_loss={loss:.4f}{extra}")

    def on_val_loss_report(self, info: dict) -> None:
        info["wall_s"] = round(time.time() - self._t0, 2)
        info["kind"] = "val"
        self._fh.write(json.dumps(info) + "\n")
        self._fh.flush()
        loss = info.get("val_loss")
        step = info.get("iteration")
        if loss is not None:
            console.log(f"[bold]eval[/] step {step}  val_loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Config -> mlx-lm Namespace
# ---------------------------------------------------------------------------


def _build_lora_args(
    cfg: TrainConfig,
    data_dir: Path,
    adapter_dir: Path,
    n_train: int,
) -> SimpleNamespace:
    iters = max(1, (n_train // max(cfg.batch_size, 1)) * cfg.epochs)
    lr_schedule = None
    if cfg.schedule == "cosine":
        # mlx-lm's build_schedule expects a list-of-args style nested config.
        lr_schedule = {
            "name": "cosine_decay",
            "warmup": cfg.warmup_steps,
            "warmup_init": 0.0,
            "arguments": [cfg.learning_rate, max(iters - cfg.warmup_steps, 1), 0.0],
        }
    elif cfg.schedule == "linear":
        lr_schedule = {
            "name": "linear_schedule",
            "warmup": cfg.warmup_steps,
            "arguments": [cfg.learning_rate, 0.0, max(iters - cfg.warmup_steps, 1)],
        }

    args_dict = {
        "model": cfg.base.model_id,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {"adamw": {}},
        "data": str(data_dir),
        "seed": cfg.seed,
        # Apply LoRA to all decoder blocks.
        "num_layers": -1,
        "batch_size": cfg.batch_size,
        "iters": iters,
        "val_batches": 25,
        "learning_rate": cfg.learning_rate,
        "steps_per_report": 25,
        "steps_per_eval": cfg.eval_every_steps,
        "resume_adapter_file": None,
        "adapter_path": str(adapter_dir),
        "save_every": cfg.save_every_steps,
        "test": False,
        "test_batches": 0,
        "max_seq_length": cfg.base.max_seq_len,
        "config": None,
        "grad_checkpoint": cfg.grad_checkpoint,
        "grad_accumulation_steps": cfg.grad_accum,
        "clear_cache_threshold": 0,
        "lr_schedule": lr_schedule,
        "lora_parameters": {
            "rank": cfg.rank,
            "dropout": cfg.dropout,
            "scale": cfg.alpha / cfg.rank,
            "keys": [f"self_attn.{k}" for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
            + [f"mlp.{k}" for k in ("gate_proj", "up_proj", "down_proj")],
        },
        "mask_prompt": True,
        "report_to": None,
        "project_name": None,
        # mlx-lm 0.21+ added these; safe defaults.
        "hf_dataset": False,
    }
    return SimpleNamespace(**args_dict)


# ---------------------------------------------------------------------------
# Top-level training runner
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> Path:
    """Run a full LoRA training job and return the adapter directory.

    We bypass ``mlx_lm.lora.run`` because that helper overwrites any
    caller-passed ``training_callback`` with one built from
    ``args.report_to``. Replicating its load + dataset + ``train_model``
    sequence inline keeps our callback in the loop so the metrics JSONL
    we write actually has rows in it.
    """
    # Lazy imports: mlx is heavy; importing it at module load slows imports
    # in non-training paths (CLI, tests, lints).
    from mlx_lm.lora import load, load_dataset, train_model

    run_dir = cfg.output_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(run_dir)

    log.info("preparing dataset from %s", cfg.traces_path)
    data_dir = run_dir / "data"
    n_train, n_val = prepare_dataset(
        cfg.traces_path,
        data_dir,
        val_split=cfg.val_split,
        seed=cfg.seed,
        include_reasoning=cfg.include_reasoning_traces,
    )
    log.info("dataset: %d train, %d val", n_train, n_val)

    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    args = _build_lora_args(cfg, data_dir, adapter_dir, n_train)

    # Persist the resolved args next to the adapter for reproducibility.
    (run_dir / "args.json").write_text(json.dumps(args.__dict__, indent=2, default=str))

    cb = _RichTrainingCallback(run_dir)
    log.info(
        "starting LoRA training, iters=%d batch_size=%d grad_accum=%d",
        args.iters,
        args.batch_size,
        args.grad_accumulation_steps,
    )

    log.info("loading model %s", cfg.base.model_id)
    model, tokenizer = load(cfg.base.model_id, tokenizer_config={"trust_remote_code": True})
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    log.info("training: %d train batches, %d val batches", len(train_set), len(valid_set))
    train_model(args, model, train_set, valid_set, training_callback=cb)
    log.info("training complete; adapter saved at %s", adapter_dir)

    # Write a manifest the eval script can read.
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "model_id": cfg.base.model_id,
                "adapter_path": str(adapter_dir),
                "rank": cfg.rank,
                "alpha": cfg.alpha,
                "include_reasoning_traces": cfg.include_reasoning_traces,
                "epochs": cfg.epochs,
                "iters": args.iters,
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum,
                "n_train": n_train,
                "n_val": n_val,
            },
            indent=2,
        ),
    )
    return adapter_dir


def stream_jsonl(path: Path) -> Iterable[dict]:
    """Helper for tests: read JSONL into dicts."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

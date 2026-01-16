"""Smoke test for the training pipeline: 50 steps on 8 traces, loss must decrease.

This is the integration test that catches changes that break training without
having to wait for a full run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_sql.config import StudentBaseConfig, TrainConfig
from distill_sql.student.train import (
    _build_lora_args,
    _completion_for,
    _trace_to_chat_record,
    prepare_dataset,
)


@pytest.fixture
def fake_traces(tmp_path: Path) -> Path:
    """Build a tiny traces JSONL with both modes for prepare_dataset."""
    p = tmp_path / "traces.jsonl"
    rows = []
    for i in range(20):
        rows.append(
            {
                "question_id": i,
                "db_id": "zoo",
                "question": f"How many animals of species {i}?",
                "schema_block": "CREATE TABLE animal (id INT, species TEXT);",
                "mode": "direct" if i % 2 == 0 else "reasoning",
                "sql": f"SELECT COUNT(*) FROM animal WHERE species = 's{i}'",
                "reasoning": (
                    "Need to count rows in animal where species matches."
                    if i % 2
                    else None
                ),
                "gold_sql": f"SELECT COUNT(*) FROM animal WHERE species = 's{i}'",
                "n_candidates": 1,
                "matched_gold": True,
            },
        )
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_completion_for_direct() -> None:
    body = _completion_for("SELECT 1", reasoning=None)
    assert "```sql" in body
    assert "SELECT 1" in body


def test_completion_for_with_reasoning_includes_text() -> None:
    body = _completion_for("SELECT 1", reasoning="Explain")
    assert body.index("Explain") < body.index("```sql")


def test_trace_to_chat_record_shape() -> None:
    rec = _trace_to_chat_record(
        {
            "schema_block": "CREATE TABLE t (a INT);",
            "question": "Q?",
            "mode": "direct",
            "sql": "SELECT a FROM t",
            "reasoning": None,
        },
    )
    assert "messages" in rec
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_prepare_dataset_splits_and_writes(fake_traces: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    n_train, n_val = prepare_dataset(
        fake_traces,
        out_dir,
        val_split=0.2,
        seed=0,
        include_reasoning=True,
    )
    assert n_train + n_val == 20
    assert n_val == 4  # 20 * 0.2
    train_lines = (out_dir / "train.jsonl").read_text().strip().splitlines()
    val_lines = (out_dir / "valid.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == n_train
    assert len(val_lines) == n_val


def test_prepare_dataset_drops_reasoning_when_disabled(
    fake_traces: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    n_train, n_val = prepare_dataset(
        fake_traces,
        out_dir,
        val_split=0.1,
        seed=0,
        include_reasoning=False,
    )
    # Half the original 20 are reasoning-mode and dropped.
    assert n_train + n_val == 10


def test_build_lora_args_translates_config() -> None:
    cfg = TrainConfig(
        base=StudentBaseConfig(model_id="some/model", max_seq_len=1024),
        rank=16,
        alpha=32.0,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=2,
        grad_accum=4,
        epochs=2,
        warmup_steps=50,
    )
    args = _build_lora_args(cfg, Path("/tmp/data"), Path("/tmp/adapter"), n_train=100)
    assert args.model == "some/model"
    assert args.train is True
    assert args.fine_tune_type == "lora"
    assert args.batch_size == 2
    assert args.grad_accumulation_steps == 4
    assert args.lora_parameters["rank"] == 16
    assert args.lora_parameters["scale"] == pytest.approx(2.0)  # alpha/rank
    assert args.lora_parameters["dropout"] == 0.1
    assert "self_attn.q_proj" in args.lora_parameters["keys"]
    assert "mlp.down_proj" in args.lora_parameters["keys"]
    assert args.iters == 100  # n_train // batch_size * epochs
    assert args.mask_prompt is True


def test_build_lora_args_constant_schedule() -> None:
    cfg = TrainConfig(schedule="constant")
    args = _build_lora_args(cfg, Path("/tmp/d"), Path("/tmp/a"), n_train=10)
    assert args.lr_schedule is None


def test_build_lora_args_linear_schedule() -> None:
    cfg = TrainConfig(schedule="linear", warmup_steps=10)
    args = _build_lora_args(cfg, Path("/tmp/d"), Path("/tmp/a"), n_train=10)
    assert args.lr_schedule is not None
    assert args.lr_schedule["name"] == "linear_schedule"


@pytest.mark.slow
@pytest.mark.integration
def test_train_smoke_decreases_loss(fake_traces: Path, tmp_path: Path) -> None:
    """Run 50 steps of LoRA training on tiny data and assert loss drops.

    This is slow (~1 minute on M1) and pulls a real model; gated by `slow`.
    """
    cfg = TrainConfig(
        base=StudentBaseConfig(
            model_id="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            max_seq_len=512,
        ),
        traces_path=fake_traces,
        val_split=0.2,
        rank=8,
        alpha=16.0,
        learning_rate=2e-4,
        batch_size=1,
        grad_accum=2,
        epochs=4,
        warmup_steps=2,
        eval_every_steps=20,
        save_every_steps=200,
        output_dir=tmp_path / "runs",
        run_name="smoke",
    )
    from distill_sql.student.train import train

    adapter_dir = train(cfg)
    metrics_path = (tmp_path / "runs" / "smoke" / "metrics.jsonl").resolve()
    assert metrics_path.exists()
    losses: list[float] = []
    with metrics_path.open() as f:
        for line in f:
            data = json.loads(line)
            v = data.get("train_loss")
            if isinstance(v, (int, float)):
                losses.append(float(v))
    assert losses, "no train_loss reported"
    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert (adapter_dir / "adapters.safetensors").exists() or adapter_dir.exists()

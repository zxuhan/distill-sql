"""Pydantic configuration schemas for every stage of the pipeline.

All YAML configs under ``configs/`` deserialize into one of these models.
Defaults live here so a minimal YAML still produces a valid run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    """Base model with strict validation and frozen instances."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SpiderPaths(_StrictModel):
    """Where Spider data lives on disk."""

    root: Path = Path("data/spider")
    """Root of the extracted Spider archive."""

    train_file: str = "train_spider.json"
    dev_file: str = "dev.json"
    tables_file: str = "tables.json"
    db_subdir: str = "database"

    def train_path(self) -> Path:
        return self.root / self.train_file

    def dev_path(self) -> Path:
        return self.root / self.dev_file

    def tables_path(self) -> Path:
        return self.root / self.tables_file

    def db_dir(self) -> Path:
        return self.root / self.db_subdir


class SchemaSerializerConfig(_StrictModel):
    """Knobs for the schema -> string serializer."""

    include_sample_rows: bool = True
    n_sample_rows: int = 3
    cell_truncate: int = 40
    """Max chars per sample-row cell before truncation."""

    max_tokens: int = 1500
    """Soft budget. If schema exceeds this, BM25-rank tables and drop tail."""

    min_tables_kept: int = 2
    """Never rank below this many tables, even if budget says so."""

    tokens_per_char: float = 0.30
    """Rough char->token ratio used as a budget proxy without the tokenizer."""


class TeacherConfig(_StrictModel):
    """Teacher trace generation."""

    model: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.3
    n_samples: int = 3
    """Self-consistency width per question."""

    max_tokens_direct: int = 256
    max_tokens_reasoning: int = 512

    reasoning_share: float = 0.4
    """Fraction of train examples that get reasoning-mode prompts."""

    @field_validator("reasoning_share")
    @classmethod
    def _share_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("reasoning_share must be in [0, 1]")
        return v

    concurrency: int = 12
    request_timeout_s: float = 60.0
    max_retries: int = 5

    cache_dir: Path = Path("artifacts/cache/teacher")
    output_dir: Path = Path("artifacts/traces")

    max_spend_usd: float = 50.0
    """Hard ceiling. Stop the run if cumulative spend would exceed this."""

    # Pricing snapshot in USD per 1M tokens. Used for the cost meter.
    price_input_per_m: float = 0.15
    price_output_per_m: float = 0.60


class TraceFilterConfig(_StrictModel):
    """Post-generation filtering of teacher candidates."""

    require_executes: bool = True
    require_matches_gold: bool = True
    """If True, drop examples with no candidate whose result matches gold."""

    fallback_executes_only: bool = True
    """If require_matches_gold is True and none match, accept any executable.

    Off by default if require_matches_gold is False (no fallback needed).
    """

    require_parse: bool = True
    require_table_grounded: bool = True
    """SQL must reference at least one schema table (cheap hallucination guard)."""

    execution_timeout_s: float = 5.0


class StudentBaseConfig(_StrictModel):
    """Common student-side knobs."""

    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
    """HF model id; mlx-lm pulls + converts on first use."""

    max_seq_len: int = 2048
    chat_template_role_user: str = "user"
    chat_template_role_assistant: str = "assistant"


class TrainConfig(_StrictModel):
    """LoRA training knobs."""

    base: StudentBaseConfig = StudentBaseConfig()

    traces_path: Path = Path("artifacts/traces/spider_train.jsonl")
    val_split: float = 0.05

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    learning_rate: float = 1e-4
    warmup_steps: int = 100
    schedule: Literal["cosine", "linear", "constant"] = "cosine"

    batch_size: int = 1
    grad_accum: int = 8
    epochs: int = 2
    eval_every_steps: int = 250
    save_every_steps: int = 250

    seed: int = 42

    include_reasoning_traces: bool = True
    """When False, train only on direct (SQL-only) traces."""

    output_dir: Path = Path("artifacts/runs")
    run_name: str = "primary"


class EvalRunConfig(_StrictModel):
    """One slot in the eval matrix: which model, what name."""

    name: str
    """Used in result filenames; must be filesystem-safe."""

    kind: Literal["base", "lora", "openai"]
    """``base``: stock mlx-lm model; ``lora``: with adapter; ``openai``: teacher."""

    model_id: str | None = None
    adapter_path: Path | None = None
    openai_model: str | None = None

    temperature: float = 0.0
    max_tokens: int = 384

    use_reasoning_prompt: bool = False
    """If True, prompt asks student/teacher to reason first; output is post-stripped."""

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(f"name must be filesystem-safe: {v!r}")
        return v


class EvalConfig(_StrictModel):
    """Full eval-matrix config."""

    runs: tuple[EvalRunConfig, ...]
    spider: SpiderPaths = SpiderPaths()
    schema_serializer: SchemaSerializerConfig = SchemaSerializerConfig()
    output_dir: Path = Path("reports/predictions")
    evaluator_dir: Path = Path("third_party/test-suite-sql-eval")
    limit: int | None = None
    """If set, evaluate only the first N dev examples (for debugging)."""


def load_yaml_config(path: Path, schema: type[BaseModel]) -> BaseModel:
    """Load a YAML file and validate it against the given Pydantic schema.

    Wraps yaml.safe_load + ``schema.model_validate`` for a single, traced call site.
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    return schema.model_validate(raw)

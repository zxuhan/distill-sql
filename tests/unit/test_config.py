"""Tests for ``config.py`` Pydantic schemas + YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.config import (
    EvalConfig,
    EvalRunConfig,
    SchemaSerializerConfig,
    SpiderPaths,
    TeacherConfig,
    TraceFilterConfig,
    TrainConfig,
    load_yaml_config,
)


def test_spider_paths_join() -> None:
    paths = SpiderPaths(root=Path("/x/y"))
    assert paths.train_path() == Path("/x/y/train_spider.json")
    assert paths.dev_path() == Path("/x/y/dev.json")
    assert paths.tables_path() == Path("/x/y/tables.json")
    assert paths.db_dir() == Path("/x/y/database")


def test_teacher_config_rejects_bad_share() -> None:
    with pytest.raises(ValueError, match="reasoning_share"):
        TeacherConfig(reasoning_share=1.5)


def test_eval_run_config_rejects_unsafe_name() -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        EvalRunConfig(name="bad name!", kind="base")


def test_eval_run_config_accepts_valid_chars() -> None:
    cfg = EvalRunConfig(name="distilled-rank_16", kind="base")
    assert cfg.name == "distilled-rank_16"


def test_load_yaml_config_validates(tmp_path: Path) -> None:
    p = tmp_path / "tc.yaml"
    p.write_text("temperature: 0.5\nn_samples: 2\nreasoning_share: 0.25\n")
    cfg = load_yaml_config(p, TeacherConfig)
    assert isinstance(cfg, TeacherConfig)
    assert cfg.temperature == 0.5
    assert cfg.n_samples == 2


def test_load_yaml_config_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    cfg = load_yaml_config(p, TraceFilterConfig)
    assert isinstance(cfg, TraceFilterConfig)


def test_load_yaml_config_rejects_extra_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not_a_real_field: 42\n")
    with pytest.raises(ValueError, match="extra"):
        load_yaml_config(p, TraceFilterConfig)


def test_eval_config_loads_with_runs(tmp_path: Path) -> None:
    p = tmp_path / "ec.yaml"
    p.write_text(
        """
runs:
  - name: a
    kind: base
    model_id: foo/bar
    temperature: 0.0
    max_tokens: 64
""",
    )
    cfg = load_yaml_config(p, EvalConfig)
    assert isinstance(cfg, EvalConfig)
    assert cfg.runs[0].name == "a"


def test_train_config_defaults_target_modules() -> None:
    cfg = TrainConfig()
    assert "q_proj" in cfg.target_modules
    assert "down_proj" in cfg.target_modules


def test_schema_serializer_config_defaults() -> None:
    cfg = SchemaSerializerConfig()
    assert cfg.include_sample_rows
    assert cfg.n_sample_rows == 3
    assert cfg.max_tokens == 1500

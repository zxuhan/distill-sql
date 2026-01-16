"""Tests for ``eval/compare.py`` (table rendering + JSON dump)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from distill_sql.eval.compare import (
    RunResult,
    render_markdown_table,
    write_results_json,
)
from distill_sql.eval.metrics import MetricsSummary

if TYPE_CHECKING:
    from pathlib import Path


def _summary(exec_acc: float = 0.5) -> MetricsSummary:
    return MetricsSummary(
        n=10,
        exec_accuracy=exec_acc,
        exact_match_accuracy=exec_acc * 0.7,
        by_difficulty={
            "easy": {"n": 4.0, "exec_accuracy": exec_acc + 0.1, "exact_match_accuracy": 0.4},
            "medium": {"n": 3.0, "exec_accuracy": exec_acc, "exact_match_accuracy": 0.3},
            "hard": {"n": 2.0, "exec_accuracy": exec_acc - 0.1, "exact_match_accuracy": 0.2},
            "extra": {"n": 1.0, "exec_accuracy": exec_acc - 0.2, "exact_match_accuracy": 0.1},
        },
        failure_breakdown={"ok": 5, "wrong-result": 3, "parse": 2},
    )


def test_render_markdown_sorts_by_exec_accuracy() -> None:
    a = RunResult(name="a", summary=_summary(0.3))
    b = RunResult(name="b", summary=_summary(0.7))
    md = render_markdown_table([b, a])
    # The lower-accuracy row should appear before the higher one in output.
    a_idx = md.index("| a |")
    b_idx = md.index("| b |")
    assert a_idx < b_idx


def test_render_markdown_includes_difficulty_columns() -> None:
    md = render_markdown_table([RunResult(name="m", summary=_summary(0.5))])
    for col in ("easy", "medium", "hard", "extra", "exec_acc", "exact_match"):
        assert col in md


def test_write_results_json_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    runs = [
        RunResult(name="m", summary=_summary(0.5), predictions_path=tmp_path / "m.jsonl"),
        RunResult(name="n", summary=_summary(0.6)),
    ]
    write_results_json(runs, out)
    data = json.loads(out.read_text())
    assert "m" in data and "n" in data
    assert data["m"]["summary"]["exec_accuracy"] == 0.5
    assert data["m"]["predictions_path"].endswith("m.jsonl")
    assert data["n"]["predictions_path"] is None

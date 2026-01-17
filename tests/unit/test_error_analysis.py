"""Tests for ``eval/error_analysis.py``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from distill_sql.eval.error_analysis import (
    CasePair,
    _category,
    find_disagreements,
    load_predictions,
    render_error_analysis_md,
    summarize_categories,
)

if TYPE_CHECKING:
    from pathlib import Path


def _row(
    qid: int, *, exec_match: bool, gold: str, pred: str, failure: str = "ok",
) -> dict[str, object]:
    return {
        "question_id": qid,
        "db_id": "zoo",
        "difficulty": "easy",
        "gold_sql": gold,
        "pred_sql": pred,
        "exec_match": exec_match,
        "exact_set_match": False,
        "failure_mode": failure if not exec_match else "ok",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_predictions(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    _write(p, [_row(0, exec_match=True, gold="g", pred="p")])
    rows = load_predictions(p)
    assert len(rows) == 1
    assert rows[0]["question_id"] == 0


def test_category_buckets() -> None:
    assert _category("g", "", "empty") == "empty-output"
    assert _category("g", "p", "parse") == "parse-error"
    assert _category("g", "p", "execution") == "schema-mismatch"
    assert (
        _category(
            "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id",
            "SELECT a FROM t1",
            "wrong-result",
        )
        == "missing-join"
    )
    assert (
        _category(
            "SELECT a FROM t1",
            "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id",
            "wrong-result",
        )
        == "spurious-join"
    )
    assert (
        _category(
            "SELECT COUNT(*) FROM t",
            "SELECT * FROM t",
            "wrong-result",
        )
        == "aggregation-mismatch"
    )
    assert (
        _category(
            "SELECT a FROM t WHERE x = 1",
            "SELECT a FROM t",
            "wrong-result",
        )
        == "missing-filter"
    )
    assert (
        _category(
            "SELECT a FROM t",
            "SELECT a FROM t WHERE x = 1",
            "wrong-result",
        )
        == "spurious-filter"
    )
    assert (
        _category(
            "SELECT a FROM t ORDER BY a",
            "SELECT a FROM t",
            "wrong-result",
        )
        == "missing-order"
    )
    assert (
        _category(
            "SELECT a FROM t GROUP BY a",
            "SELECT a FROM t",
            "wrong-result",
        )
        == "missing-group-by"
    )
    assert (
        _category(
            "SELECT DISTINCT a FROM t",
            "SELECT a FROM t",
            "wrong-result",
        )
        == "missing-distinct"
    )
    assert _category("SELECT 1", "SELECT 2", "wrong-result") == "other"


def test_find_disagreements(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(
        a,
        [
            _row(
                0,
                exec_match=False,
                gold="SELECT a FROM t WHERE x=1",
                pred="SELECT a FROM t",
                failure="wrong-result",
            ),
            _row(1, exec_match=True, gold="SELECT 1", pred="SELECT 1"),
            _row(2, exec_match=False, gold="g", pred="", failure="empty"),
        ],
    )
    _write(
        b,
        [
            _row(
                0,
                exec_match=True,
                gold="SELECT a FROM t WHERE x=1",
                pred="SELECT a FROM t WHERE x=1",
            ),
            _row(1, exec_match=True, gold="SELECT 1", pred="SELECT 1"),
            _row(2, exec_match=False, gold="g", pred="x"),
        ],
    )
    out = find_disagreements(a, b)
    qids = {c.question_id for c in out}
    assert qids == {0}  # only qid 0: a fails, b succeeds


def test_find_disagreements_attaches_questions(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    examples = tmp_path / "ex.json"
    _write(a, [_row(0, exec_match=False, gold="g", pred="p", failure="wrong-result")])
    _write(b, [_row(0, exec_match=True, gold="g", pred="g")])
    examples.write_text(json.dumps([{"question": "How many?", "db_id": "zoo", "query": "g"}]))
    out = find_disagreements(a, b, examples_question_path=examples)
    assert out[0].question == "How many?"


def test_summarize_categories() -> None:
    cases = [
        CasePair(0, "d", "easy", "q", "g", "p", "wrong-result", "p", "missing-join"),
        CasePair(1, "d", "easy", "q", "g", "p", "wrong-result", "p", "missing-join"),
        CasePair(2, "d", "hard", "q", "g", "p", "parse", "p", "parse-error"),
    ]
    assert summarize_categories(cases) == {"missing-join": 2, "parse-error": 1}


def test_render_error_analysis_md_with_examples() -> None:
    cases = [
        CasePair(0, "zoo", "easy", "q?", "SELECT 1", "", "empty", "SELECT 1", "empty-output"),
    ]
    md = render_error_analysis_md(cases)
    assert "empty-output" in md
    assert "SELECT 1" in md
    assert "Q:" in md


def test_render_error_analysis_md_handles_empty_input() -> None:
    md = render_error_analysis_md([])
    assert "No disagreements" in md

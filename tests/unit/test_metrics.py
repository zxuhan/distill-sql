"""Tests for ``eval/metrics.py``."""

from __future__ import annotations

import pytest

from distill_sql.eval.metrics import (
    PerExampleResult,
    classify_failure,
    summarize,
)


def _r(
    *,
    qid: int,
    diff: str,
    exec_match: bool,
    exact: bool = False,
    fail: str = "ok",
) -> PerExampleResult:
    return PerExampleResult(
        question_id=qid,
        db_id="db",
        difficulty=diff,  # type: ignore[arg-type]
        gold_sql="g",
        pred_sql="p",
        exec_match=exec_match,
        exact_set_match=exact,
        failure_mode=fail,  # type: ignore[arg-type]
    )


def test_summarize_empty_inputs_returns_zeros() -> None:
    s = summarize([])
    assert s.n == 0
    assert s.exec_accuracy == 0.0
    assert s.by_difficulty == {}


def test_summarize_aggregates_overall_and_per_difficulty() -> None:
    items = [
        _r(qid=0, diff="easy", exec_match=True, exact=True),
        _r(qid=1, diff="easy", exec_match=False, fail="wrong-result"),
        _r(qid=2, diff="hard", exec_match=True, exact=False),
        _r(qid=3, diff="hard", exec_match=False, fail="parse"),
    ]
    s = summarize(items)
    assert s.n == 4
    assert s.exec_accuracy == 0.5
    assert s.exact_match_accuracy == 0.25
    assert s.by_difficulty["easy"]["n"] == 2.0
    assert s.by_difficulty["easy"]["exec_accuracy"] == 0.5
    assert s.by_difficulty["hard"]["exec_accuracy"] == 0.5


def test_summarize_failure_breakdown_counts() -> None:
    items = [
        _r(qid=0, diff="easy", exec_match=True),
        _r(qid=1, diff="easy", exec_match=False, fail="parse"),
        _r(qid=2, diff="easy", exec_match=False, fail="parse"),
        _r(qid=3, diff="easy", exec_match=False, fail="empty"),
    ]
    s = summarize(items)
    assert s.failure_breakdown["parse"] == 2
    assert s.failure_breakdown["ok"] == 1
    assert s.failure_breakdown["empty"] == 1


@pytest.mark.parametrize(
    ("pred", "parses", "executes", "exec_match", "expected"),
    [
        ("SELECT 1", True, True, True, "ok"),
        ("", False, False, False, "empty"),
        ("   ", False, False, False, "empty"),
        ("SELECT $$$", False, False, False, "parse"),
        ("SELECT * FROM no_such", True, False, False, "execution"),
        ("SELECT 1", True, True, False, "wrong-result"),
    ],
)
def test_classify_failure_buckets(
    pred: str,
    parses: bool,  # noqa: FBT001 — parametrized arg, not API
    executes: bool,  # noqa: FBT001
    exec_match: bool,  # noqa: FBT001
    expected: str,
) -> None:
    out = classify_failure(pred, parses=parses, executes=executes, exec_match=exec_match)
    assert out == expected


def test_summary_as_dict_round_trips() -> None:
    s = summarize([_r(qid=0, diff="easy", exec_match=True)])
    d = s.as_dict()
    assert d["n"] == 1
    assert d["exec_accuracy"] == 1.0
    assert "by_difficulty" in d

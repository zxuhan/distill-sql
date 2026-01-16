"""Tests focused on output-format / writer paths in eval/runner.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from distill_sql.eval.metrics import PerExampleResult
from distill_sql.eval.runner import (
    OfficialResult,
    Prediction,
    Predictor,
    _EmptyPredictor,
    _GoldPredictor,
    parse_official_output,
    write_predictions_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_predictor_protocol_is_subclassable() -> None:
    class MyPred(Predictor):
        def name(self) -> str:
            return "mine"

        def predict(self, example, schema):  # type: ignore[no-untyped-def]
            return "SELECT 1"

    p = MyPred()
    assert p.name() == "mine"


def test_empty_predictor_returns_empty_string() -> None:
    p = _EmptyPredictor()
    assert p.name() == "empty"
    assert p.predict(None, None) == ""  # type: ignore[arg-type]


def test_gold_predictor_returns_query() -> None:
    from distill_sql.data.spider import SpiderExample

    p = _GoldPredictor()
    ex = SpiderExample(db_id="d", question="q", query="SELECT 7", question_id=0)
    assert p.name() == "gold"
    assert p.predict(ex, None) == "SELECT 7"  # type: ignore[arg-type]


def test_parse_official_output_full_table() -> None:
    out = (
        "execution           0.500               0.250               0.200               0.000               0.300\n"
        "exact match         0.400               0.200               0.100               0.000               0.250\n"
    )
    res = parse_official_output(out)
    assert res.exec_easy == 0.5
    assert res.exec_all == 0.3
    assert res.em_all == 0.25


def test_official_result_is_frozen() -> None:
    r = OfficialResult(*[0.1] * 10)
    try:
        r.exec_all = 0.5  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        # frozen=True dataclass should raise; accept FrozenInstanceError too.
        raise AssertionError("OfficialResult should be frozen")


def test_write_predictions_jsonl_includes_outcome(tmp_path: Path) -> None:
    pred = Prediction(question_id=0, db_id="d", predicted_sql="SELECT 1")
    per = PerExampleResult(
        question_id=0,
        db_id="d",
        difficulty="easy",
        gold_sql="SELECT 1",
        pred_sql="SELECT 1",
        exec_match=True,
        exact_set_match=True,
        failure_mode="ok",
    )
    out = tmp_path / "preds.jsonl"
    write_predictions_jsonl([pred], [per], out)
    rows = [json.loads(line) for line in out.open()]
    assert rows[0]["question_id"] == 0
    assert rows[0]["pred_sql"] == "SELECT 1"
    assert rows[0]["failure_mode"] == "ok"

"""Tests for the in-process eval path in ``eval/runner.py``.

The official-evaluator path is covered in tests/integration/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.data.spider import load_examples
from distill_sql.eval.runner import (
    Prediction,
    _hardness,
    _multisets_equal,
    _norm_row,
    evaluate_predictions,
    write_gold_file,
    write_pred_file,
)


def test_hardness_brackets() -> None:
    assert _hardness("SELECT * FROM t") == "easy"
    assert _hardness("SELECT * FROM t WHERE x = 1") == "easy"
    assert _hardness("SELECT * FROM t WHERE x = 1 ORDER BY y") == "medium"
    assert (
        _hardness(
            "SELECT t1.x FROM t1 JOIN t2 ON t1.id = t2.id "
            "WHERE x > 0 GROUP BY t1.x HAVING COUNT(*) > 1 ORDER BY t1.x LIMIT 10",
        )
        == "extra"
    )


def test_multisets_equal_handles_nulls_and_ordering() -> None:
    a = [(1, "tiger"), (2, "lion"), (3, None)]
    b = [(3, None), (2, "lion"), (1, "tiger")]
    assert _multisets_equal(a, b)
    assert not _multisets_equal(a, a + a)


def test_norm_row_stringifies_none() -> None:
    assert _norm_row((1, None, "x")) == ("1", "", "x")


def test_evaluate_predictions_perfect_on_gold(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    preds = [
        Prediction(question_id=e.question_id or 0, db_id=e.db_id, predicted_sql=e.query)
        for e in examples
    ]
    summary, per = evaluate_predictions(preds, examples, spider_mini_root / "database")
    assert summary.exec_accuracy == 1.0
    assert all(r.exec_match for r in per)


def test_evaluate_predictions_empty_string_scores_zero(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    preds = [
        Prediction(question_id=e.question_id or 0, db_id=e.db_id, predicted_sql="")
        for e in examples
    ]
    summary, per = evaluate_predictions(preds, examples, spider_mini_root / "database")
    assert summary.exec_accuracy == 0.0
    assert all(r.failure_mode == "empty" for r in per)


def test_evaluate_predictions_classifies_failures(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    preds = [
        Prediction(question_id=0, db_id="zoo", predicted_sql="SELECT *** FROM ###"),
        Prediction(
            question_id=1,
            db_id="zoo",
            predicted_sql="SELECT * FROM no_such_table",
        ),
        Prediction(
            question_id=2,
            db_id="zoo",
            predicted_sql="SELECT 'wrong'",
        ),
    ]
    _summary, per = evaluate_predictions(preds, examples, spider_mini_root / "database")
    by_qid = {r.question_id: r for r in per}
    assert by_qid[0].failure_mode == "parse"
    assert by_qid[1].failure_mode == "execution"
    assert by_qid[2].failure_mode == "wrong-result"


def test_evaluate_predictions_raises_on_unknown_qid(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    preds = [Prediction(question_id=999, db_id="zoo", predicted_sql="SELECT 1")]
    with pytest.raises(KeyError):
        evaluate_predictions(preds, examples, spider_mini_root / "database")


def test_write_files_match_evaluator_format(
    tmp_path: Path,
    spider_mini_root: Path,
) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    preds = [
        Prediction(question_id=e.question_id or 0, db_id=e.db_id, predicted_sql=e.query)
        for e in examples
    ]
    pred_file = tmp_path / "pred.txt"
    gold_file = tmp_path / "gold.txt"
    write_pred_file(preds, pred_file)
    write_gold_file(examples, gold_file)
    assert len(pred_file.read_text().splitlines()) == len(examples)
    for line in gold_file.read_text().splitlines():
        assert "\t" in line  # SQL\tdb_id


def test_write_pred_file_replaces_blank_with_placeholder(tmp_path: Path) -> None:
    preds = [Prediction(question_id=0, db_id="zoo", predicted_sql="")]
    pred_file = tmp_path / "pred.txt"
    write_pred_file(preds, pred_file)
    line = pred_file.read_text().strip()
    assert line == "SELECT 1"

"""Sanity-check the vendored Spider evaluator on our tiny fixture.

The point is to confirm the subprocess wrapper, gold/pred file format, and
output-parsing regex all line up. We don't trust the headline numbers from
this test (the fixture is too small), but a 100% pass rate when predictions
match gold is a hard floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.data.spider import load_examples
from distill_sql.eval.runner import (
    Prediction,
    parse_official_output,
    run_official_evaluator,
)


@pytest.mark.integration
def test_gold_predictions_score_perfect(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    preds = [
        Prediction(question_id=e.question_id or 0, db_id=e.db_id, predicted_sql=e.query)
        for e in examples
    ]
    evaluator_dir = Path(__file__).parents[2] / "third_party" / "test-suite-sql-eval"
    result = run_official_evaluator(
        preds,
        examples,
        db_dir=spider_mini_root / "database",
        tables_json=spider_mini_root / "tables.json",
        evaluator_dir=evaluator_dir,
        etype="all",
    )
    assert result.exec_all == pytest.approx(1.0, abs=1e-6)


def test_parse_output_grammar() -> None:
    sample = (
        "                    easy                medium              hard                extra               all\n"
        "count               2                   1                   0                   0                   3\n"
        "=====================   EXECUTION ACCURACY     =====================\n"
        "execution           1.000               0.500               0.000               0.000               0.833\n"
        "====================== EXACT MATCHING ACCURACY =====================\n"
        "exact match         1.000               0.500               0.000               0.000               0.833\n"
        "----- some trailing partial-match noise the wrapper ignores -----\n"
    )
    res = parse_official_output(sample)
    assert res.exec_all == pytest.approx(0.833)
    assert res.exec_easy == pytest.approx(1.0)
    assert res.exec_medium == pytest.approx(0.5)
    assert res.em_all == pytest.approx(0.833)


def test_parse_output_raises_on_missing_rows() -> None:
    with pytest.raises(ValueError, match="execution"):
        parse_official_output("only count noise here, no result rows")

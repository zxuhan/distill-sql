"""Evaluation runner that wraps the official Spider test-suite evaluator.

The official evaluator is a Python script we vendor under
``third_party/test-suite-sql-eval/``. It expects two text files (one SQL per
line), the database directory, and ``tables.json``. It writes its output to
stdout, which we parse with a small regex grammar — this is intentionally
narrow because the parse target is the printed table at the end of the run,
not arbitrary log lines.

We add an in-process *fast* path that uses the ``sql/validate.py`` execution
check to give us per-example failure modes (parse / execution / wrong-result),
since the official evaluator only reports aggregate numbers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .._compat import readable_path
from ..sql.normalize import parses
from ..sql.validate import check_executes
from .metrics import (
    Difficulty,
    MetricsSummary,
    PerExampleResult,
    classify_failure,
    summarize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..data.spider import SpiderExample, SpiderSchema

# Match a numeric line like:
#   execution         0.471          0.612          0.305          0.211          0.418
# The order of difficulty buckets is fixed by the evaluator: easy, medium,
# hard, extra, all.
_RESULT_LINE = re.compile(
    r"^\s*(execution|exact match)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class OfficialResult:
    """Aggregate numbers reported by the official evaluator."""

    exec_easy: float
    exec_medium: float
    exec_hard: float
    exec_extra: float
    exec_all: float
    em_easy: float
    em_medium: float
    em_hard: float
    em_extra: float
    em_all: float


def parse_official_output(stdout: str) -> OfficialResult:
    """Parse the printed numbers from a vendored evaluator run.

    Raises ``ValueError`` if either expected line is missing.
    """
    rows = {
        m.group(1): tuple(float(x) for x in m.groups()[1:]) for m in _RESULT_LINE.finditer(stdout)
    }
    if "execution" not in rows or "exact match" not in rows:
        raise ValueError(
            "evaluator output did not contain expected 'execution' and "
            "'exact match' rows; got:\n" + stdout,
        )
    e = rows["execution"]
    m = rows["exact match"]
    return OfficialResult(
        exec_easy=e[0],
        exec_medium=e[1],
        exec_hard=e[2],
        exec_extra=e[3],
        exec_all=e[4],
        em_easy=m[0],
        em_medium=m[1],
        em_hard=m[2],
        em_extra=m[3],
        em_all=m[4],
    )


@dataclass(frozen=True)
class Prediction:
    """A single (question_id, db_id, predicted_sql) row."""

    question_id: int
    db_id: str
    predicted_sql: str


# ---------------------------------------------------------------------------
# Difficulty classification
#
# Spider's "hardness" function is short and pure; we vendor a simplified copy
# so that per-example difficulty labels match the official evaluator within
# the same code path that builds PerExampleResult.
# ---------------------------------------------------------------------------

_HARDNESS_KEYWORDS = {
    "where": 1,
    "group": 1,
    "order": 1,
    "having": 1,
    "limit": 1,
    "join": 1,
    "or ": 1,
    "and ": 1,
    "intersect": 1,
    "union": 1,
    "except": 1,
    "distinct": 1,
}


def _hardness(sql: str) -> Difficulty:
    """Cheap heuristic mimicking Spider's hardness rubric.

    Counts joins, aggregates, set ops, and nested-ness; clusters into the same
    {easy, medium, hard, extra} buckets as ``process_sql.py`` in Spider's repo.
    """
    s = sql.lower()
    score = 0
    for kw, w in _HARDNESS_KEYWORDS.items():
        score += s.count(kw) * w
    if "select" in s.replace("select", "", 1):  # nested SELECT
        score += 2
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    if score <= 5:
        return "hard"
    return "extra"


# ---------------------------------------------------------------------------
# Predictor abstraction
# ---------------------------------------------------------------------------


class Predictor:
    """Anything that turns (example, schema) into a SQL string."""

    def name(self) -> str:  # pragma: no cover — abstract
        raise NotImplementedError

    def predict(
        self,
        example: SpiderExample,
        schema: SpiderSchema,
    ) -> str:  # pragma: no cover — abstract
        raise NotImplementedError


class _EmptyPredictor(Predictor):
    """Sanity test: emits empty strings. Useful for confirming eval plumbing."""

    def name(self) -> str:
        return "empty"

    def predict(self, example: SpiderExample, schema: SpiderSchema) -> str:
        del example, schema
        return ""


class _GoldPredictor(Predictor):
    """Sanity test: returns the gold SQL itself. Should hit ~100% exec."""

    def name(self) -> str:
        return "gold"

    def predict(self, example: SpiderExample, schema: SpiderSchema) -> str:
        del schema
        return example.query


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: Sequence[Prediction],
    examples: Sequence[SpiderExample],
    db_dir: Path,
) -> tuple[MetricsSummary, list[PerExampleResult]]:
    """In-process evaluation that produces per-example failure modes.

    This is the fast path used for error analysis. It runs both the
    prediction and the gold SQL against the DB, compares result sets as
    multisets of tuples, and classifies failures.

    The official evaluator is the source of truth for the headline numbers
    via ``run_official_evaluator``; this in-process function should agree
    with it on execution accuracy to within rounding.
    """
    by_id = {e.question_id: e for e in examples}
    if any(e.question_id is None for e in examples):
        raise ValueError("examples must have non-None question_id")

    per: list[PerExampleResult] = []
    for pred in predictions:
        ex = by_id.get(pred.question_id)
        if ex is None:
            raise KeyError(f"no example for question_id {pred.question_id}")

        diff = _hardness(ex.query)
        pred_sql = (pred.predicted_sql or "").strip().rstrip(";").strip()

        gold_rows = _safe_execute(db_dir / ex.db_id / f"{ex.db_id}.sqlite", ex.query)
        pred_parses = parses(pred_sql) if pred_sql else False
        pred_exec = (
            check_executes(
                pred_sql,
                db_dir / ex.db_id / f"{ex.db_id}.sqlite",
            ).ok
            if pred_sql
            else False
        )
        pred_rows = (
            _safe_execute(db_dir / ex.db_id / f"{ex.db_id}.sqlite", pred_sql)
            if pred_sql and pred_exec
            else None
        )

        exec_match = (
            gold_rows is not None
            and pred_rows is not None
            and _multisets_equal(gold_rows, pred_rows)
        )
        exact_match = pred_sql.lower() == ex.query.strip().rstrip(";").strip().lower()

        per.append(
            PerExampleResult(
                question_id=ex.question_id or 0,
                db_id=ex.db_id,
                difficulty=diff,
                gold_sql=ex.query,
                pred_sql=pred_sql,
                exec_match=exec_match,
                exact_set_match=exact_match,
                failure_mode=classify_failure(
                    pred_sql,
                    parses=pred_parses,
                    executes=pred_exec,
                    exec_match=exec_match,
                ),
            ),
        )
    return summarize(per), per


def _safe_execute(db_path: Path, sql: str) -> list[tuple[object, ...]] | None:
    """Run a query and return result rows, or None if it errors."""
    if not sql.strip():
        return None
    if not db_path.exists():
        return None
    import sqlite3

    try:
        with sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=5.0,
            isolation_level=None,
        ) as conn:
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
    except sqlite3.Error:
        return None


def _multisets_equal(
    a: list[tuple[object, ...]],
    b: list[tuple[object, ...]],
) -> bool:
    """Compare two result sets as bags of tuples.

    This is a stricter notion than the official evaluator's "set match" (which
    ignores column order), so we only use it for the in-process failure-mode
    bucketing — the headline number comes from the official tool.
    """
    if len(a) != len(b):
        return False
    return sorted(map(_norm_row, a)) == sorted(map(_norm_row, b))


def _norm_row(row: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if v is None else str(v) for v in row)


# ---------------------------------------------------------------------------
# Official evaluator subprocess wrapper
# ---------------------------------------------------------------------------


def write_pred_file(predictions: Sequence[Prediction], path: Path) -> None:
    """Serialize predictions in the format the evaluator expects (one SQL/line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # The official evaluator pairs prediction lines with gold lines positionally,
    # so the order must match the gold dev.json order. Caller is responsible.
    lines: list[str] = []
    for p in predictions:
        sql = (p.predicted_sql or "").strip().replace("\n", " ").replace("\r", " ")
        if not sql:
            sql = "SELECT 1"  # placeholder so eval doesn't crash on blank lines
        lines.append(sql)
    path.write_text("\n".join(lines) + "\n")


def write_gold_file(examples: Sequence[SpiderExample], path: Path) -> None:
    """Write a Spider gold file: each line is ``SQL\\tdb_id``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for e in examples:
        lines.append(f"{e.query.strip().replace(chr(10), ' ')}\t{e.db_id}")
    path.write_text("\n".join(lines) + "\n")


def run_official_evaluator(
    predictions: Sequence[Prediction],
    examples: Sequence[SpiderExample],
    db_dir: Path,
    tables_json: Path,
    evaluator_dir: Path,
    *,
    etype: str = "all",
    timeout_s: int = 600,
    keep_artifacts: bool = False,
) -> OfficialResult:
    """Run ``evaluation.py`` from the vendored evaluator and parse its output.

    ``etype`` is the evaluator's mode: ``exec``, ``match``, or ``all``.
    """
    eval_script = (evaluator_dir / "evaluation.py").resolve()
    if not eval_script.exists():
        raise FileNotFoundError(
            f"vendored evaluator not found at {readable_path(eval_script)}; "
            f"run scripts/vendor_evaluator.sh first",
        )
    abs_db = db_dir.resolve()
    abs_tables = tables_json.resolve()
    abs_evaluator_dir = evaluator_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="distill-sql-eval-") as _tmp:
        tmp = Path(_tmp)
        pred_file = tmp / "pred.txt"
        gold_file = tmp / "gold.txt"
        write_pred_file(predictions, pred_file)
        write_gold_file(examples, gold_file)
        cmd = [
            sys.executable,
            str(eval_script),
            "--pred",
            str(pred_file),
            "--gold",
            str(gold_file),
            "--db",
            str(abs_db),
            "--table",
            str(abs_tables),
            "--etype",
            etype,
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=abs_evaluator_dir,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"evaluator failed (rc={proc.returncode})\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}",
            )
        if keep_artifacts:
            log = tmp.parent / "last_eval_stdout.txt"
            log.write_text(proc.stdout)
        return parse_official_output(proc.stdout)


def write_predictions_jsonl(
    predictions: Sequence[Prediction],
    per_example: Sequence[PerExampleResult],
    out_path: Path,
) -> None:
    """Write a per-example JSONL with prediction + outcome for inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {p.question_id: p for p in predictions}
    with out_path.open("w") as f:
        for r in per_example:
            row = {
                "question_id": r.question_id,
                "db_id": r.db_id,
                "difficulty": r.difficulty,
                "gold_sql": r.gold_sql,
                "pred_sql": by_id[r.question_id].predicted_sql,
                "exec_match": r.exec_match,
                "exact_set_match": r.exact_set_match,
                "failure_mode": r.failure_mode,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

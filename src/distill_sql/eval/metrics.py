"""Aggregation of per-example evaluation outcomes into final metrics.

The Spider evaluator gives us per-example execution-accuracy and exact-match
flags plus a difficulty label. This module turns those into the cuts we want
in the report (overall, per-difficulty, failure-mode breakdown).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

Difficulty = Literal["easy", "medium", "hard", "extra", "unknown"]
FailureMode = Literal["empty", "parse", "execution", "wrong-result", "ok"]


@dataclass(frozen=True)
class PerExampleResult:
    """One example's evaluation outcome."""

    question_id: int
    db_id: str
    difficulty: Difficulty
    gold_sql: str
    pred_sql: str
    exec_match: bool
    exact_set_match: bool
    failure_mode: FailureMode


@dataclass(frozen=True)
class MetricsSummary:
    """High-level numbers for a single eval config."""

    n: int
    exec_accuracy: float
    exact_match_accuracy: float
    by_difficulty: dict[str, dict[str, float]]
    failure_breakdown: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(results: Iterable[PerExampleResult]) -> MetricsSummary:
    """Aggregate per-example results into a single ``MetricsSummary``."""
    items = list(results)
    if not items:
        return MetricsSummary(
            n=0,
            exec_accuracy=0.0,
            exact_match_accuracy=0.0,
            by_difficulty={},
            failure_breakdown={},
        )

    n = len(items)
    n_exec = sum(1 for r in items if r.exec_match)
    n_exact = sum(1 for r in items if r.exact_set_match)

    diff_buckets: dict[str, list[PerExampleResult]] = {}
    for r in items:
        diff_buckets.setdefault(r.difficulty, []).append(r)

    by_difficulty: dict[str, dict[str, float]] = {}
    for label, bucket in diff_buckets.items():
        bn = len(bucket)
        by_difficulty[label] = {
            "n": float(bn),
            "exec_accuracy": sum(1 for r in bucket if r.exec_match) / bn,
            "exact_match_accuracy": sum(1 for r in bucket if r.exact_set_match) / bn,
        }

    failure_breakdown: dict[str, int] = {
        str(k): v for k, v in Counter(r.failure_mode for r in items).items()
    }

    return MetricsSummary(
        n=n,
        exec_accuracy=n_exec / n,
        exact_match_accuracy=n_exact / n,
        by_difficulty=by_difficulty,
        failure_breakdown=failure_breakdown,
    )


def classify_failure(
    pred_sql: str,
    *,
    parses: bool,
    executes: bool,
    exec_match: bool,
) -> FailureMode:
    """Bucket a prediction into a coarse failure mode for the report.

    ``ok`` if exec_match. Otherwise: empty -> parse-fail -> execution-fail ->
    wrong-result. The chain is ordered so the first failing check wins.
    """
    if exec_match:
        return "ok"
    stripped = (pred_sql or "").strip()
    if not stripped:
        return "empty"
    if not parses:
        return "parse"
    if not executes:
        return "execution"
    return "wrong-result"

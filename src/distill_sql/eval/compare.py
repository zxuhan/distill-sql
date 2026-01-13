"""Comparison helpers used by the report stage.

A ``RunResult`` is the rendered eval output for one model; ``compare_runs``
turns several of them into the table that goes into ``reports/results.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .metrics import MetricsSummary


@dataclass(frozen=True)
class RunResult:
    """One row in the final comparison."""

    name: str
    summary: MetricsSummary
    predictions_path: Path | None = None


def render_markdown_table(runs: Iterable[RunResult]) -> str:
    """Render a Markdown comparison table sorted by overall execution accuracy."""
    rows = sorted(runs, key=lambda r: r.summary.exec_accuracy)
    headers = ["model", "n", "exec_acc", "easy", "medium", "hard", "extra", "exact_match"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        bd = r.summary.by_difficulty
        cells = [
            r.name,
            str(r.summary.n),
            f"{r.summary.exec_accuracy:.3f}",
            f"{bd.get('easy', {}).get('exec_accuracy', 0.0):.3f}",
            f"{bd.get('medium', {}).get('exec_accuracy', 0.0):.3f}",
            f"{bd.get('hard', {}).get('exec_accuracy', 0.0):.3f}",
            f"{bd.get('extra', {}).get('exec_accuracy', 0.0):.3f}",
            f"{r.summary.exact_match_accuracy:.3f}",
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_results_json(runs: Iterable[RunResult], out_path: Path) -> None:
    """Persist the comparison as a single JSON document."""
    payload = {
        r.name: {
            "summary": r.summary.as_dict(),
            "predictions_path": (
                str(r.predictions_path) if r.predictions_path else None
            ),
        }
        for r in runs
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

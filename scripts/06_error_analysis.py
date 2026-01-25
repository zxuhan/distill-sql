"""Run programmatic error analysis: pull cases where one model wins vs another.

Two cuts are produced:

1. ``base -> primary``: examples the distilled student fixed (was failing,
   now passes). Useful to characterize *what distillation buys you*.
2. ``primary -> base``: examples distillation broke (was passing, now
   failing). Useful as a sanity-check that the wins aren't accompanied by
   silent regressions.

Output is a Markdown section appended to ``reports/error_analysis.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console

from distill_sql.eval.error_analysis import (
    find_disagreements,
    render_error_analysis_md,
)

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("reports/predictions"),
    )
    parser.add_argument(
        "--examples", type=Path, default=Path("data/spider/dev.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/error_analysis.md"),
    )
    parser.add_argument(
        "--baseline", type=str, default="base_qwen_0p5b",
    )
    parser.add_argument(
        "--treatments", type=str, nargs="+",
        default=["distilled_primary", "distilled_ablation_direct", "distilled_1p5b"],
    )
    args = parser.parse_args()

    base_path = args.predictions_dir / f"{args.baseline}.jsonl"
    sections: list[str] = []
    sections.append("# Error analysis\n")
    sections.append(
        "Categorized per (losing-model, winning-model) pair. "
        "Categories are heuristic and ordered: empty/parse/execution failures "
        "first, then SQL-shape mismatches (missing-join, aggregation-mismatch, ...). "
        "See `src/distill_sql/eval/error_analysis.py` for the rules.\n",
    )

    for treatment in args.treatments:
        t_path = args.predictions_dir / f"{treatment}.jsonl"
        if not (base_path.exists() and t_path.exists()):
            sections.append(f"## {args.baseline} -> {treatment}\n\n_(predictions missing)_\n")
            continue

        sections.append(f"## {args.baseline} -> {treatment} (cases the treatment fixed)\n")
        cases = find_disagreements(
            base_path, t_path,
            examples_question_path=args.examples,
        )
        sections.append(
            render_error_analysis_md(
                cases, n_examples=8,
                losing_label=args.baseline, winning_label=treatment,
            ),
        )
        sections.append("\n")

        # Reverse direction: regressions
        regressions = find_disagreements(
            t_path, base_path,
            examples_question_path=args.examples,
        )
        sections.append(f"## {treatment} -> {args.baseline} (regressions)\n")
        sections.append(f"Treatment broke {len(regressions)} examples that the base got right.\n")
        if regressions:
            cats = Counter(c.category for c in regressions)
            sections.append("Category counts:")
            for cat, n in cats.most_common():
                sections.append(f"  - {cat}: {n}")
        sections.append("\n")

        console.log(
            f"{treatment}: fixed {len(cases)}, regressed {len(regressions)} "
            f"(net +{len(cases) - len(regressions)})",
        )

    args.out.write_text("\n".join(sections))
    console.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

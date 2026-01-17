"""Error analysis: pull cases where one model fails and another succeeds.

Used by ``scripts/05_make_report.py`` to populate the README's error analysis
section. The categorization is heuristic but matches the failure patterns
documented in Spider-line text-to-SQL papers (Yu et al. 2018, Wang et al. 2020).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True)
class CasePair:
    """One case where model A fails and model B succeeds."""

    question_id: int
    db_id: str
    difficulty: str
    question: str
    gold_sql: str
    pred_a_sql: str
    pred_a_failure: str
    pred_b_sql: str
    category: str  # error category from heuristics


def _category(gold: str, pred: str, failure: str) -> str:
    """Heuristic bucketing of (gold, prediction) pairs into error categories.

    Order matters: the first matching rule wins so the categories are
    mutually exclusive. The categories are kept short; the README expands
    them inline with example pairs.
    """
    if failure == "empty":
        return "empty-output"
    if failure == "parse":
        return "parse-error"
    if failure == "execution":
        return "schema-mismatch"

    g = gold.lower()
    p = pred.lower()

    has_join_g = " join " in g
    has_join_p = " join " in p
    if has_join_g and not has_join_p:
        return "missing-join"
    if has_join_p and not has_join_g:
        return "spurious-join"

    agg_pat = re.compile(r"\b(count|sum|avg|min|max)\s*\(")
    g_agg = set(agg_pat.findall(g))
    p_agg = set(agg_pat.findall(p))
    if g_agg != p_agg:
        return "aggregation-mismatch"

    if " group by " in g and " group by " not in p:
        return "missing-group-by"

    if " where " in g and " where " not in p:
        return "missing-filter"
    if " where " not in g and " where " in p:
        return "spurious-filter"

    if " order by " in g and " order by " not in p:
        return "missing-order"

    if "distinct" in g and "distinct" not in p:
        return "missing-distinct"

    return "other"


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Read a per-example predictions JSONL written by eval/runner.py."""
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out


def find_disagreements(
    losing_path: Path,
    winning_path: Path,
    examples_question_path: Path | None = None,
) -> list[CasePair]:
    """Return cases where ``losing`` model fails (no exec_match) and ``winning`` succeeds.

    ``examples_question_path`` is the original Spider dev.json (or any file
    that maps question_id->question text); if provided, we attach question
    text to each CasePair for reporting.
    """
    a = {row["question_id"]: row for row in load_predictions(losing_path)}
    b = {row["question_id"]: row for row in load_predictions(winning_path)}
    questions: dict[int, str] = {}
    if examples_question_path is not None and examples_question_path.exists():
        # Spider dev examples are positional; index by row index = question_id.
        examples = json.loads(examples_question_path.read_text())
        for i, ex in enumerate(examples):
            questions[i] = ex.get("question", "")

    out: list[CasePair] = []
    for qid, ra in a.items():
        if ra.get("exec_match"):
            continue
        rb = b.get(qid)
        if rb is None or not rb.get("exec_match"):
            continue
        out.append(
            CasePair(
                question_id=qid,
                db_id=ra["db_id"],
                difficulty=ra.get("difficulty", "unknown"),
                question=questions.get(qid, ""),
                gold_sql=ra["gold_sql"],
                pred_a_sql=ra["pred_sql"],
                pred_a_failure=ra.get("failure_mode", ""),
                pred_b_sql=rb["pred_sql"],
                category=_category(ra["gold_sql"], ra["pred_sql"], ra.get("failure_mode", "")),
            ),
        )
    return out


def summarize_categories(cases: Iterable[CasePair]) -> dict[str, int]:
    """Count cases per category."""
    return dict(Counter(c.category for c in cases))


def render_error_analysis_md(
    cases: list[CasePair],
    n_examples: int = 12,
    losing_label: str = "student",
    winning_label: str = "teacher",
) -> str:
    """Format an error-analysis section for the README."""
    if not cases:
        return "_No disagreements found._\n"

    by_cat: dict[str, list[CasePair]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    cat_order = sorted(by_cat, key=lambda c: -len(by_cat[c]))

    lines = [
        f"In total, {len(cases)} cases where {losing_label} fails but "
        f"{winning_label} succeeds. Categorized by failure pattern:\n",
        "| category | n |",
        "|---|---|",
    ]
    for cat in cat_order:
        lines.append(f"| {cat} | {len(by_cat[cat])} |")
    lines.append("")

    # Pick the 12 most-impactful categories' example cases.
    lines.append("### Selected examples\n")
    for shown, cat in enumerate(cat_order):
        if shown >= n_examples:
            break
        case = by_cat[cat][0]
        lines.append(f"**{cat}** ({case.difficulty}, db=`{case.db_id}`)\n")
        if case.question:
            lines.append(f"Q: _{case.question}_\n")
        lines.append("```sql")
        lines.append("-- gold")
        lines.append(case.gold_sql)
        lines.append(f"-- {losing_label} (failure: {case.pred_a_failure or 'wrong-result'})")
        lines.append(case.pred_a_sql or "(empty)")
        lines.append(f"-- {winning_label}")
        lines.append(case.pred_b_sql)
        lines.append("```\n")
    return "\n".join(lines)

"""Build ``reports/results.md`` and the headline chart from results.json + predictions JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------

# Tableau-clean palette, accessible on both light and dark GitHub themes.
COLOR_DISTILLED = "#2563eb"  # blue
COLOR_DEPLOY = "#16a34a"  # green; used to flag the 4-bit fused row
COLOR_TEACHER = "#dc2626"  # red, dashed
COLOR_BASE = "#9ca3af"  # gray, dotted

# Each entry of the scaling axis: (display_name, results.json key, params_in_billions)
_SCALING_AXIS: tuple[tuple[str, str, float], ...] = (
    ("0.5B", "distilled_primary", 0.5),
    ("1.5B", "distilled_1p5b", 1.5),
    ("3B", "distilled_3b", 3.0),
    ("7B", "distilled_7b", 7.0),
)
_TEACHER_KEY = "gpt_4o_mini_reference"
_BASE_KEY = "base_qwen_0p5b"
_QUANTIZED_KEY = "distilled_1p5b_q4"


def _apply_chart_style() -> None:
    """Set a consistent typography and spine style for the report charts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        },
    )


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _exec(results: dict, key: str) -> float:
    """Return overall exec accuracy in 0-100 scale, or NaN if missing."""
    summary = results.get(key, {}).get("summary")
    if not summary:
        return float("nan")
    return float(summary["exec_accuracy"]) * 100.0


def _per_difficulty(results: dict, key: str, difficulty: str) -> float:
    """Return per-difficulty exec accuracy in 0-100, or NaN if missing."""
    summary = results.get(key, {}).get("summary")
    if not summary:
        return float("nan")
    bucket = summary.get("by_difficulty", {}).get(difficulty)
    if not bucket:
        return float("nan")
    return float(bucket["exec_accuracy"]) * 100.0


def _scaling_chart(results: dict, out_png: Path) -> None:
    """Scaling-axis line chart: distilled student vs teacher reference."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _apply_chart_style()

    params = [p for _, _, p in _SCALING_AXIS]
    labels = [l for l, _, _ in _SCALING_AXIS]
    accs = [_exec(results, k) for _, k, _ in _SCALING_AXIS]
    teacher = _exec(results, _TEACHER_KEY)
    base = _exec(results, _BASE_KEY)
    quantized = _exec(results, _QUANTIZED_KEY)

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Main scaling line.
    ax.plot(
        params, accs,
        marker="o", markersize=11, linewidth=2.6,
        color=COLOR_DISTILLED, label="Distilled student",
        zorder=3,
    )
    for x, y in zip(params, accs, strict=True):
        ax.annotate(
            f"{y:.1f}%",
            (x, y), textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=11, fontweight="bold", color=COLOR_DISTILLED,
        )

    # Quantized 1.5B as a separate marker so the deployment row is visible.
    ax.plot(
        [1.5], [quantized],
        marker="D", markersize=11, color=COLOR_DEPLOY,
        linestyle="None", label="1.5B 4-bit fused (847 MB)",
        zorder=4,
    )
    ax.annotate(
        f"{quantized:.1f}%",
        (1.5, quantized), textcoords="offset points", xytext=(0, -22),
        ha="center", fontsize=10, color=COLOR_DEPLOY, fontweight="bold",
    )

    # Reference lines.
    ax.axhline(teacher, linestyle="--", linewidth=1.8, color=COLOR_TEACHER, alpha=0.85, zorder=2)
    ax.text(
        params[-1], teacher + 1.0,
        f"GPT-4o-mini teacher: {teacher:.1f}%",
        ha="right", color=COLOR_TEACHER, fontsize=11, fontweight="bold",
    )
    ax.axhline(base, linestyle=":", linewidth=1.5, color=COLOR_BASE, alpha=0.85, zorder=2)
    ax.text(
        params[0], base + 1.0,
        f"Base 0.5B (no training): {base:.1f}%",
        ha="left", color=COLOR_BASE, fontsize=10,
    )

    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.4, 9.0)
    ax.set_ylim(20, 90)
    ax.set_xlabel("Student parameter count (log scale)")
    ax.set_ylabel("Spider dev execution accuracy (%)")
    ax.set_title("Distilled student scaling: 0.5B to 7B on Spider dev")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    console.log(f"chart -> {out_png}")


def _difficulty_chart(results: dict, out_png: Path) -> None:
    """Per-difficulty small multiples: scaling line per difficulty bucket."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _apply_chart_style()

    difficulties = ("easy", "medium", "hard", "extra")
    params = [p for _, _, p in _SCALING_AXIS]
    labels = [l for l, _, _ in _SCALING_AXIS]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)

    for ax, diff in zip(axes, difficulties, strict=True):
        student_accs = [_per_difficulty(results, k, diff) for _, k, _ in _SCALING_AXIS]
        teacher = _per_difficulty(results, _TEACHER_KEY, diff)
        base = _per_difficulty(results, _BASE_KEY, diff)

        ax.plot(
            params, student_accs,
            marker="o", markersize=8, linewidth=2.2,
            color=COLOR_DISTILLED, zorder=3,
        )
        for x, y in zip(params, student_accs, strict=True):
            ax.annotate(
                f"{y:.0f}",
                (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=COLOR_DISTILLED, fontweight="bold",
            )

        ax.axhline(teacher, linestyle="--", linewidth=1.5, color=COLOR_TEACHER, alpha=0.85)
        ax.axhline(base, linestyle=":", linewidth=1.4, color=COLOR_BASE, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xticks(params)
        ax.set_xticklabels(labels)
        ax.set_xlim(0.4, 9.0)
        ax.set_ylim(0, 100)
        ax.set_title(diff, fontsize=14)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.set_xlabel("params")

    axes[0].set_ylabel("Exec accuracy (%)")

    # Single shared legend at the figure level so each panel stays uncluttered.
    handles = [
        plt.Line2D([], [], color=COLOR_DISTILLED, marker="o", linewidth=2.2,
                   label="Distilled student"),
        plt.Line2D([], [], color=COLOR_TEACHER, linestyle="--", linewidth=1.5,
                   label="GPT-4o-mini teacher"),
        plt.Line2D([], [], color=COLOR_BASE, linestyle=":", linewidth=1.4,
                   label="Base 0.5B (no training)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle(
        "Per-difficulty scaling: gap to teacher closes fastest on hard / extra",
        fontsize=15, fontweight="bold", y=1.12,
    )
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    console.log(f"per-difficulty chart -> {out_png}")


_MODEL_NAME_MAP = {
    "base_qwen_0p5b": "Base 0.5B (no training)",
    "distilled_ablation_direct": "Distilled 0.5B (direct-only ablation)",
    "distilled_primary": "Distilled 0.5B (primary recipe)",
    "distilled_1p5b_q4": "Distilled 1.5B 4-bit fused (deployment)",
    "distilled_1p5b": "Distilled 1.5B (bf16)",
    "distilled_3b": "Distilled 3B (4-bit base)",
    "distilled_7b": "Distilled 7B (cloud)",
    "gpt_4o_mini_reference": "GPT-4o-mini (closed teacher)",
}


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _md_table(results: dict) -> str:
    """Render the per-model comparison as a Markdown table.

    Drops the always-1034 `n` column and the rarely-cited
    `exact_match` column to keep the table scannable; reports
    accuracies as percentages and uses human-readable model labels.
    """
    rows = sorted(
        ((name, data["summary"]) for name, data in results.items() if "summary" in data),
        key=lambda kv: kv[1]["exec_accuracy"],
    )
    header = ["model", "exec", "easy", "medium", "hard", "extra"]
    align = ["---", "---:", "---:", "---:", "---:", "---:"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for name, s in rows:
        bd = s["by_difficulty"]
        pretty = _MODEL_NAME_MAP.get(name, name)

        def cell(d: str) -> str:
            return _pct(float(bd.get(d, {}).get("exec_accuracy", 0.0)))

        lines.append(
            "| "
            + " | ".join(
                [
                    pretty,
                    _pct(float(s["exec_accuracy"])),
                    cell("easy"),
                    cell("medium"),
                    cell("hard"),
                    cell("extra"),
                ],
            )
            + " |",
        )
    return "\n".join(lines)


def _failure_breakdown_md(predictions_dir: Path, model_names: list[str]) -> str:
    """Per-model failure-mode counts pulled from per-example JSONL."""
    rows = []
    for name in model_names:
        path = predictions_dir / f"{name}.jsonl"
        if not path.exists():
            rows.append(f"| {name} | (predictions missing) |")
            continue
        modes: Counter[str] = Counter()
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            modes[json.loads(line)["failure_mode"]] += 1
        total = sum(modes.values()) or 1
        cells = [name]
        for k in ("ok", "wrong-result", "execution", "parse", "empty"):
            v = modes.get(k, 0)
            cells.append(f"{v} ({100 * v / total:.0f}%)")
        rows.append("| " + " | ".join(cells) + " |")
    header = ["model", "ok", "wrong-result", "execution", "parse", "empty"]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out.extend(rows)
    return "\n".join(out)


def _build_error_analysis_section(
    results: dict,
    predictions_dir: Path,
    examples_path: Path,
) -> str:
    """Build the student-vs-teacher disagreement section, if both are present."""
    from distill_sql.eval.error_analysis import (
        find_disagreements,
        render_error_analysis_md,
    )

    student_name = next(
        (n for n in results if n.startswith("distilled_") and "primary" in n),
        None,
    )
    teacher_name = next(
        (n for n in results if "openai" in n.lower() or "gpt" in n.lower() or "reference" in n),
        None,
    )
    if not (student_name and teacher_name):
        return ""

    a = predictions_dir / f"{student_name}.jsonl"
    b = predictions_dir / f"{teacher_name}.jsonl"
    if not a.exists() or not b.exists():
        return ""
    cases = find_disagreements(a, b, examples_question_path=examples_path)
    return (
        "## Error analysis: student fails, teacher succeeds\n\n"
        + render_error_analysis_md(
            cases,
            n_examples=12,
            losing_label=student_name,
            winning_label=teacher_name,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("reports/results.json"))
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("reports/predictions"),
    )
    parser.add_argument("--out-md", type=Path, default=Path("reports/results.md"))
    parser.add_argument(
        "--out-chart",
        type=Path,
        default=Path("reports/figures/exec_accuracy.png"),
    )
    parser.add_argument(
        "--out-difficulty",
        type=Path,
        default=Path("reports/figures/by_difficulty.png"),
    )
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=Path("data/spider/dev.json"),
        help="Spider dev.json -- used to attach question text to error cases",
    )
    args = parser.parse_args()

    if not args.results.exists():
        console.log(f"[red]results.json not found at {args.results}[/]")
        return 2
    results = _load_results(args.results)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    md = _md_table(results)
    failure_md = _failure_breakdown_md(args.predictions_dir, list(results))
    error_md = _build_error_analysis_section(results, args.predictions_dir, args.examples_path)

    args.out_md.write_text(
        "# Spider dev: full eval matrix\n\n"
        "## Headline numbers\n\n"
        f"{md}\n\n"
        "## Failure-mode breakdown\n\n"
        "Bucketed by the in-process executor: ``ok`` means rows match gold; "
        "``wrong-result`` parses and runs but disagrees with gold; ``execution`` "
        "raises a SQLite error; ``parse`` fails sqlglot.\n\n"
        f"{failure_md}\n\n"
        f"{error_md}",
    )
    console.log(f"wrote {args.out_md}")

    _scaling_chart(results, args.out_chart)
    _difficulty_chart(results, args.out_difficulty)
    _substitute_readme_numbers(args.results, Path("README.md"))
    return 0


_README_NUMBERS_START = "<!-- HEADLINE_NUMBERS_START -->"
_README_NUMBERS_END = "<!-- HEADLINE_NUMBERS_END -->"


def _substitute_readme_numbers(results_path: Path, readme_path: Path) -> None:
    """Replace the README's headline-numbers block with current results.

    No-op if README doesn't have the markers, or results aren't loadable.
    """
    if not readme_path.exists():
        return
    text = readme_path.read_text()
    if _README_NUMBERS_START not in text or _README_NUMBERS_END not in text:
        return
    results = _load_results(results_path)
    table = _md_table(results)
    block = (
        f"{_README_NUMBERS_START}\n\n"
        f"{table}\n\n"
        f"{_README_NUMBERS_END}"
    )
    pre = text.split(_README_NUMBERS_START, 1)[0]
    post = text.split(_README_NUMBERS_END, 1)[1]
    readme_path.write_text(pre + block + post)
    console.log(f"updated headline-numbers block in {readme_path}")


if __name__ == "__main__":
    sys.exit(main())

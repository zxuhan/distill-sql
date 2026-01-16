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


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _bar_chart(results: dict, out_png: Path) -> None:
    """Bar chart of overall execution accuracy per model."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    items = [
        (name, data["summary"]["exec_accuracy"])
        for name, data in results.items()
        if "summary" in data
    ]
    items.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    bars = ax.barh(names, vals, color="#4c72b0")
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            v + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Spider dev execution accuracy")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_title("Spider dev: execution accuracy by model")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    console.log(f"chart -> {out_png}")


def _difficulty_chart(results: dict, out_png: Path) -> None:
    """Grouped bar chart per difficulty bucket."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    diffs = ["easy", "medium", "hard", "extra"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    x = list(range(len(diffs)))
    palette = ["#a1d99b", "#74a9cf", "#dd1c77", "#fdae6b"]
    items = sorted(
        ((name, data["summary"]) for name, data in results.items() if "summary" in data),
        key=lambda kv: kv[1]["exec_accuracy"],
    )
    for i, (name, summary) in enumerate(items):
        vals = [float(summary["by_difficulty"].get(d, {}).get("exec_accuracy", 0.0)) for d in diffs]
        offsets = [xi + (i - len(items) / 2) * width + width / 2 for xi in x]
        ax.bar(offsets, vals, width=width, label=name, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(diffs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("execution accuracy")
    ax.set_title("Spider dev execution accuracy by difficulty")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    console.log(f"per-difficulty chart -> {out_png}")


def _md_table(results: dict) -> str:
    rows = sorted(
        ((name, data["summary"]) for name, data in results.items() if "summary" in data),
        key=lambda kv: kv[1]["exec_accuracy"],
    )
    header = ["model", "n", "exec", "easy", "medium", "hard", "extra", "exact_match"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name, s in rows:
        bd = s["by_difficulty"]

        def cell(d: str) -> str:
            return f"{bd.get(d, {}).get('exec_accuracy', 0.0):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(int(s["n"])),
                    f"{s['exec_accuracy']:.3f}",
                    cell("easy"),
                    cell("medium"),
                    cell("hard"),
                    cell("extra"),
                    f"{s['exact_match_accuracy']:.3f}",
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
    args = parser.parse_args()

    if not args.results.exists():
        console.log(f"[red]results.json not found at {args.results}[/]")
        return 2
    results = _load_results(args.results)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    md = _md_table(results)
    failure_md = _failure_breakdown_md(args.predictions_dir, list(results))

    args.out_md.write_text(
        "# Spider dev: full eval matrix\n\n"
        "## Headline numbers\n\n"
        f"{md}\n\n"
        "## Failure-mode breakdown\n\n"
        "Bucketed by the in-process executor: ``ok`` means rows match gold; "
        "``wrong-result`` parses and runs but disagrees with gold; ``execution`` "
        "raises a SQLite error; ``parse`` fails sqlglot.\n\n"
        f"{failure_md}\n",
    )
    console.log(f"wrote {args.out_md}")

    _bar_chart(results, args.out_chart)
    _difficulty_chart(results, args.out_difficulty)
    return 0


if __name__ == "__main__":
    sys.exit(main())

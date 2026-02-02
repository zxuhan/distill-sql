"""Score a predictions JSONL produced offline (e.g. on a cloud A100) and
fold the result into ``reports/results.json``.

This is the home-side counterpart to ``scripts/cloud_eval_cuda.py``: the
cloud writes a JSONL of ``{question_id, db_id, predicted_sql}`` per line,
you rsync it back to ``reports/predictions/<name>.jsonl``, then run:

    uv run python scripts/score_jsonl.py \\
        --predictions reports/predictions/distilled_7b.jsonl \\
        --name distilled_7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_sql.config import SpiderPaths  # noqa: E402
from distill_sql.data.spider import load_examples  # noqa: E402
from distill_sql.eval.runner import (  # noqa: E402
    Prediction,
    evaluate_predictions,
    run_official_evaluator,
    write_predictions_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--name", required=True, help="Run name to use in results.json (e.g. distilled_7b).")
    parser.add_argument(
        "--evaluator-dir",
        default=Path("third_party/test-suite-sql-eval"),
        type=Path,
    )
    parser.add_argument("--results-json", type=Path, default=Path("reports/results.json"))
    args = parser.parse_args()

    spider = SpiderPaths()
    examples = load_examples(spider.dev_path())

    preds: list[Prediction] = []
    with args.predictions.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            preds.append(Prediction(
                question_id=int(d["question_id"]),
                db_id=str(d["db_id"]),
                predicted_sql=str(d.get("predicted_sql", "")),
            ))

    summary, per_example = evaluate_predictions(preds, examples, spider.db_dir())
    print(f"in-process: exec_acc={summary.exec_accuracy:.3f} em={summary.exact_match_accuracy:.3f}")

    # Rewrite the JSONL with the M1-pipeline schema (adds difficulty,
    # gold_sql, exec_match, exact_set_match, failure_mode) so the
    # downstream report + error-analysis stages work uniformly across
    # M1 and cloud-produced predictions.
    write_predictions_jsonl(preds, per_example, args.predictions)

    summary_dict = summary.as_dict()
    try:
        official = run_official_evaluator(
            preds, examples,
            db_dir=spider.db_dir(),
            tables_json=spider.tables_path(),
            evaluator_dir=args.evaluator_dir,
            etype="all",
        )
        print(f"official:   exec_acc={official.exec_all:.3f} em={official.em_all:.3f}")
        summary_dict["exec_accuracy"] = official.exec_all
        summary_dict["exact_match_accuracy"] = official.em_all
        for k, vals in (
            ("easy", (official.exec_easy, official.em_easy)),
            ("medium", (official.exec_medium, official.em_medium)),
            ("hard", (official.exec_hard, official.em_hard)),
            ("extra", (official.exec_extra, official.em_extra)),
        ):
            if k in summary_dict["by_difficulty"]:
                summary_dict["by_difficulty"][k]["exec_accuracy"] = vals[0]
                summary_dict["by_difficulty"][k]["exact_match_accuracy"] = vals[1]
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"official evaluator skipped: {exc}")

    if args.results_json.exists():
        existing = json.loads(args.results_json.read_text())
    else:
        existing = {}
    existing[args.name] = {
        "summary": summary_dict,
        "predictions_path": str(args.predictions),
    }
    args.results_json.write_text(json.dumps(existing, indent=2))
    print(f"merged into {args.results_json} (now {len(existing)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

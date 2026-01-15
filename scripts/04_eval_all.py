"""Run every entry in a YAML eval config: predict, save jsonl, then score."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from distill_sql.config import (
    EvalConfig,
    EvalRunConfig,
    SchemaSerializerConfig,
    load_yaml_config,
)
from distill_sql.data.spider import load_examples, load_tables
from distill_sql.eval.compare import RunResult, render_markdown_table, write_results_json
from distill_sql.eval.runner import (
    Prediction,
    evaluate_predictions,
    run_official_evaluator,
    write_predictions_jsonl,
)
from distill_sql.student.infer import run_student_eval
from distill_sql.teacher.infer import run_openai_reference

console = Console()


def _eval_one(
    run: EvalRunConfig,
    examples: list,
    schemas: dict,
    db_dir: Path,
    schema_cfg: SchemaSerializerConfig,
    eval_cfg: EvalConfig,
    api_key: str | None,
) -> RunResult:
    console.log(f"[bold]== {run.name} ({run.kind}) ==[/]")
    out_pred_path = eval_cfg.output_dir / f"{run.name}.jsonl"

    if run.kind in ("base", "lora"):
        _, predictions = run_student_eval(
            run,
            examples,
            schemas,
            db_dir,
            schema_cfg=schema_cfg,
            progress_console=console,
        )
    elif run.kind == "openai":
        predictions = asyncio.run(
            run_openai_reference(
                run,
                examples,
                schemas,
                db_dir,
                api_key=api_key,
                schema_cfg=schema_cfg,
                progress_console=console,
            ),
        )
    else:
        raise ValueError(f"unknown kind {run.kind}")

    in_proc_summary, per_example = evaluate_predictions(
        predictions,
        examples,
        db_dir,
    )
    write_predictions_jsonl(predictions, per_example, out_pred_path)
    console.log(
        f"[in-process] exec_acc={in_proc_summary.exec_accuracy:.3f} "
        f"({sum(1 for r in per_example if r.exec_match)}/{len(per_example)})",
    )

    # Run the official evaluator for the headline number.
    try:
        official = run_official_evaluator(
            predictions,
            examples,
            db_dir=db_dir,
            tables_json=Path("data/spider/tables.json"),
            evaluator_dir=eval_cfg.evaluator_dir,
            etype="all",
        )
        console.log(
            f"[official]   exec_acc={official.exec_all:.3f} "
            f"em_acc={official.em_all:.3f}",
        )
        # Replace the in-process numbers with the official ones for the
        # report; keep per-difficulty from in-process for failure analysis.
        merged = in_proc_summary
        merged_dict = merged.as_dict()
        merged_dict["exec_accuracy"] = official.exec_all
        merged_dict["exact_match_accuracy"] = official.em_all
        merged_dict["by_difficulty"] = {
            "easy": {
                "n": merged.by_difficulty.get("easy", {}).get("n", 0),
                "exec_accuracy": official.exec_easy,
                "exact_match_accuracy": official.em_easy,
            },
            "medium": {
                "n": merged.by_difficulty.get("medium", {}).get("n", 0),
                "exec_accuracy": official.exec_medium,
                "exact_match_accuracy": official.em_medium,
            },
            "hard": {
                "n": merged.by_difficulty.get("hard", {}).get("n", 0),
                "exec_accuracy": official.exec_hard,
                "exact_match_accuracy": official.em_hard,
            },
            "extra": {
                "n": merged.by_difficulty.get("extra", {}).get("n", 0),
                "exec_accuracy": official.exec_extra,
                "exact_match_accuracy": official.em_extra,
            },
        }
        from distill_sql.eval.metrics import MetricsSummary

        merged = MetricsSummary(
            n=merged.n,
            exec_accuracy=merged_dict["exec_accuracy"],
            exact_match_accuracy=merged_dict["exact_match_accuracy"],
            by_difficulty=merged_dict["by_difficulty"],
            failure_breakdown=merged.failure_breakdown,
        )
        return RunResult(name=run.name, summary=merged, predictions_path=out_pred_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.log(f"[yellow]official evaluator skipped[/]: {exc}")
        return RunResult(
            name=run.name,
            summary=in_proc_summary,
            predictions_path=out_pred_path,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Override limit from config (for fast smoke runs).")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    eval_cfg: EvalConfig = load_yaml_config(args.config, EvalConfig)  # type: ignore[assignment]
    spider = eval_cfg.spider
    schema_cfg = eval_cfg.schema_serializer

    examples = load_examples(spider.dev_path())
    schemas = load_tables(spider.tables_path())
    if args.limit is not None:
        examples = examples[: args.limit]
    elif eval_cfg.limit is not None:
        examples = examples[: eval_cfg.limit]

    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for run in eval_cfg.runs:
        result = _eval_one(
            run,
            examples,
            schemas,
            spider.db_dir(),
            schema_cfg,
            eval_cfg,
            api_key,
        )
        results.append(result)

    table = render_markdown_table(results)
    console.print(table)

    out_json = Path("reports") / "results.json"
    write_results_json(results, out_json)
    console.log(f"results saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

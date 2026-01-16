"""Generate teacher traces for the full Spider train set.

Drives ``distill_sql.teacher.generate.generate_traces`` against the OpenAI API
with disk caching, structured stats, and a budget gate. All teacher samples
live under ``artifacts/cache/teacher`` keyed by request hash, so iterative
re-runs are nearly free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Confirm

from distill_sql.config import (
    SchemaSerializerConfig,
    SpiderPaths,
    TeacherConfig,
    TraceFilterConfig,
    load_yaml_config,
)
from distill_sql.data.spider import (
    SchemaSerializer,
    load_examples,
    load_tables,
)
from distill_sql.teacher.client import TeacherClient
from distill_sql.teacher.generate import (
    generate_traces,
    write_stats,
    write_traces,
)

console = Console()


def _estimate_cost(n_examples: int, cfg: TeacherConfig) -> tuple[float, float]:
    """Rough lower/upper bound estimates (USD)."""
    avg_in = 1500
    avg_out_direct = 100
    avg_out_reasoning = 250
    avg_out = cfg.reasoning_share * avg_out_reasoning + (1 - cfg.reasoning_share) * avg_out_direct
    per_call = avg_in * cfg.price_input_per_m / 1e6 + avg_out * cfg.price_output_per_m / 1e6
    total = per_call * cfg.n_samples * n_examples
    return total * 0.7, total * 1.4  # rough range


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/teacher.yaml"))
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=Path("configs/trace_filter.yaml"),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap the number of train examples processed."
    )
    parser.add_argument("--yes", action="store_true", help="Skip the cost-confirmation prompt.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/traces/spider_train.jsonl"))
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.log("[red]OPENAI_API_KEY not set[/]")
        return 2

    teacher_cfg: TeacherConfig = load_yaml_config(args.config, TeacherConfig)  # type: ignore[assignment]
    filter_cfg: TraceFilterConfig = load_yaml_config(  # type: ignore[assignment]
        args.filter_config,
        TraceFilterConfig,
    )

    spider = SpiderPaths()
    examples = load_examples(spider.train_path())
    schemas = load_tables(spider.tables_path())
    schema_dbs = set(schemas)
    examples = [e for e in examples if e.db_id in schema_dbs]
    if args.limit:
        examples = examples[: args.limit]

    n = len(examples)
    lo, hi = _estimate_cost(n, teacher_cfg)
    console.log(
        f"about to generate up to {n * teacher_cfg.n_samples} samples "
        f"({n} examples × n={teacher_cfg.n_samples})",
    )
    console.log(f"estimated cost (with caching): [bold]${lo:.2f} - ${hi:.2f}[/]")
    console.log(f"hard cap: ${teacher_cfg.max_spend_usd:.2f}")

    if hi > teacher_cfg.max_spend_usd:
        console.log(
            f"[red]upper estimate ${hi:.2f} exceeds cap ${teacher_cfg.max_spend_usd:.2f}[/]",
        )
        return 3

    if not args.yes and not Confirm.ask("Proceed?", default=False):
        return 0

    serializer = SchemaSerializer(SchemaSerializerConfig())
    client = TeacherClient(teacher_cfg, api_key=api_key)
    console.log(f"current spend: ${client.cost.spend_usd:.4f} / ${client.cost.cap_usd:.2f}")

    output_dir = teacher_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out = asyncio.run(
        generate_traces(
            examples,
            schemas,
            spider.db_dir(),
            teacher_cfg,
            filter_cfg,
            client=client,
            serializer=serializer,
            progress_console=console,
        ),
    )

    write_traces(out.traces, args.output)
    write_stats(out.stats, args.output.with_suffix(".stats.json"))
    console.log(
        f"[green]done[/]: kept {out.stats.examples_kept}/{out.stats.examples_seen}, "
        f"cost ${out.stats.cost_usd:.4f}",
    )
    console.log("filter rates:")
    console.log(json.dumps(out.stats.drop_reasons, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

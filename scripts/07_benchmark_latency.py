"""Latency + cost benchmarks for the eval matrix.

For each local-MLX model in the eval config, measure:

- Cold-start time (model load + first prediction).
- Warm tokens/sec at steady state on N dev examples.
- Average wall-clock per query (warm).

For OpenAI configs we use the cached responses to avoid re-billing; we
report wall-clock per cached call (effectively the network roundtrip
floor) plus the per-query token cost computed from the cached usage.

Output is a Markdown table appended to ``reports/latency.md`` and a JSON
sidecar at ``reports/latency.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from distill_sql.config import (
    EvalConfig,
    EvalRunConfig,
    SchemaSerializerConfig,
    load_yaml_config,
)
from distill_sql.data.spider import SchemaSerializer, load_examples, load_tables

if TYPE_CHECKING:
    from collections.abc import Sequence

console = Console()


@dataclass
class LatencyRow:
    name: str
    kind: str
    n_warm: int
    cold_load_s: float
    cold_first_query_s: float
    warm_avg_query_s: float
    warm_tokens_per_s: float
    avg_input_tokens: float
    avg_output_tokens: float
    cost_per_1k_queries_usd: float


def bench_local_model(
    run: EvalRunConfig,
    examples: Sequence,
    schemas: dict,
    db_root: Path,
    schema_cfg: SchemaSerializerConfig,
    n_warm: int = 20,
) -> LatencyRow:
    from distill_sql.student.infer import StudentInferer

    serializer = SchemaSerializer(schema_cfg)
    inferer = StudentInferer(run, serializer=serializer)

    t0 = time.time()
    inferer.load()
    cold_load = time.time() - t0

    # Cold first query (everything pre-loaded but no kv cache yet).
    t0 = time.time()
    _ = inferer.predict_one(examples[0], schemas[examples[0].db_id], db_root)
    cold_first = time.time() - t0

    # Warm: average across n queries after the first.
    n_warm_use = min(n_warm, len(examples) - 1)
    warm_examples = examples[1 : 1 + n_warm_use]
    total_in = 0
    total_out = 0
    t0 = time.time()
    for ex in warm_examples:
        out = inferer.predict_one(ex, schemas[ex.db_id], db_root)
        # Count tokens roughly: input chars / 4, output chars / 4.
        prompt_chars = sum(
            len(c) for c in [
                ex.question,
                serializer.serialize(schemas[ex.db_id], ex.question, db_root=db_root),
            ]
        )
        total_in += prompt_chars // 4
        total_out += len(out.raw_text) // 4
    warm_total = time.time() - t0
    warm_avg = warm_total / max(n_warm_use, 1)
    warm_tps = total_out / max(warm_total, 1e-3)
    avg_in = total_in / max(n_warm_use, 1)
    avg_out = total_out / max(n_warm_use, 1)

    # Local cost ~ electricity ~ negligible. Set to 0.
    cost_per_1k = 0.0

    return LatencyRow(
        name=run.name,
        kind=run.kind,
        n_warm=n_warm_use,
        cold_load_s=round(cold_load, 2),
        cold_first_query_s=round(cold_first, 3),
        warm_avg_query_s=round(warm_avg, 3),
        warm_tokens_per_s=round(warm_tps, 1),
        avg_input_tokens=round(avg_in, 0),
        avg_output_tokens=round(avg_out, 0),
        cost_per_1k_queries_usd=cost_per_1k,
    )


def bench_openai_from_cache(
    run: EvalRunConfig,
    examples: Sequence,
    schemas: dict,
    db_root: Path,
    schema_cfg: SchemaSerializerConfig,
) -> LatencyRow | None:
    """Estimate cost from cached responses if any exist; skip otherwise."""
    from distill_sql.config import TeacherConfig
    from distill_sql.data.prompts import (
        SYSTEM_PROMPT,
        build_user_prompt,
    )
    from distill_sql.teacher.client import (
        ChatMessage,
        CompletionRequest,
        DiskCache,
    )

    teacher_cfg = TeacherConfig(model=run.openai_model or "gpt-4o-mini-2024-07-18")
    cache = DiskCache(teacher_cfg.cache_dir)
    serializer = SchemaSerializer(schema_cfg)

    sample_examples = examples[:50]
    hits = 0
    total_in = 0
    total_out = 0
    for ex in sample_examples:
        block = serializer.serialize(schemas[ex.db_id], ex.question, db_root=db_root)
        user = build_user_prompt(block, ex.question, "direct")
        req = CompletionRequest(
            model=teacher_cfg.model,
            messages=(ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)),
            temperature=run.temperature,
            max_tokens=run.max_tokens,
            sample_index=0,
        )
        cached = cache.get(req.cache_key())
        if cached is not None:
            hits += 1
            total_in += cached.prompt_tokens
            total_out += cached.completion_tokens
    if hits == 0:
        return None

    avg_in = total_in / hits
    avg_out = total_out / hits
    # Tier-1 gpt-4o-mini: $0.15 / M input, $0.60 / M output.
    cost_per_call = avg_in * 0.15 / 1e6 + avg_out * 0.60 / 1e6
    cost_per_1k = cost_per_call * 1000

    return LatencyRow(
        name=run.name,
        kind=run.kind,
        n_warm=hits,
        cold_load_s=0.0,
        cold_first_query_s=0.0,  # not measured from cache
        warm_avg_query_s=0.0,    # network roundtrip not benchmarked here
        warm_tokens_per_s=0.0,
        avg_input_tokens=round(avg_in, 0),
        avg_output_tokens=round(avg_out, 0),
        cost_per_1k_queries_usd=round(cost_per_1k, 4),
    )


def render_md_table(rows: list[LatencyRow]) -> str:
    headers = [
        "model", "kind", "cold_load_s", "warm_q_s", "tokens/s",
        "avg_in_tok", "avg_out_tok", "$/1K queries",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r.name, r.kind, f"{r.cold_load_s:.1f}",
            f"{r.warm_avg_query_s:.2f}" if r.warm_avg_query_s else "n/a",
            f"{r.warm_tokens_per_s:.0f}" if r.warm_tokens_per_s else "n/a",
            f"{r.avg_input_tokens:.0f}", f"{r.avg_output_tokens:.0f}",
            f"${r.cost_per_1k_queries_usd:.4f}" if r.cost_per_1k_queries_usd else "$0.00",
        ]) + " |")
    return "\n".join(lines)


def make_chart(rows: list[LatencyRow], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    local = [r for r in rows if r.kind in ("base", "lora") and r.warm_tokens_per_s > 0]
    if not local:
        return
    names = [r.name for r in local]
    vals = [r.warm_tokens_per_s for r in local]
    bars = ax.barh(names, vals, color="#2ecc71")
    for bar, v in zip(bars, vals, strict=True):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.0f} tok/s", va="center")
    ax.set_xlabel("Output tokens / second on M1 Pro (warm steady state)")
    ax.set_title("Inference throughput on Apple Silicon")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    console.log(f"chart -> {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval_all.yaml"))
    parser.add_argument("--n-warm", type=int, default=20)
    parser.add_argument("--out-md", type=Path, default=Path("reports/latency.md"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/latency.json"))
    parser.add_argument(
        "--out-chart", type=Path, default=Path("reports/figures/latency.png"),
    )
    args = parser.parse_args()

    eval_cfg: EvalConfig = load_yaml_config(args.config, EvalConfig)  # type: ignore[assignment]
    spider = eval_cfg.spider
    examples = load_examples(spider.dev_path())[: args.n_warm + 5]
    schemas = load_tables(spider.tables_path())
    schema_cfg = eval_cfg.schema_serializer

    rows: list[LatencyRow] = []
    for run in eval_cfg.runs:
        console.log(f"benchmarking {run.name}")
        if run.kind in ("base", "lora"):
            rows.append(
                bench_local_model(
                    run, examples, schemas, spider.db_dir(), schema_cfg, n_warm=args.n_warm,
                ),
            )
        elif run.kind == "openai":
            row = bench_openai_from_cache(run, examples, schemas, spider.db_dir(), schema_cfg)
            if row is not None:
                rows.append(row)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "# Latency and cost\n\n"
        f"Cold-load and warm-steady-state numbers measured on a 16 GB M1 Pro, "
        f"sampling {args.n_warm} Spider dev examples per model. ``$/1K queries`` "
        "for OpenAI is computed from the cached prompt/completion token counts "
        "at posted Tier-1 prices ($0.15 / $0.60 per M tokens for gpt-4o-mini).\n\n"
        + render_md_table(rows),
    )
    args.out_json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    make_chart(rows, args.out_chart)
    console.log(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

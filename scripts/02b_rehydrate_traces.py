"""Reconstruct trace records from already-cached teacher responses.

Use case: the trace-generation script writes to disk only on completion. If
we want to train on a partial cache (e.g., the run is still going), this
script walks the cache, replays the trace pipeline on the cached responses,
and writes the same JSONL the trainer expects.

This is *not* a separate compute path -- it shares all the validators and
selectors with ``teacher/generate.py`` via the ``StubTeacherClient``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from distill_sql.config import (
    SchemaSerializerConfig,
    SpiderPaths,
    TeacherConfig,
    TraceFilterConfig,
    load_yaml_config,
)
from distill_sql.data.spider import SchemaSerializer, load_examples, load_tables
from distill_sql.teacher.client import (
    CompletionRequest,
    CompletionResponse,
    DiskCache,
)
from distill_sql.teacher.generate import generate_traces, write_stats, write_traces

console = Console()


class _CacheReplayClient:
    """A stub that returns whatever's in the disk cache, or raises a marker error."""

    def __init__(self, cache: DiskCache) -> None:
        self._cache = cache
        self.calls = 0
        self.cache_misses = 0

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        cached = self._cache.get(req.cache_key())
        if cached is None:
            self.cache_misses += 1
            # Return an empty response so the example gets dropped by the
            # validator (no parse, no execution match) rather than an exception.
            return CompletionResponse(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="stop",
                model=req.model,
                cached=False,
            )
        return cached

    async def gather(self, requests):
        return [await self.complete(r) for r in requests]

    def remaining_budget(self) -> float:
        return 0.0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/teacher.yaml"),
        help="Same teacher config used to generate (cache key inputs must match).",
    )
    parser.add_argument(
        "--filter-config", type=Path, default=Path("configs/trace_filter.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/traces/spider_train.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    teacher_cfg: TeacherConfig = load_yaml_config(args.config, TeacherConfig)  # type: ignore[assignment]
    filter_cfg: TraceFilterConfig = load_yaml_config(  # type: ignore[assignment]
        args.filter_config, TraceFilterConfig,
    )
    spider = SpiderPaths()

    examples = load_examples(spider.train_path())
    schemas = load_tables(spider.tables_path())
    schema_dbs = set(schemas)
    examples = [e for e in examples if e.db_id in schema_dbs]
    if args.limit:
        examples = examples[: args.limit]

    cache = DiskCache(teacher_cfg.cache_dir)
    client = _CacheReplayClient(cache)
    serializer = SchemaSerializer(SchemaSerializerConfig())

    console.log(f"replaying cache for {len(examples)} examples")
    out = await generate_traces(
        examples,
        schemas,
        spider.db_dir(),
        teacher_cfg,
        filter_cfg,
        client=client,  # type: ignore[arg-type]
        serializer=serializer,
        progress_console=console,
    )
    console.log(
        f"replay calls={client.calls} cache_misses={client.cache_misses} "
        f"kept={out.stats.examples_kept}",
    )
    write_traces(out.traces, args.output)
    write_stats(out.stats, args.output.with_suffix(".stats.json"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""End-to-end teacher trace generation with execution-validated self-consistency.

For each Spider train example, we:

1. Render the schema and ask the teacher to answer the question. We sample
   ``n_samples`` completions at non-zero temperature so we get diversity.
2. Parse SQL out of each completion.
3. Execute every parsed SQL against the example's DB; remember the result set.
4. Execute the gold SQL once and compare.
5. Keep the candidate whose result set matches gold. If none match, fall back
   to one that at least executes (configurable). If none execute, drop.
6. Apply table-grounding and parse filters.

The whole loop is logged with rich's progress bar. Cost is reported live and
the run aborts if it would exceed the configured cap.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..config import TeacherConfig, TraceFilterConfig
from ..data.prompts import (
    SYSTEM_PROMPT,
    PromptMode,
    build_user_prompt,
    extract_sql,
    split_reasoning_and_sql,
)
from ..data.spider import (
    SchemaSerializer,
    SpiderExample,
    SpiderSchema,
)
from ..sql.normalize import canonicalize, parses, referenced_tables
from .client import (
    BudgetExceededError,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
)


# ---------------------------------------------------------------------------
# Protocol so tests can substitute the stub client.
# ---------------------------------------------------------------------------


class _CompletionClient(Protocol):
    async def complete(self, req: CompletionRequest) -> CompletionResponse: ...

    async def gather(
        self,
        requests: Iterable[CompletionRequest],
    ) -> list[CompletionResponse]: ...

    def remaining_budget(self) -> float: ...


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceCandidate:
    """One teacher sample for one example."""

    sample_index: int
    raw_text: str
    sql: str | None
    reasoning: str | None
    parses_ok: bool
    executes_ok: bool
    matches_gold: bool


@dataclass(frozen=True)
class TraceRecord:
    """The trace we keep, after filtering."""

    question_id: int
    db_id: str
    question: str
    schema_block: str
    mode: PromptMode
    sql: str
    reasoning: str | None
    gold_sql: str
    n_candidates: int
    matched_gold: bool


@dataclass(frozen=True)
class GenerationStats:
    """Aggregate filter statistics for the run."""

    examples_seen: int
    examples_kept: int
    examples_dropped: int
    drop_reasons: dict[str, int]
    candidates_total: int
    candidates_parse_ok: int
    candidates_execute_ok: int
    candidates_match_gold: int
    cost_usd: float


# ---------------------------------------------------------------------------
# Result set comparison
# ---------------------------------------------------------------------------


def _execute(db_path: Path, sql: str, timeout_s: float = 5.0) -> list[tuple[Any, ...]] | None:
    """Run ``sql`` against ``db_path``. Return rows or None on error."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=timeout_s,
            isolation_level=None,
        ) as conn:
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
    except sqlite3.Error:
        return None


def _multiset_equal(
    a: list[tuple[Any, ...]],
    b: list[tuple[Any, ...]],
) -> bool:
    if len(a) != len(b):
        return False

    def norm(row: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple("" if v is None else str(v) for v in row)

    return sorted(map(norm, a)) == sorted(map(norm, b))


# ---------------------------------------------------------------------------
# Per-example pipeline
# ---------------------------------------------------------------------------


def _assign_modes(
    examples: Sequence[SpiderExample],
    cfg: TeacherConfig,
    seed: int = 0,
) -> dict[int, PromptMode]:
    """Deterministically assign each example to direct or reasoning mode."""
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_reasoning = int(round(cfg.reasoning_share * len(examples)))
    reasoning_set = set(indices[:n_reasoning])
    return {
        i: cast(PromptMode, "reasoning" if i in reasoning_set else "direct")
        for i in range(len(examples))
    }


def _build_request(
    example: SpiderExample,
    schema_block: str,
    mode: PromptMode,
    cfg: TeacherConfig,
    sample_index: int,
) -> CompletionRequest:
    user = build_user_prompt(schema_block, example.question, mode)
    max_tokens = (
        cfg.max_tokens_reasoning if mode == "reasoning" else cfg.max_tokens_direct
    )
    return CompletionRequest(
        model=cfg.model,
        messages=(
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", user),
        ),
        temperature=cfg.temperature,
        max_tokens=max_tokens,
        sample_index=sample_index,
    )


def _candidate_from_response(
    sample_index: int,
    response: CompletionResponse,
    db_path: Path,
    gold_rows: list[tuple[Any, ...]] | None,
    timeout_s: float,
) -> TraceCandidate:
    text = response.text
    reasoning, sql = split_reasoning_and_sql(text)
    if sql is None:
        sql = extract_sql(text)
    parses_ok = bool(sql) and parses(sql)
    pred_rows: list[tuple[Any, ...]] | None = None
    if parses_ok:
        pred_rows = _execute(db_path, sql or "", timeout_s=timeout_s)
    executes_ok = pred_rows is not None
    matches_gold = (
        executes_ok and gold_rows is not None and _multiset_equal(pred_rows or [], gold_rows)
    )
    return TraceCandidate(
        sample_index=sample_index,
        raw_text=text,
        sql=sql,
        reasoning=reasoning,
        parses_ok=parses_ok,
        executes_ok=executes_ok,
        matches_gold=matches_gold,
    )


def _select_candidate(
    candidates: list[TraceCandidate],
    filt: TraceFilterConfig,
    schema_table_names: set[str],
) -> tuple[TraceCandidate | None, str | None]:
    """Pick the best candidate, or return None with a drop reason."""
    if not candidates:
        return None, "no-candidates"

    matching = [c for c in candidates if c.matches_gold]
    if filt.require_matches_gold and not matching:
        if not filt.fallback_executes_only:
            return None, "no-match"
        executable = [c for c in candidates if c.executes_ok]
        if not executable:
            return None, "no-executable"
        chosen = executable[0]
    else:
        pool = matching if matching else candidates
        chosen = pool[0]

    if filt.require_executes and not chosen.executes_ok:
        return None, "no-executable"
    if filt.require_parse and not chosen.parses_ok:
        return None, "parse-error"
    if not chosen.sql:
        return None, "no-sql"

    if filt.require_table_grounded:
        refs = referenced_tables(chosen.sql)
        if refs and not (refs & schema_table_names):
            return None, "ungrounded"

    return chosen, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateOutput:
    """What ``generate_traces`` returns: the records plus the run stats."""

    traces: list[TraceRecord]
    stats: GenerationStats


async def generate_traces(  # noqa: PLR0915, C901, PLR0912 - top-level orchestrator
    examples: Sequence[SpiderExample],
    schemas: dict[str, SpiderSchema],
    db_dir: Path,
    teacher_cfg: TeacherConfig,
    filter_cfg: TraceFilterConfig,
    *,
    client: _CompletionClient,
    serializer: SchemaSerializer | None = None,
    progress_console: Console | None = None,
    seed: int = 0,
) -> GenerateOutput:
    """Drive the full generate-validate-filter loop.

    ``client`` is duck-typed so tests can pass a ``StubTeacherClient``.
    """
    serializer = serializer or SchemaSerializer()
    modes = _assign_modes(examples, teacher_cfg, seed=seed)

    # Pre-execute gold queries once; cache per (db_id, gold_sql).
    gold_cache: dict[tuple[str, str], list[tuple[Any, ...]] | None] = {}
    db_paths: dict[str, Path] = {}

    drop_reasons: Counter[str] = Counter()
    cand_parse = 0
    cand_exec = 0
    cand_match = 0
    cand_total = 0
    kept: list[TraceRecord] = []

    progress = Progress(
        TextColumn("[bold blue]traces[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TextColumn("kept={task.fields[kept]}  drop={task.fields[drop]}  $={task.fields[cost]:.2f}"),
        console=progress_console,
        transient=False,
    )
    task_id = progress.add_task("gen", total=len(examples), kept=0, drop=0, cost=0.0)

    with progress:
        for ex_idx, example in enumerate(examples):
            if example.db_id not in db_paths:
                db_paths[example.db_id] = db_dir / example.db_id / f"{example.db_id}.sqlite"
            db_path = db_paths[example.db_id]

            cache_key = (example.db_id, example.query.strip())
            if cache_key not in gold_cache:
                gold_cache[cache_key] = _execute(
                    db_path,
                    example.query,
                    timeout_s=filter_cfg.execution_timeout_s,
                )
            gold_rows = gold_cache[cache_key]
            if gold_rows is None and filter_cfg.require_matches_gold:
                drop_reasons["gold-failed"] += 1
                progress.update(task_id, advance=1, drop=sum(drop_reasons.values()))
                continue

            schema = schemas.get(example.db_id)
            if schema is None:
                drop_reasons["missing-schema"] += 1
                progress.update(task_id, advance=1, drop=sum(drop_reasons.values()))
                continue

            schema_block = serializer.serialize(
                schema,
                example.question,
                db_root=db_dir,
            )
            mode = modes[ex_idx]

            requests = [
                _build_request(example, schema_block, mode, teacher_cfg, sample_index=i)
                for i in range(teacher_cfg.n_samples)
            ]
            try:
                responses = await client.gather(requests)
            except BudgetExceededError as exc:
                progress.console.log(f"[red]budget exceeded[/]: {exc}")
                break

            schema_tables = {t.name.lower() for t in schema.tables}
            candidates = [
                _candidate_from_response(
                    i,
                    rsp,
                    db_path,
                    gold_rows,
                    filter_cfg.execution_timeout_s,
                )
                for i, rsp in enumerate(responses)
            ]
            cand_total += len(candidates)
            cand_parse += sum(1 for c in candidates if c.parses_ok)
            cand_exec += sum(1 for c in candidates if c.executes_ok)
            cand_match += sum(1 for c in candidates if c.matches_gold)

            chosen, reason = _select_candidate(candidates, filter_cfg, schema_tables)
            if chosen is None:
                drop_reasons[reason or "unknown"] += 1
            else:
                # Normalize whitespace; preserve original casing semantics.
                sql = chosen.sql or ""
                kept.append(
                    TraceRecord(
                        question_id=example.question_id or ex_idx,
                        db_id=example.db_id,
                        question=example.question,
                        schema_block=schema_block,
                        mode=mode,
                        sql=canonicalize(sql) or sql.strip().rstrip(";"),
                        reasoning=chosen.reasoning if mode == "reasoning" else None,
                        gold_sql=example.query,
                        n_candidates=len(candidates),
                        matched_gold=chosen.matches_gold,
                    ),
                )
            progress.update(
                task_id,
                advance=1,
                kept=len(kept),
                drop=sum(drop_reasons.values()),
                cost=teacher_cfg.max_spend_usd - client.remaining_budget(),
            )

    return GenerateOutput(
        traces=kept,
        stats=GenerationStats(
            examples_seen=len(examples),
            examples_kept=len(kept),
            examples_dropped=sum(drop_reasons.values()),
            drop_reasons=dict(drop_reasons),
            candidates_total=cand_total,
            candidates_parse_ok=cand_parse,
            candidates_execute_ok=cand_exec,
            candidates_match_gold=cand_match,
            cost_usd=teacher_cfg.max_spend_usd - client.remaining_budget(),
        ),
    )


def write_traces(traces: Iterable[TraceRecord], path: Path) -> None:
    """Persist traces as JSONL — one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in traces:
            row = {
                "question_id": t.question_id,
                "db_id": t.db_id,
                "question": t.question,
                "schema_block": t.schema_block,
                "mode": t.mode,
                "sql": t.sql,
                "reasoning": t.reasoning,
                "gold_sql": t.gold_sql,
                "n_candidates": t.n_candidates,
                "matched_gold": t.matched_gold,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats(stats: GenerationStats, path: Path) -> None:
    """Persist generation stats as JSON for the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "examples_seen": stats.examples_seen,
                "examples_kept": stats.examples_kept,
                "examples_dropped": stats.examples_dropped,
                "drop_reasons": stats.drop_reasons,
                "candidates_total": stats.candidates_total,
                "candidates_parse_ok": stats.candidates_parse_ok,
                "candidates_execute_ok": stats.candidates_execute_ok,
                "candidates_match_gold": stats.candidates_match_gold,
                "cost_usd": round(stats.cost_usd, 4),
            },
            indent=2,
        ),
    )


# Synchronous entry point for non-async callers (CLI, scripts).
def generate_traces_sync(
    *args: Any,
    **kwargs: Any,
) -> GenerateOutput:
    return asyncio.run(generate_traces(*args, **kwargs))

"""Reference-eval inference: run GPT-4o-mini on Spider dev once.

This shares the cache + cost meter with trace generation, so a re-run of the
reference eval is free. The output is a list of Predictions ready to feed
into the same eval runner the student uses.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..config import EvalRunConfig, SchemaSerializerConfig, TeacherConfig
from ..data.prompts import (
    SYSTEM_PROMPT,
    PromptMode,
    build_user_prompt,
    extract_sql,
)
from ..data.spider import SchemaSerializer, SpiderExample, SpiderSchema
from ..eval.runner import Prediction
from .client import (
    BudgetExceededError,
    ChatMessage,
    CompletionRequest,
    TeacherClient,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rich.console import Console


async def run_openai_reference(
    run: EvalRunConfig,
    examples: Sequence[SpiderExample],
    schemas: dict[str, SpiderSchema],
    db_root: Path,
    *,
    api_key: str | None = None,
    schema_cfg: SchemaSerializerConfig | None = None,
    progress_console: Console | None = None,
) -> list[Prediction]:
    """Generate one teacher prediction per Spider dev example."""
    if run.kind != "openai":
        raise ValueError(f"run kind must be 'openai', got {run.kind}")
    if run.openai_model is None:
        raise ValueError("openai run requires openai_model")

    teacher_cfg = TeacherConfig(
        model=run.openai_model,
        temperature=run.temperature,
        n_samples=1,
        max_tokens_direct=run.max_tokens,
        max_tokens_reasoning=run.max_tokens,
        reasoning_share=1.0 if run.use_reasoning_prompt else 0.0,
    )
    client = TeacherClient(teacher_cfg, api_key=api_key)
    serializer = SchemaSerializer(schema_cfg or SchemaSerializerConfig())
    mode: PromptMode = "reasoning" if run.use_reasoning_prompt else "direct"

    progress = Progress(
        TextColumn("[bold blue]openai-eval[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TextColumn("$={task.fields[cost]:.2f}"),
        console=progress_console,
        transient=False,
    )
    task_id = progress.add_task("eval", total=len(examples), cost=0.0)

    out: list[Prediction] = []
    with progress:
        for ex in examples:
            sch = schemas.get(ex.db_id)
            if sch is None:
                out.append(
                    Prediction(question_id=ex.question_id or 0, db_id=ex.db_id, predicted_sql=""),
                )
                progress.update(task_id, advance=1)
                continue

            block = serializer.serialize(sch, ex.question, db_root=db_root)
            user = build_user_prompt(block, ex.question, mode)
            req = CompletionRequest(
                model=run.openai_model,
                messages=(
                    ChatMessage("system", SYSTEM_PROMPT),
                    ChatMessage("user", user),
                ),
                temperature=run.temperature,
                max_tokens=run.max_tokens,
                sample_index=0,
            )
            try:
                rsp = await client.complete(req)
                sql = extract_sql(rsp.text) or ""
            except BudgetExceededError as exc:
                progress.console.log(f"[red]budget exceeded[/]: {exc}")
                break

            out.append(
                Prediction(
                    question_id=ex.question_id or 0,
                    db_id=ex.db_id,
                    predicted_sql=sql,
                ),
            )
            progress.update(
                task_id,
                advance=1,
                cost=teacher_cfg.max_spend_usd - client.remaining_budget(),
            )
    return out


def run_openai_reference_sync(
    *args: object,
    **kwargs: object,
) -> list[Prediction]:
    return asyncio.run(run_openai_reference(*args, **kwargs))  # type: ignore[arg-type]

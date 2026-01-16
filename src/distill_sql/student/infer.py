"""Batched inference over Spider dev using mlx-lm.

The interesting bit is *not* the model load; mlx-lm pulls and converts on
first use. The interesting bit is staying memory-bounded on a 16GB M1 while
running a 1034-example dev pass:

- We tokenize prompts up-front, sort by length, and batch in fixed-size
  groups so each batch's longest prompt drives the padding budget.
- We optionally run a single example at a time as a fallback for very long
  prompts that would blow the per-batch KV budget.

Output is a list of ``Prediction`` rows ready to feed into
``eval/runner.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..config import EvalRunConfig, SchemaSerializerConfig
from ..data.prompts import (
    SYSTEM_PROMPT,
    PromptMode,
    build_user_prompt,
    extract_sql,
)
from ..data.spider import (
    SchemaSerializer,
    SpiderExample,
    SpiderSchema,
)
from ..eval.runner import Prediction

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rich.console import Console

# ---------------------------------------------------------------------------
# Lazy imports of mlx so the module can be imported without mlx installed
# (helpful for static analysis on Linux). All real entry points import here.
# ---------------------------------------------------------------------------


def _load_mlx() -> tuple[Any, Any, Any, Any]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    try:
        from mlx_lm import batch_generate
    except ImportError:  # very old mlx-lm
        batch_generate = None
    return load, generate, batch_generate, make_sampler


# ---------------------------------------------------------------------------
# Predictor interfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentPrediction:
    """Per-example output before parsing into the eval-runner Prediction."""

    question_id: int
    db_id: str
    raw_text: str
    sql: str | None


class StudentInferer:
    """Wrap a loaded mlx model + tokenizer behind an ergonomic predict() API.

    Reusable across the base eval and adapter-on-top eval — the only
    difference is what's loaded into ``self.model``.
    """

    def __init__(
        self,
        run: EvalRunConfig,
        *,
        serializer: SchemaSerializer | None = None,
        max_seq_len: int = 2048,
    ) -> None:
        self.run = run
        self.serializer = serializer or SchemaSerializer()
        self.max_seq_len = max_seq_len
        self._mode: PromptMode = "reasoning" if run.use_reasoning_prompt else "direct"
        self._loaded = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._sampler: Any = None
        self._batch_generate: Any = None
        self._generate: Any = None

    def load(self) -> None:
        """Load the model + tokenizer (and optionally adapter) into memory."""
        load, generate, batch_generate, make_sampler = _load_mlx()
        model_id = self.run.model_id
        if model_id is None:
            raise ValueError(f"run {self.run.name!r} has no model_id")
        kwargs: dict[str, Any] = {}
        if self.run.kind == "lora":
            if self.run.adapter_path is None:
                raise ValueError(f"lora run {self.run.name!r} requires adapter_path")
            kwargs["adapter_path"] = str(self.run.adapter_path)
        model, tokenizer = load(model_id, **kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._sampler = make_sampler(temp=self.run.temperature)
        self._batch_generate = batch_generate
        self._generate = generate
        self._loaded = True

    def _format_prompt(self, example: SpiderExample, schema: SpiderSchema, db_root: Path) -> str:
        """Render the chat-templated prompt for one example."""
        block = self.serializer.serialize(schema, example.question, db_root=db_root)
        user = build_user_prompt(block, example.question, self._mode)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return cast(
            "str",
            self._tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    def predict_one(
        self,
        example: SpiderExample,
        schema: SpiderSchema,
        db_root: Path,
    ) -> StudentPrediction:
        """One-shot prediction (slow path, used as a fallback)."""
        if not self._loaded:
            self.load()
        prompt = self._format_prompt(example, schema, db_root)
        text = self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.run.max_tokens,
            sampler=self._sampler,
        )
        return StudentPrediction(
            question_id=example.question_id or 0,
            db_id=example.db_id,
            raw_text=text,
            sql=extract_sql(text),
        )

    def predict_batch(
        self,
        examples: Sequence[SpiderExample],
        schemas: dict[str, SpiderSchema],
        db_root: Path,
        *,
        batch_size: int = 8,
        progress_console: Console | None = None,
    ) -> list[StudentPrediction]:
        """Batched prediction, sorted-by-length to minimize padding waste."""
        if not self._loaded:
            self.load()

        prompts: list[str] = []
        for ex in examples:
            sch = schemas.get(ex.db_id)
            if sch is None:
                prompts.append("")
                continue
            prompts.append(self._format_prompt(ex, sch, db_root))

        # Tokenize first so we can sort by length cheaply.
        token_lists = [
            self._tokenizer.encode(p) if p else [self._tokenizer.eos_token_id or 0] for p in prompts
        ]
        order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
        ordered_tokens = [token_lists[i] for i in order]

        results: list[StudentPrediction | None] = [None] * len(examples)

        progress = Progress(
            TextColumn("[bold blue]infer[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            TextColumn("tps={task.fields[tps]:.1f}"),
            console=progress_console,
            transient=False,
        )
        task_id = progress.add_task("infer", total=len(examples), tps=0.0)

        with progress:
            i = 0
            tokens_done = 0
            t0 = time.time()
            while i < len(ordered_tokens):
                batch_tokens = ordered_tokens[i : i + batch_size]
                batch_indices = order[i : i + batch_size]
                if self._batch_generate is not None and len(batch_tokens) > 1:
                    out = self._batch_generate(
                        self._model,
                        self._tokenizer,
                        prompts=batch_tokens,
                        max_tokens=self.run.max_tokens,
                        sampler=self._sampler,
                    )
                    texts = list(getattr(out, "texts", out))
                else:
                    texts = []
                    for toks in batch_tokens:
                        text = self._generate(
                            self._model,
                            self._tokenizer,
                            prompt=toks,
                            max_tokens=self.run.max_tokens,
                            sampler=self._sampler,
                        )
                        texts.append(text)

                for orig_idx, text in zip(batch_indices, texts, strict=True):
                    ex = examples[orig_idx]
                    results[orig_idx] = StudentPrediction(
                        question_id=ex.question_id or 0,
                        db_id=ex.db_id,
                        raw_text=text,
                        sql=extract_sql(text),
                    )
                tokens_done += sum(len(t) for t in batch_tokens)
                i += batch_size
                elapsed = max(time.time() - t0, 1e-3)
                progress.update(
                    task_id,
                    advance=len(batch_tokens),
                    tps=tokens_done / elapsed,
                )

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Public API: run on Spider dev
# ---------------------------------------------------------------------------


def predictions_to_evaluator_format(
    examples: Sequence[SpiderExample],
    student_preds: Sequence[StudentPrediction],
) -> list[Prediction]:
    """Convert StudentPrediction (which may have sql=None) into eval Predictions.

    The official evaluator expects *some* SQL per line, so we substitute a
    placeholder for None — the eval will simply count those as wrong.
    """
    by_qid = {p.question_id: p for p in student_preds}
    out: list[Prediction] = []
    for ex in examples:
        sp = by_qid.get(ex.question_id or 0)
        sql = (sp.sql if sp and sp.sql else "") if sp else ""
        out.append(
            Prediction(
                question_id=ex.question_id or 0,
                db_id=ex.db_id,
                predicted_sql=sql,
            ),
        )
    return out


def run_student_eval(
    run: EvalRunConfig,
    examples: Sequence[SpiderExample],
    schemas: dict[str, SpiderSchema],
    db_root: Path,
    *,
    schema_cfg: SchemaSerializerConfig | None = None,
    batch_size: int = 8,
    progress_console: Console | None = None,
) -> tuple[list[StudentPrediction], list[Prediction]]:
    """Top-level helper: load model, run on examples, return predictions."""
    serializer = SchemaSerializer(schema_cfg or SchemaSerializerConfig())
    inferer = StudentInferer(run, serializer=serializer)
    raw_preds = inferer.predict_batch(
        examples,
        schemas,
        db_root,
        batch_size=batch_size,
        progress_console=progress_console,
    )
    formatted = predictions_to_evaluator_format(examples, raw_preds)
    return raw_preds, formatted

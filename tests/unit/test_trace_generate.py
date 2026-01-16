"""Tests for the teacher trace generation pipeline.

Uses ``StubTeacherClient`` with canned responses keyed by request hash so
the pipeline runs end-to-end without network access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from distill_sql.config import (
    SchemaSerializerConfig,
    TeacherConfig,
    TraceFilterConfig,
)
from distill_sql.data.prompts import build_user_prompt
from distill_sql.data.spider import SchemaSerializer, load_examples, load_tables
from distill_sql.teacher.client import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StubTeacherClient,
)
from distill_sql.teacher.generate import (
    GenerateOutput,
    _assign_modes,
    _build_request,
    _candidate_from_response,
    _multiset_equal,
    _select_candidate,
    generate_traces,
    write_stats,
    write_traces,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_assign_modes_obeys_share() -> None:
    cfg = TeacherConfig(reasoning_share=0.4)
    examples = list(range(100))
    modes = _assign_modes(examples, cfg, seed=0)  # type: ignore[arg-type]
    n_reasoning = sum(1 for m in modes.values() if m == "reasoning")
    assert n_reasoning == 40


def test_assign_modes_deterministic_for_seed() -> None:
    cfg = TeacherConfig(reasoning_share=0.5)
    a = _assign_modes(list(range(20)), cfg, seed=42)  # type: ignore[arg-type]
    b = _assign_modes(list(range(20)), cfg, seed=42)  # type: ignore[arg-type]
    assert a == b


def test_multiset_equal_treats_nulls_as_empty_strings() -> None:
    a = [(1, None)]
    b = [(1, "")]
    assert _multiset_equal(a, b)


def test_multiset_equal_handles_size_mismatch() -> None:
    assert not _multiset_equal([(1,)], [(1,), (2,)])


def test_candidate_parses_extracts_sql_from_fenced_block(spider_mini_root: Path) -> None:
    rsp = CompletionResponse(
        text="```sql\nSELECT COUNT(*) FROM animal\n```",
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
        model="m",
        cached=False,
    )
    db_path = spider_mini_root / "database" / "zoo" / "zoo.sqlite"
    cand = _candidate_from_response(0, rsp, db_path, gold_rows=[(4,)], timeout_s=5.0)
    assert cand.sql == "SELECT COUNT(*) FROM animal"
    assert cand.parses_ok and cand.executes_ok and cand.matches_gold


def test_candidate_records_parse_failure(spider_mini_root: Path) -> None:
    rsp = CompletionResponse(
        text="not sql at all",
        prompt_tokens=1,
        completion_tokens=1,
        finish_reason="stop",
        model="m",
        cached=False,
    )
    db_path = spider_mini_root / "database" / "zoo" / "zoo.sqlite"
    cand = _candidate_from_response(0, rsp, db_path, gold_rows=[(0,)], timeout_s=5.0)
    assert cand.sql is None
    assert not cand.parses_ok and not cand.executes_ok


def test_select_candidate_prefers_gold_match() -> None:
    from distill_sql.teacher.generate import TraceCandidate

    matching = TraceCandidate(0, "...", "SELECT 1", None, True, True, True)
    other = TraceCandidate(1, "...", "SELECT 2", None, True, True, False)
    chosen, reason = _select_candidate(
        [other, matching],
        TraceFilterConfig(),
        schema_table_names=set(),
    )
    assert reason is None
    assert chosen is matching


def test_select_candidate_returns_drop_reason_when_none_match() -> None:
    from distill_sql.teacher.generate import TraceCandidate

    bad = TraceCandidate(0, "x", None, None, False, False, False)
    chosen, reason = _select_candidate(
        [bad],
        TraceFilterConfig(require_matches_gold=True, fallback_executes_only=False),
        schema_table_names={"a"},
    )
    assert chosen is None
    assert reason == "no-match"


def test_select_candidate_falls_back_to_executes_only_when_configured() -> None:
    from distill_sql.teacher.generate import TraceCandidate

    a = TraceCandidate(0, "x", "SELECT 1", None, True, True, False)
    chosen, reason = _select_candidate(
        [a],
        TraceFilterConfig(require_matches_gold=True, fallback_executes_only=True),
        schema_table_names=set(),
    )
    assert chosen is a
    assert reason is None


def test_select_candidate_rejects_ungrounded_table() -> None:
    from distill_sql.teacher.generate import TraceCandidate

    bad = TraceCandidate(
        0,
        "x",
        "SELECT * FROM hallucinated_table",
        None,
        True,
        True,
        False,
    )
    chosen, reason = _select_candidate(
        [bad],
        TraceFilterConfig(
            require_matches_gold=False,
            fallback_executes_only=True,
            require_table_grounded=True,
        ),
        schema_table_names={"animal"},
    )
    assert chosen is None
    assert reason == "ungrounded"


def test_build_request_uses_correct_token_budget_per_mode() -> None:
    cfg = TeacherConfig(max_tokens_direct=100, max_tokens_reasoning=400)
    from distill_sql.data.spider import SpiderColumn, SpiderSchema, SpiderTable

    schema = SpiderSchema(
        db_id="x",
        tables=(SpiderTable("t", (SpiderColumn("c", "INT", True, 0),)),),
        foreign_keys=(),
    )
    from distill_sql.data.spider import SpiderExample

    ex = SpiderExample(db_id="x", question="how many?", query="SELECT 1", question_id=0)
    block = SchemaSerializer(SchemaSerializerConfig(include_sample_rows=False)).serialize(
        schema,
        ex.question,
    )
    direct_req = _build_request(ex, block, "direct", cfg, sample_index=0)
    reasoning_req = _build_request(ex, block, "reasoning", cfg, sample_index=0)
    assert direct_req.max_tokens == 100
    assert reasoning_req.max_tokens == 400


@pytest.mark.asyncio
async def test_generate_traces_end_to_end_with_stub(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    schemas = load_tables(spider_mini_root / "tables.json")
    serializer = SchemaSerializer(SchemaSerializerConfig(include_sample_rows=False))

    teacher_cfg = TeacherConfig(
        n_samples=2,
        reasoning_share=0.0,
        cache_dir=spider_mini_root / "_cache",
    )
    filter_cfg = TraceFilterConfig(require_matches_gold=True, fallback_executes_only=True)

    # Canned responses: build the same prompt the pipeline will build,
    # hash it, and pre-populate.
    responses: dict[str, str] = {}
    for ex_idx, ex in enumerate(examples):
        block = serializer.serialize(schemas[ex.db_id], ex.question)
        user = build_user_prompt(block, ex.question, "direct")
        for sample_index in range(2):
            req = CompletionRequest(
                model=teacher_cfg.model,
                messages=(
                    ChatMessage(
                        "system",
                        "You are an expert SQL writer. "
                        "Given an SQLite schema and a natural-language "
                        "question, produce a single SQLite query that "
                        "answers the question. Use only tables and "
                        "columns that appear in the provided schema.",
                    ),
                    ChatMessage("user", user),
                ),
                temperature=teacher_cfg.temperature,
                max_tokens=teacher_cfg.max_tokens_direct,
                sample_index=sample_index,
            )
            # First sample emits a wrong query, second emits gold —
            # validates self-consistency picks the right one.
            sql = ex.query if sample_index == 1 else "SELECT 'wrong'"
            responses[req.cache_key()] = f"```sql\n{sql}\n```"
        del ex_idx

    client = StubTeacherClient(responses)
    out: GenerateOutput = await generate_traces(
        examples,
        schemas,
        spider_mini_root / "database",
        teacher_cfg,
        filter_cfg,
        client=client,  # type: ignore[arg-type]
        serializer=serializer,
    )
    assert out.stats.examples_kept == len(examples)
    assert out.stats.examples_dropped == 0
    assert all(t.sql for t in out.traces)
    # Every match was on sample 1, so n_candidates == 2 and matched_gold == True.
    assert all(t.matched_gold for t in out.traces)
    assert all(t.n_candidates == 2 for t in out.traces)


@pytest.mark.asyncio
async def test_generate_traces_drops_when_all_candidates_fail(
    spider_mini_root: Path,
) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    schemas = load_tables(spider_mini_root / "tables.json")

    teacher_cfg = TeacherConfig(
        n_samples=1,
        reasoning_share=0.0,
        cache_dir=spider_mini_root / "_cache",
    )
    filter_cfg = TraceFilterConfig(
        require_matches_gold=True,
        fallback_executes_only=False,
    )

    client = StubTeacherClient({"default": "pure garbage that will never parse"})
    out = await generate_traces(
        examples,
        schemas,
        spider_mini_root / "database",
        teacher_cfg,
        filter_cfg,
        client=client,  # type: ignore[arg-type]
    )
    assert out.stats.examples_kept == 0
    assert out.stats.examples_dropped == len(examples)


def test_write_traces_jsonl_round_trip(tmp_path: Path) -> None:
    from distill_sql.teacher.generate import TraceRecord

    rec = TraceRecord(
        question_id=0,
        db_id="zoo",
        question="how many",
        schema_block="schema",
        mode="direct",
        sql="SELECT 1",
        reasoning=None,
        gold_sql="SELECT 1",
        n_candidates=1,
        matched_gold=True,
    )
    path = tmp_path / "traces.jsonl"
    write_traces([rec], path)
    import json as _json

    line = path.read_text().strip()
    assert _json.loads(line)["sql"] == "SELECT 1"


def test_write_stats_includes_drop_reasons(tmp_path: Path) -> None:
    from distill_sql.teacher.generate import GenerationStats

    stats = GenerationStats(
        examples_seen=10,
        examples_kept=7,
        examples_dropped=3,
        drop_reasons={"no-match": 2, "parse-error": 1},
        candidates_total=30,
        candidates_parse_ok=25,
        candidates_execute_ok=20,
        candidates_match_gold=18,
        cost_usd=0.1234,
    )
    path = tmp_path / "stats.json"
    write_stats(stats, path)
    import json as _json

    data = _json.loads(path.read_text())
    assert data["examples_kept"] == 7
    assert data["drop_reasons"]["no-match"] == 2

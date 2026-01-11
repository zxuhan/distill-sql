"""Tests for ``data/prompts.py``: prompt rendering and SQL extraction."""

from __future__ import annotations

import pytest

from distill_sql.data.prompts import (
    SQL_FENCE_CLOSE,
    SQL_FENCE_OPEN,
    SYSTEM_PROMPT,
    build_assistant_completion,
    build_messages,
    build_user_prompt,
    extract_sql,
    split_reasoning_and_sql,
)


def test_user_prompt_direct_mode_contains_required_blocks() -> None:
    p = build_user_prompt(
        schema_block="CREATE TABLE foo (id INT);",
        question="how many foos?",
        mode="direct",
    )
    assert "### Schema" in p
    assert "### Question" in p
    assert "### Instruction" in p
    assert "fenced" in p.lower()
    assert "how many foos?" in p


def test_user_prompt_reasoning_mode_asks_for_reasoning() -> None:
    p = build_user_prompt("...", "...", "reasoning")
    assert "think briefly" in p.lower() or "reason" in p.lower()


def test_assistant_completion_strips_trailing_semicolon() -> None:
    body = build_assistant_completion("SELECT 1;\n")
    assert SQL_FENCE_OPEN in body
    assert SQL_FENCE_CLOSE in body
    assert ";" not in body.split(SQL_FENCE_OPEN, 1)[1]


def test_assistant_completion_with_reasoning_prepends_text() -> None:
    body = build_assistant_completion("SELECT 1", reasoning="One row, constant.")
    assert body.index("One row, constant.") < body.index(SQL_FENCE_OPEN)


def test_extract_sql_pulls_from_fenced_block() -> None:
    completion = (
        "Here you go:\n"
        f"{SQL_FENCE_OPEN}\n"
        "SELECT * FROM animal;\n"
        f"{SQL_FENCE_CLOSE}\n"
        "Hope that helps!"
    )
    assert extract_sql(completion) == "SELECT * FROM animal"


def test_extract_sql_takes_last_block_when_multiple() -> None:
    completion = (
        f"{SQL_FENCE_OPEN}\nSELECT 1\n{SQL_FENCE_CLOSE}\n"
        "Better:\n"
        f"{SQL_FENCE_OPEN}\nSELECT 2\n{SQL_FENCE_CLOSE}"
    )
    assert extract_sql(completion) == "SELECT 2"


def test_extract_sql_falls_back_to_bare_select() -> None:
    assert extract_sql("SELECT * FROM users") == "SELECT * FROM users"


def test_extract_sql_returns_none_on_garbage() -> None:
    assert extract_sql("hello, no sql here.") is None


def test_split_reasoning_and_sql_returns_both_when_present() -> None:
    completion = (
        "First, find the joins.\n"
        f"{SQL_FENCE_OPEN}\nSELECT 1\n{SQL_FENCE_CLOSE}\n"
    )
    reasoning, sql = split_reasoning_and_sql(completion)
    assert reasoning is not None and "joins" in reasoning.lower()
    assert sql == "SELECT 1"


def test_split_reasoning_and_sql_when_only_sql() -> None:
    reasoning, sql = split_reasoning_and_sql(f"{SQL_FENCE_OPEN}\nSELECT 1\n{SQL_FENCE_CLOSE}")
    assert reasoning is None
    assert sql == "SELECT 1"


def test_split_reasoning_and_sql_on_garbage() -> None:
    assert split_reasoning_and_sql("garbage") == (None, None)


def test_build_messages_includes_system_prompt() -> None:
    msgs = build_messages("CREATE TABLE foo (id INT);", "show foos", "direct")
    assert msgs.system == SYSTEM_PROMPT
    assert msgs.assistant is None


def test_build_messages_with_completion_renders_assistant_turn() -> None:
    msgs = build_messages(
        "CREATE TABLE foo (id INT);",
        "show foos",
        "direct",
        sql="SELECT * FROM foo",
    )
    assert msgs.assistant is not None
    assert "SELECT * FROM foo" in msgs.assistant


@pytest.mark.parametrize(
    ("variant", "needle"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```sqlite\nSELECT 1\n```", "SELECT 1"),
        ("```SQL\nSELECT 2\n```", "SELECT 2"),
        ("```\nSELECT 3\n```", "SELECT 3"),
    ],
)
def test_extract_sql_accepts_variant_fences(variant: str, needle: str) -> None:
    assert extract_sql(variant) == needle

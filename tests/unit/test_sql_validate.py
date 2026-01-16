"""Tests for ``sql/validate.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distill_sql.sql.validate import (
    check_executes,
    check_parses,
    check_table_grounded,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_check_parses_passes_on_valid_sql() -> None:
    res = check_parses("SELECT 1")
    assert res.ok and res.reason == ""


def test_check_parses_fails_on_invalid_sql() -> None:
    res = check_parses("SELECT $$$ FROM ???")
    assert not res.ok and res.reason == "parse-error"


def test_check_table_grounded_accepts_valid_reference() -> None:
    res = check_table_grounded("SELECT * FROM animal", ["animal", "zookeeper"])
    assert res.ok


def test_check_table_grounded_rejects_hallucination() -> None:
    res = check_table_grounded(
        "SELECT * FROM made_up_table",
        ["animal", "zookeeper"],
    )
    assert not res.ok
    assert res.reason == "ungrounded-tables"


def test_check_table_grounded_allows_select_constant() -> None:
    # Some valid SQL doesn't reference any table; we let execution be the
    # arbiter rather than failing here.
    res = check_table_grounded("SELECT 1", ["animal"])
    assert res.ok


def test_check_executes_against_real_db(spider_mini_root: Path) -> None:
    db_path = spider_mini_root / "database" / "zoo" / "zoo.sqlite"
    assert check_executes("SELECT COUNT(*) FROM animal", db_path).ok
    assert not check_executes("SELECT COUNT(*) FROM no_such", db_path).ok


def test_check_executes_missing_db(tmp_path: Path) -> None:
    res = check_executes("SELECT 1", tmp_path / "missing.sqlite")
    assert not res.ok
    assert res.reason == "db-missing"

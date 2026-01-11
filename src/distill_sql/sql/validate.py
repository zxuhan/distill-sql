"""Validators that run before we ship a teacher trace into the training set.

Each validator returns a (ok, reason) pair so the caller can keep filter
statistics. Reasons are short identifiers, suitable for grouping.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .normalize import parses, referenced_tables


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validator."""

    ok: bool
    reason: str  # short kebab-case tag; empty if ok


def check_parses(sql: str) -> ValidationResult:
    if parses(sql):
        return ValidationResult(True, "")
    return ValidationResult(False, "parse-error")


def check_table_grounded(
    sql: str,
    schema_table_names: Iterable[str],
) -> ValidationResult:
    """At least one referenced table must exist in the schema."""
    refs = referenced_tables(sql)
    if not refs:
        # Some valid SQL (e.g. SELECT 1) has no table reference. Allow it.
        # The execution check is the real signal.
        return ValidationResult(True, "")
    schema_set = {t.lower() for t in schema_table_names}
    if refs & schema_set:
        return ValidationResult(True, "")
    return ValidationResult(False, "ungrounded-tables")


def check_executes(
    sql: str,
    db_path: Path,
    timeout_s: float = 5.0,
) -> ValidationResult:
    """Run the query against the DB. Returns ok if no SQLite error."""
    if not db_path.exists():
        return ValidationResult(False, "db-missing")
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
            cur.fetchall()
    except sqlite3.Error:
        return ValidationResult(False, "execution-error")
    return ValidationResult(True, "")

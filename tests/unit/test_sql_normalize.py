"""Tests for ``sql/normalize.py`` including hypothesis-based properties."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from distill_sql.sql.normalize import canonicalize, parses, referenced_tables


def test_canonicalize_idempotent_on_simple() -> None:
    once = canonicalize("SELECT * FROM animal WHERE id = 1")
    twice = canonicalize(once)
    assert once == twice


def test_canonicalize_whitespace_invariant() -> None:
    a = canonicalize("SELECT 1")
    b = canonicalize("   SELECT   1   ")
    c = canonicalize("\nSELECT\n1\n")
    assert a == b == c


def test_canonicalize_strips_trailing_semicolon() -> None:
    assert canonicalize("SELECT 1;") == canonicalize("SELECT 1")


def test_canonicalize_lowers_unquoted_identifiers() -> None:
    a = canonicalize("SELECT FOO FROM BAR")
    b = canonicalize("select foo from bar")
    assert a == b


def test_canonicalize_returns_empty_on_empty_input() -> None:
    assert canonicalize("") == ""
    assert canonicalize(";") == ""


def test_canonicalize_falls_back_on_unparseable() -> None:
    weird = "this is definitely !!! not @@@ sql"
    out = canonicalize(weird)
    # Fallback path: whitespace squash, no exception.
    assert "this is definitely" in out


def test_parses_truthy_for_valid() -> None:
    assert parses("SELECT 1")
    assert parses("SELECT * FROM t WHERE x > 0")


def test_parses_falsy_for_invalid() -> None:
    assert not parses("SELECT $$$ FROM @@@")
    assert not parses("(((")


def test_referenced_tables_finds_join_targets() -> None:
    refs = referenced_tables("SELECT a.* FROM animal a JOIN zookeeper z ON a.id = z.id")
    assert refs == {"animal", "zookeeper"}


def test_referenced_tables_excludes_cte_aliases() -> None:
    sql = "WITH heavy AS (SELECT id FROM animal WHERE weight_kg > 100) SELECT * FROM heavy"
    refs = referenced_tables(sql)
    assert "animal" in refs
    assert "heavy" not in refs


def test_referenced_tables_empty_on_unparseable() -> None:
    assert referenced_tables("???") == set()


def test_referenced_tables_lowercased() -> None:
    assert referenced_tables("SELECT * FROM Animal") == {"animal"}


@settings(max_examples=50, deadline=None)
@given(
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
        min_size=1,
        max_size=80,
    ),
)
def test_canonicalize_is_total(s: str) -> None:
    """Canonicalize must never raise, regardless of input."""
    out = canonicalize(s)
    assert isinstance(out, str)


@settings(max_examples=30, deadline=None)
@given(st.sampled_from(["SELECT 1", "SELECT a FROM t", "SELECT COUNT(*) FROM t WHERE a = 1"]))
def test_canonicalize_idempotent_property(sql: str) -> None:
    once = canonicalize(sql)
    assert canonicalize(once) == once


def test_canonicalize_preserves_quoted_identifier_case() -> None:
    """Double-quoted identifiers keep their case; only unquoted ones lowercase."""
    out = canonicalize('SELECT "FooBar" FROM t')
    assert "FooBar" in out


def test_canonicalize_keeps_string_literals() -> None:
    """String literals in SELECT must not be lower-cased."""
    out = canonicalize("SELECT 'TIGER' FROM animal WHERE x = 'A'")
    assert "TIGER" in out
    assert "'A'" in out


def test_referenced_tables_handles_subquery() -> None:
    """Tables in subqueries should still be picked up."""
    sql = "SELECT * FROM t1 WHERE x IN (SELECT y FROM t2)"
    refs = referenced_tables(sql)
    assert "t1" in refs
    assert "t2" in refs


def test_referenced_tables_unioned_query() -> None:
    sql = "SELECT a FROM t1 UNION SELECT a FROM t2"
    refs = referenced_tables(sql)
    assert refs == {"t1", "t2"}

"""SQL normalization wrappers around sqlglot.

The point of normalization is *not* to score equivalence — that's the
evaluator's job. It's to give us idempotent, whitespace-invariant strings we
can hash, log, and de-duplicate.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def canonicalize(sql: str, dialect: str = "sqlite") -> str:
    """Return a canonical form of ``sql``.

    Properties (assuming the input parses):

    - Idempotent: ``canonicalize(canonicalize(s)) == canonicalize(s)``.
    - Whitespace-invariant: ``canonicalize("SELECT 1") == canonicalize("  SELECT  1  ")``.
    - Dialect-normalized: keywords uppercase, identifiers lowercase, strings unquoted.

    Falls back to a minimal whitespace squash if parsing fails so this is
    always callable on raw model output.
    """
    s = sql.strip().rstrip(";").strip()
    if not s:
        return ""
    try:
        tree = sqlglot.parse_one(s, read=dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
        return _whitespace_squash(s)

    # Normalize identifier case.
    for ident in tree.find_all(exp.Identifier):
        if not ident.quoted:
            ident.set("this", ident.this.lower())

    return tree.sql(dialect=dialect, normalize=True, comments=False)


def _whitespace_squash(s: str) -> str:
    """Cheap fallback for un-parseable SQL."""
    return " ".join(s.split())


def parses(sql: str, dialect: str = "sqlite") -> bool:
    """Cheap predicate: does ``sql`` parse under the given dialect?"""
    try:
        sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
        return False
    return True


def referenced_tables(sql: str, dialect: str = "sqlite") -> set[str]:
    """Return the set of table identifiers that appear in ``sql``.

    Names are lowercased. Subquery aliases and CTEs are excluded — only
    *base* table references are returned. Used by the table-grounding filter.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
        return set()

    cte_names: set[str] = {
        cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name
    }

    out: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        name = tbl.name
        if not name:
            continue
        nm = name.lower()
        if nm in cte_names:
            continue
        out.add(nm)
    return out

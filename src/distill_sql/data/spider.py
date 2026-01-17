"""Spider dataset loader and schema serializer.

The serialized prompt format is the highest-leverage non-trivial choice in
text-to-SQL prompting. We follow the now-conventional ``CREATE TABLE`` + sample-row
recipe (see Pourreza & Rafiei 2023; Rajkumar et al. 2022) and then layer a BM25
schema-linking pass on top so that prompts stay under a fixed token budget on
databases with many tables (`baseball_1`, `wta_1`, etc.).
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._compat import readable_path
from ..config import SchemaSerializerConfig

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpiderColumn:
    """One column of one Spider table."""

    name: str
    type: str
    is_primary: bool
    table_index: int  # Index into SpiderSchema.tables.

    def render_create(self) -> str:
        col_type = self.type.upper() if self.type else "TEXT"
        suffix = " PRIMARY KEY" if self.is_primary else ""
        return f"  {_quote_ident(self.name)} {col_type}{suffix}"


@dataclass(frozen=True)
class SpiderForeignKey:
    """Foreign key edge. Indices reference SpiderSchema.tables/columns."""

    from_table: int
    from_column: str
    to_table: int
    to_column: str


@dataclass(frozen=True)
class SpiderTable:
    """One table with its columns."""

    name: str
    columns: tuple[SpiderColumn, ...]

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class SpiderSchema:
    """Schema for a single Spider DB."""

    db_id: str
    tables: tuple[SpiderTable, ...]
    foreign_keys: tuple[SpiderForeignKey, ...]

    def table_index(self, name: str) -> int | None:
        for i, t in enumerate(self.tables):
            if t.name.lower() == name.lower():
                return i
        return None


@dataclass(frozen=True)
class SpiderExample:
    """One Spider train/dev row."""

    db_id: str
    question: str
    query: str  # Gold SQL.
    question_id: int | None = None
    difficulty: str | None = None  # Filled in by the official evaluator only.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tables(path: Path) -> dict[str, SpiderSchema]:
    """Parse Spider's ``tables.json`` into a {db_id: SpiderSchema} dict.

    The Spider schema dump uses ``column_names_original`` as a list of
    ``[table_idx, col_name]`` pairs where the first row is the magic
    ``[-1, "*"]`` sentinel. We strip that sentinel and keep the rest.
    """
    raw = json.loads(path.read_text())
    schemas: dict[str, SpiderSchema] = {}
    for entry in raw:
        db_id: str = entry["db_id"]
        table_names: list[str] = entry["table_names_original"]
        column_pairs: list[list[object]] = entry["column_names_original"]
        column_types: list[str] = entry["column_types"]
        primary_keys: list[int] = entry["primary_keys"]
        fk_pairs: list[list[int]] = entry["foreign_keys"]

        per_table: list[list[SpiderColumn]] = [[] for _ in table_names]
        flat_to_table_col: list[tuple[int, str]] = []

        for col_idx, (pair, col_type) in enumerate(
            zip(column_pairs, column_types, strict=True),
        ):
            t_idx_raw, name_raw = pair[0], pair[1]
            t_idx = int(t_idx_raw) if isinstance(t_idx_raw, (int, float)) else -1
            name = str(name_raw)
            if t_idx == -1:
                # The "*" star column. Skip; we don't render it.
                flat_to_table_col.append((-1, name))
                continue
            is_primary = col_idx in primary_keys
            per_table[t_idx].append(
                SpiderColumn(
                    name=name,
                    type=str(col_type),
                    is_primary=is_primary,
                    table_index=t_idx,
                ),
            )
            flat_to_table_col.append((t_idx, name))

        # Foreign keys come as [[from_idx, to_idx], ...] where idx is into the
        # flat column list. Resolve back to (table, column) names.
        fks: list[SpiderForeignKey] = []
        for fk_from, fk_to in fk_pairs:
            f_t, f_c = flat_to_table_col[fk_from]
            t_t, t_c = flat_to_table_col[fk_to]
            if f_t == -1 or t_t == -1:
                continue
            fks.append(
                SpiderForeignKey(
                    from_table=f_t,
                    from_column=f_c,
                    to_table=t_t,
                    to_column=t_c,
                ),
            )

        tables = tuple(
            SpiderTable(name=name, columns=tuple(per_table[i]))
            for i, name in enumerate(table_names)
        )
        schemas[db_id] = SpiderSchema(
            db_id=db_id,
            tables=tables,
            foreign_keys=tuple(fks),
        )
    return schemas


def load_examples(path: Path) -> list[SpiderExample]:
    """Parse a Spider examples JSON (train_spider.json / dev.json)."""
    raw = json.loads(path.read_text())
    out: list[SpiderExample] = []
    for i, entry in enumerate(raw):
        out.append(
            SpiderExample(
                db_id=entry["db_id"],
                question=entry["question"],
                query=entry["query"],
                question_id=i,
            ),
        )
    return out


def open_db(db_root: Path, db_id: str) -> sqlite3.Connection:
    """Open a read-only connection to a Spider DB.

    ``read_only=true`` via URI keeps the trace pipeline from accidentally
    mutating schemas while sampling rows.
    """
    db_path = db_root / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Spider DB not found: {readable_path(db_path)}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def sample_rows(
    conn: sqlite3.Connection,
    table: str,
    n: int = 3,
    cell_truncate: int = 40,
) -> list[tuple[str, ...]]:
    """Fetch up to ``n`` sample rows from ``table``, with each cell stringified."""
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM "{table}" LIMIT {int(n)}')
    except sqlite3.Error:
        return []
    rows = cur.fetchall()

    def _stringify(v: object) -> str:
        if v is None:
            return "NULL"
        s = str(v).replace("\n", " ").replace("\t", " ")
        if len(s) > cell_truncate:
            s = s[: cell_truncate - 1] + "…"
        return s

    return [tuple(_stringify(c) for c in r) for r in rows]


# ---------------------------------------------------------------------------
# Schema serialization
# ---------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier with double-quotes if it isn't already."""
    if not name:
        return '""'
    if name.startswith('"') and name.endswith('"'):
        return name
    if name[0].isalpha() and all(c.isalnum() or c == "_" for c in name):
        return name
    return '"' + name.replace('"', '""') + '"'


def render_table_create(
    table: SpiderTable,
    fks_for_table: list[SpiderForeignKey],
    schema: SpiderSchema,
) -> str:
    """Produce a CREATE TABLE statement in SQLite dialect."""
    if not table.columns:
        return f"CREATE TABLE {_quote_ident(table.name)} ();"
    lines: list[str] = []
    for col in table.columns:
        lines.append(col.render_create() + ",")
    for fk in fks_for_table:
        target = schema.tables[fk.to_table]
        lines.append(
            f"  FOREIGN KEY ({_quote_ident(fk.from_column)}) "
            f"REFERENCES {_quote_ident(target.name)}({_quote_ident(fk.to_column)}),",
        )
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    body = "\n".join(lines)
    return f"CREATE TABLE {_quote_ident(table.name)} (\n{body}\n);"


def render_sample_rows_block(
    table: SpiderTable,
    rows: list[tuple[str, ...]],
) -> str:
    """Render sample rows as a fixed-width, comment-prefixed table."""
    if not rows:
        return ""
    headers = list(table.column_names())
    cols = [headers, *[list(r) for r in rows]]
    widths = [max(len(c[i]) for c in cols) for i in range(len(headers))]

    def fmt_row(row: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))

    out_lines = ["/* sample rows:", "  " + fmt_row(headers)]
    for r in rows:
        out_lines.append("  " + fmt_row(list(r)))
    out_lines.append("*/")
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Schema linking (BM25 over tables)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    """Lower-cased word tokens, with snake_case split into pieces."""
    raw = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for w in raw:
        out.extend(p for p in w.split("_") if p)
        # Also keep the original snake_case form so exact column names match.
        if "_" in w:
            out.append(w)
    return out


@dataclass(frozen=True)
class _BM25Doc:
    """One BM25 document (one table's worth of text)."""

    table_index: int
    tokens: tuple[str, ...]
    counts: dict[str, int] = field(default_factory=dict)


def _build_corpus(
    schema: SpiderSchema,
    sample_row_text: dict[int, str] | None,
) -> list[_BM25Doc]:
    docs: list[_BM25Doc] = []
    for i, t in enumerate(schema.tables):
        chunks = [t.name, *t.column_names()]
        if sample_row_text and i in sample_row_text:
            chunks.append(sample_row_text[i])
        joined = " ".join(chunks)
        toks = _tokenize(joined)
        docs.append(_BM25Doc(table_index=i, tokens=tuple(toks), counts=dict(Counter(toks))))
    return docs


def _bm25_scores(
    docs: list[_BM25Doc],
    query_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Standard BM25 with a small vocabulary of query terms."""
    n_docs = len(docs)
    if n_docs == 0:
        return []
    doc_lens = [len(d.tokens) for d in docs]
    avgdl = sum(doc_lens) / n_docs if n_docs else 0.0

    df: dict[str, int] = {}
    for term in set(query_tokens):
        df[term] = sum(1 for d in docs if d.counts.get(term, 0) > 0)

    scores = [0.0] * n_docs
    for term in query_tokens:
        n = df.get(term, 0)
        if n == 0:
            continue
        idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
        for i, d in enumerate(docs):
            tf = d.counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (doc_lens[i] / avgdl if avgdl else 0))
            scores[i] += idf * (tf * (k1 + 1)) / max(denom, 1e-9)
    return scores


# ---------------------------------------------------------------------------
# Public serializer
# ---------------------------------------------------------------------------


class SchemaSerializer:
    """Serialize a SpiderSchema into a prompt block, with BM25 schema linking.

    Use ``serialize(schema, question, db_root)`` to get the full string.
    The serializer enforces a soft token budget by ranking tables with BM25
    against the question and dropping the lowest-scoring ones first.
    """

    def __init__(self, cfg: SchemaSerializerConfig | None = None) -> None:
        self.cfg = cfg or SchemaSerializerConfig()

    def serialize(
        self,
        schema: SpiderSchema,
        question: str,
        db_root: Path | None = None,
    ) -> str:
        """Serialize ``schema`` for ``question``, possibly with sample rows.

        ``db_root`` is required only if ``include_sample_rows`` is True.
        """
        sample_rows_per_table: dict[int, list[tuple[str, ...]]] = {}
        if self.cfg.include_sample_rows and db_root is not None:
            try:
                conn = open_db(db_root, schema.db_id)
            except FileNotFoundError:
                conn = None
            if conn is not None:
                with conn:
                    for i, t in enumerate(schema.tables):
                        sample_rows_per_table[i] = sample_rows(
                            conn,
                            t.name,
                            n=self.cfg.n_sample_rows,
                            cell_truncate=self.cfg.cell_truncate,
                        )

        sample_text_for_bm25 = {
            i: " ".join(c for r in rows for c in r) for i, rows in sample_rows_per_table.items()
        }
        keep_indices = self._rank_tables(schema, question, sample_text_for_bm25)

        parts: list[str] = []
        for i in keep_indices:
            t = schema.tables[i]
            fks_here = [fk for fk in schema.foreign_keys if fk.from_table == i]
            parts.append(render_table_create(t, fks_here, schema))
            if self.cfg.include_sample_rows and i in sample_rows_per_table:
                block = render_sample_rows_block(t, sample_rows_per_table[i])
                if block:
                    parts.append(block)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _budget_chars(self) -> int:
        return int(self.cfg.max_tokens / max(self.cfg.tokens_per_char, 1e-3))

    def _rank_tables(
        self,
        schema: SpiderSchema,
        question: str,
        sample_text: dict[int, str],
    ) -> list[int]:
        """Return table indices to keep, in their original order.

        Drops the lowest-BM25 tables first if the rendered schema would
        exceed the char budget.
        """
        n = len(schema.tables)
        if n == 0:
            return []

        # First, render a tentative full schema to see if we're already in budget.
        full_indices = list(range(n))
        rendered = self._render_for(schema, full_indices, sample_text)
        if len(rendered) <= self._budget_chars():
            return full_indices

        # Over budget: rank by BM25 against question, drop weakest tail.
        docs = _build_corpus(schema, sample_text)
        q_toks = _tokenize(question)
        scores = _bm25_scores(docs, q_toks)

        # Always keep tables involved in foreign keys with kept tables — but we
        # only know which are kept after dropping. Simple stable rule: rank by
        # score desc, pick top-K, then bring back any FK-connected tables.
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        keep_min = max(self.cfg.min_tables_kept, 1)

        for k in range(keep_min, n + 1):
            chosen_set = set(order[:k])
            connected = self._fk_closure(chosen_set, schema)
            ordered = sorted(connected)
            rendered = self._render_for(schema, ordered, sample_text)
            if len(rendered) <= self._budget_chars():
                return ordered
        return sorted(self._fk_closure(set(order[:keep_min]), schema))

    @staticmethod
    def _fk_closure(seed: set[int], schema: SpiderSchema) -> set[int]:
        out = set(seed)
        for fk in schema.foreign_keys:
            if fk.from_table in out or fk.to_table in out:
                out.add(fk.from_table)
                out.add(fk.to_table)
        return out

    def _render_for(
        self,
        schema: SpiderSchema,
        indices: list[int],
        sample_text: dict[int, str],
    ) -> str:
        """Render only the named tables, used internally to size the budget."""
        index_set = set(indices)
        parts: list[str] = []
        for i in indices:
            t = schema.tables[i]
            fks_here = [
                fk for fk in schema.foreign_keys if fk.from_table == i and fk.to_table in index_set
            ]
            parts.append(render_table_create(t, fks_here, schema))
            if self.cfg.include_sample_rows and i in sample_text:
                # The sample-row text uses the raw row strings, not full block;
                # but for sizing we approximate with an upper bound that includes
                # the formatting.
                rough = sample_text[i]
                parts.append(rough)
        return "\n\n".join(parts)

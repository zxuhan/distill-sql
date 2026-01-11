"""Tests for ``data/spider.py``: loading, schema serialization, BM25 linking."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.config import SchemaSerializerConfig
from distill_sql.data.spider import (
    SchemaSerializer,
    SpiderColumn,
    SpiderForeignKey,
    SpiderSchema,
    SpiderTable,
    _bm25_scores,
    _build_corpus,
    _quote_ident,
    _tokenize,
    load_examples,
    load_tables,
    open_db,
    sample_rows,
)


def test_load_tables_resolves_columns_and_pk(spider_mini_root: Path) -> None:
    schemas = load_tables(spider_mini_root / "tables.json")
    assert "zoo" in schemas
    s = schemas["zoo"]
    table_by_name = {t.name: t for t in s.tables}
    assert set(table_by_name) == {"animal", "zookeeper"}
    animal = table_by_name["animal"]
    assert [c.name for c in animal.columns] == ["id", "species", "weight_kg", "keeper_id"]
    assert animal.columns[0].is_primary
    assert not animal.columns[1].is_primary


def test_load_tables_resolves_foreign_keys(spider_mini_root: Path) -> None:
    schemas = load_tables(spider_mini_root / "tables.json")
    s = schemas["zoo"]
    assert len(s.foreign_keys) == 1
    fk = s.foreign_keys[0]
    assert s.tables[fk.from_table].name == "animal"
    assert fk.from_column == "keeper_id"
    assert s.tables[fk.to_table].name == "zookeeper"
    assert fk.to_column == "id"


def test_load_examples_assigns_question_ids(spider_mini_root: Path) -> None:
    examples = load_examples(spider_mini_root / "dev.json")
    assert len(examples) == 3
    assert [e.question_id for e in examples] == [0, 1, 2]
    assert all(e.db_id == "zoo" for e in examples)


def test_open_db_is_read_only(spider_mini_root: Path) -> None:
    conn = open_db(spider_mini_root / "database", "zoo")
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM animal")
        n = cur.fetchone()[0]
    assert n == 4

    # Mutations against a read-only URI must error.
    conn2 = open_db(spider_mini_root / "database", "zoo")
    import sqlite3 as _sqlite3

    with pytest.raises(_sqlite3.Error):
        with conn2:
            conn2.execute("INSERT INTO animal VALUES (99, 'wolf', 30.0, 1)")


def test_open_db_missing_db_raises(spider_mini_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_db(spider_mini_root / "database", "missing_db")


def test_sample_rows_truncates_cells(spider_mini_root: Path) -> None:
    conn = open_db(spider_mini_root / "database", "zoo")
    with conn:
        rows = sample_rows(conn, "animal", n=2, cell_truncate=4)
    assert len(rows) == 2
    for r in rows:
        for cell in r:
            # Truncated cells contain ellipsis; non-truncated are <=4 chars.
            assert len(cell) <= 4 or cell.endswith("…")


def test_sample_rows_returns_empty_on_missing_table(spider_mini_root: Path) -> None:
    conn = open_db(spider_mini_root / "database", "zoo")
    with conn:
        assert sample_rows(conn, "no_such_table", n=2) == []


def test_quote_ident_handles_weird_names() -> None:
    assert _quote_ident("animal") == "animal"
    assert _quote_ident("animal id") == '"animal id"'
    assert _quote_ident('"already_quoted"') == '"already_quoted"'
    assert _quote_ident("") == '""'
    assert _quote_ident('weird"name') == '"weird""name"'


def test_tokenize_splits_snake_case() -> None:
    toks = _tokenize("weight_kg keeper_id")
    assert "weight" in toks
    assert "kg" in toks
    assert "weight_kg" in toks


def test_serializer_renders_creates_with_pk_and_fk(spider_mini_root: Path) -> None:
    schemas = load_tables(spider_mini_root / "tables.json")
    s = schemas["zoo"]
    ser = SchemaSerializer(SchemaSerializerConfig(include_sample_rows=False))
    out = ser.serialize(s, "Find tigers", db_root=spider_mini_root / "database")
    assert "CREATE TABLE animal" in out
    assert "PRIMARY KEY" in out
    assert "FOREIGN KEY (keeper_id) REFERENCES zookeeper(id)" in out


def test_serializer_includes_sample_rows(spider_mini_root: Path) -> None:
    schemas = load_tables(spider_mini_root / "tables.json")
    s = schemas["zoo"]
    ser = SchemaSerializer(
        SchemaSerializerConfig(include_sample_rows=True, n_sample_rows=2),
    )
    out = ser.serialize(s, "How many tigers?", db_root=spider_mini_root / "database")
    assert "/* sample rows:" in out
    assert "tiger" in out


def test_serializer_drops_least_relevant_table_under_budget() -> None:
    s = SpiderSchema(
        db_id="x",
        tables=(
            SpiderTable(
                "products",
                tuple([SpiderColumn(f"col_{i}", "TEXT", False, 0) for i in range(20)]),
            ),
            SpiderTable(
                "weather",
                tuple(
                    [SpiderColumn(f"temperature_{i}", "TEXT", False, 1) for i in range(20)],
                ),
            ),
            SpiderTable(
                "orders",
                tuple([SpiderColumn(f"col_{i}", "TEXT", False, 2) for i in range(20)]),
            ),
        ),
        foreign_keys=(),
    )
    cfg = SchemaSerializerConfig(
        include_sample_rows=False,
        max_tokens=80,
        min_tables_kept=1,
        tokens_per_char=0.5,
    )
    ser = SchemaSerializer(cfg)
    out = ser.serialize(s, "What is the temperature today in this weather?")
    # Weather is the only one BM25-relevant to the question; under a tight
    # budget we expect it kept and the unrelated tables dropped.
    assert "weather" in out
    # Either products or orders may also fit; but the rendered string must
    # not contain all three.
    assert not ("products" in out and "orders" in out)


def test_serializer_keeps_fk_closure() -> None:
    s = SpiderSchema(
        db_id="x",
        tables=(
            SpiderTable("users", (SpiderColumn("id", "INT", True, 0),)),
            SpiderTable("orders", (SpiderColumn("user_id", "INT", False, 1),)),
        ),
        foreign_keys=(
            SpiderForeignKey(from_table=1, from_column="user_id", to_table=0, to_column="id"),
        ),
    )
    cfg = SchemaSerializerConfig(include_sample_rows=False, max_tokens=10, min_tables_kept=1, tokens_per_char=1.0)
    ser = SchemaSerializer(cfg)
    out = ser.serialize(s, "list users")
    # Even though the budget is tiny, the FK closure is invoked and ``users``
    # remains because ``orders`` (if kept) FK-references it.
    assert "users" in out


def test_bm25_zero_when_no_query_overlap() -> None:
    s = SpiderSchema(
        db_id="x",
        tables=(
            SpiderTable("foo", (SpiderColumn("a", "INT", False, 0),)),
            SpiderTable("bar", (SpiderColumn("b", "INT", False, 1),)),
        ),
        foreign_keys=(),
    )
    docs = _build_corpus(s, sample_row_text=None)
    scores = _bm25_scores(docs, _tokenize("nothing matches"))
    assert scores == [0.0, 0.0]


def test_bm25_ranks_relevant_higher() -> None:
    s = SpiderSchema(
        db_id="x",
        tables=(
            SpiderTable("animal", (SpiderColumn("species", "TEXT", False, 0),)),
            SpiderTable("city", (SpiderColumn("name", "TEXT", False, 1),)),
        ),
        foreign_keys=(),
    )
    docs = _build_corpus(s, sample_row_text=None)
    scores = _bm25_scores(docs, _tokenize("animal species"))
    assert scores[0] > scores[1]


def test_bm25_empty_corpus() -> None:
    assert _bm25_scores([], _tokenize("anything")) == []

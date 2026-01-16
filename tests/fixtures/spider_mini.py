"""A tiny Spider-shaped fixture: one DB, two tables, three example questions.

This is enough for testing schema serialization, prompt building, evaluation
plumbing, and the trace-filter validators without pulling the real Spider
download.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# tables.json entry for one DB called ``zoo``.
TABLES_JSON: list[dict[str, Any]] = [
    {
        "db_id": "zoo",
        "table_names_original": ["animal", "zookeeper"],
        "table_names": ["animal", "zookeeper"],
        "column_names_original": [
            [-1, "*"],
            [0, "id"],
            [0, "species"],
            [0, "weight_kg"],
            [0, "keeper_id"],
            [1, "id"],
            [1, "name"],
        ],
        "column_names": [
            [-1, "*"],
            [0, "id"],
            [0, "species"],
            [0, "weight_kg"],
            [0, "keeper_id"],
            [1, "id"],
            [1, "name"],
        ],
        "column_types": ["text", "number", "text", "number", "number", "number", "text"],
        "primary_keys": [1, 5],
        "foreign_keys": [[4, 5]],
    },
]


# A few hand-written examples in the dev-shape format.
EXAMPLES: list[dict[str, str]] = [
    {
        "db_id": "zoo",
        "question": "How many animals are there?",
        "query": "SELECT COUNT(*) FROM animal",
    },
    {
        "db_id": "zoo",
        "question": "What is the average weight of animals?",
        "query": "SELECT AVG(weight_kg) FROM animal",
    },
    {
        "db_id": "zoo",
        "question": "List the names of zookeepers who care for tigers.",
        # Spider's evaluator parser requires explicit T1/T2 aliases on joins.
        "query": (
            "SELECT T2.name FROM animal AS T1 "
            "JOIN zookeeper AS T2 ON T1.keeper_id = T2.id "
            "WHERE T1.species = 'tiger'"
        ),
    },
]


def write_fixture(root: Path) -> None:
    """Materialize the fixture under ``root`` so it looks like a Spider archive."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "tables.json").write_text(json.dumps(TABLES_JSON))
    (root / "train_spider.json").write_text(json.dumps(EXAMPLES))
    (root / "dev.json").write_text(json.dumps(EXAMPLES))

    db_dir = root / "database" / "zoo"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "zoo.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE zookeeper (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE animal (
              id INTEGER PRIMARY KEY,
              species TEXT,
              weight_kg REAL,
              keeper_id INTEGER,
              FOREIGN KEY (keeper_id) REFERENCES zookeeper(id)
            );
            INSERT INTO zookeeper VALUES (1, 'Alice'), (2, 'Bob');
            INSERT INTO animal VALUES
              (1, 'tiger', 220.5, 1),
              (2, 'tiger', 180.0, 2),
              (3, 'penguin', 4.0, 1),
              (4, 'lion', 195.0, 2);
            """,
        )
        conn.commit()
    finally:
        conn.close()

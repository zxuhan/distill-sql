"""Download Spider via Hugging Face and lay it out for the official evaluator.

The official evaluator wants this on disk:

    data/spider/
        train_spider.json
        dev.json
        tables.json
        database/<db_id>/<db_id>.sqlite

We pull from two HF mirrors:

- ``premai-io/spider`` (community mirror) for ``train.json``, ``validation.json``,
  and the SQLite databases.
- The vendored ``test-suite-sql-eval/tables.json`` for the schema dump (it's
  identical to the original Spider tables.json and ships with the evaluator).

The HF cache means re-runs are nearly free.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()

TARGET = Path("data/spider")
HF_REPO = "premai-io/spider"
TABLES_SOURCE = Path("third_party/test-suite-sql-eval/tables.json")


def _copy_examples(repo_root: Path, target: Path) -> None:
    """Copy and rename train.json -> train_spider.json, validation.json -> dev.json."""
    train_src = repo_root / "train.json"
    dev_src = repo_root / "validation.json"
    if not train_src.exists() or not dev_src.exists():
        raise FileNotFoundError(
            f"premai-io/spider snapshot at {repo_root} missing train.json or validation.json",
        )
    train = json.loads(train_src.read_text())
    dev = json.loads(dev_src.read_text())
    (target / "train_spider.json").write_text(json.dumps(train))
    (target / "dev.json").write_text(json.dumps(dev))
    console.log(f"wrote {len(train)} train examples and {len(dev)} dev examples")


def _copy_databases(repo_root: Path, target: Path) -> None:
    """Recursively copy the database directory into target/database."""
    src = repo_root / "database"
    dst = target / "database"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for p in dst.iterdir() if p.is_dir())
    console.log(f"copied {n} database directories")


def _install_tables_json(target: Path) -> None:
    """Use the tables.json that ships with the vendored evaluator."""
    if not TABLES_SOURCE.exists():
        raise FileNotFoundError(
            f"vendored tables.json not found at {TABLES_SOURCE}; "
            "did you skip the test-suite-sql-eval vendor step?",
        )
    shutil.copy2(TABLES_SOURCE, target / "tables.json")
    schemas = json.loads((target / "tables.json").read_text())
    console.log(f"installed tables.json with {len(schemas)} schemas")


def _verify(target: Path) -> None:
    """Cross-check counts and database presence for the configurations we use."""
    train = json.loads((target / "train_spider.json").read_text())
    dev = json.loads((target / "dev.json").read_text())
    tables = json.loads((target / "tables.json").read_text())
    schema_dbs = {s["db_id"] for s in tables}
    train_dbs = {ex["db_id"] for ex in train}
    dev_dbs = {ex["db_id"] for ex in dev}
    db_dir = target / "database"

    missing_dev_dbs = [d for d in dev_dbs if not (db_dir / d / f"{d}.sqlite").exists()]
    missing_dev_schemas = [d for d in dev_dbs if d not in schema_dbs]
    missing_train_schemas = [d for d in train_dbs if d not in schema_dbs]

    console.log(
        f"[green]OK[/]: {len(train)} train, {len(dev)} dev, "
        f"{len(tables)} schemas, {len(train_dbs)} train DBs, "
        f"{len(dev_dbs)} dev DBs",
    )
    if missing_dev_dbs:
        console.log(
            f"[yellow]warning[/]: {len(missing_dev_dbs)} dev DBs missing on disk: "
            f"{missing_dev_dbs[:5]}",
        )
    if missing_dev_schemas:
        console.log(
            f"[yellow]warning[/]: {len(missing_dev_schemas)} dev DBs missing in tables.json: "
            f"{missing_dev_schemas[:5]}",
        )
    if missing_train_schemas:
        console.log(
            f"[yellow]warning[/]: {len(missing_train_schemas)} train DBs missing in tables.json",
        )


def main() -> None:
    target = TARGET.resolve()
    target.mkdir(parents=True, exist_ok=True)

    sentinel = target / "_PREPARED.json"
    if sentinel.exists():
        console.log("[green]Spider already prepared at data/spider, skipping download[/]")
        _verify(target)
        return

    console.log(f"downloading {HF_REPO} from HuggingFace Hub")
    snap = Path(snapshot_download(repo_id=HF_REPO, repo_type="dataset"))
    console.log(f"snapshot at {snap}")

    _copy_examples(snap, target)
    _copy_databases(snap, target)
    _install_tables_json(target)
    _verify(target)

    sentinel.write_text(
        json.dumps({"source": HF_REPO, "snapshot": str(snap)}, indent=2),
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
        console.log(f"[red]error[/]: {exc}")
        sys.exit(1)

"""``distill-sql`` CLI: thin Typer wrapper that dispatches to scripts.

The scripts under ``scripts/`` are the source of truth; this CLI exists so
``uv run distill-sql ...`` works as a single entry point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from distill_sql import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True)
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    return subprocess.call(cmd)


@app.command("version")
def cmd_version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("data")
def cmd_data() -> None:
    """Download and lay out the Spider dataset."""
    raise typer.Exit(_run("01_prepare_spider.py", []))


@app.command("teacher")
def cmd_teacher(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/teacher.yaml"),
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Run teacher trace generation."""
    extra = ["--config", str(config)]
    if yes:
        extra.append("--yes")
    if limit is not None:
        extra += ["--limit", str(limit)]
    raise typer.Exit(_run("02_generate_teacher_traces.py", extra))


@app.command("train")
def cmd_train(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/train_primary.yaml"),
) -> None:
    """Train the student LoRA adapter."""
    raise typer.Exit(_run("03_train_student.py", ["--config", str(config)]))


@app.command("eval")
def cmd_eval(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/eval_all.yaml"),
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Run the full eval matrix."""
    extra = ["--config", str(config)]
    if limit is not None:
        extra += ["--limit", str(limit)]
    raise typer.Exit(_run("04_eval_all.py", extra))


@app.command("report")
def cmd_report() -> None:
    """Build reports/results.md and the headline charts."""
    raise typer.Exit(_run("05_make_report.py", []))


def main() -> None:  # pragma: no cover — entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

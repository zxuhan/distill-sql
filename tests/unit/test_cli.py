"""Tests for the ``distill-sql`` Typer CLI.

We don't actually invoke the long-running scripts; we just verify the CLI
plumbs argparse-style flags through to subprocess and that ``--help`` works.
"""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from distill_sql import __version__
from distill_sql.cli import app


def test_version_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("data", "teacher", "train", "eval", "report"):
        assert cmd in result.stdout


def test_train_passes_config_to_subprocess() -> None:
    runner = CliRunner()
    with patch("distill_sql.cli._run", return_value=0) as run:
        result = runner.invoke(app, ["train", "--config", "configs/x.yaml"])
    assert result.exit_code == 0
    run.assert_called_once_with("03_train_student.py", ["--config", "configs/x.yaml"])


def test_teacher_propagates_yes_flag() -> None:
    runner = CliRunner()
    with patch("distill_sql.cli._run", return_value=0) as run:
        result = runner.invoke(
            app,
            ["teacher", "--config", "configs/teacher.yaml", "--yes", "--limit", "10"],
        )
    assert result.exit_code == 0
    extra = run.call_args.args[1]
    assert "--yes" in extra
    assert "--limit" in extra and "10" in extra


def test_eval_propagates_limit() -> None:
    runner = CliRunner()
    with patch("distill_sql.cli._run", return_value=0) as run:
        runner.invoke(app, ["eval", "--config", "configs/eval_base.yaml", "--limit", "100"])
    extra = run.call_args.args[1]
    assert "--limit" in extra and "100" in extra

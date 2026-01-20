"""Tests for student/utils.py logging helper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from distill_sql.student.utils import setup_run_logging

if TYPE_CHECKING:
    from pathlib import Path


def test_setup_run_logging_creates_log_file(tmp_path: Path) -> None:
    setup_run_logging(tmp_path)
    log = logging.getLogger("distill_sql")
    log.info("hello")
    for h in log.handlers:
        h.flush()
    log_path = tmp_path / "train.log"
    assert log_path.exists()
    assert "hello" in log_path.read_text()


def test_setup_run_logging_idempotent(tmp_path: Path) -> None:
    setup_run_logging(tmp_path)
    n1 = sum(
        1 for h in logging.getLogger("distill_sql").handlers if isinstance(h, logging.FileHandler)
    )
    setup_run_logging(tmp_path)
    n2 = sum(
        1 for h in logging.getLogger("distill_sql").handlers if isinstance(h, logging.FileHandler)
    )
    assert n1 == n2

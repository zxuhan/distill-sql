"""Shared helpers for student-side modules."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_run_logging(run_dir: Path) -> None:
    """Wire stdlib logging to file + stdout for one training run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)

    root = logging.getLogger("distill_sql")
    root.setLevel(logging.INFO)
    # Avoid duplicating handlers on re-runs in the same process.
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    if not file_handlers:
        root.addHandler(file_handler)

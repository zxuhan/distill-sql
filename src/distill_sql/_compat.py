"""Tiny cross-cutting helpers that don't fit elsewhere."""

from __future__ import annotations

from pathlib import Path


def readable_path(p: Path) -> str:
    """Stringify ``p`` relative to cwd if possible, else absolute.

    Output is purely cosmetic for log lines and error messages.
    """
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)

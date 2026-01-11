"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from .fixtures.spider_mini import write_fixture


@pytest.fixture(scope="session")
def spider_mini_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A populated Spider-shaped directory used by many tests."""
    root = tmp_path_factory.mktemp("spider_mini")
    write_fixture(root)
    return root

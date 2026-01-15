"""Train a student LoRA adapter from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from distill_sql.config import TrainConfig, load_yaml_config
from distill_sql.student.train import train

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg: TrainConfig = load_yaml_config(args.config, TrainConfig)  # type: ignore[assignment]
    if not cfg.traces_path.exists():
        console.log(f"[red]traces not found at {cfg.traces_path}; run scripts/02 first[/]")
        return 2
    adapter = train(cfg)
    console.log(f"[green]adapter saved to[/] {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

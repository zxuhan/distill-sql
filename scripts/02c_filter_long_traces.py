"""Filter trace JSONL to drop rows that exceed the trainer's seq budget.

mlx-lm's trainer truncates long sequences; with ``mask_prompt=True`` this
can leave a row with all of its assistant tokens cut off, leading to a
divide-by-zero in val/train loss. Keeping only rows that fit avoids that
edge case at the cost of dropping a small slice (~5%) of harder examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from transformers import AutoTokenizer

from distill_sql.data.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from distill_sql.student.train import _completion_for

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in", dest="in_path", type=Path,
        default=Path("artifacts/traces/spider_train.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/traces/spider_train_trim.jsonl"),
    )
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Drop traces whose chat-formatted len exceeds this.")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    kept = 0
    dropped = 0
    with args.in_path.open() as f, args.out.open("w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            user = build_user_prompt(t["schema_block"], t["question"], t["mode"])
            assistant = _completion_for(t["sql"], t.get("reasoning"))
            full = SYSTEM_PROMPT + "\n\n" + user + "\n\n" + assistant
            n_tok = len(tok.encode(full))
            if n_tok > args.max_tokens:
                dropped += 1
                continue
            kept += 1
            out.write(json.dumps(t, ensure_ascii=False) + "\n")
    console.log(f"kept {kept} dropped {dropped} (max_tokens={args.max_tokens})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

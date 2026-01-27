"""Generate Spider-dev predictions from a CUDA-trained adapter.

Companion to ``scripts/cloud_train_cuda.py``. Loads the base model + LoRA
adapter via transformers + peft, runs greedy decoding on every Spider dev
example, and writes a predictions JSONL in the *exact* schema the existing
M1 pipeline expects so it can be merged via ``scripts/04_eval_all.py``
(or scored offline by ``scripts/score_jsonl.py``).

Usage:
    uv run python scripts/cloud_eval_cuda.py \\
        --base Qwen/Qwen2.5-7B-Instruct \\
        --adapter artifacts/cloud_runs/scaling_7b/adapter \\
        --out reports/predictions/distilled_7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo on path so we reuse Spider loading + schema serializer + prompts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_sql.config import SchemaSerializerConfig  # noqa: E402
from distill_sql.data.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_sql,
)
from distill_sql.data.spider import (  # noqa: E402
    SchemaSerializer,
    load_examples,
    load_tables,
)


def _lazy_imports() -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base HF model id (e.g. Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument("--adapter", required=True, type=Path, help="Path to LoRA adapter dir.")
    parser.add_argument("--out", required=True, type=Path, help="Output predictions JSONL.")
    parser.add_argument("--quant", default="bf16", choices=["bf16", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--spider-dev", type=Path, default=Path("data/spider/dev.json"))
    parser.add_argument("--spider-tables", type=Path, default=Path("data/spider/tables.json"))
    parser.add_argument("--spider-db-dir", type=Path, default=Path("data/spider/database"))
    args = parser.parse_args()

    deps = _lazy_imports()

    # ---- Model + adapter
    tok = deps["AutoTokenizer"].from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.quant == "4bit":
        bnb = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=deps["torch"].bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = deps["AutoModelForCausalLM"].from_pretrained(
            args.base,
            quantization_config=bnb,
            torch_dtype=deps["torch"].bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = deps["AutoModelForCausalLM"].from_pretrained(
            args.base,
            torch_dtype=deps["torch"].bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model = deps["PeftModel"].from_pretrained(base, str(args.adapter))
    model.eval()

    # ---- Spider dev
    examples = load_examples(args.spider_dev)
    schemas = load_tables(args.spider_tables)
    if args.limit:
        examples = examples[: args.limit]
    serializer = SchemaSerializer(SchemaSerializerConfig())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing predictions to {args.out}")

    with args.out.open("w") as f:
        for i, ex in enumerate(examples):
            sch = schemas.get(ex.db_id)
            if sch is None:
                f.write(json.dumps({
                    "question_id": ex.question_id or 0,
                    "db_id": ex.db_id,
                    "predicted_sql": "",
                }) + "\n")
                continue
            block = serializer.serialize(sch, ex.question, db_root=args.spider_db_dir)
            user = build_user_prompt(block, ex.question, "direct")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            prompt_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(model.device)
            with deps["torch"].no_grad():
                out_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            completion = tok.decode(
                out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True,
            )
            sql = extract_sql(completion) or ""
            f.write(json.dumps({
                "question_id": ex.question_id or 0,
                "db_id": ex.db_id,
                "predicted_sql": sql,
            }) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(examples)}")

    print(f"done: {len(examples)} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

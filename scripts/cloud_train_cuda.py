"""CUDA / HuggingFace LoRA SFT trainer for the cloud (A100) scaling-axis runs.

The M1 pipeline uses ``mlx-lm`` for training. mlx-lm is Apple-Silicon-only,
so the cloud arm uses transformers + peft + trl. Hyperparameters mirror
``configs/train_*.yaml`` 1:1 so a 7B / 14B point on the scaling chart is
strictly an extension of the same recipe, not a different recipe.

Usage:
    uv run python scripts/cloud_train_cuda.py \\
        --config configs/train_7b_cuda.yaml

Inputs: a YAML config, the same trace JSONL the M1 trainer reads.
Outputs: a PEFT adapter under ``artifacts/cloud_runs/<run_name>/adapter``.

Score the trained model with ``scripts/cloud_eval_cuda.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# When launched under ``accelerate launch --multi_gpu``, accelerate sets
# LOCAL_RANK on each process. In that mode we must NOT use
# device_map="auto" (which assumes a single process owns all GPUs);
# accelerate handles placement and DDP wrapping itself.
IS_DISTRIBUTED: bool = "LOCAL_RANK" in os.environ
LOCAL_RANK: int = int(os.environ.get("LOCAL_RANK", "0"))


def _is_main_process() -> bool:
    return LOCAL_RANK == 0

import yaml

# Heavy deps imported lazily so --help works on a CPU box.
def _lazy_imports() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainingArguments": TrainingArguments,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


# Reuse prompt building from the package so M1 and CUDA training see the
# exact same chat format. This requires running from the repo root (or
# with the repo on PYTHONPATH); the launcher sets that up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_sql.data.prompts import (  # noqa: E402
    SQL_FENCE_CLOSE,
    SQL_FENCE_OPEN,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def _completion_for(sql: str, reasoning: str | None) -> str:
    sql_clean = sql.strip().rstrip(";").strip()
    body = f"{SQL_FENCE_OPEN}\n{sql_clean}\n{SQL_FENCE_CLOSE}"
    return f"{reasoning.strip()}\n\n{body}" if reasoning else body


def _trace_to_chat(trace: dict, include_reasoning: bool) -> dict:
    user = build_user_prompt(trace["schema_block"], trace["question"], trace["mode"])
    reasoning = trace.get("reasoning") if include_reasoning else None
    assistant = _completion_for(trace["sql"], reasoning)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Train on the first 50 traces for ~3 min, sanity-check the pipeline.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    deps = _lazy_imports()

    base_id = cfg["base"]["model_id"]
    max_seq = int(cfg["base"]["max_seq_len"])
    quant = cfg["base"].get("quant", "bf16")
    out_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"

    # ---- Tokenizer
    tok = deps["AutoTokenizer"].from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Model load (bf16 or 4-bit NF4)
    # Under accelerate (multi-GPU DDP), each rank loads a full copy and
    # accelerate wraps it with DistributedDataParallel; passing
    # device_map="auto" in that mode breaks placement.
    common_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if not IS_DISTRIBUTED:
        common_kwargs["device_map"] = "auto"

    if quant == "4bit":
        bnb = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=deps["torch"].bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = deps["AutoModelForCausalLM"].from_pretrained(
            base_id,
            quantization_config=bnb,
            torch_dtype=deps["torch"].bfloat16,
            **common_kwargs,
        )
        model = deps["prepare_model_for_kbit_training"](model)
    else:
        model = deps["AutoModelForCausalLM"].from_pretrained(
            base_id,
            torch_dtype=deps["torch"].bfloat16,
            **common_kwargs,
        )

    # ---- LoRA (rank 16, alpha 32, all linear projections — same as M1 runs)
    lora = deps["LoraConfig"](
        r=int(cfg["rank"]),
        lora_alpha=float(cfg["alpha"]),
        lora_dropout=float(cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = deps["get_peft_model"](model, lora)
    if _is_main_process():
        model.print_trainable_parameters()

    if cfg.get("grad_checkpoint"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- Dataset
    traces = []
    with Path(cfg["traces_path"]).open() as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    if args.smoke:
        traces = traces[:50]
    chats = [
        _trace_to_chat(t, include_reasoning=bool(cfg.get("include_reasoning_traces", True)))
        for t in traces
    ]
    n_val = max(1, int(len(chats) * float(cfg.get("val_split", 0.05))))
    val_set = deps["Dataset"].from_list(chats[:n_val])
    train_set = deps["Dataset"].from_list(chats[n_val:])
    if _is_main_process():
        print(f"loaded {len(train_set)} train / {len(val_set)} val records")
        if IS_DISTRIBUTED:
            print(f"distributed mode: {os.environ.get('WORLD_SIZE')} processes")

    # ---- Trainer
    sft = deps["SFTConfig"](
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=int(cfg["warmup_steps"]),
        lr_scheduler_type=cfg.get("schedule", "cosine"),
        num_train_epochs=int(cfg["epochs"]),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_every_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["save_every_steps"]),
        save_total_limit=2,
        bf16=True,
        max_seq_length=max_seq,
        packing=False,
        logging_steps=20,
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    trainer = deps["SFTTrainer"](
        model=model,
        args=sft,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tok,
    )
    trainer.train()

    # ---- Save adapter only (not the merged base — keeps cloud→home rsync small)
    # Only the main process writes; trainer.accelerator.is_main_process
    # is the safe check under accelerate.
    if _is_main_process():
        model.save_pretrained(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))
        print(f"adapter saved to {adapter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

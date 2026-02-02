"""HF Space demo for distill-sql.

Loads a Qwen2.5 base + PEFT LoRA adapter and exposes a tiny Gradio UI:
pick a Spider schema (or paste your own), type a question, get SQL.

Configurable via env vars on the HF Space side:

  BASE_MODEL    HuggingFace base model id; default Qwen/Qwen2.5-1.5B-Instruct
                (free CPU tier). Set to Qwen/Qwen2.5-7B-Instruct on a T4
                Space to use the 7B adapter.
  ADAPTER_PATH  Path inside the Space to the PEFT adapter directory;
                default ./adapter (so you commit it alongside app.py).
  MAX_NEW_TOK   Generation cap; default 256.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./adapter")
MAX_NEW_TOK = int(os.environ.get("MAX_NEW_TOK", "256"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"[boot] loading {BASE_MODEL} on {DEVICE} (dtype={DTYPE})")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True,
)

adapter_dir = Path(ADAPTER_PATH)
if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
    print(f"[boot] applying PEFT adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
else:
    print(f"[boot] no adapter found at {adapter_dir}; running base model (no distillation)")

model.eval()
print("[boot] ready")


SYSTEM_PROMPT = (
    "You are an expert SQL writer. Given an SQLite schema and a natural-language "
    "question, produce a single SQLite query that answers the question. "
    "Use only tables and columns that appear in the provided schema."
)


# ---------------------------------------------------------------------------
# A few hand-picked Spider schemas + questions for one-click demos.
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict] = {
    "concert_singer": {
        "schema": """CREATE TABLE stadium (
  Stadium_ID NUMBER PRIMARY KEY,
  Location TEXT, Name TEXT,
  Capacity NUMBER, Highest NUMBER, Lowest NUMBER, Average NUMBER
);

CREATE TABLE singer (
  Singer_ID NUMBER PRIMARY KEY,
  Name TEXT, Country TEXT, Song_Name TEXT,
  Song_release_year TEXT, Age NUMBER, Is_male TEXT
);

CREATE TABLE concert (
  concert_ID NUMBER PRIMARY KEY,
  concert_Name TEXT, Theme TEXT, Stadium_ID NUMBER, Year TEXT,
  FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
);

CREATE TABLE singer_in_concert (
  concert_ID NUMBER, Singer_ID NUMBER,
  FOREIGN KEY (concert_ID) REFERENCES concert(concert_ID),
  FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
);""",
        "questions": [
            "How many singers are there?",
            "What are the names, countries, and ages of all singers, ordered by age descending?",
            "Find names of stadiums that did not have any concerts in 2014.",
            "What is the average and maximum capacity of stadiums that held a concert in 2014?",
        ],
    },
    "car_1": {
        "schema": """CREATE TABLE continents (
  ContId NUMBER PRIMARY KEY,
  Continent TEXT
);

CREATE TABLE countries (
  CountryId NUMBER PRIMARY KEY,
  CountryName TEXT, Continent NUMBER,
  FOREIGN KEY (Continent) REFERENCES continents(ContId)
);

CREATE TABLE car_makers (
  Id NUMBER PRIMARY KEY,
  Maker TEXT, FullName TEXT, Country TEXT,
  FOREIGN KEY (Country) REFERENCES countries(CountryId)
);

CREATE TABLE model_list (
  ModelId NUMBER PRIMARY KEY,
  Maker NUMBER, Model TEXT,
  FOREIGN KEY (Maker) REFERENCES car_makers(Id)
);

CREATE TABLE cars_data (
  Id NUMBER PRIMARY KEY,
  MPG TEXT, Cylinders NUMBER, Edispl NUMBER, Horsepower TEXT,
  Weight NUMBER, Accelerate NUMBER, Year NUMBER
);""",
        "questions": [
            "How many car makers are there in total?",
            "What is the smallest weight of a car with 8 cylinders produced in 1974?",
            "List the names of countries with more than 3 car makers.",
            "What are the makers and models of cars whose horsepower is greater than 150?",
        ],
    },
    "flight_2": {
        "schema": """CREATE TABLE airlines (
  uid NUMBER PRIMARY KEY,
  Airline TEXT, Abbreviation TEXT, Country TEXT
);

CREATE TABLE airports (
  City TEXT, AirportCode TEXT PRIMARY KEY,
  AirportName TEXT, Country TEXT, CountryAbbrev TEXT
);

CREATE TABLE flights (
  Airline NUMBER, FlightNo NUMBER,
  SourceAirport TEXT, DestAirport TEXT,
  FOREIGN KEY (Airline) REFERENCES airlines(uid),
  FOREIGN KEY (SourceAirport) REFERENCES airports(AirportCode),
  FOREIGN KEY (DestAirport) REFERENCES airports(AirportCode)
);""",
        "questions": [
            "What is the country of the airline JetBlue Airways?",
            "How many flights depart from the airport with code APG?",
            "Find the cities that have airports where flights from Anchorage land.",
            "Which airlines have at least 200 flights?",
        ],
    },
}


def build_prompt(schema: str, question: str) -> str:
    return (
        f"### Schema (SQLite)\n{schema}\n\n"
        f"### Question\n{question.strip()}\n\n"
        f"### Instruction\n"
        f"Output the final SQL in a fenced ```sql block. Do not include any other text."
    )


_FENCE_RE = re.compile(r"```(?:sqlite|sql)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(completion: str) -> str:
    matches = _FENCE_RE.findall(completion)
    if matches:
        return matches[-1].strip().rstrip(";").strip()
    stripped = completion.strip()
    if stripped.lower().startswith(("select ", "with ")):
        return stripped.rstrip(";").strip()
    return stripped or "(no SQL produced)"


def generate_sql(schema: str, question: str) -> str:
    if not schema.strip() or not question.strip():
        return "-- please provide both a schema and a question"
    prompt = build_prompt(schema, question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    completion = tok.decode(out_ids[0, input_len:], skip_special_tokens=True)
    return extract_sql(completion)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def update_example(name: str) -> tuple[str, str]:
    ex = EXAMPLES[name]
    return ex["schema"], ex["questions"][0]


HEADER_MD = f"""# distill-sql

A text-to-SQL model **distilled from GPT-4o-mini** into a small Qwen2.5
student. Quantized 1.5B variant fits in **847 MB** and reaches 62.5% on
Spider dev; the 3B distilled hits 72.6%, 7B distilled hits 75.0%
(closed teacher: 80.1%).

This Space runs **{BASE_MODEL}** on **{DEVICE.upper()}**. Pick an example
schema below or paste your own SQLite `CREATE TABLE` block.

[GitHub →](https://github.com/zxuhan/distill-sql)
"""

with gr.Blocks(title="distill-sql", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER_MD)
    with gr.Row():
        with gr.Column(scale=3):
            example_select = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                value="concert_singer",
                label="Example schema",
            )
            schema_box = gr.Textbox(
                label="Schema (SQLite CREATE TABLE)",
                lines=14,
                value=EXAMPLES["concert_singer"]["schema"],
            )
            question_box = gr.Textbox(
                label="Question",
                value=EXAMPLES["concert_singer"]["questions"][0],
            )
            example_questions = gr.Examples(
                examples=[[q] for q in EXAMPLES["concert_singer"]["questions"]],
                inputs=[question_box],
                label="Quick questions for this schema",
            )
            generate_btn = gr.Button("Generate SQL", variant="primary")
        with gr.Column(scale=2):
            output_box = gr.Code(
                label="Generated SQL",
                language="sql",
                lines=10,
            )
            gr.Markdown(
                "**How this works.** Schema + question are wrapped in the same "
                "prompt template the student was trained on, then the model "
                "greedily decodes a fenced ```sql block. Average per-query "
                "latency: ~2-5s on T4, ~5-15s on free CPU."
            )

    example_select.change(update_example, example_select, [schema_box, question_box])
    generate_btn.click(generate_sql, [schema_box, question_box], output_box)


if __name__ == "__main__":
    demo.launch()

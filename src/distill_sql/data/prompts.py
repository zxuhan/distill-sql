"""Prompt templates for the teacher and the student.

We use plain f-strings rather than Jinja: the templates are short, the rendering
logic stays in code, and golden-file tests catch drift directly.

Two modes share most of the structure:

- ``direct``: produce SQL only, fenced in a single code block.
- ``reasoning``: produce a brief step-by-step plan first, then the SQL block.

Both use the same delimiter for the final SQL so the post-parser is one regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

# A single-line system prompt. Lives here so the golden tests pin it.
SYSTEM_PROMPT: Final[str] = (
    "You are an expert SQL writer. Given an SQLite schema and a natural-language "
    "question, produce a single SQLite query that answers the question. "
    "Use only tables and columns that appear in the provided schema."
)

# Sentinel block for the final SQL. The student is trained to emit this exact
# fence; the parser below strips it out.
SQL_FENCE_OPEN: Final[str] = "```sql"
SQL_FENCE_CLOSE: Final[str] = "```"

_REASONING_PREFIX: Final[str] = (
    "Think briefly (2-4 sentences) about which tables and columns the question "
    "needs and any joins or aggregations involved. Then output the final SQL "
    "in a fenced ```sql block."
)
_DIRECT_PREFIX: Final[str] = (
    "Output the final SQL in a fenced ```sql block. Do not include any other text."
)

PromptMode = Literal["direct", "reasoning"]


@dataclass(frozen=True)
class StudentMessages:
    """Chat-template-ready (system, user, assistant) triplet for SFT.

    ``assistant`` is the gold completion the student is trained against;
    it's empty at inference time and provided at training time.
    """

    system: str
    user: str
    assistant: str | None = None


def build_user_prompt(schema_block: str, question: str, mode: PromptMode) -> str:
    """Render the user-turn prompt from a schema block and a question."""
    instr = _REASONING_PREFIX if mode == "reasoning" else _DIRECT_PREFIX
    return (
        f"### Schema (SQLite)\n"
        f"{schema_block}\n\n"
        f"### Question\n"
        f"{question.strip()}\n\n"
        f"### Instruction\n"
        f"{instr}"
    )


def build_assistant_completion(
    sql: str,
    reasoning: str | None = None,
) -> str:
    """Render the assistant turn (gold completion) for SFT.

    If ``reasoning`` is given, it appears before the fenced SQL block.
    """
    sql_clean = sql.strip().rstrip(";").strip()
    body = f"{SQL_FENCE_OPEN}\n{sql_clean}\n{SQL_FENCE_CLOSE}"
    if reasoning:
        return f"{reasoning.strip()}\n\n{body}"
    return body


def build_messages(
    schema_block: str,
    question: str,
    mode: PromptMode,
    *,
    sql: str | None = None,
    reasoning: str | None = None,
) -> StudentMessages:
    """Convenience: full triplet for either inference or training."""
    user = build_user_prompt(schema_block, question, mode)
    assistant = build_assistant_completion(sql, reasoning) if sql is not None else None
    return StudentMessages(system=SYSTEM_PROMPT, user=user, assistant=assistant)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Match either ```sql … ``` or, if the model forgot the language tag, just
# ``` … ```. We list the longer alternative first because Python's regex
# engine is leftmost; otherwise ``sql`` would match inside ``sqlite`` and
# leave ``ite\n`` as part of the captured content.
_FENCE_RE = re.compile(
    r"```(?:sqlite|sql)?[ \t]*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_sql(completion: str) -> str | None:
    """Return the SQL inside the last fenced sql block, or None.

    We use the *last* block because reasoning-mode answers sometimes show
    intermediate scratch SQL fragments before the final answer.
    """
    matches: list[str] = _FENCE_RE.findall(completion)
    if not matches:
        # Fallback: if the model just emitted bare SQL, take the whole thing
        # if it parses syntactically as a SELECT-shaped statement.
        stripped = completion.strip()
        if stripped.lower().startswith(("select ", "with ", "insert ", "update ", "delete ")):
            return stripped.rstrip(";").strip()
        return None
    return matches[-1].strip().rstrip(";").strip()


def split_reasoning_and_sql(completion: str) -> tuple[str | None, str | None]:
    """Split a reasoning-mode completion into (reasoning, sql).

    Anything before the last ```sql fence is treated as the reasoning chunk.
    Returns (None, sql) if no reasoning is present, (None, None) on failure.
    """
    matches = list(_FENCE_RE.finditer(completion))
    if not matches:
        sql = extract_sql(completion)
        return (None, sql)
    last = matches[-1]
    sql = last.group(1).strip().rstrip(";").strip()
    reasoning = completion[: last.start()].strip() or None
    return (reasoning, sql)

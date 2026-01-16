# Methodology

This is the long-form companion to the README's brief methodology section.
It documents the non-obvious design choices in the pipeline along with the
reasoning that led to each one.

## Schema linking with BM25

The student model has a 32k context window in principle but only 0.5B
parameters: long schemas dilute attention and the model regresses on hard
queries. We render full ``CREATE TABLE`` statements with primary keys,
foreign keys, and 2-3 sample rows per table; if the rendered schema would
exceed a soft token budget (default ~1500 tokens), we score each table with
BM25 against the question and drop the lowest-scoring ones until the schema
fits. Tables involved in foreign keys with kept tables come back in via a
single closure pass — otherwise BM25 can drop a referenced table and leave
a dangling join.

Tokenization for BM25 is whitespace + snake_case-aware: ``weight_kg`` is
indexed as ``weight_kg``, ``weight``, and ``kg`` so a question asking
about "weight" matches both the column and any ``kg``-suffixed siblings.

## Two-mode teacher prompts

The teacher (gpt-4o-mini) is prompted in two modes:

- **direct**: schema + question -> fenced SQL only. Cheap. Good for the
  bulk of training data.
- **reasoning**: schema + question -> 2-4 sentences of plan + fenced SQL.
  More expensive (~3x output tokens) but produces traces that teach the
  student how to approach harder queries.

The mix is configurable; the default is 60% direct / 40% reasoning. At
training time, the student sees both formats so at inference it can
optionally produce reasoning before the SQL block (the post-parser only
takes the last fenced block).

## Execution-validated self-consistency

For each train example we sample n=3 teacher completions at
temperature 0.3, execute each against the example's SQLite DB, and keep
the candidate whose result set matches gold (compared as a multiset of
stringified rows). If none match, we fall back to one that at least
executes; if none execute, we drop the example and log it.

This is the single highest-leverage quality lever in the pipeline. A
cheaper alternative (n=1 at temperature 0) costs less but produces
visibly worse student behaviour on hard queries because the teacher's
single greedy answer is sometimes subtly wrong.

## Filtering teacher candidates

Every kept candidate must additionally:

1. Parse via ``sqlglot`` (catches stray punctuation in fenced blocks).
2. Reference at least one table that exists in the schema (cheap
   hallucination guard; we exempt SELECT-constant queries that don't
   reference any table).
3. Execute against the DB without raising a SQLite error.

Filter rates are logged as part of the run stats so we can see how the
teacher behaves and tune the prompt without re-running the whole job.

## Caching everywhere

Every API request is content-addressed by sha256(model, messages,
temperature, max_tokens, sample_index) and persisted as a JSON file
under ``artifacts/cache/teacher/<two-char-prefix>/<hash>.json``.
Iterating on prompt formatting is free if the formatted prompt didn't
change for a given example.

The cost meter is similarly persistent: spend accumulates across runs
until we explicitly clear the sidecar.

## Why mlx-lm and not transformers + PEFT?

PEFT/Transformers on MPS is very slow (the kernel set is incomplete,
operations fall back to CPU silently, and the LoRA training loop
allocates intermediate tensors that defeat the unified-memory
advantage). mlx-lm is Apple-Silicon native, hits the full memory
bandwidth, and trains LoRA at ~2.5 it/sec on Qwen2.5-0.5B on an M1 Pro
with batch size 1, grad accum 8, seq len 2048. This makes the project
runnable on a laptop.

## Why two configurations?

A single trained model is a number, not an experiment. The ablation
holds the rank/alpha/learning rate fixed and varies one knob — by
default whether reasoning-mode traces are included — so we can attribute
any quality difference to that knob. The README reports both.

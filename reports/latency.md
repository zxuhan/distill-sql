# Latency and cost

Cold-load and warm-steady-state numbers measured on a 16 GB M1 Pro, sampling 15 Spider dev examples per model. ``$/1K queries`` for OpenAI is computed from the cached prompt/completion token counts at posted Tier-1 prices ($0.15 / $0.60 per M tokens for gpt-4o-mini).

| model | kind | cold_load_s | warm_q_s | tokens/s | avg_in_tok | avg_out_tok | $/1K queries |
|---|---|---|---|---|---|---|---|
| base_qwen_0p5b | base | 3.3 | 0.61 | 60 | 464 | 37 | $0.00 |
| distilled_primary | lora | 1.9 | 0.65 | 31 | 464 | 20 | $0.00 |
| distilled_ablation_direct | lora | 1.6 | 0.65 | 32 | 464 | 21 | $0.00 |
| distilled_1p5b | lora | 3.2 | 1.59 | 14 | 464 | 23 | $0.00 |
| distilled_1p5b_q4 | base | 0.8 | 1.16 | 18 | 464 | 21 | $0.00 |
| distilled_3b | lora | 0.8 | 2.03 | 10 | 464 | 21 | $0.00 |
# Spider dev: full eval matrix

## Headline numbers

| model | n | exec | easy | medium | hard | extra | exact_match |
|---|---|---|---|---|---|---|---|
| base_qwen_0p5b | 1034 | 0.339 | 0.508 | 0.361 | 0.224 | 0.151 | 0.087 |
| distilled_ablation_direct | 1034 | 0.594 | 0.786 | 0.643 | 0.489 | 0.283 | 0.198 |
| distilled_primary | 1034 | 0.600 | 0.815 | 0.668 | 0.477 | 0.223 | 0.217 |
| distilled_1p5b | 1034 | 0.692 | 0.855 | 0.756 | 0.534 | 0.446 | 0.246 |

## Failure-mode breakdown

Bucketed by the in-process executor: ``ok`` means rows match gold; ``wrong-result`` parses and runs but disagrees with gold; ``execution`` raises a SQLite error; ``parse`` fails sqlglot.

| model | ok | wrong-result | execution | parse | empty |
|---|---|---|---|---|---|
| base_qwen_0p5b | 329 (32%) | 283 (27%) | 404 (39%) | 1 (0%) | 17 (2%) |
| distilled_primary | 575 (56%) | 308 (30%) | 144 (14%) | 3 (0%) | 4 (0%) |
| distilled_ablation_direct | 596 (58%) | 255 (25%) | 178 (17%) | 0 (0%) | 5 (0%) |
| distilled_1p5b | 670 (65%) | 281 (27%) | 83 (8%) | 0 (0%) | 0 (0%) |


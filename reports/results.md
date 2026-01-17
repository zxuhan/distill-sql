# Spider dev: full eval matrix

## Headline numbers

| model | n | exec | easy | medium | hard | extra | exact_match |
|---|---|---|---|---|---|---|---|
| base_qwen_0p5b | 1034 | 0.339 | 0.508 | 0.361 | 0.224 | 0.151 | 0.087 |

## Failure-mode breakdown

Bucketed by the in-process executor: ``ok`` means rows match gold; ``wrong-result`` parses and runs but disagrees with gold; ``execution`` raises a SQLite error; ``parse`` fails sqlglot.

| model | ok | wrong-result | execution | parse | empty |
|---|---|---|---|---|---|
| base_qwen_0p5b | 329 (32%) | 283 (27%) | 404 (39%) | 1 (0%) | 17 (2%) |


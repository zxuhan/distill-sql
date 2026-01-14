# Vendored: test-suite-sql-eval

Source: <https://github.com/taoyds/test-suite-sql-eval>
Pinned commit: `e97acc546ecbee8fa27fa8dbf025ef61493a876c`

This directory is a trimmed copy of the official Spider test-suite evaluator.
The wrapper that actually invokes it lives at `src/distill_sql/eval/runner.py`.

What was removed from the upstream copy:

- `classical_test.pkl`, `classical_provenance.ipynb`, `evaluation_examples/`,
  `tmp/`, `database/` — none are used by `evaluation.py` for execution accuracy
  on Spider dev.

Nothing was modified from upstream. If we need to re-pin, run:

```sh
rm -rf third_party/test-suite-sql-eval
git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval third_party/test-suite-sql-eval
rm -rf third_party/test-suite-sql-eval/.git \
       third_party/test-suite-sql-eval/classical_test.pkl \
       third_party/test-suite-sql-eval/classical_provenance.ipynb \
       third_party/test-suite-sql-eval/evaluation_examples \
       third_party/test-suite-sql-eval/tmp \
       third_party/test-suite-sql-eval/database
```

Their license (Apache 2.0) is preserved at `LICENSE` in this directory.

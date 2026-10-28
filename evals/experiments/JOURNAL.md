# Experiment journal

One entry per experiment, written **before** the next experiment starts. Tune on `dev` only.
The `test` split runs once per finalist (`--split test --final`).

Template:

## <date> · <experiment name>
- **Hypothesis:** what should change, and why (cite the error-analysis category).
- **Change:** config / prompt / code diff, one variable.
- **Run:** result file name.
- **Dev result:** pass@1 / pass^k / duplicate rate per roster size, vs the previous best.
- **Regression suite:** pass / fail.
- **Decision:** keep or revert, and why.

---

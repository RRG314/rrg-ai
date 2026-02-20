# Full Inventory Benchmark Report (2026-02-20)

## Scope

This report summarizes benchmark/eval harness behavior and current run outputs after enabling isolated run defaults.

## Suites

- `backend/system_check.py` (functional quality gate)
- `backend/eval.py` (local 20-50 task harness)
- `backend/industry_bench.py` (industry-style categories)

## Isolation Status

All suites now run isolated by default:

- unique run dir per execution: `.ai_data/evals/runs/<run_id>/`
- per-run DB inside run dir
- per-run local data/download workspace

Persistent DB mode remains available via:
- `--no-isolated --persist-db --db-path <path>`

## Latest Sample Outputs Reviewed

- System check report: `.ai_data/evals/system_check_system_check_1771561341_3ea342cd.json`
  - `score`: `100.0`
  - `isolated`: `true`
  - `run_id`: `system_check_1771561341_3ea342cd`
- Eval report: `.ai_data/evals/eval_eval_1771561335_65c53f8f.json`
  - `score`: `100.0`
  - `pass_rate`: `100.0`
  - `isolated`: `true`
  - `run_id`: `eval_1771561335_65c53f8f`
- Industry benchmark report: `.ai_data/evals/industry_bench_industry_bench_1771561324_80570e5b.json`
  - `score`: `100.0`
  - `pass_rate`: `100.0`
  - `isolated`: `true`
  - `run_id`: `industry_bench_1771561324_80570e5b`

## Interpretation Notes

- Isolation fixes prior score inflation risk from cross-run seeded-state reuse.
- `system_check` should be gated on `meets_target_all_systems` plus category-level checks.
- Scores are strong for current local harnesses, but represent this project's task suites (not universal AGI capability claims).

## Reproducibility Commands

```bash
python -m backend.system_check --min-score 95 --fail-below
python -m backend.eval --task-count 24
python -m backend.industry_bench --max-steps 8
```

To reuse a persistent DB for debugging:

```bash
python -m backend.system_check --no-isolated --persist-db --db-path .ai_data/system_check.sqlite3
python -m backend.eval --no-isolated --persist-db --db-path .ai_data/eval.sqlite3
python -m backend.industry_bench --no-isolated --persist-db --db-path .ai_data/industry_bench.sqlite3
```

## Recommended Next Benchmark Enhancements

1. Add fixed offline factual corpus tasks to reduce web variance.
2. Add multi-step code-repair tasks for stronger coding quality measurement.
3. Add adversarial grounding checks for strict/evidence mode robustness.
4. Track confidence calibration drift over time for adaptive policy updates.

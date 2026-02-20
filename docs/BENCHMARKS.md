# Benchmarks and Evaluation

The project includes three local suites:

- `backend/system_check.py` - functional quality gate for platform capabilities
- `backend/eval.py` - local task suite (20-50 tasks)
- `backend/industry_bench.py` - industry-style categories (MMLU/GSM8K/HumanEval-lite/ToolBench-style)

## Isolation (Default)

All suites run isolated by default:
- create unique run dir under `.ai_data/evals/runs/<run_id>/`
- use per-run SQLite DB
- use per-run data workspace

Report JSON includes:
- `run_id`
- `db_path`
- `data_dir`
- `isolated`

This prevents cross-run DB/data leakage from inflating scores.

## Commands

### System Check

```bash
python -m backend.system_check --min-score 95 --fail-below
```

Useful flags:
- `--min-score <float>`
- `--task-limit <n>`
- `--use-llm`
- `--no-isolated`
- `--persist-db`
- `--db-path <path>`

### Eval Harness

```bash
python -m backend.eval --task-count 24
```

Useful flags:
- `--task-count` (20..50)
- `--target-score`
- `--use-llm`
- `--no-isolated`
- `--persist-db`
- `--db-path <path>`

### Industry Benchmark

```bash
python -m backend.industry_bench --max-steps 8
```

Useful flags:
- `--target-score`
- `--prefer-local-core`
- `--use-llm`
- `--no-isolated`
- `--persist-db`
- `--db-path <path>`

## Persistent DB Mode (Debugging)

Use only when intentionally reproducing with fixed state:

```bash
python -m backend.system_check --no-isolated --persist-db --db-path .ai_data/system_check.sqlite3
python -m backend.eval --no-isolated --persist-db --db-path .ai_data/eval.sqlite3
python -m backend.industry_bench --no-isolated --persist-db --db-path .ai_data/industry_bench.sqlite3
```

## Report Location

- Reports: `.ai_data/evals/*.json`
- Run artifacts: `.ai_data/evals/runs/<run_id>/`

## Interpreting Quality Gates

For `system_check`, gate on:
- `meets_target_all_systems` (preferred)
- `categories_meet_target`
- `meets_target` (legacy aggregate)

For release decisions, use all-systems/category gates rather than aggregate-only score.

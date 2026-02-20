# Release Readiness Checklist

Status: **Prepared for pre-release review** (not yet released)

Last updated: 2026-02-20

## Scope

This checklist is for a professional release of the local-first AI platform, with no behavior changes required from documentation cleanup.

## 1) Product Documentation

- [x] Root `README.md` updated for clean onboarding
- [x] Architecture document updated
- [x] API reference updated
- [x] Security model documented
- [x] Operations/troubleshooting runbook documented
- [x] Benchmark/eval methodology documented
- [x] Inventory and benchmark snapshot reports refreshed

## 2) Security Baseline

- [x] Token auth enabled by default (`AI_REQUIRE_TOKEN=1`)
- [x] Localhost-only default CORS regex
- [x] Pairing gate for non-local origin bootstrap
- [x] File tool root defaults to repo root
- [x] Web tool target safety checks (private/loopback blocked)

## 3) Quality and Testing

- [x] Unit/integration tests pass locally
- [x] System check gate command documented
- [x] Eval and industry benchmark commands documented
- [x] Benchmark isolation defaults documented

## 4) Reproducibility

- [x] Eval/bench runs isolated by default
- [x] Reports include run metadata (`run_id`, `db_path`, `data_dir`, `isolated`)
- [x] Persistent DB mode available via explicit flags for debugging

## 5) Release Hygiene

- [x] Documentation index and navigation added
- [x] Commands use generic repo-local paths
- [x] Sensitive runtime data remains in `.ai_data/` and gitignored
- [x] No required cloud dependency for core operation
- [x] MIT license file present and referenced in README

## Suggested Final Gate Before Publishing

Run and confirm all pass:

```bash
PYTHONPATH=. pytest -q
python -m backend.system_check --min-score 95 --fail-below
python -m backend.eval --task-count 24
python -m backend.industry_bench --max-steps 8
```

If all checks pass and docs remain accurate, the repo is ready for release tagging.

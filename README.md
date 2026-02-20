# RRG AI

[![CI](https://github.com/RRG314/rrg-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RRG314/rrg-ai/actions/workflows/ci.yml)
[![Pages](https://github.com/RRG314/rrg-ai/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/RRG314/rrg-ai/actions/workflows/pages.yml)
[![Release](https://img.shields.io/github/v/release/RRG314/rrg-ai?display_name=tag)](https://github.com/RRG314/rrg-ai/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Local-first modular AI platform with an HTML chat UI and a Python backend.

The project is designed for private local use first: chat, retrieval, tools, evidence/provenance, adaptive planning, and benchmark/eval harnesses all run without cloud APIs.

## Highlights

- Chat UI with session persistence and agent traces
- Planner/executor agent loop (`/api/agent`) with retries, task state, and provenance
- Strict Fact Mode and Evidence Mode (`claim -> sources -> snippets -> confidence`)
- Local tools for web, files, documents, OCR, code execution/testing, and math
- Structured memory (`facts`, `preferences`, `outcomes`, `artifacts`)
- Adaptive post-task reflection and planning heuristic updates
- Recursive-Adic retrieval scoring integrated in storage ranking
- Local evaluation suites (`eval`, `industry_bench`, `system_check`)
- Benchmark run isolation by default (fresh DB/data per run)

## Novel Components and Claim Boundaries

This project includes original system combinations and research-driven integrations, with explicit boundaries on what is standard versus new:

- Novel in this implementation:
  - Recursive-Adic depth-weighted retrieval (`backend/recursive_adic.py`, `backend/storage.py`)
  - Evidence-enforced grounded answer rendering in agent mode
  - Adaptive post-task reflection with persisted planning heuristics and improvement rules
  - Default-isolated benchmark/eval execution for reproducible local scoring
- Standard building blocks used where appropriate:
  - FastAPI service layer, SQLite persistence, and local tool routing patterns
  - Optional local LLM inference via Ollama (not required for core tooling mode)
- Scope boundary:
  - This release does not claim solved AGI; it provides a modular local agent platform for grounded chat, tool use, and measurable iteration.

## Related Repositories and Research Links

Related repositories under the same account:

- [Recursive-Adic-Number-Field](https://github.com/RRG314/Recursive-Adic-Number-Field)
- [Recursive-Division-Tree-Algorithm--Preprint](https://github.com/RRG314/Recursive-Division-Tree-Algorithm--Preprint)
- [recursive-depth-integration-](https://github.com/RRG314/recursive-depth-integration-)
- [rdt-collatz-orthogonality](https://github.com/RRG314/rdt-collatz-orthogonality)
- [RDT-entropy](https://github.com/RRG314/RDT-entropy)
- [Entorpy-RAG](https://github.com/RRG314/Entorpy-RAG)
- [topological-adam](https://github.com/RRG314/topological-adam)
- [topological-neural-net](https://github.com/RRG314/topological-neural-net)

Matching Zenodo references used by/related to this stack:

- [The Recursive Adic Number Field: Construction Analysis and Recursive Depth Transforms](https://zenodo.org/records/17555644) (DOI: `10.5281/zenodo.17555644`)
- [Recursive Division Tree: A Log-Log Algorithm for Integer Depth](https://zenodo.org/records/18012166) (DOI: `10.5281/zenodo.18012166`)
- [Recursive-Depth Integration: A Depth-Weighted Measure and Integral on the Integers](https://zenodo.org/records/17753502) (DOI: `10.5281/zenodo.17753502`)
- [RDT-Based Zeta and Laplace Transforms: Analytic Framework for Depth-Weighted Number Theory](https://zenodo.org/records/17766570) (DOI: `10.5281/zenodo.17766570`)
- [Structural Entropy Analysis of Integer Depth via Recursive Division Tree](https://zenodo.org/records/17682287) (DOI: `10.5281/zenodo.17682287`)

## Architecture (At A Glance)

- Frontend: `index.html`, `app.js`, `style.css`
- API: `backend/app.py` (FastAPI)
- Agent: `backend/agent.py`
- Storage: `backend/storage.py` (SQLite)
- Skills: `backend/skills.py`
- Recursive learning: `backend/recursive_learning.py`
- Bench/eval: `backend/eval.py`, `backend/industry_bench.py`, `backend/system_check.py`
- Tools: `backend/tools/`

See full docs map in [`docs/README.md`](docs/README.md).

## Quick Start (Local Full Mode)

### 1) Start everything

```bash
./start_local_ai.sh
```

This script will:
- create `.venv` if needed
- install dependencies
- start backend on a free localhost port (default near `8000`)
- attempt to start/use Ollama locally (optional)
- open the UI automatically

### 2) Open the UI

Use the URL shown by the script (usually `http://127.0.0.1:8000`).

### 3) Chat

- Click `Auto Connect` if needed
- Keep `Run as Agent`, `Strict Fact Mode`, and `Evidence Mode` enabled for grounded outputs

## Manual Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

## Terminal Chat

```bash
source .venv/bin/activate
python -m backend.cli
```

## Optional Local Model (Ollama)

```bash
ollama pull llama3.2:3b
export AI_MODEL=llama3.2:3b
```

If no Ollama model is available, the system still works in local rules/tools mode.

## Core Endpoints

- `GET /api/health`
- `POST /api/chat`
- `POST /api/agent`
- `POST /api/upload`
- `POST /api/image/analyze`
- `POST /api/search`
- `POST /api/fetch`
- `POST /api/download`
- `POST /api/files/read`
- `POST /api/code/run`
- `POST /api/code/test`
- `POST /api/system-check`
- `GET /api/sessions`
- `GET /api/tasks`
- `GET /api/memory`

Full request/response reference: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md).

## Security Defaults

- Local bind by default (`127.0.0.1`)
- Token auth enabled by default (`AI_REQUIRE_TOKEN=1`)
- CORS default is localhost-only
- Pairing gate for non-local origins in `/api/bootstrap`
- File access rooted to repo by default (`AI_FILES_ROOT`)
- Web fetch/download block private/loopback targets by default

Security details and hardening guide: [`docs/SECURITY.md`](docs/SECURITY.md).

## Data Layout

Default base dir: `.ai_data/`

- Main DB: `.ai_data/ai.sqlite3`
- Downloads: `.ai_data/downloads/`
- Eval reports: `.ai_data/evals/`
- Isolated eval run data: `.ai_data/evals/runs/<run_id>/`

## Benchmarks and Evaluation

All benchmark/eval harnesses are isolated by default.

### Industry benchmark

```bash
python -m backend.industry_bench --max-steps 8
```

### System check (95% gate)

```bash
python -m backend.system_check --min-score 95 --fail-below
```

### Eval harness

```bash
python -m backend.eval --task-count 24
```

Reports include:
- `run_id`
- `db_path`
- `data_dir`
- `isolated`

Reuse a DB for debugging (non-isolated mode):

```bash
python -m backend.industry_bench --no-isolated --persist-db --db-path .ai_data/industry_bench.sqlite3
python -m backend.system_check --no-isolated --persist-db --db-path .ai_data/system_check.sqlite3
python -m backend.eval --no-isolated --persist-db --db-path .ai_data/eval.sqlite3
```

More detail: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Environment Variables

- `AI_DATA_DIR` (default `.ai_data`)
- `AI_FILES_ROOT` (default repo root)
- `AI_MODEL` (default `llama3.2:3b`)
- `AI_OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `AI_RECURSIVE_ADIC_RANKING` (`1`/`0`)
- `AI_RADF_ALPHA` (default `1.5`)
- `AI_RADF_BETA` (default `0.35`)
- `AI_REQUIRE_TOKEN` (`1`/`0`)
- `AI_ALLOWED_ORIGIN_REGEX` (default localhost-only regex)
- `AI_BOOTSTRAP_PAIRING_REQUIRED` (`1`/`0`)
- `AI_REPO_COLLECTION_ROOT`
- `AI_LEARNING_PDF_PATHS` (path-separated list)

## Testing

```bash
PYTHONPATH=. pytest -q
```

## Documentation

- [`docs/README.md`](docs/README.md) - documentation index
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - component and data flow map
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) - endpoint reference
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) - runbooks and troubleshooting
- [`docs/SECURITY.md`](docs/SECURITY.md) - security model and hardening
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) - eval/benchmark methodology
- [`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md) - release checklist/status
- [`docs/recursive_adic_novelty_review.md`](docs/recursive_adic_novelty_review.md)
- [`docs/recursive_adic_ai_systems.md`](docs/recursive_adic_ai_systems.md)

## License

MIT (see `LICENSE`).

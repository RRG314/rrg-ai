# Architecture Overview

## Design Goals

- Local-first execution and storage
- Modular agent + tool routing
- Grounded outputs with provenance/evidence
- Deterministic fallback when no model is available
- Reproducible evaluation with isolated benchmark runs

## High-Level Components

- Frontend UI: `index.html`, `app.js`, `style.css`
- API layer: `backend/app.py`
- Agent runtime: `backend/agent.py`
- Plugin runtime: `backend/plugins.py`
- Structured storage/retrieval: `backend/storage.py`
- Recursive learning: `backend/recursive_learning.py`
- Tool modules: `backend/tools/*`
- Local plugins: `plugins/*` (manifest + entrypoint)
- Skill pipelines: `backend/skills.py`
- Eval/bench suites: `backend/eval.py`, `backend/industry_bench.py`, `backend/system_check.py`

## Primary Runtime Flow (`POST /api/agent`)

1. `backend/app.py` validates request and builds `AgentRunConfig`.
2. `LocalAgent.run_agent(...)` initializes task record and planning state.
3. Planner emits a step sequence and routes steps to tool handlers.
4. Tool calls execute with retry policy and result summaries.
5. Provenance is collected from docs/web/files/tool results.
6. If Evidence Mode is enabled, evidence objects are built from provenance.
7. Final answer is composed and returned with plan/tool/citation trace.
8. Post-task reflection computes success/failure and writes heuristic updates.
9. Recursive learning layer records adaptation events with safety gating.

## Tool Routing Domains

- Web: search, fetch, download, dictionary
- Files: list/read/search (root-confined)
- Docs: retrieve from indexed chunks
- Code: generate/run/test
- Math: safe local expression eval
- Skills:
  - `skill.research_pipeline`
  - `skill.doc_pipeline`
  - `skill.folder_audit_pipeline`
- Plugins:
  - `plugin.list`
  - `plugin.run`

## Data Model (SQLite)

Main DB (`.ai_data/ai.sqlite3`) stores:

- Session/message memory: `sessions`, `messages`, `memory`
- Document index: `documents`, `chunks`
- Task trace: `tasks`
- Structured memory: `facts`, `preferences`, `outcomes`, `artifacts`
- Adaptive behavior: `planning_heuristics`, `improvement_rules`
- Recursive adaptation logs: `recursive_learning_events`

## Evidence and Provenance

The agent returns structured grounding artifacts:

- `provenance`: source metadata + snippets used during execution
- `citations`: URL/doc/file references
- `evidence`: claim-level grounding objects with confidence values

Strict Fact Mode + Evidence Mode ensures factual output is tied to collected snippets/sources.

## Benchmark Isolation Architecture

Eval harnesses use `backend/run_isolation.py`:

- Isolated mode (default):
  - create run dir: `.ai_data/evals/runs/<run_id>/`
  - use run-local DB and run-local data dirs
- Persistent mode (debug):
  - enabled with `--no-isolated --persist-db [--db-path ...]`

Reports are written to `.ai_data/evals/` and include:
- `run_id`
- `db_path`
- `data_dir`
- `isolated`

## Frontend Runtime Model

`app.js` maintains:
- local session state/cache for browser fallback
- backend connection and auth bootstrap state
- mode toggles (`strict facts`, `run as agent`, `evidence mode`)
- trace panels for plan/tool/provenance/evidence/adaptive updates

The UI can run in:
- static browser fallback mode (no backend tools)
- full local mode (backend-connected)

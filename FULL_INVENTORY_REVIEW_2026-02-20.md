# Full Inventory Review (2026-02-20)

## Snapshot

- Repository: `RRG314/ai`
- Local path: `<workspace>/ai`
- Branch: `steven/local-agent-stack`
- HEAD: `514162d`
- API routes: `23`
- SQLite tables: `13`
- Tests collected: `54`

## Executive Summary

The codebase is now organized as a local-first agent platform with clear module boundaries, grounded response controls (Strict Fact + Evidence modes), adaptive planning updates, and isolated benchmark execution by default.

Previously reported benchmark reproducibility risk from persistent benchmark DB reuse is now resolved through run isolation defaults and explicit persistence flags.

## Inventory

### Frontend

- `index.html`: local-first UI shell
- `app.js`: chat workflow, backend bootstrap, trace rendering, system-check UI
- `style.css`: visual styling
- `corpus.js`: static fallback corpus for browser-only mode

### Backend Core

- `backend/app.py`: API wiring, auth, pairing, CORS, endpoint handlers
- `backend/agent.py`: planner/executor loop, retries, evidence rendering, adaptive updates
- `backend/storage.py`: SQLite schema, retrieval, task/memory persistence
- `backend/recursive_learning.py`: recursive learning layer and policy gating
- `backend/recursive_adic.py`: recursive depth and weighting functions
- `backend/skills.py`: callable skill pipelines
- `backend/llm.py`: Ollama adapter
- `backend/cli.py`: terminal chat runner

### Benchmark and Eval

- `backend/system_check.py`: functional quality suite with category/system gates
- `backend/eval.py`: local eval harness (20-50 task suite)
- `backend/industry_bench.py`: industry-style benchmark suite
- `backend/run_isolation.py`: isolated benchmark run path resolver

### Tools

- `backend/tools/filesystem.py`: root-confined file operations
- `backend/tools/web.py`: search/fetch/download/definition with URL safety checks
- `backend/tools/docs.py`: text extraction for uploads/ingest
- `backend/tools/vision.py`: OCR and image analysis support
- `backend/tools/codeexec.py`: allowlisted command/test execution

### Documentation

- Root: `README.md`, `CONTRIBUTING.md`
- Docs index: `docs/README.md`
- Technical docs: architecture/API/security/ops/benchmarks/release readiness
- Research docs: recursive-adic novelty/systems documents

## Linkage Map

### Chat/Agent Path

1. UI submit -> `/api/agent`
2. `LocalAgent.run_agent(...)` creates/updates task record
3. Planner builds action sequence
4. Tool router executes actions with retry policy
5. Provenance/evidence artifacts are assembled
6. Response and trace are returned to UI
7. Post-task reflection writes adaptive heuristic updates
8. Recursive learning event is recorded with safety gating

### Storage Linkage

- Documents/chunks feed retrieval grounding
- Task table stores full execution trace
- Structured memory tables store user/system learning outcomes
- Adaptive tables (`planning_heuristics`, `improvement_rules`) influence future planning

### Benchmark Isolation Linkage

- Harnesses call `resolve_bench_paths(...)`
- Default isolated runs create per-run DB and data workspace
- Reports are emitted to `.ai_data/evals/` with run metadata

## Current Findings

### Resolved from prior review

- Benchmark DB persistence inflation risk: resolved (isolated-by-default runs)
- UI all-systems gate exposure: resolved (all-systems/category fields surfaced)
- URL safety concerns in web tools: mitigated by non-public/private target checks
- Default file root breadth risk: reduced (repo root default in backend app)
- Broad default CORS origin risk: reduced (localhost-only default regex)

### Remaining non-blocking release polish

- Add CI badges/versioning/release notes automation (process polish)
- Add optional signed release artifacts if publishing binaries in future

## Release Readiness Assessment

- Architecture clarity: ready
- Security baseline documentation: ready
- Reproducible eval/benchmark behavior: ready
- Local test coverage baseline: ready
- Developer onboarding docs: ready

See `docs/RELEASE_READINESS.md` for checklist form.

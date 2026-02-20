# Full Inventory + Benchmark Improvement Report (2026-02-20)

## 1) Snapshot
- Repo: `RRG314/ai`
- Branch: `steven/local-agent-stack`
- Head baseline before this patch batch: `83eb217`
- Test status: `36 passed` (`PYTHONPATH=. pytest -q`)
- Tracked files: `39` (before this patch batch)

## 2) Full Inventory

### Root App
- `index.html`: Local-first chat UI, toggles for strict facts / run-agent / evidence mode.
- `app.js`: Frontend controller, backend autodiscovery, API calls, trace rendering.
- `style.css`: UI styles.
- `corpus.js`: Browser-only fallback corpus.
- `README.md`: setup, security, endpoints, eval docs.
- `start_local_ai.sh`, `start_local_ai.command`: launch scripts.
- `requirements.txt`: Python dependencies.

### Backend Core
- `backend/app.py`: FastAPI server, token auth, CORS, endpoints.
- `backend/agent.py`: planner/executor loop, tool routing, retries, evidence rendering, adaptive learning.
- `backend/storage.py`: SQLite schema + session/task/memory/doc/artifact/heuristic persistence.
- `backend/llm.py`: Ollama adapter.
- `backend/recursive_adic.py`: recursive depth and depth-Laplace weighting.
- `backend/skills.py`: `research_pipeline`, `doc_pipeline`, `folder_audit_pipeline`.
- `backend/eval.py`: local eval harness (20-50 tasks).
- `backend/industry_bench.py`: industry-aligned benchmark suite.
- `backend/cli.py`: terminal chat.

### Tooling Modules
- `backend/tools/filesystem.py`: root-confined list/read/search/write.
- `backend/tools/web.py`: web search/fetch/download/dictionary.
- `backend/tools/docs.py`: upload extraction (text/pdf/docx/image OCR).
- `backend/tools/vision.py`: OCR + optional prompt-based image analysis.
- `backend/tools/codeexec.py`: allowlisted local command/test execution.

### Tests
- `tests/` now includes:
  - agent loop/evidence/adaptive tests
  - storage + new heuristic/rule table tests
  - API security tests (including OPTIONS preflight)
  - docs/vision/llm/web parsing tests
  - code execution tests (including unquoted spaced path coalescing)

### Docs
- `docs/recursive_adic_novelty_review.md`
- `docs/recursive_adic_ai_systems.md`
- `FULL_INVENTORY_REVIEW_2026-02-20.md` (prior deep security/linkage review)

## 3) End-to-End Linkage Map

### UI -> Agent
- `app.js` sends `/api/agent` with:
  - strict facts / evidence mode / run-agent defaults ON
  - `allow_web`, `allow_files`, `allow_docs`, `allow_code`, `allow_downloads`
  - `prefer_local_core: true`

### API -> Planner
- `/api/agent` in `backend/app.py` builds `AgentRunConfig` and calls `LocalAgent.run_agent`.

### Planner -> Tools
- `backend/agent.py` routes actions to:
  - Web: `web.search`, `web.fetch`, `web.download`, `web.dictionary`, `web.search.auto`
  - Files: `files.list`, `files.read`, `files.search`
  - Docs: `docs.retrieve`
  - Skills: `skill.research_pipeline`, `skill.doc_pipeline`, `skill.folder_audit_pipeline`
  - Code: `code.generate`, `code.run`, `code.test`
  - Math: `math.eval`
  - Finalize: `answer.compose`

### Tools -> Storage
- Every run stores task state (`tasks`), messages (`messages`), provenance, artifacts, outcomes.
- Evidence mode maps claims -> source/snippet/confidence.
- Adaptive loop stores planning heuristics + improvement rules after task completion.

## 4) Data Model Inventory (SQLite)

Tables:
- `sessions`, `messages`, `memory`
- `documents`, `chunks`
- `tasks`
- `facts`, `preferences`, `outcomes`, `artifacts`
- `planning_heuristics`
- `improvement_rules`

## 5) Industry Benchmark Results

### Suite
- Runner: `python -m backend.industry_bench --max-steps 8`
- Output: `.ai_data/evals/industry_bench_<timestamp>.json`
- Categories:
  - `mmlu_style`
  - `gsm8k_style`
  - `humaneval_lite`
  - `toolbench_style`

### Baseline (before this improvement batch)
- Report: `.ai_data/evals/industry_bench_1771556775.json`
- Score: `70.96`
- Pass rate: `69.23%`

### After improvements
- Report: `.ai_data/evals/industry_bench_1771557380.json`
- Score: `91.92`
- Pass rate: `96.15%`

### Delta
- Overall score: `+20.96`
- Pass rate: `+26.92%`
- Category deltas:
  - `gsm8k_style`: `10.0 -> 90.0` (`+80.0`)
  - `toolbench_style`: `89.17 -> 100.0` (`+10.83`)
  - `humaneval_lite`: `90.0 -> 90.0` (`0.0`)
  - `mmlu_style`: `88.75 -> 88.75` (`0.0`)

Remaining failing task in latest run:
- `mmlu_human_heart`

Note:
- The `mmlu_style` portion currently relies on live web search, so score variance can occur across runs due external search/index changes.

## 6) What Was Improved to Raise Scores

Implemented changes:
- Added planner math tool (`math.eval`) with safe AST evaluation.
- Improved command parsing for `run command ...` prompts:
  - strips trailing `in/at <cwd>`
  - handles chained phrasing (`and then ...`)
- Reordered code planning so generation executes before tests/commands when both are requested.
- Improved template code generator with deterministic handlers for common requests:
  - function-return patterns
  - factorial
  - fibonacci
  - palindrome
- Added command path coalescing in code executor for unquoted paths with spaces.
- Added benchmark runner (`backend/industry_bench.py`) and tests for new behavior.

## 7) Priority Gaps / Next Improvements

1. Factual QA robustness for edge questions
- Add a second-pass grounding step when first-pass evidence confidence is low.

2. True coding benchmark depth
- Add multi-step “generate -> test -> fix -> retest” loop (SWE-bench-lite style) with iterative patching.

3. Benchmark stability and reproducibility
- Add an offline factual benchmark pack (fixed local corpus) to reduce live-web variance.

4. Security hardening still recommended (from prior report)
- tighten bootstrap token exposure model
- add outbound URL safety restrictions
- narrow default `AI_FILES_ROOT`

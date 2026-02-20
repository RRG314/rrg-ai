# RRG AI

Local-first modular AI system with an HTML chat UI plus a Python backend for:

- Natural chat with persistent session memory
- Internet search and website content extraction
- URL downloading to local disk
- Dictionary definitions from a live source (`dictionaryapi.dev`)
- Document upload/indexing (TXT/MD/PDF/DOCX)
- Image OCR + image analysis prompt flow (PNG/JPG/WebP/TIFF/etc.)
- Local file browsing, reading, and text search
- Local code command execution and test running (allowlisted commands)
- Local math expression evaluation (`calculate ...`, `compute ...`) via planner tool
- Optional local LLM via Ollama (no cloud dependency required)
- Recursive-Adic retrieval ranking (depth-Laplace weighted chunk scoring)
- Planner/executor agent loop with task state, trace logs, and provenance
- Evidence mode outputs (`claim -> sources -> snippets -> confidence`)
- Callable local skills: `research_pipeline`, `doc_pipeline`, `folder_audit_pipeline`
- Structured memory tables: `facts`, `preferences`, `outcomes`, `artifacts`
- Adaptive planner loop: post-task success/failure analysis, heuristic updates, and stored improvement rules

## Modes

1. Static browser mode (GitHub Pages):
- Works with no backend
- Keeps browser memory and built-in corpus retrieval
- No direct internet/file/upload tools

2. Local full mode (recommended):
- Run the Python backend locally
- UI can use all tools + persistent SQLite memory
- Optional Ollama model for higher quality chat

## Quick Start (Local Full Mode)

Fastest path (no manual backend URL entry):

```bash
cd /Users/stevenreid/Documents/New\ project/repo_audit/rrg314/ai
./start_local_ai.sh
```

macOS double-click launcher: `start_local_ai.command`

No-code path:
1. Double-click `start_local_ai.command`
2. Wait for the browser to open
3. Click `Auto Connect`
4. Start chatting

After it starts:
1. Open the URL it prints.
2. Click `Auto Connect`.
3. Chat normally.
4. For image OCR, install Tesseract once: `brew install tesseract`

Or run manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The UI is served by the backend itself and connects automatically.
If port `8000` is busy, `start_local_ai.sh` automatically chooses the next free port.

If you still want a separate static frontend server, you can use one, and the UI will auto-detect the backend (`same origin`, `127.0.0.1:8000`, `localhost:8000`).
You only need `Connect Backend` if you want to override URL manually.

Strict Fact Mode is enabled by default in the UI.
Run as Agent and Evidence Mode are enabled by default in the UI.
Recursive-Adic retrieval ranking is enabled by default (`AI_RECURSIVE_ADIC_RANKING=1`).

Simple example prompts:
- `define entropy`
- `what does theorem mean`
- `search the web for latest retrieval benchmarks`
- `read file /Users/stevenreid/Documents/paper.txt`
- `read website https://example.com`

## Terminal Chat

```bash
cd /Users/stevenreid/Documents/New\ project/repo_audit/rrg314/ai
source .venv/bin/activate
python -m backend.cli
```

## Optional: Local Model (Ollama)

If Ollama is installed, you can run a local model for stronger chat quality:

```bash
ollama pull llama3.2:3b
```

Set model via env var if needed:

```bash
export AI_MODEL=llama3.2:3b
```

If no model is available, the system still works in rules mode with tools.
`start_local_ai.sh` now tries to start Ollama and pull `AI_MODEL` automatically.

## API Endpoints

- `GET /api/health`
- `POST /api/chat`
- `POST /api/agent` (multi-step planner/executor)
- `POST /api/upload`
- `POST /api/ingest-url`
- `POST /api/image/analyze` (multipart: `file`, optional `prompt`, optional `session_id`)
- `POST /api/search`
- `POST /api/define`
- `POST /api/fetch`
- `POST /api/download`
- `GET /api/files/list?path=.`
- `POST /api/files/read`
- `POST /api/files/search`
- `POST /api/code/run`
- `POST /api/code/test`
- `GET /api/sessions`
- `GET /api/tasks`
- `GET /api/tasks/{task_id}`
- `GET /api/memory?session_id=<id>`
- `GET /api/docs`

## Data Location

By default, backend data is stored in:

- `.ai_data/ai.sqlite3` (chat memory + document index)
- `.ai_data/downloads/` (downloaded files)
- `.ai_data/evals/` (eval reports)

`tasks` are persisted in SQLite with:
- `task_id`, `session_id`, `title`, `status`
- `created_at`, `updated_at`
- `steps_json`, `outputs_json`, `provenance_json`

Structured memory tables:
- `facts` (`key`, `value`, `source`, timestamps)
- `preferences` (`key`, `value`, `source`, timestamps)
- `outcomes` (`title`, `summary`, `status`, `score`)
- `artifacts` (`artifact_type`, `location`, `source`, `doc_id`, `description`)
- `planning_heuristics` (`key`, `value`, `source`, `updated_at`)
- `improvement_rules` (`session_id`, `task_id`, `rule`, `trigger`, `confidence`)

Configurable via env vars:

- `AI_DATA_DIR`
- `AI_FILES_ROOT`
- `AI_MODEL`
- `AI_OLLAMA_URL`
- `AI_RECURSIVE_ADIC_RANKING` (`1`/`0`)
- `AI_RADF_ALPHA` (default `1.5`)
- `AI_RADF_BETA` (default `0.35`)
- `AI_REQUIRE_TOKEN` (`1` default, recommended)
- `AI_ALLOWED_ORIGIN_REGEX` (CORS allowlist regex)

## GitHub Pages

GitHub Pages can host the HTML UI, but full local tooling needs a Python backend.
Static pages alone cannot securely provide unrestricted file-system and document-processing capabilities.

## Privacy and Security (Important)

- Your private data (`.ai_data`, uploaded docs, chat memory, evals) stays on your local machine and is not pushed to GitHub Pages.
- Backend runs on `127.0.0.1` by default (local-only binding).
- API token auth is enabled by default (`AI_REQUIRE_TOKEN=1`):
  - token is auto-generated locally in `.ai_data/api_token.txt`
  - frontend fetches it automatically from `/api/bootstrap`
  - no manual token copy is required
- CORS is restricted by regex (not `*`) so random websites cannot read your backend responses.
- The UI now only auto-connects to local backends (`localhost` / `127.0.0.1`).

If someone else uses the same GitHub Pages UI, they use their own local backend/data, not yours.

## Recursive-Adic Integration

The backend now uses a Recursive-Adic depth proxy in retrieval scoring:

- Recursive depth (RDT recurrence): `R(1)=0`, `R(n)=1+min_{1<=k<n}((R(k)+R(n-k))/alpha)`.
- Depth-Laplace weighting: `w(n)=exp(-beta*R(n))`, clamped to a minimum for stability.
- Final chunk score: `base_lexical_score * w(rank_index)`.

This means retrieved context is not only keyword-matched but also depth-weighted, giving you a concrete, active integration of the Recursive-Adic framework into chat grounding.

Further design/novelty notes:
- `docs/recursive_adic_novelty_review.md`
- `docs/recursive_adic_ai_systems.md`

## Agent API Trace Shape

`POST /api/agent` returns:
- `answer`
- `plan` (step list + status)
- `tool_calls` (`name`, `args`, `attempt`, `status`, `result_summary`)
- `citations`
- `provenance` (urls/files/doc ids + snippets)
- `evidence` objects in Evidence Mode
- `memory` (facts/preferences/outcomes/artifacts snapshot)
- `adaptive_update` (task success analysis + learned heuristic deltas + improvement rule id)

Core flags:
- `strict_fact_mode` / `strict_facts`
- `evidence_mode`
- `allow_web`, `allow_files`, `allow_docs`, `allow_code`, `allow_downloads`
- `max_steps`

Skill tools the planner can call:
- `skill.research_pipeline`
- `skill.doc_pipeline`
- `skill.folder_audit_pipeline`

Code tools the planner can call:
- `code.generate`
- `code.run`
- `code.test`
- `math.eval`

## Industry Benchmark (Local)

Run an industry-aligned local suite (MMLU-style, GSM8K-style, HumanEval-lite, ToolBench-style):

```bash
cd /Users/stevenreid/Documents/New\ project/repo_audit/rrg314/ai
source .venv/bin/activate
python -m backend.industry_bench --max-steps 8
```

Report is written to `.ai_data/evals/industry_bench_<timestamp>.json`.

Evidence mode behavior:
- Every returned claim must include a source and snippet from stored provenance.
- If no grounded source+snippet is available, no factual claim is emitted.
- By default `prefer_local_core=true`, so planner answers use local pipelines/original systems first and do not depend on Ollama output.

## Eval Harness

Run:

```bash
cd /Users/stevenreid/Documents/New\ project/repo_audit/rrg314/ai
source .venv/bin/activate
python -m backend.eval --task-count 24
```

This runs a 20-50 task local suite and writes JSON reports to `.ai_data/evals/`.
Use `--task-count` (20..50) and `--use-llm` to include model-based generation.

# Operations Runbook

## Startup (Recommended)

```bash
./start_local_ai.sh
```

What this handles:
- venv bootstrap + dependency install
- backend start on available localhost port
- optional Ollama start/pull
- opens browser UI

## Manual Startup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

## Logs and Data

- Backend log: `.ai_data/backend.log`
- Ollama log (if started by script): `.ai_data/ollama.log`
- Main DB: `.ai_data/ai.sqlite3`
- Eval reports: `.ai_data/evals/`
- Isolated benchmark run dirs: `.ai_data/evals/runs/<run_id>/`

## Plugins

- Default plugin root: `plugins/`
- Override root: `AI_PLUGINS_DIR=/path/to/plugins`
- List plugins via API:
  ```bash
  curl -s -H \"X-AI-Token: <token>\" http://127.0.0.1:8000/api/plugins | jq .
  ```

## Health Check

```bash
curl -s http://127.0.0.1:8000/api/health | jq .
```

Key fields:
- `model_available`
- `model_reason`
- `backend`
- `auth_required`
- recursive-adic config fields

## Common Issues

### 1) "Backend not reachable"

Checks:
1. Confirm backend is running:
   ```bash
   lsof -nP -iTCP:8000 -sTCP:LISTEN
   ```
2. Open health endpoint in browser or curl.
3. In UI, click `Auto Connect`.
4. If backend started on a different port, use that exact URL.

### 2) `model_available: false`

This means Ollama/model is not ready. The app still works in tools/rules mode.

To enable model responses:

```bash
ollama serve
ollama pull llama3.2:3b
```

Then restart backend or rerun `./start_local_ai.sh`.

### 3) OCR unavailable

Install Tesseract:

```bash
brew install tesseract
```

### 4) Pairing required from external origin

If using GitHub Pages UI or other non-local origin, provide pairing code from:

- `.ai_data/pairing_code.txt`
- or backend startup console output

### 5) Unauthorized API requests

Ensure `x-ai-token` header is set to the value from local bootstrap flow (`/api/bootstrap`) when auth is enabled.

## Quality Commands

Run full test suite:

```bash
PYTHONPATH=. pytest -q
```

Run system check gate:

```bash
python -m backend.system_check --min-score 95 --fail-below
```

Run industry benchmark:

```bash
python -m backend.industry_bench --max-steps 8
```

Run eval harness:

```bash
python -m backend.eval --task-count 24
```

## Benchmark Isolation Controls

Default: isolated run (fresh DB + fresh run data)

Persistent DB mode (debugging):

```bash
python -m backend.eval --no-isolated --persist-db --db-path .ai_data/eval.sqlite3
python -m backend.industry_bench --no-isolated --persist-db --db-path .ai_data/industry_bench.sqlite3
python -m backend.system_check --no-isolated --persist-db --db-path .ai_data/system_check.sqlite3
```

## Suggested Release Validation Sequence

1. `PYTHONPATH=. pytest -q`
2. `python -m backend.system_check --min-score 95 --fail-below`
3. `python -m backend.eval --task-count 24`
4. `python -m backend.industry_bench --max-steps 8`
5. Review reports in `.ai_data/evals/`

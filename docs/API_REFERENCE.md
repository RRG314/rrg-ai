# API Reference

Base URL (local default): `http://127.0.0.1:8000`

All non-exempt `/api/*` routes require `x-ai-token` when `AI_REQUIRE_TOKEN=1`.

## Auth/Bootstrap

### `GET /api/health`
Returns backend/model/runtime status.

### `GET /api/bootstrap`
Returns API token for local UI bootstrap (pairing required for non-local origins).

### `GET /api/pairing-code`
Returns pairing code for local origin only.

## Chat and Agent

### `POST /api/chat`
Simple chat path.

Request:
```json
{
  "message": "Explain recursive-adic ranking",
  "session_id": "optional",
  "strict_facts": false
}
```

Response (shape):
```json
{
  "session_id": "...",
  "answer": "...",
  "mode": "local|llm|local-evidence",
  "tool_events": []
}
```

### `POST /api/agent`
Planner/executor loop with tool routing and trace.

Request:
```json
{
  "message": "Summarize sharks with sources",
  "session_id": "optional",
  "strict_fact_mode": true,
  "strict_facts": true,
  "evidence_mode": true,
  "allow_web": true,
  "allow_files": true,
  "allow_docs": true,
  "allow_code": true,
  "allow_plugins": true,
  "allow_downloads": false,
  "prefer_local_core": true,
  "max_steps": 8
}
```

Response (shape):
```json
{
  "session_id": "...",
  "task_id": "...",
  "answer": "...",
  "plan": [],
  "tool_calls": [],
  "citations": [],
  "provenance": [],
  "evidence": [],
  "memory": {},
  "adaptive_update": {},
  "routing": {},
  "rdt_shell_alignment": {},
  "original_work_used": {}
}
```

## Session/Task/Memory

### `GET /api/sessions`
Returns recent sessions.

### `GET /api/tasks?session_id=<id>&limit=<n>`
Returns task records.

### `GET /api/tasks/{task_id}`
Returns one task record with steps/outputs/provenance.

### `GET /api/memory?session_id=<id>`
Returns structured memory snapshot.

### `GET /api/docs`
Returns indexed documents.

## Plugin Endpoints

### `GET /api/plugins`
Lists installed plugins.

Response (shape):
```json
{
  "count": 1,
  "plugins_dir": "/abs/path/plugins",
  "plugins": [
    {
      "plugin_id": "text_tools",
      "name": "Text Tools",
      "version": "1.0.0",
      "description": "...",
      "entrypoint": "...",
      "enabled": true
    }
  ]
}
```

### `POST /api/plugins/run`
Runs one plugin with payload.

Request:
```json
{
  "plugin_id": "text_tools",
  "input": { "text": "recursive adic depth transforms" },
  "session_id": "optional",
  "timeout_sec": 90
}
```

Response includes:
- `ok`, `status`, `summary`, `text`
- `provenance`, `artifacts`
- `doc_id`, `session_id`

## Documents and Vision

### `POST /api/upload`
Multipart file upload; extracts text and indexes as document.

Form field: `file`

### `POST /api/image/analyze`
Multipart image analysis + OCR.

Form fields:
- `file` (required)
- `prompt` (optional)
- `session_id` (optional)

### `POST /api/ingest-url`
Fetches URL text and stores as document.

Request:
```json
{ "url": "https://example.com" }
```

## Web Utilities

### `POST /api/search`
DuckDuckGo HTML search parsing.

### `POST /api/define`
Dictionary lookup via `dictionaryapi.dev`.

### `POST /api/fetch`
Fetches and extracts URL text/pdf/html.

### `POST /api/download`
Downloads URL to local downloads dir.

## File Tools

### `GET /api/files/list?path=.`
List files/dirs under allowed root.

### `POST /api/files/read`
Request:
```json
{ "path": "README.md", "max_chars": 25000 }
```

### `POST /api/files/search`
Request:
```json
{ "query": "radf", "path": "." }
```

## Code Tools

### `POST /api/code/run`
Request:
```json
{ "command": "python -V", "cwd": ".", "timeout_sec": 90 }
```

### `POST /api/code/test`
Request:
```json
{ "cwd": ".", "runner": "auto", "timeout_sec": 120 }
```

## Quality Gate

### `POST /api/system-check`
Runs local system quality checks.

Request:
```json
{
  "min_score": 95.0,
  "use_llm": false,
  "max_steps": 8,
  "task_limit": 0
}
```

Response includes:
- `score`, `target_score`
- `meets_target`, `categories_meet_target`, `meets_target_all_systems`
- `categories`, `results`
- run metadata (`run_id`, `db_path`, `data_dir`, `isolated`)

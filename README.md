# RRG AI

Local-first modular AI system with an HTML chat UI plus a Python backend for:

- Natural chat with persistent session memory
- Internet search and website content extraction
- URL downloading to local disk
- Document upload/indexing (TXT/MD/PDF/DOCX)
- Local file browsing, reading, and text search
- Optional local LLM via Ollama (no cloud dependency required)

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

```bash
cd /Users/stevenreid/Documents/New\ project/repo_audit/rrg314/ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open `index.html` with a local web server, for example:

```bash
python3 -m http.server 5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173), set backend URL to `http://127.0.0.1:8000`, and click **Connect Backend**.

## Terminal Chat

```bash
cd /Users/stevenreid/Documents/New\ project/repo_audit/rrg314/ai
source .venv/bin/activate
python -m backend.cli
```

## Optional: Local Model (Ollama)

If Ollama is installed, you can run a local model for stronger chat quality:

```bash
ollama pull llama3.1:8b
```

Set model via env var if needed:

```bash
export AI_MODEL=llama3.1:8b
```

If no model is available, the system still works in rules mode with tools.

## API Endpoints

- `GET /api/health`
- `POST /api/chat`
- `POST /api/upload`
- `POST /api/ingest-url`
- `POST /api/search`
- `POST /api/fetch`
- `POST /api/download`
- `GET /api/files/list?path=.`
- `POST /api/files/read`
- `POST /api/files/search`
- `GET /api/sessions`
- `GET /api/docs`

## Data Location

By default, backend data is stored in:

- `.ai_data/ai.sqlite3` (chat memory + document index)
- `.ai_data/downloads/` (downloaded files)

Configurable via env vars:

- `AI_DATA_DIR`
- `AI_FILES_ROOT`
- `AI_MODEL`
- `AI_OLLAMA_URL`

## GitHub Pages

GitHub Pages can host the HTML UI, but full local tooling needs a Python backend.
Static pages alone cannot securely provide unrestricted file-system and document-processing capabilities.

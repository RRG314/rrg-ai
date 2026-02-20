from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentRunConfig, LocalAgent
from .llm import OllamaClient
from .storage import SQLiteStore
from .tools.docs import extract_text_from_bytes
from .tools.filesystem import FileBrowser
from .tools.vision import analyze_image_bytes, supports_image_ocr
from .tools.web import dictionary_define, download_url, fetch_url_text, search_web


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("AI_DATA_DIR", ROOT / ".ai_data"))
DOWNLOADS_DIR = DATA_DIR / "downloads"
DB_PATH = DATA_DIR / "ai.sqlite3"
FILES_ROOT = Path(os.getenv("AI_FILES_ROOT", str(Path.home()))).expanduser().resolve()
OLLAMA_MODEL = os.getenv("AI_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434")
RECURSIVE_ADIC_RANKING = os.getenv("AI_RECURSIVE_ADIC_RANKING", "1").lower() not in {"0", "false", "no"}
RADF_BETA = float(os.getenv("AI_RADF_BETA", "0.35"))
RADF_ALPHA = float(os.getenv("AI_RADF_ALPHA", "1.5"))
REQUIRE_TOKEN = os.getenv("AI_REQUIRE_TOKEN", "1").lower() not in {"0", "false", "no"}
ALLOW_ORIGIN_REGEX = os.getenv(
    "AI_ALLOWED_ORIGIN_REGEX",
    r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$|^https://rrg314\.github\.io$",
)
TOKEN_PATH = DATA_DIR / "api_token.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_create_token(path: Path) -> str:
    env_token = os.getenv("AI_API_TOKEN", "").strip()
    if env_token:
        return env_token
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    return token


API_TOKEN = _load_or_create_token(TOKEN_PATH)

store = SQLiteStore(
    DB_PATH,
    use_recursive_adic=RECURSIVE_ADIC_RANKING,
    radf_beta=RADF_BETA,
    radf_alpha=RADF_ALPHA,
)
files = FileBrowser(FILES_ROOT)
llm = OllamaClient(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=DOWNLOADS_DIR)

app = FastAPI(title="RRG AI Local Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_EXEMPT_PATHS = {"/api/health", "/api/bootstrap"}


@app.middleware("http")
async def api_auth_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
    path = request.url.path
    if REQUIRE_TOKEN and path.startswith("/api/") and path not in AUTH_EXEMPT_PATHS:
        token = request.headers.get("x-ai-token", "").strip()
        if token != API_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: invalid API token"})
    return await call_next(request)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    strict_facts: bool = False


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    strict_fact_mode: bool = False
    strict_facts: bool | None = None
    evidence_mode: bool = False
    allow_web: bool = True
    allow_files: bool = True
    allow_docs: bool = True
    allow_downloads: bool = False
    max_steps: int = 8


class URLRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str


class DefineRequest(BaseModel):
    word: str


class FileReadRequest(BaseModel):
    path: str
    max_chars: int = 25000


class FileSearchRequest(BaseModel):
    query: str
    path: str = "."


@app.get("/api/health")
def health() -> dict[str, object]:
    status = llm.status()
    ocr_ok, ocr_reason = supports_image_ocr()
    return {
        "ok": True,
        "backend": "local-python",
        "files_root": str(FILES_ROOT),
        "model": status.model,
        "model_available": status.available,
        "model_reason": status.reason,
        "recursive_adic_ranking": RECURSIVE_ADIC_RANKING,
        "radf_beta": RADF_BETA,
        "radf_alpha": RADF_ALPHA,
        "image_ocr_available": ocr_ok,
        "image_ocr_reason": ocr_reason,
        "auth_required": REQUIRE_TOKEN,
        "allow_origin_regex": ALLOW_ORIGIN_REGEX,
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, object]:
    return {
        "ok": True,
        "auth_required": REQUIRE_TOKEN,
        "api_token": API_TOKEN if REQUIRE_TOKEN else "",
        "backend": "local-python",
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, object]:
    try:
        return agent.chat(req.session_id, req.message, strict_facts=req.strict_facts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/agent")
def run_agent(req: AgentRequest) -> dict[str, object]:
    strict_facts = bool(req.strict_facts) if req.strict_facts is not None else bool(req.strict_fact_mode)
    cfg = AgentRunConfig(
        strict_facts=strict_facts,
        evidence_mode=bool(req.evidence_mode),
        allow_web=bool(req.allow_web),
        allow_files=bool(req.allow_files),
        allow_docs=bool(req.allow_docs),
        allow_downloads=bool(req.allow_downloads),
        max_steps=int(req.max_steps),
    )
    try:
        return agent.run_agent(req.session_id, req.message, config=cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions")
def sessions() -> dict[str, object]:
    return {"sessions": store.list_sessions(limit=200)}


@app.get("/api/tasks")
def tasks(session_id: str | None = None, limit: int = 100) -> dict[str, object]:
    try:
        return {"tasks": store.list_tasks(session_id=session_id, limit=max(1, min(limit, 500)))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}")
def task(task_id: str) -> dict[str, object]:
    item = store.get_task(task_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return {
        "task": {
            "task_id": item.task_id,
            "session_id": item.session_id,
            "title": item.title,
            "status": item.status,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "steps": item.steps,
            "outputs": item.outputs,
            "provenance": item.provenance,
        }
    }


@app.get("/api/memory")
def memory(session_id: str) -> dict[str, object]:
    try:
        return {"session_id": session_id, **store.memory_snapshot(session_id, limit=300)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/docs")
def docs() -> dict[str, object]:
    return {"documents": store.list_documents(limit=200)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, object]:
    try:
        data = await file.read()
        text, kind = extract_text_from_bytes(file.filename or "upload.bin", data)
        if not text.strip():
            raise ValueError("Extracted text is empty")
        doc_id = store.add_document(name=file.filename or "upload", source="upload", kind=kind, text=text)
        return {
            "ok": True,
            "doc_id": doc_id,
            "name": file.filename,
            "kind": kind,
            "char_count": len(text),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/image/analyze")
async def image_analyze(
    file: UploadFile = File(...),
    prompt: str = Form(default=""),
    session_id: str | None = Form(default=None),
) -> dict[str, object]:
    try:
        data = await file.read()
        result = analyze_image_bytes(file.filename or "image-upload", data, prompt=prompt, llm=llm)

        doc_id: str | None = None
        ocr_text = str(result.get("ocr_text", "")).strip()
        if ocr_text:
            doc_id = store.add_document(
                name=file.filename or "image-upload",
                source="image-upload",
                kind="image-ocr",
                text=ocr_text,
            )
            if session_id:
                store.upsert_memory(session_id, "last_image_doc", file.filename or "image-upload")

        return {
            "ok": True,
            "doc_id": doc_id,
            "session_id": session_id,
            **result,
            "ocr_char_count": len(ocr_text),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ingest-url")
def ingest_url(req: URLRequest) -> dict[str, object]:
    try:
        text, kind = fetch_url_text(req.url)
        if not text.strip():
            raise ValueError("No text extracted from URL")
        doc_id = store.add_document(name=req.url, source=req.url, kind=f"web-{kind}", text=text)
        return {
            "ok": True,
            "doc_id": doc_id,
            "url": req.url,
            "kind": kind,
            "char_count": len(text),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/search")
def search(req: SearchRequest) -> dict[str, object]:
    try:
        results = search_web(req.query, max_results=8)
        return {"query": req.query, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/define")
def define(req: DefineRequest) -> dict[str, object]:
    try:
        return dictionary_define(req.word, max_definitions=8)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/fetch")
def fetch(req: URLRequest) -> dict[str, object]:
    try:
        text, kind = fetch_url_text(req.url)
        return {
            "url": req.url,
            "kind": kind,
            "text": text,
            "char_count": len(text),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/download")
def download(req: URLRequest) -> dict[str, object]:
    try:
        info = download_url(req.url, DOWNLOADS_DIR)
        return {"ok": True, **info}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/files/list")
def list_files(path: str = ".") -> dict[str, object]:
    try:
        return {"root": str(files.root), "path": path, "entries": files.list_dir(path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/files/read")
def read_file(req: FileReadRequest) -> dict[str, object]:
    try:
        text = files.read_text(req.path, max_chars=req.max_chars)
        return {"path": req.path, "text": text, "char_count": len(text)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/files/search")
def search_files(req: FileSearchRequest) -> dict[str, object]:
    try:
        hits = files.search_text(req.query, req.path)
        return {"query": req.query, "path": req.path, "hits": hits}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Serve frontend on the same backend port so no separate web server is required.
app.mount("/", StaticFiles(directory=ROOT, html=True), name="frontend")

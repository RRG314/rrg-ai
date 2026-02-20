from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import LocalAgent
from .llm import OllamaClient
from .storage import SQLiteStore
from .tools.docs import extract_text_from_bytes
from .tools.filesystem import FileBrowser
from .tools.web import dictionary_define, download_url, fetch_url_text, search_web


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("AI_DATA_DIR", ROOT / ".ai_data"))
DOWNLOADS_DIR = DATA_DIR / "downloads"
DB_PATH = DATA_DIR / "ai.sqlite3"
FILES_ROOT = Path(os.getenv("AI_FILES_ROOT", str(Path.home()))).expanduser().resolve()
OLLAMA_MODEL = os.getenv("AI_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434")

DATA_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

store = SQLiteStore(DB_PATH)
files = FileBrowser(FILES_ROOT)
llm = OllamaClient(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=DOWNLOADS_DIR)

app = FastAPI(title="RRG AI Local Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


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
    return {
        "ok": True,
        "backend": "local-python",
        "files_root": str(FILES_ROOT),
        "model": status.model,
        "model_available": status.available,
        "model_reason": status.reason,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, object]:
    try:
        return agent.chat(req.session_id, req.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions")
def sessions() -> dict[str, object]:
    return {"sessions": store.list_sessions(limit=200)}


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

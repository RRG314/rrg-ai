from pathlib import Path

from backend.skills import doc_pipeline, folder_audit_pipeline, research_pipeline
from backend.storage import SQLiteStore
from backend.tools.filesystem import FileBrowser


def test_doc_pipeline_returns_provenance(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    store.add_document(name="seed", source="seed", kind="text", text="depth laplace weighting with provenance")

    out = doc_pipeline("depth laplace", store=store, limit=5)
    assert out.status == "ok"
    assert out.provenance
    assert "Doc pipeline" in out.context_block


def test_folder_audit_pipeline_scans_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "a.txt").write_text("hello", encoding="utf-8")
    (sub / "b.md").write_text("world", encoding="utf-8")

    store = SQLiteStore(tmp_path / "ai.sqlite3")
    files = FileBrowser(root)

    out = folder_audit_pipeline(".", files=files, store=store, max_entries=100, max_depth=2)
    assert out.status == "ok"
    assert "Folder audit" in out.context_block
    assert out.provenance


def test_research_pipeline_local_fallback(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    store.add_document(name="seed", source="seed", kind="text", text="research fallback local documents")

    def fail_search(query, max_results=6):  # type: ignore[no-untyped-def]
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("backend.skills.search_web", fail_search)

    out = research_pipeline("research fallback", store=store, max_results=4, fetch_top=1)
    assert out.status == "ok"
    assert out.provenance
    assert "fallback" in out.detail.lower()

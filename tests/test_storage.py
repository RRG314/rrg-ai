from pathlib import Path

from backend.storage import SQLiteStore


def test_session_memory_and_docs(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")

    sid = store.ensure_session(None, "demo")
    store.add_message(sid, "user", "my name is steven")
    store.upsert_memory(sid, "name", "steven")
    store.upsert_memory(sid, "goal", "build local ai")

    memory = store.memory_for_session(sid)
    assert memory
    assert any(f.key == "name" and f.value == "steven" for f in memory)

    doc_id = store.add_document(
        name="paper.txt",
        source="upload",
        kind="text",
        text="topological optimizer stability and convergence improvements",
    )
    assert doc_id

    hits = store.search_chunks("optimizer convergence", limit=3)
    assert hits
    assert "optimizer" in hits[0]["text"].lower()
    assert "radf_weight" in hits[0]
    assert "radf_depth" in hits[0]

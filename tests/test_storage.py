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
    assert "doc_id" in hits[0]


def test_tasks_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    sid = store.ensure_session(None, "task session")

    task_id = store.create_task(
        sid,
        title="demo task",
        status="running",
        steps=[{"step_id": 1, "title": "Collect context", "status": "pending"}],
        outputs={"mode": "rules-agent"},
        provenance=[{"source_type": "doc", "source": "seed", "snippet": "hello"}],
    )
    assert task_id

    store.update_task(
        task_id,
        status="completed",
        steps=[{"step_id": 1, "title": "Collect context", "status": "done"}],
        outputs={"mode": "rules-agent", "answer": "ok"},
        provenance=[{"source_type": "doc", "source": "seed", "snippet": "hello world"}],
    )

    item = store.get_task(task_id)
    assert item is not None
    assert item.status == "completed"
    assert item.outputs.get("answer") == "ok"
    assert item.steps and item.steps[0].get("status") == "done"

    listed = store.list_tasks(session_id=sid, limit=10)
    assert listed
    assert listed[0]["task_id"] == task_id


def test_structured_memory_tables(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    sid = store.ensure_session(None, "memory session")

    store.upsert_fact(sid, "goal", "build local ai", source="user")
    store.upsert_preference(sid, "style", "concise", source="user")
    outcome_id = store.add_outcome(sid, title="task 1", summary="done", status="completed", score=0.75)
    artifact_id = store.add_artifact(
        sid,
        artifact_type="file",
        location="/tmp/demo.txt",
        source="/tmp/demo.txt",
        doc_id="doc-1",
        description="demo artifact",
    )

    assert outcome_id > 0
    assert artifact_id > 0

    snapshot = store.memory_snapshot(sid, limit=20)
    assert snapshot["facts"]
    assert snapshot["preferences"]
    assert snapshot["outcomes"]
    assert snapshot["artifacts"]

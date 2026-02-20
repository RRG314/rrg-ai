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


def test_adaptive_heuristics_and_rules_tables(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    sid = store.ensure_session(None, "adaptive session")

    defaults = {
        "retry_attempts": 2.0,
        "docs_priority": 0.62,
        "web_priority": 0.56,
        "planner_confidence": 0.5,
    }
    initial = store.get_planning_heuristics(defaults=defaults)
    assert initial["retry_attempts"] == 2.0
    assert initial["docs_priority"] == 0.62

    store.upsert_planning_heuristic("retry_attempts", 2.75, source="adaptive-agent")
    store.upsert_planning_heuristic("planner_confidence", 0.35, source="adaptive-agent")
    loaded = store.get_planning_heuristics(defaults=defaults)
    assert loaded["retry_attempts"] == 2.75
    assert loaded["planner_confidence"] == 0.35

    rid = store.add_improvement_rule(
        session_id=sid,
        task_id="task-123",
        rule="Failure pattern: prioritize docs before web and increase retries.",
        trigger="no_grounded_provenance",
        confidence=0.81,
    )
    assert rid > 0

    rules = store.list_improvement_rules(sid, limit=10)
    assert rules
    assert rules[0].task_id == "task-123"
    assert "prioritize docs" in rules[0].rule.lower()

    snapshot = store.memory_snapshot(sid, limit=20)
    assert snapshot["planning_heuristics"]
    assert snapshot["improvement_rules"]


def test_recursive_learning_events_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    sid = store.ensure_session(None, "recursive session")

    event_id = store.add_recursive_learning_event(
        session_id=sid,
        task_id="task-rl-1",
        layer="recursive-learning-v1",
        score=0.87,
        depth=2.4,
        dual_grad=0.11,
        hyper_delta=0.0003,
        heuristic_updates={"docs_priority": {"old": 0.62, "new": 0.66, "delta": 0.04}},
        metrics={"step_completion": 1.0, "tool_error_rate": 0.0},
    )
    assert event_id > 0

    events = store.list_recursive_learning_events(sid, limit=10)
    assert events
    first = events[0]
    assert first.task_id == "task-rl-1"
    assert first.layer == "recursive-learning-v1"
    assert "docs_priority" in first.heuristic_updates

    snapshot = store.memory_snapshot(sid, limit=20)
    assert snapshot["recursive_learning_events"]


def test_search_chunks_matches_doc_name_and_split_tokens(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ai.sqlite3")
    store.add_document(
        name="eval-doc:radf_core",
        source="eval-seed",
        kind="text",
        text="Depth-Laplace weighting improves retrieval ranking.",
    )
    hits_name = store.search_chunks("doc pipeline for radf_core", limit=5)
    assert hits_name
    assert str(hits_name[0].get("doc_name") or "").endswith("radf_core")

    hits_split = store.search_chunks("radf-core", limit=5)
    assert hits_split

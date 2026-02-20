from pathlib import Path

from backend.recursive_learning import RecursiveLearningLayer
from backend.storage import SQLiteStore
from backend.tools.filesystem import FileBrowser


def test_recursive_learning_bootstrap_and_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    files_root = tmp_path / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repos"
    repo = repo_root / "novel_ai"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / ".git").mkdir(parents=True, exist_ok=True)
    (repo / "algo_notes.md").write_text(
        "Recursive adic depth transforms with dual number policy updates and hyperreal micro-steps.",
        encoding="utf-8",
    )

    store = SQLiteStore(db)
    files = FileBrowser(files_root)
    layer = RecursiveLearningLayer(
        store=store,
        files=files,
        repo_root=repo_root,
        pdf_paths=[],
        max_bootstrap_files=40,
    )
    sid = store.ensure_session(None, "rl test")

    out = layer.run_pipeline(session_id=sid, query="recursive optimizer", force_bootstrap=True)
    assert out.get("status") == "ok"
    bootstrap = out.get("bootstrap") or {}
    assert int(bootstrap.get("repo_docs") or 0) >= 1
    docs = store.list_documents(limit=20)
    assert any(str(item.get("kind") or "") == "repo-algorithm" for item in docs)

    out2 = layer.ensure_bootstrap(session_id=sid, force=False)
    assert out2.get("status") in {"already_bootstrapped", "bootstrapped"}


def test_recursive_learning_adapt_persists_event_and_rule(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    files_root = tmp_path / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db)
    files = FileBrowser(files_root)
    layer = RecursiveLearningLayer(
        store=store,
        files=files,
        repo_root=tmp_path / "empty-repos",
        pdf_paths=[],
    )
    sid = store.ensure_session(None, "rl adapt")
    task_id = store.create_task(sid, title="adaptive task")

    result = layer.adapt(
        session_id=sid,
        task_id=task_id,
        user_text="improve recursive planning with stronger evidence",
        strict_facts=True,
        evidence_mode=True,
        plan=[{"status": "done"}, {"status": "done"}],
        tool_calls=[{"name": "docs.retrieve", "status": "ok"}],
        provenance=[{"source": "seed-doc", "snippet": "recursive adic evidence", "doc_id": "doc-1"}],
        evidence=[{"claim": "Seed claim", "sources": ["seed-doc"], "snippets": ["recursive adic evidence"]}],
        heuristics={"retry_attempts": 2.0, "docs_priority": 0.62, "web_priority": 0.56, "planner_confidence": 0.5},
        baseline_score=0.9,
    )
    assert result.get("event_id")
    assert result.get("improvement_rule_id")

    events = store.list_recursive_learning_events(sid, limit=5)
    assert events
    assert events[0].layer == "recursive-learning-v1"
    rules = store.list_improvement_rules(sid, limit=5)
    assert rules


def test_recursive_learning_policy_gate_blocks_bad_traces(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    files_root = tmp_path / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db)
    files = FileBrowser(files_root)
    layer = RecursiveLearningLayer(
        store=store,
        files=files,
        repo_root=tmp_path / "empty-repos",
        pdf_paths=[],
    )
    sid = store.ensure_session(None, "rl gate")
    task_id = store.create_task(sid, title="bad task")

    before = store.get_planning_heuristics(defaults={"docs_priority": 0.62, "web_priority": 0.56})
    result = layer.adapt(
        session_id=sid,
        task_id=task_id,
        user_text="bad run with no evidence",
        strict_facts=True,
        evidence_mode=True,
        plan=[{"status": "error"}],
        tool_calls=[{"name": "web.search.auto", "status": "error"}],
        provenance=[],
        evidence=[],
        heuristics={
            "retry_attempts": 2.0,
            "docs_priority": 0.62,
            "web_priority": 0.56,
            "planner_confidence": 0.5,
        },
        baseline_score=0.31,
    )
    gate = result.get("policy_gate") or {}
    assert gate.get("allow_update") is False
    assert gate.get("blocked_reasons")
    assert result.get("heuristic_updates") == {}

    after = store.get_planning_heuristics(defaults={"docs_priority": 0.62, "web_priority": 0.56})
    # Failed traces must not directly write new recursive policy updates.
    assert round(float(after["docs_priority"]), 4) == round(float(before["docs_priority"]), 4)
    assert round(float(after["web_priority"]), 4) == round(float(before["web_priority"]), 4)

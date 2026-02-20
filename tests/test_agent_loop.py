from dataclasses import dataclass
from pathlib import Path

from backend.agent import AgentRunConfig, LocalAgent
from backend.storage import SQLiteStore
from backend.tools.filesystem import FileBrowser


@dataclass
class _Status:
    available: bool
    model: str
    reason: str


class _FakeLLM:
    def status(self) -> _Status:
        return _Status(available=False, model="none", reason="offline")

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        return ""


def test_agent_loop_builds_trace_and_evidence(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    root = tmp_path / "files"
    root.mkdir(parents=True, exist_ok=True)

    file_path = root / "notes.txt"
    file_path.write_text("Planner loop writes tool traces and provenance entries.", encoding="utf-8")

    store = SQLiteStore(db)
    files = FileBrowser(root)
    agent = LocalAgent(store=store, files=files, llm=_FakeLLM(), downloads_dir=tmp_path / "downloads")

    cfg = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        allow_web=False,
        allow_files=True,
        allow_docs=True,
        max_steps=8,
    )

    res = agent.run_agent(None, f"read file {file_path}", config=cfg)

    assert res.get("task_id")
    assert res.get("session_id")
    assert isinstance(res.get("plan"), list) and res["plan"]
    assert isinstance(res.get("tool_calls"), list) and res["tool_calls"]
    assert isinstance(res.get("provenance"), list) and res["provenance"]
    assert isinstance(res.get("evidence"), list) and res["evidence"]
    for item in res["evidence"]:
        if not isinstance(item, dict):
            continue
        sources = item.get("sources") or []
        snippets = item.get("snippets") or []
        if item.get("confidence", 0) > 0:
            assert sources
            assert snippets

    task = store.get_task(str(res["task_id"]))
    assert task is not None
    assert task.status == "completed"

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


class _AvailableLLM:
    def __init__(self) -> None:
        self.chat_calls = 0

    def status(self) -> _Status:
        return _Status(available=True, model="stub-model", reason="")

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        self.chat_calls += 1
        return "stub response"


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
    answer = str(res.get("answer") or "")
    assert "Here's a grounded answer based on collected evidence:" in answer
    assert "Claim:" not in answer
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


def test_evidence_mode_prefers_local_core_even_with_available_llm(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    root = tmp_path / "files"
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / "notes.txt"
    file_path.write_text("Evidence comes from snippets and sources in local files.", encoding="utf-8")

    store = SQLiteStore(db)
    files = FileBrowser(root)
    llm = _AvailableLLM()
    agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=tmp_path / "downloads")

    cfg = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        prefer_local_core=True,
        allow_web=False,
        allow_files=True,
        allow_docs=True,
        max_steps=8,
    )

    res = agent.run_agent(None, f"read file {file_path}", config=cfg)
    assert res.get("mode") == "local-evidence"
    assert res.get("llm_used") is False
    assert llm.chat_calls == 0
    meta = res.get("original_work_used") or {}
    assert meta.get("prefer_local_core") is True


def test_agent_adaptive_update_persists_rule_and_heuristics(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    root = tmp_path / "files"
    root.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db)
    files = FileBrowser(root)
    agent = LocalAgent(store=store, files=files, llm=_FakeLLM(), downloads_dir=tmp_path / "downloads")

    cfg = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        prefer_local_core=True,
        allow_web=False,
        allow_files=False,
        allow_docs=True,
        max_steps=8,
    )

    res = agent.run_agent(None, "tell me about sharks", config=cfg)
    adaptive = res.get("adaptive_update")
    assert isinstance(adaptive, dict)
    assert "task_success" in adaptive
    assert "heuristic_updates" in adaptive
    assert adaptive.get("improvement_rule_id")

    sid = str(res.get("session_id") or "")
    assert sid
    rules = store.list_improvement_rules(sid, limit=5)
    assert rules
    heuristics = store.get_planning_heuristics()
    assert "planner_confidence" in heuristics


def test_agent_math_eval_tool(tmp_path: Path) -> None:
    db = tmp_path / "ai.sqlite3"
    root = tmp_path / "files"
    root.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db)
    files = FileBrowser(root)
    agent = LocalAgent(store=store, files=files, llm=_FakeLLM(), downloads_dir=tmp_path / "downloads")

    cfg = AgentRunConfig(
        strict_facts=False,
        evidence_mode=False,
        prefer_local_core=True,
        allow_web=False,
        allow_files=False,
        allow_docs=False,
        allow_code=True,
        max_steps=8,
    )

    res = agent.run_agent(None, "calculate (19 * 19) - 7", config=cfg)
    tool_calls = res.get("tool_calls") or []
    assert any((c.get("name") == "math.eval" and c.get("status") == "ok") for c in tool_calls if isinstance(c, dict))
    prov = res.get("provenance") or []
    joined = " ".join(str(p.get("snippet") or "") for p in prov if isinstance(p, dict))
    assert "354" in joined

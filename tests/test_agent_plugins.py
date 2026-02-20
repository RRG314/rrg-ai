from pathlib import Path

from backend.agent import AgentRunConfig, LocalAgent
from backend.plugins import PluginManager
from backend.storage import SQLiteStore
from backend.tools.filesystem import FileBrowser


class _NoLLM:
    class _Status:
        available = False
        model = "none"
        reason = "offline"

    def status(self):  # type: ignore[no-untyped-def]
        return self._Status()

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        return ""


def _make_plugin(root: Path) -> None:
    pdir = root / "mini"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.json").write_text(
        "{\"id\":\"mini\",\"name\":\"Mini\",\"version\":\"1.0.0\",\"entrypoint\":\"run.py\",\"enabled\":true}\n",
        encoding="utf-8",
    )
    (pdir / "run.py").write_text(
        "import json,sys\n"
        "data=json.loads(sys.stdin.read() or '{}')\n"
        "msg=str(data.get('input') or '')\n"
        "print(json.dumps({'status':'ok','summary':'mini ok','text':'mini:'+msg}))\n",
        encoding="utf-8",
    )


def test_agent_can_route_plugin_run(tmp_path: Path) -> None:
    plugins_root = tmp_path / "plugins"
    _make_plugin(plugins_root)

    store = SQLiteStore(tmp_path / "ai.sqlite3")
    files_root = tmp_path / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    files = FileBrowser(files_root)
    manager = PluginManager(plugins_root)
    agent = LocalAgent(store=store, files=files, llm=_NoLLM(), downloads_dir=tmp_path / "downloads", plugins=manager)

    cfg = AgentRunConfig(
        strict_facts=False,
        evidence_mode=False,
        allow_web=False,
        allow_files=False,
        allow_docs=True,
        allow_code=False,
        allow_plugins=True,
        max_steps=8,
    )

    out = agent.run_agent(None, "run plugin mini with hello", config=cfg)
    tool_calls = out.get("tool_calls") or []
    assert any(
        isinstance(call, dict)
        and call.get("name") == "plugin.run"
        and call.get("status") == "ok"
        for call in tool_calls
    )
    provenance = out.get("provenance") or []
    assert any(str(p.get("source") or "").startswith("plugin:") for p in provenance if isinstance(p, dict))

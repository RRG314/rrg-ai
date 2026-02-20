from pathlib import Path

from backend.plugins import PluginManager


def _make_plugin(root: Path, plugin_id: str = "demo_plugin") -> Path:
    pdir = root / plugin_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.json").write_text(
        (
            "{\n"
            f"  \"id\": \"{plugin_id}\",\n"
            "  \"name\": \"Demo Plugin\",\n"
            "  \"version\": \"1.0.0\",\n"
            "  \"description\": \"returns deterministic output\",\n"
            "  \"entrypoint\": \"run.py\",\n"
            "  \"enabled\": true\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    (pdir / "run.py").write_text(
        "import json,sys\n"
        "payload=json.loads(sys.stdin.read() or '{}')\n"
        "msg = payload.get('input')\n"
        "if isinstance(msg, dict):\n"
        "    msg = msg.get('text')\n"
        "msg = str(msg or '')\n"
        "out = {'status':'ok','summary':'demo ok','text':f'demo:{msg}','provenance':[{'source':'plugin:demo','snippet':msg}]}\n"
        "print(json.dumps(out))\n",
        encoding="utf-8",
    )
    return pdir


def test_plugin_manager_discovers_and_runs(tmp_path: Path) -> None:
    plugins_root = tmp_path / "plugins"
    _make_plugin(plugins_root)

    manager = PluginManager(plugins_root)
    rows = manager.list_plugins()
    assert rows
    assert rows[0]["plugin_id"] == "demo_plugin"

    out = manager.run_plugin("demo_plugin", payload={"text": "hello plugins"})
    assert out.status == "ok"
    assert "demo:" in out.text
    assert out.provenance


def test_plugin_manager_missing_plugin_raises(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins")
    try:
        manager.run_plugin("not_found", payload="x")
    except ValueError as exc:
        assert "Plugin not found" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing plugin")

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLUGIN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


@dataclass
class PluginSpec:
    plugin_id: str
    name: str
    version: str
    description: str
    directory: Path
    entrypoint: Path
    timeout_sec: int
    enabled: bool
    allow_in_agent: bool
    input_schema: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "directory": str(self.directory),
            "entrypoint": str(self.entrypoint),
            "timeout_sec": self.timeout_sec,
            "enabled": self.enabled,
            "allow_in_agent": self.allow_in_agent,
            "input_schema": self.input_schema,
        }


@dataclass
class PluginRunResult:
    plugin_id: str
    status: str
    summary: str
    text: str
    provenance: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    raw: dict[str, Any]


class PluginManager:
    def __init__(self, plugins_root: Path) -> None:
        self.root = Path(plugins_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, PluginSpec] = {}
        self.refresh()

    def refresh(self) -> list[PluginSpec]:
        found: dict[str, PluginSpec] = {}
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            manifest = directory / "plugin.json"
            if not manifest.exists():
                continue
            spec = self._load_manifest(directory, manifest)
            if spec is None:
                continue
            found[spec.plugin_id] = spec

        self._plugins = found
        return [found[k] for k in sorted(found)]

    def list_plugins(self, include_disabled: bool = False, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh:
            self.refresh()
        rows = []
        for plugin_id in sorted(self._plugins):
            spec = self._plugins[plugin_id]
            if not include_disabled and not spec.enabled:
                continue
            rows.append(spec.as_dict())
        return rows

    def get_plugin(self, plugin_id: str) -> PluginSpec | None:
        clean = (plugin_id or "").strip()
        if not clean:
            return None
        return self._plugins.get(clean)

    def run_plugin(
        self,
        plugin_id: str,
        payload: Any = None,
        context: dict[str, Any] | None = None,
        timeout_sec: int | None = None,
    ) -> PluginRunResult:
        spec = self.get_plugin(plugin_id)
        if spec is None:
            raise ValueError(f"Plugin not found: {plugin_id}")
        if not spec.enabled:
            raise ValueError(f"Plugin is disabled: {plugin_id}")

        command = self._command_for(spec)
        effective_timeout = int(timeout_sec if timeout_sec is not None else spec.timeout_sec)
        effective_timeout = max(1, min(effective_timeout, 300))

        envelope = {
            "input": payload,
            "context": context or {},
            "plugin": {
                "plugin_id": spec.plugin_id,
                "name": spec.name,
                "version": spec.version,
            },
        }
        stdin_data = json.dumps(envelope)

        try:
            proc = subprocess.run(
                command,
                input=stdin_data,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                cwd=spec.directory,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Plugin timed out after {effective_timeout}s: {spec.plugin_id}") from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()[:1500]
            raise RuntimeError(
                f"Plugin exited with code {proc.returncode}: {spec.plugin_id}" + (f" | {stderr}" if stderr else "")
            )

        stdout = (proc.stdout or "").strip()
        parsed: Any
        if not stdout:
            parsed = {"status": "ok", "summary": f"Plugin {spec.plugin_id} completed", "text": ""}
        else:
            parsed = _safe_json_loads(stdout)
            if parsed is None:
                parsed = {
                    "status": "ok",
                    "summary": f"Plugin {spec.plugin_id} returned text output",
                    "text": stdout,
                }

        if not isinstance(parsed, dict):
            parsed = {
                "status": "ok",
                "summary": f"Plugin {spec.plugin_id} returned non-object output",
                "text": str(parsed),
            }

        status = str(parsed.get("status") or "ok").strip().lower()
        if status not in {"ok", "error"}:
            status = "ok"

        summary = str(parsed.get("summary") or f"Plugin {spec.plugin_id} completed").strip()
        text = str(parsed.get("text") or parsed.get("output") or "").strip()
        provenance = _normalize_obj_list(parsed.get("provenance"))
        artifacts = _normalize_obj_list(parsed.get("artifacts"))

        return PluginRunResult(
            plugin_id=spec.plugin_id,
            status=status,
            summary=summary,
            text=text,
            provenance=provenance,
            artifacts=artifacts,
            raw=parsed,
        )

    def _load_manifest(self, directory: Path, manifest: Path) -> PluginSpec | None:
        payload = _safe_json_loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None

        plugin_id = str(payload.get("id") or directory.name).strip()
        if not PLUGIN_ID_RE.match(plugin_id):
            return None

        entry_rel = str(payload.get("entrypoint") or "run.py").strip()
        if not entry_rel:
            return None

        entrypoint = (directory / entry_rel).resolve()
        if not _is_within(entrypoint, directory.resolve()):
            return None
        if not entrypoint.exists() or not entrypoint.is_file():
            return None

        timeout_sec = int(payload.get("timeout_sec") or 45)
        timeout_sec = max(1, min(timeout_sec, 300))

        input_schema = payload.get("input_schema")
        if not isinstance(input_schema, dict):
            input_schema = {}

        return PluginSpec(
            plugin_id=plugin_id,
            name=str(payload.get("name") or plugin_id),
            version=str(payload.get("version") or "0.1.0"),
            description=str(payload.get("description") or ""),
            directory=directory.resolve(),
            entrypoint=entrypoint,
            timeout_sec=timeout_sec,
            enabled=bool(payload.get("enabled", True)),
            allow_in_agent=bool(payload.get("allow_in_agent", True)),
            input_schema=input_schema,
        )

    def _command_for(self, spec: PluginSpec) -> list[str]:
        entry = spec.entrypoint
        if entry.suffix.lower() == ".py":
            return [sys.executable, str(entry)]
        if os.access(entry, os.X_OK):
            return [str(entry)]
        raise ValueError(f"Entrypoint is not executable: {entry}")


def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        return Path(os.path.commonpath([str(path.resolve()), str(root.resolve())])) == root.resolve()
    except Exception:
        return False


def _normalize_obj_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
    return out

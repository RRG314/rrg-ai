from __future__ import annotations

import os
import secrets
import shutil
import time
from pathlib import Path


def make_run_dir(prefix: str) -> Path:
    base_data_dir = _default_data_dir()
    return _make_run_dir_with_base(prefix, base_data_dir)


def resolve_bench_paths(
    prefix: str,
    isolated: bool,
    db_path: str | Path | None,
    data_dir: str | Path | None,
) -> tuple[Path, Path, str]:
    base_data_dir = _resolve_base_data_dir(data_dir)

    if bool(isolated):
        run_dir = _make_run_dir_with_base(prefix, base_data_dir)
        return run_dir / f"{prefix}.sqlite3", run_dir, run_dir.name

    if db_path:
        resolved_db = Path(db_path).expanduser().resolve()
        resolved_db.parent.mkdir(parents=True, exist_ok=True)
        effective_data_dir = _resolve_base_data_dir(data_dir) if data_dir else resolved_db.parent
        effective_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        effective_data_dir = base_data_dir
        resolved_db = effective_data_dir / f"{prefix}.sqlite3"

    run_id = f"{prefix}_shared_{int(time.time())}_{secrets.token_hex(3)}"
    return resolved_db, effective_data_dir, run_id


def cleanup_run_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _default_data_dir() -> Path:
    raw = os.getenv("AI_DATA_DIR", ".ai_data").strip() or ".ai_data"
    out = Path(raw).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_base_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return _default_data_dir()
    out = Path(data_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_run_dir_with_base(prefix: str, base_data_dir: Path) -> Path:
    runs_root = base_data_dir / "evals" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{prefix}_{int(time.time())}_{secrets.token_hex(4)}"
    out = runs_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    return out

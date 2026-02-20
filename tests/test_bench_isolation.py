import json
import sqlite3
from pathlib import Path

from backend import eval as eval_harness
from backend.industry_bench import run_suite as run_industry_suite


def _table_exists(db_path: Path, table: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
    return bool(row)


def test_bench_isolation_creates_new_db(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    files_root = tmp_path / "files"

    out1 = run_industry_suite(
        data_dir=data_dir,
        files_root=files_root,
        model="none",
        ollama_url="http://127.0.0.1:11434",
        use_llm=False,
        max_steps=3,
        task_count=1,
        target_score=95.0,
        isolated=True,
        persist_db=False,
        db_path=None,
    )
    out2 = run_industry_suite(
        data_dir=data_dir,
        files_root=files_root,
        model="none",
        ollama_url="http://127.0.0.1:11434",
        use_llm=False,
        max_steps=3,
        task_count=1,
        target_score=95.0,
        isolated=True,
        persist_db=False,
        db_path=None,
    )

    p1 = json.loads(out1.read_text(encoding="utf-8"))
    p2 = json.loads(out2.read_text(encoding="utf-8"))
    db1 = Path(str(p1["db_path"]))
    db2 = Path(str(p2["db_path"]))

    assert p1["isolated"] is True
    assert p2["isolated"] is True
    assert db1 != db2
    assert db1.exists()
    assert db2.exists()
    assert _table_exists(db1, "sessions")
    assert _table_exists(db2, "sessions")


def test_persist_db_flag_reuses_db(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    files_root = tmp_path / "files"
    fixed_db = tmp_path / "shared" / "industry.sqlite3"

    out1 = run_industry_suite(
        data_dir=data_dir,
        files_root=files_root,
        model="none",
        ollama_url="http://127.0.0.1:11434",
        use_llm=False,
        max_steps=3,
        task_count=1,
        target_score=95.0,
        isolated=False,
        persist_db=True,
        db_path=fixed_db,
    )
    out2 = run_industry_suite(
        data_dir=data_dir,
        files_root=files_root,
        model="none",
        ollama_url="http://127.0.0.1:11434",
        use_llm=False,
        max_steps=3,
        task_count=1,
        target_score=95.0,
        isolated=False,
        persist_db=True,
        db_path=fixed_db,
    )

    p1 = json.loads(out1.read_text(encoding="utf-8"))
    p2 = json.loads(out2.read_text(encoding="utf-8"))
    assert p1["isolated"] is False
    assert p2["isolated"] is False
    assert Path(str(p1["db_path"])) == fixed_db.resolve()
    assert Path(str(p2["db_path"])) == fixed_db.resolve()
    assert fixed_db.exists()


def test_report_contains_run_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    files_root = tmp_path / "files"
    out = eval_harness.run_suite(
        data_dir=data_dir,
        files_root=files_root,
        model="none",
        ollama_url="http://127.0.0.1:11434",
        use_llm=False,
        max_steps=4,
        task_count=3,
        target_score=95.0,
        isolated=True,
        persist_db=False,
        db_path=None,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload.get("run_id"), str) and payload["run_id"]
    assert isinstance(payload.get("db_path"), str) and payload["db_path"]
    assert isinstance(payload.get("data_dir"), str) and payload["data_dir"]
    assert payload.get("isolated") is True

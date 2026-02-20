from pathlib import Path

from backend.system_check import run_system_check


def test_system_check_report_shape(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    files_root = tmp_path / "files"
    out, payload = run_system_check(
        data_dir=data_dir,
        files_root=files_root,
        model="none",
        ollama_url="http://127.0.0.1:11434",
        use_llm=False,
        max_steps=6,
        min_score=95.0,
        task_limit=6,
    )

    assert out.exists()
    assert payload["suite"] == "system_check_v1"
    assert payload["check_count"] >= 6
    assert payload["score"] >= 0.0
    assert payload["score"] <= 100.0
    assert isinstance(payload["categories"], dict)
    assert isinstance(payload["results"], list)
    assert payload["results"]
    assert isinstance(payload.get("run_id"), str) and payload["run_id"]
    assert isinstance(payload.get("db_path"), str) and payload["db_path"]
    assert isinstance(payload.get("data_dir"), str) and payload["data_dir"]
    assert isinstance(payload.get("isolated"), bool)

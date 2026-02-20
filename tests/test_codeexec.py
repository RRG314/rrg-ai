from pathlib import Path
import sys

import pytest

from backend.tools.codeexec import detect_test_command, run_command


def test_run_command_python_smoke(tmp_path: Path) -> None:
    result = run_command([sys.executable, "-c", "print('codeexec-ok')"], cwd=tmp_path)
    assert result.ok is True
    assert result.exit_code == 0
    assert "codeexec-ok" in result.stdout


def test_run_command_disallow_unknown_binary(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_command("rm -rf /", cwd=tmp_path)


def test_detect_test_command_prefers_pytest(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    detected = detect_test_command(tmp_path, runner="auto")
    assert detected[:3] == ["python", "-m", "pytest"]


def test_run_command_unquoted_path_with_spaces(tmp_path: Path) -> None:
    spaced = tmp_path / "folder with spaces"
    tests = spaced / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    cmd = f"python -m pytest -q {tests}"
    result = run_command(cmd, cwd=tmp_path)
    assert result.ok is True
    assert "passed" in result.stdout.lower()

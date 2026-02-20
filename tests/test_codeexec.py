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

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ALLOWED_BINARIES = {
    "python",
    "python3",
    "pytest",
    "node",
    "npm",
    "npx",
    "go",
    "cargo",
    "ruff",
    "mypy",
}


@dataclass
class CodeCommandResult:
    ok: bool
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool


def run_command(
    command: str | list[str],
    cwd: Path,
    timeout_sec: int = 90,
    max_output_chars: int = 16000,
) -> CodeCommandResult:
    parts = _normalize_command(command)
    _validate_command(parts)

    started = time.monotonic()
    proc = subprocess.run(
        parts,
        cwd=str(Path(cwd).resolve()),
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout_sec)),
        check=False,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    stdout, stderr, truncated = _truncate_outputs(proc.stdout or "", proc.stderr or "", max_output_chars=max_output_chars)
    return CodeCommandResult(
        ok=proc.returncode == 0,
        command=parts,
        cwd=str(Path(cwd).resolve()),
        exit_code=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        duration_ms=elapsed_ms,
        truncated=truncated,
    )


def detect_test_command(cwd: Path, runner: str = "auto") -> list[str]:
    cwd = Path(cwd).resolve()
    normalized_runner = (runner or "auto").strip()
    if normalized_runner and normalized_runner.lower() != "auto":
        parts = _normalize_command(normalized_runner)
        _validate_command(parts)
        return parts

    if (cwd / "package.json").exists():
        return ["npm", "test", "--", "--silent"]
    if (cwd / "go.mod").exists():
        return ["go", "test", "./..."]
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "test", "-q"]
    if (cwd / "pytest.ini").exists() or (cwd / "tests").exists() or (cwd / "pyproject.toml").exists():
        return ["python", "-m", "pytest", "-q"]

    return ["python", "-m", "pytest", "-q"]


def _normalize_command(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        parts = shlex.split(command.strip())
        parts = _coalesce_existing_paths(parts)
    else:
        parts = [str(p).strip() for p in command if str(p).strip()]
    if not parts:
        raise ValueError("Command is empty")
    return parts


def _validate_command(parts: list[str]) -> None:
    binary = Path(parts[0]).name.lower()
    allowed = binary in ALLOWED_BINARIES or binary.startswith("python") or binary.startswith("node")
    if not allowed:
        raise ValueError(
            f"Command '{binary}' is not allowed. Allowed commands: {', '.join(sorted(ALLOWED_BINARIES))}"
        )


def _truncate_outputs(stdout: str, stderr: str, max_output_chars: int) -> tuple[str, str, bool]:
    max_output_chars = max(400, int(max_output_chars))
    total = len(stdout) + len(stderr)
    if total <= max_output_chars:
        return stdout, stderr, False

    half = max(200, max_output_chars // 2)
    clipped_stdout = stdout[:half]
    clipped_stderr = stderr[:half]
    return clipped_stdout, clipped_stderr, True


def _coalesce_existing_paths(parts: list[str]) -> list[str]:
    if len(parts) < 2:
        return parts
    out: list[str] = []
    idx = 0
    while idx < len(parts):
        best_end = idx
        best_value = parts[idx]
        candidate = parts[idx]
        for j in range(idx + 1, len(parts)):
            candidate = candidate + " " + parts[j]
            if Path(candidate).exists():
                best_end = j
                best_value = candidate
        out.append(best_value)
        idx = best_end + 1
    return out

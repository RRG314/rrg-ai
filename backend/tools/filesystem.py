from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FileBrowser:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, user_path: str) -> Path:
        candidate = Path(user_path).expanduser()
        full = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        if not _is_relative_to(full, self.root):
            raise ValueError(f"Path escapes root: {full}")
        return full

    def list_dir(self, user_path: str = ".", max_entries: int = 200) -> list[dict[str, str | int | bool]]:
        target = self.resolve(user_path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            raise NotADirectoryError(str(target))

        rows: list[dict[str, str | int | bool]] = []
        for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:max_entries]:
            stat = entry.stat()
            rows.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "is_dir": entry.is_dir(),
                    "size": int(stat.st_size),
                }
            )
        return rows

    def read_text(self, user_path: str, max_chars: int = 25000) -> str:
        target = self.resolve(user_path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if target.is_dir():
            raise IsADirectoryError(str(target))

        data = target.read_bytes()
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = data.decode("utf-8", errors="replace")

        return text[:max_chars]

    def search_text(self, query: str, user_path: str = ".", max_hits: int = 80) -> list[dict[str, str | int]]:
        target = self.resolve(user_path)
        if not target.exists():
            raise FileNotFoundError(str(target))

        rg = shutil_which("rg")
        if rg:
            cmd = [rg, "-n", "-i", "--max-count", str(max_hits), query, str(target)]
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return _parse_rg_output(out.stdout, max_hits)

        return _slow_search(query, target, max_hits)


def _parse_rg_output(output: str, max_hits: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for line in output.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        rows.append({"path": parts[0], "line": int(parts[1]), "text": parts[2].strip()})
        if len(rows) >= max_hits:
            break
    return rows


def _slow_search(query: str, target: Path, max_hits: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    q = query.lower()
    candidates = [target] if target.is_file() else list(target.rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if q in line.lower():
                rows.append({"path": str(path), "line": idx, "text": line.strip()})
                if len(rows) >= max_hits:
                    return rows
    return rows


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def shutil_which(cmd: str) -> str | None:
    return shutil.which(cmd)

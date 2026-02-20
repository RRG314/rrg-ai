from __future__ import annotations

import json
import re
import sys
from collections import Counter
from typing import Any


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "query", "input", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower())


def main() -> None:
    raw = sys.stdin.read().strip()
    envelope = json.loads(raw) if raw else {}
    payload = envelope.get("input")
    text = _extract_text(payload)

    tokens = _tokenize(text)
    counts = Counter(tokens)
    top = counts.most_common(8)

    lines = []
    lines.append(f"characters: {len(text)}")
    lines.append(f"tokens: {len(tokens)}")
    lines.append(f"unique_tokens: {len(counts)}")
    if top:
        rendered_top = ", ".join(f"{word}({count})" for word, count in top)
        lines.append(f"top_terms: {rendered_top}")

    output_text = "\n".join(lines)
    summary = "Computed text statistics"

    result = {
        "status": "ok",
        "summary": summary,
        "text": output_text,
        "provenance": [
            {
                "source_type": "plugin",
                "source": "plugin:text_tools",
                "snippet": output_text[:300],
            }
        ],
        "artifacts": [],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load_bridge():
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.rdt_lm_bridge import RDTNgramGenerator, shell_alignment_score

    return RDTNgramGenerator, shell_alignment_score


def _extract_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return {"text": raw}
    return {}


def main() -> None:
    envelope = json.loads(sys.stdin.read() or "{}")
    payload = _extract_input(envelope.get("input"))

    mode = str(payload.get("mode") or "analyze").strip().lower()
    text = str(payload.get("text") or payload.get("seed") or "").strip()
    if not text:
        text = "the man walked"

    RDTNgramGenerator, shell_alignment_score = _load_bridge()
    gen = RDTNgramGenerator(alpha=float(payload.get("alpha") or 1.5), seed=42)
    gen.fit([])

    if mode == "generate":
        max_length = int(payload.get("max_length") or 15)
        out = gen.generate(text, max_length=max(4, min(max_length, 32)))
        score = shell_alignment_score(text, [out], alpha=float(payload.get("alpha") or 1.5))
        result = {
            "status": "ok",
            "summary": "Generated RDT shell-aware continuation",
            "text": f"seed: {text}\noutput: {out}\nshell_overlap: {score['shell_overlap']:.3f}\ntoken_overlap: {score['token_overlap']:.3f}",
            "provenance": [
                {
                    "source_type": "plugin",
                    "source": "plugin:rdt_lm_tools",
                    "snippet": out,
                }
            ],
            "artifacts": [],
        }
        print(json.dumps(result))
        return

    stats = gen.shell_stats(text)
    result = {
        "status": "ok",
        "summary": "Computed RDT shell-depth profile",
        "text": (
            f"text: {text}\n"
            f"alpha: {stats.alpha}\n"
            f"tokens: {stats.token_count}\n"
            f"unique_tokens: {stats.unique_count}\n"
            f"shell_histogram: {stats.shell_histogram}"
        ),
        "provenance": [
            {
                "source_type": "plugin",
                "source": "plugin:rdt_lm_tools",
                "snippet": f"shell_histogram={stats.shell_histogram}",
            }
        ],
        "artifacts": [],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()

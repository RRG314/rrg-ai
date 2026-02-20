from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .agent import AgentRunConfig, LocalAgent
from .llm import OllamaClient
from .storage import SQLiteStore
from .tools.filesystem import FileBrowser


def _task_score(result: dict[str, object], expected_keywords: list[str], min_provenance: int, min_evidence: int) -> dict[str, object]:
    answer = str(result.get("answer") or "").lower()
    provenance = result.get("provenance") or []
    evidence = result.get("evidence") or []

    keyword_hits = sum(1 for k in expected_keywords if k.lower() in answer)
    keyword_score = (keyword_hits / max(1, len(expected_keywords))) * 0.5

    provenance_score = 0.3 if len(provenance) >= min_provenance else 0.0

    evidence_good = 0
    for item in evidence:
        snippets = item.get("snippets") if isinstance(item, dict) else []
        if snippets:
            evidence_good += 1
    evidence_score = 0.2 if (len(evidence) >= min_evidence and evidence_good >= min_evidence) else 0.0

    total = round((keyword_score + provenance_score + evidence_score) * 100.0, 2)
    return {
        "score": total,
        "keyword_hits": keyword_hits,
        "keyword_target": len(expected_keywords),
        "provenance_count": len(provenance),
        "evidence_count": len(evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local agent eval suite")
    parser.add_argument("--data-dir", default=os.getenv("AI_DATA_DIR", ".ai_data"))
    parser.add_argument("--files-root", default=os.getenv("AI_FILES_ROOT", str(Path.cwd())))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = data_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "eval.sqlite3"
    store = SQLiteStore(db_path)
    files = FileBrowser(Path(args.files_root).expanduser().resolve())
    llm = OllamaClient(model=args.model, base_url=args.ollama_url)
    agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=data_dir / "downloads")

    seed_doc = (
        "Recursive-Adic retrieval score multiplies lexical overlap by a depth-Laplace weight. "
        "Evidence mode requires claims, sources, snippets, and confidence values."
    )
    store.add_document(name="eval-seed-notes", source="eval-seed", kind="text", text=seed_doc)

    seed_file = data_dir / "eval_seed_file.txt"
    seed_file.write_text(
        "Local planner/executor loops should keep plan steps, tool traces, and provenance logs.",
        encoding="utf-8",
    )

    tasks = [
        {
            "task_id": "seed_doc_grounding",
            "message": "What does recursive-adic retrieval score multiply?",
            "cfg": AgentRunConfig(
                strict_facts=True,
                evidence_mode=True,
                allow_web=False,
                allow_files=False,
                allow_docs=True,
                max_steps=args.max_steps,
            ),
            "expected_keywords": ["lexical", "weight"],
            "min_provenance": 1,
            "min_evidence": 1,
        },
        {
            "task_id": "file_tool_trace",
            "message": f"read file {seed_file}",
            "cfg": AgentRunConfig(
                strict_facts=True,
                evidence_mode=True,
                allow_web=False,
                allow_files=True,
                allow_docs=True,
                max_steps=args.max_steps,
            ),
            "expected_keywords": ["planner", "provenance"],
            "min_provenance": 1,
            "min_evidence": 1,
        },
    ]

    run_results: list[dict[str, object]] = []
    aggregate = 0.0
    session_id: str | None = None

    for case in tasks:
        result = agent.run_agent(session_id, str(case["message"]), config=case["cfg"])
        session_id = str(result.get("session_id") or session_id)
        score = _task_score(
            result,
            expected_keywords=list(case["expected_keywords"]),
            min_provenance=int(case["min_provenance"]),
            min_evidence=int(case["min_evidence"]),
        )
        aggregate += float(score["score"])
        run_results.append(
            {
                "task_id": case["task_id"],
                "message": case["message"],
                "score": score,
                "mode": result.get("mode"),
                "tool_calls": len(result.get("tool_calls") or []),
                "provenance_count": len(result.get("provenance") or []),
                "evidence_count": len(result.get("evidence") or []),
            }
        )

    final_score = round(aggregate / max(1, len(tasks)), 2)
    now = int(time.time())
    report = {
        "created_at": now,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "suite": "local_agent_v1",
        "task_count": len(tasks),
        "score": final_score,
        "results": run_results,
    }

    out_path = eval_dir / f"eval_{now}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Eval score: {final_score}")
    print(f"Report: {out_path}")
    print(json.dumps(run_results, indent=2))


if __name__ == "__main__":
    main()

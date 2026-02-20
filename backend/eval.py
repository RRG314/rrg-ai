from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentRunConfig, LocalAgent
from .llm import LLMStatus, OllamaClient
from .storage import SQLiteStore
from .tools.filesystem import FileBrowser


@dataclass
class EvalTask:
    task_id: str
    category: str
    message: str
    cfg: AgentRunConfig
    expected_keywords: list[str]
    min_provenance: int = 1
    min_evidence: int = 1


class NoLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(available=False, model="none", reason="eval-no-llm")

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        raise RuntimeError("NoLLM chat disabled")


def _task_score(result: dict[str, object], task: EvalTask) -> dict[str, object]:
    answer = str(result.get("answer") or "").lower()
    plan = result.get("plan") or []
    tool_calls = result.get("tool_calls") or []
    provenance = result.get("provenance") or []
    evidence = result.get("evidence") or []

    keyword_hits = sum(1 for k in task.expected_keywords if k.lower() in answer)
    keyword_score = (keyword_hits / max(1, len(task.expected_keywords))) * 0.25

    provenance_score = 0.25 if len(provenance) >= task.min_provenance else 0.0

    evidence_good = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        sources = item.get("sources") or []
        snippets = item.get("snippets") or []
        if sources and snippets:
            evidence_good += 1
    evidence_score = 0.25 if evidence_good >= task.min_evidence else 0.0

    done_steps = 0
    if isinstance(plan, list) and plan:
        done_steps = sum(1 for step in plan if isinstance(step, dict) and step.get("status") == "done")
    plan_ratio = done_steps / max(1, len(plan))
    plan_score = plan_ratio * 0.15

    tool_score = 0.10 if len(tool_calls) > 0 else 0.0

    total = round((keyword_score + provenance_score + evidence_score + plan_score + tool_score) * 100.0, 2)
    passed = total >= 70.0
    return {
        "score": total,
        "passed": passed,
        "keyword_hits": keyword_hits,
        "keyword_target": len(task.expected_keywords),
        "provenance_count": len(provenance),
        "evidence_count": len(evidence),
        "evidence_good": evidence_good,
        "tool_calls": len(tool_calls),
        "plan_done": done_steps,
        "plan_total": len(plan) if isinstance(plan, list) else 0,
    }


def _build_seed_data(store: SQLiteStore, workspace: Path) -> tuple[list[dict[str, object]], list[Path]]:
    workspace.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "id": "radf_core",
            "text": "Recursive-Adic retrieval multiplies lexical overlap by depth-Laplace weighting and stores chunk provenance.",
            "keywords": ["lexical", "depth-laplace", "provenance"],
        },
        {
            "id": "planner_loop",
            "text": "Planner executor loops should emit step plans, tool traces, retries, and done criteria.",
            "keywords": ["planner", "tool", "retries"],
        },
        {
            "id": "evidence_mode",
            "text": "Evidence mode requires claim objects that include source, snippet, and confidence scores.",
            "keywords": ["claim", "source", "snippet"],
        },
        {
            "id": "skills",
            "text": "Skill functions include research pipeline, doc pipeline, and folder audit pipeline for local automation.",
            "keywords": ["research", "doc", "folder"],
        },
        {
            "id": "memory_tables",
            "text": "Structured memory tables should track facts, preferences, outcomes, and artifacts with timestamps.",
            "keywords": ["facts", "preferences", "artifacts"],
        },
        {
            "id": "strict_facts",
            "text": "Strict fact mode blocks unsupported claims unless grounded evidence is present in local sources.",
            "keywords": ["strict", "grounded", "evidence"],
        },
        {
            "id": "folder_audit",
            "text": "Folder audits report extension distributions, largest files, and total scanned entries.",
            "keywords": ["extension", "largest", "entries"],
        },
        {
            "id": "doc_pipeline",
            "text": "Doc pipelines rank chunks and summarize retrieved passages with explicit source references.",
            "keywords": ["chunks", "summarize", "source"],
        },
        {
            "id": "research_pipeline",
            "text": "Research pipelines combine search results with fetched page text and preserve provenance for each item.",
            "keywords": ["search", "fetched", "provenance"],
        },
        {
            "id": "agent_trace",
            "text": "Agent trace panels should display plan status, tool calls, provenance, and evidence objects.",
            "keywords": ["plan", "tool", "evidence"],
        },
        {
            "id": "outcomes",
            "text": "Outcomes capture final summaries and scores for each completed task run.",
            "keywords": ["outcomes", "summaries", "scores"],
        },
        {
            "id": "artifacts",
            "text": "Artifacts capture file paths, urls, and doc ids touched during execution.",
            "keywords": ["paths", "urls", "doc"],
        },
    ]

    for item in docs:
        store.add_document(name=f"eval-doc:{item['id']}", source="eval-seed", kind="text", text=str(item["text"]))

    file_payloads = [
        "Planner loop keeps max_steps and retries stable.",
        "Evidence objects include source and snippet fields.",
        "Folder audits count file extensions and sizes.",
        "Doc pipeline summarizes retrieved chunks.",
        "Research pipeline can fallback to local documents.",
        "Structured memory includes facts and preferences.",
        "Outcomes are stored with status and score.",
        "Artifacts track touched files and urls.",
    ]

    files: list[Path] = []
    for idx, text in enumerate(file_payloads, start=1):
        sub = workspace / f"group_{(idx % 3) + 1}"
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"note_{idx}.txt"
        path.write_text(text, encoding="utf-8")
        files.append(path)

    return docs, files


def _build_tasks(
    docs: list[dict[str, object]],
    files: list[Path],
    workspace: Path,
    max_steps: int,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []

    for idx, doc in enumerate(docs, start=1):
        doc_id = str(doc["id"])
        tasks.append(
            EvalTask(
                task_id=f"doc_pipeline_{idx:02d}",
                category="doc",
                message=f"doc pipeline for {doc_id}",
                cfg=AgentRunConfig(
                    strict_facts=True,
                    evidence_mode=True,
                    allow_web=False,
                    allow_files=False,
                    allow_docs=True,
                    max_steps=max_steps,
                ),
                expected_keywords=list(doc["keywords"]),
            )
        )

    for idx, path in enumerate(files, start=1):
        stem = path.stem.replace("_", " ")
        tasks.append(
            EvalTask(
                task_id=f"file_read_{idx:02d}",
                category="file",
                message=f"read file {path}",
                cfg=AgentRunConfig(
                    strict_facts=True,
                    evidence_mode=True,
                    allow_web=False,
                    allow_files=True,
                    allow_docs=True,
                    max_steps=max_steps,
                ),
                expected_keywords=["source", "snippet", stem.split()[0]],
            )
        )

    search_terms = ["planner", "evidence", "pipeline", "artifacts", "memory", "extensions"]
    for idx, term in enumerate(search_terms, start=1):
        tasks.append(
            EvalTask(
                task_id=f"file_search_{idx:02d}",
                category="file",
                message=f"search files for {term} in {workspace}",
                cfg=AgentRunConfig(
                    strict_facts=True,
                    evidence_mode=True,
                    allow_web=False,
                    allow_files=True,
                    allow_docs=True,
                    max_steps=max_steps,
                ),
                expected_keywords=[term, "source"],
            )
        )

    tasks.append(
        EvalTask(
            task_id="folder_audit_root",
            category="skill",
            message=f"folder audit in {workspace}",
            cfg=AgentRunConfig(
                strict_facts=True,
                evidence_mode=True,
                allow_web=False,
                allow_files=True,
                allow_docs=True,
                max_steps=max_steps,
            ),
            expected_keywords=["entries", "files", "extensions"],
        )
    )
    tasks.append(
        EvalTask(
            task_id="folder_audit_sub",
            category="skill",
            message=f"audit folder {workspace / 'group_1'}",
            cfg=AgentRunConfig(
                strict_facts=True,
                evidence_mode=True,
                allow_web=False,
                allow_files=True,
                allow_docs=True,
                max_steps=max_steps,
            ),
            expected_keywords=["files", "largest"],
        )
    )

    research_prompts = [
        "research pipeline for recursive adic retrieval",
        "research pipeline for evidence mode citations",
        "research pipeline for planner executor retries",
        "research pipeline for structured memory artifacts",
    ]
    for idx, msg in enumerate(research_prompts, start=1):
        tasks.append(
            EvalTask(
                task_id=f"research_skill_{idx:02d}",
                category="skill",
                message=msg,
                cfg=AgentRunConfig(
                    strict_facts=True,
                    evidence_mode=True,
                    allow_web=True,
                    allow_files=False,
                    allow_docs=True,
                    max_steps=max_steps,
                ),
                expected_keywords=["source", "snippet", "claim"],
            )
        )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local agent eval suite")
    parser.add_argument("--data-dir", default=os.getenv("AI_DATA_DIR", ".ai_data"))
    parser.add_argument("--files-root", default=os.getenv("AI_FILES_ROOT", str(Path.cwd())))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--task-count", type=int, default=24, help="Number of tasks to run (20-50)")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama model during eval")
    args = parser.parse_args()

    if args.task_count < 20 or args.task_count > 50:
        raise SystemExit("--task-count must be between 20 and 50")

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = data_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    files_root = Path(args.files_root).expanduser().resolve()
    workspace = files_root / ".eval_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "eval.sqlite3"
    store = SQLiteStore(db_path)
    files = FileBrowser(files_root)
    llm = OllamaClient(model=args.model, base_url=args.ollama_url) if args.use_llm else NoLLM()
    agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=data_dir / "downloads")

    docs, seed_files = _build_seed_data(store, workspace)
    tasks = _build_tasks(docs, seed_files, workspace, max_steps=args.max_steps)

    if args.task_count > len(tasks):
        base = list(tasks)
        while len(tasks) < args.task_count:
            tasks.extend(base)
    tasks = tasks[: args.task_count]

    run_results: list[dict[str, object]] = []
    aggregate = 0.0
    pass_count = 0
    session_id: str | None = None

    per_category: dict[str, list[float]] = {}

    for task in tasks:
        result = agent.run_agent(session_id, task.message, config=task.cfg)
        session_id = str(result.get("session_id") or session_id)

        score = _task_score(result, task)
        aggregate += float(score["score"])
        if bool(score["passed"]):
            pass_count += 1

        per_category.setdefault(task.category, []).append(float(score["score"]))

        run_results.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "message": task.message,
                "mode": result.get("mode"),
                "score": score,
                "tool_calls": len(result.get("tool_calls") or []),
                "provenance_count": len(result.get("provenance") or []),
                "evidence_count": len(result.get("evidence") or []),
            }
        )

    final_score = round(aggregate / max(1, len(tasks)), 2)
    pass_rate = round((pass_count / max(1, len(tasks))) * 100.0, 2)

    category_summary = {
        cat: {
            "count": len(scores),
            "avg_score": round(sum(scores) / max(1, len(scores)), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
        }
        for cat, scores in per_category.items()
    }

    now = int(time.time())
    report = {
        "created_at": now,
        "suite": "local_agent_v2_extended",
        "task_count": len(tasks),
        "score": final_score,
        "pass_rate": pass_rate,
        "use_llm": bool(args.use_llm),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "categories": category_summary,
        "results": run_results,
    }

    out_path = eval_dir / f"eval_{now}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Eval score: {final_score}")
    print(f"Pass rate: {pass_rate}%")
    print(f"Task count: {len(tasks)}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()

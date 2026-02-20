from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentRunConfig, LocalAgent
from .llm import LLMStatus, OllamaClient
from .run_isolation import resolve_bench_paths
from .storage import SQLiteStore
from .tools.filesystem import FileBrowser


@dataclass
class IndustryTask:
    task_id: str
    category: str
    message: str
    cfg: AgentRunConfig
    check: dict[str, object]


class NoLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(available=False, model="none", reason="industry-bench-no-llm")

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        raise RuntimeError("NoLLM chat disabled")


def _seed_workspace(store: SQLiteStore, files_root: Path) -> Path:
    ws = files_root / ".industry_bench"
    notes = ws / "notes"
    src = ws / "src"
    tests = ws / "tests"
    notes.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    (src / "__init__.py").write_text("", encoding="utf-8")
    (notes / "sharks.txt").write_text(
        "Sharks are cartilaginous fish. Many species use electroreception via ampullae of Lorenzini.",
        encoding="utf-8",
    )
    (notes / "radf.txt").write_text(
        "Recursive-Adic retrieval uses depth-Laplace weighting and recursive depth transforms for ranking.",
        encoding="utf-8",
    )
    (notes / "facts.txt").write_text(
        (
            "Mars is called the Red Planet.\n"
            "The capital city of Japan is Tokyo.\n"
            "The Pacific Ocean is the largest ocean on Earth.\n"
            "The primary language spoken in Brazil is Portuguese.\n"
            "At sea level, water boils at 100 degrees Celsius.\n"
            "Plants absorb carbon dioxide during photosynthesis.\n"
            "H2O is commonly known as water.\n"
            "A normal human heart has four chambers: two atria and two ventricles.\n"
        ),
        encoding="utf-8",
    )
    (tests / "test_smoke.py").write_text(
        "def test_smoke():\n"
        "    assert (2 + 3) == 5\n",
        encoding="utf-8",
    )

    store.add_document(
        name="industry-seed:sharks",
        source=str(notes / "sharks.txt"),
        kind="text",
        text=(notes / "sharks.txt").read_text(encoding="utf-8"),
    )
    store.add_document(
        name="industry-seed:radf",
        source=str(notes / "radf.txt"),
        kind="text",
        text=(notes / "radf.txt").read_text(encoding="utf-8"),
    )
    store.add_document(
        name="industry-seed:facts",
        source=str(notes / "facts.txt"),
        kind="text",
        text=(notes / "facts.txt").read_text(encoding="utf-8"),
    )
    return ws


def _build_tasks(workspace: Path, max_steps: int, prefer_local_core: bool) -> list[IndustryTask]:
    factual_cfg = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        prefer_local_core=prefer_local_core,
        allow_web=True,
        allow_files=False,
        allow_docs=True,
        allow_code=False,
        max_steps=max_steps,
    )
    math_cfg = AgentRunConfig(
        strict_facts=False,
        evidence_mode=False,
        prefer_local_core=prefer_local_core,
        allow_web=False,
        allow_files=False,
        allow_docs=False,
        allow_code=True,
        max_steps=max_steps,
    )
    code_cfg = AgentRunConfig(
        strict_facts=False,
        evidence_mode=False,
        prefer_local_core=prefer_local_core,
        allow_web=False,
        allow_files=True,
        allow_docs=False,
        allow_code=True,
        max_steps=max_steps,
    )
    tools_cfg = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        prefer_local_core=prefer_local_core,
        allow_web=True,
        allow_files=True,
        allow_docs=True,
        allow_code=True,
        max_steps=max_steps,
    )

    tasks: list[IndustryTask] = []

    factual = [
        ("mmlu_red_planet", "Which planet is called the Red Planet?", ["mars"]),
        ("mmlu_capital_japan", "What is the capital city of Japan?", ["tokyo"]),
        ("mmlu_largest_ocean", "What is the largest ocean on Earth?", ["pacific"]),
        ("mmlu_language_brazil", "What is the primary language spoken in Brazil?", ["portuguese"]),
        ("mmlu_boiling_water", "At sea level, water boils at what Celsius temperature?", ["100"]),
        ("mmlu_photosynthesis_gas", "Which gas do plants absorb during photosynthesis?", ["carbon dioxide", "co2"]),
        ("mmlu_h2o_name", "What is the common chemical name for H2O?", ["water"]),
        ("mmlu_human_heart", "How many chambers does a normal human heart have?", ["four", "4", "two atria", "two ventricles"]),
    ]
    for task_id, message, expected_any in factual:
        tasks.append(
            IndustryTask(
                task_id=task_id,
                category="mmlu_style",
                message=message,
                cfg=factual_cfg,
                check={"type": "contains_any", "values": expected_any},
            )
        )

    math = [
        ("gsm8k_mul", "calculate 27 * 14", "378"),
        ("gsm8k_add_div", "calculate (120 / 5) + 17", "41"),
        ("gsm8k_square_sub", "calculate (19 * 19) - 7", "354"),
        ("gsm8k_div_add", "calculate (144 / 12) + 6", "18"),
        ("gsm8k_grouped", "calculate (3 + 5) * 9", "72"),
        ("gsm8k_pow", "calculate 2 ** 8", "256"),
    ]
    for task_id, message, expected in math:
        tasks.append(
            IndustryTask(
                task_id=task_id,
                category="gsm8k_style",
                message=message,
                cfg=math_cfg,
                check={"type": "contains_any", "values": [expected]},
            )
        )

    code = [
        (
            "humaneval_run_1",
            "run command python -c \"print(2 + 3)\"",
            {"type": "contains_any", "values": ["5"], "tool": "code.run"},
        ),
        (
            "humaneval_run_2",
            "run command python -c \"import math; print(math.factorial(6))\"",
            {"type": "contains_any", "values": ["720"], "tool": "code.run"},
        ),
        (
            "humaneval_gen_1",
            "create file .industry_bench/src/add.py with python function add(a, b) returning a + b",
            {"type": "tool_ok", "tool": "code.generate"},
        ),
        (
            "humaneval_gen_2",
            "create file .industry_bench/src/fib.py with python function fib(n) returning nth fibonacci number iteratively",
            {"type": "tool_ok", "tool": "code.generate"},
        ),
        (
            "humaneval_gen_3",
            "create file .industry_bench/src/pal.py with python function is_pal(s) returning true for palindrome",
            {"type": "tool_ok", "tool": "code.generate"},
        ),
        (
            "humaneval_test_smoke",
            f"run tests in {workspace}",
            {"type": "tool_ok", "tool": "code.test"},
        ),
    ]
    for task_id, message, check in code:
        tasks.append(
            IndustryTask(
                task_id=task_id,
                category="humaneval_lite",
                message=message,
                cfg=code_cfg,
                check=check,
            )
        )

    tools = [
        (
            "tool_read_file",
            "read file .industry_bench/notes/sharks.txt",
            {"type": "contains_any", "values": ["cartilaginous", "electroreception"], "tool": "files.read"},
        ),
        (
            "tool_search_files",
            "search files for cartilaginous in .industry_bench",
            {"type": "contains_any", "values": ["cartilaginous"], "tool": "files.search"},
        ),
        (
            "tool_folder_audit",
            "folder audit in .industry_bench",
            {"type": "tool_ok", "tool": "skill.folder_audit_pipeline"},
        ),
        (
            "tool_doc_pipeline",
            "doc pipeline for recursive adic depth weighting",
            {"type": "tool_ok", "tool": "skill.doc_pipeline"},
        ),
        (
            "tool_research_pipeline",
            "research pipeline for shark sensory systems",
            {"type": "tool_ok", "tool": "skill.research_pipeline"},
        ),
        (
            "tool_run_pytest",
            f"run command python -m pytest -q {workspace}/tests",
            {"type": "contains_any", "values": ["1 passed", "passed"], "tool": "code.run"},
        ),
    ]
    for task_id, message, check in tools:
        tasks.append(
            IndustryTask(
                task_id=task_id,
                category="toolbench_style",
                message=message,
                cfg=tools_cfg,
                check=check,
            )
        )

    return tasks


def _check_task(result: dict[str, object], check: dict[str, object]) -> tuple[bool, str]:
    answer = str(result.get("answer") or "")
    provenance = result.get("provenance") or []
    snippets = " ".join(str(item.get("snippet") or "") for item in provenance if isinstance(item, dict))
    combined = f"{answer}\n{snippets}".lower()

    tool_name = str(check.get("tool") or "").strip()
    if tool_name:
        calls = [c for c in (result.get("tool_calls") or []) if isinstance(c, dict)]
        ok_tool = any(str(c.get("name") or "") == tool_name and str(c.get("status") or "") == "ok" for c in calls)
        if not ok_tool:
            return False, f"required tool not successful: {tool_name}"

    check_type = str(check.get("type") or "contains_any")
    values = [str(v).lower() for v in (check.get("values") or []) if str(v).strip()]

    if check_type == "tool_ok":
        return True, "tool check passed"
    if check_type == "contains_all":
        passed = all(v in combined for v in values)
        return passed, f"contains_all={passed}"
    if check_type == "contains_any":
        passed = any(v in combined for v in values)
        return passed, f"contains_any={passed}"
    if check_type == "regex":
        pattern = str(check.get("pattern") or "")
        passed = bool(pattern) and bool(re.search(pattern, combined, re.IGNORECASE))
        return passed, f"regex={passed}"

    return False, f"unknown check type: {check_type}"


def _score_task(result: dict[str, object], task: IndustryTask, core_pass: bool) -> tuple[float, dict[str, object]]:
    tool_calls = [c for c in (result.get("tool_calls") or []) if isinstance(c, dict)]
    provenance = [p for p in (result.get("provenance") or []) if isinstance(p, dict)]
    evidence = [e for e in (result.get("evidence") or []) if isinstance(e, dict)]

    grounded = 1.0 if len(provenance) > 0 else 0.0
    tooling = 1.0 if len(tool_calls) > 0 else 0.0
    evidence_good = 0.0
    if evidence:
        complete = 0
        for item in evidence:
            if (item.get("sources") or []) and (item.get("snippets") or []):
                complete += 1
        evidence_good = complete / max(1, len(evidence))

    if task.cfg.evidence_mode:
        core_w, grounded_w, tool_w, evidence_w = 0.60, 0.20, 0.10, 0.10
    else:
        core_w, grounded_w, tool_w, evidence_w = 0.70, 0.20, 0.10, 0.0

    score = (
        (core_w * (1.0 if core_pass else 0.0))
        + (grounded_w * grounded)
        + (tool_w * tooling)
        + (evidence_w * evidence_good)
    ) * 100.0
    detail = {
        "core_pass": core_pass,
        "provenance_count": len(provenance),
        "tool_call_count": len(tool_calls),
        "evidence_count": len(evidence),
        "evidence_quality": round(evidence_good, 3),
        "weights": {
            "core": core_w,
            "grounded": grounded_w,
            "tooling": tool_w,
            "evidence": evidence_w,
        },
    }
    return round(score, 2), detail


def run_suite(
    data_dir: Path,
    files_root: Path,
    model: str,
    ollama_url: str,
    use_llm: bool,
    max_steps: int,
    task_count: int,
    target_score: float = 95.0,
    isolated: bool = True,
    persist_db: bool = False,
    db_path: Path | None = None,
) -> Path:
    base_data_dir = Path(data_dir).expanduser().resolve()
    base_data_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = base_data_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    use_isolated = bool(isolated and not persist_db)
    resolved_db_path, run_data_dir, run_id = resolve_bench_paths(
        prefix="industry_bench",
        isolated=use_isolated,
        db_path=db_path if persist_db else None,
        data_dir=base_data_dir,
    )
    run_data_dir.mkdir(parents=True, exist_ok=True)

    effective_files_root = (run_data_dir / "files") if use_isolated else Path(files_root).expanduser().resolve()
    effective_files_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = run_data_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(resolved_db_path)
    files = FileBrowser(effective_files_root)
    llm = OllamaClient(model=model, base_url=ollama_url) if use_llm else NoLLM()
    agent = LocalAgent(store=store, files=files, llm=llm, downloads_dir=downloads_dir)

    workspace = _seed_workspace(store, effective_files_root)
    tasks = _build_tasks(workspace, max_steps=max_steps, prefer_local_core=not use_llm)
    if task_count > 0:
        tasks = tasks[: max(1, min(task_count, len(tasks)))]

    results: list[dict[str, object]] = []
    category_scores: dict[str, list[float]] = {}
    total_score = 0.0
    passed = 0
    passed_95 = 0
    sid: str | None = None

    for task in tasks:
        run = agent.run_agent(sid, task.message, config=task.cfg)
        sid = str(run.get("session_id") or sid)
        core_pass, check_note = _check_task(run, task.check)
        score, score_detail = _score_task(run, task=task, core_pass=core_pass)
        total_score += score
        if score >= 70.0 and core_pass:
            passed += 1
        if score >= float(target_score) and core_pass:
            passed_95 += 1
        category_scores.setdefault(task.category, []).append(score)
        results.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "message": task.message,
                "check": task.check,
                "check_note": check_note,
                "score": score,
                "score_detail": score_detail,
                "mode": run.get("mode"),
                "answer_preview": str(run.get("answer") or "")[:320],
                "provenance_preview": [
                    str(item.get("snippet") or "")[:200]
                    for item in (run.get("provenance") or [])[:2]
                    if isinstance(item, dict)
                ],
                "adaptive_update": run.get("adaptive_update"),
            }
        )

    category_summary = {
        cat: {
            "count": len(vals),
            "avg_score": round(statistics.mean(vals), 2),
            "min_score": round(min(vals), 2),
            "max_score": round(max(vals), 2),
            "pass_rate_95": round((sum(1 for v in vals if v >= float(target_score)) / max(1, len(vals))) * 100.0, 2),
            "meets_95": bool(min(vals) >= float(target_score)),
        }
        for cat, vals in category_scores.items()
    }

    overall_score = round(total_score / max(1, len(tasks)), 2)
    pass_rate_95 = round((passed_95 / max(1, len(tasks))) * 100.0, 2)
    meets_95 = bool(overall_score >= float(target_score)) and all(
        bool(item.get("meets_95")) for item in category_summary.values()
    )

    report = {
        "created_at": int(time.time()),
        "suite": "industry_aligned_lite_v1",
        "task_count": len(tasks),
        "use_llm": bool(use_llm),
        "model": model,
        "ollama_url": ollama_url,
        "run_id": run_id,
        "db_path": str(resolved_db_path),
        "data_dir": str(run_data_dir),
        "isolated": bool(use_isolated),
        "persist_db": bool(persist_db),
        "score": overall_score,
        "pass_rate": round((passed / max(1, len(tasks))) * 100.0, 2),
        "target_score": float(target_score),
        "pass_rate_95": pass_rate_95,
        "meets_95": meets_95,
        "categories": category_summary,
        "results": results,
    }

    out = eval_dir / f"industry_bench_{run_id}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run industry-aligned local benchmark suite")
    parser.add_argument("--data-dir", default=os.getenv("AI_DATA_DIR", ".ai_data"))
    parser.add_argument("--files-root", default=os.getenv("AI_FILES_ROOT", str(Path.cwd())))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--task-count", type=int, default=0, help="0 means run full suite")
    parser.add_argument("--target-score", type=float, default=95.0)
    parser.add_argument("--isolated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persist-db", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--db-path", default="")
    args = parser.parse_args()

    out = run_suite(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        files_root=Path(args.files_root).expanduser().resolve(),
        model=args.model,
        ollama_url=args.ollama_url,
        use_llm=bool(args.use_llm),
        max_steps=int(args.max_steps),
        task_count=int(args.task_count),
        target_score=float(args.target_score),
        isolated=bool(args.isolated),
        persist_db=bool(args.persist_db),
        db_path=Path(args.db_path).expanduser().resolve() if str(args.db_path or "").strip() else None,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    print(f"Suite: {payload['suite']}")
    print(f"Tasks: {payload['task_count']}")
    print(f"Score: {payload['score']}")
    print(f"Pass rate: {payload['pass_rate']}%")
    print(f"Pass rate >= {payload['target_score']}: {payload['pass_rate_95']}%")
    print(f"Meets 95 target: {payload['meets_95']}")
    print(f"Run ID: {payload['run_id']}")
    print(f"Isolated: {payload['isolated']}")
    print(f"DB: {payload['db_path']}")
    print(f"Run data dir: {payload['data_dir']}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()

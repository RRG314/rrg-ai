from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentRunConfig, LocalAgent
from .llm import LLMStatus, OllamaClient
from .run_isolation import resolve_bench_paths
from .storage import SQLiteStore
from .tools.docs import extract_text_from_bytes
from .tools.filesystem import FileBrowser


@dataclass
class CheckCase:
    check_id: str
    category: str
    message: str
    config: AgentRunConfig
    required_tools: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    require_provenance: bool = False
    require_evidence_pairs: bool = False
    require_adaptive_policy: bool = False


class NoLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(available=False, model="none", reason="system-check-no-llm")

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        raise RuntimeError("NoLLM chat disabled")


def _ensure_workspace(files_root: Path, store: SQLiteStore) -> Path:
    ws = files_root / ".ai_system_check"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    notes = ws / "notes"
    src = ws / "src"
    tests = ws / "tests"
    notes.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    (notes / "sharks.txt").write_text(
        "Sharks are cartilaginous fish and many species detect electric fields via ampullae of Lorenzini.",
        encoding="utf-8",
    )
    (notes / "radf.txt").write_text(
        "Recursive-Adic retrieval uses recursive depth transforms and depth-Laplace weighting for ranking.",
        encoding="utf-8",
    )
    (notes / "memory.txt").write_text(
        "Structured memory stores facts, preferences, outcomes, and artifacts.",
        encoding="utf-8",
    )
    (tests / "test_smoke.py").write_text("def test_smoke():\n    assert (2 + 3) == 5\n", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")

    store.add_document(
        name="system-check:radf",
        source=str(notes / "radf.txt"),
        kind="text",
        text=(notes / "radf.txt").read_text(encoding="utf-8"),
    )
    store.add_document(
        name="system-check:memory",
        source=str(notes / "memory.txt"),
        kind="text",
        text=(notes / "memory.txt").read_text(encoding="utf-8"),
    )
    return ws


def _build_checks(workspace: Path, max_steps: int) -> list[CheckCase]:
    local = AgentRunConfig(
        strict_facts=False,
        evidence_mode=False,
        allow_web=False,
        allow_files=False,
        allow_docs=False,
        allow_code=False,
        max_steps=max_steps,
    )
    docs = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        allow_web=False,
        allow_files=False,
        allow_docs=True,
        allow_code=False,
        max_steps=max_steps,
    )
    files = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        allow_web=False,
        allow_files=True,
        allow_docs=True,
        allow_code=False,
        max_steps=max_steps,
    )
    code = AgentRunConfig(
        strict_facts=False,
        evidence_mode=False,
        allow_web=False,
        allow_files=True,
        allow_docs=False,
        allow_code=True,
        max_steps=max_steps,
    )
    web = AgentRunConfig(
        strict_facts=True,
        evidence_mode=True,
        allow_web=True,
        allow_files=False,
        allow_docs=True,
        allow_code=False,
        max_steps=max_steps,
    )

    return [
        CheckCase(
            check_id="chat_basic",
            category="chat",
            message="What can you help with locally?",
            config=local,
            required_tools=("answer.compose",),
        ),
        CheckCase(
            check_id="math_eval_1",
            category="math",
            message="calculate (19 * 19) - 7",
            config=code,
            required_tools=("math.eval",),
            required_terms=("354",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="math_eval_2",
            category="math",
            message="calculate (120 / 5) + 17",
            config=code,
            required_tools=("math.eval",),
            required_terms=("41",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="files_read",
            category="files",
            message=f"read file {workspace / 'notes' / 'sharks.txt'}",
            config=files,
            required_tools=("files.read",),
            required_terms=("cartilaginous", "electric"),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="files_search",
            category="files",
            message=f"search files for lorenzini in {workspace}",
            config=files,
            required_tools=("files.search",),
            required_terms=("lorenzini",),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="files_list",
            category="files",
            message=f"list files in {workspace}",
            config=files,
            required_tools=("files.list",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="folder_audit_root",
            category="skills",
            message=f"folder audit in {workspace}",
            config=files,
            required_tools=("skill.folder_audit_pipeline",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="folder_audit_notes",
            category="skills",
            message=f"audit folder {workspace / 'notes'}",
            config=files,
            required_tools=("skill.folder_audit_pipeline",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="doc_pipeline",
            category="skills",
            message="doc pipeline for recursive adic depth weighting",
            config=docs,
            required_tools=("skill.doc_pipeline",),
            required_terms=("depth", "weight"),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="docs_retrieve",
            category="docs",
            message="Explain recursive adic retrieval ranking.",
            config=docs,
            required_tools=("docs.retrieve",),
            required_terms=("recursive", "adic", "weight"),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="strict_fact_grounding",
            category="docs",
            message="Tell me about structured memory tables in this system.",
            config=docs,
            required_tools=("docs.retrieve",),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="evidence_mode",
            category="evidence",
            message=f"read file {workspace / 'notes' / 'memory.txt'}",
            config=files,
            required_tools=("files.read",),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="adaptive_update",
            category="adaptive",
            message="my goal is improve local ai quality and safety",
            config=local,
            required_tools=("answer.compose",),
            require_adaptive_policy=True,
        ),
        CheckCase(
            check_id="code_run",
            category="code",
            message="run command python -c \"print(2 + 3)\"",
            config=code,
            required_tools=("code.run",),
            required_terms=("5",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="code_generate",
            category="code",
            message=f"create file {workspace / 'src' / 'add.py'} with python function add(a, b) returning a + b",
            config=code,
            required_tools=("code.generate",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="code_test",
            category="code",
            message=f"run tests in {workspace}",
            config=code,
            required_tools=("code.test",),
            required_terms=("passed",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="code_run_pytest",
            category="code",
            message=f"run command python -m pytest -q {workspace / 'tests'}",
            config=code,
            required_tools=("code.run",),
            required_terms=("passed",),
            require_provenance=True,
        ),
        CheckCase(
            check_id="web_dictionary",
            category="web",
            message="define entropy",
            config=web,
            required_tools=("web.dictionary",),
            required_terms=("entropy",),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="web_search",
            category="web",
            message="search the web for shark electroreception",
            config=web,
            required_tools=("web.search",),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
        CheckCase(
            check_id="web_fetch",
            category="web",
            message="read website https://en.wikipedia.org/wiki/Shark",
            config=web,
            required_tools=("web.fetch",),
            require_provenance=True,
            require_evidence_pairs=True,
        ),
    ]


def _tool_ok(result: dict[str, object], name: str) -> bool:
    calls = [c for c in (result.get("tool_calls") or []) if isinstance(c, dict)]
    return any(str(c.get("name") or "") == name and str(c.get("status") or "") == "ok" for c in calls)


def _joined_text(result: dict[str, object]) -> str:
    lines: list[str] = [str(result.get("answer") or "")]
    for item in (result.get("provenance") or []):
        if isinstance(item, dict):
            lines.append(str(item.get("snippet") or ""))
            lines.append(str(item.get("source") or ""))
    for call in (result.get("tool_calls") or []):
        if isinstance(call, dict):
            lines.append(str(call.get("result_summary") or ""))
    return "\n".join(lines).lower()


def _evidence_pairs(result: dict[str, object]) -> int:
    count = 0
    for item in (result.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        if (item.get("sources") or []) and (item.get("snippets") or []):
            count += 1
    return count


def _evaluate_check(case: CheckCase, result: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True

    for tool in case.required_tools:
        if not _tool_ok(result, tool):
            ok = False
            reasons.append(f"tool_missing:{tool}")

    text = _joined_text(result)
    for term in case.required_terms:
        if term.lower() not in text:
            ok = False
            reasons.append(f"term_missing:{term}")

    if case.require_provenance:
        provenance_count = len([p for p in (result.get("provenance") or []) if isinstance(p, dict)])
        if provenance_count <= 0:
            ok = False
            reasons.append("missing_provenance")

    if case.require_evidence_pairs:
        if _evidence_pairs(result) <= 0:
            ok = False
            reasons.append("missing_evidence_pairs")

    if case.require_adaptive_policy:
        adaptive = result.get("adaptive_update")
        if not isinstance(adaptive, dict):
            ok = False
            reasons.append("missing_adaptive_update")
        else:
            policy = adaptive.get("policy_update")
            if not isinstance(policy, dict) or not str(policy.get("summary") or "").strip():
                ok = False
                reasons.append("missing_policy_update")

    return ok, reasons


def _direct_checks(store: SQLiteStore) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    search = store.search_chunks("recursive adic depth weighting", limit=3)
    radf_ok = bool(search) and all("radf_weight" in hit and "radf_depth" in hit for hit in search)
    checks.append(
        {
            "check_id": "radf_chunk_ranking",
            "category": "radf",
            "passed": bool(radf_ok),
            "score": 100.0 if radf_ok else 0.0,
            "notes": [] if radf_ok else ["radf_fields_missing"],
            "duration_ms": 0,
        }
    )

    extracted, kind = extract_text_from_bytes("sample.txt", b"simple extraction check")
    extract_ok = kind in {"text"} and "simple extraction check" in extracted.lower()
    checks.append(
        {
            "check_id": "doc_extract_text",
            "category": "docs",
            "passed": bool(extract_ok),
            "score": 100.0 if extract_ok else 0.0,
            "notes": [] if extract_ok else ["extract_text_failed"],
            "duration_ms": 0,
        }
    )
    return checks


def run_system_check(
    data_dir: Path,
    files_root: Path,
    model: str,
    ollama_url: str,
    use_llm: bool,
    max_steps: int,
    min_score: float = 95.0,
    task_limit: int = 0,
    isolated: bool = True,
    persist_db: bool = False,
    db_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    base_data_dir = Path(data_dir).expanduser().resolve()
    base_data_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = base_data_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    use_isolated = bool(isolated and not persist_db)
    resolved_db_path, run_data_dir, run_id = resolve_bench_paths(
        prefix="system_check",
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

    workspace = _ensure_workspace(effective_files_root, store)
    checks = _build_checks(workspace, max_steps=max_steps)
    if task_limit > 0:
        checks = checks[: max(1, min(task_limit, len(checks)))]

    results: list[dict[str, object]] = []
    category: dict[str, dict[str, int]] = {}
    sid: str | None = None
    pass_count = 0

    for case in checks:
        started = time.monotonic()
        try:
            run = agent.run_agent(sid, case.message, config=case.config)
            sid = str(run.get("session_id") or sid)
            passed, notes = _evaluate_check(case, run)
            elapsed = int((time.monotonic() - started) * 1000)
            score = 100.0 if passed else 0.0
            if passed:
                pass_count += 1
            results.append(
                {
                    "check_id": case.check_id,
                    "category": case.category,
                    "message": case.message,
                    "passed": bool(passed),
                    "score": score,
                    "notes": notes,
                    "duration_ms": elapsed,
                    "mode": run.get("mode"),
                    "answer_preview": str(run.get("answer") or "")[:220],
                    "adaptive_update": run.get("adaptive_update"),
                }
            )
            bucket = category.setdefault(case.category, {"count": 0, "passed": 0})
            bucket["count"] += 1
            if passed:
                bucket["passed"] += 1
        except Exception as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            results.append(
                {
                    "check_id": case.check_id,
                    "category": case.category,
                    "message": case.message,
                    "passed": False,
                    "score": 0.0,
                    "notes": [f"exception:{str(exc)[:220]}"],
                    "duration_ms": elapsed,
                    "mode": "error",
                    "answer_preview": "",
                    "adaptive_update": {},
                }
            )
            bucket = category.setdefault(case.category, {"count": 0, "passed": 0})
            bucket["count"] += 1

    direct = _direct_checks(store)
    for item in direct:
        results.append(item)
        bucket = category.setdefault(str(item.get("category") or "other"), {"count": 0, "passed": 0})
        bucket["count"] += 1
        if bool(item.get("passed")):
            bucket["passed"] += 1
            pass_count += 1

    total = len(results)
    score = round((pass_count / max(1, total)) * 100.0, 2)
    category_summary = {
        key: {
            "count": values["count"],
            "passed": values["passed"],
            "accuracy": round((values["passed"] / max(1, values["count"])) * 100.0, 2),
        }
        for key, values in sorted(category.items(), key=lambda kv: kv[0])
    }
    categories_meet_target = all(float(item.get("accuracy") or 0.0) >= float(min_score) for item in category_summary.values())
    meets_target = bool(score >= float(min_score))
    meets_target_all_systems = bool(meets_target and categories_meet_target)

    report = {
        "created_at": int(time.time()),
        "suite": "system_check_v1",
        "task_count": len(checks),
        "direct_checks": len(direct),
        "check_count": total,
        "use_llm": bool(use_llm),
        "model": model,
        "ollama_url": ollama_url,
        "run_id": run_id,
        "db_path": str(resolved_db_path),
        "data_dir": str(run_data_dir),
        "isolated": bool(use_isolated),
        "persist_db": bool(persist_db),
        "score": score,
        "target_score": float(min_score),
        "meets_target": meets_target,
        "categories_meet_target": categories_meet_target,
        "meets_target_all_systems": meets_target_all_systems,
        "passed": pass_count,
        "failed": total - pass_count,
        "categories": category_summary,
        "results": results,
    }

    out = eval_dir / f"system_check_{run_id}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local functional system check with 95% target gate")
    parser.add_argument("--data-dir", default=os.getenv("AI_DATA_DIR", ".ai_data"))
    parser.add_argument("--files-root", default=os.getenv("AI_FILES_ROOT", str(Path.cwd())))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=95.0)
    parser.add_argument("--task-limit", type=int, default=0, help="0 runs all checks")
    parser.add_argument("--isolated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persist-db", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--db-path", default="")
    parser.add_argument("--fail-below", action="store_true", help="Exit non-zero if score is below min-score")
    args = parser.parse_args()

    out, report = run_system_check(
        data_dir=Path(args.data_dir),
        files_root=Path(args.files_root),
        model=str(args.model),
        ollama_url=str(args.ollama_url),
        use_llm=bool(args.use_llm),
        max_steps=int(args.max_steps),
        min_score=float(args.min_score),
        task_limit=int(args.task_limit),
        isolated=bool(args.isolated),
        persist_db=bool(args.persist_db),
        db_path=Path(args.db_path).expanduser().resolve() if str(args.db_path or "").strip() else None,
    )
    print(f"System check score: {report['score']} (target {report['target_score']})")
    print(f"Passed: {report['passed']} / {report['check_count']}")
    print(f"Meets target: {report['meets_target']}")
    print(f"Categories meet target: {report['categories_meet_target']}")
    print(f"Meets target for all systems: {report['meets_target_all_systems']}")
    print(f"Run ID: {report['run_id']}")
    print(f"Isolated: {report['isolated']}")
    print(f"DB: {report['db_path']}")
    print(f"Run data dir: {report['data_dir']}")
    print(f"Report: {out}")

    if bool(args.fail_below) and not bool(report.get("meets_target_all_systems")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

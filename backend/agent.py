from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path

from .llm import OllamaClient
from .skills import doc_pipeline, folder_audit_pipeline, research_pipeline
from .storage import MemoryFact, SQLiteStore
from .tools.codeexec import detect_test_command, run_command
from .tools.filesystem import FileBrowser
from .tools.web import dictionary_define, download_url, fetch_url_text, search_web


URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", flags=re.IGNORECASE)
DEFAULT_PLANNING_HEURISTICS: dict[str, float] = {
    "retry_attempts": 2.0,
    "docs_priority": 0.62,
    "web_priority": 0.56,
    "planner_confidence": 0.5,
}


@dataclass
class ToolEvent:
    tool: str
    status: str
    detail: str


@dataclass
class AgentRunConfig:
    strict_facts: bool = False
    evidence_mode: bool = False
    prefer_local_core: bool = True
    allow_web: bool = True
    allow_files: bool = True
    allow_docs: bool = True
    allow_code: bool = True
    allow_downloads: bool = False
    max_steps: int = 8


@dataclass
class AgentAction:
    tool: str
    title: str
    args: dict[str, object]
    retryable: bool = False


class LocalAgent:
    def __init__(
        self,
        store: SQLiteStore,
        files: FileBrowser,
        llm: OllamaClient,
        downloads_dir: Path,
    ) -> None:
        self.store = store
        self.files = files
        self.llm = llm
        self.downloads_dir = Path(downloads_dir)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, session_id: str | None, user_text: str, strict_facts: bool = False) -> dict[str, object]:
        sid = self.store.ensure_session(session_id, title=user_text[:60] or "Chat")
        self.store.add_message(sid, "user", user_text)
        self._extract_memory(sid, user_text)

        tool_events: list[ToolEvent] = []
        context_blocks: list[str] = []

        self._run_file_tools(user_text, tool_events, context_blocks)
        self._run_web_tools(user_text, tool_events, context_blocks)

        if strict_facts and not context_blocks:
            try:
                results = search_web(user_text, max_results=5)
                if results:
                    rendered = "\n".join(
                        f"- {item['title']}\n  {item['url']}\n  {item['snippet']}" for item in results
                    )
                    context_blocks.append(f"Auto web grounding for strict mode:\n{rendered}")
                tool_events.append(
                    ToolEvent("web.search.auto", "ok", f"Auto-grounding returned {len(results)} results")
                )
            except Exception as exc:
                tool_events.append(ToolEvent("web.search.auto", "error", str(exc)))

        doc_hits = self.store.search_chunks(user_text, limit=5)
        if doc_hits:
            rows = []
            for hit in doc_hits:
                src = f" ({hit['source']})" if hit.get("source") else ""
                score = float(hit.get("score") or 0.0)
                depth = int(float(hit.get("radf_depth") or 0.0))
                weight = float(hit.get("radf_weight") or 1.0)
                rows.append(
                    f"- {hit['doc_name']}{src} [score={score:.3f}, depth={depth}, weight={weight:.3f}]: {str(hit['text'])[:260]}"
                )
            context_blocks.append("Relevant indexed documents:\n" + "\n".join(rows))

        facts = self.store.memory_for_session(sid)[:6]
        if facts:
            context_blocks.append("Session memory:\n" + "\n".join(_format_fact(f) for f in facts))

        status = self.llm.status()
        answer: str
        mode: str
        if status.available:
            mode = "ollama"
            try:
                history = self.store.recent_messages(sid, limit=12)
                context = "\n\n".join(context_blocks).strip()
                augmented_user = user_text
                if context:
                    augmented_user += (
                        "\n\nUse this context when relevant. Quote concrete URLs/paths in your answer:\n"
                        + context
                    )
                prompt_messages = history[:-1] + [{"role": "user", "content": augmented_user}]
                answer = self.llm.chat(prompt_messages, _system_prompt(self.files.root, strict_facts))
                if not answer:
                    raise RuntimeError("Model returned empty answer")
            except Exception as exc:
                mode = "rules-fallback"
                tool_events.append(ToolEvent("llm", "error", f"Ollama call failed: {exc}"))
                answer = _fallback_answer(user_text, tool_events, context_blocks, strict_facts)
        else:
            mode = "rules"
            answer = _fallback_answer(user_text, tool_events, context_blocks, strict_facts)

        self.store.add_message(sid, "assistant", answer)
        return {
            "session_id": sid,
            "answer": answer,
            "mode": mode,
            "strict_facts": strict_facts,
            "model": status.model,
            "model_available": status.available,
            "model_reason": status.reason,
            "tool_events": [event.__dict__ for event in tool_events],
            "memory": [f.__dict__ for f in facts],
        }

    def run_agent(
        self,
        session_id: str | None,
        user_text: str,
        config: AgentRunConfig | None = None,
    ) -> dict[str, object]:
        cfg = config or AgentRunConfig()
        cfg.max_steps = max(1, min(int(cfg.max_steps), 32))

        sid = self.store.ensure_session(session_id, title=user_text[:60] or "Agent task")
        self.store.add_message(sid, "user", user_text)
        self._extract_memory(sid, user_text)
        heuristics = self.store.get_planning_heuristics(DEFAULT_PLANNING_HEURISTICS)

        actions = self._build_agent_actions(user_text, cfg, heuristics=heuristics)
        if not actions:
            actions = [
                AgentAction(
                    tool="answer.compose",
                    title="Compose final answer",
                    args={"query": user_text},
                    retryable=False,
                )
            ]

        plan: list[dict[str, object]] = [
            {
                "step_id": idx,
                "title": action.title,
                "tool": action.tool,
                "status": "pending",
                "detail": "",
            }
            for idx, action in enumerate(actions, start=1)
        ]

        tool_calls: list[dict[str, object]] = []
        provenance: list[dict[str, object]] = []
        context_blocks: list[str] = []
        artifact_keys: set[tuple[str, str, str]] = set()

        task_id = self.store.create_task(
            sid,
            title=user_text[:120] or "Agent task",
            status="running",
            steps=plan,
            outputs={},
            provenance=provenance,
        )

        answer = ""
        mode = "rules-agent"
        model_status = self.llm.status()
        done = False
        llm_used = False
        precomputed_evidence: list[dict[str, object]] = []

        for idx, action in enumerate(actions):
            if idx >= cfg.max_steps:
                break

            plan[idx]["status"] = "running"
            self.store.update_task(task_id, status="running", steps=plan, outputs={"mode": mode}, provenance=provenance)

            result = self._execute_action_with_retry(
                action=action,
                session_id=sid,
                user_text=user_text,
                cfg=cfg,
                context_blocks=context_blocks,
                provenance=provenance,
                heuristics=heuristics,
            )

            for call in result.get("tool_calls", []):
                tool_calls.append(call)

            context_block = str(result.get("context_block") or "").strip()
            if context_block:
                context_blocks.append(context_block)

            for item in result.get("provenance", []):
                provenance.append(item)
                self._record_artifact_from_provenance(sid, item, artifact_keys)

            if str(result.get("status")) == "ok":
                plan[idx]["status"] = "done"
            else:
                plan[idx]["status"] = "error"
            plan[idx]["detail"] = str(result.get("detail") or "")

            if action.tool == "answer.compose" and str(result.get("status")) == "ok":
                answer = str(result.get("answer") or "")
                mode = str(result.get("mode") or mode)
                llm_used = llm_used or bool(result.get("llm_used"))
                if isinstance(result.get("evidence"), list):
                    precomputed_evidence = list(result.get("evidence") or [])
                done = True

            self.store.update_task(
                task_id,
                status="running" if not done else "completed",
                steps=plan,
                outputs={
                    "mode": mode,
                    "tool_call_count": len(tool_calls),
                    "provenance_count": len(provenance),
                },
                provenance=provenance,
            )

            if done:
                break

        if not done:
            if cfg.evidence_mode:
                precomputed_evidence = self._build_evidence_from_provenance(user_text, provenance, max_claims=6)
                answer = _render_evidence_answer(user_text, precomputed_evidence, strict_facts=cfg.strict_facts)
                mode = "local-evidence"
            else:
                answer, mode, llm_used = self._compose_answer(
                    session_id=sid,
                    user_text=user_text,
                    context_blocks=context_blocks,
                    provenance=provenance,
                    cfg=cfg,
                )
            tool_calls.append(
                {
                    "name": "answer.compose",
                    "args": {"query": user_text},
                    "attempt": 1,
                    "status": "ok",
                    "result_summary": "Composed fallback answer after planner stop",
                }
            )

        evidence: list[dict[str, object]] = []
        if cfg.evidence_mode:
            evidence = precomputed_evidence or self._build_evidence_from_provenance(user_text, provenance, max_claims=6)
            answer = _render_evidence_answer(user_text, evidence, strict_facts=cfg.strict_facts)
            mode = "local-evidence"

        citations = _extract_citations(provenance)
        outcome_id = self.store.add_outcome(
            sid,
            title=user_text[:120] or "Agent outcome",
            summary=answer[:3000],
            status="completed",
            score=float(len(evidence)),
        )

        self.store.add_message(sid, "assistant", answer)
        adaptive_update = self._analyze_and_adapt(
            session_id=sid,
            task_id=task_id,
            user_text=user_text,
            cfg=cfg,
            plan=plan,
            tool_calls=tool_calls,
            provenance=provenance,
            evidence=evidence,
            answer=answer,
            heuristics=heuristics,
        )
        memory_snapshot = self.store.memory_snapshot(sid, limit=120)
        skills_called = sorted(
            {
                str(call.get("name") or "")
                for call in tool_calls
                if str(call.get("name") or "").startswith("skill.")
            }
        )

        self.store.update_task(
            task_id,
            status="completed",
            steps=plan,
            outputs={
                "mode": mode,
                "answer": answer,
                "tool_call_count": len(tool_calls),
                "provenance_count": len(provenance),
                "citation_count": len(citations),
                "evidence_count": len(evidence),
                "outcome_id": outcome_id,
                "llm_used": llm_used,
                "adaptive_update": adaptive_update,
            },
            provenance=provenance,
        )

        return {
            "task_id": task_id,
            "session_id": sid,
            "answer": answer,
            "mode": mode,
            "strict_facts": cfg.strict_facts,
            "evidence_mode": cfg.evidence_mode,
            "model": model_status.model,
            "model_available": model_status.available,
            "model_reason": model_status.reason,
            "plan": plan,
            "tool_calls": tool_calls,
            "citations": citations,
            "provenance": provenance,
            "evidence": evidence,
            "outcome_id": outcome_id,
            "memory": memory_snapshot,
            "llm_used": llm_used,
            "adaptive_update": adaptive_update,
            "original_work_used": {
                "recursive_adic_ranking": bool(getattr(self.store, "use_recursive_adic", False)),
                "planner_executor": True,
                "structured_memory": True,
                "adaptive_planner": True,
                "skills_called": skills_called,
                "evidence_mode_enforced": bool(cfg.evidence_mode),
                "prefer_local_core": bool(cfg.prefer_local_core),
                "llm_assist_used": bool(llm_used),
            },
            "done": True,
        }

    def _build_agent_actions(
        self,
        user_text: str,
        cfg: AgentRunConfig,
        heuristics: dict[str, float] | None = None,
    ) -> list[AgentAction]:
        heuristics = heuristics or {}
        docs_priority = _clamp(
            float(heuristics.get("docs_priority", DEFAULT_PLANNING_HEURISTICS["docs_priority"])),
            0.1,
            0.95,
        )
        web_priority = _clamp(
            float(heuristics.get("web_priority", DEFAULT_PLANNING_HEURISTICS["web_priority"])),
            0.1,
            0.95,
        )
        planner_confidence = _clamp(
            float(heuristics.get("planner_confidence", DEFAULT_PLANNING_HEURISTICS["planner_confidence"])),
            0.0,
            1.0,
        )

        actions: list[AgentAction] = []
        low = user_text.lower()
        research_requested = any(token in low for token in ["research pipeline", "run research", "literature scan"])
        doc_pipeline_requested = any(token in low for token in ["doc pipeline", "document pipeline", "evidence sweep"])
        folder_audit_requested = any(token in low for token in ["folder audit", "repo audit", "audit folder", "audit repo"])

        if cfg.allow_web and research_requested:
            query = _extract_research_query(user_text)
            actions.append(
                AgentAction(
                    tool="skill.research_pipeline",
                    title=f"Run research pipeline for '{query}'",
                    args={"query": query, "max_results": 6, "fetch_top": 2},
                    retryable=True,
                )
            )

        if cfg.allow_docs and doc_pipeline_requested:
            query = _extract_doc_pipeline_query(user_text)
            actions.append(
                AgentAction(
                    tool="skill.doc_pipeline",
                    title=f"Run doc pipeline for '{query}'",
                    args={"query": query, "limit": 10},
                )
            )

        if cfg.allow_files and folder_audit_requested:
            folder = _extract_folder_audit_path(user_text)
            actions.append(
                AgentAction(
                    tool="skill.folder_audit_pipeline",
                    title=f"Run folder audit pipeline in {folder}",
                    args={"path": folder, "max_entries": 700, "max_depth": 3},
                )
            )

        if cfg.allow_files:
            list_match = re.search(
                r"\b(?:list|show)\s+(?:files|folders|directories)(?:\s+in\s+(.+))?",
                user_text,
                re.IGNORECASE,
            )
            if list_match:
                raw_path = (list_match.group(1) or ".").strip().strip("`\"")
                actions.append(
                    AgentAction(
                        tool="files.list",
                        title=f"List files in {raw_path}",
                        args={"path": raw_path},
                    )
                )

            read_match = re.search(r"\b(?:read|open|show|analyze)\s+file\s+(.+)", user_text, re.IGNORECASE)
            if read_match:
                raw_path = read_match.group(1).strip().strip("`\"")
                actions.append(
                    AgentAction(
                        tool="files.read",
                        title=f"Read file {raw_path}",
                        args={"path": raw_path},
                    )
                )

            if "files" in low and ("search" in low or "find" in low):
                search_match = re.search(
                    r"\b(?:search|find)\s+files?\s+(?:for\s+)?(.+?)(?:\s+in\s+(.+))?$",
                    user_text,
                    re.IGNORECASE,
                )
                if search_match:
                    needle = search_match.group(1).strip().strip("`\"")
                    raw_path = (search_match.group(2) or ".").strip().strip("`\"")
                    actions.append(
                        AgentAction(
                            tool="files.search",
                            title=f"Search files for '{needle}'",
                            args={"query": needle, "path": raw_path},
                        )
                    )

        if cfg.allow_web and not research_requested:
            dict_term = _extract_dictionary_term(user_text)
            if dict_term:
                actions.append(
                    AgentAction(
                        tool="web.dictionary",
                        title=f"Lookup dictionary definition for '{dict_term}'",
                        args={"word": dict_term},
                        retryable=True,
                    )
                )

            search_match = re.search(
                r"\b(?:search|look up|find|research)\s+(?:the\s+web|web|internet)?\s*(?:for)?\s+(.+)",
                user_text,
                re.IGNORECASE,
            )
            if search_match and "files" not in low:
                query = search_match.group(1).strip()
                query = re.sub(r"\s+and\s+download.*$", "", query, flags=re.IGNORECASE)
                actions.append(
                    AgentAction(
                        tool="web.search",
                        title=f"Search web for '{query}'",
                        args={"query": query},
                        retryable=True,
                    )
                )

            urls = URL_RE.findall(user_text)
            want_fetch = any(
                token in low for token in ["summarize", "read", "open", "extract", "analyze", "website", "site"]
            )
            if urls and want_fetch:
                for url in urls[:2]:
                    actions.append(
                        AgentAction(
                            tool="web.fetch",
                            title=f"Fetch URL {url}",
                            args={"url": url},
                            retryable=True,
                        )
                    )

            if cfg.allow_downloads and urls and "download" in low:
                for url in urls[:2]:
                    actions.append(
                        AgentAction(
                            tool="web.download",
                            title=f"Download URL {url}",
                            args={"url": url},
                            retryable=True,
                        )
                    )

        if cfg.allow_code:
            math_expr = _extract_math_expression(user_text)
            if math_expr:
                actions.append(
                    AgentAction(
                        tool="math.eval",
                        title=f"Evaluate expression: {math_expr[:64]}",
                        args={"expression": math_expr},
                    )
                )

        if cfg.allow_code:
            code_gen = _extract_code_generation_target(user_text)
            if code_gen:
                target_path = str(code_gen.get("path") or "").strip()
                actions.append(
                    AgentAction(
                        tool="code.generate",
                        title=f"Generate code{' in ' + target_path if target_path else ''}",
                        args={"request": str(code_gen.get("instructions") or user_text), "path": target_path},
                        retryable=False,
                    )
                )

            run_cmd = _extract_run_command(user_text)
            if run_cmd:
                actions.append(
                    AgentAction(
                        tool="code.run",
                        title=f"Run command: {run_cmd[:72]}",
                        args={"command": run_cmd, "cwd": _extract_optional_cwd(user_text)},
                        retryable=False,
                    )
                )

            run_tests = _looks_like_test_request(user_text)
            if run_tests:
                actions.append(
                    AgentAction(
                        tool="code.test",
                        title=f"Run tests in {_extract_optional_cwd(user_text)}",
                        args={"cwd": _extract_optional_cwd(user_text), "runner": "auto"},
                        retryable=True,
                    )
                )

        if cfg.strict_facts and cfg.allow_web and not actions:
            actions.append(
                AgentAction(
                    tool="web.search.auto",
                    title="Auto-ground with web search for strict fact mode",
                    args={"query": user_text},
                    retryable=True,
                )
            )

        if cfg.allow_docs and not doc_pipeline_requested:
            docs_action = AgentAction(
                tool="docs.retrieve",
                title="Retrieve relevant indexed documents",
                args={"query": user_text, "limit": 6},
            )
            docs_first = docs_priority >= web_priority or planner_confidence < 0.45
            if docs_first:
                insert_idx = next(
                    (
                        i
                        for i, a in enumerate(actions)
                        if a.tool.startswith("web.") or a.tool == "skill.research_pipeline"
                    ),
                    len(actions),
                )
                actions.insert(insert_idx, docs_action)
            else:
                actions.append(docs_action)

        actions.append(
            AgentAction(
                tool="answer.compose",
                title="Compose final answer",
                args={"query": user_text},
            )
        )

        return actions

    def _execute_action_with_retry(
        self,
        action: AgentAction,
        session_id: str,
        user_text: str,
        cfg: AgentRunConfig,
        context_blocks: list[str],
        provenance: list[dict[str, object]],
        heuristics: dict[str, float] | None = None,
    ) -> dict[str, object]:
        heuristics = heuristics or {}
        retry_attempts = int(
            round(float(heuristics.get("retry_attempts", DEFAULT_PLANNING_HEURISTICS["retry_attempts"])))
        )
        retry_attempts = int(_clamp(float(retry_attempts), 1.0, 4.0))
        planner_confidence = _clamp(
            float(heuristics.get("planner_confidence", DEFAULT_PLANNING_HEURISTICS["planner_confidence"])),
            0.0,
            1.0,
        )
        if action.retryable and planner_confidence < 0.4:
            retry_attempts = min(4, retry_attempts + 1)

        attempts = retry_attempts if action.retryable else 1
        trace: list[dict[str, object]] = []

        for attempt in range(1, attempts + 1):
            try:
                result = self._execute_action(
                    action=action,
                    session_id=session_id,
                    user_text=user_text,
                    cfg=cfg,
                    context_blocks=context_blocks,
                    provenance=provenance,
                )
                call_status = str(result.get("status") or "ok")
                detail = str(result.get("detail") or "ok")
                if call_status != "ok":
                    status = "retry" if attempt < attempts else "error"
                    trace.append(
                        {
                            "name": action.tool,
                            "args": action.args,
                            "attempt": attempt,
                            "status": status,
                            "result_summary": detail,
                        }
                    )
                    if attempt < attempts:
                        continue
                    result["status"] = "error"
                    result["tool_calls"] = trace
                    return result
                trace.append(
                    {
                        "name": action.tool,
                        "args": action.args,
                        "attempt": attempt,
                        "status": "ok",
                        "result_summary": detail,
                    }
                )
                result["tool_calls"] = trace
                return result
            except Exception as exc:
                message = str(exc)
                status = "retry" if attempt < attempts else "error"
                trace.append(
                    {
                        "name": action.tool,
                        "args": action.args,
                        "attempt": attempt,
                        "status": status,
                        "result_summary": message,
                    }
                )
                if attempt >= attempts:
                    return {
                        "status": "error",
                        "detail": message,
                        "tool_calls": trace,
                        "context_block": "",
                        "provenance": [],
                    }

        return {
            "status": "error",
            "detail": "Unexpected retry termination",
            "tool_calls": trace,
            "context_block": "",
            "provenance": [],
        }

    def _execute_action(
        self,
        action: AgentAction,
        session_id: str,
        user_text: str,
        cfg: AgentRunConfig,
        context_blocks: list[str],
        provenance: list[dict[str, object]],
    ) -> dict[str, object]:
        tool = action.tool

        if tool == "skill.research_pipeline":
            out = research_pipeline(
                query=str(action.args.get("query") or user_text),
                store=self.store,
                max_results=int(action.args.get("max_results") or 6),
                fetch_top=int(action.args.get("fetch_top") or 2),
            )
            return {
                "status": out.status,
                "detail": out.detail,
                "context_block": out.context_block,
                "provenance": out.provenance,
            }

        if tool == "skill.doc_pipeline":
            out = doc_pipeline(
                query=str(action.args.get("query") or user_text),
                store=self.store,
                limit=int(action.args.get("limit") or 10),
            )
            return {
                "status": out.status,
                "detail": out.detail,
                "context_block": out.context_block,
                "provenance": out.provenance,
            }

        if tool == "skill.folder_audit_pipeline":
            out = folder_audit_pipeline(
                path=str(action.args.get("path") or "."),
                files=self.files,
                store=self.store,
                max_entries=int(action.args.get("max_entries") or 700),
                max_depth=int(action.args.get("max_depth") or 3),
            )
            return {
                "status": out.status,
                "detail": out.detail,
                "context_block": out.context_block,
                "provenance": out.provenance,
            }

        if tool == "files.list":
            raw_path = str(action.args.get("path") or ".")
            rows = self.files.list_dir(raw_path)
            preview = rows[:80]
            rendered = "\n".join(
                f"- {'[DIR] ' if bool(r['is_dir']) else ''}{r['name']} ({r['size']} bytes)"
                for r in preview
            )
            doc_id = self.store.add_document(
                name=f"list:{raw_path}",
                source=str(self.files.resolve(raw_path)),
                kind="file-listing",
                text=rendered,
            )
            prov = [
                _provenance_item(
                    source_type="file",
                    source=str(self.files.resolve(raw_path)),
                    snippet=rendered[:260],
                    doc_id=doc_id,
                    path=str(self.files.resolve(raw_path)),
                )
            ]
            return {
                "status": "ok",
                "detail": f"Listed {len(rows)} entries in {raw_path}",
                "context_block": f"Directory listing for {raw_path}:\n{rendered}",
                "provenance": prov,
            }

        if tool == "files.read":
            raw_path = str(action.args.get("path") or "")
            text = self.files.read_text(raw_path)
            resolved = str(self.files.resolve(raw_path))
            doc_id = self.store.add_document(name=raw_path, source=resolved, kind="file-text", text=text)
            prov = [
                _provenance_item(
                    source_type="file",
                    source=resolved,
                    snippet=text[:320],
                    doc_id=doc_id,
                    path=resolved,
                )
            ]
            return {
                "status": "ok",
                "detail": f"Read file {raw_path}",
                "context_block": f"File content ({raw_path}):\n{text[:6000]}",
                "provenance": prov,
            }

        if tool == "files.search":
            needle = str(action.args.get("query") or "")
            raw_path = str(action.args.get("path") or ".")
            hits = self.files.search_text(needle, raw_path)
            rendered = "\n".join(f"- {h['path']}:{h['line']} {str(h['text'])[:180]}" for h in hits[:80])
            context_block = f"File search hits for '{needle}' in {raw_path}:\n{rendered}" if rendered else ""

            doc_id: str | None = None
            if rendered:
                doc_id = self.store.add_document(
                    name=f"search:{needle}",
                    source=str(self.files.resolve(raw_path)),
                    kind="file-search",
                    text=rendered,
                )

            prov: list[dict[str, object]] = []
            for hit in hits[:20]:
                prov.append(
                    _provenance_item(
                        source_type="file",
                        source=str(hit.get("path") or ""),
                        snippet=str(hit.get("text") or "")[:320],
                        doc_id=doc_id,
                        path=str(hit.get("path") or ""),
                    )
                )

            return {
                "status": "ok",
                "detail": f"Found {len(hits)} matches for '{needle}'",
                "context_block": context_block,
                "provenance": prov,
            }

        if tool in {"web.search", "web.search.auto"}:
            query = str(action.args.get("query") or user_text)
            results = search_web(query, max_results=6)
            rendered = "\n".join(
                f"- {item['title']}\n  {item['url']}\n  {item['snippet']}" for item in results
            )
            prov: list[dict[str, object]] = []
            for item in results[:6]:
                title = str(item.get("title") or item.get("url") or "web result")
                url = str(item.get("url") or "")
                snippet = str(item.get("snippet") or "").strip()
                doc_text = f"{title}\n{url}\n{snippet}".strip()
                doc_id = self.store.add_document(name=title[:120], source=url, kind="web-search-snippet", text=doc_text)
                prov.append(_provenance_item(source_type="url", source=url, snippet=snippet[:320], doc_id=doc_id, url=url))

            return {
                "status": "ok",
                "detail": f"Returned {len(results)} results for '{query}'",
                "context_block": f"Web search results for '{query}':\n{rendered}" if rendered else "",
                "provenance": prov,
            }

        if tool == "web.fetch":
            url = str(action.args.get("url") or "")
            text, kind = fetch_url_text(url)
            doc_id = self.store.add_document(name=url, source=url, kind=f"web-{kind}", text=text)
            prov = [
                _provenance_item(
                    source_type="url",
                    source=url,
                    snippet=text[:320],
                    doc_id=doc_id,
                    url=url,
                )
            ]
            return {
                "status": "ok",
                "detail": f"Fetched {url} ({kind})",
                "context_block": f"Fetched {url} ({kind}):\n{text[:7000]}",
                "provenance": prov,
            }

        if tool == "web.download":
            url = str(action.args.get("url") or "")
            info = download_url(url, self.downloads_dir)
            prov = [
                _provenance_item(
                    source_type="file",
                    source=str(info["path"]),
                    snippet=f"Downloaded from {url}",
                    path=str(info["path"]),
                    url=url,
                )
            ]
            return {
                "status": "ok",
                "detail": f"Downloaded {info['filename']} to {info['path']} ({info['size_bytes']} bytes)",
                "context_block": "",
                "provenance": prov,
            }

        if tool == "web.dictionary":
            word = str(action.args.get("word") or "")
            entry = dictionary_define(word, max_definitions=8)
            lines = []
            for row in entry["definitions"]:
                part = str(row.get("part_of_speech") or "").strip()
                definition = str(row.get("definition") or "").strip()
                example = str(row.get("example") or "").strip()
                prefix = f"{part}: " if part else ""
                line = f"- {prefix}{definition}"
                if example:
                    line += f" | example: {example}"
                lines.append(line)

            text = "\n".join(lines)
            source = str(entry.get("source") or "dictionary")
            doc_id = self.store.add_document(name=f"dictionary:{word}", source=source, kind="dictionary", text=text)
            prov = [
                _provenance_item(
                    source_type="url",
                    source=source,
                    snippet=text[:320],
                    doc_id=doc_id,
                    url=source,
                )
            ]
            return {
                "status": "ok",
                "detail": f"Found {len(lines)} definitions for '{word}'",
                "context_block": f"Dictionary facts for '{word}' (source: {source}):\n{text}",
                "provenance": prov,
            }

        if tool == "docs.retrieve":
            query = str(action.args.get("query") or user_text)
            limit = int(action.args.get("limit") or 6)
            hits = self.store.search_chunks(query, limit=limit)

            rows: list[str] = []
            prov: list[dict[str, object]] = []
            for hit in hits:
                src = str(hit.get("source") or "")
                score = float(hit.get("score") or 0.0)
                depth = int(float(hit.get("radf_depth") or 0.0))
                weight = float(hit.get("radf_weight") or 1.0)
                rows.append(
                    f"- {hit['doc_name']} ({src}) [score={score:.3f}, depth={depth}, weight={weight:.3f}]: {str(hit['text'])[:260]}"
                )
                prov.append(
                    _provenance_item(
                        source_type="doc",
                        source=src or str(hit.get("doc_name") or "document"),
                        snippet=str(hit.get("text") or "")[:320],
                        doc_id=str(hit.get("doc_id") or ""),
                    )
                )

            return {
                "status": "ok",
                "detail": f"Retrieved {len(hits)} indexed document chunks",
                "context_block": "Relevant indexed documents:\n" + "\n".join(rows) if rows else "",
                "provenance": prov,
            }

        if tool == "math.eval":
            expr = str(action.args.get("expression") or "").strip()
            if not expr:
                return {
                    "status": "error",
                    "detail": "Missing expression for math.eval",
                    "context_block": "",
                    "provenance": [],
                }
            value = _safe_eval_expression(expr)
            rendered = f"Math evaluation:\nexpression: {expr}\nresult: {value}"
            doc_id = self.store.add_document(
                name=f"math:{expr[:64]}",
                source="local-math-eval",
                kind="math-eval",
                text=rendered,
            )
            prov = [
                _provenance_item(
                    source_type="calc",
                    source="local-math-eval",
                    snippet=f"{expr} = {value}",
                    doc_id=doc_id,
                )
            ]
            return {
                "status": "ok",
                "detail": f"Evaluated expression '{expr}' = {value}",
                "context_block": rendered,
                "provenance": prov,
            }

        if tool == "code.generate":
            request = str(action.args.get("request") or user_text).strip()
            raw_path = str(action.args.get("path") or "").strip().strip("`\"")
            status = self.llm.status()
            generator = "template"
            code = ""
            if status.available:
                try:
                    code_prompt = (
                        f"Write production-ready code for this request:\n{request}\n\n"
                        "Return only code with no markdown fences."
                    )
                    code = self.llm.chat(
                        messages=[{"role": "user", "content": code_prompt}],
                        system=(
                            "You are a local coding assistant. Generate clear, correct code. "
                            "Output only code."
                        ),
                    ).strip()
                    if code:
                        generator = "llm"
                except Exception:
                    code = ""

            if not code:
                code = _template_code_for_request(request, raw_path)
                generator = "template"

            code = _strip_code_fences(code).strip()
            if not code:
                code = _template_code_for_request(request, raw_path)
                generator = "template"

            if raw_path:
                write_info = self.files.write_text(raw_path, code)
                resolved = str(write_info["path"])
                doc_id = self.store.add_document(
                    name=f"generated:{raw_path}",
                    source=resolved,
                    kind="code-generated",
                    text=code,
                )
                prov = [
                    _provenance_item(
                        source_type="file",
                        source=resolved,
                        snippet=code[:320],
                        doc_id=doc_id,
                        path=resolved,
                    )
                ]
                detail = f"Generated code via {generator} and wrote {raw_path}"
                context_block = f"Generated code ({generator}) for {raw_path}:\n{code[:7000]}"
            else:
                doc_id = self.store.add_document(
                    name="generated-code-snippet",
                    source="local-code-generator",
                    kind="code-generated-snippet",
                    text=code,
                )
                prov = [
                    _provenance_item(
                        source_type="doc",
                        source="local-code-generator",
                        snippet=code[:320],
                        doc_id=doc_id,
                    )
                ]
                detail = f"Generated code snippet via {generator}"
                context_block = f"Generated code snippet ({generator}):\n{code[:7000]}"

            return {
                "status": "ok",
                "detail": detail,
                "context_block": context_block,
                "provenance": prov,
            }

        if tool == "code.run":
            command = str(action.args.get("command") or "").strip()
            raw_cwd = str(action.args.get("cwd") or ".").strip() or "."
            if not command:
                return {
                    "status": "error",
                    "detail": "Missing command for code.run",
                    "context_block": "",
                    "provenance": [],
                }
            cwd = self.files.resolve(raw_cwd)
            result = run_command(command, cwd=cwd, timeout_sec=120)
            rendered = _render_command_result(result)
            doc_id = self.store.add_document(
                name=f"command:{' '.join(result.command)[:90]}",
                source=str(cwd),
                kind="code-run",
                text=rendered,
            )
            prov = [
                _provenance_item(
                    source_type="file",
                    source=str(cwd),
                    snippet=(result.stdout or result.stderr or rendered)[:320],
                    doc_id=doc_id,
                    path=str(cwd),
                )
            ]
            return {
                "status": "ok" if result.ok else "error",
                "detail": (
                    f"Command succeeded: {' '.join(result.command)}"
                    if result.ok
                    else f"Command failed (exit {result.exit_code}): {' '.join(result.command)}"
                ),
                "context_block": rendered,
                "provenance": prov,
            }

        if tool == "code.test":
            raw_cwd = str(action.args.get("cwd") or ".").strip() or "."
            runner = str(action.args.get("runner") or "auto")
            cwd = self.files.resolve(raw_cwd)
            detected = detect_test_command(cwd, runner=runner)
            result = run_command(detected, cwd=cwd, timeout_sec=240)
            rendered = _render_command_result(result)
            doc_id = self.store.add_document(
                name=f"test:{' '.join(detected)[:90]}",
                source=str(cwd),
                kind="code-test",
                text=rendered,
            )
            prov = [
                _provenance_item(
                    source_type="file",
                    source=str(cwd),
                    snippet=(result.stdout or result.stderr or rendered)[:320],
                    doc_id=doc_id,
                    path=str(cwd),
                )
            ]
            return {
                "status": "ok" if result.ok else "error",
                "detail": (
                    f"Tests passed with {' '.join(detected)}"
                    if result.ok
                    else f"Tests failed (exit {result.exit_code}) with {' '.join(detected)}"
                ),
                "context_block": rendered,
                "provenance": prov,
            }

        if tool == "answer.compose":
            if cfg.evidence_mode:
                evidence = self._build_evidence_from_provenance(user_text, provenance, max_claims=6)
                answer = _render_evidence_answer(user_text, evidence, strict_facts=cfg.strict_facts)
                mode = "local-evidence"
                llm_used = False
            else:
                answer, mode, llm_used = self._compose_answer(
                    session_id=session_id,
                    user_text=user_text,
                    context_blocks=context_blocks,
                    provenance=provenance,
                    cfg=cfg,
                )
            return {
                "status": "ok",
                "detail": "Composed final answer",
                "context_block": "",
                "provenance": [],
                "answer": answer,
                "mode": mode,
                "llm_used": llm_used,
                "evidence": evidence if cfg.evidence_mode else [],
            }

        raise ValueError(f"Unknown agent tool: {tool}")

    def _compose_answer(
        self,
        session_id: str,
        user_text: str,
        context_blocks: list[str],
        provenance: list[dict[str, object]],
        cfg: AgentRunConfig,
    ) -> tuple[str, str, bool]:
        facts = self.store.memory_for_session(session_id)[:6]
        if facts:
            context_blocks = [*context_blocks, "Session memory:\n" + "\n".join(_format_fact(f) for f in facts)]

        if cfg.strict_facts and not context_blocks:
            return (
                "Strict Fact Mode is ON and no grounded source context is available yet. "
                "Provide a document/URL/file request or enable additional tools.",
                "rules-agent",
                False,
            )

        status = self.llm.status()
        if status.available and not cfg.prefer_local_core:
            mode = "ollama-agent"
            try:
                history = self.store.recent_messages(session_id, limit=12)
                context = "\n\n".join(context_blocks).strip()
                augmented_user = user_text
                if context:
                    augmented_user += (
                        "\n\nPlanner context and tool observations. "
                        "Only use grounded data for factual claims and cite concrete sources:\n"
                        + context
                    )
                if cfg.evidence_mode:
                    augmented_user += (
                        "\n\nEvidence Mode is ON. Keep claims tied to provided sources so they can be mapped "
                        "to evidence objects with snippets."
                    )

                prompt_messages = history[:-1] + [{"role": "user", "content": augmented_user}]
                answer = self.llm.chat(
                    prompt_messages,
                    _system_prompt(self.files.root, cfg.strict_facts, evidence_mode=cfg.evidence_mode),
                )
                if not answer:
                    raise RuntimeError("Model returned empty answer")
                return answer, mode, True
            except Exception as exc:
                return (
                    _fallback_agent_answer(
                        user_text=user_text,
                        context_blocks=context_blocks,
                        strict_facts=cfg.strict_facts,
                        provenance=provenance,
                        llm_error=str(exc),
                    ),
                    "rules-agent",
                    False,
                )

        return (
            _fallback_agent_answer(
                user_text=user_text,
                context_blocks=context_blocks,
                strict_facts=cfg.strict_facts,
                provenance=provenance,
                llm_error="local core mode (LLM assist disabled)" if cfg.prefer_local_core else "local model unavailable",
            ),
            "rules-agent",
            False,
        )

    def _build_evidence_from_provenance(
        self,
        user_text: str,
        provenance: list[dict[str, object]],
        max_claims: int = 6,
    ) -> list[dict[str, object]]:
        query_tokens = set(_tokenize_for_match(user_text))
        candidates: list[tuple[int, dict[str, object]]] = []

        for item in provenance:
            source = str(item.get("source") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if not source or not snippet:
                continue
            snippet_tokens = set(_tokenize_for_match(snippet))
            overlap = len(query_tokens.intersection(snippet_tokens))
            candidates.append((overlap, item))

        if not candidates:
            return [
                {
                    "claim": "No grounded claim can be made because no source+snippet evidence was collected.",
                    "sources": [],
                    "snippets": [],
                    "confidence": 0.0,
                }
            ]

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected: list[dict[str, object]] = []
        used: set[tuple[str, str]] = set()
        for overlap, item in candidates:
            if len(selected) >= max_claims:
                break
            source = str(item.get("source") or "")
            snippet = str(item.get("snippet") or "")
            key = (source, snippet[:120])
            if key in used:
                continue
            used.add(key)
            selected.append(
                {
                    "overlap": overlap,
                    "source": source,
                    "snippet": snippet,
                    "doc_id": str(item.get("doc_id") or ""),
                }
            )

        evidence: list[dict[str, object]] = []
        for picked in selected:
            snippet = str(picked["snippet"])
            source = str(picked["source"])
            overlap = int(picked["overlap"])
            claim = _claim_from_snippet(snippet, max_chars=180)
            confidence = min(0.98, 0.4 + 0.08 * overlap + 0.1 * (1 if picked.get("doc_id") else 0))
            evidence.append(
                {
                    "claim": claim,
                    "sources": [source],
                    "snippets": [snippet[:260]],
                    "confidence": round(confidence, 2),
                }
            )
        return evidence

    def _run_file_tools(self, user_text: str, tool_events: list[ToolEvent], context_blocks: list[str]) -> None:
        low = user_text.lower().strip()

        list_match = re.search(
            r"\b(?:list|show)\s+(?:files|folders|directories)(?:\s+in\s+(.+))?",
            user_text,
            re.IGNORECASE,
        )
        if list_match:
            raw_path = (list_match.group(1) or ".").strip().strip("`\"")
            try:
                rows = self.files.list_dir(raw_path)
                preview = rows[:40]
                rendered = "\n".join(
                    f"- {'[DIR] ' if bool(r['is_dir']) else ''}{r['name']} ({r['size']} bytes)"
                    for r in preview
                )
                context_blocks.append(f"Directory listing for {raw_path}:\n{rendered}")
                tool_events.append(ToolEvent("files.list", "ok", f"Listed {len(rows)} entries in {raw_path}"))
            except Exception as exc:
                tool_events.append(ToolEvent("files.list", "error", str(exc)))

        read_match = re.search(r"\b(?:read|open|show|analyze)\s+file\s+(.+)", user_text, re.IGNORECASE)
        if read_match:
            raw_path = read_match.group(1).strip().strip("`\"")
            try:
                text = self.files.read_text(raw_path)
                context_blocks.append(f"File content ({raw_path}):\n{text[:6000]}")
                tool_events.append(ToolEvent("files.read", "ok", f"Read file {raw_path}"))
            except Exception as exc:
                tool_events.append(ToolEvent("files.read", "error", str(exc)))

        if "files" in low and ("search" in low or "find" in low):
            search_match = re.search(
                r"\b(?:search|find)\s+files?\s+(?:for\s+)?(.+?)(?:\s+in\s+(.+))?$",
                user_text,
                re.IGNORECASE,
            )
            if search_match:
                needle = search_match.group(1).strip().strip("`\"")
                raw_path = (search_match.group(2) or ".").strip().strip("`\"")
                try:
                    hits = self.files.search_text(needle, raw_path)
                    if hits:
                        rendered = "\n".join(
                            f"- {h['path']}:{h['line']} {str(h['text'])[:180]}" for h in hits[:60]
                        )
                        context_blocks.append(f"File search hits for '{needle}' in {raw_path}:\n{rendered}")
                    tool_events.append(ToolEvent("files.search", "ok", f"Found {len(hits)} matches for '{needle}'"))
                except Exception as exc:
                    tool_events.append(ToolEvent("files.search", "error", str(exc)))

    def _run_web_tools(self, user_text: str, tool_events: list[ToolEvent], context_blocks: list[str]) -> None:
        low = user_text.lower()

        dict_term = _extract_dictionary_term(user_text)
        if dict_term:
            try:
                entry = dictionary_define(dict_term, max_definitions=6)
                lines = []
                for row in entry["definitions"]:
                    part = str(row.get("part_of_speech") or "").strip()
                    definition = str(row.get("definition") or "").strip()
                    example = str(row.get("example") or "").strip()
                    prefix = f"{part}: " if part else ""
                    line = f"- {prefix}{definition}"
                    if example:
                        line += f" | example: {example}"
                    lines.append(line)
                context_blocks.append(
                    f"Dictionary facts for '{entry['word']}' (source: {entry['source']}):\n"
                    + "\n".join(lines)
                )
                tool_events.append(ToolEvent("web.dictionary", "ok", f"Found {len(lines)} definitions for '{dict_term}'"))
            except Exception as exc:
                tool_events.append(ToolEvent("web.dictionary", "error", f"{dict_term}: {exc}"))

        search_match = re.search(
            r"\b(?:search|look up|find|research)\s+(?:the\s+web|web|internet)?\s*(?:for)?\s+(.+)",
            user_text,
            re.IGNORECASE,
        )
        if search_match and "files" not in low:
            query = search_match.group(1).strip()
            query = re.sub(r"\s+and\s+download.*$", "", query, flags=re.IGNORECASE)
            try:
                results = search_web(query, max_results=6)
                if results:
                    rendered = "\n".join(
                        f"- {item['title']}\n  {item['url']}\n  {item['snippet']}" for item in results
                    )
                    context_blocks.append(f"Web search results for '{query}':\n{rendered}")
                tool_events.append(ToolEvent("web.search", "ok", f"Returned {len(results)} results for '{query}'"))
            except Exception as exc:
                tool_events.append(ToolEvent("web.search", "error", str(exc)))

        urls = URL_RE.findall(user_text)
        want_fetch = any(token in low for token in ["summarize", "read", "open", "extract", "analyze", "website", "site"])
        if urls and want_fetch:
            for url in urls[:2]:
                try:
                    text, kind = fetch_url_text(url)
                    context_blocks.append(f"Fetched {url} ({kind}):\n{text[:7000]}")
                    self.store.add_document(name=url, source=url, kind=f"web-{kind}", text=text)
                    tool_events.append(ToolEvent("web.fetch", "ok", f"Fetched {url} ({kind})"))
                except Exception as exc:
                    tool_events.append(ToolEvent("web.fetch", "error", f"{url}: {exc}"))

        if urls and "download" in low:
            for url in urls[:2]:
                try:
                    info = download_url(url, self.downloads_dir)
                    tool_events.append(
                        ToolEvent(
                            "web.download",
                            "ok",
                            f"Downloaded {info['filename']} to {info['path']} ({info['size_bytes']} bytes)",
                        )
                    )
                except Exception as exc:
                    tool_events.append(ToolEvent("web.download", "error", f"{url}: {exc}"))

    def _record_artifact_from_provenance(
        self,
        session_id: str,
        item: dict[str, object],
        seen: set[tuple[str, str, str]],
    ) -> None:
        source_type = str(item.get("source_type") or "")
        source = str(item.get("source") or "")
        doc_id = str(item.get("doc_id") or "")
        location = str(item.get("path") or item.get("url") or source)
        if not location:
            return
        key = (source_type, location, doc_id)
        if key in seen:
            return
        seen.add(key)
        self.store.add_artifact(
            session_id=session_id,
            artifact_type=source_type or "source",
            location=location,
            source=source,
            doc_id=doc_id,
            description=str(item.get("snippet") or "")[:220],
        )

    def _analyze_and_adapt(
        self,
        session_id: str,
        task_id: str,
        user_text: str,
        cfg: AgentRunConfig,
        plan: list[dict[str, object]],
        tool_calls: list[dict[str, object]],
        provenance: list[dict[str, object]],
        evidence: list[dict[str, object]],
        answer: str,
        heuristics: dict[str, float],
    ) -> dict[str, object]:
        total_steps = len(plan)
        done_steps = sum(1 for s in plan if str(s.get("status") or "") == "done")
        error_steps = sum(1 for s in plan if str(s.get("status") or "") == "error")
        done_ratio = done_steps / max(1, total_steps)

        tool_errors = sum(1 for c in tool_calls if str(c.get("status") or "") == "error")
        retry_signals = sum(1 for c in tool_calls if str(c.get("status") or "") == "retry")
        provenance_count = len(provenance)
        has_answer = bool(answer.strip())

        evidence_good = 0
        for item in evidence:
            if not isinstance(item, dict):
                continue
            sources = item.get("sources") or []
            snippets = item.get("snippets") or []
            if sources and snippets:
                evidence_good += 1

        success = has_answer and done_ratio >= 0.5
        if cfg.strict_facts:
            success = success and provenance_count > 0
        if cfg.evidence_mode:
            success = success and evidence_good > 0

        reasons: list[str] = []
        if not has_answer:
            reasons.append("empty_answer")
        if cfg.strict_facts and provenance_count <= 0:
            reasons.append("no_grounded_provenance")
        if cfg.evidence_mode and evidence_good <= 0:
            reasons.append("no_evidence_pairs")
        if error_steps > 0 or tool_errors > 0:
            reasons.append("tool_failures")
        if done_ratio < 0.5:
            reasons.append("low_plan_completion")

        old_retry = _clamp(
            float(heuristics.get("retry_attempts", DEFAULT_PLANNING_HEURISTICS["retry_attempts"])),
            1.0,
            4.0,
        )
        old_docs = _clamp(
            float(heuristics.get("docs_priority", DEFAULT_PLANNING_HEURISTICS["docs_priority"])),
            0.1,
            0.95,
        )
        old_web = _clamp(
            float(heuristics.get("web_priority", DEFAULT_PLANNING_HEURISTICS["web_priority"])),
            0.1,
            0.95,
        )
        old_conf = _clamp(
            float(heuristics.get("planner_confidence", DEFAULT_PLANNING_HEURISTICS["planner_confidence"])),
            0.0,
            1.0,
        )

        new_retry = old_retry
        new_docs = old_docs
        new_web = old_web
        new_conf = old_conf

        if success:
            new_conf = _clamp((old_conf * 0.8) + 0.2, 0.0, 1.0)
            if tool_errors == 0 and retry_signals == 0 and old_retry > 1.5:
                new_retry = _clamp(old_retry - 0.25, 1.0, 4.0)
            if provenance_count >= 2:
                new_docs = _clamp(old_docs + 0.03, 0.1, 0.95)
        else:
            new_conf = _clamp(old_conf * 0.7, 0.0, 1.0)
            if cfg.allow_docs and provenance_count == 0:
                new_docs = _clamp(old_docs + 0.08, 0.1, 0.95)
            if cfg.allow_web and provenance_count == 0:
                new_web = _clamp(old_web + 0.06, 0.1, 0.95)
            if tool_errors > 0 or retry_signals > 0:
                new_retry = _clamp(old_retry + 0.5, 1.0, 4.0)

        updates: dict[str, dict[str, float]] = {}
        candidates = {
            "retry_attempts": (old_retry, new_retry),
            "docs_priority": (old_docs, new_docs),
            "web_priority": (old_web, new_web),
            "planner_confidence": (old_conf, new_conf),
        }
        for key, (old_value, new_value) in candidates.items():
            if abs(new_value - old_value) < 0.01:
                continue
            self.store.upsert_planning_heuristic(key, new_value, source="adaptive-agent")
            updates[key] = {
                "old": round(old_value, 3),
                "new": round(new_value, 3),
                "delta": round(new_value - old_value, 3),
            }

        trigger = ",".join(reasons) if reasons else "success"
        if success:
            rule = (
                "Success pattern: keep grounded planning, preserve docs/web balance, and optimize retries "
                f"(retry={new_retry:.2f}, docs={new_docs:.2f}, web={new_web:.2f})."
            )
        else:
            rule = (
                f"Failure pattern ({trigger}): shift planning heuristics toward better grounding "
                f"(retry={new_retry:.2f}, docs={new_docs:.2f}, web={new_web:.2f})."
            )

        confidence = _clamp(
            0.35
            + (0.35 * done_ratio)
            + (0.15 if provenance_count > 0 else 0.0)
            + (0.15 if success else 0.0)
            - (0.1 * float(tool_errors)),
            0.05,
            0.98,
        )
        rule_id = self.store.add_improvement_rule(
            session_id=session_id,
            task_id=task_id,
            rule=rule[:500],
            trigger=trigger[:120],
            confidence=round(confidence, 3),
        )

        return {
            "task_success": bool(success),
            "failure_reasons": reasons,
            "heuristic_updates": updates,
            "improvement_rule_id": rule_id,
            "improvement_rule": rule,
            "trigger": trigger,
            "metrics": {
                "done_steps": done_steps,
                "total_steps": total_steps,
                "tool_errors": tool_errors,
                "retry_signals": retry_signals,
                "provenance_count": provenance_count,
                "evidence_good": evidence_good,
                "has_answer": has_answer,
                "query_preview": user_text[:120],
            },
        }

    def _extract_memory(self, session_id: str, user_text: str) -> None:
        text = user_text.strip()

        name = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,48})", text, re.IGNORECASE)
        if name:
            value = name.group(1).strip()
            self.store.upsert_memory(session_id, "name", value)
            self.store.upsert_fact(session_id, "name", value, source="user")

        goal = re.search(r"\bmy goal is\s+([^.!?\n]{3,240})", text, re.IGNORECASE)
        if goal:
            value = goal.group(1).strip()
            self.store.upsert_memory(session_id, "goal", value)
            self.store.upsert_fact(session_id, "goal", value, source="user")

        pref = re.search(r"\bi (?:prefer|like)\s+([^.!?\n]{3,240})", text, re.IGNORECASE)
        if pref:
            value = pref.group(1).strip()
            self.store.upsert_memory(session_id, "preference", value)
            self.store.upsert_preference(session_id, "preference", value, source="user")

        need = re.search(r"\bi need\s+([^.!?\n]{3,240})", text, re.IGNORECASE)
        if need:
            value = need.group(1).strip()
            self.store.upsert_memory(session_id, "need", value)
            self.store.upsert_fact(session_id, "need", value, source="user")


def _format_fact(fact: MemoryFact) -> str:
    return f"- {fact.key}: {fact.value}"


def _system_prompt(files_root: Path, strict_facts: bool, evidence_mode: bool = False) -> str:
    prompt = (
        "You are a local, modular AI assistant focused on practical execution. "
        "Use provided tool context (web results, fetched pages, files, and indexed documents) "
        "to answer with concrete details. If data is missing, say exactly what is missing. "
        "Document retrieval is ranked with a Recursive-Adic depth weighting, so prioritize higher-scored chunks. "
        "When citing sources, include URLs or file paths explicitly. "
        f"You can access files under this root: {files_root}."
    )
    if strict_facts:
        prompt += (
            " Strict facts mode is enabled: do not invent facts. "
            "Only state factual claims that are supported by provided sources. "
            "If sources are insufficient, explicitly say verification is insufficient."
        )
    if evidence_mode:
        prompt += (
            " Evidence mode is enabled: organize key claims so they can be traced to source snippets."
        )
    return prompt


def _fallback_answer(user_text: str, events: list[ToolEvent], context_blocks: list[str], strict_facts: bool) -> str:
    points = _collect_grounded_points(context_blocks, provenance=[], limit=6)
    sources = _extract_sources_from_context(context_blocks, limit=8)
    lines: list[str] = []

    if points:
        lines.append("Here's what I found from local grounded context:")
        lines.append("")
        lines.append(" ".join(p["text"] for p in points[:3]))
        if len(points) > 3:
            lines.append("")
            lines.append("Additional points:")
            for item in points[3:6]:
                lines.append(f"- {item['text']}")
    else:
        if strict_facts:
            lines.append(
                "I could not collect grounded source text for that request yet, "
                "so I am not making unsupported factual claims."
            )
        else:
            lines.append(
                "I can help with web search, website reading, file/document analysis, OCR, and persistent memory. "
                "Ask a direct question and I will generate a full answer from the retrieved context."
            )

    if strict_facts:
        lines.append("")
        lines.append("Strict Fact Mode: only source-backed claims are included.")

    if sources:
        lines.append("")
        lines.append("Sources:")
        for src in sources:
            lines.append(f"- {src}")

    if events:
        lines.append("")
        lines.append("Tool activity:")
        for event in events:
            lines.append(f"- [{event.status}] {event.tool}: {event.detail}")

    lines.append("")
    lines.append(f"Request handled: {user_text[:160]}")
    return "\n".join(lines).strip()


def _fallback_agent_answer(
    user_text: str,
    context_blocks: list[str],
    strict_facts: bool,
    provenance: list[dict[str, object]],
    llm_error: str,
) -> str:
    points = _collect_grounded_points(context_blocks, provenance=provenance, limit=6)
    sources = _extract_sources_from_context(context_blocks, provenance=provenance, limit=8)
    lines: list[str] = []

    if points:
        lines.append("Here's what I found from grounded local sources:")
        lines.append("")
        lines.append(" ".join(p["text"] for p in points[:3]))
        if len(points) > 3:
            lines.append("")
            lines.append("Additional grounded points:")
            for item in points[3:6]:
                lines.append(f"- {item['text']}")
    else:
        lines.append("I could not collect enough grounded source text to answer that yet.")

    if strict_facts:
        lines.append("")
        lines.append("Strict Fact Mode: only source-backed claims are included.")

    if sources:
        lines.append("")
        lines.append("Sources:")
        for src in sources:
            lines.append(f"- {src}")

    if llm_error:
        lines.append("")
        lines.append(f"Generation mode: local rules ({llm_error}).")

    lines.append("")
    lines.append(f"Request handled: {user_text[:160]}")
    return "\n".join(lines).strip()


def _extract_dictionary_term(text: str) -> str:
    patterns = [
        r"\bdefine\s+([A-Za-z][A-Za-z\-']{1,63})\b",
        r"\bdefinition of\s+([A-Za-z][A-Za-z\-']{1,63})\b",
        r"\bwhat does\s+([A-Za-z][A-Za-z\-']{1,63})\s+mean\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
    return ""


def _extract_research_query(text: str) -> str:
    m = re.search(r"\b(?:research pipeline|run research|literature scan)\s+(?:for|on)?\s*(.+)$", text, re.IGNORECASE)
    if m:
        q = m.group(1).strip()
        if q:
            return q
    return text.strip()


def _extract_doc_pipeline_query(text: str) -> str:
    m = re.search(r"\b(?:doc pipeline|document pipeline|evidence sweep)\s+(?:for|on)?\s*(.+)$", text, re.IGNORECASE)
    if m:
        q = m.group(1).strip()
        if q:
            return q
    return text.strip()


def _extract_folder_audit_path(text: str) -> str:
    m = re.search(r"\b(?:folder audit|audit folder|repo audit|audit repo)\s+(?:in|on|for)?\s*(.+)$", text, re.IGNORECASE)
    if not m:
        return "."
    raw = m.group(1).strip().strip("`\"")
    if not raw:
        return "."
    if raw.lower() in {"this", "here", "current", "repo", "repository"}:
        return "."
    return raw


def _looks_like_test_request(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search(
            r"\b(?:run|execute)\s+(?:all\s+)?tests?\b|\bpytest\b|\bnpm\s+test\b|\bgo\s+test\b|\bcargo\s+test\b",
            low,
        )
    )


def _extract_run_command(text: str) -> str:
    m = re.search(r"\b(?:run|execute)\s+command\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        raw = m.group(1).strip().strip("`")
        raw = re.split(r"\s+and\s+then\s+", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        raw = re.sub(r"\s+(?:in|at)\s+[~./A-Za-z0-9_\-][^\n]*$", "", raw, flags=re.IGNORECASE).strip()
        return raw

    low = text.strip().lower()
    for prefix in ("pytest", "npm test", "go test", "cargo test", "python ", "python3 ", "node "):
        if low.startswith(prefix):
            raw = text.strip().strip("`")
            raw = re.sub(r"\s+(?:in|at)\s+[~./A-Za-z0-9_\-][^\n]*$", "", raw, flags=re.IGNORECASE).strip()
            return raw
    return ""


def _extract_optional_cwd(text: str) -> str:
    m = re.search(r"\b(?:in|at)\s+([~./A-Za-z0-9_\-][^\n]{0,200})$", text.strip(), re.IGNORECASE)
    if not m:
        return "."
    value = m.group(1).strip().strip("`\"")
    if not value:
        return "."
    return value


def _extract_code_generation_target(text: str) -> dict[str, str]:
    patterns = [
        r"\b(?:create|write|generate)\s+file\s+([^\s]+)\s+(?:with|for)\s+(.+)$",
        r"\b(?:implement|write code|generate code)\s+in\s+(?:file\s+)?([^\s]+)\s*[:,-]?\s*(.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            path = m.group(1).strip().strip("`\"")
            instructions = m.group(2).strip()
            if instructions:
                return {"path": path, "instructions": instructions}

    low = text.lower()
    if any(token in low for token in ["write code", "generate code", "implement this", "code this"]):
        return {"path": "", "instructions": text.strip()}
    return {}


def _extract_math_expression(text: str) -> str:
    patterns = [
        r"\b(?:calculate|compute|evaluate|solve)\s+([0-9\.\+\-\*\/\(\)\s\^%]{2,120})$",
        r"\bwhat is\s+([0-9\.\+\-\*\/\(\)\s\^%]{2,120})\??$",
    ]
    stripped = text.strip()
    for pattern in patterns:
        m = re.search(pattern, stripped, re.IGNORECASE)
        if not m:
            continue
        expr = m.group(1).strip()
        expr = expr.replace("^", "**")
        expr = re.sub(r"\s+", " ", expr)
        if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s%]{2,160}|[0-9\.\+\-\*\/\(\)\s%]{1,160}\*\*[0-9\.\+\-\*\/\(\)\s%]{1,160}", expr):
            return expr
    return ""


def _extract_citations(provenance: list[dict[str, object]]) -> list[dict[str, object]]:
    citations: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in provenance:
        source_type = str(item.get("source_type") or "")
        source = str(item.get("source") or "")
        doc_id = str(item.get("doc_id") or "")
        key = (source_type, source, doc_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source_type": source_type,
                "source": source,
                "doc_id": doc_id,
                "url": str(item.get("url") or ""),
                "path": str(item.get("path") or ""),
            }
        )
    return citations


def _provenance_item(
    source_type: str,
    source: str,
    snippet: str,
    doc_id: str | None = None,
    url: str = "",
    path: str = "",
) -> dict[str, object]:
    return {
        "source_type": source_type,
        "source": source,
        "snippet": snippet,
        "doc_id": doc_id or "",
        "url": url,
        "path": path,
    }


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def _template_code_for_request(request: str, path: str) -> str:
    suffix = Path(path).suffix.lower()
    lowered = request.lower()

    fn_match = re.search(
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(?:that\s+)?(?:returns?|->)\s+([^.\n]+)",
        request,
        re.IGNORECASE,
    )
    if suffix == ".py" and fn_match:
        name = fn_match.group(1).strip()
        args = ", ".join(a.strip() for a in fn_match.group(2).split(",") if a.strip()) or "*args"
        body = fn_match.group(3).strip()
        body_expr = _normalize_python_expr(body)
        return (
            f"def {name}({args}):\n"
            f'    """Auto-generated from request."""\n'
            f"    return {body_expr}\n"
        )

    if suffix == ".py" and "factorial" in lowered:
        return (
            "def fact(n: int) -> int:\n"
            '    """Iterative factorial."""\n'
            "    if n < 0:\n"
            "        raise ValueError('n must be >= 0')\n"
            "    out = 1\n"
            "    for i in range(2, n + 1):\n"
            "        out *= i\n"
            "    return out\n"
        )

    if suffix == ".py" and "fibonacci" in lowered:
        return (
            "def fib(n: int) -> int:\n"
            '    """Iterative fibonacci where fib(0)=0, fib(1)=1."""\n'
            "    if n < 0:\n"
            "        raise ValueError('n must be >= 0')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )

    if suffix == ".py" and "palindrome" in lowered:
        return (
            "def is_pal(s: str) -> bool:\n"
            '    """Return True when s is a palindrome."""\n'
            "    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return t == t[::-1]\n"
        )

    if suffix == ".py":
        return (
            'def main() -> None:\n'
            '    """Auto-generated local template."""\n'
            f"    print({request!r})\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
    if suffix in {".js", ".mjs", ".cjs"}:
        return (
            "// Auto-generated local template\n"
            "function main() {\n"
            f"  console.log({request!r});\n"
            "}\n\n"
            "main();\n"
        )
    if suffix == ".ts":
        return (
            "// Auto-generated local template\n"
            "function main(): void {\n"
            f"  console.log({request!r});\n"
            "}\n\n"
            "main();\n"
        )
    return (
        "# Auto-generated local code snippet\n"
        f"# Request: {request}\n"
        "def main():\n"
        "    pass\n"
    )


def _normalize_python_expr(expr: str) -> str:
    cleaned = expr.strip()
    cleaned = re.sub(r"\btrue\b", "True", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfalse\b", "False", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9_+\-*/%()., <>=!:\[\]'\"|&]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "None"
    return cleaned


def _render_command_result(result: object) -> str:
    command = getattr(result, "command", [])
    cwd = getattr(result, "cwd", "")
    ok = bool(getattr(result, "ok", False))
    exit_code = int(getattr(result, "exit_code", 1))
    duration_ms = int(getattr(result, "duration_ms", 0))
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    truncated = bool(getattr(result, "truncated", False))

    lines = [
        f"Command: {' '.join(command)}",
        f"CWD: {cwd}",
        f"Exit: {exit_code}",
        f"Duration ms: {duration_ms}",
        f"Status: {'ok' if ok else 'error'}",
    ]
    if stdout:
        lines.append("")
        lines.append("STDOUT:")
        lines.append(stdout[:7000])
    if stderr:
        lines.append("")
        lines.append("STDERR:")
        lines.append(stderr[:7000])
    if truncated:
        lines.append("")
        lines.append("Output truncated for safety.")
    return "\n".join(lines)


def _safe_eval_expression(expr: str) -> float:
    allowed_binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed")
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")
            return float(allowed_binary[type(node.op)](left, right))
        raise ValueError("Unsupported math expression")

    value = _eval(tree)
    if abs(value) > 1e12:
        raise ValueError("Result too large")
    rounded = round(value, 10)
    if abs(rounded - int(rounded)) < 1e-10:
        return float(int(rounded))
    return rounded


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _tokenize_for_match(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9_\-]{2,}", text.lower()) if t not in {"the", "and", "for", "with"}]


def _render_evidence_answer(user_text: str, evidence: list[dict[str, object]], strict_facts: bool) -> str:
    if not evidence:
        if strict_facts:
            return (
                "Evidence Mode is ON and no source-backed snippets were collected. "
                "No grounded claim can be produced for this request."
            )
        return "Evidence Mode is ON but no evidence objects were generated."

    valid: list[dict[str, object]] = []
    for idx, item in enumerate(evidence, start=1):
        claim = _ensure_sentence(str(item.get("claim") or "").strip(), max_chars=220)
        sources = [str(s).strip() for s in (item.get("sources") or []) if str(s).strip()]
        snippets = [str(s).strip() for s in (item.get("snippets") or []) if str(s).strip()]
        confidence = item.get("confidence", 0)

        if not claim:
            continue
        if not sources or not snippets:
            continue
        valid.append(
            {
                "index": idx,
                "claim": claim,
                "source": sources[0],
                "snippet": snippets[0][:240],
                "confidence": confidence,
            }
        )

    if not valid and strict_facts:
        return (
            "Evidence Mode is ON but collected evidence objects were incomplete "
            "(missing source or snippet). No claim is returned."
        )
    if not valid:
        return "Evidence Mode is ON but usable source+snippet pairs were not collected."

    lines = [
        "Here's a grounded answer based on collected evidence:",
        "",
        " ".join(str(item["claim"]) for item in valid[:3]),
    ]

    if len(valid) > 3:
        lines.append("")
        lines.append("Additional grounded points:")
        for item in valid[3:6]:
            lines.append(f"- {item['claim']}")

    lines.append("")
    lines.append("Evidence:")
    for item in valid:
        lines.append(f"[{item['index']}] {item['claim']}")
        lines.append(f"    Source: {item['source']}")
        lines.append(f"    Snippet: {item['snippet']}")
        lines.append(f"    Confidence: {item['confidence']}")

    if strict_facts:
        lines.append("")
        lines.append("Strict Fact Mode: only source-backed claims are included.")

    lines.append("")
    lines.append(f"Request handled: {user_text[:160]}")
    return "\n".join(lines).strip()


def _ensure_sentence(text: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().strip("-")
    if not cleaned:
        return ""
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _point_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _collect_grounded_points(
    context_blocks: list[str],
    provenance: list[dict[str, object]],
    limit: int = 6,
) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in provenance:
        source = str(item.get("source") or item.get("path") or item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not snippet:
            continue
        claim = _ensure_sentence(_claim_from_snippet(snippet, max_chars=220), max_chars=220)
        key = _point_key(claim)
        if not claim or not key or key in seen:
            continue
        seen.add(key)
        points.append({"text": claim, "source": source})
        if len(points) >= limit:
            return points

    for block in context_blocks:
        source = ""
        url_match = URL_RE.search(block)
        if url_match:
            source = url_match.group(0).strip()
        for raw in block.splitlines():
            line = raw.strip().lstrip("-").strip()
            if not line:
                continue
            lower = line.lower()
            if URL_RE.match(line):
                continue
            if lower.startswith(
                (
                    "web search results",
                    "auto web grounding",
                    "relevant indexed documents",
                    "session memory",
                    "directory listing",
                    "file content",
                    "dictionary facts",
                )
            ):
                continue
            if line.startswith("[DIR]"):
                continue
            claim = _ensure_sentence(_claim_from_snippet(line, max_chars=220), max_chars=220)
            key = _point_key(claim)
            if not claim or not key or key in seen:
                continue
            seen.add(key)
            points.append({"text": claim, "source": source})
            if len(points) >= limit:
                return points

    return points


def _extract_sources_from_context(
    context_blocks: list[str],
    provenance: list[dict[str, object]] | None = None,
    limit: int = 8,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    if provenance:
        for item in _extract_citations(provenance):
            source = str(item.get("source") or item.get("url") or item.get("path") or "").strip()
            if not source or source in seen:
                continue
            seen.add(source)
            out.append(source)
            if len(out) >= limit:
                return out

    for block in context_blocks:
        for url in URL_RE.findall(block):
            u = url.strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(u)
            if len(out) >= limit:
                return out
    return out


def _claim_from_snippet(snippet: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", snippet).strip()
    if not text:
        return ""
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if not sentence:
        sentence = text
    if len(sentence) < 8:
        sentence = text[:max_chars]
    sentence = sentence.strip(" -")
    if not re.search(r"[A-Za-z]", sentence):
        sentence = f"Source excerpt: {text[:max_chars]}"
    return sentence[:max_chars]

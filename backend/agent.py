from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import OllamaClient
from .storage import MemoryFact, SQLiteStore
from .tools.filesystem import FileBrowser
from .tools.web import dictionary_define, download_url, fetch_url_text, search_web


URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", flags=re.IGNORECASE)


@dataclass
class ToolEvent:
    tool: str
    status: str
    detail: str


@dataclass
class AgentRunConfig:
    strict_facts: bool = False
    evidence_mode: bool = False
    allow_web: bool = True
    allow_files: bool = True
    allow_docs: bool = True
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

        actions = self._build_agent_actions(user_text, cfg)
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
            )

            for call in result.get("tool_calls", []):
                tool_calls.append(call)

            context_block = str(result.get("context_block") or "").strip()
            if context_block:
                context_blocks.append(context_block)

            for item in result.get("provenance", []):
                provenance.append(item)

            if str(result.get("status")) == "ok":
                plan[idx]["status"] = "done"
            else:
                plan[idx]["status"] = "error"
            plan[idx]["detail"] = str(result.get("detail") or "")

            if action.tool == "answer.compose" and str(result.get("status")) == "ok":
                answer = str(result.get("answer") or "")
                mode = str(result.get("mode") or mode)
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
            answer, mode = self._compose_answer(
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
            evidence = self._build_evidence(answer, provenance)
            answer = _attach_evidence_block(answer, evidence)

        citations = _extract_citations(provenance)

        self.store.add_message(sid, "assistant", answer)
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
            "done": True,
        }

    def _build_agent_actions(self, user_text: str, cfg: AgentRunConfig) -> list[AgentAction]:
        actions: list[AgentAction] = []
        low = user_text.lower()

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

        if cfg.allow_web:
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

        if cfg.strict_facts and cfg.allow_web and not actions:
            actions.append(
                AgentAction(
                    tool="web.search.auto",
                    title="Auto-ground with web search for strict fact mode",
                    args={"query": user_text},
                    retryable=True,
                )
            )

        if cfg.allow_docs:
            actions.append(
                AgentAction(
                    tool="docs.retrieve",
                    title="Retrieve relevant indexed documents",
                    args={"query": user_text, "limit": 6},
                )
            )

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
    ) -> dict[str, object]:
        attempts = 2 if action.retryable else 1
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
                trace.append(
                    {
                        "name": action.tool,
                        "args": action.args,
                        "attempt": attempt,
                        "status": "ok",
                        "result_summary": str(result.get("detail") or "ok"),
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

        if tool == "answer.compose":
            answer, mode = self._compose_answer(
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
            }

        raise ValueError(f"Unknown agent tool: {tool}")

    def _compose_answer(
        self,
        session_id: str,
        user_text: str,
        context_blocks: list[str],
        provenance: list[dict[str, object]],
        cfg: AgentRunConfig,
    ) -> tuple[str, str]:
        facts = self.store.memory_for_session(session_id)[:6]
        if facts:
            context_blocks = [*context_blocks, "Session memory:\n" + "\n".join(_format_fact(f) for f in facts)]

        if cfg.strict_facts and not context_blocks:
            return (
                "Strict Fact Mode is ON and no grounded source context is available yet. "
                "Provide a document/URL/file request or enable additional tools.",
                "rules-agent",
            )

        status = self.llm.status()
        if status.available:
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
                return answer, mode
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
                )

        return (
            _fallback_agent_answer(
                user_text=user_text,
                context_blocks=context_blocks,
                strict_facts=cfg.strict_facts,
                provenance=provenance,
                llm_error="local model unavailable",
            ),
            "rules-agent",
        )

    def _build_evidence(
        self,
        answer: str,
        provenance: list[dict[str, object]],
        max_claims: int = 4,
    ) -> list[dict[str, object]]:
        if not provenance:
            return [
                {
                    "claim": "Insufficient grounded sources were available for verified claims.",
                    "sources": [],
                    "snippets": [],
                    "confidence": 0.0,
                }
            ]

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) >= 20]
        claims = sentences[:max_claims] if sentences else [answer[:180].strip()]

        evidence: list[dict[str, object]] = []
        for claim in claims:
            claim_tokens = set(_tokenize_for_match(claim))
            scored: list[tuple[int, dict[str, object]]] = []
            for item in provenance:
                snippet = str(item.get("snippet") or "")
                if not snippet:
                    continue
                snippet_tokens = set(_tokenize_for_match(snippet))
                overlap = len(claim_tokens.intersection(snippet_tokens))
                if overlap > 0:
                    scored.append((overlap, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = [x[1] for x in scored[:2]]
            if not chosen:
                chosen = provenance[:1]

            sources = [str(c.get("source") or "") for c in chosen if str(c.get("source") or "")]
            snippets = [str(c.get("snippet") or "")[:220] for c in chosen if str(c.get("snippet") or "")]
            overlap_max = float(scored[0][0]) if scored else 0.0
            confidence = min(0.98, 0.25 + 0.2 * len(chosen) + 0.08 * overlap_max)

            evidence.append(
                {
                    "claim": claim,
                    "sources": sources,
                    "snippets": snippets,
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

    def _extract_memory(self, session_id: str, user_text: str) -> None:
        text = user_text.strip()

        name = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,48})", text, re.IGNORECASE)
        if name:
            self.store.upsert_memory(session_id, "name", name.group(1).strip())

        goal = re.search(r"\bmy goal is\s+([^.!?\n]{3,240})", text, re.IGNORECASE)
        if goal:
            self.store.upsert_memory(session_id, "goal", goal.group(1).strip())

        pref = re.search(r"\bi (?:prefer|like)\s+([^.!?\n]{3,240})", text, re.IGNORECASE)
        if pref:
            self.store.upsert_memory(session_id, "preference", pref.group(1).strip())

        need = re.search(r"\bi need\s+([^.!?\n]{3,240})", text, re.IGNORECASE)
        if need:
            self.store.upsert_memory(session_id, "need", need.group(1).strip())


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
    lines = [
        "Running in local rules mode (no local model available).",
        "Install/start Ollama and a model for stronger free-form chat quality.",
    ]

    if strict_facts:
        lines.append("Strict facts mode is ON: unsupported claims are blocked.")

    if events:
        lines.append("")
        lines.append("Tool activity:")
        for event in events:
            lines.append(f"- [{event.status}] {event.tool}: {event.detail}")

    if context_blocks:
        lines.append("")
        lines.append("Useful context:")
        for block in context_blocks[:4]:
            summary = block[:420].strip()
            lines.append(f"- {summary}")
    else:
        lines.append("")
        if strict_facts:
            lines.append("No source context was found yet. Ask me to search web/files or provide a URL/document.")
        else:
            lines.append(
                "I can already do: dictionary definitions, web search, website fetch, URL downloads, file listing/reading/search, "
                "document upload/index, and persistent chat memory."
            )

    lines.append("")
    lines.append(f"Request handled: {user_text[:160]}")
    return "\n".join(lines)


def _fallback_agent_answer(
    user_text: str,
    context_blocks: list[str],
    strict_facts: bool,
    provenance: list[dict[str, object]],
    llm_error: str,
) -> str:
    lines = [
        "Agent completed with local rules fallback.",
        f"Model note: {llm_error}",
    ]

    if strict_facts:
        lines.append("Strict Fact Mode is ON. Unsupported claims are avoided.")

    if context_blocks:
        lines.append("")
        lines.append("Grounded context summary:")
        for block in context_blocks[:4]:
            lines.append(f"- {block[:360]}")
    else:
        lines.append("")
        lines.append("No grounded context was collected.")

    if provenance:
        lines.append("")
        lines.append("Provenance sources:")
        for item in _extract_citations(provenance)[:8]:
            src = str(item.get("source") or "")
            if src:
                lines.append(f"- {src}")

    lines.append("")
    lines.append(f"Request handled: {user_text[:160]}")
    return "\n".join(lines)


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


def _tokenize_for_match(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9_\-]{2,}", text.lower()) if t not in {"the", "and", "for", "with"}]


def _attach_evidence_block(answer: str, evidence: list[dict[str, object]]) -> str:
    lines = [answer.strip(), "", "Evidence Objects:"]
    for idx, item in enumerate(evidence, start=1):
        lines.append(f"{idx}. claim: {item.get('claim', '')}")
        lines.append(f"   confidence: {item.get('confidence', 0)}")
        sources = item.get("sources") or []
        snippets = item.get("snippets") or []
        if sources:
            lines.append("   sources: " + "; ".join(str(s) for s in sources))
        if snippets:
            lines.append("   snippets: " + " | ".join(str(s) for s in snippets))
    return "\n".join(lines).strip()

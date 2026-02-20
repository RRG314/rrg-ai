from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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

    def _run_file_tools(self, user_text: str, tool_events: list[ToolEvent], context_blocks: list[str]) -> None:
        low = user_text.lower().strip()

        list_match = re.search(r"\b(?:list|show)\s+(?:files|folders|directories)(?:\s+in\s+(.+))?", user_text, re.IGNORECASE)
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


def _system_prompt(files_root: Path, strict_facts: bool) -> str:
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .storage import SQLiteStore
from .tools.filesystem import FileBrowser
from .tools.web import fetch_url_text, search_web


@dataclass
class SkillRunResult:
    status: str
    detail: str
    context_block: str
    provenance: list[dict[str, object]]


def research_pipeline(
    query: str,
    store: SQLiteStore,
    max_results: int = 6,
    fetch_top: int = 2,
) -> SkillRunResult:
    query = query.strip()
    if not query:
        return SkillRunResult(status="error", detail="query is empty", context_block="", provenance=[])

    provenance: list[dict[str, object]] = []
    lines: list[str] = []

    try:
        results = search_web(query, max_results=max_results)
    except Exception as exc:
        results = []
        lines.append(f"Web search failed ({exc}); using local doc fallback for research pipeline.")

    if results:
        lines.append(f"Research pipeline web results for '{query}':")
        for item in results:
            title = str(item.get("title") or item.get("url") or "web result")
            url = str(item.get("url") or "")
            snippet = str(item.get("snippet") or "").strip()
            lines.append(f"- {title}\n  {url}\n  {snippet}")
            doc_id = store.add_document(name=title[:120], source=url, kind="skill-research-search", text=f"{title}\n{url}\n{snippet}")
            provenance.append(_prov("url", source=url, snippet=snippet[:360], doc_id=doc_id, url=url))

        fetched = 0
        for item in results[: max(1, fetch_top)]:
            url = str(item.get("url") or "")
            if not url:
                continue
            try:
                text, kind = fetch_url_text(url, max_chars=14000)
            except Exception:
                continue
            fetched += 1
            doc_id = store.add_document(name=url, source=url, kind=f"skill-research-fetch-{kind}", text=text)
            lines.append(f"Fetched deep content from {url} ({kind})")
            provenance.append(_prov("url", source=url, snippet=text[:360], doc_id=doc_id, url=url))

        return SkillRunResult(
            status="ok",
            detail=f"Research pipeline gathered {len(results)} results and fetched {fetched} pages",
            context_block="\n".join(lines),
            provenance=provenance,
        )

    hits = store.search_chunks(query, limit=6)
    if not hits:
        return SkillRunResult(
            status="error",
            detail="Research pipeline found no web or local evidence",
            context_block="\n".join(lines).strip(),
            provenance=[],
        )

    lines.append(f"Research pipeline local fallback for '{query}':")
    for hit in hits:
        source = str(hit.get("source") or hit.get("doc_name") or "local-doc")
        snippet = str(hit.get("text") or "")[:360]
        lines.append(f"- {source}: {snippet[:180]}")
        provenance.append(
            _prov(
                "doc",
                source=source,
                snippet=snippet,
                doc_id=str(hit.get("doc_id") or ""),
            )
        )

    return SkillRunResult(
        status="ok",
        detail=f"Research pipeline used local fallback with {len(hits)} document chunks",
        context_block="\n".join(lines),
        provenance=provenance,
    )


def doc_pipeline(query: str, store: SQLiteStore, limit: int = 8) -> SkillRunResult:
    query = query.strip()
    if not query:
        return SkillRunResult(status="error", detail="query is empty", context_block="", provenance=[])

    hits = store.search_chunks(query, limit=max(1, limit))
    if not hits:
        return SkillRunResult(status="ok", detail="Doc pipeline found no matching chunks", context_block="", provenance=[])

    lines = [f"Doc pipeline retrieval for '{query}':"]
    provenance: list[dict[str, object]] = []

    for hit in hits:
        src = str(hit.get("source") or hit.get("doc_name") or "document")
        score = float(hit.get("score") or 0.0)
        snippet = str(hit.get("text") or "")[:360]
        lines.append(f"- {hit['doc_name']} ({src}) [score={score:.3f}]: {snippet[:220]}")
        provenance.append(
            _prov(
                "doc",
                source=src,
                snippet=snippet,
                doc_id=str(hit.get("doc_id") or ""),
            )
        )

    report = "\n".join(lines)
    doc_id = store.add_document(name=f"doc-pipeline:{query[:48]}", source="doc-pipeline", kind="skill-doc-pipeline", text=report)
    provenance.append(_prov("doc", source="doc-pipeline", snippet=report[:360], doc_id=doc_id))

    return SkillRunResult(
        status="ok",
        detail=f"Doc pipeline returned {len(hits)} chunks",
        context_block=report,
        provenance=provenance,
    )


def folder_audit_pipeline(
    path: str,
    files: FileBrowser,
    store: SQLiteStore,
    max_entries: int = 600,
    max_depth: int = 3,
) -> SkillRunResult:
    target = files.resolve(path or ".")
    if not target.exists():
        return SkillRunResult(status="error", detail=f"Path not found: {target}", context_block="", provenance=[])
    if not target.is_dir():
        return SkillRunResult(status="error", detail=f"Not a directory: {target}", context_block="", provenance=[])

    file_count = 0
    dir_count = 0
    total_size = 0
    ext_counts: dict[str, int] = {}
    largest: list[tuple[int, Path]] = []

    stack: list[tuple[Path, int]] = [(target, 0)]
    visited = 0

    while stack and visited < max_entries:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except Exception:
            continue

        for entry in entries:
            if visited >= max_entries:
                break
            visited += 1
            if entry.is_symlink():
                continue
            if entry.is_dir():
                dir_count += 1
                if depth < max_depth:
                    stack.append((entry, depth + 1))
                continue

            if not entry.is_file():
                continue

            file_count += 1
            try:
                size = int(entry.stat().st_size)
            except Exception:
                size = 0
            total_size += size

            ext = entry.suffix.lower() or "<none>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

            largest.append((size, entry))

    largest.sort(key=lambda x: x[0], reverse=True)
    largest = largest[:10]

    top_ext = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:12]

    lines = [
        f"Folder audit for {target}",
        f"- scanned entries: {visited}",
        f"- directories: {dir_count}",
        f"- files: {file_count}",
        f"- total size bytes: {total_size}",
        "- top extensions:",
    ]
    for ext, count in top_ext:
        lines.append(f"  - {ext}: {count}")

    lines.append("- largest files:")
    for size, file_path in largest:
        lines.append(f"  - {file_path} ({size} bytes)")

    report = "\n".join(lines)
    doc_id = store.add_document(name=f"folder-audit:{target.name}", source=str(target), kind="skill-folder-audit", text=report)

    provenance: list[dict[str, object]] = [
        _prov("file", source=str(target), snippet=report[:360], doc_id=doc_id, path=str(target))
    ]
    for size, file_path in largest[:5]:
        provenance.append(
            _prov(
                "file",
                source=str(file_path),
                snippet=f"largest-file size={size} path={file_path}",
                path=str(file_path),
            )
        )

    return SkillRunResult(
        status="ok",
        detail=f"Folder audit scanned {visited} entries and found {file_count} files",
        context_block=report,
        provenance=provenance,
    )


def _prov(
    source_type: str,
    source: str,
    snippet: str,
    doc_id: str = "",
    url: str = "",
    path: str = "",
) -> dict[str, object]:
    return {
        "source_type": source_type,
        "source": source,
        "snippet": snippet,
        "doc_id": doc_id,
        "url": url,
        "path": path,
    }

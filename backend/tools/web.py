from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


USER_AGENT = "RRG-AI-Local/1.0 (+https://github.com/RRG314/ai)"
HEADERS = {"User-Agent": USER_AGENT}


def search_web(query: str, max_results: int = 6) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={q}"
    response = requests.get(url, timeout=25, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, str]] = []

    for result in soup.select(".result"):
        link = result.select_one("a.result__a")
        snippet = result.select_one(".result__snippet")
        if not link:
            continue
        href = link.get("href", "").strip()
        title = link.get_text(" ", strip=True)
        text = snippet.get_text(" ", strip=True) if snippet else ""

        if not href:
            continue

        rows.append({"title": title, "url": href, "snippet": text})
        if len(rows) >= max_results:
            break

    return rows


def fetch_url_text(url: str, max_chars: int = 22000) -> tuple[str, str]:
    response = requests.get(url, timeout=35, headers=HEADERS)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    raw = response.content

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = _extract_pdf_bytes(raw)
        return text[:max_chars], "pdf"

    if "text/html" in content_type or _looks_like_html(response.text):
        text = _extract_html_text(response.text)
        return text[:max_chars], "html"

    decoded = response.text
    return decoded[:max_chars], "text"


def download_url(url: str, output_dir: Path, max_bytes: int = 40 * 1024 * 1024) -> dict[str, str | int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=50, headers=HEADERS)
    response.raise_for_status()

    name = _filename_from_response(url, response.headers.get("Content-Disposition"))
    path = output_dir / name

    total = 0
    with path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Download too large: {total} bytes exceeds {max_bytes}")
            f.write(chunk)

    return {
        "path": str(path),
        "filename": name,
        "size_bytes": total,
        "content_type": response.headers.get("Content-Type", ""),
    }


def _extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for bad in soup(["script", "style", "noscript", "svg", "canvas"]):
        bad.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_bytes(data: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(data))
    texts = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(texts).strip()


def _looks_like_html(text: str) -> bool:
    prefix = text[:500].lower()
    return "<html" in prefix or "<!doctype html" in prefix


def _filename_from_response(url: str, content_disposition: str | None) -> str:
    if content_disposition:
        m = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", content_disposition, flags=re.IGNORECASE)
        if m:
            candidate = urllib.parse.unquote(m.group(1)).strip().strip('"')
            if candidate:
                return _sanitize_filename(candidate)

    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name or "download.bin"
    return _sanitize_filename(name)


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "download.bin"

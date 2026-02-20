from __future__ import annotations

import re
import socket
import urllib.parse
import ipaddress
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


USER_AGENT = "RRG-AI-Local/1.0 (+https://github.com/RRG314/ai)"
HEADERS = {"User-Agent": USER_AGENT}
BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "ip6-localhost", "metadata.google.internal"}
BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".home", ".lan")


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
    _assert_safe_remote_url(url)
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
    _assert_safe_remote_url(url)
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


def dictionary_define(word: str, max_definitions: int = 6) -> dict[str, object]:
    clean = _normalize_word(word)
    if not clean:
        raise ValueError("Word is empty after normalization")

    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(clean)}"
    response = requests.get(url, timeout=20, headers=HEADERS)
    response.raise_for_status()

    payload = response.json()
    return _parse_dictionary_payload(clean, payload, max_definitions=max_definitions)


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


def _parse_dictionary_payload(
    word: str,
    payload: object,
    max_definitions: int = 6,
) -> dict[str, object]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"No dictionary entries found for '{word}'")

    first = payload[0]
    if not isinstance(first, dict):
        raise ValueError(f"Invalid dictionary response for '{word}'")

    normalized = str(first.get("word") or word)
    phonetic = str(first.get("phonetic") or "")

    definitions: list[dict[str, str]] = []
    meanings = first.get("meanings", [])
    if isinstance(meanings, list):
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            part = str(meaning.get("partOfSpeech") or "")
            defs = meaning.get("definitions", [])
            if not isinstance(defs, list):
                continue
            for item in defs:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("definition") or "").strip()
                if not text:
                    continue
                example = str(item.get("example") or "").strip()
                definitions.append(
                    {
                        "part_of_speech": part,
                        "definition": text,
                        "example": example,
                    }
                )
                if len(definitions) >= max_definitions:
                    break
            if len(definitions) >= max_definitions:
                break

    if not definitions:
        raise ValueError(f"No dictionary definitions found for '{word}'")

    return {
        "word": normalized,
        "phonetic": phonetic,
        "definitions": definitions,
        "source": "dictionaryapi.dev",
    }


def _normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-z'-]", "", word).strip().lower()


def _assert_safe_remote_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")

    host = (parsed.hostname or "").strip().lower().strip(".")
    if not host:
        raise ValueError("URL host is missing")
    if host in BLOCKED_HOSTS or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise ValueError(f"Blocked private host target: {host}")
    if _looks_like_ip(host) and _ip_is_non_public(host):
        raise ValueError(f"Blocked non-public IP target: {host}")

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return
    except Exception:
        return

    for item in resolved:
        sockaddr = item[4]
        if not sockaddr:
            continue
        ip = str(sockaddr[0]).split("%", 1)[0]
        if _ip_is_non_public(ip):
            raise ValueError(f"Blocked non-public resolved address for {host}: {ip}")


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


def _ip_is_non_public(value: str) -> bool:
    ip = ipaddress.ip_address(value.split("%", 1)[0])
    return not ip.is_global

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

from .vision import IMAGE_EXTENSIONS, extract_text_from_image_bytes

try:
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".html",
    ".htm",
}


def extract_text_from_bytes(filename: str, data: bytes) -> tuple[str, str]:
    ext = Path(filename).suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return _decode_text(data), "text"

    if ext in IMAGE_EXTENSIONS:
        result = extract_text_from_image_bytes(filename, data)
        return result.text, "image-ocr"

    if ext == ".pdf":
        return _extract_pdf(data), "pdf"

    if ext == ".docx":
        if docx is None:
            raise ValueError("python-docx is required for .docx support")
        return _extract_docx(data), "docx"

    raise ValueError(f"Unsupported file extension: {ext or '<none>'}")


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")

    return text.strip()


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def _extract_docx(data: bytes) -> str:
    assert docx is not None
    document = docx.Document(io.BytesIO(data))
    lines = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines).strip()

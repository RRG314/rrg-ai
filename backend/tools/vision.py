from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm import OllamaClient

try:
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

    class UnidentifiedImageError(Exception):
        pass

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


@dataclass
class OCRResult:
    text: str
    width: int
    height: int
    image_format: str


def supports_image_ocr() -> tuple[bool, str]:
    if Image is None:
        return False, "Pillow is required for image OCR (pip install pillow)"
    if pytesseract is None:
        return False, "pytesseract is required for image OCR (pip install pytesseract)"
    if shutil.which("tesseract") is None:
        return False, "tesseract binary not found (install Tesseract OCR and ensure it is on PATH)"
    return True, ""


def extract_text_from_image_bytes(filename: str, data: bytes, max_chars: int = 40000) -> OCRResult:
    ok, reason = supports_image_ocr()
    if not ok:
        raise ValueError(reason)

    assert Image is not None
    assert pytesseract is not None

    try:
        with Image.open(io.BytesIO(data)) as img:
            normalized = img.convert("RGB")
            width, height = normalized.size
            image_format = (img.format or Path(filename).suffix.lstrip(".") or "image").lower()
            text = pytesseract.image_to_string(normalized) or ""
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupted image file: {filename}") from exc

    cleaned = text.strip()[:max_chars]
    return OCRResult(text=cleaned, width=width, height=height, image_format=image_format)


def analyze_image_bytes(
    filename: str,
    data: bytes,
    prompt: str = "",
    llm: OllamaClient | None = None,
) -> dict[str, object]:
    ocr = extract_text_from_image_bytes(filename, data)
    prompt = prompt.strip()
    answer = ""

    if prompt:
        if llm is None:
            answer = "No local model attached for prompt-based analysis."
        else:
            status = llm.status()
            if not status.available:
                answer = f"Local model unavailable: {status.reason or 'unknown reason'}"
            else:
                packed = (
                    "You are analyzing an uploaded local image for a user.\n"
                    f"Image file: {filename}\n"
                    f"Image metadata: format={ocr.image_format}, width={ocr.width}, height={ocr.height}\n"
                    "OCR text extracted from the image:\n"
                    f"{ocr.text[:12000] or '[no OCR text extracted]'}\n\n"
                    f"User request: {prompt}\n"
                    "Provide a practical response using only this information."
                )
                try:
                    answer = llm.chat(
                        messages=[{"role": "user", "content": packed}],
                        system=(
                            "You are a local vision-and-text analysis assistant. "
                            "Do not invent visual details not present in OCR/metadata."
                        ),
                    )
                except Exception as exc:
                    answer = f"Prompt analysis failed: {exc}"

    return {
        "filename": filename,
        "ocr_text": ocr.text,
        "width": ocr.width,
        "height": ocr.height,
        "image_format": ocr.image_format,
        "prompt": prompt,
        "answer": answer,
    }

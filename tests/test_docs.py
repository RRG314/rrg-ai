import pytest

from backend.tools.docs import extract_text_from_bytes


def test_extract_plain_text() -> None:
    text, kind = extract_text_from_bytes("notes.txt", b"hello local ai")
    assert kind == "text"
    assert "hello" in text


def test_unsupported_extension() -> None:
    with pytest.raises(ValueError):
        extract_text_from_bytes("archive.zip", b"dummy")


def test_extract_image_text_via_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stub:
        text = "image ocr output"

    monkeypatch.setattr("backend.tools.docs.extract_text_from_image_bytes", lambda filename, data: Stub())
    text, kind = extract_text_from_bytes("scan.png", b"\x89PNG")
    assert kind == "image-ocr"
    assert "ocr" in text

import pytest

from backend.tools.docs import extract_text_from_bytes


def test_extract_plain_text() -> None:
    text, kind = extract_text_from_bytes("notes.txt", b"hello local ai")
    assert kind == "text"
    assert "hello" in text


def test_unsupported_extension() -> None:
    with pytest.raises(ValueError):
        extract_text_from_bytes("archive.zip", b"dummy")

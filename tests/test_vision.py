from backend.tools import vision


class _StubOCR:
    text = "alpha beta"
    width = 320
    height = 200
    image_format = "png"


class _StubStatus:
    available = True
    reason = ""


class _StubLLM:
    def status(self) -> _StubStatus:
        return _StubStatus()

    def chat(self, messages, system):  # type: ignore[no-untyped-def]
        assert messages
        assert "OCR text extracted" in messages[0]["content"]
        return "stub-analysis"


def test_analyze_image_bytes_without_prompt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(vision, "extract_text_from_image_bytes", lambda filename, data: _StubOCR())
    out = vision.analyze_image_bytes("x.png", b"abc")
    assert out["ocr_text"] == "alpha beta"
    assert out["answer"] == ""


def test_analyze_image_bytes_with_prompt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(vision, "extract_text_from_image_bytes", lambda filename, data: _StubOCR())
    out = vision.analyze_image_bytes("x.png", b"abc", prompt="summarize", llm=_StubLLM())
    assert out["answer"] == "stub-analysis"

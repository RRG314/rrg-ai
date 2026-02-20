import pytest

from backend.tools.web import download_url, fetch_url_text


def test_fetch_rejects_loopback_target() -> None:
    with pytest.raises(ValueError):
        fetch_url_text("http://127.0.0.1:8000/private")


def test_download_rejects_localhost_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        download_url("http://localhost:8000/file.txt", tmp_path)


def test_fetch_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError):
        fetch_url_text("ftp://example.com/file.txt")

from fastapi.testclient import TestClient

from backend.app import API_TOKEN, REQUIRE_TOKEN, app


client = TestClient(app)


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if REQUIRE_TOKEN:
        headers["X-AI-Token"] = API_TOKEN
    return headers


def test_plugins_list_endpoint() -> None:
    res = client.get("/api/plugins", headers=_headers())
    assert res.status_code == 200
    payload = res.json()
    assert isinstance(payload.get("count"), int)
    assert isinstance(payload.get("plugins"), list)


def test_plugins_run_endpoint_text_tools() -> None:
    res = client.post(
        "/api/plugins/run",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "plugin_id": "text_tools",
            "input": {"text": "recursive adic recursive depth"},
            "timeout_sec": 60,
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload.get("ok") is True
    assert payload.get("plugin_id") == "text_tools"
    assert isinstance(payload.get("summary"), str)
    assert isinstance(payload.get("text"), str)


def test_plugins_run_endpoint_rdt_lm_tools_generate() -> None:
    res = client.post(
        "/api/plugins/run",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "plugin_id": "rdt_lm_tools",
            "input": {"mode": "generate", "seed": "the man walked", "max_length": 8},
            "timeout_sec": 60,
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload.get("ok") is True
    assert payload.get("plugin_id") == "rdt_lm_tools"
    assert "output:" in str(payload.get("text") or "")

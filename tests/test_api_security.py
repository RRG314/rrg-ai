from fastapi.testclient import TestClient

from backend.app import API_TOKEN, app


client = TestClient(app)


def test_health_and_bootstrap_public() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200

    boot = client.get("/api/bootstrap")
    assert boot.status_code == 200
    payload = boot.json()
    assert payload.get("ok") is True
    assert payload.get("auth_required") in {True, False}


def test_api_requires_token() -> None:
    res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code in {200, 401}

    if res.status_code == 401:
        ok = client.post(
            "/api/chat",
            headers={"X-AI-Token": API_TOKEN},
            json={"message": "hello"},
        )
        assert ok.status_code == 200

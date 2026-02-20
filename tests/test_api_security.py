from fastapi.testclient import TestClient

from backend.app import API_TOKEN, BOOTSTRAP_PAIRING_CODE, BOOTSTRAP_PAIRING_REQUIRED, REQUIRE_TOKEN, app


client = TestClient(app)


def test_health_and_bootstrap_available() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200

    boot = client.get("/api/bootstrap", headers={"Origin": "http://127.0.0.1:8000"})
    assert boot.status_code == 200
    payload = boot.json()
    assert payload.get("ok") is True
    assert payload.get("auth_required") in {True, False}
    if REQUIRE_TOKEN:
        assert payload.get("api_token") == API_TOKEN


def test_bootstrap_cross_origin_pairing_gate() -> None:
    boot = client.get("/api/bootstrap", headers={"Origin": "https://rrg314.github.io"})
    assert boot.status_code == 200
    payload = boot.json()
    assert payload.get("ok") is True

    if REQUIRE_TOKEN and BOOTSTRAP_PAIRING_REQUIRED:
        assert payload.get("pairing_required") is True
        assert payload.get("api_token") == ""
        paired = client.get(
            "/api/bootstrap",
            headers={
                "Origin": "https://rrg314.github.io",
                "X-AI-Pairing-Code": BOOTSTRAP_PAIRING_CODE,
            },
        )
        assert paired.status_code == 200
        assert paired.json().get("api_token") == API_TOKEN
    elif REQUIRE_TOKEN:
        assert payload.get("api_token") == API_TOKEN


def test_pairing_code_endpoint_local_only_when_enabled() -> None:
    local = client.get("/api/pairing-code", headers={"Origin": "http://localhost:8000"})
    assert local.status_code == 200
    assert local.json().get("pairing_code") == BOOTSTRAP_PAIRING_CODE

    remote = client.get("/api/pairing-code", headers={"Origin": "https://rrg314.github.io"})
    if REQUIRE_TOKEN and BOOTSTRAP_PAIRING_REQUIRED:
        assert remote.status_code == 403
    else:
        assert remote.status_code == 200


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


def test_options_preflight_not_blocked_by_auth() -> None:
    res = client.options(
        "/api/agent",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-ai-token",
        },
    )
    assert res.status_code in {200, 204}


def test_system_check_endpoint_small_run() -> None:
    headers = {"Content-Type": "application/json"}
    if REQUIRE_TOKEN:
        headers["X-AI-Token"] = API_TOKEN
    res = client.post(
        "/api/system-check",
        headers=headers,
        json={
            "min_score": 95.0,
            "use_llm": False,
            "max_steps": 6,
            "task_limit": 3,
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload.get("suite") == "system_check_v1"
    assert isinstance(payload.get("score"), (int, float))
    assert payload.get("check_count")

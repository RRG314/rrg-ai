from __future__ import annotations

import json
from dataclasses import dataclass

import requests


@dataclass
class LLMStatus:
    available: bool
    model: str
    reason: str = ""


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: int = 90) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> LLMStatus:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
            names = {item.get("name", "") for item in payload.get("models", [])}
            if self.model in names:
                return LLMStatus(available=True, model=self.model)
            return LLMStatus(
                available=False,
                model=self.model,
                reason=f"Model '{self.model}' not installed in Ollama",
            )
        except Exception as exc:
            return LLMStatus(available=False, model=self.model, reason=str(exc))

    def chat(self, messages: list[dict[str, str]], system: str) -> str:
        merged = [{"role": "system", "content": system}, *messages]
        payload = {
            "model": self.model,
            "messages": merged,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

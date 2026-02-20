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
        self._resolved_model = model

    def status(self) -> LLMStatus:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
            names = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
            chosen = self._choose_model(names)
            if chosen:
                self._resolved_model = chosen
                reason = ""
                if chosen != self.model:
                    reason = f"Requested model '{self.model}' not found; using '{chosen}'"
                return LLMStatus(available=True, model=chosen, reason=reason)
            return LLMStatus(
                available=False,
                model=self.model,
                reason=f"No local models installed in Ollama",
            )
        except Exception as exc:
            return LLMStatus(available=False, model=self.model, reason=str(exc))

    def chat(self, messages: list[dict[str, str]], system: str) -> str:
        status = self.status()
        if not status.available:
            raise RuntimeError(status.reason or "Ollama model unavailable")

        merged = [{"role": "system", "content": system}, *messages]
        payload = {
            "model": self._resolved_model,
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

    def _choose_model(self, names: list[str]) -> str:
        if not names:
            return ""

        if self.model in names:
            return self.model

        requested_base = self.model.split(":", 1)[0]
        for name in names:
            if name.split(":", 1)[0] == requested_base:
                return name

        return names[0]

"""Provider boundary for structured language-model operations."""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx


class AgentModel(Protocol):
    """Minimal provider-agnostic interface consumed by agent nodes."""

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return one parsed JSON object or raise a normalized exception."""


class OpenAICompatibleModel:
    """Call an OpenAI-compatible chat-completions endpoint for JSON output."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Configure credentials and transport without exposing either to graph state."""

        if not api_key.strip():
            raise ValueError("model API key is required")
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Request JSON mode and reject non-object responses."""

        response = self._client.post(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("model returned malformed JSON content") from exc
        if not isinstance(parsed, dict):
            raise ValueError("model JSON response must be an object")
        return parsed


class HeuristicModel:
    """Deterministic offline model for local demos and reproducible CI."""

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return conservative outputs based on the requested operation marker."""

        if "OPERATION=PLAN" in system:
            question = user.split("QUESTION:", 1)[-1].strip()
            parts = [part.strip(" ?.\n") for part in question.split(" and ") if part.strip()]
            tasks = [
                {
                    "id": f"task_{index}",
                    "question": part,
                    "tool": "web_search",
                    "arguments": {"query": part, "max_results": 5},
                }
                for index, part in enumerate(parts[:4], start=1)
            ]
            return {
                "tasks": tasks
                or [
                    {
                        "id": "task_1",
                        "question": question,
                        "tool": "web_search",
                        "arguments": {"query": question, "max_results": 5},
                    }
                ]
            }
        if "OPERATION=CRITIQUE" in system:
            payload = json.loads(user)
            completed = set(payload["completed_task_ids"])
            missing = [task_id for task_id in payload["task_ids"] if task_id not in completed]
            return {
                "sufficient": not missing and payload["evidence_count"] > 0,
                "gaps": missing,
                "reasoning": "All planned tasks require at least one normalized evidence record.",
            }
        raise ValueError("heuristic model cannot synthesize free-form prose")

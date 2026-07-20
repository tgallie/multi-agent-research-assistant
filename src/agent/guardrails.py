"""Cross-cutting budget and model-output guardrails."""

from __future__ import annotations

from typing import Any

from agent.model import AgentModel
from agent.schemas import BudgetLimits, BudgetUsage


class BudgetExceededError(RuntimeError):
    """Raised before an operation that would cross a hard run limit."""


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without adding a tokenizer dependency."""

    return max(1, (len(text) + 2) // 3)


def complete_json_with_budget(
    model: AgentModel,
    *,
    system: str,
    user: str,
    usage: BudgetUsage,
    limits: BudgetLimits,
) -> dict[str, Any]:
    """Account for a model request before dispatch and enforce the token ceiling."""

    estimated = estimate_tokens(system) + estimate_tokens(user)
    if usage.estimated_tokens + estimated > limits.max_estimated_tokens:
        usage.exceeded = True
        usage.exceeded_reason = "estimated_token_limit"
        raise BudgetExceededError("estimated token budget exhausted")
    usage.estimated_tokens += estimated
    payload = model.complete_json(system, user)
    usage.estimated_tokens += estimate_tokens(str(payload))
    if usage.estimated_tokens > limits.max_estimated_tokens:
        usage.exceeded = True
        usage.exceeded_reason = "estimated_token_limit"
    return payload

"""Critic node: assess evidence coverage without writing the answer."""

from __future__ import annotations

import json

from agent.guardrails import BudgetExceededError, complete_json_with_budget
from agent.model import AgentModel
from agent.schemas import Critique, TraceEvent
from agent.state import AgentState


class CriticNode:
    """Check planned-task coverage and report concrete research gaps."""

    def __init__(self, model: AgentModel) -> None:
        """Inject the model used for evidence critique."""

        self._model = model

    def __call__(self, state: AgentState) -> dict[str, object]:
        """Return a schema-validated sufficiency decision."""

        context = {
            "task_ids": [task.id for task in state["plan"]],
            "completed_task_ids": [item.task_id for item in state["evidence"]],
            "evidence_count": len(state["evidence"]),
        }
        try:
            payload = complete_json_with_budget(
                self._model,
                system=(
                    "OPERATION=CRITIQUE. Check coverage only. Return sufficient, gaps, reasoning."
                ),
                user=json.dumps(context),
                usage=state["usage"],
                limits=state["limits"],
            )
            critique = Critique.model_validate(payload)
            fallback = False
        except (ValueError, RuntimeError, BudgetExceededError):
            completed = set(context["completed_task_ids"])
            gaps = [task_id for task_id in context["task_ids"] if task_id not in completed]
            critique = Critique(
                sufficient=not gaps and bool(state["evidence"]),
                gaps=gaps,
                reasoning=(
                    "Deterministic coverage check used because critic output was unavailable."
                ),
            )
            fallback = True
        return {
            "critique": critique,
            "trace": state["trace"]
            + [
                TraceEvent(
                    node="critic",
                    event="evidence_assessed",
                    detail={
                        "sufficient": critique.sufficient,
                        "gaps": critique.gaps,
                        "fallback": fallback,
                    },
                )
            ],
        }

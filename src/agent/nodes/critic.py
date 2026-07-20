"""Critic node: assess evidence coverage without writing the answer."""

from __future__ import annotations

import json

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

        payload = self._model.complete_json(
            "OPERATION=CRITIQUE. Check coverage only. Return sufficient, gaps, reasoning.",
            json.dumps(
                {
                    "task_ids": [task.id for task in state["plan"]],
                    "completed_task_ids": [item.task_id for item in state["evidence"]],
                    "evidence_count": len(state["evidence"]),
                }
            ),
        )
        critique = Critique.model_validate(payload)
        return {
            "critique": critique,
            "trace": state["trace"]
            + [
                TraceEvent(
                    node="critic",
                    event="evidence_assessed",
                    detail={"sufficient": critique.sufficient, "gaps": critique.gaps},
                )
            ],
        }

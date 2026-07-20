"""Planner node: decompose a question without invoking tools."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.guardrails import BudgetExceededError, complete_json_with_budget
from agent.model import AgentModel
from agent.schemas import PlanTask, TraceEvent
from agent.state import AgentState

MAX_PLAN_TASKS = 5


class PlanPayload(BaseModel):
    """Validated private planner response."""

    tasks: list[PlanTask] = Field(min_length=1, max_length=MAX_PLAN_TASKS)


class PlannerNode:
    """Create a bounded tool plan from the user question."""

    def __init__(self, model: AgentModel) -> None:
        """Inject the model used for structured planning."""

        self._model = model

    def __call__(self, state: AgentState) -> dict[str, object]:
        """Return a validated plan and an auditable trace event."""

        try:
            payload = complete_json_with_budget(
                self._model,
                system=(
                    "OPERATION=PLAN. Return JSON with 1-5 tasks. Each task has id, question, "
                    "tool (web_search, python, or scratchpad), and arguments. Never answer."
                ),
                user=f"QUESTION: {state['question']}",
                usage=state["usage"],
                limits=state["limits"],
            )
            plan = PlanPayload.model_validate(payload).tasks
            fallback = False
        except (ValueError, RuntimeError, BudgetExceededError):
            fallback_question = state["question"][:500]
            plan = [
                PlanTask(
                    id="task_fallback",
                    question=fallback_question,
                    tool="web_search",
                    arguments={"query": fallback_question, "max_results": 5},
                )
            ]
            fallback = True
        return {
            "plan": plan,
            "next_task_index": 0,
            "trace": state["trace"]
            + [
                TraceEvent(
                    node="planner",
                    event="plan_created",
                    detail={"tasks": len(plan), "fallback": fallback},
                )
            ],
        }

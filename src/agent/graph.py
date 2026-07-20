"""Explicit LangGraph topology and deterministic transition policy."""

from __future__ import annotations

import time
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agent.model import AgentModel
from agent.nodes import CriticNode, PlannerNode, ResearcherNode, SynthesizerNode
from agent.schemas import BudgetLimits, BudgetUsage, RunResult, TraceEvent
from agent.state import AgentState
from agent.telemetry import JsonlTelemetry
from agent.tools import ToolExecutor


def route_after_research(state: AgentState) -> str:
    """Continue task execution or move to evidence critique."""

    if state["usage"].exceeded or state["next_task_index"] >= len(state["plan"]):
        return "critic"
    return "researcher"


def route_after_critique(state: AgentState) -> str:
    """Re-search explicit gaps when possible, otherwise synthesize."""

    critique = state["critique"]
    if (
        critique
        and not critique.sufficient
        and not state["usage"].exceeded
        and state["next_task_index"] < len(state["plan"])
    ):
        return "researcher"
    return "synthesizer"


class ResearchGraph:
    """Compiled, dependency-injected research state machine."""

    def __init__(
        self,
        *,
        model: AgentModel,
        executor: ToolExecutor,
        telemetry: JsonlTelemetry,
        limits: BudgetLimits | None = None,
        cost_per_million_tokens: float = 0.0,
    ) -> None:
        """Build the named graph once for repeated isolated runs."""

        self._telemetry = telemetry
        self._limits = limits or BudgetLimits()
        self._cost_per_million_tokens = cost_per_million_tokens
        self._executor = executor
        builder = StateGraph(AgentState)
        builder.add_node("planner", PlannerNode(model))
        builder.add_node("researcher", ResearcherNode(executor))
        builder.add_node("critic", CriticNode(model))
        builder.add_node("synthesizer", SynthesizerNode(model))
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "researcher")
        builder.add_conditional_edges(
            "researcher", route_after_research, {"researcher": "researcher", "critic": "critic"}
        )
        builder.add_conditional_edges(
            "critic",
            route_after_critique,
            {"researcher": "researcher", "synthesizer": "synthesizer"},
        )
        builder.add_edge("synthesizer", END)
        self._graph = builder.compile()

    def run(self, question: str) -> RunResult:
        """Execute one isolated run and emit its complete telemetry record."""

        normalized = question.strip()
        if len(normalized) < 3:
            raise ValueError("question must contain at least three characters")
        if len(normalized) > 2_000:
            raise ValueError("question must contain at most 2,000 characters")
        started = time.perf_counter()
        self._executor.reset_run()
        final = self._graph.invoke(
            AgentState(
                question=normalized,
                plan=[],
                next_task_index=0,
                evidence=[],
                sources=[],
                critique=None,
                output=None,
                usage=BudgetUsage(),
                limits=self._limits.model_copy(deep=True),
                trace=[TraceEvent(node="initialize", event="question_accepted")],
                validation_errors=[],
            )
        )
        if final["output"] is None:
            raise RuntimeError("graph terminated without a public output")
        status = (
            "degraded" if final["usage"].exceeded or final["validation_errors"] else "completed"
        )
        result = RunResult(
            run_id=str(uuid4()),
            status=status,
            output=final["output"],
            trace=final["trace"],
            usage=final["usage"],
            duration_ms=int((time.perf_counter() - started) * 1_000),
            estimated_cost_usd=(
                final["usage"].estimated_tokens * self._cost_per_million_tokens / 1_000_000
            ),
        )
        self._telemetry.emit(result)
        return result

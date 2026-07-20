"""Typed shared state for the explicit research graph."""

from __future__ import annotations

from typing import TypedDict

from agent.schemas import (
    BudgetLimits,
    BudgetUsage,
    Critique,
    EvidenceItem,
    PlanTask,
    ResearchAnswer,
    Source,
    TraceEvent,
)


class AgentState(TypedDict):
    """Complete state passed between named graph nodes."""

    question: str
    plan: list[PlanTask]
    next_task_index: int
    evidence: list[EvidenceItem]
    sources: list[Source]
    critique: Critique | None
    output: ResearchAnswer | None
    usage: BudgetUsage
    limits: BudgetLimits
    trace: list[TraceEvent]
    validation_errors: list[str]

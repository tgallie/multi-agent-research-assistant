"""Public data contracts for research runs and tool boundaries."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields at trust boundaries."""

    model_config = ConfigDict(extra="forbid")


class Source(StrictModel):
    """Canonical evidence source exposed in a final answer."""

    id: str = Field(pattern=r"^src_[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=300)
    url: HttpUrl
    snippet: str = Field(min_length=1, max_length=4_000)


class ResearchAnswer(StrictModel):
    """Validated public response returned by every run."""

    answer: str = Field(min_length=1, max_length=20_000)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source] = Field(max_length=25)
    reasoning_summary: str = Field(min_length=1, max_length=4_000)
    budget_exceeded: bool = False

    @field_validator("sources")
    @classmethod
    def unique_source_ids(cls, sources: list[Source]) -> list[Source]:
        """Reject ambiguous duplicate citation identifiers."""

        ids = [source.id for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique")
        return sources


class ToolStatus(StrEnum):
    """Normalized outcomes for untrusted tool calls."""

    OK = "ok"
    ERROR = "error"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ToolResult(StrictModel):
    """Sanitized output returned by a research tool."""

    tool_name: str
    status: ToolStatus
    content: str = Field(default="", max_length=12_000)
    sources: list[Source] = Field(default_factory=list, max_length=10)
    error: str | None = Field(default=None, max_length=1_000)
    latency_ms: int = Field(ge=0)


class TraceEvent(StrictModel):
    """Safe, serializable event in a run's decision trail."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node: str
    event: str
    detail: dict[str, Any] = Field(default_factory=dict)


class BudgetLimits(StrictModel):
    """Hard resource ceilings for one run."""

    max_iterations: int = Field(default=4, ge=1, le=20)
    max_tool_calls: int = Field(default=8, ge=1, le=50)
    max_estimated_tokens: int = Field(default=12_000, ge=500, le=200_000)
    max_output_retries: int = Field(default=1, ge=0, le=3)


class BudgetUsage(StrictModel):
    """Mutable counters compared with `BudgetLimits`."""

    iterations: int = 0
    tool_calls: int = 0
    estimated_tokens: int = 0
    output_retries: int = 0
    exceeded: bool = False
    exceeded_reason: str | None = None


class PlanTask(StrictModel):
    """One bounded research task produced by the planner."""

    id: str = Field(pattern=r"^task_[a-zA-Z0-9_-]+$")
    question: str = Field(min_length=3, max_length=500)
    tool: Literal["web_search", "python", "scratchpad"]
    arguments: dict[str, Any]
    attempts: int = Field(default=0, ge=0, le=5)


class EvidenceItem(StrictModel):
    """Sanitized evidence attached to one plan task."""

    task_id: str
    tool: str
    content: str = Field(max_length=12_000)
    source_ids: list[str] = Field(default_factory=list)


class Critique(StrictModel):
    """Evidence sufficiency decision produced by the critic."""

    sufficient: bool
    gaps: list[str] = Field(default_factory=list, max_length=10)
    reasoning: str = Field(min_length=1, max_length=2_000)


class RunResult(StrictModel):
    """Complete programmatic result including optional trace metadata."""

    run_id: str
    status: Literal["completed", "degraded", "failed"]
    output: ResearchAnswer
    trace: list[TraceEvent]
    usage: BudgetUsage
    duration_ms: int = Field(ge=0)

"""Thin one-hop research slice used to de-risk the end-to-end contract."""

from __future__ import annotations

import time
from uuid import uuid4

from agent.schemas import (
    BudgetLimits,
    BudgetUsage,
    ResearchAnswer,
    RunResult,
    ToolStatus,
    TraceEvent,
)
from agent.telemetry import JsonlTelemetry
from agent.tools.base import ToolExecutor


def run_single_hop(
    question: str,
    executor: ToolExecutor,
    telemetry: JsonlTelemetry,
    limits: BudgetLimits | None = None,
) -> RunResult:
    """Search once and synthesize a transparent evidence extract."""

    normalized_question = question.strip()
    if len(normalized_question) < 3:
        raise ValueError("question must contain at least three characters")
    started = time.perf_counter()
    run_id = str(uuid4())
    usage = BudgetUsage()
    active_limits = limits or BudgetLimits(max_iterations=1, max_tool_calls=1)
    trace = [TraceEvent(node="initialize", event="question_accepted")]
    result = executor.execute(
        "web_search",
        {"query": normalized_question, "max_results": 5},
        usage,
        active_limits,
    )
    trace.append(
        TraceEvent(
            node="researcher",
            event="tool_completed",
            detail={
                "tool": result.tool_name,
                "status": result.status,
                "sources": len(result.sources),
            },
        )
    )
    if result.status == ToolStatus.OK and result.sources:
        source_lines = "\n".join(f"- {source.snippet} [{source.id}]" for source in result.sources)
        answer = ResearchAnswer(
            answer=f"Evidence gathered for: {normalized_question}\n\n{source_lines}",
            confidence=min(0.85, 0.45 + 0.08 * len(result.sources)),
            sources=result.sources,
            reasoning_summary=(
                "Single-hop mode reports normalized search evidence without inference."
            ),
        )
        status = "completed"
    else:
        answer = ResearchAnswer(
            answer=(
                "No usable web evidence was returned; the question could not be answered reliably."
            ),
            confidence=0.0,
            sources=[],
            reasoning_summary=f"The search tool ended with status={result.status}.",
            budget_exceeded=usage.exceeded,
        )
        status = "degraded"
    trace.append(TraceEvent(node="synthesizer", event="output_created"))
    run = RunResult(
        run_id=run_id,
        status=status,
        output=answer,
        trace=trace,
        usage=usage,
        duration_ms=int((time.perf_counter() - started) * 1_000),
    )
    telemetry.emit(run)
    return run

"""Researcher node: execute exactly one planned tool call."""

from __future__ import annotations

from agent.schemas import EvidenceItem, ToolStatus, TraceEvent
from agent.state import AgentState
from agent.tools import ToolExecutor


class ResearcherNode:
    """Execute the next task through the centralized tool boundary."""

    def __init__(self, executor: ToolExecutor) -> None:
        """Inject the allow-listed executor."""

        self._executor = executor

    def __call__(self, state: AgentState) -> dict[str, object]:
        """Execute one task and append only sanitized evidence."""

        usage = state["usage"]
        usage.iterations += 1
        if usage.iterations > state["limits"].max_iterations:
            usage.exceeded = True
            usage.exceeded_reason = "iteration_limit"
            return {
                "usage": usage,
                "trace": state["trace"] + [TraceEvent(node="researcher", event="budget_exhausted")],
            }
        index = state["next_task_index"]
        if index >= len(state["plan"]):
            return {"trace": state["trace"]}
        task = state["plan"][index]
        result = self._executor.execute(task.tool, task.arguments, usage, state["limits"])
        evidence = list(state["evidence"])
        sources = list(state["sources"])
        if result.status == ToolStatus.OK and result.content:
            evidence.append(
                EvidenceItem(
                    task_id=task.id,
                    tool=task.tool,
                    content=result.content,
                    source_ids=[source.id for source in result.sources],
                )
            )
            known_ids = {source.id for source in sources}
            for source in result.sources:
                if source.id not in known_ids:
                    sources.append(source)
                    known_ids.add(source.id)
        return {
            "evidence": evidence,
            "sources": sources,
            "next_task_index": index + 1,
            "usage": usage,
            "trace": state["trace"]
            + [
                TraceEvent(
                    node="researcher",
                    event="tool_completed",
                    detail={
                        "task_id": task.id,
                        "tool": task.tool,
                        "status": result.status,
                        "source_count": len(result.sources),
                    },
                )
            ],
        }

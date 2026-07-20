"""Shared validation and execution boundary for untrusted tools."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from agent.schemas import BudgetLimits, BudgetUsage, ToolResult, ToolStatus

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_TOOL_CONTENT_CHARACTERS = 12_000


class ResearchTool(Protocol):
    """Interface implemented by every allow-listed research tool."""

    name: str
    args_schema: type[BaseModel]

    def invoke(self, arguments: BaseModel) -> ToolResult:
        """Execute validated arguments and return normalized output."""


def sanitize_text(value: object, *, limit: int = MAX_TOOL_CONTENT_CHARACTERS) -> str:
    """Normalize untrusted text and cap its context footprint."""

    text = CONTROL_CHARACTERS.sub("", str(value)).strip()
    return text[:limit]


class ToolExecutor:
    """Validate, allow-list, invoke, and account for all tool calls."""

    def __init__(self, tools: list[ResearchTool]) -> None:
        """Register tools and reject duplicate names."""

        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return stable allow-listed tool names."""

        return tuple(sorted(self._tools))

    def execute(
        self,
        name: str,
        raw_arguments: Mapping[str, Any],
        usage: BudgetUsage,
        limits: BudgetLimits,
    ) -> ToolResult:
        """Run a validated tool call without leaking exceptions to the graph."""

        if usage.tool_calls >= limits.max_tool_calls:
            usage.exceeded = True
            usage.exceeded_reason = "tool_call_limit"
            return ToolResult(
                tool_name=name,
                status=ToolStatus.REJECTED,
                error="tool-call budget exhausted",
                latency_ms=0,
            )
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                status=ToolStatus.REJECTED,
                error=f"tool is not allow-listed; available={self.names}",
                latency_ms=0,
            )
        try:
            arguments = tool.args_schema.model_validate(dict(raw_arguments))
        except ValidationError as exc:
            return ToolResult(
                tool_name=name,
                status=ToolStatus.REJECTED,
                error=sanitize_text(exc, limit=1_000),
                latency_ms=0,
            )

        usage.tool_calls += 1
        started = time.perf_counter()
        try:
            result = tool.invoke(arguments)
        except TimeoutError:
            return ToolResult(
                tool_name=name,
                status=ToolStatus.TIMEOUT,
                error="tool timed out",
                latency_ms=int((time.perf_counter() - started) * 1_000),
            )
        except Exception as exc:  # Tool implementations are an untrusted boundary.
            return ToolResult(
                tool_name=name,
                status=ToolStatus.ERROR,
                error=sanitize_text(exc, limit=1_000),
                latency_ms=int((time.perf_counter() - started) * 1_000),
            )
        return result.model_copy(
            update={
                "content": sanitize_text(result.content),
                "error": sanitize_text(result.error, limit=1_000) if result.error else None,
            }
        )

    def reset_run(self) -> None:
        """Reset stateful tools before a new isolated graph run."""

        for tool in self._tools.values():
            reset = getattr(tool, "reset_run", None)
            if callable(reset):
                reset()

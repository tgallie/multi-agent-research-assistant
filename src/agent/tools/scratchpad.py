"""Run-scoped in-memory scratchpad tool."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from agent.schemas import ToolResult, ToolStatus
from agent.tools.base import sanitize_text


class ScratchpadArgs(BaseModel):
    """Validated scratchpad operation."""

    operation: Literal["write", "read", "list"]
    key: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    value: str | None = Field(default=None, max_length=8_000)


class ScratchpadTool:
    """Store intermediate findings without persisting across runs."""

    name = "scratchpad"
    args_schema = ScratchpadArgs

    def __init__(self) -> None:
        """Create an empty, isolated store."""

        self._values: dict[str, str] = {}

    def invoke(self, arguments: BaseModel) -> ToolResult:
        """Perform a validated read, write, or key listing."""

        args = ScratchpadArgs.model_validate(arguments)
        started = time.perf_counter()
        if args.operation == "write":
            if args.key is None or args.value is None:
                return self._rejected("write requires key and value")
            self._values[args.key] = sanitize_text(args.value, limit=8_000)
            content = f"stored:{args.key}"
        elif args.operation == "read":
            if args.key is None:
                return self._rejected("read requires key")
            if args.key not in self._values:
                return self._rejected(f"key not found: {args.key}")
            content = self._values[args.key]
        else:
            content = "\n".join(sorted(self._values))
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            content=content,
            latency_ms=int((time.perf_counter() - started) * 1_000),
        )

    def reset_run(self) -> None:
        """Erase all findings so graph instances are safe to reuse."""

        self._values.clear()

    def _rejected(self, error: str) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.REJECTED,
            error=error,
            latency_ms=0,
        )

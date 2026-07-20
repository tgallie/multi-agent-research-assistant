"""Constrained Python calculation tool for non-adversarial research tasks."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from typing import ClassVar

from pydantic import BaseModel, Field

from agent.schemas import ToolResult, ToolStatus
from agent.tools.base import sanitize_text


class PythonArgs(BaseModel):
    """Validated Python calculation request."""

    expression: str = Field(min_length=1, max_length=2_000)


class PythonSandboxTool:
    """Evaluate small arithmetic expressions in an isolated subprocess."""

    name = "python"
    args_schema = PythonArgs
    _allowed_nodes: ClassVar[tuple[type[ast.AST], ...]] = (
        ast.Expression,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Subscript,
        ast.Slice,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
    )
    _runner = (
        "import json,sys\n"
        "expr=json.loads(sys.stdin.read())['expression']\n"
        "value=eval(compile(expr,'<sandbox>','eval'),{'__builtins__':{}},{})\n"
        "print(json.dumps({'result':value},default=str))\n"
    )

    def __init__(self, timeout_seconds: float = 2.0) -> None:
        """Configure a strict wall-clock limit."""

        if not 0.1 <= timeout_seconds <= 10.0:
            raise ValueError("timeout_seconds must be between 0.1 and 10")
        self._timeout_seconds = timeout_seconds

    def invoke(self, arguments: BaseModel) -> ToolResult:
        """Validate the AST, evaluate it in isolated mode, and normalize output."""

        args = PythonArgs.model_validate(arguments)
        tree = ast.parse(args.expression, mode="eval")
        rejected = [
            type(node).__name__
            for node in ast.walk(tree)
            if not isinstance(node, self._allowed_nodes)
        ]
        if rejected:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.REJECTED,
                error=f"expression contains forbidden syntax: {sorted(set(rejected))}",
                latency_ms=0,
            )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", self._runner],
                input=json.dumps({"expression": args.expression}),
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("python calculation timed out") from exc
        latency_ms = int((time.perf_counter() - started) * 1_000)
        if completed.returncode != 0:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=sanitize_text(completed.stderr, limit=1_000) or "calculation failed",
                latency_ms=latency_ms,
            )
        payload = json.loads(completed.stdout)
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            content=sanitize_text(payload.get("result")),
            latency_ms=latency_ms,
        )

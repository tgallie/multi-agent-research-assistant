"""Tests for untrusted tool boundaries."""

from agent.schemas import BudgetLimits, BudgetUsage, ToolStatus
from agent.tools import PythonSandboxTool, ScratchpadTool, ToolExecutor
from agent.tools.base import sanitize_text


def test_python_tool_calculates_arithmetic() -> None:
    result = ToolExecutor([PythonSandboxTool()]).execute(
        "python",
        {"expression": "(17 * 3) + 2"},
        BudgetUsage(),
        BudgetLimits(),
    )

    assert result.status == ToolStatus.OK
    assert result.content == "53"


def test_python_tool_rejects_import_and_calls() -> None:
    result = ToolExecutor([PythonSandboxTool()]).execute(
        "python",
        {"expression": "__import__('os').getcwd()"},
        BudgetUsage(),
        BudgetLimits(),
    )

    assert result.status == ToolStatus.REJECTED
    assert "forbidden" in (result.error or "")


def test_scratchpad_is_run_scoped_and_sanitized() -> None:
    scratchpad = ScratchpadTool()
    executor = ToolExecutor([scratchpad])
    usage = BudgetUsage()
    limits = BudgetLimits()

    write = executor.execute(
        "scratchpad",
        {"operation": "write", "key": "finding", "value": "safe\x00text"},
        usage,
        limits,
    )
    read = executor.execute("scratchpad", {"operation": "read", "key": "finding"}, usage, limits)

    assert write.status == ToolStatus.OK
    assert read.content == "safetext"
    assert (
        ScratchpadTool().invoke(ScratchpadTool.args_schema(operation="read", key="finding")).status
        == ToolStatus.REJECTED
    )


def test_executor_rejects_unknown_tool_without_spending_budget() -> None:
    usage = BudgetUsage()
    result = ToolExecutor([]).execute("shell", {}, usage, BudgetLimits())

    assert result.status == ToolStatus.REJECTED
    assert usage.tool_calls == 0


def test_executor_stops_at_tool_budget() -> None:
    usage = BudgetUsage(tool_calls=1)
    result = ToolExecutor([ScratchpadTool()]).execute(
        "scratchpad", {"operation": "list"}, usage, BudgetLimits(max_tool_calls=1)
    )

    assert result.status == ToolStatus.REJECTED
    assert usage.exceeded is True
    assert usage.exceeded_reason == "tool_call_limit"


def test_sanitizer_removes_control_characters_and_caps_length() -> None:
    assert sanitize_text("a\x00b" * 10, limit=5) == "ababa"

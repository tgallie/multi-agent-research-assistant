"""Integration tests for explicit graph routing with mocked dependencies."""

from pathlib import Path

from pydantic import BaseModel

from agent.graph import ResearchGraph
from agent.model import HeuristicModel
from agent.schemas import BudgetLimits, Source, ToolResult, ToolStatus
from agent.telemetry import JsonlTelemetry
from agent.tools import ToolExecutor


class SearchArgs(BaseModel):
    query: str
    max_results: int


class FakeSearch:
    name = "web_search"
    args_schema = SearchArgs

    def invoke(self, arguments: BaseModel) -> ToolResult:
        args = SearchArgs.model_validate(arguments)
        slug = str(abs(hash(args.query)))[:8]
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            content=f"Verified evidence for {args.query}",
            sources=[
                Source(
                    id=f"src_{slug}",
                    title="Fixture source",
                    url=f"https://example.com/{slug}",
                    snippet=f"Verified evidence for {args.query}",
                )
            ],
            latency_ms=1,
        )


def test_graph_runs_named_roles_and_returns_cited_fallback(tmp_path: Path) -> None:
    graph = ResearchGraph(
        model=HeuristicModel(),
        executor=ToolExecutor([FakeSearch()]),
        telemetry=JsonlTelemetry(tmp_path / "runs.jsonl"),
    )

    result = graph.run("Compare alpha and beta")

    assert [event.node for event in result.trace] == [
        "initialize",
        "planner",
        "researcher",
        "researcher",
        "critic",
        "synthesizer",
    ]
    assert result.output.sources
    assert all(source.id in result.output.answer for source in result.output.sources)
    assert result.status == "degraded"  # Offline model intentionally triggers synthesis fallback.


def test_graph_degrades_gracefully_when_iteration_budget_is_exhausted(tmp_path: Path) -> None:
    graph = ResearchGraph(
        model=HeuristicModel(),
        executor=ToolExecutor([FakeSearch()]),
        telemetry=JsonlTelemetry(tmp_path / "runs.jsonl"),
        limits=BudgetLimits(max_iterations=1, max_tool_calls=5),
    )

    result = graph.run("Compare alpha and beta")

    assert result.status == "degraded"
    assert result.output.budget_exceeded is True
    assert result.usage.exceeded_reason == "iteration_limit"
    assert result.output.answer

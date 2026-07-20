"""End-to-end tests for the first runnable slice."""

from pathlib import Path

from pydantic import BaseModel

from agent.schemas import Source, ToolResult, ToolStatus
from agent.telemetry import JsonlTelemetry
from agent.thin import run_single_hop
from agent.tools import ToolExecutor


class FakeArgs(BaseModel):
    query: str
    max_results: int


class FakeSearch:
    name = "web_search"
    args_schema = FakeArgs

    def invoke(self, arguments: BaseModel) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            content="trusted after normalization",
            sources=[
                Source(
                    id="src_example",
                    title="Example",
                    url="https://example.com/research",
                    snippet="A relevant fact.",
                )
            ],
            latency_ms=1,
        )


def test_single_hop_returns_cited_evidence_and_writes_telemetry(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"

    result = run_single_hop(
        "What is the relevant fact?",
        ToolExecutor([FakeSearch()]),
        JsonlTelemetry(log_path),
    )

    assert result.status == "completed"
    assert result.output.sources[0].id in result.output.answer
    assert result.usage.tool_calls == 1
    assert result.trace[-1].node == "synthesizer"
    assert result.run_id in log_path.read_text(encoding="utf-8")

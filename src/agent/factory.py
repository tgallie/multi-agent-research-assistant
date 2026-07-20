"""Composition root shared by CLI and web UI."""

from __future__ import annotations

from pydantic import BaseModel

from agent.config import Settings
from agent.graph import ResearchGraph
from agent.model import HeuristicModel, OpenAICompatibleModel
from agent.schemas import ToolResult, ToolStatus
from agent.telemetry import JsonlTelemetry
from agent.tools import PythonSandboxTool, ScratchpadTool, ToolExecutor, WebSearchTool


class UnavailableSearchArgs(BaseModel):
    """Permissive schema used to normalize missing search credentials."""

    query: str
    max_results: int = 5


class UnavailableSearchTool:
    """Return a typed error when no web-search provider is configured."""

    name = "web_search"
    args_schema = UnavailableSearchArgs

    def invoke(self, arguments: BaseModel) -> ToolResult:
        """Explain the configuration gap without raising."""

        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.ERROR,
            error="set TAVILY_API_KEY or SERPAPI_API_KEY to enable web research",
            latency_ms=0,
        )


def build_graph(settings: Settings | None = None) -> ResearchGraph:
    """Build the production graph from environment-backed configuration."""

    active = settings or Settings()
    if active.openai_api_key:
        model = OpenAICompatibleModel(
            api_key=active.openai_api_key,
            base_url=active.openai_base_url,
            model=active.openai_model,
            timeout_seconds=active.request_timeout_seconds,
        )
    else:
        model = HeuristicModel()
    if active.tavily_api_key:
        search = WebSearchTool(
            provider="tavily",
            api_key=active.tavily_api_key,
            timeout_seconds=active.request_timeout_seconds,
        )
    elif active.serpapi_api_key:
        search = WebSearchTool(
            provider="serpapi",
            api_key=active.serpapi_api_key,
            timeout_seconds=active.request_timeout_seconds,
        )
    else:
        search = UnavailableSearchTool()
    return ResearchGraph(
        model=model,
        executor=ToolExecutor([search, PythonSandboxTool(), ScratchpadTool()]),
        telemetry=JsonlTelemetry(active.agent_log_path),
        limits=active.budget_limits(),
    )

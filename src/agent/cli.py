"""Command-line entrypoint for research runs."""

import json

import typer

from agent.config import Settings
from agent.telemetry import JsonlTelemetry
from agent.thin import run_single_hop
from agent.tools import ToolExecutor, WebSearchTool

app = typer.Typer(no_args_is_help=True)


@app.command()
def query(question: str, trace: bool = typer.Option(False, "--trace")) -> None:
    """Run the current research pipeline for one question."""

    settings = Settings()
    if settings.tavily_api_key:
        tool = WebSearchTool(provider="tavily", api_key=settings.tavily_api_key)
    elif settings.serpapi_api_key:
        tool = WebSearchTool(provider="serpapi", api_key=settings.serpapi_api_key)
    else:
        raise typer.BadParameter("set TAVILY_API_KEY or SERPAPI_API_KEY")
    result = run_single_hop(
        question,
        ToolExecutor([tool]),
        JsonlTelemetry(settings.agent_log_path),
        settings.budget_limits(),
    )
    payload = result.model_dump(mode="json") if trace else result.output.model_dump(mode="json")
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()

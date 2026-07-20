"""Minimal FastAPI surface for live research demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent.factory import build_graph
from agent.schemas import RunResult


class RunnableGraph(Protocol):
    """Small graph interface injected into the API for testability."""

    def run(self, question: str) -> RunResult:
        """Run one research request."""


class ResearchRequest(BaseModel):
    """Validated API request body."""

    question: str = Field(min_length=3, max_length=2_000)


def create_app(graph: RunnableGraph | None = None) -> FastAPI:
    """Create an application with an injectable research graph."""

    app = FastAPI(title="Guarded Research Agent", version="0.1.0")
    app.state.graph = graph or build_graph()
    template_path = Path(__file__).parents[2] / "ui" / "index.html"

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report process liveness without calling external providers."""

        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the self-contained demonstration client."""

        return template_path.read_text(encoding="utf-8")

    @app.post("/api/research", response_model=RunResult)
    def research(payload: ResearchRequest, request: Request) -> RunResult:
        """Execute a bounded research run and return output plus trace."""

        return request.app.state.graph.run(payload.question)

    return app


def main() -> None:
    """Run the local demonstration server."""

    uvicorn.run("agent.api:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

"""API tests with the graph and all external calls mocked."""

from fastapi.testclient import TestClient

from agent.api import create_app
from agent.schemas import BudgetUsage, ResearchAnswer, RunResult, TraceEvent


class FakeGraph:
    def run(self, question: str) -> RunResult:
        return RunResult(
            run_id="run-api",
            status="completed",
            output=ResearchAnswer(
                answer=f"Answer for {question}",
                confidence=0.8,
                sources=[],
                reasoning_summary="Mocked graph response.",
            ),
            trace=[TraceEvent(node="planner", event="plan_created")],
            usage=BudgetUsage(),
            duration_ms=5,
        )


def test_health_does_not_invoke_graph() -> None:
    response = TestClient(create_app(FakeGraph())).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_returns_structured_run() -> None:
    response = TestClient(create_app(FakeGraph())).post(
        "/api/research", json={"question": "A valid question"}
    )

    assert response.status_code == 200
    assert response.json()["output"]["confidence"] == 0.8
    assert response.json()["trace"][0]["node"] == "planner"


def test_research_endpoint_rejects_short_question() -> None:
    response = TestClient(create_app(FakeGraph())).post("/api/research", json={"question": "?"})

    assert response.status_code == 422

"""Tests for deterministic evaluation and failure bucketing."""

from agent.evaluation import EvalCase, score_case, summarize
from agent.schemas import BudgetUsage, ResearchAnswer, RunResult, TraceEvent


def make_result(*, answer: str, tool_calls: int, budget_exceeded: bool = False) -> RunResult:
    return RunResult(
        run_id="run-test",
        status="degraded" if budget_exceeded else "completed",
        output=ResearchAnswer(
            answer=answer,
            confidence=0.5,
            sources=[],
            reasoning_summary="Test result",
            budget_exceeded=budget_exceeded,
        ),
        trace=[
            TraceEvent(node="researcher", event="tool_completed", detail={"tool": "web_search"})
        ],
        usage=BudgetUsage(tool_calls=tool_calls, exceeded=budget_exceeded),
        duration_ms=1,
    )


def test_score_case_buckets_independent_failures() -> None:
    case = EvalCase(
        id="q-test",
        question="Find two facts",
        expected_keywords=["alpha", "beta"],
        reference_facts="alpha beta",
        minimum_tool_calls=2,
    )

    score = score_case(case, make_result(answer="alpha", tool_calls=1, budget_exceeded=True))

    assert score.keyword_score == 0.5
    assert set(score.failures) == {
        "premature_stop",
        "hallucinated_citation",
        "budget_exceeded",
        "fact_check_failed",
    }


def test_summarize_aggregates_failure_counts() -> None:
    case = EvalCase(
        id="q-test",
        question="Find two facts",
        expected_keywords=["alpha", "beta"],
        reference_facts="alpha beta",
        minimum_tool_calls=2,
    )
    score = score_case(case, make_result(answer="alpha", tool_calls=1))

    summary = summarize([score, score])

    assert summary.cases == 2
    assert summary.mean_keyword_score == 0.5
    assert summary.failure_counts["premature_stop"] == 2


def test_score_case_detects_wrong_tool_choice() -> None:
    case = EvalCase(
        id="q-python",
        question="Calculate a value",
        expected_keywords=["four", "4"],
        reference_facts="four is 4",
        required_tools=["python"],
        minimum_tool_calls=1,
    )

    score = score_case(case, make_result(answer="four is 4", tool_calls=1))

    assert "wrong_tool_choice" in score.failures

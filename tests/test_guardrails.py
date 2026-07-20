"""Tests for token budgets and structured-output citation retries."""

from typing import Any

import pytest

from agent.guardrails import BudgetExceededError, complete_json_with_budget
from agent.nodes.synthesizer import SynthesizerNode
from agent.schemas import (
    BudgetLimits,
    BudgetUsage,
    EvidenceItem,
    Source,
)
from agent.state import AgentState


class StaticModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def make_state() -> AgentState:
    source = Source(
        id="src_known",
        title="Known",
        url="https://example.com/known",
        snippet="Known evidence",
    )
    return AgentState(
        question="What is known?",
        plan=[],
        next_task_index=0,
        evidence=[
            EvidenceItem(
                task_id="task_1",
                tool="web_search",
                content="Known evidence",
                source_ids=[source.id],
            )
        ],
        sources=[source],
        critique=None,
        output=None,
        usage=BudgetUsage(),
        limits=BudgetLimits(max_output_retries=1),
        trace=[],
        validation_errors=[],
    )


def answer_payload(source_id: str) -> dict[str, Any]:
    return {
        "answer": f"Known evidence [{source_id}]",
        "confidence": 0.8,
        "sources": [
            {
                "id": source_id,
                "title": "Known",
                "url": "https://example.com/known",
                "snippet": "Known evidence",
            }
        ],
        "reasoning_summary": "The answer follows the provided source.",
        "budget_exceeded": False,
    }


def test_synthesizer_retries_unknown_citation_then_accepts_known_source() -> None:
    model = StaticModel([answer_payload("src_hallucinated"), answer_payload("src_known")])
    state = make_state()

    update = SynthesizerNode(model)(state)

    assert update["output"].sources[0].id == "src_known"
    assert model.calls == 2
    assert state["usage"].output_retries == 1
    assert "unknown source" in update["validation_errors"][0]


def test_synthesizer_replaces_model_source_metadata_with_canonical_evidence() -> None:
    payload = answer_payload("src_known")
    payload["sources"][0]["url"] = "https://attacker.example/changed"
    payload["sources"][0]["snippet"] = "Invented metadata"
    state = make_state()

    update = SynthesizerNode(StaticModel([payload]))(state)

    assert str(update["output"].sources[0].url) == "https://example.com/known"
    assert update["output"].sources[0].snippet == "Known evidence"


def test_model_call_is_refused_before_crossing_token_budget() -> None:
    usage = BudgetUsage(estimated_tokens=490)

    with pytest.raises(BudgetExceededError):
        complete_json_with_budget(
            StaticModel([{}]),
            system="long system instruction",
            user="long user input",
            usage=usage,
            limits=BudgetLimits(max_estimated_tokens=500),
        )

    assert usage.exceeded is True
    assert usage.exceeded_reason == "estimated_token_limit"

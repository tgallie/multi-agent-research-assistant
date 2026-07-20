"""Reproducible multi-hop evaluation and failure bucketing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from pydantic import BaseModel, Field

from agent.graph import ResearchGraph
from agent.model import AgentModel, HeuristicModel
from agent.schemas import RunResult, Source, ToolResult, ToolStatus
from agent.telemetry import JsonlTelemetry
from agent.tools import ToolExecutor

app = typer.Typer(no_args_is_help=True)


class EvalCase(BaseModel):
    """One labeled multi-hop evaluation example."""

    id: str
    question: str
    expected_keywords: list[str] = Field(min_length=2)
    reference_facts: str
    required_tools: list[str] = Field(default_factory=lambda: ["web_search"])
    minimum_tool_calls: int = Field(default=2, ge=1)


class CaseScore(BaseModel):
    """Deterministic and optional judge scores for one case."""

    case_id: str
    keyword_score: float = Field(ge=0, le=1)
    citation_valid: bool
    judge_score: float | None = Field(default=None, ge=0, le=1)
    tool_calls: int
    failures: list[str]


class EvalSummary(BaseModel):
    """Aggregate evaluation artifact."""

    cases: int
    mean_keyword_score: float
    citation_valid_rate: float
    mean_judge_score: float | None
    failure_counts: dict[str, int]
    scores: list[CaseScore]


class FixtureSearchArgs(BaseModel):
    """Search arguments accepted by the deterministic eval fixture."""

    query: str
    max_results: int = 5


class FixtureSearchTool:
    """Return case-specific public facts without a network dependency."""

    name = "web_search"
    args_schema = FixtureSearchArgs

    def __init__(self, case: EvalCase) -> None:
        """Bind a fixture tool to one eval case to prevent cross-case leakage."""

        self._case = case

    def invoke(self, arguments: BaseModel) -> ToolResult:
        """Return labeled facts as citation-bearing normalized evidence."""

        FixtureSearchArgs.model_validate(arguments)
        source = Source(
            id=f"src_{self._case.id}",
            title=f"Evaluation fixture for {self._case.id}",
            url=f"https://example.com/eval/{self._case.id}",
            snippet=self._case.reference_facts,
        )
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            content=self._case.reference_facts,
            sources=[source],
            latency_ms=0,
        )


def load_cases(path: Path) -> list[EvalCase]:
    """Load and validate a JSONL evaluation set."""

    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case ids must be unique")
    return cases


def score_case(case: EvalCase, result: RunResult, judge: AgentModel | None = None) -> CaseScore:
    """Score fact coverage, provenance, tool behavior, and optional answer quality."""

    answer_lower = result.output.answer.lower()
    matched = sum(keyword.lower() in answer_lower for keyword in case.expected_keywords)
    keyword_score = matched / len(case.expected_keywords)
    source_ids = {source.id for source in result.output.sources}
    citation_valid = bool(source_ids) and all(
        source_id in result.output.answer for source_id in source_ids
    )
    tools_used = [
        str(event.detail.get("tool")) for event in result.trace if event.event == "tool_completed"
    ]
    failures: list[str] = []
    if not set(case.required_tools).issubset(tools_used):
        failures.append("wrong_tool_choice")
    if result.usage.tool_calls < case.minimum_tool_calls:
        failures.append("premature_stop")
    if not citation_valid:
        failures.append("hallucinated_citation")
    if result.output.budget_exceeded:
        failures.append("budget_exceeded")
    if keyword_score < 1.0:
        failures.append("fact_check_failed")
    judge_score = _judge(case, result, judge) if judge else None
    return CaseScore(
        case_id=case.id,
        keyword_score=keyword_score,
        citation_valid=citation_valid,
        judge_score=judge_score,
        tool_calls=result.usage.tool_calls,
        failures=failures,
    )


def _judge(case: EvalCase, result: RunResult, judge: AgentModel) -> float:
    payload = judge.complete_json(
        "Score answer correctness and grounding from 0 to 1. Return {score, explanation}.",
        json.dumps(
            {
                "question": case.question,
                "reference_facts": case.reference_facts,
                "answer": result.output.answer,
                "sources": [source.model_dump(mode="json") for source in result.output.sources],
            }
        ),
    )
    score = payload.get("score")
    if not isinstance(score, int | float) or not 0 <= score <= 1:
        raise ValueError("judge returned an invalid score")
    return float(score)


def summarize(scores: list[CaseScore]) -> EvalSummary:
    """Aggregate case scores and explicit failure-mode counts."""

    if not scores:
        raise ValueError("at least one case score is required")
    judged = [score.judge_score for score in scores if score.judge_score is not None]
    failures = Counter(failure for score in scores for failure in score.failures)
    return EvalSummary(
        cases=len(scores),
        mean_keyword_score=sum(score.keyword_score for score in scores) / len(scores),
        citation_valid_rate=sum(score.citation_valid for score in scores) / len(scores),
        mean_judge_score=sum(judged) / len(judged) if judged else None,
        failure_counts=dict(sorted(failures.items())),
        scores=scores,
    )


def render_markdown(summary: EvalSummary, *, mode: str) -> str:
    """Render the reviewer-facing evaluation table and failure breakdown."""

    judge = f"{summary.mean_judge_score:.3f}" if summary.mean_judge_score is not None else "not run"
    failures = summary.failure_counts or {"none": 0}
    rows = "\n".join(f"| {name} | {count} |" for name, count in failures.items())
    return f"""# Evaluation Results

Mode: **{mode}**

| Cases | Mean keyword/fact score | Citation-valid rate | LLM judge |
|---:|---:|---:|---:|
| {summary.cases} | {summary.mean_keyword_score:.3f} | {summary.citation_valid_rate:.3f} | {judge} |

## Failure buckets

| Failure mode | Count |
|---|---:|
{rows}

The offline fixture mode validates graph traversal, multi-call behavior, deterministic fact checks,
and citation plumbing. It does **not** measure live search recall or model synthesis quality.
Run live evaluation with configured provider keys before using these numbers for model selection.
"""


@app.command()
def run(
    dataset: Path = Path("eval/questions.jsonl"),
    output: Path = Path("eval/results.md"),
) -> None:
    """Run the deterministic offline baseline and write its Markdown artifact."""

    scores: list[CaseScore] = []
    for case in load_cases(dataset):
        graph = ResearchGraph(
            model=HeuristicModel(),
            executor=ToolExecutor([FixtureSearchTool(case)]),
            telemetry=JsonlTelemetry(Path("eval/runs/offline.jsonl")),
        )
        scores.append(score_case(case, graph.run(case.question)))
    summary = summarize(scores)
    output.write_text(
        render_markdown(summary, mode="offline deterministic fixture"), encoding="utf-8"
    )
    typer.echo(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    app()

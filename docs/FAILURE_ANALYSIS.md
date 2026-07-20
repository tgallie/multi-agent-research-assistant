# Failure-Mode Analysis

## Scope and reading of the baseline

The committed offline run covers 16 labeled, multi-call questions. It scored 1.000 on deterministic keyword coverage and citation validity, with no observed failure buckets. That is a wiring baseline, not an estimate of real-world answer quality: case-bound fixtures guarantee retrieval of the labeled facts, the heuristic model deliberately avoids free-form synthesis, and no LLM judge ran without credentials.

A zero-failure fixture result is useful only if the detectors themselves are tested. The test suite therefore injects malformed model output, unknown citations, insufficient tool calls, missing tools, and exhausted budgets.

## Taxonomy and response

| Bucket | Detection signal | Likely causes | Automated response | Operator action |
|---|---|---|---|---|
| `wrong_tool_choice` | Required eval tools are absent from trace | Weak plan, ambiguous tool descriptions, provider outage masked as planning | Critic names uncovered tasks; bounded re-search | Inspect plan/tool traces and confusion by question class |
| `premature_stop` | Tool calls below labeled minimum or planned tasks lack evidence | Critic false positive, task-index bug, model refusal | Deterministic coverage fallback sends missing tasks back once | Compare task completion and critic decision distributions |
| `hallucinated_citation` | Final source ID is unknown or cited source ID is absent from answer | Model invented provenance, output schema was copied incorrectly | Reject output, retry with validation error, then evidence-only fallback | Audit prompt and provider-specific JSON behavior |
| `budget_exceeded` | Usage reaches token, tool, or iteration ceiling | Over-decomposition, repeated low-yield search, oversized evidence | Stop new work, preserve trace, synthesize best effort with flag | Tune per-class limits only after inspecting marginal evidence gain |
| `fact_check_failed` | Expected facts/keywords are absent | Retrieval miss, contradiction, synthesis omission, stale label | Record independently of judge score | Inspect source recall before changing synthesis prompts |

Buckets are non-exclusive. A run can, for example, stop early because the tool budget is exhausted and also fail the fact check. Preserving that overlap is more useful than forcing one root-cause label.

## Fault-injection coverage

| Injected condition | Expected defense | Test |
|---|---|---|
| Synthesizer returns an unknown source ID | Reject and retry once with validation feedback | `test_synthesizer_retries_unknown_citation_then_accepts_known_source` |
| Model request would cross token cap | Refuse call before dispatch and set explicit reason | `test_model_call_is_refused_before_crossing_token_budget` |
| Graph has one iteration for two tasks | Return evidence gathered so far with `budget_exceeded=true` | `test_graph_degrades_gracefully_when_iteration_budget_is_exhausted` |
| Evaluation result has one call where two are required | Add `premature_stop` independently of other failures | `test_score_case_buckets_independent_failures` |
| Planner requests a non-allow-listed tool | Reject without spending tool budget | `test_executor_rejects_unknown_tool_without_spending_budget` |
| Tool output contains control characters | Strip before evidence enters graph state | `test_scratchpad_is_run_scoped_and_sanitized` |

## What a live evaluation should answer

Before changing model or prompt configuration, run the same set against a live search provider and an LLM judge, then inspect:

1. source recall before answer quality, because synthesis cannot repair missing evidence;
2. failure rate by question pattern, not only the aggregate mean;
3. critic re-search yield, measured as additional accepted evidence per retry;
4. token and latency percentiles for successful and degraded runs;
5. citation validity separately from factual correctness.

The first production tuning candidate would be planner tool selection. The offline heuristic selects web search for each clause and does not yet learn when a Python calculation should follow retrieval. That limitation is visible in traces and should be evaluated with labels before adding planner prompt complexity.


---
title: "Microsoft Agent Framework (Python) — Evaluation"
description: "Quality gates for agents and workflows: evaluate_agent, evaluate_workflow, LocalEvaluator, @evaluator decorator, keyword_check and tool_call helpers — all from agent-framework-core 1.1.0."
framework: microsoft-agent-framework
language: python
---

# Evaluation — Python

`agent_framework` ships a lightweight evaluation harness built into the core package. It runs a list of queries against an agent (or pre-recorded responses), extracts a normalised `EvalItem` per query, then applies a set of **checks** that produce pass/fail + score results. The same pattern scales to workflows via `evaluate_workflow`.

Everything is in-process by default — no Azure AI Foundry / Microsoft Foundry dependency required. Plug a cloud evaluator in alongside `LocalEvaluator` when you want LLM-judge / risk / groundedness checks.

Verified against `agent-framework-core==1.1.0` (`agent_framework._evaluation`). This module is marked `experimental` — API may evolve.

## The three building blocks

| Primitive | Role |
|---|---|
| `EvalItem` | One row in the test set — query, response, conversation, expected output, expected tool calls, tools, context |
| `EvalCheck` | A function `(EvalItem) -> CheckResult \| Awaitable[CheckResult]` — one rule |
| `Evaluator` | Protocol that runs many checks over many items — `LocalEvaluator` is the built-in implementation |

## Minimal example

```python
import asyncio
from agent_framework import (
    Agent,
    LocalEvaluator,
    evaluate_agent,
    keyword_check,
    tool,
    tool_called_check,
)
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is 22°C."


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a friendly weather assistant.",
        tools=[get_weather],
    )

    local = LocalEvaluator(
        keyword_check("weather"),
        tool_called_check("get_weather"),
    )

    [results] = await evaluate_agent(
        agent=agent,
        queries=[
            "What's the weather in Amsterdam?",
            "Is it raining in Seattle?",
        ],
        evaluators=local,
    )

    print(f"passed: {results.result_counts['passed']}  failed: {results.result_counts['failed']}")
    for item in results.items:
        print(item.status, item.input_text, "→", [s.name for s in item.scores if not s.passed])


asyncio.run(main())
```

`evaluate_agent` returns `list[EvalResults]` — one entry per evaluator you pass. With a single evaluator, unpack it directly.

## Built-in checks

```python
from agent_framework import (
    keyword_check,
    tool_called_check,
    tool_calls_present,
    tool_call_args_match,
)
```

| Check | What it asserts |
|---|---|
| `keyword_check(*keywords, case_sensitive=False)` | Every keyword appears in `item.response`. |
| `tool_called_check(*tool_names, mode="all" \| "any")` | Named tools were actually invoked. |
| `tool_calls_present(item)` | Every tool in `item.expected_tool_calls` was invoked at least once (unordered, extras allowed). |
| `tool_call_args_match(item)` | Expected tool calls ran AND their arguments match. |

## Custom checks with `@evaluator`

Plain Python functions become checks via the `@evaluator` decorator. Parameter names are introspected — return a bool, float, dict, or `CheckResult`.

```python
from agent_framework import evaluator, CheckResult, LocalEvaluator


@evaluator
def mentions_celsius(response: str) -> bool:
    return "°C" in response or "celsius" in response.lower()


@evaluator(name="response_length")
def short_enough(response: str) -> float:
    # Float ≥ 0.5 = pass
    return 1.0 if len(response) < 500 else 0.3


@evaluator
async def llm_judge(query: str, response: str) -> CheckResult:
    # Use any client you like — this is the typical LLM-as-judge pattern.
    score = await my_judge_client.score(query=query, response=response)
    return CheckResult(
        passed=score >= 0.7,
        reason=f"judge score {score:.2f}",
        check_name="llm_judge",
    )


local = LocalEvaluator(mentions_celsius, short_enough, llm_judge)
```

Supported parameter names (pick any subset): `query`, `response`, `expected_output`, `conversation`, `tools`, `context`.

## Ground-truth comparisons

Pass expected outputs and tool calls; checks that care about them use those fields on `EvalItem`.

```python
from agent_framework import ExpectedToolCall

results = await evaluate_agent(
    agent=agent,
    queries=["What's 2+2?", "Capital of France?"],
    expected_output=["4", "Paris"],
    expected_tool_calls=[
        [],                                               # no tools expected
        [ExpectedToolCall("lookup", {"topic": "Paris"})], # one expected tool call
    ],
    evaluators=LocalEvaluator(tool_call_args_match, keyword_check("Paris")),
)
```

### How `tool_call_args_match` actually matches

The built-in `tool_call_args_match` check does a **subset match** on the arguments — expected keys must be present with the expected values, but the actual call may carry additional arguments that aren't asserted against:

```python
# Expected
ExpectedToolCall("get_weather", {"location": "NYC"})

# Passes — actual has the expected key/value, plus one extra
{"location": "NYC", "unit": "celsius"}

# Fails — value mismatch
{"location": "Seattle"}

# Passes when arguments=None — only the name is checked
ExpectedToolCall("get_weather", arguments=None)
```

This lets you write tight assertions that don't break when the agent (or the model) starts passing extra optional arguments to a tool.

### Multi-tool sequences

`expected_tool_calls` is a list per query — use it to assert the agent followed a specific plan:

```python
results = await evaluate_agent(
    agent=agent,
    queries=["Plan a trip to Paris for next weekend"],
    expected_tool_calls=[[
        ExpectedToolCall("get_weather", {"location": "Paris"}),
        ExpectedToolCall("search_flights", {"destination": "Paris"}),
        ExpectedToolCall("search_hotels"),       # only name, args unchecked
    ]],
    evaluators=LocalEvaluator(tool_call_args_match),
)
```

`tool_call_args_match` is **order-insensitive** — each expected call is matched against the actual call list by name, not by position. Use a custom `@evaluator` if you need to enforce ordering.

## Pre-recorded responses

Evaluate the agent you already have (no re-running) — useful for replaying production transcripts:

```python
# Get a response once
response = await agent.run("What's the weather?")

# Score it later — or in CI against stored transcripts
results = await evaluate_agent(
    agent=agent,                       # still needed for tool definitions
    queries="What's the weather?",
    responses=response,
    evaluators=LocalEvaluator(keyword_check("weather")),
)
```

## Measuring consistency

`num_repetitions=N` runs each query N times, giving you N rows per query. Combine with a determinism check to spot variance:

```python
await evaluate_agent(
    agent=agent,
    queries="What's the weather in Tokyo?",
    evaluators=LocalEvaluator(keyword_check("Tokyo")),
    num_repetitions=5,
)
```

## Workflow evaluation

```python
from agent_framework import evaluate_workflow

results = await evaluate_workflow(
    workflow=research_pipeline,
    queries=["Quantum sensors", "Photonics"],
    evaluators=LocalEvaluator(keyword_check("summary")),
)
```

`evaluate_workflow` runs the workflow end-to-end for each query, extracts the final output, and produces the same `EvalItem` shape. Tool definitions and conversation history are pulled from the workflow's agent executors automatically.

## Cloud + local composition

Microsoft Foundry ships a richer evaluator with groundedness, relevance, safety, and PII checks. Install `agent-framework-foundry` and mix it in — `evaluate_agent` accepts a list:

```python
from agent_framework.foundry import FoundryEvals

foundry_evals = FoundryEvals(project_client=project, model="gpt-4o-mini")

all_results = await evaluate_agent(
    agent=agent,
    queries=[...],
    evaluators=[LocalEvaluator(keyword_check("weather")), foundry_evals],
)

for result in all_results:
    print(result.provider, result.result_counts)
```

One `EvalResults` comes back per evaluator — local first, then foundry. Merge them or publish each separately.

## Failing the build

`EvalResults.result_counts["failed"]` is non-zero when any item fails any check. In CI, assert it's zero (or call `EvalResults.raise_if_failed()` when available) — the test run halts and the failure reasons surface in `EvalItemResult.scores`:

```python
[results] = await evaluate_agent(agent=agent, queries=queries, evaluators=local)
assert results.result_counts["failed"] == 0, results.error
```

### Actionable failure output for CI

`EvalItemResult.scores` carries per-check `EvalScoreResult` entries. Walk them to print a machine-readable diff plus a human-readable reason per failure:

```python
import json
import sys
from agent_framework import LocalEvaluator, evaluate_agent, keyword_check


async def ci_gate(agent, queries: list[str]) -> None:
    local = LocalEvaluator(keyword_check("weather"), tool_called_check("get_weather"))
    [results] = await evaluate_agent(agent=agent, queries=queries, evaluators=local)

    failures = [item for item in results.items if item.status == "fail"]
    if not failures:
        print(f"pass: {results.result_counts['passed']}/{len(results.items)}")
        return

    for item in failures:
        failed_checks = [s for s in item.scores if not s.passed]
        print(json.dumps({
            "id": item.item_id,
            "input": item.input_text,
            "output": item.output_text,
            "failed": [
                {"check": s.name, "reason": (s.sample or {}).get("reason")}
                for s in failed_checks
            ],
        }, indent=2))

    sys.exit(1)
```

Plug that into a pytest test or a standalone CI step — the failing check name plus `reason` is enough to triage most regressions without opening the transcript UI.

## Patterns

**Smoke test in CI.** A small `LocalEvaluator` with `keyword_check` + `tool_called_check` catches regressions caused by prompt edits without spending judge tokens.

**Nightly regression.** Record production transcripts → evaluate offline with `responses=...` and an LLM judge → post pass/fail trend to a dashboard.

**Per-PR quality gate.** Run `evaluate_agent` against a small curated test set on every PR; block merge if failures appear. Pair with [Observability](./microsoft_agent_framework_python_observability/) so regressions surface as traces too.

**Compare two models.** Build two agents — one per `OpenAIChatClient(model=...)` — and call `evaluate_agent` on each. Since `LocalEvaluator` is cheap, run thousands of queries locally in minutes.

**Evaluate orchestrations.** `evaluate_workflow` accepts any `Workflow`, including those produced by `SequentialBuilder`, `GroupChatBuilder`, and `MagenticBuilder`. Same checks, same score aggregation.

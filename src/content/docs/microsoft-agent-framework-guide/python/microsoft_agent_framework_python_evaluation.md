---
title: "Microsoft Agent Framework (Python) — Evaluation"
description: "Quality gates for agents and workflows: evaluate_agent, evaluate_workflow, LocalEvaluator, @evaluator decorator, keyword_check and tool_call helpers — all from agent-framework-core 1.5.0."
framework: microsoft-agent-framework
language: python
---

# Evaluation — Python

`agent_framework` ships a lightweight evaluation harness built into the core package. It runs a list of queries against an agent (or pre-recorded responses), extracts a normalised `EvalItem` per query, then applies a set of **checks** that produce pass/fail + score results. The same pattern scales to workflows via `evaluate_workflow`.

Everything is in-process by default — no Azure AI Foundry / Microsoft Foundry dependency required. Plug a cloud evaluator in alongside `LocalEvaluator` when you want LLM-judge / risk / groundedness checks.

Verified against `agent-framework-core==1.5.0` (`agent_framework._evaluation`). This module is marked `experimental` — API may evolve.

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

### Reference — what each `@evaluator` parameter receives

The framework introspects the function signature and only passes the parameters you declare. Mix and match:

| Parameter | Type | What it carries |
|---|---|---|
| `query` | `str` | The user query for this `EvalItem` (last user message under `LAST_TURN` split, or the whole user side under `FULL`). |
| `response` | `str` | The assistant response under the same split. |
| `expected_output` | `str` | Ground-truth answer if you provided one via `evaluate_agent(expected_output=...)`. |
| `conversation` | `list[Message]` | The full conversation, untouched by splitter. Inspect tool calls, system prompts, multi-turn flow. |
| `tools` | `list[FunctionTool]` | The tools the agent had registered when it produced the response. |
| `context` | `Optional[str]` | Grounding context provided to `evaluate_agent(context=...)`. |

A single check can pull whatever combination it needs:

```python
from agent_framework import CheckResult, evaluator, FunctionTool, Message


@evaluator(name="cite_only_documented_tools")
def only_documented_tools(
    response: str,
    conversation: list[Message],
    tools: list[FunctionTool],
) -> CheckResult:
    """Fail if the response cites tool names the agent doesn't actually have."""
    declared = {t.name for t in tools}
    cited = {
        c.name
        for msg in conversation
        for c in (msg.contents or [])
        if c.type == "function_call" and c.name
    }
    hallucinated = cited - declared
    return CheckResult(
        passed=not hallucinated,
        reason=(
            "all cited tools are declared"
            if not hallucinated
            else f"hallucinated tool names: {sorted(hallucinated)}"
        ),
        check_name="cite_only_documented_tools",
    )
```

You can return a plain `bool` or `float` for simple checks; reach for `CheckResult` when you want the failure reason in `EvalScoreResult.sample["reason"]` so it propagates to your CI logs and dashboards.

### Async LLM-judge with rate limiting

When the judge is itself an LLM call, async + a semaphore is the right shape — the framework awaits async checks transparently:

```python
import asyncio
from agent_framework import CheckResult, evaluator
from agent_framework.openai import OpenAIChatClient

judge_client = OpenAIChatClient(model="gpt-4o-mini")
judge_semaphore = asyncio.Semaphore(8)        # cap concurrency to 8 in-flight judges

JUDGE_PROMPT = """\
Score the assistant response on a 0.0 – 1.0 scale for factual accuracy
given the user's question and the grounding context. Reply with the score on
the first line and a one-line reason on the second.
"""


@evaluator(name="factuality_judge")
async def factuality(query: str, response: str, context: str) -> CheckResult:
    async with judge_semaphore:
        result = await judge_client.get_response(
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {query}\nContext: {context}\nResponse: {response}",
                },
            ],
        )
    score_line, _, reason = result.text.partition("\n")
    try:
        score = float(score_line.strip())
    except ValueError:
        return CheckResult(
            passed=False,
            reason=f"could not parse score from judge output: {result.text[:80]!r}",
            check_name="factuality_judge",
        )
    return CheckResult(
        passed=score >= 0.7,
        reason=f"score={score:.2f} — {reason.strip()}",
        check_name="factuality_judge",
    )
```

`LocalEvaluator` runs every check for every item via `asyncio.gather`, so the semaphore is what actually bounds spend on the judge model.

## Conversation splits — choosing what "response" means

`EvalItem` stores a full conversation and derives `query` / `response` from it via a `ConversationSplitter` strategy. The built-in `ConversationSplit` enum gives you two strategies out of the box; anything callable with signature `(list[Message]) -> (list[Message], list[Message])` satisfies the `ConversationSplitter` protocol.

```python
from agent_framework import ConversationSplit, EvalItem, Message

conversation = [
    Message(role="system", contents=["You are an assistant."]),
    Message(role="user", contents=["What's 2+2?"]),
    Message(role="assistant", contents=["4"]),
    Message(role="user", contents=["Square that."]),
    Message(role="assistant", contents=["16"]),
]

# Default strategy — last_turn. Query = everything up to the last user msg; response = after.
item = EvalItem(conversation=conversation)
assert item.query == "Square that."
assert item.response == "16"

# FULL strategy — evaluate the whole trajectory against the first user msg.
item_full = EvalItem(conversation=conversation, split_strategy=ConversationSplit.FULL)
assert item_full.query == "What's 2+2?"
assert "4" in item_full.response and "16" in item_full.response
```

### Custom splitter — evaluate just before a tool call

A custom splitter is a plain callable. Here's one that splits just before the agent called a retrieval tool — perfect for evaluating whether the agent generated a *good* retrieval query, independent of whether the tool returned good results:

```python
from agent_framework import ConversationSplitter, EvalItem, Message


def split_before_tool(tool_name: str) -> ConversationSplitter:
    """Return a splitter that isolates everything up to a named tool call."""

    def _split(conversation: list[Message]) -> tuple[list[Message], list[Message]]:
        for i, msg in enumerate(conversation):
            for c in msg.contents or []:
                if getattr(c, "type", None) == "function_call" and getattr(c, "name", None) == tool_name:
                    return conversation[:i], conversation[i:]
        # No matching tool call — fall back to the static last-turn splitter.
        return EvalItem._split_last_turn_static(conversation)

    return _split


item = EvalItem(
    conversation=recorded_transcript,
    split_strategy=split_before_tool("retrieve_docs"),
)
```

### Splitter + evaluate_agent

Pass the splitter through `evaluate_agent` via `conversation_split=` so every recorded item uses the same strategy:

```python
from agent_framework import ConversationSplit, LocalEvaluator, evaluate_agent, keyword_check

results = await evaluate_agent(
    agent=agent,
    queries=queries,
    conversation_split=ConversationSplit.FULL,
    evaluators=LocalEvaluator(keyword_check("summary")),
)

# Or with a custom splitter (any callable matching the ConversationSplitter protocol):
results = await evaluate_agent(
    agent=agent,
    queries=queries,
    conversation_split=split_before_tool("retrieve_docs"),
    evaluators=LocalEvaluator(keyword_check("retrieved")),
)
```

Use `ConversationSplit.LAST_TURN` when you care about the latest answer and `ConversationSplit.FULL` when you care about the whole trajectory (did the agent stay on-task across N turns?). Drop to a custom splitter when the evaluation boundary depends on domain-specific signals — tool calls, explicit handoffs, state transitions.

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

## Custom `Evaluator` — going beyond `LocalEvaluator`

`LocalEvaluator` is a single concrete implementation of the `Evaluator` protocol — a `name: str` attribute plus one async method:

```python
class Evaluator(Protocol):
    name: str

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str = "Eval",
    ) -> EvalResults: ...
```

Roll your own evaluator when you need behaviour that doesn't fit `LocalEvaluator`'s "every check must pass" rule — weighted scoring, golden-dataset comparisons that need warm caches, federation across multiple backends, or per-item parallelism budgets.

### Weighted scorer

Aggregate multiple scorers into one numeric score, with a configurable pass threshold:

```python
from collections.abc import Awaitable, Callable, Sequence
from agent_framework import (
    EvalItem,
    EvalItemResult,
    EvalResults,
    EvalScoreResult,
    LocalEvaluator,
    evaluate_agent,
    keyword_check,
)


class WeightedScorer:
    """Aggregate multiple scorers into one weighted pass/fail decision.

    Each scorer returns a float in ``[0, 1]``. The item passes overall when
    the weighted average meets ``threshold``. Per-scorer ``passed`` flags use
    ``per_scorer_threshold`` (default ``0.5``) so individual results stay
    interpretable independently of the global aggregate.
    """

    def __init__(
        self,
        scorers: dict[str, tuple[float, Callable[[EvalItem], float | Awaitable[float]]]],
        *,
        threshold: float = 0.7,
        per_scorer_threshold: float = 0.5,
        name: str = "weighted",
    ) -> None:
        self.name = name
        self.scorers = scorers
        self.threshold = threshold
        self.per_scorer_threshold = per_scorer_threshold
        total_weight = sum(weight for weight, _ in scorers.values())
        if total_weight <= 0:
            raise ValueError("scorer weights must sum to a positive number")
        self._total_weight = total_weight

    async def _score_one(self, item: EvalItem) -> tuple[float, list[EvalScoreResult]]:
        per_scorer_scores: list[EvalScoreResult] = []
        weighted_sum = 0.0
        for name, (weight, fn) in self.scorers.items():
            raw = fn(item)
            value = await raw if hasattr(raw, "__await__") else raw
            score = max(0.0, min(1.0, float(value)))
            weighted_sum += weight * score
            # Per-scorer pass uses its own cutoff so a passing aggregate doesn't
            # mask a failing individual scorer in `per_evaluator` counts.
            per_scorer_scores.append(
                EvalScoreResult(name=name, score=score, passed=score >= self.per_scorer_threshold)
            )
        return weighted_sum / self._total_weight, per_scorer_scores

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str = "Weighted",
    ) -> EvalResults:
        passed = 0
        failed = 0
        result_items: list[EvalItemResult] = []
        per_check: dict[str, dict[str, int]] = {
            name: {"passed": 0, "failed": 0, "errored": 0} for name in self.scorers
        }
        for idx, item in enumerate(items):
            score, per_scorer = await self._score_one(item)
            item_passed = score >= self.threshold
            if item_passed:
                passed += 1
            else:
                failed += 1
            for s in per_scorer:
                per_check[s.name]["passed" if s.passed else "failed"] += 1
            result_items.append(
                EvalItemResult(
                    item_id=str(idx),
                    status="pass" if item_passed else "fail",
                    scores=[*per_scorer, EvalScoreResult(name="weighted", score=score, passed=item_passed)],
                    input_text=item.query,
                    output_text=item.response,
                )
            )
        return EvalResults(
            provider=self.name,
            eval_id="weighted",
            run_id=eval_name,
            status="completed",
            result_counts={"passed": passed, "failed": failed, "errored": 0},
            per_evaluator=per_check,
            items=result_items,
        )


def length_score(item: EvalItem) -> float:
    # Reward responses between 50 and 400 chars; penalise anything outside.
    n = len(item.response)
    if 50 <= n <= 400:
        return 1.0
    return max(0.0, 1.0 - abs(n - 200) / 800)


def cites_temperature(item: EvalItem) -> float:
    return 1.0 if "°C" in item.response or "celsius" in item.response.lower() else 0.0


scorer = WeightedScorer(
    scorers={
        "length": (1.0, length_score),
        "temperature_cited": (3.0, cites_temperature),  # 3× weight
    },
    threshold=0.7,
)

# Run side-by-side with LocalEvaluator — evaluate_agent accepts a list.
all_results = await evaluate_agent(
    agent=agent,
    queries=["What's the weather in Paris?", "How hot is it in Cairo?"],
    evaluators=[LocalEvaluator(keyword_check("weather")), scorer],
)

local_results, weighted_results = all_results
print(weighted_results.per_evaluator)
# {'length': {...}, 'temperature_cited': {...}}
```

The custom evaluator slots into the same pipeline as `LocalEvaluator` and Microsoft Foundry — `evaluate_agent` returns one `EvalResults` per evaluator, in registration order, so callers stay framework-agnostic.

### Federating across backends

Wrap two evaluators behind one `Evaluator` so callers see them as a single backend:

```python
class FederatedEvaluator:
    """Run two evaluators sequentially and combine their pass/fail counts."""

    def __init__(self, *backends, name: str = "federated") -> None:
        self.name = name
        self.backends = backends

    async def evaluate(self, items, *, eval_name="Federated") -> EvalResults:
        all_results = [await b.evaluate(items, eval_name=eval_name) for b in self.backends]
        merged_counts = {"passed": 0, "failed": 0, "errored": 0}
        merged_per_check: dict[str, dict[str, int]] = {}
        merged_items: list[EvalItemResult] = []
        for r in all_results:
            for k, v in r.result_counts.items():
                merged_counts[k] = merged_counts.get(k, 0) + v
            merged_per_check.update(r.per_evaluator)
            merged_items.extend(r.items)
        return EvalResults(
            provider=self.name,
            eval_id="federated",
            run_id=eval_name,
            status="completed",
            result_counts=merged_counts,
            per_evaluator=merged_per_check,
            items=merged_items,
        )
```

This is the pattern Microsoft Foundry's `FoundryEvals` uses internally to combine groundedness, relevance, and safety into a single result object — with a custom `Evaluator` you can do the same for whichever scorers you have.

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

`EvalResults.result_counts["failed"]` is non-zero when any item fails any check. Call `results.raise_for_status()` to raise `ValueError` when failures are present — the test run halts and the failure reasons surface in `EvalItemResult.scores`. Pass an optional message to customise the exception text:

```python
[results] = await evaluate_agent(agent=agent, queries=queries, evaluators=local)
results.raise_for_status("weather agent quality gate failed")
# → ValueError: weather agent quality gate failed (2 failed, 0 errored)
```

Convenience properties let you branch without walking the count dict:

```python
print(results.passed)     # int — items that passed all checks
print(results.failed)     # int — items that failed at least one check
print(results.total)      # int — total items scored
print(results.all_passed) # bool — True iff failed == 0
```

### Actionable failure output for CI

`EvalItemResult.scores` carries per-check `EvalScoreResult` entries. Walk them to print a machine-readable diff plus a human-readable reason per failure:

```python
import json
import sys
from agent_framework import (
    LocalEvaluator,
    evaluate_agent,
    keyword_check,
    tool_called_check,
)


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

## Running `LocalEvaluator` directly on `EvalItem`s

`evaluate_agent` is the common entry point — it runs the agent, builds `EvalItem`s, and scores them. For offline regression tests against a recorded corpus, skip the agent entirely and feed `EvalItem`s into `LocalEvaluator.evaluate(...)`:

```python
from agent_framework import (
    Content,
    ConversationSplit,
    EvalItem,
    ExpectedToolCall,
    LocalEvaluator,
    Message,
    keyword_check,
    tool_calls_present,
)


items = [
    # Item 1 — a real production trace with a tool call. Note the function_call
    # and function_result Content entries: tool_calls_present inspects
    # conversation for these, matching them against expected_tool_calls.
    EvalItem(
        conversation=[
            Message(role="user", contents=[Content.from_text("What's the weather in Oslo?")]),
            Message(
                role="assistant",
                contents=[Content.from_function_call(
                    call_id="call_1",
                    name="get_weather",
                    arguments={"location": "Oslo"},
                )],
            ),
            Message(
                role="tool",
                contents=[Content.from_function_result(
                    call_id="call_1",
                    result={"temp_c": -2, "condition": "snow"},
                )],
            ),
            Message(role="assistant", contents=[Content.from_text("It's -2°C and snowing in Oslo.")]),
        ],
        expected_tool_calls=[ExpectedToolCall("get_weather", {"location": "Oslo"})],
    ),
    # Item 2 — plain text-only trace. No tool calls expected.
    EvalItem(
        conversation=[
            Message(role="user", contents=[Content.from_text("Summarise this doc.")]),
            Message(role="assistant", contents=[Content.from_text("The doc is about X, Y, Z.")]),
        ],
        expected_output="The doc is about X, Y, Z.",
        split_strategy=ConversationSplit.LAST_TURN,
    ),
]

# tool_calls_present reads item.expected_tool_calls — it's a no-op on items
# that don't set it, so the second item passes the check trivially. Use
# tool_called_check(name) only when your conversation actually contains
# function_call Content entries for that name.
local = LocalEvaluator(keyword_check("°C"), tool_calls_present)
results = await local.evaluate(items, eval_name="offline-regression")

print(results.status, results.result_counts)     # completed {'passed': 1, 'failed': 1, 'errored': 0}
for item in results.items:
    print(item.item_id, item.status, [s.name for s in item.scores if not s.passed])
```

- Item 1 passes both checks (response contains `°C`, expected `get_weather` call is present).
- Item 2 fails `keyword_check("°C")` (its response doesn't mention temperature).

Which check to use depends on what the recorded trace carries. `tool_called_check(name)` walks the conversation for `function_call` content and fails if that content isn't present — the right choice when you trust the trace format. `tool_calls_present` and `tool_call_args_match` compare the conversation's actual calls against `EvalItem.expected_tool_calls` — the right choice when different items have different expectations or some items have none.

This keeps the evaluator loop cheap — no LLM calls, no network — ideal for replaying production traces in CI.

### Reading per-check breakdowns from `EvalResults`

Every `EvalResults` returned by `LocalEvaluator.evaluate()` carries a `per_evaluator` map keyed by check name. Use it to summarise which checks failed most often without walking every item:

```python
results = await local.evaluate(items)

for check_name, counts in results.per_evaluator.items():
    total = counts["passed"] + counts["failed"] + counts["errored"]
    pass_rate = counts["passed"] / total if total else 0
    print(f"{check_name}: {pass_rate:.0%} passed ({counts})")
```

Plot those over time and you get a per-check regression dashboard — cheap, entirely local, driven by `LocalEvaluator` output.

### Returning `CheckResult` for rich failure context

The minimal `@evaluator` return type is `bool`, but return `CheckResult` when you want the failure reason to surface in `EvalScoreResult.sample["reason"]` and eventually in your CI output. That's the only way to attach a message the triage engineer will see next Monday morning:

```python
from agent_framework import CheckResult, evaluator


@evaluator
def no_hallucinated_prices(response: str, context: str) -> CheckResult:
    import re
    prices_in_response = set(re.findall(r"\$\d+(?:\.\d{2})?", response))
    prices_in_context  = set(re.findall(r"\$\d+(?:\.\d{2})?", context or ""))
    hallucinated = prices_in_response - prices_in_context

    return CheckResult(
        passed=not hallucinated,
        reason=(
            "all prices grounded in context"
            if not hallucinated
            else f"hallucinated prices not in context: {sorted(hallucinated)}"
        ),
        check_name="no_hallucinated_prices",
    )
```

When it fails, every `EvalItemResult.scores` entry for that check keeps `sample={"reason": "hallucinated prices not in context: ['$19.99']"}` — drop that into Slack and the on-call engineer can act without opening the transcript UI.

## `EvalItem.per_turn_items` — evaluating multi-turn conversations

`EvalItem.per_turn_items` is a static helper that splits a full multi-turn conversation into one `EvalItem` per user turn. Each item gets cumulative context — the query messages for turn N include everything up to and including the Nth user message, while the response messages cover the agent's reply up to the next user turn. Use it to evaluate how the agent performs at each step of a long conversation, not just at the final answer:

```python
import asyncio
from agent_framework import (
    Agent,
    Content,
    EvalItem,
    LocalEvaluator,
    Message,
    keyword_check,
    tool,
    tool_called_check,
)
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(location: str) -> str:
    """Get the current weather."""
    return f"The weather in {location} is 18°C."


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a weather assistant.",
        tools=[get_weather],
    )

    # Record a real multi-turn conversation
    session = agent.create_session()
    await agent.run("What's the weather in Paris?", session=session)
    await agent.run("And in Tokyo?", session=session)
    await agent.run("Which city is warmer?", session=session)

    # Reconstruct the conversation from session history (or load from storage)
    # For this example we build the conversation manually to demonstrate the helper:
    conversation = [
        Message(role="user",      contents=[Content.from_text("What's the weather in Paris?")]),
        Message(role="assistant", contents=[Content.from_text("It's 18°C in Paris.")]),
        Message(role="user",      contents=[Content.from_text("And in Tokyo?")]),
        Message(role="assistant", contents=[Content.from_text("It's 18°C in Tokyo too.")]),
        Message(role="user",      contents=[Content.from_text("Which city is warmer?")]),
        Message(role="assistant", contents=[Content.from_text("Both are the same temperature — 18°C.")]),
    ]

    # One EvalItem per user turn, each with cumulative history as context
    items = EvalItem.per_turn_items(
        conversation,
        tools=[get_weather],
        context="Weather assistant test suite",
    )
    print(f"Generated {len(items)} eval items")   # → 3

    local = LocalEvaluator(keyword_check("18°C"))
    results = await local.evaluate(items, eval_name="per-turn-weather")

    print(f"passed={results.passed}/{results.total}")
    for item in results.items:
        print(f"  turn {item.item_id}: {item.status} — query: {item.input_text!r}")


asyncio.run(main())
```

`per_turn_items` is particularly useful for:
- **Regression testing conversation flows** — check that the agent's mid-conversation answers are also correct, not just the final one.
- **Evaluating grounding drift** — an LLM-judge evaluator can check whether the agent's reasoning stays consistent across turns.
- **Tool-call coverage per turn** — pair with `tool_called_check` to assert the agent called the right tool at the right step.

## `AgentEvalConverter` — bridging to cloud evaluators

When you want to send evaluation items to a cloud provider (such as Microsoft Foundry) that expects OpenAI-style message dicts, `AgentEvalConverter` handles the type conversion from agent-framework's `Message` / `Content` / `FunctionTool` types. All methods are static:

```python
import asyncio
from agent_framework import (
    AgentEvalConverter,
    Content,
    EvalItem,
    Message,
    tool,
)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: shipped"


# A recorded conversation with a tool call
conversation = [
    Message(role="user",      contents=[Content.from_text("Where is order #99?")]),
    Message(
        role="assistant",
        contents=[Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "99"})],
    ),
    Message(
        role="tool",
        contents=[Content.from_function_result(call_id="c1", result="Order 99: shipped")],
    ),
    Message(role="assistant", contents=[Content.from_text("Order #99 has been shipped.")]),
]

# Convert a single message to Foundry-compatible format
foundry_msg = AgentEvalConverter.convert_message(conversation[-1])
# [{"role": "assistant", "content": [{"type": "text", "text": "Order #99 has been shipped."}]}]

# Convert the whole conversation (tool calls become "tool_call" typed entries)
foundry_conv = AgentEvalConverter.convert_messages(conversation)
print(f"{len(foundry_conv)} Foundry-format messages")

# Extract registered tools from an agent as JSON schema dicts for Foundry
# (matches the tool definitions the model saw when it produced the response)
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), tools=[lookup_order])
tool_defs = AgentEvalConverter.extract_tools(agent)
# [{"type": "function", "function": {"name": "lookup_order", "description": "...", "parameters": {...}}}]

# Convert an agent response directly to an EvalItem for offline scoring
async def main():
    response = await agent.run("Where is order #99?")
    item = AgentEvalConverter.to_eval_item(
        query="Where is order #99?",
        response=response,
        agent=agent,
        context="Order management assistant",
    )
    print(item.query, "→", item.response)

asyncio.run(main())
```

`AgentEvalConverter.to_eval_item` is the fastest path from a live `AgentResponse` to an `EvalItem` you can feed to any evaluator — including cloud providers that need the Foundry message format:

```python
from agent_framework.foundry import FoundryEvals

# Collect responses
test_queries = ["Where is order #99?", "Has order #42 shipped?"]
responses = []
for query in test_queries:
    r = await agent.run(query)
    responses.append(AgentEvalConverter.to_eval_item(query=query, response=r, agent=agent))

# Score with Foundry (groundedness, relevance, safety, …)
foundry = FoundryEvals(project_client=project, model="gpt-4o-mini")
[foundry_results] = await foundry.evaluate(responses, eval_name="order-agent-v2")
foundry_results.raise_for_status()
```

## Patterns

**Smoke test in CI.** A small `LocalEvaluator` with `keyword_check` + `tool_called_check` catches regressions caused by prompt edits without spending judge tokens.

**Nightly regression.** Record production transcripts → evaluate offline with `responses=...` and an LLM judge → post pass/fail trend to a dashboard.

**Per-PR quality gate.** Run `evaluate_agent` against a small curated test set on every PR; block merge if failures appear. Pair with [Observability](./microsoft_agent_framework_python_observability/) so regressions surface as traces too.

**Compare two models.** Build two agents — one per `OpenAIChatClient(model=...)` — and call `evaluate_agent` on each. Since `LocalEvaluator` is cheap, run thousands of queries locally in minutes.

**Evaluate orchestrations.** `evaluate_workflow` accepts any `Workflow`, including those produced by `SequentialBuilder`, `GroupChatBuilder`, and `MagenticBuilder`. Same checks, same score aggregation.

**Multi-turn evaluation.** Use `EvalItem.per_turn_items` to split recorded conversations and evaluate every intermediate response, not just the final one. Feed those items directly to `LocalEvaluator.evaluate()` for cheap CI coverage of conversation flows.

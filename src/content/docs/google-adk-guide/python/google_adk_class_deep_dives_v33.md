---
title: "Class deep dives — volume 33 (10 additional classes)"
description: "Source-verified deep dives into 10 additional google-adk 2.3.0 classes: LoggingPlugin, InMemoryEvalSetsManager, _VertexAiEvalFacade hierarchy, SafetyEvaluatorV1, MultiTurnToolUseQualityV1Evaluator, session migration pipeline, AgentBuilderAssistant, yaml_utils, and the V1 SQLAlchemy schema models."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 33"
  order: 102
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source at
`<site-packages>/google/adk/` on
**google-adk == 2.3.0**. Path varies by environment; run `pip show google-adk` to find yours.
No documentation or blog posts were used as primary sources.
</Aside>

## 1 · `LoggingPlugin` — terminal-friendly 12-callback debugging plugin

**Module:** `google.adk.plugins.logging_plugin`

`LoggingPlugin` implements every hook on `BasePlugin` and prints a structured, emoji-prefixed trace to stdout using ANSI grey colour (`\033[90m`). It is the first plugin you should add when debugging locally — it requires no configuration and produces zero overhead in production when removed.

### Key implementation facts (verified from source)

- **12 hooks, all implemented** — `on_user_message_callback`, `before_run_callback`, `after_run_callback`, `before_agent_callback`, `after_agent_callback`, `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`, `on_event_callback`, `on_model_error_callback`, `on_tool_error_callback`. Every callback returns `None` so it never short-circuits the pipeline.
- **`_log(msg)`** — prepends `[{self.name}]`, wraps in ANSI grey, and calls `print()`. There is no Python `logging` integration; output goes directly to stdout.
- **`_format_content(content, max_length=200)`** — iterates `content.parts` and labels each: `text: '<str>'`, `function_call: <name>`, `function_response: <name>`, `code_execution_result`, or `other_part`. Truncates text parts to `max_length` characters.
- **`_format_args(args, max_length=300)`** — calls `str(args)` and truncates to `max_length` characters. Safe for any dict type.
- **`before_model_callback` logs** — model name, agent name, first 200 chars of system instruction (if present), and list of available tool names from `llm_request.tools_dict`.
- **`after_model_callback` logs** — agent name, error code + message (if error), content (via `_format_content`), `partial`, `turn_complete`, and token usage from `usage_metadata.prompt_token_count` / `candidates_token_count`.
- **`on_event_callback` logs** — event ID, author, content, `is_final_response()`, function call names, function response names, and `long_running_tool_ids`.
- **`before/after_tool_callback` logs** — tool name, agent name, function call ID, and arguments (via `_format_args`). The `after_tool_callback` also logs the result dict.
- **Default name** — `"logging_plugin"`. Pass a custom `name=` to distinguish instances when running multiple agents with separate loggers.

### Example 1 — attach to a runner and trace a full invocation

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types


async def main():
    def greet(name: str) -> str:
        """Greet the user by name."""
        return f"Hello, {name}!"

    agent = LlmAgent(
        name="greeter",
        model="gemini-2.0-flash",
        instruction="You are a friendly greeter. Use the greet tool.",
        tools=[greet],
    )

    plugin = LoggingPlugin(name="debug")   # grey logs tagged [debug]

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="demo",
        session_service=session_service,
        plugins=[plugin],
    )

    session = await session_service.create_session(app_name="demo", user_id="u1")
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text="Greet Alice")]),
    ):
        if event.is_final_response():
            print("Final:", event.content.parts[0].text)

asyncio.run(main())
# Each plugin hook prints grey lines like:
# [debug] 🚀 USER MESSAGE RECEIVED
# [debug]    Invocation ID: abc123
# [debug] 🤖 AGENT STARTING
# [debug]    Agent Name: greeter
# [debug] 🧠 LLM REQUEST
# [debug]    Available Tools: ['greet']
# ...
```

### Example 2 — listing all 12 callback hooks available on `LoggingPlugin`

```python
import asyncio
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


async def main():
    agent = LlmAgent(
        name="failer",
        model="gemini-2.0-flash",
        instruction="Answer questions.",
    )
    plugin = LoggingPlugin()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="demo",
        session_service=session_service,
        plugins=[plugin],
    )

    await session_service.create_session(app_name="demo", user_id="u1")

    # LoggingPlugin implements all 12 BasePlugin hooks.
    print("LoggingPlugin hooks:", [
        m for m in dir(plugin)
        if m.endswith("_callback")
    ])

asyncio.run(main())
```

### Example 3 — subclassing `LoggingPlugin` to redirect output to `logging`

```python
import logging
from typing import Optional
from google.adk.plugins.logging_plugin import LoggingPlugin

log = logging.getLogger("adk.trace")


class StructuredLoggingPlugin(LoggingPlugin):
    """Routes LoggingPlugin output to Python's logging instead of stdout."""

    def _log(self, message: str) -> None:
        # Strip ANSI codes and forward to the Python logger at DEBUG level.
        clean = message.replace("\033[90m", "").replace("\033[0m", "")
        log.debug(clean)


# Usage: swap LoggingPlugin() for StructuredLoggingPlugin() anywhere.
plugin = StructuredLoggingPlugin(name="structured")
# Now all callbacks write to log.debug("...") — safe for production logging pipelines.
```

---

## 2 · `InMemoryEvalSetsManager` — zero-dependency eval set storage for testing

**Module:** `google.adk.evaluation.in_memory_eval_sets_manager`

`InMemoryEvalSetsManager` implements the full `EvalSetsManager` abstract interface using nested Python dicts. It is the simplest way to set up eval pipelines in tests without a database or GCS bucket.

### Key implementation facts (verified from source)

- **Dual-dict storage** — maintains two nested dicts: `_eval_sets: dict[app_name, dict[eval_set_id, EvalSet]]` and `_eval_cases: dict[app_name, dict[eval_set_id, dict[eval_case_id, EvalCase]]]`. The outer key in both is `app_name`, enabling multi-app isolation.
- **`_ensure_app_exists(app_name)`** — called at the start of every mutating method; creates empty inner dicts for the app on first use. This avoids `KeyError` without requiring a separate `create_app` call.
- **List synchronisation** — `EvalSet.eval_cases` is a `list[EvalCase]`. `InMemoryEvalSetsManager` keeps this list in sync with the `_eval_cases` dict on every `add_eval_case`, `update_eval_case`, and `delete_eval_case` call. Both representations must be consistent.
- **Duplicate guards** — `create_eval_set` raises `ValueError` if `eval_set_id` already exists. `add_eval_case` raises `ValueError` if `eval_case_id` already exists in the set.
- **`NotFoundError` on missing keys** — `update_eval_case`, `delete_eval_case`, `get_eval_case` return `None` or raise `NotFoundError` (from `google.adk.errors.not_found_error`) when the set or case doesn't exist. This mirrors the behaviour of GCS and disk-backed implementations.
- **`creation_timestamp`** — set to `time.time()` at `create_eval_set` time and never updated.
- **No thread-safety primitives** — the implementation uses plain dict operations without locks. Safe for single-threaded test runners; wrap with a lock for async concurrent tests.

### Example 1 — create a set, add cases, retrieve and delete

```python
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types


manager = InMemoryEvalSetsManager()

# Create a new eval set for the "search_agent" app.
manager.create_eval_set(app_name="search_agent", eval_set_id="baseline_v1")

# Add two eval cases.
case_a = EvalCase(
    eval_id="q_weather",
    conversation=[
        Invocation(
            user_content=types.Content(parts=[types.Part(text="What's the weather in Paris?")]),
        )
    ],
)
case_b = EvalCase(
    eval_id="q_capital",
    conversation=[
        Invocation(
            user_content=types.Content(parts=[types.Part(text="What is the capital of France?")]),
        )
    ],
)
manager.add_eval_case("search_agent", "baseline_v1", case_a)
manager.add_eval_case("search_agent", "baseline_v1", case_b)

# List all eval sets.
sets = manager.list_eval_sets("search_agent")
print(sets)  # ['baseline_v1']

# Retrieve a specific case.
retrieved = manager.get_eval_case("search_agent", "baseline_v1", "q_weather")
assert retrieved.eval_id == "q_weather"

# Delete a case.
manager.delete_eval_case("search_agent", "baseline_v1", "q_capital")
eval_set = manager.get_eval_set("search_agent", "baseline_v1")
print(len(eval_set.eval_cases))  # 1
```

### Example 2 — update an eval case

```python
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types


manager = InMemoryEvalSetsManager()
manager.create_eval_set("my_app", "set_1")

original = EvalCase(
    eval_id="test_001",
    conversation=[
        Invocation(
            user_content=types.Content(parts=[types.Part(text="Hello")]),
        )
    ],
)
manager.add_eval_case("my_app", "set_1", original)

# Update the case with a richer conversation.
updated = EvalCase(
    eval_id="test_001",
    conversation=[
        Invocation(
            user_content=types.Content(parts=[types.Part(text="Hello, who are you?")]),
        )
    ],
    tags=["greeting"],
)
manager.update_eval_case("my_app", "set_1", updated)

result = manager.get_eval_case("my_app", "set_1", "test_001")
assert result.tags == ["greeting"]
# The EvalSet.eval_cases list is also updated in-place.
eval_set = manager.get_eval_set("my_app", "set_1")
assert eval_set.eval_cases[0].tags == ["greeting"]
```

### Example 3 — use `InMemoryEvalSetsManager` as a test fixture

```python
import pytest
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.errors.not_found_error import NotFoundError
from google.genai import types


@pytest.fixture
def manager():
    m = InMemoryEvalSetsManager()
    m.create_eval_set("app", "test_set")
    return m


def test_get_nonexistent_case_returns_none(manager):
    result = manager.get_eval_case("app", "test_set", "nonexistent")
    assert result is None


def test_get_nonexistent_set_returns_none(manager):
    result = manager.get_eval_set("app", "no_such_set")
    assert result is None


def test_duplicate_set_raises_value_error(manager):
    with pytest.raises(ValueError, match="already exists"):
        manager.create_eval_set("app", "test_set")   # second creation fails


def test_missing_app_list_returns_empty(manager):
    result = manager.list_eval_sets("no_such_app")
    assert result == []


def test_delete_missing_case_raises_not_found(manager):
    with pytest.raises(NotFoundError):
        manager.delete_eval_case("app", "test_set", "ghost_case")
```

---

## 3 · `_VertexAiEvalFacade` + `_SingleTurnVertexAiEvalFacade` + `_MultiTurnVertexiAiEvalFacade` — Vertex Gen AI Eval SDK bridge

**Module:** `google.adk.evaluation.vertex_ai_eval_facade`

These three private classes form the bridge between ADK's `Evaluator` interface and Google's Vertex Gen AI Eval SDK (`vertexai.Client.evals.evaluate`). They power `SafetyEvaluatorV1`, `MultiTurnToolUseQualityV1Evaluator`, and any future Vertex-backed evaluators.

### Key implementation facts (verified from source)

- **`_VertexAiEvalFacade.__init__`** — reads `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_API_KEY` from environment. API key takes priority; if absent, both project and location must be set or a descriptive `ValueError` is raised that includes instructions for the `.env` file.
- **`_perform_eval(dataset, metrics)`** — the only method that calls the external service: `self._client.evals.evaluate(dataset=dataset, metrics=metrics)`. Isolated for unit-test patching.
- **`_get_score(eval_result)`** — reads `eval_result.summary_metrics[0].mean_score`; returns `None` if the result is empty, `mean_score` is not a `float`, or is `math.nan`. This prevents NaN from propagating into `EvaluationResult.overall_score`.
- **`_get_eval_status(score)`** — returns `EvalStatus.PASSED` if `score >= threshold`, `EvalStatus.FAILED` if score is below, or `EvalStatus.NOT_EVALUATED` if `score is None`.
- **`_SingleTurnVertexAiEvalFacade`** — evaluates each `Invocation` independently. Builds a single-row `pandas.DataFrame` with `{"prompt": ..., "reference": ..., "response": ...}` and calls `_perform_eval` once per invocation. Aggregates scores as a running mean.
- **`_MultiTurnVertexiAiEvalFacade`** — evaluates all invocations as a single conversation. Marks the first `N-1` turns `NOT_EVALUATED` (score=`None`). On the last turn, it assembles a `vertexai.types.EvalCase` with `AgentData` (agents dict + turns list) and calls `_perform_eval` once for the entire conversation. Score applies only to the last turn.
- **Multi-turn turn mapping** — each `Invocation` becomes a `ConversationTurn(turn_index=i, events=[...], turn_id=invocation.invocation_id)`. The events list starts with a user `AgentEvent`, then all `invocation.intermediate_data.invocation_events`, then a final agent `AgentEvent`.
- **Agent config extraction** — `_get_agent_details()` scans all invocations and builds a `dict[agent_name, AgentConfig]`; the first occurrence wins. `AgentConfig` carries `agent_id`, `instruction`, and `tool_declarations` taken from `invocation.app_details.agent_details`.

### Example 1 — auth patterns for the Vertex AI eval client

```python
import os

# Pattern A: Vertex AI project + location (recommended for production on GCP)
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# Pattern B: API key (simpler for dev; limited to supported regions)
# os.environ["GOOGLE_API_KEY"] = "AIza..."

# The facade reads these at __init__ time. API key takes priority.
# If neither pattern is set, ValueError is raised with remediation instructions.

# SafetyEvaluatorV1 and MultiTurnToolUseQualityV1Evaluator both use
# _VertexAiEvalFacade under the hood — no separate client setup needed.
from google.adk.evaluation.eval_metrics import EvalMetric

metric = EvalMetric(metric_name="safety", threshold=0.5)
# Instantiating SafetyEvaluatorV1(metric) will construct the Vertex client here.
print("Auth env vars set:", bool(os.environ.get("GOOGLE_CLOUD_PROJECT")))
```

### Example 2 — single-turn evaluation: how `_SingleTurnVertexAiEvalFacade` builds the DataFrame

```python
# Illustrates the data flow without making a real API call.
# In real use: SafetyEvaluatorV1(metric).evaluate_invocations(invocations)
from google.genai import types
from google.adk.evaluation.eval_case import Invocation
import pandas as pd

user_content = types.Content(parts=[types.Part(text="Is aspirin safe?")])
final_response = types.Content(parts=[types.Part(text="Yes, in recommended doses.")])

invocation = Invocation(
    user_content=user_content,
    final_response=final_response,
)

# Internally the facade builds this DataFrame for each invocation:
def _get_text(content):
    if content and content.parts:
        return "\n".join(p.text for p in content.parts if p.text)
    return ""

eval_case_dict = {
    "prompt": _get_text(invocation.user_content),          # "Is aspirin safe?"
    "reference": None,                                       # no reference for safety
    "response": _get_text(invocation.final_response),       # "Yes, in recommended doses."
}
df = pd.DataFrame([eval_case_dict])
print(df)
# Passed to: client.evals.evaluate(dataset=EvaluationDataset(eval_dataset_df=df),
#                                   metrics=[PrebuiltMetric.SAFETY])
```

### Example 3 — multi-turn evaluation: conversation turn structure

```python
# Shows how _MultiTurnVertexiAiEvalFacade maps ADK Invocations to Vertex turns.
from google.genai import types
from google.adk.evaluation.eval_case import Invocation, InvocationEvent

turn1 = Invocation(
    invocation_id="inv-001",
    user_content=types.Content(parts=[types.Part(text="What tools do you have?")]),
    final_response=types.Content(parts=[types.Part(text="I have search and calculator.")]),
)
turn2 = Invocation(
    invocation_id="inv-002",
    user_content=types.Content(parts=[types.Part(text="Use the calculator: 2+2")]),
    final_response=types.Content(parts=[types.Part(text="4")]),
)

# _MultiTurnVertexiAiEvalFacade produces:
# - ConversationTurn(turn_index=0) -> NOT_EVALUATED (score=None)
# - ConversationTurn(turn_index=1) -> score from a single client.evals.evaluate() call
# The AgentData.turns list contains BOTH turns; the model sees the full context.
print("First N-1 turns are NOT_EVALUATED, last turn gets the score.")
print(f"Turn 1 id: {turn1.invocation_id}, Turn 2 id: {turn2.invocation_id}")
```

---

## 4 · `SafetyEvaluatorV1` — Vertex AI safety metric for ADK agents

**Module:** `google.adk.evaluation.safety_evaluator`

`SafetyEvaluatorV1` wraps `vertexai.types.PrebuiltMetric.SAFETY` as a drop-in ADK `Evaluator`. It scores each agent response for harmlessness on a `[0, 1]` scale via the Vertex Gen AI Eval SDK.

### Key implementation facts (verified from source)

- **Single-turn metric** — delegates to `_SingleTurnVertexAiEvalFacade`, which evaluates each `Invocation` independently. No reference response is required.
- **Score range `[0, 1]`** — values closer to 1 are safer. The `threshold` in `EvalMetric` determines pass/fail.
- **`V1` suffix convention** — the suffix signals that alternative safety implementations (different models, prompting strategies) may appear as `SafetyEvaluatorV2` etc. in future versions.
- **Requires GCP credentials** — `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` or `GOOGLE_API_KEY` must be set before instantiation; see `_VertexAiEvalFacade.__init__`.
- **`expected_invocations_required=False`** — no golden reference needed. `_SingleTurnVertexAiEvalFacade` receives `expected_invocations_required=False`, so missing expected invocations are auto-filled with `None`.

### Example 1 — evaluate a batch of responses for safety

```python
import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

from google.adk.evaluation.safety_evaluator import SafetyEvaluatorV1
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_case import Invocation
from google.genai import types

metric = EvalMetric(metric_name="safety", threshold=0.7)
evaluator = SafetyEvaluatorV1(eval_metric=metric)

invocations = [
    Invocation(
        user_content=types.Content(parts=[types.Part(text="How do I bake a cake?")]),
        final_response=types.Content(parts=[types.Part(text="Here is a simple recipe...")]),
    ),
    Invocation(
        user_content=types.Content(parts=[types.Part(text="Tell me a joke.")]),
        final_response=types.Content(parts=[types.Part(text="Why did the chicken cross the road?")]),
    ),
]

# Requires live Vertex AI credentials to run.
# result = evaluator.evaluate_invocations(invocations)
# print(result.overall_score)         # e.g. 0.95
# print(result.overall_eval_status)   # EvalStatus.PASSED or EvalStatus.FAILED
# for r in result.per_invocation_results:
#     print(r.score, r.eval_status)
print(f"Evaluator threshold: {metric.threshold}")
print(f"Evaluator metric: {metric.metric_name}")
```

### Example 2 — use `SafetyEvaluatorV1` inside `AgentEvaluator`

```python
# SafetyEvaluatorV1 integrates with the EvalConfig metric registry.
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluation_constants import EvalConstants

# The metric_name must match a key in MetricEvaluatorRegistry.
# Built-in name for safety: "safety"
config = EvalConfig(
    criteria=[
        EvalMetric(metric_name="safety", threshold=0.7),
    ]
)
print("EvalConfig criteria:", config.criteria)
# When AgentEvaluator runs with this config, MetricEvaluatorRegistry
# resolves "safety" -> SafetyEvaluatorV1 automatically.
```

### Example 3 — patch `_perform_eval` for offline unit testing

```python
from unittest.mock import patch, MagicMock
import os

os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

from google.adk.evaluation.safety_evaluator import SafetyEvaluatorV1
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types


def mock_summary_metrics(score: float):
    m = MagicMock()
    m.mean_score = score
    return [m]

mock_result = MagicMock()
mock_result.summary_metrics = mock_summary_metrics(0.95)

with patch(
    "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval",
    return_value=mock_result,
):
    metric = EvalMetric(metric_name="safety", threshold=0.7)
    evaluator = SafetyEvaluatorV1(eval_metric=metric)
    invocations = [
        Invocation(
            user_content=types.Content(parts=[types.Part(text="Hello")]),
            final_response=types.Content(parts=[types.Part(text="Hi!")]),
        )
    ]
    result = evaluator.evaluate_invocations(invocations)
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.overall_score == 0.95
    print("Offline safety eval passed:", result.overall_eval_status)
```

---

## 5 · `MultiTurnToolUseQualityV1Evaluator` — Vertex AI multi-turn tool quality metric

**Module:** `google.adk.evaluation.multi_turn_tool_use_quality_evaluator`

`MultiTurnToolUseQualityV1Evaluator` uses Vertex AI's `RubricMetric.MULTI_TURN_TOOL_USE_QUALITY` to score how well an agent called tools across a multi-turn conversation. It is a *reference-free* metric — no golden tool-call sequence is needed.

### Key implementation facts (verified from source)

- **Multi-turn metric** — delegates to `_MultiTurnVertexiAiEvalFacade`. Only the last turn receives a numeric score; all prior turns are marked `NOT_EVALUATED`.
- **`RubricMetric` vs `PrebuiltMetric`** — `SafetyEvaluatorV1` uses `vertexai.types.PrebuiltMetric.SAFETY`; this evaluator uses `vertexai.types.RubricMetric.MULTI_TURN_TOOL_USE_QUALITY`. The facade handles both via the same `_perform_eval()` dispatch.
- **Reference-free** — `expected_invocations` is accepted but treated as `None` in the facade (not passed to the Vertex API). Omit it in calls.
- **`V1` suffix** — same convention as `SafetyEvaluatorV1`; alternative tool-quality strategies may appear in future versions.
- **Score range `[0, 1]`** — higher scores reflect better tool selection, sequencing, and argument construction across the conversation.
- **Requires GCP credentials** — same as `SafetyEvaluatorV1`: `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` or `GOOGLE_API_KEY`.

### Example 1 — evaluate multi-turn tool use quality

```python
import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

from google.adk.evaluation.multi_turn_tool_use_quality_evaluator import (
    MultiTurnToolUseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_case import Invocation, IntermediateData, InvocationEvent
from google.genai import types

metric = EvalMetric(metric_name="multi_turn_tool_use_quality", threshold=0.6)
evaluator = MultiTurnToolUseQualityV1Evaluator(eval_metric=metric)

# Turn 1: user asks a question; agent calls 'search' tool
tool_call = types.Content(parts=[
    types.Part(function_call=types.FunctionCall(name="search", args={"query": "Paris weather"}))
])
tool_result = types.Content(parts=[
    types.Part(function_response=types.FunctionResponse(name="search", response={"result": "Cloudy, 18°C"}))
])

turn1 = Invocation(
    invocation_id="t1",
    user_content=types.Content(parts=[types.Part(text="What's the weather in Paris?")]),
    intermediate_data=IntermediateData(invocation_events=[
        InvocationEvent(author="agent", content=tool_call),
        InvocationEvent(author="tool", content=tool_result),
    ]),
    final_response=types.Content(parts=[types.Part(text="It's cloudy and 18°C in Paris.")]),
)
turn2 = Invocation(
    invocation_id="t2",
    user_content=types.Content(parts=[types.Part(text="And in London?")]),
    final_response=types.Content(parts=[types.Part(text="London is 15°C and rainy.")]),
)

# result = evaluator.evaluate_invocations([turn1, turn2])
# print(result.per_invocation_results[0].eval_status)  # NOT_EVALUATED
# print(result.per_invocation_results[1].score)        # e.g. 0.82
print(f"Metric threshold: {metric.threshold}")
```

### Example 2 — difference between `PrebuiltMetric` (single-turn) and `RubricMetric` (multi-turn)

```python
# Both are accessed via vertexai.types but serve different evaluation strategies.
# This snippet shows the conceptual difference without live credentials.

# PrebuiltMetric = single-turn, curated by Google, minimal configuration.
#   Used by: SafetyEvaluatorV1
#   Example: PrebuiltMetric.SAFETY, PrebuiltMetric.GROUNDEDNESS

# RubricMetric = multi-turn, rubric-based LLM judge.
#   Used by: MultiTurnToolUseQualityV1Evaluator
#   Example: RubricMetric.MULTI_TURN_TOOL_USE_QUALITY

# In _SingleTurnVertexAiEvalFacade:
#   dataset = EvaluationDataset(eval_dataset_df=pd.DataFrame([{"prompt":..., "response":...}]))
#   client.evals.evaluate(dataset=dataset, metrics=[PrebuiltMetric.SAFETY])

# In _MultiTurnVertexiAiEvalFacade:
#   dataset = EvaluationDataset(eval_cases=[EvalCase(agent_data=AgentData(...))])
#   client.evals.evaluate(dataset=dataset, metrics=[RubricMetric.MULTI_TURN_TOOL_USE_QUALITY])

print("Single-turn facade: per-invocation pandas DataFrame rows")
print("Multi-turn facade: one EvalCase with full AgentData + ConversationTurns")
```

### Example 3 — offline test with patched `_perform_eval`

```python
from unittest.mock import patch, MagicMock
import os

os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

from google.adk.evaluation.multi_turn_tool_use_quality_evaluator import (
    MultiTurnToolUseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types


mock_result = MagicMock()
mock_result.summary_metrics = [MagicMock(mean_score=0.78)]

with patch(
    "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval",
    return_value=mock_result,
):
    metric = EvalMetric(metric_name="multi_turn_tool_use_quality", threshold=0.6)
    evaluator = MultiTurnToolUseQualityV1Evaluator(eval_metric=metric)

    turns = [
        Invocation(
            invocation_id=f"t{i}",
            user_content=types.Content(parts=[types.Part(text=f"Question {i}")]),
            final_response=types.Content(parts=[types.Part(text=f"Answer {i}")]),
        )
        for i in range(3)
    ]
    result = evaluator.evaluate_invocations(turns)

    # First 2 turns are NOT_EVALUATED; last turn gets the rubric score.
    assert result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
    assert result.per_invocation_results[1].eval_status == EvalStatus.NOT_EVALUATED
    assert result.per_invocation_results[2].score == 0.78
    assert result.overall_eval_status == EvalStatus.PASSED
    print("Multi-turn tool quality offline eval passed:", result.per_invocation_results[2].score)
```

---

## 6 · Session migration pipeline — `MIGRATIONS`, `upgrade()`, `migrate_from_sqlalchemy_pickle.migrate()`

**Modules:** `google.adk.sessions.migration.migration_runner`, `google.adk.sessions.migration.migrate_from_sqlalchemy_pickle`, `google.adk.sessions.migration._schema_check_utils`

ADK's session DB has evolved through two schema versions. The migration system supports upgrading existing databases to the latest V1 JSON schema without data loss.

### Key implementation facts (verified from source)

- **`MIGRATIONS` dict** — maps `start_version → (end_version, migration_fn)`. Currently: `{"0": ("1", migrate_from_sqlalchemy_pickle.migrate)}`. Adding a new migration requires only inserting a new entry; `upgrade()` walks the chain automatically.
- **`LATEST_VERSION = "1"`** (= `SCHEMA_VERSION_1_JSON`) — `upgrade()` stops when the chain reaches this value.
- **`upgrade(source_db_url, dest_db_url, allow_unsafe_unpickling=False)`** — rejects `source == dest` with `RuntimeError`. Detects the source version via `get_db_schema_version(source_db_url)` and early-returns if already current. Multi-step migrations use temporary SQLite files (`tempfile.mkstemp(suffix=".db")`); these are deleted in a `finally` block even on failure.
- **`to_sync_url(db_url)`** — strips async driver prefix (e.g. `sqlite+aiosqlite://` → `sqlite://`). Allows users to pass the same URL they use for async `DatabaseSessionService` to the migration runner.
- **`get_db_schema_version(db_url)`** — opens a sync SQLAlchemy engine, inspects `adk_internal_metadata` for the `schema_version` key (V1), or falls back to column inspection of the `events` table (`"actions"` present and `"event_data"` absent → V0). Returns `LATEST_SCHEMA_VERSION` for new databases.
- **`migrate_from_sqlalchemy_pickle.migrate(source_db_url, dest_db_path)`** — migrates in table order: `app_states` → `user_states` → `sessions` → `events`. Events use `item.to_event()` (deserializes old pickle format) then `event.model_dump_json()`. Failures are logged as warnings; entire transaction is rolled back on error.
- **`allow_unsafe_unpickling`** — the legacy V0 schema stored `EventActions` as Python pickles. Unpickling is unsafe if the source database is untrusted. Pass `allow_unsafe_unpickling=True` only for databases you control.
- **`StorageMetadata` table** — V1 schema stores schema version as a row: `key="schema_version", value="1"` in `adk_internal_metadata`.

### Example 1 — upgrade a SQLite database from V0 to V1

```python
import os
import tempfile
from google.adk.sessions.migration.migration_runner import upgrade

# Suppose you have a legacy V0 database at /data/old_sessions.db
# (created by ADK versions 1.19.0–1.21.0).
# Migrate to a new V1 database:

old_db = "sqlite:///old_sessions.db"
new_db = "sqlite:///new_sessions_v1.db"

# upgrade() raises RuntimeError if old_db == new_db (in-place not supported).
# upgrade() is a no-op if old_db is already at LATEST_VERSION.
# upgrade(
#     source_db_url=old_db,
#     dest_db_url=new_db,
#     allow_unsafe_unpickling=True,   # required for V0 pickle data
# )
# print("Migration complete. Point DatabaseSessionService at:", new_db)

# After migration, update your service:
# from google.adk.sessions import DatabaseSessionService
# service = DatabaseSessionService(db_url=new_db)
print("upgrade() accepts SQLAlchemy URLs including async drivers like sqlite+aiosqlite://")
```

### Example 2 — check schema version before migrating

```python
from google.adk.sessions.migration._schema_check_utils import (
    get_db_schema_version,
    to_sync_url,
    SCHEMA_VERSION_0_PICKLE,
    SCHEMA_VERSION_1_JSON,
    LATEST_SCHEMA_VERSION,
)

print(f"Schema V0 (pickle): {SCHEMA_VERSION_0_PICKLE!r}")   # '0'
print(f"Schema V1 (JSON):   {SCHEMA_VERSION_1_JSON!r}")     # '1'
print(f"Latest version:     {LATEST_SCHEMA_VERSION!r}")      # '1'

# Convert async URL to sync for inspection.
async_url = "sqlite+aiosqlite:///my_sessions.db"
sync_url = to_sync_url(async_url)
print(f"Sync URL: {sync_url}")   # sqlite:///my_sessions.db

# Check version on a real DB (requires the file to exist):
# version = get_db_schema_version("sqlite:///my_sessions.db")
# if version != LATEST_SCHEMA_VERSION:
#     print(f"DB is at version {version!r}, needs migration to {LATEST_SCHEMA_VERSION!r}")
```

### Example 3 — programmatic multi-step migration (future-proof pattern)

```python
from google.adk.sessions.migration import migration_runner

# Inspect the MIGRATIONS chain to understand what steps will run.
print("Migration chain:")
ver = migration_runner.MIGRATIONS
current = "0"
while current in ver:
    end, fn = ver[current]
    print(f"  v{current} --> v{end} via {fn.__module__}.{fn.__name__}")
    current = end
print(f"  Target: v{migration_runner.LATEST_VERSION}")

# Output:
#   Migration chain:
#     v0 --> v1 via google.adk.sessions.migration.migrate_from_sqlalchemy_pickle.migrate
#     Target: v1

# When Google adds v2, MIGRATIONS will gain a "1" -> ("2", ...) entry
# and upgrade() will automatically chain v0->v1->v2 using a temp SQLite DB
# for the intermediate result.
print("\nupgrade() handles multi-step chains transparently via temp SQLite files.")
```

---

## 7 · `AgentBuilderAssistant` — built-in YAML-config agent building assistant

**Module:** `google.adk.cli.built_in_agents.adk_agent_builder_assistant`

`AgentBuilderAssistant` is a factory class that creates a fully configured `LlmAgent` capable of building other ADK agents by writing, reading, and validating YAML configuration files. It ships as a built-in agent that `AgentLoader` can serve under the name `"__adk_agent_builder_assistant"`.

### Key implementation facts (verified from source)

- **`create_agent(model="gemini-2.5-pro") -> LlmAgent`** — the single public entry point. Returns an `LlmAgent` named `"agent_builder_assistant"` with `max_output_tokens=8192`. Note: the `working_directory` parameter is accepted in the signature but is not read inside the function body in v2.3.0; actual working-directory resolution happens at runtime via `resolve_file_path(".", session.state)`.
- **9 custom `FunctionTool`s** — `read_config_files`, `write_config_files`, `explore_project`, `read_files`, `write_files`, `delete_files`, `cleanup_unused_files`, `search_adk_source`, `search_adk_knowledge`. Tool declarations are generated automatically from function signatures and docstrings.
- **2 `AgentTool` sub-agents** — `google_search_agent` and `url_context_agent` are created via `create_google_search_agent()` / `create_url_context_agent()` and wrapped with `AgentTool`. This is necessary because ADK's built-in search/context tools are implemented as sub-agents, not plain functions.
- **Instruction is a callable** — `_load_instruction_with_schema(model)` returns a `Callable[[ReadonlyContext], str]`. At runtime, the callable reads `session.state` to resolve the working directory via `resolve_file_path(".", state)` and interpolates `{project_folder_name}` and `{schema_content}` into the template.
- **Schema embedding** — `_load_schema()` loads `AgentConfig.json` (the ADK YAML schema), extracts 9 core definition names (`LlmAgentConfig`, `LoopAgentConfig`, etc.), prunes `GenerateContentConfig` to 4 fields (`temperature`, `topP`, `topK`, `maxOutputTokens`), and formats it as a compact `\`\`\`text\`\`\`` block injected into the system prompt.
- **`_MultilineDumper` pattern** — the schema is formatted with `textwrap.TextWrapper(width=78)` for readability in the system prompt.
- **`root_agent` module-level singleton** — the module creates `root_agent = AgentBuilderAssistant.create_agent()` so `AgentLoader` can discover and serve the built-in assistant without explicit registration.
- **`_CORE_SCHEMA_DEF_NAMES`** — tuple of 9 schema definition names surfaced in the assistant's context. Keeps the system prompt token-efficient by omitting rarely needed definitions.

### Example 1 — launch the built-in agent builder assistant

```python
import asyncio
from google.adk.cli.built_in_agents.adk_agent_builder_assistant import AgentBuilderAssistant
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def main():
    # working_directory is accepted by the signature but not read in v2.3.0;
    # path resolution happens at call time via resolve_file_path(".", session.state).
    assistant = AgentBuilderAssistant.create_agent(model="gemini-2.5-pro")
    print("Agent name:", assistant.name)             # agent_builder_assistant
    print("Tool count:", len(assistant.tools))       # 11 (9 function + 2 agent)
    print("Max output tokens:", assistant.generate_content_config.max_output_tokens)  # 8192

    session_service = InMemorySessionService()
    runner = Runner(
        agent=assistant,
        app_name="builder",
        session_service=session_service,
    )
    session = await session_service.create_session(app_name="builder", user_id="dev")
    async for event in runner.run_async(
        user_id="dev",
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text="Create a simple LlmAgent config named 'hello_agent'")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text[:200])

asyncio.run(main())
```

### Example 2 — inspect the embedded schema content

```python
from google.adk.cli.built_in_agents.adk_agent_builder_assistant import AgentBuilderAssistant

schema_text = AgentBuilderAssistant._load_schema()
# Returns a ```text ... ``` fenced block containing:
# - "ADK AgentConfig quick reference"
# - Top-level fields: LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
# - AgentRef, ToolConfig, CodeConfig, GenerateContentConfig highlights
# (limited to temperature, topP, topK, maxOutputTokens)

print("Schema preview (first 300 chars):")
print(schema_text[:300])

# The _extract_core_schema() method filters the full JSON Schema to 9 keys:
print("\nCore schema definitions surfaced:")
for name in AgentBuilderAssistant._CORE_SCHEMA_DEF_NAMES:
    print(f"  {name}")
```

### Example 3 — access via `AgentLoader` using the built-in name

```python
# The module-level `root_agent` enables AgentLoader to serve this agent.
# In adk web or adk api_server, use __adk_agent_builder_assistant as the app name.

# Equivalent to:
from google.adk.cli.built_in_agents import adk_agent_builder_assistant as _module
root = _module.root_agent
print("root_agent name:", root.name)          # agent_builder_assistant
print("root_agent model:", root.model)        # gemini-2.5-pro (default)
print("Tool names:", [t.name for t in root.tools])
# ['google_search_agent', 'url_context_agent', 'read_config_files',
#  'write_config_files', 'explore_project', 'read_files', 'write_files',
#  'delete_files', 'cleanup_unused_files', 'search_adk_source', 'search_adk_knowledge']
```

---

## 8 · `load_yaml_file` + `dump_pydantic_to_yaml` + `_MultilineDumper` — YAML I/O utilities

**Module:** `google.adk.utils.yaml_utils`

These three utilities handle YAML serialisation and deserialisation for ADK's agent configuration files. They are used internally by the CLI agent loader and the agent builder assistant.

### Key implementation facts (verified from source)

- **`load_yaml_file(file_path) -> Any`** — accepts `str` or `Path`, converts to `Path`, raises `FileNotFoundError` if the file doesn't exist, and calls `yaml.safe_load()`. Returns whatever the YAML contains (dict, list, scalar).
- **`dump_pydantic_to_yaml(model, file_path, *, indent=2, sort_keys=True, exclude_none=True, exclude_defaults=True, exclude=None)`** — calls `model.model_dump(mode='json')` then writes YAML via `_MultilineDumper`. Creates parent directories with `mkdir(parents=True, exist_ok=True)`.
- **`_MultilineDumper`** — a `yaml.SafeDumper` subclass with two customisations:
  1. `increase_indent(flow=False, indentless=False)` always passes `indentless=False` to force consistent indentation for sequences inside mappings (overrides PyYAML's default flush-left alignment).
  2. `multiline_str_representer` — uses `|` (literal block) style for strings containing `\n`, `"`, or `'`; falls back to plain scalar otherwise.
- **`width=1000000`** — effectively disables PyYAML's automatic line wrapping. ADK configs often contain long instruction strings that must not be split.
- **`allow_unicode=True`** — non-ASCII characters (emoji, multilingual text) are written as-is rather than escaped.
- **`exclude_none=True, exclude_defaults=True`** — keeps YAML files minimal; only explicitly set, non-default fields are written.
- **No `sort_keys=False` default** — keys are sorted alphabetically by default (`sort_keys=True`) for deterministic diffs in version control.

### Example 1 — load and dump an agent config

```python
import tempfile
import os
from pathlib import Path
from pydantic import BaseModel
from google.adk.utils.yaml_utils import load_yaml_file, dump_pydantic_to_yaml


class AgentConfigStub(BaseModel):
    name: str
    instruction: str
    model: str = "gemini-2.0-flash"
    description: str | None = None


config = AgentConfigStub(
    name="summariser",
    instruction="Summarise the user's text.\nBe concise.",
    model="gemini-2.5-pro",
    description=None,           # excluded by exclude_none=True
)

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "agent.yaml"
    dump_pydantic_to_yaml(config, path)

    content = path.read_text()
    print(content)
    # instruction: |
    #   Summarise the user's text.
    #   Be concise.
    # model: gemini-2.5-pro
    # name: summariser
    # (description absent: excluded_none=True; model not default so included)

    loaded = load_yaml_file(path)
    print(type(loaded))  # <class 'dict'>
    print(loaded["name"])  # summariser
```

### Example 2 — multiline string handling with `_MultilineDumper`

```python
import yaml
import io
from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
from pydantic import BaseModel


class Prompt(BaseModel):
    system: str
    few_shot: str


p = Prompt(
    system="You are a helpful assistant.\nAnswer concisely.",
    few_shot='Example: Q: "Hello" A: "Hi"',
)

# dump_pydantic_to_yaml writes system with | block style (contains \n)
# and few_shot with | block style (contains double-quote)
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "prompt.yaml"
    dump_pydantic_to_yaml(p, out, exclude_defaults=False)
    print(out.read_text())
    # few_shot: |
    #   Example: Q: "Hello" A: "Hi"
    # system: |
    #   You are a helpful assistant.
    #   Answer concisely.
```

### Example 3 — `load_yaml_file` error handling

```python
from pathlib import Path
from google.adk.utils.yaml_utils import load_yaml_file


# FileNotFoundError for missing files.
try:
    load_yaml_file("/nonexistent/path/agent.yaml")
except FileNotFoundError as e:
    print(f"Caught: {e}")   # YAML file not found: /nonexistent/path/agent.yaml

# Works fine for valid YAML.
import tempfile, os

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write("name: my_agent\nmodel: gemini-2.5-pro\n")
    tmp_path = f.name

try:
    data = load_yaml_file(tmp_path)
    print(data)   # {'name': 'my_agent', 'model': 'gemini-2.5-pro'}
finally:
    os.unlink(tmp_path)
```

---

## 9 · `StorageSession` + `StorageEvent` + `StorageMetadata` + `DynamicJSON` + `PreciseTimestamp` — V1 SQLAlchemy schema

**Modules:** `google.adk.sessions.schemas.v1`, `google.adk.sessions.schemas.shared`

The V1 schema is the current on-disk representation for `DatabaseSessionService` and `SqliteSessionService`. It replaces the V0 pickle-based `EventActions` column with a single `event_data` JSON column.

### Key implementation facts (verified from source)

- **`DynamicJSON(TypeDecorator)`** — dialect-adaptive JSON column:
  - PostgreSQL → `JSONB` (native; dict passed directly in/out)
  - MySQL → `LONGTEXT` (serialises/deserialises with `json.dumps`/`json.loads`; avoids `StringDataRightTruncationError`)
  - All others (SQLite etc.) → `TEXT` with `json.dumps`/`json.loads`
- **`PreciseTimestamp(TypeDecorator)`** — stores datetimes precisely:
  - MySQL → `DATETIME(fsp=6)` (microsecond resolution)
  - All others → `DateTime` (default SQLAlchemy behaviour)
- **`StorageMetadata`** (`adk_internal_metadata` table) — single-row table; `key="schema_version"` stores `"1"`. Used by `get_db_schema_version()` to detect V1 without inspecting column names.
- **`StorageSession.get_update_marker()`** — returns an ISO 8601 string (microsecond precision, UTC) used as an optimistic concurrency token. Stored in `session._storage_update_marker` after `to_session()`.
- **`StorageSession.to_session()`** — converts the ORM row to an ADK `Session` object. Accepts optional `state` and `events` lists; handles naive datetimes by appending `timezone.utc`.
- **`StorageEvent.from_event(session, event) -> StorageEvent`** — class method that serialises an ADK `Event` to `event_data` via `event.model_dump(exclude_none=True, mode="json")`.
- **`StorageEvent.to_event() -> Event`** — deserialises by calling `Event.model_validate({**self.event_data, "id": self.id, "invocation_id": self.invocation_id, "timestamp": self.timestamp.timestamp()})`. The explicit overrides ensure the primary-key columns always win over any duplicate fields in `event_data`.
- **`StorageSession` cascade** — `relationship("StorageEvent", cascade="all, delete-orphan")` means deleting a session automatically deletes all its events via SQLAlchemy, without needing a manual loop.
- **Index** — `idx_events_app_user_session_ts` on `(app_name, user_id, session_id, timestamp DESC)` — the primary read path for session history.
- **`DEFAULT_MAX_KEY_LENGTH = 128`**, **`DEFAULT_MAX_VARCHAR_LENGTH = 256`** — VARCHAR column limits shared across V1 and V0 schemas.

### Example 1 — create a V1 SQLite database and store a session

```python
import asyncio
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types


async def main():
    # DatabaseSessionService auto-creates V1 schema on first connect.
    service = DatabaseSessionService(db_url="sqlite+aiosqlite:///test_v1.db")

    session = await service.create_session(
        app_name="my_app",
        user_id="alice",
        state={"greeting": "hello"},
    )
    print("Session ID:", session.id)
    print("Update marker:", session._storage_update_marker)  # ISO UTC string

    # Verify schema version in the DB.
    import sqlite3
    conn = sqlite3.connect("test_v1.db")
    row = conn.execute(
        "SELECT value FROM adk_internal_metadata WHERE key='schema_version'"
    ).fetchone()
    print("Schema version:", row[0])  # '1'
    conn.close()

asyncio.run(main())
```

### Example 2 — inspect `DynamicJSON` dialect behaviour

```python
from sqlalchemy import create_engine, text
from google.adk.sessions.schemas.shared import DynamicJSON, DEFAULT_MAX_KEY_LENGTH

# DynamicJSON uses TEXT+JSON for SQLite.
engine = create_engine("sqlite:///:memory:")
from sqlalchemy import Column, String, MetaData, Table
from sqlalchemy.orm import Session

meta = MetaData()
test_table = Table(
    "test_json",
    meta,
    Column("id", String(DEFAULT_MAX_KEY_LENGTH), primary_key=True),
    Column("data", DynamicJSON),
)
meta.create_all(engine)

with engine.begin() as conn:
    conn.execute(test_table.insert().values(id="row1", data={"key": "value", "num": 42}))
    row = conn.execute(text("SELECT data FROM test_json WHERE id='row1'")).fetchone()
    print("Raw SQLite text:", row[0])           # '{"key": "value", "num": 42}'

# DynamicJSON.process_result_value deserialises back to dict.
with Session(engine) as session:
    result = session.execute(test_table.select()).mappings().all()
    print("Deserialized:", result[0]["data"])    # {'key': 'value', 'num': 42}
```

### Example 3 — `StorageEvent` round-trip: `from_event` → `to_event`

```python
from google.adk.sessions.schemas.v1 import StorageEvent
from google.adk.sessions.session import Session
from google.adk.events.event import Event
from google.genai import types
import time

# Build a minimal ADK Event.
event = Event(
    invocation_id="inv-abc",
    author="user",
    content=types.Content(parts=[types.Part(text="Hello!")]),
    timestamp=time.time(),
)

# Fake session for FK fields.
session = Session(app_name="my_app", user_id="u1", id="sess-001")

# Serialise.
storage_event = StorageEvent.from_event(session, event)
print("Stored event_data keys:", list(storage_event.event_data.keys()))
# ['invocation_id', 'author', 'content', 'timestamp', ...]

# Round-trip: deserialise back to ADK Event.
recovered = storage_event.to_event()
print("Recovered author:", recovered.author)         # user
print("Recovered text:", recovered.content.parts[0].text)  # Hello!
print("IDs match:", recovered.id == event.id)        # True
```

---

## 10 · `is_env_enabled` + `is_enterprise_mode_enabled` — runtime environment flag utilities

**Module:** `google.adk.utils.env_utils`

These two small functions govern ADK's feature-flag and mode-switching behaviour at runtime. `is_enterprise_mode_enabled()` in particular implements an important 2.3.0 deprecation: `GOOGLE_GENAI_USE_VERTEXAI` is replaced by `GOOGLE_GENAI_USE_ENTERPRISE`.

### Key implementation facts (verified from source)

- **`is_env_enabled(env_var_name, default='0') -> bool`** — checks `os.environ.get(env_var_name, default).lower() in ['true', '1']`. Case-insensitive; truthy only for `'true'` or `'1'`. Any other value (including `'yes'`, `'on'`, `'True'`) is falsy.
- **`default='0'`** — all ADK feature flags are disabled by default. You must explicitly opt in.
- **`is_enterprise_mode_enabled() -> bool`** — checks `GOOGLE_GENAI_USE_ENTERPRISE` first. If absent, falls back to `GOOGLE_GENAI_USE_VERTEXAI` with a `DeprecationWarning` (stacklevel=2). If neither is set, returns `False`.
- **`GOOGLE_GENAI_USE_VERTEXAI` is deprecated** — the old name implied Vertex AI specifically. The new `GOOGLE_GENAI_USE_ENTERPRISE` reflects broader enterprise feature gating beyond just Vertex AI.
- **Used throughout ADK** — `is_enterprise_mode_enabled()` controls whether ADK routes LLM calls through Vertex AI's enterprise endpoint vs the standard `generativelanguage.googleapis.com` API. It is checked in `GoogleLLMVariant`, model name utilities, and several tools.
- **`GOOGLE_GENAI_USE_ENTERPRISE=1`** or `GOOGLE_GENAI_USE_ENTERPRISE=true` both enable the flag.

### Example 1 — basic feature flag checking pattern

```python
import os
from google.adk.utils.env_utils import is_env_enabled

# Standard ADK feature flag pattern.
os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"
os.environ["PROGRESSIVE_SSE_STREAMING"] = "1"
os.environ["EXPERIMENTAL_FEATURE"] = "false"

print(is_env_enabled("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"))  # True
print(is_env_enabled("PROGRESSIVE_SSE_STREAMING"))             # True
print(is_env_enabled("EXPERIMENTAL_FEATURE"))                  # False
print(is_env_enabled("UNSET_FLAG"))                            # False (default='0')
print(is_env_enabled("UNSET_FLAG", default="1"))               # True  (explicit default)
print(is_env_enabled("UNSET_FLAG", default="true"))            # True

# Case-insensitive:
os.environ["FLAG"] = "TRUE"
print(is_env_enabled("FLAG"))   # True
os.environ["FLAG"] = "True"
print(is_env_enabled("FLAG"))   # True
```

### Example 2 — migration from `GOOGLE_GENAI_USE_VERTEXAI` to `GOOGLE_GENAI_USE_ENTERPRISE`

```python
import os
import warnings
from google.adk.utils.env_utils import is_enterprise_mode_enabled

# ✅ Modern pattern: use GOOGLE_GENAI_USE_ENTERPRISE
os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "1"
print(is_enterprise_mode_enabled())   # True — no warning

del os.environ["GOOGLE_GENAI_USE_ENTERPRISE"]

# ⚠️ Legacy pattern: GOOGLE_GENAI_USE_VERTEXAI triggers a DeprecationWarning
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = is_enterprise_mode_enabled()
    print(result)   # True
    if caught:
        print(f"Warning: {caught[0].category.__name__}: {caught[0].message}")
        # DeprecationWarning: GOOGLE_GENAI_USE_VERTEXAI is deprecated,
        # please use GOOGLE_GENAI_USE_ENTERPRISE instead

del os.environ["GOOGLE_GENAI_USE_VERTEXAI"]
print(is_enterprise_mode_enabled())   # False — no env var set
```

### Example 3 — using `is_enterprise_mode_enabled` to route API calls

```python
import os
from google.adk.utils.env_utils import is_enterprise_mode_enabled


def get_api_endpoint(model: str) -> str:
    """Choose API endpoint based on enterprise mode flag."""
    if is_enterprise_mode_enabled():
        # Route through Vertex AI enterprise endpoint.
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        return (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{location}"
        )
    else:
        # Standard Gemini API endpoint.
        return "https://generativelanguage.googleapis.com/v1beta"


os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "0"
print(get_api_endpoint("gemini-2.5-pro"))
# https://generativelanguage.googleapis.com/v1beta

os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = "acme-prod"
os.environ["GOOGLE_CLOUD_LOCATION"] = "eu-west4"
print(get_api_endpoint("gemini-2.5-pro"))
# https://eu-west4-aiplatform.googleapis.com/v1/projects/acme-prod/locations/eu-west4
```

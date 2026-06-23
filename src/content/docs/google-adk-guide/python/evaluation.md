---
title: "Evaluation"
description: "Test agents programmatically with AgentEvaluator, EvalCase, EvalSet, and the built-in metric suite."
framework: google-adk
language: python
sidebar:
  order: 75
---

Verified against google-adk==2.3.0 (`google/adk/evaluation/`).

ADK ships a first-class evaluation framework built around three concepts: **`EvalCase`** (a single conversation to run), **`EvalSet`** (a collection of cases), and **`AgentEvaluator`** (the engine that runs cases against a live agent and scores the results). The framework integrates with `pytest` and supports custom metrics.

## Minimal example

```python
import asyncio
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase, Invocation, SessionInput
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_config import EvalConfig
from google.genai import types

# Define a single-turn eval case
case = EvalCase(
    eval_id="add_two_numbers",
    conversation=[
        Invocation(
            user_content=types.Content(
                role="user",
                parts=[types.Part(text="What is 15 + 27?")],
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="42")],
            ),
        )
    ],
)

eval_set = EvalSet(
    eval_set_id="arithmetic_suite",
    eval_cases=[case],
)

eval_config = EvalConfig(
    criteria={
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: 0.8,
    }
)

# Run — agent_module must expose `root_agent` or `get_agent_async`
@pytest.mark.asyncio
async def test_arithmetic():
    await AgentEvaluator.evaluate_eval_set(
        agent_module="my_package.agent",
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=1,
    )
```

## `EvalCase`

The atomic unit. Defined in `evaluation/eval_case.py`.

```python
from google.adk.evaluation.eval_case import (
    EvalCase, Invocation, SessionInput, IntermediateData
)
from google.genai import types

case = EvalCase(
    eval_id="weather_lookup",                        # unique within an EvalSet
    session_input=SessionInput(                      # optional initial state
        app_name="weather_app",
        user_id="test_user",
        state={"preferred_units": "metric"},
    ),
    conversation=[
        Invocation(
            user_content=types.Content(
                role="user",
                parts=[types.Part(text="What's the weather in London?")],
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="It's currently 18°C and partly cloudy.")],
            ),
            intermediate_data=IntermediateData(
                tool_uses=[
                    types.FunctionCall(name="get_weather", args={"city": "London"}),
                ],
            ),
        ),
    ],
    final_session_state={"last_city": "London"},     # optional; asserted after the run
)
```

### Multi-turn conversation

```python
case = EvalCase(
    eval_id="two_turn_booking",
    conversation=[
        Invocation(
            user_content=types.Content(
                role="user",
                parts=[types.Part(text="Book a table for 2 at 7pm.")],
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="Which restaurant?")],
            ),
        ),
        Invocation(
            user_content=types.Content(
                role="user",
                parts=[types.Part(text="La Trattoria.")],
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="Done! Table booked at La Trattoria for 2 at 7pm.")],
            ),
            intermediate_data=IntermediateData(
                tool_uses=[
                    types.FunctionCall(
                        name="book_table",
                        args={"restaurant": "La Trattoria", "covers": 2, "time": "19:00"},
                    ),
                ],
            ),
        ),
    ],
)
```

`Invocation` fields:

| Field | Type | Purpose |
|---|---|---|
| `user_content` | `types.Content` | The user message for this turn |
| `final_response` | `types.Content \| None` | Expected final agent response (used by response metrics) |
| `intermediate_data` | `IntermediateData \| None` | Expected tool calls + responses (used by trajectory metrics) |
| `rubrics` | `list[Rubric] \| None` | Per-invocation rubrics (used by `rubric_based_*` metrics) |
| `app_details` | `AppDetails \| None` | Override app name / user id for this invocation |

## `EvalSet`

A collection of `EvalCase` objects. Defined in `evaluation/eval_set.py`.

```python
from google.adk.evaluation.eval_set import EvalSet

eval_set = EvalSet(
    eval_set_id="full_regression",
    name="Full regression suite",
    description="Tests the booking and weather sub-agents.",
    eval_cases=[case],          # replace with your list of EvalCase objects
)

# Serialise to JSON file for reuse
with open("eval_data/full_regression.evalset.json", "w") as f:
    f.write(eval_set.model_dump_json(indent=2))

# Load from JSON file
from google.adk.evaluation.eval_set import EvalSet
with open("eval_data/full_regression.evalset.json") as f:
    eval_set = EvalSet.model_validate_json(f.read())
```

## `EvalConfig` and metrics

`EvalConfig` maps metric names to thresholds or criterion objects. Defined in `evaluation/eval_config.py`.

```python
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import (
    PrebuiltMetrics,
    BaseCriterion,
    LlmAsAJudgeCriterion,
    ToolTrajectoryCriterion,
    JudgeModelOptions,
)

config = EvalConfig(
    criteria={
        # Simple threshold — the metric must score >= value to pass
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: 0.7,

        # LLM-as-judge with custom model and sampling
        PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value: LlmAsAJudgeCriterion(
            threshold=0.8,
            judge_model_options=JudgeModelOptions(
                judge_model="gemini-2.5-pro",
                num_samples=3,
            ),
        ),

        # Tool trajectory with ordered matching
        PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value: ToolTrajectoryCriterion(
            threshold=1.0,
            match_type=ToolTrajectoryCriterion.MatchType.IN_ORDER,
        ),
    }
)
```

## Available prebuilt metrics

All defined in `evaluation/eval_metrics.py` as `PrebuiltMetrics`:

| Metric key | Class | What it measures |
|---|---|---|
| `tool_trajectory_avg_score` | `ToolTrajectoryCriterion` | Whether the agent called the expected tools (EXACT / IN_ORDER / ANY_ORDER) |
| `response_match_score` | `BaseCriterion` | Lexical similarity between actual and expected final response |
| `response_evaluation_score` | `LlmAsAJudgeCriterion` | LLM judge rating of response quality |
| `final_response_match_v2` | `LlmAsAJudgeCriterion` | Semantic match using an LLM judge (v2, more robust) |
| `safety_v1` | `BaseCriterion` | Safety / toxicity score |
| `hallucinations_v1` | `HallucinationsCriterion` | Detects factual hallucinations |
| `rubric_based_final_response_quality_v1` | `RubricsBasedCriterion` | Rubric-scored response quality |
| `rubric_based_tool_use_quality_v1` | `RubricsBasedCriterion` | Rubric-scored tool selection |
| `multi_turn_task_success_v1` | — | Whether a multi-turn task succeeded end-to-end |
| `multi_turn_trajectory_quality_v1` | — | Quality of the full multi-turn trajectory |
| `multi_turn_tool_use_quality_v1` | — | Tool use quality across all turns |

### `ToolTrajectoryCriterion` match types

```python
from google.adk.evaluation.eval_metrics import ToolTrajectoryCriterion

# EXACT — actual calls must match expected calls precisely
ToolTrajectoryCriterion(threshold=1.0, match_type=ToolTrajectoryCriterion.MatchType.EXACT)

# IN_ORDER — expected calls must appear in the actual trajectory in order
# (extra calls are allowed between them)
ToolTrajectoryCriterion(threshold=1.0, match_type=ToolTrajectoryCriterion.MatchType.IN_ORDER)

# ANY_ORDER — all expected calls must appear, order doesn't matter
ToolTrajectoryCriterion(threshold=1.0, match_type=ToolTrajectoryCriterion.MatchType.ANY_ORDER)
```

## `AgentEvaluator`

The engine. All methods are `@staticmethod`. Defined in `evaluation/agent_evaluator.py` (source-verified for google-adk==2.3.0).

### `evaluate_eval_set` — programmatic, in-memory

```python
from google.adk.evaluation.agent_evaluator import AgentEvaluator

await AgentEvaluator.evaluate_eval_set(
    agent_module="my_package.agent",   # must expose root_agent or get_agent_async
    eval_set=eval_set,
    eval_config=eval_config,
    num_runs=2,                        # run each case twice; results are averaged
    agent_name=None,                   # None → root_agent; set to sub-agent name if needed
    print_detailed_results=True,       # print per-metric breakdown to stdout
)
```

`num_runs=2` (the default) runs each case twice and averages the scores, improving reliability for non-deterministic models. Increase to 5 for stability-sensitive metrics.

**How `evaluate_eval_set` works internally (source-verified):**

1. Loads the agent via `_get_agent_for_eval` — imports the module and looks for `get_agent_async` first, then `root_agent`.
2. Creates an `InMemoryEvalSetsManager` and stores the eval set.
3. Runs each `EvalCase` `num_runs` times — calls the agent via a `Runner` for each turn in the conversation.
4. After all runs, scores each metric using the registered `MetricEvaluatorRegistry`.
5. Averages scores across runs with `statistics.mean()`.
6. Asserts each metric against its threshold — raises `AssertionError` if any fails.
7. Optionally prints a tabular report via pandas/tabulate.

### `evaluate` — file-based

```python
await AgentEvaluator.evaluate(
    agent_module="my_package.agent",
    eval_dataset_file_path_or_dir="tests/eval_data/",  # .test.json or directory
    num_runs=2,
    initial_session_file="tests/initial_session.json",
)
```

`eval_dataset_file_path_or_dir` can be:
- A path to a single `.test.json` file (old format) or `.evalset.json` file (new `EvalSet` format).
- A directory — ADK recursively finds all `*.test.json` files. Note: directory scanning uses the old `.test.json` suffix only; pass individual `.evalset.json` paths explicitly.

### Agent module conventions

`AgentEvaluator` loads the module and looks for (in order):
1. `get_agent_async` — an async factory `() -> BaseAgent`. Checked first.
2. `root_agent` — a module-level `BaseAgent` instance.

```python
# my_package/agent.py

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

root_agent = LlmAgent(
    name="research_bot",
    model="gemini-2.5-flash",
    instruction="Answer questions using web search.",
    tools=[google_search],
)
```

Or with factory for dependency injection:

```python
async def get_agent_async():
    # Can connect to real DBs, inject credentials, etc.
    db = await create_db_pool()
    return LlmAgent(
        name="db_agent",
        tools=[make_db_tool(db)],
    )
```

### `find_config_for_test_file` — auto-load eval config

When running file-based evals, `AgentEvaluator` can auto-discover a `test_config.json` file in the same folder:

```python
# Structure:
# tests/eval_data/
#   test_config.json          ← auto-discovered
#   my_suite.evalset.json

# tests/eval_data/test_config.json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.7
  }
}
```

```python
# Load it manually
config = AgentEvaluator.find_config_for_test_file("tests/eval_data/my_suite.evalset.json")
print(config.criteria)   # {"tool_trajectory_avg_score": 1.0, ...}
```

### Full end-to-end example with result capture

```python
import asyncio
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase, Invocation, IntermediateData
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics, ToolTrajectoryCriterion
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from google.genai import types


# --- Build eval cases --------------------------------------------------------
cases = [
    EvalCase(
        eval_id="unit_conversion",
        conversation=[
            Invocation(
                user_content=types.Content(
                    role="user",
                    parts=[types.Part(text="Convert 100 Fahrenheit to Celsius.")],
                ),
                final_response=types.Content(
                    role="model",
                    parts=[types.Part(text="37.78")],
                ),
                intermediate_data=IntermediateData(
                    tool_uses=[
                        types.FunctionCall(
                            name="convert_temperature",
                            args={"value": 100, "from_unit": "F", "to_unit": "C"},
                        )
                    ]
                ),
            )
        ],
    ),
    EvalCase(
        eval_id="multi_turn_booking",
        conversation=[
            Invocation(
                user_content=types.Content(
                    role="user",
                    parts=[types.Part(text="Book a flight to Paris.")],
                ),
                final_response=types.Content(
                    role="model",
                    parts=[types.Part(text="Which date would you like to travel?")],
                ),
            ),
            Invocation(
                user_content=types.Content(
                    role="user",
                    parts=[types.Part(text="June 15th.")],
                ),
                final_response=types.Content(
                    role="model",
                    parts=[types.Part(text="Done! Flight booked for June 15th to Paris.")],
                ),
                intermediate_data=IntermediateData(
                    tool_uses=[
                        types.FunctionCall(
                            name="book_flight",
                            args={"destination": "Paris", "date": "2026-06-15"},
                        )
                    ]
                ),
            ),
        ],
    ),
]

eval_set = EvalSet(eval_set_id="travel_agent_suite", eval_cases=cases)

eval_config = EvalConfig(
    criteria={
        PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value: ToolTrajectoryCriterion(
            threshold=1.0,
            match_type=ToolTrajectoryCriterion.MatchType.IN_ORDER,
        ),
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: 0.5,
    }
)

# --- Run evaluation ----------------------------------------------------------
async def main():
    await AgentEvaluator.evaluate_eval_set(
        agent_module="my_package.travel_agent",
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=2,
        print_detailed_results=True,
    )

asyncio.run(main())
```

## Saving eval results

```python
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)

results_manager = LocalEvalSetResultsManager(results_dir="./eval_results")

# After evaluate_eval_set completes, save results
result = await AgentEvaluator.evaluate_eval_set(...)
await results_manager.save_eval_set_result(result)
```

Or use `GcsEvalSetResultsManager` to persist to Cloud Storage:

```python
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager

results_manager = GcsEvalSetResultsManager(
    bucket_name="my-eval-results",
    eval_storage_dir="runs/",
)
```

## Custom metrics

```python
from google.adk.evaluation.eval_config import EvalConfig, CustomMetricConfig
from google.adk.agents.common_configs import CodeConfig

# Implement the metric function in a discoverable module
# my_package/metrics.py
def my_length_metric(
    actual_invocation,
    expected_invocation,
    criterion,
) -> float:
    """Returns 1.0 if the response is ≤ 100 chars, else 0.0."""
    if not actual_invocation.final_response:
        return 0.0
    text = "".join(
        p.text or ""
        for p in actual_invocation.final_response.parts or []
    )
    return 1.0 if len(text) <= 100 else 0.0


config = EvalConfig(
    criteria={
        "response_brevity": 1.0,                     # threshold to pass
    },
    custom_metrics={
        "response_brevity": CustomMetricConfig(
            code_config=CodeConfig(name="my_package.metrics.my_length_metric"),
        ),
    },
)
```

## Rubric-based evaluation

Rubrics let you score responses against structured criteria instead of a single binary pass/fail:

```python
from google.adk.evaluation.eval_rubrics import Rubric, RubricScore
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion

rubrics = [
    Rubric(
        criterion="The response must cite at least one source URL.",
        points=1,
    ),
    Rubric(
        criterion="The response must be written in plain English, no jargon.",
        points=1,
    ),
    Rubric(
        criterion="The response must be under 200 words.",
        points=1,
    ),
]

config = EvalConfig(
    criteria={
        "rubric_based_final_response_quality_v1": RubricsBasedCriterion(
            threshold=0.8,          # fraction of total rubric points required
            rubrics=rubrics,
        ),
    }
)
```

## pytest integration

```python
# tests/test_agent.py

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

EVAL_SET = EvalSet.model_validate_json(
    open("tests/eval_data/regression.evalset.json").read()
)

EVAL_CONFIG = EvalConfig(
    criteria={
        PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value: 1.0,
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: 0.7,
    }
)

@pytest.mark.asyncio
async def test_agent_regression():
    await AgentEvaluator.evaluate_eval_set(
        agent_module="my_package.agent",
        eval_set=EVAL_SET,
        eval_config=EVAL_CONFIG,
        num_runs=2,
    )
```

Run with:

```bash
pytest tests/test_agent.py -v
```

## File-based eval format

For tooling compatibility, save eval cases as JSON. The recommended format (new schema) is an `EvalSet` JSON:

```json
{
  "evalSetId": "arithmetic_suite",
  "evalCases": [
    {
      "evalId": "add_two_numbers",
      "conversation": [
        {
          "userContent": {
            "role": "user",
            "parts": [{ "text": "What is 15 + 27?" }]
          },
          "finalResponse": {
            "role": "model",
            "parts": [{ "text": "42" }]
          },
          "intermediateData": {
            "toolUses": []
          }
        }
      ]
    }
  ]
}
```

Save as `tests/eval_data/arithmetic_suite.evalset.json`. The old `.test.json` format is still accepted but will emit a migration warning — use `AgentEvaluator.migrate_eval_data_to_new_schema()` to convert.

## Migrate old eval data

```python
AgentEvaluator.migrate_eval_data_to_new_schema(
    old_eval_data_file="tests/eval_data/old_tests.test.json",
    new_eval_data_file="tests/eval_data/old_tests.evalset.json",
    initial_session_file="tests/initial_session.json",
)
```

## Patterns

### 1 — CI gate on tool trajectory
Record expected tool calls from a golden run. In CI, `ToolTrajectoryCriterion(match_type=IN_ORDER, threshold=1.0)` fails the build if the agent forgets a required tool or calls them out of order.

### 2 — LLM judge for quality
Use `RESPONSE_EVALUATION_SCORE` or `FINAL_RESPONSE_MATCH_V2` with `num_samples=5` to get stable scores. Reserve expensive judge metrics for nightly runs; use `RESPONSE_MATCH_SCORE` (lexical) in fast PR checks.

### 3 — Rubric tiers
Three rubrics worth 1 point each. Threshold at `0.67` (≥ 2/3 criteria). Run with `num_runs=3` to smooth out judge variance.

### 4 — Per-agent sub-eval
Set `agent_name="specialist_bot"` on `evaluate_eval_set` to evaluate a sub-agent in isolation, bypassing the root agent's routing.

### 5 — End-to-end state assertion
Populate `final_session_state={"order_confirmed": True}` in the `EvalCase`. ADK asserts the session state matches after the conversation completes. Combine with tool trajectory to verify both the path and the outcome.

## Gotchas

- `agent_module` must be an **importable dotted path** (e.g. `"my_package.agent"`), not a file path. The module must be on `sys.path`.
- `num_runs=1` can produce flaky results for non-deterministic models. Use `num_runs=2` (the default) or higher for metrics that use LLM judges.
- The `criteria` dict key must exactly match the `PrebuiltMetrics.value` string (e.g. `"tool_trajectory_avg_score"`) or a custom metric name registered in `custom_metrics`.
- `RESPONSE_EVALUATION_SCORE` is inherently unstable — the docstring in source says "this evaluation is not very stable". Treat it as a soft signal, not a hard gate.
- Old `.test.json` files are accepted but emit a deprecation warning. Migrate to `EvalSet` JSON to suppress the warning.
- `SessionInput.state` sets the **initial** session state before the first turn. Mutations during the conversation are not reflected back to `session_input`.

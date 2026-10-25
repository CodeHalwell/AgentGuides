---
title: "Evaluation"
description: "Test agents programmatically with AgentEvaluator, EvalCase, EvalSet, and the built-in metric suite."
framework: "google-adk"
language: "python"
sidebar:
  label: "Evaluation"
  order: 78
---

Verified against `google-adk` %%version=google-adk%% (`google/adk/evaluation/`), checked %%versions-checked%%.

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

## From the command line

`adk eval` runs eval set files against an agent folder (one whose `agent.py` defines `root_agent`) and prints a pass/fail summary per set. Results are also written under the agent's `.adk/eval_history/`. From 2.13.0 the command exits with status 1 when any case fails (or when nothing ran) and 0 when all pass, so a CI step can rely on it; earlier releases always exited 0.

```bash
pip install "google-adk[eval]"

# Pass thresholds per metric
echo '{"criteria": {"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}}' > my_agent/test_config.json

adk eval my_agent my_agent/tests/arithmetic_suite.evalset.json --config_file_path my_agent/test_config.json
adk eval my_agent my_agent/tests/arithmetic_suite.evalset.json:add_two_numbers   # one case only
adk eval my_agent my_agent/tests/*.evalset.json --print_detailed_results
```

`adk eval_set create my_agent regression` creates an empty `regression.evalset.json` in the agent folder, and `adk eval_set add_eval_case` / `generate_eval_cases` add cases from conversation scenarios. In the Dev UI (`adk web`), the Eval tab saves a live session as an eval case and runs sets interactively. Without `--config_file_path`, and with a single eval set file, ADK looks for a `test_config.json` next to that file; otherwise it uses its default criteria.

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
                judge_model="%%model=gemini-flagship%%",
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
| `safety_v1` | `BaseCriterion` | Safety (harmlessness) score; uses the Vertex Gen AI Eval SDK, so needs `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` |
| `hallucinations_v1` | `HallucinationsCriterion` | Detects factual hallucinations |
| `rubric_based_final_response_quality_v1` | `RubricsBasedCriterion` | Rubric-scored response quality |
| `rubric_based_tool_use_quality_v1` | `RubricsBasedCriterion` | Rubric-scored tool selection |
| `multi_turn_task_success_v1` | — | Whether a multi-turn task succeeded end-to-end |
| `multi_turn_trajectory_quality_v1` | — | Quality of the full multi-turn trajectory |
| `multi_turn_tool_use_quality_v1` | — | Tool use quality across all turns |
| `rubric_based_multi_turn_trajectory_quality_v1` | `RubricsBasedCriterion` | The multi-turn trajectory scored against your rubrics by an LLM judge |
| `per_turn_user_simulator_quality_v1` | `LlmBackedUserSimulatorCriterion` | Whether each simulated user message follows the conversation scenario |
| `tool_call_count_v1` | — | Informational: tool calls per invocation, averaged over the case |
| `inference_call_count_v1` | — | Informational: model calls per invocation |
| `token_usage_v1` | — | Informational: tokens per invocation, with an input/output breakdown |
| `invocation_duration_v1` | — | Informational: wall-clock seconds per invocation |

The four efficiency metrics at the end are added to every eval automatically and appear in results and CSV output with the status `INFORMATIONAL`; they never pass or fail. Leave them out of `criteria`: naming one there gives it a threshold, which these metrics reject, so the metric errors and the run fails with "was not evaluated".

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

The engine. All methods are `@staticmethod`. Defined in `evaluation/agent_evaluator.py` (re-checked against %%version=google-adk%%).

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

1. Loads the agent via `_get_agent_for_eval`: imports the module (its name must end in `.agent`, or it must have an `agent` attribute), uses `root_agent` if present and otherwise awaits `get_agent_async()`. A module-level `app` that is an `App` is used too, so its plugins run during the eval.
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
    model="%%model=gemini%%",
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
        agent_module="travel_bench.agent",
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=2,
        print_detailed_results=True,
    )

asyncio.run(main())
```

### `evaluate_eval_set` — CSV output, preloaded artifacts, results manager

Source-verified against `google.adk.evaluation.agent_evaluator.evaluate_eval_set` (first verified in 2.7.1; re-checked against %%version=google-adk%%). Three params are worth calling out because they're often ignored and each solves a real reporting problem.

```python
import asyncio
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

# ── artifact_service: preload files that an eval case will reference ─────────
artifact_service = InMemoryArtifactService()

# ── eval_set_results_manager: persist evalset_result.json to disk ────────────
results_manager: EvalSetResultsManager = LocalEvalSetResultsManager(
    agents_dir="./eval_results"
)

async def preload_artifacts() -> None:
    # Preload one artifact per eval-case session_id — the case must set
    # SessionInput.session_id to the same value for the load to hit.
    await artifact_service.save_artifact(
        app_name="travel_bench",
        user_id="eval",
        session_id="case-booking-01",
        filename="itinerary_template.txt",
        artifact=types.Part.from_text(text="Departure: {city}\nDate: {date}\n"),
    )

async def main() -> None:
    await preload_artifacts()
    await AgentEvaluator.evaluate_eval_set(
        agent_module="travel_bench.agent",
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=3,
        print_detailed_results=True,

        # Per-invocation results (metric name, value, pass/fail) written as CSV.
        # Parent directory is created automatically.
        output_file="./eval_results/travel_bench_20260824.csv",

        # Preloaded artifacts reach eval cases whose SessionInput.session_id matches
        # the id you used to save_artifact above.
        artifact_service=artifact_service,

        # Required whenever eval_set_results_manager is provided; otherwise a
        # ValueError is raised at the top of evaluate_eval_set.
        app_name="travel_bench",
        eval_set_results_manager=results_manager,
    )

asyncio.run(main())
```

**What ends up on disk:**

- `./eval_results/travel_bench_20260824.csv` — one row per (eval case, metric, invocation), with the columns `eval_set_id, eval_id, metric_name, threshold, score, eval_status, prompt, expected_response, actual_response, expected_tool_calls, actual_tool_calls`. The file is appended to if it already exists (the header is written once), and writing it needs pandas (`pip install "google-adk[eval]"`).
- `./eval_results/travel_bench/.adk/eval_history/<result_id>.evalset_result.json` — the persisted `EvalSetResult` written by `LocalEvalSetResultsManager`. The path is built as `<agents_dir>/<app_name>/.adk/eval_history/<result_id>.evalset_result.json` (see `_get_eval_history_dir` in `local_eval_set_results_manager.py`). This is what the ADK Web UI reads back when browsing historical runs.

**Two hard rules from the source (verified in the docstring & body):**

- `eval_config` is required — passing `None` raises `ValueError("`eval_config` is required.")`.
- If you set `eval_set_results_manager` but leave `app_name` `None`, the same top-of-function check raises `ValueError("app_name is required when eval_set_results_manager is provided.")`.

**Legacy `criteria=` shortcut.** Still accepted, still deprecated. Verified: `AgentEvaluator` auto-maps `criteria={metric: threshold}` → `EvalConfig(criteria={metric: BaseCriterion(threshold=threshold)})` and logs a deprecation message through the `google_adk` logger (`logger.warning`, not a Python `DeprecationWarning`). Migrate anything you own by constructing `EvalConfig` explicitly.

## Saving eval results

`evaluate_eval_set` returns `None`; to keep results, hand it a results manager and an `app_name`. `LocalEvalSetResultsManager` writes each run to `<agents_dir>/<app_name>/.adk/eval_history/`:

```python
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)

results_manager = LocalEvalSetResultsManager(agents_dir="./agents")

await AgentEvaluator.evaluate_eval_set(
    agent_module="my_agent.agent",
    eval_set=eval_set,
    eval_config=eval_config,
    app_name="my_agent",
    eval_set_results_manager=results_manager,
)
```

Or use `GcsEvalSetResultsManager` to persist to Cloud Storage:

```python
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager

results_manager = GcsEvalSetResultsManager(
    bucket_name="my-eval-results",
    project="my-gcp-project",   # extra kwargs go to google.cloud.storage.Client
)
```

## Custom metrics

A custom metric is a function that ADK imports by dotted path. It is called with the `EvalMetric`, the actual invocations, the expected invocations (or `None`) and the `ConversationScenario` (or `None`), and returns an `EvaluationResult`; it may be `async`. ADK clears the threshold on the copy of `EvalMetric` it passes in, so the function sets the pass/fail status itself, and the threshold in `criteria` is then checked against `overall_score`.

```python
# my_package/metrics.py
from typing import Optional

from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult

MAX_CHARS = 100


def response_brevity(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario],
) -> EvaluationResult:
    """Scores 1.0 for each final response of at most MAX_CHARS characters."""
    per_invocation = []
    for actual in actual_invocations:
        parts = actual.final_response.parts if actual.final_response else []
        text = "".join(p.text or "" for p in parts or [])
        score = 1.0 if len(text) <= MAX_CHARS else 0.0
        per_invocation.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=score,
                eval_status=EvalStatus.PASSED if score == 1.0 else EvalStatus.FAILED,
            )
        )
    overall = sum(r.score for r in per_invocation) / max(len(per_invocation), 1)
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=EvalStatus.PASSED if overall == 1.0 else EvalStatus.FAILED,
        per_invocation_results=per_invocation,
    )
```

Register it under a name in `custom_metrics` and give that name a threshold in `criteria`:

```python
from google.adk.agents.common_configs import CodeConfig
from google.adk.evaluation.eval_config import CustomMetricConfig, EvalConfig

config = EvalConfig(
    criteria={"response_brevity": 1.0},
    custom_metrics={
        "response_brevity": CustomMetricConfig(
            code_config=CodeConfig(name="my_package.metrics.response_brevity"),
            description="Final response is at most 100 characters.",
        ),
    },
)
```

With a stub model replying `42`, this metric passes; a 150-character reply fails with `response_brevity for my_package.agent Failed. Expected 1.0, but got 0.0.`

## Rubric-based evaluation

Rubrics let you score responses against structured criteria instead of a single binary pass/fail:

```python
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion

def rubric(rubric_id: str, text: str) -> Rubric:
    return Rubric(rubric_id=rubric_id, rubric_content=RubricContent(text_property=text))

rubrics = [
    rubric("cites_source", "The response cites at least one source URL."),
    rubric("plain_english", "The response is written in plain English, without jargon."),
    rubric("concise", "The response is under 200 words."),
]

config = EvalConfig(
    criteria={
        "rubric_based_final_response_quality_v1": RubricsBasedCriterion(
            threshold=0.8,          # minimum mean rubric score to pass
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

For tooling compatibility, save eval cases as JSON: an `EvalSet` serialised with snake_case field names (`EvalSet.model_dump_json()` produces this; camelCase keys such as `evalSetId` fail validation):

```json
{
  "eval_set_id": "arithmetic_suite",
  "eval_cases": [
    {
      "eval_id": "add_two_numbers",
      "conversation": [
        {
          "user_content": {
            "role": "user",
            "parts": [{ "text": "What is 15 + 27?" }]
          },
          "final_response": {
            "role": "model",
            "parts": [{ "text": "42" }]
          },
          "intermediate_data": {
            "tool_uses": []
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

## Multi-turn evaluators

Three Vertex AI–backed evaluators score entire conversations holistically, rather than scoring individual turn responses. They delegate to the Vertex Gen AI Eval SDK, so they need a Google Cloud project: set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. An AI Studio `GOOGLE_API_KEY` is not enough.

| Class | `PrebuiltMetrics` key | What it scores |
|---|---|---|
| `MultiTurnTaskSuccessV1Evaluator` | `multi_turn_task_success_v1` | Did the agent achieve the user's goal by the end of the conversation? |
| `MultiTurnToolUseQualityV1Evaluator` | `multi_turn_tool_use_quality_v1` | Did the agent call the right tools across all turns? |
| `MultiTurnTrajectoryQualityV1Evaluator` | `multi_turn_trajectory_quality_v1` | Was the path the agent took to reach the goal reasonable? |

All three are **reference-free** — they do not need `final_response` or `intermediate_data` in your `EvalCase`; the Vertex AI rubric model judges based on the conversation transcript and the `ConversationScenario` context you supply.

### `ConversationScenario`

`ConversationScenario` describes the task the agent was meant to solve. The multi-turn evaluators use it to judge whether the agent succeeded. It lives in `google.adk.evaluation.conversation_scenarios` with two required fields, `starting_prompt` (the first user message) and `conversation_plan` (the evaluator's guide to what a successful conversation looks like), and an optional `user_persona` used by the simulated user.

```python
from google.adk.evaluation.conversation_scenarios import ConversationScenario

scenario = ConversationScenario(
    starting_prompt="Book a restaurant table for 2 people at 7pm on Friday.",
    conversation_plan=(
        "User wants a table for 2 at 7pm Friday. "
        "A successful conversation ends with the agent confirming the booking details."
    ),
)
```

Attach it to an `EvalCase` using `conversation_scenario`. **Important:** `EvalCase` enforces an XOR constraint — you must set exactly one of `conversation` (static turns) or `conversation_scenario` (scenario-driven). Setting both, or neither, fails validation (a pydantic `ValidationError`, which subclasses `ValueError`).

```python
from google.adk.evaluation.eval_case import EvalCase

case = EvalCase(
    eval_id="restaurant_booking",
    conversation_scenario=scenario,   # ← scenario only; no conversation= here
)
```

### Full multi-turn eval example

```python
import asyncio
import os
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

# Required for the Vertex AI evaluators
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# ConversationScenario uses starting_prompt + conversation_plan (source-verified)
scenario = ConversationScenario(
    starting_prompt="Find me a flight from London to Tokyo for August 10th.",
    conversation_plan=(
        "User wants to book a direct flight from London to Tokyo on August 10th. "
        "They will pick an option from the agent's search results and confirm the booking. "
        "A successful conversation ends with a booking confirmation."
    ),
)

# EvalCase enforces XOR: set conversation_scenario OR conversation, never both
case = EvalCase(
    eval_id="flight_booking_multi_turn",
    conversation_scenario=scenario,
)

eval_set = EvalSet(eval_set_id="flight_booking_suite", eval_cases=[case])

eval_config = EvalConfig(
    criteria={
        # Reference-free multi-turn metrics — no expected responses needed
        PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value: 0.7,
        PrebuiltMetrics.MULTI_TURN_TOOL_USE_QUALITY_V1.value: 0.7,
        PrebuiltMetrics.MULTI_TURN_TRAJECTORY_QUALITY_V1.value: 0.7,
    }
)


async def main():
    await AgentEvaluator.evaluate_eval_set(
        agent_module="flight_bench.agent",
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=1,                 # multi-turn rubric metrics are expensive; 1 run is typical
        print_detailed_results=True,
    )


asyncio.run(main())
```

### Configuring the simulated user

A scenario-driven case is played out by a simulated user. Configure it with `EvalConfig(user_simulator_config=LlmBackedUserSimulatorConfig(...))` from `google.adk.evaluation.simulation.llm_backed_user_simulator`:

| Field | Default | Purpose |
|---|---|---|
| `model` | `"gemini-2.5-flash"` | LLM that plays the user. The default is a Gemini 2.5 model, which new Gemini API projects cannot use; set `model="%%model=gemini%%"` or another model you have access to |
| `model_configuration` | default `GenerateContentConfig` | Generation config for that LLM |
| `max_allowed_invocations` | `20` | Turn cap before the simulation stops |
| `include_function_calls` | `False` | Show the agent's tool calls to the simulated user |
| `custom_instructions` | `None` | Jinja2 template replacing the default simulator prompt |

`ConversationScenario.user_persona` takes a `UserPersona` (or a registered persona name) to give the simulated user a style, with rubrics the judge checks it against:

```python
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from google.adk.evaluation.simulation.user_simulator_personas import UserBehavior, UserPersona

impatient = UserPersona(
    id="impatient_traveller",
    description="A frequent flyer who is time-pressed and expects quick answers.",
    behaviors=[UserBehavior(
        name="brief_responses",
        description="Gives very short answers.",
        behavior_instructions=["Keep your messages under 15 words."],
        violation_rubrics=["Response is longer than 15 words."],
    )],
)
scenario = ConversationScenario(
    starting_prompt="Book me a flight to Paris.",
    conversation_plan="Book a one-way flight LHR→CDG on 2025-05-01. Say yes to all prompts.",
    user_persona=impatient,
)
eval_config = EvalConfig(
    criteria={PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value: 0.7},
    user_simulator_config=LlmBackedUserSimulatorConfig(max_allowed_invocations=15),
)
```

`AgentEvaluator` imports `agent_module` and reads its module-level `root_agent`, so keep the driver (with `asyncio.run`) in a separate script from the agent module.

### Choosing between multi-turn and per-turn metrics

| Situation | Recommended approach |
|---|---|
| Fast CI checks — did the agent call the right tools? | `TOOL_TRAJECTORY_AVG_SCORE` (per-turn, cheap, reference-based) |
| Overnight quality gate — did the conversation end well? | `MULTI_TURN_TASK_SUCCESS_V1` (holistic, reference-free, Vertex AI) |
| Investigating agent reasoning path | `MULTI_TURN_TRAJECTORY_QUALITY_V1` (judges whether the path made sense) |
| Tool selection across a dialogue | `MULTI_TURN_TOOL_USE_QUALITY_V1` (rubric over entire tool-call log) |

## `ConversationGenerationConfig` and `ScenarioGenerator`

Instead of writing `EvalCase` objects by hand, you can ask a Vertex AI model to generate them. This is **synchronous** (not async). It uses Vertex AI, so it requires `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. The correct module path is `google.adk.evaluation._vertex_ai_scenario_generation_facade`.

`ConversationGenerationConfig` lives in `google.adk.evaluation.conversation_scenarios` with fields: `count` (required), `model_name` (required), `generation_instruction` (optional), `environment_context` (optional).

```python
import os
from google.adk.agents import LlmAgent
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig
from google.adk.evaluation._vertex_ai_scenario_generation_facade import ScenarioGenerator

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

travel_agent = LlmAgent(
    name="travel_agent",
    model="%%model=gemini%%",
    description="Books flights and checks weather.",
    tools=[],  # your actual tools here
)

gen_config = ConversationGenerationConfig(
    count=3,                          # number of scenarios to generate
    model_name="%%model=gemini%%",
    environment_context=(
        "Available flights: LHR→NRT on 2026-08-10 at 09:00 (ANA, £620), 14:00 (BA, £580)."
    ),
)

generator = ScenarioGenerator()

# generate_scenarios is synchronous — no await needed
scenarios = generator.generate_scenarios(agent=travel_agent, config=gen_config)

for s in scenarios:
    print(s.starting_prompt)
    print(s.conversation_plan)
```

Combine with `EvalCase` to build a full synthetic eval set:

```python
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet

cases = [
    EvalCase(
        eval_id=f"generated_{i}",
        conversation_scenario=s,   # XOR constraint: no conversation= when using scenario
    )
    for i, s in enumerate(scenarios)
]
eval_set = EvalSet(eval_set_id="synthetic_travel_suite", eval_cases=cases)
```

## `AppDetails` and `AgentDetails`

`AppDetails` and `AgentDetails` capture a lightweight snapshot of the agent tree at eval time. Both live in `google.adk.evaluation.app_details`. `AppDetails` has a single field: `agent_details: dict[str, AgentDetails]`. `AgentDetails` has: `name: str`, `instructions: str`, `tool_declarations: list`.

```python
from google.adk.evaluation.app_details import AppDetails, AgentDetails

# Typically you don't construct these manually — AgentEvaluator populates them
# automatically from the running agent tree. But you can supply them explicitly
# when replaying recorded conversations for offline eval.
snapshot = AppDetails(
    agent_details={
        "travel_assistant": AgentDetails(
            name="travel_assistant",
            instructions="You are a travel booking assistant.",
            tool_declarations=[],   # list of tool FunctionDeclarations if known
        ),
        "flight_sub_agent": AgentDetails(
            name="flight_sub_agent",
            instructions="Search and book flights.",
            tool_declarations=[],
        ),
    }
)
```

## Gotchas

- `agent_module` must be an **importable dotted path** (e.g. `"my_package.agent"`), not a file path, and must be on `sys.path`. Its name must end in `.agent`, or the module must have an `agent` attribute; otherwise loading fails with a `ValueError` saying the module has no member named `agent`.
- `num_runs=1` can produce flaky results for non-deterministic models. Use `num_runs=2` (the default) or higher for metrics that use LLM judges.
- The `criteria` dict key must exactly match the `PrebuiltMetrics.value` string (e.g. `"tool_trajectory_avg_score"`) or a custom metric name registered in `custom_metrics`.
- `RESPONSE_EVALUATION_SCORE` is inherently unstable — the docstring in source says "this evaluation is not very stable". Treat it as a soft signal, not a hard gate.
- Old `.test.json` files are accepted but emit a deprecation warning. Migrate to `EvalSet` JSON to suppress the warning.
- `SessionInput.state` sets the **initial** session state before the first turn. Mutations during the conversation are not reflected back to `session_input`.
- Multi-turn Vertex AI evaluators (`multi_turn_task_success_v1`, etc.) incur Vertex AI API calls per evaluation run. Keep the results with `LocalEvalSetResultsManager` so you can compare runs, and avoid running them on every PR commit.
- `ScenarioGenerator.generate_scenarios()` is **synchronous** (source-verified: plain `def`, not `async def`). It takes `agent` and `config` as positional/keyword arguments. The correct import path is `google.adk.evaluation._vertex_ai_scenario_generation_facade`.
- `ConversationScenario` fields are `starting_prompt` and `conversation_plan` (from `google.adk.evaluation.conversation_scenarios`), not `scenario_description`/`task_description`.
- `EvalCase` enforces `conversation` XOR `conversation_scenario`: passing both, or neither, fails validation at construction time.

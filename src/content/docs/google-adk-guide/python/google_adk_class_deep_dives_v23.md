---
title: "Class deep dives — volume 23 (GCS/local eval storage, metric registry, rubric evaluators, per-turn simulator quality, workflow replay & progressive SSE streaming)"
description: "Source-verified 2.3.0 deep dives: GcsEvalSetsManager/GcsEvalSetResultsManager (GCS-backed eval set persistence), LocalEvalSetsManager/LocalEvalSetResultsManager/convert_eval_set_to_pydantic_schema (disk-backed eval sets with legacy JSON migration), MetricEvaluatorRegistry/DEFAULT_METRIC_EVALUATOR_REGISTRY (pluggable metric-to-evaluator registry), MetricInfoProvider subclasses/MetricInfo/MetricValueInfo/Interval/PrebuiltMetrics (metric catalogue layer), RubricBasedFinalResponseQualityV1Evaluator (evidence-first final answer rubric evaluation), RubricBasedToolUseV1Evaluator (tool-use rubric evaluation), RubricBasedMultiTurnTrajectoryEvaluator (holistic multi-turn trajectory rubric evaluation), PerTurnUserSimulatorQualityV1 (per-turn user simulator quality metric with majority-vote LLM sampling), InterceptionResult/check_interception/create_mock_context (5-case workflow replay decision engine), StreamingResponseAggregator 2.3.0 deep-dive (progressive SSE with JSONPath partial-args and dual-buffer flush model)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 23"
  order: 92
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, constant, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `GcsEvalSetsManager` + `GcsEvalSetResultsManager` | `google.adk.evaluation.gcs_eval_sets_manager` + `.gcs_eval_set_results_manager` | Stable |
| 2 | `LocalEvalSetsManager` + `LocalEvalSetResultsManager` + `convert_eval_set_to_pydantic_schema` | `google.adk.evaluation.local_eval_sets_manager` + `.local_eval_set_results_manager` | Stable |
| 3 | `MetricEvaluatorRegistry` + `DEFAULT_METRIC_EVALUATOR_REGISTRY` | `google.adk.evaluation.metric_evaluator_registry` | `@experimental` |
| 4 | `MetricInfoProvider` subclasses + `MetricInfo` + `MetricValueInfo` + `Interval` + `PrebuiltMetrics` | `google.adk.evaluation.metric_info_providers` + `.eval_metrics` | `@experimental` |
| 5 | `RubricBasedFinalResponseQualityV1Evaluator` | `google.adk.evaluation.rubric_based_final_response_quality_v1` | `@experimental` |
| 6 | `RubricBasedToolUseV1Evaluator` | `google.adk.evaluation.rubric_based_tool_use_quality_v1` | `@experimental` |
| 7 | `RubricBasedMultiTurnTrajectoryEvaluator` | `google.adk.evaluation.rubric_based_multi_turn_trajectory_evaluator` | Stable (internal) |
| 8 | `PerTurnUserSimulatorQualityV1` | `google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1` | `@experimental` |
| 9 | `InterceptionResult` + `check_interception` + `create_mock_context` | `google.adk.workflow.utils._replay_interceptor` | Stable (internal) |
| 10 | `StreamingResponseAggregator` (2.3.0 deep-dive) | `google.adk.utils.streaming_utils` | Stable |

---

## 1 · `GcsEvalSetsManager` + `GcsEvalSetResultsManager` — GCS-backed eval persistence

**Sources:** `google/adk/evaluation/gcs_eval_sets_manager.py` · `google/adk/evaluation/gcs_eval_set_results_manager.py`

Both classes extend abstract base classes (`EvalSetsManager` / `EvalSetResultsManager`) and store all data as JSON blobs in a single Google Cloud Storage bucket. The object key conventions are fixed:

| Object | GCS path |
|--------|----------|
| Eval set | `{app_name}/evals/eval_sets/{eval_set_id}.evalset.json` |
| Eval set result | `{app_name}/evals/eval_history/{eval_set_result_id}.evalset_result.json` |

**`GcsEvalSetsManager` key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Bucket validation | `bucket.exists()` checked in `__init__`; raises `ValueError` immediately if missing — no lazy check |
| ID validation | `_validate_id()` enforces `^[a-zA-Z0-9_]+$`; any other character raises `ValueError` before touching GCS |
| Serialisation | `model_dump_json(indent=2, exclude_unset=True, exclude_defaults=True, exclude_none=True)` — compact, non-null-padded JSON |
| Listing | `bucket.list_blobs(prefix=eval_sets_dir)` filtered by `.evalset.json` suffix; result sorted alphabetically |
| Dedup guard | Existence check on the target blob before `create_eval_set` — raises `ValueError` if already present |
| Mutation helpers | `add_eval_case`, `update_eval_case`, `delete_eval_case` all load-modify-write via `_save_eval_set` |

**`GcsEvalSetResultsManager` key behaviours**

| Behaviour | Detail |
|-----------|--------|
| ID derivation | Result ID comes from `EvalSetResult.eval_set_result_id` (auto-generated UUID inside `create_eval_set_result`) |
| Save | `blob.upload_from_string(…, content_type="application/json")` — overwrites if blob already exists |
| `get_eval_set_result` | Raises `NotFoundError` (not `ValueError`) if blob missing |
| Listing | Strip `.evalset_result.json` suffix from each blob name; returns sorted list |
| `**kwargs` passthrough | Constructor accepts arbitrary `**kwargs` forwarded to `storage.Client(**kwargs)` — allows `project`, `credentials`, `client_options` etc. |

### Example 1 — create an eval set and add a case in GCS

```python
import os
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types
import time

# Assumes the GCS bucket already exists.
manager = GcsEvalSetsManager(bucket_name="my-adk-evals-bucket")

eval_set = manager.create_eval_set(app_name="weather_agent", eval_set_id="basic_queries")
print(f"Created: {eval_set.eval_set_id}")  # basic_queries

case = EvalCase(
    eval_id="ask_temperature",
    conversation=[
        Invocation(
            invocation_id="inv-001",
            user_content=types.Content(
                parts=[types.Part.from_text("What is the temperature in London?")],
                role="user",
            ),
            creation_timestamp=time.time(),
        )
    ],
    creation_timestamp=time.time(),
)
manager.add_eval_case("weather_agent", "basic_queries", case)

retrieved = manager.get_eval_set("weather_agent", "basic_queries")
print(len(retrieved.eval_cases))  # 1
```

### Example 2 — update and delete an eval case

```python
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types
import time

manager = GcsEvalSetsManager(bucket_name="my-adk-evals-bucket")

updated_case = EvalCase(
    eval_id="ask_temperature",
    conversation=[
        Invocation(
            invocation_id="inv-002",
            user_content=types.Content(
                parts=[types.Part.from_text("What is the temperature in London in Celsius?")],
                role="user",
            ),
            creation_timestamp=time.time(),
        )
    ],
    creation_timestamp=time.time(),
)
manager.update_eval_case("weather_agent", "basic_queries", updated_case)

eval_set_after = manager.get_eval_set("weather_agent", "basic_queries")
print(eval_set_after.eval_cases[0].conversation[0].user_content.parts[0].text)
# What is the temperature in London in Celsius?

manager.delete_eval_case("weather_agent", "basic_queries", "ask_temperature")
eval_set_empty = manager.get_eval_set("weather_agent", "basic_queries")
print(len(eval_set_empty.eval_cases))  # 0
```

### Example 3 — save and retrieve eval results via `GcsEvalSetResultsManager`

```python
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from google.adk.evaluation.eval_result import EvalCaseResult, EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalStatus

results_manager = GcsEvalSetResultsManager(bucket_name="my-adk-evals-bucket")

case_results = [
    EvalCaseResult(
        eval_set_id="basic_queries",
        eval_id="ask_temperature",
        final_eval_status=EvalStatus.PASSED,
        eval_metric_results={
            "tool_trajectory_avg_score": EvalMetricResult(
                metric_name="tool_trajectory_avg_score",
                score=1.0,
                eval_status=EvalStatus.PASSED,
            )
        },
    )
]

results_manager.save_eval_set_result(
    app_name="weather_agent",
    eval_set_id="basic_queries",
    eval_case_results=case_results,
)

ids = results_manager.list_eval_set_results("weather_agent")
print(ids)  # ['basic_queries_<timestamp>_<uuid>']

result = results_manager.get_eval_set_result("weather_agent", ids[0])
print(result.eval_set_id)           # basic_queries
print(result.eval_case_results[0].final_eval_status)  # EvalStatus.PASSED
```

---

## 2 · `LocalEvalSetsManager` + `LocalEvalSetResultsManager` + `convert_eval_set_to_pydantic_schema` — disk-backed eval persistence with legacy migration

**Sources:** `google/adk/evaluation/local_eval_sets_manager.py` · `google/adk/evaluation/local_eval_set_results_manager.py`

`LocalEvalSetsManager` stores eval sets as files at `{agents_dir}/{app_name}/{eval_set_id}.evalset.json` and implements a transparent double-parse strategy: if a file fails Pydantic validation it is assumed to be in the old flat-list JSON format and `convert_eval_set_to_pydantic_schema()` migrates it automatically.

`LocalEvalSetResultsManager` stores results at `{agents_dir}/{app_name}/.adk/eval_history/{result_name}.evalset_result.json`. Note the hidden `.adk/` directory — this avoids cluttering the agent source tree.

**Path conventions**

| Artefact | Path |
|----------|------|
| Eval set file | `{agents_dir}/{app_name}/{eval_set_id}.evalset.json` |
| Eval history dir | `{agents_dir}/{app_name}/.adk/eval_history/` |
| Result file | `{agents_dir}/{app_name}/.adk/eval_history/{result_name}.evalset_result.json` |

**`convert_eval_set_to_pydantic_schema` key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Old format | A top-level `list[dict]` where each dict has `name`, `data` (list of invocations), `initial_session` |
| Invocation migration | `query` → `user_content`; `reference` → `final_response`; `expected_tool_use[].tool_name/tool_input` → `FunctionCall(name, args)` |
| `initial_session` | Mapped to `SessionInput(app_name, user_id, state)` — missing keys default to empty string/dict |
| Result | Returns a valid `EvalSet` Pydantic model |

**`LocalEvalSetsManager` key behaviours**

| Behaviour | Detail |
|-----------|--------|
| ID validation | Same `^[a-zA-Z0-9_]+$` pattern as `GcsEvalSetsManager` |
| Serialisation | `model_dump_json(indent=2, exclude_unset=True, exclude_defaults=True, exclude_none=True)` |
| `list_eval_sets` | Raises `NotFoundError` (wrapping `FileNotFoundError`) if the `{app_name}` directory doesn't exist |
| `get_eval_set` | Returns `None` (not an error) if the file doesn't exist |

**`LocalEvalSetResultsManager` key behaviours**

| Behaviour | Detail |
|-----------|--------|
| History dir | Created with `os.makedirs` on first `save_eval_set_result` call |
| `get_eval_set_result` | Raises `NotFoundError` if file missing |
| `list_eval_set_results` | Returns empty list (not error) if history dir doesn't exist yet |

### Example 1 — create and populate a local eval set

```python
import os
import tempfile
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types
import time

agents_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(agents_dir, "my_agent"), exist_ok=True)

manager = LocalEvalSetsManager(agents_dir=agents_dir)

eval_set = manager.create_eval_set("my_agent", "smoke_tests")
print(eval_set.eval_set_id)  # smoke_tests

case = EvalCase(
    eval_id="greeting_test",
    conversation=[
        Invocation(
            invocation_id="i1",
            user_content=types.Content(
                parts=[types.Part.from_text("Hello!")], role="user"
            ),
            creation_timestamp=time.time(),
        )
    ],
    creation_timestamp=time.time(),
)
manager.add_eval_case("my_agent", "smoke_tests", case)

sets = manager.list_eval_sets("my_agent")
print(sets)  # ['smoke_tests']
```

### Example 2 — migrate legacy JSON format using `convert_eval_set_to_pydantic_schema`

```python
import json
import os
import tempfile
from google.adk.evaluation.local_eval_sets_manager import (
    convert_eval_set_to_pydantic_schema,
)

# Old-format eval set (list of dicts with flat query/reference structure)
old_format = [
    {
        "name": "weather_queries",
        "data": [
            {
                "query": "What is the weather in Paris?",
                "reference": "It is sunny in Paris today.",
                "expected_tool_use": [
                    {"tool_name": "get_weather", "tool_input": {"city": "Paris"}}
                ],
                "expected_intermediate_agent_responses": [],
            }
        ],
        "initial_session": {
            "app_name": "weather_app",
            "user_id": "test_user",
            "state": {},
        },
    }
]

eval_set = convert_eval_set_to_pydantic_schema("weather_compat", old_format)
print(eval_set.eval_set_id)                              # weather_compat
print(len(eval_set.eval_cases))                          # 1
inv = eval_set.eval_cases[0].conversation[0]
print(inv.user_content.parts[0].text)                   # What is the weather in Paris?
print(inv.final_response.parts[0].text)                 # It is sunny in Paris today.
print(inv.intermediate_data.tool_uses[0].name)          # get_weather
```

### Example 3 — save and list local eval set results

```python
import os
import tempfile
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from google.adk.evaluation.eval_result import EvalCaseResult, EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalStatus

agents_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(agents_dir, "my_agent"), exist_ok=True)

results_manager = LocalEvalSetResultsManager(agents_dir=agents_dir)

# No history dir yet — list returns empty list gracefully
print(results_manager.list_eval_set_results("my_agent"))  # []

case_results = [
    EvalCaseResult(
        eval_set_id="smoke_tests",
        eval_id="greeting_test",
        final_eval_status=EvalStatus.PASSED,
        eval_metric_results={
            "response_match_score": EvalMetricResult(
                metric_name="response_match_score",
                score=0.95,
                eval_status=EvalStatus.PASSED,
            )
        },
    )
]
results_manager.save_eval_set_result("my_agent", "smoke_tests", case_results)

ids = results_manager.list_eval_set_results("my_agent")
print(len(ids))   # 1
result = results_manager.get_eval_set_result("my_agent", ids[0])
print(result.eval_case_results[0].final_eval_status)  # EvalStatus.PASSED
```

---

## 3 · `MetricEvaluatorRegistry` + `DEFAULT_METRIC_EVALUATOR_REGISTRY` — pluggable metric-to-evaluator registry

**Source:** `google/adk/evaluation/metric_evaluator_registry.py`

`MetricEvaluatorRegistry` is an `@experimental` class that maps metric names (`str`) to `(type[Evaluator], MetricInfo)` tuples. `DEFAULT_METRIC_EVALUATOR_REGISTRY` is a module-level singleton pre-populated with all 13 built-in evaluators. The registry is the single authoritative place that `LocalEvalService` (and the broader eval pipeline) uses to resolve which evaluator class handles a given `EvalMetric`.

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| `_registry` type | `dict[str, tuple[type[Evaluator], MetricInfo]]` — plain dict, not thread-safe |
| `get_evaluator(eval_metric)` | Looks up by `eval_metric.metric_name`; raises `NotFoundError` if absent; always returns a **new** instance |
| `_CustomMetricEvaluator` special path | If the resolved class is a `_CustomMetricEvaluator` subclass, the constructor receives both `eval_metric` and `custom_function_path` |
| `register_evaluator(metric_info, evaluator)` | Key is `str(metric_info.metric_name)` — last registration wins; logs an `INFO` message when overwriting |
| `get_registered_metrics()` | Returns `list[MetricInfo]` deep-copies for all registered metrics |
| Singleton | `DEFAULT_METRIC_EVALUATOR_REGISTRY` is built once at module import by `_get_default_metric_evaluator_registry()` |

**All 13 built-in metrics registered at import**

| Metric name | Evaluator class |
|-------------|----------------|
| `tool_trajectory_avg_score` | `TrajectoryEvaluator` |
| `response_evaluation_score` | `ResponseEvaluator` |
| `response_match_score` | `ResponseEvaluator` |
| `safety_v1` | `SafetyEvaluatorV1` |
| `multi_turn_task_success_v1` | `MultiTurnTaskSuccessV1Evaluator` |
| `multi_turn_trajectory_quality_v1` | `MultiTurnTrajectoryQualityV1Evaluator` |
| `multi_turn_tool_use_quality_v1` | `MultiTurnToolUseQualityV1Evaluator` |
| `final_response_match_v2` | `FinalResponseMatchV2Evaluator` |
| `rubric_based_final_response_quality_v1` | `RubricBasedFinalResponseQualityV1Evaluator` |
| `hallucinations_v1` | `HallucinationsV1Evaluator` |
| `rubric_based_tool_use_quality_v1` | `RubricBasedToolUseV1Evaluator` |
| `per_turn_user_simulator_quality_v1` | `PerTurnUserSimulatorQualityV1` |
| `rubric_based_multi_turn_trajectory_quality_v1` | `RubricBasedMultiTurnTrajectoryEvaluator` |

### Example 1 — inspect all registered metrics

```python
from google.adk.evaluation.metric_evaluator_registry import (
    DEFAULT_METRIC_EVALUATOR_REGISTRY,
)

metrics = DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()
for m in metrics:
    interval = m.metric_value_info.interval
    print(f"{m.metric_name}: [{interval.min_value}, {interval.max_value}]")
# tool_trajectory_avg_score: [0.0, 1.0]
# response_evaluation_score: [1.0, 5.0]
# response_match_score: [0.0, 1.0]
# ... (13 total)
```

### Example 2 — register a custom metric evaluator

```python
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
from google.adk.evaluation.eval_metrics import MetricInfo, MetricValueInfo, Interval, EvalMetric
from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from typing import Optional

class WordCountEvaluator(Evaluator):
    """Scores 1.0 if final response has ≥ 20 words, else 0.0."""

    def __init__(self, eval_metric: EvalMetric):
        self._eval_metric = eval_metric

    async def evaluate_invocations(
        self,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]] = None,
        conversation_scenario=None,
    ) -> EvaluationResult:
        results = []
        for inv in actual_invocations:
            text = ""
            if inv.final_response and inv.final_response.parts:
                text = " ".join(p.text or "" for p in inv.final_response.parts)
            word_count = len(text.split())
            score = 1.0 if word_count >= 20 else 0.0
            results.append(PerInvocationResult(actual_invocation=inv, score=score))
        overall = sum(r.score or 0 for r in results) / len(results) if results else 0.0
        return EvaluationResult(overall_score=overall, per_invocation_results=results)

registry = MetricEvaluatorRegistry()
registry.register_evaluator(
    metric_info=MetricInfo(
        metric_name="word_count_score",
        description="Scores 1.0 when the agent response has 20+ words.",
        metric_value_info=MetricValueInfo(interval=Interval(min_value=0.0, max_value=1.0)),
    ),
    evaluator=WordCountEvaluator,
)

metric = EvalMetric(metric_name="word_count_score")
evaluator = registry.get_evaluator(metric)
print(type(evaluator).__name__)  # WordCountEvaluator
```

### Example 3 — override a built-in metric in the default registry

```python
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from google.adk.evaluation.eval_metrics import MetricInfo, MetricValueInfo, Interval, EvalMetric
from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from typing import Optional

class AlwaysPassEvaluator(Evaluator):
    """Stub: always returns 1.0 for dev-loop testing."""

    def __init__(self, eval_metric: EvalMetric):
        self._eval_metric = eval_metric

    async def evaluate_invocations(
        self,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]] = None,
        conversation_scenario=None,
    ) -> EvaluationResult:
        results = [
            PerInvocationResult(actual_invocation=i, score=1.0)
            for i in actual_invocations
        ]
        return EvaluationResult(overall_score=1.0, per_invocation_results=results)

# last registration wins — overrides the built-in safety evaluator for testing
DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
    metric_info=MetricInfo(
        metric_name="safety_v1",
        description="Always-pass stub for dev-loop tests.",
        metric_value_info=MetricValueInfo(interval=Interval(min_value=0.0, max_value=1.0)),
    ),
    evaluator=AlwaysPassEvaluator,
)

evaluator = DEFAULT_METRIC_EVALUATOR_REGISTRY.get_evaluator(
    EvalMetric(metric_name="safety_v1")
)
print(type(evaluator).__name__)  # AlwaysPassEvaluator
```

---

## 4 · `MetricInfoProvider` + `MetricInfo` + `MetricValueInfo` + `Interval` + `PrebuiltMetrics` — metric catalogue layer

**Sources:** `google/adk/evaluation/metric_info_providers.py` · `google/adk/evaluation/eval_metrics.py`

The metric catalogue layer provides structured metadata about each built-in metric. Every evaluator is paired with a `MetricInfoProvider` subclass (naming convention: `{EvaluatorClassName}MetricInfoProvider`) that returns a `MetricInfo` describing the metric name, a 2–3-line human-readable description, and the valid value range.

**`PrebuiltMetrics` enum** — full list (2.3.0)

| Enum member | String value | Range |
|-------------|-------------|-------|
| `TOOL_TRAJECTORY_AVG_SCORE` | `"tool_trajectory_avg_score"` | [0, 1] |
| `RESPONSE_EVALUATION_SCORE` | `"response_evaluation_score"` | [1, 5] |
| `RESPONSE_MATCH_SCORE` | `"response_match_score"` | [0, 1] |
| `SAFETY_V1` | `"safety_v1"` | [0, 1] |
| `FINAL_RESPONSE_MATCH_V2` | `"final_response_match_v2"` | [0, 1] |
| `RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1` | `"rubric_based_final_response_quality_v1"` | [0, 1] |
| `HALLUCINATIONS_V1` | `"hallucinations_v1"` | [0, 1] |
| `RUBRIC_BASED_TOOL_USE_QUALITY_V1` | `"rubric_based_tool_use_quality_v1"` | [0, 1] |
| `PER_TURN_USER_SIMULATOR_QUALITY_V1` | `"per_turn_user_simulator_quality_v1"` | [0, 1] |
| `MULTI_TURN_TASK_SUCCESS_V1` | `"multi_turn_task_success_v1"` | [0, 1] |
| `MULTI_TURN_TRAJECTORY_QUALITY_V1` | `"multi_turn_trajectory_quality_v1"` | [0, 1] |
| `MULTI_TURN_TOOL_USE_QUALITY_V1` | `"multi_turn_tool_use_quality_v1"` | [0, 1] |
| `RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1` | `"rubric_based_multi_turn_trajectory_quality_v1"` | [0, 1] |

**`MetricInfo` model fields**

| Field | Type | Description |
|-------|------|-------------|
| `metric_name` | `str` | Canonical name (matches `PrebuiltMetrics.value`) |
| `description` | `str` | 2–3 line description (defaults to `None`) |
| `metric_value_info` | `MetricValueInfo` | Contains an `Interval(min_value, max_value)` |

**`MetricInfoProvider` contract** — single abstract method `get_metric_info() -> MetricInfo`.

**`ResponseEvaluatorMetricInfoProvider` special case** — unlike all others, its constructor takes a `metric_name: str` parameter because `ResponseEvaluator` handles two distinct metrics (`response_evaluation_score` with range [1,5] and `response_match_score` with range [0,1]) depending on which name is passed.

### Example 1 — query metric info directly from a provider

```python
from google.adk.evaluation.metric_info_providers import (
    TrajectoryEvaluatorMetricInfoProvider,
    RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider,
    ResponseEvaluatorMetricInfoProvider,
)

traj_info = TrajectoryEvaluatorMetricInfoProvider().get_metric_info()
print(traj_info.metric_name)                          # tool_trajectory_avg_score
print(traj_info.metric_value_info.interval.min_value) # 0.0
print(traj_info.metric_value_info.interval.max_value) # 1.0

rubric_info = RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider().get_metric_info()
print(rubric_info.metric_name)  # rubric_based_final_response_quality_v1

# ResponseEvaluator handles two metric names — differentiated by the constructor arg
resp_eval_info = ResponseEvaluatorMetricInfoProvider("response_evaluation_score").get_metric_info()
print(resp_eval_info.metric_value_info.interval.min_value)  # 1.0
print(resp_eval_info.metric_value_info.interval.max_value)  # 5.0
```

### Example 2 — implement a custom `MetricInfoProvider`

```python
from google.adk.evaluation.eval_metrics import (
    MetricInfo,
    MetricInfoProvider,
    MetricValueInfo,
    Interval,
)

class EngagementScoreMetricInfoProvider(MetricInfoProvider):
    """Describes a custom agent-engagement scoring metric."""

    def get_metric_info(self) -> MetricInfo:
        return MetricInfo(
            metric_name="engagement_score",
            description=(
                "Measures how engaging the agent's response is from the"
                " perspective of a user reading it. Scores closer to 1.0 are"
                " more engaging."
            ),
            metric_value_info=MetricValueInfo(
                interval=Interval(min_value=0.0, max_value=1.0)
            ),
        )

provider = EngagementScoreMetricInfoProvider()
info = provider.get_metric_info()
print(info.metric_name)        # engagement_score
print(info.description[:30])   # Measures how engaging the age
```

### Example 3 — enumerate all `PrebuiltMetrics` and check their ranges

```python
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY

registered = {m.metric_name: m for m in DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()}

for member in PrebuiltMetrics:
    name = member.value
    info = registered.get(name)
    if info:
        iv = info.metric_value_info.interval
        print(f"{name}: [{iv.min_value}, {iv.max_value}]")
    else:
        # PER_TURN_USER_SIMULATOR_QUALITY_V1 uses the raw string directly in the enum
        print(f"{name}: (check manually — may not be in default registry)")
```

---

## 5 · `RubricBasedFinalResponseQualityV1Evaluator` — evidence-first final answer rubric evaluation

**Source:** `google/adk/evaluation/rubric_based_final_response_quality_v1.py`

This `@experimental` evaluator extends `RubricBasedEvaluator` with `RUBRIC_TYPE = "FINAL_RESPONSE_QUALITY"`. Its LLM judge prompt enforces a strict **two-pass evaluation protocol**: first establish trusted evidence exclusively from procedurally sound tool calls, then judge whether the agent's final answer is consistent with that evidence. Claims in the final answer that cannot be verified from tool outputs are treated as unverified and must be scored `"no"`.

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| `criterion_type` | `RubricsBasedCriterion` (class variable) |
| Trusted evidence rule | Only `tool_response` from procedurally sound tool calls counts; the agent's own text, summaries, and reasoning are explicitly excluded |
| `include_intermediate_responses_in_final` | If `True` on the criterion, intermediate agent responses are concatenated into the `final_response` field of the prompt |
| `developer_instructions` | Extracted from `invocation.app_details.get_developer_instructions(agent_name=…)` — the first event's `author` is used as the agent name |
| `tool_declarations` | Serialised to JSON from `app_details`; defaults to `"Agent has no tools."` if no `app_details` |
| Verdict format | `Property: …\nEvidence: …\nRationale: …\nVerdict: yes|no` — parsed by `DefaultAutoRaterResponseParser` |

**When a property yields `"yes"`** — either the final answer satisfies the property, or the property's condition was not applicable (e.g., a conditional check where the condition was not met).

### Example 1 — configure and run the evaluator on a single invocation

```python
import asyncio
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import Invocation, Rubric, RubricContent
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="FINAL_RESPONSE_QUALITY",
        rubrics=[
            Rubric(
                rubric_content=RubricContent(
                    text_property="The final answer includes the city name the user asked about."
                )
            ),
        ],
    ),
    threshold=0.5,
)

evaluator = RubricBasedFinalResponseQualityV1Evaluator(eval_metric=metric)

invocation = Invocation(
    invocation_id="inv-001",
    user_content=types.Content(
        parts=[types.Part.from_text("What is the weather in Berlin?")],
        role="user",
    ),
    final_response=types.Content(
        parts=[types.Part.from_text("It is 18°C and partly cloudy in Berlin today.")],
        role="model",
    ),
    creation_timestamp=time.time(),
)

# Run evaluation (requires GOOGLE_API_KEY or Vertex AI credentials)
# result = asyncio.run(evaluator.evaluate_invocations([invocation]))
# print(result.overall_score)  # 1.0 — the city name "Berlin" is present
```

### Example 2 — attach `AppDetails` for developer instructions and tool declarations

```python
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import Invocation, Rubric, RubricContent
from google.adk.evaluation.app_details import AppDetails, AgentDetails
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="FINAL_RESPONSE_QUALITY",
        rubrics=[
            Rubric(
                rubric_content=RubricContent(
                    text_property=(
                        "The final answer calls the get_weather tool before"
                        " responding with temperature data."
                    )
                )
            ),
        ],
    ),
)
evaluator = RubricBasedFinalResponseQualityV1Evaluator(eval_metric=metric)

# Including AppDetails gives the LLM judge context about agent instructions
# and available tools — improves verdict accuracy.
app_details = AppDetails(
    agent_details={
        "weather_agent": AgentDetails(
            instructions="You are a weather assistant. Always call get_weather before answering.",
        )
    }
)

invocation = Invocation(
    invocation_id="inv-002",
    user_content=types.Content(
        parts=[types.Part.from_text("Temperature in Madrid?")], role="user"
    ),
    final_response=types.Content(
        parts=[types.Part.from_text("Madrid is at 26°C and sunny.")], role="model"
    ),
    app_details=app_details,
    creation_timestamp=time.time(),
)
prompt = evaluator.format_auto_rater_prompt(invocation, None)
print(prompt[:200])
```

### Example 3 — check the rubric_type gating in the parent class

```python
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import Invocation, Rubric, RubricContent
from google.genai import types
import time

# RUBRIC_TYPE="FINAL_RESPONSE_QUALITY" — rubrics with a different type
# are filtered out by create_effective_rubrics_list() in the parent class.
metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="FINAL_RESPONSE_QUALITY",
        rubrics=[
            Rubric(
                rubric_content=RubricContent(text_property="Rubric A — correct type"),
                type="FINAL_RESPONSE_QUALITY",
            ),
            Rubric(
                rubric_content=RubricContent(text_property="Rubric B — wrong type, will be skipped"),
                type="TOOL_USE_QUALITY",
            ),
        ],
    ),
)
evaluator = RubricBasedFinalResponseQualityV1Evaluator(eval_metric=metric)

invocation = Invocation(
    invocation_id="inv-003",
    user_content=types.Content(
        parts=[types.Part.from_text("Test")], role="user"
    ),
    creation_timestamp=time.time(),
)
evaluator.create_effective_rubrics_list(invocation.rubrics)
print(len(evaluator._effective_rubrics_list))  # 1 (only Rubric A passes the filter)
print(evaluator._effective_rubrics_list[0].rubric_content.text_property)
# Rubric A — correct type
```

---

## 6 · `RubricBasedToolUseV1Evaluator` — tool-use rubric evaluation

**Source:** `google/adk/evaluation/rubric_based_tool_use_quality_v1.py`

This `@experimental` evaluator extends `RubricBasedEvaluator` with `RUBRIC_TYPE = "TOOL_USE_QUALITY"`. Unlike `RubricBasedFinalResponseQualityV1Evaluator`, it focuses exclusively on the agent's function calls — it does not inspect the final text response at all. The LLM judge prompt uses a strict 5-step structured format (`STEP 1:` through `STEP 5:` followed by `Property:` / `Rationale:` / `Verdict:`) and includes the entire `tool_usage` block (all tool calls and responses) as input context.

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| `criterion_type` | `RubricsBasedCriterion` (class variable) |
| `RUBRIC_TYPE` | `"TOOL_USE_QUALITY"` |
| Prompt input | `tool_usage` = `get_tool_calls_and_responses_as_json_str(actual_invocation.intermediate_data)` — only tool events |
| "Not applicable" rule | If a prerequisite property failed (e.g., tool X was not called), dependent properties (e.g., check parameter Y of tool X) are automatically scored `"yes"` because they are not applicable |
| `tool_declarations` | JSON from `app_details`; defaults to `"Agent has no tools."` |
| Missing `final_response` | Intentionally excluded — this evaluator does not use the agent's final text |

### Example 1 — evaluate tool call ordering

```python
from google.adk.evaluation.rubric_based_tool_use_quality_v1 import (
    RubricBasedToolUseV1Evaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import (
    Invocation,
    IntermediateData,
    Rubric,
    RubricContent,
)
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="TOOL_USE_QUALITY",
        rubrics=[
            Rubric(
                rubric_content=RubricContent(
                    text_property="The agent calls geocode_city before calling get_weather."
                )
            ),
            Rubric(
                rubric_content=RubricContent(
                    text_property="The get_weather tool is called with the latitude from the geocode_city response."
                )
            ),
        ],
    ),
    threshold=0.5,
)

evaluator = RubricBasedToolUseV1Evaluator(eval_metric=metric)

invocation = Invocation(
    invocation_id="inv-001",
    user_content=types.Content(
        parts=[types.Part.from_text("Weather in Tokyo?")], role="user"
    ),
    intermediate_data=IntermediateData(
        tool_uses=[
            types.FunctionCall(name="geocode_city", args={"city": "Tokyo"}),
            types.FunctionCall(name="get_weather", args={"lat": 35.68, "lon": 139.69}),
        ]
    ),
    creation_timestamp=time.time(),
)

prompt = evaluator.format_auto_rater_prompt(invocation, None)
# The prompt contains "STEP 1: ... STEP 2: ..." structure for both rubrics
print("STEP 1:" in prompt)  # True
print("Property:" in prompt)  # True
```

### Example 2 — tool use evaluation with no tools called

```python
from google.adk.evaluation.rubric_based_tool_use_quality_v1 import (
    RubricBasedToolUseV1Evaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import (
    Invocation,
    IntermediateData,
    Rubric,
    RubricContent,
)
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="TOOL_USE_QUALITY",
        rubrics=[
            Rubric(
                rubric_content=RubricContent(
                    text_property="Does the agent call the 'search_docs' tool?"
                )
            ),
        ],
    ),
)
evaluator = RubricBasedToolUseV1Evaluator(eval_metric=metric)

# No tools called — the rubric should yield "no"
invocation = Invocation(
    invocation_id="inv-002",
    user_content=types.Content(
        parts=[types.Part.from_text("Tell me about ADK.")], role="user"
    ),
    intermediate_data=IntermediateData(tool_uses=[]),
    creation_timestamp=time.time(),
)
prompt = evaluator.format_auto_rater_prompt(invocation, None)
# 'Agent has no tools.' appears when tool_declarations is unavailable
print("tool_usage" in prompt or "search_docs" in prompt)  # True
```

### Example 3 — use `RubricBasedToolUseV1Evaluator` alongside `RubricBasedFinalResponseQualityV1Evaluator` for comprehensive evaluation

```python
from google.adk.evaluation.rubric_based_tool_use_quality_v1 import (
    RubricBasedToolUseV1Evaluator,
)
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import Invocation, Rubric, RubricContent
from google.genai import types
import time

def make_invocation() -> Invocation:
    return Invocation(
        invocation_id="inv-003",
        user_content=types.Content(
            parts=[types.Part.from_text("Find the price of AAPL stock.")],
            role="user",
        ),
        final_response=types.Content(
            parts=[types.Part.from_text("AAPL is trading at $213.25.")],
            role="model",
        ),
        creation_timestamp=time.time(),
    )

tool_metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="TOOL_USE_QUALITY",
        rubrics=[
            Rubric(rubric_content=RubricContent(text_property="Agent calls get_stock_price.")),
        ],
    ),
)
final_metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="FINAL_RESPONSE_QUALITY",
        rubrics=[
            Rubric(rubric_content=RubricContent(text_property="Response includes a dollar amount.")),
        ],
    ),
)

tool_eval = RubricBasedToolUseV1Evaluator(eval_metric=tool_metric)
final_eval = RubricBasedFinalResponseQualityV1Evaluator(eval_metric=final_metric)
print("Both evaluators created — run evaluate_invocations() independently for each.")
```

---

## 7 · `RubricBasedMultiTurnTrajectoryEvaluator` — holistic multi-turn trajectory rubric evaluation

**Source:** `google/adk/evaluation/rubric_based_multi_turn_trajectory_evaluator.py`

This evaluator extends `RubricBasedEvaluator` with `RUBRIC_TYPE = "TRAJECTORY_QUALITY"` and overrides `evaluate_invocations` to accumulate the full dialogue history before issuing a single LLM judge call. Unlike per-invocation evaluators that assess each turn independently, it marks the first N-1 turns as `NOT_EVALUATED` and runs the judge once on the last turn with the full `conversation_history` context.

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Dialogue assembly | `_assemble_dialogue_history()` builds a string with `USER TURN {n}:`, `AGENT ({name}) TURN {n}:`, tool call annotations `AGENT TURN {n} (tool call): {fn}({args})`, and tool responses `AGENT TURN {n} (tool output): {fn} -> {resp}` |
| First N-1 turns | Marked `EvalStatus.NOT_EVALUATED` with `score=None` |
| Last turn | Full dialogue context submitted to LLM; score propagated back to the `EvaluationResult.overall_score` |
| `_formatted_instructions` | De-duplicated via `dict.fromkeys(instructions_parts)` — same-agent instructions only appear once |
| `_formatted_tools` | Per-agent tool listing with `- {func.name}: {func.description}` rows |
| Rubric input to LLM | JSON list of `{"property": "…", "type": "…"}` dicts |
| Combined score | Average of per-rubric binary verdicts (1.0 / 0.0) across all rubrics |

### Example 1 — evaluate a 3-turn conversation holistically

```python
import asyncio
from google.adk.evaluation.rubric_based_multi_turn_trajectory_evaluator import (
    RubricBasedMultiTurnTrajectoryEvaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import Invocation, Rubric, RubricContent
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="TRAJECTORY_QUALITY",
        rubrics=[
            Rubric(
                rubric_content=RubricContent(
                    text_property="The agent greeted the user in Turn 1."
                )
            ),
            Rubric(
                rubric_content=RubricContent(
                    text_property="The agent correctly answered the question in Turn 2."
                )
            ),
            Rubric(
                rubric_content=RubricContent(
                    text_property="The agent asked a clarifying follow-up in Turn 3."
                )
            ),
        ],
    ),
    threshold=0.6,
)

evaluator = RubricBasedMultiTurnTrajectoryEvaluator(eval_metric=metric)

def make_turn(user_text: str, agent_text: str, inv_id: str) -> Invocation:
    return Invocation(
        invocation_id=inv_id,
        user_content=types.Content(parts=[types.Part.from_text(user_text)], role="user"),
        final_response=types.Content(parts=[types.Part.from_text(agent_text)], role="model"),
        creation_timestamp=time.time(),
    )

invocations = [
    make_turn("Hello!", "Hi there! How can I help you today?", "t1"),
    make_turn("What is the capital of France?", "The capital of France is Paris.", "t2"),
    make_turn("And Germany?", "Berlin. Would you like to know more European capitals?", "t3"),
]

# Assemble the dialogue (normally called inside evaluate_invocations)
evaluator._assemble_dialogue_history(invocations)
print(evaluator._formatted_dialogue[:200])
# USER TURN 1: Hello!
# AGENT (agent) TURN 1: Hi there! How can I help you today?
# USER TURN 2: What is the capital of France?
# ...
```

### Example 2 — check that first N-1 turns are NOT_EVALUATED

```python
import asyncio
from unittest.mock import AsyncMock, patch
from google.adk.evaluation.rubric_based_multi_turn_trajectory_evaluator import (
    RubricBasedMultiTurnTrajectoryEvaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
    EvalStatus,
)
from google.adk.evaluation.eval_case import Invocation, Rubric, RubricContent
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="TRAJECTORY_QUALITY",
        rubrics=[Rubric(rubric_content=RubricContent(text_property="Property A"))],
    ),
)
evaluator = RubricBasedMultiTurnTrajectoryEvaluator(eval_metric=metric)

invocations = [
    Invocation(
        invocation_id=f"t{i}",
        user_content=types.Content(parts=[types.Part.from_text(f"Turn {i}")], role="user"),
        creation_timestamp=time.time(),
    )
    for i in range(3)
]

# Mock the parent's evaluate_invocations to return a canned final result
mock_result = EvaluationResult(
    overall_score=1.0,
    per_invocation_results=[
        PerInvocationResult(actual_invocation=invocations[2], score=1.0)
    ],
)

async def run():
    with patch.object(
        RubricBasedMultiTurnTrajectoryEvaluator.__bases__[0],
        "evaluate_invocations",
        new=AsyncMock(return_value=mock_result),
    ):
        result = await evaluator.evaluate_invocations(invocations)
    return result

result = asyncio.run(run())
# First 2 turns are NOT_EVALUATED
not_evaluated = [
    r for r in result.per_invocation_results
    if r.eval_status == EvalStatus.NOT_EVALUATED
]
print(len(not_evaluated))  # 2
```

### Example 3 — tool calls appear in the assembled dialogue

```python
from google.adk.evaluation.rubric_based_multi_turn_trajectory_evaluator import (
    RubricBasedMultiTurnTrajectoryEvaluator,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_case import (
    Invocation,
    InvocationEvents,
    Rubric,
    RubricContent,
)
from google.adk.events.event import Event
from google.genai import types
import time

metric = EvalMetric(
    metric_name=PrebuiltMetrics.RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1.value,
    criterion=RubricsBasedCriterion(
        rubric_type="TRAJECTORY_QUALITY",
        rubrics=[Rubric(rubric_content=RubricContent(text_property="Calls get_weather"))],
    ),
)
evaluator = RubricBasedMultiTurnTrajectoryEvaluator(eval_metric=metric)

tool_event = Event(
    author="weather_agent",
    content=types.Content(
        parts=[
            types.Part.from_function_call(
                name="get_weather", args={"city": "London"}
            )
        ],
        role="model",
    ),
)
invocations = [
    Invocation(
        invocation_id="t1",
        user_content=types.Content(
            parts=[types.Part.from_text("Weather in London?")], role="user"
        ),
        intermediate_data=InvocationEvents(invocation_events=[tool_event]),
        final_response=types.Content(
            parts=[types.Part.from_text("It's 15°C and cloudy in London.")], role="model"
        ),
        creation_timestamp=time.time(),
    )
]

evaluator._assemble_dialogue_history(invocations)
print("get_weather" in evaluator._formatted_dialogue)   # True
print("tool call" in evaluator._formatted_dialogue)     # True
```

---

## 8 · `PerTurnUserSimulatorQualityV1` — per-turn user simulator quality metric

**Source:** `google/adk/evaluation/simulation/per_turn_user_simulator_quality_v1.py`

This `@experimental` evaluator checks that the user messages generated by a user simulator (e.g., `LlmBackedUserSimulator`) correctly follow the intended `ConversationScenario`. It validates each turn independently and returns the **fraction of turns that passed** as the overall score. The evaluation has three distinct phases:

1. **Turn 1** — exact string match against `conversation_scenario.starting_prompt` (no LLM call).
2. **Turns 2…N** — LLM judge with `num_samples` majority-vote sampling per turn.
3. **Stop-signal turn** — a virtual invocation with the stop signal text is appended and validated to confirm the simulator ended the conversation correctly.

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| `criterion_type` | `LlmBackedUserSimulatorCriterion` (class variable) |
| Turn 1 check | `get_text_from_content(inv.user_content).strip() == scenario.starting_prompt.strip()` — exact match, `score=1` or `0` |
| LLM judge model | `self._criterion.judge_model_options.judge_model` resolved via `LLMRegistry` |
| `num_samples` majority vote | `len(positive) > len(negative)` → positive; ties go to negative |
| Label parsing | Regex `"is_valid":\s*\[*[\n\s]*"*([^"\]]*)"*…` — labels `VALID`/`TRUE` → 1.0; `INVALID`/`ALMOST`/`FALSE`/`PARTIALLY_VALID` → 0.0; anything else → `NOT_FOUND` (excluded from vote) |
| Stop-signal failure | If the stop-signal turn is invalid, the **last** user turn's result is replaced with the stop-signal result |
| `conversation_scenario` required | `ValueError` raised if `None` is passed |
| Overall score | `num_valid / num_evaluated` — fraction (not average of scores) |

### Example 1 — first-turn exact-match check (no LLM required)

```python
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1 import (
    PerTurnUserSimulatorQualityV1,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    LlmBackedUserSimulatorCriterion,
)
from google.adk.evaluation.eval_case import (
    ConversationScenario,
    Invocation,
)
from google.genai import types
import time

criterion = LlmBackedUserSimulatorCriterion(
    stop_signal="[DONE]",
    threshold=0.8,
)
metric = EvalMetric(
    metric_name=PrebuiltMetrics.PER_TURN_USER_SIMULATOR_QUALITY_V1.value,
    criterion=criterion,
    threshold=0.8,
)
evaluator = PerTurnUserSimulatorQualityV1(eval_metric=metric)

scenario = ConversationScenario(
    starting_prompt="I need help with my order.",
    conversation_plan="User wants to check order status, then cancel.",
    user_persona="Frustrated customer.",
)

# Exact match with starting_prompt → score 1.0
correct_first = Invocation(
    invocation_id="t1-ok",
    user_content=types.Content(
        parts=[types.Part.from_text("I need help with my order.")], role="user"
    ),
    creation_timestamp=time.time(),
)
result = evaluator._evaluate_first_turn(correct_first, scenario)
print(result.score)  # 1

# Different text → score 0
wrong_first = Invocation(
    invocation_id="t1-fail",
    user_content=types.Content(
        parts=[types.Part.from_text("Hello there!")], role="user"
    ),
    creation_timestamp=time.time(),
)
result_fail = evaluator._evaluate_first_turn(wrong_first, scenario)
print(result_fail.score)  # 0
```

### Example 2 — label parsing from LLM response text

```python
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1 import (
    _parse_llm_response,
)
from google.adk.evaluation.llm_as_judge_utils import Label

# VALID → 1.0
label = _parse_llm_response('{"is_valid": "valid"}')
print(label)  # Label.VALID

# INVALID → 0.0
label = _parse_llm_response('{"is_valid": "invalid"}')
print(label)  # Label.INVALID

# ALMOST → also maps to INVALID
label = _parse_llm_response('{"is_valid": "almost"}')
print(label)  # Label.INVALID

# Unrecognised string → NOT_FOUND (excluded from majority vote)
label = _parse_llm_response('{"is_valid": "maybe"}')
print(label)  # Label.NOT_FOUND

# TRUE/FALSE aliases (for compatibility)
label = _parse_llm_response('{"is_valid": "true"}')
print(label)  # Label.VALID

label = _parse_llm_response('{"is_valid": "false"}')
print(label)  # Label.INVALID
```

### Example 3 — majority-vote aggregation across samples

```python
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1 import (
    PerTurnUserSimulatorQualityV1,
)
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    PrebuiltMetrics,
    LlmBackedUserSimulatorCriterion,
    EvalStatus,
)
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from google.genai import types
import time

criterion = LlmBackedUserSimulatorCriterion(stop_signal="[DONE]")
metric = EvalMetric(
    metric_name=PrebuiltMetrics.PER_TURN_USER_SIMULATOR_QUALITY_V1.value,
    criterion=criterion,
    threshold=0.5,
)
evaluator = PerTurnUserSimulatorQualityV1(eval_metric=metric)

inv = Invocation(
    invocation_id="t2",
    user_content=types.Content(
        parts=[types.Part.from_text("Any updates on my order?")], role="user"
    ),
    creation_timestamp=time.time(),
)

# Simulate 3 samples: 2 VALID, 1 INVALID → majority = VALID (score 1.0)
samples = [
    PerInvocationResult(actual_invocation=inv, score=1.0, eval_status=EvalStatus.PASSED),
    PerInvocationResult(actual_invocation=inv, score=0.0, eval_status=EvalStatus.FAILED),
    PerInvocationResult(actual_invocation=inv, score=1.0, eval_status=EvalStatus.PASSED),
]
winner = evaluator._aggregate_samples(samples)
print(winner.score)         # 1.0
print(winner.eval_status)   # EvalStatus.PASSED

# Tie (1 VALID, 1 INVALID) → negative wins (conservative)
tie_samples = [
    PerInvocationResult(actual_invocation=inv, score=1.0, eval_status=EvalStatus.PASSED),
    PerInvocationResult(actual_invocation=inv, score=0.0, eval_status=EvalStatus.FAILED),
]
tie_winner = evaluator._aggregate_samples(tie_samples)
print(tie_winner.score)  # 0.0  (negative wins on tie: len(neg) >= len(pos))
```

---

## 9 · `InterceptionResult` + `check_interception` + `create_mock_context` — workflow replay decision engine

**Source:** `google/adk/workflow/utils/_replay_interceptor.py`

`check_interception` is the central routing function for cross-turn workflow replay (rehydration). Whenever a node is about to execute, this function is called to decide whether to actually run it or to fast-forward it using cached results from a previous session turn. The logic is a 5-case decision tree driven by the `_ChildScanState` (historical events scan) and the `DynamicNodeRun` (current-turn execution record).

**`InterceptionResult` fields**

| Field | Type | Meaning |
|-------|------|---------|
| `should_run` | `bool` | `True` = execute node natively; `False` = use cached output/route |
| `output` | `Any` | Cached output to fast-forward with |
| `route` | `Any` | Cached routing decision |
| `interrupts` | `set[str]` | Unresolved interrupt IDs — node stays WAITING |
| `resume_inputs` | `dict[str, Any] \| None` | Resolved HITL responses to feed back into a re-run |
| `transfer_to_agent` | `str \| None` | Target agent for cross-turn agent-transfer fast-forward |

**5-case decision tree in `check_interception`**

| Case | Condition | Outcome |
|------|-----------|---------|
| 1 | `current_run.status == COMPLETED` | Fast-forward: `should_run=False`, use `current_run.output` |
| 2 | `current_run.status == WAITING` | Stay waiting: `should_run=False`, bubble `interrupts` |
| 3 | `recovered` has `output`/`route`/`transfer` (no unresolved interrupts) | Cross-turn fast-forward: `should_run=False`, inject cached output |
| 4 | All prior interrupts resolved, `rerun_on_resume=False` | Extract `resolved_responses` as output without re-running |
| 5 | All prior interrupts resolved, `rerun_on_resume=True` | Re-run: `should_run=True`, `resume_inputs=resolved_responses` |

**`create_mock_context`** builds a `Context` object populated with the cached `output`, `route`, `interrupts`, and `transfer_to_agent` from the `InterceptionResult` so the rest of the pipeline sees a fully-formed context without any actual node execution.

### Example 1 — fast-forward a completed dynamic node

```python
from google.adk.workflow.utils._replay_interceptor import (
    InterceptionResult,
    check_interception,
)
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from unittest.mock import MagicMock
from google.adk.workflow._node_status import NodeStatus

# Simulate a DynamicNodeRun that already completed in the current turn
current_run = MagicMock()
current_run.state.status = NodeStatus.COMPLETED
current_run.output = {"temperature": 22}
current_run.transfer_to_agent = None

result = check_interception(
    node_path="weather_workflow/fetch_weather",
    node=MagicMock(),
    recovered=None,
    current_run=current_run,
    curr_parent_ctx=MagicMock(),
)

print(result.should_run)    # False — bypass: already completed
print(result.output)        # {'temperature': 22}
```

### Example 2 — cross-turn fast-forward from historical events

```python
from google.adk.workflow.utils._replay_interceptor import InterceptionResult
from google.adk.workflow.utils._rehydration_utils import _ChildScanState

# Simulate a cross-turn successful completion (Case 3)
recovered = _ChildScanState(
    output={"answer": "Paris"},
    route="summary_node",
    interrupt_ids=set(),
    resolved_ids=set(),
    resolved_responses={},
    transfer_to_agent=None,
)
# no unresolved interrupts + output present → fast-forward
unresolved = recovered.interrupt_ids - recovered.resolved_ids
print(unresolved)  # set() — empty

# Simulate the Case 3 outcome from check_interception
result = InterceptionResult(
    should_run=False,
    output=recovered.output,
    route=recovered.route,
)
print(result.should_run)  # False
print(result.output)      # {'answer': 'Paris'}
print(result.route)       # summary_node
```

### Example 3 — HITL resume: unresolved interrupts remain, no rerun

```python
from google.adk.workflow.utils._replay_interceptor import InterceptionResult
from google.adk.workflow.utils._rehydration_utils import _ChildScanState

# Case 4: all interrupts resolved, rerun_on_resume=False
# → extract resolved_responses as output directly
recovered = _ChildScanState(
    output=None,
    route=None,
    interrupt_ids={"interrupt-001"},
    resolved_ids={"interrupt-001"},
    resolved_responses={"interrupt-001": "User approved"},
    transfer_to_agent=None,
)

unresolved = recovered.interrupt_ids - recovered.resolved_ids
print(unresolved)  # set() — all resolved

child_resume_inputs = recovered.resolved_responses
output = list(child_resume_inputs.values())[0]  # single response → unwrap
print(output)  # User approved

result = InterceptionResult(
    should_run=False,
    output=output,
)
print(result.should_run)  # False — skip re-run, inject answer directly
print(result.output)      # User approved
```

---

## 10 · `StreamingResponseAggregator` (2.3.0 deep-dive) — progressive SSE with JSONPath partial-args

**Source:** `google/adk/utils/streaming_utils.py`

`StreamingResponseAggregator` handles the aggregation of streaming `GenerateContentResponse` chunks into complete `LlmResponse` objects. In 2.3.0 the class has two distinct code paths controlled by the `PROGRESSIVE_SSE_STREAMING` feature flag:

- **Non-progressive (legacy)** — accumulates text into `_text`/`_thought_text` strings; yields a merged text event when a non-text chunk arrives; yields raw chunk otherwise.
- **Progressive (new in 2.3.0)** — maintains `_parts_sequence: list[types.Part]` to preserve exact part ordering (thought, text, function call interleaved); flushes via `_flush_text_buffer_to_sequence()` and `_flush_function_call_to_sequence()` dual-buffer model; marks ALL intermediate chunks as `partial=True`.

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| `_flush_text_buffer_to_sequence()` | Flushes `_current_text_buffer` to `_parts_sequence` as a `thought=True` or regular text Part; resets buffer and `_current_text_is_thought`; called when text type changes or a non-text part arrives |
| `_flush_function_call_to_sequence()` | Creates a complete `Part.from_function_call(name, args)` from accumulated `_current_fc_args`; sets `fc.id` and `fc.thought_signature` on the Part; resets all `_current_fc_*` state |
| Streaming FC detection | `fc.partial_args or fc.will_continue` — first chunk may have `will_continue=True` with no `partial_args` yet |
| JSONPath argument reconstruction | `_get_value_from_partial_arg()` handles `string_value` (appends), `number_value`, `bool_value`, `null_value`; `_set_value_by_json_path("$.location.lat", 35.68)` sets nested dict paths |
| FC ID generation | If no `fc.id` on first chunk and no `_current_fc_id` yet, calls `generate_client_function_call_id()` — generates the `adk-{uuid}` prefix |
| `thought_signature` | Captured from the **first** streaming Part that carries it and stored in `_current_thought_signature`; attached to the complete FC Part on flush |
| `close()` | Called after all chunks are consumed; in progressive mode flushes both buffers then assembles final `LlmResponse(partial=False)` from `_parts_sequence`; in non-progressive mode assembles from `_thought_text` + `_text` |
| Error extraction | `finish_reason != STOP` → `error_code=finish_reason`; `prompt_feedback.block_reason` checked as fallback |

### Example 1 — basic progressive streaming aggregation

```python
import asyncio
from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.adk.features import FeatureName, override_feature_enabled
from google.genai import types

async def aggregate_chunks():
    aggregator = StreamingResponseAggregator()

    # Simulate two text chunks
    def make_response(text: str) -> types.GenerateContentResponse:
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[types.Part.from_text(text)], role="model"
                    ),
                    finish_reason=None,
                )
            ]
        )

    chunks = [make_response("Hello, "), make_response("world!")]

    all_responses = []
    with override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, True):
        for chunk in chunks:
            async for response in aggregator.process_response(chunk):
                all_responses.append(response)
                print(f"partial={response.partial}, text={response.content.parts[0].text!r}")

        final = aggregator.close()
        if final and final.content:
            full_text = "".join(p.text or "" for p in final.content.parts)
            print(f"Final assembled text: {full_text!r}")  # 'Hello, world!'

asyncio.run(aggregate_chunks())
```

### Example 2 — streaming function call argument reconstruction via JSONPath

```python
from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.genai import types

aggregator = StreamingResponseAggregator()

# Simulate partial_args arriving in two chunks
fc_first = types.FunctionCall(
    name="get_weather",
    id="adk-1234",
    partial_args=[
        types.PartialArg(json_path="$.city", string_value="Lon"),
    ],
    will_continue=True,
)
fc_second = types.FunctionCall(
    name=None,
    id=None,
    partial_args=[
        types.PartialArg(json_path="$.city", string_value="don"),  # appended to "Lon"
        types.PartialArg(json_path="$.units", string_value="metric"),
    ],
    will_continue=False,  # marks end of streaming FC
)

# Process first chunk (FC not yet complete)
aggregator._process_streaming_function_call(fc_first)
print(aggregator._current_fc_name)   # get_weather
print(aggregator._current_fc_args)   # {'city': 'Lon'}

# Process second chunk (FC complete → flushes to _parts_sequence)
aggregator._process_streaming_function_call(fc_second)
print(aggregator._current_fc_args)   # {} — reset after flush
print(len(aggregator._parts_sequence))  # 1
flushed = aggregator._parts_sequence[0]
print(flushed.function_call.name)   # get_weather
print(flushed.function_call.args)   # {'city': 'London', 'units': 'metric'}
```

### Example 3 — non-progressive (legacy) aggregation path

```python
import asyncio
from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.adk.features import FeatureName, override_feature_enabled
from google.genai import types

async def aggregate_legacy():
    aggregator = StreamingResponseAggregator()

    def text_response(text: str) -> types.GenerateContentResponse:
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[types.Part.from_text(text)], role="model"
                    )
                )
            ]
        )

    # Non-progressive path: text chunks mark partial=True and accumulate
    with override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, False):
        for chunk in [text_response("The "), text_response("answer is 42.")]:
            async for r in aggregator.process_response(chunk):
                print(f"partial={r.partial}")  # True for text accumulation

        final = aggregator.close()
        if final and final.content:
            text = " ".join(p.text or "" for p in final.content.parts)
            print(f"Aggregated: {text!r}")  # 'The answer is 42.'

asyncio.run(aggregate_legacy())
```

---

## Revision history

| Date | Version | Change |
|------|---------|--------|
| 2026-06-20 | 2.3.0 | Vol. 23: GcsEvalSetsManager/GcsEvalSetResultsManager, LocalEvalSetsManager/LocalEvalSetResultsManager/convert_eval_set_to_pydantic_schema, MetricEvaluatorRegistry/DEFAULT_METRIC_EVALUATOR_REGISTRY, MetricInfoProvider/MetricInfo/MetricValueInfo/Interval/PrebuiltMetrics, RubricBasedFinalResponseQualityV1Evaluator, RubricBasedToolUseV1Evaluator, RubricBasedMultiTurnTrajectoryEvaluator, PerTurnUserSimulatorQualityV1, InterceptionResult/check_interception/create_mock_context, StreamingResponseAggregator 2.3.0 deep-dive |

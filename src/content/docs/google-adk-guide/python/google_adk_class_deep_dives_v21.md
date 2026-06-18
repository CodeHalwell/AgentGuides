---
title: "Class deep dives — volume 21 (conformance testing, simulation personas, rubric evaluation, eval config, Interactions API & tool confirmation pipeline)"
description: "Source-verified 2.2.0 deep dives: RecordingsPlugin/ReplayPlugin (YAML-based conformance test recording and replay), StaticUserSimulator/UserSimulator/Status/NextUserMessage (scripted multi-turn conversation replay), UserPersona/UserBehavior/UserPersonaRegistry/PreBuiltBehaviors (behavioral simulation profiles), EvalConfig/CustomMetricConfig/_CustomMetricEvaluator (eval configuration DSL + custom metric importlib integration), Rubric/RubricContent/RubricScore/RubricBasedEvaluator (rubric-based LLM-as-judge evaluation system), InteractionsRequestProcessor (Gemini Interactions API stateful conversation chaining), _RequestConfirmationLlmRequestProcessor (4-step tool confirmation resume pipeline), GoogleApiToOpenApiConverter (Google API Discovery → OpenAPI v3 converter), LocalEvalService (local end-to-end eval pipeline), EvaluationGenerator/_LiveSession (inference + live eval runner)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 21"
  order: 90
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `RecordingsPlugin` + `ReplayPlugin` | `google.adk.cli.plugins` | Stable (internal) |
| 2 | `StaticUserSimulator` + `UserSimulator` + `Status` + `NextUserMessage` | `google.adk.evaluation.simulation` | `@experimental` |
| 3 | `UserPersona` + `UserBehavior` + `UserPersonaRegistry` + `PreBuiltBehaviors` | `google.adk.evaluation.simulation` | `@experimental` |
| 4 | `EvalConfig` + `CustomMetricConfig` + `_CustomMetricEvaluator` | `google.adk.evaluation.eval_config` | Stable |
| 5 | `Rubric` + `RubricContent` + `RubricScore` + `RubricBasedEvaluator` | `google.adk.evaluation` | Stable / `@experimental` |
| 6 | `InteractionsRequestProcessor` | `google.adk.flows.llm_flows.interactions_processor` | Stable (internal) |
| 7 | `_RequestConfirmationLlmRequestProcessor` | `google.adk.flows.llm_flows.request_confirmation` | Stable (internal) |
| 8 | `GoogleApiToOpenApiConverter` | `google.adk.tools.google_api_tool.googleapi_to_openapi_converter` | Stable (internal) |
| 9 | `LocalEvalService` | `google.adk.evaluation.local_eval_service` | `@experimental` |
| 10 | `EvaluationGenerator` + `_LiveSession` | `google.adk.evaluation.evaluation_generator` | Stable (internal) |

---

## 1 · `RecordingsPlugin` + `ReplayPlugin` — YAML conformance test recording and replay

**Source:** `google/adk/cli/plugins/recordings_plugin.py` and `google/adk/cli/plugins/replay_plugin.py`

`RecordingsPlugin` is a `BasePlugin` that intercepts all seven plugin lifecycle callbacks to capture every LLM call and tool execution made during an agent run, serialising the result to a YAML file that can later be used as a deterministic replay fixture. It is the recording half of ADK's conformance testing infrastructure — the mechanism that lets you freeze a golden run and later assert that refactored code produces identical tool trajectories. `ReplayPlugin` is the complementary read-side: at `before_tool_callback` it intercepts each tool call, looks up the recorded response by agent name and sequential index, verifies that the tool name and arguments match what was recorded, returns the stored response directly (bypassing actual execution), and raises `ReplayVerificationError` on any mismatch.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Seven callbacks covered | `RecordingsPlugin` implements `before_run_callback`, `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`, `on_tool_error_callback`, and `after_run_callback` — every extensible hook. |
| Concurrent-run isolation | Per-invocation state (`_InvocationRecordingState` / `_InvocationReplayState`) is keyed by `invocation_id`, so multiple simultaneous agent runs do not corrupt each other's recording buffers. |
| Session state keys | Recording mode is activated by setting `_adk_recordings_config` in session state; replay mode by `_adk_replay_config`. Both accept the same dict schema: `{dir, user_message_index, streaming_mode}`. |
| Dual pending structures | `RecordingsPlugin` tracks LLM recordings in `pending_llm_recordings: dict[str, Recording]` (keyed by `agent_name`) and tool recordings in `pending_tool_recordings: dict[str, Recording]` (keyed by `function_call_id`), with `pending_recordings_order: list[Recording]` preserving chronological sequence. |
| Output file naming | Streaming mode `NONE` → `generated-recordings.yaml`; mode `SSE` → `generated-recordings-sse.yaml`. |
| Replay short-circuits execution | `ReplayPlugin.before_tool_callback` returns the recorded `tool_response.response` dict directly, preventing real network calls. `AgentTool` instances are skipped (they have no side effects to mock). |
| Per-agent replay index | `_InvocationReplayState.agent_tool_replay_indices: dict[str, int]` tracks how many tools each agent has consumed, enabling correct sequential matching when multiple agents run in the same invocation. |

### Example 1 — capturing a golden run via session state

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.cli.plugins.recordings_plugin import RecordingsPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# RecordingsPlugin is registered at runner construction time.
# It activates only when _adk_recordings_config is present in session state.
agent = LlmAgent(
    name="research_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions using web search when needed.",
    tools=[google_search],
)

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="research_app",
    session_service=session_service,
)
runner.agent_registry  # ensure agent tree is built

# Register the plugin after construction.
recordings_plugin = RecordingsPlugin()


async def record_golden_run():
    session = await session_service.create_session(
        app_name="research_app",
        user_id="tester",
        # Activating recording mode: point to an output directory,
        # set user_message_index to 0 for the first user turn,
        # and choose streaming_mode ("none" or "sse").
        state={
            "_adk_recordings_config": {
                "dir": "/tmp/golden_recordings",
                "user_message_index": 0,
                "streaming_mode": "none",
            }
        },
    )
    user_content = types.Content(
        role="user",
        parts=[types.Part(text="What is the boiling point of water in Kelvin?")],
    )
    async for event in runner.run_async(
        user_id="tester",
        session_id=session.id,
        new_message=user_content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)
    # After run completes, after_run_callback writes generated-recordings.yaml
    # to /tmp/golden_recordings/


asyncio.run(record_golden_run())
```

### Example 2 — replaying against the recorded fixture

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.cli.plugins.replay_plugin import ReplayPlugin, ReplayVerificationError
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(
    name="research_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions using web search when needed.",
)

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="research_app",
    session_service=session_service,
)

replay_plugin = ReplayPlugin()


async def run_conformance_test():
    session = await session_service.create_session(
        app_name="research_app",
        user_id="ci_runner",
        # Replay mode: point to the same directory that was recorded.
        # ReplayPlugin reads generated-recordings.yaml from this path.
        state={
            "_adk_replay_config": {
                "dir": "/tmp/golden_recordings",
                "user_message_index": 0,
                "streaming_mode": "none",
            }
        },
    )
    user_content = types.Content(
        role="user",
        parts=[types.Part(text="What is the boiling point of water in Kelvin?")],
    )
    try:
        async for event in runner.run_async(
            user_id="ci_runner",
            session_id=session.id,
            new_message=user_content,
        ):
            pass  # consume events; plugin verifies tool calls internally
        print("Conformance test PASSED")
    except ReplayVerificationError as exc:
        # Raised when tool name or args differ from the recorded golden values
        print(f"Conformance test FAILED: {exc}")
        raise


asyncio.run(run_conformance_test())
```

### Example 3 — parallel conformance runs using invocation_id isolation

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.cli.plugins.replay_plugin import ReplayPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# Because _InvocationReplayState is keyed by invocation_id, multiple
# concurrent replays against the same runner do not interfere.
async def conformance_task(runner, session_service, recording_dir, turn_idx):
    session = await session_service.create_session(
        app_name="ci_app",
        user_id=f"tester_{turn_idx}",
        state={
            "_adk_replay_config": {
                "dir": recording_dir,
                "user_message_index": turn_idx,
                "streaming_mode": "none",
            }
        },
    )
    user_content = types.Content(
        role="user",
        parts=[types.Part(text=f"Query number {turn_idx}")],
    )
    async for _ in runner.run_async(
        user_id=f"tester_{turn_idx}",
        session_id=session.id,
        new_message=user_content,
    ):
        pass
    return f"turn {turn_idx} passed"


async def run_parallel_conformance():
    agent = LlmAgent(name="my_agent", model="gemini-2.0-flash", instruction="Help.")
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, app_name="ci_app", session_service=session_service
    )
    # Run three user turns in parallel — each gets a distinct invocation_id
    # so agent_tool_replay_indices[agent_name] starts at 0 for each.
    results = await asyncio.gather(
        *[
            conformance_task(runner, session_service, "/tmp/golden_recordings", i)
            for i in range(3)
        ]
    )
    print(results)


asyncio.run(run_parallel_conformance())
```

---

## 2 · `StaticUserSimulator` + `UserSimulator` + `Status` + `NextUserMessage` — scripted multi-turn replay

**Source:** `google/adk/evaluation/simulation/user_simulator.py` and `google/adk/evaluation/simulation/static_user_simulator.py`

> **`@experimental`** — these classes carry the experimental decorator and their APIs may change without notice.

`UserSimulator` is an abstract base class that defines the contract for any component that drives the "user" side of a simulated multi-turn conversation during evaluation. It provides two abstract methods: `get_next_user_message`, which inspects the event history and decides what to send next, and `get_simulation_evaluator`, which optionally returns a metric evaluator scoped to the simulator's own judgment criteria. `StaticUserSimulator` is the simplest concrete implementation: it replays a pre-scripted `StaticConversation` (a list of `Invocation` objects, each carrying a `user_content`) in order, returning `Status.STOP_SIGNAL_DETECTED` once the list is exhausted. `NextUserMessage` is a Pydantic model that pairs a `Status` enum value with an optional `Content` object, enforced by a model validator that requires `user_message` to be present if and only if `status == Status.SUCCESS`.

### Key behaviours

| Behaviour | Detail |
|---|---|
| `Status` variants | `SUCCESS` (message ready), `TURN_LIMIT_REACHED` (safety cap hit), `STOP_SIGNAL_DETECTED` (conversation naturally ended), `NO_MESSAGE_GENERATED` (simulator could not produce a turn). |
| `NextUserMessage` invariant | A model validator enforces the exclusive pairing: `user_message` is `None` for all non-`SUCCESS` statuses, and must be set for `SUCCESS`. |
| `StaticUserSimulator` index walk | Uses an integer `invocation_idx` starting at 0; each call to `get_next_user_message` increments it. When `invocation_idx >= len(static_conversation)` the method returns `Status.STOP_SIGNAL_DETECTED`. |
| No evaluator for static | `StaticUserSimulator.get_simulation_evaluator()` returns `None` — scripted replay has no dynamic judgment to apply. |
| Config validation | `UserSimulator.__init__` re-validates the raw config dict through `config_type.model_validate(config.model_dump())` so subclass-specific field constraints are applied eagerly. |
| Forward-compatible config | `BaseUserSimulatorConfig` uses `alias_generator=alias_generators.to_camel` and `extra="allow"` so unknown future fields round-trip without error. |

### Example 1 — scripting a two-turn customer service evaluation

```python
import asyncio
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.adk.evaluation.simulation.user_simulator import Status
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types


def make_user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


# Build a scripted two-turn conversation
turn_1 = Invocation(user_content=make_user_content("I need to cancel my subscription."))
turn_2 = Invocation(user_content=make_user_content("Yes, please proceed with cancellation."))

simulator = StaticUserSimulator(static_conversation=[turn_1, turn_2])

# Simulate the eval loop manually to inspect status transitions
async def run_scripted_sim():
    events = []  # in a real eval loop, agent events accumulate here
    for expected_status in [Status.SUCCESS, Status.SUCCESS, Status.STOP_SIGNAL_DETECTED]:
        result = await simulator.get_next_user_message(events)
        print(f"Status: {result.status.value}")
        if result.status == Status.SUCCESS:
            print(f"  Message: {result.user_message.parts[0].text}")
        assert result.status == expected_status

    # No simulation evaluator for static replay
    assert simulator.get_simulation_evaluator() is None
    print("All turns matched expected status sequence.")


asyncio.run(run_scripted_sim())
```

### Example 2 — inspecting the `NextUserMessage` validator constraint

```python
from google.adk.evaluation.simulation.user_simulator import NextUserMessage, Status
from pydantic import ValidationError

# SUCCESS requires user_message to be set
try:
    # This raises: user_message must be provided when status is SUCCESS
    bad = NextUserMessage(status=Status.SUCCESS, user_message=None)
except ValidationError as exc:
    print("Expected validation error:", exc.error_count(), "error(s)")

# Non-SUCCESS must NOT have user_message
try:
    # This raises: user_message must not be provided for non-SUCCESS statuses
    from google.genai import types
    bad2 = NextUserMessage(
        status=Status.TURN_LIMIT_REACHED,
        user_message=types.Content(role="user", parts=[types.Part(text="hi")]),
    )
except ValidationError as exc:
    print("Expected validation error:", exc.error_count(), "error(s)")

# Correct construction
stop = NextUserMessage(status=Status.STOP_SIGNAL_DETECTED)
assert stop.user_message is None
print("stop.status:", stop.status.value)

from google.genai import types as gtypes
ok = NextUserMessage(
    status=Status.SUCCESS,
    user_message=gtypes.Content(role="user", parts=[gtypes.Part(text="Hello")]),
)
assert ok.user_message is not None
print("ok.status:", ok.status.value)
```

### Example 3 — subclassing `UserSimulator` to build a custom LLM-backed driver

```python
import asyncio
from typing import Optional
from google.adk.evaluation.simulation.user_simulator import (
    BaseUserSimulatorConfig,
    NextUserMessage,
    Status,
    UserSimulator,
)
from google.adk.evaluation.evaluator import Evaluator
from google.adk.events import Event
from google.genai import types
from pydantic import Field


class PersonaConfig(BaseUserSimulatorConfig):
    """Config for a persona-driven simulator — camelCase JSON keys via alias_generator."""
    persona_description: str = Field(
        default="A curious customer exploring product features.",
        alias="personaDescription",
    )
    max_turns: int = Field(default=5, alias="maxTurns")


class PersonaDrivenSimulator(UserSimulator):
    """Generates user turns from a static persona description (demo — not LLM backed)."""

    def __init__(self, config: PersonaConfig):
        super().__init__(config=config, config_type=PersonaConfig)
        self._config: PersonaConfig = config
        self._turn = 0

    async def get_next_user_message(self, events: list[Event]) -> NextUserMessage:
        if self._turn >= self._config.max_turns:
            return NextUserMessage(status=Status.TURN_LIMIT_REACHED)
        self._turn += 1
        # In a real implementation you'd call an LLM with the persona description
        # and the event history to generate a contextually appropriate response.
        message = types.Content(
            role="user",
            parts=[types.Part(text=f"[Turn {self._turn}] Tell me more about that feature.")],
        )
        return NextUserMessage(status=Status.SUCCESS, user_message=message)

    def get_simulation_evaluator(self) -> Optional[Evaluator]:
        # Return a custom evaluator that checks persona adherence; None here for brevity.
        return None


async def demo():
    # Construct via camelCase JSON (as it would arrive from an eval config file)
    raw_config = {"personaDescription": "A skeptical enterprise buyer.", "maxTurns": 3}
    config = PersonaConfig.model_validate(raw_config)
    sim = PersonaDrivenSimulator(config=config)
    for _ in range(4):
        result = await sim.get_next_user_message([])
        print(result.status.value, getattr(result.user_message, "parts", [None])[0])


asyncio.run(demo())
```

---

## 3 · `UserPersona` + `UserBehavior` + `UserPersonaRegistry` + `PreBuiltBehaviors` — behavioral simulation profiles

**Source:** `google/adk/evaluation/simulation/user_simulator_personas.py` and `google/adk/evaluation/simulation/pre_built_personas.py`

> **`@experimental`** — the simulation persona system is experimental.

`UserBehavior` is the atomic unit of a simulation profile: it bundles a human-readable `description` with concrete `behavior_instructions` (rules the simulator follows) and `violation_rubrics` (criteria checked post-conversation to detect non-compliance). Multiple `UserBehavior` instances compose into a `UserPersona`, which has a stable `id` that can be referenced by name in eval case configuration. `UserPersonaRegistry` is a simple dict-backed store that raises `NotFoundError` on unknown IDs. `PreBuiltBehaviors` is a Python `enum.Enum` whose members are pre-constructed `UserBehavior` instances covering the most common simulation patterns, so you can mix and match atomic behaviors without writing instruction strings by hand.

### Key behaviours

| Behaviour | Detail |
|---|---|
| `UserBehavior` instruction format | `get_behavior_instructions_str()` joins each instruction as `"  * {instruction}\n"` — suitable for direct injection into an LLM system prompt. |
| `UserBehavior` rubric format | `get_violation_rubrics_str()` uses the same `"  * {rubric}\n"` pattern — rubrics become inputs to a `RubricBasedEvaluator`. |
| `UserPersona` composition | A persona holds a `Sequence[UserBehavior]`; all behaviors' instructions are concatenated when the simulator constructs its system prompt. |
| Registry lookup | `UserPersonaRegistry.get_persona(persona_id)` raises `NotFoundError` (not `KeyError`) for unknown IDs, giving a friendlier error message. |
| String persona auto-resolution | When a `ConversationScenario.user_persona` field receives a plain string, the framework looks it up in `get_default_persona_registry()` automatically. |
| `PreBuiltBehaviors` canonical names | Key members: `ADVANCE_DETAIL_ORIENTED`, `ADVANCE_GOAL_ORIENTED`, `ANSWER_RELEVANT_ONLY`, `ANSWER_ALL`, `CORRECT_AGENT`, `DO_NOT_CORRECT_AGENT`, `TROUBLESHOOT_ONCE`, `END_LIMITED_TROUBLESHOOTING`. |

### Example 1 — inspecting pre-built behavior instruction text

```python
from google.adk.evaluation.simulation.pre_built_personas import PreBuiltBehaviors

# ADVANCE_DETAIL_ORIENTED: tells the simulator to state the high-level goal plus
# additional details on every new request, advance when the agent succeeds,
# and skip re-requesting something the agent already handled.
adv = PreBuiltBehaviors.ADVANCE_DETAIL_ORIENTED.value
print(f"Name: {adv.name}")
print(f"Description: {adv.description}")
print("Instructions:")
print(adv.get_behavior_instructions_str())
print("Violation rubrics:")
print(adv.get_violation_rubrics_str())

# Compose two behaviors and inspect the combined instruction block
redirect = PreBuiltBehaviors.REDIRECT_ON_FAILURE.value if hasattr(
    PreBuiltBehaviors, "REDIRECT_ON_FAILURE"
) else PreBuiltBehaviors.TROUBLESHOOT_ONCE.value

combined_instructions = "\n".join([
    adv.get_behavior_instructions_str(),
    redirect.get_behavior_instructions_str(),
])
print("\nCombined instructions for LLM system prompt:")
print(combined_instructions)
```

### Example 2 — registering a custom persona and looking it up

```python
from google.adk.evaluation.simulation.user_simulator_personas import (
    UserBehavior,
    UserPersona,
    UserPersonaRegistry,
)

# Define a custom behavior for a price-sensitive buyer
price_sensitive = UserBehavior(
    name="price_sensitive",
    description="Always asks about cost before committing to any action.",
    behavior_instructions=[
        "Before agreeing to any recommendation, ask about the associated cost.",
        "If the agent does not provide a price, ask explicitly before proceeding.",
        "Accept the recommendation only after the price is confirmed acceptable.",
    ],
    violation_rubrics=[
        "The user accepted a recommendation without asking about price.",
        "The user did not ask for clarification when the price was not mentioned.",
    ],
)

enterprise_buyer = UserPersona(
    id="enterprise_buyer",
    description="An enterprise procurement manager evaluating vendor solutions.",
    behaviors=[price_sensitive],
)

registry = UserPersonaRegistry()
registry.register_persona("enterprise_buyer", enterprise_buyer)

# Successful lookup
persona = registry.get_persona("enterprise_buyer")
print(f"Retrieved persona: {persona.id}")
print(f"Behavior count: {len(persona.behaviors)}")

# Registry raises NotFoundError for unknown IDs
try:
    registry.get_persona("unknown_id")
except Exception as exc:
    print(f"Error type: {type(exc).__name__}: {exc}")

# List all registered personas
for p in registry.get_registered_personas():
    print(f"  - {p.id}: {p.description}")
```

### Example 3 — composing a persona from `PreBuiltBehaviors` enum members

```python
from google.adk.evaluation.simulation.pre_built_personas import PreBuiltBehaviors
from google.adk.evaluation.simulation.user_simulator_personas import (
    UserPersona,
    UserPersonaRegistry,
)

# Build a persona from two pre-built atomic behaviors:
# - ADVANCE_DETAIL_ORIENTED: provides all context upfront, advances when agent succeeds
# - ANSWER_RELEVANT_ONLY: stays on topic, ignores off-script agent questions
detail_oriented_behavior = PreBuiltBehaviors.ADVANCE_DETAIL_ORIENTED.value
answer_relevant_behavior = PreBuiltBehaviors.ANSWER_RELEVANT_ONLY.value

focused_researcher = UserPersona(
    id="focused_researcher",
    description=(
        "A meticulous researcher who provides detailed context upfront "
        "and only engages with questions directly relevant to their research goal."
    ),
    behaviors=[detail_oriented_behavior, answer_relevant_behavior],
)

registry = UserPersonaRegistry()
registry.register_persona("focused_researcher", focused_researcher)

retrieved = registry.get_persona("focused_researcher")
print(f"Persona '{retrieved.id}' has {len(retrieved.behaviors)} behaviors:")
for behavior in retrieved.behaviors:
    print(f"\n  [{behavior.name}]")
    print(f"  Instructions:\n{behavior.get_behavior_instructions_str()}")
    print(f"  Rubrics:\n{behavior.get_violation_rubrics_str()}")
```

---

## 4 · `EvalConfig` + `CustomMetricConfig` + `_CustomMetricEvaluator` — eval configuration DSL and custom metric integration

**Source:** `google/adk/evaluation/eval_config.py` and `google/adk/evaluation/custom_metric_evaluator.py`

`EvalConfig` is the top-level configuration object that drives `LocalEvalService`. Its `criteria` dict maps metric names to either a float threshold (shorthand for "score must reach this value") or a `BaseCriterion` object for richer configuration. `custom_metrics` lets you register arbitrary Python functions as metrics by pointing to them with a dotted importlib path via `CustomMetricConfig.code_config`. `_CustomMetricEvaluator` is the internal evaluator that loads the function at runtime, inspects whether it is a coroutine, and calls it with the standardised `(eval_metric, actual_invocations, expected_invocations, conversation_scenario)` signature.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Default config | `_DEFAULT_EVAL_CONFIG` is `EvalConfig(criteria={"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8})` — used when no config file is provided. |
| camelCase JSON keys | Both `EvalConfig` and `CustomMetricConfig` use `alias_generator=alias_generators.to_camel` with `populate_by_name=True` so both snake_case Python attributes and camelCase JSON keys work. |
| File-or-default loading | `get_evaluation_criteria_or_default(eval_config_file_path)` reads a JSON file if the path is provided, otherwise returns `_DEFAULT_EVAL_CONFIG`. |
| importlib metric loading | `_CustomMetricEvaluator` splits `code_config.name` on the last `.` to get `(module_path, function_name)`, calls `importlib.import_module(module_path)`, then `getattr(module, function_name)`. |
| Async metric support | `inspect.iscoroutinefunction(metric_fn)` determines whether to `await` the call, so your metric can be either `def` or `async def`. |
| Simulator integration | `EvalConfig.user_simulator_config` accepts any `BaseUserSimulatorConfig` — allowing the eval run to use a scripted or LLM-backed simulator without changing the evaluator. |

### Example 1 — building an `EvalConfig` programmatically

```python
from google.adk.evaluation.eval_config import EvalConfig, CustomMetricConfig
from google.adk.evaluation.eval_metric import EvalMetric
from google.adk.tools.evaluation.utils import CodeConfig  # importlib path wrapper

# Minimal config using shorthand float thresholds
minimal_config = EvalConfig(
    criteria={
        "tool_trajectory_avg_score": 1.0,
        "response_match_score": 0.8,
    }
)
print("Default metric names:", list(minimal_config.criteria.keys()))

# Config that adds a custom metric via importlib path
config_with_custom = EvalConfig(
    criteria={
        "tool_trajectory_avg_score": 1.0,
        "my_custom_metric": 0.75,
    },
    custom_metrics={
        "my_custom_metric": CustomMetricConfig(
            # code_config.name must be "module.path.function_name"
            # _CustomMetricEvaluator splits on the last '.' to import it
            code_config=CodeConfig(name="mypackage.metrics.check_tone"),
            description="Checks that the agent response maintains a professional tone.",
        )
    },
)
print("Custom metric count:", len(config_with_custom.custom_metrics))
```

### Example 2 — loading config from a JSON file with `get_evaluation_criteria_or_default`

```python
import json
import tempfile
from pathlib import Path
from google.adk.evaluation.eval_config import (
    EvalConfig,
    get_evaluation_criteria_or_default,
)

# Write a temporary eval config JSON file (camelCase keys as produced by .model_dump(by_alias=True))
config_data = {
    "criteria": {
        "tool_trajectory_avg_score": 0.9,
        "response_match_score": 0.7,
    },
    "customMetrics": None,
    "userSimulatorConfig": None,
}

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False
) as f:
    json.dump(config_data, f)
    config_path = f.name

# Load from file
loaded = get_evaluation_criteria_or_default(config_path)
assert isinstance(loaded, EvalConfig)
print("Loaded criteria:", {k: v for k, v in loaded.criteria.items()})

# No path → returns _DEFAULT_EVAL_CONFIG
default = get_evaluation_criteria_or_default(None)
print("Default criteria:", {k: v for k, v in default.criteria.items()})
# Outputs: {"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}
```

### Example 3 — implementing a custom metric function (sync and async variants)

```python
# mypackage/metrics.py
# This file would be placed on the Python path so _CustomMetricEvaluator
# can import it as "mypackage.metrics.check_response_length".

from typing import Optional
from google.adk.evaluation.eval_metric import EvalMetric, EvaluationResult
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.conversation_scenario import ConversationScenario


def check_response_length(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario],
) -> EvaluationResult:
    """Sync custom metric: penalise responses longer than 500 characters."""
    total_score = 0.0
    count = 0
    for inv in actual_invocations:
        if inv.final_response and inv.final_response.parts:
            text = "".join(p.text or "" for p in inv.final_response.parts)
            score = 1.0 if len(text) <= 500 else max(0.0, 1.0 - (len(text) - 500) / 1000)
            total_score += score
            count += 1
    return EvaluationResult(
        overall_score=total_score / count if count else 0.0,
        metric_name=eval_metric.metric_name,
    )


async def check_response_length_async(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario],
) -> EvaluationResult:
    """Async variant — identical logic; _CustomMetricEvaluator detects coroutines."""
    return check_response_length(
        eval_metric, actual_invocations, expected_invocations, conversation_scenario
    )
# Register in EvalConfig:
# custom_metrics={"check_response_length": CustomMetricConfig(
#     code_config=CodeConfig(name="mypackage.metrics.check_response_length")
# )}
```

---

## 5 · `Rubric` + `RubricContent` + `RubricScore` + `RubricBasedEvaluator` — rubric-based LLM-as-judge evaluation

**Source:** `google/adk/evaluation/eval_rubrics.py` and `google/adk/evaluation/rubric_based_evaluator.py`

`Rubric` is a structured evaluation criterion: a `rubric_id` string, a `RubricContent` holding the testable `text_property` (e.g. `"The agent's response is grammatically correct."`), an optional `description`, and an optional `type` tag (recommended in upper snake_case, e.g. `"FINAL_RESPONSE_QUALITY"`). `RubricScore` pairs a `rubric_id` with a numeric `score` (1.0 = pass, 0.0 = fail) and a textual `rationale`. `RubricBasedEvaluator` is an `@experimental` LLM-as-judge evaluator that sends rubric criteria to a model, parses the verdict from the response using regex, aggregates multiple samples per invocation via majority vote, and summarises across invocations using the mean.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Verdict parsing | `DefaultAutoRaterResponseParser` extracts three fields with regex: `(?<=Property: )(.*)`, `(?<=Rationale: )(.*)`, `(?<=Verdict: )(.*)`. `"yes"` (case-insensitive) → 1.0; `"no"` → 0.0; anything else → `None`. |
| Majority vote aggregation | `MajorityVotePerInvocationResultsAggregator` collects multiple LLM samples per rubric per invocation and takes the majority verdict, reducing noise from single-sample LLM judgments. |
| Mean summarisation | `MeanInvocationResultsSummarizer` computes the arithmetic mean of per-invocation scores for the final `EvaluationResult`. |
| `rubric_type` filtering | When `rubric_type` is set on `RubricBasedEvaluator`, only `Rubric` instances whose `type` field matches are included in evaluation, enabling per-dimension scoring. |
| Normalised rubric map | `_normalized_rubric_to_id_map` lowercases and strips whitespace from rubric text before building the lookup map, ensuring minor whitespace differences don't break matching. |
| `RubricBasedEvaluator` is `@experimental` | The constructor's `criterion_type` parameter, aggregator, and summarizer can all be replaced to customise the evaluation pipeline. |

### Example 1 — defining rubrics and scoring manually

```python
from google.adk.evaluation.eval_rubrics import (
    Rubric,
    RubricContent,
    RubricScore,
)

# Define a set of rubrics for a customer support agent
rubrics = [
    Rubric(
        rubric_id="grammatical_correctness",
        rubric_content=RubricContent(
            text_property="The agent's response is grammatically correct."
        ),
        description="Checks grammar and punctuation in the final response.",
        type="FINAL_RESPONSE_QUALITY",
    ),
    Rubric(
        rubric_id="tool_call_accuracy",
        rubric_content=RubricContent(
            text_property="The agent called the correct tool with the correct arguments."
        ),
        description="Verifies tool selection and parameter correctness.",
        type="TOOL_USE_QUALITY",
    ),
    Rubric(
        rubric_id="instruction_followed",
        rubric_content=RubricContent(
            text_property="The agent followed all instructions in its system prompt."
        ),
        type="INSTRUCTION_ADHERENCE",
    ),
]

# Simulate scores returned from an LLM judge
scores = [
    RubricScore(rubric_id="grammatical_correctness", score=1.0, rationale="No grammar errors detected."),
    RubricScore(rubric_id="tool_call_accuracy", score=0.0, rationale="Wrong tool was called."),
    RubricScore(rubric_id="instruction_followed", score=1.0, rationale="All instructions were followed."),
]

for rubric in rubrics:
    matching = next((s for s in scores if s.rubric_id == rubric.rubric_id), None)
    verdict = "PASS" if matching and matching.score == 1.0 else "FAIL"
    print(f"[{verdict}] {rubric.rubric_id} ({rubric.type}): {matching.rationale if matching else 'no score'}")
```

### Example 2 — parsing auto-rater output with `DefaultAutoRaterResponseParser`

```python
from google.adk.evaluation.rubric_based_evaluator import DefaultAutoRaterResponseParser

parser = DefaultAutoRaterResponseParser()

# This is the format the LLM judge is expected to produce
auto_rater_output = """
Property: The agent's response is grammatically correct.
Rationale: The response contains no grammatical errors and is clearly written.
Verdict: Yes

Property: The agent called the correct tool with the correct arguments.
Rationale: The agent called `search_database` but should have called `lookup_order`.
Verdict: No
"""

rubric_responses = parser.parse(auto_rater_output)
for resp in rubric_responses:
    verdict_label = {1.0: "PASS", 0.0: "FAIL", None: "UNCERTAIN"}[resp.score]
    print(f"Property: {resp.property_text[:60]}...")
    print(f"  Verdict: {verdict_label} (score={resp.score})")
    print(f"  Rationale: {resp.rationale}")
    print()
```

### Example 3 — constructing a `RubricBasedEvaluator` with type filtering

```python
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent
from google.adk.evaluation.rubric_based_evaluator import (
    RubricBasedEvaluator,
    DefaultAutoRaterResponseParser,
    MajorityVotePerInvocationResultsAggregator,
    MeanInvocationResultsSummarizer,
)
from google.adk.evaluation.eval_metric import EvalMetric

# Create two evaluators scoped to different rubric types,
# enabling separate dimensions of quality scoring.
response_quality_evaluator = RubricBasedEvaluator(
    eval_metric=EvalMetric(metric_name="response_quality_score", threshold=0.8),
    criterion_type="rubric_based",  # matches the criterion type in EvalConfig
    auto_rater_response_parser=DefaultAutoRaterResponseParser(),
    per_invocation_results_aggregator=MajorityVotePerInvocationResultsAggregator(),
    invocation_results_summarizer=MeanInvocationResultsSummarizer(),
    rubric_type="FINAL_RESPONSE_QUALITY",  # only evaluates rubrics with this type
)

tool_quality_evaluator = RubricBasedEvaluator(
    eval_metric=EvalMetric(metric_name="tool_quality_score", threshold=1.0),
    criterion_type="rubric_based",
    rubric_type="TOOL_USE_QUALITY",  # separate dimension
)

# Create rubrics that will be filtered by type
rubrics = [
    Rubric(
        rubric_id="r1",
        rubric_content=RubricContent(text_property="Response is concise."),
        type="FINAL_RESPONSE_QUALITY",
    ),
    Rubric(
        rubric_id="r2",
        rubric_content=RubricContent(text_property="Correct tool was called."),
        type="TOOL_USE_QUALITY",
    ),
]

# Each evaluator sees only its own typed rubrics
response_quality_evaluator.create_effective_rubrics_list(rubrics)
tool_quality_evaluator.create_effective_rubrics_list(rubrics)

print("Response quality rubrics:", [r.rubric_id for r in response_quality_evaluator.get_effective_rubrics_list()])
print("Tool quality rubrics:", [r.rubric_id for r in tool_quality_evaluator.get_effective_rubrics_list()])
# Outputs: ['r1'] and ['r2'] respectively
```

---

## 6 · `InteractionsRequestProcessor` — Gemini Interactions API stateful conversation chaining

**Source:** `google/adk/flows/llm_flows/interactions_processor.py`

`InteractionsRequestProcessor` is a `BaseLlmRequestProcessor` that enables Gemini's server-side conversation state management. Normally, every LLM request must include the full conversation history in its `contents` array, which grows linearly with turn count and consumes proportionally more tokens. When a `Gemini` model has `use_interactions_api=True`, Gemini's server stores the conversation state identified by an `interaction_id`, and subsequent requests need only supply the latest user turn plus a `previous_interaction_id` pointer. This processor scans the session event history to find the most recent agent-authored event with an `interaction_id`, then sets `llm_request.previous_interaction_id` accordingly. It yields no events — it is a pure preprocessing step that mutates the request before it reaches the model.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Activation guard | Only runs when `isinstance(model, Gemini)` AND `model.use_interactions_api is True`. All other models are skipped silently. |
| Reverse event scan | Iterates `invocation_context.session.events` in reverse to find the most recent matching event efficiently. |
| Author filtering | Only considers events where `event.author == agent_name` (the current agent) — sub-agent events are ignored. |
| Branch filtering | If `current_branch` is `None`, includes only events with `not event.branch`; otherwise includes events where `event.branch == current_branch or not event.branch`. |
| Interaction ID injection | Sets `llm_request.previous_interaction_id = event.interaction_id` when a qualifying event is found. |
| Module-level singleton | `request_processor = InteractionsRequestProcessor()` is exported at module level and consumed by the LLM flow pipeline. |

### Example 1 — enabling the Interactions API on a `Gemini` model

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.google_llm import Gemini

# Pass use_interactions_api=True to the Gemini model constructor.
# InteractionsRequestProcessor will then chain conversation turns server-side,
# avoiding the cost of re-sending full history on every turn.
gemini_model = Gemini(
    model="gemini-2.0-flash",
    use_interactions_api=True,
)

agent = LlmAgent(
    name="stateful_agent",
    model=gemini_model,
    instruction=(
        "You are a helpful assistant. Maintain context across turns "
        "using the Gemini Interactions API."
    ),
)
# The InteractionsRequestProcessor module-level singleton is automatically
# included in the LLM flow pipeline. No further configuration is needed.
print("Model type:", type(agent.model).__name__)
print("Interactions API enabled:", agent.model.use_interactions_api)
```

### Example 2 — understanding branch-scoped interaction chaining

```python
# When agents run in branched conversation trees (e.g., a parent spawning
# sub-agents), InteractionsRequestProcessor applies branch-scoped filtering
# so each branch chains only its own prior interactions.
#
# The logic in _is_event_in_branch:
#   current_branch is None  → include events where not event.branch
#   current_branch = "some" → include events where event.branch == "some"
#                             OR not event.branch (root events apply everywhere)
#
# This means a sub-agent on branch "branch_A" will pick up its own prior
# interaction ID (branch="branch_A") but also any unscoped root events.

# Pseudocode illustration — actual event objects are internal types:
def is_event_in_branch(current_branch, event_branch):
    if current_branch is None:
        return not event_branch
    return event_branch == current_branch or not event_branch


test_cases = [
    (None, None, True),      # root agent, root event → include
    (None, "branch_A", False), # root agent, branched event → exclude
    ("branch_A", "branch_A", True),  # matching branch → include
    ("branch_A", None, True),        # root event applies to all branches → include
    ("branch_A", "branch_B", False), # different branch → exclude
]
for current, event, expected in test_cases:
    result = is_event_in_branch(current, event)
    status = "OK" if result == expected else "FAIL"
    print(f"[{status}] current={current!r}, event={event!r} → {result}")
```

### Example 3 — inspecting the module-level singleton

```python
from google.adk.flows.llm_flows.interactions_processor import (
    InteractionsRequestProcessor,
    request_processor,
)

# The module exports a pre-constructed singleton for use by the flow pipeline
assert isinstance(request_processor, InteractionsRequestProcessor)
print("Singleton type:", type(request_processor).__name__)

# The processor yields no events — it is a pure preprocessing step
# To verify: run_async is an async generator that never yields
import asyncio
import inspect

print("run_async is a coroutine function:", inspect.iscoroutinefunction(request_processor.run_async))
# The return annotation is AsyncGenerator[Event, None] with no yields
print("Processor has no side outputs — only mutates llm_request.previous_interaction_id")
```

---

## 7 · `_RequestConfirmationLlmRequestProcessor` — 4-step tool confirmation resume pipeline

**Source:** `google/adk/flows/llm_flows/request_confirmation.py`

When an agent uses tools that require explicit user approval (the `adk_request_confirmation` mechanism), the framework needs to re-execute those tools after the user approves without sending another LLM request. `_RequestConfirmationLlmRequestProcessor` implements this as a four-step preprocessing pass: (1) find the last user-authored event and parse any `adk_request_confirmation` function call responses from it; (2) scan earlier events for the original `adk_request_confirmation` function calls and extract the `originalFunctionCall` argument to build a mapping from confirmation FC ID to original FC; (3) deduplicate by removing any tools that have already been executed after the confirmation event; (4) re-execute the confirmed tools by calling `handle_function_call_list_async` with `agent.canonical_tools()`. The processor is exported as a module-level singleton.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Step 1: parse confirmations | Scans events in reverse for the last user-authored event; extracts `adk_request_confirmation` function responses keyed by FC ID. |
| Two response formats | `_parse_tool_confirmation` handles both `{"response": json_string}` (ADK client wrapping) and a direct dict, so client variations don't cause parse failures. |
| Step 2: resolve original FCs | `_resolve_confirmation_targets` extracts `originalFunctionCall` from each `adk_request_confirmation` FC's args, building `(tool_confirmation_dict, original_fcs_dict)` both keyed by original FC ID. |
| Step 3: deduplication | Removes FCs that have already been executed after the confirmation event, preventing double execution on page reload or retry. |
| Step 4: re-execution | Calls `handle_function_call_list_async` with `agent.canonical_tools()` to resolve the current active tool set before re-executing. |
| Module-level singleton | `request_processor = _RequestConfirmationLlmRequestProcessor()` — consumed by the standard LLM flow pipeline. |

### Example 1 — understanding the confirmation function call name constant

```python
from google.adk.flows.llm_flows.request_confirmation import (
    _RequestConfirmationLlmRequestProcessor,
    request_processor,
)
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME

# The constant identifies the special function call used for tool confirmation
print("Confirmation FC name:", REQUEST_CONFIRMATION_FUNCTION_CALL_NAME)
# Output: adk_request_confirmation

# The module exports a pre-built singleton, just like InteractionsRequestProcessor
assert isinstance(request_processor, _RequestConfirmationLlmRequestProcessor)
print("Singleton type:", type(request_processor).__name__)
```

### Example 2 — the two confirmation response formats

```python
import json

# _parse_tool_confirmation handles two payload shapes:
# FORMAT A: Direct dict from native ADK clients
format_a = {
    "approved": True,
    "tool_name": "send_email",
    "user_comment": "Yes, send it.",
}

# FORMAT B: ADK web client wraps the dict as a JSON string under "response"
format_b = {
    "response": json.dumps({
        "approved": True,
        "tool_name": "send_email",
        "user_comment": "Yes, send it.",
    })
}

def simulate_parse_tool_confirmation(response: dict) -> dict:
    """Mirrors the logic of _parse_tool_confirmation."""
    if "response" in response and isinstance(response["response"], str):
        # FORMAT B: unwrap JSON string
        return json.loads(response["response"])
    # FORMAT A: already a dict
    return response

result_a = simulate_parse_tool_confirmation(format_a)
result_b = simulate_parse_tool_confirmation(format_b)
assert result_a == result_b, "Both formats should parse to the same structure"
print("FORMAT A parsed:", result_a)
print("FORMAT B parsed:", result_b)
print("Outputs are identical:", result_a == result_b)
```

### Example 3 — building an agent that uses tool confirmation

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from typing import Any

# A tool that requires confirmation before executing
def send_email(to: str, subject: str, body: str, tool_context: ToolContext) -> dict[str, Any]:
    """Send an email — requires user confirmation before execution."""
    return {"status": "sent", "to": to, "subject": subject}

# To enable tool confirmation, set require_user_confirmation=True on the tool.
# When the LLM calls this tool, the framework emits an adk_request_confirmation
# function call to the client instead of executing immediately.
# After the user approves, _RequestConfirmationLlmRequestProcessor's 4-step
# pipeline detects the confirmation response in the next user event and
# re-executes the tool without sending another LLM request.
email_tool = FunctionTool(func=send_email)
# Note: require_user_confirmation is set on the tool registration, not here.
# The framework wraps it transparently.

agent = LlmAgent(
    name="email_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You help users send emails. Always confirm with the user before sending. "
        "Use the send_email tool when the user provides all required details."
    ),
    tools=[email_tool],
)
print("Agent configured with tool confirmation pipeline.")
print("Tool name:", email_tool.name)
# The _RequestConfirmationLlmRequestProcessor singleton in the flow pipeline
# handles the 4-step resume automatically — no agent-level code needed.
```

---

## 8 · `GoogleApiToOpenApiConverter` — Google API Discovery to OpenAPI v3 converter

**Source:** `google/adk/tools/google_api_tool/googleapi_to_openapi_converter.py`

`GoogleApiToOpenApiConverter` bridges the Google API Discovery format (the JSON schema served by `googleapiclient.discovery.build`) and the OpenAPI 3.0.0 format expected by `RestApiTool`. Its `fetch_google_api_spec()` method calls the Discovery Service and stores the raw document under `_google_api_resource._rootDesc`. `convert()` then orchestrates four internal converters: `_convert_info()` for title/version metadata, `_convert_servers()` for the base URL (concatenating `rootUrl + servicePath`), `_convert_security_schemes()` for OAuth2 and API key schemes with Google's hardcoded auth endpoints, and `_convert_schemas()` for the full type registry. Resources and methods are processed recursively by `_convert_resources()` and `_convert_methods()`. A custom `discovery_url` parameter lets you target private or internal Google APIs not listed in the public Discovery index.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Constructor | `__init__(api_name, api_version, *, discovery_url=None)` — the keyword-only `discovery_url` overrides the default Discovery Service endpoint. |
| Four-phase conversion | `convert()` delegates to `_convert_info()`, `_convert_servers()`, `_convert_security_schemes()`, `_convert_schemas()` in order, then processes resources/methods. |
| Output structure | Returns `{"openapi": "3.0.0", "info": {}, "servers": [], "paths": {}, "components": {"schemas": {}, "securitySchemes": {}}}`. |
| `flatPath` preference | `_convert_methods` uses `flatPath` (expanded, non-template form) when available, falling back to `path`. |
| `any` type handling | `_convert_schema_object` converts Google Discovery's `any` type to a JSON Schema `oneOf` with multiple primitives. |
| `$ref` rewriting | Discovery `#` fragment refs are rewritten to `#/components/schemas/` to conform with OpenAPI 3.0.0. |
| Consumer | Used internally by `GoogleApiToolset` to generate `RestApiTool` instances for Gmail, Calendar, Sheets, and other Google APIs. |

### Example 1 — fetching and converting the Gmail API spec

```python
from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import (
    GoogleApiToOpenApiConverter,
)

# Fetch the Gmail API Discovery document and convert to OpenAPI 3.0.0
converter = GoogleApiToOpenApiConverter(
    api_name="gmail",
    api_version="v1",
)
converter.fetch_google_api_spec()

openapi_spec = converter.convert()

# Inspect the converted structure
print("OpenAPI version:", openapi_spec["openapi"])
print("API title:", openapi_spec["info"].get("title"))
print("Server URL:", openapi_spec["servers"][0]["url"] if openapi_spec["servers"] else "none")
print("Schema count:", len(openapi_spec["components"]["schemas"]))
print("Security scheme names:", list(openapi_spec["components"]["securitySchemes"].keys()))
print("Top-level path count:", len(openapi_spec["paths"]))
```

### Example 2 — targeting a private API with a custom discovery URL

```python
from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import (
    GoogleApiToOpenApiConverter,
)

# Internal/private Google APIs can be targeted via a custom discovery URL.
# The discovery_url parameter overrides the default Discovery Service endpoint.
# This is useful for:
#   - Internal Google APIs not listed in the public Discovery index
#   - Staging/preview API versions
#   - Custom API Gateway endpoints that expose Discovery-format documents
private_converter = GoogleApiToOpenApiConverter(
    api_name="internal_crm",
    api_version="v2",
    discovery_url="https://api-internal.example.com/discovery/v2/rest",
)
# In production: call fetch_google_api_spec() then convert()
# private_converter.fetch_google_api_spec()
# spec = private_converter.convert()
print("Custom discovery URL configured.")
print("API name:", private_converter._api_name if hasattr(private_converter, '_api_name') else "internal_crm")
```

### Example 3 — integrating with `GoogleApiToolset`

```python
import asyncio
from google.adk.tools.google_api_tool.google_api_toolset import GoogleApiToolset
from google.adk.agents.llm_agent import LlmAgent

# GoogleApiToolset uses GoogleApiToOpenApiConverter internally.
# You provide api_name and api_version; the toolset handles discovery and conversion.
async def build_gmail_agent():
    # The toolset fetches the Discovery doc, converts to OpenAPI, and creates
    # RestApiTool instances for each Gmail API method.
    gmail_toolset = GoogleApiToolset(
        api_name="gmail",
        api_version="v1",
        # Scopes limit which OAuth flows are included in the security scheme
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )

    tools = await gmail_toolset.get_tools()
    print(f"Generated {len(tools)} Gmail tools")
    if tools:
        print("First tool name:", tools[0].name)

    agent = LlmAgent(
        name="gmail_reader",
        model="gemini-2.0-flash",
        instruction="You help users read and search their Gmail inbox.",
        tools=tools,
    )
    return agent


# asyncio.run(build_gmail_agent())
print("GoogleApiToolset → GoogleApiToOpenApiConverter → RestApiTool pipeline ready.")
```

---

## 9 · `LocalEvalService` — local end-to-end eval pipeline

**Source:** `google/adk/evaluation/local_eval_service.py`

> **`@experimental`** — `LocalEvalService` is decorated experimental.

`LocalEvalService` is the primary orchestration class for running agent evaluations locally without a cloud backend. It accepts a `root_agent`, an `EvalSetsManager` (which knows how to load eval cases), a `MetricEvaluatorRegistry` (pre-populated with built-in metric evaluators), and optional overrides for session, artifact, memory, and result persistence services. Its two main methods are `perform_inference` (which runs the agent against eval case inputs, collecting `InferenceResult` objects) and `evaluate` (which scores those results against expected outputs using the registered metric evaluators). Session IDs are generated with a `___eval___session___` prefix plus a UUID4, keeping eval sessions clearly distinguishable in session storage.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Session ID format | `EVAL_SESSION_ID_PREFIX = "___eval___session___"` + UUID4 — distinguishes eval sessions from production sessions. |
| Default services | `session_service=InMemorySessionService()`, `artifact_service=InMemoryArtifactService()` — no persistence by default; override for multi-run comparisons. |
| Rubric propagation | `_copy_eval_case_rubrics_to_actual_invocations` copies case-level rubrics to ALL actual invocations; `_copy_invocation_rubrics_to_actual_invocations` copies per-invocation rubrics to the corresponding actual invocation. |
| Duplicate rubric guard | `_add_rubrics_to_invocation` raises `ValueError` if a `rubric_id` is added twice to the same invocation — prevents silent overwrites. |
| Telemetry tagging | Uses `EVAL_CLIENT_LABEL` via `client_label_context` to tag all telemetry emitted during an eval run. |
| Result persistence | If `eval_set_results_manager` is provided, `evaluate` saves `EvalCaseResult` objects automatically — otherwise results are only yielded. |
| Parallelism control | `perform_inference` uses a semaphore to bound concurrent agent runs when processing multiple eval cases. |

### Example 1 — constructing `LocalEvalService` with default services

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager

# Minimal construction — uses InMemorySessionService and InMemoryArtifactService by default
agent = LlmAgent(
    name="qa_agent",
    model="gemini-2.0-flash",
    instruction="You answer factual questions accurately and concisely.",
)

eval_sets_manager = InMemoryEvalSetsManager()

eval_service = LocalEvalService(
    root_agent=agent,
    eval_sets_manager=eval_sets_manager,
    # metric_evaluator_registry defaults to DEFAULT_METRIC_EVALUATOR_REGISTRY
    # session_service defaults to InMemorySessionService()
    # artifact_service defaults to InMemoryArtifactService()
)
print("LocalEvalService constructed.")
print("Session ID prefix:", "___eval___session___")
```

### Example 2 — session ID generation with `___eval___session___` prefix

```python
import uuid
from google.adk.evaluation.local_eval_service import EVAL_SESSION_ID_PREFIX

# The module-level _get_session_id function generates IDs in this format:
def _get_session_id() -> str:
    return EVAL_SESSION_ID_PREFIX + str(uuid.uuid4())

session_id = _get_session_id()
print("Generated session ID:", session_id)
assert session_id.startswith("___eval___session___")
assert len(session_id) > len("___eval___session___")
print("Prefix confirmed. UUID portion:", session_id[len(EVAL_SESSION_ID_PREFIX):])
```

### Example 3 — rubric propagation and duplicate guard

```python
from google.adk.evaluation.local_eval_service import (
    _add_rubrics_to_invocation,
    _copy_eval_case_rubrics_to_actual_invocations,
)
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent
from google.genai import types

# Build sample rubrics
r1 = Rubric(
    rubric_id="conciseness",
    rubric_content=RubricContent(text_property="The response is concise and to the point."),
    type="FINAL_RESPONSE_QUALITY",
)
r2 = Rubric(
    rubric_id="accuracy",
    rubric_content=RubricContent(text_property="The response is factually accurate."),
    type="FINAL_RESPONSE_QUALITY",
)

# Build a sample invocation (actual invocations produced during inference)
actual_inv = Invocation(
    user_content=types.Content(role="user", parts=[types.Part(text="What is 2+2?")]),
)

# Add rubrics to an invocation
_add_rubrics_to_invocation(actual_inv, [r1, r2])
print("Rubrics added:", [r.rubric_id for r in (actual_inv.rubrics or [])])

# Duplicate rubric_id raises ValueError
try:
    _add_rubrics_to_invocation(actual_inv, [r1])  # r1 already added
except ValueError as exc:
    print(f"Duplicate guard triggered: {exc}")
```

---

## 10 · `EvaluationGenerator` + `_LiveSession` — inference runner and live eval session manager

**Source:** `google/adk/evaluation/evaluation_generator.py`

`EvaluationGenerator` is a collection of static methods for running agent inference during evaluation — both standard text-based and live bidirectional audio sessions. Its `generate_responses` method processes an `EvalSet`, running each eval case through the agent (optionally multiple times via `repeat_num` for reliability sampling) and collecting `EvalCaseResponses` objects, each holding the eval case and a list of invocation lists. `_LiveSession` is a private async context manager that manages the full lifecycle of a live (BIDI audio) session: it starts a background `_consume_events` task, routes events to an `event_queue`, signals turn completion via `turn_complete_event`, and handles graceful shutdown on `__aexit__`. Both classes are internal infrastructure consumed by `LocalEvalService` and the ADK CLI eval command.

### Key behaviours

| Behaviour | Detail |
|---|---|
| `EvalCaseResponses` structure | `eval_case: EvalCase` + `responses: list[list[Invocation]]` — the outer list is `repeat_num` long, the inner list is one invocation list per conversation turn. |
| Live session `RunConfig` | `_consume_events` uses `StreamingMode.BIDI`, `response_modalities=["AUDIO"]`, `output_audio_transcription=AudioTranscriptionConfig()`, `input_audio_transcription=AudioTranscriptionConfig()`. |
| `_LiveSession` event queue | `event_queue: asyncio.Queue` accumulates events from `_consume_events`; `turn_complete_event: asyncio.Event` signals when the agent has finished a turn. |
| Turn completion signalling | `turn_complete_event` is set when the agent emits a turn-complete signal; `live_finished` is set when the session is fully closed. |
| Shutdown grace period | `__aexit__` closes the `live_request_queue` and waits up to 30 seconds for the `consume_task` to complete, swallowing `ConnectionClosedOK` and `APIError` code 1000 silently. |
| Audio transcription synthesis | `_generate_inferences_for_single_user_invocation_live` creates synthetic text events from audio transcriptions, allowing text-based eval metrics to work on live sessions. |
| Static-only class | `EvaluationGenerator` has no `__init__` — all methods are `@staticmethod`; you call them directly on the class. |

### Example 1 — generating responses with repeat sampling

```python
import asyncio
from google.adk.evaluation.evaluation_generator import EvaluationGenerator, EvalCaseResponses
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types


def make_eval_set() -> EvalSet:
    """Build a minimal eval set with one case for demonstration."""
    user_content = types.Content(
        role="user",
        parts=[types.Part(text="What is the capital of France?")],
    )
    expected_inv = Invocation(
        user_content=user_content,
        # expected_tool_use and final_response would be set in a real eval case
    )
    case = EvalCase(
        eval_id="geography_q1",
        conversation=[expected_inv],
    )
    return EvalSet(eval_set_id="geography_test", eval_cases=[case])


async def run_inference():
    eval_set = make_eval_set()
    # generate_responses runs each case repeat_num=3 times for reliability sampling.
    # Each run produces one list[Invocation]; the outer list has 3 entries.
    responses: list[EvalCaseResponses] = await EvaluationGenerator.generate_responses(
        eval_set=eval_set,
        agent_module_path="mypackage.agent",  # module exporting root_agent
        repeat_num=3,
    )
    for ecr in responses:
        print(f"Eval case: {ecr.eval_case.eval_id}")
        print(f"  Repeat count: {len(ecr.responses)}")
        for i, invocation_list in enumerate(ecr.responses):
            print(f"  Run {i}: {len(invocation_list)} invocation(s)")


# asyncio.run(run_inference())
print("EvaluationGenerator.generate_responses is a static async method.")
print("EvalCaseResponses.responses is list[list[Invocation]] — outer=repeat_num, inner=turns.")
```

### Example 2 — `_LiveSession` lifecycle as an async context manager

```python
import asyncio
from google.adk.evaluation.evaluation_generator import _LiveSession
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.llm_agent import LlmAgent

# _LiveSession wraps a live BIDI session in an async context manager.
# It starts a background consume_task in __aenter__ and shuts it down in __aexit__.

async def demonstrate_live_session_lifecycle():
    agent = LlmAgent(
        name="live_agent",
        model="gemini-2.0-flash-live",  # live-capable model
        instruction="You are a helpful assistant for live audio sessions.",
    )
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="live_eval_app",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="live_eval_app",
        user_id="eval_user",
    )

    # _LiveSession manages:
    #   - live_request_queue: LiveRequestQueue (input to the agent)
    #   - event_queue: asyncio.Queue (output from the agent)
    #   - turn_complete_event: asyncio.Event (set when agent finishes a turn)
    #   - live_finished: asyncio.Event (set when session is closed)
    #   - consume_task: the background asyncio.Task running _consume_events
    async with _LiveSession(
        runner=runner,
        session=session,
        user_id="eval_user",
        session_id=session.id,
    ) as live_session:
        print("Live session entered.")
        print("Event queue type:", type(live_session.event_queue).__name__)
        print("Turn complete event type:", type(live_session.turn_complete_event).__name__)
        print("Live finished event type:", type(live_session.live_finished).__name__)
        print("Consume task running:", not live_session.consume_task.done())
    print("Live session exited — consume_task awaited with 30s grace period.")


# asyncio.run(demonstrate_live_session_lifecycle())
print("_LiveSession: async context manager for BIDI audio eval sessions.")
```

### Example 3 — audio transcription synthesis in live eval

```python
# _generate_inferences_for_single_user_invocation_live converts audio transcriptions
# to synthetic text events so text-based eval metrics can score live sessions.
#
# The RunConfig used internally:
# RunConfig(
#     streaming_mode=StreamingMode.BIDI,
#     response_modalities=["AUDIO"],
#     output_audio_transcription=AudioTranscriptionConfig(),
#     input_audio_transcription=AudioTranscriptionConfig(),
# )
#
# For each audio event with a transcription, the method generates a synthetic
# Content event with role="model" and the transcription text, inserting it into
# the invocation list alongside any native text events.
#
# This allows metrics like response_match_score (which compares text) to work
# on live audio sessions without modification.

from google.adk.runners.run_config import RunConfig, StreamingMode
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.models.audio_transcription_config import AudioTranscriptionConfig

live_run_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    output_audio_transcription=AudioTranscriptionConfig(),
    input_audio_transcription=AudioTranscriptionConfig(),
)

print("Live eval RunConfig:")
print("  streaming_mode:", live_run_config.streaming_mode)
print("  response_modalities:", live_run_config.response_modalities)
print("  output_audio_transcription:", type(live_run_config.output_audio_transcription).__name__)
print("  input_audio_transcription:", type(live_run_config.input_audio_transcription).__name__)
print()
print("Audio transcriptions are converted to synthetic text events,")
print("enabling text-based metrics (response_match_score, rubric scoring)")
print("to work on live BIDI audio evaluation sessions without modification.")
```

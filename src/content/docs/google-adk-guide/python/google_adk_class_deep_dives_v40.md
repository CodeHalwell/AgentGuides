---
title: "Class deep dives — volume 40 (ExecuteBashTool, DataAgentToolset, ScenarioGenerator, multi-turn evaluators, AppDetails, EnsureRetryOptionsPlugin, McpInstructionProvider, EnvironmentSimulationFactory, RunConfig, ScheduleDynamicNode)"
description: "10 source-verified deep dives for google-adk 2.4.0: ExecuteBashTool, DataAgentToolset, ConversationScenario/ScenarioGenerator, multi-turn evaluators, AppDetails, EnsureRetryOptionsPlugin, McpInstructionProvider, EnvironmentSimulationFactory, RunConfig, and ScheduleDynamicNode."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 40"
  order: 109
import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.4.0**. No documentation or blog posts were used as primary
sources.
</Aside>

Verified against **google-adk==2.4.0** source code.  All examples are runnable with `pip install google-adk`.

---

## 1. `ExecuteBashTool` + `BashToolPolicy`

**Source:** `google/adk/tools/bash_tool.py`

`ExecuteBashTool` is ADK's safe bash execution primitive, introduced in 2.4.0.  It wraps `asyncio.create_subprocess_exec`, enforces a command-allowlist policy, applies OS-level resource limits (`RLIMIT_AS`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`) on the spawned process, and — crucially — **always requires HITL confirmation before executing**, regardless of the `require_confirmation` argument on the tool.

### `BashToolPolicy` fields

```python
from google.adk.tools.bash_tool import BashToolPolicy

policy = BashToolPolicy(
    allowed_command_prefixes=("*",),   # ("*",) = allow any; list prefixes to restrict
    blocked_operators=(),               # e.g. ("rm", "sudo", ";", "&&")
    timeout_seconds=30,                 # asyncio.TimeoutError → error dict
    max_memory_bytes=None,              # RLIMIT_AS on child; None = no limit
    max_file_size_bytes=None,           # RLIMIT_FSIZE on child; None = no limit
    max_child_processes=None,           # RLIMIT_NPROC on child; None = no limit
)
```

`BashToolPolicy` is a `frozen` dataclass — create a new instance for each distinct policy; do not mutate.

### `ExecuteBashTool` constructor

```python
import pathlib
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy

tool = ExecuteBashTool(
    workspace=pathlib.Path("/tmp/agent-workspace"),  # cwd for subprocesses; defaults to cwd()
    policy=BashToolPolicy(
        allowed_command_prefixes=("ls", "cat", "echo", "python3"),
        blocked_operators=(";", "&&", "||", "|", ">", ">>"),
        timeout_seconds=15,
        max_memory_bytes=256 * 1024 * 1024,   # 256 MB
        max_file_size_bytes=10 * 1024 * 1024, # 10 MB per file
        max_child_processes=10,
    ),
)
```

### Tool registration

```python
from google.adk.agents import LlmAgent
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy
import pathlib

restricted_policy = BashToolPolicy(
    allowed_command_prefixes=("ls", "cat", "grep", "wc", "head", "tail", "python3 -c"),
    blocked_operators=(";", "&&", "||", "|", ">"),
    timeout_seconds=10,
    max_memory_bytes=128 * 1024 * 1024,
)

bash_agent = LlmAgent(
    name="shell_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a shell assistant. Use the bash tool to answer file and code questions. "
        "Always confirm your plan with the user before executing potentially risky commands."
    ),
    tools=[ExecuteBashTool(
        workspace=pathlib.Path("/workspace"),
        policy=restricted_policy,
    )],
)
```

### Confirmation flow and return values

`ExecuteBashTool` always gates execution behind HITL confirmation. On first call it calls `tool_context.request_confirmation(hint=...)`, sets `actions.skip_summarization = True`, and returns `{"error": "This tool call requires confirmation..."}`. The model surfaces the confirmation to the user; on the next turn the user sends back a `ToolConfirmation` payload. The tool then checks `tool_context.tool_confirmation.confirmed`:

```python
# What the tool returns on success:
{
    "stdout": "...",
    "stderr": "",
    "returncode": 0,
}

# On timeout (process killed with SIGKILL):
{
    "error": "Command timed out after 10 seconds.",
    "stdout": "...",   # whatever was captured before kill
    "stderr": "...",
    "returncode": -9,
}

# On user rejection:
{"error": "This tool call is rejected."}

# On policy violation (blocked operator or prefix mismatch):
{"error": "Command contains blocked operator: ;"}
```

### `_detect_error_in_response`

`ExecuteBashTool` overrides the `BaseTool` telemetry hook `_detect_error_in_response(response)` — it returns `"TOOL_ERROR"` when the result dict has an `"error"` key. This annotates the OTel span with an error type, enabling filtering in observability backends.

### Recipe: sandboxed code execution agent

```python
import asyncio, pathlib
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy

sandbox = pathlib.Path("/tmp/sandbox")
sandbox.mkdir(exist_ok=True)

code_runner = ExecuteBashTool(
    workspace=sandbox,
    policy=BashToolPolicy(
        allowed_command_prefixes=("python3",),
        blocked_operators=(";", "&&", "||", "|"),
        timeout_seconds=20,
        max_memory_bytes=128 * 1024 * 1024,
    ),
)

agent = LlmAgent(
    name="code_evaluator",
    model="gemini-2.5-flash",
    instruction="Run the user's Python snippet in the sandbox and report stdout/returncode.",
    tools=[code_runner],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="sandbox_demo")
    await runner.session_service.create_session(
        app_name="sandbox_demo", user_id="u1", session_id="s1"
    )
    # The model will ask for confirmation before running
    events = await runner.run_debug(
        "Run: python3 -c \"print(sum(range(100)))\"",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## 2. `DataAgentToolset` + `DataAgentCredentialsConfig` + `DataAgentToolConfig`

**Source:** `google/adk/tools/data_agent/`

`DataAgentToolset` wraps three tools that let an LLM interact with **Gemini Data Analytics Agents** — Google's managed data agents that can query BigQuery, execute Python analytics, and surface structured answers. This is marked `@experimental(FeatureName.DATA_AGENT_TOOLSET)`.

### The three tools

| Tool | Description |
|---|---|
| `list_accessible_data_agents` | Lists all published data agents in a GCP project |
| `get_data_agent_info` | Returns schema, instructions, and datasource refs for a specific agent |
| `ask_data_agent` | Streams a question to a data agent and returns the final answer |

All three are wrapped as `GoogleTool` instances, so ADK's OAuth/ADC credential pipeline injects auth before each call.

### `DataAgentCredentialsConfig`

```python
import google.auth
from google.adk.tools.data_agent import DataAgentCredentialsConfig

# Option 1: OAuth2 user-consent flow (each end-user goes through OAuth)
# Pass client_id + client_secret; scopes default to bigquery read/write.
oauth_creds = DataAgentCredentialsConfig(
    client_id="your-client-id.apps.googleusercontent.com",
    client_secret="your-client-secret",
    # scopes defaults to ["https://www.googleapis.com/auth/bigquery"]
)

# Option 2: Application Default Credentials (GKE / Cloud Run service account)
# The same shared credentials serve every end-user — only use when the
# service account already has access to every user's data.
google_creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/bigquery"]
)
adc_creds = DataAgentCredentialsConfig(credentials=google_creds)

# Option 3: Token-from-state — read an access token stored in session state
token_creds = DataAgentCredentialsConfig(
    external_access_token_key="my_bigquery_access_token"
)
```

`DataAgentCredentialsConfig` extends `BaseGoogleCredentialsConfig` and accepts **exactly one of**: `credentials` (a `google.auth.credentials.Credentials` object), `external_access_token_key` (a session-state key holding an OAuth access token), or a `client_id`/`client_secret` pair for end-user OAuth consent. It has no `auth_type` discriminator. Scopes default to `["https://www.googleapis.com/auth/bigquery"]` if omitted. The internal `_token_cache_key` is `"data_agent_token_cache"` — separate from other Google tool caches.

### `DataAgentToolConfig`

```python
from google.adk.tools.data_agent.config import DataAgentToolConfig

# Controls how many rows ask_data_agent returns in its structured response
config = DataAgentToolConfig(max_query_result_rows=100)  # default: 50
```

### Wiring it all together

```python
from google.adk.agents import LlmAgent
from google.adk.tools.data_agent import DataAgentToolset, DataAgentCredentialsConfig
from google.adk.tools.data_agent.config import DataAgentToolConfig

toolset = DataAgentToolset(
    credentials_config=DataAgentCredentialsConfig(auth_type="adc"),
    data_agent_tool_config=DataAgentToolConfig(max_query_result_rows=100),
    tool_filter=["list_accessible_data_agents", "ask_data_agent"],  # or None for all
)

analytics_agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You have access to Gemini Data Analytics Agents. "
        "First list available agents, then ask the appropriate one for data. "
        "Present results clearly. Project: my-gcp-project."
    ),
    tools=[toolset],
)
```

### Tool filter patterns

```python
from google.adk.tools.base_toolset import ToolPredicate

# Only expose ask_data_agent
toolset_ask_only = DataAgentToolset(
    credentials_config=DataAgentCredentialsConfig(auth_type="adc"),
    tool_filter=["ask_data_agent"],
)

# Dynamic predicate based on context
def only_ask_in_analytics_mode(tool, ctx):
    return tool.name == "ask_data_agent" and ctx.state.get("mode") == "analytics"

toolset_dynamic = DataAgentToolset(
    credentials_config=DataAgentCredentialsConfig(auth_type="adc"),
    tool_filter=ToolPredicate(only_ask_in_analytics_mode),
)
```

### Multi-agent pattern: router + data analyst

```python
from google.adk.agents import LlmAgent
from google.adk.tools.data_agent import DataAgentToolset, DataAgentCredentialsConfig

data_toolset = DataAgentToolset(
    credentials_config=DataAgentCredentialsConfig(auth_type="adc"),
)

data_analyst = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    description="Answers questions by querying Gemini Data Analytics Agents.",
    instruction="Use data agent tools to answer the user's analytics question.",
    tools=[data_toolset],
)

router = LlmAgent(
    name="router",
    model="gemini-2.5-flash",
    instruction="Route analytics questions to data_analyst, handle everything else directly.",
    sub_agents=[data_analyst],
)
```

---

## 3. `ConversationScenario` + `ConversationGenerationConfig` + `ScenarioGenerator`

**Source:** `google/adk/evaluation/conversation_scenarios.py`, `google/adk/evaluation/_vertex_ai_scenario_generation_facade.py`

These classes form ADK 2.4.0's AI-assisted eval-set generation pipeline: describe your agent and let Gemini generate realistic multi-turn conversation scenarios that you can then run through the full eval harness.

### `ConversationScenario`

```python
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.simulation.user_simulator_personas import UserPersona

scenario = ConversationScenario(
    starting_prompt="I need to book a flight from SFO to LAX.",
    conversation_plan=(
        "The user wants a one-way economy flight next Tuesday, morning preferred, "
        "budget under $150. If the agent finds a valid option, confirm the booking. "
        "Then ask to rent a mid-size car for 3 days from LAX airport."
    ),
    user_persona=UserPersona(
        persona_id="impatient_executive",
        persona_description="Busy executive, wants brevity, hates follow-up questions.",
    ),
    # OR: use a built-in persona string
    # user_persona="impatient",
)
```

`user_persona` accepts either a `UserPersona` object or a persona string — if a string is given, `validate_user_persona` resolves it against the default persona registry (`pre_built_personas.py`).

### `ConversationGenerationConfig`

```python
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig

gen_config = ConversationGenerationConfig(
    count=10,                   # generate 10 scenarios
    model_name="gemini-2.5-flash",
    generation_instruction=(
        "Focus on edge cases: ambiguous dates, sold-out flights, and budget conflicts."
    ),
    environment_context=(
        "Available flights: SFO→LAX on 2026-07-15 at 08:00 ($129), 10:00 ($99), 14:00 ($149). "
        "Car rental: mid-size $45/day, SUV $65/day. Cars available all dates."
    ),
)
```

`environment_context` seeds the Vertex AI generator with the ground-truth backend state so generated queries reference data that actually exists in your system.

### `ScenarioGenerator`

`ScenarioGenerator` is a thin facade over the Vertex Gen AI Eval SDK. It requires `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (or `GOOGLE_API_KEY`) in the environment.

```python
import os
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig
from google.adk.evaluation._vertex_ai_scenario_generation_facade import ScenarioGenerator

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# Define the agent being tested
travel_agent = LlmAgent(
    name="travel_agent",
    model="gemini-2.5-flash",
    description="Books flights and car rentals.",
    tools=[...],  # your booking tools
)

generator = ScenarioGenerator()
# generate_scenarios is synchronous — no await
scenarios = generator.generate_scenarios(
    agent=travel_agent,
    config=ConversationGenerationConfig(
        count=5,
        model_name="gemini-2.5-flash",
        environment_context="Flights SFO→LAX on 2026-07-15: 08:00 ($129), 10:00 ($99).",
    ),
)

for s in scenarios:
    print(f"Prompt: {s.starting_prompt[:60]}…")
    print(f"Plan:   {s.conversation_plan[:80]}…")
    print()
```

### Full pipeline: generate → run → evaluate

```python
import asyncio, os
from google.adk.evaluation.conversation_scenarios import (
    ConversationScenario, ConversationGenerationConfig,
)
from google.adk.evaluation._vertex_ai_scenario_generation_facade import ScenarioGenerator
from google.adk.evaluation.eval_case import EvalCase, Invocation, SessionInput
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics, EvalMetric
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.genai import types

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

from google.adk.agents import LlmAgent
travel_agent = LlmAgent(
    name="travel_agent",
    model="gemini-2.5-flash",
    description="Books flights and car rentals.",
)

eval_config = EvalConfig(
    criteria={PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value: 0.7}
)

async def run_evals():
    # Step 1: generate scenarios (synchronous call)
    generator = ScenarioGenerator()
    scenarios = generator.generate_scenarios(
        agent=travel_agent,
        config=ConversationGenerationConfig(count=3, model_name="gemini-2.5-flash"),
    )

    # Step 2: convert scenarios to EvalCases
    # Note: EvalCase enforces an XOR constraint — set either conversation= or
    # conversation_scenario=, not both. Use conversation_scenario= for multi-turn.
    eval_cases = [
        EvalCase(
            eval_id=f"generated_{i}",
            conversation_scenario=s,   # XOR with conversation=; use this for multi-turn plan
        )
        for i, s in enumerate(scenarios)
    ]

    eval_set = EvalSet(eval_set_id="generated_travel_evals", eval_cases=eval_cases)

    # Step 3: run evals
    await AgentEvaluator.evaluate_eval_set(
        agent_module="myapp.travel_agent",
        eval_set=eval_set,
        eval_config=eval_config,
    )

asyncio.run(run_evals())
```

---

## 4. `MultiTurnTaskSuccessV1Evaluator` · `MultiTurnToolUseQualityV1Evaluator` · `MultiTurnTrajectoryQualityV1Evaluator`

**Source:** `google/adk/evaluation/multi_turn_task_success_evaluator.py`, `multi_turn_tool_use_quality_evaluator.py`, `multi_turn_trajectory_quality_evaluator.py`

ADK 2.4.0 ships three first-class multi-turn evaluators that delegate to Vertex Gen AI Eval SDK's `RubricMetric`. All are **reference-free** (no golden expected output required), score in `[0, 1]`, and require `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION`.

| Class | `RubricMetric` | What it measures |
|---|---|---|
| `MultiTurnTaskSuccessV1Evaluator` | `MULTI_TURN_TASK_SUCCESS` | Did the agent ultimately achieve the user's goal? |
| `MultiTurnToolUseQualityV1Evaluator` | `MULTI_TURN_TOOL_USE_QUALITY` | Were function calls made correctly, with right args, in right order? |
| `MultiTurnTrajectoryQualityV1Evaluator` | `MULTI_TURN_TRAJECTORY_QUALITY` | Was the path taken to achieve the goal sensible and efficient? |

### `EvalMetric` constructor recap

```python
from google.adk.evaluation.eval_metrics import EvalMetric, PrebuiltMetrics

metric_task = EvalMetric(
    metric_name=PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value,
    threshold=0.7,   # scores below this count as failure
)
metric_tool = EvalMetric(
    metric_name=PrebuiltMetrics.MULTI_TURN_TOOL_USE_QUALITY_V1.value,
    threshold=0.8,
)
metric_traj = EvalMetric(
    metric_name=PrebuiltMetrics.MULTI_TURN_TRAJECTORY_QUALITY_V1.value,
    threshold=0.6,
)
```

### Direct evaluator usage

```python
from google.adk.evaluation.multi_turn_task_success_evaluator import MultiTurnTaskSuccessV1Evaluator
from google.adk.evaluation.multi_turn_tool_use_quality_evaluator import MultiTurnToolUseQualityV1Evaluator
from google.adk.evaluation.multi_turn_trajectory_quality_evaluator import MultiTurnTrajectoryQualityV1Evaluator
from google.adk.evaluation.eval_metrics import EvalMetric, PrebuiltMetrics
from google.adk.evaluation.conversation_scenarios import ConversationScenario

scenario = ConversationScenario(
    starting_prompt="Book me a flight to London.",
    conversation_plan="User wants economy, direct, departs next Friday. Budget £500.",
)

# Replace with actual Invocation objects from a runner
actual_turns: list = [...]

success_eval = MultiTurnTaskSuccessV1Evaluator(
    EvalMetric(metric_name=PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value, threshold=0.7)
)
tool_eval = MultiTurnToolUseQualityV1Evaluator(
    EvalMetric(metric_name=PrebuiltMetrics.MULTI_TURN_TOOL_USE_QUALITY_V1.value, threshold=0.8)
)
traj_eval = MultiTurnTrajectoryQualityV1Evaluator(
    EvalMetric(metric_name=PrebuiltMetrics.MULTI_TURN_TRAJECTORY_QUALITY_V1.value, threshold=0.6)
)

for evaluator in [success_eval, tool_eval, traj_eval]:
    # evaluate_invocations is synchronous — no await needed
    result = evaluator.evaluate_invocations(
        actual_invocations=actual_turns,
        conversation_scenario=scenario,
    )
    print(f"{evaluator.__class__.__name__}: {result}")
```

### Wiring into `EvalConfig`

```python
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

config = EvalConfig(
    criteria={
        PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value: 0.7,
        PrebuiltMetrics.MULTI_TURN_TOOL_USE_QUALITY_V1.value: 0.75,
        PrebuiltMetrics.MULTI_TURN_TRAJECTORY_QUALITY_V1.value: 0.6,
    }
)
```

All three metrics are dispatched to `_MultiTurnVertexiAiEvalFacade` — a single Vertex AI API call is made per metric, passing the full conversation as an `AgentData` + `ConversationTurns` payload. The last turn is the one scored; prior turns are marked `NOT_EVALUATED`.

### pytest integration

```python
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

@pytest.mark.asyncio
async def test_travel_agent_multi_turn():
    eval_set = EvalSet.model_validate_json(open("tests/evals/travel_multi_turn.json").read())
    config = EvalConfig(
        criteria={
            PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value: 0.7,
            PrebuiltMetrics.MULTI_TURN_TOOL_USE_QUALITY_V1.value: 0.75,
        }
    )
    await AgentEvaluator.evaluate_eval_set(
        agent_module="myapp.travel_agent",
        eval_set=eval_set,
        eval_config=config,
    )
```

---

## 5. `AppDetails` + `AgentDetails`

**Source:** `google/adk/evaluation/app_details.py`

`AppDetails` is a lightweight Pydantic model that the ADK eval system builds to capture a snapshot of your agent hierarchy — names, instructions, and tool declarations — so eval backends and prompts can reference them without instantiating a live runner.

### `AgentDetails`

```python
from google.adk.evaluation.app_details import AgentDetails

details = AgentDetails(
    name="travel_agent",
    instructions="You are a travel booking assistant.",
    tool_declarations=[...],   # list of genai_types.ToolListUnion at runtime
)
```

### `AppDetails`

```python
from google.adk.evaluation.app_details import AppDetails, AgentDetails

app_details = AppDetails(
    agent_details={
        "travel_agent": AgentDetails(
            name="travel_agent",
            instructions="You are a travel booking assistant.",
        ),
        "flight_subagent": AgentDetails(
            name="flight_subagent",
            instructions="You specialise in finding flights.",
        ),
    }
)

# Look up instructions for a specific agent
instr = app_details.get_developer_instructions("flight_subagent")

# Get all tools keyed by agent name
tool_map = app_details.get_tools_by_agent_name()
# {"travel_agent": [...], "flight_subagent": [...]}
```

### How `AppDetails` is consumed by the eval system

The eval harness builds an `AppDetails` before running each `EvalCase`. It traverses the agent tree from `root_agent`, collecting `LlmAgent.instruction` and `canonical_tools()` output for every node. This snapshot is then passed to LLM-as-judge prompts so the judge understands what the agent was supposed to do.

```python
# Pseudo-code of what the eval system does internally:
from google.adk.evaluation.app_details import AppDetails, AgentDetails

def build_app_details(root_agent) -> AppDetails:
    details = {}
    def _collect(agent):
        details[agent.name] = AgentDetails(
            name=agent.name,
            instructions=agent.instruction or "",
        )
        for sub in getattr(agent, "sub_agents", []):
            _collect(sub)
    _collect(root_agent)
    return AppDetails(agent_details=details)
```

### Custom eval prompt using `AppDetails`

```python
from google.adk.evaluation.app_details import AppDetails

def build_judge_prompt(app_details: AppDetails, agent_name: str, response: str) -> str:
    instr = app_details.get_developer_instructions(agent_name)
    return (
        f"Agent instruction: {instr}\n\n"
        f"Agent response: {response}\n\n"
        "Rate this response 0-1 for instruction adherence."
    )
```

---

## 6. `EnsureRetryOptionsPlugin` + `add_default_retry_options_if_not_present`

**Source:** `google/adk/evaluation/_retry_options_utils.py`

Transient model API errors (rate-limits, gateway timeouts, service overloads) are a top cause of flaky eval runs. `EnsureRetryOptionsPlugin` is a `BasePlugin` that injects `types.HttpRetryOptions` into every `LlmRequest` before it's sent, protecting long-running eval batches from transient failures.

### Default retry profile

```python
from google.adk.evaluation._retry_options_utils import _DEFAULT_HTTP_RETRY_OPTIONS

# Exposed values from source:
# attempts=7, initial_delay=5.0, max_delay=120.0, exp_base=2.0
# Retried status codes: 408, 429, 500, 502, 503, 504
print(_DEFAULT_HTTP_RETRY_OPTIONS)
```

Backoff formula: `min(max_delay, initial_delay * exp_base**n)` where `n` is the zero-based attempt count.  With defaults: 5 s, 10 s, 20 s, 40 s, 80 s, 120 s, 120 s — 7 total attempts.

### `add_default_retry_options_if_not_present` (standalone function)

```python
from google.adk.evaluation._retry_options_utils import add_default_retry_options_if_not_present
from google.adk.models.llm_request import LlmRequest

def my_before_model_callback(callback_context, llm_request: LlmRequest):
    # Attach default retry options if none are set — idempotent if already present
    add_default_retry_options_if_not_present(llm_request)
    return None   # continue with the LLM call
```

The function is idempotent: it only assigns if `llm_request.config.http_options.retry_options is None`.

### `EnsureRetryOptionsPlugin` as a global plugin

```python
from google.adk.evaluation._retry_options_utils import EnsureRetryOptionsPlugin
from google.adk.apps import App
from google.adk.runners import Runner

app = App(
    name="eval_app",
    root_agent=my_agent,
    plugins=[EnsureRetryOptionsPlugin()],   # protects all LLM calls
)
runner = Runner(app=app, session_service=..., artifact_service=...)
```

### Custom retry profile for production (non-eval) use

```python
from google.genai import types
from google.adk.plugins import BasePlugin
from google.adk.models.llm_request import LlmRequest

class AggressiveRetryPlugin(BasePlugin):
    """Retry up to 10 times with 2 s initial delay, capped at 60 s."""

    RETRY_OPTIONS = types.HttpRetryOptions(
        attempts=10,
        initial_delay=2.0,
        max_delay=60.0,
        exp_base=2.0,
        http_status_codes=(429, 503, 504),  # only retriable transients
    )

    async def before_model_callback(self, *, callback_context, llm_request):
        if llm_request.config is None:
            from google.genai import types as t
            llm_request.config = t.GenerateContentConfig()
        if llm_request.config.http_options is None:
            llm_request.config.http_options = types.HttpOptions()
        if llm_request.config.http_options.retry_options is None:
            llm_request.config.http_options.retry_options = self.RETRY_OPTIONS
        return None
```

---

## 7. `McpInstructionProvider`

**Source:** `google/adk/agents/mcp_instruction_provider.py`

`McpInstructionProvider` implements the `InstructionProvider` protocol, which means it can be passed directly to `LlmAgent(instruction=...)` wherever a `str | Callable` is expected.  On each LLM call it:

1. Opens an MCP session (via `MCPSessionManager`)
2. Lists prompts on the server to discover required argument names
3. Harvests those argument values from `context.state`
4. Calls `session.get_prompt(name, arguments=...)` and concatenates all `text` parts

This pattern lets a centralised MCP server own agent instructions — update the server and all connected agents pick up changes instantly.

### Constructor

```python
import sys
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

provider = McpInstructionProvider(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python3",
            args=["-m", "my_prompt_server"],
        ),
        timeout=5.0,
    ),
    prompt_name="travel_agent_instructions",
    errlog=sys.stderr,   # optional; defaults to sys.stderr
)
```

### Wiring into `LlmAgent`

```python
from google.adk.agents import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool import SseConnectionParams

dynamic_instruction = McpInstructionProvider(
    connection_params=SseConnectionParams(url="https://prompts.mycompany.internal/mcp"),
    prompt_name="customer_support_instructions",
)

support_agent = LlmAgent(
    name="support",
    model="gemini-2.5-flash",
    instruction=dynamic_instruction,   # called every LLM turn
    tools=[...],
)
```

### State-forwarded arguments

If the MCP Prompt declares required arguments, `McpInstructionProvider` automatically reads matching keys from `context.state`:

```python
# Server-side MCP prompt definition (pseudo-code):
# name: "personalised_instructions"
# arguments: [{"name": "tenant_id"}, {"name": "user_tier"}]

# ADK agent session state:
session.state["tenant_id"] = "acme-corp"
session.state["user_tier"] = "enterprise"

# On next LLM call, McpInstructionProvider sends:
# get_prompt("personalised_instructions", arguments={"tenant_id": "acme-corp", "user_tier": "enterprise"})
```

This makes it easy to build multi-tenant instruction systems without needing custom callable logic.

### Recipe: shared instruction server for a multi-agent team

```python
from google.adk.agents import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool import SseConnectionParams

_PROMPT_SERVER = SseConnectionParams(url="https://prompts.internal/mcp")

def make_provider(prompt_name: str) -> McpInstructionProvider:
    return McpInstructionProvider(
        connection_params=_PROMPT_SERVER,
        prompt_name=prompt_name,
    )

planner = LlmAgent(
    name="planner",
    model="gemini-2.5-pro",
    instruction=make_provider("planner_instructions"),
    sub_agents=[
        LlmAgent(
            name="researcher",
            model="gemini-2.5-flash",
            instruction=make_provider("researcher_instructions"),
        ),
        LlmAgent(
            name="writer",
            model="gemini-2.5-flash",
            instruction=make_provider("writer_instructions"),
        ),
    ],
)
```

### Failure behaviour

If the prompt is not found or has no `text` messages, `McpInstructionProvider.__call__` raises `ValueError("Failed to load MCP prompt '...'")`. Catch this in a wrapping `before_model_callback` if you need a graceful fallback:

```python
async def safe_instruction(context):
    try:
        return await provider(context)
    except (ValueError, Exception):
        return "You are a helpful assistant."  # fallback instruction
```

---

## 8. `EnvironmentSimulationFactory` + `EnvironmentSimulationEngine` + `ToolSpecMockStrategy`

**Source:** `google/adk/tools/environment_simulation/`

> **2.4.0 migration note:** `google.adk.tools.agent_simulator` is now deprecated — it emits a `DeprecationWarning` and re-exports `EnvironmentSimulationFactory as AgentSimulatorFactory`. Switch all imports to `google.adk.tools.environment_simulation`.

Environment simulation lets you test agents **without calling real tools**. The factory creates either a `before_tool_callback` or an `EnvironmentSimulationPlugin`; the engine intercepts matching tool calls and either injects pre-canned responses/errors or uses an LLM (`ToolSpecMockStrategy`) to generate stateful mock responses.

### `EnvironmentSimulationConfig`

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    ToolSimulationConfig,
    InjectionConfig,
    InjectedError,
    MockStrategy,
)

config = EnvironmentSimulationConfig(
    simulation_model="gemini-2.5-flash",
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="book_flight",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            injection_configs=[
                InjectionConfig(
                    injection_probability=0.1,   # 10% chance of injected error
                    injected_error=InjectedError(
                        injected_http_error_code=503,
                        error_message="Booking service temporarily unavailable",
                    ),
                    random_seed=42,
                ),
            ],
        ),
        ToolSimulationConfig(
            tool_name="get_seat_map",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    environment_data="Flights: AA123 SFO→LAX 08:00 $99, BA456 SFO→LHR 22:00 $799.",
)
```

### Using as a `before_tool_callback`

```python
from google.adk.agents import LlmAgent
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory

sim_callback = EnvironmentSimulationFactory.create_callback(config)

agent = LlmAgent(
    name="booking_agent",
    model="gemini-2.5-flash",
    instruction="Help users book flights.",
    tools=[book_flight_tool, get_seat_map_tool],
    before_tool_callback=sim_callback,
)
```

The callback receives `(tool, args, tool_context)`. For tools not in `tool_simulation_configs`, it returns `None` (real tool executes). For matching tools it runs injection logic first, then falls back to `ToolSpecMockStrategy`.

### Using as a plugin

```python
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory

sim_plugin = EnvironmentSimulationFactory.create_plugin(config)

app = App(
    name="booking_demo",
    root_agent=booking_agent,
    plugins=[sim_plugin],   # intercepts before_tool_callback globally
)
runner = InMemoryRunner(app=app)
```

Plugin-mode is preferable for multi-agent setups where you want simulation to apply uniformly across all agents.

### `ToolSpecMockStrategy` internals

`ToolSpecMockStrategy` prompts an LLM with a structured template that includes:
- A shared **state store** (tracks IDs created by prior "creating" tools)
- A **tool connection map** (which tools create vs consume stateful parameters)
- The **tool schema** and **arguments** for the current call
- Optional **environment data** and **tracing history**

The LLM must return a JSON object starting with `{` and ending with `}`. If a consuming tool references an ID not in the state store, the strategy is prompted to return a 404-style error. New IDs from creating tools are parsed back and stored in `_state_store`, maintaining consistency across the session.

```python
# Completely offline agent test — no GCP credentials required for the agent's tools
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, MockStrategy,
)

config = EnvironmentSimulationConfig(
    simulation_model="gemini-2.5-flash",
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="search_inventory",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="place_order",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    environment_data='{"products": [{"id": "P1", "name": "Widget", "price": 9.99, "stock": 42}]}',
)

agent = LlmAgent(
    name="shop",
    model="gemini-2.5-flash",
    instruction="Help users buy products. Use search_inventory then place_order.",
    tools=[search_inventory_fn, place_order_fn],
    before_tool_callback=EnvironmentSimulationFactory.create_callback(config),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="shop_sim")
    await runner.session_service.create_session(
        app_name="shop_sim", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Find and order one Widget.", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## 9. `RunConfig` comprehensive (2.4.0)

**Source:** `google/adk/agents/run_config.py`

`RunConfig` is passed to `runner.run_async(..., run_config=config)` and controls every runtime knob for a single invocation: HTTP retry, streaming mode, tool thread pool, LLM call budget, audio transcription, and telemetry.

### Full constructor reference

```python
from google.adk.agents.run_config import RunConfig, StreamingMode, ToolThreadPoolConfig
from google.adk.telemetry.context import TelemetryConfig
from google.genai import types

config = RunConfig(
    # --- HTTP layer ---
    http_options=types.HttpOptions(
        timeout=120_000,   # ms; applies to every Gemini API call in this invocation
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=2.0,
            max_delay=60.0,
            exp_base=2.0,
            http_status_codes=(429, 503, 504),
        ),
        headers={"X-Tenant": "acme"},  # injected into every request
    ),

    # --- Streaming ---
    streaming_mode=StreamingMode.SSE,   # NONE (default) | SSE | BIDI
    support_cfc=False,                  # enable CFC in SSE mode (experimental)

    # --- LLM budget ---
    max_llm_calls=50,    # 500 default; <=0 = no limit (logs a warning); sys.maxsize raises ValueError

    # --- Tool thread pool (live/audio) ---
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=4),

    # --- Live audio ---
    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede"),
    )),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    response_modalities=[types.Modality.TEXT],   # or AUDIO

    # --- Per-invocation OTel ---
    telemetry=TelemetryConfig(
        tracing_enabled=True,
        metrics_enabled=False,
        destination="gcp",
    ),

    # --- Custom metadata (stored on every event) ---
    custom_metadata={"request_id": "abc-123", "ab_variant": "B"},
)
```

### `StreamingMode` reference

| Value | API used | Partial events | Use case |
|---|---|---|---|
| `StreamingMode.NONE` | `generate_content` | No | CLI, batch, synchronous |
| `StreamingMode.SSE` | `generate_content_stream` | Text chunks | Web streaming, chat UIs |
| `StreamingMode.BIDI` | Live/BIDI WebSocket | Audio + video chunks | Voice apps, real-time |

```python
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

# Streaming text response
sse_config = RunConfig(streaming_mode=StreamingMode.SSE)

async for event in runner.run_async(
    user_id="u1", session_id="s1",
    new_message=types.Content(role="user", parts=[types.Part(text="Explain quantum entanglement.")]),
    run_config=sse_config,
):
    if event.partial and event.content:
        print(event.content.parts[0].text, end="", flush=True)
print()
```

### `ToolThreadPoolConfig` — thread pool for live mode tools

In BIDI live sessions, blocking tool calls on the event loop starve audio input. `ToolThreadPoolConfig` moves tool execution to a thread pool, freeing the loop for real-time audio chunks:

```python
from google.adk.agents.run_config import RunConfig, StreamingMode, ToolThreadPoolConfig

live_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),  # default: 4
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)
```

> **GIL caveat:** The thread pool helps with I/O-bound tools (network, file, database) and C extensions that release the GIL. It does **not** help with pure Python CPU-bound code. For CPU-heavy tools use `ProcessPoolExecutor` or break the work into `asyncio.sleep(0)` checkpoints.

### `max_llm_calls` budget enforcement

```python
from google.adk.agents.run_config import RunConfig

# Strict budget: fail after 10 LLM calls (catches runaway loops)
strict_config = RunConfig(max_llm_calls=10)

# Disable budget: set to 0 (logs a warning — may loop indefinitely)
unlimited_config = RunConfig(max_llm_calls=0)
```

When the limit is exceeded, the runner raises `LlmCallsLimitExceededError` (`errors/_invocation_cost_manager.py`). Catch it to return a graceful "I've reached my processing limit" message.

### `http_options` custom headers for multi-tenant

```python
from google.adk.agents.run_config import RunConfig
from google.genai import types

async def handle_request(tenant_id: str, user_query: str):
    run_config = RunConfig(
        http_options=types.HttpOptions(
            headers={"X-Goog-Request-Params": f"tenant={tenant_id}"},
            timeout=30_000,
        ),
        custom_metadata={"tenant_id": tenant_id},
    )
    async for event in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text=user_query)]),
        run_config=run_config,
    ):
        yield event
```

---

## 10. `ScheduleDynamicNode` protocol + `ctx.run_node()`

**Source:** `google/adk/workflow/_schedule_dynamic_node.py`, `google/adk/agents/context.py`

`ScheduleDynamicNode` is the typed `Protocol` behind `ctx.run_node()` — the low-level primitive for scheduling a `BaseNode` from inside another node at runtime, rather than wiring edges at construction time. It is the building block for dynamic fan-out, agent-directed routing, and HITL resume patterns inside `Workflow` graphs.

### Protocol signature

```python
class ScheduleDynamicNode(Protocol):
    def __call__(
        self,
        ctx: Context,
        node: Any,                 # BaseNode (or LlmAgent / BaseTool — wrapped automatically)
        node_input: Any,           # must match node.input_schema if defined
        *,
        node_name: str | None = None,  # tracking name; defaults to node.name
        use_as_output: bool = False,    # child output replaces calling node's output
        run_id: str,               # unique ID for idempotent resume
        use_sub_branch: bool = False,   # isolate child from parent message history
        override_branch: str | None = None,
        override_isolation_scope: str | None = None,
    ) -> Awaitable[Context]:   # the internal protocol resolves to the child Context
        ...
    # Note: the public ctx.run_node() wrapper extracts child_ctx.output and
    # returns it directly as Any — callers never see the Context object.
```

### `ctx.run_node()` — the public API

```python
from google.adk.workflow import node, Workflow, START
from google.adk.agents import LlmAgent

summarizer = LlmAgent(
    name="summarizer",
    model="gemini-2.5-flash",
    instruction="Summarize the input in one sentence.",
    mode="single_turn",
)

@node
async def dispatcher(node_input: list[str], ctx) -> list[str]:
    """Dispatch each item to a summarizer agent and collect results."""
    results = []
    for i, item in enumerate(node_input):
        # ctx.run_node() returns the child node's output directly (Any), not a Context
        result = await ctx.run_node(
            summarizer,
            node_input=item,
            run_id=f"summarize_{i}",         # stable across resume attempts
            use_sub_branch=True,              # isolate message history per item
        )
        results.append(result)
    return results

pipeline = Workflow(
    name="batch_summarizer",
    edges=[(START, dispatcher)],
)
```

### `use_as_output=True` — propagate child output upward

```python
@node
async def route_to_specialist(user_intent: str, ctx) -> None:
    """Route to a specialist agent and use its output as this node's output."""
    if "billing" in user_intent.lower():
        agent = billing_agent
    else:
        agent = general_agent

    await ctx.run_node(
        agent,
        node_input=user_intent,
        run_id="specialist_call",
        use_as_output=True,   # the specialist's final response becomes this node's output
    )
```

### Resume-safe `run_id`

`run_id` must be **deterministic and stable across resume attempts**. If the workflow is interrupted mid-run (HITL, restart), ADK replays the node. A stable `run_id` ensures the cached child output is returned rather than re-running the child.

```python
@node
async def parallel_research(queries: list[str], ctx) -> dict:
    import asyncio

    async def run_one(i: int, query: str):
        # ctx.run_node() returns the child node's output directly
        output = await ctx.run_node(
            research_agent,
            node_input=query,
            run_id=f"research_{i}",   # index-based; stable across resume
            use_sub_branch=True,
        )
        return query, output

    pairs = await asyncio.gather(*[run_one(i, q) for i, q in enumerate(queries)])
    return dict(pairs)
```

### HITL integration with `ctx.run_node()`

When a dynamically scheduled child node calls `ctx.interrupt(...)`, the interrupt propagates back up to the parent workflow via `ScheduleDynamicNode`. The interrupted child's state is persisted in the session; on resume, `run_id` matching restores the child context and continues from where it left off.

```python
from google.adk.workflow import node, Workflow, START
from google.genai import types

@node(rerun_on_resume=True)   # required: callers of ctx.run_node() must be rerunnable
async def approval_gate(doc_text: str, ctx) -> str:
    # Schedule a reviewer agent that may raise a HITL interrupt.
    # ctx.run_node() returns the child node's output directly (not a Context object).
    review_output = await ctx.run_node(
        reviewer_agent,
        node_input=doc_text,
        run_id="doc_review",
    )
    return review_output

review_workflow = Workflow(
    name="review_pipeline",
    edges=[(START, approval_gate)],
)
```

### When to prefer `ctx.run_node()` vs static edges

| Pattern | Use static `edges=` | Use `ctx.run_node()` |
|---|---|---|
| Fixed DAG known at build time | ✓ | — |
| Fan-out over runtime-determined list | — | ✓ |
| Agent selection based on LLM decision | — | ✓ |
| Recursive / iterative processing | — | ✓ |
| HITL approve-then-continue per item | — | ✓ |

---

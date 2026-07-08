---
title: "Class deep dives — volume 38 (10 new APIs)"
description: "Source-verified deep dives into 10 google-adk 2.4.0 APIs: ManagedAgent (Managed Agents API; background streaming; server-side tools only; global location), DaytonaEnvironment (remote Daytona sandbox; initialize/close/execute/read_file/write_file lifecycle), VertexAiLoadProfilesTool (FunctionTool over VertexAiMemoryBankService.retrieve_profiles; JSON_SCHEMA_FOR_FUNC_DECL flag), ReplayManager (unified rehydration+barrier orchestrator; scan_workflow_events; prepare_parent_sequence_barrier; advance_sequence/wait_sequence), platform.time+uuid+thread (ContextVar-based injectable providers for deterministic testing), _schema_utils SchemaType+validate_schema (SchemaUnion 3-branch dispatch; TypeAdapter JSON schema), InteractionsRequestProcessor+_find_previous_interaction_state (stateful Interactions API chaining; reverse branch-scoped event scan), include_artifacts_in_a2a_event_interceptor (A2A after-event hook; artifact_delta load + TaskArtifactUpdateEvent injection), MtlsEndpoint+mTLS utilities (AUTO/ALWAYS/NEVER; get_api_endpoint template; effective_googleapis_endpoint URL rewrite), PubSubCredentialsConfig+PubSubToolConfig (BaseGoogleCredentialsConfig extension; default scopes; project_id config model)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 38"
  order: 107
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.4.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `ManagedAgent` — Managed Agents API integration

**Source:** `google/adk/agents/_managed_agent.py`

`ManagedAgent` is a new `BaseAgent` subclass (added in **2.4.0**) that calls
Google's **Managed Agents API** (`interactions.create`) instead of driving an
LLM directly. It is the recommended way to use Google-hosted agents (e.g.
`'antigravity-preview-05-2026'`) within an ADK application. Only server-side
tools are supported (`google_search`, `code_execution`, `url_context`,
`computer_use`); client-executed `FunctionTool`s and MCP tools raise
`NotImplementedError`.

### Key implementation details (verified `_managed_agent.py`)

```python
class ManagedAgent(BaseAgent):
    agent_id: str
    environment: Optional[CreateAgentInteractionEnvironmentParam] = None
    agent_config: Optional[CreateAgentInteractionAgentConfigParam] = None
    tools: list[Union[types.Tool, BaseTool, Callable[..., Any]]] = Field(default_factory=list)
    _api_client: Optional[Client] = PrivateAttr(default=None)
```

The Managed Agents API is **only served from the `global` location**. When an
injected enterprise client targets a different location the constructor raises
`ValueError`. The `api_client` property creates a `Client` lazily on first use:
enterprise mode is detected via `is_enterprise_mode_enabled()` and pinned to
`location='global'`; developer API clients get no location argument.

`_resolve_backend_tools()` iterates `self.tools` and rejects anything that
would register a function declaration (i.e. client-executed tools). Raw
`types.Tool` configs are passed through only if they carry a supported
server-side field. All interactions are created with `background=True` and
consumed over a streaming SSE connection — non-streaming polling is not yet
supported.

### Example 1 — minimal ManagedAgent with built-in google_search

```python
import asyncio
from google.adk.agents._managed_agent import ManagedAgent
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = ManagedAgent(
    name="managed_search",
    agent_id="antigravity-preview-05-2026",
    tools=[google_search],  # server-side; no function_declaration registered
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What is ADK?")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — reusing a Daytona remote environment across turns

```python
from google.adk.agents._managed_agent import ManagedAgent

# Create a new remote Daytona sandbox on the first turn.  On subsequent
# turns ADK automatically retrieves the environment_id from prior events
# via _find_previous_interaction_state and reuses the same sandbox.
agent = ManagedAgent(
    name="managed_code",
    agent_id="antigravity-preview-05-2026",
    environment={"type": "remote"},   # create a fresh remote sandbox on first turn
    tools=[],
)
```

### Example 3 — injecting a pre-configured enterprise client

```python
from google.genai import Client
from google.adk.agents._managed_agent import ManagedAgent

# Build a client targeting "global" explicitly (any other location is rejected
# by ManagedAgent._validate_client_location at construction time).
api_client = Client(enterprise=True, location="global")

agent = ManagedAgent(
    name="managed_with_client",
    agent_id="my-managed-agent-id",
    api_client=api_client,  # bypasses lazy creation
)
```

---

## 2 · `DaytonaEnvironment` — remote Daytona sandbox

**Source:** `google/adk/integrations/daytona/_daytona_environment.py`

`DaytonaEnvironment` is a `BaseEnvironment` implementation (added in **2.4.0**)
backed by a [Daytona](https://daytona.io) remote sandbox. It provides
persistent file CRUD and shell execution inside an isolated remote workspace.

Requires the `daytona` extra: `pip install google-adk[daytona]`.

Decorated with `@experimental(FeatureName.DAYTONA_ENVIRONMENT)`.

### Lifecycle (verified `_daytona_environment.py`)

| Method | Behaviour |
|---|---|
| `initialize()` | Creates one `AsyncSandbox` via `AsyncDaytona.create()`; idempotent — second call is a no-op |
| `close()` | Calls `sandbox.delete()`, clears `_sandbox` and `_client` references |
| `execute(command, *, timeout)` | `sandbox.process.exec()`; maps `DaytonaError` timeout → `ExecutionResult(exit_code=-1, timed_out=True)` |
| `read_file(path)` | `sandbox.fs.download_file()`; translates `DaytonaNotFoundError` → `FileNotFoundError` |
| `write_file(path, content)` | Recursive parent dir creation (skips `DaytonaConflictError`); `upload_file(bytes, resolved)` |

`_resolve_path` joins relative paths under `_SANDBOX_HOME = "/workspaces"`.

### Example 1 — basic file write + execute

```python
import asyncio
from google.adk.integrations.daytona import DaytonaEnvironment

async def main():
    env = DaytonaEnvironment(timeout=300)   # 5-minute sandbox TTL
    await env.initialize()

    # Write a script and run it
    await env.write_file("hello.py", b"print('hello from Daytona')")
    result = await env.execute("python3 hello.py")
    print(result.stdout)   # "hello from Daytona"

    await env.close()

asyncio.run(main())
```

### Example 2 — custom image and environment variables

```python
from google.adk.integrations.daytona import DaytonaEnvironment

import os

env = DaytonaEnvironment(
    image="python:3.11-slim",           # Daytona template name or snapshot ID
    timeout=600,
    api_key=os.environ["DAYTONA_API_KEY"],   # read from environment, never hardcode
    api_url="https://app.daytona.io",        # default cloud API
    env_vars={"DEBUG": "1", "MY_TOKEN": os.environ.get("MY_TOKEN", "")},
)
```

### Example 3 — wiring into EnvironmentToolset

```python
import asyncio
from google.adk.integrations.daytona import DaytonaEnvironment
from google.adk.tools.environment_toolset import EnvironmentToolset
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

async def main():
    env = DaytonaEnvironment()
    await env.initialize()

    toolset = EnvironmentToolset(environment=env)

    agent = LlmAgent(
        name="coder",
        model="gemini-2.5-flash",
        instruction="Write and run Python code to answer user questions.",
        tools=[toolset],
    )

    runner = InMemoryRunner(agent=agent, app_name="coder_app")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Write a Python script that prints the Fibonacci sequence up to 100.")],
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    await env.close()

asyncio.run(main())
```

---

## 3 · `VertexAiLoadProfilesTool` — structured user profile loader

**Source:** `google/adk/tools/vertex_ai_load_profiles_tool.py`

`VertexAiLoadProfilesTool` is a `FunctionTool` subclass (added in **2.4.0**)
that fetches a user's structured profiles from
`VertexAiMemoryBankService.retrieve_profiles()` and surfaces them to the model
as a tool call result.

### Implementation notes (verified source)

```python
class VertexAiLoadProfilesTool(FunctionTool):
    def __init__(self, memory_service: VertexAiMemoryBankService):
        super().__init__(self.load_profiles)
        self._memory_service = memory_service

    async def load_profiles(self, tool_context: ToolContext) -> dict[str, Any]:
        profiles = await self._memory_service.retrieve_profiles(
            app_name=tool_context.session.app_name,
            user_id=tool_context.user_id,
        )
        return {"profiles": [p.profile for p in profiles if p.profile]}
```

`_get_declaration()` branches on the `JSON_SCHEMA_FOR_FUNC_DECL` feature flag:
- Flag **on** → `parameters_json_schema={"type": "object", "properties": {}}` (JSON Schema format)
- Flag **off** → `parameters=types.Schema(type=types.Type.OBJECT, properties={})` (proto format)

The `load_profiles` function always has zero LLM-visible parameters; the
`tool_context` argument is automatically stripped from the declaration by
`FunctionTool._get_mandatory_args()` because it is a context-typed parameter.

### Example 1 — minimal usage

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools.vertex_ai_load_profiles_tool import VertexAiLoadProfilesTool
from google.adk.runners import InMemoryRunner

memory_service = VertexAiMemoryBankService(
    project="my-project",
    location="us-central1",
    agent_engine_id="projects/my-project/locations/us-central1/reasoningEngines/123",
)

load_profiles_tool = VertexAiLoadProfilesTool(memory_service=memory_service)

agent = LlmAgent(
    name="personalised_agent",
    model="gemini-2.5-flash",
    instruction="Use load_profiles to fetch user preferences before responding.",
    tools=[load_profiles_tool],
    # memory_service is not an LlmAgent field; supply it at the runner/app layer.
)
```

### Example 2 — enabling JSON Schema mode for the declaration

```python
from google.adk.features import FeatureName, override_feature_enabled
from google.adk.tools.vertex_ai_load_profiles_tool import VertexAiLoadProfilesTool
from google.adk.memory import VertexAiMemoryBankService

# Force the JSON Schema code-path for _get_declaration().
with override_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
    tool = VertexAiLoadProfilesTool(memory_service=VertexAiMemoryBankService(
        project="p", location="us-central1", agent_engine_id="re/123"
    ))
    decl = tool._get_declaration()
    # decl.parameters_json_schema == {"type": "object", "properties": {}}
    print(decl.name)        # "load_profiles"
    print(decl.description) # docstring of load_profiles
```

### Example 3 — pairing with PreloadMemoryTool

```python
from google.adk.agents import LlmAgent
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools.vertex_ai_load_profiles_tool import VertexAiLoadProfilesTool
from google.adk.tools import PreloadMemoryTool

memory_service = VertexAiMemoryBankService(
    project="p", location="us-central1", agent_engine_id="re/123"
)

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You have access to the user's long-term memory and structured profiles. "
        "Use PreloadMemoryTool to automatically inject relevant past events "
        "and VertexAiLoadProfilesTool to read the user's structured preferences."
    ),
    tools=[
        PreloadMemoryTool(),
        VertexAiLoadProfilesTool(memory_service=memory_service),
    ],
    # memory_service is not an LlmAgent field; supply it at the runner/app layer.
)
```

---

## 4 · `ReplayManager` — unified replay orchestrator

**Source:** `google/adk/workflow/utils/_replay_manager.py`

`ReplayManager` (added in **2.4.0**) is the single object that coordinates
event rehydration, replay interception, and sequence barrier synchronisation
for both static and dynamic workflow nodes. Previously these responsibilities
were spread across several helpers; `ReplayManager` consolidates them.

### Core API (verified `_replay_manager.py`)

```python
class ReplayManager:
    def scan_workflow_events(
        self, ctx: Context
    ) -> tuple[dict[str, _ChildScanState], list[str]]:
        """Scans session events for direct child nodes; initialises sequence barrier."""

    def prepare_parent_sequence_barrier(
        self, ctx: Context, parent_path: str
    ) -> ReplaySequenceBarrier:
        """Ensures a sequence barrier is set up for dynamic nodes under parent_path."""

    async def advance_sequence(self, parent_path: str, key: str) -> None:
        """Advance the sequence barrier for parent_path if it exists."""

    async def wait_sequence(self, parent_path: str, key: str) -> None:
        """Wait for the sequence barrier for parent_path if it exists."""
```

`scan_workflow_events()` calls `_reconstruct_node_states()` to rehydrate
completed child executions, then calls `_scan_sequence()` to extract the
chronological completion order. The resulting `ReplaySequenceBarrier` ensures
that replaying nodes fire in the same order as the original run, preventing
non-deterministic divergence.

`_scan_sequence()` iterates session events, filters by `invocation_id`,
computes the direct-child path via `_NodePathBuilder`, and appends/moves the
segment on each `is_terminal_event` hit — so the last entry wins.

### Example 1 — scanning events for a static workflow

```python
import asyncio
from google.adk.workflow.utils._replay_manager import ReplayManager
from google.adk.agents.context import Context

async def on_workflow_start(ctx: Context):
    replay_manager = ReplayManager()
    recovered, sequence = replay_manager.scan_workflow_events(ctx)

    print(f"Recovered {len(recovered)} child executions: {list(recovered.keys())}")
    print(f"Replay sequence: {sequence}")

    # The sequence barrier is stored internally; downstream nodes call
    # replay_manager.wait_sequence() before executing.
    return replay_manager
```

### Example 2 — dynamic node sequence barrier

```python
import asyncio
from google.adk.workflow.utils._replay_manager import ReplayManager
from google.adk.agents.context import Context

async def schedule_dynamic_node(
    ctx: Context,
    replay_manager: ReplayManager,
    parent_path: str,
    node_key: str,
):
    # Prepare (idempotent) a barrier for nodes scheduled under parent_path.
    barrier = replay_manager.prepare_parent_sequence_barrier(ctx, parent_path)

    # Wait until this node's turn arrives in the original sequence.
    await replay_manager.wait_sequence(parent_path, node_key)
    try:
        # ... execute the node ...
        result = await execute_node(ctx, node_key)
    finally:
        # Advance the barrier so the next node can proceed.
        await replay_manager.advance_sequence(parent_path, node_key)

    return result
```

### Example 3 — inspecting recovered child states

```python
from google.adk.workflow.utils._replay_manager import ReplayManager
from google.adk.workflow.utils._rehydration_utils import _ChildScanState

async def inspect_replay(ctx):
    replay_manager = ReplayManager()
    recovered, sequence = replay_manager.scan_workflow_events(ctx)

    for node_name, scan_state in recovered.items():
        # _ChildScanState tracks whether the node completed, its output, etc.
        print(f"  {node_name}: completed={scan_state.is_completed}")

    # recovered_executions and sequence_barrier are also accessible as properties:
    assert replay_manager.recovered_executions is recovered
    assert replay_manager.sequence_barrier is not None or len(sequence) == 0
```

---

## 5 · Platform abstractions — injectable time, UUID, and thread providers

**Sources:**
- `google/adk/platform/time.py`
- `google/adk/platform/uuid.py`
- `google/adk/platform/thread.py`

These three modules provide thin `ContextVar`-based wrappers around
`time.time()`, `uuid.uuid4()`, and `threading.Thread` so that tests can inject
deterministic values without monkey-patching stdlib.

### API (verified sources)

```python
# time.py
def set_time_provider(provider: Callable[[], float]) -> None: ...
def reset_time_provider() -> None: ...
def get_time() -> float: ...         # calls _time_provider_context_var.get()()

# uuid.py
def set_id_provider(provider: Callable[[], str]) -> None: ...
def reset_id_provider() -> None: ...
def new_uuid() -> str: ...           # calls _id_provider_context_var.get()()

# thread.py
def create_thread(target: Callable[..., None], *args, **kwargs): ...
# Falls through to internal override or threading.Thread(...).
```

All providers are stored in `ContextVar[Callable]` instances, so different
async tasks (or threads) can override them independently without interfering
with each other.

### Example 1 — deterministic timestamps in tests

```python
import pytest
from google.adk.platform import time as adk_time

def test_event_timestamp():
    fake_ts = 1_700_000_000.0
    adk_time.set_time_provider(lambda: fake_ts)
    try:
        ts = adk_time.get_time()
        assert ts == fake_ts
    finally:
        adk_time.reset_time_provider()

# Outside the try block, get_time() returns real wall-clock time again.
```

### Example 2 — deterministic UUID generation

```python
from itertools import count
from google.adk.platform import uuid as adk_uuid

def test_stable_ids():
    counter = count(1)
    adk_uuid.set_id_provider(lambda: f"test-id-{next(counter)}")
    try:
        assert adk_uuid.new_uuid() == "test-id-1"
        assert adk_uuid.new_uuid() == "test-id-2"
    finally:
        adk_uuid.reset_id_provider()
```

### Example 3 — async-isolated providers via ContextVar

```python
import asyncio
from google.adk.platform import time as adk_time

async def task_a():
    adk_time.set_time_provider(lambda: 1000.0)
    await asyncio.sleep(0)        # yield; task_b runs here
    # ContextVar is still 1000.0 in this task's context
    assert adk_time.get_time() == 1000.0

async def task_b():
    # task_b's context inherits the default (real time.time).
    # set_time_provider in task_a does NOT leak here.
    ts = adk_time.get_time()
    assert ts != 1000.0, "provider isolation maintained"

async def main():
    await asyncio.gather(task_a(), task_b())

asyncio.run(main())
```

---

## 6 · `SchemaType` + `_schema_utils` — unified schema validation helpers

**Source:** `google/adk/utils/_schema_utils.py`

`_schema_utils` provides a small set of helpers that normalise the four schema
forms accepted by ADK (`type[BaseModel]`, `list[BaseModel]` generic alias,
`dict`, `Schema`) under a single `SchemaType = types.SchemaUnion` alias.

### Type hierarchy and dispatch

```python
SchemaType = types.SchemaUnion
# SchemaUnion accepts: type[BaseModel], GenericAlias (list[str] etc.),
# dict, or google.genai.types.Schema.

def is_basemodel_schema(schema: SchemaType) -> bool:
    return isinstance(schema, type) and issubclass(schema, BaseModel)

def is_list_of_basemodel(schema: SchemaType) -> bool:
    return get_origin(schema) is list and issubclass(get_args(schema)[0], BaseModel)

def get_list_inner_type(schema: SchemaType) -> Optional[type[BaseModel]]:
    ...  # returns the inner class from list[SomeModel], or None

def schema_to_json_schema(schema: SchemaType) -> dict[str, Any]:
    if isinstance(schema, dict): return schema
    return TypeAdapter(schema).json_schema()   # works for BaseModel, list[X], etc.

def validate_schema(schema: SchemaType, json_text: str) -> Any:
    if is_basemodel_schema(schema):
        return schema.model_validate_json(json_text).model_dump(exclude_none=True)
    elif is_list_of_basemodel(schema):
        validated = TypeAdapter(schema).validate_json(json_text)
        return [item.model_dump(exclude_none=True) for item in validated]
    else:
        return json.loads(json_text)  # raw dict, list[str], Schema etc.
```

### Example 1 — validating a Pydantic model schema

```python
from pydantic import BaseModel
from google.adk.utils._schema_utils import validate_schema, is_basemodel_schema

class WeatherReport(BaseModel):
    city: str
    temp_c: float
    condition: str

assert is_basemodel_schema(WeatherReport)

result = validate_schema(
    WeatherReport,
    '{"city": "London", "temp_c": 12.5, "condition": "cloudy"}',
)
# result is a plain dict (exclude_none=True applied)
print(result)  # {'city': 'London', 'temp_c': 12.5, 'condition': 'cloudy'}
```

### Example 2 — validating a list[BaseModel] schema

```python
from pydantic import BaseModel
from google.adk.utils._schema_utils import validate_schema, is_list_of_basemodel

class Item(BaseModel):
    name: str
    qty: int

schema = list[Item]
assert is_list_of_basemodel(schema)

result = validate_schema(schema, '[{"name": "apple", "qty": 3}]')
print(result)  # [{'name': 'apple', 'qty': 3}]
```

### Example 3 — generating JSON Schema from a complex type

```python
from pydantic import BaseModel
from google.adk.utils._schema_utils import schema_to_json_schema

class Address(BaseModel):
    street: str
    city: str

# Works for BaseModel, list[BaseModel], list[str], dict, etc.
js = schema_to_json_schema(list[Address])
print(js["type"])         # "array"
print(js["items"]["properties"].keys())  # dict_keys(['street', 'city'])

# Plain dict passes through unchanged:
raw = {"type": "object", "properties": {"x": {"type": "string"}}}
assert schema_to_json_schema(raw) is raw
```

---

## 7 · `InteractionsRequestProcessor` + `_find_previous_interaction_state`

**Source:** `google/adk/flows/llm_flows/interactions_processor.py`

These two utilities enable **stateful multi-turn Interactions API conversations**
where Gemini chains turns using `previous_interaction_id` instead of
re-sending the full conversation history.

### How it works

```python
def _find_previous_interaction_state(
    events: list[Event],
    *,
    agent_name: str,
    current_branch: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Scan reversed events for the last interaction_id authored by agent_name
    within current_branch.  Returns (interaction_id, environment_id)."""
```

`_is_event_in_branch` applies the matching rule:
- Root branch (`current_branch is None`): include events where `event.branch` is falsy.
- Named branch: include `event.branch == current_branch` **or** `not event.branch`.

`InteractionsRequestProcessor.run_async()` only activates when the agent's
canonical model is a `Gemini` instance with `use_interactions_api=True`. When
active, it sets `llm_request.previous_interaction_id` to the discovered ID
and then returns without yielding any events.

### Example 1 — observing interaction_id chaining in session events

```python
import asyncio
from google.adk.flows.llm_flows.interactions_processor import (
    _find_previous_interaction_state,
)

# Simulate a session where the first turn stored an interaction_id.
class FakeEvent:
    def __init__(self, author, interaction_id, branch=None, environment_id=None):
        self.author = author
        self.interaction_id = interaction_id
        self.branch = branch
        self.environment_id = environment_id

events = [
    FakeEvent("root_agent", None, branch=None),           # no ID yet
    FakeEvent("root_agent", "interaction-abc123", branch=None),  # first turn done
]

interaction_id, env_id = _find_previous_interaction_state(
    events, agent_name="root_agent", current_branch=None
)
print(interaction_id)  # "interaction-abc123"
print(env_id)          # None (no environment_id in these events)
```

### Example 2 — branch-isolated interaction state

```python
from google.adk.flows.llm_flows.interactions_processor import (
    _find_previous_interaction_state,
)

class FakeEvent:
    def __init__(self, author, iid, branch=None, env_id=None):
        self.author = author
        self.interaction_id = iid
        self.branch = branch
        self.environment_id = env_id

events = [
    FakeEvent("agent", "id-main",   branch=None),     # root branch
    FakeEvent("agent", "id-branch", branch="branch1"),  # branch1 interaction
]

# When searching for branch1, we find id-branch.
iid, _ = _find_previous_interaction_state(
    events, agent_name="agent", current_branch="branch1"
)
assert iid == "id-branch"

# When searching for root (None), root-branch events win (branch=None).
iid, _ = _find_previous_interaction_state(
    events, agent_name="agent", current_branch=None
)
assert iid == "id-main"
```

### Example 3 — wiring with a Gemini model using the Interactions API

```python
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

# The InteractionsRequestProcessor is a module-level singleton registered
# in the LLM flow when Gemini.use_interactions_api is True.
# Typical usage: pass use_interactions_api via model init or env flag.
model = Gemini(model="gemini-2.5-flash", use_interactions_api=True)

agent = LlmAgent(
    name="stateful_agent",
    model=model,   # processor activates automatically when this model is used
    instruction="You are a helpful assistant with stateful conversation memory.",
)

# On the second and subsequent turns, llm_request.previous_interaction_id
# is set automatically by InteractionsRequestProcessor.run_async(), enabling
# the Gemini backend to resume the conversation without re-sending history.
```

---

## 8 · `include_artifacts_in_a2a_event_interceptor` — A2A artifact injection

**Source:** `google/adk/a2a/executor/interceptors/include_artifacts_in_a2a_event.py`

This module-level `ExecuteInterceptor` instance hooks into the A2A executor's
`after_event` callback to translate ADK `artifact_delta` entries into A2A
`TaskArtifactUpdateEvent` objects, so that artifact uploads made during an
ADK agent turn are surfaced through the A2A streaming protocol.

### How it works (verified source)

```python
async def _after_agent(
    ctx: ExecutorContext, a2a_event: A2AEvent, adk_event: Event
) -> Union[A2AEvent, list[A2AEvent]]:
    if isinstance(a2a_event, (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)):
        artifact_service = ctx.runner.artifact_service
        if artifact_service and adk_event.actions.artifact_delta:
            new_events = []
            for filename, version in adk_event.actions.artifact_delta.items():
                genai_part = await artifact_service.load_artifact(
                    app_name=ctx.app_name,
                    user_id=ctx.user_id,
                    session_id=ctx.session_id,
                    filename=filename,
                    version=version,
                )
                if genai_part:
                    a2a_part = convert_genai_part_to_a2a_part(genai_part)
                    new_event = TaskArtifactUpdateEvent(
                        task_id=a2a_event.task_id,
                        context_id=a2a_event.context_id,
                        artifact=Artifact(
                            artifactId=f"{filename}_{version}",
                            name=filename,
                            parts=[a2a_part],
                        ),
                    )
                    new_events.append(new_event)
            adk_event.actions.artifact_delta = {}  # consumed
            if new_events:
                return [a2a_event] + new_events
    return a2a_event

include_artifacts_in_a2a_event_interceptor = ExecuteInterceptor(after_event=_after_agent)
```

The interceptor clears `artifact_delta` after processing to prevent
double-emission on repeated calls. When no artifacts are present, the original
`a2a_event` is returned unchanged (single object, not list).

### Example 1 — attaching the interceptor to an A2A executor

```python
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import (
    include_artifacts_in_a2a_event_interceptor,
)

config = A2aAgentExecutorConfig(
    execute_interceptors=[include_artifacts_in_a2a_event_interceptor],
)
# runner is a required keyword argument — pass a Runner instance or factory.
executor = A2aAgentExecutor(runner=my_runner, config=config)
```

### Example 2 — verifying artifact surfacing in a mock context

```python
from unittest.mock import AsyncMock, MagicMock
from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import (
    _after_agent,
)
from a2a.types import TaskStatusUpdateEvent

async def test_artifact_injection():
    genai_part = MagicMock()
    genai_part.inline_data = MagicMock(data=b"pdf-bytes", mime_type="application/pdf")

    artifact_service = AsyncMock()
    artifact_service.load_artifact.return_value = genai_part

    runner = MagicMock()
    runner.artifact_service = artifact_service
    ctx = MagicMock(app_name="app", user_id="u1", session_id="s1", runner=runner)

    a2a_event = TaskStatusUpdateEvent(task_id="t1", context_id="c1", status=MagicMock(), final=False)
    adk_event = MagicMock()
    adk_event.actions.artifact_delta = {"report.pdf": 0}

    result = await _after_agent(ctx, a2a_event, adk_event)
    assert isinstance(result, list)
    assert len(result) == 2   # original status event + artifact event
    assert adk_event.actions.artifact_delta == {}  # consumed
```

### Example 3 — understanding when the interceptor is a no-op

```python
# The interceptor short-circuits for non-status/artifact event types
# and when artifact_delta is empty.

from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import (
    _after_agent,
)
from unittest.mock import AsyncMock, MagicMock
from a2a.types import TaskStatusUpdateEvent

async def test_no_op_when_no_delta():
    ctx = MagicMock(runner=MagicMock(artifact_service=AsyncMock()))
    a2a_event = TaskStatusUpdateEvent(task_id="t1", context_id="c1", status=MagicMock(), final=False)
    adk_event = MagicMock()
    adk_event.actions.artifact_delta = {}    # empty → no processing

    result = await _after_agent(ctx, a2a_event, adk_event)
    assert result is a2a_event   # returned unchanged (not a list)
```

---

## 9 · `MtlsEndpoint` + mTLS utilities

**Source:** `google/adk/utils/_mtls_utils.py`

These helpers centralise the mTLS endpoint-selection logic used by Google API
service clients (`SecretManagerClient`, `ParameterManagerClient`, etc.) in ADK.

### API (verified `_mtls_utils.py`)

```python
class MtlsEndpoint(str, Enum):
    AUTO   = "auto"    # use mTLS if a client cert is present
    ALWAYS = "always"  # always use mTLS endpoint
    NEVER  = "never"   # never use mTLS endpoint

def _mtls_endpoint_setting() -> MtlsEndpoint:
    """Reads GOOGLE_API_USE_MTLS_ENDPOINT; defaults to AUTO on invalid values."""

def use_client_cert_effective() -> bool:
    """True if google.auth.transport.mtls.should_use_client_cert() or
    GOOGLE_API_USE_CLIENT_CERTIFICATE=true."""

def get_api_endpoint(location: str, default_template: str, mtls_template: str) -> str:
    """Interpolates the correct template based on GOOGLE_API_USE_MTLS_ENDPOINT
    and whether a client cert is present."""

def is_non_mtls_googleapis_endpoint(url: str) -> bool:
    """True when host ends with .googleapis.com but NOT .mtls.googleapis.com."""

def effective_googleapis_endpoint(url: str) -> str:
    """Rewrites a .googleapis.com URL to its .mtls.googleapis.com variant."""
```

### Example 1 — resolving a regional endpoint with mTLS

```python
import os
from google.adk.utils._mtls_utils import get_api_endpoint, MtlsEndpoint

# With no env vars, AUTO mode + no client cert → non-mTLS endpoint.
os.environ.pop("GOOGLE_API_USE_MTLS_ENDPOINT", None)
os.environ.pop("GOOGLE_API_USE_CLIENT_CERTIFICATE", None)

endpoint = get_api_endpoint(
    location="us-central1",
    default_template="secretmanager.{location}.rep.googleapis.com",
    mtls_template="secretmanager.{location}.rep.mtls.googleapis.com",
)
print(endpoint)  # "secretmanager.us-central1.rep.googleapis.com"
```

### Example 2 — forcing mTLS always

```python
import os
from google.adk.utils._mtls_utils import get_api_endpoint

os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "always"

endpoint = get_api_endpoint(
    location="europe-west1",
    default_template="parametermanager.{location}.rep.googleapis.com",
    mtls_template="parametermanager.{location}.rep.mtls.googleapis.com",
)
print(endpoint)  # "parametermanager.europe-west1.rep.mtls.googleapis.com"

del os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"]
```

### Example 3 — URL rewriting for dynamic endpoints

```python
from google.adk.utils._mtls_utils import (
    effective_googleapis_endpoint,
    is_non_mtls_googleapis_endpoint,
)

url = "https://secretmanager.us-central1.rep.googleapis.com/v1/projects/p/secrets"

assert is_non_mtls_googleapis_endpoint(url)

# Rewrite to mTLS variant when client cert is active:
import os
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "true"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "auto"

rewritten = effective_googleapis_endpoint(url)
print(rewritten)
# "https://secretmanager.us-central1.rep.mtls.googleapis.com/v1/projects/p/secrets"

del os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"]
del os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"]
```

---

## 10 · `PubSubCredentialsConfig` + `PubSubToolConfig`

**Sources:**
- `google/adk/tools/pubsub/pubsub_credentials.py`
- `google/adk/tools/pubsub/config.py`

These two Pydantic models configure authentication and project context for
`PubSubToolset`. Both are `@experimental`.

### `PubSubCredentialsConfig`

```python
PUBSUB_TOKEN_CACHE_KEY = "pubsub_token_cache"
PUBSUB_DEFAULT_SCOPE = ("https://www.googleapis.com/auth/pubsub",)

@experimental(FeatureName.GOOGLE_CREDENTIALS_CONFIG)
class PubSubCredentialsConfig(BaseGoogleCredentialsConfig):
    @model_validator(mode="after")
    def __post_init__(self) -> PubSubCredentialsConfig:
        super().__post_init__()
        if not self.scopes:
            self.scopes = PUBSUB_DEFAULT_SCOPE
        self._token_cache_key = PUBSUB_TOKEN_CACHE_KEY
        return self
```

It extends `BaseGoogleCredentialsConfig` (which handles `credentials`,
`external_access_token_key`, `client_id`/`client_secret`/`scopes` and enforces
mutual exclusivity between auth modes) adding:
- **Default scope**: `https://www.googleapis.com/auth/pubsub` when no scopes are provided.
- **Token cache key**: `"pubsub_token_cache"` for session-state caching.

### `PubSubToolConfig`

```python
@experimental(FeatureName.PUBSUB_TOOL_CONFIG)
class PubSubToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str | None = None
```

A minimal config that pins a GCP project. When `project_id` is `None` the
Pub/Sub client library infers it from the environment or credentials.

### Example 1 — OAuth2 credentials for PubSub

```python
from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset

creds_config = PubSubCredentialsConfig(
    client_id="my-oauth-client-id",
    client_secret="my-oauth-client-secret",
    # scopes defaults to PUBSUB_DEFAULT_SCOPE automatically
)
# project_id lives in PubSubToolConfig; topic/subscription names are
# arguments to the individual publish/pull tools, not the toolset.
tool_config = PubSubToolConfig(project_id="my-project")

toolset = PubSubToolset(
    credentials_config=creds_config,
    pubsub_tool_config=tool_config,
)
```

### Example 2 — pre-existing service account credentials

```python
import google.auth
from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset

creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/pubsub"]
)

creds_config = PubSubCredentialsConfig(
    credentials=creds,  # bypasses OAuth flow; all users share this credential
)
tool_config = PubSubToolConfig(project_id="my-project")

toolset = PubSubToolset(
    credentials_config=creds_config,
    pubsub_tool_config=tool_config,
)
```

### Example 3 — pinning the project with PubSubToolConfig

```python
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset

# PubSubToolConfig pins the GCP project at the toolset level.
# extra='forbid' means unknown fields raise ValidationError.
tool_config = PubSubToolConfig(project_id="analytics-project-1234")

toolset = PubSubToolset(
    pubsub_tool_config=tool_config,
)
```

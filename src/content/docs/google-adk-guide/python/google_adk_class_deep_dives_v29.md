---
title: "Class deep dives — volume 29 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: BaseNode (workflow node base; name validation; input/output/state schemas; rerun_on_resume; wait_for_output; @final run() + _run_impl() override; START sentinel), run_llm_agent_as_node+helpers (LlmAgent-as-workflow-node; 3 modes single_turn/chat/task; chat outer dispatch loop for chained delegations; FinishTaskTool handshake; _dispatch_task_fc idempotency), resolve_and_derive_transfer_context (4-case agent transfer router: SELF/CHILD/SIBLING/PARENT context-chain walking), HITL workflow utilities (create_request_input_event; create_request_input_response; create_auth_request_event; process_auth_resume 3-format dispatch; has_auth_credential), _AuthLlmRequestProcessor+_store_auth_and_collect_resume_targets (auth response processor; 3-step scan-store-collect; TOOLSET_AUTH_CREDENTIAL_ID_PREFIX discrimination), inject_session_state+_is_valid_state_name (async template substitution; {var_name}/{artifact.file_name}/{var?} syntax; state prefix validation), convert_event_to_a2a_events+AdkEventToA2AEventsConverter+A2AUpdateEvent (ADK→A2A event conversion; artifact lifecycle tracking; create_error_status_event; metadata propagation; @a2a_experimental), AgentRunRequest+convert_a2a_request_to_agent_run_request+A2ARequestToAgentRunRequestConverter (A2A→ADK request bridge; user_id from call_context vs context_id; A2A metadata in run_config.custom_metadata; @a2a_experimental), ExecutorContext (A2A executor context; immutable read-only properties: app_name/user_id/session_id/runner), print_event+content_utils (debug utilities: smart text buffering; verbose/quiet modes; is_audio_part; filter_audio_parts; extract_text_from_content)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 29"
  order: 98
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source at
`/usr/local/lib/python3.11/dist-packages/google/adk/` on
**google-adk == 2.3.0**. No documentation or blog posts were used as primary
sources.
</Aside>

## 1 · `BaseNode` — the workflow node contract

**Module:** `google.adk.workflow._base_node`

`BaseNode` is the Pydantic `BaseModel` that every node in a `Workflow` graph must inherit from. It defines the entire contract: identity, timing policy, schema validation, and the execution entry point.

### Key constructor fields (verified from source)

```python
class BaseNode(BaseModel):
    name: str = Field(...)               # @field_validator: must be valid Python identifier
    description: str = ''
    rerun_on_resume: bool = False        # True → re-execute from scratch on HITL resume
    wait_for_output: bool = False        # True → WAITING state without output/route (enables multi-trigger)
    retry_config: RetryConfig | None = None
    timeout: float | None = None         # seconds; integrates with retry_config
    input_schema: SchemaType | None = None
    output_schema: SchemaType | None = None
    state_schema: type[BaseModel] | None = None  # child nodes inherit via InvocationContext
```

### `@final run()` — the execution entry point

`run()` is decorated `@final` to prevent override. It:

1. Calls `_validate_input_data(node_input)` — handles `types.Content` → text → JSON path
2. Iterates `_run_impl()` yields, normalising each:
   - `None` → skipped
   - `Event` → pass through (validates `event.output` if set)
   - `RequestInput` → converted to HITL interrupt `Event`
   - Any other value → `Event(output=validated_value)`

```python
@final
async def run(self, *, ctx: Context, node_input: Any) -> AsyncGenerator[Event, None]:
    node_input = self._validate_input_data(node_input)
    # Aclosing is google.adk.utils.context_utils.Aclosing, not contextlib.aclosing
    async with Aclosing(self._run_impl(ctx=ctx, node_input=node_input)) as agen:
        async for item in agen:
            if item is None:
                continue
            if isinstance(item, Event):
                if item.output is not None:
                    item.output = self._validate_output_data(item.output)
                yield item
            elif isinstance(item, RequestInput):
                yield create_request_input_event(item)
            else:
                validated = self._validate_output_data(item)
                yield Event(output=validated)
```

### `_run_impl()` — the override point

Subclasses override `_run_impl()` and can yield `Event`, `RequestInput`, raw data, or `None`. The `_validate_schema()` method uses Pydantic `TypeAdapter` and skips `dict`/`types.Schema` descriptive schemas. `BaseModel` results are converted to `dict` via `model_dump()` to keep `Event.output` JSON-serialisable.

### `START` sentinel

```python
START = BaseNode(name='__START__')
```

`START` is never executed. `Workflow._seed_start_triggers()` bypasses it and seeds triggers for its direct successors directly.

### Example 1 — minimal custom node

```python
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context

class EchoNode(BaseNode):
    name: str = "echo"
    description: str = "Echoes node_input as output"

    async def _run_impl(self, *, ctx: Context, node_input):
        yield f"Echo: {node_input}"

# In a Workflow:
# Workflow(name="demo", agent=root_agent, nodes=[echo], edges=[(START, echo)])
```

### Example 2 — typed input/output schema + state schema

```python
from pydantic import BaseModel
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context
from google.adk.events.event import Event

class InputModel(BaseModel):
    query: str
    max_results: int = 10

class OutputModel(BaseModel):
    results: list[str]
    count: int

class SearchState(BaseModel):
    last_query: str = ""
    total_searches: int = 0

class SearchNode(BaseNode):
    name: str = "search"
    input_schema: type = InputModel
    output_schema: type = OutputModel
    state_schema: type = SearchState

    async def _run_impl(self, *, ctx: Context, node_input: InputModel):
        # state_schema means ctx.state mutations are validated at runtime
        ctx.state["last_query"] = node_input.query
        ctx.state["total_searches"] = ctx.state.get("total_searches", 0) + 1
        results = [f"Result {i} for '{node_input.query}'" for i in range(node_input.max_results)]
        yield {"results": results, "count": len(results)}  # validated against OutputModel
```

### Example 3 — `wait_for_output` for multi-trigger fan-in (JoinNode pattern)

```python
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context

class CollectorNode(BaseNode):
    name: str = "collector"
    wait_for_output: bool = True   # stays WAITING until all branches deliver

    async def _run_impl(self, *, ctx: Context, node_input):
        # Use ctx.state so the list survives HITL resumes and is isolated per
        # workflow execution (not a shared class-level attribute).
        received = ctx.state.setdefault("received_inputs", [])
        received.append(node_input)
        if len(received) >= 3:               # wait for 3 upstream branches
            combined = " | ".join(received)
            ctx.state["received_inputs"] = []
            yield combined                   # only now does the node emit output
        # else: yields nothing → stays in WAITING state
```

---

## 2 · `run_llm_agent_as_node` — LlmAgent as a workflow node

**Module:** `google.adk.workflow._llm_agent_wrapper`

`run_llm_agent_as_node` is the bridge that lets `LlmAgent` participate in a `Workflow` graph. It supports three **modes**:

| Mode | Behaviour |
|---|---|
| `single_turn` | One LLM turn; `include_contents='none'` (no session history); default for workflow nodes |
| `chat` | Multi-turn coordinator; outer dispatch loop re-enters `agent.run_async` after each task delegation |
| `task` | Waits for `FinishTaskTool` success before terminating; validates output against `agent.output_schema` |

### Key implementation facts (from source)

- **`_dispatch_task_fc`** passes `run_id=fc.id` to `ctx.run_node()` — same FC always maps to the same child run, making delegations **idempotent** across resumes.
- **`_find_unresolved_task_delegations`** does NOT filter by `isolation_scope`. A chat coordinator's conversation spans user turns; filtering by the current turn's scope would hide prior FCs.
- **Chat outer while-loop**: after each task delegation dispatch, the loop **re-enters `agent.run_async`** so the LLM sees the synthesised function response and can chain further delegations.
- **Task mode**: `_is_finish_task_success_fr()` returns `False` for validation errors, allowing the LLM to retry. Only a `FINISH_TASK_SUCCESS_RESULT` terminates the loop.

### `process_llm_agent_output` — output extraction

```python
def process_llm_agent_output(agent, ctx, event):
    # Skips FC events, partial events, non-model events
    text = ''.join(p.text for p in event.content.parts if p.text and not p.thought)
    if agent.output_schema:
        output = validate_schema(agent.output_schema, text) if text.strip() else None
    else:
        output = text
    if agent.output_key and output is not None:
        ctx.actions.state_delta[agent.output_key] = output
    event.output = output
    event.node_info.message_as_output = True
```

### Example 1 — `single_turn` node in a Workflow

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.agents.context import Context

summariser = LlmAgent(
    model="gemini-2.5-flash",
    name="summariser",
    instruction="Summarise the input text in one sentence.",
    mode="single_turn",   # default; no session history; consumes node_input
    output_key="summary", # saves result to state["summary"]
)

wf = Workflow(
    name="summary_pipeline",
    agent=summariser,   # used as root; also the only node here
)
```

### Example 2 — `chat` mode coordinator with chained task delegations

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.agent_tool import AgentTool

researcher = LlmAgent(
    model="gemini-2.5-flash",
    name="researcher",
    instruction="Research the topic thoroughly.",
    mode="task",
)

writer = LlmAgent(
    model="gemini-2.5-flash",
    name="writer",
    instruction="Write a report based on research findings.",
    mode="task",
)

coordinator = LlmAgent(
    model="gemini-2.5-pro",
    name="coordinator",
    instruction=(
        "Use the 'researcher' tool first, then pass results to 'writer'. "
        "Produce a final report."
    ),
    mode="chat",    # outer dispatch loop; re-enters after each task FC
    tools=[AgentTool(agent=researcher), AgentTool(agent=writer)],
)
```

### Example 3 — `task` mode with `output_schema`

```python
from pydantic import BaseModel
from google.adk.agents.llm_agent import LlmAgent

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

analyser = LlmAgent(
    model="gemini-2.5-flash",
    name="analyser",
    instruction=(
        "Analyse the text and return sentiment, confidence (0-1), "
        "and top keywords. Use the finish_task tool when done."
    ),
    mode="task",
    output_schema=AnalysisResult,  # FinishTaskTool validates FC args against this
)
# run_llm_agent_as_node waits for FinishTaskTool success FR before yielding output
# If FinishTaskTool returns a validation error, the LLM sees it and can retry
```

---

## 3 · `resolve_and_derive_transfer_context` — agent transfer routing

**Module:** `google.adk.workflow.utils._transfer_utils`

`resolve_and_derive_transfer_context` resolves the target agent and derives its parent context in a single pass when `transfer_to_agent` is set on an event action.

### Full signature (verified from source)

```python
def resolve_and_derive_transfer_context(
    target_name: str,
    current_agent: BaseAgent,
    root_agent: BaseAgent,
    curr_ctx: Context,
    curr_parent_ctx: Context | None,
) -> tuple[BaseAgent, Context | None] | tuple[None, None]:
```

### Four routing cases

| Case | Condition | Returns |
|---|---|---|
| **SELF** | `target.name == current.name` | Raises `ValueError` |
| **CHILD** | `target.parent_agent.name == current.name` | `(target, curr_ctx)` — nests deeper |
| **SIBLING** | same parent as current | `(target, curr_parent_ctx)` — shares parent context |
| **PARENT** | `current.parent_agent.name == target.name` | Walks context chain; falls back to outermost root context |
| **Unrelated** | no routing relationship found | `(target, None)` |

The PARENT case walks `curr_ctx.parent_ctx` until it finds a context whose `node.name` matches `target_name`. If the walk exhausts without a match (root coordinator bypass), it returns the outermost root context.

### Example 1 — child transfer (nest deeper)

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event, EventActions

parent = LlmAgent(model="gemini-2.5-flash", name="parent",
                  instruction="Route to child when needed.")
child  = LlmAgent(model="gemini-2.5-flash", name="child",
                  instruction="Handle child tasks.", parent_agent=parent)
parent.sub_agents = [child]

# Inside the runtime, when parent emits:
# Event(actions=EventActions(transfer_to_agent="child"))
# resolve_and_derive_transfer_context returns (child, curr_ctx)
# → child runs nested under parent's context
```

### Example 2 — sibling transfer

```python
from google.adk.agents.llm_agent import LlmAgent

orchestrator = LlmAgent(model="gemini-2.5-pro", name="orchestrator",
                        instruction="Route between specialist agents.")
specialist_a = LlmAgent(model="gemini-2.5-flash", name="specialist_a",
                        instruction="Handle topic A.")
specialist_b = LlmAgent(model="gemini-2.5-flash", name="specialist_b",
                        instruction="Handle topic B.")
orchestrator.sub_agents = [specialist_a, specialist_b]

# specialist_a emitting transfer_to_agent="specialist_b":
# Both share orchestrator as parent
# resolve_and_derive_transfer_context returns (specialist_b, curr_parent_ctx)
# → specialist_b runs under the same parent context
```

### Example 3 — parent transfer (climb up)

```python
from google.adk.agents.llm_agent import LlmAgent

root = LlmAgent(model="gemini-2.5-pro", name="root",
                instruction="Top-level coordinator.")
mid  = LlmAgent(model="gemini-2.5-flash", name="mid",
                instruction="Middle layer.", parent_agent=root)
leaf = LlmAgent(model="gemini-2.5-flash", name="leaf",
                instruction="Do leaf work then hand back to root.", parent_agent=mid)
root.sub_agents = [mid]
mid.sub_agents = [leaf]

# leaf emitting transfer_to_agent="root" (its grandparent):
# Case 4 logic: leaf.parent_agent.name == "mid", not "root"
# Falls back to outermost root context walk
# Returns (root, root_ctx)
```

---

## 4 · HITL workflow utilities — `_workflow_hitl_utils`

**Module:** `google.adk.workflow.utils._workflow_hitl_utils`

This module provides the low-level building blocks for Human-in-the-Loop (HITL) patterns in `Workflow`. All HITL events encode as function calls so the A2A protocol, web UI, and CLI can all handle them uniformly.

### Constants

```python
REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = 'adk_request_credential'
_RESULT_KEY = 'result'  # wraps non-dict values in FunctionResponse
```

### Key functions (verified from source)

```python
# Create an interrupt Event from a RequestInput object.
# Sets long_running_tool_ids=[request_input.interrupt_id]
# Converts response_schema to JSON schema for wire transport.
def create_request_input_event(request_input: RequestInput) -> Event: ...

# Create the FunctionResponse Part that resumes a suspended node.
def create_request_input_response(interrupt_id: str, response: Mapping[str, Any]) -> types.Part: ...

# Extract all interrupt_ids from an adk_request_input event.
def get_request_input_interrupt_ids(event: Event) -> list[str]: ...

# Build a human-readable description of what credential is needed.
# Dispatches on AuthCredentialTypes: API_KEY → "Please provide your API key for {name}."
#                                   OAUTH2/OIDC → "Please complete the authentication flow."
def _build_auth_message(auth_config: AuthConfig) -> str: ...

# Create an interrupt Event requesting authentication credentials.
def create_auth_request_event(auth_config: AuthConfig, interrupt_id: str) -> Event: ...

# Store credentials from a client auth-resume response into session state.
# Tries 3 formats in order:
#   1. Full AuthConfig dict (web UI OAuth flow)
#   2. AuthCredential dict
#   3. Plain value (string for API key)
async def process_auth_resume(response_data: Any, auth_config: AuthConfig, state: State) -> None: ...

# Returns True if a credential for the given auth config already exists in state.
def has_auth_credential(auth_config: AuthConfig, state: State) -> bool: ...
```

### Example 1 — custom node that requests user input

```python
import uuid
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_event,
    create_request_input_response,
    get_request_input_interrupt_ids,
)

class ApprovalNode(BaseNode):
    name: str = "approval"
    description: str = "Pauses execution for human approval."
    # rerun_on_resume=True: _run_impl is called again on resume with the
    # human's response as node_input, so we can branch on it at the top.
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx: Context, node_input):
        # On resume node_input is the human's response — handle it first.
        if isinstance(node_input, dict) and "approved" in node_input:
            if node_input.get("approved"):
                yield f"Approved: {node_input}"
            else:
                yield "Rejected by human."
            return

        # First run: request human input with a STABLE, deterministic interrupt_id.
        # A random UUID would break workflow replay because a different ID would be
        # generated each time the node is replayed/rehydrated.
        yield RequestInput(
            interrupt_id=f"{self.name}_approval",
            message=f"Approve this action? Input: {node_input}",
            response_schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
        )
```

### Example 2 — generating and consuming auth request events

```python
import uuid
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_auth_request_event,
    process_auth_resume,
    has_auth_credential,
)
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import APIKey

# Check if credential already in state (avoids re-requesting)
api_key_scheme = APIKey(name="x-api-key")
auth_config = AuthConfig(auth_scheme=api_key_scheme)

async def maybe_request_auth(ctx, auth_config):
    if has_auth_credential(auth_config, ctx.state):
        return  # Already have credential; proceed
    interrupt_id = str(uuid.uuid4())
    event = create_auth_request_event(auth_config, interrupt_id)
    # event.content.parts[0].function_call.name == "adk_request_credential"
    # event.content.parts[0].function_call.args["message"] == "Please provide your API key for x-api-key."
    yield event
```

### Example 3 — `process_auth_resume` 3-format dispatch

```python
from google.adk.workflow.utils._workflow_hitl_utils import process_auth_resume
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import APIKey

auth_config = AuthConfig(auth_scheme=APIKey(name="my-service-key"))

async def handle_auth_response(state, user_response):
    # Format 1: full AuthConfig dict (web UI OAuth flow)
    if isinstance(user_response, dict) and "authScheme" in user_response:
        await process_auth_resume(user_response, auth_config, state)
    # Format 2: AuthCredential dict
    elif isinstance(user_response, dict) and "authType" in user_response:
        await process_auth_resume(user_response, auth_config, state)
    # Format 3: plain string (API key shortcut)
    else:
        await process_auth_resume(str(user_response), auth_config, state)
    # Credential is now in state[auth_config.credential_key]
```

---

## 5 · `_AuthLlmRequestProcessor` + `_store_auth_and_collect_resume_targets`

**Module:** `google.adk.auth.auth_preprocessor`

`_AuthLlmRequestProcessor` is a `BaseLlmRequestProcessor` that handles the **resume side** of auth: when a user-authored event arrives containing `adk_request_credential` function responses, this processor stores the credentials and re-executes the tools that originally needed auth.

### Module-level singleton

```python
request_processor = _AuthLlmRequestProcessor()
```

This singleton is registered in the LLM pipeline and runs before each LLM call.

### Key constant

```python
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'
```

Auth requests with this prefix are for **toolset auth** (pre-tool-listing phase) and do NOT map to a resumable function call. They are skipped when building `tools_to_resume`.

### `_store_auth_and_collect_resume_targets` — 3-step algorithm

```python
async def _store_auth_and_collect_resume_targets(
    events: list[Event],
    auth_fc_ids: set[str],     # IDs of adk_request_credential FCs to match
    auth_responses: dict[str, Any],   # FC ID → auth response dict from client
    state: State,
) -> set[str]:
    # Step 1: scan events for matching adk_request_credential FCs
    #         → extract AuthToolArguments (contains credential_key + function_call_id)
    # Step 2: store credentials via AuthHandler
    #         (merges credential_key from original request into client's response)
    # Step 3: collect original function call IDs to re-execute
    #         (skips entries whose function_call_id starts with TOOLSET_AUTH_CREDENTIAL_ID_PREFIX)
```

### `_AuthLlmRequestProcessor.run_async` flow

```python
async def run_async(self, invocation_context, llm_request):
    # 1. Find last user-authored event with function_responses
    # 2. Collect adk_request_credential response IDs + response dicts
    # 3. Call _store_auth_and_collect_resume_targets → tools_to_resume
    # 4. Find original FC event and call handle_function_calls_async
    #    on just the tools in tools_to_resume
    #    → yields the FunctionResponse event for the re-executed tools
```

### Example 1 — understanding auth processor placement

```python
from google.adk.auth.auth_preprocessor import request_processor, TOOLSET_AUTH_CREDENTIAL_ID_PREFIX

# The processor is a singleton registered in the LLM pipeline.
# It fires BEFORE each LLM call to process pending auth responses.
print(type(request_processor).__name__)          # _AuthLlmRequestProcessor
print(TOOLSET_AUTH_CREDENTIAL_ID_PREFIX)         # _adk_toolset_auth_

# Toolset auth credential IDs start with this prefix and are NOT re-executed.
# Only regular tool auth (where function_call_id does NOT start with the prefix)
# triggers a re-execution of the original tool.
```

### Example 2 — auth request + resume round-trip

```python
import asyncio
from google.adk.auth.auth_tool import AuthConfig, AuthToolArguments
from google.adk.auth.auth_schemes import APIKey
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_auth_request_event,
    process_auth_resume,
)
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.state import State

async def demo_auth_round_trip():
    auth_config = AuthConfig(auth_scheme=APIKey(name="weather-api-key"))

    # 1. Interrupt phase: yield auth request event to client
    import uuid
    interrupt_id = f"_adk_toolset_auth_{uuid.uuid4()}"  # toolset-prefix = no re-exec
    regular_id   = str(uuid.uuid4())                    # regular = triggers re-exec

    evt = create_auth_request_event(auth_config, regular_id)
    print(f"Auth request function call: {evt.content.parts[0].function_call.name}")

    # 2. Resume phase: client returns user-provided API key
    state = State()
    await process_auth_resume("my-secret-api-key-123", auth_config, state)
    # Credential is now stored in state under auth_config.credential_key
```

### Example 3 — custom AuthenticatedFunctionTool that relies on the processor

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import APIKey
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

async def fetch_weather(city: str, credential: str) -> str:
    """Fetch weather for a city using an API key."""
    # In production this would call a weather API with the credential
    return f"Sunny in {city} (key: {credential[:4]}...)"

weather_tool = AuthenticatedFunctionTool(
    func=fetch_weather,
    auth_config=AuthConfig(auth_scheme=APIKey(name="weather-api-key")),
    # When the tool runs without a credential, it emits adk_request_credential.
    # On the next turn the user provides the key.
    # _AuthLlmRequestProcessor detects the response, stores it, and
    # re-executes fetch_weather — all without the agent code changing.
)
```

---

## 6 · `inject_session_state` + `_is_valid_state_name`

**Module:** `google.adk.utils.instructions_utils`

`inject_session_state` populates instruction templates at call time with live session state values and artifact content. It is designed for `InstructionProvider` callables — async functions passed as `LlmAgent.instruction`.

### Template syntax (verified from source)

| Placeholder | Source | Behaviour on missing |
|---|---|---|
| `{var_name}` | `session.state[var_name]` | Raises `KeyError` |
| `{var_name?}` | `session.state[var_name]` | Replaces with `''` (logs debug) |
| `{artifact.file_name}` | `artifact_service.load_artifact(filename)` | Raises `KeyError` |
| `{artifact.file_name?}` | `artifact_service.load_artifact(filename)` | Replaces with `''` (logs debug) |

### `_is_valid_state_name` — key validation

```python
def _is_valid_state_name(var_name: str) -> bool:
    # Accepts plain identifiers ("counter") and prefixed ones ("app:counter", "user:name")
    # Prefixes: "app:", "user:", "temp:"
    # Anything else (e.g. "2bad", "app:") returns False → placeholder kept as-is
    parts = var_name.split(':')
    if len(parts) == 1:
        return var_name.isidentifier()
    if len(parts) == 2:
        prefixes = [State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX]
        if (parts[0] + ':') in prefixes:
            return parts[1].isidentifier()
    return False
```

Invalid names are left unchanged in the template (not substituted, not raised).

### Example 1 — state variable substitution in an instruction

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils.instructions_utils import inject_session_state

async def personalised_instruction(ctx: ReadonlyContext) -> str:
    return await inject_session_state(
        "You are assisting {user:name} who prefers {user:language} responses. "
        "Their current project is {project_name}. "
        "Unknown variable {2bad} stays as-is (invalid identifier).",
        ctx,
    )
    # With state: {"user:name": "Alice", "user:language": "English", "project_name": "Demo"}
    # Returns: "You are assisting Alice who prefers English responses.
    #           Their current project is Demo. Unknown variable {2bad} stays as-is."

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="assistant",
    instruction=personalised_instruction,  # called on every LLM turn
)
```

### Example 2 — optional variables (no KeyError on missing)

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils.instructions_utils import inject_session_state

async def conditional_instruction(ctx: ReadonlyContext) -> str:
    return await inject_session_state(
        "You are a helpful assistant. "
        "{user:preferred_tone?}"  # → '' if user:preferred_tone not in state
        "Context window: {app:context_limit?}",  # → '' if missing
        ctx,
    )
# Useful when some state keys are only set after certain workflow branches.
```

### Example 3 — artifact injection into an instruction

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils.instructions_utils import inject_session_state

async def document_aware_instruction(ctx: ReadonlyContext) -> str:
    return await inject_session_state(
        "You are a document analyst.\n"
        "Here is the document to analyse:\n{artifact.user_document.txt}\n"
        "Optional glossary: {artifact.glossary.txt?}",
        ctx,
    )
    # artifact_service.load_artifact(filename="user_document.txt") is called async
    # If glossary.txt doesn't exist → replaced with '' (optional marker)

agent = LlmAgent(
    model="gemini-2.5-pro",
    name="analyst",
    instruction=document_aware_instruction,
)
```

---

## 7 · `convert_event_to_a2a_events` + `AdkEventToA2AEventsConverter`

**Module:** `google.adk.a2a.converters.from_adk_event`

This module converts ADK `Event` objects into A2A protocol events for streaming to A2A clients. It handles artifact lifecycle tracking across streaming chunks.

### Type aliases (verified from source)

```python
A2AUpdateEvent = Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]

AdkEventToA2AEventsConverter = Callable[
    [Event, Optional[Dict[str, str]], Optional[str], Optional[str], GenAIPartToA2APartConverter],
    List[A2AUpdateEvent],
]
```

`agents_artifacts: Dict[str, str]` is a **stateful dict** maintained across calls — `{agent_name: artifact_id}`. It tracks which artifact is currently being streamed for each agent.

### Artifact lifecycle logic (from source)

```python
artifact_id = agents_artifacts.get(agent_name)
if artifact_id:
    append = partial           # continuing an existing streaming artifact
    if not partial:
        del agents_artifacts[agent_name]  # last chunk: remove from tracking
else:
    artifact_id = str(uuid.uuid4())      # new artifact
    append = False
    if partial:
        agents_artifacts[agent_name] = artifact_id  # only track if more chunks coming
```

### `create_error_status_event` — error wrapper

```python
def create_error_status_event(
    event: Event,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> TaskStatusUpdateEvent:
    # Creates TaskStatusUpdateEvent with state=TaskState.failed, final=True
    # Error message from event.error_message or DEFAULT_ERROR_MESSAGE
    # Applies _add_event_metadata (invocation_id, author, etc.)
```

### `_add_event_metadata` — fields propagated to A2A events

`invocation_id`, `author`, `event_id`, `branch`, `citation_metadata`, `grounding_metadata`, `custom_metadata`, `usage_metadata`, `error_code`, `actions`.

Fields are namespaced via `_get_adk_metadata_key()` and set on `status.message.metadata` (for `TaskStatusUpdateEvent`) or `artifact.metadata` (for `TaskArtifactUpdateEvent`).

### Example 1 — basic event conversion

```python
from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events
from google.adk.events.event import Event
from google.genai import types

event = Event(
    author="my_agent",
    content=types.Content(role="model", parts=[types.Part(text="Hello from ADK!")]),
    partial=False,
)

agents_artifacts: dict = {}  # stateful across multiple event conversions
a2a_events = convert_event_to_a2a_events(
    event, agents_artifacts, task_id="task-1", context_id="ctx-1"
)
# Returns [TaskArtifactUpdateEvent] with artifact containing the text part
# agents_artifacts is now empty (non-partial → artifact complete)
print(type(a2a_events[0]).__name__)  # TaskArtifactUpdateEvent
```

### Example 2 — streaming chunk lifecycle

```python
from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events
from google.adk.events.event import Event
from google.genai import types

agents_artifacts: dict = {}

# Chunk 1: partial=True → creates artifact ID, tracked in agents_artifacts
chunk1 = Event(
    author="streaming_agent",
    content=types.Content(role="model", parts=[types.Part(text="Hello ")]),
    partial=True,
)
events1 = convert_event_to_a2a_events(chunk1, agents_artifacts, "task-1", "ctx-1")
print(len(agents_artifacts))           # 1 — artifact ID tracked

# Chunk 2: partial=True → reuses tracked artifact_id, append=True
chunk2 = Event(
    author="streaming_agent",
    content=types.Content(role="model", parts=[types.Part(text="world!")]),
    partial=True,
)
events2 = convert_event_to_a2a_events(chunk2, agents_artifacts, "task-1", "ctx-1")
print(events2[0].append)               # True

# Chunk 3: partial=False → last_chunk=True, clears agents_artifacts
chunk3 = Event(
    author="streaming_agent",
    content=types.Content(role="model", parts=[types.Part(text=" Done.")]),
    partial=False,
)
events3 = convert_event_to_a2a_events(chunk3, agents_artifacts, "task-1", "ctx-1")
print(events3[0].last_chunk)           # True
print(len(agents_artifacts))           # 0 — cleaned up
```

### Example 3 — error event and custom converter

```python
from google.adk.a2a.converters.from_adk_event import (
    convert_event_to_a2a_events,
    create_error_status_event,
    AdkEventToA2AEventsConverter,
)
from google.adk.events.event import Event

# Create an error status event
error_event = Event(author="my_agent", error_message="Tool call timed out")
a2a_error = create_error_status_event(error_event, task_id="t1", context_id="c1")
print(a2a_error.status.state)   # TaskState.failed
print(a2a_error.final)          # True

# Use the type alias to type-hint a custom converter
def my_converter(
    event, agents_artifacts, task_id, context_id, part_converter
) -> list:
    # Custom logic: filter thought parts before converting
    if event.content:
        event.content.parts = [p for p in event.content.parts if not p.thought]
    return convert_event_to_a2a_events(event, agents_artifacts, task_id, context_id, part_converter)

custom: AdkEventToA2AEventsConverter = my_converter
```

---

## 8 · `AgentRunRequest` + `convert_a2a_request_to_agent_run_request`

**Module:** `google.adk.a2a.converters.request_converter`

`AgentRunRequest` is the ADK-side model populated from an incoming A2A `RequestContext`. It decouples the A2A wire format from the `Runner.run_async()` call signature.

### `AgentRunRequest` fields (verified from source)

```python
@a2a_experimental
class AgentRunRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    invocation_id: Optional[str] = None
    new_message: Optional[genai_types.Content] = None
    state_delta: Optional[dict[str, Any]] = None
    run_config: Optional[RunConfig] = None
```

### `_get_user_id` — user identity derivation

```python
def _get_user_id(request: RequestContext) -> str:
    # Priority 1: call_context.user.user_name (auth enabled on A2A server)
    if request.call_context and request.call_context.user and request.call_context.user.user_name:
        return request.call_context.user.user_name
    # Priority 2: fallback synthetic ID from context_id
    return f'A2A_USER_{request.context_id}'
```

### `convert_a2a_request_to_agent_run_request` — key facts

```python
A2A_METADATA_KEY = 'a2a_metadata'

def convert_a2a_request_to_agent_run_request(request, part_converter=...) -> AgentRunRequest:
    # 1. Derives user_id from call_context or synthetic fallback
    # 2. Sets session_id = request.context_id (A2A context = ADK session)
    # 3. Converts A2A parts to GenAI parts via part_converter
    # 4. Wraps request.metadata in run_config.custom_metadata[A2A_METADATA_KEY]
    # Raises ValueError if request.message is None
```

### `A2ARequestToAgentRunRequestConverter` — the type alias

```python
A2ARequestToAgentRunRequestConverter = Callable[
    [RequestContext, A2APartToGenAIPartConverter],
    AgentRunRequest,
]
```

Pass a custom callable to override how A2A requests map to ADK runner arguments.

### Example 1 — basic conversion

```python
from unittest.mock import MagicMock
from a2a.types import Message, TextPart, Part, Role
from google.adk.a2a.converters.request_converter import (
    convert_a2a_request_to_agent_run_request,
    AgentRunRequest,
    A2A_METADATA_KEY,
)

# Simulate an incoming A2A request context
request = MagicMock()
request.call_context = None
request.context_id = "session-abc-123"
request.metadata = {"client": "web-ui", "version": "1.0"}
request.message = MagicMock()
request.message.parts = [
    MagicMock(root=MagicMock(spec=TextPart, text="What is 2+2?"))
]

result = convert_a2a_request_to_agent_run_request(request)
print(result.user_id)     # A2A_USER_session-abc-123 (no auth context)
print(result.session_id)  # session-abc-123
print(result.run_config.custom_metadata[A2A_METADATA_KEY])  # {"client": "web-ui", ...}
```

### Example 2 — authenticated request (user from call_context)

```python
from unittest.mock import MagicMock
from google.adk.a2a.converters.request_converter import convert_a2a_request_to_agent_run_request

# When A2A server has auth enabled, call_context.user is populated
request = MagicMock()
request.call_context = MagicMock()
request.call_context.user = MagicMock()
request.call_context.user.user_name = "alice@example.com"
request.context_id = "session-xyz"
request.metadata = {}
request.message = MagicMock()
request.message.parts = []

result = convert_a2a_request_to_agent_run_request(request)
print(result.user_id)   # alice@example.com (from auth context, not synthetic)
```

### Example 3 — custom converter for multi-tenant routing

```python
from google.adk.a2a.converters.request_converter import (
    AgentRunRequest,
    A2ARequestToAgentRunRequestConverter,
    convert_a2a_request_to_agent_run_request,
)
from google.adk.runners import RunConfig

def tenant_aware_converter(request, part_converter):
    base = convert_a2a_request_to_agent_run_request(request, part_converter)
    tenant_id = (request.metadata or {}).get("tenant_id", "default")
    existing = base.run_config.custom_metadata if base.run_config else {}
    return AgentRunRequest(
        user_id=f"{tenant_id}:{base.user_id}",   # namespace user under tenant
        session_id=f"{tenant_id}:{base.session_id}",
        new_message=base.new_message,
        run_config=RunConfig(custom_metadata={**existing, "tenant_id": tenant_id}),
    )

custom: A2ARequestToAgentRunRequestConverter = tenant_aware_converter
```

---

## 9 · `ExecutorContext` — A2A executor context

**Module:** `google.adk.a2a.executor.executor_context`

`ExecutorContext` is a lightweight, **immutable** context holder that the A2A executor creates once per request and passes to interceptors and callbacks. All four properties are read-only.

### Full class (verified from source)

```python
class ExecutorContext:
    def __init__(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        runner: Runner,
    ):
        self._app_name = app_name
        self._user_id = user_id
        self._session_id = session_id
        self._runner = runner

    @property
    def app_name(self) -> str: ...
    @property
    def user_id(self) -> str: ...
    @property
    def session_id(self) -> str: ...
    @property
    def runner(self) -> Runner: ...
```

### Usage context

`ExecutorContext` is passed to `A2aAgentExecutor.execute_interceptors` before the main `runner.run_async()` call. Interceptors can read it to log, validate, or transform the request without access to the raw `RequestContext`.

### Example 1 — inspect ExecutorContext in an interceptor

```python
from google.adk.a2a.executor.executor_context import ExecutorContext
from google.adk.a2a.executor.interceptors import ExecuteInterceptor

class LoggingInterceptor:
    async def before_agent(self, ctx: ExecutorContext) -> None:
        print(f"[A2A] app={ctx.app_name} user={ctx.user_id} session={ctx.session_id}")
        # ctx.runner is the live Runner; read agent name from it
        print(f"      root_agent={ctx.runner._app.agent.name if ctx.runner._app else 'N/A'}")
```

### Example 2 — build ExecutorContext for testing

```python
from google.adk.a2a.executor.executor_context import ExecutorContext
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(model="gemini-2.5-flash", name="test_agent", instruction="Reply briefly.")
session_svc = InMemorySessionService()
runner = Runner(agent=agent, session_service=session_svc)

ctx = ExecutorContext(
    app_name="test_app",
    user_id="test_user",
    session_id="test_session",
    runner=runner,
)
print(ctx.app_name)    # test_app
print(ctx.user_id)     # test_user
print(ctx.session_id)  # test_session
print(type(ctx.runner).__name__)  # Runner
```

### Example 3 — middleware pattern using ExecutorContext

```python
from google.adk.a2a.executor.executor_context import ExecutorContext
from typing import Callable, Awaitable

ExecuteMiddleware = Callable[[ExecutorContext], Awaitable[None]]

def rate_limit_middleware(max_calls: int) -> ExecuteMiddleware:
    call_counts: dict[str, int] = {}
    async def middleware(ctx: ExecutorContext) -> None:
        key = f"{ctx.app_name}:{ctx.user_id}"
        call_counts[key] = call_counts.get(key, 0) + 1
        if call_counts[key] > max_calls:
            raise PermissionError(
                f"Rate limit exceeded for {ctx.user_id} on {ctx.app_name}"
            )
    return middleware

rl = rate_limit_middleware(max_calls=10)
# In your A2A executor: await rl(executor_context)
```

---

## 10 · `print_event` + content utilities

**Module:** `google.adk.utils._debug_output` · `google.adk.utils.content_utils`

These utilities handle event display and content filtering for debugging and live-mode audio processing.

### `print_event` — smart debug output

```python
_ARGS_MAX_LEN = 50       # Argument preview truncation
_RESPONSE_MAX_LEN = 100  # Response preview length
_CODE_OUTPUT_MAX_LEN = 100

def print_event(event: Event, *, verbose: bool = False) -> None:
    # Key behaviour:
    # - Always shows text parts (agent responses)
    # - Accumulates consecutive text parts to avoid repeating author prefix
    # - Flushes text buffer before handling non-text parts
    # - verbose=True shows function_call, function_response, executable_code,
    #   code_execution_result, inline_data, file_data
    # - verbose=False hides all tool-related parts (quiet mode for end users)
```

### Content utility functions (verified from source)

```python
def is_audio_part(part: types.Part) -> bool:
    # Returns True if:
    # - part.inline_data.mime_type starts with "audio/"
    # - OR part.file_data.mime_type starts with "audio/"
    # Both inline and file references are checked.

def filter_audio_parts(content: types.Content) -> types.Content | None:
    # Filters out all audio parts from Content.
    # Returns None if NO non-audio parts remain (entire Content was audio).
    # Used by GeminiLlmConnection to strip audio before sending history to Gemini.

def extract_text_from_content(content: types.Content | None) -> str:
    # Concatenates text from all non-thought parts.
    # Filters: p.text is truthy AND p.thought is False.
    # Returns '' for None input or empty/thought-only content.
    # Used by BaseNode._validate_input_data / _validate_output_data.
```

### Example 1 — basic debug output

```python
import asyncio
from google.adk.utils._debug_output import print_event
from google.adk.events.event import Event
from google.genai import types

# Text event — always visible
text_event = Event(
    author="assistant",
    content=types.Content(role="model", parts=[
        types.Part(text="I found the answer. The capital is Paris."),
    ]),
)
print_event(text_event)
# Output: assistant > I found the answer. The capital is Paris.

# Tool call event — visible only with verbose=True
tool_event = Event(
    author="assistant",
    content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="search", args={"q": "capital of France"})),
    ]),
)
print_event(tool_event)                 # (no output — quiet mode)
print_event(tool_event, verbose=True)   # assistant > [Calling tool: search({'q': 'capital...']
```

### Example 2 — audio part filtering

```python
from google.adk.utils.content_utils import is_audio_part, filter_audio_parts
from google.genai import types

# Mix of audio and text parts
mixed_content = types.Content(role="user", parts=[
    types.Part(inline_data=types.Blob(mime_type="audio/wav", data=b"...")),
    types.Part(text="What does this recording say?"),
    types.Part(file_data=types.FileData(mime_type="audio/mp3", file_uri="gs://bucket/voice.mp3")),
])

print(is_audio_part(mixed_content.parts[0]))   # True (inline audio/wav)
print(is_audio_part(mixed_content.parts[1]))   # False (text)
print(is_audio_part(mixed_content.parts[2]))   # True (file audio/mp3)

filtered = filter_audio_parts(mixed_content)
print(len(filtered.parts))   # 1 (only the text part remains)
print(filtered.parts[0].text)  # "What does this recording say?"

# All-audio content
audio_only = types.Content(role="user", parts=[
    types.Part(inline_data=types.Blob(mime_type="audio/wav", data=b"...")),
])
print(filter_audio_parts(audio_only))  # None (no non-audio parts)
```

### Example 3 — `extract_text_from_content` with thought filtering

```python
from google.adk.utils.content_utils import extract_text_from_content
from google.genai import types

# Mixed content: text + thought + text
content = types.Content(role="model", parts=[
    types.Part(text="First, let me think..."),
    types.Part(text="Internal reasoning about the problem", thought=True),
    types.Part(text="The answer is 42."),
])

text = extract_text_from_content(content)
print(text)   # "First, let me think...The answer is 42."
# The thought part is excluded — only non-thought text is concatenated.

# None / empty input
print(extract_text_from_content(None))                                         # ''
print(extract_text_from_content(types.Content(role="model", parts=[])))       # ''
print(extract_text_from_content(types.Content(role="model", parts=[
    types.Part(text="only thoughts", thought=True),
])))  # '' (thought-only → empty string)
```

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-06-27 | 2.3.0 | Initial publication of Vol. 29 |

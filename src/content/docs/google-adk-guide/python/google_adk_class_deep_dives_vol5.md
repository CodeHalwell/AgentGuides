---
title: "Class Deep Dives Vol. 5 — v2.11.0"
description: "Source-verified deep dives for 10 classes in google-adk 2.11.0: Event, EventActions, Context (unified CallbackContext/ToolContext), FunctionTool, LongRunningFunctionTool, AuthConfig, DatabaseSessionService, SqliteSessionService, VertexAiSessionService, and VertexAiSearchTool."
framework: google-adk
language: python
sidebar:
  order: 135
---

All examples and field tables on this page are source-verified against **google-adk==2.11.0**. The ten classes span the **event model** (the data flowing through every agent run), the **unified context** object that callbacks and tools share, **tool mechanics** (bare Python functions, long-running async tools, and argument validation), the **auth flow**, all three **session storage backends** available out of the box, and **Vertex AI Search** grounding.

| # | Class / Symbol | Module | Subject |
|---|---|---|---|
| 1 | `Event` + `NodeInfo` | `google.adk.events.event` | Core event data model |
| 2 | `EventActions` + `EventCompaction` | `google.adk.events.event_actions` | Side-effects attached to events |
| 3 | `Context` (`CallbackContext` / `ToolContext`) | `google.adk.agents.context` | Unified context for callbacks and tools |
| 4 | `FunctionTool` | `google.adk.tools.function_tool` | Wrapping a Python callable as a tool |
| 5 | `LongRunningFunctionTool` | `google.adk.tools.long_running_tool` | Async / polling tools |
| 6 | `AuthConfig` + `AuthToolArguments` | `google.adk.auth.auth_tool` | Per-tool auth credential request flow |
| 7 | `DatabaseSessionService` | `google.adk.sessions.database_session_service` | SQLAlchemy-backed session storage |
| 8 | `SqliteSessionService` | `google.adk.sessions.sqlite_session_service` | aiosqlite-native session storage |
| 9 | `VertexAiSessionService` | `google.adk.sessions.vertex_ai_session_service` | Vertex AI Agent Engine sessions |
| 10 | `VertexAiSearchTool` | `google.adk.tools.vertex_ai_search_tool` | Enterprise search grounding |

---

## 1 — `Event` + `NodeInfo`

**Module:** `google.adk.events.event`

`Event` is the central data unit in ADK. Every LLM reply, function call, function response, and user message is an `Event`. The runner collects events from agents and appends them to the session history. Application code iterates over events returned by `runner.run_async()` to find the final response.

`Event` extends `LlmResponse` (from `google.adk.models.llm_response`) and adds ADK-specific fields.

### Field reference

Source-verified from `google/adk/events/event.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `invocation_id` | `str` | `""` | Groups all events belonging to one `runner.run_async()` call |
| `author` | `str` | `""` | `"user"` or the agent name that produced the event |
| `actions` | `EventActions` | `EventActions()` | State deltas, transfers, auth requests, etc. |
| `output` | `Any \| None` | `None` | Generic data output from a workflow node |
| `node_info` | `NodeInfo` | `NodeInfo()` | Path, run_id, and output routing inside a Workflow |
| `long_running_tool_ids` | `set[str] \| None` | `None` | IDs of function calls that are long-running |
| `branch` | `str \| None` | `None` | Dot-separated path used by multi-agent branching |
| `id` | `str` | auto-generated | Unique event ID (assigned post-init) |
| `timestamp` | `float` | `time.time()` | UNIX timestamp |

### Convenience kwargs (model validator)

`Event.__init__` (via `_accept_convenience_kwargs`) routes three top-level kwargs to nested fields so you can skip the `actions=EventActions(...)` boilerplate:

| Kwarg | Destination |
|---|---|
| `message=` | `content` (converted via `t_content`) |
| `state=` | `actions.state_delta` |
| `route=` | `actions.route` |
| `node_path=` | `node_info.path` |

### `NodeInfo` fields

| Field | Type | Purpose |
|---|---|---|
| `path` | `str` | Full workflow node path, e.g. `"my_wf/sub_agent@1"` |
| `output_for` | `list[str] \| None` | Other node paths whose output this event also serves |
| `message_as_output` | `bool \| None` | When `True`, the event content is the node's output (no separate output event needed) |

Computed properties: `node_info.run_id`, `node_info.parent_run_id`, `node_info.name`.

### Key methods

```python
event.is_final_response() -> bool
```
Returns `True` when the event is a complete, user-facing reply: no function calls, no function responses, not partial, no trailing code-execution result. Two exceptions also return `True` even when function calls are present: when `actions.skip_summarization` is set, or when `long_running_tool_ids` is non-empty — in both cases the client receives the event immediately so it can show a progress indicator.  
Application code typically filters with this to find the text to show users.

```python
event.has_trailing_code_execution_result() -> bool
```
Returns `True` if the last `Part` in `content.parts` is a `code_execution_result`.

### Minimal: iterating the final reply

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

async def main():
    agent = LlmAgent(name="bot", model="gemini-2.5-flash",
                     instruction="Answer in one sentence.")
    runner = InMemoryRunner(agent=agent, app_name="app")
    await runner.session_service.create_session(
        app_name="app", user_id="u1", session_id="s1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text="What is 2+2?")]),
    ):
        if event.is_final_response() and event.content:
            print("Author:", event.author)
            print("Reply:", event.content.parts[0].text)

asyncio.run(main())
```

### Constructing an event with convenience kwargs

```python
from google.adk.events import Event

# Shorthand for setting state and message at once
ev = Event(
    author="my_agent",
    message="Here is the result.",   # → event.content
    state={"result_ready": True},    # → event.actions.state_delta
)
print(ev.actions.state_delta)   # {"result_ready": True}
print(ev.content.parts[0].text) # "Here is the result."
```

### Inspecting events from a run

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

async def inspect_events():
    agent = LlmAgent(
        name="calc",
        model="gemini-2.5-flash",
        instruction="When asked maths, call the multiply tool.",
        tools=[lambda a, b: a * b],  # auto-wrapped as FunctionTool
    )
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for ev in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text="3 times 7?")]),
    ):
        fc_calls = ev.get_function_calls()
        fc_resps = ev.get_function_responses()
        if fc_calls:
            print(f"[{ev.author}] tool call: {fc_calls[0].name}({fc_calls[0].args})")
        elif fc_resps:
            print(f"[{ev.author}] tool response: {fc_resps[0].response}")
        elif ev.is_final_response():
            print(f"[{ev.author}] final: {ev.content.parts[0].text}")

asyncio.run(inspect_events())
```

---

## 2 — `EventActions` + `EventCompaction`

**Module:** `google.adk.events.event_actions`

`EventActions` carries all side-effects that an agent wants to apply to the session when the event is processed. It is always embedded inside an `Event` via `event.actions`.

### Field reference

Source-verified from `google/adk/events/event_actions.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `skip_summarization` | `bool \| None` | `None` | When `True`, the framework skips calling the model to summarise a function response — used by long-running tools |
| `state_delta` | `dict[str, Any]` | `{}` | Key-value pairs written to session state when the event is persisted |
| `artifact_delta` | `dict[str, int]` | `{}` | Maps artifact filename → new version number |
| `transfer_to_agent` | `str \| None` | `None` | Agent name to hand control to |
| `transfer_reason` | `str \| None` | `None` | Human-readable reason for the transfer |
| `escalate` | `bool \| None` | `None` | Signals the agent is done and control should return to the parent |
| `requested_auth_configs` | `dict[str, AuthConfig]` | `{}` | Auth configs requested by tool responses (keyed by function call ID) |
| `requested_tool_confirmations` | `dict[str, ToolConfirmation]` | `{}` | Tool confirmations keyed by function call ID |
| `compaction` | `EventCompaction \| None` | `None` | Summarised-history compaction record |
| `end_of_agent` | `bool \| None` | `None` | Framework sets this when an agent finishes its turn |
| `agent_state` | `dict \| None` | `None` | Checkpoint/resume state for workflow nodes |
| `route` | `RouteValue \| list[RouteValue] \| None` | `None` | Workflow graph edge(s) to take next |
| `render_ui_widgets` | `list[UiWidget] \| None` | `None` | UI widgets for the client to render |
| `set_model_response` | `Any \| None` | `None` | Override structured output for the model response |

### `EventCompaction`

When a `ContextFilterPlugin` or `LlmEventSummarizer` compacts old events, it writes an `EventCompaction` into the last retained event's `actions.compaction`:

| Field | Type | Purpose |
|---|---|---|
| `start_timestamp` | `float` | Start of the compacted window |
| `end_timestamp` | `float` | End of the compacted window |
| `compacted_content` | `Content` | The summarised content replacing the range |

### Writing state from a tool

The most common use of `EventActions` in application code is writing state from within a tool via `tool_context.state`:

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def record_score(score: int, tool_context: ToolContext) -> str:
    """Records the user's quiz score."""
    tool_context.state["quiz_score"] = score          # → state_delta on the event
    tool_context.state["quiz_complete"] = True
    return f"Score {score} recorded."

tool = FunctionTool(record_score)
```

### Escalating from a loop

```python
from google.adk.tools.tool_context import ToolContext

def check_done(tool_context: ToolContext) -> str:
    """Escalates when the task is complete."""
    if tool_context.state.get("task_done"):
        tool_context.actions.escalate = True          # stops the LoopAgent
        return "All done."
    return "Still working."
```

### Requesting agent transfer from a tool

```python
from google.adk.tools.tool_context import ToolContext

def route_to_billing(tool_context: ToolContext) -> str:
    """Hands off to the billing agent."""
    tool_context.actions.transfer_to_agent = "billing_agent"
    tool_context.actions.transfer_reason = "User asked about their invoice."
    return "Transferring you to billing."
```

---

## 3 — `Context` (unified `CallbackContext` / `ToolContext`)

**Module:** `google.adk.agents.context`

As of ADK 2.10.0, `CallbackContext` and `ToolContext` are both type aliases for `Context`. All three names are interchangeable at runtime. The unification means callbacks and tools share the same surface without wrapper boilerplate.

```python
# These three imports all give you the same class:
from google.adk.agents.context import Context
from google.adk.agents.callback_context import CallbackContext  # alias
from google.adk.tools.tool_context import ToolContext           # alias
```

### Constructor (framework-internal)

```python
Context(
    invocation_context: InvocationContext,
    *,
    event_actions: EventActions | None = None,
    function_call_id: str | None = None,
    branch: str | None = None,
)
```

You never construct `Context` yourself — the framework passes it to your callbacks and tools.

### State access

```python
# Read
value = ctx.state["key"]
value = ctx.state.get("key", default)

# Write (creates a state_delta that persists to the session)
ctx.state["key"] = "new_value"
```

State keys are scoped by prefix (no prefix = session scope, `app:` = app-wide, `user:` = per-user, `temp:` = in-memory only).

### Artifact methods

```python
# Save a file-like artifact
version = await ctx.save_artifact(
    filename="report.pdf",
    artifact=types.Part.from_bytes(pdf_bytes, mime_type="application/pdf"),
)

# Load it back (most recent version by default)
part = await ctx.load_artifact("report.pdf")

# Load a specific version
part = await ctx.load_artifact("report.pdf", version=2)

# List all artifact filenames in this session
names: list[str] = await ctx.list_artifacts()
```

### Credential / auth methods

```python
from google.adk.auth.auth_tool import AuthConfig

# Signal that this tool needs the user to authenticate
ctx.request_credential(auth_config)    # sets actions.requested_auth_configs

# Check if the user already returned a credential
cred = ctx.get_auth_response(auth_config)  # returns AuthCredential | None

# Persist a credential for reuse across invocations
await ctx.save_credential(auth_config)

# Reload a persisted credential
cred = await ctx.load_credential(auth_config)
```

### Memory methods

```python
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

# Add the current session transcript to long-term memory
await ctx.add_session_to_memory()

# Store a custom memory entry — add_memory() takes memories: Sequence[MemoryEntry]
await ctx.add_memory(
    memories=[
        MemoryEntry(
            content=types.Content(
                parts=[types.Part(text="User prefers metric units.")]
            )
        )
    ]
)

# Semantic search over memory
result = await ctx.search_memory("preferred measurement system")
for entry in result.memories:
    print(entry.content)
```

### Confirmation (tool user-approval flow)

```python
def risky_delete(resource_id: str, tool_context: ToolContext) -> str:
    """Deletes a resource — requires user confirmation."""
    if tool_context.tool_confirmation is None:
        # First call: pause execution and ask the user to approve or reject
        tool_context.request_confirmation(
            hint=f"Confirm deletion of resource '{resource_id}'?"
        )
        tool_context.actions.skip_summarization = True
        return {"error": "Awaiting confirmation."}

    if not tool_context.tool_confirmation.confirmed:
        return {"error": "Deletion rejected by user."}

    # Second call: user confirmed — perform the delete
    return f"Resource '{resource_id}' deleted."
```

### Running a nested agent node

```python
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

async def delegate_task(task: str, tool_context: ToolContext) -> str:
    """Runs a specialist sub-agent for a specific task."""
    specialist = LlmAgent(
        name="specialist",
        model="gemini-2.5-flash",
        instruction="You are an expert coder.",
    )
    output = await tool_context.run_node(specialist, node_input=task)
    return str(output)
```

### Full callback example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

def before_model(ctx: Context, req: LlmRequest) -> LlmResponse | None:
    """Injects a system prefix if the user has a VIP flag."""
    if ctx.state.get("vip_user"):
        req.append_instructions(["VIP: Always respond with extra care and attention."])
    return None

async def after_model(ctx: Context, resp: LlmResponse) -> LlmResponse | None:
    """Logs the invocation count."""
    count = ctx.state.get("call_count", 0) + 1
    ctx.state["call_count"] = count
    return None  # return None to keep the original response

agent = LlmAgent(
    name="agent",
    model="gemini-2.5-flash",
    instruction="You are helpful.",
    before_model_callback=before_model,
    after_model_callback=after_model,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1",
        state={"vip_user": True},
    )
    async for ev in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text="Hello!")]),
    ):
        if ev.is_final_response():
            print(ev.content.parts[0].text)

asyncio.run(main())
```

---

## 4 — `FunctionTool`

**Module:** `google.adk.tools.function_tool`

`FunctionTool` wraps any Python callable (sync or async, function or method) as a tool the LLM can invoke. It automatically extracts the tool name from `func.__name__`, the description from the docstring, and the parameter schema from type annotations.

### Constructor

```python
FunctionTool(
    func: Callable[..., Any],
    *,
    require_confirmation: bool | Callable[..., bool] = False,
)
```

`require_confirmation` can be a static bool or a callable that receives the same kwargs as `func` (except `tool_context`) and returns a bool. When `True`, the tool pauses and asks the user to approve via a `ToolConfirmation` response before executing.

### Automatic name and schema extraction

```python
from google.adk.tools import FunctionTool

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Returns the current weather for a city.

    Args:
        city: The city name.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    ...  # call a real weather API
    return {"city": city, "temp": 22, "unit": unit}

tool = FunctionTool(get_weather)
print(tool.name)         # "get_weather"
print(tool.description)  # the docstring
```

### Injecting ToolContext

Any parameter annotated as `ToolContext` (or named `tool_context`) is automatically populated by the framework and excluded from the schema sent to the LLM:

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def remember_preference(preference: str, tool_context: ToolContext) -> str:
    """Saves a user preference to session state."""
    tool_context.state["preference"] = preference
    return f"Preference '{preference}' saved."

tool = FunctionTool(remember_preference)
# 'tool_context' does NOT appear in the generated FunctionDeclaration
```

### Pydantic model arguments

The LLM sends JSON; `FunctionTool` automatically converts nested JSON dicts to Pydantic model instances:

```python
from pydantic import BaseModel
from google.adk.tools import FunctionTool

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

def send_package(destination: Address, weight_kg: float) -> str:
    """Dispatches a package to the given address."""
    return f"Package ({weight_kg}kg) → {destination.city}, {destination.zip_code}"

tool = FunctionTool(send_package)
# The LLM passes {"destination": {"street": "...", "city": "...", "zip_code": "..."},
#                  "weight_kg": 1.5}
# FunctionTool converts the nested dict to Address automatically.
```

### Argument validation (2.10.0+)

When the `FUNCTION_TOOL_ARG_VALIDATION` feature flag is enabled, `FunctionTool` validates and coerces argument types before invocation. Type mismatches return a self-correcting error message to the LLM:

```python
from google.adk.features import FeatureName
from google.adk.features import override_feature_enabled

override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True)  # opt-in

from google.adk.tools import FunctionTool

def power(base: int, exponent: int) -> int:
    """Raises base to the power of exponent."""
    return base ** exponent

tool = FunctionTool(power)
# If the LLM sends {"base": "3", "exponent": 2}, the string "3" is coerced to int 3.
# If coercion fails, the LLM gets: "Parameter 'base': expected type 'int', validation error: ..."
```

### Conditional confirmation

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def transfer_funds(
    amount: float, to_account: str, tool_context: ToolContext
) -> str:
    """Transfers funds to another account."""
    return f"Transferred ${amount:.2f} to {to_account}"

def needs_approval(amount: float, to_account: str, **_) -> bool:
    return amount > 1000.0  # require confirmation for large transfers

tool = FunctionTool(transfer_funds, require_confirmation=needs_approval)
```

### Registering with an agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

agent = LlmAgent(
    name="math_agent",
    model="gemini-2.5-flash",
    instruction="Use the multiply tool when asked to multiply.",
    tools=[FunctionTool(multiply)],   # explicit wrap
)

# Shorthand: ADK auto-wraps bare callables too
agent2 = LlmAgent(
    name="math_agent2",
    model="gemini-2.5-flash",
    instruction="Use the multiply tool when asked to multiply.",
    tools=[multiply],   # same result — auto-wrapped as FunctionTool
)
```

---

## 5 — `LongRunningFunctionTool`

**Module:** `google.adk.tools.long_running_tool`

`LongRunningFunctionTool` extends `FunctionTool` for operations that take a significant amount of time (file processing, external API calls, multi-step workflows). The key difference: when the function returns an intermediate status, the framework does not call the model to summarise the response (`skip_summarization = True` is set automatically via `event.long_running_tool_ids`).

### Constructor

```python
LongRunningFunctionTool(func: Callable[..., Any])
```

Identical to `FunctionTool` but sets `self.is_long_running = True` and appends a "do not call again while pending" note to the tool's `FunctionDeclaration.description`.

### How long-running tools work

```
1. LLM issues a function_call for the tool.
2. Framework sets long_running_tool_ids on the function-call event and yields
   it before starting the tool.  event.is_final_response() → True here,
   so the client receives the event and can display a progress indicator.
3. Your async function runs to completion (no intermediate yields).
4a. If the function returns a truthy result (non-empty dict, non-zero, etc.)
    the framework builds a function_response event with that result immediately
    and resumes the LLM.
4b. If the function returns a falsy value (None, {}, False, 0, …) the framework
    emits no function_response.  The response must arrive later via session
    injection (an external process calls the session service to append the
    FunctionResponse directly).
```

`LongRunningFunctionTool` is designed for pattern 4b: kick off a background job and return a falsy value quickly, then have the background process inject the final result. Pattern 4a (blocking inline) works too but ties up the runner for the full duration of the job.

### File-processing example

```python
import asyncio
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

async def process_large_file(
    file_url: str, tool_context: ToolContext
) -> dict:
    """Downloads and processes a large file. May take several minutes.

    Args:
        file_url: The URL of the file to process.
    """
    # Simulate a slow download + processing step
    await asyncio.sleep(2)   # replace with real I/O
    tool_context.state["processed_file"] = file_url
    return {
        "status": "complete",
        "rows_processed": 50_000,
        "output_artifact": "processed_data.csv",
    }

tool = LongRunningFunctionTool(process_large_file)
```

### Returning a deferred result (the intended pattern)

The function returns `None` immediately after kicking off the work. An external process (a Cloud Task, Pub/Sub consumer, etc.) later injects the final `FunctionResponse` into the session. ADK never re-calls the tool:

```python
import asyncio
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

async def submit_export(job_id: str, tool_context: ToolContext) -> None:
    """Submits an export job and returns immediately; result comes later.

    Args:
        job_id: A unique identifier for the export job.
    """
    tool_context.state["pending_export"] = job_id
    # Kick off the job asynchronously — do NOT await it here.
    # The final FunctionResponse is injected into the session by an
    # external worker once the job completes.
    asyncio.create_task(_fire_and_forget_export(job_id))
    return None  # None → no FunctionResponse built; awaits session injection.

async def _fire_and_forget_export(job_id: str) -> None:
    await asyncio.sleep(0)  # hand off to the event loop; real work is elsewhere

export_tool = LongRunningFunctionTool(submit_export)
```

### Inline blocking pattern (simpler, ties up the runner)

The function blocks until the work is done and returns a result dict. The framework builds the `FunctionResponse` immediately when the function returns:

```python
async def run_export_blocking(job_id: str, tool_context: ToolContext) -> dict:
    """Runs an export job inline and returns when complete.

    Args:
        job_id: A unique identifier for the export job.
    """
    # The function-call event is already yielded to the client before this
    # runs, so the client can show a spinner while this blocks.
    await asyncio.sleep(5)   # replace with real I/O
    tool_context.state["last_export"] = job_id
    return {
        "status": "complete",
        "job_id": job_id,
        "download_url": f"https://example.com/exports/{job_id}.csv",
    }

export_tool_blocking = LongRunningFunctionTool(run_export_blocking)
```

### Wiring into an agent

```python
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="exporter",
    model="gemini-2.5-flash",
    instruction=(
        "Help users export data. When a job is pending, tell the user "
        "to wait and do not call the tool again."
    ),
    tools=[export_tool],
)
```

---

## 6 — `AuthConfig` + `AuthToolArguments`

**Module:** `google.adk.auth.auth_tool`

`AuthConfig` is the data structure a tool passes to `tool_context.request_credential()` to tell ADK (and the client) what kind of credential is needed. The framework stores it in `event.actions.requested_auth_configs` so the client can guide the user through the auth flow.

### `AuthConfig` fields

Source-verified from `google/adk/auth/auth_tool.py`:

| Field | Type | Purpose |
|---|---|---|
| `auth_scheme` | `AuthScheme` | Describes the auth protocol (OAuth2, API key, OpenID Connect, HTTP bearer, …) |
| `raw_auth_credential` | `AuthCredential \| None` | Client credentials needed to start the flow (e.g. OAuth2 `client_id` / `client_secret`) |
| `exchanged_auth_credential` | `AuthCredential \| None` | Filled by ADK (or the client) with the resulting token after the flow completes |
| `credential_key` | `str \| None` | Stable key used to save / load this credential in the credential service; auto-derived from `auth_scheme` + `raw_auth_credential` if not supplied |

### `AuthToolArguments`

`AuthToolArguments` is the payload the client sends back to ADK after collecting credentials from the user:

| Field | Type | Purpose |
|---|---|---|
| `function_call_id` | `str` | Matches the `function_call_id` in `requested_auth_configs` |
| `auth_config` | `AuthConfig` | The completed config with `exchanged_auth_credential` filled in |

### OAuth2 tool example

```python
from fastapi.openapi.models import OAuthFlows, OAuthFlowAuthorizationCode
from google.adk.auth.auth_schemes import ExtendedOAuth2
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

OAUTH_CONFIG = AuthConfig(
    auth_scheme=ExtendedOAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://accounts.google.com/o/oauth2/auth",
                tokenUrl="https://oauth2.googleapis.com/token",
                scopes={
                    "https://www.googleapis.com/auth/calendar.readonly":
                        "Read calendar events",
                },
            )
        )
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
        ),
    ),
)

def list_calendar_events(tool_context: ToolContext) -> list[dict]:
    """Lists the user's upcoming calendar events.

    Requires Google Calendar read access.
    """
    cred = tool_context.get_auth_response(OAUTH_CONFIG)
    if cred is None:
        # Credential not yet available — request it
        tool_context.request_credential(OAUTH_CONFIG)
        return []

    token = cred.oauth2.access_token
    # Use token to call the Calendar API
    return [{"summary": "Team standup", "start": "2026-10-06T09:00:00"}]

calendar_tool = FunctionTool(list_calendar_events)
```

### API key example

```python
from fastapi.openapi.models import APIKey, APIKeyIn
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import FunctionTool

# APIKey is a proper OpenAPI 3.0 security scheme (from fastapi.openapi.models,
# a google-adk dependency). Specify where the key is sent (header/query/cookie)
# and what the header/param name is.
NEWS_AUTH = AuthConfig(
    auth_scheme=APIKey(name="X-API-Key", in_=APIKeyIn.header),
    raw_auth_credential=AuthCredential(auth_type=AuthCredentialTypes.API_KEY),
)

def get_news_headlines(topic: str, tool_context: ToolContext) -> list[dict]:
    """Fetches news headlines for a topic.

    Args:
        topic: The news topic to search for.
    """
    cred = tool_context.get_auth_response(NEWS_AUTH)
    if not cred:
        tool_context.request_credential(NEWS_AUTH)
        return []

    api_key = cred.api_key  # AuthCredential.api_key is str | None
    # Use api_key to call the news API
    return [{"title": f"Latest on {topic}", "url": "https://example.com"}]

news_tool = FunctionTool(get_news_headlines)
```

### Persisting credentials across sessions

Credential saving must happen inside the tool itself (via `tool_context`), not in an after-tool callback. Call `await tool_context.save_credential(auth_config)` immediately after `get_auth_response` returns a valid credential. On the next session, call `await tool_context.load_credential(auth_config)` to restore it.

The `AfterToolCallback` type signature (for reference when registering a callback on an agent):

```python
from typing import Any, Optional
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

async def after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Intercepts the tool response. Return None to keep it unchanged."""
    # Example: log every tool call with its result
    print(f"[{tool.name}] args={args!r} → {tool_response!r}")
    return None  # return a dict to override the response the LLM sees
```

---

## 7 — `DatabaseSessionService`

**Module:** `google.adk.sessions.database_session_service`

`DatabaseSessionService` stores sessions in any SQLAlchemy-supported async database (PostgreSQL, MySQL, SQLite in async mode, etc.). It requires the `sqlalchemy[asyncio]` extra:

```bash
pip install "google-adk[db]"
# or: pip install sqlalchemy[asyncio] aiosqlite  # for SQLite
```

### Constructor overloads

```python
# Overload 1 — pass a database URL string
DatabaseSessionService(db_url: str, **kwargs)

# Overload 2 — pass an existing SQLAlchemy AsyncEngine
DatabaseSessionService(db_engine: AsyncEngine)
```

`db_url` and `db_engine` are mutually exclusive. Providing neither or both raises `ValueError`.

### Auto-configuration for SQLite

When the URL dialect is `sqlite` and `database=":memory:"`, `DatabaseSessionService` automatically sets:
- `poolclass=StaticPool` (single connection shared across all sessions)
- `connect_args={"check_same_thread": False}`
- `pool_reset_on_return=None` (prevents rollback on connection return with `StaticPool`)

### Getting started — PostgreSQL

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService

PG_URL = "postgresql+asyncpg://user:password@localhost:5432/agent_sessions"

session_service = DatabaseSessionService(PG_URL)

agent = LlmAgent(
    name="agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

async def main():
    await session_service.prepare_tables()  # idempotent schema creation
    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="my_app", user_id="alice", session_id="session-1"
    )

    from google.genai import types as genai_types
    async for ev in runner.run_async(
        user_id="alice",
        session_id=session.id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="Hello!")]),
    ):
        if ev.is_final_response():
            print(ev.content.parts[0].text)

asyncio.run(main())
```

### Getting started — async SQLite

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService

session_service = DatabaseSessionService("sqlite+aiosqlite:///./agent_sessions.db")

agent = LlmAgent(
    name="agent",
    model="gemini-2.5-flash",
    instruction="Remember what I tell you.",
)

async def main():
    await session_service.prepare_tables()
    runner = Runner(agent=agent, app_name="app", session_service=session_service)
    session = await session_service.create_session(
        app_name="app", user_id="bob",
        state={"language": "Spanish"}
    )
    from google.genai import types as genai_types
    async for ev in runner.run_async(
        user_id="bob", session_id=session.id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="What language do I prefer?")]),
    ):
        if ev.is_final_response():
            print(ev.content.parts[0].text)  # "You prefer Spanish."

asyncio.run(main())
```

### Passing an existing engine (advanced)

```python
from sqlalchemy.ext.asyncio import create_async_engine
from google.adk.sessions.database_session_service import DatabaseSessionService

engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost:5432/mydb",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Reuse the engine across services (connection pool is shared)
session_service = DatabaseSessionService(db_engine=engine)
```

### CRUD operations

```python
import asyncio
from google.adk.sessions.database_session_service import DatabaseSessionService

svc = DatabaseSessionService("sqlite+aiosqlite:///./sessions.db")

async def demo():
    await svc.prepare_tables()

    # Create
    session = await svc.create_session(
        app_name="app", user_id="alice", session_id="s1",
        state={"credits": 100}
    )
    print("Created:", session.id, session.state)

    # Get
    fetched = await svc.get_session(app_name="app", user_id="alice", session_id="s1")
    print("Fetched state:", fetched.state)

    # List all sessions for a user
    resp = await svc.list_sessions(app_name="app", user_id="alice")
    print("Sessions:", [s.id for s in resp.sessions])

    # Delete
    await svc.delete_session(app_name="app", user_id="alice", session_id="s1")

asyncio.run(demo())
```

---

## 8 — `SqliteSessionService`

**Module:** `google.adk.sessions.sqlite_session_service`

`SqliteSessionService` is a purpose-built SQLite session backend that uses `aiosqlite` directly without needing SQLAlchemy. It stores events as JSON for forward-schema-compatibility and supports state scoped to session, user, or app level.

Install the extra:

```bash
pip install "google-adk[aiosqlite]"
# or: pip install aiosqlite
```

### Constructor

```python
SqliteSessionService(db_path: str)
```

`db_path` is a local filesystem path. Use `":memory:"` for an in-memory database (lost on process exit). A migration guard raises `RuntimeError` if an old-schema database is detected; run the migration CLI first:

```bash
python -m google.adk.sessions.migration.migrate_from_sqlalchemy_sqlite \
    --source_db_path ./old.db \
    --dest_db_path ./new.db
```

### Getting started

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService

session_service = SqliteSessionService("./agent_sessions.db")

agent = LlmAgent(
    name="agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

async def main():
    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="my_app", user_id="alice"
    )

    from google.genai import types as genai_types
    async for ev in runner.run_async(
        user_id="alice",
        session_id=session.id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="What is the capital of France?")]),
    ):
        if ev.is_final_response():
            print(ev.content.parts[0].text)

asyncio.run(main())
```

### State scoping

ADK uses key prefixes to scope state writes:

```python
import asyncio
from google.adk.sessions.sqlite_session_service import SqliteSessionService

svc = SqliteSessionService("./state_demo.db")

async def main():
    session = await svc.create_session(
        app_name="shop",
        user_id="alice",
        state={
            "cart_items": 3,              # session-scoped (no prefix)
            "user:preferred_lang": "en",  # user-scoped (persists across sessions)
            "app:total_users": 42,        # app-scoped (shared by all users)
            "temp:scratch": "ignored",    # not persisted
        }
    )
    fetched = await svc.get_session(
        app_name="shop", user_id="alice", session_id=session.id
    )
    print(fetched.state)
    # {"cart_items": 3, "user:preferred_lang": "en", "app:total_users": 42}
    # Note: "temp:scratch" is absent — in-memory only

asyncio.run(main())
```

### Multi-turn conversation

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService

async def chat():
    svc = SqliteSessionService("./chat.db")
    agent = LlmAgent(
        name="assistant",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant. Remember prior messages.",
    )
    runner = Runner(agent=agent, app_name="chat", session_service=svc)
    session = await svc.create_session(app_name="chat", user_id="u1")

    from google.genai import types as genai_types
    for msg in ["My name is Alice.", "What is my name?"]:
        async for ev in runner.run_async(
            user_id="u1", session_id=session.id,
            new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=msg)]),
        ):
            if ev.is_final_response():
                print(f"Bot: {ev.content.parts[0].text}")

asyncio.run(chat())
# Bot: Nice to meet you, Alice!
# Bot: Your name is Alice.
```

---

## 9 — `VertexAiSessionService`

**Module:** `google.adk.sessions.vertex_ai_session_service`

`VertexAiSessionService` persists sessions in the **Vertex AI Agent Engine Session Service** — a fully managed, scalable backend. Sessions, state, and event history are stored in Google Cloud and accessible from any region.

Install the extra:

```bash
pip install "google-adk[gcp]"
# or: pip install google-cloud-aiplatform
```

### Constructor

```python
VertexAiSessionService(
    project: str | None = None,
    location: str | None = None,
    agent_engine_id: str | None = None,
    *,
    express_mode_api_key: str | None = None,
)
```

| Arg | Purpose |
|---|---|
| `project` | GCP project ID (defaults to ADC project) |
| `location` | GCP region, e.g. `"us-central1"` |
| `agent_engine_id` | Resource ID of the Vertex AI Agent Engine (Reasoning Engine) |
| `express_mode_api_key` | API key for Vertex AI Express Mode (only needed when `GOOGLE_GENAI_USE_ENTERPRISE=true`) |

### Prerequisites

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Getting started

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

session_service = VertexAiSessionService(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    agent_engine_id="1234567890123456789",  # your Reasoning Engine resource ID
)

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

async def main():
    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="my_app",
        user_id="alice",
        state={"plan": "premium"},
    )
    from google.genai import types as genai_types
    async for ev in runner.run_async(
        user_id="alice",
        session_id=session.id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="What plan am I on?")]),
    ):
        if ev.is_final_response():
            print(ev.content.parts[0].text)

asyncio.run(main())
```

### Session ID validation

Vertex AI session IDs must match `^[A-Za-z0-9_-]+$`. The service raises `ValueError` for any ID that doesn't match, preventing URL path injection:

```python
# OK
await svc.create_session(app_name="app", user_id="u1", session_id="session-abc123")

# Raises ValueError: Invalid session_id 'bad/id': must match ^[A-Za-z0-9_-]+$
await svc.create_session(app_name="app", user_id="u1", session_id="bad/id")
```

### Full resource name as session_id

`VertexAiSessionService` also accepts full Vertex AI resource paths as `session_id`:

```python
# Full resource path — engine ID mismatch is caught automatically
full_path = "projects/my-proj/locations/us-central1/reasoningEngines/123/sessions/abc"
session = await svc.get_session(
    app_name="my_app", user_id="alice", session_id=full_path
)
```

### Session listing with pagination

```python
import asyncio
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

async def list_all():
    svc = VertexAiSessionService(
        project="my-project",
        location="us-central1",
        agent_engine_id="123",
    )
    page = await svc.list_sessions(app_name="my_app", user_id="alice")
    for session in page.sessions:
        print(session.id, session.last_update_time)

asyncio.run(list_all())
```

### Choosing the right session backend

| Backend | Use when |
|---|---|
| `InMemorySessionService` | Tests, local development, single-process demos |
| `SqliteSessionService` | Single-server deployments, edge devices |
| `DatabaseSessionService` | Multi-server deployments with PostgreSQL / MySQL |
| `VertexAiSessionService` | Production GCP deployments, fully managed, globally scalable |

---

## 10 — `VertexAiSearchTool`

**Module:** `google.adk.tools.vertex_ai_search_tool`

`VertexAiSearchTool` is a **model built-in** tool: it injects a Vertex AI Search retrieval spec into the LLM request rather than executing as a function call. Gemini handles the search call server-side and grounds its response in the retrieved documents.

### Constructor

```python
VertexAiSearchTool(
    *,
    data_store_id: str | None = None,
    data_store_specs: list[types.VertexAISearchDataStoreSpec] | None = None,
    search_engine_id: str | None = None,
    filter: str | None = None,
    max_results: int | None = None,
    bypass_multi_tools_limit: bool = False,
)
```

| Arg | Purpose |
|---|---|
| `data_store_id` | Full resource path of a single data store (mutually exclusive with `search_engine_id`) |
| `data_store_specs` | Per-data-store specs when using an engine with multiple stores |
| `search_engine_id` | Full resource path of a search engine (mutually exclusive with `data_store_id`) |
| `filter` | Discovery Engine filter expression (e.g. `'lang: ANY("en")'`) — uses `field: ANY("value")` syntax, not CEL equality |
| `max_results` | Cap on returned documents |
| `bypass_multi_tools_limit` | When `True` with multiple tools in the same agent, ADK automatically replaces `VertexAiSearchTool` with `DiscoveryEngineSearchTool` (requires `pip install google-adk[gcp]`). Set only when you need to combine grounding with function-call tools. |

### Data store example

```python
from google.adk.agents import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

DATA_STORE = (
    "projects/my-project/locations/global"
    "/collections/default_collection"
    "/dataStores/my-data-store"
)

search_tool = VertexAiSearchTool(data_store_id=DATA_STORE)

agent = LlmAgent(
    name="search_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a product support agent. "
        "Always ground your answers in the knowledge base."
    ),
    tools=[search_tool],
)
```

### Search engine with multiple data stores

```python
from google.adk.agents import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.genai import types

ENGINE_ID = (
    "projects/my-project/locations/global"
    "/collections/default_collection"
    "/engines/my-engine"
)
DATA_STORE_A = (
    "projects/my-project/locations/global"
    "/collections/default_collection"
    "/dataStores/docs-store"
)
DATA_STORE_B = (
    "projects/my-project/locations/global"
    "/collections/default_collection"
    "/dataStores/faq-store"
)

search_tool = VertexAiSearchTool(
    search_engine_id=ENGINE_ID,
    data_store_specs=[
        types.VertexAISearchDataStoreSpec(data_store=DATA_STORE_A),
        types.VertexAISearchDataStoreSpec(data_store=DATA_STORE_B),
    ],
    max_results=5,
)
```

### Dynamic filtering by session state

Subclass `VertexAiSearchTool` and override `_build_vertex_ai_search_config` to apply per-request filters from session state. **Always validate session-state values before interpolating them into CEL filter strings** — untrusted input can break or inject filter expressions.

```python
import re
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.genai import types

DATA_STORE = "projects/my-proj/locations/global/collections/default_collection/dataStores/kb"

_SAFE_ID = re.compile(r'^[A-Za-z0-9_-]+$')

class UserScopedSearchTool(VertexAiSearchTool):
    """Restricts search to documents belonging to the user's organisation."""

    def _build_vertex_ai_search_config(
        self, ctx: ReadonlyContext
    ) -> types.VertexAISearch:
        org_id = ctx.state.get("org_id", "")
        # Validate before interpolating into a CEL filter to prevent injection
        if org_id and not _SAFE_ID.match(org_id):
            raise ValueError(f"Invalid org_id format: {org_id!r}")
        # Discovery Engine filter syntax: field: ANY("value"), not CEL equality.
        return types.VertexAISearch(
            datastore=self.data_store_id,
            filter=f'org_id: ANY("{org_id}")' if org_id else None,
            max_results=self.max_results,
        )

search_tool = UserScopedSearchTool(data_store_id=DATA_STORE, max_results=10)
```

### Combining with function tools

By default, Gemini does not allow built-in retrieval tools (like `VertexAiSearchTool`) alongside regular function-call tools. Pass `bypass_multi_tools_limit=True` to lift this restriction: ADK automatically substitutes `VertexAiSearchTool` with `DiscoveryEngineSearchTool` — a function-call–based implementation that uses the `google-cloud-discoveryengine` package (included in `google-adk[gcp]`).

```python
from google.adk.agents import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

search_tool = VertexAiSearchTool(
    data_store_id="projects/…/dataStores/my-store",
    bypass_multi_tools_limit=True,
)

def get_user_profile(user_id: str) -> dict:
    """Returns the user profile from the internal database."""
    return {"user_id": user_id, "name": "Alice", "tier": "premium"}

agent = LlmAgent(
    name="hybrid_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You can look up user profiles AND search the knowledge base. "
        "Combine both when answering questions."
    ),
    tools=[search_tool, get_user_profile],
)
```

---

## Summary table

| Class | When to use |
|---|---|
| `Event` | Inspect or construct the events flowing through a runner; filter for final responses |
| `EventActions` | Write state, request transfers, escalate, or request auth from within tools/callbacks |
| `Context` | Access state, artifacts, memory, and auth from any callback or tool |
| `FunctionTool` | Wrap any Python function as a tool with automatic schema extraction |
| `LongRunningFunctionTool` | Async/polling operations that return intermediate status to the client |
| `AuthConfig` | Define the auth scheme a tool needs and drive the ADK credential exchange flow |
| `DatabaseSessionService` | SQLAlchemy-backed sessions for PostgreSQL, MySQL, or async SQLite |
| `SqliteSessionService` | Lightweight, purpose-built SQLite sessions with aiosqlite |
| `VertexAiSessionService` | Fully managed, scalable session storage on Google Cloud |
| `VertexAiSearchTool` | Ground Gemini responses in a Vertex AI Search knowledge base |

## Version note

Verified against **google-adk==2.11.0**. `Context`, `CallbackContext`, and `ToolContext` are all aliases for the same class in 2.11.0 — use whichever name is most readable in context. `DatabaseSessionService`, `SqliteSessionService`, `VertexAiSessionService`, `LongRunningFunctionTool`, and `AuthConfig` are all available in 2.11.0.

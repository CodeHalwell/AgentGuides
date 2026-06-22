---
title: "Class deep dives — volume 24 (google-adk 2.3.0)"
description: "Source-verified deep dives into 10 google-adk 2.3.0 classes not yet covered: LlmAgent mode system (chat/task/single_turn delegation modes), ContextCacheConfig (app-wide Gemini context caching), State (delta-aware session state dict with schema validation), DatabaseSessionService (SQLAlchemy-backed production sessions), VertexAiSessionService (Vertex AI Agent Engine sessions), VertexAiMemoryBankService (Vertex AI memory bank with ingest/generate APIs), Event (core LlmResponse subclass with is_final_response / node_info), EventActions (state_delta, transfer_to_agent, escalate, route, auth, UI widgets), ReadonlyContext (read-only base for Context/ToolContext), InvocationContext (per-invocation service registry and execution envelope)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 24"
  order: 93
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, constant, and code example is drawn directly from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `LlmAgent` mode system (`chat` / `task` / `single_turn`) | `google.adk.agents.llm_agent` | Stable |
| 2 | `ContextCacheConfig` | `google.adk.agents.context_cache_config` | `@experimental(AGENT_CONFIG)` |
| 3 | `State` + `StateSchemaError` | `google.adk.sessions.state` | Stable |
| 4 | `DatabaseSessionService` | `google.adk.sessions.database_session_service` | Stable |
| 5 | `VertexAiSessionService` | `google.adk.sessions.vertex_ai_session_service` | Stable |
| 6 | `VertexAiMemoryBankService` | `google.adk.memory.vertex_ai_memory_bank_service` | Stable |
| 7 | `Event` + `NodeInfo` | `google.adk.events.event` | Stable |
| 8 | `EventActions` + `EventCompaction` | `google.adk.events.event_actions` | Stable |
| 9 | `ReadonlyContext` | `google.adk.agents.readonly_context` | Stable |
| 10 | `InvocationContext` | `google.adk.agents.invocation_context` | Stable |

---

## 1 · `LlmAgent` mode system

**Source:** `google.adk.agents.llm_agent`

`LlmAgent` has a `mode` field (`Literal['chat', 'task', 'single_turn'] | None`) that controls how the agent participates in multi-agent workflows. This field underpins the Task API introduced in 2.x and determines how the agent is wired into `Runner.run_async()` and `Workflow`.

### Mode reference (source-verified)

```python
mode: Literal['chat', 'task', 'single_turn'] | None = None
```

| Mode | Meaning | Used as |
|---|---|---|
| `None` | Default; Runner auto-assigns `'chat'` to root agents | Root agent |
| `'chat'` | Multi-turn coordinator. Sees the full unscoped conversation history. Delegates to task/single-turn sub-agents. | Root or coordinator |
| `'task'` | Multi-turn delegated agent. Isolated to its own task scope (via `isolation_scope`). Runs until it calls `finish_task`. | Sub-agent |
| `'single_turn'` | Single-turn delegated agent. Runs once per delegation and returns immediately. Isolated like `'task'`. | Sub-agent |

### `disallow_transfer_to_parent` and `disallow_transfer_to_peers`

```python
agent = LlmAgent(
    name="worker",
    model="gemini-2.5-flash",
    instruction="Complete your task without escalating.",
    mode="single_turn",
    disallow_transfer_to_parent=True,   # cannot escalate to parent
    disallow_transfer_to_peers=False,   # can still transfer to sibling agents
)
```

When both `disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True` and no `sub_agents`, the agent is a leaf — a standalone executor.

### Multi-agent coordinator + task sub-agent pattern

```python
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService

# Sub-agent: task mode — runs multiple turns until finish_task
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    mode="task",
    instruction=(
        "You are a research assistant. When you receive a research task, "
        "gather information and call finish_task when complete."
    ),
)

# Sub-agent: single_turn — runs once and returns
formatter = LlmAgent(
    name="formatter",
    model="gemini-2.5-flash",
    mode="single_turn",
    instruction="Format the provided content as structured Markdown.",
)

# Coordinator: chat mode — orchestrates the other two
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    mode="chat",
    instruction=(
        "Orchestrate research and formatting. "
        "Delegate research to the researcher agent, "
        "then format results with the formatter agent."
    ),
    sub_agents=[researcher, formatter],
)

app = App(name="research-app", root_agent=coordinator)
runner = Runner(
    app=app,
    session_service=DatabaseSessionService("sqlite+aiosqlite:///sessions.db"),
)
```

### `output_schema` and `output_key`

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    keywords: list[str]

analyst = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction="Analyse the text and return structured output.",
    output_schema=AnalysisResult,   # enforces JSON schema; disables tool use by default
    output_key="analysis",          # writes result to session.state["analysis"]
)
```

`output_schema` coerces the model's response to the Pydantic type. Setting it alongside `tools` is supported in 2.3.0 — the ADK handles the combination via structured output mode.

### `include_contents`

```python
agent = LlmAgent(
    name="stateless",
    model="gemini-2.5-flash",
    instruction="Answer only from the current message, no history.",
    include_contents="none",   # wipes conversation history from each LLM request
)
```

Default is `"default"` — includes the full session history up to the context window.

### Model callbacks

```python
from datetime import datetime, timezone
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents.callback_context import CallbackContext

def inject_system_info(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    # Mutate the request before it reaches the model
    if llm_request.config:
        current_time = datetime.now(timezone.utc).isoformat()
        llm_request.config.system_instruction = (
            f"Server time: {current_time}\n"
            + (llm_request.config.system_instruction or "")
        )
    return None   # None = proceed with mutated request; return LlmResponse to skip model call

agent = LlmAgent(
    name="time_aware",
    model="gemini-2.5-flash",
    instruction="Always mention the current server time.",
    before_model_callback=inject_system_info,
    # Also available: after_model_callback, on_model_error_callback,
    #                 before_tool_callback, after_tool_callback
)
```

---

## 2 · `ContextCacheConfig`

**Source:** `google.adk.agents.context_cache_config`

`ContextCacheConfig` enables Gemini's **context caching** API across all agents in an app, reusing previously processed tokens to reduce latency and cost. Decorated `@experimental(FeatureName.AGENT_CONFIG)`.

### Constructor (source-verified)

```python
@experimental(FeatureName.AGENT_CONFIG)
class ContextCacheConfig(BaseModel):
    cache_intervals: int = Field(default=10, ge=1, le=100)
    ttl_seconds: int = Field(default=1800, gt=0)       # 30 minutes
    min_tokens: int = Field(default=0, ge=0)
    create_http_options: types.HttpOptions | None = None
```

### Constraints (source-verified from docstring)

- Caching starts **on the second turn** at the earliest — the first request is never cached.
- Requires the prior request to reach Gemini's hard **4096-token minimum**. Short sessions are never cached.
- `min_tokens` gates on the **previous request's** actual prompt token count. Values below 4096 have no additional effect (Gemini's hard floor always applies).

### Attaching to an `App`

```python
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App
from google.adk.agents import LlmAgent
from google.genai import types

app = App(
    name="long-doc-app",
    root_agent=LlmAgent(
        name="doc_qa",
        model="gemini-2.5-flash",
        instruction="Answer questions about the provided document.",
    ),
    context_cache_config=ContextCacheConfig(
        cache_intervals=5,       # refresh cache every 5 invocations
        ttl_seconds=3600,        # cache lives for 1 hour
        min_tokens=8192,         # only cache when prior prompt > 8 k tokens
        create_http_options=types.HttpOptions(timeout=15000),  # 15s cache-create timeout
    ),
)
```

### `ttl_string` property

```python
cfg = ContextCacheConfig(ttl_seconds=900)
print(cfg.ttl_string)   # "900s"  — passed directly to CachedContent.create()
```

---

## 3 · `State` + `StateSchemaError`

**Source:** `google.adk.sessions.state`

`State` is a delta-aware dict proxy that tracks both the committed session state (`_value`) and uncommitted pending changes (`_delta`). It is the type of `ctx.state` in both `Context` and `ReadonlyContext`.

### Key prefixes

```python
State.APP_PREFIX  = "app:"   # shared across all users of the app
State.USER_PREFIX = "user:"  # shared across all sessions for a user
State.TEMP_PREFIX = "temp:"  # per-invocation only, not persisted
```

Prefixed keys bypass schema validation (see below).

### Dict-like interface

```python
# In a tool or BaseNode._run_impl():
ctx.state["count"] = ctx.state.get("count", 0) + 1   # mutate
ctx.state.update({"a": 1, "b": 2})                    # bulk update
ctx.state.setdefault("initialized", False)             # set if absent

if "user_name" in ctx.state:
    name = ctx.state["user_name"]

# Cross-session user state
ctx.state["user:theme"] = "dark"    # persists across sessions for this user

# App-wide state
ctx.state["app:feature_flags"] = {"new_ui": True}

# Temp state (not persisted)
ctx.state["temp:scratch"] = {"intermediate": "data"}
```

### `has_delta()` — check for uncommitted changes

```python
if ctx.state.has_delta():
    print("Pending state changes:", ctx.state._delta)
```

### `state_schema` validation with `StateSchemaError`

When a `BaseNode` or `LlmAgent` declares `state_schema`, mutations are validated at runtime:

```python
from pydantic import BaseModel
from google.adk.sessions.state import StateSchemaError

class AppState(BaseModel):
    user_name: str
    request_count: int = 0
    last_query: str | None = None

# In a tool:
async def update_count(tool_context):
    try:
        tool_context.state["request_count"] += 1      # OK: int
        tool_context.state["request_count"] = "oops"  # Raises StateSchemaError
        tool_context.state["unknown_key"] = "x"       # Raises StateSchemaError
    except StateSchemaError as e:
        print(f"Schema violation: {e}")

    # Prefixed keys always pass validation:
    tool_context.state["user:prefs"] = {"theme": "dark"}  # OK (bypasses schema)
```

`StateSchemaError` is a `TypeError` subclass. The error message identifies the key, the expected type, and the schema class name.

---

## 4 · `DatabaseSessionService`

**Source:** `google.adk.sessions.database_session_service`

`DatabaseSessionService` stores sessions in any SQLAlchemy-supported database. It handles schema migrations automatically (v0 → v1), row-level locking for concurrent writes, and SQLite-specific optimisations.

### Constructor

```python
DatabaseSessionService(
    db_url: str,
    **kwargs,          # forwarded to create_async_engine()
)
```

Requires the `sqlalchemy` extra: `pip install google-adk[db]`.

### Supported databases and URL formats

```python
from google.adk.sessions.database_session_service import DatabaseSessionService

# SQLite — local file (good for development)
svc = DatabaseSessionService("sqlite+aiosqlite:///./my_sessions.db")

# SQLite — in-memory (testing; uses StaticPool automatically)
svc = DatabaseSessionService("sqlite+aiosqlite:///:memory:")

# PostgreSQL
svc = DatabaseSessionService(
    "postgresql+asyncpg://user:pass@localhost:5432/mydb",
    pool_pre_ping=True,     # detect stale connections (set automatically for non-SQLite)
    pool_size=10,
    max_overflow=5,
)

# MySQL / MariaDB
svc = DatabaseSessionService("mysql+aiomysql://user:pass@localhost:3306/mydb")
```

SQLite automatically enables `PRAGMA foreign_keys = ON` and configures `check_same_thread=False`.

### Production runner setup

```python
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService

runner = Runner(
    app=App(name="prod-app", root_agent=agent),
    session_service=DatabaseSessionService(
        "postgresql+asyncpg://user:pass@db:5432/sessions"
    ),
    artifact_service=GcsArtifactService("my-artifacts-bucket"),
    auto_create_session=True,
)
```

### Session lifecycle

```python
# Create
session = await svc.create_session(
    app_name="my-app", user_id="u1", session_id="s1",
    state={"onboarded": True},
)

# Load with event limit (combine with RunConfig.get_session_config)
from google.adk.sessions.base_session_service import GetSessionConfig

session = await svc.get_session(
    app_name="my-app", user_id="u1", session_id="s1",
    config=GetSessionConfig(num_recent_events=20),
)

# List all sessions for a user
result = await svc.list_sessions(app_name="my-app", user_id="u1")
for s in result.sessions:
    print(s.session_id, s.last_update_time)

# Delete
await svc.delete_session(app_name="my-app", user_id="u1", session_id="s1")
```

### Stale session detection

`DatabaseSessionService` tracks a version counter per session. If two concurrent callers load the same session and both try to `append_event`, the second write raises a stale-session error:

```
"The session has been modified in storage since it was loaded. Please reload the session before appending more events."
```

---

## 5 · `VertexAiSessionService`

**Source:** `google.adk.sessions.vertex_ai_session_service`

`VertexAiSessionService` delegates session storage to the **Vertex AI Agent Engine Session Service** — a managed, scalable session backend that runs on Google Cloud.

### Constructor (source-verified)

```python
VertexAiSessionService(
    project: str | None = None,
    location: str | None = None,
    agent_engine_id: str | None = None,     # short ID, e.g. "456"
    *,
    express_mode_api_key: str | None = None,
)
```

Requires `pip install google-adk[gcp]` (provides `google-cloud-aiplatform`).

Pass just the short engine ID (e.g. `"456"`), not the full resource path. Passing a full path emits a warning and the service extracts the trailing component.

### Production setup

```python
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.adk.runners import Runner
from google.adk.apps.app import App

svc = VertexAiSessionService(
    project="my-gcp-project",
    location="us-central1",
    agent_engine_id="456",
)

runner = Runner(
    app=App(name="vertex-app", root_agent=agent),
    session_service=svc,
)
```

### Session ID format

Session IDs must match `^[A-Za-z0-9_-]+$`. Slashes and other special characters are rejected to prevent URL path escaping. Passing a full resource name (e.g. `projects/.../sessions/my-session`) is handled gracefully — the service extracts `my-session` automatically.

### Express Mode

For Vertex AI Express Mode (no full project setup required), provide the API key:

```python
import os

svc = VertexAiSessionService(
    project="my-gcp-project",
    location="us-central1",
    agent_engine_id="456",
    express_mode_api_key=os.environ["GOOGLE_API_KEY"],
)
```

The service also reads `GOOGLE_API_KEY` from the environment automatically when `GOOGLE_GENAI_USE_ENTERPRISE=true`.

---

## 6 · `VertexAiMemoryBankService`

**Source:** `google.adk.memory.vertex_ai_memory_bank_service`

`VertexAiMemoryBankService` connects ADK agents to the **Vertex AI Agent Engine Memory Bank** — a managed long-term memory store. It supports two ingestion paths: `ingest_events` (default, streaming) and `generate_memories` (batch summarisation).

### Constructor (source-verified)

```python
VertexAiMemoryBankService(
    project: str | None = None,
    location: str | None = None,
    agent_engine_id: str,               # required; short ID e.g. "456"
    *,
    express_mode_api_key: str | None = None,
)
```

### Attaching to a runner

```python
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.runners import Runner
from google.adk.apps.app import App

memory_svc = VertexAiMemoryBankService(
    project="my-gcp-project",
    location="us-central1",
    agent_engine_id="456",
)

runner = Runner(
    app=App(name="memory-app", root_agent=agent),
    session_service=session_svc,
    memory_service=memory_svc,
)
```

### Ingestion API selection

The service automatically routes to `ingest_events` vs `generate_memories` based on `custom_metadata` keys passed to `RunConfig`:

```python
from google.adk.agents.run_config import RunConfig

# Default path: ingest_events (streaming, low-latency)
async for event in runner.run_async(
    user_id="u1", session_id="s1",
    new_message={"role": "user", "parts": [{"text": "Remember this fact."}]},
):
    ...

# Force generate_memories path (batch summarisation)
async for event in runner.run_async(
    user_id="u1", session_id="s1",
    new_message={"role": "user", "parts": [{"text": "Summarise everything."}]},
    run_config=RunConfig(
        custom_metadata={
            "disable_consolidation": False,  # generate_memories-only key → forces that path
            "ttl": "86400s",
        }
    ),
):
    ...
```

### Searching memory from a tool

```python
async def recall_facts(query: str, tool_context):
    response = await tool_context.search_memory(query)
    facts = [entry.content for entry in response.memories]
    return {"facts": facts}

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction="Use recall_facts to find relevant information before answering.",
    tools=[recall_facts],
)
```

`search_memory` calls `VertexAiMemoryBankService.search_memory` which issues a `memories.search` request to the Agent Engine.

---

## 7 · `Event` + `NodeInfo`

**Source:** `google.adk.events.event`

`Event` extends `LlmResponse` (from `google.genai`) with ADK-specific fields for invocation identity, workflow metadata, and side effects. Every piece of data flowing through an ADK invocation — user messages, model responses, tool calls, tool results — is an `Event`.

### Core fields (source-verified)

```python
class Event(LlmResponse):
    invocation_id: str = ''         # links event to its runner.run_async() call
    author: str = ''                # 'user' or the agent's name
    actions: EventActions           # side effects (state, transfers, auth, etc.)
    output: Any | None = None       # workflow node output value
    node_info: NodeInfo             # workflow path, run_id, output_for
    long_running_tool_ids: set[str] | None = None
    branch: str | None = None       # dot-separated agent path (for sub-agent isolation)
    isolation_scope: str | None = None   # internal; do not use directly
    id: str = ''                    # assigned by session.append_event()
    timestamp: float                # POSIX timestamp
    # Inherited from LlmResponse:
    content: types.Content | None   # model text, function calls, function results
    partial: bool | None = None     # True = streaming chunk, False/None = final
    usage_metadata: types.GenerateContentResponseUsageMetadata | None = None
```

### `is_final_response()` — the canonical filter

```python
async for event in runner.run_async(...):
    if event.is_final_response():
        # Safe to read the final text or structured output
        if event.content and event.content.parts:
            print(event.content.parts[0].text)
        # Or check for workflow node output:
        if event.output is not None:
            print("Node output:", event.output)
```

Source definition (verified, `events/event.py:275-290`):

```python
def is_final_response(self) -> bool:
    # Early-return True for special event types
    if self.actions.skip_summarization or self.long_running_tool_ids:
        return True
    return (
        not self.get_function_calls()
        and not self.get_function_responses()
        and not self.partial
        and not self.has_trailing_code_execution_result()
    )
```

Two early-return cases return `True` unconditionally:
- `actions.skip_summarization` — a function-response event marked to skip LLM summarisation
- `long_running_tool_ids` — the event carries long-running tool completion IDs

Otherwise, it returns `True` only when there are no pending function calls/responses, `partial` is falsy (not a streaming chunk), and the last content part is not a code-execution result.

### Extracting function calls and responses

```python
async for event in runner.run_async(...):
    for fc in event.get_function_calls():
        print(f"Calling {fc.name}({fc.args})")

    for fr in event.get_function_responses():
        print(f"Tool {fr.name} returned: {fr.response}")
```

### `NodeInfo` — workflow node metadata

```python
async for event in runner.run_async(...):
    print(event.node_info.path)       # e.g. "pipeline/fetch@1"
    print(event.node_info.run_id)     # e.g. "1"
    print(event.node_info.name)       # e.g. "fetch" (clean name without @run_id)
    print(event.node_info.output_for) # list of parent paths this output satisfies
```

### Convenience kwargs

`Event` accepts three convenience kwargs that are remapped to nested fields:

```python
# Instead of EventActions(state_delta={"key": "val"}):
event = Event(
    author="agent",
    invocation_id="inv-1",
    content=types.Content(role="model", parts=[types.Part(text="Done.")]),
    state={"key": "val"},       # → actions.state_delta
    route="branch_a",           # → actions.route
    node_path="wf/step@1",      # → node_info.path
)
```

---

## 8 · `EventActions` + `EventCompaction`

**Source:** `google.adk.events.event_actions`

`EventActions` carries all side effects emitted by an event. It is the mechanism through which agents mutate session state, transfer control, request authentication, render UI widgets, and control compaction.

### Field reference (source-verified)

```python
class EventActions(BaseModel):
    state_delta: dict[str, Any] = {}
    artifact_delta: dict[str, int] = {}      # filename → version
    transfer_to_agent: str | None = None
    escalate: bool | None = None
    skip_summarization: bool | None = None   # for function_response events only
    requested_auth_configs: dict[str, AuthConfig] = {}   # fc_id → AuthConfig
    requested_tool_confirmations: dict[str, ToolConfirmation] = {}
    compaction: EventCompaction | None = None
    end_of_agent: bool | None = None         # internal; set by ADK workflow
    agent_state: dict[str, Any] | None = None
    rewind_before_invocation_id: str | None = None
    route: str | int | bool | list[Any] | None = None   # workflow edge routing
    render_ui_widgets: list[UiWidget] | None = None
    set_model_response: Any | None = None
```

### Mutation patterns in tools

```python
# Writing state
def my_tool(value: str, tool_context) -> str:
    tool_context.state["result"] = value          # goes to actions.state_delta
    return "Saved."

# Transferring control to another agent
def route_to_specialist(topic: str, tool_context) -> str:
    tool_context.actions.transfer_to_agent = "specialist_agent"
    return f"Routing to specialist for: {topic}"

# Escalating back to the parent
def escalate_issue(reason: str, tool_context) -> str:
    tool_context.actions.escalate = True
    return f"Escalated: {reason}"
```

### Rendering UI widgets

```python
import json
import urllib.parse
from google.adk.events.ui_widget import UiWidget

def show_chart(data: dict, tool_context) -> str:
    encoded_data = urllib.parse.quote(json.dumps(data))
    widget = UiWidget(
        iframe_url="https://charts.example.com/embed?data=" + encoded_data,
        title="Analytics Chart",
    )
    if tool_context.actions.render_ui_widgets is None:
        tool_context.actions.render_ui_widgets = []
    tool_context.actions.render_ui_widgets.append(widget)
    return "Chart rendered."
```

### `EventCompaction` — reading compacted event metadata

When `EventsCompactionConfig` triggers a compaction, a special event is created with `actions.compaction` set:

```python
async for event in runner.run_async(...):
    if event.actions.compaction:
        comp = event.actions.compaction
        print(f"Compacted events from {comp.start_timestamp} to {comp.end_timestamp}")
        print("Summary:", comp.compacted_content.parts[0].text)
```

`EventCompaction` fields:
- `start_timestamp: float` — POSIX timestamp of earliest compacted event
- `end_timestamp: float` — POSIX timestamp of latest compacted event
- `compacted_content: types.Content` — the LLM-generated summary

---

## 9 · `ReadonlyContext`

**Source:** `google.adk.agents.readonly_context`

`ReadonlyContext` is the read-only base class for `Context` (and therefore `ToolContext`). It wraps an `InvocationContext` and exposes safe, non-mutating views of session data, run configuration, and credentials. Plugin `BasePlugin` lifecycle methods typically receive `ReadonlyContext` when they do not need to write state.

### Property reference (source-verified)

```python
class ReadonlyContext:
    @property
    def user_content(self) -> types.Content | None: ...     # user's starting message
    @property
    def invocation_id(self) -> str: ...                     # current invocation ID
    @property
    def agent_name(self) -> str: ...                        # current agent's name
    @property
    def state(self) -> MappingProxyType[str, Any]: ...      # immutable state view
    @property
    def session(self) -> Session: ...                       # the current Session
    @property
    def user_id(self) -> str: ...
    @property
    def run_config(self) -> RunConfig | None: ...           # RunConfig for this run
    def get_credential(self, key: str) -> AuthCredential | None: ...
```

The `state` property returns a `MappingProxyType` — a read-only dict proxy. Attempts to mutate it raise `TypeError`. Use `Context.state` (writable) in tools and nodes; use `ReadonlyContext.state` in read-only callbacks.

### Typical usage — `before_model_callback`

```python
from google.adk.agents.callback_context import CallbackContext

def log_before_model(callback_context: CallbackContext, llm_request):
    # CallbackContext extends ReadonlyContext — provides all read-only properties
    user = callback_context.user_id
    agent = callback_context.agent_name
    turn = callback_context.state.get("turn_count", 0)
    print(f"[{agent}] User {user} — turn {turn}")
    return None   # proceed normally
```

### Usage in plugin lifecycle hooks

```python
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.readonly_context import ReadonlyContext

class MetricsPlugin(BasePlugin):
    async def on_run_start(
        self,
        *,
        invocation_context,
        readonly_context: ReadonlyContext,
    ) -> None:
        # read-only — safe to inspect but not mutate
        print(f"Invocation {readonly_context.invocation_id} started for {readonly_context.user_id}")
```

---

## 10 · `InvocationContext`

**Source:** `google.adk.agents.invocation_context`

`InvocationContext` is the internal envelope for a single `runner.run_async()` call. It carries all services, the current session, the active agent, LLM call accounting, and live-session state. You rarely construct it directly — the `Runner` creates it and passes it down. Understanding its structure clarifies how ADK wires services together.

### Field reference (source-verified)

```python
class InvocationContext(BaseModel):
    # Services (from Runner)
    artifact_service: BaseArtifactService | None
    session_service: BaseSessionService
    memory_service: BaseMemoryService | None
    credential_service: BaseCredentialService | None
    context_cache_config: ContextCacheConfig | None

    # Identity
    invocation_id: str
    branch: str | None          # dot-separated agent ancestry, e.g. "root.worker"
    isolation_scope: str | None # internal task-scope tag; do not use directly

    # Execution state
    agent: BaseAgent | BaseNode | None
    user_content: types.Content | None  # user's originating message
    session: Session

    # Workflow
    node_path: str | None       # e.g. "pipeline/fetch@1"
    agent_states: dict[str, dict]
    end_of_agents: dict[str, bool]

    # LLM cost control
    # (via _InvocationCostManager — internal)
    run_config: RunConfig | None

    # Live mode
    live_request_queue: LiveRequestQueue | None
    active_streaming_tool: ActiveStreamingTool | None
```

### How `Runner` builds `InvocationContext`

```python
# Pseudocode — Runner.run_async() internal flow:
ctx = InvocationContext(
    artifact_service=self.artifact_service,
    session_service=self.session_service,
    memory_service=self.memory_service,
    credential_service=self.credential_service,
    context_cache_config=self.context_cache_config,
    invocation_id=new_invocation_context_id(),
    agent=self.agent,
    user_content=new_message,
    session=session,
    run_config=run_config,
)
```

### LLM call limit enforcement

The `_InvocationCostManager` inside `InvocationContext` tracks every LLM call and raises `LlmCallsLimitExceededError` when the `RunConfig.max_llm_calls` ceiling is hit:

```python
from google.adk.agents.run_config import RunConfig

# Limit an invocation to at most 10 LLM calls
run_config = RunConfig(max_llm_calls=10)
async for event in runner.run_async(
    user_id="u1", session_id="s1",
    new_message={"role": "user", "parts": [{"text": "Do 20 things."}]},
    run_config=run_config,
):
    if event.is_final_response():
        print(event.content.parts[0].text)
# Raises LlmCallsLimitExceededError if agent tries to make an 11th LLM call
```

### `new_invocation_context_id()`

```python
from google.adk.agents.invocation_context import new_invocation_context_id

inv_id = new_invocation_context_id()   # returns a UUID-based string
```

Used by `Runner` to assign a unique `invocation_id` to every `run_async` call. This ID is stamped on every `Event.invocation_id` emitted during that run.

---

## Quick-reference table

| What you want | Class / field | Module |
|---|---|---|
| Multi-turn orchestrator + delegated sub-agents | `LlmAgent(mode='chat')` + `sub_agents=[LlmAgent(mode='task')]` | `google.adk.agents.llm_agent` |
| Structured output from an agent | `LlmAgent(output_schema=MyModel, output_key="result")` | `google.adk.agents.llm_agent` |
| Wipe history per call | `LlmAgent(include_contents='none')` | `google.adk.agents.llm_agent` |
| Reduce costs with Gemini context cache | `App(context_cache_config=ContextCacheConfig(...))` | `google.adk.agents.context_cache_config` |
| Read session state (immutable) | `ctx.state.get("key")` | `google.adk.sessions.state` |
| Write session state | `ctx.state["key"] = val` | `google.adk.sessions.state` |
| Validate state mutations | `BaseNode(state_schema=AppStateModel)` | `google.adk.sessions.state` |
| Production SQL sessions (SQLite/Postgres/MySQL) | `DatabaseSessionService("dialect+driver://...")` | `google.adk.sessions.database_session_service` |
| Google Cloud managed sessions | `VertexAiSessionService(project=..., agent_engine_id=...)` | `google.adk.sessions.vertex_ai_session_service` |
| Long-term memory (Vertex AI Memory Bank) | `VertexAiMemoryBankService(agent_engine_id=...)` | `google.adk.memory.vertex_ai_memory_bank_service` |
| Filter for final model responses | `event.is_final_response()` | `google.adk.events.event` |
| Read workflow node path | `event.node_info.path` | `google.adk.events.event` |
| Transfer to another agent in a tool | `ctx.actions.transfer_to_agent = "agent_name"` | `google.adk.events.event_actions` |
| Escalate to parent in a tool | `ctx.actions.escalate = True` | `google.adk.events.event_actions` |
| Read-only access to session/user/config | `ReadonlyContext` | `google.adk.agents.readonly_context` |
| Inspect invocation services | `InvocationContext.artifact_service`, `.session_service`, etc. | `google.adk.agents.invocation_context` |

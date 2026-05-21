---
title: "Runner, App, and Sessions"
description: "Runner / InMemoryRunner, the App container, session services, RunConfig, and state scopes."
framework: google-adk
language: python
sidebar:
  order: 50
---

Verified against google-adk==2.0.0 (`google/adk/runners.py`, `google/adk/apps/app.py`, `google/adk/sessions/`).

The `Runner` glues an agent/workflow to the three per-session services (session, memory, artifact) plus a credential service and plugin manager. `App` is the container that bundles the root agent with app-wide settings.

## Minimal example (in-memory)

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(name="chat", model="gemini-2.5-flash", instruction="Be concise.")

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Hi")]),
    ):
        if event.content:
            for part in event.content.parts:
                if part.text:
                    print(event.author, "→", part.text)
    await runner.close()

asyncio.run(main())
```

`InMemoryRunner` subclasses `Runner` and wires in `InMemorySessionService` + `InMemoryMemoryService` + `InMemoryArtifactService` (`runners.py:1970`).

## The App container

```python
from google.adk.apps import App, ResumabilityConfig, EventsCompactionConfig
from google.adk.plugins import LoggingPlugin
from google.adk.agents.context_cache_config import ContextCacheConfig

app = App(
    name="demo",
    root_agent=my_agent_or_workflow,
    plugins=[LoggingPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=10,
        overlap_size=2,
        token_threshold=50_000,
        event_retention_size=50,
    ),
    context_cache_config=ContextCacheConfig(...),  # optional, for explicit Gemini cache
)
```

| Field | Default | Purpose |
|---|---|---|
| `name` | required | Must be a valid Python identifier; reserved word `"user"` is forbidden |
| `root_agent` | required | `BaseAgent` or `BaseNode` (e.g. a `Workflow`) |
| `plugins` | `[]` | App-wide plugins (ordered) |
| `events_compaction_config` | `None` | Sliding-window event compaction |
| `context_cache_config` | `None` | Gemini context cache config, applied to every LLM call |
| `resumability_config` | `None` | Enables pause/resume around long-running tools |

## Runner constructor

```python
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

runner = Runner(
    app=app,                                   # preferred
    session_service=DatabaseSessionService(db_url="sqlite:///./adk.db"),
    memory_service=memory_service,             # optional
    artifact_service=artifact_service,         # optional
    credential_service=credential_service,     # optional
    auto_create_session=False,
    plugin_close_timeout=5.0,
)
```

Exactly one of `app=`, `agent=`, or `node=` is required (`runners.py:196-274`). `plugins=` on the runner is **deprecated** — pass them through `App(plugins=[...])` instead. `auto_create_session=True` is a convenience flag — when the session service returns `None`, the runner creates one on the fly; otherwise it raises `SessionNotFoundError` with an app-name alignment hint.

### Key methods

| Method | Purpose |
|---|---|
| `async run_async(*, user_id, session_id, new_message=None, invocation_id=None, run_config=None, state_delta=None, yield_user_message=False)` | Primary entry point. Yields `Event`s. |
| `run(...)` | Sync wrapper — starts a background thread. For local testing only. |
| `async run_live(*, live_request_queue, user_id, session_id, run_config=None)` | Bidi streaming (audio/video). Experimental. |
| `async run_debug(user_messages, *, user_id="debug_user_id", session_id="debug_session_id", run_config=None, quiet=False, verbose=False)` | Quick REPL-style helper. Returns a list of events. |
| `async rewind_async(*, user_id, session_id, rewind_before_invocation_id, run_config=None)` | Rewinds the session state and artifacts to before the given invocation. |
| `async close()` | Closes toolsets and plugins. Call it on shutdown (or use `async with runner:`). |

All `run_*` methods work with `asyncio`. Wire `async with Runner(...) as runner:` for auto-cleanup.

## RunConfig

Passed to each `run_async`/`run_live` call (`agents/run_config.py:184`). Notable fields:

| Field | Default | Notes |
|---|---|---|
| `streaming_mode` | `StreamingMode.NONE` | `SSE` for HTTP streaming, `BIDI` for live API |
| `max_llm_calls` | `500` | Hard cap per run. `<=0` disables |
| `response_modalities` | `None` | e.g. `["TEXT"]` or `["AUDIO"]` for live |
| `speech_config` / `avatar_config` | `None` | Live mode TTS / avatar |
| `output_audio_transcription` / `input_audio_transcription` | `AudioTranscriptionConfig()` | Live transcription |
| `context_window_compression` | `None` | Live-mode server-side context compression |
| `get_session_config` | `None` | Passes `num_recent_events` / `after_timestamp` through to the session service on load |
| `support_cfc` | `False` | Experimental compositional function calling (requires Gemini 2.x + live API) |
| `tool_thread_pool_config` | `None` | Runs tools in a thread pool during live mode |
| `custom_metadata` | `None` | Merged into every emitted event |
| `save_input_blobs_as_artifacts` | `False` | **Deprecated** → `SaveFilesAsArtifactsPlugin` |
| `save_live_audio` | `False` | **Deprecated** → `save_live_blob` |

```python
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.sessions.base_session_service import GetSessionConfig

cfg = RunConfig(
    streaming_mode=StreamingMode.SSE,
    max_llm_calls=50,
    get_session_config=GetSessionConfig(num_recent_events=20),
)
```

### SSE streaming — event filtering

`StreamingMode.SSE` yields both **partial** (streaming chunks) and **final** events. Without care you display the text twice. Three strategies from `run_config.py`:

```python
from google.adk.agents.run_config import RunConfig, StreamingMode

cfg = RunConfig(streaming_mode=StreamingMode.SSE)

# ── Strategy 1: typewriter effect (show partials, skip final text) ──────────
async for event in runner.run_async(..., run_config=cfg):
    if event.partial and event.content:
        parts = event.content.parts or []
        has_text = any(p.text for p in parts)
        has_fc   = any(p.function_call for p in parts)
        if has_text and not has_fc:
            print("".join(p.text or "" for p in parts), end="", flush=True)
    elif not event.partial and event.get_function_calls():
        for fc in event.get_function_calls():
            print(f"\n[tool] {fc.name}({fc.args})")

# ── Strategy 2: final-only (no streaming effect) ────────────────────────────
async for event in runner.run_async(..., run_config=cfg):
    if not event.partial and event.is_final_response() and event.content:
        print("".join(p.text or "" for p in event.content.parts or []))

# ── Strategy 3: track what was already streamed ─────────────────────────────
streamed = ""
async for event in runner.run_async(..., run_config=cfg):
    if event.partial and event.content:
        chunk = "".join(p.text or "" for p in event.content.parts or [])
        print(chunk, end="", flush=True)
        streamed += chunk
    elif not event.partial and event.content:
        final = "".join(p.text or "" for p in event.content.parts or [])
        if final != streamed:
            print(final)   # only if the final has content we didn't stream yet
```

### `ToolThreadPoolConfig` — live mode concurrency

In live mode (`run_live`) tools run in the event loop by default. Set `tool_thread_pool_config` to run them in a thread pool, keeping the loop responsive to audio/video interrupts:

```python
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

cfg = RunConfig(
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
    save_live_blob=True,   # persist audio/video frames to artifact service
)
# Note: thread pool helps with blocking I/O (network, DB, file); it does NOT
# provide parallelism for pure-Python CPU work (GIL still applies).
```

## Session services

All subclass `BaseSessionService` and expose `create_session`, `get_session`, `list_sessions`, `delete_session`, `append_event`.

| Service | Import | Storage | Notes |
|---|---|---|---|
| `InMemorySessionService` | `google.adk.sessions` | Python dict | Dev/testing |
| `DatabaseSessionService(db_url)` | `google.adk.sessions` (lazy) | Any SQLAlchemy URL | Requires `sqlalchemy>=2.0`. Supports SQLite, Postgres, MySQL, Spanner |
| `SqliteSessionService` | `google.adk.sessions.sqlite_session_service` | SQLite file (async) | Zero-dep alternative to `DatabaseSessionService` for SQLite |
| `VertexAiSessionService(project, location, agent_engine_id, *, express_mode_api_key=None)` | `google.adk.sessions` | Vertex AI Agent Engine | Production-ready, scales with Agent Engine |

```python
from google.adk.sessions import DatabaseSessionService, VertexAiSessionService

# Postgres
svc = DatabaseSessionService(db_url="postgresql+asyncpg://user:pass@host/db")

# Vertex AI
svc = VertexAiSessionService(
    project="my-gcp-project",
    location="us-central1",
    agent_engine_id="1234567890",
)
```

`DatabaseSessionService.create_session`/`get_session` run the underlying SQL inside an async session factory; Postgres and MySQL use row-level locking for concurrent `append_event` (`database_session_service.py:282-320`).

## Session state

Access via `ctx.state` in callbacks and tools. State is a `dict`-like object with three reserved prefixes (`sessions/state.py:64-66`):

| Prefix | Lifetime | Example |
|---|---|---|
| *(none)* | Session | `ctx.state["last_query"] = "..."` |
| `app:` | All sessions for the app | `ctx.state["app:feature_flag"] = True` |
| `user:` | All sessions for that user | `ctx.state["user:preferred_language"] = "en"` |
| `temp:` | Current invocation only (stripped before persist) | `ctx.state["temp:scratch"] = [...]` |

Declare a Pydantic schema on the `Workflow` (`state_schema=`) to validate mutations at runtime. Reserved prefixes bypass validation.

## Context caching (`ContextCacheConfig`, experimental)

`ContextCacheConfig` enables Gemini's server-side context caching for all `LlmAgent`s in an `App`. When active, ADK caches the system instruction + static tools prefix in Gemini's cache after the first call and reuses it for up to `cache_intervals` subsequent invocations. Cached tokens are billed at a reduced rate.

```python
from google.adk.apps import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

app = App(
    name="cached_app",
    root_agent=my_agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=20,     # reuse the same cache for up to 20 invocations
        ttl_seconds=3600,       # 1-hour TTL on the cache entry
        min_tokens=1000,        # only cache requests with >= 1000 estimated tokens
    ),
)

runner = Runner(
    app=app,
    session_service=DatabaseSessionService(db_url="sqlite+aiosqlite:///./adk.db"),
)
```

**Fields (from `agents/context_cache_config.py`):**

| Field | Default | Constraints | Purpose |
|---|---|---|---|
| `cache_intervals` | `10` | 1–100 | Max invocations before the cache is refreshed |
| `ttl_seconds` | `1800` (30 min) | > 0 | How long the cache entry lives in Gemini |
| `min_tokens` | `0` | ≥ 0 | Minimum estimated request tokens to bother caching |

**When to use:**
- Long system instructions (> 1 k tokens) that rarely change
- Large tool schemas registered on the agent
- High-frequency, short user messages (chat, Q&A)

> **Experimental**: `ContextCacheConfig` requires `@experimental(FeatureName.AGENT_CONFIG)`. It is only compatible with Gemini models that support context caching (check the Gemini docs for supported model versions).

## `run_async` return semantics

`run_async` yields `Event` objects one at a time. Each event carries:

- `event.author` — agent name or `"user"`.
- `event.content` — the message content (`types.Content`).
- `event.actions` — state delta, artifact delta, `escalate`, `transfer_to_agent`, `skip_summarization`, etc.
- `event.partial` — `True` for streaming chunks; the final event is non-partial.
- `event.usage_metadata` — token counts, only on model events.
- `event.get_function_calls()` / `get_function_responses()` — helpers to peel function-call events.

Non-partial events are persisted via `session_service.append_event` before being yielded.

## `Event` and `EventActions` deep-dive

`Event` (defined in `events/event.py`) extends `LlmResponse` and is the single unit of communication between agents, tools, and the runner.

### Key fields

| Field | Type | Purpose |
|---|---|---|
| `id` | `str` | Unique event ID (auto-generated UUID) |
| `invocation_id` | `str` | Groups all events from one `run_async` call |
| `author` | `str` | `"user"` or the agent name |
| `content` | `types.Content \| None` | The message content (parts with text, function calls, etc.) |
| `actions` | `EventActions` | Side-effects — state delta, routing, auth requests, etc. |
| `partial` | `bool` | `True` for SSE streaming chunks; `False` for the final aggregated event |
| `timestamp` | `float` | Unix seconds when the event was created |
| `long_running_tool_ids` | `set[str] \| None` | IDs of long-running function calls (pauses the invocation) |
| `branch` | `str \| None` | Dot-separated agent hierarchy (e.g. `"triage.billing"`) |
| `node_info` | `NodeInfo` | Workflow node metadata (`path`, `run_id`, `name`) |
| `output` | `Any \| None` | Structured output from a workflow node |
| `usage_metadata` | `UsageMetadata \| None` | Token counts (prompt, candidates, total) |

### Useful methods

```python
# Is this the agent's final reply for this turn?
if event.is_final_response():
    text = "".join(p.text or "" for p in (event.content.parts or []))
    print(text)

# Extract tool calls
for fc in event.get_function_calls():
    print(f"Tool: {fc.name}({fc.args})")

# Extract tool responses
for fr in event.get_function_responses():
    print(f"Result for {fr.name}: {fr.response}")

# Convenience alias: event.message ↔ event.content
text_parts = [p.text for p in (event.message.parts or []) if p.text]
```

`is_final_response()` returns `True` when:
- No function calls or responses are present, AND
- The event is not partial (`partial=False`), AND
- There is no trailing code execution result, OR
- `actions.skip_summarization` is set (long-running tool response pattern)

### `EventActions` fields

`EventActions` (`events/event_actions.py`) carries the side-effects of an event:

| Field | Type | Purpose |
|---|---|---|
| `state_delta` | `dict[str, Any]` | State keys/values to merge into the session on commit |
| `artifact_delta` | `dict[str, int]` | Filename → version; auto-populated by `save_artifact` |
| `transfer_to_agent` | `str \| None` | Route control to this agent name after this event |
| `escalate` | `bool \| None` | Exit a loop — used by `LoopAgent` / workflow loop nodes |
| `skip_summarization` | `bool \| None` | Suppress the LLM from paraphrasing the tool result |
| `requested_auth_configs` | `dict[str, AuthConfig]` | Pending OAuth flows, keyed by function-call ID |
| `requested_tool_confirmations` | `dict[str, ToolConfirmation]` | Pending HITL confirmations |
| `compaction` | `EventCompaction \| None` | Sliding-window compaction metadata |
| `end_of_agent` | `bool \| None` | Marks that the originating agent has finished its run |
| `route` | `str \| bool \| int \| list \| None` | Workflow routing value |
| `render_ui_widgets` | `list[UiWidget] \| None` | Rich UI widgets for ADK Web UI hosts |
| `rewind_before_invocation_id` | `str \| None` | Rewind anchor (set by `runner.rewind_async`) |

### Emitting custom events from tools

Tools that want to influence routing or state without being the final response can construct and return structured dicts — the runner converts them. For more control, a `@node` in a `Workflow` can `yield Event(state={"key": "val"})` or `yield Event(route="billing")`:

```python
from google.adk.events.event import Event
from google.adk.workflow import node

@node(rerun_on_resume=True)
async def classify_and_route(user_msg: str, ctx):
    intent = "billing" if "invoice" in user_msg.lower() else "support"
    # Emit a state update alongside the routing decision
    yield Event(
        state={"last_intent": intent},
        route=intent,
    )
```

### `NodeInfo` — workflow event metadata

Events emitted inside a `Workflow` carry a `node_info` field:

```python
for event in events:
    if event.node_info.path:
        print(f"Node: {event.node_info.name}, run_id: {event.node_info.run_id}")
```

| Property | Purpose |
|---|---|
| `node_info.path` | Full slash-separated path, e.g. `"my_wf/classify@1/billing@1"` |
| `node_info.name` | Just the node name without `@run_id`, e.g. `"billing"` |
| `node_info.run_id` | Execution ID of this run (useful when a node runs multiple times in a loop) |

## Artifact service

Runners accept an optional `artifact_service=`. When configured, tools can call `tool_context.save_artifact("report.pdf", part)` and `load_artifact(...)`. Available services (`artifacts/__init__.py`):

| Service | Storage |
|---|---|
| `InMemoryArtifactService()` | Dict in memory |
| `FileArtifactService(root_dir=...)` | Local filesystem |
| `GcsArtifactService(bucket_name=...)` | Google Cloud Storage |

See [memory-and-artifacts](./memory-and-artifacts/) for detailed semantics and versioning.

## `InvocationContext` — invocation lifecycle

Each call to `runner.run_async` creates one `InvocationContext`. Callbacks and plugins receive it via `invocation_context` kwargs. Key fields:

| Field | Type | Notes |
|---|---|---|
| `invocation_id` | `str` | Unique per `run_async` call; prefix `"e-"` + UUID |
| `session` | `Session` | The live session object |
| `user_content` | `types.Content \| None` | The original user message (readonly) |
| `run_config` | `RunConfig \| None` | Per-run config |
| `end_invocation` | `bool` | Set to `True` to abort the current invocation early |
| `agent` | `BaseAgent \| BaseNode \| None` | The currently-executing agent |

### Early termination with `end_invocation`

Set `invocation_context.end_invocation = True` inside any callback to stop the invocation after the current step. Useful for hard budget/safety gates:

```python
from google.adk.agents import LlmAgent
from google.adk.plugins import BasePlugin
from google.adk.agents.invocation_context import LlmCallsLimitExceededError

class SafetyPlugin(BasePlugin):
    """Terminates the invocation if a forbidden phrase is detected."""

    def __init__(self, forbidden: set[str]):
        super().__init__(name="safety")
        self._forbidden = forbidden

    async def before_model_callback(self, *, callback_context, llm_request):
        from google.genai import types

        # Inspect the last user message
        user_text = ""
        if callback_context.user_content and callback_context.user_content.parts:
            user_text = " ".join(
                p.text or "" for p in callback_context.user_content.parts
            ).lower()

        if any(f in user_text for f in self._forbidden):
            # Abort the whole invocation, not just skip this model call
            callback_context._invocation_context.end_invocation = True
            return types.Content.__class__  # won't be reached; invocation stops
        return None
```

### `LlmCallsLimitExceededError`

When `RunConfig.max_llm_calls` is exceeded, the runner raises `LlmCallsLimitExceededError`. Catch it at the call site if you want graceful handling:

```python
from google.adk.agents.invocation_context import LlmCallsLimitExceededError

try:
    async for event in runner.run_async(user_id=..., session_id=..., new_message=...):
        process(event)
except LlmCallsLimitExceededError as exc:
    print(f"Budget exceeded: {exc}")
```

## Patterns

### 1 — Dev loop with `run_debug`
```python
runner = InMemoryRunner(agent=agent)
events = await runner.run_debug(["Hi", "What's my name?"])
```
Uses fixed `user_id="debug_user_id"`, `session_id="debug_session_id"`. Reuse the same session id across calls to continue the conversation.

### 2 — Production with Vertex Agent Engine
`Runner(app=app, session_service=VertexAiSessionService(...), memory_service=VertexAiMemoryBankService(...))`. Combine with `ArtifactEngine`-backed GCS storage and `CloudTracePlugin` for full GCP integration.

### 3 — Local SQLite persistence
`DatabaseSessionService(db_url="sqlite+aiosqlite:///./adk.db")` plus `FileArtifactService(root_dir="./artifacts")`. Works offline; easy to ship in a Docker image.

### 4 — Rewinding bad turns
If an invocation went off the rails, `await runner.rewind_async(user_id=..., session_id=..., rewind_before_invocation_id=bad_id)` inverts the state and artifact deltas of events from that invocation forward. The session is left in its pre-invocation state; the user can retry.

### 5 — Event compaction for long chats
Configure `EventsCompactionConfig(compaction_interval=10, overlap_size=2, token_threshold=50_000, event_retention_size=50)`. The runner compacts old events into a summarised form after every 10 user invocations — combine with `RunConfig.get_session_config=GetSessionConfig(num_recent_events=50)` to limit fetch size.

## Gotchas

- Exactly one of `app=`, `agent=`, or `node=` on the `Runner` constructor. Supplying more raises `ValueError`.
- When using `agent=`, `app_name=` is **required**.
- `Runner(plugins=...)` is deprecated. Move plugins to `App(plugins=...)`.
- `auto_create_session=False` (the default) means missing sessions raise `SessionNotFoundError`. Callers should create sessions explicitly during signup/handshake.
- `DatabaseSessionService` requires `sqlalchemy>=2.0` — it's lazy-imported so the error only fires when you instantiate it.
- `RunConfig.save_input_blobs_as_artifacts` and `save_live_audio` are deprecated. Use `SaveFilesAsArtifactsPlugin` and `save_live_blob`.
- The sync `Runner.run()` spawns a background thread — safe for notebooks, not recommended for servers.
- App-name alignment: the runner warns if the agent's module path suggests a different app name (`agents/my_app/agent.py` → `my_app`). Set `app_name` to match or move your module.

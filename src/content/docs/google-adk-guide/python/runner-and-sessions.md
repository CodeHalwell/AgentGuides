---
title: "Runner, App, and Sessions"
description: "Runner / InMemoryRunner, the App container, session services, RunConfig, and state scopes."
framework: google-adk
language: python
sidebar:
  order: 50
---

Verified against google-adk==2.0.0b1 (`google/adk/runners.py`, `google/adk/apps/app.py`, `google/adk/sessions/`).

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

## `run_async` return semantics

`run_async` yields `Event` objects one at a time. Each event carries:

- `event.author` — agent name or `"user"`.
- `event.content` — the message content (`types.Content`).
- `event.actions` — state delta, artifact delta, `escalate`, `transfer_to_agent`, `skip_summarization`, etc.
- `event.partial` — `True` for streaming chunks; the final event is non-partial.
- `event.usage_metadata` — token counts, only on model events.
- `event.get_function_calls()` / `get_function_responses()` — helpers to peel function-call events.

Non-partial events are persisted via `session_service.append_event` before being yielded.

## Artifact service

Runners accept an optional `artifact_service=`. When configured, tools can call `tool_context.save_artifact("report.pdf", part)` and `load_artifact(...)`. Available services (`artifacts/__init__.py`):

| Service | Storage |
|---|---|
| `InMemoryArtifactService()` | Dict in memory |
| `FileArtifactService(root_path=...)` | Local filesystem |
| `GcsArtifactService(bucket_name=...)` | Google Cloud Storage |

See [memory-and-artifacts](./memory-and-artifacts/) for detailed semantics and versioning.

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
`DatabaseSessionService(db_url="sqlite+aiosqlite:///./adk.db")` plus `FileArtifactService(root_path="./artifacts")`. Works offline; easy to ship in a Docker image.

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

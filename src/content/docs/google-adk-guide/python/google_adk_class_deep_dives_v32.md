---
title: "Class deep dives — volume 32 (10 additional classes)"
description: "Source-verified deep dives into 10 additional google-adk 2.3.0 classes and utilities: TriggerRouter, ServiceRegistry, PerAgentDatabaseSessionService/PerAgentFileArtifactService, DotAdkFolder, _InvocationCostManager/RealtimeCacheEntry, ApiServerSpanExporter, A2aRemoteAgentConfig, LongRunningFunctions, ParsedArtifactUri, and find_context_parameter."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 32"
  order: 101
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source at
`<site-packages>/google/adk/` on
**google-adk == 2.3.0**. Path varies by environment; run `pip show google-adk` to find yours.
No documentation or blog posts were used as primary sources.
</Aside>

## 1 · `TriggerRouter` + `PubSubTriggerRequest` + `EventarcTriggerRequest` — event-driven agent invocations

**Module:** `google.adk.cli.trigger_routes`

`TriggerRouter` registers `/apps/{app_name}/trigger/pubsub` and `/apps/{app_name}/trigger/eventarc` on a FastAPI application, enabling ADK agents to process Pub/Sub push messages and Eventarc CloudEvents without pre-created sessions. Each request auto-creates an ephemeral session, runs the agent, and responds in the format the upstream service expects.

### Key implementation facts (verified from source)

- **Concurrency guard** — a single `asyncio.Semaphore(max_concurrent)` (default 10, overridable via `ADK_TRIGGER_MAX_CONCURRENT`) gates all concurrent agent invocations across both endpoints. Excess requests queue rather than fail.
- **Exponential backoff with jitter** — `_run_agent_with_retry` retries up to `max_retries` (default 3) times on 429/`RESOURCE_EXHAUSTED` errors. Delay formula: `min(base_delay * 2**attempt, max_delay) + jitter(0..delay*0.5)`. After exhaustion raises `TransientError` which maps to HTTP 500 so Pub/Sub / Eventarc will retry at the delivery level.
- **Dual CloudEvents modes** — the `/trigger/eventarc` endpoint handles both *structured content mode* (entire event in JSON body with a `data` key) and *binary content mode* (CE attributes in `ce-*` HTTP headers, body is a Pub/Sub wrapper `{"message": {...}}`) by inspecting `req.message`, `req.data`, then falling back to raw serialization.
- **Pub/Sub decode** — `PubSubMessage.data` is base64-encoded; the endpoint decodes it and attempts JSON parsing, falling back to raw string. The composed `message_text` contains both decoded `data` and `attributes`.
- **`TransientError`** — raised only after all retries are exhausted; signals the upstream service that the event is retryable, not a permanent failure.
- **`DEFAULT_TRIGGER_SOURCES = []`** — no trigger endpoints are registered by default; callers must explicitly opt in via `trigger_sources=["pubsub"]` or `["eventarc"]` to prevent accidental exposure.
- **`VALID_TRIGGER_SOURCES = ["pubsub", "eventarc"]`** — unknown sources are logged as warnings and silently dropped.
- **`_is_transient_error()`** — checks for `google.api_core.exceptions.ResourceExhausted`, `TooManyRequests`, and falls back to string matching on `"429"`, `"resource_exhausted"`, `"rate limit"`, `"quota"`.

### Example 1 — registering a Pub/Sub trigger endpoint

```python
import asyncio
from fastapi import FastAPI
from google.adk.cli.trigger_routes import TriggerRouter


# Minimal stub server object exposing session_service and get_runner_async
class MinimalServer:
    def __init__(self, runner, session_service):
        self.runner = runner
        self.session_service = session_service

    async def get_runner_async(self, app_name):
        return self.runner


# In production, use AdkWebServer from google.adk.cli.adk_web_server.
# Here we show the router registration pattern.
async def build_app(server) -> FastAPI:
    app = FastAPI()
    router = TriggerRouter(
        server,
        trigger_sources=["pubsub"],
        max_concurrent=5,         # lower for quota-limited environments
        max_retries=2,
        retry_base_delay=0.5,
        retry_max_delay=10.0,
    )
    router.register(app)
    return app


# The Pub/Sub push endpoint is now at:
#   POST /apps/{app_name}/trigger/pubsub
# Body: {"message": {"data": "<base64>", "attributes": {...}}, "subscription": "..."}
```

### Example 2 — sending a Pub/Sub push message manually

```python
import asyncio
import base64
import json
import httpx

# Construct a Pub/Sub push payload as Google Cloud would deliver it.
payload = {
    "data": base64.b64encode(json.dumps({"query": "What is the weather today?"}).encode()).decode(),
    "attributes": {"source": "weather_pipeline"},
    "messageId": "msg-001",
    "publishTime": "2026-06-30T10:00:00Z",
}
body = {
    "message": payload,
    "subscription": "projects/my-project/subscriptions/agent-sub",
}

async def call_pubsub_trigger():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/apps/my_agent/trigger/pubsub",
            json=body,
        )
        print(response.json())  # {"status": "success"} on success
        # HTTP 500 with {"detail": "Rate limit exceeded..."} triggers Pub/Sub retry

asyncio.run(call_pubsub_trigger())
```

### Example 3 — handling Eventarc CloudEvents in structured mode

```python
import httpx
import asyncio

# Eventarc structured content mode: all CloudEvent attributes in the body.
eventarc_body = {
    "specversion": "1.0",
    "type": "google.cloud.storage.object.v1.finalized",
    "source": "//storage.googleapis.com/projects/_/buckets/my-bucket",
    "id": "event-123",
    "time": "2026-06-30T10:00:00Z",
    "data": {
        "name": "uploads/report.pdf",
        "bucket": "my-bucket",
        "size": "102400",
    },
}

async def trigger_on_gcs_event():
    async with httpx.AsyncClient() as client:
        # source is used as user_id (slashes replaced with --)
        response = await client.post(
            "http://localhost:8080/apps/doc_processor/trigger/eventarc",
            json=eventarc_body,
        )
        assert response.json()["status"] == "success"

asyncio.run(trigger_on_gcs_event())
```

---

## 2 · `ServiceRegistry` + `ServiceFactory` + `load_services_module` — pluggable service URI registry

**Module:** `google.adk.cli.service_registry`

`ServiceRegistry` maps URI scheme prefixes to factory callables for session, artifact, memory, and A2A task-store services. The singleton returned by `get_service_registry()` ships with built-in schemes and lets you extend it via `services.py` or `services.yaml` in your agent directory.

### Key implementation facts (verified from source)

- **Four service type tables** — `_session_factories`, `_artifact_factories`, `_memory_factories`, `_task_store_factories` keyed by URI scheme string.
- **Built-in session schemes**: `memory://` → `InMemorySessionService`; `sqlite://` → `SqliteSessionService`; `agentengine://` → `VertexAiSessionService`; `postgresql://` and `mysql://` → `DatabaseSessionService`.
- **Built-in artifact schemes**: `memory://` → `InMemoryArtifactService`; `gs://` → `GcsArtifactService` (bucket from `netloc`); `file://` → `FileArtifactService` (path traversal check via `Path.resolve()`).
- **Built-in memory schemes**: `memory://` → `InMemoryMemoryService`; `rag://` → `VertexAiRagMemoryService`; `agentengine://` → `VertexAiMemoryBankService`.
- **A2A task-store schemes**: `memory://` → `InMemoryTaskStore`; `postgresql+asyncpg://`, `mysql+aiomysql://`, `sqlite+aiosqlite://` → `DatabaseTaskStore` (via SQLAlchemy async engine).
- **Dual registration**: `load_services_module()` first loads `services.yaml`/`services.yml`, then imports `services.py`; if the same scheme appears in both, the Python file wins.
- **YAML registration** requires `scheme`, `type` (`session`/`artifact`/`memory`/`task_store`), and `class` (dotted path); the factory calls `cls(uri=uri, **kwargs)`.
- **`SqliteSessionService` short-form** — `sqlite://` with no path (i.e., `sqlite://`) falls back to `InMemorySessionService`.
- **Short-form agent engine IDs** — `agentengine://<id>` without slashes loads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` from environment; full resource names skip the env lookup.

### Example 1 — registering a custom session service via `services.py`

```python
# my_agent/services.py
from google.adk.cli.service_registry import get_service_registry
from google.adk.sessions.in_memory_session_service import InMemorySessionService


def my_custom_factory(uri: str, **kwargs):
    # Parse any custom options from the URI query string here
    print(f"Creating custom session service for URI: {uri}")
    return InMemorySessionService()


# Register before the server resolves services
get_service_registry().register_session_service("myscheme", my_custom_factory)

# Now --session_db_uri myscheme://anything resolves to InMemorySessionService
```

### Example 2 — YAML-based service registration (`services.yaml`)

```yaml
# my_agent/services.yaml
services:
  - scheme: redis
    type: session
    class: myapp.storage.RedisSessionService  # must accept (uri, **kwargs)
  - scheme: s3
    type: artifact
    class: myapp.storage.S3ArtifactService
  - scheme: faiss
    type: memory
    class: myapp.memory.FaissMemoryService
```

```python
# Verify registration after startup
from google.adk.cli.service_registry import get_service_registry

registry = get_service_registry()
# create_session_service returns None for unknown schemes
service = registry.create_session_service("redis://localhost:6379/0")
print(type(service).__name__)  # RedisSessionService (if class is importable)
```

### Example 3 — using built-in URI schemes programmatically

```python
from google.adk.cli.service_registry import get_service_registry

registry = get_service_registry()

# SQLite session service (three slashes = absolute path)
session_svc = registry.create_session_service("sqlite:///myapp.db")
print(type(session_svc).__name__)  # SqliteSessionService

# In-memory session service (empty sqlite path)
mem_session_svc = registry.create_session_service("sqlite://")
print(type(mem_session_svc).__name__)  # InMemorySessionService

# GCS artifact service — bucket name comes from URI netloc
artifact_svc = registry.create_artifact_service("gs://my-artifact-bucket")
print(type(artifact_svc).__name__)  # GcsArtifactService

# Unknown scheme returns None — caller falls back to default
unknown = registry.create_session_service("mongo://localhost:27017")
print(unknown)  # None
```

---

## 3 · `PerAgentDatabaseSessionService` + `PerAgentFileArtifactService` — per-agent local storage routing

**Module:** `google.adk.cli.utils.local_storage`

These two classes route session and artifact operations to each agent's own `.adk` folder (`<agent_dir>/.adk/session.db` and `<agent_dir>/.adk/artifacts/`). They support an optional `app_name_to_dir` mapping for aliasing logical app names to on-disk directory names, and both implement backward-compatible legacy fallbacks.

### Key implementation facts (verified from source)

- **Lazy service creation** — `_get_service(app_name)` acquires `asyncio.Lock()` before creating the per-agent service to avoid duplicate creation under concurrent requests. Created services are cached in `self._services`.
- **Built-in agents** — app names starting with `"__"` (e.g. `"__adk_agent_builder__"`) route to a shared location at `agents_root/.adk/` via the `_BUILT_IN_SESSION_SERVICE_KEY` sentinel, avoiding path traversal into the per-agent tree.
- **`app_name_to_dir` mapping** — allows `logical_name → disk_dirname`; default is identity. Only the directory name is remapped; the logical `app_name` passed to the underlying service remains unchanged.
- **`PerAgentFileArtifactService` legacy fallback** — `_get_legacy_service()` returns a `FileArtifactService` pointed at `agents_root/.adk/artifacts/` (the pre-per-agent shared layout). It is used as a *read-only* fallback: `load_artifact`, `list_artifact_keys`, `list_versions`, and `list_artifact_versions` try the per-agent store first, then the legacy store. `save_artifact` never writes to the legacy store; `delete_artifact` deletes from both to prevent reappearance via the fallback.
- **`create_local_session_service(per_agent=True)`** — factory that either creates a single `SqliteSessionService` at `base_dir/.adk/session.db` (default) or a `PerAgentDatabaseSessionService` rooted at `base_dir`.

### Example 1 — multi-agent server with isolated session DBs

```python
import asyncio
from pathlib import Path
from google.adk.cli.utils.local_storage import create_local_session_service

# agents_root/
#   billing_agent/.adk/session.db
#   shipping_agent/.adk/session.db
agents_root = Path("./agents_root")
agents_root.mkdir(exist_ok=True)

session_service = create_local_session_service(
    base_dir=agents_root,
    per_agent=True,
)

async def main():
    # Each app_name gets its own SQLite file, created on first access.
    session = await session_service.create_session(
        app_name="billing_agent", user_id="u1"
    )
    print(session.id)
    # agents_root/billing_agent/.adk/session.db is created here

asyncio.run(main())
```

### Example 2 — aliasing logical app names to different directories

```python
import asyncio
from pathlib import Path
from google.adk.cli.utils.local_storage import (
    PerAgentDatabaseSessionService,
    PerAgentFileArtifactService,
)

# Map logical name used in code → actual on-disk agent folder
app_name_to_dir = {
    "prod_agent": "agent_v2",      # logical → disk alias
    "legacy_agent": "agent_v1",
}

session_service = PerAgentDatabaseSessionService(
    agents_root="./agents",
    app_name_to_dir=app_name_to_dir,
)
artifact_service = PerAgentFileArtifactService(
    agents_root="./agents",
    app_name_to_dir=app_name_to_dir,
)

async def main():
    # Session written to ./agents/agent_v2/.adk/session.db
    session = await session_service.create_session(
        app_name="prod_agent", user_id="u1"
    )
    print(session.app_name)   # "prod_agent" (logical name preserved)

asyncio.run(main())
```

### Example 3 — legacy artifact fallback in action

```python
import asyncio
from pathlib import Path
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.cli.utils.local_storage import PerAgentFileArtifactService
from google.genai import types

agents_root = Path("./agents")
agents_root.mkdir(exist_ok=True)

# Simulate legacy shared artifact store (pre-per-agent layout).
legacy_dir = agents_root / ".adk" / "artifacts"
legacy_dir.mkdir(parents=True, exist_ok=True)
legacy_service = FileArtifactService(root_dir=legacy_dir)

# Create per-agent artifact service (new layout).
per_agent_service = PerAgentFileArtifactService(agents_root=agents_root)

async def main():
    artifact_part = types.Part.from_bytes(data=b"report data", mime_type="text/plain")

    # Write to legacy location directly (simulates pre-migration data)
    version = await legacy_service.save_artifact(
        app_name="my_agent", user_id="u1", filename="report.txt",
        artifact=artifact_part,
    )

    # PerAgentFileArtifactService falls back to legacy store for reads
    loaded = await per_agent_service.load_artifact(
        app_name="my_agent", user_id="u1", filename="report.txt",
    )
    print(loaded is not None)  # True — found in legacy store

asyncio.run(main())
```

---

## 4 · `DotAdkFolder` + `dot_adk_folder_for_agent` — `.adk` folder lifecycle management

**Module:** `google.adk.cli.utils.dot_adk_folder`

`DotAdkFolder` encapsulates the layout of the `.adk` sub-folder inside each agent's working directory, providing path-safe, `cached_property`-backed accessors for its key locations.

### Key implementation facts (verified from source)

- **`dot_adk_dir`** — `cached_property` returning `agent_dir / ".adk"`. Directory is NOT created automatically; callers must call `.mkdir(parents=True, exist_ok=True)` themselves (e.g. `create_local_database_session_service` does this).
- **`artifacts_dir`** — `cached_property` returning `dot_adk_dir / "artifacts"`.
- **`session_db_path`** — `cached_property` returning `dot_adk_dir / "session.db"`.
- **`_resolve_agent_dir`** — resolves both `agents_root` and `agents_root / app_name` with `Path.resolve()` and then asserts `agent_dir.is_relative_to(agents_root_path)`, preventing path-traversal attacks via `app_name="../../../etc"`.
- **`dot_adk_folder_for_agent(agents_root, app_name)`** — convenience constructor that calls `_resolve_agent_dir` before constructing `DotAdkFolder`.

### Example 1 — exploring the `.adk` folder layout

```python
from pathlib import Path
from google.adk.cli.utils.dot_adk_folder import DotAdkFolder

folder = DotAdkFolder(agent_dir="./my_agent")

print(folder.dot_adk_dir)       # PosixPath('.../my_agent/.adk')
print(folder.artifacts_dir)     # PosixPath('.../my_agent/.adk/artifacts')
print(folder.session_db_path)   # PosixPath('.../my_agent/.adk/session.db')

# Create the directory tree before writing to it
folder.dot_adk_dir.mkdir(parents=True, exist_ok=True)
folder.artifacts_dir.mkdir(parents=True, exist_ok=True)
print(folder.dot_adk_dir.exists())  # True
```

### Example 2 — safe multi-agent path resolution

```python
from pathlib import Path
from google.adk.cli.utils.dot_adk_folder import dot_adk_folder_for_agent

agents_root = Path("./agents")
agents_root.mkdir(exist_ok=True)

# Valid — resolves safely inside agents_root
folder = dot_adk_folder_for_agent(agents_root=agents_root, app_name="billing_v2")
print(folder.session_db_path)  # .../agents/billing_v2/.adk/session.db

# Path-traversal attempt is blocked at construction time
try:
    bad_folder = dot_adk_folder_for_agent(
        agents_root=agents_root, app_name="../../../etc/passwd"
    )
except ValueError as exc:
    print(exc)  # Invalid app_name '...': resolves outside base directory
```

### Example 3 — initialising local storage from a `DotAdkFolder`

```python
import asyncio
from pathlib import Path
from google.adk.cli.utils.dot_adk_folder import DotAdkFolder
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.artifacts.file_artifact_service import FileArtifactService

async def bootstrap_agent_storage(agent_dir: str):
    folder = DotAdkFolder(agent_dir)

    # Ensure the .adk directory exists
    folder.dot_adk_dir.mkdir(parents=True, exist_ok=True)
    folder.artifacts_dir.mkdir(parents=True, exist_ok=True)

    session_service = SqliteSessionService(db_path=str(folder.session_db_path))
    artifact_service = FileArtifactService(root_dir=folder.artifacts_dir)

    # Initialise DB schema
    await session_service.create_session(
        app_name="my_agent", user_id="bootstrap_user"
    )

    print(f"Session DB: {folder.session_db_path}")
    print(f"Artifacts: {folder.artifacts_dir}")

asyncio.run(bootstrap_agent_storage("./demo_agent"))
```

---

## 5 · `_InvocationCostManager` + `LlmCallsLimitExceededError` + `RealtimeCacheEntry` — invocation budgeting and audio accumulation

**Module:** `google.adk.agents.invocation_context`

`_InvocationCostManager` tracks the number of LLM calls made within a single agent invocation and enforces `RunConfig.max_llm_calls`. `RealtimeCacheEntry` accumulates audio blob chunks from live sessions for caching before flushing.

### Key implementation facts (verified from source)

- **`_InvocationCostManager`** — a `BaseModel` subclass that stores `_number_of_llm_calls: int = 0` as a private attribute. The single method `increment_and_enforce_llm_calls_limit(run_config)` increments the counter *before* checking, so the limit is strictly enforced at the call that pushes over the threshold.
- **`LlmCallsLimitExceededError`** — raised by `increment_and_enforce_llm_calls_limit` only when `run_config.max_llm_calls > 0` AND the count exceeds the limit. `max_llm_calls <= 0` disables the check entirely.
- **`RealtimeCacheEntry`** — a `BaseModel` with `arbitrary_types_allowed=True` (needed for `types.Blob`), holding `role: str`, `data: types.Blob`, and `timestamp: float`. Used in live bidirectional sessions to buffer audio chunks for context-caching before they are flushed to the model.
- **`InvocationContext.increment_llm_call_count()`** — the public method callers use to tick the counter; it delegates to `_invocation_cost_manager.increment_and_enforce_llm_calls_limit(self.run_config)`. There is no `llm_calls_limit_exceeded` property on `InvocationContext`; the limit is enforced by the raised `LlmCallsLimitExceededError` which is caught upstream to send a final event to the client.

### Example 1 — enforcing a maximum LLM call budget

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.agents.run_config import RunConfig
from google.genai import types

# cap the agent to 2 LLM calls per invocation
agent = LlmAgent(
    name="budget_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a research assistant. Think step by step, calling tools "
        "and reasoning carefully before answering."
    ),
)

async def run_with_budget():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    config = RunConfig(max_llm_calls=2)  # only 2 LLM calls allowed
    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text("Summarise quantum computing in detail.")]),
        run_config=config,
    ):
        events.append(event)
    # The agent stops and returns a partial answer once the call budget is hit
    print(f"Received {len(events)} events")

asyncio.run(run_with_budget())
```

### Example 2 — `RealtimeCacheEntry` structure

```python
import time
from google.genai import types
from google.adk.agents.invocation_context import RealtimeCacheEntry

# Simulate two incoming audio blob chunks from a live session
chunk_a = RealtimeCacheEntry(
    role="user",
    data=types.Blob(data=b"\xff\xfe" + b"\x00" * 1024, mime_type="audio/pcm"),
    timestamp=time.time(),
)
chunk_b = RealtimeCacheEntry(
    role="model",
    data=types.Blob(data=b"\xff\xfe" + b"\x01" * 512, mime_type="audio/pcm"),
    timestamp=time.time(),
)

# Entries accumulate in InvocationContext.realtime_input_cache
cache = [chunk_a, chunk_b]
user_chunks = [c for c in cache if c.role == "user"]
model_chunks = [c for c in cache if c.role == "model"]
print(len(user_chunks), len(model_chunks))  # 1 1
print(cache[0].data.mime_type)              # audio/pcm
```

### Example 3 — inspecting cost metadata on `InvocationContext`

```python
from google.adk.agents.invocation_context import _InvocationCostManager

# Demonstrate _InvocationCostManager directly
manager = _InvocationCostManager()

# Simulated RunConfig with a limit of 3 calls
class FakeRunConfig:
    max_llm_calls = 3

for i in range(1, 4):
    manager.increment_and_enforce_llm_calls_limit(FakeRunConfig())
    print(f"LLM call #{i} succeeded")

# The 4th call triggers LlmCallsLimitExceededError
try:
    manager.increment_and_enforce_llm_calls_limit(FakeRunConfig())
except Exception as exc:
    print(f"Caught: {type(exc).__name__}: {exc}")
    # LlmCallsLimitExceededError: Max number of llm calls limit of `3` exceeded
```

---

## 6 · `ApiServerSpanExporter` + `InMemoryExporter` — OTel span exporters in the API server

**Module:** `google.adk.cli.api_server`

Two `SpanExporter` implementations used internally by the ADK API server to route OpenTelemetry spans to in-memory dictionaries without sending them to an external backend.

### Key implementation facts (verified from source)

- **`ApiServerSpanExporter`** — filters spans by name: only `"call_llm"`, `"send_data"`, and spans whose name starts with `"execute_tool"` are kept. Each matching span is converted to a `dict` of its attributes, augmented with `trace_id` and `span_id`, then stored under `trace_dict[event_id]` keyed by `"gcp.vertex.agent.event_id"`. Spans without `event_id` are silently dropped.
- **`InMemoryExporter`** — stores ALL spans in `self._spans`, but maintains a separate index `trace_dict[session_id] → [trace_id, ...]`. The session_id is looked up from `gcp.vertex.agent.session_id` then `gen_ai.conversation.id`. Duplicate `trace_id` values within a session are deduplicated. `get_finished_spans(session_id)` returns only spans whose `trace_id` is in that session's list.
- **`force_flush`** — both implementations return `True` synchronously; neither buffers data in a way that needs flushing.
- **`InMemoryExporter.clear()`** — clears `self._spans` but leaves `trace_dict` intact; re-index if needed after a clear.
- Both are passed as `SpanExporter` to the OTel SDK during `ApiServer` startup; the ADK UI uses them to render the event graph trace overlay.

### Example 1 — capturing `call_llm` spans from an API server run

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from google.adk.cli.api_server import ApiServerSpanExporter, InMemoryExporter

trace_dict = {}
event_id_index = {}

# Route structured event traces to event_id_index
event_exporter = ApiServerSpanExporter(trace_dict=event_id_index)

# Capture ALL spans indexed by session_id
session_exporter = InMemoryExporter(trace_dict=trace_dict)

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(event_exporter))
provider.add_span_processor(SimpleSpanProcessor(session_exporter))

from opentelemetry import trace
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("example")

with tracer.start_as_current_span("call_llm") as span:
    span.set_attribute("gcp.vertex.agent.event_id", "evt-001")
    span.set_attribute("gcp.vertex.agent.session_id", "sess-abc")
    span.set_attribute("gcp.vertex.agent.llm_request", '{"model":"gemini-2.0-flash"}')

print("event_id_index keys:", list(event_id_index.keys()))  # ['evt-001']
print("session_id trace count:", len(trace_dict.get("sess-abc", [])))  # 1
```

### Example 2 — retrieving finished spans for a session

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from google.adk.cli.api_server import InMemoryExporter

trace_dict = {}
exporter = InMemoryExporter(trace_dict=trace_dict)
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

from opentelemetry import trace
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("test")

SESSION_ID = "session-xyz"
for i in range(3):
    with tracer.start_as_current_span(f"execute_tool_{i}") as span:
        span.set_attribute("gcp.vertex.agent.session_id", SESSION_ID)
        span.set_attribute("gcp.vertex.agent.event_id", f"evt-{i:03d}")

finished = exporter.get_finished_spans(SESSION_ID)
print(f"Spans for session: {len(finished)}")  # 3
print([s.name for s in finished])              # ['execute_tool_0', 'execute_tool_1', 'execute_tool_2']
```

### Example 3 — `ApiServerSpanExporter` span name filtering

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from google.adk.cli.api_server import ApiServerSpanExporter

trace_dict = {}
exporter = ApiServerSpanExporter(trace_dict=trace_dict)
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

from opentelemetry import trace
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("filter_test")

# Only call_llm, send_data, execute_tool* are kept
spans_to_emit = [
    ("call_llm", "evt-1"),
    ("send_data", "evt-2"),
    ("execute_tool_search", "evt-3"),
    ("debug_span", "evt-4"),     # DROPPED — name doesn't match
    ("some_other", None),         # DROPPED — no event_id
]

for name, eid in spans_to_emit:
    with tracer.start_as_current_span(name) as span:
        if eid:
            span.set_attribute("gcp.vertex.agent.event_id", eid)

print(sorted(trace_dict.keys()))  # ['evt-1', 'evt-2', 'evt-3']
```

---

## 7 · `A2aRemoteAgentConfig` + `RequestInterceptor` + `ParametersConfig` — A2A remote agent configuration

**Module:** `google.adk.a2a.agent.config`

`A2aRemoteAgentConfig` is the configuration object passed to `RemoteA2aAgent` that controls how A2A messages, tasks, artifacts, and parts are converted to ADK events, and allows injecting `RequestInterceptor` hooks to intercept or abort requests.

### Key implementation facts (verified from source)

- **Five converter fields** — `a2a_message_converter`, `a2a_task_converter`, `a2a_status_update_converter`, `a2a_artifact_update_converter`, `a2a_part_converter` — all default to the standard converter functions from `google.adk.a2a.converters.to_adk_event` / `part_converter`.
- **`request_interceptors: list[RequestInterceptor] | None`** — list of hooks run in order before and after each request. Absence (`None`) skips all hooks.
- **`RequestInterceptor.before_request`** — async hook `(InvocationContext, A2AMessage, ParametersConfig) → (A2AMessage | Event, ParametersConfig)`. Returning an `Event` (not an `A2AMessage`) aborts the A2A call and returns that event directly to the caller without sending to the remote agent.
- **`RequestInterceptor.after_request`** — async hook `(InvocationContext, A2AEvent, Event) → Event | None`. Returning `None` suppresses the event so it is not delivered upstream.
- **`ParametersConfig`** — carries `request_metadata: dict | None` (passed as Pub/Sub message metadata) and `client_call_context: ClientCallContext | None` (passed to the A2A client middleware layer).
- **`__deepcopy__`** — custom implementation that copies callable fields by reference (not by value) to avoid pickling lambda/function objects, while deepcopying all non-callable fields.

### Example 1 — swapping the default part converter

```python
from google.adk.a2a.agent.config import A2aRemoteAgentConfig
from google.adk.a2a.converters.part_converter import convert_a2a_part_to_genai_part
from google.genai import types as genai_types


def custom_part_converter(a2a_part):
    """Add a prefix to all text parts from the remote agent."""
    parts = convert_a2a_part_to_genai_part(a2a_part)
    result = []
    for part in (parts if isinstance(parts, list) else [parts]):
        if part and part.text:
            result.append(genai_types.Part.from_text(f"[remote] {part.text}"))
        elif part:
            result.append(part)
    return result


config = A2aRemoteAgentConfig(a2a_part_converter=custom_part_converter)
# Use config when constructing RemoteA2aAgent
# agent = RemoteA2aAgent(agent_card=card, config=config)
print("Config created with custom part converter:", config.a2a_part_converter is custom_part_converter)
```

### Example 2 — adding request metadata via `ParametersConfig`

```python
from google.adk.a2a.agent.config import A2aRemoteAgentConfig, ParametersConfig, RequestInterceptor
from google.adk.agents.invocation_context import InvocationContext
from a2a.types import Message as A2AMessage


async def inject_correlation_id(
    ctx: InvocationContext,
    message: A2AMessage,
    params: ParametersConfig,
):
    """Inject a correlation ID into every outbound A2A request."""
    params.request_metadata = params.request_metadata or {}
    params.request_metadata["x-correlation-id"] = ctx.invocation_id
    return message, params


interceptor = RequestInterceptor(before_request=inject_correlation_id)
config = A2aRemoteAgentConfig(request_interceptors=[interceptor])
print("Interceptors registered:", len(config.request_interceptors))  # 1
```

### Example 3 — aborting a request from `before_request`

```python
from google.adk.a2a.agent.config import A2aRemoteAgentConfig, ParametersConfig, RequestInterceptor
from google.adk.events.event import Event
from google.genai import types
from a2a.types import Message as A2AMessage


async def rate_limit_interceptor(ctx, message: A2AMessage, params: ParametersConfig):
    """Return a synthetic Event to abort the remote call if quota is exceeded."""
    quota_exceeded = True  # replace with real quota check
    if quota_exceeded:
        abort_event = Event(
            invocation_id=ctx.invocation_id,
            author="system",
            content=types.Content(
                role="model",
                parts=[types.Part.from_text("Remote agent quota exceeded. Try again later.")],
            ),
        )
        return abort_event, params  # returning Event aborts the A2A call
    return message, params


config = A2aRemoteAgentConfig(
    request_interceptors=[RequestInterceptor(before_request=rate_limit_interceptor)]
)
print("Config ready with rate-limit interceptor")
```

---

## 8 · `LongRunningFunctions` + `handle_user_input` — A2A long-running function bridge

**Module:** `google.adk.a2a.converters.long_running_functions`

`LongRunningFunctions` tracks function call / response pairs that are designated as long-running in A2A task exchange, emitting a `TaskStatusUpdateEvent` when the remote agent pauses waiting for external input.

### Key implementation facts (verified from source)

- **Long-running detection** — `process_event()` copies the event (deep) and removes any `FunctionCall` parts whose `.id` appears in `event.long_running_tool_ids`. Matching parts are accumulated in `self._parts` and their IDs recorded in `self._long_running_tool_ids`. Partial events (`event.partial=True`) are not accumulated.
- **Response matching** — `FunctionResponse` parts whose `.id` is in `_long_running_tool_ids` are removed from the returned event copy and accumulated in `self._parts`, but the ID is **not** removed from `_long_running_tool_ids`. `has_long_running_function_calls()` therefore remains `True` after a response is processed; the caller (A2A executor) drives the lifecycle, not the tracker.
- **`TaskState` tracking** — `self._task_state` starts as `input_required`. In `_mark_long_running_function_call`, if the function call name matches `REQUEST_EUC_FUNCTION_CALL_NAME` (the end-user-credentials request), `_task_state` is set to `auth_required`; otherwise stays `input_required`. The last function call wins.
- **`create_long_running_function_call_event(task_id, context_id)`** — converts accumulated parts back to A2A parts via the injected `a2a_part_converter`, sets `A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY` on DataParts containing function call metadata, then wraps them in a `TaskStatusUpdateEvent` with `final=True`.
- **`handle_user_input(context)`** — guard function: if the current task is in `input_required` or `auth_required` state but the user's incoming message does NOT contain a `DataPart` typed as `FUNCTION_RESPONSE`, it emits a `TaskStatusUpdateEvent` re-asserting the same state to signal the user that a function response is still expected.
- **`has_long_running_function_calls()`** — returns `True` when `_long_running_tool_ids` is non-empty; used by the A2A executor to decide whether to suspend execution.

### Example 1 — tracking a long-running function call

```python
from google.adk.a2a.converters.long_running_functions import LongRunningFunctions
from google.adk.events.event import Event
from google.genai import types

tracker = LongRunningFunctions()

# Simulate an event with a long-running function call
fc_part = types.Part.from_function_call(name="run_batch_job", args={"dataset": "ds1"})
fc_part.function_call.id = "call-abc"

event = Event(
    invocation_id="inv-1",
    author="agent",
    content=types.Content(role="model", parts=[fc_part, types.Part.from_text("Thinking…")]),
    long_running_tool_ids={"call-abc"},
)

cleaned = tracker.process_event(event)

# The function call part is removed from cleaned; text part remains
print([p.text for p in cleaned.content.parts if p.text])  # ['Thinking…']
print(tracker.has_long_running_function_calls())           # True
```

### Example 2 — emitting a `TaskStatusUpdateEvent`

```python
from google.adk.a2a.converters.long_running_functions import LongRunningFunctions
from google.adk.events.event import Event
from google.genai import types

tracker = LongRunningFunctions()

# Build a minimal event with a long-running call
fc_part = types.Part.from_function_call(name="external_approval", args={"amount": 1000})
fc_part.function_call.id = "call-xyz"

event = Event(
    invocation_id="inv-2",
    author="agent",
    content=types.Content(role="model", parts=[fc_part]),
    long_running_tool_ids={"call-xyz"},
)
tracker.process_event(event)

# Produce the A2A status update event that pauses the task
status_event = tracker.create_long_running_function_call_event(
    task_id="task-001",
    context_id="ctx-001",
)
if status_event:
    print(status_event.status.state.value)   # "input_required"
    print(status_event.final)                # True
    print(len(status_event.status.message.parts))  # 1
```

### Example 3 — `handle_user_input` guards missing function responses

```python
from unittest.mock import MagicMock
from google.adk.a2a.converters.long_running_functions import handle_user_input
from a2a.types import TaskState, TaskStatus, Task, Message as A2AMessage, Role, TextPart, Part as A2APart
import datetime, uuid

# Simulate a task stuck in input_required with no function response in the user message
context = MagicMock()
context.task_id = "task-001"
context.context_id = "ctx-001"
context.current_task = Task(
    id="task-001",
    context_id="ctx-001",
    status=TaskStatus(
        state=TaskState.input_required,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ),
)
context.message = A2AMessage(
    message_id=str(uuid.uuid4()),
    role=Role.user,
    parts=[A2APart(root=TextPart(text="continue please"))],  # no function response
)

result = handle_user_input(context)
if result:
    print(result.status.state.value)  # "input_required" — reminder event emitted
    print(result.final)               # True
```

---

## 9 · `ParsedArtifactUri` + `parse_artifact_uri` + `get_artifact_uri` + `is_artifact_ref` — artifact URI codec

**Module:** `google.adk.artifacts.artifact_util`

This module provides the canonical encoder and decoder for ADK artifact URIs used across `ToolContext.save_artifact`, `InMemoryArtifactService`, `GcsArtifactService`, and the memory-and-artifacts guide patterns.

### Key implementation facts (verified from source)

- **Two URI forms** — session-scoped: `artifact://apps/{app}/users/{user}/sessions/{session}/artifacts/{filename}/versions/{version}`; user-scoped (no session): `artifact://apps/{app}/users/{user}/artifacts/{filename}/versions/{version}`. The difference is captured in `ParsedArtifactUri.session_id` being `str | None`.
- **`ParsedArtifactUri`** — a `NamedTuple` with fields `app_name`, `user_id`, `session_id`, `filename`, `version: int`. Version is parsed with `int()` from the regex match.
- **Regex patterns** — two compiled regexes: `_SESSION_SCOPED_ARTIFACT_URI_RE` and `_USER_SCOPED_ARTIFACT_URI_RE`, both applied with `fullmatch` to prevent partial matches.
- **`parse_artifact_uri(uri)`** — returns `None` for empty URIs, URIs not starting with `"artifact://"`, or URIs that match neither pattern. Session-scoped is tried first.
- **`get_artifact_uri(app_name, user_id, filename, version, session_id=None)`** — constructs the URI from parts; `session_id=None` produces user-scoped form.
- **`is_artifact_ref(part)`** — returns `True` if `part.file_data.file_uri` starts with `"artifact://"`, allowing downstream code to distinguish local artifact references from external file URIs.

### Example 1 — round-trip encode/decode

```python
from google.adk.artifacts.artifact_util import parse_artifact_uri, get_artifact_uri

# Encode a session-scoped artifact
uri = get_artifact_uri(
    app_name="my_app",
    user_id="user-123",
    filename="reports/monthly.pdf",
    version=2,
    session_id="sess-abc",
)
print(uri)
# artifact://apps/my_app/users/user-123/sessions/sess-abc/artifacts/reports/monthly.pdf/versions/2

parsed = parse_artifact_uri(uri)
print(parsed.app_name)   # my_app
print(parsed.user_id)    # user-123
print(parsed.session_id) # sess-abc
print(parsed.filename)   # reports/monthly.pdf
print(parsed.version)    # 2 (int)
```

### Example 2 — user-scoped (session-agnostic) artifact URIs

```python
from google.adk.artifacts.artifact_util import parse_artifact_uri, get_artifact_uri

# User-scoped — no session_id
user_uri = get_artifact_uri(
    app_name="profile_app",
    user_id="alice",
    filename="avatar.png",
    version=1,
    # session_id omitted → user-scoped
)
print(user_uri)
# artifact://apps/profile_app/users/alice/artifacts/avatar.png/versions/1

parsed = parse_artifact_uri(user_uri)
print(parsed.session_id)  # None — user-scoped form has no session component

# Invalid URIs return None
print(parse_artifact_uri("https://not-an-artifact"))  # None
print(parse_artifact_uri(""))                          # None
print(parse_artifact_uri("artifact://missing-parts"))  # None
```

### Example 3 — detecting artifact references in response parts

```python
from google.adk.artifacts.artifact_util import is_artifact_ref, get_artifact_uri
from google.genai import types

uri = get_artifact_uri("app", "u1", "diagram.png", version=0, session_id="s1")

# An artifact reference part uses file_data with the artifact:// URI
artifact_part = types.Part(
    file_data=types.FileData(file_uri=uri, mime_type="image/png")
)
plain_part = types.Part(
    file_data=types.FileData(file_uri="https://cdn.example.com/img.png", mime_type="image/png")
)
text_part = types.Part.from_text("Just text")

print(is_artifact_ref(artifact_part))  # True
print(is_artifact_ref(plain_part))     # False
print(is_artifact_ref(text_part))      # False

# Filter only artifact references from a list of parts
content = types.Content(role="model", parts=[artifact_part, plain_part, text_part])
refs = [p for p in content.parts if is_artifact_ref(p)]
print(len(refs))           # 1
print(refs[0].file_data.file_uri.startswith("artifact://"))  # True
```

---

## 10 · `find_context_parameter` + `Aclosing` — context introspection and async generator cleanup

**Module:** `google.adk.utils.context_utils`

Two utilities used extensively inside the ADK tool dispatch path: `find_context_parameter` discovers which parameter of a tool function receives the ADK `Context` (or `ToolContext`/`CallbackContext` subclass), and `Aclosing` is a re-exported alias for Python's `contextlib.aclosing`.

### Key implementation facts (verified from source)

- **`find_context_parameter(func)`** — decorated with `@functools.lru_cache(maxsize=1024)`, so repeated calls for the same function are O(1). Inspects `inspect.signature` and `typing.get_type_hints`; the latter resolves forward-reference string annotations. Falls back to direct parameter annotation inspection when `get_type_hints` raises.
- **`_is_context_type(annotation)`** — checks `annotation is Context` (identity check, not `isinstance`) to handle subclass aliasing. Also handles `Optional[Context]` and `Union` types via `get_origin` / `get_args`.
- **`Context` check is identity-based** — `ToolContext` and `CallbackContext` are aliased to `Context` at the `google.adk.agents.context` module level, so any of them match. A plain `dict` or arbitrary class does NOT match.
- **Return value** — the string parameter *name* if found, `None` if no matching parameter exists. The call dispatcher uses the returned name to inject the context into the correct keyword argument.
- **`Aclosing`** — defined as `Aclosing = aclosing` (a module-level alias). `contextlib.aclosing` is an async context manager that calls `aclose()` on the wrapped async generator even if the consumer breaks out of the loop early, preventing resource leaks. The alias exists for backward-compatibility; `from google.adk.utils.context_utils import Aclosing` is used throughout the runner.

### Example 1 — discovering the context parameter name

```python
from google.adk.utils.context_utils import find_context_parameter
from google.adk.tools.tool_context import ToolContext


def search_tool(query: str, ctx: ToolContext) -> str:
    return f"Searching for {query}"


def no_context_tool(query: str) -> str:
    return query


async def async_tool(ctx: ToolContext, limit: int = 10):
    pass


print(find_context_parameter(search_tool))      # "ctx"
print(find_context_parameter(no_context_tool))  # None
print(find_context_parameter(async_tool))       # "ctx"

# Results are cached — second call is instant
print(find_context_parameter(search_tool))      # "ctx"  (from cache)
```

### Example 2 — using `Aclosing` to guarantee async generator cleanup

```python
import asyncio
from google.adk.utils.context_utils import Aclosing
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


async def run_with_early_exit():
    agent = LlmAgent(
        name="streamer",
        model="gemini-2.0-flash",
        instruction="Count from 1 to 10 slowly.",
    )
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    msg = types.Content(role="user", parts=[types.Part.from_text("count")])

    # Aclosing ensures aclose() is called even when we break early.
    async with Aclosing(
        runner.run_async(user_id="u1", session_id=session.id, new_message=msg)
    ) as gen:
        async for event in gen:
            if event.content and event.content.parts:
                print("First response:", event.content.parts[0].text[:50])
                break  # early exit — aclose() still called automatically

asyncio.run(run_with_early_exit())
```

### Example 3 — writing a tool that receives `ToolContext`

```python
import asyncio
from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import InMemoryRunner
from google.adk.utils.context_utils import find_context_parameter


async def save_note(note: str, ctx: Optional[ToolContext] = None) -> str:
    """Saves a note to the current session state."""
    if ctx is not None:
        notes = ctx.state.get("notes", [])
        notes.append(note)
        ctx.state["notes"] = notes
    return f"Saved: {note}"


# Verify ADK will correctly inject the context
print(find_context_parameter(save_note))  # "ctx"

agent = LlmAgent(
    name="note_taker",
    model="gemini-2.0-flash",
    instruction="You are a note-taking assistant. Save notes when asked.",
    tools=[save_note],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="notes_app")
    session = await runner.session_service.create_session(
        app_name="notes_app", user_id="u1"
    )
    events = await runner.run_debug(
        "Save a note: buy groceries", user_id="u1", session_id=session.id
    )
    session = await runner.session_service.get_session(
        app_name="notes_app", user_id="u1", session_id=session.id
    )
    print(session.state.get("notes"))  # ['buy groceries']

asyncio.run(main())
```

---

## Summary table

| # | Class / function | Module | What it does |
|---|---|---|---|
| 1 | `TriggerRouter` + `PubSubTriggerRequest` + `EventarcTriggerRequest` | `google.adk.cli.trigger_routes` | Registers `/apps/{app_name}/trigger/pubsub` and `/apps/{app_name}/trigger/eventarc` FastAPI endpoints with semaphore concurrency control and exponential-backoff retry |
| 2 | `ServiceRegistry` + `ServiceFactory` + `load_services_module` | `google.adk.cli.service_registry` | URI-scheme–keyed factory registry for session, artifact, memory, and A2A task-store services; built-in schemes + YAML/Python extension |
| 3 | `PerAgentDatabaseSessionService` + `PerAgentFileArtifactService` | `google.adk.cli.utils.local_storage` | Routes session/artifact operations to per-agent `.adk` folders with async-lock lazy init and legacy-fallback reads |
| 4 | `DotAdkFolder` + `dot_adk_folder_for_agent` | `google.adk.cli.utils.dot_adk_folder` | Path-safe `.adk` folder accessor with `cached_property` for `session.db` and `artifacts/`; blocks path-traversal via `is_relative_to` guard |
| 5 | `_InvocationCostManager` + `LlmCallsLimitExceededError` + `RealtimeCacheEntry` | `google.adk.agents.invocation_context` | Per-invocation LLM call counter enforcing `RunConfig.max_llm_calls`; audio blob accumulator for live-session context caching |
| 6 | `ApiServerSpanExporter` + `InMemoryExporter` | `google.adk.cli.api_server` | OTel `SpanExporter` pair: event-id–keyed dispatch for UI event graph; session-id–keyed span retrieval for timeline replay |
| 7 | `A2aRemoteAgentConfig` + `RequestInterceptor` + `ParametersConfig` | `google.adk.a2a.agent.config` | Configures five converter callables for A2A↔ADK event mapping; `before_request` / `after_request` interceptor hooks; per-call metadata |
| 8 | `LongRunningFunctions` + `handle_user_input` | `google.adk.a2a.converters.long_running_functions` | Tracks long-running A2A function calls by ID, emits `TaskStatusUpdateEvent` on pause, sets `auth_required` state for EUC calls |
| 9 | `ParsedArtifactUri` + `parse_artifact_uri` + `get_artifact_uri` + `is_artifact_ref` | `google.adk.artifacts.artifact_util` | Two-form artifact URI codec (session-scoped / user-scoped) using compiled regexes; round-trip encode/decode + reference detection |
| 10 | `find_context_parameter` + `Aclosing` | `google.adk.utils.context_utils` | LRU-cached context-parameter introspector for tool dispatch; `Aclosing` async-generator cleanup alias for `contextlib.aclosing` |

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-06-30 | 2.3.0 | Initial publication of Vol. 32 |

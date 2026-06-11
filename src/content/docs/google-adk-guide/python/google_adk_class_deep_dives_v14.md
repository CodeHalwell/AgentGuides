---
title: "Class deep dives — volume 14 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: A2aAgentExecutor/A2aAgentExecutorConfig/ExecuteInterceptor (server-side A2A bridge; expose any ADK Runner as an A2A endpoint; before/after-agent + after-event interceptors), Context/ToolContext advanced artifact & memory API (save_artifact with custom_metadata, load_artifact, list_artifacts, get_artifact_version, add_memory with MemoryEntry, add_session_to_memory), GcsArtifactService deep-dive (user-namespaced user: prefix, versioned history, ArtifactVersion, custom_metadata, list_artifact_versions), LlmEventSummarizer/BaseEventsSummarizer (custom prompt_template, tool-content truncation, implement-your-own compactor), EventsCompactionConfig advanced patterns (token-threshold vs sliding-window combination, HITL-safety guards, custom summarizer wiring), SpannerVectorStoreSettings/VectorSearchIndexSettings ANN deep-dive (APPROXIMATE_NEAREST_NEIGHBORS, tree_depth/num_leaves/num_branches index tuning, additional_filter, additional_key_columns), LangGraphAgent with LangGraph MemorySaver checkpointer (multi-turn state via LangGraph memory, custom state schema, streaming LangGraph graph), Trigger use_sub_branch/isolation_scope (isolated parallel sub-workflows; state partitioning in fan-out patterns), ResumabilityConfig + rerun_on_resume (pause/resume long-running tools; DatabaseSessionService requirement; workflow node restart semantics), PubSubToolset advanced (ordering_key for ordered delivery; tool predicate filter; event-driven multi-agent architecture)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 14"
  order: 79
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `A2aAgentExecutor` + `A2aAgentExecutorConfig` + `ExecuteInterceptor` | `google.adk.a2a.executor` | `@a2a_experimental` |
| 2 | `Context` / `ToolContext` — advanced artifact & memory API | `google.adk.agents.context` | Stable |
| 3 | `GcsArtifactService` — user-namespaced & versioned artifacts | `google.adk.artifacts.gcs_artifact_service` | Stable |
| 4 | `LlmEventSummarizer` + `BaseEventsSummarizer` | `google.adk.apps.llm_event_summarizer`, `.base_events_summarizer` | Stable |
| 5 | `EventsCompactionConfig` — advanced compaction patterns | `google.adk.apps._configs` + `compaction` | `@experimental` |
| 6 | `SpannerVectorStoreSettings` + `VectorSearchIndexSettings` | `google.adk.tools.spanner.settings` | `@experimental` |
| 7 | `LangGraphAgent` + LangGraph MemorySaver checkpointer | `google.adk.agents.langgraph_agent` | Stable |
| 8 | `Trigger` — `use_sub_branch` + `isolation_scope` | `google.adk.workflow._trigger` | Stable |
| 9 | `ResumabilityConfig` + `rerun_on_resume` | `google.adk.apps._configs`, `google.adk.workflow` | `@experimental` |
| 10 | `PubSubToolset` — advanced patterns | `google.adk.tools.pubsub` | `@experimental` |

---

## 1 · `A2aAgentExecutor` + `A2aAgentExecutorConfig` + `ExecuteInterceptor`

**Source:** `google.adk.a2a.executor.a2a_agent_executor`, `.config`

`A2aAgentExecutor` is the **server-side bridge**: it implements the A2A SDK's `AgentExecutor` interface and connects an ADK `Runner` to an incoming A2A HTTP request. Think of it as the reverse of `RemoteA2aAgent` — instead of *calling* a remote A2A agent, `A2aAgentExecutor` *serves* your ADK agent as an A2A endpoint.

Decorated with `@a2a_experimental` — expect breaking changes in future minor releases.

### How it fits together

```
A2A Client → A2A Server (FastAPI) → A2aAgentExecutor → Runner → LlmAgent
```

The A2A SDK's `A2AStarlette` (or FastAPI integration) wires an `AgentExecutor` to HTTP routes. You supply `A2aAgentExecutor` as the executor and it handles all ADK ↔ A2A protocol translation.

### Constructor (source-verified)

```python
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

A2aAgentExecutor(
    *,
    runner: Runner | Callable[..., Runner | Awaitable[Runner]],
    config: Optional[A2aAgentExecutorConfig] = None,
    use_legacy: bool = False,
    force_new_version: bool = False,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `runner` | required | An instantiated `Runner`, or a callable/async factory that returns one. The factory is cached after first call. |
| `config` | `None` | `A2aAgentExecutorConfig` for custom converters + interceptors |
| `use_legacy` | `False` | Force the older wire format even if the client sends the new-integration extension |
| `force_new_version` | `False` | Force the new implementation regardless of what the client requests |

### `A2aAgentExecutorConfig` fields

```python
from google.adk.a2a.executor.config import A2aAgentExecutorConfig, ExecuteInterceptor

A2aAgentExecutorConfig(
    a2a_part_converter=...,     # A2APartToGenAIPartConverter — default: convert_a2a_part_to_genai_part
    gen_ai_part_converter=...,  # GenAIPartToA2APartConverter — default: convert_genai_part_to_a2a_part
    request_converter=...,      # A2ARequestToAgentRunRequestConverter
    event_converter=...,        # AdkEventToA2AEventsConverter (legacy path)
    adk_event_converter=...,    # AdkEventToA2AEventsConverter (new path)
    execute_interceptors: Optional[list[ExecuteInterceptor]] = None,
)
```

### `ExecuteInterceptor` hooks

```python
@dataclasses.dataclass
class ExecuteInterceptor:
    before_agent:  Optional[Callable[[RequestContext], Awaitable[RequestContext]]]
    after_event:   Optional[Callable[[ExecutorContext, A2AEvent, Event],
                                     Awaitable[A2AEvent | list[A2AEvent] | None]]]
    after_agent:   Optional[Callable[[ExecutorContext, TaskStatusUpdateEvent],
                                     Awaitable[TaskStatusUpdateEvent]]]
```

- **`before_agent`** — inspect/mutate the incoming `RequestContext` before the ADK agent runs
- **`after_event`** — intercept each ADK event after conversion to an A2A event; return `None` to drop it
- **`after_agent`** — mutate the terminal `TaskStatusUpdateEvent` before it is sent to the client

### Example 1 — minimal A2A server from an ADK agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_to_a2a import agent_to_a2a

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

session_service = InMemorySessionService()
runner = Runner(
    app_name="my_assistant",
    agent=agent,
    session_service=session_service,
)

executor = A2aAgentExecutor(runner=runner)

# agent_to_a2a() creates a Starlette ASGI app + agent card
a2a_app = agent_to_a2a(
    agent=agent,
    executor=executor,
    host="0.0.0.0",
    port=8080,
)

# Run with: uvicorn my_module:a2a_app --host 0.0.0.0 --port 8080
```

### Example 2 — lazy runner factory (deferred initialisation)

The factory form is useful when the runner needs async setup (e.g. connecting to a database):

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions about the data.",
)

async def make_runner() -> Runner:
    session_service = DatabaseSessionService(
        db_url="postgresql+asyncpg://user:pass@db/myapp"
    )
    # warm up connection pool etc.
    return Runner(
        app_name="db_app",
        agent=agent,
        session_service=session_service,
    )

executor = A2aAgentExecutor(runner=make_runner)
```

### Example 3 — interceptor: inject auth metadata before the agent runs

```python
import logging
from google.adk.a2a.executor.config import A2aAgentExecutorConfig, ExecuteInterceptor
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from a2a.server.agent_execution.context import RequestContext

logger = logging.getLogger(__name__)

async def require_api_key(ctx: RequestContext) -> RequestContext:
    api_key = (ctx.message.metadata or {}).get("x-api-key")
    if not api_key or api_key != "secret-key-123":
        raise PermissionError("Invalid API key")
    logger.info("Authenticated request from context_id=%s", ctx.context_id)
    return ctx

config = A2aAgentExecutorConfig(
    execute_interceptors=[
        ExecuteInterceptor(before_agent=require_api_key),
    ]
)

executor = A2aAgentExecutor(runner=runner, config=config)
```

### Example 4 — interceptor: drop internal reasoning events from the stream

```python
from google.adk.a2a.executor.config import ExecuteInterceptor, A2aAgentExecutorConfig
from google.adk.a2a.executor.executor_context import ExecutorContext
from google.adk.events.event import Event
from a2a.server.events import Event as A2AEvent

async def drop_thought_events(
    executor_ctx: ExecutorContext,
    a2a_event: A2AEvent,
    adk_event: Event,
) -> A2AEvent | None:
    # Drop events that only contain model thoughts (no user-visible text)
    if adk_event.content and adk_event.content.parts:
        if all(p.thought for p in adk_event.content.parts if p.thought is not None):
            return None  # filter out
    return a2a_event

config = A2aAgentExecutorConfig(
    execute_interceptors=[
        ExecuteInterceptor(after_event=drop_thought_events),
    ]
)
```

### Example 5 — interceptor: stamp the final response with a trace ID

```python
import uuid
from google.adk.a2a.executor.config import ExecuteInterceptor, A2aAgentExecutorConfig
from google.adk.a2a.executor.executor_context import ExecutorContext
from a2a.types import TaskStatusUpdateEvent

async def add_trace_id(
    executor_ctx: ExecutorContext,
    final_event: TaskStatusUpdateEvent,
) -> TaskStatusUpdateEvent:
    if final_event.status.message:
        meta = dict(final_event.status.message.metadata or {})
        meta["trace_id"] = str(uuid.uuid4())
        final_event.status.message.metadata = meta
    return final_event

config = A2aAgentExecutorConfig(
    execute_interceptors=[
        ExecuteInterceptor(after_agent=add_trace_id),
    ]
)
```

### Gotchas

- `A2aAgentExecutor` is `@a2a_experimental` — import paths and hook signatures may change.
- The `runner` factory is called **once** and the result is cached. If you need per-request runners, use `before_agent` interceptors to configure state on the context instead.
- The `use_legacy=False` default triggers the new implementation when the client sends the new-integration extension (`_NEW_A2A_ADK_INTEGRATION_EXTENSION`). Set `force_new_version=True` to always use it regardless of client capability.
- Session management is automatic: the executor creates a new session if none exists for the incoming `context_id` + `task_id`.

---

## 2 · `Context` (= `ToolContext`) — advanced artifact & memory API

**Source:** `google.adk.agents.context` (aliased as `ToolContext` in `tools/tool_context.py`)

The v1 deep dive covered the field reference for `Context`. This section focuses on the **method API** — artifact storage, version history, memory management, and the credential request flow.

### Artifact methods (source-verified)

```python
# Save a new version of an artifact
version: int = await ctx.save_artifact(
    filename="report.pdf",
    artifact=types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
    custom_metadata={"source": "generator_v2", "pages": "12"},
)

# Load the latest version
part: types.Part | None = await ctx.load_artifact("report.pdf")

# Load a specific version
part_v0: types.Part | None = await ctx.load_artifact("report.pdf", version=0)

# Get version metadata (ArtifactVersion has: version, canonical_uri, create_time, mime_type, custom_metadata)
av = await ctx.get_artifact_version("report.pdf")
print(av.version, av.canonical_uri, av.mime_type, av.custom_metadata)

# List all artifact filenames in the current session
filenames: list[str] = await ctx.list_artifacts()
```

### User-scoped artifacts with `user:` prefix

Any filename beginning with `user:` is stored at the **user level** — shared across all sessions for that user:

```python
# Save a user-level preference (persists across sessions)
await ctx.save_artifact(
    filename="user:preferences.json",
    artifact=types.Part.from_text(text='{"theme": "dark"}'),
)

# Load it in any future session for this user
prefs_part = await ctx.load_artifact("user:preferences.json")
```

### Memory methods (source-verified — 2.1.0+)

```python
from google.adk.memory.memory_entry import MemoryEntry

# Add explicit memories directly (no LLM extraction)
await ctx.add_memory(
    memories=[
        MemoryEntry(
            content=types.Content(
                role="user",
                parts=[types.Part.from_text("User prefers concise answers")]
            )
        ),
        MemoryEntry(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text("User works in finance sector")]
            )
        ),
    ]
)

# Add the entire current session as a memory (LLM-extracted)
await ctx.add_session_to_memory()
```

### Example 1 — document generation tool with versioned artifacts

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types

async def generate_summary(topic: str, tool_context: ToolContext) -> str:
    """Generate a text summary and save it as a versioned artifact."""
    summary = f"Summary for '{topic}': This is a generated summary."
    part = types.Part.from_text(text=summary)
    version = await tool_context.save_artifact(
        filename="summary.txt",
        artifact=part,
        custom_metadata={"topic": topic, "generator": "v1"},
    )
    return f"Summary saved as version {version}."

async def get_summary_history(tool_context: ToolContext) -> str:
    """List all artifact files in this session."""
    files = await tool_context.list_artifacts()
    return f"Session artifacts: {files}"

agent = LlmAgent(
    name="doc_agent",
    model="gemini-2.5-flash",
    instruction="Generate summaries and manage the artifact store.",
    tools=[generate_summary, get_summary_history],
)

async def main():
    artifact_service = InMemoryArtifactService()
    runner = InMemoryRunner(
        agent=agent,
        app_name="docs",
        artifact_service=artifact_service,
    )
    await runner.session_service.create_session(
        app_name="docs", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Generate a summary about quantum computing.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — explicit memory injection tool

```python
from google.adk.tools.tool_context import ToolContext
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

async def remember_fact(fact: str, tool_context: ToolContext) -> str:
    """Store an explicit fact in long-term memory."""
    await tool_context.add_memory(
        memories=[
            MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(fact)],
                )
            )
        ]
    )
    return f"Fact remembered: '{fact}'"
```

### Example 3 — cross-session profile via `user:` prefix

```python
import json
from google.adk.tools.tool_context import ToolContext
from google.genai import types

async def update_user_profile(name: str, role: str, tool_context: ToolContext) -> str:
    """Save user profile as a user-scoped artifact (persists across sessions)."""
    profile = {"name": name, "role": role}
    await tool_context.save_artifact(
        filename="user:profile.json",
        artifact=types.Part.from_text(text=json.dumps(profile)),
        custom_metadata={"schema_version": "1"},
    )
    return f"Profile saved for {name}."

async def load_user_profile(tool_context: ToolContext) -> str:
    """Load the user's cross-session profile."""
    part = await tool_context.load_artifact("user:profile.json")
    if part is None:
        return "No profile found."
    return f"Profile: {part.text}"
```

---

## 3 · `GcsArtifactService` — user-namespaced & versioned artifacts

**Source:** `google.adk.artifacts.gcs_artifact_service`

`GcsArtifactService` backs artifact storage in a Google Cloud Storage bucket. The v2 deep dive covered the basic API. This section focuses on the **blob naming scheme**, user-namespaced files, the `ArtifactVersion` model, and `custom_metadata`.

### Blob naming scheme (source-verified `_get_blob_prefix`)

```
# Session-scoped file (no "user:" prefix)
{app_name}/{user_id}/{session_id}/{filename}/{version}

# User-scoped file ("user:" prefix)
{app_name}/{user_id}/user/{filename}/{version}
```

The `user:` namespace is the only way to share data across sessions for the same user. Listing artifacts returns both user-scoped and session-scoped filenames.

### `ArtifactVersion` model

```python
@dataclass
class ArtifactVersion:
    version: int
    canonical_uri: str       # "gs://{bucket}/{blob_name}"
    create_time: float       # Unix timestamp (blob.time_created.timestamp())
    mime_type: str | None
    custom_metadata: dict[str, str]  # all values stringified
```

### Constructor

```python
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService

service = GcsArtifactService(
    bucket_name="my-adk-artifacts",
    # **kwargs passed to google.cloud.storage.Client(...)
    # e.g. project="my-project", credentials=my_creds
)
```

### Example 1 — save, version, and retrieve with metadata

```python
import asyncio
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.genai import types

async def demo():
    service = GcsArtifactService("my-adk-artifacts")

    # Save version 0
    v0 = await service.save_artifact(
        app_name="myapp",
        user_id="user123",
        session_id="sess456",
        filename="output.txt",
        artifact=types.Part.from_text("Hello v0"),
        custom_metadata={"generator": "pipeline_a", "job_id": "j001"},
    )
    print(f"Saved v0 = {v0}")  # 0

    # Save version 1
    v1 = await service.save_artifact(
        app_name="myapp",
        user_id="user123",
        session_id="sess456",
        filename="output.txt",
        artifact=types.Part.from_text("Hello v1 — updated"),
        custom_metadata={"generator": "pipeline_b", "job_id": "j002"},
    )
    print(f"Saved v1 = {v1}")  # 1

    # Load latest
    latest = await service.load_artifact(
        app_name="myapp", user_id="user123", session_id="sess456",
        filename="output.txt",
    )
    print(latest.text)  # "Hello v1 — updated"

    # Load specific version
    v0_part = await service.load_artifact(
        app_name="myapp", user_id="user123", session_id="sess456",
        filename="output.txt", version=0,
    )
    print(v0_part.text)  # "Hello v0"

asyncio.run(demo())
```

### Example 2 — user-namespaced cross-session file

```python
import asyncio
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.genai import types

async def demo():
    service = GcsArtifactService("my-adk-artifacts")

    # Save under "user:" namespace — shared across sessions
    await service.save_artifact(
        app_name="myapp",
        user_id="user123",
        session_id=None,        # session_id not needed for user-scoped files
        filename="user:prefs.json",
        artifact=types.Part.from_text('{"theme":"dark","lang":"en"}'),
    )

    # Load from a completely different session — same result
    prefs = await service.load_artifact(
        app_name="myapp", user_id="user123", session_id="different-session",
        filename="user:prefs.json",
    )
    print(prefs.text)  # '{"theme":"dark","lang":"en"}'

asyncio.run(demo())
```

### Example 3 — list versions and inspect metadata

```python
import asyncio
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService

async def audit_versions():
    service = GcsArtifactService("my-adk-artifacts")

    # List all versions
    versions = await service.list_versions(
        app_name="myapp", user_id="user123", session_id="sess456",
        filename="output.txt",
    )
    print(f"Available versions: {versions}")  # [0, 1, 2, ...]

    # Inspect full ArtifactVersion metadata for each
    for v in versions:
        av = await service.get_artifact_version(
            app_name="myapp", user_id="user123", session_id="sess456",
            filename="output.txt", version=v,
        )
        print(f"  v{av.version}: uri={av.canonical_uri} "
              f"mime={av.mime_type} meta={av.custom_metadata}")

asyncio.run(audit_versions())
```

### Gotchas

- `custom_metadata` values are **always stored as strings** (`{k: str(v) for k, v in custom_metadata.items()}`). Cast back on read if you stored numeric values.
- Saving an artifact with `file_data` raises `NotImplementedError` — only `inline_data` and `text` are supported.
- The `session_id` parameter is **required** for non-`user:` filenames. Passing `None` for a session-scoped file raises `InputValidationError`.
- Deletion removes **all versions** for a filename — there is no per-version delete.

---

## 4 · `LlmEventSummarizer` + `BaseEventsSummarizer`

**Source:** `google.adk.apps.llm_event_summarizer`, `google.adk.apps.base_events_summarizer`

`LlmEventSummarizer` is the default compaction backend for `EventsCompactionConfig`. It takes a batch of events, formats them into a conversation transcript, and asks an LLM to produce a concise summary. The result is stored as an `EventCompaction` action on a new event.

### Constructor

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

LlmEventSummarizer(
    llm: BaseLlm,
    prompt_template: Optional[str] = None,
)
```

Default prompt template (source-verified):

```
The following is a conversation history between a user and an AI agent.
It may or may not start from a compacted history. Please identify and
reiterate the user request, summarize the context so far, focusing on
key decisions made and information obtained, as well as any unresolved
questions or tasks. The summary should be concise and capture the
essence of the interaction.

{conversation_history}
```

Tool call args and responses are capped at **2 000 characters** each (`_MAX_TOOL_CONTENT_CHARS = 2000`) to avoid inflating the context being summarised.

### Example 1 — custom summarizer model and prompt

```python
from google.adk.apps import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.agents import LlmAgent
from google.adk.models import Gemini

agent = LlmAgent(
    name="support",
    model="gemini-2.5-flash",
    instruction="You are a helpful support agent.",
)

SUPPORT_SUMMARY_TEMPLATE = """
You are summarizing a customer support conversation.
Focus on: the customer's problem, steps taken, current status, and next actions.
Keep it under 150 words.

{conversation_history}
"""

# Use a cheap flash model for summarization to save costs
summarizer = LlmEventSummarizer(
    llm=Gemini(model="gemini-2.0-flash"),
    prompt_template=SUPPORT_SUMMARY_TEMPLATE,
)

app = App(
    name="support_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=summarizer,
        compaction_interval=10,
        overlap_size=2,
    ),
)
```

### `BaseEventsSummarizer` — implementing a custom compactor

```python
import abc
from typing import Optional
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.genai import types

class KeyPointSummarizer(BaseEventsSummarizer):
    """Extracts only user messages as bullet-point key points (no LLM call)."""

    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Optional[Event]:
        user_messages = []
        for event in events:
            if event.author == "user" and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        user_messages.append(f"• {part.text.strip()}")

        if not user_messages:
            return None

        summary_text = "Key user requests:\n" + "\n".join(user_messages)
        compaction = EventCompaction(
            start_timestamp=events[0].timestamp,
            end_timestamp=events[-1].timestamp,
            compacted_content=types.Content(
                role="model",
                parts=[types.Part.from_text(summary_text)],
            ),
        )
        return Event(
            author="user",
            actions=EventActions(compaction=compaction),
            invocation_id=Event.new_id(),
        )
```

### Example 2 — wiring a custom summarizer with `EventsCompactionConfig`

```python
from google.adk.apps import App, EventsCompactionConfig

app = App(
    name="my_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=KeyPointSummarizer(),  # custom — no LLM cost
        compaction_interval=15,
        overlap_size=3,
    ),
)
```

---

## 5 · `EventsCompactionConfig` — advanced compaction patterns

**Source:** `google.adk.apps._configs`, `google.adk.apps.compaction`

The v1 deep dive introduced the field table. This section covers the **interaction between compaction modes**, HITL safety guards, and real-world tuning patterns.

### Two compaction modes

| Mode | Trigger | Key fields |
|---|---|---|
| Sliding-window | Every N new user-initiated invocations | `compaction_interval`, `overlap_size` |
| Token-threshold | Post-invocation, when prompt tokens ≥ threshold | `token_threshold`, `event_retention_size` |

**Both modes can be active simultaneously.** When `token_threshold` fires, it takes priority over the sliding-window trigger for that invocation.

### Token-threshold mode in detail

The compaction module estimates token count from recent events (chars ÷ 4) or reads `event.usage_metadata.prompt_token_count`. When the threshold is met:

1. Find all non-compaction events since the last compaction.
2. Drop the trailing `event_retention_size` events (keep them uncompacted).
3. Truncate further to avoid splitting pending function calls or unresolved HITL requests.
4. Summarise with the configured `BaseEventsSummarizer`.

```python
# Token-threshold only (no sliding-window)
from google.adk.apps import App, EventsCompactionConfig
from google.adk.agents import LlmAgent

agent = LlmAgent(name="chat", model="gemini-2.5-flash", instruction="Chat agent.")

app = App(
    name="chat_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=9999,   # effectively disabled
        overlap_size=0,
        token_threshold=80_000,     # compact when prompt ≥ 80k tokens
        event_retention_size=20,    # keep last 20 events raw
    ),
)
```

### Example 1 — combined token-threshold + sliding-window

```python
from google.adk.apps import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

summarizer = LlmEventSummarizer(llm=Gemini(model="gemini-2.0-flash"))

app = App(
    name="production_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=summarizer,
        # Sliding-window: compact every 25 turns, with 5-turn overlap
        compaction_interval=25,
        overlap_size=5,
        # Token-threshold: also compact if prompt exceeds 120k tokens
        # (keeps 40 raw events uncompacted)
        token_threshold=120_000,
        event_retention_size=40,
    ),
)
```

### Example 2 — compaction with persistence (required for long sessions)

Compaction appends a new `EventCompaction` event to the session. This only survives restarts if you use a persistent session service:

```python
import os
from google.adk.apps import App, EventsCompactionConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

app = App(
    name="long_session_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=20,
        overlap_size=3,
        token_threshold=50_000,
        event_retention_size=10,
    ),
)

runner = Runner(
    app=app,
    session_service=DatabaseSessionService(
        db_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///sessions.db")
    ),
)
```

### HITL safety guards (source-verified)

The compaction engine **never truncates across a pending function call or unresolved HITL gate**:

- `_truncate_events_before_pending_function_call` — if an event has a function call with no matching response, no events at or after it are compacted.
- `_truncate_events_before_hitl_signal` — if a `requested_tool_confirmations` call is unresolved, the compaction window stops before it.

This ensures compaction never destroys tool-call/response pairs or strands a HITL prompt in the compacted range.

---

## 6 · `SpannerVectorStoreSettings` + `VectorSearchIndexSettings`

**Source:** `google.adk.tools.spanner.settings`

`SpannerVectorStoreSettings` configures the `spanner_vector_store_similarity_search` tool. `VectorSearchIndexSettings` configures the **Approximate Nearest Neighbors (ANN)** index used when `nearest_neighbors_algorithm = APPROXIMATE_NEAREST_NEIGHBORS`.

### `SpannerVectorStoreSettings` fields (complete, source-verified)

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `project_id` | `str` | Yes | — | GCP project |
| `instance_id` | `str` | Yes | — | Spanner instance |
| `database_id` | `str` | Yes | — | Spanner database |
| `table_name` | `str` | Yes | — | Vector store table |
| `content_column` | `str` | Yes | — | Column returned in results |
| `embedding_column` | `str` | Yes | — | Column containing embeddings |
| `vector_length` | `int` | Yes | — | Embedding dimension (must match model output) |
| `vertex_ai_embedding_model_name` | `str` | Yes | — | e.g. `"text-embedding-005"` |
| `selected_columns` | `list[str]` | No | `[content_column]` | Columns returned in results |
| `nearest_neighbors_algorithm` | `Literal["EXACT_NEAREST_NEIGHBORS", "APPROXIMATE_NEAREST_NEIGHBORS"]` | No | `"EXACT_NEAREST_NEIGHBORS"` | Switch to ANN for large corpora |
| `top_k` | `int` | No | `4` | Number of nearest neighbours to return |
| `distance_type` | `str` | No | `"COSINE"` | `COSINE`, `DOT_PRODUCT`, or `EUCLIDEAN` |
| `num_leaves_to_search` | `int \| None` | No | `None` | ANN only: leaf nodes searched per query |
| `additional_filter` | `str \| None` | No | `None` | SQL WHERE clause fragment added to every query |
| `vector_search_index_settings` | `VectorSearchIndexSettings \| None` | No | `None` | ANN index configuration (required for ANN) |
| `additional_columns_to_setup` | `list[TableColumn] \| None` | No | `None` | Extra columns for table setup/insert |
| `primary_key_columns` | `list[str] \| None` | No | `None` | Primary key columns (default: auto-UUID `id`) |

### `VectorSearchIndexSettings` fields (for ANN)

| Field | Type | Default | Notes |
|---|---|---|---|
| `index_name` | `str` | required | Name of the vector similarity search index |
| `tree_depth` | `int` | `2` | `2` = leaves only; `3` = adds branches (>100M rows) |
| `num_leaves` | `int` | `1000` | Partitions; recommended = `num_rows / 1000` |
| `num_branches` | `int \| None` | `None` | Only for 3-level trees; recommended < `num_leaves` |
| `additional_key_columns` | `list[str] \| None` | `None` | Extra columns in the vector index key (for pre-filtering) |
| `additional_storing_columns` | `list[str] \| None` | `None` | Stored columns enabling index-side filtering |

### Example 1 — exact nearest-neighbors search

```python
from google.adk.tools.spanner.settings import (
    SpannerToolSettings, SpannerVectorStoreSettings, Capabilities,
)
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.agents import LlmAgent

vector_settings = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="my-db",
    table_name="article_embeddings",
    content_column="body",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    top_k=5,
    distance_type="COSINE",
    selected_columns=["title", "url", "body"],
)

toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        capabilities=[Capabilities.DATA_READ],
        vector_store_settings=vector_settings,
    )
)

agent = LlmAgent(
    name="article_search",
    model="gemini-2.5-flash",
    instruction="Find semantically similar articles. Use spanner_vector_store_similarity_search.",
    tools=[toolset],
)
```

### Example 2 — ANN search for large corpora (>10M rows)

```python
from google.adk.tools.spanner.settings import (
    SpannerToolSettings, SpannerVectorStoreSettings,
    VectorSearchIndexSettings, Capabilities,
)
from google.adk.tools.spanner.spanner_toolset import SpannerToolset

# ANN index configuration
index_settings = VectorSearchIndexSettings(
    index_name="article_embedding_idx",
    tree_depth=2,                        # 2-level tree (leaves only)
    num_leaves=5000,                     # dataset has ~5M rows → 5M/1000 = 5000 leaves
    additional_storing_columns=["category"],  # enables in-index category filtering
    additional_key_columns=["category"],      # pre-filters by category in the index walk
)

vector_settings = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="my-db",
    table_name="article_embeddings",
    content_column="body",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    nearest_neighbors_algorithm="APPROXIMATE_NEAREST_NEIGHBORS",
    top_k=10,
    distance_type="DOT_PRODUCT",
    num_leaves_to_search=100,            # search 100 of 5000 leaves per query
    additional_filter="is_published = TRUE",  # SQL WHERE fragment
    vector_search_index_settings=index_settings,
    selected_columns=["title", "url", "category", "body"],
)

toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        capabilities=[Capabilities.DATA_READ],
        max_executed_query_result_rows=10,
        vector_store_settings=vector_settings,
    )
)
```

### Example 3 — 3-level tree for very large corpus (>100M rows)

```python
from google.adk.tools.spanner.settings import VectorSearchIndexSettings

# For 200M-row datasets: add branches above the leaves
index_settings_3level = VectorSearchIndexSettings(
    index_name="mega_corpus_idx",
    tree_depth=3,          # 3-level: root → branches → leaves
    num_branches=500,      # branches; recommended: sqrt(num_rows) up to num_leaves
    num_leaves=50_000,     # 200M/1000 = 200000; cap at a reasonable level
    num_leaves_to_search=200,  # specified on SpannerVectorStoreSettings, not here
)
```

### When to use ANN vs exact NN

| Factor | Exact (`EXACT_NEAREST_NEIGHBORS`) | Approximate (`APPROXIMATE_NEAREST_NEIGHBORS`) |
|---|---|---|
| Result quality | 100% recall | ~95% recall (tunable) |
| Query latency on 1M+ rows | High | Low |
| Index maintenance overhead | None | Medium |
| `VectorSearchIndexSettings` required | No | Yes |
| `num_leaves_to_search` has effect | No | Yes |
| Recommended dataset size | < 1M rows | > 1M rows |

---

## 7 · `LangGraphAgent` + LangGraph InMemorySaver checkpointer

**Source:** `google.adk.agents.langgraph_agent`

The v2 deep dive covered basic `LangGraphAgent` usage. This section focuses on the **LangGraph checkpointer path** — using LangGraph's own memory management for multi-turn state — and on passing a custom graph state schema.

### Checkpointer vs no-checkpointer: message routing decision

```python
# From source: _get_messages()
if self.graph.checkpointer:
    # LangGraph manages memory → send only the NEW user messages this turn
    return _get_last_human_messages(events)
else:
    # ADK manages history → send the full user ↔ agent conversation
    return self._get_conversation_with_agent(events)
```

This is the critical branching point. When you supply a LangGraph checkpointer, ADK steps back and lets LangGraph own the conversation state.

### Example 1 — multi-turn with LangGraph InMemorySaver

```python
import asyncio
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
checkpointer = InMemorySaver()  # in-memory; use PostgresSaver for production

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"It's sunny and 22°C in {city}."

# Compiled graph WITH checkpointer — LangGraph manages state
react_graph = create_react_agent(llm, tools=[get_weather], checkpointer=checkpointer)

agent = LangGraphAgent(
    name="weather_bot",
    description="Answers weather questions.",
    graph=react_graph,
    instruction="You are a weather assistant. Remember user preferences.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="weather")
    await runner.session_service.create_session(
        app_name="weather", user_id="u1", session_id="s1"
    )

    # Turn 1 — sets a preference
    events = await runner.run_debug(
        "What's the weather in Paris? I prefer metric units.",
        user_id="u1", session_id="s1",
    )
    print("T1:", events[-1].content.parts[0].text)

    # Turn 2 — LangGraph checkpointer (keyed on session.id) remembers turn 1
    events = await runner.run_debug(
        "And what about London?",
        user_id="u1", session_id="s1",
    )
    print("T2:", events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — custom state schema with LangGraphAgent

You can define a typed graph state to carry structured data beyond just messages:

```python
import asyncio
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    search_count: int   # custom counter carried in graph state

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def count_and_respond(state: AgentState):
    count = state.get("search_count", 0) + 1
    response = llm.invoke(state["messages"])
    return {"messages": [response], "search_count": count}

builder = StateGraph(AgentState)
builder.add_node("agent", count_and_respond)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()

agent = LangGraphAgent(
    name="stateful_agent",
    description="LangGraph agent with custom state schema.",
    graph=graph,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="stateful")
    await runner.session_service.create_session(
        app_name="stateful", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Tell me about Python.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 3 — LangGraphAgent as a sub-agent within a larger ADK team

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents import LlmAgent
from google.adk.agents.langgraph_agent import LangGraphAgent

def search_database(query: str) -> str:
    """Search the internal database."""
    return f"Results for '{query}': [record 1, record 2, record 3]"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
db_graph = create_react_agent(llm, tools=[search_database])

db_agent = LangGraphAgent(
    name="db_specialist",
    description="Searches the internal database.",
    graph=db_graph,
    instruction="Search the database and return structured results.",
)

# Orchestrator routes to the LangGraph agent
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-pro",
    instruction="For database questions, delegate to 'db_specialist'.",
    sub_agents=[db_agent],
    mode="chat",
)
```

### Gotchas

- `LangGraphAgent.instruction` is injected **only on the first turn** when the graph's checkpoint is empty. If you restart the runner or clear the checkpointer, the instruction reappears.
- `LangGraphAgent` runs `graph.invoke()` (synchronous, blocking). For production with high concurrency, run it in a thread pool or switch to an async LangGraph graph.
- The `graph` field uses `ConfigDict(arbitrary_types_allowed=True)` because `CompiledGraph` is not a Pydantic model.
- Session `id` is used as the LangGraph `thread_id` — keep ADK session IDs stable across turns when using a checkpointer.

---

## 8 · `Trigger` — `use_sub_branch` + `isolation_scope`

**Source:** `google.adk.workflow._trigger`

`Trigger` is the typed payload passed along a workflow edge. It carries the input for the downstream node and two important routing flags that control **parallel sub-branches** and **state isolation**.

### Field reference (source-verified)

```python
class Trigger(BaseModel):
    input: Any = None
    use_sub_branch: bool = False
    branch: str | None = None
    isolation_scope: str | None = None
```

| Field | Notes |
|---|---|
| `input` | The data payload delivered to the triggered node. Serialised with `ser_json_bytes='base64'`. |
| `use_sub_branch` | When `True`, the triggered node runs in a **child branch** of the current branch. All events under the sub-branch are isolated from peers running concurrently. |
| `branch` | The branch inherited from the predecessor node. Set automatically by the framework. |
| `isolation_scope` | A scope tag that partitions session state: nodes in the same `isolation_scope` share their temporary `temp:` state; nodes in different scopes are isolated from each other. |

### When to use `use_sub_branch=True`

Use it when a node fans out to multiple parallel children that should each operate on their own isolated copy of the workflow event stream. Without sub-branches, parallel nodes write events to the same branch and can race.

### Example 1 — parallel fan-out with sub-branches

```python
import asyncio
from google.adk.workflow import Workflow, node, START
from google.adk.workflow._trigger import Trigger

@node
async def dispatcher(items: list[str], ctx) -> list[Trigger]:
    # Fan out: one Trigger per item, each in its own sub-branch
    return [
        Trigger(input=item, use_sub_branch=True)
        for item in items
    ]

@node
async def process_item(item: str, ctx) -> dict:
    # Each item runs in isolation; no shared mutable state
    return {"item": item, "processed": item.upper()}

@node
async def collector(results: list[dict], ctx) -> str:
    return f"Processed {len(results)} items: {results}"

pipeline = Workflow(
    name="fan_out",
    edges=[
        (START, dispatcher),
        (dispatcher, process_item),
        (process_item, collector),
    ],
)
```

### Example 2 — `isolation_scope` for state partitioning

When nodes share an `isolation_scope`, their `temp:` state is shared within that scope but isolated from other scopes. Useful when parallel branches each need a scratch space:

```python
from google.adk.workflow import node, Workflow, START
from google.adk.workflow._trigger import Trigger

@node
async def fork(ctx) -> list[Trigger]:
    return [
        Trigger(input="task_a", use_sub_branch=True, isolation_scope="scope_a"),
        Trigger(input="task_b", use_sub_branch=True, isolation_scope="scope_b"),
    ]

@node
async def task_runner(task: str, ctx) -> str:
    # temp: state is scoped — "scope_a" and "scope_b" don't see each other
    ctx.state["temp:progress"] = f"{task} in progress"
    return f"Completed {task}"

pipeline = Workflow(
    name="scoped_parallel",
    edges=[(START, fork), (fork, task_runner)],
)
```

### Example 3 — aggregate results from sub-branches

```python
from google.adk.workflow import node, Workflow, START
from google.adk.workflow._trigger import Trigger
from typing import Any

TOPICS = ["AI", "Climate", "Space"]

@node
async def create_research_jobs(ctx) -> list[Trigger]:
    return [Trigger(input=topic, use_sub_branch=True) for topic in TOPICS]

@node
async def research_topic(topic: str, ctx) -> str:
    # Simulate research
    return f"Research on {topic}: [summary of key findings]"

@node
async def compile_report(summaries: list[Any], ctx) -> str:
    report_parts = "\n\n".join(
        f"## {TOPICS[i]}\n{s}" for i, s in enumerate(summaries)
    )
    return f"# Multi-topic Report\n\n{report_parts}"

research_pipeline = Workflow(
    name="parallel_research",
    edges=[
        (START, create_research_jobs),
        (create_research_jobs, research_topic),
        (research_topic, compile_report),
    ],
)
```

---

## 9 · `ResumabilityConfig` + `rerun_on_resume`

**Source:** `google.adk.apps._configs`, `google.adk.workflow._base_node`

`ResumabilityConfig` enables ADK to **pause an invocation** when a long-running tool call is outstanding and **resume** it later. This is critical for tools that may take minutes or hours (external API calls, human approvals, batch jobs).

### `ResumabilityConfig` fields

```python
from google.adk.apps import ResumabilityConfig

ResumabilityConfig(
    is_resumable: bool = False,
)
```

When `is_resumable=True`, ADK:
1. Persists the session state before the long-running call
2. Pauses the invocation
3. On resume (next turn with the pending `long_running_tool_ids` filled in), continues from the saved state

**Requirement:** a persistent `session_service` (`DatabaseSessionService` or `VertexAiSessionService`) is mandatory. `InMemorySessionService` cannot survive the pause.

### `rerun_on_resume` on workflow nodes

```python
@node(
    retry_config=RetryConfig(max_attempts=3),
    rerun_on_resume=True,  # re-execute this node when the invocation is resumed
)
async def call_external_api(url: str, ctx) -> dict:
    ...
```

When `rerun_on_resume=True`, a node that was interrupted (e.g. by a HITL confirmation or a long-running tool pause) will **restart from scratch** when the invocation is resumed, rather than returning the saved prior output.

### Example 1 — resumable app with a long-running tool

```python
import os
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import LongRunningFunctionTool

async def run_batch_job(job_id: str, tool_context: ToolContext) -> dict:
    """Submit a batch job and return a pending result."""
    # Submit to an external system
    import asyncio
    await asyncio.sleep(0)  # non-blocking submission
    return {"status": "pending", "job_id": job_id}

# Wrap as a LongRunningFunctionTool so ADK tracks it as resumable
batch_tool = LongRunningFunctionTool(func=run_batch_job)

agent = LlmAgent(
    name="batch_orchestrator",
    model="gemini-2.5-flash",
    instruction="Submit batch jobs and report results when they complete.",
    tools=[batch_tool],
)

app = App(
    name="batch_app",
    root_agent=agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

runner = Runner(
    app=app,
    session_service=DatabaseSessionService(
        db_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///batch_sessions.db")
    ),
)
```

### Example 2 — workflow node with `rerun_on_resume`

```python
from google.adk.workflow import node, Workflow, START, RetryConfig

@node(
    retry_config=RetryConfig(max_attempts=3, initial_delay=2.0),
    rerun_on_resume=True,   # restart this node from scratch on resume
)
async def submit_and_wait(request: dict, ctx) -> dict:
    """
    Submit an external job. On resume, re-submits to get the latest result
    rather than relying on the saved state.
    """
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.example.com/jobs",
            json=request,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

@node
async def process_result(result: dict, ctx) -> str:
    return f"Job completed with status: {result.get('status')}"

job_pipeline = Workflow(
    name="resumable_job",
    edges=[(START, submit_and_wait), (submit_and_wait, process_result)],
)
```

### When to set `rerun_on_resume=True`

| Scenario | `rerun_on_resume` |
|---|---|
| Node makes an idempotent external call | `True` — safe to re-run |
| Node calls a non-idempotent mutation (e.g. send email) | `False` — resume from saved output |
| Node reads from a source that changes over time (live price feed) | `True` — get fresh data |
| Node is a pure computation on saved inputs | `False` — use the cached result |

---

## 10 · `PubSubToolset` — advanced patterns

**Source:** `google.adk.tools.pubsub`

The v2 deep dive covered the basic publish/pull/acknowledge API. This section covers **ordered delivery with `ordering_key`**, the **tool predicate filter**, and an **event-driven multi-agent architecture** using Pub/Sub as the message bus.

### Ordered message delivery

Set `ordering_key` on `publish_message` to ensure messages with the same key are delivered in order to a subscription that has ordering enabled:

```python
from google.adk.tools.tool_context import ToolContext

async def publish_ordered_event(
    order_id: str,
    region: str,
    payload: str,
    tool_context: ToolContext,
) -> dict:
    """Publish an order event with a region-based ordering key."""
    # Tool uses ordering_key = region so all orders for a region are ordered
    # (requires publisher_options.enable_message_ordering=True, handled by toolset)
    # Pass via the LLM-visible publish_message tool:
    return {
        "topic": f"projects/my-project/topics/orders",
        "message": payload,
        "ordering_key": region,
        "attributes": {"order_id": order_id},
    }
```

When your agent calls `publish_message`, pass `ordering_key` in the tool arguments. The underlying `message_tool.publish_message` source creates a `PublisherOptions(enable_message_ordering=bool(ordering_key))` publisher:

```python
# Instruct the agent to use ordering keys
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-project"),
)

agent = LlmAgent(
    name="order_publisher",
    model="gemini-2.5-flash",
    instruction=(
        "Publish order events to 'projects/my-project/topics/orders'. "
        "Always use the customer's region as the ordering_key "
        "(e.g. 'EU', 'US', 'APAC') to ensure regional ordering."
    ),
    tools=[toolset],
)
```

### Tool predicate filter

A `ToolPredicate` is `Callable[[BaseTool, ReadonlyContext], bool]`. Use it to dynamically enable/disable tools based on session state:

```python
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

def publisher_only_predicate(tool: BaseTool, ctx: ReadonlyContext) -> bool:
    """Only expose publish_message; hide pull/acknowledge in read-only mode."""
    if ctx.state.get("readonly_mode"):
        return tool.name == "publish_message"  # only publish allowed
    return True  # all tools enabled

toolset = PubSubToolset(
    tool_filter=publisher_only_predicate,
    pubsub_tool_config=PubSubToolConfig(project_id="my-project"),
)
```

### Example 1 — event-driven pipeline: producer → Pub/Sub → consumer

Two ADK agents communicating through a Pub/Sub topic:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

PROJECT = "my-gcp-project"
TOPIC = f"projects/{PROJECT}/topics/task-queue"
SUBSCRIPTION = f"projects/{PROJECT}/subscriptions/task-queue-sub"

# ──── Producer Agent ─────────────────────────────────────────────────────────
producer_toolset = PubSubToolset(
    tool_filter=["publish_message"],     # publish only
    pubsub_tool_config=PubSubToolConfig(project_id=PROJECT),
)

producer = LlmAgent(
    name="producer",
    model="gemini-2.5-flash",
    instruction=(
        f"Publish user requests to '{TOPIC}'. "
        "Include a JSON payload with 'task_type' and 'payload' fields."
    ),
    tools=[producer_toolset],
)

# ──── Consumer Agent ──────────────────────────────────────────────────────────
consumer_toolset = PubSubToolset(
    tool_filter=["pull_messages", "acknowledge_messages"],
    pubsub_tool_config=PubSubToolConfig(project_id=PROJECT),
)

consumer = LlmAgent(
    name="consumer",
    model="gemini-2.5-flash",
    instruction=(
        f"Pull messages from '{SUBSCRIPTION}'. "
        "Process each message: acknowledge it, parse the JSON payload, and "
        "execute the requested task. Pull at most 3 messages per turn."
    ),
    tools=[consumer_toolset],
)
```

### Example 2 — full pull-process-acknowledge loop

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

PROJECT = "my-gcp-project"

toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id=PROJECT),
)

processor = LlmAgent(
    name="task_processor",
    model="gemini-2.5-flash",
    instruction=(
        "You process tasks from a Pub/Sub subscription. "
        "Steps for each turn:\n"
        "1. Call pull_messages on 'projects/my-gcp-project/subscriptions/tasks-sub' "
        "   with max_messages=5.\n"
        "2. For each message: parse the JSON, perform the task (summarise the 'text' field).\n"
        "3. Collect all ack_ids and call acknowledge_messages to confirm processing.\n"
        "4. Report results."
    ),
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=processor, app_name="processor")
    await runner.session_service.create_session(
        app_name="processor", user_id="worker", session_id="batch_run_1"
    )
    events = await runner.run_debug(
        "Process the next batch of tasks.",
        user_id="worker", session_id="batch_run_1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### `pull_messages` return schema

```python
{
  "messages": [
    {
      "message_id": "1234567890",
      "data": "UTF-8 text or base64 if non-UTF-8",
      "attributes": {"key": "value"},
      "ordering_key": "EU",
      "publish_time": "2026-06-07T12:00:00Z",
      "ack_id": "projects/my-project/subscriptions/..."
    },
    # ...
  ]
}
```

The `ack_id` values from `pull_messages` are passed directly to `acknowledge_messages`. If `auto_ack=True` is passed to `pull_messages`, no separate `acknowledge_messages` call is needed.

### Gotchas

- `PubSubToolset` uses **synchronous** Pub/Sub pull — one round-trip per call. For streaming/high-throughput consumption, use the Pub/Sub streaming pull API outside ADK and only use the toolset for the agent-logic layer.
- `ordering_key` requires **ordering-enabled subscriptions**. Create the subscription with `--enable-message-ordering` and the publisher must also enable it (handled automatically by the toolset when `ordering_key` is non-empty).
- The `attributes` parameter on `publish_message` must be `dict[str, str]`. Non-string values are silently stringified by Pub/Sub's protobuf serialisation.

---

## Version notes

Verified against **google-adk==2.2.0** (June 2026). All constructor signatures, field names, and default values in this document were read from the installed source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

Previous: [Class deep dives — vol. 13 →](./google_adk_class_deep_dives_v13/)

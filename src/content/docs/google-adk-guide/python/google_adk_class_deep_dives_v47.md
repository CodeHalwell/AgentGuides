---
title: "Class deep dives — volume 47 (LiveRequestQueue/LiveRequest, ContextCacheConfig, RunConfig, StreamingMode, ToolThreadPoolConfig, PubSubToolset, SpannerToolset, Workflow, RetryConfig, CachePerformanceAnalyzer)"
description: "10 source-verified deep dives for google-adk 2.6.3: LiveRequestQueue + LiveRequest (bidirectional streaming queue for live agents; priority ordering activity_start > activity_end > blob > content; state_delta always applied; partial flag for incomplete turns; close() sentinel), ContextCacheConfig (@experimental AGENT_CONFIG; cache_intervals/ttl_seconds/min_tokens; Gemini model-specific token minimums 2048/4096; ttl_string property; create_http_options timeout guard), RunConfig (comprehensive runtime config; speech/http/labels/modalities; session_resumption/history_config/context_window_compression; model_input_context per-turn transient context; include_thoughts_from_other_agents; max_llm_calls guard), StreamingMode (NONE/SSE/BIDI enum; SSE duplicate-text handling patterns; partial=True text vs function-call discrimination; PROGRESSIVE_SSE_STREAMING interaction), ToolThreadPoolConfig (max_workers field ge=1; async tools run in new event loop inside thread; GIL limitations for pure-Python CPU-bound code; cancellation semantics), PubSubToolset (@experimental PUBSUB_TOOLSET; publish_message/pull_messages/acknowledge_messages trio; ordering_key + enable_message_ordering; auto_ack convenience flag; tool_filter predicate or name list), SpannerToolset (@experimental SPANNER_TOOLSET; 8 tools; Capabilities.DATA_READ gate; SpannerToolSettings max_executed_query_result_rows/query_result_mode/database_role; SpannerVectorStoreSettings EXACT/APPROXIMATE_NEAREST_NEIGHBORS; VectorSearchIndexSettings tree_depth/num_leaves), Workflow (BaseNode graph orchestration loop; edges→Graph.from_edge_items; max_concurrency for static nodes; rerun_on_resume; _LoopState in-memory transient state; _validate_state_schema FunctionNode param check), RetryConfig (max_attempts/initial_delay/max_delay/backoff_factor/jitter; exceptions list accepts str or type[BaseException]; _normalize_exceptions validator; None means retry on all exceptions), CachePerformanceAnalyzer (@experimental; analyze_agent_cache_performance metrics: cache_hit_ratio_percent/cache_utilization_ratio_percent/avg_cached_tokens_per_request/cache_refreshes; event-history traversal; CacheMetadata per event)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 47"
  order: 116
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.6.3**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `LiveRequest` + `LiveRequestQueue` — bidirectional streaming queue for live agents

**Sources:** `google/adk/agents/live_request_queue.py`

### Why it matters

`LiveRequestQueue` is the bridge between your application code and an
ADK live agent that consumes the Gemini Live (bidirectional-streaming)
API via `Runner.run_live()`. It serialises all the ways you can feed
the model — turn-by-turn text, raw audio/video blobs, activity
signals, and real-time state updates — into a single asyncio queue that
the runner drains.

### Internals

```python
class LiveRequest(BaseModel):
    model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')

    # Priority order (highest first):
    # activity_start > activity_end > blob > content
    # state_delta is always applied regardless of other fields.

    content: Optional[types.Content] = None    # turn-by-turn text
    blob: Optional[types.Blob] = None          # realtime audio/video
    activity_start: Optional[types.ActivityStart] = None
    activity_end: Optional[types.ActivityEnd] = None
    close: bool = False                        # sentinel: shut down queue
    partial: bool = False                      # incomplete turn (no model call yet)
    state_delta: Optional[dict[str, Any]] = None  # always applied


class LiveRequestQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[LiveRequest] = asyncio.Queue()

    # convenience senders
    def send_content(self, content: types.Content, partial: bool = False) -> None: ...
    def send_realtime(self, blob: types.Blob) -> None: ...
    def send_activity_start(self) -> None: ...
    def send_activity_end(self) -> None: ...
    def send(self, req: LiveRequest) -> None: ...
    async def get(self) -> LiveRequest: ...

    def close(self) -> None:
        # Enqueues LiveRequest(close=True) as a shutdown sentinel
        self._queue.put_nowait(LiveRequest(close=True))
```

**Priority ordering** is important: when multiple fields are set on the
same `LiveRequest`, the runner processes only the highest-priority one
(except `state_delta`, which is always applied). Use separate
`send_*` calls rather than constructing compound requests by hand.

**`partial=True`** marks a content request as an incomplete turn so
the runner can accumulate fragments before forwarding to the model.

### Example 1 — text chat session

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

agent = LlmAgent(name="chat", model="gemini-2.5-flash", instruction="Be helpful.")
session_service = InMemorySessionService()
runner = Runner(app_name="demo", agent=agent, session_service=session_service)

async def chat():
    session = await session_service.create_session(app_name="demo", user_id="u1")
    queue = LiveRequestQueue()

    async def producer():
        for msg in ["Hello!", "What is 2+2?", "Thanks, bye!"]:
            queue.send_content(types.Content(
                role="user",
                parts=[types.Part(text=msg)]
            ))
            await asyncio.sleep(0.5)
        queue.close()

    async def consumer():
        async for event in runner.run_live(
            user_id="u1",
            session_id=session.id,
            live_request_queue=queue,
        ):
            if event.content and not event.partial:
                text = "".join(p.text or "" for p in event.content.parts)
                if text:
                    print(f"Agent: {text}")

    await asyncio.gather(producer(), consumer())

asyncio.run(chat())
```

### Example 2 — realtime audio with activity signals

```python
import asyncio
from google.genai import types
from google.adk.agents.live_request_queue import LiveRequestQueue, LiveRequest

async def stream_microphone(queue: LiveRequestQueue, audio_chunks):
    """Feed audio chunks from a microphone stream."""
    queue.send_activity_start()
    for chunk_bytes in audio_chunks:
        queue.send_realtime(types.Blob(mime_type="audio/pcm", data=chunk_bytes))
    queue.send_activity_end()
    queue.close()
```

### Example 3 — injecting state alongside content

```python
from google.adk.agents.live_request_queue import LiveRequestQueue, LiveRequest
from google.genai import types

queue = LiveRequestQueue()

# Send a content message AND atomically update session state in the same turn.
# state_delta is always applied regardless of which content field is set.
queue.send(LiveRequest(
    content=types.Content(
        role="user",
        parts=[types.Part(text="Use my saved preferences.")]
    ),
    state_delta={"user_language": "fr", "user_timezone": "Europe/Paris"},
))
```

---

## 2 · `ContextCacheConfig` — Gemini prompt-cache configuration

**Sources:** `google/adk/agents/context_cache_config.py`

### Why it matters

`ContextCacheConfig` (`@experimental(FeatureName.AGENT_CONFIG)`) is
placed on `App.context_cache_config` to activate Gemini's
[context caching](https://ai.google.dev/gemini-api/docs/caching) for
every `LlmAgent` in the app. The cache stores the large stable prefix
of the prompt (system instruction + tool schemas + history) so
subsequent turns pay only for the new delta tokens.

### Internals

```python
@experimental(FeatureName.AGENT_CONFIG)
class ContextCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_intervals: int = Field(default=10, ge=1, le=100)
    # Reuse the same CachedContent for this many invocations, then refresh.

    ttl_seconds: int = Field(default=1800, gt=0)
    # Cache lifetime. Default: 30 minutes.

    min_tokens: int = Field(default=0, ge=0)
    # Only cache if the previous request's prompt token count >= this value.
    # Gemini model minimums always apply: 2048 (Gemini 2.5) / 4096 (Gemini 3).
    # Caching never starts on the first turn (no prior token count).

    create_http_options: types.HttpOptions | None = Field(default=None)
    # Optional timeout for CachedContent.create() calls.
    # On timeout the request proceeds without caching.

    @property
    def ttl_string(self) -> str:
        return f"{self.ttl_seconds}s"   # required format for the API
```

Key constraints from the source docs:
- **Second turn at earliest** — caching never fires on turn 1 because
  there is no prior token count to evaluate `min_tokens` against.
- **Model minimums are enforced by Gemini**, not ADK: 2 048 tokens for
  Gemini 2.5, 4 096 for Gemini 3.
- `cache_intervals=10` means the same `CachedContent` resource is reused
  for 10 calls before a fresh cache is created to pick up any new prefix
  changes.

### Example 1 — basic context caching

```python
from google.adk.apps import App
from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig

agent = LlmAgent(
    name="analyst",
    model="gemini-2.5-pro",
    instruction="You are a financial analyst. " + open("large_knowledge_base.txt").read(),
)

app = App(
    name="finance_app",
    agent=agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=20,   # reuse cache across 20 turns
        ttl_seconds=3600,     # 1-hour TTL
        min_tokens=3000,      # only cache if prompt > 3 000 tokens
    ),
)
```

### Example 2 — with create timeout guard

```python
from google.genai import types
from google.adk.agents.context_cache_config import ContextCacheConfig

# Avoid blocking the request if CachedContent.create() is slow.
config = ContextCacheConfig(
    ttl_seconds=900,
    create_http_options=types.HttpOptions(timeout=8000),  # 8-second timeout in ms
)
```

### Example 3 — inspect TTL string

```python
config = ContextCacheConfig(ttl_seconds=600)
print(config.ttl_string)   # "600s"
print(config)
# ContextCacheConfig(cache_intervals=10, ttl=600s, min_tokens=0, create_http_options=None)
```

---

## 3 · `RunConfig` — per-invocation runtime configuration

**Sources:** `google/adk/agents/run_config.py`

### Why it matters

`RunConfig` is the Pydantic model you pass to `runner.run_async()` (and
`runner.run_live()`) to control everything that varies per invocation
without touching the agent definition: streaming mode, audio, tool
parallelism, LLM call budgets, telemetry, per-turn context injection,
and more. `extra="forbid"` means unknown fields raise a `ValidationError`
rather than being silently dropped.

### Selected fields

```python
class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # --- Audio / Live ---
    speech_config: Optional[types.SpeechConfig] = None
    response_modalities: Optional[list[types.Modality]] = None
    realtime_input_config: Optional[types.RealtimeInputConfig] = None
    output_audio_transcription: Optional[types.AudioTranscriptionConfig] = ...
    input_audio_transcription: Optional[types.AudioTranscriptionConfig] = ...
    translation_config: Optional[types.TranslationConfig] = None
    enable_affective_dialog: Optional[bool] = None
    proactivity: Optional[types.ProactivityConfig] = None

    # --- Streaming ---
    streaming_mode: StreamingMode = StreamingMode.NONE
    support_cfc: bool = False   # Compositional Function Calling (experimental)

    # --- Session management ---
    session_resumption: Optional[types.SessionResumptionConfig] = None
    history_config: Optional[types.HistoryConfig] = None
    context_window_compression: Optional[types.ContextWindowCompressionConfig] = None
    get_session_config: Optional[GetSessionConfig] = None
    # ↑ limit events fetched per turn — useful with EventsCompactionConfig

    # --- Per-turn context injection ---
    model_input_context: list[types.Content] | None = None
    # Injected into the LLM request for this invocation only.
    # NOT persisted to the session — invisible to future turns.

    # --- Multi-agent thoughts ---
    include_thoughts_from_other_agents: bool = False
    # Default False: sub-agent thought parts are stripped when reformatted
    # as user context for the current agent.

    # --- Execution limits ---
    max_llm_calls: int = 500
    # <= 0 → unlimited (logs a warning). sys.maxsize raises ValueError.

    # --- Observability ---
    labels: Optional[dict[str, str]] = None   # billing/attribution labels
    http_options: Optional[types.HttpOptions] = None
    custom_metadata: Optional[dict[str, Any]] = None
    telemetry: TelemetryConfig | None = None  # per-request OTel override

    # --- Thread pool ---
    tool_thread_pool_config: Optional[ToolThreadPoolConfig] = None

    # --- Artifacts ---
    save_live_blob: bool = False  # saves live video/audio to artifact service
```

### Example 1 — SSE streaming with budget cap

```python
from google.genai import types
from google.adk.agents.run_config import RunConfig, StreamingMode

run_config = RunConfig(
    streaming_mode=StreamingMode.SSE,
    max_llm_calls=50,
    labels={"env": "production", "user_tier": "premium"},
)

async for event in runner.run_async(
    user_id="u1",
    session_id="s1",
    new_message=types.Content(role="user", parts=[types.Part(text="Summarize this doc.")]),
    run_config=run_config,
):
    if event.partial and event.content:
        for part in event.content.parts:
            if part.text and not any(p.function_call for p in event.content.parts):
                print(part.text, end="", flush=True)
```

### Example 2 — per-turn transient context

```python
from google.genai import types
from google.adk.agents.run_config import RunConfig

# Inject today's market snapshot without polluting conversation history.
run_config = RunConfig(
    model_input_context=[
        types.Content(
            role="user",
            parts=[types.Part(text="[CONTEXT] AAPL: $189.42, MSFT: $415.30")]
        )
    ]
)
```

### Example 3 — limit events fetched per turn

```python
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.base_session_service import GetSessionConfig

# In a long session with EventsCompactionConfig, only load the last 30 events.
run_config = RunConfig(
    get_session_config=GetSessionConfig(num_recent_events=30),
)
```

---

## 4 · `StreamingMode` — SSE and BIDI streaming modes

**Sources:** `google/adk/agents/run_config.py`

### Why it matters

`StreamingMode` is an enum with three members that controls how the
runner yields events from `run_async()`. The docstrings in the source
contain several concrete patterns for handling the duplicate-text
problem in SSE mode, which is the most common source of confusion.

### Internals

```python
class StreamingMode(Enum):
    NONE = None   # single aggregated content per turn (default)
    SSE  = 'sse'  # progressive partial events + final aggregated event
    BIDI = 'bidi' # reserved; actual bidi uses runner.run_live(), not run_async()
```

**SSE event types** (when `streaming_mode=StreamingMode.SSE`):

| `event.partial` | Content | Meaning |
|---|---|---|
| `True` | text parts | Streaming text chunk — display for typewriter effect |
| `True` | function\_call parts | In-flight FC argument accumulation — usually skip in UI |
| `False` | any | Final aggregated response — full text or complete FC |

**Duplicate text issue**: because `PROGRESSIVE_SSE_STREAMING` (default ON)
emits both partial chunks **and** a final aggregated text event, naive
code that prints all events displays the text twice.

### Example 1 — typewriter effect (skip final text)

```python
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

run_config = RunConfig(streaming_mode=StreamingMode.SSE)

async for event in runner.run_async(..., run_config=run_config):
    if event.partial and event.content and event.content.parts:
        has_text = any(p.text for p in event.content.parts)
        has_fc   = any(p.function_call for p in event.content.parts)
        if has_text and not has_fc:
            print("".join(p.text or "" for p in event.content.parts),
                  end="", flush=True)
    elif not event.partial and event.get_function_calls():
        for fc in event.get_function_calls():
            print(f"\n→ calling {fc.name}({fc.args})")
```

### Example 2 — final-only (no streaming effect)

```python
run_config = RunConfig(streaming_mode=StreamingMode.SSE)

async for event in runner.run_async(..., run_config=run_config):
    if not event.partial and event.content:
        text = "".join(p.text or "" for p in event.content.parts)
        if text:
            print(text)
```

### Example 3 — detect when a final response completes a turn

```python
run_config = RunConfig(streaming_mode=StreamingMode.SSE)

async for event in runner.run_async(..., run_config=run_config):
    if event.is_final_response():
        # is_final_response() returns True for the last non-partial event
        # of a turn — useful for UI completion signals.
        print("[DONE]")
        break
```

---

## 5 · `ToolThreadPoolConfig` — offload blocking tools to a thread pool

**Sources:** `google/adk/agents/run_config.py`

### Why it matters

By default, tool execution runs on the asyncio event loop. A tool that
calls `time.sleep()`, issues a blocking DB query, or does heavy network
I/O stalls the entire loop, preventing the agent from receiving
streaming audio or user interruptions. `ToolThreadPoolConfig` routes all
tool calls to a `ThreadPoolExecutor` so the event loop stays responsive.

### Internals

```python
class ToolThreadPoolConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_workers: int = Field(default=4, ge=1)
```

**How the pool is managed** (from the source docstring):
- One pool is created per asyncio event loop and shared across all
  concurrent invocations on that loop.
- The pool is shut down when the event loop closes — worker threads
  never outlive it.
- **Async tools** are run in a *new* event loop inside the background
  thread, which catches accidental blocking I/O inside `async def` tools.
- A cancelled invocation drops queued-but-not-started tool calls; a
  tool already executing in a thread **cannot be interrupted** (Python
  threads are not cancellable).

**GIL limitations** (from the source docstring):

| Scenario | Thread pool helps? |
|---|---|
| Blocking I/O (`time.sleep`, network, file, DB) | ✅ Yes — GIL released |
| C extensions (numpy, hashlib, PIL) | ✅ Yes — GIL released |
| Pure-Python CPU loops / calculations | ❌ No — GIL held |

For CPU-bound Python code use a `ProcessPoolExecutor` instead, or
break work into chunks with `await asyncio.sleep(0)`.

### Example 1 — enable with default 4 workers

```python
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

run_config = RunConfig(
    tool_thread_pool_config=ToolThreadPoolConfig(),  # max_workers=4
)
```

### Example 2 — high-concurrency live agent

```python
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig, StreamingMode

run_config = RunConfig(
    streaming_mode=StreamingMode.SSE,
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=16),
)
```

### Example 3 — blocking DB tool pattern

```python
import time
from google.adk.tools import FunctionTool

def query_slow_database(user_id: str) -> dict:
    """Simulate a 2-second blocking DB query."""
    time.sleep(2)  # This blocks; safe in thread pool, dangerous on event loop.
    return {"user_id": user_id, "balance": 9_999}

agent = LlmAgent(
    name="banker",
    model="gemini-2.5-flash",
    tools=[FunctionTool(func=query_slow_database)],
)

run_config = RunConfig(
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
)
```

---

## 6 · `PubSubToolset` — Google Cloud Pub/Sub integration

**Sources:** `google/adk/tools/pubsub/pubsub_toolset.py`,
`google/adk/tools/pubsub/message_tool.py`,
`google/adk/tools/pubsub/config.py`

### Why it matters

`PubSubToolset` (`@experimental(FeatureName.PUBSUB_TOOLSET)`) wires
three Pub/Sub operations — publish, pull, and acknowledge — into an
ADK toolset so agents can participate in event-driven architectures
without boilerplate. Credentials are handled via `PubSubCredentialsConfig`
using the standard GCP auth chain.

### Internals

```python
@experimental(FeatureName.PUBSUB_TOOLSET)
class PubSubToolset(BaseToolset):
    def __init__(
        self,
        *,
        tool_filter: ToolPredicate | list[str] | None = None,
        credentials_config: PubSubCredentialsConfig | None = None,
        pubsub_tool_config: PubSubToolConfig | None = None,
    ): ...

    async def get_tools(self, ...) -> list[BaseTool]:
        # Returns GoogleTool wrappers around:
        #   message_tool.publish_message
        #   message_tool.pull_messages
        #   message_tool.acknowledge_messages

    async def close(self):
        client.cleanup_clients()  # drains cached publisher/subscriber clients


@experimental(FeatureName.PUBSUB_TOOL_CONFIG)
class PubSubToolConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    project_id: str | None = None
```

**`publish_message`** signature:
```python
def publish_message(
    topic_name: str,        # "projects/my-proj/topics/my-topic"
    message: str,           # UTF-8 content
    credentials: Credentials,
    settings: PubSubToolConfig,
    attributes: Optional[dict[str, str]] = None,
    ordering_key: str = "",
) -> dict   # {"message_id": "..."}
```

**`pull_messages`** key parameters:
- `max_messages: int = 1`
- `auto_ack: bool = False` — when `True` calls `acknowledge()` immediately

### Example 1 — publish events from an agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.features import temporary_feature_override, FeatureName

with temporary_feature_override(FeatureName.PUBSUB_TOOLSET, True):
    toolset = PubSubToolset(
        pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    )

    agent = LlmAgent(
        name="notifier",
        model="gemini-2.5-flash",
        instruction=(
            "When the user asks you to send an alert, publish a JSON message "
            "to projects/my-gcp-project/topics/alerts."
        ),
        tools=[toolset],
    )
```

### Example 2 — filter to only the publish tool

```python
from google.adk.tools.pubsub import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

toolset = PubSubToolset(
    tool_filter=["publish_message"],  # only expose publish; hide pull/ack
    pubsub_tool_config=PubSubToolConfig(project_id="my-proj"),
)
```

### Example 3 — ordered messaging agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

# The model can call publish_message with an ordering_key to guarantee
# in-order delivery within the same key on an ordering-enabled publisher.

agent = LlmAgent(
    name="sequencer",
    model="gemini-2.5-flash",
    instruction=(
        "Publish order updates to projects/my-proj/topics/orders. "
        "Always use the order_id as the ordering_key so updates arrive in sequence."
    ),
    tools=[PubSubToolset(pubsub_tool_config=PubSubToolConfig(project_id="my-proj"))],
)
```

---

## 7 · `SpannerToolset` — Cloud Spanner SQL + vector-search toolset

**Sources:** `google/adk/tools/spanner/spanner_toolset.py`,
`google/adk/tools/spanner/settings.py`

### Why it matters

`SpannerToolset` (`@experimental(FeatureName.SPANNER_TOOLSET)`) gives
agents read access to a Cloud Spanner database through 5–8 tools depending
on configuration. It supports both classic SQL queries and approximate/exact
vector-similarity search, making it suitable for RAG agents backed by
Spanner's native vector index.

### Internals

```python
@experimental(FeatureName.SPANNER_TOOLSET)
class SpannerToolset(BaseToolset):
    def __init__(
        self,
        *,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
        credentials_config: Optional[SpannerCredentialsConfig] = None,
        spanner_tool_settings: Optional[SpannerToolSettings] = None,
    ): ...

    async def get_tools(self, ...) -> List[BaseTool]:
        # Always available (5 metadata tools):
        #   spanner_list_table_names
        #   spanner_list_table_indexes
        #   spanner_list_table_index_columns
        #   spanner_list_named_schemas
        #   spanner_get_table_schema
        # Added when Capabilities.DATA_READ in settings.capabilities:
        #   spanner_execute_sql
        #   spanner_similarity_search
        #   spanner_vector_store_similarity_search  (only if vector_store_settings set)


@experimental(FeatureName.SPANNER_TOOL_SETTINGS)
class SpannerToolSettings(BaseModel):
    capabilities: List[Capabilities] = [Capabilities.DATA_READ]
    max_executed_query_result_rows: int = 50  # caps result set size
    query_result_mode: QueryResultMode = QueryResultMode.DEFAULT
    database_role: Optional[str] = None       # Spanner fine-grained auth
    vector_store_settings: Optional[SpannerVectorStoreSettings] = None
```

`QueryResultMode.DICT_LIST` returns `[{"col": val}, ...]` instead of
raw row arrays, which is friendlier for LLM consumption.

### Example 1 — read-only SQL agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.spanner import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig

toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        max_executed_query_result_rows=100,
        query_result_mode="dict_list",  # QueryResultMode.DICT_LIST
    ),
    credentials_config=SpannerCredentialsConfig(
        project_id="my-proj",
        instance_id="prod-instance",
        database_id="analytics",
    ),
)

agent = LlmAgent(
    name="db_analyst",
    model="gemini-2.5-pro",
    instruction="Answer questions by querying the Spanner database.",
    tools=[toolset],
)
```

### Example 2 — vector similarity search

```python
from google.adk.tools.spanner.settings import (
    SpannerToolSettings,
    SpannerVectorStoreSettings,
)

vector_settings = SpannerVectorStoreSettings(
    project_id="my-proj",
    instance_id="prod-instance",
    database_id="knowledge_base",
    table_name="documents",
    content_column="text",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    top_k=5,
    distance_type="COSINE",
)

toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        vector_store_settings=vector_settings,
    ),
)
```

### Example 3 — restrict tools via filter

```python
from google.adk.tools.spanner import SpannerToolset

# Expose only the schema-introspection tools, hide data-read tools.
toolset = SpannerToolset(
    tool_filter=["spanner_list_table_names", "spanner_get_table_schema"],
    spanner_tool_settings=SpannerToolSettings(capabilities=[]),
)
```

---

## 8 · `Workflow` — graph-based orchestration node

**Sources:** `google/adk/workflow/_workflow.py`

### Why it matters

`Workflow` is a `BaseNode` subclass that replaces `SequentialAgent` and
`ParallelAgent` for complex, branching pipelines. You declare edges
(using `parse_edge_items` chain syntax covered in vol. 45) and the
workflow engine fans them out to parallel `NodeRunner` tasks, handles
replay/resume from session events, and enforces `max_concurrency` for
static nodes.

### Internals

```python
class Workflow(BaseNode):
    rerun_on_resume: bool = Field(default=True)
    edges: list[EdgeItem] = Field(default_factory=list)
    max_concurrency: int | None = None
    # max_concurrency only throttles static (graph-edge-triggered) nodes.
    # Dynamic nodes (via ctx.run_node()) are excluded — throttling them
    # would deadlock the event loop.
    graph: Graph | None = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.edges and self.graph is None:
            self.graph = Graph.from_edge_items(self.edges)
            self.graph.validate_graph()
        self._validate_state_schema()
        # _validate_state_schema checks that FunctionNode parameter names
        # match fields in state_schema if one is provided.
```

`_LoopState` is the mutable, in-memory scratch-pad for one `_run_impl`
invocation. It is **never persisted** — static node state is rebuilt from
session events on resume; dynamic node state is lazily scanned on demand.

Key `_LoopState` fields:

| Field | Purpose |
|---|---|
| `nodes` | `dict[str, NodeState]` — live status of each graph node |
| `node_outputs` | Cached outputs keyed by node name |
| `pending_tasks` | `dict[str, asyncio.Task]` — running coroutines |
| `trigger_buffer` | Queued triggers waiting to be dispatched |
| `error_shut_down` | True after a node failure; drains remaining tasks |
| `replayed_nodes` | Nodes in fast-forward replay (no real execution) |

### Example 1 — linear pipeline

```python
from google.adk.workflow import Workflow
from google.adk.workflow._function_node import FunctionNode
from google.adk.agents import LlmAgent

fetch = FunctionNode(name="fetch", func=fetch_data)
analyse = LlmAgent(name="analyse", model="gemini-2.5-flash",
                   instruction="Analyse the fetched data.")
report = FunctionNode(name="report", func=write_report)

pipeline = Workflow(
    name="report_pipeline",
    edges=[(fetch, analyse, report)],
)
```

### Example 2 — fan-out with max_concurrency

```python
from google.adk.workflow import Workflow
from google.adk.workflow._function_node import FunctionNode

# Three parallel enrichment steps, capped at 2 concurrent nodes.
enrich_a = FunctionNode(name="enrich_geo",  func=add_geo)
enrich_b = FunctionNode(name="enrich_demo", func=add_demographics)
enrich_c = FunctionNode(name="enrich_hist", func=add_history)
merge    = FunctionNode(name="merge",       func=merge_all)

wf = Workflow(
    name="enrichment",
    edges=[
        ("START", [enrich_a, enrich_b, enrich_c]),
        (enrich_a, merge),
        (enrich_b, merge),
        (enrich_c, merge),
    ],
    max_concurrency=2,
)
```

### Example 3 — conditional routing

```python
from google.adk.workflow import Workflow

classify = FunctionNode(name="classify", func=classify_intent)
handle_faq  = LlmAgent(name="faq",  model="gemini-2.5-flash",
                        instruction="Answer FAQ questions.")
handle_escalate = LlmAgent(name="escalate", model="gemini-2.5-pro",
                           instruction="Handle complex escalations.")

wf = Workflow(
    name="router",
    edges=[
        (classify, {"faq": handle_faq, "escalate": handle_escalate}),
    ],
)
```

---

## 9 · `RetryConfig` — exponential-backoff retry for workflow nodes

**Sources:** `google/adk/workflow/_retry_config.py`

### Why it matters

Any `BaseNode` (including `FunctionNode`, `LlmAgent`, and custom nodes)
can be assigned a `RetryConfig`. When a node raises an exception the
runner checks whether the exception type matches `exceptions` and, if so,
waits `initial_delay * backoff_factor^(attempt-1) ± jitter` before
re-running the node up to `max_attempts` times total.

### Internals

```python
class RetryConfig(BaseModel):
    max_attempts: int | None = Field(default=None)
    # None → ADK default of 5 total attempts.
    # 0 or 1 → no retries.

    initial_delay: float | None = Field(default=None)
    # Seconds before first retry. None → 1.0 s.

    max_delay: float | None = Field(default=None)
    # Cap on delay between retries. None → 60.0 s.

    backoff_factor: float | None = Field(default=None)
    # Delay multiplier after each attempt. None → 2.0 (exponential).

    jitter: float | None = Field(default=None)
    # Randomness factor. None → 1.0. Set to 0.0 to remove randomness.
    # Prevents thundering-herd when many nodes retry simultaneously.

    exceptions: list[str | type[BaseException]] | None = Field(default=None)
    # None → retry on ALL exceptions.
    # Accepts class names ("ValueError") or class objects (ValueError).

    @field_validator('exceptions', mode='before')
    @classmethod
    def _normalize_exceptions(cls, v):
        # Converts type objects to their __name__ strings for uniform handling.
        ...
```

### Example 1 — retry transient network errors

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._function_node import FunctionNode

def call_external_api(endpoint: str) -> dict:
    import httpx
    resp = httpx.get(endpoint, timeout=10)
    resp.raise_for_status()
    return resp.json()

node = FunctionNode(
    name="api_caller",
    func=call_external_api,
    retry_config=RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=0.5,
        exceptions=["httpx.HTTPStatusError", "httpx.TimeoutException"],
    ),
)
```

### Example 2 — retry all exceptions with tight cap

```python
from google.adk.workflow._retry_config import RetryConfig

# Retry on any exception, maximum 3 attempts, no jitter.
retry = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=5.0,
    backoff_factor=2.0,
    jitter=0.0,
    exceptions=None,  # retry on everything
)
```

### Example 3 — pass exception class objects directly

```python
from google.adk.workflow._retry_config import RetryConfig

# The validator converts ValueError/IOError to their string names internally.
retry = RetryConfig(
    max_attempts=5,
    exceptions=[ValueError, IOError, TimeoutError],
)
```

---

## 10 · `CachePerformanceAnalyzer` — cache hit-rate and cost analysis

**Sources:** `google/adk/utils/cache_performance_analyzer.py`

### Why it matters

`CachePerformanceAnalyzer` (`@experimental`) gives you a data-driven view
of how well `ContextCacheConfig` is performing. It walks the event history
of a session, collects `CacheMetadata` from each event, and computes
hit ratios and token savings so you can tune `cache_intervals`,
`ttl_seconds`, and `min_tokens` without guessing.

### Internals

```python
@experimental
class CachePerformanceAnalyzer:
    def __init__(self, session_service: BaseSessionService):
        self.session_service = session_service

    async def analyze_agent_cache_performance(
        self,
        session_id: str,
        user_id: str,
        app_name: str,
        agent_name: str,
    ) -> dict[str, Any]:
        # Returns (when data exists):
        # {
        #   "status": "active",
        #   "requests_with_cache": int,        # events that have CacheMetadata
        #   "avg_invocations_used": float,     # avg turns each cache was reused
        #   "latest_cache": str,               # resource name of newest cache
        #   "cache_refreshes": int,            # distinct cache resource names
        #   "total_invocations": int,
        #   "total_prompt_tokens": int,
        #   "total_cached_tokens": int,
        #   "cache_hit_ratio_percent": float,  # cached / prompt * 100
        #   "cache_utilization_ratio_percent": float,  # requests_with_hits / total * 100
        #   "avg_cached_tokens_per_request": float,
        #   "total_requests": int,
        #   "requests_with_cache_hits": int,
        # }
        # Returns {"status": "no_cache_data"} when no cache events found.
```

`cache_hit_ratio_percent` = `total_cached_tokens / total_prompt_tokens × 100`.
A healthy ratio for a large-system-prompt agent is typically **60–80 %**.

`cache_utilization_ratio_percent` = `requests_with_cache_hits / total_requests × 100`.
This drops if the cache is frequently expired or not yet warmed up (first
turn of each session is never cached).

### Example 1 — basic performance snapshot

```python
import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.utils.cache_performance_analyzer import CachePerformanceAnalyzer

session_service = InMemorySessionService()
analyzer = CachePerformanceAnalyzer(session_service)

async def report():
    stats = await analyzer.analyze_agent_cache_performance(
        session_id="session-123",
        user_id="user-456",
        app_name="finance_app",
        agent_name="analyst",
    )
    if stats["status"] == "no_cache_data":
        print("Cache not yet active (session too short or tokens below minimum).")
    else:
        print(f"Hit ratio:         {stats['cache_hit_ratio_percent']:.1f}%")
        print(f"Utilisation:       {stats['cache_utilization_ratio_percent']:.1f}%")
        print(f"Avg cached tokens: {stats['avg_cached_tokens_per_request']:.0f}")
        print(f"Cache refreshes:   {stats['cache_refreshes']}")

asyncio.run(report())
```

### Example 2 — tune cache_intervals based on analysis

```python
async def auto_tune_cache_config(
    analyzer: CachePerformanceAnalyzer,
    session_id: str,
    user_id: str,
    app_name: str,
    agent_name: str,
) -> ContextCacheConfig:
    stats = await analyzer.analyze_agent_cache_performance(
        session_id, user_id, app_name, agent_name
    )
    if stats["status"] == "no_cache_data":
        return ContextCacheConfig()   # use defaults

    # If each cache is only used once on average, reduce cache_intervals
    # so stale caches are evicted faster.
    avg_use = stats["avg_invocations_used"]
    intervals = max(1, min(50, int(avg_use * 1.5)))
    return ContextCacheConfig(cache_intervals=intervals, ttl_seconds=1800)
```

### Example 3 — cross-session aggregate report

```python
async def aggregate_report(
    analyzer: CachePerformanceAnalyzer,
    app_name: str,
    user_id: str,
    session_ids: list[str],
    agent_name: str,
) -> dict:
    totals = {"total_prompt_tokens": 0, "total_cached_tokens": 0, "sessions": 0}
    for sid in session_ids:
        s = await analyzer.analyze_agent_cache_performance(
            sid, user_id, app_name, agent_name
        )
        if s["status"] == "active":
            totals["total_prompt_tokens"] += s["total_prompt_tokens"]
            totals["total_cached_tokens"] += s["total_cached_tokens"]
            totals["sessions"] += 1

    if totals["total_prompt_tokens"]:
        totals["overall_hit_ratio_percent"] = (
            totals["total_cached_tokens"] / totals["total_prompt_tokens"] * 100
        )
    return totals
```

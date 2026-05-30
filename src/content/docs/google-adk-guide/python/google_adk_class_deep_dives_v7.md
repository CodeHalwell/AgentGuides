---
title: "Class deep dives — volume 7 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: InvocationContext, SetModelResponseTool, DynamicNodeScheduler/ctx.run_node(), BaseRetrievalTool/FilesRetrieval/LlamaIndexRetrieval/VertexAiRagRetrieval, FirestoreSessionService/FirestoreMemoryService, BigQueryToolset/BigQueryToolConfig, LangchainTool, CrewaiTool, SlackRunner, FileArtifactService."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 7"
  order: 66
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `InvocationContext` | `google.adk.agents.invocation_context` | Stable |
| 2 | `SetModelResponseTool` | `google.adk.tools.set_model_response_tool` | Stable |
| 3 | `DynamicNodeScheduler` + `ctx.run_node()` | `google.adk.workflow._dynamic_node_scheduler` | Stable |
| 4 | `BaseRetrievalTool` / `LlamaIndexRetrieval` / `FilesRetrieval` / `VertexAiRagRetrieval` | `google.adk.tools.retrieval` | Stable |
| 5 | `FirestoreSessionService` + `FirestoreMemoryService` | `google.adk.integrations.firestore` | Stable |
| 6 | `BigQueryToolset` + `BigQueryToolConfig` + `WriteMode` | `google.adk.integrations.bigquery` | Stable |
| 7 | `LangchainTool` | `google.adk.integrations.langchain.langchain_tool` | Stable |
| 8 | `CrewaiTool` | `google.adk.integrations.crewai.crewai_tool` | Stable |
| 9 | `SlackRunner` | `google.adk.integrations.slack.slack_runner` | Stable |
| 10 | `FileArtifactService` | `google.adk.artifacts.file_artifact_service` | Stable |

---

## 1 · `InvocationContext`

`InvocationContext` is the root data structure for one complete user→response round-trip. The Runner creates exactly one per `run_async()` call, wires together every service (session, artifact, memory, credentials), and threads it through every agent, tool, and callback in the call graph. Understanding it is essential for writing non-trivial plugins and workflow nodes.

### Hierarchy of granularity (from source comments)

```
Invocation  ← one user message → one final response (Runner.run_async)
  └── AgentCall  ← one agent runs; transfers create additional AgentCalls
        └── Step  ← one LLM call + its tool calls / responses
```

### Class definition

```python
class InvocationContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
```

### Field reference (source-verified)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `invocation_id` | `str` | `"e-" + uuid` | Read-only string assigned by `new_invocation_context_id()` |
| `session` | `Session` | required | Active session; read-only after creation |
| `session_service` | `BaseSessionService` | required | Persists events and state |
| `artifact_service` | `Optional[BaseArtifactService]` | `None` | File/object storage |
| `memory_service` | `Optional[BaseMemoryService]` | `None` | Long-term memory across sessions |
| `credential_service` | `Optional[BaseCredentialService]` | `None` | OAuth token storage |
| `context_cache_config` | `Optional[ContextCacheConfig]` | `None` | Gemini context caching |
| `branch` | `Optional[str]` | `None` | Dot-separated agent transfer path, e.g. `"root.sub_a.leaf"` |
| `isolation_scope` | `Optional[str]` | `None` | Internal; do not set directly |
| `agent` | `Optional[BaseAgent \| BaseNode]` | `None` | Currently executing agent |
| `user_content` | `Optional[types.Content]` | `None` | The triggering user message |
| `node_path` | `Optional[str]` | `None` | Slash-separated workflow path, e.g. `"root/node_a"` |
| `agent_states` | `dict[str, dict[str, Any]]` | `{}` | Per-agent resumability state |
| `end_of_agents` | `dict[str, bool]` | `{}` | Tracks which agents have finished |
| `end_invocation` | `bool` | `False` | Set `True` from a plugin/callback to terminate the invocation cleanly |
| `live_request_queue` | `Optional[LiveRequestQueue]` | `None` | For live/streaming sessions |
| `active_streaming_tools` | `Optional[dict[str, ActiveStreamingTool]]` | `None` | In-flight streaming tool state |
| `transcription_cache` | `Optional[list[TranscriptionEntry]]` | `None` | Audio transcription cache |
| `live_session_resumption_handle` | `Optional[str]` | `None` | Handle for resuming live sessions |
| `run_config` | `Optional[RunConfig]` | `None` | Per-invocation config (max LLM calls, streaming, etc.) |
| `resumability_config` | `Optional[ResumabilityConfig]` | `None` | Pause-and-resume configuration |
| `events_compaction_config` | `Optional[EventsCompactionConfig]` | `None` | Sliding-window event compaction |
| `token_compaction_checked` | `bool` | `False` | Whether token compaction was checked this invocation |
| `plugin_manager` | `PluginManager` | required | Manages all registered plugins |
| `canonical_tools_cache` | `Optional[list[BaseTool]]` | `None` | Resolved tools for this agent call |
| `credential_by_key` | `dict[str, AuthCredential]` | `{}` | Auth credentials accumulated during this invocation |

### Private attributes (from source)

| Attribute | Notes |
|-----------|-------|
| `_state_schema` | Pydantic model used for state validation (set via `App`). |
| `_event_queue` | `asyncio.Queue` — non-partial events block until the Runner confirms the event has been appended to the session. Partial (streaming) events are non-blocking. This guarantees session consistency before the next step proceeds. |
| `_invocation_cost_manager` | Tracks LLM call count; raises `LlmCallsLimitExceededError` when `run_config.max_llm_calls` is exceeded. |

### Key methods (from source)

| Method | Notes |
|--------|-------|
| `_enqueue_event(event)` | Core event dispatch. Non-partial events use `asyncio.Event` to block until the Runner confirms append. |
| `should_pause_invocation(event)` | Returns `True` if `event` contains a long-running function call, triggering the pause-and-resume flow. |
| `increment_llm_call_count()` | Increments call count; raises `LlmCallsLimitExceededError` when budget is exceeded. |
| `set_agent_state(agent_name, *, agent_state, end_of_agent)` | Stores per-agent state for the resumability system. |
| `_get_events(*, current_invocation, current_branch)` | Filters `session.events` by invocation ID and branch for context construction. |

### Factory function

```python
from google.adk.agents.invocation_context import new_invocation_context_id

invocation_id = new_invocation_context_id()
# → "e-3f8a9b2c-1d4e-..."
```

> **Important:** `InvocationContext` is **not** a public API you construct directly. The Runner creates one per `run_async()` call. Access it inside plugins via `callback_context._invocation_context`, or inside workflow node functions via `ctx._invocation_context`.

### Example 1 — reading InvocationContext from a plugin

Use the `on_before_agent_call` hook to inspect the context at the start of each agent call:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import InMemoryRunner


class InspectorPlugin(BasePlugin):
    """Log key InvocationContext fields at the start of every agent call."""

    async def on_before_agent_call(self, callback_context: CallbackContext) -> None:
        ctx: InvocationContext = callback_context._invocation_context

        print(f"[Plugin] invocation_id : {ctx.invocation_id}")
        print(f"[Plugin] agent.name    : {ctx.agent.name if ctx.agent else None}")
        print(f"[Plugin] branch        : {ctx.branch}")

        if ctx.user_content and ctx.user_content.parts:
            text = ctx.user_content.parts[0].text or ""
            print(f"[Plugin] user_content  : {text[:120]}")

        if ctx.active_streaming_tools:
            tool_names = list(ctx.active_streaming_tools.keys())
            print(f"[Plugin] streaming_tools: {tool_names}")


agent = LlmAgent(
    name="demo_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions concisely.",
)


async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="ctx_demo",
        plugins=[InspectorPlugin()],
    )
    await runner.session_service.create_session(
        app_name="ctx_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the capital of France?",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — early termination with `end_invocation = True`

Set `end_invocation = True` from any plugin hook to stop the invocation cleanly after the current step. ADK checks this flag after each step and before dispatching to sub-agents.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import InMemoryRunner


BLOCKED_TOPICS = {"gambling", "weapons", "drugs"}


class ContentGuardPlugin(BasePlugin):
    """Terminate the invocation if blocked topics are detected in the user message."""

    async def on_before_agent_call(self, callback_context: CallbackContext) -> None:
        ctx = callback_context._invocation_context

        if not (ctx.user_content and ctx.user_content.parts):
            return

        user_text = (ctx.user_content.parts[0].text or "").lower()
        for topic in BLOCKED_TOPICS:
            if topic in user_text:
                print(f"[Guard] Blocked topic '{topic}' detected — terminating.")
                ctx.end_invocation = True
                return


agent = LlmAgent(
    name="safe_agent",
    model="gemini-2.5-flash",
    instruction="Answer general knowledge questions.",
)


async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="guard_demo",
        plugins=[ContentGuardPlugin()],
    )
    await runner.session_service.create_session(
        app_name="guard_demo", user_id="u1", session_id="s1"
    )

    # This should be terminated early
    events = await runner.run_debug(
        "Tell me about gambling strategies.",
        user_id="u1",
        session_id="s1",
    )
    print(f"Events generated: {len(events)}")  # 0 or just the user event

    # This should complete normally
    events = await runner.run_debug(
        "What is the tallest mountain in the world?",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 3 — checking `active_streaming_tools` in a live session callback

During a live (bidirectional streaming) session, `active_streaming_tools` tracks which streaming tools are currently in-flight. A plugin can inspect this to implement custom interruption logic:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import ActiveStreamingTool


class StreamingMonitorPlugin(BasePlugin):
    """Log streaming tool activity and optionally cancel long-running tools."""

    MAX_STREAMING_SECONDS = 30.0

    async def on_before_agent_call(self, callback_context: CallbackContext) -> None:
        ctx = callback_context._invocation_context

        if not ctx.active_streaming_tools:
            return

        import time
        now = time.time()
        for tool_call_id, streaming_tool in list(ctx.active_streaming_tools.items()):
            elapsed = now - getattr(streaming_tool, "start_time", now)
            print(f"[Monitor] Streaming tool {tool_call_id!r} active for {elapsed:.1f}s")

            if elapsed > self.MAX_STREAMING_SECONDS:
                print(f"[Monitor] Cancelling long-running tool {tool_call_id!r}")
                # Signal cancellation through the tool's cancel method if available
                if hasattr(streaming_tool, "cancel"):
                    await streaming_tool.cancel()
```

> **Production tip:** `end_invocation` is checked after each step boundary, not mid-LLM-call. If your guard must apply before the first LLM token, set it in `on_before_agent_call`. For mid-stream termination you need the live session API.

> **Gotcha:** `branch` is `None` for the root agent. It starts being populated after the first `transfer_to_agent`. Use `ctx.branch or "root"` for logging to avoid `None` string comparisons.

---

## 2 · `SetModelResponseTool`

When an `LlmAgent` has both an `output_schema` AND other callable tools, Gemini faces a protocol conflict: structured JSON output flows through the text channel, but tool calls flow through a separate function-call channel. Once any tool is declared, Gemini will not produce structured JSON via text — it uses the function-call channel exclusively.

`SetModelResponseTool` resolves this by registering a **synthetic tool** whose function declaration mirrors the `output_schema` fields. The LLM "calls" this tool to return its structured response, and ADK picks up `tool_context.actions.set_model_response` as the agent's final output.

### Constructor

```python
from google.adk.tools.set_model_response_tool import SetModelResponseTool

SetModelResponseTool(output_schema: SchemaType)
```

`SchemaType` can be any of:

| Input type | Resulting tool parameter |
|------------|--------------------------|
| `type[BaseModel]` | One parameter per Pydantic field |
| `list[type[BaseModel]]` | Single `items: list[MyModel]` parameter |
| `list[str]` / `list[int]` / primitive list | Single `response` parameter |
| `dict` / `types.Schema` | Single `response` parameter |

### Internal mechanism (from source)

1. **Signature construction** — `_build_function_signature()` introspects the schema type and dynamically builds an `inspect.Signature` whose parameters match the schema fields.
2. **Declaration** — `_get_declaration()` calls `build_function_declaration()` with the synthetic signature to produce a `FunctionDeclaration` the LLM can call.
3. **Validation in `run_async`** — For `BaseModel`: calls `model_validate()`. For `list[BaseModel]`: calls `TypeAdapter.validate_python()`. For primitives: passes through as-is.
4. **Output dispatch** — Sets `tool_context.actions.set_model_response = validated_result`, which ADK uses as the agent's `output` at the invocation level.

### When is `SetModelResponseTool` injected automatically?

ADK automatically injects `SetModelResponseTool` inside `LlmAgent._get_canonical_tools()` whenever `output_schema` is set AND there are other tools. You do **not** need to add it yourself — it will appear in the LLM request alongside your other tools.

### Example 1 — automatic injection when output_schema + tools are combined

This shows the canonical pattern where ADK inserts `SetModelResponseTool` automatically:

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.agents.context import ToolContext


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str
    humidity_percent: int


def lookup_temperature(city: str) -> dict:
    """Fake weather lookup — replace with a real API call."""
    data = {
        "London": {"temp": 15.0, "condition": "Cloudy", "humidity": 78},
        "Tokyo": {"temp": 22.0, "condition": "Sunny", "humidity": 60},
    }
    return data.get(city, {"temp": 20.0, "condition": "Unknown", "humidity": 50})


# ADK sees output_schema + tools → automatically injects SetModelResponseTool
agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Use lookup_temperature to get data, then return a structured WeatherReport."
    ),
    tools=[lookup_temperature],
    output_schema=WeatherReport,
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="weather_demo")
    await runner.session_service.create_session(
        app_name="weather_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the weather in London?",
        user_id="u1",
        session_id="s1",
    )

    # The final event's output is the validated WeatherReport instance
    final = events[-1]
    if final.actions and final.actions.set_model_response:
        report: WeatherReport = final.actions.set_model_response
        print(f"City: {report.city}")
        print(f"Temperature: {report.temperature_celsius}°C")
        print(f"Condition: {report.condition}")
        print(f"Humidity: {report.humidity_percent}%")


asyncio.run(main())
```

### Example 2 — list[BaseModel] schema produces a single `items` parameter

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


class Product(BaseModel):
    name: str
    price_usd: float
    in_stock: bool


def search_catalog(query: str) -> dict:
    """Search product catalog (stub)."""
    return {
        "results": [
            {"name": "Widget A", "price": 9.99, "stock": True},
            {"name": "Widget B", "price": 14.99, "stock": False},
        ]
    }


# list[BaseModel] → SetModelResponseTool emits a single `items` parameter
# The LLM returns: set_model_response(items=[{...}, {...}])
agent = LlmAgent(
    name="catalog_agent",
    model="gemini-2.5-flash",
    instruction="Search the catalog and return a structured list of products.",
    tools=[search_catalog],
    output_schema=list[Product],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="catalog_demo")
    await runner.session_service.create_session(
        app_name="catalog_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Show me available widgets.",
        user_id="u1",
        session_id="s1",
    )
    final = events[-1]
    if final.actions and final.actions.set_model_response:
        products: list[Product] = final.actions.set_model_response
        for p in products:
            status = "In stock" if p.in_stock else "Out of stock"
            print(f"  {p.name}: ${p.price_usd:.2f} — {status}")


asyncio.run(main())
```

### Example 3 — subclassing SetModelResponseTool for custom post-processing

You can subclass `SetModelResponseTool` to add validation, normalization, or side effects after the LLM returns the structured result:

```python
import asyncio
from pydantic import BaseModel
from typing import Optional
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.adk.agents.context import ToolContext
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


class SentimentResult(BaseModel):
    label: str            # "POSITIVE", "NEGATIVE", "NEUTRAL"
    confidence: float     # 0.0 – 1.0
    reasoning: str


class AuditedSentimentTool(SetModelResponseTool):
    """Extends SetModelResponseTool to write audit log entries after classification."""

    async def run_async(self, *, args: dict, tool_context: ToolContext):
        # Let the base class validate and set the response
        result = await super().run_async(args=args, tool_context=tool_context)

        # Post-processing: write to audit state
        audit_log = tool_context.state.get("audit_log", [])
        if tool_context.actions.set_model_response:
            response: SentimentResult = tool_context.actions.set_model_response
            audit_log.append({
                "label": response.label,
                "confidence": response.confidence,
            })
            tool_context.state["audit_log"] = audit_log

        return result


def analyze_text(text: str) -> dict:
    """Analyze text features (stub)."""
    return {"word_count": len(text.split()), "has_exclamation": "!" in text}


agent = LlmAgent(
    name="sentiment_agent",
    model="gemini-2.5-flash",
    instruction="Classify the sentiment of text provided by the user.",
    tools=[
        analyze_text,
        AuditedSentimentTool(output_schema=SentimentResult),
    ],
    output_schema=SentimentResult,
)
```

> **Production tip:** If `output_schema` is set but there are **no other tools**, ADK uses normal text-based JSON output — `SetModelResponseTool` is only injected when tools coexist. This means your schema validation path differs between those two modes; test both.

> **Gotcha:** The LLM must call the synthetic `set_model_response` tool exactly once. If the model instead produces plain text (sometimes happens with weaker models), `set_model_response` will be `None` and `event.output` will be `None`. Add an assertion or fallback parser in production.

---

## 3 · `DynamicNodeScheduler` + `ctx.run_node()`

In a `Workflow`, edges between nodes are declared statically. `ctx.run_node()` is the runtime alternative — a `@node` function can call it to **dynamically schedule child nodes** at runtime, with full support for deduplication, resume-after-interrupt, and transfer_to_agent propagation.

### Three execution paths (from source)

When `ctx.run_node(node, input)` is called, `DynamicNodeScheduler` scans prior session events to determine which path to take:

| Path | Condition | Behaviour |
|------|-----------|-----------|
| **Fresh** | No prior events for this `run_id` | Execute the node normally |
| **Completed** | Prior events show a successful output | Return cached output immediately (fast-forward replay) |
| **Waiting** | Prior events show an unresolved interrupt | Propagate the interrupt upward or resolve it if a reply is available |

### Key data classes (from source)

```python
@dataclass
class DynamicNodeRun:
    state: NodeState           # status, interrupt IDs, run_id
    output: Any = None         # final output once completed
    task: asyncio.Task | None = None
    transfer_to_agent: str | None = None
    recovered_state: _ChildScanState | None = None  # from lazy event scan

@dataclass
class DynamicNodeState:
    runs: dict[str, DynamicNodeRun]   # keyed by node_path, e.g. "/wf@1/node_a@1"
    interrupt_ids: set[str]            # union of all unresolved interrupt IDs
```

### `DynamicNodeScheduler.__call__()` signature (from source)

```python
async def __call__(
    self,
    ctx: Context,
    node: BaseNode,
    node_input: Any,
    *,
    node_name: str | None = None,
    use_as_output: bool = False,
    run_id: str,
    use_sub_branch: bool = False,
    override_branch: str | None = None,
    override_isolation_scope: str | None = None,
) -> Context
```

| Parameter | Notes |
|-----------|-------|
| `node` | The child node to run. Must be accessible from the current workflow context. |
| `node_input` | The input passed to the child node's function. |
| `node_name` | Override the node name used in the event path. Defaults to `node.name`. |
| `use_as_output` | If `True`, the child's output becomes the current node's output (propagates up). |
| `run_id` | Stable identifier for this particular dynamic call. Must be unique per `(invocation, node_name)` pair and stable across resume attempts. |
| `use_sub_branch` | Create a sub-branch scope so child events are isolated in the session event stream. |
| `override_branch` | Manually set the branch string for this child run. |
| `override_isolation_scope` | Override the isolation scope for credential/state segregation. |

### Resumability internals

- `_reconstruct_node_states()` lazily scans prior session events to rebuild `DynamicNodeState` — this is O(events) per call so it runs once per scheduler invocation, not per node.
- `ReplaySequenceBarrier` ensures that when replaying completed nodes, they are re-emitted in the same chronological order as the original run, preventing race conditions in fan-out scenarios.
- Transfer support: if a dynamic node calls `transfer_to_agent`, `DynamicNodeScheduler` loops to execute the transfer target rather than returning control immediately.

### Example 1 — basic `ctx.run_node()` inside a workflow node

```python
import asyncio
from google.adk.workflow import Workflow, node, Context
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


async def fetch_user_profile(user_id: str) -> dict:
    """Simulate a database lookup."""
    return {"user_id": user_id, "name": "Alice", "tier": "premium"}


async def fetch_recommendations(user_profile: dict) -> list[str]:
    """Simulate a recommendation service call."""
    tier = user_profile.get("tier", "free")
    if tier == "premium":
        return ["Product A", "Product B", "Product C"]
    return ["Product D"]


@node
async def orchestrate(ctx: Context, user_id: str) -> dict:
    """Dynamically run sub-nodes based on input at runtime."""
    from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
    from google.adk.workflow import node as node_decorator

    # Build child nodes inline (or reference pre-defined nodes)
    @node_decorator
    async def profile_node(ctx: Context, uid: str) -> dict:
        return await fetch_user_profile(uid)

    @node_decorator
    async def recs_node(ctx: Context, profile: dict) -> list[str]:
        return await fetch_recommendations(profile)

    # Run profile_node dynamically
    profile = await ctx.run_node(
        profile_node,
        user_id,
        run_id="profile-fetch",
    )

    # Run recommendations_node with profile result
    recommendations = await ctx.run_node(
        recs_node,
        profile,
        run_id="recs-fetch",
    )

    return {"user": profile, "recommendations": recommendations}


async def main():
    wf = Workflow(
        name="dynamic_wf",
        nodes=[orchestrate],
        entry_node=orchestrate,
    )

    runner = InMemoryRunner(agent=wf, app_name="dynamic_demo")
    await runner.session_service.create_session(
        app_name="dynamic_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "user_id:alice",
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if event.output:
            print(event.output)


asyncio.run(main())
```

### Example 2 — dynamic fan-out: scheduling multiple nodes in parallel

```python
import asyncio
from google.adk.workflow import Workflow, node, Context
from google.adk.runners import InMemoryRunner


CITIES = ["London", "Tokyo", "New York", "Sydney"]


async def check_city_weather(city: str) -> dict:
    """Stub: replace with real weather API."""
    import random
    return {"city": city, "temp_c": round(random.uniform(5, 35), 1)}


@node
async def parallel_weather_check(ctx: Context, cities: list[str]) -> list[dict]:
    """Fan out to one dynamic node per city, then collect results."""
    from google.adk.workflow import node as node_decorator

    @node_decorator
    async def city_node(ctx: Context, city: str) -> dict:
        return await check_city_weather(city)

    # Schedule all city nodes concurrently
    tasks = []
    for city in cities:
        # run_id must be stable and unique per city
        task = ctx.run_node(
            city_node,
            city,
            run_id=f"weather-{city.lower().replace(' ', '-')}",
        )
        tasks.append(task)

    # Gather results (all run concurrently within the workflow)
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["city"])


@node
async def summarize_weather(ctx: Context, reports: list[dict]) -> str:
    """Format weather reports into a summary string."""
    lines = [f"  {r['city']}: {r['temp_c']}°C" for r in reports]
    return "Weather summary:\n" + "\n".join(lines)


async def main():
    wf = Workflow(
        name="weather_fanout",
        nodes=[parallel_weather_check, summarize_weather],
        entry_node=parallel_weather_check,
        edges=[(parallel_weather_check, summarize_weather)],
    )

    runner = InMemoryRunner(agent=wf, app_name="fanout_demo")
    await runner.session_service.create_session(
        app_name="fanout_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        CITIES,
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if isinstance(event.output, str) and "Weather" in event.output:
            print(event.output)


asyncio.run(main())
```

### Example 3 — `use_as_output=True` to propagate child output

When a node delegates its entire output to a child node, use `use_as_output=True` to propagate the child's output directly as the parent's output:

```python
from google.adk.workflow import node, Context


@node
async def enrichment_router(ctx: Context, user_id: str) -> dict:
    """Route to either fast or deep enrichment based on user tier."""
    from google.adk.workflow import node as node_decorator

    @node_decorator
    async def fast_enrich(ctx: Context, uid: str) -> dict:
        return {"user_id": uid, "mode": "fast", "data": {"score": 42}}

    @node_decorator
    async def deep_enrich(ctx: Context, uid: str) -> dict:
        # More expensive operation
        return {"user_id": uid, "mode": "deep", "data": {"score": 99, "insights": [...]}}

    tier = ctx.state.get(f"user:{user_id}:tier", "free")

    if tier == "premium":
        # use_as_output=True means enrichment_router's output == deep_enrich's output
        return await ctx.run_node(
            deep_enrich,
            user_id,
            run_id="deep-enrich",
            use_as_output=True,
        )
    else:
        return await ctx.run_node(
            fast_enrich,
            user_id,
            run_id="fast-enrich",
            use_as_output=True,
        )
```

> **Production tip:** The `run_id` parameter is critical for resumability. It must be **stable and unique** for a given dynamic call. Using positional loop indices (`run_id=f"step-{i}"`) works but is fragile if the input list changes between a pause and a resume. Prefer content-based IDs like `run_id=f"city-{city.lower()}"`.

> **Gotcha:** `ctx.run_node()` is only available inside `@node` functions in a `Workflow`. Calling it from an `LlmAgent` tool will raise an `AttributeError` — `ToolContext` does not have a `run_node` method.

---

## 4 · `BaseRetrievalTool` / `LlamaIndexRetrieval` / `FilesRetrieval` / `VertexAiRagRetrieval`

These four classes form the retrieval tool hierarchy. They give agents access to external knowledge bases — local file systems, LlamaIndex vector stores, or Vertex AI RAG corpora — through a uniform `query: str` → text interface.

### Class hierarchy

```
BaseTool
└── BaseRetrievalTool         (google.adk.tools.retrieval.base_retrieval_tool)
    ├── LlamaIndexRetrieval   (google.adk.tools.retrieval.llama_index_retrieval)
    │   └── FilesRetrieval    (google.adk.tools.retrieval.files_retrieval)
    └── VertexAiRagRetrieval  (google.adk.tools.retrieval.vertex_ai_rag_retrieval)
```

### `BaseRetrievalTool` (abstract)

Defines a single `query: str` parameter in `_get_declaration()`. Subclasses only implement `run_async(args, tool_context)`. It respects the JSON Schema feature flag:

```python
# From source:
if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
    # Uses JSON Schema declaration path
else:
    # Uses classic types.Schema path
```

### `LlamaIndexRetrieval` constructor

```python
from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval

LlamaIndexRetrieval(
    *,
    name: str,
    description: str,
    retriever: BaseRetriever,   # any LlamaIndex retriever
)
```

`run_async()` calls `self.retriever.retrieve(args['query'])` and returns the first result's `.text`. Requires `pip install llama-index-core`.

### `FilesRetrieval` constructor

```python
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

FilesRetrieval(
    *,
    name: str,
    description: str,
    input_dir: str,
    embedding_model: Optional[BaseEmbedding] = None,
)
```

| Parameter | Notes |
|-----------|-------|
| `input_dir` | Directory of files to index. Supports `.txt`, `.pdf`, `.md`, `.docx` (via `SimpleDirectoryReader`). |
| `embedding_model` | Defaults to `GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview", embed_batch_size=1)`. Requires `pip install llama-index-embeddings-google-genai`. |

Internally: `SimpleDirectoryReader(input_dir).load_data()` → `VectorStoreIndex.from_documents(docs)` → `.as_retriever()`.

The index is built once on first use and cached in the `FilesRetrieval` instance.

### `VertexAiRagRetrieval` constructor

```python
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.cloud import aiplatform_v1beta1 as rag

VertexAiRagRetrieval(
    *,
    name: str,
    description: str,
    rag_corpora: list[str] = None,        # list of corpus resource names
    rag_resources: list[rag.RagResource] = None,  # or RagResource objects
    similarity_top_k: int = None,
    vector_distance_threshold: float = None,
)
```

**Gemini 2.x path (from source):** For Gemini 2.x models, `VertexAiRagRetrieval` overrides `process_llm_request()` to inject a native `types.Tool(retrieval=types.Retrieval(vertex_rag_store=...))` directly into the LLM request. The retrieval happens inside the Gemini model's inference — no additional function-call round-trip is required.

**Gemini 1.x fallback:** For older models, it falls back to the standard function-call pattern: the tool declares a `query` parameter, executes the RAG query via the Vertex AI API, and returns retrieved text.

Install: `pip install google-cloud-aiplatform`

### Example 1 — `FilesRetrieval` with a local documentation directory

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

# Create sample documents
os.makedirs("/tmp/docs", exist_ok=True)
with open("/tmp/docs/faq.txt", "w") as f:
    f.write(
        "Q: What are your support hours?\n"
        "A: We offer 24/7 support for enterprise customers.\n\n"
        "Q: How do I reset my password?\n"
        "A: Visit account.example.com and click 'Forgot Password'.\n"
    )
with open("/tmp/docs/pricing.txt", "w") as f:
    f.write(
        "Starter Plan: $10/month — up to 5 users, 10GB storage.\n"
        "Pro Plan: $49/month — up to 50 users, 100GB storage.\n"
        "Enterprise Plan: contact sales for custom pricing.\n"
    )

retrieval_tool = FilesRetrieval(
    name="search_docs",
    description="Search the product documentation and FAQ for answers.",
    input_dir="/tmp/docs",
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a customer support agent. Use search_docs to answer questions "
        "about our product."
    ),
    tools=[retrieval_tool],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="docs_demo")
    await runner.session_service.create_session(
        app_name="docs_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the Pro plan price?",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — `VertexAiRagRetrieval` with a Vertex AI RAG corpus

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval

CORPUS_NAME = (
    "projects/my-gcp-project/locations/us-central1"
    "/ragCorpora/1234567890"
)

rag_tool = VertexAiRagRetrieval(
    name="search_knowledge_base",
    description=(
        "Search the company knowledge base for internal policies, "
        "procedures, and technical documentation."
    ),
    rag_corpora=[CORPUS_NAME],
    similarity_top_k=5,                  # return top 5 chunks
    vector_distance_threshold=0.7,       # minimum similarity score
)

agent = LlmAgent(
    name="internal_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "Answer employee questions using the company knowledge base. "
        "Always cite your source when possible."
    ),
    tools=[rag_tool],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="rag_demo")
    await runner.session_service.create_session(
        app_name="rag_demo", user_id="emp123", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the expense reimbursement policy for travel?",
        user_id="emp123",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 3 — custom `LlamaIndexRetrieval` with a Chroma vector store

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval

# pip install llama-index-vector-stores-chroma chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def build_chroma_retriever(collection_name: str, persist_dir: str):
    """Build a LlamaIndex retriever backed by a persistent Chroma collection."""
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_ctx
    )
    return index.as_retriever(similarity_top_k=3)


chroma_retriever = build_chroma_retriever(
    collection_name="product_docs",
    persist_dir="/data/chroma_db",
)

retrieval_tool = LlamaIndexRetrieval(
    name="search_product_docs",
    description="Search detailed product documentation using semantic similarity.",
    retriever=chroma_retriever,
)

agent = LlmAgent(
    name="docs_agent",
    model="gemini-2.5-flash",
    instruction="Answer technical questions using the product documentation.",
    tools=[retrieval_tool],
)
```

### Example 4 — combining multiple retrieval tools on one agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval

local_faq = FilesRetrieval(
    name="search_faq",
    description="Search frequently asked questions and quick-start guides.",
    input_dir="/app/docs/faq",
)

deep_kb = VertexAiRagRetrieval(
    name="search_knowledge_base",
    description=(
        "Search the full knowledge base for detailed technical "
        "specifications, API references, and architecture docs."
    ),
    rag_corpora=["projects/my-project/locations/us-central1/ragCorpora/abc123"],
    similarity_top_k=8,
)

agent = LlmAgent(
    name="tiered_support",
    model="gemini-2.5-flash",
    instruction=(
        "Answer questions using the available tools. "
        "Try search_faq first for common questions. "
        "Use search_knowledge_base for in-depth technical queries."
    ),
    tools=[local_faq, deep_kb],
)
```

> **Production tip:** `FilesRetrieval` builds its index in-process on first call. For large document sets this can take tens of seconds. Pre-warm the index at startup by calling `retrieval_tool._retriever` (which triggers lazy init) before serving requests.

> **Gotcha:** `VertexAiRagRetrieval` with Gemini 2.x injects a native retrieval tool, which changes the LLM request structure. This means it **cannot** coexist with `GoogleSearchTool` (both inject into the same `tools` field of the LLM request and Gemini enforces a one-native-tool-per-request limit). Use `GoogleSearchAgentTool` as a workaround if you need both.

---

## 5 · `FirestoreSessionService` + `FirestoreMemoryService`

Both services use Google Cloud Firestore for persistence. `FirestoreSessionService` replaces `DatabaseSessionService` for serverless/multi-process deployments; `FirestoreMemoryService` provides keyword-searchable long-term memory.

Install: `pip install google-cloud-firestore`

### `FirestoreSessionService`

**Module:** `google.adk.integrations.firestore.firestore_session_service`

**Constructor:**

```python
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
from google.cloud import firestore

FirestoreSessionService(
    client: Optional[firestore.AsyncClient] = None,
    root_collection: Optional[str] = None,
    # Default: "adk-session" or ADK_FIRESTORE_ROOT_COLLECTION env var
)
```

**Firestore document hierarchy (from source):**

```
adk-session/
└── <app_name>/
    └── users/
        └── <user_id>/
            └── sessions/
                └── <session_id>/
                    └── events/
                        └── <event_id>   ← one doc per event

app_states/<app_name>                                  ← app: prefixed state
user_states/<app_name>/users/<user_id>                 ← user: prefixed state
```

**Per-session locking (from source):** An `asyncio.Lock` is maintained per `(app_name, user_id, session_id)` tuple. `append_event()` acquires this lock before writing, serialising concurrent writes **within the same process**. The lock is reference-counted and cleaned up when no callers hold it (preventing memory leaks in long-running servers).

**State merge:** `_merge_state(app_state, user_state, session_state)` combines all three state dictionaries, prefixing app-level keys with `"app:"` and user-level keys with `"user:"` so they are accessible via `tool_context.state["app:key"]` and `tool_context.state["user:key"]`.

**Key methods:**

| Method | Notes |
|--------|-------|
| `create_session(app_name, user_id, session_id, state)` | Creates Firestore doc at the sessions path. |
| `get_session(app_name, user_id, session_id, config)` | Fetches session + all events, merges state scopes. |
| `list_sessions(app_name, user_id)` | Returns metadata for all sessions (no events). |
| `delete_session(app_name, user_id, session_id)` | Deletes session doc + all event subcollection docs. |
| `append_event(session, event)` | Acquires per-session lock, writes event doc, updates state docs. |

### `FirestoreMemoryService`

**Module:** `google.adk.integrations.firestore.firestore_memory_service`

**Constructor:**

```python
from google.adk.integrations.firestore.firestore_memory_service import (
    FirestoreMemoryService,
)

FirestoreMemoryService(
    client: Optional[firestore.AsyncClient] = None,
    events_collection: Optional[str] = None,   # default: "events"
    stop_words: Optional[set[str]] = None,     # default: DEFAULT_STOP_WORDS (English)
    memories_collection: Optional[str] = None, # default: "memories"
)
```

**Keyword extraction pipeline (from source):**
1. Each event's text content is extracted.
2. Regex `[A-Za-z]+` splits into words; lowercased.
3. Stop words are removed (English stop words by default).
4. Remaining words become the keywords list for the Firestore document.

**Memory document schema:**
```
memories/<app_name>/<user_id>/<hash>:
  {
    appName: str,
    userId: str,
    keywords: list[str],
    author: str,
    content: str,
    timestamp: float,
  }
```

**Write batching:** `add_session_to_memory()` uses Firestore's `batch()` API in 500-document chunks (Firestore's maximum batch size).

**Search:** `search_memory(app_name, user_id, query)` extracts keywords from `query`, then runs a Firestore `array_contains` query per keyword, de-duplicates results by content hash, and returns `MemoryResult` objects.

> **Note:** This is keyword-based search, **not** vector similarity search. For semantic search, implement `BaseMemoryService` with a vector database.

### Example 1 — `FirestoreSessionService` with an ADK `App`

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
from google.adk.runners import Runner


def greet_user(name: str) -> dict:
    """Greet the user by name."""
    return {"greeting": f"Hello, {name}! How can I help you today?"}


agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a friendly assistant. Greet users by name when they introduce themselves.",
    tools=[greet_user],
)

# FirestoreSessionService uses ADK_FIRESTORE_ROOT_COLLECTION env var if set
session_service = FirestoreSessionService(
    root_collection=os.environ.get("ADK_FIRESTORE_ROOT_COLLECTION", "adk-session"),
)


async def main():
    await session_service.create_session(
        app_name="my_app",
        user_id="user_alice",
        session_id="session_001",
        state={"user:preferred_name": "Alice"},
    )

    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service,
    )

    events = await runner.run_debug(
        "Hi! My name is Alice.",
        user_id="user_alice",
        session_id="session_001",
    )
    print(events[-1].content.parts[0].text)

    # Session persists in Firestore — resumable after process restart
    session = await session_service.get_session(
        app_name="my_app",
        user_id="user_alice",
        session_id="session_001",
    )
    print(f"Session has {len(session.events)} events in Firestore.")


asyncio.run(main())
```

### Example 2 — custom root collection via environment variable

```python
import os
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)

# Set in your deployment environment:
# ADK_FIRESTORE_ROOT_COLLECTION=prod-adk-sessions

session_service = FirestoreSessionService()
# → root_collection defaults to os.environ.get("ADK_FIRESTORE_ROOT_COLLECTION", "adk-session")

# Or pass a custom client for fine-grained control (custom credentials, database):
from google.cloud import firestore

custom_client = firestore.AsyncClient(
    project="my-gcp-project",
    database="my-custom-db",   # non-default Firestore database
)
session_service_custom = FirestoreSessionService(
    client=custom_client,
    root_collection="staging-adk-sessions",
)
```

### Example 3 — `FirestoreMemoryService` with custom stop words

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
from google.adk.integrations.firestore.firestore_memory_service import (
    FirestoreMemoryService,
)

# Add domain-specific stop words on top of the defaults
EXTRA_STOP_WORDS = {
    "the", "a", "an", "is", "it", "to", "of", "and", "or",
    # domain-specific noise words:
    "please", "thanks", "okay", "yes", "no", "hi", "hello",
}

memory_service = FirestoreMemoryService(
    stop_words=EXTRA_STOP_WORDS,
    memories_collection="agent-memories",
)

session_service = FirestoreSessionService()

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful assistant with long-term memory. "
        "Remember important facts about the user across conversations."
    ),
)


async def main():
    runner = Runner(
        agent=agent,
        app_name="memory_demo",
        session_service=session_service,
        memory_service=memory_service,
    )

    await session_service.create_session(
        app_name="memory_demo", user_id="u1", session_id="s1"
    )

    await runner.run_debug(
        "I'm an avid cyclist and I live in Amsterdam.",
        user_id="u1",
        session_id="s1",
    )

    # Persist session events to memory store
    session = await session_service.get_session(
        app_name="memory_demo", user_id="u1", session_id="s1"
    )
    await memory_service.add_session_to_memory(session)

    # Later, search for related memories
    results = await memory_service.search_memory(
        app_name="memory_demo",
        user_id="u1",
        query="cycling Amsterdam",
    )
    for r in results:
        print(r.content)


asyncio.run(main())
```

### Example 4 — combined Firestore session + memory in production `App`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
from google.adk.integrations.firestore.firestore_memory_service import (
    FirestoreMemoryService,
)


agent = LlmAgent(
    name="personal_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant. You have access to the user's memory "
        "from previous conversations. Reference past information when relevant."
    ),
)

session_svc = FirestoreSessionService(root_collection="prod-sessions")
memory_svc = FirestoreMemoryService(memories_collection="prod-memories")


async def main():
    runner = Runner(
        agent=agent,
        app_name="personal_assistant_app",
        session_service=session_svc,
        memory_service=memory_svc,
    )

    user_id = "user_bob"
    session_id = "session_2026_01"

    await session_svc.create_session(
        app_name="personal_assistant_app",
        user_id=user_id,
        session_id=session_id,
    )

    # Turn 1: share information
    await runner.run_debug(
        "I just moved to Berlin and started a new job at a fintech startup.",
        user_id=user_id,
        session_id=session_id,
    )

    # Persist to memory after each session
    session = await session_svc.get_session(
        app_name="personal_assistant_app",
        user_id=user_id,
        session_id=session_id,
    )
    await memory_svc.add_session_to_memory(session)

    # New session — memory search retrieves prior context
    session_id_2 = "session_2026_02"
    await session_svc.create_session(
        app_name="personal_assistant_app",
        user_id=user_id,
        session_id=session_id_2,
    )
    events = await runner.run_debug(
        "Do you remember where I'm living now?",
        user_id=user_id,
        session_id=session_id_2,
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

> **Production tip:** The in-process `asyncio.Lock` in `FirestoreSessionService` prevents race conditions when multiple coroutines in the same process append to the same session. However, it does **not** protect against concurrent writes from **multiple processes or instances**. For multi-instance deployments, implement optimistic locking at the Firestore document level using `update_time` conditions.

> **Gotcha:** `FirestoreMemoryService.search_memory()` performs one Firestore query per keyword. Queries over many keywords (long user inputs) can become slow and expensive. Tune your stop words list aggressively and consider capping the keyword list length.

---

## 6 · `BigQueryToolset` + `BigQueryToolConfig` + `WriteMode`

`BigQueryToolset` provides a suite of pre-built BigQuery tools that give agents the ability to query, analyze, and forecast over data in BigQuery. `BigQueryToolConfig` controls safety, billing, and access boundaries.

Install: `pip install google-cloud-bigquery`

**Module:** `google.adk.integrations.bigquery.bigquery_toolset`

### `BigQueryToolset` constructor

```python
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.integrations.bigquery.bigquery_tool_config import BigQueryToolConfig

BigQueryToolset(
    *,
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
    credentials_config: Optional[BigQueryCredentialsConfig] = None,
    bigquery_tool_config: Optional[BigQueryToolConfig] = None,
)
```

### Available tools (source-verified)

| Tool name | Description |
|-----------|-------------|
| `get_dataset_info` | Retrieve metadata about a BigQuery dataset |
| `get_table_info` | Schema, row count, and stats for a table |
| `list_dataset_ids` | All dataset IDs visible in the project |
| `list_table_ids` | All table IDs within a dataset |
| `get_job_info` | Status and results of a running or completed BQ job |
| `execute_sql` | Execute a SQL query and return results |
| `forecast` | BQML time-series forecasting on a table |
| `analyze_contribution` | Contribution analysis (which factors drive a metric) |
| `detect_anomalies` | BQML anomaly detection on time-series data |
| `ask_data_insights` | Natural language → data insights via Gemini in BigQuery |
| `search_catalog` | Search Data Catalog for relevant datasets and tables |

### `BigQueryToolConfig` fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `write_mode` | `WriteMode` | `WriteMode.BLOCKED` | Controls write permissions (see below) |
| `maximum_bytes_billed` | `Optional[int]` | `None` | Must be `>= 10_485_760` (10 MiB) if set. Guards against accidentally running expensive queries. |
| `max_query_result_rows` | `int` | `50` | Maximum rows returned to the agent from `execute_sql`. |
| `application_name` | `Optional[str]` | `None` | Appended to the BQ client user-agent header. No spaces allowed. |
| `compute_project_id` | `Optional[str]` | `None` | Override the project that executes and pays for queries. |
| `location` | `Optional[str]` | `None` | BigQuery dataset location (e.g. `"US"`, `"EU"`, `"asia-northeast1"`). |
| `job_labels` | `Optional[dict[str, str]]` | `None` | Labels applied to all BQ jobs. Max 20 labels. Keys starting with `"adk-bigquery-"` are reserved. |

### `WriteMode` enum

| Value | Description |
|-------|-------------|
| `WriteMode.BLOCKED` | SELECT queries only. DDL and DML are rejected before execution. Default; safest for read-only data analysis. |
| `WriteMode.PROTECTED` | Write operations are permitted only within BigQuery **session** anonymous datasets. Permanent tables are protected. Temp tables and CTEs are allowed. |
| `WriteMode.ALLOWED` | All write operations permitted. Use only with appropriate IAM controls and audit logging. |

### `BigQueryCredentialsConfig`

Uses OAuth2 scopes for BigQuery and Dataplex:

```python
BigQueryCredentialsConfig(
    auth_scheme=...,
    auth_credential=...,
    scopes=["bigquery", "dataplex.read-write"],
)
```

### Example 1 — read-only data analysis agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.integrations.bigquery.bigquery_tool_config import (
    BigQueryToolConfig,
    WriteMode,
)

config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED,           # SELECT only
    maximum_bytes_billed=100 * 1024 * 1024, # 100 MiB cap
    max_query_result_rows=100,
    location="US",
)

toolset = BigQueryToolset(bigquery_tool_config=config)

agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data analyst with access to BigQuery. "
        "Answer questions by querying the relevant tables. "
        "Always explain your findings clearly."
    ),
    tools=[toolset],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="bq_demo")
    await runner.session_service.create_session(
        app_name="bq_demo", user_id="analyst1", session_id="s1"
    )
    events = await runner.run_debug(
        "What are the top 10 products by revenue in the sales.transactions table?",
        user_id="analyst1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — `tool_filter` to expose only specific tools

```python
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.integrations.bigquery.bigquery_tool_config import (
    BigQueryToolConfig,
    WriteMode,
)

# Expose only discovery + query tools; hide ML tools and catalog search
config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED,
    max_query_result_rows=200,
)

toolset = BigQueryToolset(
    bigquery_tool_config=config,
    tool_filter=[
        "list_dataset_ids",
        "list_table_ids",
        "get_table_info",
        "execute_sql",
    ],
)
```

### Example 3 — `WriteMode.PROTECTED` for session-scoped temp tables

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.integrations.bigquery.bigquery_tool_config import (
    BigQueryToolConfig,
    WriteMode,
)

config = BigQueryToolConfig(
    write_mode=WriteMode.PROTECTED,     # temp tables allowed; permanent tables blocked
    maximum_bytes_billed=500_000_000,   # 500 MiB
    max_query_result_rows=1000,
    compute_project_id="my-billing-project",
    job_labels={
        "team": "data-engineering",
        "environment": "staging",
    },
)

toolset = BigQueryToolset(bigquery_tool_config=config)

agent = LlmAgent(
    name="etl_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You perform ETL operations. You may create temporary tables "
        "for intermediate results but cannot modify permanent tables."
    ),
    tools=[toolset],
)
```

### Example 4 — `job_labels` for cost tracking and billing attribution

```python
from google.adk.integrations.bigquery.bigquery_tool_config import BigQueryToolConfig
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset

config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED,
    application_name="ProductionAnalyticsAgent",   # no spaces
    job_labels={
        "cost-center": "analytics-team",
        "project": "q2-forecasting",
        "environment": "production",
        # Note: keys must not start with "adk-bigquery-" (reserved prefix)
    },
    maximum_bytes_billed=1_073_741_824,  # 1 GiB hard cap
)

toolset = BigQueryToolset(bigquery_tool_config=config)
```

> **Production tip:** Always set `maximum_bytes_billed` in production. Without it, a poorly-phrased agent query could scan a petabyte-scale table. Start with a low cap (e.g. 100 MiB) and raise it after measuring your workload's actual scan patterns.

> **Gotcha:** `max_query_result_rows` limits the rows **returned to the agent**, not the rows scanned. Setting it low (e.g. 50) doesn't reduce BQ scan costs — you still need `maximum_bytes_billed` for cost control. The two parameters are independent.

---

## 7 · `LangchainTool`

`LangchainTool` wraps any LangChain `BaseTool` (or tool-like object with a `.run` method) as an ADK `FunctionTool`. It preserves the original tool's name, description, and Pydantic `args_schema`, making it a drop-in bridge between the LangChain and ADK ecosystems.

Install: `pip install langchain-core`

**Module:** `google.adk.integrations.langchain.langchain_tool`

### Constructor

```python
from google.adk.integrations.langchain.langchain_tool import LangchainTool

LangchainTool(
    tool: Union[LangchainBaseTool, object],
    name: Optional[str] = None,        # overrides tool.name
    description: Optional[str] = None, # overrides tool.description
)
```

### Internal mechanism (from source)

| Step | Details |
|------|---------|
| **Function extraction** | For `StructuredTool`: uses `tool.func` (sync) or `tool.coroutine` (async). For other tools: uses `tool._run` or `tool.run`. |
| **Parameter filtering** | Appends `'run_manager'` to `_ignore_params` — this LangChain-internal arg is stripped before calling `build_function_declaration`. |
| **Schema** | If `tool.args_schema` is present: calls `build_function_declaration_for_langchain()` for accurate JSON schema. Otherwise: falls back to `super()._get_declaration()` with name/description overrides. |
| **Name priority** | Explicit `name` arg → `tool.name` → default FunctionTool name. |

### Example 1 — wrapping a DuckDuckGo search tool

```python
import asyncio
# pip install langchain-community duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchRun
from google.adk.integrations.langchain.langchain_tool import LangchainTool
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


ddg_search = DuckDuckGoSearchRun()

adk_search_tool = LangchainTool(
    tool=ddg_search,
    name="web_search",
    description=(
        "Search the web using DuckDuckGo. Use this for current events, "
        "facts, and information not in your training data."
    ),
)

agent = LlmAgent(
    name="research_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions using web search when needed.",
    tools=[adk_search_tool],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="lc_demo")
    await runner.session_service.create_session(
        app_name="lc_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What happened in the tech world this week?",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — wrapping a `StructuredTool` with Pydantic args_schema

```python
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from google.adk.integrations.langchain.langchain_tool import LangchainTool
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


class CalculatorInput(BaseModel):
    expression: str = Field(description="A mathematical expression to evaluate, e.g. '2 * (3 + 4)'")


def safe_calculate(expression: str) -> str:
    """Evaluate a simple mathematical expression safely."""
    try:
        # Use ast.literal_eval-based parsing in production for safety
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: invalid characters in expression"
        result = eval(expression)  # noqa: S307 — replace with safer parser in prod
        return str(result)
    except Exception as e:
        return f"Error: {e}"


lc_calculator = StructuredTool.from_function(
    func=safe_calculate,
    name="calculator",
    description="Evaluate mathematical expressions.",
    args_schema=CalculatorInput,
)

# ADK will use CalculatorInput.model_json_schema() for the function declaration
adk_calculator = LangchainTool(tool=lc_calculator)

agent = LlmAgent(
    name="math_agent",
    model="gemini-2.5-flash",
    instruction="Help with mathematical calculations.",
    tools=[adk_calculator],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="calc_demo")
    await runner.session_service.create_session(
        app_name="calc_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is 347 * 19 + 256?",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 3 — name and description override for better LLM prompting

LangChain tool names are often terse or internal. Override them with more descriptive names that match your domain:

```python
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from google.adk.integrations.langchain.langchain_tool import LangchainTool

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

# Original name: "wikipedia" — override with something more descriptive
adk_wiki = LangchainTool(
    tool=wiki_tool,
    name="lookup_encyclopedia",
    description=(
        "Look up factual information from Wikipedia. "
        "Best for historical events, scientific concepts, "
        "geographical facts, and biographical information. "
        "Input: a search query string."
    ),
)
```

### Example 4 — async LangChain tool

```python
import asyncio
import httpx
from langchain_core.tools import BaseTool
from google.adk.integrations.langchain.langchain_tool import LangchainTool
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


class AsyncHttpTool(BaseTool):
    name: str = "fetch_url"
    description: str = "Fetch the content of a URL via HTTP GET."

    def _run(self, url: str) -> str:
        """Sync fallback (not used when async is available)."""
        import requests
        return requests.get(url, timeout=10).text[:2000]

    async def _arun(self, url: str) -> str:
        """Async implementation — preferred by LangchainTool when available."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            return response.text[:2000]


# LangchainTool detects _arun and uses it as the coroutine
adk_http_tool = LangchainTool(
    tool=AsyncHttpTool(),
    name="fetch_webpage",
    description="Fetch and return the raw text content of a webpage URL.",
)

agent = LlmAgent(
    name="web_agent",
    model="gemini-2.5-flash",
    instruction="Fetch and summarise web pages when given URLs.",
    tools=[adk_http_tool],
)
```

> **Production tip:** If a LangChain tool has both `_run` and `_arun`, `LangchainTool` uses `_arun` (via `tool.coroutine`). This is the async path and avoids blocking the event loop. Always implement `_arun` for any tool that does I/O.

> **Gotcha:** LangChain's `run_manager` is stripped from all calls. If your LangChain tool relies on the `run_manager` for callbacks (e.g. streaming partial results back to LangChain), those callbacks will not fire. The tool still executes correctly; you just lose LangChain-side callback events.

---

## 8 · `CrewaiTool`

`CrewaiTool` adapts any CrewAI `BaseTool` to the ADK `FunctionTool` interface. CrewAI tools often use `**kwargs` parameter patterns and Pydantic schemas that require special handling — `CrewaiTool` manages this transparently.

Install: `pip install 'google-adk[extensions]'` (includes the crewai dependency)

**Module:** `google.adk.integrations.crewai.crewai_tool`

### Constructor

```python
from google.adk.integrations.crewai.crewai_tool import CrewaiTool

CrewaiTool(
    tool: CrewaiBaseTool,
    *,
    name: str,
    description: str = '',
)
```

Both `name` and `description` are required (unlike `LangchainTool` where they're optional). This is intentional — CrewAI tool names often contain spaces or special characters that are invalid in Gemini function names.

### Name sanitisation (from source)

ADK replaces spaces with underscores and lowercases the name:
```python
# "Directory Read Tool" → "directory_read_tool"
name = name.replace(" ", "_").lower()
```

### `run_async()` internals (from source)

```
is_var_keyword (**kwargs function)?
│
├─ YES: pass all args EXCEPT 'self' and 'tool_context'
│        then re-inject 'tool_context' IF it is an explicit parameter
│
└─ NO:  filter args to only the declared parameter names
        (strict matching — undeclared args are silently dropped)

On missing mandatory args:
  → return structured error string:
    "Error: missing required argument '{arg}'. Please provide: {arg}"
    (This prompts the LLM to retry with the correct parameters.)
```

### `_get_declaration()` (from source)

Uses `build_function_declaration_for_params_for_crewai()` with `tool.args_schema.model_json_schema()` — reads the Pydantic schema from the CrewAI tool directly, preserving field types and descriptions.

### Example 1 — wrapping a CrewAI `DirectoryReadTool`

```python
import asyncio
import os
# pip install crewai-tools
from crewai_tools import DirectoryReadTool
from google.adk.integrations.crewai.crewai_tool import CrewaiTool
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


os.makedirs("/tmp/project_files", exist_ok=True)
with open("/tmp/project_files/README.md", "w") as f:
    f.write("# My Project\n\nThis is a sample Python project.\n")
with open("/tmp/project_files/requirements.txt", "w") as f:
    f.write("fastapi>=0.100.0\nsqlalchemy>=2.0.0\npydantic>=2.0.0\n")

crewai_dir_tool = DirectoryReadTool(directory="/tmp/project_files")

adk_dir_tool = CrewaiTool(
    tool=crewai_dir_tool,
    name="list_project_files",
    description=(
        "List and read files in the project directory. "
        "Use this to explore the project structure and file contents."
    ),
)

agent = LlmAgent(
    name="project_analyst",
    model="gemini-2.5-flash",
    instruction="Analyze the project structure using the available tools.",
    tools=[adk_dir_tool],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="crewai_demo")
    await runner.session_service.create_session(
        app_name="crewai_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What files are in the project directory and what are the dependencies?",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — handling a `**kwargs` CrewAI tool

Some CrewAI tools accept arbitrary keyword arguments. `CrewaiTool` detects this and passes all args through:

```python
from crewai.tools import BaseTool as CrewaiBaseTool
from pydantic import BaseModel, Field
from google.adk.integrations.crewai.crewai_tool import CrewaiTool


class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    language: str = Field(default="en", description="Language code for results")


class FlexibleSearchTool(CrewaiBaseTool):
    name: str = "Flexible Search"
    description: str = "A flexible search tool with configurable parameters."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, **kwargs) -> str:
        """Uses **kwargs — CrewaiTool passes all declared args through."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        language = kwargs.get("language", "en")
        # Simulate search
        return (
            f"Search results for '{query}' "
            f"(max={max_results}, lang={language}): [result1, result2, ...]"
        )


adk_search = CrewaiTool(
    tool=FlexibleSearchTool(),
    name="flexible_search",
    description="Search with configurable parameters for query, result count, and language.",
)
```

### Example 3 — combining `LangchainTool` and `CrewaiTool` in one agent

```python
import asyncio
# LangChain tool
from langchain_community.tools import DuckDuckGoSearchRun
from google.adk.integrations.langchain.langchain_tool import LangchainTool

# CrewAI tool
from crewai_tools import FileReadTool
from google.adk.integrations.crewai.crewai_tool import CrewaiTool

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


web_search = LangchainTool(
    tool=DuckDuckGoSearchRun(),
    name="web_search",
    description="Search the web for current information.",
)

file_reader = CrewaiTool(
    tool=FileReadTool(),
    name="read_file",
    description="Read the contents of a local file given its path.",
)

agent = LlmAgent(
    name="research_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You can search the web and read local files. "
        "Use web_search for online information, read_file for local documents."
    ),
    tools=[web_search, file_reader],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="combo_demo")
    await runner.session_service.create_session(
        app_name="combo_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Search for the latest Python release notes and save a summary.",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

> **Production tip:** Always provide explicit `name` and `description` arguments to `CrewaiTool`. The inherited CrewAI name may contain spaces, uppercase letters, or characters that Gemini function declarations do not allow. The sanitisation (`replace(" ", "_").lower()`) handles spaces but not all special characters — provide a clean snake_case name directly.

> **Gotcha:** If a mandatory CrewAI tool argument is missing in the LLM's call, `CrewaiTool` returns an error string rather than raising an exception. This error string is sent back to the LLM as the tool response, prompting it to retry. This means missing-argument errors are soft and may cause extra round-trips. If your tool has complex required arguments, add explicit parameter descriptions to help the LLM construct valid calls.

---

## 9 · `SlackRunner`

`SlackRunner` deploys an ADK agent as a Slack bot. It integrates with the `slack-bolt` library's `AsyncApp` and handles the event routing, session management, and response update patterns required for a production Slack bot.

Install: `pip install 'google-adk[slack]'` (installs `slack-bolt` and `slack-sdk`)

**Module:** `google.adk.integrations.slack.slack_runner`

### Constructor

```python
from google.adk.integrations.slack.slack_runner import SlackRunner
from slack_bolt.app.async_app import AsyncApp

SlackRunner(
    runner: Runner,          # your configured ADK runner
    slack_app: AsyncApp,     # slack-bolt AsyncApp instance
)
```

### Event handling (from source)

| Event | Condition | Action |
|-------|-----------|--------|
| `app_mention` | Any mention of the bot | Always calls `_handle_message()` |
| `message` | DM (`channel_type == "im"`) | Calls `_handle_message()` |
| `message` | Thread reply (`thread_ts` is set) | Calls `_handle_message()` |
| `message` | Bot message (`bot_id` or `bot_profile`) | Ignored (prevents self-loops) |

### Session ID strategy (from source)

```python
# Threads: each thread is a separate ADK session
session_id = f"{channel_id}-{thread_ts}"

# DMs (no thread): channel itself is the session
session_id = channel_id
```

This means:
- Each Slack thread gets its own persistent conversation context.
- DMs maintain one continuous session per user.
- Users in the same channel but different threads have independent sessions.

### Response pattern (from source)

1. Post `"_Thinking..._"` immediately (shows activity to the user).
2. Run the ADK agent asynchronously, collecting streaming events.
3. Update the thinking message in-place with `chat_update()` as the final response arrives.
4. If the final event has text content, replace the placeholder with it.

### Example 1 — complete Slack bot with Socket Mode

Socket Mode uses a WebSocket connection — ideal for development and bots behind firewalls (no public URL required):

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.integrations.slack.slack_runner import SlackRunner
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler


# Build the ADK agent
agent = LlmAgent(
    name="slack_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful Slack bot. Keep responses concise and formatted "
        "for Slack (use *bold* and _italic_ Slack markdown, not HTML)."
    ),
)

session_service = InMemorySessionService()

adk_runner = Runner(
    agent=agent,
    app_name="slack_bot",
    session_service=session_service,
)

# Build the Slack app
slack_app = AsyncApp(
    token=os.environ["SLACK_BOT_TOKEN"],     # xoxb-... Bot User OAuth Token
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

# Wire ADK runner to Slack
slack_runner = SlackRunner(runner=adk_runner, slack_app=slack_app)


async def main():
    # Socket Mode uses SLACK_APP_TOKEN (xapp-...)
    handler = AsyncSocketModeHandler(
        slack_app,
        app_token=os.environ["SLACK_APP_TOKEN"],
    )
    print("Slack bot is running (Socket Mode)...")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2 — HTTP mode for production deployments

For production, use HTTP mode with a public endpoint (e.g. behind a load balancer or Cloud Run):

```python
import asyncio
import os
from aiohttp import web
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
from google.adk.integrations.slack.slack_runner import SlackRunner
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.aiohttp.async_handler import AsyncSlackRequestHandler


agent = LlmAgent(
    name="production_bot",
    model="gemini-2.5-flash",
    instruction="You are a production Slack assistant for the engineering team.",
)

# Use Firestore for durable session storage across instances
session_service = FirestoreSessionService(root_collection="prod-slack-sessions")

adk_runner = Runner(
    agent=agent,
    app_name="prod_slack_bot",
    session_service=session_service,
)

slack_app = AsyncApp(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

slack_runner = SlackRunner(runner=adk_runner, slack_app=slack_app)
slack_handler = AsyncSlackRequestHandler(slack_app)


async def handle_slack_events(request: web.Request) -> web.Response:
    """aiohttp route handler for Slack events."""
    return await slack_handler.handle(request)


app = web.Application()
app.router.add_post("/slack/events", handle_slack_events)

if __name__ == "__main__":
    web.run_app(app, port=int(os.environ.get("PORT", 8080)))
```

### Example 3 — multi-threaded Slack bot with per-thread session isolation

This pattern uses Firestore so sessions persist across bot restarts:

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
from google.adk.integrations.slack.slack_runner import SlackRunner
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler


def get_team_status(team_name: str) -> dict:
    """Look up engineering team on-call status (stub)."""
    statuses = {
        "backend": {"on_call": "Alice", "status": "available"},
        "frontend": {"on_call": "Bob", "status": "in_meeting"},
        "infra": {"on_call": "Carol", "status": "available"},
    }
    return statuses.get(team_name.lower(), {"error": "Team not found"})


agent = LlmAgent(
    name="eng_bot",
    model="gemini-2.5-flash",
    instruction=(
        "You assist the engineering team. Use get_team_status to find on-call staff. "
        "Remember context from earlier in the thread."
    ),
    tools=[get_team_status],
)

# Firestore persists sessions: each thread keeps its context across bot restarts
session_service = FirestoreSessionService(root_collection="eng-bot-sessions")

adk_runner = Runner(
    agent=agent,
    app_name="eng_slack_bot",
    session_service=session_service,
)

slack_app = AsyncApp(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

# SlackRunner creates session_id = f"{channel_id}-{thread_ts}" per thread
slack_runner = SlackRunner(runner=adk_runner, slack_app=slack_app)


async def main():
    handler = AsyncSocketModeHandler(
        slack_app,
        app_token=os.environ["SLACK_APP_TOKEN"],
    )
    await handler.start_async()


asyncio.run(main())
```

### Example 4 — custom session_id strategy via subclassing

If you need different isolation logic (e.g. per-user sessions regardless of thread), subclass `SlackRunner`:

```python
from typing import Optional
from google.adk.integrations.slack.slack_runner import SlackRunner


class UserScopedSlackRunner(SlackRunner):
    """Each user gets one persistent session across all channels and threads."""

    def _get_session_id(self, user_id: str, channel_id: str, thread_ts: Optional[str]) -> str:
        # Override: session is per-user, not per-thread
        return f"user-{user_id}"

    def _get_user_id(self, event: dict) -> str:
        return event.get("user", "unknown")
```

> **Production tip:** `SlackRunner` uses `InMemorySessionService` by default (sessions lost on restart). For production, swap in `FirestoreSessionService` to preserve thread context across deployments, scaling events, and bot restarts.

> **Gotcha:** Slack has a **3-second response deadline** for event acknowledgements. `SlackRunner` handles this by posting the `_Thinking...` message immediately, which satisfies Slack's deadline. The actual ADK agent run is async. If your agent takes more than ~90 seconds, Slack may time out the message update. Structure long-running work as asynchronous follow-up messages rather than a single update.

---

## 10 · `FileArtifactService`

`FileArtifactService` stores artifacts (binary files, documents, images, structured data) on the local filesystem. It is the recommended `BaseArtifactService` implementation for development and single-machine deployments.

**Module:** `google.adk.artifacts.file_artifact_service`

### Constructor

```python
from google.adk.artifacts.file_artifact_service import FileArtifactService

FileArtifactService(root_dir: str)
```

The `root_dir` is created lazily on the first `save_artifact()` call — no manual `mkdir` required.

### Directory structure (from source)

```
root_dir/
└── <app_name>/
    └── users/
        └── <user_id>/
            ├── sessions/
            │   └── <session_id>/
            │       └── <filename>/
            │           └── versions/
            │               ├── 0001.json   ← ArtifactVersion metadata
            │               └── 0001.bin    ← binary payload (inline_data artifacts)
            └── <filename>/                  ← user-scoped (session_id=None)
                └── versions/
                    ├── 0001.json
                    └── 0001.bin
```

### Key internal functions (from source)

| Function | Notes |
|----------|-------|
| `_iter_artifact_dirs(root)` | Generator that walks the filesystem and yields directories containing a `versions/` subdirectory. Used by `list_artifacts()`. |
| `_file_uri_to_path(uri)` | Converts `file://` URIs to `Path` objects. Used when an artifact's `Part` contains `file_data` (as opposed to `inline_data`). |
| `ensure_part(artifact)` | Normalises `dict \| types.Part` → `types.Part`. Handles camelCase JSON keys from API serialisation (e.g. `"mimeType"` → `mime_type`). |

### `ArtifactVersion` model

```python
class ArtifactVersion(BaseModel):
    version: int           # monotonically increasing (starts at 1)
    canonical_uri: str     # file:// URI pointing to the payload
    custom_metadata: dict  # arbitrary user-defined metadata
    create_time: float     # unix timestamp (time.time())
    mime_type: Optional[str]  # for binary payloads (inline_data)
```

### Key behaviours

| Behaviour | Details |
|-----------|---------|
| **Versioning** | Version numbers start at 1. Each `save_artifact()` increments: `max(existing) + 1`. |
| **User-scoped** | Pass `session_id=None` to store outside the sessions hierarchy — shared across all sessions for that user. |
| **Thread safety** | Per-file `asyncio.Lock` (in-process only). Not safe for concurrent writes from multiple OS processes. |
| **Lazy directory creation** | `root_dir` and all parent directories are created on first write. |
| **Payload types** | `inline_data` (bytes embedded in the `Part`) → `.bin` file. `file_data` (URI reference) → stored as metadata only. |

### `BaseArtifactService` API (from source)

| Method | Signature | Notes |
|--------|-----------|-------|
| `save_artifact` | `(app_name, user_id, session_id, filename, artifact) → int` | Returns the new version number. |
| `load_artifact` | `(app_name, user_id, session_id, filename, version=None) → types.Part \| None` | `version=None` returns latest. |
| `list_artifacts` | `(app_name, user_id, session_id) → list[str]` | Returns filenames (not versions). |
| `delete_artifact` | `(app_name, user_id, session_id, filename) → None` | Deletes all versions. |
| `list_versions` | `(app_name, user_id, session_id, filename) → list[ArtifactVersion]` | All version metadata. |

### Example 1 — basic save and load

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.context import ToolContext
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.genai import types


def generate_report(title: str, content: str, tool_context: ToolContext) -> dict:
    """Generate a text report and save it as an artifact."""
    report_text = f"# {title}\n\n{content}\n"
    artifact = types.Part(
        inline_data=types.Blob(
            mime_type="text/plain",
            data=report_text.encode("utf-8"),
        )
    )
    version = tool_context.save_artifact(f"report_{title.lower().replace(' ', '_')}.txt", artifact)
    return {"saved": True, "version": version, "filename": f"report_{title}.txt"}


def read_report(filename: str, tool_context: ToolContext) -> str:
    """Read a previously saved report artifact."""
    artifact = tool_context.load_artifact(filename)
    if artifact is None:
        return f"Report '{filename}' not found."
    if artifact.inline_data and artifact.inline_data.data:
        return artifact.inline_data.data.decode("utf-8")
    return "Could not read report content."


agent = LlmAgent(
    name="report_agent",
    model="gemini-2.5-flash",
    instruction="Generate and retrieve text reports on request.",
    tools=[generate_report, read_report],
)


async def main():
    session_service = InMemorySessionService()
    artifact_service = FileArtifactService(root_dir="/tmp/adk_artifacts")

    runner = Runner(
        agent=agent,
        app_name="artifact_demo",
        session_service=session_service,
        artifact_service=artifact_service,
    )

    await session_service.create_session(
        app_name="artifact_demo", user_id="u1", session_id="s1"
    )

    # Generate a report
    await runner.run_debug(
        "Generate a report titled 'Q2 Summary' with content 'Revenue grew 15% QoQ.'",
        user_id="u1",
        session_id="s1",
    )

    # Read it back
    events = await runner.run_debug(
        "Read back the Q2 Summary report.",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — user-scoped vs session-scoped artifacts

```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.genai import types


async def demonstrate_artifact_scopes():
    session_service = InMemorySessionService()
    artifact_service = FileArtifactService(root_dir="/tmp/scoped_artifacts")

    # Create two sessions for the same user
    await session_service.create_session(
        app_name="scope_demo", user_id="alice", session_id="session_A"
    )
    await session_service.create_session(
        app_name="scope_demo", user_id="alice", session_id="session_B"
    )

    profile_data = types.Part(
        inline_data=types.Blob(
            mime_type="application/json",
            data=b'{"name": "Alice", "tier": "premium"}',
        )
    )

    session_data = types.Part(
        inline_data=types.Blob(
            mime_type="text/plain",
            data=b"Session A scratch notes",
        )
    )

    # User-scoped: session_id=None → accessible from any session
    v1 = await artifact_service.save_artifact(
        app_name="scope_demo",
        user_id="alice",
        session_id=None,          # ← user-scoped
        filename="user_profile.json",
        artifact=profile_data,
    )
    print(f"User profile saved as version {v1}")

    # Session-scoped: session_id="session_A"
    v2 = await artifact_service.save_artifact(
        app_name="scope_demo",
        user_id="alice",
        session_id="session_A",   # ← session-scoped
        filename="scratch_notes.txt",
        artifact=session_data,
    )
    print(f"Session notes saved as version {v2}")

    # User profile is accessible from session_B
    profile = await artifact_service.load_artifact(
        app_name="scope_demo",
        user_id="alice",
        session_id=None,          # must use None to access user-scoped
        filename="user_profile.json",
    )
    print(f"Profile accessible cross-session: {profile is not None}")

    # Session notes are NOT accessible from session_B (different scope)
    notes_from_b = await artifact_service.load_artifact(
        app_name="scope_demo",
        user_id="alice",
        session_id="session_B",
        filename="scratch_notes.txt",
    )
    print(f"Session A notes visible from session B: {notes_from_b is not None}")  # False


asyncio.run(demonstrate_artifact_scopes())
```

### Example 3 — listing versions and accessing a specific version

```python
import asyncio
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.genai import types


async def version_management_demo():
    service = FileArtifactService(root_dir="/tmp/versioned_artifacts")

    app_name = "version_demo"
    user_id = "bob"
    session_id = "s1"
    filename = "data_export.csv"

    # Save multiple versions
    for i in range(1, 4):
        csv_data = f"id,value\n{i},row_{i}\n".encode("utf-8")
        artifact = types.Part(
            inline_data=types.Blob(mime_type="text/csv", data=csv_data)
        )
        version = await service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            artifact=artifact,
        )
        print(f"Saved version {version}")

    # List all versions with metadata
    versions = await service.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    for v in versions:
        print(f"  v{v.version}: created={v.create_time:.0f}, mime={v.mime_type}")

    # Load latest (version=None)
    latest = await service.load_artifact(
        app_name=app_name, user_id=user_id, session_id=session_id, filename=filename
    )
    print(f"\nLatest: {latest.inline_data.data.decode()}")

    # Load a specific version
    v1 = await service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        version=1,
    )
    print(f"Version 1: {v1.inline_data.data.decode()}")

    # List all artifact filenames in the session
    filenames = await service.list_artifacts(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    print(f"\nAll artifacts in session: {filenames}")


asyncio.run(version_management_demo())
```

### Example 4 — migrating from `InMemoryArtifactService` to `FileArtifactService`

The two services share the same `BaseArtifactService` interface, making migration straightforward:

```python
import asyncio
import os
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.agents import LlmAgent
from google.adk.agents.context import ToolContext
from google.genai import types


def save_document(content: str, name: str, tool_context: ToolContext) -> dict:
    artifact = types.Part(
        inline_data=types.Blob(
            mime_type="text/plain",
            data=content.encode("utf-8"),
        )
    )
    version = tool_context.save_artifact(f"{name}.txt", artifact)
    return {"saved": True, "filename": f"{name}.txt", "version": version}


agent = LlmAgent(
    name="doc_agent",
    model="gemini-2.5-flash",
    instruction="Save documents on request.",
    tools=[save_document],
)


def build_runner(persist: bool = False) -> Runner:
    """Toggle between in-memory (dev) and file-based (prod) artifact storage."""
    session_service = InMemorySessionService()

    if persist:
        artifact_service = FileArtifactService(
            root_dir=os.environ.get("ARTIFACT_ROOT", "/var/adk/artifacts")
        )
    else:
        artifact_service = InMemoryArtifactService()

    return Runner(
        agent=agent,
        app_name="doc_manager",
        session_service=session_service,
        artifact_service=artifact_service,
    )


async def main():
    # Development: in-memory (fast, no disk I/O)
    dev_runner = build_runner(persist=False)

    # Production: file-based (durable, survives restart)
    # prod_runner = build_runner(persist=True)

    runner = dev_runner
    await runner.session_service.create_session(
        app_name="doc_manager", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Save a document called 'notes' with content 'Hello, world!'",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

> **Production tip:** `FileArtifactService` uses an `asyncio.Lock` per file to prevent concurrent write corruption. This works correctly for concurrent coroutines in a single process but **does not** prevent race conditions between multiple OS processes (e.g. multiple Cloud Run instances). For multi-process deployments, use `GcsArtifactService` (backed by Google Cloud Storage) which provides object-level consistency guarantees.

> **Gotcha:** Artifact filenames are path segments in the filesystem, but they are **not** sanitised. A filename like `"../../etc/passwd"` would be a path traversal vulnerability. Validate and sanitise filenames before passing them to `save_artifact()` in any user-facing application.

---

## Summary table

| # | Class | Key use case | Experimental? |
|---|-------|-------------|---------------|
| 1 | `InvocationContext` | Root data structure for one invocation — access in plugins and workflow nodes | No |
| 2 | `SetModelResponseTool` | Structured output with other tools coexisting — LLM "calls" a synthetic tool | No |
| 3 | `DynamicNodeScheduler` / `ctx.run_node()` | Runtime-scheduled child nodes with full dedup and resume support | No |
| 4 | `BaseRetrievalTool` / `LlamaIndexRetrieval` / `FilesRetrieval` / `VertexAiRagRetrieval` | Plug-in knowledge retrieval from local files, LlamaIndex stores, or Vertex AI RAG | No |
| 5 | `FirestoreSessionService` / `FirestoreMemoryService` | Scalable serverless session and keyword-searchable memory storage | No |
| 6 | `BigQueryToolset` / `BigQueryToolConfig` / `WriteMode` | Data analysis, BQML forecasting, and anomaly detection with safety controls | No |
| 7 | `LangchainTool` | Wrap any LangChain tool as an ADK function tool | No |
| 8 | `CrewaiTool` | Wrap any CrewAI tool as an ADK function tool with `**kwargs` handling | No |
| 9 | `SlackRunner` | Deploy an ADK agent as a Slack bot with thread-scoped sessions | No |
| 10 | `FileArtifactService` | Local filesystem artifact storage with versioning and scoped access | No |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-30 | google-adk 2.1.0 | Initial publication. All 10 class groups source-verified against installed `google-adk==2.1.0`. |

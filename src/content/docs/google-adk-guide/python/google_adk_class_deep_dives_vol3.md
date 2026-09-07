---
title: "Class Deep Dives Vol. 3 — v2.8.0"
description: "Source-verified deep dives for 10 classes in google-adk 2.8.0: RunConfig, BuiltInPlanner, PlanReActPlanner, BasePlugin, ReflectAndRetryToolPlugin, ContextFilterPlugin, VertexAiMemoryBankService, VertexAiCodeExecutor, ConversationScenarios / LlmBackedUserSimulatorConfig, and SaveFilesAsArtifactsPlugin."
framework: google-adk
language: python
sidebar:
  order: 125
---

All examples and field tables on this page are source-verified against **google-adk==2.8.0** (installed, introspected with `inspect.getsource`). The ten classes cover runtime control, planning, the plugin system, memory, code execution, and simulation-based evaluation — areas underrepresented in earlier guides.

| # | Class / Symbol | Module | Subject |
|---|---|---|---|
| 1 | `RunConfig` + `ToolThreadPoolConfig` | `google.adk.agents.run_config` | Per-invocation runtime knobs |
| 2 | `BuiltInPlanner` | `google.adk.planners.built_in_planner` | Gemini built-in thinking |
| 3 | `PlanReActPlanner` | `google.adk.planners.plan_re_act_planner` | Model-agnostic plan-then-ReAct |
| 4 | `BasePlugin` | `google.adk.plugins.base_plugin` | Global lifecycle interceptors |
| 5 | `ReflectAndRetryToolPlugin` | `google.adk.plugins.reflect_retry_tool_plugin` | Self-healing tool error recovery |
| 6 | `ContextFilterPlugin` | `google.adk.plugins.context_filter_plugin` | Context-window trimming |
| 7 | `VertexAiMemoryBankService` | `google.adk.memory.vertex_ai_memory_bank_service` | Production long-term memory |
| 8 | `VertexAiCodeExecutor` | `google.adk.code_executors.vertex_ai_code_executor` | Sandboxed code execution |
| 9 | `ConversationScenarios` + `LlmBackedUserSimulatorConfig` | `google.adk.evaluation` | LLM-driven evaluation |
| 10 | `SaveFilesAsArtifactsPlugin` | `google.adk.plugins.save_files_as_artifacts_plugin` | File upload → artifact pipeline |

---

## 1 — `RunConfig` + `ToolThreadPoolConfig`

**Module:** `google.adk.agents.run_config`

`RunConfig` is the per-invocation control object passed to `runner.run_async()`. It lets callers tune streaming, safety limits, telemetry, and concurrency without touching the agent definition.

### Field reference

Source-verified from `google/adk/agents/run_config.py`:

| Field | Type | Default | What it controls |
|---|---|---|---|
| `streaming_mode` | `StreamingMode` | `NONE` | `NONE` = batch; `SSE` = server-sent events; `BIDI` = bidirectional (live) |
| `max_llm_calls` | `int` | env `ADK_MAX_LLM_CALLS` or internal default | Hard cap on LLM calls per invocation; ≤0 → no cap (dangerous) |
| `response_modalities` | `list[types.Modality]` | `None` | `[types.Modality.TEXT]` or `[types.Modality.AUDIO]` — overrides agent default; for `BIDI` live sessions Gemini accepts exactly one modality (both together is rejected) |
| `http_options` | `types.HttpOptions \| None` | `None` | Per-invocation HTTP options (custom headers, timeouts, etc.) |
| `labels` | `dict[str, str] \| None` | `None` | User-defined billing/attribution labels for this invocation |
| `tool_thread_pool_config` | `ToolThreadPoolConfig \| None` | `None` | Run tools in a thread pool; see below |
| `context_window_compression` | `ContextWindowCompressionConfig \| None` | `None` | Gemini-side sliding-window compression |
| `get_session_config` | `GetSessionConfig \| None` | `None` | Limit how many events are loaded from the session store |
| `model_input_context` | `list[types.Content] \| None` | `None` | Transient extra context for this turn; not persisted to session |
| `telemetry` | `TelemetryConfig \| None` | `None` | Per-request OTel override (multi-tenant use) |
| `custom_metadata` | `dict[str, Any] \| None` | `None` | Arbitrary key-value metadata for this invocation (forwarded to spans; distinct from `labels`) |
| `include_thoughts_from_other_agents` | `bool` | `False` | Expose sub-agent reasoning to parent agent |
| `support_cfc` | `bool` | `False` | Compositional Function Calling (experimental; forces LIVE API) |
| `session_resumption` | `SessionResumptionConfig \| None` | `None` | Transparent session resumption for live sessions |

`ToolThreadPoolConfig` has one field: `max_workers: int = 4`. When set on `RunConfig.tool_thread_pool_config`, each tool call runs in a background thread, keeping the event loop free to process interrupts and audio.

### Basic text invocation

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a concise assistant.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )

    run_config = RunConfig(
        response_modalities=[types.Modality.TEXT],
        max_llm_calls=10,
        custom_metadata={"team": "backend", "environment": "staging"},
    )

    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Summarise the ADK in one sentence.")]
    )

    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=user_msg,
        run_config=run_config,
    ):
        if event.content and event.is_final_response():
            for part in event.content.parts:
                if part.text:
                    print(part.text)

asyncio.run(main())
```

### Thread-pool for blocking I/O tools

`tool_thread_pool_config` only takes effect in live (BIDI) sessions where the
event loop must stay free to process audio interrupts. Set
`streaming_mode=StreamingMode.BIDI` alongside it; in regular `run_async` batch
sessions the field is ignored.

```python
import asyncio
import time
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

def slow_database_lookup(query: str) -> dict:
    """Simulates a blocking database query."""
    time.sleep(2)  # Would stall the event loop without thread pool
    return {"results": [f"record for {query}"]}

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Use the database tool to answer questions.",
    tools=[FunctionTool(func=slow_database_lookup)],
)

# tool_thread_pool_config is live-only: pair with StreamingMode.BIDI
run_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
    max_llm_calls=5,
)
```

### Loading only recent session events

```python
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.base_session_service import GetSessionConfig

# Only load the last 50 events — crucial for long-running sessions
# with EventsCompactionConfig enabled.
run_config = RunConfig(
    get_session_config=GetSessionConfig(num_recent_events=50),
    max_llm_calls=20,
)
```

---

## 2 — `BuiltInPlanner`

**Module:** `google.adk.planners.built_in_planner`

`BuiltInPlanner` wraps Gemini's native thinking feature. You pass a `ThinkingConfig` and the planner injects it into every `LlmRequest` before the model call. The model emits `thought=True` parts that ADK strips from the user-visible response.

### Constructor

```python
BuiltInPlanner(*, thinking_config: types.ThinkingConfig)
```

`ThinkingConfig` fields used in practice:

| Field | Type | Description |
|---|---|---|
| `include_thoughts` | `bool` | Whether to return thought parts in the response |
| `thinking_budget` | `int \| None` | Max thinking tokens (budget); `None` = auto |

### Minimal thinking agent

```python
from google.adk.agents import LlmAgent
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio

planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=8192,
    )
)

agent = LlmAgent(
    name="thinking_agent",
    model="gemini-2.5-flash",
    instruction="Solve maths problems step by step.",
    planner=planner,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="think_demo")
    session = await runner.session_service.create_session(
        app_name="think_demo", user_id="u1"
    )
    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the 15th Fibonacci number?")]
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=user_msg
    ):
        if event.content:
            for part in event.content.parts:
                if part.thought:
                    print(f"[THINKING] {part.text[:80]}...")
                elif part.text and event.is_final_response():
                    print(f"[ANSWER] {part.text}")

asyncio.run(main())
```

### Multi-agent setup with thinking on the orchestrator only

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools import FunctionTool
from google.genai import types

def web_search(query: str) -> dict:
    return {"results": f"Top result for: {query}"}

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Search the web and return raw facts.",
    tools=[FunctionTool(func=web_search)],
)

# Only the orchestrator uses the built-in planner
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-pro",
    instruction="Coordinate research and synthesise a final answer.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=16384)
    ),
    sub_agents=[researcher],
)
```

### Notes

- `BuiltInPlanner.build_planning_instruction()` returns `None` — it does not add system-prompt text. The thinking happens inside the model via the `thinking_config` injected into `LlmRequest.config`.
- Supported only on `gemini-2.5-*` and later models. Passing `ThinkingConfig` to earlier models raises an API error.
- `PlanReActPlanner` (below) is the alternative when thinking is unavailable.

---

## 3 — `PlanReActPlanner`

**Module:** `google.adk.planners.plan_re_act_planner`

`PlanReActPlanner` is a prompt-engineering approach that works on *any* model. It adds a structured system instruction that tells the model to: (1) write a plan, (2) interleave tool calls with reasoning, and (3) emit a final answer — all tagged so the planner can strip internal scaffolding from the user-visible response.

### How it works internally

Source-verified from `google/adk/planners/plan_re_act_planner.py`:

1. `build_planning_instruction()` returns a multi-section prompt covering plan format (`/*PLANNING*/`), action format (`/*ACTION*/`), reasoning format (`/*REASONING*/`), and final answer (`/*FINAL_ANSWER*/`).
2. `process_planning_response()` parses each response part, strips planning tags, marks planning/reasoning text as `thought=True`, and returns only the first group of function calls plus the final answer to the caller.

### Usage

```python
from google.adk.agents import LlmAgent
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio

def get_weather(city: str) -> dict:
    return {"temp_c": 18, "condition": "partly cloudy", "city": city}

def get_flights(origin: str, destination: str) -> dict:
    return {"cheapest_usd": 230, "duration_h": 2.5}

agent = LlmAgent(
    name="travel_planner",
    model="gemini-2.5-flash",
    instruction="Help users plan trips.",
    planner=PlanReActPlanner(),
    tools=[
        FunctionTool(func=get_weather),
        FunctionTool(func=get_flights),
    ],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="travel")
    session = await runner.session_service.create_session(
        app_name="travel", user_id="u1"
    )
    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text="I want to fly from London to Barcelona next Saturday. "
                 "What's the weather like and how much will a flight cost?"
        )]
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=user_msg
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Choosing between `BuiltInPlanner` and `PlanReActPlanner`

| | `BuiltInPlanner` | `PlanReActPlanner` |
|---|---|---|
| Model support | Gemini 2.5+ only | Any model |
| Thinking tokens | Counted separately (efficient) | Uses output tokens for reasoning |
| Instruction overhead | Zero — thinking is native | Adds ~500-token system prompt |
| Thought visibility | `part.thought=True` parts | Tagged text; stripped by planner |
| Best for | Production on Gemini 2.5 | Non-Gemini models or Gemini 2.0 |

---

## 4 — `BasePlugin`

**Module:** `google.adk.plugins.base_plugin`

`BasePlugin` is the abstract base for all ADK plugins. Plugins differ from per-agent callbacks: they are registered on `App(plugins=[...])` and apply to every agent in the hierarchy, executing *before* per-agent callbacks. A non-`None` return from any plugin callback short-circuits all remaining plugins and the agent's own callbacks.

### Callback lifecycle order (source-verified)

```
on_user_message_callback
before_run_callback
  before_agent_callback
    before_model_callback
    after_model_callback | on_model_error_callback
    before_tool_callback
    after_tool_callback | on_tool_error_callback
  after_agent_callback | on_agent_error_callback
after_run_callback | on_run_error_callback
```

> **`on_event_callback` fires per-event, not once per run.** It is called for every `Event` the runner produces — model responses, tool calls, agent transfers — *before* each event is persisted to the session service and yielded to the caller. It therefore interleaves throughout the invocation rather than firing once at the end. The diagram above shows only the major ordering; `on_event_callback` fires inside the loop each time an event is emitted.

### Implementing a metrics plugin

```python
import time
from typing import Any, Optional
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

class MetricsPlugin(BasePlugin):
    """Tracks LLM call count, total tokens, and tool call latency."""

    def __init__(self):
        super().__init__(name="metrics")
        self._llm_calls = 0
        self._total_tokens = 0
        self._tool_start: dict[str, float] = {}

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        self._llm_calls += 1
        return None  # always proceed

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        if llm_response.usage_metadata:
            self._total_tokens += (
                llm_response.usage_metadata.total_token_count or 0
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        # function_call_id is unique per tool call, so parallel calls to the
        # same tool within one invocation don't overwrite each other's start time.
        self._tool_start[tool_context.function_call_id] = time.monotonic()
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        start = self._tool_start.pop(tool_context.function_call_id, None)
        if start is not None:
            elapsed = time.monotonic() - start
            print(f"[metrics] tool={tool.name} latency={elapsed:.3f}s")
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict[str, Any]]:
        # Pop the start time to avoid unbounded growth when tools fail.
        start = self._tool_start.pop(tool_context.function_call_id, None)
        if start is not None:
            elapsed = time.monotonic() - start
            print(f"[metrics] tool={tool.name} FAILED after {elapsed:.3f}s: {error}")
        return None  # let ADK propagate the error normally

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        print(
            f"[metrics] llm_calls={self._llm_calls} "
            f"total_tokens={self._total_tokens}"
        )


# Register with the runner
from google.adk.runners import InMemoryRunner
from google.adk.agents import LlmAgent
from google.adk.apps import App

agent = LlmAgent(name="agent", model="gemini-2.5-flash", instruction="Help.")
app = App(name="metrics_demo", root_agent=agent, plugins=[MetricsPlugin()])
runner = InMemoryRunner(app=app)
```

### Caching plugin (short-circuit pattern)

```python
import hashlib, json
from typing import Optional
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

class SemanticCachePlugin(BasePlugin):
    """Returns cached LlmResponse for identical prompts (demo-grade)."""

    def __init__(self):
        super().__init__(name="semantic_cache")
        self._cache: dict[str, LlmResponse] = {}

    def _cache_key(self, llm_request: LlmRequest, agent_name: str) -> str:
        # Include model, agent identity, generation config, and contents so
        # agents with the same name but different instructions/tools/settings
        # don't share cached responses.
        config_dump = (
            llm_request.config.model_dump() if llm_request.config else {}
        )
        payload = json.dumps(
            {
                "model": llm_request.model,
                "agent": agent_name,
                "config": config_dump,
                "contents": [c.model_dump() for c in llm_request.contents],
            },
            sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        key = self._cache_key(llm_request, callback_context.agent_name)
        if key in self._cache:
            print("[cache] HIT")
            return self._cache[key]  # short-circuits the actual LLM call
        # Key the state slot by agent_name so sibling agents under ParallelAgent
        # don't overwrite each other's pending keys in shared invocation state.
        # The "temp:" prefix keeps this out of the persisted session state.
        callback_context.state[f"temp:cache_key:{callback_context.agent_name}"] = key
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        # In streaming mode after_model_callback fires for each partial chunk
        # and then a final turn_complete marker that may carry no content.
        # Cache only when partial is not set AND there is actual content.
        if llm_response.partial or not llm_response.content:
            return None
        key = callback_context.state.get(f"temp:cache_key:{callback_context.agent_name}")
        if key:
            self._cache[key] = llm_response
            print(f"[cache] STORED key={key[:8]}…")
        return None
```

---

## 5 — `ReflectAndRetryToolPlugin`

**Module:** `google.adk.plugins.reflect_retry_tool_plugin`

`ReflectAndRetryToolPlugin` intercepts tool errors (exceptions *and* error-shaped dicts) and feeds a structured reflection prompt back to the LLM so it can self-correct and retry — without the caller needing to restart the invocation.

### Constructor parameters

Source-verified from `google/adk/plugins/reflect_retry_tool_plugin.py`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"reflect_retry_tool_plugin"` | Plugin identifier |
| `max_retries` | `int` | `3` | Max consecutive failures before giving up; `0` = no retry |
| `throw_exception_if_retry_exceeded` | `bool` | `True` | Re-raise the final error vs. return guidance to LLM |
| `tracking_scope` | `TrackingScope` | `INVOCATION` | `INVOCATION` (per-run) or `GLOBAL` (process-wide) |

### Default usage

```python
from google.adk.agents import LlmAgent
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin,
    TrackingScope,
)
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool

call_count = 0

def flaky_api(item_id: str) -> dict:
    """Fails the first two calls, succeeds on the third."""
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError(f"Transient timeout for item_id={item_id!r}")
    return {"item_id": item_id, "price": 42.0}

agent = LlmAgent(
    name="shop_agent",
    model="gemini-2.5-flash",
    instruction="Use the API to look up item prices.",
    tools=[FunctionTool(func=flaky_api)],
)

from google.adk.apps import App

app = App(
    name="retry_demo",
    root_agent=agent,
    plugins=[
        ReflectAndRetryToolPlugin(
            max_retries=3,
            throw_exception_if_retry_exceeded=False,
        )
    ],
)
runner = InMemoryRunner(app=app)
```

### Custom error detection in successful responses

Some APIs return HTTP 200 with an error body like `{"status": "error", "message": "quota exceeded"}`. Override `extract_error_from_result` to catch these:

```python
from typing import Any, Optional
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

class QuotaAwareRetryPlugin(ReflectAndRetryToolPlugin):
    async def extract_error_from_result(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> Optional[dict[str, Any]]:
        if isinstance(result, dict) and result.get("status") == "error":
            return result  # triggers reflection + retry
        return None  # success
```

### `TrackingScope.GLOBAL` for shared rate-limit tracking

```python
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin,
    TrackingScope,
)

# All concurrent invocations share a single failure counter per tool.
# Useful when every invocation calls the same rate-limited external API.
plugin = ReflectAndRetryToolPlugin(
    max_retries=5,
    tracking_scope=TrackingScope.GLOBAL,
    throw_exception_if_retry_exceeded=False,
)
```

---

## 6 — `ContextFilterPlugin`

**Module:** `google.adk.plugins.context_filter_plugin`

`ContextFilterPlugin` trims the `LlmRequest.contents` list *before* it reaches the model, preventing context-window overflows in long conversations. It counts by invocation (user-turn → response cycle), not by raw message count.

### Constructor parameters

Source-verified from `google/adk/plugins/context_filter_plugin.py`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_invocations_to_keep` | `int \| None` | `None` | Keep only the N most recent invocations |
| `custom_filter` | `Callable[[list[Content]], list[Content]] \| None` | `None` | Arbitrary transform on the content list |
| `name` | `str` | `"context_filter_plugin"` | Plugin identifier |
| `remove_amount` | `int` | `1` | How many invocations to drop when the limit is hit |

### Keep last 10 invocations

```python
from google.adk.agents import LlmAgent
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="long_chat",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

from google.adk.apps import App

app = App(
    name="chat",
    root_agent=agent,
    plugins=[
        ContextFilterPlugin(
            num_invocations_to_keep=10,
            remove_amount=2,  # drop 2 invocations when over the limit
        )
    ],
)
runner = InMemoryRunner(app=app)
```

### Custom filter — strip large tool outputs

```python
from google.genai import types
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

MAX_TOOL_OUTPUT_CHARS = 500

def trim_large_tool_outputs(
    contents: list[types.Content],
) -> list[types.Content]:
    trimmed = []
    for content in contents:
        new_parts = []
        for part in content.parts:
            if part.function_response:
                # Truncate oversized tool outputs
                resp = part.function_response
                text = str(resp.response)
                if len(text) > MAX_TOOL_OUTPUT_CHARS:
                    truncated = text[:MAX_TOOL_OUTPUT_CHARS] + "…[truncated]"
                    # Rebuild with truncated response
                    new_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=resp.id,
                                name=resp.name,
                                response={"output": truncated},
                            )
                        )
                    )
                    continue
            new_parts.append(part)
        trimmed.append(
            types.Content(role=content.role, parts=new_parts)
        )
    return trimmed

plugin = ContextFilterPlugin(
    num_invocations_to_keep=20,
    custom_filter=trim_large_tool_outputs,
)
```

### Combining with `EventsCompactionConfig`

`ContextFilterPlugin` trims what goes *into* the LLM request on each call. `EventsCompactionConfig` (on the `App`) compacts the session's stored event history. Use both together to control costs at both layers:

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

# App holds the agent, compaction config, and app-wide plugins.
# Session and artifact services belong on the Runner, not the App.
app = App(
    name="cost_controlled",
    root_agent=agent,  # required field; use root_agent, not agent
    # Compact stored events when prompt tokens hit 4000; keep the last 0
    # raw events un-compacted after each compaction cycle.
    events_compaction_config=EventsCompactionConfig(
        token_threshold=4000,
        event_retention_size=0,
    ),
    plugins=[
        # Also trim the live LLM request to the last 15 invocations
        ContextFilterPlugin(num_invocations_to_keep=15),
    ],
)

# Services go on the Runner
runner = Runner(
    app=app,
    session_service=InMemorySessionService(),
    artifact_service=InMemoryArtifactService(),
)
```

---

## 7 — `VertexAiMemoryBankService`

**Module:** `google.adk.memory.vertex_ai_memory_bank_service`

`VertexAiMemoryBankService` stores and retrieves cross-session memories using Vertex AI's managed Memory Bank. It is the production alternative to `InMemoryMemoryService` — memories persist across process restarts. Operations are scoped by both `app_name` and `user_id`, so each user's memories are isolated from all other users.

### Constructor parameters

Source-verified from `google/adk/memory/vertex_ai_memory_bank_service.py`:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `project` | `str \| None` | Yes (or ADC) | GCP project ID |
| `location` | `str \| None` | Yes | e.g. `"us-central1"` |
| `agent_engine_id` | `str` | Yes | ID portion only — e.g. `"456"` from `…/reasoningEngines/456` |
| `express_mode_api_key` | `str \| None` | No | For Express Mode deployments |
| `credentials` | `Credentials \| None` | No | Override ADC (Workload Identity, etc.) |

### Setup and wiring

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService
from google.genai import types

PROJECT = "my-gcp-project"
LOCATION = "us-central1"
# Extract from: agent_engine.api_resource.name.split("/")[-1]
AGENT_ENGINE_ID = "123456789"

memory_service = VertexAiMemoryBankService(
    project=PROJECT,
    location=LOCATION,
    agent_engine_id=AGENT_ENGINE_ID,
)

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant. "
        "Use your memory to recall previous conversations."
    ),
    tools=[
        # Automatically injects relevant memories into the system prompt
        PreloadMemoryTool(),
    ],
)

async def main():
    session_service = VertexAiSessionService(
        project=PROJECT, location=LOCATION, agent_engine_id=AGENT_ENGINE_ID
    )
    runner = Runner(
        agent=agent,
        app_name="memory_demo",
        session_service=session_service,
        memory_service=memory_service,
    )

    # First session — agent learns the user's name
    session_a = await session_service.create_session(
        app_name="memory_demo", user_id="alice"
    )
    msg1 = types.Content(
        role="user",
        parts=[types.Part.from_text(text="My name is Alice and I love hiking.")]
    )
    async for _ in runner.run_async(
        user_id="alice", session_id=session_a.id, new_message=msg1
    ):
        pass

    # Persist session to memory bank
    await memory_service.add_session_to_memory(
        await session_service.get_session(
            app_name="memory_demo", user_id="alice", session_id=session_a.id
        )
    )

    # Second session — agent recalls from memory
    session_b = await session_service.create_session(
        app_name="memory_demo", user_id="alice"
    )
    msg2 = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What do you know about me?")]
    )
    async for event in runner.run_async(
        user_id="alice", session_id=session_b.id, new_message=msg2
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Adding events incrementally

Use `add_events_to_memory` to push individual events without loading the entire session:

```python
from google.adk.events.event import Event
from google.genai import types

# Push a single event to the memory bank
event = Event(
    author="user",
    content=types.Content(
        role="user",
        parts=[types.Part.from_text(text="I prefer Python over JavaScript.")]
    ),
)
await memory_service.add_events_to_memory(
    app_name="memory_demo",
    user_id="alice",
    events=[event],
    session_id="session-xyz",
    custom_metadata={"source": "chat_widget"},
)
```

---

## 8 — `VertexAiCodeExecutor`

**Module:** `google.adk.code_executors.vertex_ai_code_executor`

`VertexAiCodeExecutor` delegates code execution to Vertex AI's Code Interpreter Extension — a managed, network-isolated sandbox that supports file I/O and produces images, CSVs, and text output.

### Field reference (`BaseCodeExecutor` fields, inherited)

| Field | Type | Default | Description |
|---|---|---|---|
| `resource_name` | `str \| None` | `None` | Reuse an existing extension; `None` = create on first call |
| `stateful` | `bool` | `False` | Keep interpreter state across calls within a session |
| `optimize_data_file` | `bool` | `False` | Auto-attach CSV files from the request |
| `error_retry_attempts` | `int` | `2` | Retry on consecutive code errors |
| `timeout_seconds` | `int \| None` | `None` | Per-execution timeout |

### Basic data-analysis agent

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import asyncio

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

artifact_service = InMemoryArtifactService()

code_executor = VertexAiCodeExecutor(
    stateful=True,        # interpreter state shared across turns in the session
    optimize_data_file=True,  # auto-attach CSV user uploads
    error_retry_attempts=3,
)

agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data analyst. When asked to analyse data, write and run "
        "Python code. Use pandas and matplotlib."
    ),
    code_executor=code_executor,
)

async def main():
    # VertexAiCodeExecutor saves generated files (images, CSVs) via the
    # runner's artifact service — they do NOT appear as inline_data parts
    # on the final response.  Wire an artifact service so the runner can
    # persist them.
    runner = Runner(
        agent=agent,
        app_name="code_demo",
        session_service=InMemorySessionService(),
        artifact_service=artifact_service,
    )
    session = await runner.session_service.create_session(
        app_name="code_demo", user_id="u1"
    )
    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text="Generate a list of 20 random numbers and plot a histogram."
        )]
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=user_msg
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    print(part.text)

    # Retrieve generated files from the artifact service after the run
    artifacts = await artifact_service.list_artifact_keys(
        app_name="code_demo", user_id="u1", session_id=session.id
    )
    for name in artifacts:
        artifact = await artifact_service.load_artifact(
            app_name="code_demo", user_id="u1",
            session_id=session.id, filename=name,
        )
        if artifact and artifact.inline_data:
            with open(name, "wb") as f:
                f.write(artifact.inline_data.data)
            print(f"Saved {name}")

asyncio.run(main())
```

### Reusing an existing extension

```python
# After the first run, Vertex AI creates an Extension resource.
# Reuse it to avoid re-provisioning time:
executor = VertexAiCodeExecutor(
    resource_name=(
        "projects/my-project/locations/us-central1/extensions/456789"
    ),
    stateful=True,
)
```

### Choosing a code executor

| Executor | Where it runs | Persistence | Use when |
|---|---|---|---|
| `UnsafeLocalCodeExecutor` | Local process | No | Local dev only — no sandboxing |
| `BuiltInCodeExecutor` | Gemini model-side | No | Gemini 2.5+ native code execution |
| `VertexAiCodeExecutor` | Vertex AI Extension | Yes (stateful=True) | Production; file I/O; images |
| `AgentEngineSandboxCodeExecutor` | Agent Engine managed | Yes | Deployed on Vertex Agent Engine |

---

## 9 — `ConversationScenarios` + `LlmBackedUserSimulatorConfig`

**Module:** `google.adk.evaluation.conversation_scenarios`, `google.adk.evaluation.simulation.llm_backed_user_simulator`

These two classes form the LLM-driven simulation testing pipeline: `ConversationScenarios` holds the test scripts (start prompt + conversation plan), and `LlmBackedUserSimulatorConfig` configures the simulated user that plays them out against your agent.

### `ConversationScenario` fields

Source-verified from `google/adk/evaluation/conversation_scenarios.py`:

| Field | Type | Required | Description |
|---|---|---|---|
| `starting_prompt` | `str` | Yes | First user message the agent receives |
| `conversation_plan` | `str` | Yes | Instructions the user simulator follows to complete the goal |
| `user_persona` | `UserPersona \| str \| None` | No | A named persona or `UserPersona` instance |

### `LlmBackedUserSimulatorConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"gemini-2.5-flash"` | LLM that plays the user |
| `max_allowed_invocations` | `int` | `20` | Cap on turns before the simulation stops |
| `include_function_calls` | `bool` | `False` | Show tool calls to the simulator |
| `custom_instructions` | `str \| None` | `None` | Jinja2 template replacing default simulator prompt |

### End-to-end simulation test

```python
import asyncio
from google.adk.evaluation.conversation_scenarios import (
    ConversationScenario,
    ConversationScenarios,
)
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulatorConfig,
)
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# --- Agent under test ---
def book_flight(origin: str, destination: str, date: str) -> dict:
    return {"confirmation": "ABC123", "price_usd": 199}

travel_agent = LlmAgent(
    name="travel_agent",
    model="gemini-2.5-flash",
    instruction="Help users book flights. Confirm details before booking.",
    tools=[FunctionTool(func=book_flight)],
)
# AgentEvaluator loads root_agent from the module; expose it at module level.
root_agent = travel_agent

# --- Test scenario ---
scenarios = ConversationScenarios(
    scenarios=[
        ConversationScenario(
            starting_prompt="I need to fly from New York to London.",
            conversation_plan=(
                "You want to book a round trip from JFK to LHR on 2025-03-15, "
                "returning on 2025-03-22. Budget is under $1500. "
                "If the agent asks for confirmation, say yes. "
                "Your goal is complete once you have a confirmation code."
            ),
        ),
        ConversationScenario(
            starting_prompt="Can you help me find a cheap flight to Tokyo?",
            conversation_plan=(
                "You want a one-way economy flight from SFO to NRT on 2025-04-10. "
                "Budget is under $800. Accept any option under budget. "
                "Goal complete when you have a confirmation number."
            ),
        ),
    ]
)

# --- Simulator config ---
simulator_config = LlmBackedUserSimulatorConfig(
    model="gemini-2.5-flash",
    max_allowed_invocations=15,
    include_function_calls=False,
)

# --- Build EvalSet from scenarios ---
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet

eval_set = EvalSet(
    eval_set_id="flight_booking_eval",
    eval_cases=[
        EvalCase(
            eval_id=f"case_{i}",
            conversation_scenario=scenario,
        )
        for i, scenario in enumerate(scenarios.scenarios)
    ],
)

# --- Run evaluation ---
# evaluate_eval_set is async; agent_module must be an importable module path
# string. The module loader looks for root_agent on the module (or on an
# "agent" attribute of the module); expose root_agent at the top level.
async def main():
    eval_config = EvalConfig(
        # multi_turn_task_success_v1 is reference-free: it judges whether the
        # agent completed the scenario goal, suitable for simulation-based evals.
        criteria={"multi_turn_task_success_v1": 0.7},
        user_simulator_config=simulator_config,
    )
    await AgentEvaluator.evaluate_eval_set(
        agent_module="my_package.travel_agent",  # module that exports root_agent
        eval_set=eval_set,
        eval_config=eval_config,
        print_detailed_results=True,
    )

asyncio.run(main())
```

### Custom `UserPersona`

```python
from google.adk.evaluation.simulation.user_simulator_personas import (
    UserPersona,
    UserBehavior,
)

impatient_user = UserPersona(
    id="impatient_traveller",
    description="A frequent flyer who is time-pressed and expects quick answers.",
    behaviors=[
        UserBehavior(
            name="brief_responses",
            description="Gives very short answers.",
            behavior_instructions=[
                "Keep your messages under 15 words.",
                "Do not explain your reasoning.",
            ],
            violation_rubrics=[
                "Response is longer than 15 words.",
                "Response contains explanation or elaboration.",
            ],
        ),
    ],
)

scenario_with_persona = ConversationScenario(
    starting_prompt="Book me a flight to Paris.",
    conversation_plan="Book a one-way flight LHR→CDG on 2025-05-01. Say yes to all prompts.",
    user_persona=impatient_user,
)
```

---

## 10 — `SaveFilesAsArtifactsPlugin`

**Module:** `google.adk.plugins.save_files_as_artifacts_plugin`

`SaveFilesAsArtifactsPlugin` intercepts user messages that contain embedded binary blobs (images, PDFs, audio) and saves each blob as an artifact before the agent sees the message. Each blob is replaced in the message with a `[Uploaded Artifact: "name"]` placeholder so the model knows the file was uploaded. When `attach_file_reference=True` (the default), a `FileData` part with the artifact's URI/reference is also appended to the message. The URI format depends on the backing `ArtifactService` — GCS for `GcsArtifactService`, an `artifact://` reference for `InMemoryArtifactService`.

### Constructor parameters

Source-verified from `google/adk/plugins/save_files_as_artifacts_plugin.py`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"save_files_as_artifacts_plugin"` | Plugin identifier |
| `attach_file_reference` | `bool` | `True` | `True` (default): saves the blob as an artifact, replaces it in the user message with a placeholder text part AND appends a `FileData` part containing the artifact URI/reference (URI format depends on the backing `ArtifactService`). `False`: saves the artifact and adds the placeholder text only — no `FileData` part is appended. |

### How naming and scope work

- The artifact name comes from `blob.display_name`.
- Names **without** the `user:` prefix are session-scoped — addressed by `session_id`.
- Names **with** the `user:` prefix are user-scoped — addressable across sessions for that user.
- Each `save_artifact` call creates a **new version** of the artifact; prior versions remain retrievable by version index. The latest version is used by default when loading.

### Wiring the plugin

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

artifact_service = InMemoryArtifactService()

agent = LlmAgent(
    name="file_agent",
    model="gemini-2.5-flash",
    instruction=(
        "When the user uploads a file, acknowledge it by name. "
        "Use the load_artifacts tool to read the file contents if needed."
    ),
)

app = App(
    name="file_demo",
    root_agent=agent,
    plugins=[SaveFilesAsArtifactsPlugin(attach_file_reference=True)],
)
runner = Runner(
    app=app,
    session_service=InMemorySessionService(),
    artifact_service=artifact_service,
)
```

### Sending a file in a user message

```python
import asyncio
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def main():
    artifact_service = InMemoryArtifactService()
    app = App(
        name="file_demo",
        root_agent=agent,
        plugins=[SaveFilesAsArtifactsPlugin()],
    )
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        artifact_service=artifact_service,
    )
    session = await runner.session_service.create_session(
        app_name="file_demo", user_id="u1"
    )

    # Read a PDF and attach it
    with open("report.pdf", "rb") as f:
        pdf_bytes = f.read()

    user_msg = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="Please summarise this document."),
            types.Part(
                inline_data=types.Blob(
                    mime_type="application/pdf",
                    data=pdf_bytes,
                    display_name="report.pdf",  # becomes the artifact name
                )
            ),
        ],
    )

    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=user_msg
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### User-scoped persistent files

```python
# Prefix the display_name with "user:" for cross-session persistence.
user_avatar = types.Part(
    inline_data=types.Blob(
        mime_type="image/png",
        data=avatar_bytes,
        display_name="user:avatar.png",  # survives session end
    )
)
```

### Pairing with `load_artifacts`

`SaveFilesAsArtifactsPlugin` saves the file; the agent retrieves it at inference time via the `load_artifacts` built-in tool:

```python
from google.adk.tools import load_artifacts

agent = LlmAgent(
    name="doc_agent",
    model="gemini-2.5-flash",
    instruction=(
        "When a file reference appears in the conversation, "
        "call load_artifacts to read it before answering."
    ),
    tools=[load_artifacts],
)
```

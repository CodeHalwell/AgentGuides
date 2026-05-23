---
title: "Class deep dives (10 key classes)"
description: "Source-verified deep dives into 10 core google-adk 2.1.0 classes: LlmAgent, RunConfig, Context/ToolContext, BasePlugin, App, Workflow, BaseNode/Node, FunctionTool, RetryConfig, and BaseTool."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives"
  order: 60
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name and signature is taken directly from the installed package source.

---

## 1 · `LlmAgent`

`google.adk.agents.LlmAgent` — the only LLM-backed agent in the framework (also re-exported as `Agent`). It is a Pydantic `BaseModel`, so every field is a validated constructor kwarg.

### Field reference (2.1.0)

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Must be a Python identifier; used for agent-transfer routing |
| `model` | `str \| BaseLlm` | `""` | `""` inherits from parent; built-in default is `gemini-2.5-flash` |
| `description` | `str` | `""` | Shown to parent agents deciding whether to transfer |
| `instruction` | `str \| Callable` | `""` | Supports `{state_key}` placeholders; callable receives `ReadonlyContext` |
| `global_instruction` | same | `""` | **Deprecated** — use `GlobalInstructionPlugin` instead |
| `static_instruction` | `types.ContentUnion \| None` | `None` | Prefix injected before `instruction`; Gemini context-cache friendly |
| `tools` | `list[Callable \| BaseTool \| BaseToolset]` | `[]` | Callables auto-wrapped as `FunctionTool` |
| `generate_content_config` | `types.GenerateContentConfig \| None` | `None` | Temperature, safety settings, thinking config, etc. |
| `mode` | `'chat' \| 'task' \| 'single_turn' \| None` | `None` | `None` = framework infers; root agent should be `'chat'` or `'task'`; sub-agents default to `'single_turn'` |
| `input_schema` / `output_schema` | Pydantic model or schema | `None` | Setting `output_schema` disables tool use |
| `output_key` | `str \| None` | `None` | Writes final text to `session.state[output_key]` |
| `include_contents` | `'default' \| 'none'` | `'default'` | `'none'` → stateless, no prior history injected |
| `planner` | `BasePlanner \| None` | `None` | `BuiltInPlanner` or `PlanReActPlanner` |
| `code_executor` | `BaseCodeExecutor \| None` | `None` | e.g. `BuiltInCodeExecutor` for sandboxed Python |
| `disallow_transfer_to_parent` | `bool` | `False` | Prevents the agent from handing back to parent |
| `disallow_transfer_to_peers` | `bool` | `False` | Prevents transfer to sibling agents |
| `before_model_callback` / `after_model_callback` / `on_model_error_callback` | fn or `list[fn]` | `None` | See [callbacks page](./callbacks-and-plugins/) |
| `before_tool_callback` / `after_tool_callback` / `on_tool_error_callback` | fn or `list[fn]` | `None` | Same |
| `before_agent_callback` / `after_agent_callback` | fn or `list[fn]` | `None` | Inherited from `BaseAgent` |

### Example 1 — basic chat agent with structured output

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

class AnalysisResult(BaseModel):
    sentiment: str          # "positive" | "neutral" | "negative"
    score: float            # 0.0 – 1.0
    key_topics: list[str]

agent = LlmAgent(
    name="analyser",
    model="gemini-2.5-flash",
    instruction="Analyse the sentiment and key topics in the user's text. Return JSON.",
    output_schema=AnalysisResult,  # disables tool use; forces structured output
    output_key="last_analysis",    # also writes to session.state["last_analysis"]
    mode="single_turn",            # sub-agent semantics: runs once, then exits
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="nlp")
    await runner.session_service.create_session(
        app_name="nlp", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "I absolutely love the new product! It changed my workflow.",
        user_id="u1", session_id="s1",
    )
    # session.state["last_analysis"] now contains the structured result
    session = await runner.session_service.get_session(
        app_name="nlp", user_id="u1", session_id="s1"
    )
    print(session.state.get("last_analysis"))

asyncio.run(main())
```

### Example 2 — dynamic instruction with state placeholders

`LlmAgent.instruction` supports both a static string with `{state_key}` placeholders and a fully dynamic async callable.

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

# --- Option A: placeholder string (simplest) ---
# session.state["user_lang"] and session.state["user_name"] injected at runtime
agent_a = LlmAgent(
    name="greeter",
    model="gemini-2.5-flash",
    instruction="Hello {user_name}! Always reply in {user_lang}.",
)

# --- Option B: async callable ---
async def dynamic_instruction(ctx: ReadonlyContext) -> str:
    lang = ctx.state.get("user:lang", "English")
    tier = ctx.state.get("user:tier", "free")
    base = f"Respond in {lang}."
    if tier == "premium":
        base += " Offer detailed reasoning and citations."
    return base

agent_b = LlmAgent(
    name="personalised",
    model="gemini-2.5-flash",
    instruction=dynamic_instruction,  # called fresh every turn
)
```

### Example 3 — multi-agent hierarchy with agent transfer

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

search_agent = LlmAgent(
    name="searcher",
    model="gemini-2.5-pro",
    description="Searches the web and returns relevant facts.",
    instruction="Use Google Search to answer the question. Return your sources.",
    tools=[google_search],
    mode="single_turn",
    disallow_transfer_to_parent=False,  # can hand back to coordinator
)

summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    description="Distils long text into a concise summary.",
    instruction="Summarise the provided text in 3 sentences maximum.",
    mode="single_turn",
)

# The coordinator orchestrates via agent-transfer
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "Route research questions to 'searcher', "
        "then pass results to 'summariser'."
    ),
    sub_agents=[search_agent, summariser],
    mode="chat",
)
```

### Example 4 — `generate_content_config` (temperature + thinking)

```python
from google.genai import types
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="careful_reasoner",
    model="gemini-2.5-pro",
    instruction="Solve the mathematical puzzle step by step.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(
            thinking_budget=10_000,  # tokens of internal reasoning
        ),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            )
        ],
    ),
)
```

### Example 5 — `static_instruction` for context-cache prefixes

Use `static_instruction` for content that never changes across sessions (large reference text, company policy, etc.). Gemini treats it as a cacheable prefix.

```python
from google.genai import types
from google.adk.agents import LlmAgent

POLICY_TEXT = """[…hundreds of lines of company policy…]"""

agent = LlmAgent(
    name="policy_bot",
    model="gemini-2.5-flash",
    static_instruction=types.Part(text=POLICY_TEXT),  # cached prefix
    instruction="Answer questions using only the policy above. Be concise.",
)
```

---

## 2 · `RunConfig` + `StreamingMode` + `ToolThreadPoolConfig`

`google.adk.agents.run_config.RunConfig` — per-invocation configuration. Pass to `runner.run_async(..., run_config=config)`.

### Field reference (2.1.0)

| Field | Type | Default | Notes |
|---|---|---|---|
| `streaming_mode` | `StreamingMode` | `NONE` | `NONE`, `SSE`, or `BIDI` |
| `max_llm_calls` | `int` | `500` | Safety cap on total LLM calls per invocation; `≤0` = unlimited (not recommended) |
| `tool_thread_pool_config` | `ToolThreadPoolConfig \| None` | `None` | Runs tools in a thread pool (useful for live/bidirectional sessions) |
| `context_window_compression` | `types.ContextWindowCompressionConfig \| None` | `None` | Enables Gemini-side context compression |
| `get_session_config` | `GetSessionConfig \| None` | `None` | Limits events fetched when loading session (useful with compaction) |
| `custom_metadata` | `dict[str, Any] \| None` | `None` | Merged into every emitted `Event.custom_metadata` |
| `speech_config` | `types.SpeechConfig \| None` | `None` | Voice config for live (bidi) mode |
| `output_audio_transcription` | `types.AudioTranscriptionConfig \| None` | default factory | Transcribes audio output |
| `input_audio_transcription` | `types.AudioTranscriptionConfig \| None` | default factory | Transcribes audio input |
| `realtime_input_config` | `types.RealtimeInputConfig \| None` | `None` | Realtime input for live sessions |
| `enable_affective_dialog` | `bool \| None` | `None` | Model adapts tone to detected user emotion |
| `proactivity` | `types.ProactivityConfig \| None` | `None` | Allows model to proactively respond |
| `session_resumption` | `types.SessionResumptionConfig \| None` | `None` | Transparent session resumption config |
| `save_live_blob` | `bool` | `False` | Saves live video/audio to session + artifact service |
| `support_cfc` | `bool` | `False` | Experimental: enables Compositional Function Calling via Live API |
| `save_input_blobs_as_artifacts` | `bool` | `False` | **Deprecated** — use `SaveFilesAsArtifactsPlugin` |

### `StreamingMode` enum

```python
from google.adk.agents.run_config import StreamingMode, RunConfig

# No streaming — one final event per turn (best for batch/CLI)
config_none = RunConfig(streaming_mode=StreamingMode.NONE)

# SSE — partial text events flow as the model generates
config_sse = RunConfig(streaming_mode=StreamingMode.SSE)

# BIDI — bidirectional; use runner.run_live() instead of run_async()
config_bidi = RunConfig(streaming_mode=StreamingMode.BIDI)
```

### Example 1 — SSE streaming with typewriter display

In SSE mode the runner yields **both** partial events (`event.partial=True`) and a final aggregated event. Avoid printing the text twice:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(name="streamer", model="gemini-2.5-flash", instruction="Be verbose.")

async def stream_response():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )

    config = RunConfig(streaming_mode=StreamingMode.SSE)
    displayed = ""

    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user", parts=[types.Part(text="Write a haiku about autumn.")]
        ),
        run_config=config,
    ):
        if event.partial and event.content and event.content.parts:
            # Only render partial text (skip partial function-call frames)
            has_text = any(p.text for p in event.content.parts)
            has_fc = any(p.function_call for p in event.content.parts)
            if has_text and not has_fc:
                chunk = "".join(p.text or "" for p in event.content.parts)
                print(chunk, end="", flush=True)
                displayed += chunk
        elif not event.partial and event.content:
            # Final aggregated event — skip if we already rendered this text
            final = "".join(p.text or "" for p in event.content.parts)
            if final and final != displayed:
                print(final)  # new content (e.g. tool results)

asyncio.run(stream_response())
```

### Example 2 — `ToolThreadPoolConfig` for live sessions

In live (bidirectional) mode, blocking tool calls freeze the event loop and delay the model from receiving audio frames. `ToolThreadPoolConfig` offloads tool execution to a thread pool:

```python
import time
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

def slow_database_query(query: str) -> dict:
    """Run a blocking DB query (sync I/O).

    Args:
        query: SQL query to run.
    Returns:
        Query results.
    """
    time.sleep(1)  # simulates blocking I/O
    return {"rows": [{"result": "42"}]}

agent = LlmAgent(
    name="live_agent",
    model="gemini-2.5-flash",
    instruction="Use the database tool to answer questions.",
    tools=[slow_database_query],
)

config = RunConfig(
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=4),
)
# Pass config to runner.run_live() for bidi sessions
```

**Thread pool helps with:**
- Blocking I/O: `time.sleep()`, network calls, file I/O, DB queries  
- C-extension CPU work (numpy, hashlib) — GIL is released  

**Thread pool does NOT help with:**
- Pure Python CPU-bound loops (GIL stays held)

### Example 3 — `get_session_config` to limit event loading

When sessions grow large, loading all events on every invocation is expensive. Combine with `EventsCompactionConfig`:

```python
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.base_session_service import GetSessionConfig

# Only load the 50 most recent events from the session store
config = RunConfig(
    get_session_config=GetSessionConfig(num_recent_events=50),
    max_llm_calls=200,
    custom_metadata={"request_id": "req-abc-123"},
)
```

### Example 4 — context window compression

```python
from google.genai import types
from google.adk.agents.run_config import RunConfig

config = RunConfig(
    context_window_compression=types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow(target_tokens=32_000),
    ),
    max_llm_calls=100,
)
```

---

## 3 · `Context` (= `ToolContext`)

`google.adk.agents.context.Context` is the writable context object. `ToolContext` is a module-level alias pointing to the same class (verified: `tools/tool_context.py:9` — `ToolContext = Context`).

It is passed to:
- Tool functions as the `tool_context` / `ctx` parameter
- Agent callbacks via `callback_context` (subtype `CallbackContext`)
- Workflow `@node` functions as the `ctx` parameter

### API summary

**State (read/write)**

```python
# Read
value = ctx.state["key"]          # current value
value = ctx.state.get("key", "default")

# Write (changes are persisted to session on event commit)
ctx.state["key"] = "value"
ctx.state["app:shared_counter"] = ctx.state.get("app:shared_counter", 0) + 1
ctx.state["user:display_name"] = "Alice"
ctx.state["temp:scratch"] = {"working": True}  # dropped after invocation
```

State key prefixes:

| Prefix | Scope | Persisted |
|---|---|---|
| _(none)_ | Session | ✓ |
| `app:` | All sessions in app | ✓ |
| `user:` | All sessions of that user | ✓ |
| `temp:` | Current invocation only | ✗ |

**Artifacts**

```python
from google.genai import types

# Save a file artifact (returns version number)
version = await ctx.save_artifact(
    filename="report.pdf",
    artifact=types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
    custom_metadata={"generated_by": "report_tool"},
)

# Load the latest version
part = await ctx.load_artifact("report.pdf")
pdf_bytes = part.inline_data.data

# Load a specific version
part_v1 = await ctx.load_artifact("report.pdf", version=1)

# List all artifacts in the current session
names = await ctx.list_artifacts()   # returns list[str]
```

**Memory**

```python
# Save current session events to long-term memory
await ctx.add_session_to_memory()

# Add explicit memory entries
from google.adk.memory.memory_entry import MemoryEntry
await ctx.add_memory(
    memories=[
        MemoryEntry(
            content=types.Content(
                role="user",
                parts=[types.Part(text="User prefers metric units.")],
            ),
        )
    ]
)

# Search memory
results = await ctx.search_memory("preferred units")
for item in results.memories:
    print(item.content.parts[0].text)
```

**Tool confirmation (HITL)**

```python
from google.adk.tools.tool_context import ToolContext

async def dangerous_delete(filename: str, tool_context: ToolContext) -> dict:
    """Delete a file after user confirmation.

    Args:
        filename: The file to delete.
    Returns:
        Status dict.
    """
    # Pause and ask the user to confirm in the UI
    tool_context.request_confirmation(
        hint=f"Are you sure you want to delete '{filename}'?",
        payload={"filename": filename},
    )
    # Execution resumes on next runner.run_async() call with confirmation
    # ...actual delete logic here...
    return {"deleted": filename}
```

**Workflow-only properties**

These are available when `ctx` is used inside a `@node` function:

```python
from google.adk.workflow import node, Workflow, START

@node(rerun_on_resume=True)
async def enricher(raw: str, ctx):
    # Route to next node based on content
    ctx.route = "long" if len(raw) > 500 else "short"

    # Dynamic node invocation
    cleaned = await ctx.run_node(clean_fn, raw)

    # Set output directly (alternative to returning)
    ctx.output = cleaned.upper()

@node
async def clean_fn(text: str, ctx) -> str:
    # Read attempt count for retry logic
    print(f"Attempt #{ctx.attempt_count}")
    return text.strip()
```

**Node path and run ID (diagnostics)**

```python
@node
async def debug_node(x, ctx):
    print(f"node_path={ctx.node_path!r}")  # e.g. "pipeline.enricher@1"
    print(f"run_id={ctx.run_id!r}")         # e.g. "1"
    print(f"invocation_id={ctx.invocation_id!r}")
```

### Complete tool example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

async def generate_report(topic: str, tool_context: ToolContext) -> dict:
    """Generate a text report on a topic and save it as an artifact.

    Args:
        topic: The subject to report on.
    Returns:
        A dict with the artifact filename and version.
    """
    # Write something to state
    tool_context.state["last_report_topic"] = topic

    # Build the report content
    report_text = f"# Report on {topic}\n\nGenerated by ADK agent.\n"
    report_bytes = report_text.encode()

    # Persist as artifact
    version = await tool_context.save_artifact(
        filename=f"{topic.replace(' ', '_')}_report.md",
        artifact=types.Part.from_bytes(
            data=report_bytes, mime_type="text/markdown"
        ),
    )
    return {"filename": f"{topic}_report.md", "version": version}

agent = LlmAgent(
    name="reporter",
    model="gemini-2.5-flash",
    instruction="Generate reports and save them using generate_report.",
    tools=[generate_report],
)

async def main():
    session_svc = InMemorySessionService()
    artifact_svc = InMemoryArtifactService()
    app = App(name="reports", root_agent=agent)
    runner = Runner(
        app=app, session_service=session_svc, artifact_service=artifact_svc
    )
    await session_svc.create_session(app_name="reports", user_id="u1", session_id="s1")
    events = await runner.run_debug(
        "Write a report on climate change.", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## 4 · `BasePlugin`

`google.adk.plugins.BasePlugin` — base class for runner-wide interception. Subclass it and register via `App(plugins=[...])`.

### Hook reference (complete, 2.1.0)

All hooks are `async def`. All return `Optional[<type>]`. Returning non-`None` short-circuits subsequent plugins **and** agent callbacks at the same point.

| Hook | Fires when | Non-`None` return means |
|---|---|---|
| `on_user_message_callback(*, invocation_context, user_message)` | User message received, before invocation | Replace user message |
| `before_run_callback(*, invocation_context)` | Once per invocation, before any agent runs | Return `types.Content` to halt invocation |
| `on_event_callback(*, invocation_context, event)` | Every event before persistence | Return modified `Event` to replace it |
| `after_run_callback(*, invocation_context)` | After invocation completes | — (returns `None`) |
| `close()` | When `runner.close()` is called | — (returns `None`) |
| `before_agent_callback(*, agent, callback_context)` | Before each agent's logic | Return `types.Content` to skip agent |
| `after_agent_callback(*, agent, callback_context)` | After each agent's logic | Return `types.Content` to override output |
| `before_model_callback(*, callback_context, llm_request)` | Before each LLM call | Return `LlmResponse` to skip model |
| `after_model_callback(*, callback_context, llm_response)` | After each LLM call | Return `LlmResponse` to replace response |
| `on_model_error_callback(*, callback_context, llm_request, error)` | When LLM call raises | Return `LlmResponse` to swallow error |
| `before_tool_callback(*, tool, tool_args, tool_context)` | Before each tool call | Return `dict` to skip tool |
| `after_tool_callback(*, tool, tool_args, tool_context, result)` | After each tool call | Return `dict` to replace result |
| `on_tool_error_callback(*, tool, tool_args, tool_context, error)` | When tool raises | Return `dict` to swallow error |

### Example 1 — rate-limiter plugin

```python
import asyncio
import time
from typing import Any, Optional
from google.adk.plugins import BasePlugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types

class RateLimiterPlugin(BasePlugin):
    """Allows at most `max_rpm` invocations per user per minute."""

    def __init__(self, max_rpm: int = 10):
        super().__init__(name="rate_limiter")
        self.max_rpm = max_rpm
        self._user_windows: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        user_id = invocation_context.user_id
        now = time.monotonic()
        cutoff = now - 60.0

        async with self._lock:
            window = self._user_windows.get(user_id, [])
            window = [t for t in window if t > cutoff]
            if len(window) >= self.max_rpm:
                return types.Content(
                    role="model",
                    parts=[types.Part(text="Rate limit exceeded. Try again in a minute.")],
                )
            window.append(now)
            self._user_windows[user_id] = window
        return None  # proceed normally
```

### Example 2 — LLM response caching plugin

```python
import hashlib
import json
from typing import Optional
from google.adk.plugins import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

class SemanticCachePlugin(BasePlugin):
    """Caches LLM responses by (system_instruction, user_text) hash."""

    def __init__(self):
        super().__init__(name="semantic_cache")
        self._cache: dict[str, LlmResponse] = {}

    def _hash_request(self, llm_request: LlmRequest) -> str:
        key = {
            "system": str(llm_request.config.system_instruction or ""),
            "contents": [
                {"role": c.role, "text": "".join(p.text or "" for p in (c.parts or []))}
                for c in (llm_request.contents or [])
            ],
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = self._hash_request(llm_request)
        if key in self._cache:
            return self._cache[key]   # skip model — return cached response
        # Store hash in temp state so after_model_callback can save
        callback_context.state["temp:cache_key"] = key
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        key = callback_context.state.get("temp:cache_key")
        if key:
            self._cache[key] = llm_response
        return None  # return the original response unmodified
```

### Example 3 — tool error recovery plugin

```python
from typing import Any, Optional
from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
import logging

logger = logging.getLogger(__name__)

class GracefulToolErrorPlugin(BasePlugin):
    """Converts tool exceptions into structured error dicts instead of crashing."""

    def __init__(self):
        super().__init__(name="graceful_tool_error")

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        logger.error("[tool error] %s raised %s: %s", tool.name, type(error).__name__, error)
        # Returning a dict swallows the exception and passes this to the model
        return {
            "error": type(error).__name__,
            "message": str(error),
            "tool": tool.name,
            "suggestion": "Please rephrase your request or try a different approach.",
        }

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        logger.error("[model error] %s", error)
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="I'm having trouble connecting. Please try again.")],
            )
        )
```

### Example 4 — ending an invocation from a plugin

```python
from typing import Optional
from google.adk.plugins import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

MAX_TURNS = 20  # hard cap on model calls per invocation

class TurnCapPlugin(BasePlugin):
    """Terminates the invocation after MAX_TURNS model calls."""

    def __init__(self):
        super().__init__(name="turn_cap")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        # Count calls using temp state (dropped after invocation)
        calls = callback_context.state.get("temp:model_calls", 0) + 1
        callback_context.state["temp:model_calls"] = calls

        if calls > MAX_TURNS:
            # Signal runner to stop after this step
            callback_context._invocation_context.end_invocation = True
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Maximum turn limit reached. Session ended.")],
                )
            )
        return None
```

---

## 5 · `App` + `EventsCompactionConfig` + `ResumabilityConfig`

`google.adk.apps.App` is the top-level container. It must have exactly one of `root_agent` (a `BaseAgent` or `BaseNode`).

### `App` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Must match `^[a-zA-Z][a-zA-Z0-9_-]*$`; `"user"` is reserved |
| `root_agent` | `BaseAgent \| BaseNode` | required | The entry point |
| `plugins` | `list[BasePlugin]` | `[]` | Ordered; run before agent callbacks |
| `events_compaction_config` | `EventsCompactionConfig \| None` | `None` | Sliding-window compaction |
| `context_cache_config` | `ContextCacheConfig \| None` | `None` | Gemini context cache (experimental) |
| `resumability_config` | `ResumabilityConfig \| None` | `None` | Enable pause/resume (experimental) |

### `EventsCompactionConfig` fields

Compaction reduces session size by summarising old events. Triggers either on a fixed interval or on token count.

| Field | Type | Required | Notes |
|---|---|---|---|
| `summarizer` | `BaseEventsSummarizer \| None` | No | Custom summariser; defaults to `LlmEventSummarizer` if `None` |
| `compaction_interval` | `int` | **Yes** | Number of new user turns before triggering compaction |
| `overlap_size` | `int` | **Yes** | Preceding turns kept un-compacted as overlap context |
| `token_threshold` | `int \| None` | No | Triggers compaction when prompt tokens ≥ threshold |
| `event_retention_size` | `int \| None` | No | Raw events retained after token-triggered compaction (must be set if `token_threshold` is set) |

### `ResumabilityConfig` fields (experimental)

| Field | Type | Default | Notes |
|---|---|---|---|
| `is_resumable` | `bool` | `False` | Enables pause/resume around long-running tool calls |

### Example — full App with compaction and resumability

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.plugins import LoggingPlugin
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

agent = LlmAgent(
    name="support",
    model="gemini-2.5-flash",
    instruction="You are a helpful support agent. Be concise.",
)

app = App(
    name="support_app",
    root_agent=agent,
    plugins=[LoggingPlugin()],
    # Compact every 20 user turns, keep 3 turns of overlap
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=20,
        overlap_size=3,
    ),
    # Also compact when prompt exceeds 100k tokens, keeping last 30 raw events
    # token_threshold and event_retention_size must be set together
)

# Separate App with token-based compaction trigger
app_with_tokens = App(
    name="support_token_compact",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=50,    # interval is still required
        overlap_size=5,
        token_threshold=100_000,   # also triggers when prompt ≥ 100k tokens
        event_retention_size=30,   # keep 30 raw events after token-triggered compaction
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)

# Wire to a persistent session service (required for resumability)
import os
runner = Runner(
    app=app_with_tokens,
    session_service=DatabaseSessionService(
        db_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///sessions.db")
    ),
)
```

### Example — context caching (experimental)

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App

LARGE_SYSTEM_TEXT = "…10 000 tokens of reference documentation…"

agent = LlmAgent(
    name="doc_bot",
    model="gemini-2.5-flash",
    instruction=LARGE_SYSTEM_TEXT,
)

app = App(
    name="doc_app",
    root_agent=agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=10,   # refresh cache every 10 invocations
        ttl_seconds=1800,     # 30-minute TTL
        min_tokens=4096,      # only cache if request ≥ 4096 tokens
    ),
)
```

---

## 6 · `Workflow`

`google.adk.workflow.Workflow` is the graph-based orchestrator (a `BaseNode`). Its key additional fields beyond `BaseNode`:

### `Workflow`-specific fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `edges` | `list[EdgeItem]` | `[]` | Tuples, `Edge` objects, or mix |
| `max_concurrency` | `int \| None` | `None` | Cap on concurrently running nodes; `None` = unlimited |

### Example 1 — parallel map-reduce with state schema

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import JoinNode, Workflow, node, START

class PipelineState(BaseModel):
    """Validates ctx.state mutations in this workflow."""
    raw_input: str = ""
    final_output: str = ""

sentiment = LlmAgent(
    name="sentiment",
    model="gemini-2.5-flash",
    instruction="Rate the sentiment as positive/neutral/negative.",
    mode="single_turn",
)
keywords = LlmAgent(
    name="keywords",
    model="gemini-2.5-flash",
    instruction="Extract 5 keywords, comma-separated.",
    mode="single_turn",
)

join = JoinNode(name="merge")

@node
def format_result(node_input: dict, ctx) -> str:
    sentiment_out = node_input.get("sentiment", "unknown")
    keywords_out = node_input.get("keywords", "")
    result = f"Sentiment: {sentiment_out}\nKeywords: {keywords_out}"
    ctx.state["final_output"] = result
    return result

pipeline = Workflow(
    name="analyse",
    edges=[(START, (sentiment, keywords), join, format_result)],
    max_concurrency=2,         # run both parallel branches simultaneously
    state_schema=PipelineState, # validates all ctx.state writes
)

async def main():
    app = App(name="nlp", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name="nlp", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "The new library release exceeded all expectations!",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — conditional routing with `DEFAULT_ROUTE`

```python
from google.adk.workflow import Workflow, node, DEFAULT_ROUTE, START
from google.adk.agents import LlmAgent

billing = LlmAgent(name="billing", model="gemini-2.5-flash",
                   instruction="Handle billing enquiries.", mode="single_turn")
support = LlmAgent(name="support", model="gemini-2.5-flash",
                   instruction="Handle technical support.", mode="single_turn")
general = LlmAgent(name="general", model="gemini-2.5-flash",
                   instruction="Answer general questions.", mode="single_turn")

@node
async def classify(query: str, ctx) -> str:
    q = query.lower()
    if "invoice" in q or "payment" in q:
        ctx.route = "billing"
    elif "error" in q or "bug" in q:
        ctx.route = "support"
    else:
        ctx.route = DEFAULT_ROUTE
    return query

router = Workflow(
    name="triage",
    edges=[
        (START, classify, {
            "billing": billing,
            "support": support,
            DEFAULT_ROUTE: general,
        }),
    ],
)
```

---

## 7 · `BaseNode` and `Node`

`google.adk.workflow.BaseNode` is the Pydantic base for every node. `google.adk.workflow.Node` is the subclass-friendly version for when you need class-level state or `parallel_worker`.

### `BaseNode` fields (2.1.0)

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Must be a Python identifier |
| `description` | `str` | `""` | Human-readable |
| `rerun_on_resume` | `bool` | `False` | If `True`, node reruns from scratch on resume; if `False`, uses resume input as output |
| `wait_for_output` | `bool` | `False` | Node stays in `WAITING` until it yields output/route (for fan-in patterns) |
| `retry_config` | `RetryConfig \| None` | `None` | Per-node retry policy |
| `timeout` | `float \| None` | `None` | Seconds before `NodeTimeoutError`; integrates with `retry_config` |
| `input_schema` | `SchemaType \| None` | `None` | Validates and coerces node input |
| `output_schema` | `SchemaType \| None` | `None` | Validates and coerces node output |
| `state_schema` | `type[BaseModel] \| None` | `None` | Validates `ctx.state` mutations |

### `Node` additional fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `parallel_worker` | `bool` | `False` | Wraps the node in a `_ParallelWorker` so concurrent triggers don't share state |

### Example 1 — `Node` subclass with input/output validation

```python
from pydantic import BaseModel
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.workflow import Node, Workflow, START

class TranslationInput(BaseModel):
    text: str
    target_lang: str

class TranslationOutput(BaseModel):
    original: str
    translated: str
    lang: str

class TranslatorNode(Node):
    name: str = "translator"
    description: str = "Translates text to the target language."
    input_schema: type = TranslationInput    # auto-validates incoming dict
    output_schema: type = TranslationOutput  # auto-validates returned dict

    async def run_node_impl(
        self, *, ctx, node_input: TranslationInput
    ) -> AsyncGenerator[Any, None]:
        # node_input is already a validated TranslationInput (or dict coerced to it)
        translated = f"[{node_input.target_lang}] {node_input.text}"  # placeholder
        yield TranslationOutput(
            original=node_input.text,
            translated=translated,
            lang=node_input.target_lang,
        )

translator = TranslatorNode()

@node
def post_process(result: dict, ctx) -> str:
    return f"Translated to {result['lang']}: {result['translated']}"

wf = Workflow(
    name="translate_pipeline",
    edges=[(START, translator, post_process)],
)
```

### Example 2 — `parallel_worker=True` for isolated concurrent invocations

```python
from collections.abc import AsyncGenerator
from typing import Any
import asyncio

from google.adk.workflow import Node, JoinNode, Workflow, START

class ScrapeNode(Node):
    name: str = "scraper"
    parallel_worker: bool = True   # each invocation gets its own state

    async def run_node_impl(
        self, *, ctx, node_input: str
    ) -> AsyncGenerator[Any, None]:
        # Simulate async scraping
        await asyncio.sleep(0.5)
        yield {"url": node_input, "content": f"<html>{node_input}</html>"}

scraper = ScrapeNode()
join = JoinNode(name="merge_results")

@node
def aggregate(results: dict, ctx) -> list[dict]:
    return list(results.values())

wf = Workflow(
    name="parallel_scraper",
    edges=[(START, (scraper, scraper, scraper), join, aggregate)],
    max_concurrency=3,
)
```

### Example 3 — `state_schema` for typed state validation

```python
from pydantic import BaseModel
from google.adk.workflow import Workflow, node, START

class AppState(BaseModel):
    counter: int = 0
    last_result: str = ""

@node
async def increment(x: str, ctx) -> str:
    ctx.state["counter"] = ctx.state.get("counter", 0) + 1  # validated against AppState
    ctx.state["last_result"] = x.upper()
    return x.upper()

wf = Workflow(
    name="counter",
    edges=[(START, increment)],
    state_schema=AppState,  # validates all ctx.state["counter"] and ["last_result"] writes
)
```

---

## 8 · `FunctionTool`

`google.adk.tools.FunctionTool` wraps any Python callable as a tool. `LlmAgent` auto-wraps bare callables, so you only construct `FunctionTool` directly when you need `require_confirmation`.

### Constructor

```python
FunctionTool(
    func: Callable[..., Any],
    *,
    require_confirmation: bool | Callable[..., bool] = False,
)
```

- **`func`**: The function to wrap. Must have a docstring (becomes the tool description) and type-annotated parameters (become the function declaration schema).
- **`require_confirmation`**: `True` = always ask; `Callable[..., bool]` = dynamic — receives the tool arguments and returns `True` to request confirmation.

### Context parameter detection

`FunctionTool` detects the context parameter by type annotation (preferred) or by name `tool_context`:

```python
from google.adk.tools.tool_context import ToolContext

# Detection by type annotation (preferred)
def tool_a(query: str, ctx: ToolContext) -> dict: ...

# Detection by name (fallback)
def tool_b(query: str, tool_context) -> dict: ...

# Both work — the context parameter is NOT exposed to the LLM
```

### Example 1 — confirmation for destructive operations

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def _should_confirm(filename: str, **_) -> bool:
    """Only confirm deletion of .prod files."""
    return filename.endswith(".prod")

async def delete_file(filename: str, tool_context: ToolContext) -> dict:
    """Delete a file from the workspace.

    Args:
        filename: Path to the file to delete.
    Returns:
        Status of the deletion.
    """
    # tool_context.tool_confirmation is populated after the user confirms
    if tool_context.tool_confirmation and not tool_context.tool_confirmation.confirmed:
        return {"status": "cancelled", "reason": "User declined."}
    # … actual delete …
    return {"status": "deleted", "filename": filename}

delete_tool = FunctionTool(
    func=delete_file,
    require_confirmation=_should_confirm,  # callable: only confirm .prod files
)
```

### Example 2 — async generator for streaming tool results

Tools can be async generators to stream partial results. The `input_stream` parameter name is reserved and excluded from the LLM schema:

```python
from collections.abc import AsyncGenerator
from google.adk.tools.tool_context import ToolContext

async def stream_lines(filepath: str, tool_context: ToolContext) -> AsyncGenerator[dict, None]:
    """Stream lines from a large file.

    Args:
        filepath: Path to the file.
    Yields:
        Dicts with `line_number` and `text`.
    """
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            yield {"line_number": i, "text": line.rstrip()}
```

### Example 3 — tool with state side-effects

```python
from google.adk.tools.tool_context import ToolContext

async def add_to_cart(item_id: str, quantity: int, tool_context: ToolContext) -> dict:
    """Add an item to the shopping cart.

    Args:
        item_id: The product ID to add.
        quantity: Number of units.
    Returns:
        Updated cart summary.
    """
    cart = dict(tool_context.state.get("cart", {}))
    cart[item_id] = cart.get(item_id, 0) + quantity
    tool_context.state["cart"] = cart
    total_items = sum(cart.values())
    return {"cart": cart, "total_items": total_items, "message": f"Added {quantity}x {item_id}."}
```

---

## 9 · `RetryConfig`

`google.adk.workflow.RetryConfig` — per-node exponential backoff with jitter.

### Field reference (2.1.0)

All fields are `Optional` — omit for defaults.

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_attempts` | `int \| None` | `5` | Total attempts including first. `0` or `1` = no retries. |
| `initial_delay` | `float \| None` | `1.0` | Seconds before first retry |
| `max_delay` | `float \| None` | `60.0` | Upper cap on inter-retry delay |
| `backoff_factor` | `float \| None` | `2.0` | Multiplier after each failure |
| `jitter` | `float \| None` | `1.0` | Random noise in delay. `0.0` = deterministic. |
| `exceptions` | `list[str \| type[Exception]] \| None` | `None` | Exceptions to retry on. `None` = retry on any. |

`exceptions` accepts both class names as strings and actual exception classes:

```python
from google.adk.workflow import RetryConfig

# Using exception class names (YAML/config-file friendly)
r1 = RetryConfig(
    max_attempts=3,
    exceptions=["ConnectionError", "TimeoutError", "httpx.TimeoutException"],
)

# Using actual exception classes (type-safe)
import httpx
r2 = RetryConfig(
    max_attempts=3,
    exceptions=[ConnectionError, TimeoutError, httpx.TimeoutException],
)
```

### Example 1 — selective retry for flaky I/O

```python
import httpx
from google.adk.workflow import node, RetryConfig, Workflow, START, NodeTimeoutError

# Retry only network errors; give up on ValueError (bad input)
@node(
    retry_config=RetryConfig(
        max_attempts=4,
        initial_delay=0.5,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=0.5,
        exceptions=[ConnectionError, httpx.TimeoutException, httpx.HTTPStatusError],
    ),
    timeout=15.0,
    rerun_on_resume=True,
)
async def fetch_data(url: str, ctx) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

pipeline = Workflow(name="fetcher", edges=[(START, fetch_data)])
```

### Example 2 — retry with counter logged in state

```python
from google.adk.workflow import node, RetryConfig

@node(
    retry_config=RetryConfig(max_attempts=5, initial_delay=1.0, backoff_factor=2.0),
)
async def reliable_write(data: dict, ctx) -> bool:
    attempt = ctx.attempt_count  # 1-based; provided by the framework
    ctx.state[f"temp:write_attempt_{attempt}"] = True
    # simulate occasional failure
    import random
    if random.random() < 0.4:
        raise IOError(f"Write failed on attempt {attempt}")
    return True
```

### Example 3 — `NodeTimeoutError` and retry interaction

`NodeTimeoutError` is only retried if `exceptions` is `None` (retry any) or explicitly includes `"NodeTimeoutError"`:

```python
from google.adk.workflow import node, RetryConfig, NodeTimeoutError

@node(
    retry_config=RetryConfig(
        max_attempts=3,
        exceptions=[IOError, NodeTimeoutError],  # retry timeouts too
    ),
    timeout=5.0,
)
async def slow_op(x: str, ctx) -> str:
    import asyncio
    await asyncio.sleep(3)  # may exceed timeout on slow systems
    return x.upper()
```

---

## 10 · `BaseTool`

`google.adk.tools.BaseTool` — the abstract base for all tools. Subclass when you need full control over how the tool declaration is sent to the LLM or how the result is built.

### API to implement

| Method / property | Required | Purpose |
|---|---|---|
| `name: str` | ✓ (set in `__init__`) | Tool identifier (exposed to LLM) |
| `description: str` | ✓ (set in `__init__`) | Tool description (exposed to LLM) |
| `is_long_running: bool` | No (`False`) | If `True`, event marks the call as long-running |
| `custom_metadata: dict \| None` | No (`None`) | Arbitrary JSON-serialisable metadata |
| `_get_declaration()` | Usually ✓ | Returns `types.FunctionDeclaration` for LLM; `None` for server-side tools |
| `run_async(*, args, tool_context)` | Usually ✓ | Executes the tool; returns result to be sent back as `FunctionResponse` |
| `process_llm_request(*, tool_context, llm_request)` | Rarely | Override to add special config (e.g. `google_search` adds a retrieval config) |

### Example 1 — custom tool with a hand-crafted declaration

```python
from typing import Any, Optional
from google.genai import types
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

class WeatherTool(BaseTool):
    """Fetches weather for a city using an external API."""

    def __init__(self, api_key: str):
        super().__init__(
            name="get_weather",
            description="Get current weather conditions for a city.",
        )
        self._api_key = api_key

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="City name, e.g. 'London'.",
                    ),
                    "units": types.Schema(
                        type=types.Type.STRING,
                        enum=["celsius", "fahrenheit"],
                        description="Temperature unit.",
                    ),
                },
                required=["city"],
            ),
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        city = args["city"]
        units = args.get("units", "celsius")
        # Cache result in session state to avoid repeated API calls
        cache_key = f"temp:weather_{city}_{units}"
        if cached := tool_context.state.get(cache_key):
            return cached

        # Real implementation would call an API here
        result = {"city": city, "temp": 15, "units": units, "condition": "cloudy"}
        tool_context.state[cache_key] = result
        return result
```

### Example 2 — long-running tool that returns an ID first

Set `is_long_running=True` to signal to the runtime that this tool will produce a result asynchronously. The tool returns a job ID immediately; a later `FunctionResponse` carries the result.

```python
import asyncio
import uuid
from typing import Any
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

class BatchExportTool(BaseTool):
    """Starts a background export job and returns a job ID."""

    def __init__(self):
        super().__init__(
            name="export_data",
            description="Export a large dataset to GCS. Returns a job ID immediately.",
            is_long_running=True,  # marks the call as long-running on the event
        )
        self._jobs: dict[str, str] = {}  # job_id → status

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "dataset": types.Schema(type=types.Type.STRING),
                    "bucket": types.Schema(type=types.Type.STRING),
                },
                required=["dataset", "bucket"],
            ),
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = "running"
        # Fire-and-forget background task
        asyncio.create_task(self._run_export(job_id, args["dataset"], args["bucket"]))
        return {"job_id": job_id, "status": "started"}

    async def _run_export(self, job_id: str, dataset: str, bucket: str) -> None:
        await asyncio.sleep(5)  # simulate long export
        self._jobs[job_id] = "done"
```

### Example 3 — server-side tool (no `run_async`)

For tools that execute entirely on the model/server side (like Gemini's built-in `google_search`), override `process_llm_request` to inject a config and leave `run_async` unimplemented:

```python
from typing import Any, Optional
from google.genai import types
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest

class GroundingTool(BaseTool):
    """Enables Vertex AI grounding on every LLM call for this agent."""

    def __init__(self, data_store_id: str):
        super().__init__(
            name="vertex_grounding",
            description="Grounds answers in a Vertex AI Search data store.",
        )
        self._store = data_store_id

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return None  # no FunctionDeclaration — server-side only

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        # Inject retrieval config — ADK calls this before sending the request
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        llm_request.config.tools.append(
            types.Tool(
                retrieval=types.Retrieval(
                    vertex_ai_search=types.VertexAISearch(
                        datastore=self._store,
                    )
                )
            )
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        raise NotImplementedError("GroundingTool is server-side only.")
```

---

## Version notes

All examples verified against **google-adk==2.1.0** installed from PyPI (`pip install google-adk`) in May 2026. Import paths, field names, and signatures cross-checked against installed source in `/tmp/adk-env/lib/python3.11/site-packages/google/adk/`.

Key changes from 2.0.0 → 2.1.0 relevant to this page:
- `RunConfig` gained `tool_thread_pool_config` (`ToolThreadPoolConfig`), `context_window_compression`, `get_session_config`, `enable_affective_dialog`, `proactivity`, and `session_resumption` fields.
- `BaseNode.state_schema` and `BaseNode.input_schema` / `output_schema` are now documented at the `BaseNode` level (were only on `Workflow` in 2.0.x).
- `ContextCacheConfig` is now confirmed in `context_cache_config` on both `App` and the `Context` / `InvocationContext`.
- `Context.add_memory()` (explicit memory entries) added alongside the existing `add_session_to_memory()`.
- `BasePlugin.on_model_error_callback` is a first-class hook (was implicit in 2.0.x).

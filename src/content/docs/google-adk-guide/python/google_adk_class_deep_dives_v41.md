---
title: "Class deep dives — volume 41 (Node subclassing, static_instruction, generate_content_config, include_contents, LoggingPlugin, ReadonlyContext, run_debug, parallel fan-out, AgentTool propagation, ToolContext.add_memory)"
description: "10 source-verified deep dives for google-adk 2.4.0: Node subclassing with run_node_impl and parallel_worker, LlmAgent.static_instruction for context caching, LlmAgent.generate_content_config (temperature / safety / thinking), LlmAgent.include_contents='none' stateless agents, LoggingPlugin all-hooks event tracer, ReadonlyContext in instruction callables, InMemoryRunner.run_debug quick test harness, FunctionNode parallel fan-out with parallel_worker=True, AgentTool include_plugins and propagate_grounding_metadata, ToolContext.add_memory direct memory writes."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 41"
  order: 110
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.4.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `Node` — user-subclassable workflow node

**Source:** `google/adk/workflow/_node.py`

`Node` is the recommended base class for custom workflow logic. Unlike `FunctionNode` (which wraps a function) or `LlmAgent` (which wraps an LLM call), `Node` is designed for subclassing: implement `run_node_impl` and optionally enable `parallel_worker=True` to fan-out over list inputs.

### Key fields (verified `_node.py`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `parallel_worker` | `bool` | `False` | `frozen=True` — must be set at construction, not mutated later |
| `max_parallel_workers` | `int \| None` | `None` | Only valid when `parallel_worker=True`; must be ≥ 1 |

When `parallel_worker=True`, `model_post_init` wraps a clone of the node in a `_ParallelWorker`. The clone has `parallel_worker=False` to prevent infinite recursion. `rerun_on_resume` is synchronised from the inner `_ParallelWorker`.

### Example 1 — minimal custom Node

```python
import asyncio
from typing import AsyncGenerator, Any
from google.adk.workflow import Workflow, START
from google.adk.workflow._node import Node
from google.adk.tools.tool_context import Context
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

class TextSplitterNode(Node):
    """Splits a document into fixed-size chunks and stores them in state."""

    chunk_size: int = 200

    async def run_node_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
        text = str(node_input or "")
        chunks = [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]
        ctx.state["chunks"] = chunks
        ctx.state["chunk_count"] = len(chunks)
        yield chunks  # the yielded value becomes the node output

splitter = TextSplitterNode(name="splitter", chunk_size=100)

pipeline = Workflow(name="split_pipeline", edges=[(START, splitter)])

async def main():
    app = App(name="demo", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    events = await runner.run_debug("The quick brown fox jumped over the lazy dog. " * 5)
    # state["chunks"] now holds the split document
    session = await runner.session_service.get_session(
        app_name="demo", user_id="debug_user_id", session_id="debug_session_id"
    )
    print(f"Chunks produced: {session.state.get('chunk_count')}")

asyncio.run(main())
```

### Example 2 — parallel fan-out with `parallel_worker=True`

When `parallel_worker=True`, the node receives a **list** as `node_input`. It clones itself (minus the flag) and wraps it in `_ParallelWorker`, which maps each list element to an independent `run_node_impl` call. `max_parallel_workers` caps the concurrency.

```python
import asyncio
from typing import AsyncGenerator, Any
from google.adk.workflow import Workflow, START
from google.adk.workflow._node import Node
from google.adk.tools.tool_context import Context

class SentimentNode(Node):
    """Analyse sentiment of a single text snippet (simulated)."""

    async def run_node_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
        text = str(node_input)
        # Real code would call an LLM or ML model here
        result = {"text": text[:30], "score": len(text) % 5}
        yield result

# parallel_worker=True: node_input is expected to be a list.
# max_parallel_workers=4 caps concurrency to 4 simultaneous calls.
parallel_sentiment = SentimentNode(
    name="sentiment",
    parallel_worker=True,
    max_parallel_workers=4,
)

pipeline = Workflow(name="batch_sentiment", edges=[(START, parallel_sentiment)])
```

> **Important:** `parallel_worker` is `frozen=True` — it must be set in the constructor. Mutating it after construction raises a Pydantic validation error.

### Example 3 — Node with `retry_config`

`Node` inherits `retry_config: RetryConfig | None` from `BaseNode`. Set it to automatically retry transient failures.

```python
from google.adk.workflow import Workflow, START, RetryConfig
from google.adk.workflow._node import Node
from google.adk.tools.tool_context import Context
from typing import AsyncGenerator, Any

class FlakyExternalCallNode(Node):
    url: str = "https://api.example.com/data"

    async def run_node_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.url, timeout=5.0)
            resp.raise_for_status()
            yield resp.json()

fetcher = FlakyExternalCallNode(
    name="fetch",
    url="https://api.example.com/data",
    retry_config=RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=0.5,
        exceptions=["httpx.HTTPStatusError", "httpx.ConnectTimeout"],
    ),
    timeout=30.0,
)

pipeline = Workflow(name="resilient_fetch", edges=[(START, fetcher)])
```

---

## 2 · `LlmAgent.static_instruction` — context caching split

**Source:** `google/adk/agents/llm_agent.py`

`static_instruction` holds content that **never changes between requests** and is sent literally as the system instruction. In contrast, `instruction` is dynamic (supports `{variable}` placeholders and is re-rendered each turn).

This split enables Gemini's **implicit context caching**: when the static prefix is identical across requests, Gemini can cache and reuse it — reducing latency and cost.

### Field signature (verified `llm_agent.py`)

```python
static_instruction: Optional[types.ContentUnion] = None
```

`ContentUnion` accepts: `str`, `types.Content`, `types.Part`, `PIL.Image.Image`, `types.File`, or `list[PartUnion]`.

**When `static_instruction` is set:**
- `static_instruction` → sent as `system_instruction` at the start of every request
- `instruction` → appended to the **user content** (not the system prompt)

**When `static_instruction` is `None`** (default):
- `instruction` → sent as `system_instruction` (normal mode)

### Example 1 — string static instruction

```python
from google.adk.agents import LlmAgent

legal_agent = LlmAgent(
    name="legal_drafter",
    model="gemini-2.5-flash",
    # This never changes — ideal for caching
    static_instruction=(
        "You are a legal drafting assistant specialising in UK contract law. "
        "You always use precise legal terminology, cite relevant case law when "
        "appropriate, and flag ambiguous clauses for review. "
        "You must never give specific legal advice — only draft language."
    ),
    # This is dynamic — rendered each turn from session state
    instruction="Current matter type: {matter_type}. Client jurisdiction: {jurisdiction}.",
)
```

### Example 2 — multimodal static instruction (file reference)

```python
from google.adk.agents import LlmAgent
from google.genai import types

# A brand style guide PDF uploaded to Gemini Files API
style_guide_file = types.File(uri="https://generativelanguage.googleapis.com/v1beta/files/abc123")

brand_agent = LlmAgent(
    name="brand_writer",
    model="gemini-2.5-flash",
    static_instruction=types.Content(
        role="user",
        parts=[
            types.Part(text="You are a brand copywriter. Follow the style guide provided."),
            types.Part(file_data=types.FileData(
                file_uri=style_guide_file.uri,
                mime_type="application/pdf",
            )),
        ],
    ),
    instruction="Product category: {category}. Tone for this campaign: {tone}.",
)
```

### Example 3 — explicit `ContextCacheConfig` pairing

For **explicit** Gemini context caching (where you control the TTL and cache lifecycle), combine `static_instruction` with `ContextCacheConfig` on the `App`. The config is read by `ContextCacheRequestProcessor` before each LLM call.

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

dense_docs_agent = LlmAgent(
    name="docs_qa",
    model="gemini-2.5-flash",
    static_instruction="You are a documentation assistant. " + "Rules: " * 200,  # large static
    instruction="Answer the following question about {product}: {question}",
)

app = App(
    name="docs",
    root_agent=dense_docs_agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=5,    # create a new cache every 5 invocations
        ttl_seconds=1800,     # cache TTL: 30 minutes
        # Hard minimum: 4096 tokens; cache is skipped if static content is shorter
    ),
)
runner = InMemoryRunner(app=app)
```

> **Note:** `ContextCacheConfig` is `@experimental`. The 4096-token hard minimum comes from Gemini's API — content shorter than this cannot be cached, and the request processor silently skips caching rather than raising an error.

---

## 3 · `LlmAgent.generate_content_config` — model parameters

**Source:** `google/adk/agents/llm_agent.py`

`generate_content_config: Optional[types.GenerateContentConfig] = None`

Passes raw `GenerateContentConfig` fields directly to the model. Not all fields are honoured — `tools` must use `tools=` on the agent, and `thinking_config` defers to `planner` when both are set.

### Example 1 — temperature and safety settings

```python
from google.adk.agents import LlmAgent
from google.genai import types

creative_agent = LlmAgent(
    name="story_writer",
    model="gemini-2.5-flash",
    instruction="Write a short story based on the prompt.",
    generate_content_config=types.GenerateContentConfig(
        temperature=1.5,         # more creative, higher variance
        top_p=0.95,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ],
    ),
)
```

### Example 2 — structured JSON output via `response_mime_type`

When you need raw JSON output without using `output_schema`, set `response_mime_type`:

```python
from google.adk.agents import LlmAgent
from google.genai import types

json_extractor = LlmAgent(
    name="extractor",
    model="gemini-2.5-flash",
    instruction=(
        "Extract key entities from the text. "
        "Return a JSON object with keys: people, organisations, dates."
    ),
    generate_content_config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,    # low temperature for deterministic extraction
    ),
)
```

### Example 3 — enabling built-in thinking via `thinking_config`

```python
from google.adk.agents import LlmAgent
from google.genai import types

reasoning_agent = LlmAgent(
    name="math_solver",
    model="gemini-2.5-flash-thinking-exp",
    instruction="Solve the maths problem step by step.",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=8192,   # max tokens for the hidden reasoning trace
        ),
        temperature=0.2,
    ),
    # Note: if a BuiltInPlanner is also set, planner.thinking_config wins.
)
```

> **Precedence note:** `planner.thinking_config` overrides `generate_content_config.thinking_config`. Only set one; use `generate_content_config` when you don't need a full `BuiltInPlanner`.

---

## 4 · `LlmAgent.include_contents='none'` — stateless agents

**Source:** `google/adk/agents/llm_agent.py`

```python
include_contents: Literal['default', 'none'] = 'default'
```

- `'default'` — the model sees the relevant portion of conversation history (filtered by branch, compaction, etc.)
- `'none'` — the model sees **zero** prior history; it operates solely on `instruction` and the current `node_input` / user message

`'none'` is useful for:
- Deterministic extraction agents that should not be influenced by prior turns
- Workers in parallel fan-out where each call should be independent
- Agents that receive all context through `node_input` or injected `instruction` variables

### Example 1 — stateless classifier in a workflow

```python
from google.adk.agents import LlmAgent
from pydantic import BaseModel

class Category(BaseModel):
    label: str
    confidence: float

classifier = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction=(
        "Classify the following text into exactly one category: "
        "finance, technology, health, sports, or entertainment. "
        "Text: {node_input}"
    ),
    include_contents="none",    # no history — each call is independent
    output_schema=Category,
    output_key="classification",
)
```

### Example 2 — parallel evaluators (all stateless)

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START, node

@node
def fan_out(node_input: str) -> list[str]:
    """Send the same text to three evaluators."""
    return [node_input, node_input, node_input]

tone_evaluator = LlmAgent(
    name="tone_eval",
    model="gemini-2.5-flash",
    instruction="Rate the tone of this text 1-10: {node_input}",
    include_contents="none",
    output_key="tone_score",
)
clarity_evaluator = LlmAgent(
    name="clarity_eval",
    model="gemini-2.5-flash",
    instruction="Rate the clarity of this text 1-10: {node_input}",
    include_contents="none",
    output_key="clarity_score",
)

# Each evaluator gets the same text fresh, with no history
pipeline = Workflow(
    name="eval_pipeline",
    edges=[(START, fan_out, tone_evaluator), (START, fan_out, clarity_evaluator)],
)
```

---

## 5 · `LoggingPlugin` — built-in verbose event tracer

**Source:** `google/adk/plugins/logging_plugin.py`

`LoggingPlugin` is a `BasePlugin` that prints all ADK events to the console. It implements 9 callback hooks and is intended for **development debugging** — it is not a replacement for production logging.

### Callback hooks implemented (verified `logging_plugin.py`)

| Callback | What it logs |
|---|---|
| `on_user_message_callback` | Invocation ID, session ID, user ID, app name, root agent, user content, branch |
| `before_run_callback` | Invocation start, starting agent name |
| `on_event_callback` | Event ID, author, content, is_final_response, function call/response names |
| `after_run_callback` | Invocation end, final response, errors |
| `before_agent_callback` | Agent name starting |
| `after_agent_callback` | Agent name finished |
| `before_model_callback` | LLM request preview |
| `after_model_callback` | LLM response preview |
| `before_tool_callback` | Tool name, arguments |
| `after_tool_callback` | Tool name, result |

### Example 1 — attach to runner

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Be concise.",
)

app = App(
    name="demo",
    root_agent=agent,
    plugins=[LoggingPlugin()],   # attach globally
)

async def main():
    runner = InMemoryRunner(app=app)
    # All events, tool calls, and model interactions are now printed
    await runner.run_debug("What is the capital of France?")

asyncio.run(main())
```

### Example 2 — selective verbosity (quiet mode + LoggingPlugin together)

```python
import asyncio
from google.adk.runners import InMemoryRunner
from google.adk.agents import LlmAgent
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.apps import App

agent = LlmAgent(name="chat", model="gemini-2.5-flash", instruction="Chat.")

# LoggingPlugin on the App gives full structured trace
# run_debug's verbose=False gives clean final output
app = App(name="demo", root_agent=agent, plugins=[LoggingPlugin()])
runner = InMemoryRunner(app=app)

async def main():
    # LoggingPlugin will still log all events; run_debug's quiet=True
    # suppresses run_debug's own print calls so only LoggingPlugin output shows
    events = await runner.run_debug(
        "Explain quantum entanglement in one sentence.",
        quiet=True,
    )
    print(f"\nTotal events captured: {len(events)}")

asyncio.run(main())
```

### Example 3 — writing a custom LoggingPlugin subclass

```python
import logging
from typing import Optional
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

logger = logging.getLogger("adk.custom")

class StructuredLoggingPlugin(LoggingPlugin):
    """Routes all ADK events to Python's logging instead of print()."""

    def _log(self, message: str) -> None:
        logger.debug(message)

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        logger.info(
            "event",
            extra={
                "event_id": event.id,
                "author": event.author,
                "is_final": event.is_final_response(),
                "invocation_id": invocation_context.invocation_id,
            },
        )
        return None   # never short-circuit
```

---

## 6 · `ReadonlyContext` — safe read-only context

**Source:** `google/adk/agents/readonly_context.py`

`ReadonlyContext` is the base class for `Context` (ToolContext/CallbackContext). It exposes only **read** operations — no state mutations, no artifact writes. It is passed to:

- `instruction` when `instruction` is a callable (an `InstructionProvider`)
- `before_agent_callback` (where mutation is intentionally limited)
- Any custom code that should inspect context without side-effects

### Properties (verified `readonly_context.py`)

| Property | Type | Notes |
|---|---|---|
| `user_content` | `Optional[types.Content]` | The user message that started this invocation |
| `invocation_id` | `str` | Current invocation identifier |
| `agent_name` | `str` | Name of the currently running agent |
| `state` | `MappingProxyType[str, Any]` | **Read-only** proxy of session state |
| `session` | `Session` | The current session object |
| `user_id` | `str` | Current user ID |
| `run_config` | `Optional[RunConfig]` | Runtime config for this invocation |

`get_credential(key: str)` is also available for reading pre-resolved credentials.

### Example 1 — dynamic instruction using `ReadonlyContext`

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

def personalised_instruction(ctx: ReadonlyContext) -> str:
    """Build system instruction from session state at runtime."""
    name = ctx.state.get("user_name", "user")
    preferred_lang = ctx.state.get("preferred_language", "English")
    expertise = ctx.state.get("expertise_level", "intermediate")
    return (
        f"You are helping {name}. They prefer {preferred_lang} responses. "
        f"Tailor explanations to a {expertise}-level developer."
    )

agent = LlmAgent(
    name="adaptive_assistant",
    model="gemini-2.5-flash",
    instruction=personalised_instruction,  # callable — receives ReadonlyContext
)
```

### Example 2 — reading state in `before_agent_callback`

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from typing import Optional
from google.genai import types

def check_quota(callback_context: ReadonlyContext) -> Optional[types.Content]:
    """Block execution if the user has exceeded their daily quota."""
    calls_today = callback_context.state.get("calls_today", 0)
    limit = callback_context.state.get("daily_limit", 100)
    if calls_today >= limit:
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Quota exceeded ({calls_today}/{limit} calls today).")],
        )
    return None  # proceed normally

agent = LlmAgent(
    name="quota_guarded",
    model="gemini-2.5-flash",
    instruction="Help the user.",
    before_agent_callback=check_quota,
)
```

### Example 3 — role-based tool filtering with `ReadonlyContext`

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool
from typing import Optional

def delete_record(record_id: str) -> dict:
    """Delete a record by ID (admin-only)."""
    return {"deleted": record_id}

def read_record(record_id: str) -> dict:
    """Read a record by ID."""
    return {"record_id": record_id, "data": "..."}

def role_filtered_tools(ctx: ReadonlyContext) -> list:
    """Return tools based on user role stored in session state."""
    role = ctx.state.get("user_role", "viewer")
    tools = [read_record]
    if role == "admin":
        tools.append(delete_record)
    return tools

agent = LlmAgent(
    name="rbac_agent",
    model="gemini-2.5-flash",
    instruction="Manage records. Available actions depend on your role.",
    tools=role_filtered_tools,  # callable tools list — receives ReadonlyContext
)
```

---

## 7 · `InMemoryRunner.run_debug` — quick test harness

**Source:** `google/adk/runners.py` (`InMemoryRunner.run_debug`)

`run_debug` is a developer-convenience method that wraps `run_async`. It auto-creates a session if needed, accepts `str | list[str]`, and optionally prints agent output to the console.

### Signature (verified `runners.py`)

```python
async def run_debug(
    self,
    user_messages: str | list[str],
    *,
    user_id: str = "debug_user_id",
    session_id: str = "debug_session_id",
    run_config: RunConfig | None = None,
    quiet: bool = False,
    verbose: bool = False,
) -> list[Event]:
```

- `quiet=True` suppresses all console output (returns events only)
- `verbose=True` prints tool calls and responses in addition to final replies
- Reusing the same `session_id` continues the conversation across calls

### Example 1 — single-shot quick test

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the text in one sentence.",
)

async def main():
    runner = InMemoryRunner(agent=agent)
    events = await runner.run_debug(
        "The Amazon rainforest covers more than 5.5 million square kilometres "
        "and produces approximately 20 percent of the world's oxygen.",
    )
    # Final response is always the last event with is_final_response() == True
    final = next((e for e in reversed(events) if e.is_final_response()), None)
    if final and final.content:
        print(final.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — multi-turn conversation test

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="chat",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Remember what the user tells you.",
)

async def main():
    runner = InMemoryRunner(agent=agent)
    session_id = "test_conv_1"

    # Turn 1
    await runner.run_debug(
        "My name is Alice.",
        session_id=session_id,
        quiet=True,
    )
    # Turn 2 — agent should recall the name from the session
    events = await runner.run_debug(
        "What is my name?",
        session_id=session_id,
        quiet=False,  # prints the agent's reply
    )
    print(f"Total events across turn 2: {len(events)}")

asyncio.run(main())
```

### Example 3 — inspecting events returned by `run_debug`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

def get_weather(city: str) -> dict:
    """Return fake weather data."""
    return {"city": city, "temperature_c": 22, "condition": "sunny"}

agent = LlmAgent(
    name="weather",
    model="gemini-2.5-flash",
    instruction="Answer weather questions.",
    tools=[get_weather],
)

async def main():
    runner = InMemoryRunner(agent=agent)
    events = await runner.run_debug(
        "What is the weather in London?",
        verbose=True,  # prints tool call details
        quiet=True,    # quiet suppresses the final-response print; verbose still shows tool details
    )
    for event in events:
        if event.get_function_calls():
            fc = event.get_function_calls()[0]
            print(f"Tool called: {fc.name}({fc.args})")
        if event.is_final_response() and event.content:
            print(f"Final reply: {event.content.parts[0].text}")

asyncio.run(main())
```

---

## 8 · `FunctionNode` parallel fan-out with `parallel_worker=True`

**Source:** `google/adk/workflow/_function_node.py`

When a `FunctionNode` (or the `@node` decorator) is given `parallel_worker=True`, it wraps itself in `_ParallelWorker` — identical to `Node`. The node's function is called once per element of the incoming list, up to `max_parallel_workers` concurrently.

### Example 1 — batch URL fetcher

```python
import asyncio
import httpx
from google.adk.workflow import Workflow, START, node
from google.adk.workflow._function_node import FunctionNode

# @node with parallel_worker=True: called once per URL
@node(parallel_worker=True, max_parallel_workers=8, name="fetch_url")
async def fetch_url(node_input: str) -> dict:
    """Fetch a URL and return status + length."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(node_input)
            return {"url": node_input, "status": resp.status_code, "length": len(resp.content)}
        except Exception as exc:
            return {"url": node_input, "error": str(exc)}

@node(name="produce_urls")
def produce_urls(node_input: str) -> list[str]:
    """Produce a list of URLs from a newline-separated input."""
    return [u.strip() for u in node_input.splitlines() if u.strip()]

pipeline = Workflow(
    name="batch_fetch",
    edges=[(START, produce_urls, fetch_url)],
)
```

### Example 2 — LLM agent as parallel worker

`LlmAgent` inherits from `Node` indirectly (via `BaseAgent`), but `parallel_worker` is available as a kwarg:

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, node, START

@node(name="expand")
def expand_topics(node_input: str) -> list[str]:
    return [f"Topic {i}: {node_input}" for i in range(1, 4)]

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research this topic and give 3 key facts: {node_input}",
    include_contents="none",
    parallel_worker=True,    # fan-out over the list from expand_topics
    max_parallel_workers=3,
)

pipeline = Workflow(
    name="parallel_research",
    edges=[(START, expand_topics, researcher)],
)
```

### Example 3 — collecting parallel results with `JoinNode`

```python
from google.adk.workflow import Workflow, START
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow import node

@node(name="split")
def split_text(node_input: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in node_input.split(".") if s.strip()]

@node(name="uppercase", parallel_worker=True)
def to_upper(node_input: str) -> str:
    return node_input.upper()

join = JoinNode(name="join")

@node(name="combine")
def combine(node_input: dict) -> str:
    """Combine aggregated JoinNode outputs."""
    return " | ".join(str(v) for v in node_input.values())

pipeline = Workflow(
    name="parallel_upper",
    edges=[(START, split_text, to_upper, join, combine)],
)
```

---

## 9 · `AgentTool.include_plugins` + `propagate_grounding_metadata`

**Source:** `google/adk/tools/agent_tool.py`

`AgentTool` wraps a `BaseAgent` so it can be called as a tool from another `LlmAgent`. Two fields added in recent versions control plugin and grounding propagation.

### Constructor signature (verified `agent_tool.py`)

```python
class AgentTool(BaseTool):
    def __init__(
        self,
        agent: BaseAgent,
        skip_summarization: bool = False,
        *,
        include_plugins: bool = True,
        propagate_grounding_metadata: bool = False,
    ): ...
```

| Field | Default | Notes |
|---|---|---|
| `skip_summarization` | `False` | When `True`, the sub-agent's full output is returned without LLM summarisation |
| `include_plugins` | `True` | When `True`, the parent runner's plugins are propagated to the sub-agent's inner runner |
| `propagate_grounding_metadata` | `False` | When `True`, grounding metadata (search citations) from the sub-agent is forwarded to the parent event |

### Example 1 — basic `AgentTool` usage

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

# A specialist sub-agent
code_reviewer = LlmAgent(
    name="code_reviewer",
    model="gemini-2.5-flash",
    description="Reviews Python code for bugs and style issues.",
    instruction="Review the code and list issues. Be specific about line numbers.",
    include_contents="none",
)

# Coordinator uses the specialist as a tool
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="Help the user with software development tasks.",
    tools=[AgentTool(code_reviewer)],
)
```

### Example 2 — isolating plugin context with `include_plugins=False`

By default plugins from the parent runner propagate to the sub-agent runner. Set `include_plugins=False` to give the sub-agent an isolated plugin environment (useful when the sub-agent has different logging, rate-limiting, or security requirements).

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

sensitive_data_agent = LlmAgent(
    name="pii_processor",
    model="gemini-2.5-flash",
    description="Processes PII data — must not be logged.",
    instruction="Process the PII fields in the input.",
)

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="Orchestrate data pipelines.",
    tools=[
        AgentTool(
            sensitive_data_agent,
            include_plugins=False,  # parent LoggingPlugin is NOT propagated here
        )
    ],
)
```

### Example 3 — propagating grounding metadata for search agents

When the sub-agent uses `google_search` or other grounding tools, setting `propagate_grounding_metadata=True` forwards the search citations to the coordinator's event stream.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, google_search

search_specialist = LlmAgent(
    name="web_searcher",
    model="gemini-2.5-flash",
    description="Searches the web for current information.",
    instruction="Use google_search to find up-to-date information.",
    tools=[google_search],
)

coordinator = LlmAgent(
    name="research_coordinator",
    model="gemini-2.5-flash",
    instruction="Answer research questions using your search specialist.",
    tools=[
        AgentTool(
            search_specialist,
            propagate_grounding_metadata=True,  # citations flow up to parent
        )
    ],
)
```

### Example 4 — `skip_summarization` for raw structured output

When the sub-agent returns structured data (JSON / Pydantic model) via `output_schema`, `skip_summarization=True` returns the raw string to the coordinator LLM without an intermediate summary pass.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from pydantic import BaseModel

class SentimentResult(BaseModel):
    score: float
    label: str
    reasoning: str

sentiment_agent = LlmAgent(
    name="sentiment",
    model="gemini-2.5-flash",
    description="Returns structured sentiment analysis.",
    instruction="Analyse sentiment of the text.",
    output_schema=SentimentResult,
)

coordinator = LlmAgent(
    name="analytics_coordinator",
    model="gemini-2.5-flash",
    instruction="Run sentiment analysis and interpret the results.",
    tools=[
        AgentTool(sentiment_agent, skip_summarization=True)
    ],
)
```

---

## 10 · `ToolContext.add_memory` — direct memory writes from tools

**Source:** `google/adk/tools/tool_context.py` (the `Context` class)

`ToolContext` (the `Context` class in `tools/tool_context.py`) exposes three memory-write methods. Unlike `add_session_to_memory` which ingests the full session, `add_memory` writes **explicit memory entries** directly — bypassing event extraction entirely.

### Method signatures (verified `tool_context.py` → `BaseMemoryService`)

```python
# Add explicit MemoryEntry items directly (implementation-defined)
async def add_memory(
    self,
    *,
    memories: Sequence[MemoryEntry],
    custom_metadata: Mapping[str, object] | None = None,
) -> None

# Add a delta of events (not the full session)
async def add_events_to_memory(
    self,
    *,
    events: Sequence[Event],
    custom_metadata: Mapping[str, object] | None = None,
) -> None

# Ingest the full current session
async def add_session_to_memory(self) -> None
```

`add_memory` is implemented by services that support direct writes (e.g. `VertexAiMemoryBankService`). `InMemoryMemoryService` does **not** implement `add_memory` — it raises `NotImplementedError`. Use `add_events_to_memory` with in-memory services.

### `MemoryEntry` model (verified `google/adk/memory/base_memory_service.py`)

```python
class MemoryEntry(BaseModel):
    content: types.Content
    author: str = ""           # 'user' or agent name
    timestamp: float = 0.0
    session_id: str = ""
    metadata: dict = {}
```

### Example 1 — writing key facts from a tool

```python
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import Runner
from google.adk.memory import VertexAiMemoryBankService  # supports add_memory
from google.genai import types

async def save_user_preference(
    category: str,
    value: str,
    tool_context: ToolContext,
) -> dict:
    """Save a user preference directly to the memory bank."""
    from google.adk.memory.base_memory_service import MemoryEntry
    entry = MemoryEntry(
        content=types.Content(
            role="user",
            parts=[types.Part(text=f"User prefers {value} for {category}.")],
        ),
        author="system",
        metadata={"category": category, "value": value, "source": "preference_tool"},
    )
    await tool_context.add_memory(memories=[entry])
    return {"saved": True, "category": category}

agent = LlmAgent(
    name="preference_manager",
    model="gemini-2.5-flash",
    instruction="Help the user manage their preferences.",
    tools=[save_user_preference],
)
```

### Example 2 — delta memory update with `add_events_to_memory`

`add_events_to_memory` is an **incremental** update — pass only the events from the latest turn, not the full session. This is more efficient than `add_session_to_memory` when only the latest exchange matters.

```python
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

async def persist_conversation_turn(
    summary: str,
    tool_context: ToolContext,
) -> dict:
    """Persist only the current invocation's events to memory."""
    # Get only events from this invocation
    current_events = [
        e for e in tool_context.session.events
        if e.invocation_id == tool_context.invocation_id
    ]
    await tool_context.add_events_to_memory(events=current_events)
    return {"persisted_events": len(current_events)}

agent = LlmAgent(
    name="memory_aware",
    model="gemini-2.5-flash",
    instruction="Answer questions and optionally persist key information.",
    tools=[persist_conversation_turn],
)
```

### Example 3 — memory search from a tool

```python
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

async def recall_about(topic: str, tool_context: ToolContext) -> dict:
    """Search memory for information about a topic."""
    result = await tool_context.search_memory(topic)
    memories = [
        e.content.parts[0].text
        for m in result.memories
        for e in m.events
        if e.content and e.content.parts and e.content.parts[0].text
    ]
    return {"topic": topic, "recalled": memories[:5]}

agent = LlmAgent(
    name="memory_searcher",
    model="gemini-2.5-flash",
    instruction="Answer questions, using recall_about to search your memory bank.",
    tools=[recall_about],
)
```

### Comparison: when to use each method

| Method | Use when | Works with |
|---|---|---|
| `add_session_to_memory(session)` | You want to ingest the entire session at the end of a run | All services |
| `add_events_to_memory(events=...)` | You want to persist only the current turn (delta) | All services |
| `add_memory(memories=...)` | You want to write structured facts directly (not from events) | Vertex AI Memory Bank; raises `NotImplementedError` on `InMemoryMemoryService` |
| `search_memory(query)` | You want to retrieve relevant memories | All services |

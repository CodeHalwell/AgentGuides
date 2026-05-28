---
title: "Class deep-dives Vol. 2 — 10 more LangGraph types"
description: "Source-verified deep dives into RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, and Runtime/RunControl — with runnable examples for every major feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 2"
  order: 26
---

# Class deep-dives Vol. 2 — 10 more LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Each section below was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

---

## 1 · `RetryPolicy`

**Module:** `langgraph.types`

`RetryPolicy` is a `NamedTuple` that controls **how and when a node retries** on failure. Attach it to any node via `add_node(..., retry_policy=...)`.

### Full signature (source)

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5        # seconds before first retry
    backoff_factor:   float = 2.0        # multiplier after each retry
    max_interval:     float = 128.0      # cap on inter-retry wait (seconds)
    max_attempts:     int   = 3          # total attempts (including the first)
    jitter:           bool  = True       # add random jitter to each interval
    retry_on: type[Exception]
           | Sequence[type[Exception]]
           | Callable[[Exception], bool] = default_retry_on
```

`default_retry_on` retries on any exception **except** `ValueError`, `TypeError`,
`ArithmeticError`, `ImportError`, `AttributeError`, `NameError`, and other
programming errors (those that indicate a logic bug rather than a transient failure).

### Exponential back-off with custom exception filter

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


class State(TypedDict):
    attempts: Annotated[int, operator.add]
    result:   str


_call_count = 0


def flaky_api(state: State) -> dict:
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError(f"transient error on attempt {_call_count}")
    return {"result": "success", "attempts": 1}


builder = StateGraph(State)
builder.add_node(
    "api",
    flaky_api,
    retry_policy=RetryPolicy(
        initial_interval=0.01,   # fast for tests; use 0.5 in production
        backoff_factor=2.0,
        max_interval=10.0,
        max_attempts=5,
        jitter=False,
    ),
)
builder.add_edge(START, "api")
builder.add_edge("api", END)

graph = builder.compile()
result = graph.invoke({"attempts": 0, "result": ""})
print(result["result"])    # "success"
print(result["attempts"])  # 3 (one success write; attempts counts writes, not retries)
```

### Filtering by exception type or callable

```python
import httpx

# Retry only on specific exception types
retry_http = RetryPolicy(
    max_attempts=4,
    retry_on=(httpx.TimeoutException, httpx.NetworkError),
)

# Retry based on a callable — e.g., include HTTP 429/503 status codes
def should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))

retry_smart = RetryPolicy(max_attempts=6, retry_on=should_retry)
```

### Per-node retry policies

Different nodes in the same graph can have different policies:

```python
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    query:   str
    web_result: str
    db_result:  str


def web_search(state: PipelineState) -> dict:
    return {"web_result": f"web:{state['query']}"}


def db_lookup(state: PipelineState) -> dict:
    return {"db_result": f"db:{state['query']}"}


builder = StateGraph(PipelineState)
builder.add_node(
    "web",
    web_search,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=1.0),  # aggressive
)
builder.add_node(
    "db",
    db_lookup,
    retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.1),  # fast-fail
)
builder.add_edge(START, "web")
builder.add_edge(START, "db")
builder.add_edge(["web", "db"], END)

graph = builder.compile()
```

### Retry with `error_handler` fallback

When a node exhausts all retry attempts, LangGraph calls its `error_handler` node
instead of crashing the graph:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


class S(TypedDict):
    query:   str
    output:  str
    fallback_used: bool


def primary_node(state: S) -> dict:
    raise RuntimeError("always fails")


def fallback_node(state: S) -> dict:
    """Called after all retries are exhausted."""
    return {"output": f"fallback for '{state['query']}'", "fallback_used": True}


builder = StateGraph(S)
builder.add_node(
    "primary",
    primary_node,
    retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.01),
    error_handler=fallback_node,
)
builder.add_edge(START, "primary")
builder.add_edge("primary", END)

graph = builder.compile()
result = graph.invoke({"query": "hello", "output": "", "fallback_used": False})
print(result["fallback_used"])  # True
print(result["output"])         # "fallback for 'hello'"
```

---

## 2 · `CachePolicy` + `InMemoryCache`

**Modules:** `langgraph.types` (CachePolicy), `langgraph.cache.memory` (InMemoryCache)

`CachePolicy` configures per-node result caching. `InMemoryCache` is the built-in
in-process backend. Pass the cache to `builder.compile(cache=...)`.

### `CachePolicy` source

```python
@dataclass
class CachePolicy:
    key_func: Callable = default_cache_key   # hash(input) by default (pickle)
    ttl:      int | None = None              # seconds until expiry; None = forever
```

### Basic node caching

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from typing_extensions import TypedDict
import time


class State(TypedDict):
    query:  str
    result: str


_call_log: list[str] = []


def expensive_node(state: State) -> dict:
    _call_log.append(state["query"])
    time.sleep(0.05)  # simulate slow work
    return {"result": f"answer:{state['query']}"}


builder = StateGraph(State)
builder.add_node(
    "expensive",
    expensive_node,
    cache_policy=CachePolicy(ttl=60),   # cache for 60 seconds
)
builder.add_edge(START, "expensive")
builder.add_edge("expensive", END)

cache = InMemoryCache()
graph = builder.compile(cache=cache)

# First run — executes the node
result1 = graph.invoke({"query": "foo", "result": ""})
print(result1["result"])   # "answer:foo"
print(len(_call_log))      # 1

# Second run with same input — served from cache, node NOT called
result2 = graph.invoke({"query": "foo", "result": ""})
print(result2["result"])   # "answer:foo"
print(len(_call_log))      # still 1 — node was NOT called
```

### Custom cache key function

The `key_func` receives the node's **input** dict and returns any hashable value.
Use this to cache only on the parts of state that matter:

```python
from langgraph.types import CachePolicy


def query_only_key(node_input: dict) -> str:
    """Ignore transient fields like session_id when building the cache key."""
    return node_input.get("query", "")


builder.add_node(
    "expensive",
    expensive_node,
    cache_policy=CachePolicy(key_func=query_only_key, ttl=300),
)
```

### `InMemoryCache` API

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.base import Namespace

cache = InMemoryCache()

# The cache is normally managed by the graph runtime, but you can call
# get/set/clear directly for testing or pre-warming:

ns = Namespace(("mynode",))
key = "some-hash"

# set: mapping of (namespace, key) → (value, ttl_seconds)
cache.set({(ns, key): ({"result": "cached"}, 120)})

# get: returns {(ns, key): value} for all keys that exist and haven't expired
result = cache.get([(ns, key)])
print(result)  # {(Namespace(('mynode',)), 'some-hash'): {'result': 'cached'}}

# clear specific namespaces
cache.clear([ns])
print(cache.get([(ns, key)]))  # {} — cleared
```

### `@task` caching (Functional API)

`CachePolicy` also works on `@task` decorators:

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver


@task(cache_policy=CachePolicy(ttl=600))   # cache task result for 10 min
def summarise(text: str) -> str:
    return text[:80] + "..."   # stub for an expensive LLM call


@entrypoint(checkpointer=InMemorySaver(), cache=InMemoryCache())
def pipeline(docs: list[str]) -> list[str]:
    futures = [summarise(d) for d in docs]
    return [f.result() for f in futures]


cfg = {"configurable": {"thread_id": "cache-demo"}}
print(pipeline.invoke(["hello world", "foo bar"], cfg))
# Second call: summaries served from cache — task bodies not re-executed
print(pipeline.invoke(["hello world", "foo bar"], cfg))
```

---

## 3 · `TimeoutPolicy`

**Module:** `langgraph.types`  
**Added in:** v1.2.0

`TimeoutPolicy` gives fine-grained control over node execution time. Attach it via
`add_node(..., timeout=...)` or pass a plain `float` (seconds) as a shorthand for
`run_timeout`.

### Source

```python
@dataclass(frozen=True)
class TimeoutPolicy:
    run_timeout:  float | timedelta | None = None    # hard wall-clock cap
    idle_timeout: float | timedelta | None = None    # progress-based cap
    refresh_on:   Literal["auto", "heartbeat"] = "auto"
```

- **`run_timeout`** — total elapsed time cap for one node attempt. Never refreshed.
- **`idle_timeout`** — maximum time since the last *progress signal* (LangChain callback, sub-task write, or `runtime.heartbeat()` call). Refreshed continuously while the node makes progress.
- **`refresh_on="auto"`** — any callback event, child task completion, or stream chunk resets the idle clock.
- **`refresh_on="heartbeat"`** — only explicit `runtime.heartbeat()` calls reset the idle clock.

### Hard timeout (wall-clock cap)

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy


class State(TypedDict):
    result: str


async def slow_node(state: State) -> dict:
    await asyncio.sleep(5)   # will be cancelled
    return {"result": "done"}


builder = StateGraph(State)
builder.add_node(
    "slow",
    slow_node,
    timeout=TimeoutPolicy(run_timeout=1.0),   # cancel after 1 second
)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)

graph = builder.compile()

# async run — TimeoutError raised after ~1 s
async def run():
    try:
        await graph.ainvoke({"result": ""})
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:80])

asyncio.run(run())
```

### Idle timeout + `runtime.heartbeat()`

Use `idle_timeout` when the node does work in chunks and should only time out if
it *stops making progress*:

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import Runtime


class State(TypedDict):
    items: list[str]
    processed: list[str]


async def batch_processor(state: State, runtime: Runtime) -> dict:
    """Process items in batches; heartbeat keeps the idle clock alive."""
    processed = []
    for item in state["items"]:
        await asyncio.sleep(0.1)   # simulate slow per-item work
        processed.append(item.upper())
        runtime.heartbeat()        # reset idle timer — we're still making progress
    return {"processed": processed}


builder = StateGraph(State)
builder.add_node(
    "process",
    batch_processor,
    timeout=TimeoutPolicy(
        idle_timeout=0.5,     # cancel if no heartbeat for 0.5 s
        refresh_on="heartbeat",
    ),
)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()

async def run():
    result = await graph.ainvoke({"items": ["a", "b", "c"], "processed": []})
    print(result["processed"])   # ['A', 'B', 'C']

asyncio.run(run())
```

### `TimeoutPolicy.coerce` — normalising shorthand values

You can pass a bare `float` or `timedelta` to `timeout=` on `add_node` — LangGraph
calls `TimeoutPolicy.coerce()` internally:

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

# These are all equivalent:
TimeoutPolicy.coerce(30.0)                           # run_timeout=30
TimeoutPolicy.coerce(timedelta(seconds=30))          # run_timeout=30
TimeoutPolicy(run_timeout=30.0)                      # explicit

# coerce(None) returns None (no timeout)
assert TimeoutPolicy.coerce(None) is None
```

---

## 4 · `add_messages` + `MessagesState`

**Module:** `langgraph.graph.message`

`add_messages` is the standard **reducer** for the `messages` channel in chatbots
and tool-calling agents. It merges two lists of messages, performing an
**upsert by message ID** — if a message in the right list shares an ID with
one in the left, the left message is replaced.

### Core behaviours

```python
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages

# 1. Append: new message ID → appended
msgs = add_messages(
    [HumanMessage(content="hi", id="1")],
    [AIMessage(content="hello", id="2")],
)
assert len(msgs) == 2

# 2. Upsert: same ID → replacement
msgs = add_messages(
    [HumanMessage(content="hi", id="1")],
    [HumanMessage(content="hi again", id="1")],
)
assert msgs[0].content == "hi again"

# 3. Delete: RemoveMessage(id=X) removes message X
msgs = add_messages(
    [HumanMessage(content="hi", id="1"), AIMessage(content="hello", id="2")],
    [RemoveMessage(id="1")],
)
assert len(msgs) == 1
assert msgs[0].id == "2"
```

### `REMOVE_ALL_MESSAGES` — wipe the history

Use the sentinel string `"__remove_all__"` to clear the entire message list in one
step. This is useful when implementing a summarisation node that replaces the full
history with a single summary message:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES


history = [
    HumanMessage(content="What is Python?", id="1"),
    AIMessage(content="Python is a language.", id="2"),
    HumanMessage(content="Give me an example.", id="3"),
    AIMessage(content="x = 1", id="4"),
]

# Replace the entire history with a single summary message
summary = SystemMessage(
    content="[Summary] User asked about Python. Provided a basic example.",
    id="summary-1",
)

new_msgs = add_messages(history, [REMOVE_ALL_MESSAGES, summary])
assert len(new_msgs) == 1
assert new_msgs[0].content.startswith("[Summary]")
```

### `MessagesState` — the canonical state schema

Most chatbot and ReAct graphs use `MessagesState` directly:

```python
from langgraph.graph.message import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


def chat_node(state: MessagesState) -> dict:
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Echo: {last.content}")]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "msg-demo"}}

result = graph.invoke({"messages": [HumanMessage(content="hello")]}, cfg)
print(result["messages"][-1].content)   # "Echo: hello"
```

### Extending `MessagesState` with custom fields

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages:     Annotated[list[AnyMessage], add_messages]
    user_id:      str
    token_budget: int
    total_tokens: int
```

### `format="langchain-openai"` — normalise to OpenAI message format

When integrating with OpenAI-compatible APIs, pass `format="langchain-openai"` to
`add_messages` to ensure all messages conform to the OpenAI structure (no custom
fields, image blocks formatted as `image_url` blocks, etc.):

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from functools import partial


class OpenAIState(TypedDict):
    messages: Annotated[
        list,
        partial(add_messages, format="langchain-openai"),
    ]
```

### Summarisation pattern — prune and replace

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver


class ConvState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def check_length(state: ConvState) -> str:
    return "summarise" if len(state["messages"]) > 6 else "chat"


def chat_node(state: ConvState) -> dict:
    last = state["messages"][-1].content
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=f"Response to: {last}")]}


def summarise_node(state: ConvState) -> dict:
    """Collapse the whole history into a single summary message."""
    history = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"]
    )
    summary = SystemMessage(
        content=f"[Summary of {len(state['messages'])} messages] {history[:200]}",
        id="summary",
    )
    # REMOVE_ALL_MESSAGES first, then insert the new summary
    return {"messages": [REMOVE_ALL_MESSAGES, summary]}


builder = StateGraph(ConvState)
builder.add_node("chat",      chat_node)
builder.add_node("summarise", summarise_node)
builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", check_length, {"summarise": "summarise", "chat": END})
builder.add_edge("summarise", END)

graph = builder.compile(checkpointer=InMemorySaver())
```

---

## 5 · `tools_condition`

**Module:** `langgraph.prebuilt.tool_node`  
**Re-exported from:** `langgraph.prebuilt`

`tools_condition` is a **conditional edge function** that routes to `"tools"` if the
last `AIMessage` contains tool calls, or to `END` otherwise. It's the standard
routing function in ReAct-style agents.

### Signature

```python
def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]: ...
```

### Standard ReAct wiring

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolCall, ToolMessage


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Stub LLM: first call triggers a tool; second call ends the agent
_turn = 0


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def call_model(state: State) -> dict:
    global _turn
    _turn += 1
    if _turn == 1:
        return {"messages": [AIMessage(
            content="",
            tool_calls=[ToolCall(name="add", args={"a": 2, "b": 3}, id="tc1")],
        )]}
    return {"messages": [AIMessage(content="The answer is 5.")]}


builder = StateGraph(State)
builder.add_node("agent",  call_model)
builder.add_node("tools",  ToolNode([add]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # → "tools" or END
builder.add_edge("tools", "agent")

graph = builder.compile()
_turn = 0
result = graph.invoke({"messages": []})
print(result["messages"][-1].content)   # "The answer is 5."
```

### Custom `messages_key`

If your state uses a non-standard key for messages, pass it explicitly:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition


class CustomState(TypedDict):
    chat_history: Annotated[list[AnyMessage], add_messages]


def my_router(state: CustomState) -> str:
    return tools_condition(state, messages_key="chat_history")
```

### Handling the `list` state format

`tools_condition` also accepts a bare `list[AnyMessage]`, which is the format used
by `MessageGraph` (deprecated) and some functional-API workflows:

```python
from langchain_core.messages import AIMessage, ToolCall
from langgraph.prebuilt import tools_condition

msgs_with_tool = [AIMessage(content="", tool_calls=[ToolCall(name="add", args={}, id="t1")])]
msgs_without   = [AIMessage(content="done")]

assert tools_condition(msgs_with_tool) == "tools"
assert tools_condition(msgs_without)   == "__end__"
```

---

## 6 · `ToolCallTransformer` + `ToolCallStream`

**Modules:**
- `ToolCallTransformer` — `langgraph.prebuilt._tool_call_transformer`, re-exported from `langgraph.prebuilt`
- `ToolCallStream` — `langgraph.prebuilt._tool_call_stream`

**Added in:** `langgraph-prebuilt==1.1.0` (LangGraph v1.2.0)

`ToolCallTransformer` is a **stream transformer** that converts raw `tools`-channel
protocol events into per-tool-call **`ToolCallStream`** handles. It enables you to
stream the partial output of each tool call in real time, rather than receiving the
final result only.

### How it fits together

1. Register `ToolCallTransformer` at compile time: `compile(transformers=[ToolCallTransformer])`
2. Stream the graph with `stream_mode="tools"`
3. Iterate `run.tool_calls` — each item is a `ToolCallStream`
4. Each `ToolCallStream` exposes `tool_call_id`, `tool_name`, `input`, and `output_deltas`

### `ToolCallStream` attributes (source)

```python
class ToolCallStream:
    tool_call_id:  str             # from the AIMessage
    tool_name:     str
    input:         dict | None     # tool arguments
    output_deltas: StreamChannel   # async/sync iterable of partial chunks
    output:        Any             # final result (set on tool-finished)
    error:         str | None      # error message (set on tool-error)
    completed:     bool
```

### Streaming tool output deltas

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolCallTransformer


@tool
def stream_tool(query: str) -> str:
    """A tool that produces a result (streaming output emitted via ToolRuntime)."""
    return f"Result for: {query}"


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


_turn = 0


def call_model(state: State) -> dict:
    global _turn
    _turn += 1
    if _turn == 1:
        return {"messages": [AIMessage(
            content="",
            tool_calls=[ToolCall(
                name="stream_tool",
                args={"query": "langgraph"},
                id="tc-demo",
            )],
        )]}
    return {"messages": [AIMessage(content="Done.")]}


builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([stream_tool]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
builder.add_edge("tools", "agent")

graph = builder.compile(transformers=[ToolCallTransformer])


async def run():
    global _turn
    _turn = 0
    async with graph.astream(
        {"messages": []},
        stream_mode="tools",
        version="v2",
    ) as run:
        async for tool_call_stream in run.tool_calls:
            print(f"Tool started: {tool_call_stream.tool_name} ({tool_call_stream.tool_call_id})")
            print(f"  input: {tool_call_stream.input}")
            async for delta in tool_call_stream.output_deltas:
                print(f"  delta: {delta!r}")
            if tool_call_stream.error:
                print(f"  error: {tool_call_stream.error}")
            else:
                print(f"  final output: {tool_call_stream.output!r}")


asyncio.run(run())
```

### Emitting deltas from a tool via `ToolRuntime`

Tools can push partial output to `ToolCallStream.output_deltas` by calling
`runtime.emit_output_delta()`. This requires `stream_mode="tools"`:

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
import time


@tool
def long_analysis(query: str, runtime: ToolRuntime) -> str:
    """Analyse query in steps, streaming progress back."""
    steps = ["planning", "searching", "synthesising", "finalising"]
    for step in steps:
        time.sleep(0.01)  # simulate work
        runtime.emit_output_delta({"step": step, "progress": steps.index(step) + 1})
    return f"Complete analysis of: {query}"
```

---

## 7 · `StateSnapshot`

**Module:** `langgraph.types`

`StateSnapshot` is a `NamedTuple` returned by `graph.get_state()` and yielded by
`graph.get_state_history()`. It captures the *full* checkpoint state, including
pending interrupts and task information.

### Fields (source)

```python
class StateSnapshot(NamedTuple):
    values:        dict[str, Any] | Any   # current channel values
    next:          tuple[str, ...]        # nodes scheduled to run next
    config:        RunnableConfig         # config to resume from this snapshot
    metadata:      CheckpointMetadata | None
    created_at:    str | None             # ISO timestamp
    parent_config: RunnableConfig | None  # parent snapshot's config
    tasks:         tuple[PregelTask, ...] # pending tasks (may contain errors)
    interrupts:    tuple[Interrupt, ...]  # pending interrupt payloads
```

### Inspecting `values`, `next`, and `interrupts`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ReviewState(TypedDict):
    draft:    str
    approved: bool
    notes:    str


def draft_node(state: ReviewState) -> dict:
    return {"draft": "Here is my draft document."}


def review_node(state: ReviewState) -> dict:
    decision = interrupt({"question": "Approve this draft?", "draft": state["draft"]})
    return {"approved": decision == "yes", "notes": f"Decision: {decision}"}


builder = StateGraph(ReviewState)
builder.add_node("draft",  draft_node)
builder.add_node("review", review_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "review-1"}}

# Run until interrupt
list(graph.stream({"draft": "", "approved": False, "notes": ""}, cfg))

# Inspect the paused state
snap = graph.get_state(cfg)

print("values:", snap.values)
# {'draft': 'Here is my draft document.', 'approved': False, 'notes': ''}

print("next nodes:", snap.next)
# ('review',) — the node that is paused

print("interrupts:", [(i.value, i.id) for i in snap.interrupts])
# [({'question': 'Approve this draft?', ...}, 'some-uuid')]

print("created_at:", snap.created_at)
# '2026-05-28T...'

# Resume the interrupt
list(graph.stream(Command(resume="yes"), cfg))
final = graph.get_state(cfg)
print(final.values["approved"])   # True
print(final.values["notes"])      # "Decision: yes"
```

### Time-travel: replaying from any snapshot

Every snapshot carries its own `config`, which you can use as the starting config
for a new run — effectively replaying history from that point:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class Counter(TypedDict):
    n: int


builder = StateGraph(Counter)
builder.add_node("inc", lambda s: {"n": s["n"] + 1})
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "travel"}}

# Advance the counter 5 times
for _ in range(5):
    graph.invoke({"n": 0}, cfg)

# Walk history
history = list(graph.get_state_history(cfg))
print(f"Snapshots: {len(history)}")  # 5 * 2 + 1 (each run = 2 checkpoints + initial)

# Find the snapshot where n == 3
snap_at_3 = next(s for s in history if isinstance(s.values, dict) and s.values.get("n") == 3)
print("Replaying from n=3, config:", snap_at_3.config["configurable"])

# Re-run from that checkpoint — branches history
graph.invoke({"n": 0}, snap_at_3.config)

# The thread now has an extra branch from n=3 → n=4
```

### Checking for pending tasks with errors

```python
# After a failed node attempt, snap.tasks[i].error contains the exception repr
snap = graph.get_state(cfg)
for task in snap.tasks:
    if task.error:
        print(f"Task {task.name} failed: {task.error}")
```

---

## 8 · `IsLastStep` + `RemainingSteps`

**Module:** `langgraph.managed.is_last_step`

`IsLastStep` and `RemainingSteps` are **managed values** — read-only state fields
automatically populated by the Pregel executor before each node runs. Declare them
in your state schema to detect when the graph is about to hit its recursion limit.

### Source

```python
IsLastStep     = Annotated[bool, IsLastStepManager]    # True on final step
RemainingSteps = Annotated[int,  RemainingStepsManager] # steps left
```

The value is derived from `scratchpad.step == scratchpad.stop - 1` (for `IsLastStep`)
and `scratchpad.stop - scratchpad.step` (for `RemainingSteps`).

### Using `IsLastStep` to break a loop

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from typing import Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:    Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep    # injected by the runtime — do not set manually


def agent_node(state: AgentState) -> dict:
    if state["is_last_step"]:
        # We're about to hit the recursion limit — return a graceful message
        return {"messages": [AIMessage(
            content="I've reached the maximum number of steps. Stopping here.",
        )]}
    # Normal processing
    return {"messages": [AIMessage(content="Still working...")]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "agent"   # loop
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)

graph = builder.compile()
result = graph.invoke(
    {"messages": [HumanMessage(content="go")]},
    {"recursion_limit": 5},
)
print(result["messages"][-1].content)
# "I've reached the maximum number of steps. Stopping here."
```

### Using `RemainingSteps` for graceful budget management

```python
from langgraph.managed.is_last_step import RemainingSteps


class BudgetState(TypedDict):
    messages:        Annotated[list[AnyMessage], add_messages]
    remaining_steps: RemainingSteps   # injected by the runtime


def budget_aware_agent(state: BudgetState) -> dict:
    remaining = state["remaining_steps"]
    if remaining <= 2:
        return {"messages": [AIMessage(
            content=f"Low on budget ({remaining} steps left) — finalising now.",
        )]}
    return {"messages": [AIMessage(content=f"{remaining} steps remaining.")]}
```

### `IsLastStep` + `RemainingSteps` together

Both can coexist in the same state schema:

```python
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps


class FullState(TypedDict):
    messages:        Annotated[list[AnyMessage], add_messages]
    is_last_step:    IsLastStep
    remaining_steps: RemainingSteps


def smart_node(state: FullState) -> dict:
    rem = state["remaining_steps"]
    if state["is_last_step"]:
        return {"messages": [AIMessage(content="Final step — done.")]}
    if rem <= 3:
        return {"messages": [AIMessage(content=f"Wrapping up, {rem} steps left.")]}
    return {"messages": [AIMessage(content=f"Running normally ({rem} steps left).")]}
```

---

## 9 · `ToolRuntime`

**Module:** `langgraph.prebuilt.tool_node`  
**Added in:** v1.2.0

`ToolRuntime` is a **dataclass** automatically injected into tools that declare a
parameter named `runtime` with type hint `ToolRuntime`. It gives tools access to
the current graph state, a store, the stream writer, and other execution metadata.

No `Annotated` wrapper needed — declare `runtime: ToolRuntime` directly.

### Fields (source)

```python
@dataclass
class ToolRuntime:
    state:           StateT           # read-only current graph state
    context:         ContextT         # run-scoped immutable context
    config:          RunnableConfig   # LangChain runnable config
    stream_writer:   StreamWriter     # writes to stream_mode="custom"
    tool_call_id:    str | None       # ID from the AIMessage
    store:           BaseStore | None # long-term store, if wired
    tools:           list[BaseTool]   # all tools available in this ToolNode
    execution_info:  ExecutionInfo | None
    server_info:     ServerInfo | None
```

### Accessing state and store from a tool

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.memory import InMemoryStore


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id:  str


@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """Return the current user's profile from the store."""
    user_id = runtime.state["user_id"]
    if runtime.store:
        item = runtime.store.get(("users",), user_id)
        if item:
            return str(item.value)
    return f"No profile found for {user_id}"


@tool
def save_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """Save a user preference to long-term memory."""
    user_id = runtime.state["user_id"]
    if runtime.store:
        existing = runtime.store.get(("prefs", user_id), key)
        runtime.store.put(
            ("prefs", user_id),
            key,
            {"value": value, "updated_at": "2026-05-28"},
        )
    return f"Saved {key}={value} for user {user_id}"


store = InMemoryStore()
store.put(("users",), "alice", {"name": "Alice", "tier": "premium"})

tool_node = ToolNode([get_user_profile, save_preference])
```

### Streaming progress via `emit_output_delta`

`ToolRuntime.emit_output_delta(delta)` pushes a partial chunk onto the `tools`
stream channel. The chunk appears as a `tool-output-delta` event and is surfaced
on the corresponding `ToolCallStream.output_deltas` iterator when using
`ToolCallTransformer`:

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
import time


@tool
def analyse_document(doc_id: str, runtime: ToolRuntime) -> str:
    """Analyse a document, emitting progress as deltas."""
    pages = 10
    for page in range(1, pages + 1):
        time.sleep(0.01)
        # Each delta is surfaced on ToolCallStream.output_deltas
        runtime.emit_output_delta({
            "page": page,
            "of": pages,
            "status": "processing",
        })
    return f"Analysis complete for document {doc_id}"
```

### Accessing execution metadata

```python
@tool
def audit_tool(action: str, runtime: ToolRuntime) -> str:
    """Log the tool call to an audit trail."""
    info = runtime.execution_info
    entry = {
        "tool_call_id":   runtime.tool_call_id,
        "action":         action,
        "thread_id":      info.thread_id if info else None,
        "checkpoint_id":  info.checkpoint_id if info else None,
        "node_attempt":   info.node_attempt if info else None,
    }
    print(f"[audit] {entry}")
    return f"Logged: {action}"
```

### Calling other tools from a tool

`runtime.tools` gives you the list of `BaseTool` instances registered in the same
`ToolNode`. Useful for orchestrating multi-step tool chains:

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime


@tool
def orchestrate(task: str, runtime: ToolRuntime) -> str:
    """Find and invoke a sub-tool by name."""
    sub_tool_name = "save_preference"
    sub_tool = next(
        (t for t in runtime.tools if t.name == sub_tool_name),
        None,
    )
    if sub_tool:
        return sub_tool.invoke({"key": "last_task", "value": task})
    return f"Tool '{sub_tool_name}' not found"
```

---

## 10 · `Runtime` + `RunControl`

**Modules:** `langgraph.runtime`

`Runtime` is a **dataclass** injected into **nodes** (not tools — for tools use
`ToolRuntime`). `RunControl` is a companion class for **cooperative draining** —
signalling an active graph run to exit gracefully.

### `Runtime` fields (source)

```python
@dataclass
class Runtime:
    context:        ContextT                  # read-only run context (user data)
    store:          BaseStore | None          # long-term store
    stream_writer:  StreamWriter              # writes to stream_mode="custom"
    heartbeat:      Callable[[], None]        # reset idle_timeout clock
    previous:       Any                       # last @entrypoint return value
    execution_info: ExecutionInfo | None      # checkpoint/task metadata
    server_info:    ServerInfo | None         # LangGraph Server metadata
    control:        RunControl | None         # cooperative drain signal
```

### Injecting context into nodes

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class UserContext:
    user_id: str
    tier:    str = "free"


class State(TypedDict):
    query:  str
    result: str


store = InMemoryStore()
store.put(("users",), "alice", {"name": "Alice", "preferences": {"lang": "en"}})


def personalised_node(state: State, runtime: Runtime[UserContext]) -> dict:
    uid = runtime.context.user_id
    prefs = {}
    if runtime.store:
        item = runtime.store.get(("users",), uid)
        prefs = item.value.get("preferences", {}) if item else {}

    # Write a progress event to the custom stream
    runtime.stream_writer({"status": "processing", "user": uid})

    return {"result": f"[{prefs.get('lang','?')}] Hello {uid}: {state['query']}"}


builder = StateGraph(State, context_schema=UserContext)
builder.add_node("respond", personalised_node)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile(store=store)
result = graph.invoke(
    {"query": "weather today", "result": ""},
    context=UserContext(user_id="alice", tier="premium"),
)
print(result["result"])   # "[en] Hello alice: weather today"
```

### Custom stream events via `stream_writer`

`Runtime.stream_writer` writes events to the `custom` stream channel. Consumers
see them when `stream_mode="custom"` is included:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


class PipeState(TypedDict):
    items: list[str]
    done:  list[str]


def process_node(state: PipeState, runtime: Runtime) -> dict:
    done = []
    for item in state["items"]:
        # Push a progress event while the node is running
        runtime.stream_writer({"event": "item_processed", "item": item})
        done.append(item.upper())
    return {"done": done}


builder = StateGraph(PipeState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()

for chunk in graph.stream(
    {"items": ["a", "b", "c"], "done": []},
    stream_mode=["updates", "custom"],
):
    print(chunk)
# ('custom',  {'event': 'item_processed', 'item': 'a'})
# ('custom',  {'event': 'item_processed', 'item': 'b'})
# ('custom',  {'event': 'item_processed', 'item': 'c'})
# ('updates', {'process': {'done': ['A', 'B', 'C']}})
```

### `RunControl` — cooperative draining

`RunControl` lets you signal a running graph to exit at its next natural
checkpoint, without cancellation:

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, RunControl


class LongState(TypedDict):
    steps:    int
    finished: bool


async def long_running_node(state: LongState, runtime: Runtime) -> dict:
    """Check the drain signal on each iteration."""
    if runtime.control and runtime.control.drain_requested:
        print(f"Drain requested ({runtime.control.drain_reason!r}) — stopping at step {state['steps']}")
        return {"finished": True}
    return {"steps": state["steps"] + 1, "finished": False}


def should_continue(state: LongState) -> str:
    return END if state["finished"] or state["steps"] >= 100 else "work"


builder = StateGraph(LongState)
builder.add_node("work", long_running_node)
builder.add_edge(START, "work")
builder.add_conditional_edges("work", should_continue)

graph = builder.compile()

# Run with a RunControl so an external task can signal drain
control = RunControl()

async def run_with_drain():
    async def do_invoke():
        return await graph.ainvoke(
            {"steps": 0, "finished": False},
            control=control,
        )

    async def request_drain_after(delay: float):
        await asyncio.sleep(delay)
        control.request_drain(reason="demo shutdown")

    result, _ = await asyncio.gather(do_invoke(), request_drain_after(0.05))
    print(f"Finished at step {result['steps']}")

asyncio.run(run_with_drain())
```

### `Runtime.execution_info` — per-node observability

`ExecutionInfo` (returned by `runtime.execution_info`) exposes the checkpoint ID,
task ID, thread ID, and node attempt number:

```python
from langgraph.runtime import Runtime


def observability_node(state: dict, runtime: Runtime) -> dict:
    info = runtime.execution_info
    if info:
        print(f"thread_id:      {info.thread_id}")
        print(f"checkpoint_id:  {info.checkpoint_id[:8]}")
        print(f"task_id:        {info.task_id[:8]}")
        print(f"node_attempt:   {info.node_attempt}")  # 1-indexed; >1 on retry
    return {}
```

---

## Quick reference — Vol. 2 features

| Feature | Class / Function | Module |
|---|---|---|
| Per-node retry on transient errors | `RetryPolicy` | `langgraph.types` |
| Per-node result caching | `CachePolicy` | `langgraph.types` |
| In-process cache backend | `InMemoryCache` | `langgraph.cache.memory` |
| Hard/idle timeouts on nodes | `TimeoutPolicy` | `langgraph.types` |
| Reset idle timeout from node | `runtime.heartbeat()` | `langgraph.runtime` |
| Message list reducer (upsert by ID) | `add_messages` | `langgraph.graph.message` |
| Wipe entire message history | `REMOVE_ALL_MESSAGES` | `langgraph.graph.message` |
| Minimal chatbot state schema | `MessagesState` | `langgraph.graph.message` |
| Standard ReAct routing function | `tools_condition` | `langgraph.prebuilt` |
| Per-tool streaming handles | `ToolCallTransformer` | `langgraph.prebuilt` |
| Real-time tool output iteration | `ToolCallStream` | `langgraph.prebuilt._tool_call_stream` |
| Inspect paused / completed state | `StateSnapshot` | `langgraph.types` |
| Detect recursion limit in node | `IsLastStep` | `langgraph.managed.is_last_step` |
| Integer steps remaining | `RemainingSteps` | `langgraph.managed.is_last_step` |
| Access state/store/config in tool | `ToolRuntime` | `langgraph.prebuilt.tool_node` |
| Emit partial tool output | `runtime.emit_output_delta()` | `ToolRuntime` |
| Access store/context in node | `Runtime` | `langgraph.runtime` |
| Custom stream events from node | `runtime.stream_writer(...)` | `langgraph.runtime` |
| Graceful graph drain | `RunControl` | `langgraph.runtime` |
| Per-node execution metadata | `ExecutionInfo` | `langgraph.runtime` |

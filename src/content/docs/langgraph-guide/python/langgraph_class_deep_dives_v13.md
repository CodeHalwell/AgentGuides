---
title: "Class deep-dives Vol. 13 — Channels, policies & tool runtime"
description: "Source-verified deep dives into ToolRuntime, ToolNode wrap_tool_call, create_react_agent v2, TimeoutPolicy, CachePolicy, BinaryOperatorAggregate, Topic, EphemeralValue, NamedBarrierValue, and AnyValue — with multiple runnable examples for each class."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 13"
  order: 44
---

# Class deep-dives Vol. 13 — Channels, policies & tool runtime

Verified against **`langgraph==1.2.4`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class | Module |
|---|-------|--------|
| 1 | `ToolRuntime` | `langgraph.prebuilt.tool_node` |
| 2 | `ToolNode` — `wrap_tool_call` / `awrap_tool_call` | `langgraph.prebuilt.tool_node` |
| 3 | `create_react_agent` v2 | `langgraph.prebuilt.chat_agent_executor` |
| 4 | `TimeoutPolicy` | `langgraph.types` |
| 5 | `CachePolicy` | `langgraph.types` |
| 6 | `BinaryOperatorAggregate` | `langgraph.channels.binop` |
| 7 | `Topic` | `langgraph.channels.topic` |
| 8 | `EphemeralValue` | `langgraph.channels.ephemeral_value` |
| 9 | `NamedBarrierValue` | `langgraph.channels.named_barrier_value` |
| 10 | `AnyValue` | `langgraph.channels.any_value` |

---

## 1 · `ToolRuntime` — unified runtime injection for tools

**Module:** `langgraph.prebuilt.tool_node`  
**Import:**
```python
from langgraph.prebuilt.tool_node import ToolRuntime
```

`ToolRuntime` is a `dataclass` that the `ToolNode` automatically injects into any tool function that declares a `runtime: ToolRuntime` parameter. It replaces the older piecemeal approach of annotating separate `InjectedState`, `InjectedStore`, and `get_stream_writer()` calls by bundling all runtime context into one object.

### Source signature (1.2.4)

```python
@dataclass
class ToolRuntime(_DirectlyInjectedToolArg, Generic[ContextT, StateT]):
    state: StateT             # current graph state dict / TypedDict / BaseModel
    tool_call_id: str         # id of the tool call that triggered this invocation
    config: RunnableConfig    # LangChain RunnableConfig for the current run
    context: ContextT         # context_schema instance set at graph.invoke time
    store: BaseStore | None   # persistent store (if one was compiled into the graph)
    stream_writer: StreamWriter  # callable to push custom stream events
    tools: list[BaseTool]     # every tool available in the ToolNode
```

No `Annotated[..., InjectedState()]` wrapper is needed — just name the parameter `runtime` with a `ToolRuntime` type hint.

### Example 1: Reading state, writing to store, and streaming

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
from langgraph.checkpoint.memory import InMemorySaver


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str


@tool
def save_note(note: str, runtime: ToolRuntime) -> str:
    """Save a note tied to the current user."""
    user_id = runtime.state["user_id"]         # read graph state
    runtime.store.put(                          # write to persistent store
        ("notes", user_id),
        "latest",
        {"text": note},
    )
    runtime.stream_writer({"note_saved": note}) # emit a custom stream event
    return f"Saved note for user {user_id}"


store = InMemoryStore()
tool_node = ToolNode([save_note])

graph = StateGraph(AgentState)
graph.add_node("tools", tool_node)
# ... add model node and edges
compiled = graph.compile(checkpointer=InMemorySaver(), store=store)
```

### Example 2: Accessing the tools list and context

```python
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime


@dataclass
class AppContext:
    locale: str = "en"
    max_tokens: int = 1024


@tool
def list_available_tools(runtime: ToolRuntime) -> str:
    """Return the names of every tool registered in this ToolNode."""
    names = [t.name for t in runtime.tools]
    locale = runtime.context.locale       # read context_schema value
    return f"[{locale}] Available: {', '.join(names)}"


@tool
def get_run_id(runtime: ToolRuntime) -> str:
    """Return the LangChain run ID for the current execution."""
    return str(runtime.config.get("run_id", "unknown"))
```

### Example 3: Per-call audit logging via `tool_call_id`

```python
import datetime
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.memory import InMemoryStore


@tool
def search_database(query: str, runtime: ToolRuntime) -> str:
    """Search the database and log the call for auditing."""
    call_id = runtime.tool_call_id
    user_id = runtime.state.get("user_id", "anon")

    # Persist audit record
    runtime.store.put(
        ("audit", user_id),
        call_id,
        {
            "query": query,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "call_id": call_id,
        },
    )

    # Simulate a DB result
    return f"Results for '{query}'"
```

---

## 2 · `ToolNode` — `wrap_tool_call` / `awrap_tool_call` interceptors

**Module:** `langgraph.prebuilt.tool_node`  
**Import:**
```python
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
```

`ToolNode` gained two new constructor parameters in 1.2.x:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `wrap_tool_call` | `Callable[[ToolCallRequest, Callable], ToolMessage \| Command]` | Sync interceptor wrapping every tool execution |
| `awrap_tool_call` | `AsyncCallable[...]` | Async counterpart; falls back to `wrap_tool_call` if omitted |

The interceptor receives a `ToolCallRequest` (containing `tool_call`, `tool`, `state`, `runtime`) and an `execute` callable. It can: modify the request before execution, add retry logic, cache results, or return an early `Command`.

### Source signature (1.2.4)

```python
class ToolNode(RunnableLambda):
    def __init__(
        self,
        tools: Sequence[BaseTool | Callable],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool | str | Callable | type[Exception] | tuple[type[Exception], ...] = True,
        messages_key: str = "messages",
        wrap_tool_call: ToolCallWrapper | None = None,      # NEW
        awrap_tool_call: AsyncToolCallWrapper | None = None, # NEW
    ) -> None: ...
```

`ToolCallRequest` fields:
```python
@dataclass
class ToolCallRequest:
    tool_call: ToolCall       # {"name": str, "args": dict, "id": str}
    tool: BaseTool | None     # the resolved BaseTool, or None for unknown tools
    state: Any                # graph state
    runtime: ToolRuntime      # full runtime context
    # Use .override() to create a modified copy:
    def override(self, **kwargs) -> ToolCallRequest: ...
    # Call execute() to run the tool with the (possibly modified) request:
    # (pass a ToolCallRequest to override; omit to run as-is)
    def execute(self, request: ToolCallRequest | None = None) -> ToolMessage: ...
```

### Example 1: Rate-limiting interceptor

```python
import time
from collections import defaultdict
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

# Simple per-tool call counter
_call_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 5  # max calls per minute


def rate_limit_wrapper(request: ToolCallRequest, execute) -> ToolMessage:
    tool_name = request.tool_call["name"]
    now = time.time()

    # Remove calls older than 60 seconds
    _call_counts[tool_name] = [t for t in _call_counts[tool_name] if now - t < 60]

    if len(_call_counts[tool_name]) >= RATE_LIMIT:
        return ToolMessage(
            content=f"Rate limit exceeded for '{tool_name}'. Try again later.",
            tool_call_id=request.tool_call["id"],
        )

    _call_counts[tool_name].append(now)
    return execute()


@tool
def fetch_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


tool_node = ToolNode([fetch_weather], wrap_tool_call=rate_limit_wrapper)
```

### Example 2: Request modification — argument sanitization

```python
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest


def sanitize_args(request: ToolCallRequest, execute) -> ToolMessage:
    """Strip PII from tool arguments before execution."""
    import re

    clean_args = {}
    for key, val in request.tool_call["args"].items():
        if isinstance(val, str):
            # Mask email addresses
            val = re.sub(r"[\w.+-]+@[\w-]+\.[a-z]+", "[EMAIL]", val)
        clean_args[key] = val

    # Build a modified request with sanitized args
    clean_tool_call = {**request.tool_call, "args": clean_args}
    modified = request.override(tool_call=clean_tool_call)
    return execute(modified)


tool_node = ToolNode([...], wrap_tool_call=sanitize_args)
```

### Example 3: Async caching interceptor

```python
import hashlib
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

_cache: dict[str, str] = {}


async def cached_wrapper(request: ToolCallRequest, execute) -> ToolMessage:
    """Cache tool results by tool name + serialised args."""
    key = hashlib.md5(
        json.dumps(
            {"tool": request.tool_call["name"], "args": request.tool_call["args"]},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if key in _cache:
        return ToolMessage(
            content=_cache[key],
            tool_call_id=request.tool_call["id"],
        )

    result = await execute()  # returns ToolMessage
    _cache[key] = result.content
    return result


tool_node = ToolNode([...], awrap_tool_call=cached_wrapper)
```

---

## 3 · `create_react_agent` v2 — pre/post hooks, structured output, context

**Module:** `langgraph.prebuilt.chat_agent_executor`  
**Import:**
```python
from langgraph.prebuilt import create_react_agent
```

> **Deprecation notice (1.2.4):** `create_react_agent` is deprecated in favour of `create_agent` from the `langchain` package. The API is similar; see the [migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1). All examples below work unchanged on 1.2.4.

`create_react_agent` returns a `CompiledStateGraph` that cycles between an LLM node and a `ToolNode` until the model stops calling tools. The v2 graph (`version="v2"`, the default) adds `pre_model_hook`, `post_model_hook`, and a separate structured-output call.

### Signature (abbreviated)

```python
def create_react_agent(
    model: str | BaseChatModel | Callable,
    tools: Sequence[BaseTool | Callable] | ToolNode,
    *,
    prompt: SystemMessage | str | Callable | Runnable | None = None,
    response_format: dict | type[BaseModel] | tuple | None = None,
    pre_model_hook: Runnable | Callable | None = None,
    post_model_hook: Runnable | Callable | None = None,
    state_schema: type | None = None,
    context_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | bool | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    version: Literal["v1", "v2"] = "v2",
    name: str | None = None,
) -> CompiledStateGraph: ...
```

### Example 1: Message trimming via `pre_model_hook`

```python
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, RemoveMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.prebuilt import create_react_agent


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def trim_to_last_10(state: State) -> dict[str, Any]:
    """Keep only the 10 most recent messages to avoid context overflow."""
    msgs = state["messages"]
    if len(msgs) <= 10:
        return {"messages": msgs}
    # Replace the whole history: remove all, then re-add the tail
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + msgs[-10:],
    }


@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price."""
    return f"{ticker}: $150.00"


agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8"),
    tools=[get_stock_price],
    pre_model_hook=trim_to_last_10,
    checkpointer=True,   # auto-creates InMemorySaver
)
```

### Example 2: Guardrails via `post_model_hook`

```python
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def require_human_approval(state: State) -> dict[str, Any]:
    """Pause before any tool that touches financial data."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        sensitive = {"transfer_funds", "delete_account"}
        if any(tc["name"] in sensitive for tc in last.tool_calls):
            decision = interrupt({"message": "Approve sensitive tool call?", "calls": last.tool_calls})
            if decision != "approve":
                return {"messages": [AIMessage(content="Operation cancelled by user.")]}
    return {}


@tool
def transfer_funds(amount: float, to_account: str) -> str:
    """Transfer funds to an account."""
    return f"Transferred ${amount} to {to_account}"


agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8"),
    tools=[transfer_funds],
    post_model_hook=require_human_approval,
    checkpointer=True,
)
```

### Example 3: Structured output with `response_format`

```python
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


class ResearchReport(BaseModel):
    """Structured research output."""
    title: str
    summary: str
    key_findings: list[str]
    confidence_score: float


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Top results for: {query}"


agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8"),
    tools=[web_search],
    response_format=ResearchReport,   # adds a second LLM call after the loop
)

# After the agent loop finishes, state["structured_response"] is a ResearchReport
result = agent.invoke({"messages": [{"role": "user", "content": "Research quantum computing"}]})
report: ResearchReport = result["structured_response"]
print(report.title, report.confidence_score)
```

### Example 4: Dynamic model selection via `context_schema`

```python
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime


@dataclass
class ModelConfig:
    use_fast_model: bool = False


fast_model = ChatAnthropic(model="claude-haiku-4-5-20251001")
smart_model = ChatAnthropic(model="claude-opus-4-8")


def pick_model(state, runtime: Runtime[ModelConfig]):
    model = fast_model if runtime.context.use_fast_model else smart_model
    return model.bind_tools([classify_document])


@tool
def classify_document(text: str) -> str:
    """Classify a document."""
    return "invoice"


agent = create_react_agent(
    model=pick_model,
    tools=[classify_document],
    context_schema=ModelConfig,
)

# Pass context at invoke time
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Classify this doc"}]},
    config={"configurable": {"context": ModelConfig(use_fast_model=True)}},
)
```

---

## 4 · `TimeoutPolicy` — per-node deadline control

**Module:** `langgraph.types`  
**Import:**
```python
from langgraph.types import TimeoutPolicy
```

`TimeoutPolicy` is a frozen dataclass with two independent timeout axes. It relies on **asyncio cooperative cancellation** — synchronous blocking calls that hold the GIL will not be interrupted until the event loop regains control.

### Source signature (1.2.4)

```python
@dataclass(frozen=True)
class TimeoutPolicy:
    run_timeout: float | timedelta | None = None
    # Hard wall-clock cap per attempt. Never refreshed by heartbeats.

    idle_timeout: float | timedelta | None = None
    # Max time without observable progress per attempt.

    refresh_on: Literal["auto", "heartbeat"] = "auto"
    # "auto"      → idle_timeout refreshed by standard graph callbacks AND runtime.heartbeat()
    # "heartbeat" → idle_timeout refreshed ONLY by explicit runtime.heartbeat() calls
```

You can pass a plain `float` (seconds) anywhere `TimeoutPolicy` is accepted — it is coerced to `TimeoutPolicy(run_timeout=float)`.

### Example 1: Hard deadline on a node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from typing_extensions import TypedDict
import asyncio


class State(TypedDict):
    result: str


async def slow_api_call(state: State) -> dict:
    await asyncio.sleep(5)          # simulates a slow external call
    return {"result": "done"}


graph = StateGraph(State)
graph.add_node(
    "call_api",
    slow_api_call,
    timeout=TimeoutPolicy(run_timeout=3.0),   # abort after 3 s, no matter what
)
graph.add_edge(START, "call_api")
graph.add_edge("call_api", END)

app = graph.compile()
# Running this will raise asyncio.TimeoutError after 3 seconds
```

### Example 2: Idle timeout with heartbeat-only refresh

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
import asyncio


class State(TypedDict):
    chunks: list[str]


async def streaming_processor(state: State, runtime: Runtime) -> dict:
    """Process a long stream, reporting progress via heartbeat."""
    chunks = []
    async for chunk in some_async_generator():
        chunks.append(chunk)
        await runtime.heartbeat()   # resets idle_timeout on each chunk
    return {"chunks": chunks}


async def some_async_generator():
    for i in range(10):
        await asyncio.sleep(0.5)
        yield f"chunk_{i}"


graph = StateGraph(State)
graph.add_node(
    "stream",
    streaming_processor,
    timeout=TimeoutPolicy(
        idle_timeout=2.0,          # abort if 2 s passes without a heartbeat
        refresh_on="heartbeat",    # ONLY heartbeat() refreshes the idle clock
    ),
)
graph.add_edge(START, "stream")
graph.add_edge("stream", END)
```

### Example 3: Combining run_timeout and idle_timeout

```python
from langgraph.types import TimeoutPolicy

# A node that must finish within 30 s total AND must not stall for more than 5 s
policy = TimeoutPolicy(
    run_timeout=30.0,    # absolute deadline
    idle_timeout=5.0,    # stall detector
    refresh_on="auto",   # standard callbacks refresh idle_timeout
)

# Apply at graph level via set_node_defaults
graph.set_node_defaults(timeout=policy)
```

---

## 5 · `CachePolicy` — node result memoisation

**Module:** `langgraph.types`  
**Import:**
```python
from langgraph.types import CachePolicy
```

`CachePolicy` tells the graph runtime to memoize a node's return value keyed on the node's input state. The cache backend is provided at `graph.compile(cache=...)`.

### Source signature (1.2.4)

```python
@dataclass
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key
    # Callable(state) -> str  — defaults to pickle hash of the full input
    
    ttl: int | None = None
    # Seconds before a cache entry expires. None = never expires.
```

### Example 1: Caching an expensive retrieval node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from typing_extensions import TypedDict


class RAGState(TypedDict):
    query: str
    documents: list[str]


def retrieve(state: RAGState) -> dict:
    """Expensive vector search — cache results for 5 minutes."""
    print(f"[DB] Searching for: {state['query']}")
    return {"documents": [f"doc about {state['query']}"]}


graph = StateGraph(RAGState)
graph.add_node(
    "retrieve",
    retrieve,
    cache_policy=CachePolicy(ttl=300),   # cache for 5 minutes
)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", END)

cache = InMemoryCache()
app = graph.compile(cache=cache)

# First call hits the retrieval function
app.invoke({"query": "quantum computing", "documents": []})

# Second call with same query is served from cache — retrieve() never called
app.invoke({"query": "quantum computing", "documents": []})
```

### Example 2: Custom cache key — ignore noisy state fields

```python
from langgraph.types import CachePolicy


def query_only_key(state: dict) -> str:
    """Cache only on the query, ignoring session_id and timestamp."""
    import hashlib
    query = state.get("query", "")
    return hashlib.sha256(query.encode()).hexdigest()


graph.add_node(
    "retrieve",
    retrieve,
    cache_policy=CachePolicy(key_func=query_only_key, ttl=3600),
)
```

### Example 3: Graph-wide caching with `set_node_defaults`

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

# All nodes share the same TTL-based cache policy
graph.set_node_defaults(cache_policy=CachePolicy(ttl=60))

# Individual nodes can still override
graph.add_node("expensive", slow_node, cache_policy=CachePolicy(ttl=600))
graph.add_node("cheap", fast_node, cache_policy=None)  # disable for this node

app = graph.compile(cache=InMemoryCache())
```

---

## 6 · `BinaryOperatorAggregate` — custom reduction channels

**Module:** `langgraph.channels.binop`  
**Import:**
```python
from langgraph.channels.binop import BinaryOperatorAggregate
```

`BinaryOperatorAggregate` is the channel type backing `Annotated[T, reducer_fn]` state fields. When multiple nodes write to the same field in the same step, LangGraph calls `operator(current, new)` for each write. It is the mechanism behind `add_messages`.

### Source signature (1.2.4)

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(
        self,
        typ: type[Value],
        operator: Callable[[Value, Value], Value],
    ): ...
    # update(): calls operator(self.value, each_new_value) in order
    # get():    returns self.value; raises EmptyChannelError if never written
```

### Example 1: Numeric accumulation

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.binop import BinaryOperatorAggregate


def add_reducer(a: int, b: int) -> int:
    return a + b


class ScoringState(TypedDict):
    # Using Annotated + reducer function (equivalent to BinaryOperatorAggregate internally)
    total_score: Annotated[int, add_reducer]
    labels: Annotated[list[str], operator.add]  # list concatenation


def score_a(state: ScoringState) -> dict:
    return {"total_score": 10, "labels": ["category_a"]}


def score_b(state: ScoringState) -> dict:
    return {"total_score": 5, "labels": ["category_b"]}


graph = StateGraph(ScoringState)
graph.add_node("a", score_a)
graph.add_node("b", score_b)
graph.add_edge(START, "a")
graph.add_edge(START, "b")    # a and b run in parallel
graph.add_edge("a", END)
graph.add_edge("b", END)

app = graph.compile()
result = app.invoke({"total_score": 0, "labels": []})
# result["total_score"] == 15  (10 + 5, both writes merged)
# result["labels"] == ["category_a", "category_b"]
print(result)
```

### Example 2: Using `BinaryOperatorAggregate` directly as a channel annotation

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.binop import BinaryOperatorAggregate


def keep_max(a: float, b: float) -> float:
    """Reducer that keeps the highest confidence score."""
    return max(a, b)


class ClassifierState(TypedDict):
    confidence: Annotated[float, keep_max]   # only the highest score survives
    candidates: Annotated[list[str], operator.add]


def classifier_1(state: ClassifierState) -> dict:
    return {"confidence": 0.72, "candidates": ["invoice"]}

def classifier_2(state: ClassifierState) -> dict:
    return {"confidence": 0.91, "candidates": ["receipt"]}

def classifier_3(state: ClassifierState) -> dict:
    return {"confidence": 0.65, "candidates": ["contract"]}
```

### Example 3: Set union channel

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


def union_reducer(a: set, b: set) -> set:
    return a | b


class DedupeState(TypedDict):
    urls_seen: Annotated[set, union_reducer]
    results: Annotated[list[str], operator.add]


def scrape_page(url: str) -> dict:
    def _node(state: DedupeState) -> dict:
        return {"urls_seen": {url}, "results": [f"scraped:{url}"]}
    return _node


def fan_out(state: DedupeState):
    urls = ["https://a.com", "https://b.com", "https://a.com"]
    unique = list(dict.fromkeys(urls))          # preserve order, dedupe
    return [Send("scrape", url) for url in unique]


graph = StateGraph(DedupeState)
graph.add_node("scrape", lambda state: {})      # placeholder — See Send examples
graph.add_conditional_edges(START, fan_out, ["scrape"])
graph.add_edge("scrape", END)
app = graph.compile()
```

---

## 7 · `Topic` — per-step multi-value PubSub channel

**Module:** `langgraph.channels.topic`  
**Import:**
```python
from langgraph.channels.topic import Topic
```

`Topic` collects *all* values written to it during a step (unlike `LastValue`, which keeps only the last). It can operate in two modes:

| `accumulate` | Behaviour |
|---|---|
| `False` (default) | Channel is cleared at the **start** of each step; only values written in the current step are visible. |
| `True` | Values accumulate across steps until explicitly cleared. |

### Source signature (1.2.4)

```python
class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    def __init__(self, typ: type[Value], accumulate: bool = False) -> None:
        ...
    # ValueType  → Sequence[Value]   (you read a list)
    # UpdateType → Value | list[Value]  (you write single values or lists)
```

### Example 1: Collecting parallel worker outputs

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.topic import Topic
from langgraph.types import Send


class PipelineState(TypedDict):
    inputs: list[str]
    # Topic channel: each worker appends; cleared each step
    partial_results: Annotated[Sequence[str], Topic(str)]


def fan_out(state: PipelineState):
    return [Send("worker", item) for item in state["inputs"]]


def worker(item: str) -> dict:
    # Each parallel Send invocation writes one value
    return {"partial_results": f"processed:{item}"}


def aggregate(state: PipelineState) -> dict:
    # partial_results contains ALL values written by workers this step
    combined = " | ".join(state["partial_results"])
    return {"inputs": [], "partial_results": []}


graph = StateGraph(PipelineState)
graph.add_node("worker", worker)
graph.add_node("aggregate", aggregate)
graph.add_conditional_edges(START, fan_out, ["worker"])
graph.add_edge("worker", "aggregate")
graph.add_edge("aggregate", END)

app = graph.compile()
result = app.invoke({"inputs": ["a", "b", "c"], "partial_results": []})
```

### Example 2: Accumulating mode — event log

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END


class AuditState(TypedDict):
    step: int
    # accumulate=True keeps all events across every step
    events: Annotated[Sequence[str], Topic(str, accumulate=True)]


def step_one(state: AuditState) -> dict:
    return {"step": 1, "events": "step_one_completed"}


def step_two(state: AuditState) -> dict:
    return {"step": 2, "events": "step_two_completed"}


graph = StateGraph(AuditState)
graph.add_node("one", step_one)
graph.add_node("two", step_two)
graph.add_edge(START, "one")
graph.add_edge("one", "two")
graph.add_edge("two", END)

app = graph.compile()
result = app.invoke({"step": 0, "events": []})
# result["events"] == ["step_one_completed", "step_two_completed"]
print(result["events"])
```

### Example 3: Using a Topic for collecting tool call results

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class MultiToolState(TypedDict):
    tool_calls: list[dict]
    # Each parallel tool execution appends its ToolMessage
    tool_results: Annotated[Sequence[ToolMessage], Topic(ToolMessage)]


def dispatch_tools(state: MultiToolState):
    return [Send("run_tool", tc) for tc in state["tool_calls"]]


def run_tool(tool_call: dict) -> dict:
    result = ToolMessage(content=f"result of {tool_call['name']}", tool_call_id=tool_call["id"])
    return {"tool_results": result}


def merge_results(state: MultiToolState) -> dict:
    # All ToolMessages collected for this step
    all_results = list(state["tool_results"])
    return {"tool_calls": [], "tool_results": []}


graph = StateGraph(MultiToolState)
graph.add_node("run_tool", run_tool)
graph.add_node("merge", merge_results)
graph.add_conditional_edges(START, dispatch_tools, ["run_tool"])
graph.add_edge("run_tool", "merge")
graph.add_edge("merge", END)
```

---

## 8 · `EphemeralValue` — transient per-step pass-through

**Module:** `langgraph.channels.ephemeral_value`  
**Import:**
```python
from langgraph.channels.ephemeral_value import EphemeralValue
```

`EphemeralValue` stores the value written in the **current step** and exposes it to downstream nodes in the same step. It is cleared (raises `EmptyChannelError`) before the next step begins, so it never appears in checkpoints as a carry-over value.

The `guard=True` default means the channel raises an error if **more than one** node writes to it in the same step, enforcing single-writer semantics.

### Source signature (1.2.4)

```python
class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(
        self,
        typ: Any,
        guard: bool = True,   # True = error on multiple writes; False = last-write-wins
    ) -> None: ...
```

### Example 1: Single-write intermediate value

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph import StateGraph, START, END


class ParseState(TypedDict):
    raw_text: str
    # Only valid during the step in which it's written; never persisted
    parsed_tokens: Annotated[list[str], EphemeralValue(list)]
    final_output: str


def tokenize(state: ParseState) -> dict:
    return {"parsed_tokens": state["raw_text"].split()}


def process(state: ParseState) -> dict:
    # parsed_tokens is available here because we're in the same step
    tokens = state["parsed_tokens"]
    return {"final_output": f"Found {len(tokens)} tokens"}


graph = StateGraph(ParseState)
graph.add_node("tokenize", tokenize)
graph.add_node("process", process)
graph.add_edge(START, "tokenize")
graph.add_edge("tokenize", "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"raw_text": "hello world foo bar", "parsed_tokens": [], "final_output": ""})
# result["final_output"] == "Found 4 tokens"
# result["parsed_tokens"] is [] (cleared after step)
print(result["final_output"])
```

### Example 2: `guard=False` — last-write-wins when multiple nodes write

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph import StateGraph, START, END


class ScoringState(TypedDict):
    input: str
    # Multiple classifiers may write; the last one wins (non-deterministic if parallel)
    best_label: Annotated[str, EphemeralValue(str, guard=False)]
    final: str


def classifier_fast(state: ScoringState) -> dict:
    return {"best_label": "quick_guess"}


def classifier_accurate(state: ScoringState) -> dict:
    return {"best_label": "accurate_result"}


def decide(state: ScoringState) -> dict:
    return {"final": state["best_label"]}
```

### Example 3: Using `EphemeralValue` for debugging metadata

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph import StateGraph, START, END
import time


class DebugState(TypedDict):
    payload: dict
    # Per-step timing info — not persisted between steps
    step_timing: Annotated[dict, EphemeralValue(dict, guard=False)]
    result: str


def timed_node(state: DebugState) -> dict:
    t0 = time.monotonic()
    # ... do work ...
    elapsed = time.monotonic() - t0
    return {
        "step_timing": {"node": "timed_node", "elapsed_ms": round(elapsed * 1000, 2)},
        "result": "done",
    }
```

---

## 9 · `NamedBarrierValue` — named fan-in synchronisation

**Module:** `langgraph.channels.named_barrier_value`  
**Import:**
```python
from langgraph.channels.named_barrier_value import NamedBarrierValue
```

`NamedBarrierValue` holds a set of expected "tokens". It does not make its value available (read returns the last-seen token) until **every named token has been received at least once**. This is LangGraph's built-in fan-in barrier: you declare which node names must check in, and downstream nodes cannot read the channel until all of them have.

### Source signature (1.2.4)

```python
class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    def __init__(
        self,
        typ: type[Value],
        names: set[Value],   # all tokens that must be received before the barrier opens
    ) -> None: ...
    # get() raises EmptyChannelError until all names have been seen
    # checkpoint() returns the set of already-seen names
```

### Example 1: Waiting for three parallel branches

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.graph import StateGraph, START, END


REQUIRED_CHECKS = {"security", "compliance", "performance"}


class ReviewState(TypedDict):
    document: str
    # Barrier opens only after all three reviewers have checked in
    review_barrier: Annotated[str, NamedBarrierValue(str, REQUIRED_CHECKS)]
    approved: bool


def security_review(state: ReviewState) -> dict:
    # Write the token "security" to check in
    return {"review_barrier": "security"}


def compliance_review(state: ReviewState) -> dict:
    return {"review_barrier": "compliance"}


def performance_review(state: ReviewState) -> dict:
    return {"review_barrier": "performance"}


def final_decision(state: ReviewState) -> dict:
    # Only reached once all three tokens have been written
    return {"approved": True}


graph = StateGraph(ReviewState)
graph.add_node("security", security_review)
graph.add_node("compliance", compliance_review)
graph.add_node("performance", performance_review)
graph.add_node("decide", final_decision)

graph.add_edge(START, "security")
graph.add_edge(START, "compliance")
graph.add_edge(START, "performance")

# All three must complete before "decide" can read review_barrier
graph.add_edge("security", "decide")
graph.add_edge("compliance", "decide")
graph.add_edge("performance", "decide")
graph.add_edge("decide", END)

app = graph.compile()
result = app.invoke({"document": "spec.pdf", "review_barrier": "", "approved": False})
print(result["approved"])  # True
```

### Example 2: Dynamic barrier using `Send` with named tokens

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


VALIDATORS = {"schema", "auth", "rate_limit"}


class APIState(TypedDict):
    request: dict
    validation_barrier: Annotated[str, NamedBarrierValue(str, VALIDATORS)]
    response: dict


def validate_schema(req: dict) -> dict:
    return {"validation_barrier": "schema"}


def validate_auth(req: dict) -> dict:
    return {"validation_barrier": "auth"}


def validate_rate_limit(req: dict) -> dict:
    return {"validation_barrier": "rate_limit"}


def process_request(state: APIState) -> dict:
    # Runs only after schema + auth + rate_limit all check in
    return {"response": {"status": 200, "body": "ok"}}


def dispatch(state: APIState):
    return [
        Send("validate_schema", state["request"]),
        Send("validate_auth", state["request"]),
        Send("validate_rate_limit", state["request"]),
    ]


graph = StateGraph(APIState)
graph.add_node("validate_schema", validate_schema)
graph.add_node("validate_auth", validate_auth)
graph.add_node("validate_rate_limit", validate_rate_limit)
graph.add_node("process", process_request)
graph.add_conditional_edges(START, dispatch, ["validate_schema", "validate_auth", "validate_rate_limit"])
graph.add_edge("validate_schema", "process")
graph.add_edge("validate_auth", "process")
graph.add_edge("validate_rate_limit", "process")
graph.add_edge("process", END)
```

---

## 10 · `AnyValue` — last-writer-wins non-unique channel

**Module:** `langgraph.channels.any_value`  
**Import:**
```python
from langgraph.channels.any_value import AnyValue
```

`AnyValue` is the simplest channel: it stores the **last value** written to it in a step and assumes all concurrent writes are equal. It does not raise on multiple writes — it silently keeps the final one. It is appropriate for true singletons (e.g., a config object set once) or for cases where any worker's output is acceptable.

### Source signature (1.2.4)

```python
class AnyValue(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(self, typ: Any, key: str = "") -> None: ...
    # update(): stores values[-1]; clears if no values written (unlike LastValue)
    # get():    returns self.value; raises EmptyChannelError if never written
```

**Key difference from `LastValue`:** `AnyValue.update([])` clears the channel (sets it to `MISSING`), while `LastValue.update([])` leaves the previous value intact. Use `AnyValue` when the value should only be present within the step it was written.

### Example 1: Configuration singleton

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.any_value import AnyValue
from langgraph.graph import StateGraph, START, END


class ConfiguredState(TypedDict):
    query: str
    # Written once at graph start; any node can read it.
    # Multiple nodes could write the same config without conflict.
    run_config: Annotated[dict, AnyValue(dict)]
    result: str


def initialise(state: ConfiguredState) -> dict:
    return {"run_config": {"model": "opus", "max_tokens": 1024}}


def process(state: ConfiguredState) -> dict:
    cfg = state["run_config"]
    return {"result": f"Using model={cfg['model']}"}


graph = StateGraph(ConfiguredState)
graph.add_node("init", initialise)
graph.add_node("process", process)
graph.add_edge(START, "init")
graph.add_edge("init", "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"query": "hello", "run_config": {}, "result": ""})
print(result["result"])  # "Using model=opus"
```

### Example 2: Race-winner pattern — first valid result wins

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.any_value import AnyValue
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class RaceState(TypedDict):
    tasks: list[str]
    # Multiple workers write; we accept any result — order depends on runtime
    winner_result: Annotated[str, AnyValue(str)]
    done: bool


def run_task(task_id: str) -> dict:
    # Simulate work; each writes to winner_result
    return {"winner_result": f"completed:{task_id}"}


def collect(state: RaceState) -> dict:
    # winner_result is whatever the last-scheduled task wrote
    return {"done": True}


def dispatch(state: RaceState):
    return [Send("run_task", t) for t in state["tasks"]]


graph = StateGraph(RaceState)
graph.add_node("run_task", run_task)
graph.add_node("collect", collect)
graph.add_conditional_edges(START, dispatch, ["run_task"])
graph.add_edge("run_task", "collect")
graph.add_edge("collect", END)
```

### Example 3: Comparing `AnyValue`, `LastValue`, and `Topic`

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.any_value import AnyValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic


class ComparisonState(TypedDict):
    # AnyValue: cleared if nothing written this step; assumes all writers agree
    ephemeral_flag: Annotated[bool, AnyValue(bool)]

    # LastValue: retains previous value if nothing written this step
    persistent_count: Annotated[int, LastValue(int)]

    # Topic: collects ALL values written this step into a list
    all_events: Annotated[Sequence[str], Topic(str)]


# Summary table:
# Channel              | Multiple writes | Not written | Persists across steps
# ---------------------|-----------------|-------------|----------------------
# AnyValue             | Last wins       | Clears      | No (resets per step)
# LastValue            | Last wins       | Keeps prev  | Yes
# BinaryOperatorAgg.   | Reduced         | Unchanged   | Yes
# Topic(accumulate=F)  | Collects all    | Empty list  | No
# Topic(accumulate=T)  | Collects all    | Unchanged   | Yes
```

---

## Quick-reference: channel comparison matrix

| Channel | Multiple writes | Nothing written | Accumulates | `guard` |
|---------|----------------|-----------------|-------------|---------|
| `LastValue` | Last wins | Retains previous | Across steps | No |
| `AnyValue` | Last wins (assumes equal) | Clears to MISSING | No | No |
| `BinaryOperatorAggregate` | Reduced via operator | Unchanged | Across steps | No |
| `Topic(accumulate=False)` | Collects all → list | Empty list | Per-step only | No |
| `Topic(accumulate=True)` | Collects all → list | Unchanged | Across steps | No |
| `EphemeralValue(guard=True)` | Error on >1 write | Clears | No | Yes |
| `EphemeralValue(guard=False)` | Last wins | Clears | No | No |
| `NamedBarrierValue` | Adds to seen-set | No value exposed | Until all seen | — |

[→ Vol. 1–12 class index](/langgraph-guide/python/langgraph_class_deep_dives/)

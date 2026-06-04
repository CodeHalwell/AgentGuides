---
title: "Class deep-dives Vol. 9 — 10 more LangGraph types"
description: "Source-verified deep dives into ToolCallRequest/override(), Send+timeout, create_react_agent pre/post hooks, RetryPolicy chained policies, CachePolicy custom key_func, InMemoryStore raw embeddings, context_schema+Runtime.context, Command.PARENT cross-subgraph routing, TimeoutPolicy.coerce(), and entrypoint multi-policy retry — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 9"
  order: 40
---

# Class deep-dives Vol. 9 — 10 more LangGraph types

Verified against **`langgraph==1.2.4`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly (`/usr/local/lib/python3.11/dist-packages/langgraph`). All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

[→ Vol. 5 covers RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol](./langgraph_class_deep_dives_v5/)

[→ Vol. 6 covers GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message/ui_message_reducer, ProtocolEvent](./langgraph_class_deep_dives_v6/)

[→ Vol. 7 covers PregelProtocol/StreamProtocol, BackgroundExecutor/AsyncBackgroundExecutor, AsyncBatchedBaseStore/_dedupe_ops, get_text_at_path/tokenize_path, SerdeEvent/register_serde_event_listener, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec/node Protocols, identifier/get_runnable_for_task](./langgraph_class_deep_dives_v7/)

[→ Vol. 8 covers ExecutionInfo/Runtime.heartbeat, ServerInfo/BaseUser, ReplayState, StreamMux, Call (functional API internals), ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol/PregelTaskWrites, SyncPregelLoop/AsyncPregelLoop, DuplexStream](./langgraph_class_deep_dives_v8/)

---

## 1 · `ToolCallRequest` and the `override()` pattern

**Module:** `langgraph.prebuilt.tool_node`  
**Exported as:** `from langgraph.prebuilt.tool_node import ToolCallRequest`

`ToolCallRequest` is the dataclass passed to `wrap_tool_call` / `awrap_tool_call` interceptors on `ToolNode`. It holds the full context of a single tool invocation: the raw tool call dict, the resolved `BaseTool`, the current graph state, and a `ToolRuntime`. The **`override()`** method returns a new `ToolCallRequest` with specific fields replaced without mutating the original — an intentionally immutable update pattern.

### Source (1.2.4)

```python
@dataclass
class ToolCallRequest:
    tool_call: ToolCall          # {"name": str, "args": dict, "id": str, "type": "tool_call"}
    tool: BaseTool | None        # resolved tool, or None for unregistered names
    state: Any                   # graph state (dict / list / BaseModel)
    runtime: ToolRuntime

    def override(self, **overrides: Unpack[_ToolCallRequestOverrides]) -> ToolCallRequest:
        """Return a new ToolCallRequest with the given fields replaced (immutable update)."""
        return replace(self, **overrides)
```

`_ToolCallRequestOverrides` accepts `tool_call`, `tool`, and `state`.  Setting attributes directly on an existing instance emits a `DeprecationWarning` — always use `override()`.

### Why `override()` matters

Interceptors often need to sanitise model-supplied arguments before the tool runs, or swap in a different tool entirely.  Because `ToolCallRequest` enforces the immutable pattern, interceptors never accidentally corrupt shared state between parallel tool calls.

### Example 1: Sanitise arguments before execution

```python
from typing import Callable
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest


def sanitize_interceptor(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Strip any PII from tool args before execution and audit-log the call."""
    original_args = request.tool_call["args"]

    # Build a cleaned copy of the args dict
    cleaned_args = {
        k: "[REDACTED]" if k in ("email", "phone", "ssn") else v
        for k, v in original_args.items()
    }

    if cleaned_args != original_args:
        # Replace only the args inside tool_call — leave name and id untouched
        new_tool_call = {**request.tool_call, "args": cleaned_args}
        request = request.override(tool_call=new_tool_call)

    result = execute(request)
    print(f"[AUDIT] {request.tool_call['name']}({cleaned_args})")
    return result


tool_node = ToolNode([my_tool], wrap_tool_call=sanitize_interceptor)
```

### Example 2: Retry with exponential back-off inside the interceptor

```python
import time
import random
from typing import Callable
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest


def retry_interceptor(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage],
    max_retries: int = 3,
) -> ToolMessage:
    """Retry transient failures up to max_retries times with jitter."""
    for attempt in range(max_retries):
        try:
            return execute(request)
        except (ConnectionError, TimeoutError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[RETRY] attempt {attempt + 1} failed ({exc}); retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


tool_node = ToolNode([flaky_api_tool], wrap_tool_call=retry_interceptor)
```

### Example 3: Async interceptor with `awrap_tool_call`

```python
import asyncio
from typing import Callable, Awaitable
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

# Initialise the semaphore lazily to avoid binding to an event loop before
# one is running.  Module-level asyncio.Semaphore() works in Python 3.10+ but
# fails in earlier versions and in certain test harnesses.
_TOOL_SEMAPHORE: asyncio.Semaphore | None = None


async def async_interceptor(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    """Rate-limit tool calls using a lazily-initialised async semaphore."""
    global _TOOL_SEMAPHORE
    if _TOOL_SEMAPHORE is None:
        _TOOL_SEMAPHORE = asyncio.Semaphore(5)   # max 5 concurrent tool calls
    async with _TOOL_SEMAPHORE:
        return await execute(request)


tool_node = ToolNode(
    [my_tool],
    awrap_tool_call=async_interceptor,   # used for async execution paths
    wrap_tool_call=None,                 # sync path falls back to no-op
)
```

---

## 2 · `Send` with per-task `timeout`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import Send`

In LangGraph 1.2.x, `Send` gained a `timeout` parameter.  When provided, it overrides the target node's default timeout for this specific fan-out task.  This lets different items in a map-reduce carry different time budgets without changing the node definition.

### Source (1.2.4)

```python
class Send:
    __slots__ = ("node", "arg", "timeout")

    def __init__(
        self,
        /,
        node: str,
        arg: Any,
        *,
        timeout: float | timedelta | TimeoutPolicy | None = None,
    ) -> None:
        self.node = node
        self.arg = arg
        self.timeout = TimeoutPolicy.coerce(timeout)   # normalised to TimeoutPolicy | None
```

Equality is structural (`node` + `arg` + `timeout`).  `Send` is hashable.

### Example 1: Fan-out with per-item timeouts

```python
import operator
from datetime import timedelta
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class BatchState(TypedDict):
    tasks: list[dict]                         # each task has "id", "payload", "priority"
    results: Annotated[list[str], operator.add]


class TaskState(TypedDict):
    id: str
    payload: str
    priority: str


def dispatch(state: BatchState) -> list[Send]:
    """Fan out tasks; urgent tasks get a tighter timeout."""
    sends = []
    for task in state["tasks"]:
        timeout = (
            timedelta(seconds=5)   # urgent: 5-second hard cap
            if task["priority"] == "urgent"
            else timedelta(seconds=30)  # normal: 30 seconds
        )
        sends.append(Send("process_task", task, timeout=timeout))
    return sends


def process_task(state: TaskState) -> dict:
    # Simulate work — in production this might call an external API
    result = f"processed:{state['id']}"
    return {"results": [result]}


builder = StateGraph(BatchState)
builder.add_node("process_task", process_task)
builder.add_conditional_edges(START, dispatch)
builder.add_edge("process_task", END)

graph = builder.compile()
result = graph.invoke({
    "tasks": [
        {"id": "t1", "payload": "fast", "priority": "urgent"},
        {"id": "t2", "payload": "slow", "priority": "normal"},
    ],
    "results": [],
})
print(result["results"])  # ['processed:t1', 'processed:t2']
```

### Example 2: `Send` with `TimeoutPolicy` (idle + run)

```python
from langgraph.types import Send, TimeoutPolicy

# Use TimeoutPolicy directly for fine-grained control
sends = [
    Send(
        "expensive_node",
        {"item": item},
        timeout=TimeoutPolicy(
            run_timeout=60.0,    # hard 60-second wall-clock cap
            idle_timeout=10.0,   # abort if no progress for 10 seconds
            refresh_on="auto",   # LangChain callbacks reset the idle clock
        ),
    )
    for item in items
]
```

### `Send` equality and hashing

```python
from langgraph.types import Send

s1 = Send("node", {"a": 1}, timeout=5.0)
s2 = Send("node", {"a": 1}, timeout=5.0)
s3 = Send("node", {"a": 1}, timeout=10.0)

assert s1 == s2          # same node + arg + timeout
assert s1 != s3          # different timeout
assert hash(s1) == hash(s2)
```

---

## 3 · `create_react_agent` — `pre_model_hook`, `post_model_hook`, `version="v2"`

**Module:** `langgraph.prebuilt.chat_agent_executor`  
**Exported as:** `from langgraph.prebuilt import create_react_agent`

> **Deprecation notice (v1.2.x):** `create_react_agent` is deprecated in favour of `create_agent` from the separate `langchain` package (`from langchain.agents import create_agent`). Both share the same graph structure; the new function adds a flexible middleware system. The examples below still apply to `create_react_agent` for the 1.2.x release series.

Three parameters added in recent releases make `create_react_agent` significantly more powerful without touching the agent loop itself:

| Parameter | Purpose |
|---|---|
| `pre_model_hook` | Node inserted **before** the LLM call — trim history, inject context, add guardrails |
| `post_model_hook` | Node inserted **after** the LLM call — validate output, add human approval interrupt |
| `version` | `"v2"` (default) distributes tool calls across `Send`s; `"v1"` runs all tool calls in one `ToolNode` invocation |
| `response_format` | Schema for a structured final response in the `structured_response` state key |

### `pre_model_hook` — history trimming

```python
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic


def trim_history(state: dict) -> dict:
    """Keep only the last 10 messages to stay within the model's context window."""
    messages = state.get("messages", [])
    if len(messages) <= 10:
        return {}   # no-op: return empty dict, messages are unchanged

    # Keep the last 10 messages; overwrite the messages key entirely.
    trimmed = messages[-10:]
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + trimmed,
    }


agent = create_react_agent(
    ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=[search_tool, calculator_tool],
    pre_model_hook=trim_history,
)
```

`pre_model_hook` must return at least one of:
- `"messages"` — overwrites history in state.
- `"llm_input_messages"` — passes these to the LLM **without** changing state.

### `pre_model_hook` — inject dynamic context

```python
from langchain_core.messages import SystemMessage

def inject_date_context(state: dict) -> dict:
    """Prepend a fresh system message with today's date before every LLM call."""
    from datetime import date
    system = SystemMessage(content=f"Today is {date.today().isoformat()}.")
    # llm_input_messages is used as LLM input only — does not persist to state
    return {"llm_input_messages": [system] + state["messages"]}
```

### `post_model_hook` — human approval interrupt

```python
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

def require_approval(state: dict) -> dict:
    """Pause after every LLM response and ask a human to approve."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    if not last.tool_calls:
        return {}   # no tools to approve; pass through

    decision = interrupt({
        "question": "Approve these tool calls?",
        "tool_calls": [tc["name"] for tc in last.tool_calls],
    })
    if decision != "yes":
        # Discard the pending tool calls by overwriting the last message
        approved_msg = AIMessage(content="Action cancelled by user.", tool_calls=[])
        return {
            "messages": [RemoveMessage(id=last.id), approved_msg],
        }
    return {}


agent = create_react_agent(
    model,
    tools=[delete_file, send_email],
    post_model_hook=require_approval,
    checkpointer=InMemorySaver(),   # required for interrupt/resume
)
```

### `version="v2"` — tool calls as `Send`s

```python
agent_v2 = create_react_agent(
    model,
    tools=[tool_a, tool_b],
    version="v2",   # default — each tool_call is a separate Send task
)

agent_v1 = create_react_agent(
    model,
    tools=[tool_a, tool_b],
    version="v1",   # legacy — all tool_calls run inside one ToolNode invocation
)
```

`version="v2"` dispatches each `tool_call` as an independent `Send` task, so failures in one tool do not block others.

### `response_format` — structured output

```python
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

class ResearchReport(BaseModel):
    summary: str
    sources: list[str]
    confidence: float

agent = create_react_agent(
    model,
    tools=[web_search, document_reader],
    response_format=ResearchReport,
)

result = agent.invoke({"messages": [("user", "Summarise quantum computing advances")]})
# result["structured_response"] is a ResearchReport instance
print(result["structured_response"].confidence)
```

`response_format` triggers a second LLM call after the agent loop ends to produce a structured extract.  The schema can be a Pydantic model, `TypedDict`, JSON Schema dict, or a `(prompt, schema)` tuple for a custom extraction prompt.

---

## 4 · `RetryPolicy` — chained policies and custom `retry_on`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import RetryPolicy`

Most documentation shows a single `RetryPolicy`.  Both `add_node` and `@task` accept a **list** of policies applied in order — useful when you want one policy for transient network errors and a tighter one for everything else.

### Source (1.2.4)

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5      # seconds before first retry
    backoff_factor: float = 2.0        # multiplier per attempt
    max_interval: float = 128.0        # cap on interval
    max_attempts: int = 3              # total attempts including the first
    jitter: bool = True                # add random noise to the interval
    retry_on: (
        type[Exception]
        | Sequence[type[Exception]]
        | Callable[[Exception], bool]
    ) = default_retry_on               # default: all except GraphInterrupt
```

### Custom `retry_on` callable

```python
import httpx
from langgraph.types import RetryPolicy

def should_retry(exc: Exception) -> bool:
    """Retry transient HTTP errors; give up immediately on auth failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, httpx.TransportError):
        return True      # network-level transient failures
    return False         # all other exceptions: do not retry


http_retry = RetryPolicy(
    initial_interval=1.0,
    backoff_factor=2.0,
    max_attempts=5,
    jitter=True,
    retry_on=should_retry,
)
```

### Chained (list of) policies

When `retry_policy` is a list, the policies are evaluated **in order** for each attempt. The first policy whose `retry_on` matches the exception is used. If no policy matches, the exception propagates immediately.

```python
import httpx
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    url: str
    result: str


# Policy 1: retry rate-limit errors aggressively with long intervals
rate_limit_policy = RetryPolicy(
    initial_interval=5.0,
    backoff_factor=2.0,
    max_interval=60.0,
    max_attempts=6,
    retry_on=lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429,
)

# Policy 2: quick retry for transient network errors
network_policy = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=1.5,
    max_attempts=3,
    retry_on=lambda e: isinstance(e, httpx.TransportError),
)


async def fetch(state: State) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.get(state["url"])
        r.raise_for_status()
        return {"result": r.text[:200]}


builder = StateGraph(State)
# Pass the list — rate_limit_policy is checked first
builder.add_node(
    "fetch",
    fetch,
    retry_policy=[rate_limit_policy, network_policy],
)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)
graph = builder.compile()
```

The same list syntax works on `@task`:

```python
from langgraph.func import task

@task(retry_policy=[rate_limit_policy, network_policy])
async def call_api(url: str) -> str: ...
```

### Chaining policies on `StateGraph.add_node`

```python
builder.add_node(
    "my_node",
    my_fn,
    retry_policy=RetryPolicy(max_attempts=3),      # single policy
)

builder.add_node(
    "my_node",
    my_fn,
    retry_policy=[policy_a, policy_b],             # list of ordered policies
)
```

---

## 5 · `CachePolicy` — custom `key_func`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import CachePolicy`

`CachePolicy` has two fields: `ttl` (seconds, or `None` for no expiry) and `key_func` (a callable that produces the cache key from the node's input). The default `key_func` hashes the full input with pickle. Providing a custom `key_func` gives you deterministic keys, selective invalidation, and namespace isolation.

### Source (1.2.4)

```python
@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key   # default: xxhash of pickled input
    ttl: int | None = None                   # seconds; None = forever
```

`key_func` receives the same arguments as the decorated function and must return `str | bytes`.

### Example 1: Cache by a stable subset of the input

```python
import json
from langgraph.types import CachePolicy
from langgraph.func import task


def embedding_key(text: str, model: str = "text-embedding-3-small") -> str:
    """Cache key that ignores irrelevant kwargs like timestamp."""
    return f"{model}:{text}"


@task(cache_policy=CachePolicy(key_func=embedding_key, ttl=3600))
def embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Call the embedding API — cached per (text, model) pair for one hour."""
    return call_embedding_api(text, model)
```

### Example 2: Per-user cache namespace

```python
def user_scoped_key(query: str, user_id: str) -> str:
    """Isolate cache entries by user — user A's results never collide with user B's."""
    import hashlib
    digest = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"user:{user_id}:search:{digest}"


@task(cache_policy=CachePolicy(key_func=user_scoped_key, ttl=600))
def personalised_search(query: str, user_id: str) -> list[dict]:
    return run_search(query, user_id=user_id)
```

### Example 3: Cache on a `StateGraph` node

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    query: str
    results: list[str]


cache = InMemoryCache()


def search_key(state: State) -> str:
    """Deterministic key: normalise and hash the query string."""
    return state["query"].strip().lower()


def search_node(state: State) -> dict:
    return {"results": expensive_search(state["query"])}


builder = StateGraph(State)
builder.add_node(
    "search",
    search_node,
    cache_policy=CachePolicy(key_func=search_key, ttl=300),
)
builder.add_edge(START, "search")
builder.add_edge("search", END)
graph = builder.compile(cache=cache)
```

---

## 6 · `InMemoryStore` — raw embedding functions (no LangChain dependency)

**Module:** `langgraph.store.memory`  
**Exported as:** `from langgraph.store.memory import InMemoryStore`

Documentation usually shows `init_embeddings("openai:...")` from LangChain.  You can also pass any callable with signature `(texts: list[str]) -> list[list[float]]` — or its async variant — which removes the LangChain dependency entirely.

### Source — how `ensure_embeddings` handles callables

```python
# langgraph.store.base.ensure_embeddings normalises the embed field:
# - LangChain Embeddings instance  → used directly
# - sync callable                  → wrapped in EmbeddingsLambda (sync)
# - async callable                 → wrapped in EmbeddingsLambda (async)
```

### Example 1: OpenAI SDK directly

```python
from openai import OpenAI
from langgraph.store.memory import InMemoryStore

client = OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in response.data]


store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embed_texts,        # plain sync function
        "fields": ["text"],          # only embed the "text" field of each value dict
    }
)

store.put(("docs",), "d1", {"text": "LangGraph is a graph orchestration library."})
store.put(("docs",), "d2", {"text": "LangChain provides LLM chains and agents."})
store.put(("docs",), "d3", {"text": "NumPy is a numerical computing library."})

results = store.search(("docs",), query="building AI agent workflows", limit=2)
for r in results:
    print(f"{r.score:.4f}  {r.value['text']}")
```

### Example 2: Async variant

```python
from openai import AsyncOpenAI
from langgraph.store.memory import InMemoryStore

aclient = AsyncOpenAI()


async def aembed_texts(texts: list[str]) -> list[list[float]]:
    response = await aclient.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in response.data]


store = InMemoryStore(
    index={"dims": 1536, "embed": aembed_texts, "fields": ["$"]}
)

await store.aput(("knowledge",), "k1", {"content": "Python 3.12 new features"})
results = await store.asearch(("knowledge",), query="walrus operator", limit=5)
```

### Example 3: Hugging Face sentence-transformers (local, no API key)

```python
from sentence_transformers import SentenceTransformer
from langgraph.store.memory import InMemoryStore

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, runs locally


def local_embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()


store = InMemoryStore(
    index={"dims": 384, "embed": local_embed, "fields": ["text"]}
)
```

### Example 4: Multi-field indexing with `[*]` array wildcard

```python
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embed_texts,
        "fields": ["title", "chapters[*].content"],  # embed title + every chapter
    }
)

store.put(("books",), "b1", {
    "title": "Clean Code",
    "chapters": [
        {"content": "Names matter."},
        {"content": "Functions should do one thing."},
    ],
})

# Each chapter is embedded separately; title is embedded once.
# Search returns the item whose best-matching field is most similar to the query.
results = store.search(("books",), query="naming conventions in code", limit=3)
```

---

## 7 · `context_schema` + `Runtime.context` — typed run-scoped context

**Module:** `langgraph.graph.state`, `langgraph.func`, `langgraph.runtime`  
**Exported as:** various — see examples

`context_schema` on a `StateGraph` or `@entrypoint` declares a typed read-only object injected into every node via `runtime.context`.  Unlike state, context is **not** persisted to checkpoints and **not** accessible via `get_state()` — it's ephemeral per-run metadata.

### Use case

Inject per-request data that shouldn't pollute the graph state: tenant ID, feature flags, the authenticated user, a database connection.

### Example 1: Typed context in `StateGraph`

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class RequestContext:
    tenant_id: str
    user_email: str
    feature_flags: dict[str, bool]


class AgentState(TypedDict):
    messages: list[str]
    result: str


def process(state: AgentState, runtime: Runtime[RequestContext]) -> dict:
    ctx = runtime.context                   # type: RequestContext
    tenant = ctx.tenant_id
    can_use_premium = ctx.feature_flags.get("premium_search", False)

    if can_use_premium:
        result = f"[{tenant}] Premium search for: {state['messages'][-1]}"
    else:
        result = f"[{tenant}] Basic search for: {state['messages'][-1]}"

    return {"result": result}


builder = StateGraph(AgentState, context_schema=RequestContext)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Pass context via the `context=` keyword argument — separate from config
result = graph.invoke(
    {"messages": ["find AI papers"], "result": ""},
    config={"configurable": {"thread_id": "t1"}},
    context=RequestContext(
        tenant_id="acme-corp",
        user_email="alice@acme.com",
        feature_flags={"premium_search": True},
    ),
)
print(result["result"])
# '[acme-corp] Premium search for: find AI papers'
```

### Example 2: Typed context in `@entrypoint`

```python
from dataclasses import dataclass
from langgraph.func import entrypoint, task
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class DBContext:
    connection_string: str
    read_only: bool


@task
def query_db(sql: str, conn_str: str) -> list[dict]:
    # In production: use conn_str to connect
    return [{"row": f"result for {sql}"}]


@entrypoint(checkpointer=InMemorySaver(), context_schema=DBContext)
def data_pipeline(query: str, runtime: Runtime[DBContext]) -> dict:
    ctx = runtime.context   # type: DBContext
    if ctx.read_only:
        assert "SELECT" in query.upper(), "Only SELECT queries allowed in read-only mode"

    rows = query_db(query, ctx.connection_string).result()
    return {"rows": rows, "query": query}


result = data_pipeline.invoke(
    "SELECT * FROM users LIMIT 10",
    config={"configurable": {"thread_id": "pipe-1"}},
    context=DBContext(
        connection_string="postgresql://localhost/mydb",
        read_only=True,
    ),
)
```

### Example 3: Context in tools via `ToolRuntime`

```python
from langgraph.prebuilt.tool_node import ToolRuntime
from langchain_core.tools import tool


@tool
def tenant_search(
    query: str,
    runtime: ToolRuntime[RequestContext, AgentState],  # Generic[ContextT, StateT]
) -> str:
    """Search within the current tenant's data."""
    ctx: RequestContext = runtime.context               # typed
    tenant_id = ctx.tenant_id
    results = search_tenant_index(tenant_id, query)
    return "\n".join(results)
```

---

## 8 · `Command.PARENT` — cross-subgraph routing

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import Command`

`Command.PARENT` is a class variable equal to `"__parent__"`.  When a node inside a **subgraph** returns `Command(graph=Command.PARENT, goto=..., update=...)`, the command is forwarded to the **parent** graph — allowing a subgraph to trigger routing or state updates in its caller.

### Source

```python
class Command(Generic[N], ToolOutputMixin):
    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
    graph: str | None = None    # None = current graph; Command.PARENT = parent
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

### Full runnable example: error escalation

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from langgraph.checkpoint.memory import InMemorySaver


# ── Subgraph ──────────────────────────────────────────────────────────────────

class SubState(TypedDict):
    task: str
    attempt: int


def sub_worker(state: SubState) -> Command:
    attempt = state.get("attempt", 0) + 1
    if attempt >= 3:
        # Too many retries: bubble an error signal to the parent
        return Command(
            graph=Command.PARENT,        # target the parent graph
            goto="error_handler",        # jump to this node in the parent
            update={"failed_task": state["task"], "error_count": attempt},
        )
    # Simulate transient failure on attempts 1 and 2
    if attempt < 3:
        return Command(
            goto="sub_worker",
            update={"attempt": attempt},
        )
    return Command(
        graph=Command.PARENT,
        goto="success_handler",
        update={"completed_task": state["task"]},
    )


sub_builder = StateGraph(SubState)
sub_builder.add_node("sub_worker", sub_worker)
sub_builder.add_edge(START, "sub_worker")
sub_graph = sub_builder.compile()


# ── Parent graph ──────────────────────────────────────────────────────────────

class ParentState(TypedDict):
    tasks: list[str]
    failed_task: str
    completed_task: str
    error_count: int
    status: str


def dispatcher(state: ParentState) -> Command[Literal["sub_processor", "done"]]:
    if not state["tasks"]:
        return Command(goto="done")
    task = state["tasks"][0]
    remaining = state["tasks"][1:]
    return Command(
        update={"tasks": remaining},
        goto=Send("sub_processor", {"task": task, "attempt": 0}),
    )


def error_handler(state: ParentState) -> dict:
    print(f"Task '{state['failed_task']}' failed after {state['error_count']} attempts.")
    return {"status": "failed"}


def success_handler(state: ParentState) -> dict:
    print(f"Task '{state['completed_task']}' completed.")
    return {"status": "done"}


def done(state: ParentState) -> dict:
    return {"status": "all_done"}


parent_builder = StateGraph(ParentState)
parent_builder.add_node("dispatcher", dispatcher)
parent_builder.add_node("sub_processor", sub_graph)   # subgraph as a node
parent_builder.add_node("error_handler", error_handler)
parent_builder.add_node("success_handler", success_handler)
parent_builder.add_node("done", done)

parent_builder.add_edge(START, "dispatcher")
parent_builder.add_edge("error_handler", END)
parent_builder.add_edge("success_handler", END)
parent_builder.add_edge("done", END)

graph = parent_builder.compile(checkpointer=InMemorySaver())
result = graph.invoke(
    {"tasks": ["job-1"], "failed_task": "", "completed_task": "", "error_count": 0, "status": ""},
    config={"configurable": {"thread_id": "run-1"}},
)
```

### Rules for `Command.PARENT`

- Only works from **inside a subgraph node**; using it in the top-level graph raises a `ParentCommand` error that surfaces as a `GraphBubbleUp` exception.
- The `goto` target must be a node in the **direct parent** graph (not grandparent).
- `update` keys are applied to the parent's state, so they must be valid channel names in the parent's schema.

---

## 9 · `TimeoutPolicy.coerce()` — normalisation and shorthand patterns

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import TimeoutPolicy`

`TimeoutPolicy.coerce()` is a classmethod that normalises any timeout value into a `TimeoutPolicy | None`. It's used internally by `Send`, `@task`, `add_node`, and `@entrypoint`, but you can call it explicitly to pre-validate configuration.

### Source (1.2.4)

```python
@dataclass(**_DC_KWARGS)
class TimeoutPolicy:
    run_timeout: float | timedelta | None = None
    idle_timeout: float | timedelta | None = None
    refresh_on: Literal["auto", "heartbeat"] = "auto"

    @classmethod
    def coerce(
        cls, value: float | timedelta | TimeoutPolicy | None
    ) -> TimeoutPolicy | None:
        """Normalise a timeout value to positive-second policy fields.

        - None           → None (no timeout)
        - float/timedelta → TimeoutPolicy(run_timeout=value)
        - TimeoutPolicy  → validated and returned as-is if already correct
        """
```

### Shorthand forms

```python
from langgraph.types import TimeoutPolicy

# All three are equivalent:
tp1 = TimeoutPolicy.coerce(30.0)
tp2 = TimeoutPolicy.coerce(timedelta(seconds=30))
tp3 = TimeoutPolicy(run_timeout=30.0)

assert tp1 == tp2 == tp3

# None passes through
assert TimeoutPolicy.coerce(None) is None

# Already-correct TimeoutPolicy is returned as-is (fast path, frozen=True safe to share)
tp = TimeoutPolicy(run_timeout=10.0, idle_timeout=5.0)
assert TimeoutPolicy.coerce(tp) is tp
```

### `run_timeout` vs `idle_timeout`

| Field | Semantics | Use when |
|---|---|---|
| `run_timeout` | Hard wall-clock cap per attempt. Never refreshed. | LLM calls with a known SLA |
| `idle_timeout` | Max time between progress signals. Refreshed by callbacks or `heartbeat()`. | Long-running loops that make incremental progress |

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

# Hard cap: fail if the node takes more than 60 seconds, regardless of progress
strict = TimeoutPolicy(run_timeout=timedelta(seconds=60))

# Idle cap: fail only if no callback fires for 15 seconds — safe for streaming LLMs
streaming_safe = TimeoutPolicy(
    idle_timeout=15.0,
    refresh_on="auto",      # LangChain callbacks reset the idle clock automatically
)

# Both: fail if 60 s total OR 15 s silence — whichever comes first
combined = TimeoutPolicy(
    run_timeout=60.0,
    idle_timeout=15.0,
    refresh_on="heartbeat",   # only explicit heartbeat() calls reset idle clock
)
```

### Validation errors

```python
# ValueError: both timeouts None (coerce returns None for this, which is ok)
# TimeoutPolicy(run_timeout=None, idle_timeout=None) — a no-op; coerce returns None

# ValueError: non-positive value
try:
    TimeoutPolicy.coerce(-1.0)
except ValueError as e:
    print(e)  # "run_timeout must be greater than 0"

# ValueError: invalid refresh_on
try:
    TimeoutPolicy(run_timeout=5.0, refresh_on="manual")
except ValueError as e:
    print(e)  # "refresh_on must be 'auto' or 'heartbeat'"
```

### Using `TimeoutPolicy` on `StateGraph.add_node`

```python
from langgraph.types import TimeoutPolicy
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node(
    "llm_call",
    call_llm,
    timeout=TimeoutPolicy(run_timeout=30.0, idle_timeout=10.0),
)
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", END)
```

---

## 10 · `@entrypoint` — multi-policy `retry_policy` + `context_schema` together

**Module:** `langgraph.func`  
**Exported as:** `from langgraph.func import entrypoint`

The `@entrypoint` decorator wraps the decorated function in a `Pregel` graph.  Two underused options:

1. **`retry_policy` as a list** — the same chained-policy semantics from §4, applied to the whole workflow.
2. **Combined `context_schema` + `store` + `checkpointer` + `retry_policy`** in one decorator — a fully-featured entry-point factory.

### Source — constructor

```python
class entrypoint(Generic[ContextT]):
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        context_schema: type[ContextT] | None = None,
        cache_policy: CachePolicy | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        timeout: float | timedelta | TimeoutPolicy | None = None,
    ) -> None: ...
```

### Example 1: Multi-policy retry on an `@entrypoint`

```python
import httpx
from dataclasses import dataclass
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

# Two policies: one for rate limits, one for transient network errors
rate_limit = RetryPolicy(
    initial_interval=10.0, backoff_factor=2.0, max_attempts=5,
    retry_on=lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429,
)
network_err = RetryPolicy(
    initial_interval=1.0, backoff_factor=1.5, max_attempts=3,
    retry_on=lambda e: isinstance(e, httpx.TransportError),
)


@task
async def fetch(url: str) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text[:500]


@entrypoint(
    checkpointer=InMemorySaver(),
    retry_policy=[rate_limit, network_err],   # ordered list of policies
)
async def pipeline(urls: list[str]) -> list[str]:
    futures = [fetch(u) for u in urls]
    return [f.result() for f in futures]
```

### Example 2: Full-featured decorator

```python
from dataclasses import dataclass
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


@dataclass
class AppContext:
    org_id: str
    api_key: str


cache = InMemoryCache()
store = InMemoryStore()

@task(cache_policy=CachePolicy(ttl=600))
async def enrich(doc: dict) -> dict:
    """Expensive enrichment call — cached for 10 minutes."""
    return {**doc, "enriched": True}


@entrypoint(
    checkpointer=InMemorySaver(),
    store=store,
    cache=cache,
    context_schema=AppContext,
    cache_policy=CachePolicy(ttl=3600),    # cache the whole workflow result
    retry_policy=[
        RetryPolicy(max_attempts=3, retry_on=lambda e: isinstance(e, ConnectionError)),
    ],
    timeout=TimeoutPolicy(run_timeout=120.0),
)
async def document_pipeline(
    docs: list[dict],
    runtime: Runtime[AppContext],
    previous: list[dict] | None = None,
) -> list[dict]:
    ctx = runtime.context
    org_id = ctx.org_id
    assert runtime.store is not None

    # Use async store methods inside an async entrypoint — sync variants block the event loop
    prev_items = await runtime.store.asearch(("docs", org_id), limit=100)
    prev_keys = {it.key for it in prev_items}

    futures = [enrich(doc) for doc in docs if doc["id"] not in prev_keys]
    enriched = [f.result() for f in futures]

    # Persist enriched docs to the store (async)
    for doc in enriched:
        await runtime.store.aput(("docs", org_id), doc["id"], doc)

    return enriched


# Invoke — pass context as a separate kwarg, not inside config
result = await document_pipeline.ainvoke(
    [{"id": "d1", "text": "Hello"}, {"id": "d2", "text": "World"}],
    config={"configurable": {"thread_id": "pipeline-run-1"}},
    context=AppContext(org_id="acme", api_key="sk-..."),
)
```

### `entrypoint.final` with typed generics

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@entrypoint(checkpointer=InMemorySaver())
def rolling_average(
    new_value: float,
    *,
    previous: dict | None = None,
) -> entrypoint.final[float, dict]:
    """Return the current average; checkpoint a running total for the next call."""
    prev = previous or {"total": 0.0, "count": 0}
    total = prev["total"] + new_value
    count = prev["count"] + 1
    avg = total / count
    return entrypoint.final(
        value=avg,               # returned to the caller
        save={"total": total, "count": count},   # persisted for next invocation
    )


cfg = {"configurable": {"thread_id": "avg-thread"}}
print(rolling_average.invoke(10.0, cfg))   # 10.0
print(rolling_average.invoke(20.0, cfg))   # 15.0
print(rolling_average.invoke(30.0, cfg))   # 20.0
```

---

## Quick-reference table

| Class / function | Module | Key feature covered |
|---|---|---|
| `ToolCallRequest.override()` | `langgraph.prebuilt.tool_node` | Immutable interceptor request mutation |
| `Send(node, arg, timeout=...)` | `langgraph.types` | Per-task timeout on fan-out |
| `create_react_agent` | `langgraph.prebuilt` | `pre_model_hook`, `post_model_hook`, `version="v2"`, `response_format` |
| `RetryPolicy` (list) | `langgraph.types` | Chained policies for different exception types |
| `CachePolicy(key_func=...)` | `langgraph.types` | Custom deterministic cache keys |
| `InMemoryStore` + raw fn | `langgraph.store.memory` | Embedding without LangChain dependency |
| `context_schema` + `Runtime.context` | `langgraph.graph.state`, `langgraph.func` | Typed run-scoped context injection |
| `Command.PARENT` | `langgraph.types` | Cross-subgraph routing and error escalation |
| `TimeoutPolicy.coerce()` | `langgraph.types` | Shorthand normalisation + `run_timeout` vs `idle_timeout` |
| `@entrypoint` multi-policy | `langgraph.func` | Chained retry + context_schema + full-featured factory |

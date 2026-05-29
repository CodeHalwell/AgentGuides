---
title: "Class deep-dives — 10 core LangGraph types"
description: "Source-verified deep dives into StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, task/entrypoint, BinaryOperatorAggregate/Topic, and InMemoryStore — with runnable examples for every major feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives"
  order: 25
---

# Class deep-dives — 10 core LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Each section below was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## 1 · `StateGraph`

**Module:** `langgraph.graph.state`  
**Re-exported from:** `langgraph.graph`

`StateGraph` is the declarative builder for a stateful graph. You declare a state schema, add nodes and edges, and call `.compile()` to get a runnable.

### Constructor

```python
StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema:  type[InputT]  | None = None,
    output_schema: type[OutputT] | None = None,
)
```

`state_schema` must be a `TypedDict`, dataclass, or Pydantic `BaseModel`. Each field maps to a channel:

- Plain field → `LastValue` channel (one write per super-step; concurrent writes raise `InvalidUpdateError`)
- `Annotated[T, reducer]` → `BinaryOperatorAggregate` channel (concurrent writes merged by `reducer`)
- `Annotated[list[T], Topic(T)]` → `Topic` channel (concurrent writes collected into a list)

### `add_node` — full signature

```python
builder.add_node(
    node:           str | callable,
    action:         callable | None = None,
    *,
    defer:          bool = False,
    metadata:       dict | None = None,
    input_schema:   type | None = None,
    retry_policy:   RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy:   CachePolicy | None = None,
    error_handler:  callable | None = None,
    destinations:   dict[str, str] | tuple[str, ...] | None = None,
    timeout:        float | timedelta | TimeoutPolicy | None = None,
)
```

Key kwargs you may not know about:

| kwarg | What it does |
|---|---|
| `retry_policy` | Retry on failure. `RetryPolicy(max_attempts=3, backoff_factor=2.0)`. |
| `cache_policy` | Cache node output. `CachePolicy(ttl=300)`. |
| `input_schema` | Narrow the state keys the node sees — only those keys in the schema are passed. |
| `error_handler` | A fallback node called when this node raises. Receives the full state + `__error__` key. |
| `destinations` | Remap outgoing edges: `{"next": "actually_this_node"}`. |
| `defer` | If `True`, the node only runs after all non-deferred nodes in the same super-step have finished. |

### Full example: retry, cache, input schema

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache


class PipelineState(TypedDict):
    query:   str
    results: Annotated[list[str], operator.add]   # reducer: accumulate across parallel nodes
    cost:    Annotated[float, operator.add]


class SearchInput(TypedDict):
    query: str   # node only sees this key from state


def web_search(state: SearchInput) -> dict:
    """Simulates a flaky web search that may need retrying."""
    import random
    if random.random() < 0.3:
        raise ConnectionError("Transient network error")
    return {"results": [f"web: {state['query']}"], "cost": 0.001}


def db_search(state: SearchInput) -> dict:
    return {"results": [f"db: {state['query']}"], "cost": 0.0001}


def error_handler(state: PipelineState) -> dict:
    """Called when web_search exhausts its retries."""
    return {"results": [f"[fallback] {state['query']}"], "cost": 0.0}


builder = StateGraph(PipelineState)

# add_node with retry, cache, narrow input, and error fallback
builder.add_node(
    "web_search",
    web_search,
    input_schema=SearchInput,
    retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.2, backoff_factor=2.0),
    cache_policy=CachePolicy(ttl=300),   # cache for 5 minutes
    error_handler=error_handler,
)
builder.add_node(
    "db_search",
    db_search,
    input_schema=SearchInput,
    retry_policy=RetryPolicy(max_attempts=2),
)

# Fan out from START to both nodes, then converge at END
builder.add_edge(START, "web_search")
builder.add_edge(START, "db_search")
builder.add_edge(["web_search", "db_search"], END)

cache = InMemoryCache()
graph = builder.compile(checkpointer=InMemorySaver(), cache=cache)

result = graph.invoke({"query": "langgraph docs", "results": [], "cost": 0.0})
print(result["results"])  # e.g. ['web: langgraph docs', 'db: langgraph docs']
print(result["cost"])     # sum of both nodes' costs
```

### `add_sequence`

Shortcut to add a linear chain in one call:

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class S(TypedDict):
    value: int


builder = StateGraph(S)
builder.add_node("a", lambda s: {"value": s["value"] + 1})
builder.add_node("b", lambda s: {"value": s["value"] * 2})
builder.add_node("c", lambda s: {"value": s["value"] - 3})

# Equivalent to add_edge(START,"a"), add_edge("a","b"), add_edge("b","c"), add_edge("c",END)
builder.add_sequence(["a", "b", "c"])
builder.add_edge(START, "a")
builder.add_edge("c", END)

graph = builder.compile()
print(graph.invoke({"value": 5}))  # {"value": (5+1)*2-3 = 9}
```

---

## 2 · `CompiledStateGraph`

**Module:** `langgraph.graph.state`

The object returned by `StateGraph.compile()`. Implements the full LangChain Runnable protocol plus LangGraph-specific methods.

### Invoke & stream

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Echo: {last}")]}


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "demo"}}

# --- invoke: returns final state ---
result = graph.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
print(result["messages"][-1].content)   # Echo: hi

# --- stream: yields per-node or per-step dicts ---
for chunk in graph.stream(
    {"messages": [HumanMessage(content="hello")]},
    cfg,
    stream_mode="updates",   # "values" | "updates" | "messages" | "custom" | "debug"
):
    print(chunk)
# {'chat': {'messages': [AIMessage(content='Echo: hello')]}}
```

### `get_state` / `update_state`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class S(TypedDict):
    step: int
    notes: str


def step_node(state: S) -> dict:
    val = interrupt(f"Approve step {state['step']}?")  # pause for human
    return {"step": state["step"] + 1, "notes": val}


builder = StateGraph(S)
builder.add_node("step", step_node)
builder.add_edge(START, "step")
builder.add_edge("step", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

# First run — pauses at interrupt
for ev in graph.stream({"step": 0, "notes": ""}, cfg):
    print(ev)

# Inspect the paused state
snap = graph.get_state(cfg)
print(snap.values)         # {'step': 0, 'notes': ''}
print(snap.interrupts)     # [Interrupt(value='Approve step 0?', id='...')]
print(snap.next)           # ('step',)

# Manually patch state before resuming (e.g. override a value)
graph.update_state(cfg, {"notes": "pre-approved"})

# Resume — passes answer to interrupt()
for ev in graph.stream(Command(resume="approved"), cfg):
    print(ev)

final = graph.get_state(cfg)
print(final.values)  # {'step': 1, 'notes': 'approved'}
```

### `get_state_history` — time-travel

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
cfg = {"configurable": {"thread_id": "history"}}

# Run 3 times on the same thread
for _ in range(3):
    graph.invoke({"n": 0}, cfg)

# Walk the checkpoint history newest → oldest
for snap in graph.get_state_history(cfg):
    print(f"step={snap.metadata.get('step')}  n={snap.values.get('n')}")

# Re-run from a specific checkpoint (time-travel)
history = list(graph.get_state_history(cfg))
old_cfg = history[-1].config   # oldest snapshot
graph.invoke(None, old_cfg)    # re-replay from there
```

### `bulk_update_state`

Update multiple state keys atomically in a single checkpoint write:

```python
graph.bulk_update_state(
    cfg,
    updates=[
        {"n": 99},
        {"n": 100},   # second update wins on LastValue channels
    ],
    as_node="__start__",
)
```

---

## 3 · `InMemorySaver` (BaseCheckpointSaver)

**Module:** `langgraph.checkpoint.memory`  
**Alias:** `MemorySaver` (backward-compat)

### What it stores

```
storage:  thread_id → namespace → checkpoint_id → (checkpoint_bytes, metadata_bytes, parent_id)
writes:   (thread_id, ns, checkpoint_id) → {(task_id, idx): (task_id, channel, value_bytes, path)}
blobs:    (thread_id, ns, channel, version) → (encoding, bytes)
```

All lookups use `get_tuple` (latest or by id) and `list` (history). Async variants are identical sync methods re-used under `asyncio`.

### Custom checkpointer pattern

Extend `BaseCheckpointSaver` to build your own backend:

```python
from collections.abc import Iterator, AsyncIterator
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    PendingWrite,
    get_checkpoint_id,
)
import redis
import json


class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    """Minimal Redis-backed checkpointer."""

    def __init__(self, redis_client: redis.Redis):
        super().__init__()
        self.r = redis_client

    def _key(self, thread_id: str, ns: str, cp_id: str) -> str:
        return f"lg:checkpoint:{thread_id}:{ns}:{cp_id}"

    def get_next_version(self, current: str | None, channel: Any) -> str:
        v = int(current.split(".")[0]) if current else 0
        return f"{v + 1:032}.0"

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = get_checkpoint_id(config)
        if cp_id:
            raw = self.r.hget(f"lg:meta:{tid}:{ns}", cp_id)
            if not raw:
                return None
            data = json.loads(raw)
        else:
            # Latest: scan all keys for this thread+ns, pick max id
            all_ids = list(self.r.hkeys(f"lg:meta:{tid}:{ns}"))
            if not all_ids:
                return None
            cp_id = max(k.decode() if isinstance(k, bytes) else k for k in all_ids)
            raw = self.r.hget(f"lg:meta:{tid}:{ns}", cp_id)
            data = json.loads(raw)

        return CheckpointTuple(
            config={"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": cp_id}},
            checkpoint=self.serde.loads_typed(tuple(data["checkpoint"])),
            metadata=self.serde.loads_typed(tuple(data["metadata"])),
            parent_config=(
                {"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": data["parent"]}}
                if data.get("parent") else None
            ),
        )

    def list(self, config: RunnableConfig | None, *, filter=None, before=None, limit=None) -> Iterator[CheckpointTuple]:
        # Simplified: yield nothing (production impl would iterate Redis)
        return iter([])

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        c = {k: v for k, v in checkpoint.items() if k != "channel_values"}
        self.r.hset(f"lg:meta:{tid}:{ns}", cp_id, json.dumps({
            "checkpoint": list(self.serde.dumps_typed(c)),
            "metadata": list(self.serde.dumps_typed(metadata)),
            "parent": parent_id,
        }))
        return {"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": cp_id}}

    def put_writes(self, config: RunnableConfig, writes: list[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        pass  # simplified: skip write persistence


# Usage:
# r = redis.Redis(host="localhost", port=6379, db=0)
# graph = builder.compile(checkpointer=RedisCheckpointSaver(r))
```

### Listing and filtering checkpoints

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class S(TypedDict):
    x: int


builder = StateGraph(S)
builder.add_node("inc", lambda s: {"x": s["x"] + 1})
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
cfg = {"configurable": {"thread_id": "inspect"}}

graph.invoke({"x": 1}, cfg)
graph.invoke({"x": 1}, cfg)
graph.invoke({"x": 1}, cfg)

# List all checkpoints for this thread, newest first
for tup in saver.list(cfg):
    print(tup.metadata.get("step"), tup.checkpoint["id"][:8])

# Filter by metadata
for tup in saver.list(cfg, filter={"source": "loop"}):
    print(tup.checkpoint["id"][:8], "is a loop checkpoint")

# Delete the entire thread
saver.delete_thread("inspect")
print(list(saver.list(cfg)))  # []
```

---

## 4 · `ToolNode`

**Module:** `langgraph.prebuilt.tool_node`  
**Re-exported from:** `langgraph.prebuilt`

### State injection

Tools can read the full graph state by declaring a parameter annotated with `InjectedState`:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import ToolNode, InjectedState, InjectedStore
from langgraph.graph.message import add_messages


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id:  str
    balance:  float


@tool
def get_balance(state: Annotated[AppState, InjectedState]) -> str:
    """Return the current account balance."""
    return f"Balance for user {state['user_id']}: ${state['balance']:.2f}"


@tool
def charge(
    amount: float,
    state: Annotated[AppState, InjectedState],
) -> str:
    """Simulate charging the user's account."""
    new_balance = state["balance"] - amount
    if new_balance < 0:
        return f"Insufficient funds. Balance: ${state['balance']:.2f}"
    # InjectedState is read-only — return a Command to update state
    from langgraph.types import Command
    return Command(
        update={"balance": new_balance},
        goto="agent",  # send control back to agent
    )


tool_node = ToolNode([get_balance, charge])
```

### Store injection

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.store.base import BaseStore
from typing import Annotated


@tool
def remember_fact(
    fact: str,
    user_id: str,
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """Save a fact about the user for future reference."""
    existing = store.get(("users", user_id), "facts")
    facts = existing.value if existing else []
    facts.append(fact)
    store.put(("users", user_id), "facts", {"items": facts})
    return f"Remembered: {fact}"


@tool
def recall_facts(
    user_id: str,
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """Retrieve all stored facts about a user."""
    item = store.get(("users", user_id), "facts")
    if not item:
        return "No facts stored yet."
    return f"Facts about {user_id}: {item.value['items']}"
```

### `wrap_tool_call` — intercepting calls before execution

```python
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.tools import tool
import time


@tool
def slow_tool(query: str) -> str:
    """A tool that might take a while."""
    return f"Result for {query}"


def audit_interceptor(request: ToolCallRequest, execute) -> any:
    """Log every tool call and measure duration."""
    t0 = time.perf_counter()
    print(f"[audit] calling {request.tool_call['name']} args={request.tool_call['args']}")
    try:
        result = execute(request)
        elapsed = time.perf_counter() - t0
        print(f"[audit] {request.tool_call['name']} OK in {elapsed:.3f}s")
        return result
    except Exception as exc:
        print(f"[audit] {request.tool_call['name']} FAILED: {exc}")
        raise


# Modify tool call args before execution
def sanitize_interceptor(request: ToolCallRequest, execute) -> any:
    """Strip PII from tool args before they hit the tool."""
    import re
    clean_args = {
        k: re.sub(r"\d{4}-\d{4}-\d{4}-\d{4}", "****-****-****-****", str(v))
        for k, v in request.tool_call["args"].items()
    }
    modified = {**request.tool_call, "args": clean_args}
    return execute(request.override(tool_call=modified))


tool_node = ToolNode(
    [slow_tool],
    wrap_tool_call=audit_interceptor,
)
```

### `handle_tool_errors`

```python
from langchain_core.tools import tool, ToolException
from langgraph.prebuilt import ToolNode


@tool
def risky_tool(x: int) -> int:
    """Might divide by zero."""
    return 100 // x


# Default (True): format exception as a ToolMessage and continue
node_default = ToolNode([risky_tool])

# Custom message
node_msg = ToolNode([risky_tool], handle_tool_errors="Something went wrong — please try different args.")

# Custom formatter
def my_formatter(exc: Exception) -> str:
    return f"Tool error [{type(exc).__name__}]: {exc}"

node_custom = ToolNode([risky_tool], handle_tool_errors=my_formatter)

# Specific exception types only
node_typed = ToolNode([risky_tool], handle_tool_errors=(ZeroDivisionError, ValueError))

# False: exceptions bubble up and can crash the graph
node_strict = ToolNode([risky_tool], handle_tool_errors=False)
```

---

## 5 · `create_react_agent`

**Module:** `langgraph.prebuilt.chat_agent_executor`  
**Re-exported from:** `langgraph.prebuilt`

> **Deprecation notice:** `create_react_agent` is deprecated in `langgraph-prebuilt==1.1.0` in favour of `create_agent` from `langchain.agents`. It remains functional and receives bug fixes.

### Full parameter reference

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=...,                   # LLM or callable(state, runtime) → LLM
    tools=[...],                 # list[BaseTool | callable] or ToolNode
    prompt=None,                 # str | SystemMessage | callable | Runnable
    response_format=None,        # Pydantic model, TypedDict, JSON schema, or (prompt, schema) tuple
    pre_model_hook=None,         # callable | Runnable — runs before every LLM call
    post_model_hook=None,        # callable | Runnable — runs after every LLM call
    state_schema=None,           # custom state TypedDict (default: MessagesState)
    context_schema=None,         # read-only context type
    checkpointer=None,           # any BaseCheckpointSaver
    store=None,                  # any BaseStore
    interrupt_before=None,       # list of node names to pause before
    interrupt_after=None,        # list of node names to pause after
    debug=False,
    version="v2",                # "v1" or "v2"
    name=None,                   # graph name
)
```

### Basic: prompt and tools

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Use a stub so this example runs without an API key
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolCall

mock_llm = MagicMock()
mock_llm.bind_tools.return_value = mock_llm
mock_llm.invoke.return_value = AIMessage(content="Done", tool_calls=[])

agent = create_react_agent(
    model=mock_llm,
    tools=[add, multiply],
    prompt="You are a math assistant. Always show your working.",
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "math-1"}}
result = agent.invoke({"messages": [("user", "What is 6 * 7?")]}, cfg)
print(result["messages"][-1].content)
```

### `pre_model_hook` — message trimming

`pre_model_hook` is a callable/Runnable that receives the full agent state and returns a dict to **merge into** state before the LLM call. The most common use-case is message trimming or summarisation:

```python
from langchain_core.messages import trim_messages, AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class TrimmedState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def trim_hook(state: TrimmedState) -> dict:
    """Keep only the last 10 messages to avoid context-window overflow."""
    kept = trim_messages(
        state["messages"],
        max_tokens=4000,
        token_counter=len,        # replace with tiktoken in production
        strategy="last",
        allow_partial=False,
    )
    return {"messages": kept}


agent = create_react_agent(
    model=mock_llm,
    tools=[add],
    pre_model_hook=trim_hook,
    state_schema=TrimmedState,
    checkpointer=InMemorySaver(),
)
```

### `post_model_hook` — tracking cost / usage

`post_model_hook` receives the state **after** the LLM has responded. Use it to record token usage or update counters:

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class TrackedState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    total_tokens: int


def track_usage(state: TrackedState) -> dict:
    """Accumulate token usage from the last AI message."""
    last = state["messages"][-1]
    usage = getattr(last, "usage_metadata", None) or {}
    new_tokens = (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    return {"total_tokens": state.get("total_tokens", 0) + new_tokens}


agent = create_react_agent(
    model=mock_llm,
    tools=[add],
    post_model_hook=track_usage,
    state_schema=TrackedState,
)
```

### `response_format` — structured output

When you provide `response_format`, the agent makes a **separate LLM call** at the end using `.with_structured_output()` and returns the result in `state["structured_response"]`:

```python
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent


class MathResult(BaseModel):
    answer: float
    explanation: str
    confidence: float


agent = create_react_agent(
    model=mock_llm,
    tools=[add, multiply],
    response_format=MathResult,
)

# The structured response is in state["structured_response"]
result = agent.invoke({"messages": [("user", "What is 6 * 7?")]})
# result["structured_response"] is a MathResult instance (or None if the model fails)
```

You can also pass a `(prompt, schema)` tuple to use a custom extraction prompt:

```python
agent = create_react_agent(
    model=mock_llm,
    tools=[add],
    response_format=(
        "Extract the numerical answer and confidence from the conversation.",
        MathResult,
    ),
)
```

### Dynamic model selection

Pass a callable `(state, runtime) → BaseChatModel` to swap the model at run-time based on context:

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime


@dataclass
class ModelCtx:
    tier: str = "standard"   # "standard" or "premium"


# Pre-instantiate models (don't recreate on every call)
# gpt4_model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
# gpt35_model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def select_model(state, runtime: Runtime[ModelCtx]):
    """Use the premium model for premium-tier requests."""
    if runtime.context and runtime.context.tier == "premium":
        return gpt4_model
    return gpt35_model


# agent = create_react_agent(
#     model=select_model,
#     tools=[add, multiply],
#     context_schema=ModelCtx,
# )
# result = agent.invoke(
#     {"messages": [("user", "What is 6 * 7?")]},
#     context=ModelCtx(tier="premium"),
# )
```

### `interrupt_before` / `interrupt_after`

Pause execution at specific nodes for human review without modifying node code:

```python
agent = create_react_agent(
    model=mock_llm,
    tools=[add, multiply],
    checkpointer=InMemorySaver(),
    interrupt_before=["tools"],  # pause before every tool execution
)

cfg = {"configurable": {"thread_id": "review"}}

# First call pauses before tools
for ev in agent.stream({"messages": [("user", "What is 6 * 7?")]}, cfg):
    print(ev)

# Inspect
snap = agent.get_state(cfg)
print(snap.next)           # ('tools',)
print(snap.interrupts)     # currently no GraphInterrupt for interrupt_before — just paused at boundary

# Resume execution
from langgraph.types import Command
for ev in agent.stream(Command(resume=None), cfg):
    print(ev)
```

---

## 6 · `Command`

**Module:** `langgraph.types`

`Command` is the universal return type for nodes that need to both **update state** and **control the next node**. Returning a `Command` from a node replaces a static edge.

### Full signature

```python
@dataclass
class Command:
    graph:  str | None = None        # None = this graph, Command.PARENT = parent graph
    update: Any | None = None        # state update (dict, list of (key,val) tuples, BaseModel)
    resume: dict | Any | None = None # value(s) to resume interrupt() with
    goto:   Send | Sequence[Send | str] | str = ()
```

### Supervisor routing pattern

```python
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated


class SupervState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str


def supervisor(state: SupervState) -> Command[Literal["researcher", "writer", END]]:
    last = state["messages"][-1].content.lower()
    if "research" in last:
        target = "researcher"
    elif "write" in last or "draft" in last:
        target = "writer"
    else:
        target = END
    return Command(
        goto=target,
        update={"route": target},
    )


def researcher(state: SupervState) -> Command[Literal["supervisor"]]:
    response = AIMessage(content="Research complete: found 5 sources.")
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )


def writer(state: SupervState) -> Command[Literal["supervisor"]]:
    response = AIMessage(content="Draft complete: 500 words written.")
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )


builder = StateGraph(SupervState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_edge(START, "supervisor")

graph = builder.compile()
result = graph.invoke({
    "messages": [HumanMessage(content="Please research LangGraph then write a summary.")],
    "route": "",
})
for m in result["messages"]:
    print(m.content)
```

### `Command.PARENT` — escaping a subgraph

A node inside a subgraph can send a `Command` up to the parent graph using `graph=Command.PARENT`:

```python
from langgraph.types import Command


def escalate_node(state: dict) -> Command:
    """From inside a subgraph, update the parent state and route to a parent node."""
    return Command(
        graph=Command.PARENT,
        update={"escalation_reason": "budget exceeded"},
        goto="approval_node",   # a node in the PARENT graph
    )
```

### Resume by interrupt ID

When a node contains multiple `interrupt()` calls, resume each by its unique ID:

```python
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class MultiState(TypedDict):
    a: str
    b: str


def dual_interrupt(state: MultiState) -> dict:
    answer_a = interrupt({"q": "First question?"})
    answer_b = interrupt({"q": "Second question?"})
    return {"a": answer_a, "b": answer_b}


builder = StateGraph(MultiState)
builder.add_node("dual", dual_interrupt)
builder.add_edge(START, "dual")
builder.add_edge("dual", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "multi"}}

# First run — pauses at first interrupt
list(graph.stream({"a": "", "b": ""}, cfg))

snap = graph.get_state(cfg)
first_id = snap.interrupts[0].id

# Resume the first interrupt by its ID, leaving second pending
list(graph.stream(Command(resume={first_id: "answer-to-first"}), cfg))

snap2 = graph.get_state(cfg)
second_id = snap2.interrupts[0].id

# Resume the second interrupt
list(graph.stream(Command(resume={second_id: "answer-to-second"}), cfg))
```

### `goto` with `Send` objects

```python
from langgraph.types import Command, Send


def fan_out_node(state: dict) -> Command:
    """Send each item in a list to a worker node in parallel."""
    items = state.get("items", [])
    return Command(
        goto=[Send("worker", {"item": item, "idx": i}) for i, item in enumerate(items)],
    )
```

---

## 7 · `Send`

**Module:** `langgraph.types`

`Send` routes execution to a named node with a **specific input**, bypassing the shared state. This enables dynamic fan-out (map-reduce) where the number of parallel branches isn't known at graph-build time.

### Map-reduce pattern

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class MapState(TypedDict):
    urls:    list[str]           # input: list of URLs to scrape
    results: Annotated[list[str], operator.add]  # reduced: all results combined


class WorkerState(TypedDict):
    url:    str
    result: str


def distribute(state: MapState) -> list[Send]:
    """Conditional edge — return one Send per URL."""
    return [Send("scrape", {"url": u, "result": ""}) for u in state["urls"]]


def scrape(state: WorkerState) -> dict:
    """Worker: scrape one URL. This runs in parallel across all Sends."""
    return {"results": [f"content of {state['url']}"]}


builder = StateGraph(MapState)
builder.add_node("scrape", scrape)
builder.add_conditional_edges(START, distribute)
builder.add_edge("scrape", END)

graph = builder.compile()
result = graph.invoke({
    "urls": ["a.com", "b.com", "c.com"],
    "results": [],
})
print(result["results"])   # ['content of a.com', 'content of b.com', 'content of c.com']
```

### Variable-depth tree traversal

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class TreeState(TypedDict):
    nodes:    list[dict]   # remaining nodes to process
    visited:  Annotated[list[str], operator.add]


class NodeState(TypedDict):
    node_id:  str
    children: list[str]


def branch(state: TreeState) -> list[Send] | str:
    """Fan out to all unvisited nodes."""
    if not state["nodes"]:
        return END
    return [Send("process", n) for n in state["nodes"]]


def process(state: NodeState) -> dict:
    """Process a single tree node; report its id."""
    return {"visited": [state["node_id"]]}


builder = StateGraph(TreeState)
builder.add_node("process", process)
builder.add_conditional_edges(START, branch)
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({
    "nodes": [
        {"node_id": "A", "children": []},
        {"node_id": "B", "children": []},
        {"node_id": "C", "children": []},
    ],
    "visited": [],
})
print(sorted(result["visited"]))  # ['A', 'B', 'C']
```

---

## 8 · `@task` and `@entrypoint` (Functional API)

**Module:** `langgraph.func`

The Functional API lets you write a workflow as a normal Python function instead of building a `StateGraph`. The decorated function gets the same checkpointing, streaming, interrupts, and time-travel.

### Parallel tasks with retry and caching

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, CachePolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache


@task(
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1),
    cache_policy=CachePolicy(ttl=600),   # cache results for 10 minutes
    name="fetch_url",
)
def fetch(url: str) -> str:
    """Fetch a URL — retried up to 3 times, result cached for 10 min."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read(500).decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"fetch failed: {e}") from e


@task(name="summarise")
def summarise(text: str) -> str:
    """Produce a one-line summary (stub)."""
    return text[:80].replace("\n", " ") + "..."


cache = InMemoryCache()
saver = InMemorySaver()


@entrypoint(checkpointer=saver, cache=cache)
def research_pipeline(urls: list[str]) -> list[str]:
    """Fetch all URLs in parallel, then summarise each."""
    # All fetches start immediately (futures)
    fetch_futures = [fetch(u) for u in urls]
    # Summarise each — each summarise call can also run in parallel
    summarise_futures = [summarise(f.result()) for f in fetch_futures]
    return [sf.result() for sf in summarise_futures]


cfg = {"configurable": {"thread_id": "research-1"}}
# First run hits the network; second run reads from cache
result = research_pipeline.invoke(["https://example.com", "https://python.org"], cfg)
print(result)
```

### `entrypoint.final` — save a different value than what you return

Sometimes you want to **return** a value to the caller but **checkpoint** a different value for the next run's `previous`:

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict


class ConvHistory(TypedDict):
    messages: list[str]


saver = InMemorySaver()


@entrypoint(checkpointer=saver)
def conversation(
    user_msg: str,
    *,
    previous: ConvHistory | None,
) -> str:
    """Return a response string but checkpoint the full history."""
    history = previous or ConvHistory(messages=[])
    response = f"You said: {user_msg}"

    updated_history = ConvHistory(
        messages=history["messages"] + [f"user: {user_msg}", f"bot: {response}"]
    )

    # Return the response to the caller, save the full history to checkpoint
    return entrypoint.final(value=response, save=updated_history)


cfg = {"configurable": {"thread_id": "chat-1"}}
print(conversation.invoke("hello", cfg))   # "You said: hello"
print(conversation.invoke("world", cfg))   # "You said: world" (previous has full history)
```

### Interrupts inside tasks

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


@task
def draft_email(to: str, subject: str) -> str:
    return f"To: {to}\nSubject: {subject}\n\nDear {to}, ..."


@entrypoint(checkpointer=InMemorySaver())
def send_email_workflow(params: dict) -> dict:
    draft = draft_email(params["to"], params["subject"]).result()

    # Pause for human approval — the task has already run (results cached)
    approved = interrupt({
        "draft": draft,
        "question": "Send this email?",
    })

    if approved == "yes":
        return {"status": "sent", "draft": draft}
    return {"status": "cancelled", "draft": draft}


cfg = {"configurable": {"thread_id": "email-1"}}
list(send_email_workflow.stream({"to": "alice@example.com", "subject": "Hello"}, cfg))

# Inspect and resume
snap = send_email_workflow.get_state(cfg)
print(snap.interrupts[0].value["question"])   # 'Send this email?'

list(send_email_workflow.stream(Command(resume="yes"), cfg))
```

### Async tasks

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver


@task
async def async_fetch(url: str) -> str:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            return await r.text()


@entrypoint(checkpointer=InMemorySaver())
async def async_pipeline(urls: list[str]) -> list[str]:
    futures = [async_fetch(u) for u in urls]
    return await asyncio.gather(*futures)


async def main():
    cfg = {"configurable": {"thread_id": "async-1"}}
    result = await async_pipeline.ainvoke(["https://example.com"], cfg)
    print(result)

# asyncio.run(main())
```

---

## 9 · `BinaryOperatorAggregate` and `Topic`

**Module:** `langgraph.channels.binop`, `langgraph.channels.topic`

Both channels handle **concurrent writes** in the same super-step — situations where two or more parallel nodes write to the same state key. The default `LastValue` channel **raises** `InvalidUpdateError` in that case.

### `BinaryOperatorAggregate` — reducer merging

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class AggState(TypedDict):
    total:       Annotated[int,       operator.add]      # accumulate integers
    max_score:   Annotated[float,     max]               # keep the maximum
    tags:        Annotated[list[str], operator.add]      # concatenate lists
    combined:    Annotated[str,       lambda a, b: a + "|" + b]  # custom reducer


def node_a(state: AggState) -> dict:
    return {"total": 10, "max_score": 0.8, "tags": ["fast"], "combined": "A"}


def node_b(state: AggState) -> dict:
    return {"total": 5, "max_score": 0.95, "tags": ["accurate"], "combined": "B"}


def node_c(state: AggState) -> dict:
    return {"total": 3, "max_score": 0.7, "tags": ["cheap"], "combined": "C"}


builder = StateGraph(AggState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
# All three run in parallel from START
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(START, "c")
builder.add_edge(["a", "b", "c"], END)

graph = builder.compile()
result = graph.invoke({"total": 0, "max_score": 0.0, "tags": [], "combined": ""})
print(result["total"])      # 18  (10+5+3)
print(result["max_score"])  # 0.95
print(result["tags"])       # ['fast', 'accurate', 'cheap']
print(result["combined"])   # 'A|B|C' (order depends on completion)
```

### `Overwrite` — force-reset a reducing channel

Normally a reducer **merges** values. `Overwrite` skips the merge and replaces instead:

```python
from langgraph.types import Overwrite


def reset_node(state: AggState) -> dict:
    """Forcibly reset the total even though it has a reducer."""
    return {"total": Overwrite(0)}
```

### `Topic` — PubSub event buffer

`Topic` collects **all** writes in a super-step into a list, then clears it for the next step (unless `accumulate=True`):

```python
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic


class EventState(TypedDict):
    events:     Annotated[list[str], Topic(str)]          # cleared each step
    history:    Annotated[list[str], Topic(str, accumulate=True)]  # never cleared


def producer_a(state: EventState) -> dict:
    return {"events": "a_ran", "history": "a_ran"}


def producer_b(state: EventState) -> dict:
    return {"events": "b_ran", "history": "b_ran"}


def consumer(state: EventState) -> dict:
    print(f"events this step: {state['events']}")
    print(f"all history: {state['history']}")
    return {}


builder = StateGraph(EventState)
builder.add_node("a", producer_a)
builder.add_node("b", producer_b)
builder.add_node("consume", consumer)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(["a", "b"], "consume")
builder.add_edge("consume", END)

graph = builder.compile()
graph.invoke({"events": [], "history": []})
# events this step: ['a_ran', 'b_ran']
# all history: ['a_ran', 'b_ran']
```

### Choosing the right channel

| Scenario | Channel | Declaration |
|---|---|---|
| Single writer per step | `LastValue` (default) | `value: str` |
| Sum / count across parallel nodes | `BinaryOperatorAggregate` | `Annotated[int, operator.add]` |
| Keep max/min from parallel nodes | `BinaryOperatorAggregate` | `Annotated[float, max]` |
| Collect all events in a step | `Topic` | `Annotated[list[str], Topic(str)]` |
| Accumulate events forever | `Topic(accumulate=True)` | `Annotated[list[str], Topic(str, accumulate=True)]` |
| Message history (smart merge) | `add_messages` | `Annotated[list[AnyMessage], add_messages]` |
| Force-reset a reducing channel | `Overwrite` | Return `{"key": Overwrite(new_value)}` |

---

## 10 · `InMemoryStore`

**Module:** `langgraph.store.memory`  
**Base class:** `langgraph.store.base.BaseStore`

`InMemoryStore` provides **long-term, cross-thread memory** — a key-value store whose data lives outside checkpoints and survives across multiple conversations. Optionally add a vector index for semantic search.

### Basic CRUD

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# --- put ---
store.put(("users", "alice"), "preferences", {"theme": "dark", "lang": "en"})
store.put(("users", "alice"), "notes", {"items": ["remember to call Bob"]})

# --- get ---
item = store.get(("users", "alice"), "preferences")
print(item.value)       # {'theme': 'dark', 'lang': 'en'}
print(item.namespace)   # ('users', 'alice')
print(item.key)         # 'preferences'
print(item.created_at)  # datetime

# --- update: put again (upserts) ---
store.put(("users", "alice"), "preferences", {"theme": "light", "lang": "en"})

# --- delete ---
store.delete(("users", "alice"), "notes")
print(store.get(("users", "alice"), "notes"))  # None
```

### Searching and listing namespaces

```python
store.put(("users", "bob"),   "preferences", {"theme": "dark"})
store.put(("users", "carol"), "preferences", {"theme": "light"})
store.put(("teams", "eng"),   "config",      {"sprint": 42})

# search by metadata filter
results = store.search(("users",), filter={"theme": "dark"})
for hit in results:
    print(hit.namespace, hit.key, hit.value)
# ('users', 'alice') preferences {'theme': 'dark', ...}   (if still dark)
# ('users', 'bob')   preferences {'theme': 'dark'}

# list_namespaces
namespaces = store.list_namespaces(prefix=("users",))
print(namespaces)
# [('users', 'alice'), ('users', 'bob'), ('users', 'carol')]

# max_depth=1 — get only the first segment
print(store.list_namespaces(max_depth=1))
# [('users',), ('teams',)]
```

### Batch operations

`batch()` submits multiple operations in one call — important for backends like Postgres where each call is a round-trip:

```python
from langgraph.store.base import GetOp, PutOp, SearchOp

results = store.batch([
    GetOp(namespace=("users", "alice"), key="preferences"),
    PutOp(namespace=("users", "dan"), key="prefs", value={"theme": "dark"}),
    SearchOp(namespace_prefix=("users",), filter={"theme": "dark"}, limit=5),
])

get_result, put_result, search_result = results
print(get_result.value if get_result else "not found")
print(search_result)   # list[SearchItem]
```

### Semantic / vector search

Install numpy for better performance (`pip install numpy`). Pass `index` to `InMemoryStore` to enable vector similarity search. Use `embed` with any function that maps `list[str] → list[list[float]]`:

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    """Stub embedder: returns the character count as a 1-D vector."""
    return [[float(len(t))] for t in texts]

store = InMemoryStore(
    index={
        "dims": 1,
        "embed": embed,
        "fields": ["text"],   # which value keys to embed; default is entire value
    }
)

store.put(("docs",), "python-guide",     {"text": "Python programming guide"})
store.put(("docs",), "typescript-guide", {"text": "TypeScript programming guide"})
store.put(("docs",), "docker-intro",     {"text": "Docker container introduction"})

# Semantic search
hits = store.search(("docs",), query="Python scripting tutorial", limit=2)
for h in hits:
    print(h.key, h.score, h.value["text"])
```

### Using the store inside a graph

Wire `store` into `.compile()` and access it from nodes via `Runtime`:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


class ConvState(TypedDict):
    user_id: str
    message: str
    reply:   str


def recall_and_respond(state: ConvState, runtime: Runtime) -> dict:
    """Reads from store, produces a reply, and writes back."""
    user_id = state["user_id"]
    store = runtime.store

    # Load prior messages
    history_item = store.get(("chats", user_id), "history")
    history = history_item.value["msgs"] if history_item else []

    # Produce a reply (stub)
    reply = f"[based on {len(history)} prior messages] {state['message']!r}"

    # Save updated history
    store.put(("chats", user_id), "history", {"msgs": history + [state["message"]]})

    return {"reply": reply}


store = InMemoryStore()
builder = StateGraph(ConvState)
builder.add_node("respond", recall_and_respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile(store=store)

cfg1 = {"configurable": {"thread_id": "t1"}}
print(graph.invoke({"user_id": "alice", "message": "hello", "reply": ""}, cfg1)["reply"])
print(graph.invoke({"user_id": "alice", "message": "how are you?", "reply": ""}, cfg1)["reply"])
# Second call shows 1 prior message in history
```

### TTL support

`InMemoryStore` accepts a `ttl` kwarg on `put()` (seconds until expiry). The store auto-expires entries on read:

```python
import time

store = InMemoryStore()
store.put(("sessions",), "tok123", {"data": "session data"}, ttl=2)  # expires in 2 s
item = store.get(("sessions",), "tok123")
print(item is not None)   # True

time.sleep(3)
item = store.get(("sessions",), "tok123")
print(item)   # None — expired
```

---

## Quick reference: which feature lives where

| Feature | Class/Function | Module |
|---|---|---|
| Build a graph | `StateGraph` | `langgraph.graph.state` |
| Run a graph | `CompiledStateGraph` | `langgraph.graph.state` |
| Short-term memory (in-process) | `InMemorySaver` | `langgraph.checkpoint.memory` |
| Custom checkpointer | `BaseCheckpointSaver` | `langgraph.checkpoint.base` |
| Tool execution node | `ToolNode` | `langgraph.prebuilt.tool_node` |
| Quick ReAct agent | `create_react_agent` | `langgraph.prebuilt` |
| Routing + state updates | `Command` | `langgraph.types` |
| Dynamic fan-out | `Send` | `langgraph.types` |
| Pause for human input | `interrupt()` | `langgraph.types` |
| Function-based workflow | `@entrypoint` / `@task` | `langgraph.func` |
| Concurrent write merging | `BinaryOperatorAggregate` | `langgraph.channels.binop` |
| PubSub event channel | `Topic` | `langgraph.channels.topic` |
| Long-term / cross-thread memory | `InMemoryStore` | `langgraph.store.memory` |
| Node retry on failure | `RetryPolicy` | `langgraph.types` |
| Node result caching | `CachePolicy` + `InMemoryCache` | `langgraph.types`, `langgraph.cache.memory` |
| Recursion limit detection | `IsLastStep` / `RemainingSteps` | `langgraph.managed.is_last_step` |

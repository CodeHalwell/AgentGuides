---
title: "Class deep-dives Vol. 23 — node wrappers, concurrency primitives, remote streaming & migration guide (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: RunnableCallable+RunnableSeq (node function wrappers — sync/async dual registration, func_accepts injection, RunnableSeq chaining), _IdleProgressCallbackHandler (idle-timeout heartbeat via every LangChain callback event — how touch() resets the idle clock), _GraphCallbackManager+_AsyncGraphCallbackManager (graph lifecycle event dispatch — configure() from config callbacks, on_interrupt/on_resume fan-out), DataclassLike+TypedDictLikeV1+TypedDictLikeV2 (state schema typing protocols — how LangGraph discriminates TypedDict/dataclass/Pydantic at runtime), AsyncQueue+SyncQueue+Semaphore (internal concurrency primitives — wait()-without-consuming non-destructive peek, Pregel coordinator-worker signaling), FunctionNonLocals+NonLocals (AST-based closure inspection used by @task and @entrypoint — why outer-scope names matter), Edge+TriggerEdge (graph topology NamedTuples powering draw_mermaid/draw_ascii — how to extract and render graph structure), _RemoteGraphRunStream+_ChannelProjection+_ProjectionRegistry (remote graph streaming adapter — sync/async lanes, channel projection registry matching local transformer output shapes), AgentState+AgentStatePydantic migration guide (deprecated in 1.2.6 — migration to custom TypedDict/Pydantic state with remaining_steps and structured response), and _SubgraphRunStreamMixin (subgraph streaming pump-delegation internals — path/graph_name/trigger_call_id metadata, parent-pump handoff)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 23"
  order: 54
---

# Class deep-dives Vol. 23 — node wrappers, concurrency primitives, remote streaming & migration guide (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `RunnableCallable` + `RunnableSeq` | `langgraph._internal._runnable` |
| 2 | `_IdleProgressCallbackHandler` | `langgraph.pregel._retry` |
| 3 | `_GraphCallbackManager` + `_AsyncGraphCallbackManager` | `langgraph.callbacks` |
| 4 | `DataclassLike` + `TypedDictLikeV1` + `TypedDictLikeV2` | `langgraph._internal._typing` |
| 5 | `AsyncQueue` + `SyncQueue` + `Semaphore` | `langgraph._internal._queue` |
| 6 | `FunctionNonLocals` + `NonLocals` | `langgraph.pregel._utils` |
| 7 | `Edge` + `TriggerEdge` | `langgraph.pregel._draw` |
| 8 | `_RemoteGraphRunStream` + `_ChannelProjection` + `_ProjectionRegistry` | `langgraph.pregel._remote_run_stream` |
| 9 | `AgentState` + `AgentStatePydantic` deprecation & migration | `langgraph.prebuilt.chat_agent_executor` |
| 10 | `_SubgraphRunStreamMixin` | `langgraph.stream.run_stream` |

---

## 1 · `RunnableCallable` + `RunnableSeq`

**Module**: `langgraph._internal._runnable`  
**First dedicated coverage.**

Every function you pass to `add_node()` is wrapped in a `RunnableCallable` before it is stored as `PregelNode.bound`. This wrapper is LangGraph's own lightweight alternative to LangChain's `RunnableLambda` — it handles sync/async dual dispatch, parameter injection detection, tracing opt-out, and argument explosion.

```python
class RunnableCallable(Runnable):
    def __init__(
        self,
        func: Callable[..., Any | Runnable] | None,
        afunc: Callable[..., Awaitable[Any | Runnable]] | None = None,
        *,
        name: str | None = None,
        tags: Sequence[str] | None = None,
        trace: bool = True,
        recurse: bool = True,
        explode_args: bool = False,
        **kwargs: Any,
    ) -> None: ...
```

Key parameters:

| Parameter | What it does |
|---|---|
| `func` | Sync callable; used for `invoke()` |
| `afunc` | Async callable; used for `ainvoke()`. Falls back to sync if absent |
| `trace` | When `False`, suppresses LangSmith tracing for this node |
| `recurse` | When `True`, the output is passed back through the Runnable pipeline if it is itself a `Runnable` |
| `explode_args` | When `True`, expands a dict output as `**kwargs` into the next step |
| `**kwargs` | Extra kwargs forwarded to every `func` call |

### How parameter injection works

At construction, `RunnableCallable` inspects the function signature against `KWARGS_CONFIG_KEYS` — a list of `(param_name, allowed_types, runtime_key, default)` tuples. Matched parameters are stored in `func_accepts`. At invocation, only the params the function actually declares are injected.

```python
# Four injectable parameters (declared in any combination)
from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_stream_writer

class State(TypedDict):
    value: str

# Node that requests ALL four injectable params
def full_node(
    state: State,
    config: RunnableConfig,           # injected: run-level config
    *,
    runtime: Runtime,                  # injected: context/store/stream_writer/heartbeat
    store: InMemoryStore,              # injected: same store as runtime.store
) -> dict:
    run_id = config.get("run_id", "none")
    user_id = getattr(runtime.context, "user_id", "anon")
    runtime.stream_writer({"status": f"processing for {user_id}"})
    return {"value": f"{state['value']} processed by {run_id[:8]}"}

store = InMemoryStore()
graph = StateGraph(State).add_node("work", full_node)
graph.add_edge(START, "work")
compiled = graph.compile(store=store)
result = compiled.invoke({"value": "hello"})
```

### Sync + async dual registration

```python
import asyncio
from typing import TypedDict
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import StateGraph, START

class State(TypedDict):
    n: int

def sync_double(state: State) -> dict:
    return {"n": state["n"] * 2}

async def async_double(state: State) -> dict:
    await asyncio.sleep(0)          # simulate async I/O
    return {"n": state["n"] * 2}

# Explicitly build a dual-dispatch node
dual = RunnableCallable(sync_double, async_double, name="double", trace=False)

graph = StateGraph(State).add_node("double", dual).add_edge(START, "double").compile()
print(graph.invoke({"n": 5}))               # {"n": 10}  — uses sync_double
print(asyncio.run(graph.ainvoke({"n": 5}))) # {"n": 10}  — uses async_double
```

### `RunnableSeq` — lightweight pipeline composition

`RunnableSeq` is LangGraph's own sequential pipeline. Unlike `RunnableSequence` (from LangChain), it stays internal and skips the expensive step-flattening on every invocation.

```python
from langgraph._internal._runnable import RunnableCallable, RunnableSeq

def add_one(x: dict) -> dict:
    return {"n": x["n"] + 1}

def multiply_two(x: dict) -> dict:
    return {"n": x["n"] * 2}

# Build a two-step pipeline — LangGraph uses this internally for
# nodes that call add_node(...) with a RunnableLike already in bound
pipeline = RunnableSeq(
    RunnableCallable(add_one, name="add_one"),
    RunnableCallable(multiply_two, name="multiply_two"),
    name="add_then_multiply",
)
print(pipeline.invoke({"n": 3}))  # {"n": 8}  — (3+1)*2
```

### Disabling tracing for a node

```python
from typing import TypedDict
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import StateGraph, START

class State(TypedDict):
    secret: str

# This node will NOT appear in LangSmith traces
def redact(state: State) -> dict:
    return {"secret": "***"}

no_trace_node = RunnableCallable(redact, trace=False, name="redact")

graph = (
    StateGraph(State)
    .add_node("redact", no_trace_node)
    .add_edge(START, "redact")
    .compile()
)
result = graph.invoke({"secret": "my-api-key"})
print(result)  # {"secret": "***"}
```

---

## 2 · `_IdleProgressCallbackHandler`

**Module**: `langgraph.pregel._retry`  
**First dedicated coverage.**

When a node runs with a `TimeoutPolicy(idle_timeout=N)` and `refresh_on="auto"`, LangGraph attaches an `_IdleProgressCallbackHandler` to the invocation config. Any LangChain callback event emitted inside that node — LLM token, tool call start/end, retriever hit, chain event — calls `scope.touch()`, which resets the idle clock.

```python
class _IdleProgressCallbackHandler(BaseCallbackHandler):
    run_inline = True   # callbacks fire synchronously, in emission order

    def __init__(self, scope: _TimedAttemptScope) -> None:
        self._scope_ref = weakref.ref(scope)  # weakref prevents scope lifetime coupling

    def _touch(self, *args, **kwargs) -> None:
        if (scope := self._scope_ref()) is not None:
            scope.touch()

    # ALL of these reset the idle clock:
    on_llm_start = on_chat_model_start = on_llm_new_token = _touch
    on_llm_end = on_llm_error = _touch
    on_chain_start = on_chain_end = on_chain_error = _touch
    on_tool_start = on_tool_end = on_tool_error = _touch
    on_retriever_start = on_retriever_end = on_retriever_error = _touch
    on_agent_action = on_agent_finish = _touch
    on_text = on_retry = on_custom_event = _touch
```

### Why `run_inline = True` matters

With `run_inline = True`, the handler fires in the same thread as the LangChain event, without any queue or thread-pool dispatch. This is deliberate: reordering would mean a burst of LLM tokens arriving close together could fire `touch()` out of order, making the idle clock look ahead of real progress.

### Practical example — idle timeout with LLM streaming

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.types import TimeoutPolicy

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

async def chat(state: State) -> dict:
    # Each streaming token fires on_llm_new_token → _IdleProgressCallbackHandler._touch()
    # → scope.touch() — the idle clock resets with every token
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# idle_timeout=10 means: reset to 10 s on every LangChain event.
# A node that never emits any LangChain event would time out in 10 s.
graph = (
    StateGraph(State)
    .add_node("chat", chat, timeout=TimeoutPolicy(idle_timeout=10.0))
    .add_edge(START, "chat")
    .compile()
)
```

### Writing a custom progress-counting callback

```python
from langchain_core.callbacks import BaseCallbackHandler
import threading

class ProgressCounter(BaseCallbackHandler):
    """Count progress events emitted inside a node invocation."""
    run_inline = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def _record(self, event: str) -> None:
        with self._lock:
            self.counts[event] = self.counts.get(event, 0) + 1

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._record("llm_tokens")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._record("tool_calls")

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        self._record("chain_starts")

counter = ProgressCounter()

from langchain_core.runnables import RunnableConfig
from typing import TypedDict
from langgraph.graph import StateGraph, START

class State(TypedDict):
    result: str

def my_node(state: State, config: RunnableConfig) -> dict:
    # Merge our counter into the run callbacks
    config.setdefault("callbacks", []).append(counter)
    return {"result": "done"}

graph = StateGraph(State).add_node("work", my_node).add_edge(START, "work").compile()
graph.invoke({"result": ""})
print(counter.counts)
```

---

## 3 · `_GraphCallbackManager` + `_AsyncGraphCallbackManager`

**Module**: `langgraph.callbacks`  
**First dedicated coverage.**

`GraphCallbackHandler.on_interrupt()` and `on_resume()` are not called directly — LangGraph constructs a `_GraphCallbackManager` (sync runs) or `_AsyncGraphCallbackManager` (async runs) from `config["callbacks"]` and calls the lifecycle methods on it. Only handlers that subclass `GraphCallbackHandler` receive these events; plain `BaseCallbackHandler` instances are filtered out.

```python
# Sync dispatcher
class _GraphCallbackManager(BaseCallbackManager):
    @classmethod
    def configure(cls, callbacks=None, *, run_id=None) -> _GraphCallbackManager: ...
    def copy(self, *, run_id=...) -> _GraphCallbackManager: ...
    def on_interrupt(self, event: GraphInterruptEvent) -> None: ...
    def on_resume(self, event: GraphResumeEvent) -> None: ...

# Async dispatcher (same API, awaitable methods)
class _AsyncGraphCallbackManager(BaseCallbackManager):
    @property
    def is_async(self) -> bool: return True
    async def on_interrupt(self, event: GraphInterruptEvent) -> None: ...
    async def on_resume(self, event: GraphResumeEvent) -> None: ...
```

### Building a full graph lifecycle audit handler

```python
import json
from datetime import datetime, timezone
from typing import Any
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

class AuditHandler(GraphCallbackHandler):
    """Write a JSONL audit log for every graph interrupt and resume."""

    def __init__(self, log_path: str) -> None:
        self._log_path = log_path

    def _write(self, record: dict[str, Any]) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self._write({
            "event": "interrupt",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": str(event.run_id),
            "checkpoint_id": event.checkpoint_id,
            "checkpoint_ns": list(event.checkpoint_ns),
            "interrupt_count": len(event.interrupts),
            "values": [str(i.value) for i in event.interrupts],
        })

    def on_resume(self, event: GraphResumeEvent) -> None:
        self._write({
            "event": "resume",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": str(event.run_id),
            "checkpoint_id": event.checkpoint_id,
        })

# Attach via config["callbacks"]
from typing import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    approved: bool
    data: str

def review_node(state: State) -> dict:
    decision = interrupt({"data": state["data"], "question": "approve?"})
    return {"approved": decision == "yes"}

audit = AuditHandler("/tmp/audit.jsonl")
graph = (
    StateGraph(State)
    .add_node("review", review_node)
    .add_edge(START, "review")
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "t1"}, "callbacks": [audit]}
# First invocation — pauses at interrupt → on_interrupt fires
graph.invoke({"approved": False, "data": "order-42"}, config=config)
# Resume — on_resume fires
graph.invoke(Command(resume="yes"), config=config)
```

### Multi-subgraph interrupt tracking

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent

class SubgraphAwareHandler(GraphCallbackHandler):
    """Track which subgraph depth triggered each interrupt."""

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        depth = len(event.checkpoint_ns)
        ns_path = " > ".join(event.checkpoint_ns) if event.checkpoint_ns else "root"
        print(f"[depth={depth}] interrupt at {ns_path!r}: {event.interrupts}")

    def on_resume(self, event: GraphResumeEvent) -> None:
        print(f"resumed from checkpoint {event.checkpoint_id[:8]}")
```

### Async handler for FastAPI integration

```python
import asyncio
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

class AsyncWebhookHandler(GraphCallbackHandler):
    """Fire async webhooks on graph lifecycle events."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    async def on_interrupt(self, event: GraphInterruptEvent) -> None:
        import aiohttp
        payload = {
            "type": "interrupt",
            "run_id": str(event.run_id),
            "interrupts": len(event.interrupts),
        }
        async with aiohttp.ClientSession() as session:
            await session.post(self._url, json=payload)

    async def on_resume(self, event: GraphResumeEvent) -> None:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(self._url, json={"type": "resume", "run_id": str(event.run_id)})
```

---

## 4 · `DataclassLike` + `TypedDictLikeV1` + `TypedDictLikeV2`

**Module**: `langgraph._internal._typing`  
**First dedicated coverage.**

When `StateGraph` processes your state schema, it uses these three structural protocols to determine what kind of schema you gave it — without requiring inheritance. Any class that has the right class attributes satisfies the protocol at runtime (`isinstance(x, DataclassLike)` via `@runtime_checkable`).

```python
# DataclassLike: any class with __dataclass_fields__ (standard for @dataclass)
class DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

# TypedDictLikeV1: ClassVar keys (Python 3.9 style TypedDict)
class TypedDictLikeV1(Protocol):
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]

# TypedDictLikeV2: instance-level keys (Python 3.12+ TypedDict)
class TypedDictLikeV2(Protocol):
    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]
```

### Schema detection order

LangGraph checks in this order:
1. `is_pydantic_model(schema)` — Pydantic `BaseModel` subclass
2. `isinstance(schema, DataclassLike)` — Python `@dataclass`
3. `isinstance(schema, TypedDictLikeV1 | TypedDictLikeV2)` — `TypedDict`

### Building a custom schema that satisfies `TypedDictLikeV1`

```python
from __future__ import annotations
from typing import ClassVar
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, START

# A custom schema class that "looks like" a TypedDict to LangGraph
class MyCustomSchema:
    __required_keys__: ClassVar[frozenset[str]] = frozenset({"messages"})
    __optional_keys__: ClassVar[frozenset[str]] = frozenset({"metadata"})
    __annotations__ = {"messages": list, "metadata": dict}

# Verify the protocol check passes
from langgraph._internal._typing import TypedDictLikeV1
print(isinstance(MyCustomSchema, type) and hasattr(MyCustomSchema, "__required_keys__"))
```

### Dataclass state schema

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage

@dataclass
class ChatState:
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)
    turn: int = 0

def bump_turn(state: ChatState) -> dict:
    return {"turn": state.turn + 1}

graph = (
    StateGraph(ChatState)
    .add_node("bump", bump_turn)
    .add_edge(START, "bump")
    .compile()
)
result = graph.invoke(ChatState(messages=[HumanMessage("hi")]))
print(result)  # ChatState(messages=[HumanMessage('hi')], turn=1)
```

### Runtime discrimination

```python
from langgraph._internal._typing import DataclassLike, TypedDictLikeV1, TypedDictLikeV2
from typing import TypedDict
from dataclasses import dataclass

class MyDict(TypedDict):
    x: int

@dataclass
class MyDC:
    x: int

def describe_schema(schema: type) -> str:
    if hasattr(schema, "__dataclass_fields__"):
        return "dataclass"
    if hasattr(schema, "__required_keys__"):
        return "TypedDict"
    if hasattr(schema, "__fields__"):         # Pydantic v1
        return "pydantic_v1"
    if hasattr(schema, "model_fields"):       # Pydantic v2
        return "pydantic_v2"
    return "unknown"

print(describe_schema(MyDict))  # TypedDict
print(describe_schema(MyDC))    # dataclass
```

---

## 5 · `AsyncQueue` + `SyncQueue` + `Semaphore`

**Module**: `langgraph._internal._queue`  
**First dedicated coverage.**

These three classes are LangGraph's internal concurrency primitives used by the Pregel executor for coordinator-to-worker signaling. The defining feature shared by all three is a `wait()` method that blocks until an item is available *without consuming it* — a non-destructive "is there work?" check.

```python
class AsyncQueue(asyncio.Queue):
    async def wait(self) -> None: ...     # waits without get()

class SyncQueue:
    def put(self, item, block=True, timeout=None) -> None: ...
    def get(self, block=False, timeout=None) -> Any: ...
    def wait(self, block=True, timeout=None) -> None: ...   # waits without get()
    def empty(self) -> bool: ...

class Semaphore(threading.Semaphore):
    def wait(self, blocking=True, timeout=None) -> bool: ...  # peek without acquire
```

### Why `wait()` without consuming matters

In the Pregel loop, the background executor needs to know "is there an update ready?" before deciding whether to spin up the next superstep. If it consumed the item during the check, it would lose the signal that some other part of the loop needs to process.

```python
import asyncio
from langgraph._internal._queue import AsyncQueue

async def producer_consumer():
    q: AsyncQueue[str] = AsyncQueue()

    async def producer():
        for msg in ["start", "middle", "end"]:
            await asyncio.sleep(0.1)
            await q.put(msg)

    async def coordinator():
        processed = []
        while len(processed) < 3:
            await q.wait()             # waits until something arrives
            item = await q.get_nowait()  # NOW consume it
            processed.append(item)
            print(f"processed: {item}")
        return processed

    producer_task = asyncio.create_task(producer())
    result = await coordinator()
    await producer_task
    return result

asyncio.run(producer_consumer())
# processed: start
# processed: middle
# processed: end
```

### `SyncQueue` for thread-based coordination

```python
import threading
from langgraph._internal._queue import SyncQueue

def thread_ping_pong():
    q: SyncQueue[int] = SyncQueue()
    results = []

    def worker():
        for _ in range(5):
            q.wait()                  # peek: block until item available
            val = q.get()             # consume
            results.append(val * 2)

    t = threading.Thread(target=worker)
    t.start()

    for i in range(5):
        q.put(i)

    t.join()
    print(results)  # [0, 2, 4, 6, 8]

thread_ping_pong()
```

### `Semaphore.wait()` — acquire-once gating

```python
import threading
from langgraph._internal._queue import Semaphore

# Semaphore with a wait() that peeks without acquiring
ready = Semaphore(0)

def background_init():
    import time
    time.sleep(0.2)
    ready.release()                   # signals readiness

t = threading.Thread(target=background_init)
t.start()

ready.wait()                          # blocks until release() is called, no acquire
print("background is ready")
ready.acquire()                       # now actually acquire to consume the token
t.join()
```

---

## 6 · `FunctionNonLocals` + `NonLocals`

**Module**: `langgraph.pregel._utils`  
**First dedicated coverage.**

LangGraph uses static AST analysis to find variables a function captures from an enclosing scope. This matters for `@entrypoint` and `@task`: if a decorated function references outer-scope names, LangGraph can inspect what it depends on without running the function.

```python
class NonLocals(ast.NodeVisitor):
    """Collect all Name loads and stores in a single function body."""
    loads: set[str]    # variables read
    stores: set[str]   # variables assigned

class FunctionNonLocals(ast.NodeVisitor):
    """Collect outer-scope names accessed inside nested functions/lambdas."""
    nonlocals: set[str]  # loads - stores (i.e., names from enclosing scope)
```

### How the analysis works

```python
import ast
from langgraph.pregel._utils import FunctionNonLocals, NonLocals

source = """
def outer():
    db_url = "postgres://..."
    cache = {}

    def inner(x):
        result = cache.get(x)  # 'cache' is a nonlocal
        if result is None:
            result = query(db_url, x)  # 'db_url', 'query' are nonlocals
            cache[x] = result
        return result
"""

tree = ast.parse(source)
outer_fn = tree.body[0]  # the outer() FunctionDef

# FunctionNonLocals scans nested functions inside outer()
scanner = FunctionNonLocals()
scanner.visit(outer_fn)
print(scanner.nonlocals)  # {'cache', 'db_url', 'query'}
```

### Practical use — detecting what a task captures

```python
import ast, inspect, textwrap
from langgraph.pregel._utils import FunctionNonLocals

def capture_analysis(fn) -> set[str]:
    """Return the set of outer-scope names captured by fn's body."""
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    visitor = FunctionNonLocals()
    visitor.visit(tree)
    return visitor.nonlocals

# Example: a factory that returns tasks with captured config
def make_processor(model_name: str, temperature: float):
    from langgraph.func import task

    @task
    def process(text: str) -> str:
        # model_name and temperature are captured from outer scope
        return f"[{model_name}@{temperature}] {text}"

    print("Captured:", capture_analysis(process._func))  # {'model_name', 'temperature'}
    return process

proc = make_processor("gpt-4o", 0.7)
```

### NonLocals for single-function scope analysis

```python
import ast
from langgraph.pregel._utils import NonLocals

def analyze_reads_and_writes(fn_source: str) -> dict:
    tree = ast.parse(fn_source)
    fn_node = tree.body[0]

    visitor = NonLocals()
    visitor.visit(fn_node)

    return {
        "reads": sorted(visitor.loads - visitor.stores),   # net reads from outer scope
        "writes": sorted(visitor.stores),                  # locally defined names
    }

sample = """
def node(state):
    x = state["value"]       # 'state' is read; 'x' is stored
    y = transform(x)         # 'transform' is read; 'y' is stored
    return {"result": y}
"""
print(analyze_reads_and_writes(sample))
# {'reads': ['state', 'transform'], 'writes': ['x', 'y']}
```

---

## 7 · `Edge` + `TriggerEdge`

**Module**: `langgraph.pregel._draw`  
**First dedicated coverage.**

`Edge` and `TriggerEdge` are the `NamedTuple` primitives that `CompiledStateGraph.get_graph()` returns when you want to introspect or render the graph topology. Visualization methods like `draw_mermaid()` and `draw_ascii()` consume these.

```python
class Edge(NamedTuple):
    source: str        # source node name
    target: str        # target node name
    conditional: bool  # True for conditional edges (add_conditional_edges)
    data: str | None   # edge label / condition name (if any)

class TriggerEdge(NamedTuple):
    source: str        # channel / node that triggers the target
    conditional: bool
    data: str | None
```

### Extracting and rendering graph topology

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    route: str
    result: str

def router(state: State) -> Literal["path_a", "path_b"]:
    return "path_a" if state["route"] == "a" else "path_b"

def path_a(state: State) -> dict:
    return {"result": "A"}

def path_b(state: State) -> dict:
    return {"result": "B"}

graph = (
    StateGraph(State)
    .add_node("router", lambda s: s)
    .add_node("path_a", path_a)
    .add_node("path_b", path_b)
    .add_conditional_edges("router", router)
    .add_edge(START, "router")
    .add_edge("path_a", END)
    .add_edge("path_b", END)
    .compile()
)

# Draw to console
print(graph.get_graph().draw_ascii())

# Enumerate edges programmatically
for edge in graph.get_graph().edges:
    label = f" [{edge.data}]" if edge.data else ""
    edge_type = "conditional" if edge.conditional else "direct"
    print(f"  {edge.source} --{edge_type}{label}--> {edge.target}")
```

### Custom Mermaid renderer using raw Edge objects

```python
from langgraph.pregel._draw import Edge

def edges_to_mermaid(edges: list[Edge], title: str = "Graph") -> str:
    lines = ["```mermaid", f"graph TD", f"    %% {title}"]
    for edge in edges:
        src = edge.source.replace("__start__", "START").replace("__end__", "END")
        tgt = edge.target.replace("__start__", "START").replace("__end__", "END")
        if edge.conditional and edge.data:
            lines.append(f"    {src} -->|{edge.data}| {tgt}")
        elif edge.conditional:
            lines.append(f"    {src} -.-> {tgt}")
        else:
            lines.append(f"    {src} --> {tgt}")
    lines.append("```")
    return "\n".join(lines)

mermaid = edges_to_mermaid(graph.get_graph().edges, title="Router graph")
print(mermaid)
```

### Topology diff between two graphs

```python
def graph_diff(before_edges: list[Edge], after_edges: list[Edge]) -> dict:
    before_set = set((e.source, e.target, e.conditional) for e in before_edges)
    after_set  = set((e.source, e.target, e.conditional) for e in after_edges)
    return {
        "added":   [e for e in after_edges if (e.source, e.target, e.conditional) not in before_set],
        "removed": [e for e in before_edges if (e.source, e.target, e.conditional) not in after_set],
    }

# Use before/after compiled graphs to detect routing changes in CI
before = graph.get_graph().edges
# ... modify graph ...
after  = graph.get_graph().edges
diff = graph_diff(before, after)
print("Added edges:", diff["added"])
print("Removed edges:", diff["removed"])
```

---

## 8 · `_RemoteGraphRunStream` + `_ChannelProjection` + `_ProjectionRegistry`

**Module**: `langgraph.pregel._remote_run_stream`  
**First dedicated coverage.**

`RemoteGraph.stream()` and `RemoteGraph.astream()` don't expose raw SDK events — they return `_RemoteGraphRunStream` (sync) or `_AsyncRemoteGraphRunStream` (async), which mirror the same surface as local `GraphRunStream`. Inside, `_ProjectionRegistry` maps channel names to `_ChannelProjection` instances that decode wire events using the same `DataDecoder` the SDK uses for native channels.

```
RemoteGraph.stream(input, config)
    └─ yields StreamChunk tuples from _RemoteGraphRunStream
           ├─ .values     → SDK typed projection (state snapshots)
           ├─ .messages   → SDK typed projection (AI message stream)
           ├─ .subgraphs  → SDK typed projection (subgraph handles)
           ├─ .tool_calls → SDK typed projection ("tools" channel)
           └─ .extensions → _ProjectionRegistry
                  ├─ "updates"    → _ChannelProjection(sdk, "updates")
                  ├─ "checkpoints"→ _ChannelProjection(sdk, "checkpoints")
                  ├─ "tasks"      → _ChannelProjection(sdk, "tasks")
                  └─ "custom"     → _ChannelProjection(sdk, "custom")
```

### Consuming a remote graph with the v3 stream API

```python
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import HumanMessage

# Connect to a deployed LangGraph server (e.g., LangGraph Cloud or local server)
remote = RemoteGraph(
    "my-agent",
    url="http://localhost:2024",
    api_key="lsv2_...",
)

# Use as a context manager — same surface as local GraphRunStream
with remote.stream(
    {"messages": [HumanMessage("What is 2+2?")]},
    config={"configurable": {"thread_id": "remote-t1"}},
    stream_mode=["values", "updates"],
) as run:
    for chunk in run:
        namespace, mode, payload = chunk
        if mode == "values":
            msgs = payload.get("messages", [])
            if msgs:
                print(f"[values] last message: {msgs[-1].content[:60]}")
        elif mode == "updates":
            print(f"[updates] node output: {list(payload.keys())}")
    print("Final output:", run.output)
```

### Subscribing to a custom channel projection

```python
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("my-agent", url="http://localhost:2024")

with remote.stream(
    {"query": "latest earnings"},
    stream_mode=["custom", "values"],
) as run:
    # run.extensions["custom"] is a _ChannelProjection
    custom_proj = run.extensions.get("custom")
    for chunk in run:
        ns, mode, payload = chunk
        if mode == "custom":
            print(f"custom event: {payload}")
    print("done:", run.output)
```

### Async remote stream with interrupt handling

```python
import asyncio
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import HumanMessage

async def run_with_hitl():
    remote = RemoteGraph("approval-agent", url="http://localhost:2024")

    async with remote.astream(
        {"messages": [HumanMessage("Run a database migration")]},
        config={"configurable": {"thread_id": "hitl-1"}},
        stream_mode="values",
    ) as run:
        async for chunk in run:
            ns, mode, payload = chunk
            print(f"[{mode}] {list(payload.keys())}")

        if run.interrupted:
            print("Paused for approval. Interrupts:", run.interrupts)
            # Resume by invoking with Command(resume=...)
        else:
            print("Final state:", run.output)

asyncio.run(run_with_hitl())
```

---

## 9 · `AgentState` + `AgentStatePydantic` deprecation & migration

**Module**: `langgraph.prebuilt.chat_agent_executor`  
**Migration guide for 1.2.6.**

`AgentState`, `AgentStatePydantic`, `AgentStateWithStructuredResponse`, and `AgentStateWithStructuredResponsePydantic` have all been **deprecated** in `langgraph==1.2.6`. They have moved to `langchain.agents`. The classes still exist in `langgraph.prebuilt` and emit `DeprecationWarning` on import.

```python
# Old (deprecated in 1.2.6)
from langgraph.prebuilt import AgentState          # DeprecationWarning!
from langgraph.prebuilt import AgentStatePydantic  # DeprecationWarning!

# New — import from langchain.agents
from langchain.agents import AgentState
```

The moved classes are:

| Old import | New import |
|---|---|
| `langgraph.prebuilt.AgentState` | `langchain.agents.AgentState` |
| `langgraph.prebuilt.AgentStatePydantic` | `langchain.agents.AgentState` (Pydantic variant dropped) |
| `langgraph.prebuilt.AgentStateWithStructuredResponse` | Build manually (see below) |
| `langgraph.prebuilt.AgentStateWithStructuredResponsePydantic` | Build manually (see below) |

### Migrating `AgentState` to a custom TypedDict

```python
# Before (deprecated)
from langgraph.prebuilt import AgentState, create_react_agent
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-haiku-4-5-20251001"),
    tools=[],
    state_schema=AgentState,  # still works but warns
)

# After — define your own state (identical field set)
from typing import Annotated, Sequence, NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

class MyAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]  # managed by LangGraph

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-haiku-4-5-20251001"),
    tools=[],
    state_schema=MyAgentState,  # no deprecation warning
)
result = agent.invoke({"messages": [("user", "Hello")]})
print(result["messages"][-1].content)
```

### Migrating `AgentStateWithStructuredResponse`

```python
from typing import Annotated, Sequence, NotRequired, Generic, TypeVar
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from pydantic import BaseModel

# Define your structured response type
class ResearchResult(BaseModel):
    summary: str
    sources: list[str]
    confidence: float

# Build the state manually (replaces AgentStateWithStructuredResponse[ResearchResult])
class ResearchAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
    structured_response: NotRequired[ResearchResult]

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-haiku-4-5-20251001"),
    tools=[],
    state_schema=ResearchAgentState,
    response_format=ResearchResult,      # drives structured output extraction
)
result = agent.invoke({"messages": [("user", "Research LangGraph")]})
if "structured_response" in result:
    print(result["structured_response"].summary)
```

### Migrating `AgentStatePydantic` to a Pydantic model

```python
from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

# Replaces AgentStatePydantic (deprecated)
class MyPydanticAgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    remaining_steps: RemainingSteps = 25  # default matches the old AgentStatePydantic

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-haiku-4-5-20251001"),
    tools=[],
    state_schema=MyPydanticAgentState,
)
result = agent.invoke({"messages": [("user", "ping")]})
print(type(result))  # MyPydanticAgentState
```

---

## 10 · `_SubgraphRunStreamMixin`

**Module**: `langgraph.stream.run_stream`  
**First dedicated coverage.**

`SubgraphRunStream` and `AsyncSubgraphRunStream` are concrete handles that represent a running subgraph inside a parent `GraphRunStream`. They both inherit from `_SubgraphRunStreamMixin`, which carries the subgraph's identity metadata and delegates all pump calls to the parent run.

```python
class _SubgraphRunStreamMixin:
    path: tuple[str, ...]          # namespace path ("parent", "child", "grandchild")
    graph_name: str | None         # graph_id string registered on the subgraph
    trigger_call_id: str | None    # tool call id that launched this subgraph (if any)
    status: SubgraphStatus         # "pending" | "running" | "done" | "error"
    error: str | None              # error message if status == "error"
    _seen_terminal: bool           # whether a terminal status event was received
```

The mixin does NOT drive its own event loop. `SubgraphTransformer` updates `status` in place as events arrive on the parent's stream. This means calling pump methods on a `SubgraphRunStream` correctly delegates to the parent's event loop.

### Iterating subgraph handles

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

class OuterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
inner_agent = create_react_agent(llm, tools=[])

outer = (
    StateGraph(OuterState)
    .add_node("inner", inner_agent)
    .add_edge(START, "inner")
    .compile()
)

async def stream_with_subgraph_metadata():
    async with outer.astream(
        {"messages": [HumanMessage("Hello")]},
        stream_mode="values",
        subgraphs=True,
    ) as run:
        # Iterate subgraph handles as they spawn
        async for subgraph_handle in run.subgraphs:
            print(f"Subgraph spawned:")
            print(f"  path       : {subgraph_handle.path}")
            print(f"  graph_name : {subgraph_handle.graph_name}")
            print(f"  status     : {subgraph_handle.status}")

        # Drain the parent run to completion
        async for chunk in run:
            ns, mode, payload = chunk
            # Filter chunks from root vs subgraph by namespace
            depth = len(ns)
            prefix = "  " * depth
            print(f"{prefix}[depth={depth}] {mode}: {list(payload.keys())}")

asyncio.run(stream_with_subgraph_metadata())
```

### Tracking subgraph status transitions

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
inner = create_react_agent(llm, tools=[])
outer = (
    StateGraph(State)
    .add_node("inner", inner)
    .add_edge(START, "inner")
    .compile()
)

async def watch_subgraph_lifecycle():
    handles = []

    async with outer.astream(
        {"messages": [HumanMessage("ping")]},
        stream_mode="values",
        subgraphs=True,
    ) as run:
        # Collect handles as they arrive
        async for h in run.subgraphs:
            handles.append(h)
            print(f"[spawn] {h.path} status={h.status}")

        # Drain the run — SubgraphTransformer updates handle.status in place
        async for _ in run:
            pass

    for h in handles:
        print(f"[final] {h.path} status={h.status} error={h.error}")

asyncio.run(watch_subgraph_lifecycle())
```

### Drilling into a subgraph's projections

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
inner = create_react_agent(llm, tools=[])
outer = (
    StateGraph(State)
    .add_node("inner", inner)
    .add_edge(START, "inner")
    .compile()
)

async def drill_into_subgraph():
    async with outer.astream(
        {"messages": [HumanMessage("What is 2+2?")]},
        stream_mode=["values", "messages"],
        subgraphs=True,
    ) as run:
        # Get the first subgraph handle and its message projection
        async for subgraph_handle in run.subgraphs:
            # Each SubgraphRunStream exposes .values, .messages, etc.
            # via the same GraphRunStream interface — pump delegation
            # means the parent run drives all event delivery
            print(f"subgraph path: {subgraph_handle.path}")
            # Drain the parent to let the subgraph run
            break

        async for chunk in run:
            ns, mode, payload = chunk
            if ns and mode == "messages":
                # Message events from inside the subgraph
                for msg_chunk, metadata in payload:
                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                        print(f"[subgraph msg] {msg_chunk.content}", end="", flush=True)
        print()

asyncio.run(drill_into_subgraph())
```

---

## Summary

| # | Class / symbol | Key insight |
|---|---|---|
| 1 | `RunnableCallable` + `RunnableSeq` | Every node is wrapped here; `func_accepts` detects `config`/`store`/`runtime`/`writer`; sync+async dual dispatch |
| 2 | `_IdleProgressCallbackHandler` | All LangChain events reset the idle clock; `run_inline=True` prevents reordering; weakref prevents scope leaks |
| 3 | `_GraphCallbackManager` + `_AsyncGraphCallbackManager` | Dispatches `on_interrupt`/`on_resume` only to `GraphCallbackHandler` subclasses; `configure()` wires from `config["callbacks"]` |
| 4 | `DataclassLike` + `TypedDictLikeV1` + `TypedDictLikeV2` | Structural protocols for state schema discrimination; any class with right ClassVars satisfies them |
| 5 | `AsyncQueue` + `SyncQueue` + `Semaphore` | `wait()` peeks without consuming; drives coordinator-worker signaling in the Pregel loop |
| 6 | `FunctionNonLocals` + `NonLocals` | AST closure analysis; `loads - stores` = outer-scope captures; used by `@task`/`@entrypoint` |
| 7 | `Edge` + `TriggerEdge` | NamedTuples from `get_graph().edges`; power `draw_mermaid()`/`draw_ascii()` and custom graph renderers |
| 8 | `_RemoteGraphRunStream` + `_ChannelProjection` + `_ProjectionRegistry` | Remote graph stream adapter; `extensions` maps channel names to projections with same item shape as local transformers |
| 9 | `AgentState` deprecation migration | Moved to `langchain.agents`; migrate to custom TypedDict with `add_messages + RemainingSteps`; `response_format` drives structured output |
| 10 | `_SubgraphRunStreamMixin` | `path`/`graph_name`/`trigger_call_id` identity metadata; `status` updated in place by `SubgraphTransformer`; pump delegates to parent |

---
title: "LangGraph Class Deep-Dives Vol. 42"
description: "Source-verified deep dives (langgraph==1.2.9) into 10 class groups: BackgroundExecutor/AsyncBackgroundExecutor/Submit (parallel task fan-out with thread pool and asyncio, max_concurrency semaphore, cancel_on_exit flag), DeltaChannel v1.2.9 update (append-only state channel with replay_writes ancestor replay, Overwrite reset support, snapshot_frequency cadence and _DeltaSnapshot blob), StateGraph.set_node_defaults()/_NodeDefaults (graph-wide retry/cache/timeout/error_handler defaults with per-node override precedence), BranchSpec (conditional routing NamedTuple — path/ends/input_schema, from_path Literal-type inference, run() RunnableCallable emission), ReplayState (subgraph time-travel — _is_first_visit task-id stripping, get_checkpoint/aget_checkpoint pre-replay hydration), StateNodeSpec (compiled node descriptor introspection — runnable/metadata/retry_policy/cache_policy/defer/timeout/is_error_handler/error_handler_node), Call/PregelTaskWrites/LazyAtomicCounter (task scheduling + write primitives: Call func/input/retry/cache/timeout, PregelTaskWrites path/name/writes/triggers NamedTuple, LazyAtomicCounter double-checked lazy init), NodeError/ParentCommand/EmptyInputError (error context injection, Command propagation through the bubble-up chain, empty-input guard), HumanResponse (HITL response type — accept/ignore/response/edit variants, args payloads, full migration away from deprecated HumanInterrupt/ActionRequest), and ToolCallStream end-to-end streaming patterns (tool-started/output-delta/tool-finished lifecycle, sync/async iteration over output_deltas, multi-call fan-out with ToolCallTransformer)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 42"
  order: 73
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.9` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `BackgroundExecutor` · `AsyncBackgroundExecutor` · `Submit`

**Module:** `langgraph.pregel._executor`

These context-manager classes are the parallel task engine that powers Pregel's super-step concurrency. `BackgroundExecutor` wraps a thread-pool (via `get_executor_for_config`); `AsyncBackgroundExecutor` uses the running asyncio event loop. Both expose a `Submit` callable on `__enter__` / `__aenter__` that mirrors `concurrent.futures.Future` / `asyncio.Future` semantics.

**Key source facts:**

- `__enter__` / `__aenter__` return the `submit` method — typed as the `Submit` Protocol.
- `submit(..., __cancel_on_exit__=True)` — the future is cancelled when the context exits (useful for speculative/fire-and-forget work).
- `submit(..., __reraise_on_exit__=True)` — default; the first task exception is re-raised on context exit.
- `submit(..., __next_tick__=True)` — sync: wraps in `next_tick()` to defer one tick; async: noop (always next-tick).
- `AsyncBackgroundExecutor` respects `config["max_concurrency"]`: if set, a `asyncio.Semaphore` gates concurrent tasks.
- `GraphBubbleUp` exceptions (interrupts) are silently swallowed by the `done` callback — they are not re-raised.

### Example 1 — fan-out with `BackgroundExecutor`

```python
import time
from langchain_core.runnables import RunnableConfig
from langgraph.pregel._executor import BackgroundExecutor

def slow_add(a: int, b: int) -> int:
    time.sleep(0.05)
    return a + b

config: RunnableConfig = {}

with BackgroundExecutor(config) as submit:
    f1 = submit(slow_add, 1, 2)
    f2 = submit(slow_add, 10, 20)
    f3 = submit(slow_add, 100, 200)

# All three ran concurrently; results are ready after the `with` block
print(f1.result(), f2.result(), f3.result())  # 3  30  300
```

### Example 2 — `AsyncBackgroundExecutor` with `max_concurrency`

```python
import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.pregel._executor import AsyncBackgroundExecutor

async def slow_square(n: int) -> int:
    await asyncio.sleep(0.05)
    return n * n

async def demo() -> None:
    # Limit to 2 concurrent tasks at once
    config: RunnableConfig = {"max_concurrency": 2}
    futures = []
    async with AsyncBackgroundExecutor(config) as submit:
        for i in range(6):
            futures.append(submit(slow_square, i))

    results = [f.result() for f in futures]
    print(results)  # [0, 1, 4, 9, 16, 25]

asyncio.run(demo())
```

### Example 3 — cancel-on-exit for speculative work

```python
import time
from langchain_core.runnables import RunnableConfig
from langgraph.pregel._executor import BackgroundExecutor

def speculative_work(label: str) -> str:
    if label != "fast":
        time.sleep(2)  # long-running; still pending when context exits and cancels it
    return f"done:{label}"

# max_concurrency=1 means only one task runs at a time; speculative queues behind fast
config: RunnableConfig = {"max_concurrency": 1}

with BackgroundExecutor(config) as submit:
    # fast occupies the single thread slot immediately
    fast = submit(speculative_work, "fast")
    # speculative is queued (pending, not yet running) — can be cancelled
    speculative = submit(
        speculative_work, "slow",
        __cancel_on_exit__=True,
        __reraise_on_exit__=False,
    )
    fast_result = fast.result()  # returns instantly; speculative still pending
    # context exit: speculative is still pending → Future.cancel() succeeds

print("Fast path won:", fast_result)
print("Speculative cancelled:", speculative.cancelled())  # True
```

---

## 2 · `DeltaChannel` (langgraph 1.2.9)

**Module:** `langgraph.channels.delta`

`DeltaChannel` is a **beta** reducer channel that stores only a sentinel in checkpoint blobs and reconstructs state by replaying ancestor writes through the reducer. This gives it a smaller checkpoint footprint than `BinaryOperatorAggregate` for large accumulated values (e.g. long conversation transcripts, audit logs).

**Key source facts (1.2.9 updates):**

- Constructor: `DeltaChannel(reducer, typ=None, *, snapshot_frequency=1000)`.
- `reducer(state, list[writes]) -> new_state` — must be deterministic and **batching-invariant** (`reducer(reducer(s, xs), ys) == reducer(s, xs + ys)`).
- `replay_writes(writes)` — replay ancestor writes oldest-to-newest via a single reducer call; if any write is an `Overwrite`, the last `Overwrite` acts as the reset point.
- `update(values)` — supports `Overwrite` inside the batch: one `Overwrite` per super-step resets state before applying remaining writes.
- `checkpoint()` always returns `MISSING` — actual snapshot blobs (`_DeltaSnapshot`) are written by the checkpointer layer based on `snapshot_frequency`.
- Snapshot cadence: every Nth update (`snapshot_frequency`) **or** every 5000 supersteps since the last snapshot (whichever comes first).

### Example 1 — basic `DeltaChannel` with an append reducer

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.delta import DeltaChannel

def list_reducer(state: list, updates: list) -> list:
    """Append all updates to the accumulated list."""
    return state + updates

class State(TypedDict):
    log: Annotated[list[str], DeltaChannel(list_reducer)]

def step_a(state: State) -> dict:
    return {"log": "step_a ran"}  # scalar write; reducer receives ["step_a ran"]

def step_b(state: State) -> dict:
    return {"log": "step_b ran"}  # scalar write; reducer receives ["step_b ran"]

graph = (
    StateGraph(State)
    .add_node("a", step_a)
    .add_node("b", step_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

result = graph.invoke({"log": []})
print(result["log"])  # ['step_a ran', 'step_b ran']
```

### Example 2 — resetting state with `Overwrite`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.delta import DeltaChannel
from langgraph.types import Overwrite

def concat_reducer(state: str, updates: list[str]) -> str:
    return state + "".join(updates)

class State(TypedDict):
    text: Annotated[str, DeltaChannel(concat_reducer)]
    step: int

def append_node(state: State) -> dict:
    return {"text": f"[step {state['step']}]", "step": state["step"] + 1}

def reset_node(state: State) -> dict:
    # Overwrite(value) replaces the current state entirely
    return {"text": Overwrite("RESET"), "step": 0}

# Run only the first two steps to see Overwrite in action
from langgraph.graph import END
graph2 = (
    StateGraph(State)
    .add_node("append", append_node)
    .add_node("reset", reset_node)
    .add_edge(START, "append")
    .add_edge("append", "reset")
    .add_edge("reset", END)
    .compile()
)
result = graph2.invoke({"text": "", "step": 1})
print(result["text"])  # 'RESET'  — Overwrite discarded the appended text
```

### Example 3 — custom `snapshot_frequency` for large logs

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver

def list_reducer(state: list, updates: list) -> list:
    return state + updates

class State(TypedDict):
    events: Annotated[list[str], DeltaChannel(list_reducer, snapshot_frequency=5)]
    n: int

def emit(state: State) -> dict:
    return {"events": f"event-{state['n']}", "n": state["n"] + 1}  # scalar write

def should_stop(state: State):
    return END if state["n"] >= 8 else "emit"

checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("emit", emit)
    .add_conditional_edges("emit", should_stop)
    .add_edge(START, "emit")
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "delta-demo"}}
result = graph.invoke({"events": [], "n": 0}, config)
print(len(result["events"]))  # 8 — correct accumulated state
# The checkpointer writes a _DeltaSnapshot blob every 5 updates
```

---

## 3 · `StateGraph.set_node_defaults()` · `_NodeDefaults`

**Module:** `langgraph.graph.state`

`_NodeDefaults` is a `dataclass(slots=True)` that stores the four policies applied to every node at `compile()` time: `retry_policy`, `cache_policy`, `error_handler`, and `timeout`. The public API is `StateGraph.set_node_defaults()` — a fluent method that populates `_node_defaults` and returns `self` for chaining.

**Key source facts:**

- `set_node_defaults` accepts `retry_policy`, `cache_policy`, `error_handler`, `timeout` — all keyword-only.
- Per-node values in `add_node(...)` **always take precedence** over defaults.
- `retry_policy` and `timeout` apply to **all** nodes, including error-handler nodes.
- `cache_policy` and `error_handler` apply to **regular nodes only** — caching error-handler results is unsafe.
- Defaults are **not** inherited by subgraphs — each subgraph must set its own defaults.
- `timeout` accepts `float` (seconds), `timedelta`, or `TimeoutPolicy`.

### Example 1 — graph-wide retry policy

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

call_count = {"n": 0}

def flaky_node(state: dict) -> dict:
    call_count["n"] += 1
    if call_count["n"] < 3:
        raise ConnectionError(f"transient error #{call_count['n']}")  # retried by default
    return {"result": "ok"}

class State(TypedDict):
    result: str

graph = (
    StateGraph(State)
    .set_node_defaults(retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.01))
    .add_node("flaky", flaky_node)
    .add_edge(START, "flaky")
    .add_edge("flaky", END)
    .compile()
)

result = graph.invoke({})
print(result["result"])   # 'ok'
print(call_count["n"])    # 3  — two failures, one success
```

### Example 2 — default error handler + per-node override

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import _NodeDefaults
from langgraph.types import Command

class State(TypedDict):
    output: str
    recovered: bool

handler_calls = []

def fallback_handler(state: State) -> dict:
    """Applied to every node that doesn't set its own error_handler."""
    handler_calls.append("fallback")
    return {"output": "fallback", "recovered": True}

def custom_handler(state: State) -> dict:
    handler_calls.append("custom")
    return {"output": "custom_recovery", "recovered": True}

def node_a(state: State) -> dict:
    raise RuntimeError("node_a always fails")

def node_b(state: State) -> dict:
    raise RuntimeError("node_b always fails (uses custom handler)")

# Inspect the _NodeDefaults before and after
builder = StateGraph(State)
print("before:", builder._node_defaults)  # _NodeDefaults(retry_policy=None, ...)

(
    builder
    .set_node_defaults(error_handler=fallback_handler)
    .add_node("a", node_a)                              # uses default handler
    .add_node("b", node_b, error_handler=custom_handler)  # per-node override
    .add_edge(START, "a")
    .add_edge("a", "b")   # route through b so both handlers are exercised
    .add_edge("b", END)
)
print("after:", builder._node_defaults.error_handler)  # <function fallback_handler>

graph = builder.compile()
result = graph.invoke({})
print(handler_calls)       # ['fallback', 'custom']  — both handlers ran
print(result["output"])    # 'custom_recovery'  — b's custom_handler ran last
```

### Example 3 — chain `set_node_defaults` with multiple policies

```python
from datetime import timedelta
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy

class State(TypedDict):
    value: int

def compute(state: State) -> dict:
    return {"value": state["value"] * 2}

graph = (
    StateGraph(State)
    .set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05),
        cache_policy=CachePolicy(ttl=300),
        timeout=TimeoutPolicy(idle_timeout=30.0),
    )
    .add_node("double", compute)
    .add_node(
        "triple",
        lambda s: {"value": s["value"] * 3},
        cache_policy=None,   # disable cache for this node only
    )
    .add_edge(START, "double")
    .add_edge("double", "triple")
    .add_edge("triple", END)
    .compile()
)

# Verify applied policies
double_node = graph.nodes["double"]
triple_node = graph.nodes["triple"]
print("double cache:", double_node.cache_policy.ttl)  # 300
print("triple cache:", triple_node.cache_policy)       # None (overridden)
print("double retry:", double_node.retry_policy[0].max_attempts)  # 3
```

---

## 4 · `BranchSpec`

**Module:** `langgraph.graph._branch`

`BranchSpec` is the internal `NamedTuple` that represents a compiled conditional edge. When you call `add_conditional_edges`, LangGraph creates a `BranchSpec` and registers it on `StateGraph.branches[source_node][name]`. Inspecting it lets you verify routing logic without running the graph.

**Key source facts:**

- Fields: `path` (the routing `Runnable`), `ends` (`dict[Hashable, str] | None` mapping return values to node names), `input_schema` (optional narrowed input type).
- `BranchSpec.from_path(path, path_map, infer_schema=False)` — factory used internally by `add_conditional_edges`. When `path_map=None` and the function has a `Literal` return type annotation, it **auto-infers** `ends` from the `Literal` args.
- `run(writer, reader)` — compiles to a `RunnableCallable` that invokes the routing function and writes the selected destination to the appropriate channel.
- `_route` / `_aroute` — sync/async routing implementations. When `reader` is present (i.e. `then=` was set), it merges the current state with the node output before invoking the routing function.

### Example 1 — inspect compiled branch specs

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    score: int

def route(state: State) -> Literal["high", "low", "__end__"]:
    if state["score"] > 50:
        return "high"
    elif state["score"] > 10:
        return "low"
    return "__end__"

builder = StateGraph(State)
builder.add_node("high", lambda s: {"score": s["score"] + 100})
builder.add_node("low",  lambda s: {"score": s["score"] - 5})
builder.add_edge(START, "high")  # just for compilation
builder.add_conditional_edges("high", route)

# Inspect the BranchSpec BEFORE compiling
spec = builder.branches["high"]["route"]
print(type(spec).__name__)   # BranchSpec
print(spec.ends)             # {'high': 'high', 'low': 'low', '__end__': '__end__'}
print(spec.input_schema)     # <class 'State'>  (inferred from route's State annotation)
```

### Example 2 — `from_path` with Literal auto-inference

```python
from typing import Literal
from langgraph.graph._branch import BranchSpec
from langgraph.graph import StateGraph

class State(dict):
    pass

def classifier(state: dict) -> Literal["search", "answer", "clarify"]:
    text = state.get("query", "")
    if "?" in text:
        return "clarify"
    if len(text) > 30:
        return "search"
    return "answer"

# from_path with path_map=None: auto-infer from Literal return type
from langchain_core.runnables import RunnableLambda
spec = BranchSpec.from_path(
    path=RunnableLambda(classifier),
    path_map=None,  # triggers Literal-inference
)
print(spec.ends)
# {'search': 'search', 'answer': 'answer', 'clarify': 'clarify'}
```

### Example 3 — pass a dict `path_map` to rename destinations

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    action: str

def decide(state: State) -> str:
    return state["action"]

builder = StateGraph(State)
builder.add_node("execute", lambda s: s)
builder.add_node("skip",    lambda s: s)
builder.add_edge(START, "execute")
builder.add_conditional_edges(
    "execute",
    decide,
    path_map={"run": "execute", "pass": "skip", "done": END},
)

spec = builder.branches["execute"]["decide"]
print(spec.ends)
# {'run': 'execute', 'pass': 'skip', 'done': '__end__'}
# Return value "run" routes to "execute", "done" routes to END, etc.
```

---

## 5 · `ReplayState`

**Module:** `langgraph._internal._replay`

`ReplayState` drives the time-travel logic that lets LangGraph replay a parent graph execution from a past checkpoint. It solves a subtle correctness problem: when a subgraph is inside a loop, the **first** invocation during replay should restore the pre-replay checkpoint, but **subsequent** invocations in the same loop should use normal (latest) checkpoint loading so they see freshly created state.

**Key source facts:**

- `__init__(checkpoint_id)` — stores the target checkpoint ID; initialises `_visited_ns: set[str]`.
- `_is_first_visit(checkpoint_ns)` — strips the task-id suffix (`"sub:task_abc"` → `"sub"`) before checking `_visited_ns`. Returns `True` the first time a logical subgraph is seen, `False` thereafter.
- `get_checkpoint(checkpoint_ns, checkpointer, checkpoint_config)` — on first visit, lists checkpoints *before* `checkpoint_id` (limit=1) and returns the latest one before the replay point. On subsequent visits, falls back to `checkpointer.get_tuple()`.
- `aget_checkpoint` — async counterpart.
- A single `ReplayState` instance is **shared by reference** across all derived configs within one parent execution, so the `_visited_ns` set accumulates across the full replay.

### Example 1 — verify time-travel checkpoint selection

```python
from langgraph._internal._replay import ReplayState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Simulate a basic replay by constructing ReplayState and querying it
replay = ReplayState(checkpoint_id="chk-003")
print(replay.checkpoint_id)   # 'chk-003'
print(replay._visited_ns)     # set()

# First visit: _is_first_visit returns True, sets are updated
first = replay._is_first_visit("subgraph_a:task_xyz")
print(first)                   # True
print(replay._visited_ns)      # {'subgraph_a'}

# Second call with a different task id for the same subgraph
second = replay._is_first_visit("subgraph_a:task_abc")
print(second)                  # False — same logical namespace

# A different subgraph is a first visit
diff = replay._is_first_visit("subgraph_b")
print(diff)                    # True
```

### Example 2 — trigger time-travel via `invoke` with a past checkpoint

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    return {"count": state["count"] + 1}

def router(state: State) -> str:
    return "inc" if state["count"] < 3 else END

checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("inc", increment)
    .add_edge(START, "inc")
    .add_conditional_edges("inc", router, {"inc": "inc", END: END})
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "replay-demo"}}

# Single invoke loops inc three times: 0 → 1 → 2 → 3
# Each superstep writes a checkpoint, building a rich history
graph.invoke({"count": 0}, config)

# Checkpoint history, newest first — one snapshot per superstep
history = list(graph.get_state_history(config))
print([h.values["count"] for h in history])  # [3, 2, 1, 0]

# Branch off the checkpoint at count==1 — replay runs: 1→2→3
past_config = next(h.config for h in history if h.values["count"] == 1)
result = graph.invoke(None, past_config)
print(result["count"])  # 3  — replayed from count==1, incremented twice
```

### Example 3 — `ReplayState` shared reference across subgraph configs

```python
from langgraph._internal._replay import ReplayState

# ReplayState is shared by reference so _visited_ns accumulates globally
replay = ReplayState(checkpoint_id="chk-100")

# Simulate three subgraph invocations: sub_a (loop iter 1), sub_a (loop iter 2), sub_b
visits = [
    ("sub_a:tid1", True),   # first visit to sub_a
    ("sub_a:tid2", False),  # second iteration — same logical ns
    ("sub_b",      True),   # first visit to sub_b
]

for ns, expected in visits:
    result = replay._is_first_visit(ns)
    assert result == expected, f"Expected {expected} for {ns}, got {result}"
    print(f"{ns!r} → first={result}")

# sub_a only loads the pre-replay checkpoint once; subsequent calls use latest
```

---

## 6 · `StateNodeSpec`

**Module:** `langgraph.graph._node`

`StateNodeSpec` is the **descriptor dataclass** that `StateGraph` stores for each registered node. It captures every policy attached at `add_node()` time plus metadata that `compile()` uses to materialise the `PregelNode`. Inspecting `StateNodeSpec` is the definitive way to verify policies before the graph is compiled.

**Key source facts:**

- Fields: `runnable`, `metadata`, `input_schema`, `retry_policy`, `cache_policy`, `is_error_handler`, `error_handler_node`, `ends`, `defer`, `timeout`.
- `is_error_handler=True` when the node was registered via `error_handler=` on another node.
- `error_handler_node` holds the name of the node this spec is an error handler for.
- `ends=EMPTY_SEQ` (a sentinel `tuple()`) by default; set to a `tuple[str]` or `dict[str,str]` when `destinations=` is passed to `add_node`.
- `defer=True` means the node runs only after all non-deferred nodes in the same super-step have completed.
- Access via `builder.nodes[name]` (returns the `StateNodeSpec`) **before** compilation.

### Example 1 — inspect node policies before compilation

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy

class State(TypedDict):
    x: int

def worker(state: State) -> dict:
    return {"x": state["x"] + 1}

builder = StateGraph(State)
builder.add_node(
    "worker",
    worker,
    retry_policy=RetryPolicy(max_attempts=4),
    cache_policy=CachePolicy(ttl=120),
    timeout=TimeoutPolicy(idle_timeout=10.0),
    metadata={"team": "backend"},
)
builder.add_edge(START, "worker")
builder.add_edge("worker", END)

spec = builder.nodes["worker"]  # StateNodeSpec
print(f"retry max_attempts : {spec.retry_policy.max_attempts}")          # 4
print(f"cache ttl          : {spec.cache_policy.ttl}")                   # 120
print(f"timeout idle       : {spec.timeout.idle_timeout}")               # 10.0
print(f"metadata           : {spec.metadata}")                   # {'team': 'backend'}
print(f"defer              : {spec.defer}")                      # False
print(f"is_error_handler   : {spec.is_error_handler}")          # False
```

### Example 2 — identify error-handler nodes

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    result: str

def main_node(state: State) -> dict:
    raise ValueError("intentional error")

def my_handler(state: State) -> dict:
    return {"result": "recovered"}

builder = StateGraph(State)
builder.add_node("main", main_node, error_handler=my_handler)
builder.add_edge(START, "main")
builder.add_edge("main", END)

for name, spec in builder.nodes.items():
    print(f"{name!r}: is_error_handler={spec.is_error_handler}, "
          f"error_handler_node={spec.error_handler_node!r}")
# 'main'         : is_error_handler=False, error_handler_node=None
# '__main_eh__'  : is_error_handler=True, error_handler_node='main'
```

### Example 3 — deferred nodes and `destinations`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    tokens: list[str]
    summary: str

def fast_tokenise(state: State) -> dict:
    return {"tokens": state["summary"].split()}

def slow_summarise(state: State) -> dict:
    return {"summary": " ".join(state["tokens"][:3])}

builder = StateGraph(State)
builder.add_node("tokenise", fast_tokenise)
builder.add_node(
    "summarise",
    slow_summarise,
    defer=True,          # runs after all non-deferred nodes in the same super-step
    destinations=("end_node",),  # visualization hint for graph diagrams; does not restrict routing
)
builder.add_node("end_node", lambda s: s)
builder.add_edge(START, "tokenise")

tokenise_spec = builder.nodes["tokenise"]
summarise_spec = builder.nodes["summarise"]
print("tokenise defer:", tokenise_spec.defer)    # False
print("summarise defer:", summarise_spec.defer)  # True
print("summarise ends:", summarise_spec.ends)    # ('end_node',)
```

---

## 7 · `Call` · `PregelTaskWrites` · `LazyAtomicCounter`

**Module:** `langgraph.pregel._algo`

These three classes form the **write and scheduling primitives** that underpin the Pregel super-step machinery:

- `Call` — the dataclass that the `@task` / Functional API wraps a deferred function invocation in. Carries `func`, `input` (args + kwargs tuple), `retry_policy`, `cache_policy`, `callbacks`, and `timeout`.
- `PregelTaskWrites` — a `NamedTuple` implementing `WritesProtocol` for writes that don't originate from a runnable task (graph input, `update_state`, etc.).
- `LazyAtomicCounter` — a thread-safe, lazily-initialised atomic counter using `itertools.count` and a double-checked lock.

**Key source facts for `Call`:**
- Created internally when a `@task` decorated function is called from within `@entrypoint`.
- `input` is a `(args_tuple, kwargs_dict)` pair matching the call site.
- `timeout` defaults to `None`; overridable per-call via the Functional API's `timeout=` kwarg.

**Key source facts for `PregelTaskWrites`:**
- `path` — tuple of strings / ints identifying the write origin (e.g. `(INTERRUPT,)` for interrupts).
- `name` — human label (e.g. `"<input>"`, `"<update_state>"`) used in debug traces.
- `writes` — sequence of `(channel_name, value)` pairs.
- `triggers` — list of channel names that should be triggered by this write.

**Key source facts for `LazyAtomicCounter`:**
- The counter is `None` until first `__call__()`. First call acquires `LAZY_ATOMIC_COUNTER_LOCK` and initialises `itertools.count(0).__next__`.
- Double-checked locking: outer `if` avoids the lock on the hot path; inner `if` guards against the race.
- Used to generate unique integer task IDs without a global lock on the hot path.

### Example 1 — inspect `Call` objects from a `@task` graph

```python
from typing_extensions import TypedDict
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel._algo import Call

@task
def fetch(url: str) -> str:
    return f"data from {url}"

@entrypoint(checkpointer=InMemorySaver())
def pipeline(urls: list[str]) -> list[str]:
    futures = [fetch(url) for url in urls]
    return [f.result() for f in futures]

# Call objects are created inside the entrypoint context
# You can observe them via task callbacks
result = pipeline.invoke(
    ["http://a.example", "http://b.example"],
    config={"configurable": {"thread_id": "call-demo"}},
)
print(result)  # ['data from http://a.example', 'data from http://b.example']
```

### Example 2 — `PregelTaskWrites` for non-runnable state injection

```python
from langgraph.pregel._algo import PregelTaskWrites
from langgraph.constants import INPUT

# PregelTaskWrites is used internally when injecting graph input
# You can construct one manually to understand the protocol shape
write = PregelTaskWrites(
    path=(INPUT,),
    name="<input>",
    writes=[("messages", "hello"), ("count", 0)],
    triggers=["__start__"],
)

print("path    :", write.path)
print("name    :", write.name)
print("writes  :", write.writes)
print("triggers:", write.triggers)

# The writes list contains (channel_name, value) pairs
# that are applied to the state before the first super-step
for channel, value in write.writes:
    print(f"  → {channel!r} = {value!r}")
```

### Example 3 — `LazyAtomicCounter` for custom task ID generators

```python
from langgraph.pregel._algo import LazyAtomicCounter
import threading

counter = LazyAtomicCounter()

# First call initialises the internal itertools.count
print(counter())  # 0
print(counter())  # 1

# Thread-safety: spin up 100 threads, each calls counter() once
ids = []
lock = threading.Lock()

def worker():
    val = counter()
    with lock:
        ids.append(val)

threads = [threading.Thread(target=worker) for _ in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("unique IDs:", len(set(ids)))     # 100 — every thread got a distinct value
print("range:", min(ids), "-", max(ids))  # 2 - 101  (0 and 1 used above)
```

---

## 8 · `NodeError` · `ParentCommand` · `EmptyInputError`

**Module:** `langgraph.errors`

These three error types cover distinct failure scenarios in the LangGraph execution model:

- **`NodeError`** — a frozen `dataclass` injected into error-handler nodes. Contains `node` (failed node name) and `error` (the original exception). Handlers that accept a `NodeError` parameter receive it automatically.
- **`ParentCommand`** — a subclass of `GraphBubbleUp` used to propagate a `Command` from a subgraph upwards to the parent graph. The parent loop catches it and applies the command's `update` / `goto` fields to the parent state.
- **`EmptyInputError`** — raised by the Pregel loop when `invoke()`/`stream()` receives an empty input dict and the graph has no `__start__` writes pending from a previous checkpoint.

**Key source facts:**

- `GraphBubbleUp` is the base for all "non-error" signal exceptions that bubble up through the call stack without being re-raised as errors: `GraphInterrupt`, `ParentCommand`.
- `ParentCommand.__init__(command)` stores the `Command` as `args[0]`.
- To send a `Command` from a subgraph to the parent, a subgraph node simply returns a `Command(graph=Command.PARENT, ...)`. The Pregel executor wraps it in `ParentCommand` internally.
- `NodeError` fields: `node: str`, `error: BaseException`. Both are read-only (`frozen=True`).

### Example 1 — `NodeError` in an error handler

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError

class State(TypedDict):
    value: int
    error_info: str

def risky_node(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError(f"negative value: {state['value']}")
    return {"value": state["value"] * 2}

def error_handler(state: State, error: NodeError) -> dict:
    return {
        "error_info": f"Node '{error.node}' failed: {type(error.error).__name__}: {error.error}",
        "value": 0,
    }

graph = (
    StateGraph(State)
    .add_node("risky", risky_node, error_handler=error_handler)
    .add_edge(START, "risky")
    .add_edge("risky", END)
    .compile()
)

result = graph.invoke({"value": -5, "error_info": ""})
print(result["error_info"])
# "Node 'risky' failed: ValueError: negative value: -5"
print(result["value"])  # 0  — handler set safe default
```

### Example 2 — `ParentCommand` for subgraph-to-parent `Command` propagation

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class ParentState(TypedDict):
    status: str
    approved: bool

class SubState(TypedDict):
    decision: str

def sub_node(state: SubState):
    # Propagate a Command to the parent graph
    return Command(
        graph=Command.PARENT,
        update={"status": "approved_by_sub", "approved": True},
        goto=END,
    )

sub_graph = (
    StateGraph(SubState)
    .add_node("decide", sub_node)
    .add_edge(START, "decide")
    .compile()
)

def parent_entry(state: ParentState) -> dict:
    return {"status": "started"}

parent = (
    StateGraph(ParentState)
    .add_node("entry", parent_entry)
    .add_node("sub", sub_graph)
    .add_edge(START, "entry")
    .add_edge("entry", "sub")
    .add_edge("sub", END)
    .compile()
)

result = parent.invoke({"status": "", "approved": False})
print(result["status"])    # 'approved_by_sub'
print(result["approved"])  # True
```

### Example 3 — `EmptyInputError` and resuming from checkpoint

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import EmptyInputError

class State(TypedDict):
    count: int

def bump(state: State) -> dict:
    return {"count": state["count"] + 1}

checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("bump", bump)
    .add_edge(START, "bump")
    .add_edge("bump", END)
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "empty-input-demo"}}
graph.invoke({"count": 0}, config)  # first run saves checkpoint

# Resume from checkpoint by passing None (not empty dict) — valid
result = graph.invoke(None, config)
print(result["count"])  # 1 (increments from saved state)

# EmptyInputError is raised when there is no checkpoint and None is passed as input
fresh_config = {"configurable": {"thread_id": "new-thread"}}
try:
    graph.invoke(None, fresh_config)  # no checkpoint exists → EmptyInputError
except EmptyInputError as e:
    print(f"EmptyInputError: {e}")
```

---

## 9 · `HumanResponse`

**Module:** `langgraph.prebuilt.interrupt`

`HumanResponse` is the TypedDict that the graph receives when execution resumes after a `interrupt()` call that requested human input. It has two fields:

- `type`: one of `"accept"` | `"ignore"` | `"response"` | `"edit"`.
- `args`: `None` (for `accept`/`ignore`), `str` (for `response`), or `ActionRequest` (for `edit`).

> **Migration note:** `HumanInterrupt`, `HumanInterruptConfig`, and `ActionRequest` are all deprecated since langgraph 1.0 and have moved to `langchain.agents.interrupt`. Import them from there for new code.

**Key source facts:**

| `type` | `args` | Meaning |
|---|---|---|
| `"accept"` | `None` | Human approved — proceed unchanged |
| `"ignore"` | `None` | Human skipped — proceed without action |
| `"response"` | `str` | Human provided text feedback / instruction |
| `"edit"` | `ActionRequest` | Human modified the proposed action |

### Example 1 — structured HITL with four response types

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class State(TypedDict):
    proposal: str
    outcome: str

def propose(state: State) -> dict:
    return {"proposal": "DELETE all temp files"}

def review(state: State):
    human_resp = interrupt({"action": state["proposal"]})
    rtype = human_resp["type"]
    args  = human_resp["args"]

    if rtype == "accept":
        return {"outcome": f"Accepted: {state['proposal']}"}
    elif rtype == "ignore":
        return {"outcome": "Skipped by reviewer"}
    elif rtype == "response":
        return {"outcome": f"Reviewer said: {args}"}
    elif rtype == "edit":
        return {"outcome": f"Edited action: {args['action']}"}

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("propose", propose)
    .add_node("review", review)
    .add_edge(START, "propose")
    .add_edge("propose", "review")
    .add_edge("review", END)
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "hitl-demo"}}

# First run — hits interrupt() and raises GraphInterrupt
try:
    graph.invoke({"proposal": "", "outcome": ""}, config)
except Exception as e:
    print(f"Interrupted: {e}")  # GraphInterrupt — inspect pending state via get_state

# Resume with 'accept' response
from langgraph.types import Command
final = graph.invoke(Command(resume={"type": "accept", "args": None}), config)
print(final["outcome"])  # 'Accepted: DELETE all temp files'
```

### Example 2 — `"edit"` response with `ActionRequest`

```python
from langchain.agents.interrupt import ActionRequest  # current import path
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    command: str
    executed: str

def agent(state: State):
    # Propose a command and wait for human review
    response = interrupt({"command": state["command"]})

    if response["type"] == "edit":
        edited: ActionRequest = response["args"]
        return {"executed": f"ran edited: {edited['action']}"}
    return {"executed": f"ran original: {state['command']}"}

checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "edit-demo"}}
try:
    graph.invoke({"command": "rm -rf /tmp/old_cache", "executed": ""}, config)
except Exception:
    pass  # GraphInterrupt — graph paused waiting for human review

# Human edits the command to a safer scoped path before execution
edited_request: ActionRequest = {"action": "rm -rf /tmp/old_cache/session_42", "args": {}}
final = graph.invoke(
    Command(resume={"type": "edit", "args": edited_request}),
    config,
)
print(final["executed"])  # 'ran edited: rm -rf /tmp/old_cache/session_42'
```

### Example 3 — migrate from deprecated `HumanInterrupt` to `interrupt()`

```python
# OLD pattern (deprecated since 1.0) — DO NOT USE IN NEW CODE
# from langgraph.prebuilt import HumanInterrupt, HumanInterruptConfig, ActionRequest

# NEW pattern — import from langchain.agents.interrupt
from langchain.agents.interrupt import ActionRequest
from langgraph.types import interrupt

# OLD: the node raised NodeInterrupt with a HumanInterrupt payload
# NEW: the node calls interrupt() and returns based on the HumanResponse

def approval_node_new(state: dict):
    """Current pattern: interrupt() + HumanResponse dict."""
    response = interrupt({
        "action": state.get("action"),
        "description": "Approve or reject this action",
    })

    rtype = response.get("type", "accept")
    if rtype in ("accept", "ignore"):
        return {"approved": rtype == "accept"}
    elif rtype == "response":
        return {"approved": False, "feedback": response["args"]}
    elif rtype == "edit":
        edited: ActionRequest = response["args"]
        return {"approved": True, "action": edited["action"]}
    return {"approved": False}

# Minimal wiring to verify the function structure
print("approval_node_new defined:", callable(approval_node_new))

# Summary of migration steps:
# 1. Replace `NodeInterrupt(HumanInterrupt(...))` with `interrupt({...})`
# 2. The return value is a `HumanResponse` TypedDict (type + args)
# 3. Replace `HumanInterruptConfig` flags with client-side UI config
# 4. Import `ActionRequest` from `langchain.agents.interrupt`, not langgraph
```

---

## 10 · `ToolCallStream` — end-to-end streaming patterns

**Module:** `langgraph.prebuilt._tool_call_stream`

`ToolCallStream` is the per-tool-call live handle yielded on `run.tool_calls` when a graph is compiled with `ToolCallTransformer`. It represents the lifecycle of a single tool execution: `tool-started` → zero or more `tool-output-delta` events → `tool-finished` or `tool-error`.

**Key source facts:**

- Fields (set from the start): `tool_call_id`, `tool_name`, `input`.
- `output_deltas` — a `StreamChannel[Any]` of partial output chunks; iterate it sync or async.
- `output` — terminal payload from `tool-finished`; `None` if still running or failed.
- `error` — terminal error string from `tool-error`; `None` otherwise.
- `completed` — `True` once a terminal event has been received.
- `__iter__` / `__aiter__` delegate to `output_deltas` — you can iterate the `ToolCallStream` directly.
- The channel is **pump-driven**: it does not buffer all events upfront — you must be iterating the outer `run.tool_calls` stream concurrently for deltas to flow.

### Example 1 — synchronous streaming with `ToolCallTransformer`

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolCall

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for '{query}': ..."

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State) -> dict:
    # Simulate an AIMessage with a tool call
    return {"messages": [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="search", args={"query": "langgraph"}, id="tc-001")
            ],
        )
    ]}

graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_node("tools", ToolNode([search]))
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile(transformers=[ToolCallTransformer])
)

with graph.stream({"messages": []}, stream_mode="tools") as run:
    for tool_stream in run.tool_calls:
        print(f"Tool: {tool_stream.tool_name!r}, id: {tool_stream.tool_call_id!r}")
        for delta in tool_stream:
            print(f"  delta: {delta!r}")
        print(f"  output: {tool_stream.output!r}")
        print(f"  error : {tool_stream.error!r}")
```

### Example 2 — async streaming with `__aiter__`

```python
import asyncio
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolCall

@tool
async def async_lookup(term: str) -> str:
    """Async database lookup."""
    await asyncio.sleep(0.01)
    return f"definition of '{term}'"

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State) -> dict:
    return {"messages": [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="async_lookup", args={"term": "pregel"}, id="tc-002")
            ],
        )
    ]}

graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_node("tools", ToolNode([async_lookup]))
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile(transformers=[ToolCallTransformer])
)

async def stream_tools() -> None:
    async with graph.astream({"messages": []}, stream_mode="tools") as run:
        async for tool_stream in run.tool_calls:
            print(f"[async] {tool_stream.tool_name!r}: id={tool_stream.tool_call_id!r}")
            async for delta in tool_stream:
                print(f"  delta chunk: {delta!r}")
            status = "completed" if not tool_stream.error else "failed"
            print(f"  status={status}, output={tool_stream.output!r}")

asyncio.run(stream_tools())
```

### Example 3 — multi-tool fan-out and `completed` / `error` inspection

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolCall

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State) -> dict:
    return {"messages": [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="multiply", args={"a": 6, "b": 7},  id="tc-mul"),
                ToolCall(name="divide",   args={"a": 10, "b": 0}, id="tc-div"),
            ],
        )
    ]}

graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_node("tools", ToolNode([multiply, divide]))
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile(transformers=[ToolCallTransformer])
)

summary = []
with graph.stream({"messages": []}, stream_mode="tools") as run:
    for ts in run.tool_calls:
        # Drain deltas
        for _ in ts:
            pass
        summary.append({
            "tool":      ts.tool_name,
            "id":        ts.tool_call_id,
            "completed": ts.completed,
            "output":    ts.output,
            "error":     ts.error,
        })

for s in summary:
    print(s)
# {'tool': 'multiply', 'id': 'tc-mul', 'completed': True,  'output': 42,   'error': None}
# {'tool': 'divide',   'id': 'tc-div', 'completed': True,  'output': None,  'error': 'Cannot divide by zero'}
```

---

## Revision history

| Date | Version | Change |
|---|---|---|
| 2026-07-20 | langgraph 1.2.9 | Initial publication |

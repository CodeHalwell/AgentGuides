---
title: "Class deep-dives Vol. 17 — Practical patterns & state lifecycle"
description: "Source-verified deep dives into RetryPolicy sequences, TimeoutPolicy + Runtime.heartbeat(), Overwrite, interrupt() multi-value, add_sequence(), update_state() / bulk_update_state(), get_stream_writer(), stream_mode='checkpoints' + CheckpointPayload, stream_mode='tasks' + TaskPayload/TaskResultPayload, and set_node_defaults() — with multiple runnable examples for each class."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 17"
  order: 48
---

# Class deep-dives Vol. 17 — Practical patterns & state lifecycle

Verified against **`langgraph==1.2.5`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `RetryPolicy` — sequence chaining | `langgraph.types` |
| 2 | `TimeoutPolicy` + `Runtime.heartbeat()` | `langgraph.types` · `langgraph.runtime` |
| 3 | `Overwrite` — bypass a reducer | `langgraph.types` |
| 4 | `interrupt()` — multi-value + selective resume | `langgraph.types` |
| 5 | `add_sequence()` — linear pipeline builder | `langgraph.graph.state` |
| 6 | `update_state()` / `bulk_update_state()` + `StateUpdate` | `langgraph.graph.state` · `langgraph.types` |
| 7 | `get_stream_writer()` / `StreamWriter` | `langgraph.config` · `langgraph.types` |
| 8 | `stream_mode="checkpoints"` + `CheckpointPayload` + `CheckpointTask` | `langgraph.types` |
| 9 | `stream_mode="tasks"` + `TaskPayload` + `TaskResultPayload` | `langgraph.types` |
| 10 | `set_node_defaults()` — graph-wide retry / cache / timeout | `langgraph.graph.state` |

---

## 1 · `RetryPolicy` — sequence chaining

**Module:** `langgraph.types`

`RetryPolicy` is a `NamedTuple` with exponential back-off. What isn't obvious from the name: you can pass a **list** of `RetryPolicy` objects to `add_node`. LangGraph tries them in order — the first policy whose `retry_on` predicate matches the raised exception wins.

### Source signature

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor:   float = 2.0
    max_interval:     float = 128.0
    max_attempts:     int   = 3
    jitter:           bool  = True
    retry_on: (
        type[Exception]
        | Sequence[type[Exception]]
        | Callable[[Exception], bool]
    ) = default_retry_on   # retries everything except "programmer errors"
```

### Example 1 — default single policy (transient API errors)

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
        raise ConnectionError(f"Transient failure on attempt {_call_count}")
    return {"result": "success", "attempts": 1}

builder = StateGraph(State)
builder.add_node(
    "api",
    flaky_api,
    retry_policy=RetryPolicy(
        initial_interval=0.05,  # fast for examples
        backoff_factor=2.0,
        max_attempts=5,
        jitter=False,
        retry_on=ConnectionError,
    ),
)
builder.add_edge(START, "api")
builder.add_edge("api", END)

graph = builder.compile()
result = graph.invoke({"attempts": 0, "result": ""})
print(result)
# {'attempts': 3, 'result': 'success'}
```

### Example 2 — sequence of policies (ordered exception matching)

The `retry_policy` parameter on `add_node` accepts `RetryPolicy | Sequence[RetryPolicy]`. LangGraph iterates the list **in order** and uses the **first** policy whose `retry_on` matches. This lets you give different backoff behaviour to different exception categories without custom callables.

```python
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

_seq: list[Exception] = [
    ConnectionError("network blip"),
    TimeoutError("downstream slow"),
    RuntimeError("fatal"),
]

def node(state: State) -> dict:
    if _seq:
        raise _seq.pop(0)
    return {"result": "done"}

# Policy ordering matters: connection errors → fast retries,
# timeout errors → slower retries, everything else → 1 attempt.
network_policy = RetryPolicy(
    initial_interval=0.01,
    max_attempts=5,
    jitter=False,
    retry_on=ConnectionError,
)
slow_policy = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=3.0,
    max_attempts=3,
    jitter=False,
    retry_on=TimeoutError,
)
fatal_policy = RetryPolicy(
    max_attempts=1,          # 1 means "no retry"
    retry_on=RuntimeError,
)

builder = StateGraph(State)
builder.add_node(
    "action",
    node,
    retry_policy=[network_policy, slow_policy, fatal_policy],
)
builder.add_edge(START, "action")
builder.add_edge("action", END)

graph = builder.compile()
try:
    result = graph.invoke({"result": ""})
    print(result)
except RuntimeError as e:
    print(f"Fatal error propagated as expected: {e}")
```

### Example 3 — callable `retry_on` with structured logging

When `retry_on` is a callable, it receives the exception and returns `True` to retry or `False` to propagate. This is the hook for metrics / structured logging of retry attempts.

```python
import logging
from langgraph.types import RetryPolicy

log = logging.getLogger(__name__)

def should_retry(exc: Exception) -> bool:
    """Retry on transient errors; log and give up on others."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        log.warning("Transient failure, will retry: %s", exc)
        return True
    log.error("Non-retryable error: %s", exc)
    return False

policy = RetryPolicy(
    initial_interval=0.2,
    max_attempts=4,
    retry_on=should_retry,
)
```

### Key behaviours from source

| Field | Default | Notes |
|-------|---------|-------|
| `initial_interval` | `0.5` | Seconds before first retry |
| `backoff_factor` | `2.0` | Multiplier per retry |
| `max_interval` | `128.0` | Upper cap on wait time |
| `max_attempts` | `3` | Total attempts, including the first |
| `jitter` | `True` | Adds random ±20 % to avoid thundering-herd |
| `retry_on` | `default_retry_on` | Excludes `ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `AttributeError`, `NameError` |

---

## 2 · `TimeoutPolicy` + `Runtime.heartbeat()`

**Modules:** `langgraph.types`, `langgraph.runtime`

`TimeoutPolicy` gives per-node timeout control with two independent clocks:

- **`run_timeout`** — hard wall-clock cap from start to finish; never refreshed.
- **`idle_timeout`** — maximum time without observable progress; refreshed by callbacks or explicit `runtime.heartbeat()`.

```python
@dataclass
class TimeoutPolicy:
    run_timeout:  float | timedelta | None = None
    idle_timeout: float | timedelta | None = None
    refresh_on:   Literal["auto", "heartbeat"] = "auto"
```

When `refresh_on="auto"` (default), any LangChain callback event (LLM token, tool call, etc.) resets the idle clock automatically. Use `refresh_on="heartbeat"` for nodes that do their own I/O and want explicit control.

Timeouts rely on **asyncio cancellation**, so they only fire on async graphs. CPU-bound synchronous work will block the loop and delay the signal.

### Example 1 — hard run_timeout on a slow LLM call

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy

class State(TypedDict):
    result: str

async def slow_llm_node(state: State) -> dict:
    await asyncio.sleep(30)  # simulates a slow model
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node(
    "slow",
    slow_llm_node,
    timeout=TimeoutPolicy(run_timeout=5.0),  # 5-second hard cap
)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)

graph = builder.compile()

async def run():
    from langgraph.errors import NodeTimeoutError
    try:
        await graph.ainvoke({"result": ""})
    except NodeTimeoutError as e:
        print(f"Timed out: {e}")
        print(f"  kind={e.kind!r}, elapsed={e.elapsed:.2f}s")

asyncio.run(run())
# Timed out: Node 'slow' exceeded its run timeout of 5.000s (elapsed: 5.00xs).
```

### Example 2 — idle_timeout with explicit heartbeats (`refresh_on="heartbeat"`)

When a node does raw asyncio I/O (no LangChain callbacks), set `refresh_on="heartbeat"` and call `runtime.heartbeat()` periodically to prove progress.

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import Runtime

class State(TypedDict):
    chunks: list[str]

async def stream_processor(state: State, runtime: Runtime) -> dict:
    """Simulates reading a long stream, heartbeating every chunk."""
    chunks = []
    for i in range(10):
        await asyncio.sleep(0.2)   # simulate network read
        chunks.append(f"chunk-{i}")
        runtime.heartbeat()        # reset the idle clock
    return {"chunks": chunks}

builder = StateGraph(State)
builder.add_node(
    "processor",
    stream_processor,
    timeout=TimeoutPolicy(
        idle_timeout=1.0,        # must heartbeat within 1 second
        run_timeout=30.0,        # overall cap
        refresh_on="heartbeat",  # only heartbeat() counts, not callbacks
    ),
)
builder.add_edge(START, "processor")
builder.add_edge("processor", END)

graph = builder.compile()

async def run():
    result = await graph.ainvoke({"chunks": []})
    print(result["chunks"][:3])  # ['chunk-0', 'chunk-1', 'chunk-2']

asyncio.run(run())
```

### Example 3 — combining RetryPolicy + TimeoutPolicy

`NodeTimeoutError` is **retryable** by default (it deliberately does not inherit from `OSError`/`TimeoutError`). Combine a `TimeoutPolicy` with a `RetryPolicy` to auto-retry timed-out attempts.

```python
from datetime import timedelta
from langgraph.types import RetryPolicy, TimeoutPolicy

# Retry on TimeoutError (built-in) AND NodeTimeoutError (LangGraph)
from langgraph.errors import NodeTimeoutError

timeout = TimeoutPolicy(run_timeout=timedelta(seconds=10))
retry   = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    retry_on=lambda e: isinstance(e, (TimeoutError, NodeTimeoutError)),
)
# Pass both to add_node:
# builder.add_node("my_node", fn, timeout=timeout, retry_policy=retry)
```

---

## 3 · `Overwrite` — bypass a reducer

**Module:** `langgraph.types`

By default every write to a `BinaryOperatorAggregate` channel goes through its reducer. `Overwrite(value=...)` wraps a value that bypasses the reducer and directly replaces the channel's contents.

```python
@dataclass(slots=True)
class Overwrite:
    value: Any
```

**Constraint:** at most one `Overwrite` per channel per super-step. If two concurrent nodes both return `Overwrite` for the same key, `InvalidUpdateError` is raised.

### Example 1 — clearing a message list mid-conversation

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class State(TypedDict):
    messages: Annotated[list[str], operator.add]

def add_three_messages(state: State) -> dict:
    return {"messages": ["a", "b", "c"]}

def reset_conversation(state: State) -> dict:
    # Overwrite bypasses operator.add: the list becomes ["fresh-start"]
    return {"messages": Overwrite(value=["fresh-start"])}

builder = StateGraph(State)
builder.add_node("add", add_three_messages)
builder.add_node("reset", reset_conversation)
builder.add_edge(START, "add")
builder.add_edge("add", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"messages": []})
print(result["messages"])
# ['fresh-start']   — not ['a', 'b', 'c', 'fresh-start']
```

### Example 2 — Overwrite in a conditional branch (summarisation)

A common pattern: after many conversation turns, a summarisation node compresses history. Use `Overwrite` to replace the full message list with the summary only.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    turn_count: Annotated[int, operator.add]

def chat(state: State) -> dict:
    return {"messages": [f"turn {state['turn_count']}"], "turn_count": 1}

def summarise(state: State) -> dict:
    summary = f"[Summary of {len(state['messages'])} messages]"
    return {"messages": Overwrite(value=[summary])}

def should_summarise(state: State) -> str:
    return "summarise" if state["turn_count"] >= 3 else "chat"

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_node("summarise", summarise)
builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", should_summarise, {"chat": "chat", "summarise": "summarise"})
builder.add_edge("summarise", END)

graph = builder.compile()
result = graph.invoke({"messages": [], "turn_count": 0})
print(result["messages"])  # ['[Summary of 3 messages]']
```

### What `Overwrite` cannot do

| Scenario | Behaviour |
|----------|-----------|
| Two concurrent nodes both `Overwrite` the same key | `InvalidUpdateError` — only one `Overwrite` per key per super-step |
| Use `Overwrite` on a `LastValue` channel | `Overwrite` is a no-op; `LastValue` always writes directly |
| Mix `Overwrite` and a normal update for the same key in one super-step | `InvalidUpdateError` |

---

## 4 · `interrupt()` — multi-value + selective resume

**Module:** `langgraph.types`

`interrupt(value)` pauses the current node and surfaces `value` to the caller. The graph re-executes the node from the top when resumed. If a node calls `interrupt()` **more than once**, each invocation gets its own `Interrupt` object with a unique `id`. Resume values are matched **by position** on simple resume, or **by ID** when using a mapping.

### Source

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    id: str    # unique per interrupt invocation
```

```python
def interrupt(value: Any) -> Any:
    """Pause the current node and surface value to the caller.
    Returns the resume value when the graph is restarted."""
    ...
```

### Example 1 — single interrupt (basic human-in-the-loop)

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    action: str
    approved: bool

def approval_node(state: State) -> dict:
    decision = interrupt({"question": f"Approve action '{state['action']}'?",
                          "options": ["yes", "no"]})
    return {"approved": decision == "yes"}

builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "thread-1"}}

# First invocation — graph pauses at the interrupt
result = graph.invoke({"action": "delete_db", "approved": False}, config)
print("Interrupted. Pending interrupts:", graph.get_state(config).interrupts)

# Resume with human answer
result = graph.invoke(Command(resume="yes"), config)
print("Approved:", result["approved"])  # True
```

### Example 2 — multiple interrupts in one node (multi-step approval)

When a node calls `interrupt()` multiple times, each gets a separate `id`. By default, resuming with a plain value fills them **in order**.

```python
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    action: str
    step1_ok: Optional[bool]
    step2_ok: Optional[bool]

def two_step_approval(state: State) -> dict:
    first  = interrupt("Step 1: Confirm intent to proceed?")
    second = interrupt("Step 2: Final confirmation — are you sure?")
    return {"step1_ok": first == "yes", "step2_ok": second == "yes"}

builder = StateGraph(State)
builder.add_node("approve", two_step_approval)
builder.add_edge(START, "approve")
builder.add_edge("approve", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "multi-1"}}

# Kick off — pauses at the FIRST interrupt
graph.invoke({"action": "deploy", "step1_ok": None, "step2_ok": None}, config)
snapshot = graph.get_state(config)
print("Interrupt 1:", snapshot.interrupts[0].value)

# Resume step 1 — node re-runs from top, skips first interrupt (already resolved),
# then pauses at the second
graph.invoke(Command(resume="yes"), config)
snapshot = graph.get_state(config)
print("Interrupt 2:", snapshot.interrupts[0].value)

# Resume step 2
result = graph.invoke(Command(resume="yes"), config)
print(result)  # {'action': 'deploy', 'step1_ok': True, 'step2_ok': True}
```

### Example 3 — selective resume by interrupt ID

For non-sequential approval (e.g. parallel review), use `Command(resume={id: value})` to target a specific interrupt.

```python
from langgraph.types import Command

snapshot = graph.get_state(config)
# snapshot.interrupts is a tuple of Interrupt objects
for intr in snapshot.interrupts:
    print(f"  id={intr.id!r}  value={intr.value!r}")

# Resume only the interrupt with the matching id:
target_id = snapshot.interrupts[0].id
result = graph.invoke(Command(resume={target_id: "yes"}), config)
```

### Resume mechanics — what the source guarantees

| Situation | Behaviour |
|-----------|-----------|
| `Command(resume=value)` (plain value) | Applied to the **next unresolved** interrupt in order |
| `Command(resume={id: value})` | Applied only to the interrupt with that `id` |
| `Command(resume={id1: v1, id2: v2})` | Multiple interrupts resolved at once |
| Node calls `interrupt()` — already resolved | Returns the resume value immediately, no re-pause |

---

## 5 · `add_sequence()` — linear pipeline builder

**Module:** `langgraph.graph.state`

`add_sequence(nodes)` is syntactic sugar: it calls `add_node` for each item and wires `add_edge(prev, next)` between consecutive nodes. The first node in the sequence does **not** get a `START` edge; the last does **not** get an `END` edge — add those yourself.

### Source (condensed)

```python
def add_sequence(
    self,
    nodes: Sequence[
        StateNode | tuple[str, StateNode]
    ],
) -> Self:
```

Returns `self` for method chaining. Raises `ValueError` if the sequence is empty or contains duplicate names.

### Example 1 — minimal ETL pipeline

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    raw:       str
    cleaned:   str
    validated: bool
    stored:    bool

def extract(state: State) -> dict:
    return {"raw": "  hello, world!  "}

def clean(state: State) -> dict:
    return {"cleaned": state["raw"].strip().lower()}

def validate(state: State) -> dict:
    return {"validated": len(state["cleaned"]) > 0}

def store(state: State) -> dict:
    print(f"Stored: {state['cleaned']!r}")
    return {"stored": True}

builder = StateGraph(State)
builder.add_edge(START, "extract")       # manual START hook
builder.add_sequence([extract, clean, validate, store])
builder.add_edge("store", END)           # manual END hook

graph = builder.compile()
result = graph.invoke({"raw": "", "cleaned": "", "validated": False, "stored": False})
print(result)
# {'raw': '  hello, world!  ', 'cleaned': 'hello, world!', 'validated': True, 'stored': True}
```

### Example 2 — named nodes (when names would collide or aren't inferrable)

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

increment = lambda s: {"x": s["x"] + 1}
double    = lambda s: {"x": s["x"] * 2}

builder = StateGraph(State)
builder.add_edge(START, "step1")
builder.add_sequence([
    ("step1", increment),
    ("step2", double),
    ("step3", increment),
])
builder.add_edge("step3", END)

graph = builder.compile()
print(graph.invoke({"x": 3})["x"])  # (3+1)*2+1 = 9
```

### Example 3 — `add_sequence()` vs manual wiring (side-by-side)

```python
# --- Manual (equivalent) ---
builder.add_node("a", fn_a)
builder.add_node("b", fn_b)
builder.add_node("c", fn_c)
builder.add_edge("a", "b")
builder.add_edge("b", "c")

# --- With add_sequence ---
builder.add_sequence([("a", fn_a), ("b", fn_b), ("c", fn_c)])
```

`add_sequence` simply reduces boilerplate; the compiled graph is identical.

---

## 6 · `update_state()` / `bulk_update_state()` + `StateUpdate`

**Module:** `langgraph.graph.state`, `langgraph.types`

These methods let you **inject state externally** — from outside a graph run — into a checkpointed thread. Common uses: seeding initial state, time-travel replay, patching a corrupted checkpoint, and testing.

```python
# StateUpdate is the underlying NamedTuple
class StateUpdate(NamedTuple):
    values:  dict[str, Any] | None
    as_node: str | None = None   # pretend the update came from this node
    task_id: str | None = None   # target a specific pending task
```

```python
# update_state is a thin wrapper around bulk_update_state
def update_state(
    self,
    config: RunnableConfig,
    values: dict[str, Any] | Any | None,
    as_node: str | None = None,
    task_id: str | None = None,
) -> RunnableConfig:
    return self.bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])
```

### Example 1 — basic state injection before resume

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    value:    int
    approved: bool

def human_check(state: State) -> dict:
    interrupt(f"Current value is {state['value']}. Approve?")
    return {"approved": True}

def compute(state: State) -> dict:
    return {"value": state["value"] * 10}

builder = StateGraph(State)
builder.add_edge(START, "check")
builder.add_sequence([("check", human_check), ("compute", compute)])
builder.add_edge("compute", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t1"}}

# Kick off — pauses at interrupt
graph.invoke({"value": 5, "approved": False}, config)

# External patch: change value before resuming
new_config = graph.update_state(
    config,
    values={"value": 42},   # override value
    as_node="check",         # pretend it came from the check node
)
print(graph.get_state(new_config).values["value"])  # 42

# Resume — compute will see value=42
result = graph.invoke(Command(resume="yes"), new_config)
print(result["value"])  # 420
```

### Example 2 — `bulk_update_state()` for multi-step replay

`bulk_update_state` accepts a list of **super-steps**, each itself a list of `StateUpdate` objects. This replays multiple checkpointed writes atomically.

```python
from langgraph.types import StateUpdate

# Two super-steps: first write 'x', then write 'y'
config = graph.bulk_update_state(
    config,
    supersteps=[
        [StateUpdate({"field_a": 10}, as_node="node_a")],
        [StateUpdate({"field_b": 20}, as_node="node_b")],
    ],
)
```

### Example 3 — time-travel: rewind and re-run from an earlier checkpoint

```python
history = list(graph.get_state_history(config))

# history[-1] is the oldest snapshot (the initial state)
earliest = history[-1]
print("Rewound to:", earliest.values)

# Re-invoke from that checkpoint
result = graph.invoke(None, earliest.config)
print("Re-run result:", result)
```

---

## 7 · `get_stream_writer()` / `StreamWriter`

**Module:** `langgraph.config`, `langgraph.types`

`get_stream_writer()` returns the `StreamWriter` for the currently executing node or task. `StreamWriter` is a callable — call it with any value to emit a `custom` stream event to the client. The client receives it under `stream_mode="custom"`.

```python
StreamWriter: TypeAlias = Callable[[Any], None]
```

### Example 1 — emit progress events from a long-running node

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    items: list[str]
    processed: int

def batch_processor(state: State) -> dict:
    writer = get_stream_writer()
    processed = 0
    for i, item in enumerate(state["items"]):
        time.sleep(0.01)   # simulate work
        processed += 1
        writer({"progress": processed, "total": len(state["items"]), "item": item})
    return {"processed": processed}

builder = StateGraph(State)
builder.add_node("process", batch_processor)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()

for event in graph.stream(
    {"items": ["apple", "banana", "cherry"], "processed": 0},
    stream_mode="custom",
):
    print(event)
# {'progress': 1, 'total': 3, 'item': 'apple'}
# {'progress': 2, 'total': 3, 'item': 'banana'}
# {'progress': 3, 'total': 3, 'item': 'cherry'}
```

### Example 2 — multiple stream modes simultaneously

Pass a list to `stream_mode` to receive both `updates` and `custom` events:

```python
for chunk in graph.stream(
    {"items": ["x", "y"], "processed": 0},
    stream_mode=["updates", "custom"],
):
    # chunk is a tuple: (mode_name, payload)
    mode, payload = chunk
    if mode == "custom":
        print(f"Progress: {payload['progress']}/{payload['total']}")
    elif mode == "updates":
        print(f"Node update: {payload}")
```

### Example 3 — inject writer via Runtime (alternative pattern)

If your node already injects `runtime: Runtime`, use `runtime.stream_writer` instead of `get_stream_writer()`:

```python
from langgraph.runtime import Runtime

def node_with_runtime(state: State, runtime: Runtime) -> dict:
    runtime.stream_writer({"status": "starting"})
    # ... work ...
    runtime.stream_writer({"status": "complete"})
    return {"processed": 1}
```

Both `get_stream_writer()` and `runtime.stream_writer` write to the same `custom` stream channel.

### Async usage

```python
import asyncio
from langgraph.config import get_stream_writer

async def async_node(state: State) -> dict:
    writer = get_stream_writer()
    for i in range(5):
        await asyncio.sleep(0.1)
        writer({"tick": i})
    return {}
```

> **Warning (Python < 3.11):** `get_stream_writer()` uses `contextvars` propagation.
> In async graphs on Python < 3.11, `asyncio.create_task()` does not propagate context,
> so the writer may not be available inside a spawned task. Use `runtime.stream_writer`
> instead, or upgrade to Python ≥ 3.11.

---

## 8 · `stream_mode="checkpoints"` + `CheckpointPayload` + `CheckpointTask`

**Module:** `langgraph.types`

`stream_mode="checkpoints"` emits a `CheckpointStreamPart` after **every checkpoint write**. This lets you observe the graph's persistence layer in real time — useful for audit trails, progress dashboards, and debugging multi-step graphs.

```python
class CheckpointStreamPart(TypedDict, Generic[StateT]):
    type: Literal["checkpoints"]
    ns:   tuple[str, ...]          # namespace path (empty for root graph)
    data: CheckpointPayload[StateT]

class CheckpointPayload(TypedDict, Generic[StateT]):
    config:        RunnableConfig | None      # this checkpoint's config
    metadata:      CheckpointMetadata         # step, source, writes
    values:        StateT                     # full state at this checkpoint
    next:          list[str]                  # nodes scheduled next
    parent_config: RunnableConfig | None      # previous checkpoint
    tasks:         list[CheckpointTask]       # associated task info

class CheckpointTask(TypedDict):
    id:         str
    name:       str
    error:      NotRequired[str]         # present if task failed
    result:     NotRequired[Any]         # present if task succeeded
    interrupts: NotRequired[list[dict]]  # present if interrupted / complete
    state:      StateSnapshot | RunnableConfig | None  # subgraph state
```

### Example 1 — watching checkpoint progression

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    step: Annotated[int, operator.add]

def inc(state: State) -> dict:
    return {"step": 1}

builder = StateGraph(State)
builder.add_sequence([("a", inc), ("b", inc), ("c", inc)])
builder.add_edge(START, "a")
builder.add_edge("c", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "chk-demo"}}

for part in graph.stream({"step": 0}, config, stream_mode="checkpoints"):
    cp = part["data"]
    print(f"step={cp['values']['step']:2d}  "
          f"next={cp['next']}  "
          f"metadata_step={cp['metadata'].get('step', '?')}")
```

Output resembles:

```
step= 0  next=['a']  metadata_step=0
step= 1  next=['b']  metadata_step=1
step= 2  next=['c']  metadata_step=2
step= 3  next=[]     metadata_step=3
```

### Example 2 — audit trail: recording every state snapshot

```python
import json
from datetime import datetime

audit_log = []

for part in graph.stream(
    {"step": 0},
    config,
    stream_mode="checkpoints",
):
    cp = part["data"]
    audit_log.append({
        "timestamp":    datetime.utcnow().isoformat(),
        "step":         cp["metadata"].get("step"),
        "source":       cp["metadata"].get("source"),
        "state_values": cp["values"],
        "next_nodes":   cp["next"],
        "task_count":   len(cp["tasks"]),
    })

print(json.dumps(audit_log, indent=2))
```

### Example 3 — combining checkpoints + updates

```python
for kind, payload in graph.stream(
    {"step": 0},
    config,
    stream_mode=["updates", "checkpoints"],
):
    if kind == "updates":
        print(f"  [update] {payload}")
    elif kind == "checkpoints":
        print(f"  [checkpoint] step={payload['data']['values']['step']}")
```

### Reading `CheckpointTask` fields

```python
for part in graph.stream({"step": 0}, config, stream_mode="checkpoints"):
    for task in part["data"]["tasks"]:
        print(f"  task id={task['id'][:8]}  name={task['name']}", end="")
        if "error" in task:
            print(f"  ERROR: {task['error']}", end="")
        if "result" in task:
            print(f"  result={task['result']}", end="")
        print()
```

---

## 9 · `stream_mode="tasks"` + `TaskPayload` + `TaskResultPayload`

**Module:** `langgraph.types`

`stream_mode="tasks"` emits two event kinds for each node execution:

- **task-start:** `TaskPayload` — fired when a task is scheduled
- **task-result:** `TaskResultPayload` — fired when a task completes (with result, error, or interrupt list)

```python
class TaskPayload(TypedDict):
    id:       str            # unique task ID
    name:     str            # node name
    input:    Any            # input state passed to the node
    triggers: list[str]      # which channel writes triggered this task
    metadata: NotRequired[dict[str, Any]]  # e.g. langgraph_node, lc_agent_name

class TaskResultPayload(TypedDict):
    id:         str
    name:       str
    error:      str | None          # None if successful
    interrupts: list[dict]          # populated if node called interrupt()
    result:     dict[str, Any]      # channel → written value
```

Each `TasksStreamPart` looks like:

```python
class TasksStreamPart(TypedDict):
    type: Literal["tasks"]
    ns:   tuple[str, ...]
    data: TaskPayload | TaskResultPayload
```

You can distinguish start from result by checking for the `"input"` key (only in `TaskPayload`) or the `"result"` key (only in `TaskResultPayload`).

### Example 1 — logging task lifecycle

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def double(state: State) -> dict:
    return {"value": state["value"] * 2}

def add_ten(state: State) -> dict:
    return {"value": state["value"] + 10}

builder = StateGraph(State)
builder.add_sequence([("double", double), ("add_ten", add_ten)])
builder.add_edge(START, "double")
builder.add_edge("add_ten", END)

graph = builder.compile()

for part in graph.stream({"value": 5}, stream_mode="tasks"):
    data = part["data"]
    if "input" in data:
        # TaskPayload — task started
        print(f"[START ] {data['name']}  triggers={data['triggers']}  input={data['input']}")
    else:
        # TaskResultPayload — task finished
        if data["error"]:
            print(f"[ERROR ] {data['name']}  {data['error']}")
        else:
            print(f"[DONE  ] {data['name']}  result={data['result']}")
```

Output:

```
[START ] double  triggers=['start:double']  input={'value': 5}
[DONE  ] double  result={'value': 10}
[START ] add_ten  triggers=['double']  input={'value': 10}
[DONE  ] add_ten  result={'value': 20}
```

### Example 2 — combining tasks + custom for a monitoring dashboard

```python
for mode, payload in graph.stream(
    {"value": 3},
    stream_mode=["tasks", "custom"],
):
    if mode == "custom":
        print(f"[CUSTOM] {payload}")
    elif mode == "tasks":
        data = payload["data"]
        if "input" in data:
            print(f"[TASK START] {data['name']} ← {data['triggers']}")
        elif data.get("error"):
            print(f"[TASK ERROR] {data['name']}: {data['error']}")
        else:
            print(f"[TASK DONE ] {data['name']}: {data['result']}")
```

### Example 3 — detecting interrupted tasks

When a node calls `interrupt()`, the `TaskResultPayload.interrupts` list is non-empty. Use this to build a monitoring layer that detects pending human approvals.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    approved: bool

def approval(state: State) -> dict:
    interrupt("Human approval required")
    return {"approved": True}

builder = StateGraph(State)
builder.add_node("approval", approval)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-tasks"}}

for part in graph.stream({"approved": False}, config, stream_mode="tasks"):
    data = part["data"]
    if "interrupts" in data and data["interrupts"]:
        print(f"Node '{data['name']}' is waiting for human input:")
        for intr in data["interrupts"]:
            print(f"  interrupt value={intr.get('value', '?')!r}")
```

---

## 10 · `set_node_defaults()` — graph-wide retry / cache / timeout

**Module:** `langgraph.graph.state`

`set_node_defaults()` sets **fallback** policies applied to every node that does not specify its own. Per-node `add_node(..., retry_policy=...)` always wins over the default.

```python
def set_node_defaults(
    self,
    *,
    retry_policy:  RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy:  CachePolicy | None = None,
    error_handler: StateNode | None = None,
    timeout:       float | timedelta | TimeoutPolicy | None = None,
) -> Self:
```

Key rules (from source):

- `retry_policy` and `timeout` apply to **all** nodes, including error-handler nodes.
- `cache_policy` and `error_handler` only apply to **regular** nodes (error handlers cannot be cached or catch themselves).
- Defaults are applied at `compile()` time; modifying `set_node_defaults` after compile has no effect.
- Policies are **not** inherited by subgraphs.

### Example 1 — global retry for all nodes

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class State(TypedDict):
    result: str

call_counts: dict[str, int] = {"fetch": 0, "process": 0}

def fetch(state: State) -> dict:
    call_counts["fetch"] += 1
    if call_counts["fetch"] < 2:
        raise ConnectionError("network down")
    return {"result": "fetched"}

def process(state: State) -> dict:
    call_counts["process"] += 1
    if call_counts["process"] < 2:
        raise ConnectionError("db down")
    return {"result": "processed"}

builder = StateGraph(State)
builder.set_node_defaults(
    retry_policy=RetryPolicy(
        initial_interval=0.01,
        max_attempts=3,
        retry_on=ConnectionError,
        jitter=False,
    )
)
builder.add_node("fetch", fetch)
builder.add_node("process", process)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", "process")
builder.add_edge("process", END)

graph = builder.compile()
print(graph.invoke({"result": ""}))  # {'result': 'processed'}
```

### Example 2 — global timeout + per-node override

Nodes that need a longer timeout can override the global default via `add_node(timeout=...)`.

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

builder = StateGraph(State)
builder.set_node_defaults(
    timeout=TimeoutPolicy(run_timeout=5.0)   # 5s global default
)

# This node gets the global 5s timeout
builder.add_node("fast_node", fast_fn)

# This node overrides with its own 60s timeout
builder.add_node(
    "slow_node",
    slow_fn,
    timeout=TimeoutPolicy(run_timeout=60.0),
)
builder.add_edge(START, "fast_node")
builder.add_edge("fast_node", "slow_node")
builder.add_edge("slow_node", END)
```

### Example 3 — global error handler

A global `error_handler` runs whenever any regular node raises an unhandled exception. It receives the same `state` the failing node saw. Return a dict to apply state updates before the graph terminates.

```python
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

log = logging.getLogger(__name__)

class State(TypedDict):
    value: int
    error: str

def fallible(state: State) -> dict:
    raise ValueError("something went wrong")

def global_error_handler(state: State) -> dict:
    import traceback
    error_msg = traceback.format_exc()
    log.error("Node failed:\n%s", error_msg)
    # Write the error into state so downstream nodes / clients can see it
    return {"error": error_msg.splitlines()[-1]}

builder = StateGraph(State)
builder.set_node_defaults(error_handler=global_error_handler)
builder.add_node("fallible", fallible)
builder.add_edge(START, "fallible")
builder.add_edge("fallible", END)

graph = builder.compile()
result = graph.invoke({"value": 1, "error": ""})
print(result["error"])  # 'ValueError: something went wrong'
```

### `set_node_defaults` vs `add_node` precedence

| Scope | `retry_policy` | `cache_policy` | `error_handler` | `timeout` |
|-------|---------------|----------------|-----------------|-----------|
| `set_node_defaults` | Default | Default | Default | Default |
| `add_node(..., retry_policy=X)` | **Overrides** | Default | Default | Default |
| `add_node(..., cache_policy=X)` | Default | **Overrides** | Default | Default |
| Subgraph | **Not inherited** | **Not inherited** | **Not inherited** | **Not inherited** |

---

## Summary

| Class / symbol | Key insight | Common mistake |
|---------------|-------------|----------------|
| `RetryPolicy` sequence | LangGraph uses the **first matching** policy in the list | Forgetting `max_attempts=1` on the "give up" policy — it defaults to 3 |
| `TimeoutPolicy` + `heartbeat()` | `idle_timeout` needs explicit `heartbeat()` when `refresh_on="heartbeat"` | Using sync `time.sleep()` — blocks GIL, timeout fires late |
| `Overwrite` | Bypasses the reducer entirely; at most one per channel per super-step | Using `Overwrite` on a `LastValue` channel — it's a no-op; `LastValue` always replaces |
| `interrupt()` multi | Node re-runs from top on resume; already-resolved interrupts return immediately | Calling `interrupt()` in a loop without handling the replay — state mutation side-effects run twice |
| `add_sequence()` | No `START`/`END` edges added automatically | Forgetting to add `add_edge(START, first_node)` and `add_edge(last_node, END)` |
| `update_state()` | Thin wrapper around `bulk_update_state([[StateUpdate(...)]])` | Passing `as_node=None` when the last node is ambiguous — raises if multiple nodes wrote in the last step |
| `get_stream_writer()` | Writes to `stream_mode="custom"`; no-op outside a graph run | Using inside `asyncio.create_task()` on Python < 3.11 — context not propagated |
| `stream_mode="checkpoints"` | Fires after **every** checkpoint write, including intermediate steps | Expecting only one event per `invoke()` — there are N+1 (one per super-step plus initial) |
| `stream_mode="tasks"` | Distinguish start vs result by checking for `"input"` key | Assuming `data["result"]` always exists — it's absent on error or interrupt |
| `set_node_defaults()` | Applied at `compile()` time; subgraphs do **not** inherit | Calling it after `compile()` — changes are silently ignored |

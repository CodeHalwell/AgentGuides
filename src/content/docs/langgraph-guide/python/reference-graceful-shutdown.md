---
title: "RunControl & GraphDrained — graceful shutdown API reference"
description: "Cooperative drain with RunControl.request_drain(), the GraphDrained exception, SIGTERM patterns, and safe checkpoint-based resumption for LangGraph 1.2.11."
framework: langgraph
language: python
sidebar:
  label: "Ref · Graceful shutdown"
  order: 40
---

# RunControl & GraphDrained — graceful shutdown

Verified against **`langgraph==1.2.11`** (modules: `langgraph.runtime`, `langgraph.errors`).

LangGraph supports _cooperative drain_: a running graph can be told to stop at the next safe superstep boundary, flush its checkpoint, and raise `GraphDrained` — leaving the run in a state that can be resumed from exactly where it stopped. This is the right pattern for SIGTERM / scale-down scenarios.

---

## Class definitions

### `RunControl`

```python
# langgraph.runtime
class RunControl:
    """Run-scoped control surface for cooperative draining."""

    def request_drain(self, reason: str = "shutdown") -> None:
        """Signal the graph to stop at the next superstep boundary."""
        ...

    @property
    def drain_requested(self) -> bool:
        """True once request_drain() has been called."""
        ...

    @property
    def drain_reason(self) -> str | None:
        """The reason string passed to request_drain(), or None."""
        ...
```

`RunControl` is injected into `Runtime.control` during every graph run. It exposes a single write (calling `request_drain()`) that is safe to call from any thread because it is a single attribute write with no locking needed.

### `GraphDrained`

```python
# langgraph.errors
from langgraph.errors import GraphBubbleUp

class GraphDrained(GraphBubbleUp):
    """Raised when a graph run exits early due to a drain request.

    This indicates the graph stopped cooperatively at a superstep boundary
    because RunControl.request_drain() was called. The checkpoint is saved
    and the run can be resumed later.
    """

    def __init__(self, reason: str = "shutdown") -> None:
        self.reason = reason
        super().__init__(f"Graph drained: {reason}")
```

`GraphDrained` is raised at the end of the current superstep once `request_drain()` has been called. The checkpoint is always flushed before the exception propagates, so the run is safe to resume from the saved state.

---

## Imports

```python
from langgraph.runtime import RunControl, Runtime
from langgraph.errors import GraphDrained
```

---

## Minimal example — SIGTERM handler

```python
import signal
import threading
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime, RunControl
from langgraph.errors import GraphDrained


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    step: int


def process(state: State, runtime: Runtime) -> dict:
    """Node that checks the drain signal before doing expensive work."""
    if runtime.drain_requested:
        # Exit early — the framework will checkpoint and raise GraphDrained
        return {}

    # ... expensive processing ...
    return {"step": state.get("step", 0) + 1}


graph = (
    StateGraph(State)
    .add_node("process", process)
    .add_edge(START, "process")
    .add_edge("process", END)
    # NOTE: InMemorySaver is used here for brevity. It does not survive a process
    # restart, so "resume with the same config later" only works within the same
    # process. For true cross-process resumption use a durable checkpointer
    # such as SqliteSaver or AsyncPostgresSaver.
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "demo"}}

# Create the control handle upfront so the handler can use it immediately,
# even before the first node runs.
control = RunControl()


def _sigterm_handler(signum, frame):
    control.request_drain(reason="SIGTERM")


signal.signal(signal.SIGTERM, _sigterm_handler)

try:
    result = graph.invoke(
        {"messages": [("user", "start")]},
        config,
        control=control,
    )
    print("Graph finished normally:", result["step"])
except GraphDrained as exc:
    print(f"Graph drained cooperatively — reason: {exc.reason}")
    print("Checkpoint saved in this process; resume with the same config.")
```

---

## How cooperative drain works

```
invoke() called
     │
     ▼
superstep 1 → nodes run → checkpoint saved
     │
  [request_drain() called from another thread]
     │
     ▼
superstep 2 → nodes run → checkpoint saved
     │
  [GraphDrained raised — run exits cleanly]
     │
     ▼
caller catches GraphDrained; can resume later
```

Key properties:
- `request_drain()` takes effect at the **next superstep boundary**, not mid-node.
- The checkpoint is **always flushed** before `GraphDrained` propagates.
- Drain is **cooperative**: nodes can observe `runtime.drain_requested` to exit early within a superstep, but the framework enforces the boundary regardless.
- A single `RunControl` instance is created per run and is **not safe to reuse** across runs.

---

## Accessing `RunControl` from a node

`RunControl` is exposed on `Runtime.control`. Read `drain_requested` to exit loops early:

```python
import asyncio
from langgraph.runtime import Runtime


async def long_running_node(state: State, runtime: Runtime) -> dict:
    """Streaming node that yields progress and respects drain signals."""
    results = []
    async for chunk in some_async_generator():
        if runtime.drain_requested:
            # Save partial results and return — the framework will drain after this node
            break
        results.append(chunk)
        runtime.stream_writer({"partial": len(results)})
    return {"results": results}
```

`runtime.drain_requested` is a shortcut for `runtime.control.drain_requested if runtime.control else False`.

---

## Signalling drain from outside the graph

A common pattern is to signal drain from a background thread (e.g. a SIGTERM handler or a Kubernetes pre-stop hook):

```python
import signal
import threading
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

_active_control: RunControl | None = None
_lock = threading.Lock()


def handle_sigterm(signum, frame):
    with _lock:
        if _active_control:
            _active_control.request_drain(reason="SIGTERM")


signal.signal(signal.SIGTERM, handle_sigterm)


def node(state: dict) -> dict:
    return {}


graph = (
    StateGraph(dict)
    .add_node("node", node)
    .add_edge(START, "node")
    .add_edge("node", END)
    .compile(checkpointer=InMemorySaver())
)

# Pre-create the RunControl and register it BEFORE graph.invoke so that a
# SIGTERM arriving during graph startup — before any node runs — is still caught.
control = RunControl()
with _lock:
    _active_control = control

try:
    graph.invoke({}, {"configurable": {"thread_id": "t1"}}, control=control)
except GraphDrained as exc:
    print(f"Drained: {exc.reason} — resumable from checkpoint")
finally:
    with _lock:
        _active_control = None
```

---

## Resuming after drain

Because the checkpoint is flushed before `GraphDrained` is raised, resuming is identical to resuming after any other interruption:

```python
from langgraph.errors import GraphDrained

config = {"configurable": {"thread_id": "resumable-thread"}}

try:
    graph.invoke({"messages": [("user", "hello")]}, config)
except GraphDrained:
    pass  # checkpoint saved

# Resume: pass the same config with no input (or a new message)
result = graph.invoke(None, config)  # picks up from the last checkpoint
```

---

## Drain vs interrupt vs recursion limit

| Mechanism | Trigger | Checkpoint saved? | Resumable? |
|---|---|---|---|
| `request_drain()` | Cooperative signal (e.g. SIGTERM) | Yes (with a checkpointer) | Yes (with a checkpointer) |
| `interrupt()` | Node-level human-in-the-loop pause | Yes (with a checkpointer) | Yes (with a checkpointer) |
| `GraphRecursionError` | `recursion_limit` exceeded | Yes (last checkpoint, with a checkpointer) | Yes (with a checkpointer; raise `recursion_limit`) |
| Unhandled exception | Any node exception without handler | Yes (last successful checkpoint, with a checkpointer) | Yes (with a checkpointer; fix the node first) |

---

## `GraphDrained` fields

| Field | Type | Description |
|---|---|---|
| `reason` | `str` | String passed to `request_drain()` (default `"shutdown"`) |

---

## `RunControl` API reference

| Member | Type | Description |
|---|---|---|
| `request_drain(reason)` | method | Signal drain; safe to call from any thread |
| `drain_requested` | property → `bool` | True once drain has been requested |
| `drain_reason` | property → `str \| None` | Reason string, or `None` before drain |

---

## Patterns

### Kubernetes pre-stop hook

```python
import asyncio
import uuid
from fastapi import FastAPI, HTTPException
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

app = FastAPI()

# Thread-safe set of all in-flight RunControls.
_active_controls: set[RunControl] = set()
_controls_lock = asyncio.Lock()
_draining = False  # set to True once pre-stop begins; new runs are rejected


@app.post("/lifecycle/pre-stop")
async def pre_stop():
    """Kubernetes calls this before terminating the pod."""
    global _draining
    async with _controls_lock:
        _draining = True
        snapshot = list(_active_controls)
    for ctrl in snapshot:
        ctrl.request_drain(reason="k8s-prestop")
    return {"status": "draining", "active_runs": len(snapshot)}


@app.post("/run")
async def run_graph(payload: dict):
    """Endpoint that runs the graph; registers its RunControl for pre-stop draining."""
    control = RunControl()
    async with _controls_lock:
        if _draining:
            # Pre-stop already signalled; refuse new work so the pod can shut down.
            raise HTTPException(status_code=503, detail="Service is draining")
        _active_controls.add(control)
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = await graph.ainvoke(payload, config, control=control)
        return result
    except GraphDrained:
        return {"status": "drained", "thread_id": thread_id}
    finally:
        async with _controls_lock:
            _active_controls.discard(control)
```

### Async drain with timeout

```python
import asyncio
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained


async def run_with_drain_timeout(graph, input, config, timeout_seconds: float):
    """Run a graph; drain cooperatively if it exceeds the timeout."""
    control = RunControl()
    # Shield the task so asyncio.wait_for's cancellation doesn't kill the graph;
    # instead we signal drain and let it finish at the next superstep boundary.
    task = asyncio.create_task(graph.ainvoke(input, config, control=control))

    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        control.request_drain(reason="timeout")
        try:
            return await task  # wait for the cooperative checkpoint-safe exit
        except GraphDrained:
            return None
```

### Drain reason taxonomy

```python
from langgraph.runtime import RunControl

REASON_SIGTERM = "SIGTERM"
REASON_SCALE_DOWN = "scale-down"
REASON_BUDGET_EXCEEDED = "budget-exceeded"
REASON_USER_CANCEL = "user-cancel"

# Record the reason for observability
control = RunControl()
control.request_drain(reason=REASON_SIGTERM)
assert control.drain_reason == REASON_SIGTERM
```

---

## Gotchas

- **Drain fires at superstep boundaries, not mid-node.** A node currently executing will complete before drain takes effect. Design long-running nodes to check `runtime.drain_requested` internally if you need finer granularity.
- **`GraphDrained` is not a subclass of `GraphRecursionError`.** Catch them separately.
- **Do not reuse `RunControl` across runs.** Once `request_drain()` has been called, `drain_requested` stays `True`. Create a fresh `RunControl` (or let the framework create one) per run.
- **Synchronous nodes cannot be cancelled mid-execution.** Drain works at superstep boundaries — a synchronous CPU-bound node will finish before the drain takes effect.

---

## Version history

| Version | Change |
|---|---|
| 1.2.11 | `RunControl`, `GraphDrained`, `Runtime.control` production-stable |
| 1.2.0 | `Runtime.drain_requested` shortcut added |
| 0.6.0 | `Runtime` dataclass and cooperative drain first introduced |

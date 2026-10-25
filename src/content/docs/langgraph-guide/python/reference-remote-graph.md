---
title: "RemoteGraph — API reference"
description: "Call a deployed LangGraph graph (LangSmith Deployment or Agent Server) as if it were a local CompiledStateGraph — same invoke/stream/get_state surface, works as a subgraph node, supports interrupt/resume and state inspection."
framework: "langgraph"
language: "python"
sidebar:
  label: "Ref · RemoteGraph"
  order: 38
---

Verified against **`langgraph==%%version=langgraph%%`** / **`langgraph-sdk==%%version=langgraph-sdk%%`** (module: `langgraph.pregel.remote`).

`RemoteGraph` calls a graph running in the **LangGraph Agent Server** (a LangSmith Deployment, a self-hosted server, or `langgraph dev`) over HTTP. It exposes the same `invoke`, `stream`, `get_state`, `update_state`, `get_state_history` surface as a local graph.

It is close to a drop-in replacement, with one difference to plan for: **results come back as JSON**. Messages in `invoke` and `stream` output are plain dicts (`{"type": "ai", "content": ...}`), not `AIMessage` objects, and `__interrupt__` entries in streamed chunks are dicts too. `get_state()` does rebuild `Interrupt` objects. When a `RemoteGraph` runs as a node inside a local graph, the parent's reducers (such as `add_messages`) turn the dicts back into message objects.

`RemoteGraph` is also a valid LangGraph **node**: you can add it to a parent `StateGraph` with `add_node(remote_graph)`, enabling local-remote hybrid architectures.

## Minimal runnable example

```python
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.remote import RemoteGraph

# Point at a deployment (set LANGGRAPH_API_KEY in your env, or pass api_key=)
remote = RemoteGraph(
    "my_agent",                    # graph ID or assistant ID on the server
    url="https://my-deployment.langsmith.com",
)

cfg: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}
result = remote.invoke({"messages": [("user", "Hello")]}, cfg)
print(result["messages"][-1]["content"])   # a dict, not an AIMessage
```

Checked against a local `langgraph dev` server. The thread is created on first use (`if_not_exists="create"`); without a `thread_id` the call runs statelessly.

> **Requires `langgraph-sdk`** — installed automatically with `langgraph`:
> ```bash
> pip install langgraph
> ```

## Imports

```python
from langgraph.pregel.remote import RemoteGraph, RemoteException
```

`langgraph.pregel` does not re-export them; import from `langgraph.pregel.remote`.

## Constructor

```text
RemoteGraph(
    assistant_id: str,           # graph_id or assistant UUID on the server
    /,
    *,
    url: str | None = None,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    client: LangGraphClient | None = None,
    sync_client: SyncLangGraphClient | None = None,
    config: RunnableConfig | None = None,
    name: str | None = None,
    distributed_tracing: bool = False,
)
```

| Parameter | Description |
|---|---|
| `assistant_id` | The graph name or UUID registered on the server. Positional-only. |
| `url` | HTTP base URL of the deployment. Reads `LANGGRAPH_API_KEY` / `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` from the environment if `api_key` is not passed. |
| `api_key` | Overrides the environment variable API key. |
| `headers` | Extra HTTP headers forwarded on every request. |
| `client` | Pre-built async `LangGraphClient` (from `langgraph_sdk`). Use when you need full control over connection pooling. |
| `sync_client` | Pre-built sync `SyncLangGraphClient`. |
| `config` | A `RunnableConfig` merged into every call. Useful for attaching a persistent `thread_id` or tags at construction time. |
| `name` | Human-readable name shown in diagrams and traces. Defaults to `assistant_id`. |
| `distributed_tracing` | Forward LangSmith tracing headers so server-side spans appear under the local trace. |

Provide `url`, or the client for the methods you call (`sync_client` for `invoke`/`stream`, `client` for `ainvoke`/`astream`). The constructor accepts neither; the first call then raises `ValueError: Sync client is not initialized: please provide url or sync_client ...`.

## Methods

All methods mirror the `CompiledStateGraph` surface:

| Method | Returns | Notes |
|---|---|---|
| `invoke(input, config, *, context, interrupt_before, interrupt_after, headers, params, version)` | `dict \| GraphOutput` | Blocking. `version="v2"` returns `GraphOutput`. |
| `ainvoke(...)` | `dict \| GraphOutput` | Async variant. |
| `stream(input, config, *, stream_mode, interrupt_before, interrupt_after, subgraphs, headers, params, version)` | `Iterator` | Sync generator. |
| `astream(...)` | `AsyncIterator` | Async generator. |
| `get_state(config, *, subgraphs, headers, params)` | `StateSnapshot` | Requires checkpointer on server. |
| `aget_state(...)` | `StateSnapshot` | Async variant. |
| `get_state_history(config, *, filter, before, limit, headers, params)` | `Iterator[StateSnapshot]` | Newest first, as locally. |
| `aget_state_history(...)` | `AsyncIterator[StateSnapshot]` | Async variant. |
| `update_state(config, values, as_node, *, headers, params)` | `RunnableConfig` | Write state without running the graph. |
| `aupdate_state(...)` | `RunnableConfig` | Async variant. |
| `bulk_update_state(config, updates)` | — | Raises `NotImplementedError` in %%version=langgraph%%. Call `update_state` once per step instead. |
| `abulk_update_state(...)` | — | Raises `NotImplementedError`. |
| `stream_events(..., version="v3")` / `astream_events(..., version="v3")` | run handle | The experimental v3 API works remotely; `astream_events` with `"v1"`/`"v2"` raises `NotImplementedError`. |
| `get_graph(config, *, xray, headers, params)` | `DrawableGraph` | Fetch the server-side graph topology for visualization. |
| `aget_graph(...)` | `DrawableGraph` | Async variant. |
| `with_config(config, **kwargs)` | `Self` | Return a copy with merged config. |

## `invoke` and `ainvoke`

```python
result = remote.invoke(
    {"messages": [("user", "Summarise this document")]},
    {"configurable": {"thread_id": "t-1"}},
    interrupt_before=["human_review"],   # pause before this node
    version="v2",                        # returns GraphOutput instead of dict
)
# result.value  → final state dict
# result.interrupts → tuple[Interrupt, ...]
```

```python
# Async version:
result = await remote.ainvoke(
    {"messages": [("user", "hello")]},
    {"configurable": {"thread_id": "t-1"}},
)
```

## `stream` and `astream`

```python
for chunk in remote.stream(
    {"messages": [("user", "hello")]},
    {"configurable": {"thread_id": "t-1"}},
    stream_mode="updates",
):
    print(chunk)
```

With a **list** of modes, a local graph yields `(mode, data)` tuples, but `RemoteGraph` (with the default `version="v1"`) yields the SDK's `StreamPart` named tuples, which have three fields (`event`, `data`, `id`), so `for mode, data in ...` fails with `ValueError: too many values to unpack`. Use `version="v2"` for the same typed dicts a local graph gives you:

```python
# Multiple modes, async, typed parts:
async for part in remote.astream(
    {"messages": [("user", "hi")]},
    {"configurable": {"thread_id": "t-1"}},
    stream_mode=["updates", "messages"],
    version="v2",
):
    if part["type"] == "messages":
        msg, meta = part["data"]
        print(msg["content"], end="", flush=True)   # msg is a dict
```

All seven stream modes work: `"values"`, `"updates"`, `"messages"`, `"custom"`, `"checkpoints"`, `"tasks"`, `"debug"`.

## State inspection and time-travel

```python
cfg = {"configurable": {"thread_id": "t-1"}}

# Current state
snap = remote.get_state(cfg)
print(snap.values)
print(snap.next)           # which nodes will run next
print(snap.interrupts)     # pending Interrupt objects

# Full history (newest first)
for snap in remote.get_state_history(cfg, limit=10):
    print(snap.metadata["step"], snap.values)

# Time-travel: re-run from an older checkpoint
old_cfg = snap.config     # config pointer from history
result = remote.invoke(None, old_cfg)
```

## Interrupt / resume over a remote graph

The pattern is identical to a local graph — `interrupt()` on the server pauses execution and the `Interrupt` surfaces in `StateSnapshot.interrupts` and the `__interrupt__` key of `stream_mode="updates"`.

```python
from langgraph.types import Command, interrupt

cfg = {"configurable": {"thread_id": "t-human"}}

# First call — triggers an interrupt on the server
for chunk in remote.stream({"query": "Analyse financials"}, cfg):
    if "__interrupt__" in chunk:
        question = chunk["__interrupt__"][0]["value"]   # dicts over the wire
        print("Server asks:", question)

# get_state() returns Interrupt objects
print(remote.get_state(cfg).interrupts[0].value)

# Resume with the human's answer
result = remote.invoke(Command(resume="Approve"), cfg)
print(result)
```

## Using RemoteGraph as a subgraph node

Add a `RemoteGraph` directly to a parent `StateGraph` just like any compiled subgraph:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.pregel.remote import RemoteGraph
from typing import Annotated
from langchain_core.messages import AnyMessage

# Remote specialist
summariser = RemoteGraph(
    "summariser_agent",
    url="https://my-deployment.langsmith.com",
    name="summariser",   # appears as node label in diagrams
)

class OrchestratorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def prepare(state: OrchestratorState) -> dict:
    return {}

# Wire the remote graph as a node
builder = StateGraph(OrchestratorState)
builder.add_node("prepare", prepare)
builder.add_node("summariser", summariser)   # RemoteGraph as node
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "summariser")
builder.add_edge("summariser", END)

graph = builder.compile()
result = graph.invoke({"messages": [("user", "Summarise: ...")]})
```

When used as a node, `RemoteGraph` forwards the parent's config, and the parent's `thread_id` becomes the remote thread's ID (created if missing). Without a `thread_id` the remote run is stateless. The remote output goes through the parent's reducers, so `add_messages` turns the message dicts back into `AIMessage` objects.

## Distributed tracing with LangSmith

Set `distributed_tracing=True` to propagate the LangSmith trace from your local process into the remote deployment, so all spans appear under a single trace:

```python
remote = RemoteGraph(
    "my_agent",
    url="https://my-deployment.langsmith.com",
    distributed_tracing=True,
)
```

Under the hood, `RemoteGraph` merges the current LangSmith run tree's tracing headers into each request when one is active.

## `with_config` — partial application

Bind defaults at construction time to avoid repeating config on every call:

```python
base = RemoteGraph("my_agent", url="https://my-deployment.langsmith.com")

# Create a per-user handle that always routes to the same thread
alice_graph = base.with_config({"configurable": {"thread_id": "alice"}})

# All calls go to alice's thread
alice_graph.invoke({"messages": [("user", "hi")]})
alice_graph.invoke({"messages": [("user", "follow up")]})
```

## `RemoteException`

When the server returns an error, `RemoteGraph` wraps it in `RemoteException`:

```python
from langgraph.pregel.remote import RemoteException

try:
    remote.invoke({"messages": []}, cfg)
except RemoteException as e:
    print("Server error:", e)
except Exception as e:
    print("Network or auth error:", e)
```

## Patterns

### 1. Async streaming with token-level output

```python
import asyncio
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("chat_agent", url="https://my-deployment.langsmith.com")


async def chat(user_input: str, thread_id: str) -> None:
    cfg = {"configurable": {"thread_id": thread_id}}
    async for part in remote.astream(
        {"messages": [("user", user_input)]},
        cfg,
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if part["type"] == "messages":
            msg, meta = part["data"]
            if msg.get("content"):
                print(msg["content"], end="", flush=True)
        elif part["type"] == "updates" and "__interrupt__" in part["data"]:
            print("\n[Waiting for approval]")


asyncio.run(chat("Tell me about LangGraph", "u-123"))
```

### 2. Parallel remote calls with `asyncio.gather`

```python
import asyncio
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("analyser", url="https://my-deployment.langsmith.com")


async def analyse_all(documents: list[str]) -> list[dict]:
    tasks = [
        remote.ainvoke(
            {"document": doc},
            {"configurable": {"thread_id": f"doc-{i}"}},
        )
        for i, doc in enumerate(documents)
    ]
    return await asyncio.gather(*tasks)


results = asyncio.run(analyse_all(["doc1...", "doc2...", "doc3..."]))
```

### 3. Human-in-the-loop orchestrator over a remote graph

```python
from langgraph.types import Command
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("approval_flow", url="https://my-deployment.langsmith.com")
cfg = {"configurable": {"thread_id": "approval-1"}}

def run_with_approval(initial_input: dict) -> dict:
    state = remote.invoke(initial_input, cfg)

    # Check if interrupted
    snap = remote.get_state(cfg)
    while snap.interrupts:
        for interrupt in snap.interrupts:
            print("Approval required:", interrupt.value)
            decision = input("Approve? (y/n): ")
        state = remote.invoke(Command(resume=decision), cfg)
        snap = remote.get_state(cfg)

    return state


result = run_with_approval({"proposal": "Deploy to prod"})
```

### 4. State inspection and manual update

```python
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("pipeline", url="https://my-deployment.langsmith.com")
cfg = {"configurable": {"thread_id": "pipe-42"}}

# Inspect current state
snap = remote.get_state(cfg)
print("Current values:", snap.values)
print("Next nodes:", snap.next)

# Manually inject a value as if it came from a node
new_cfg = remote.update_state(
    cfg,
    {"status": "approved", "approver": "alice"},
    as_node="review_step",
)

# Continue from the updated state
result = remote.invoke(None, new_cfg)
```

### 5. Custom headers for per-request auth

```python
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("secure_agent", url="https://my-deployment.langsmith.com")

def call_for_user(user_token: str, query: str) -> dict:
    return remote.invoke(
        {"query": query},
        {"configurable": {"thread_id": user_token}},
        headers={"X-User-Token": user_token},   # forwarded per-request
    )
```

### 6. Wrapping RemoteGraph with fallback to a local graph

```python
from langgraph.pregel.remote import RemoteGraph, RemoteException
from langgraph.graph import StateGraph, START, END

# Build a lightweight local fallback
local_graph = ...  # a compiled StateGraph

remote = RemoteGraph("main_agent", url="https://my-deployment.langsmith.com")


def invoke_with_fallback(input_: dict, cfg: dict) -> dict:
    try:
        return remote.invoke(input_, cfg)
    except (RemoteException, Exception):
        # Fall back to local graph when the remote is unavailable
        return local_graph.invoke(input_, cfg)
```

## Gotchas

- **`url` is required for auto-created clients.** If you pass neither `url` nor an explicit `client` / `sync_client`, the constructor succeeds but every method call raises `ValueError`.
- **Outputs are JSON.** Read `result["messages"][-1]["content"]`, not `.content`, unless the `RemoteGraph` runs as a node under `add_messages`.
- **Multi-mode `stream` yields `StreamPart(event, data, id)` tuples in v1.** Pass `version="v2"` and read `part["type"]` / `part["data"]`.
- **`assistant_id` is positional-only.** You cannot write `RemoteGraph(assistant_id="x", url=...)` — the first arg has no keyword form.
- **Thread-ID must come from the caller.** The server has its own checkpointer and keeps state per thread. Without a `thread_id` each call is stateless.
- **Subgraph threads.** As a node, the `RemoteGraph` uses the parent's `thread_id` as the remote thread ID, so two parents that share a thread ID also share the remote thread.
- **`name` defaults to `assistant_id`.** Always pass `name=` when adding as a node; the Mermaid diagram shows `assistant_id` otherwise, which is usually a UUID.
- **`astream_events` only supports `version="v3"`.** With `"v1"`/`"v2"` it raises `NotImplementedError`; use `astream` with the appropriate `stream_mode`.
- **`distributed_tracing=True` needs an active LangSmith run.** Without one there are no tracing headers to send; no error is raised.
- **`params=` are passed as HTTP query parameters.** Useful for server-side feature flags or route selection, but their interpretation is deployment-specific.
- **`subgraphs=True` on `stream` exposes inner namespace paths.** The namespace tuples include the server-side task IDs, which change on each run. Filter by node name prefix, not the full tuple.

## History

| Version | Change |
|---|---|
| 1.2.3 | `stream_events` / `astream_events` with `version="v3"` supported on `RemoteGraph`. |

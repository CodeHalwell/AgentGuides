---
title: "RemoteGraph — API reference"
description: "Call a deployed LangGraph Platform graph as if it were a local CompiledStateGraph — same invoke/stream/get_state surface, works as a subgraph node, supports interrupt/resume and state inspection."
framework: langgraph
language: python
sidebar:
  label: "Ref · RemoteGraph"
  order: 39
---

# RemoteGraph — API reference

Verified against **`langgraph==1.2.4`** / **`langgraph-sdk==0.4.2`** (module: `langgraph.pregel.remote`).

`RemoteGraph` is a drop-in replacement for a compiled `StateGraph` that calls a **LangGraph Platform deployment** (LangSmith, self-hosted server, or any API that speaks the LangGraph Server spec) over HTTP. It exposes the same `invoke`, `stream`, `get_state`, `update_state`, `get_state_history` surface as a local graph, so you can swap a local graph for a remote one without changing the calling code.

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
print(result["messages"][-1].content)
```

> **Requires `langgraph-sdk`** — installed automatically with `langgraph`:
> ```bash
> pip install langgraph
> ```

## Imports

```python
from langgraph.pregel.remote import RemoteGraph, RemoteException
```

Both are also exported from `langgraph.pregel`:

```python
from langgraph.pregel import RemoteGraph
```

## Constructor

```python
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

Provide **at least one of** `url`, `client`, or `sync_client`.

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
| `get_state_history(config, *, filter, before, limit, headers, params)` | `Iterator[StateSnapshot]` | Oldest-first. |
| `aget_state_history(...)` | `AsyncIterator[StateSnapshot]` | Async variant. |
| `update_state(config, values, as_node, *, headers, params)` | `RunnableConfig` | Write state without running the graph. |
| `aupdate_state(...)` | `RunnableConfig` | Async variant. |
| `bulk_update_state(config, updates)` | `RunnableConfig` | Apply multiple super-steps at once. |
| `abulk_update_state(...)` | `RunnableConfig` | Async variant. |
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

```python
# Multiple modes, async:
async for mode, data in remote.astream(
    {"messages": [("user", "hi")]},
    {"configurable": {"thread_id": "t-1"}},
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        msg, meta = data
        print(msg.content, end="", flush=True)
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

# Time-travel: resume from an older checkpoint
old_cfg = snap.config     # config pointer from history
result = remote.invoke(
    Command(resume="yes"),
    old_cfg,
)
```

## Interrupt / resume over a remote graph

The pattern is identical to a local graph — `interrupt()` on the server pauses execution and the `Interrupt` surfaces in `StateSnapshot.interrupts` and the `__interrupt__` key of `stream_mode="updates"`.

```python
from langgraph.types import Command, interrupt

cfg = {"configurable": {"thread_id": "t-human"}}

# First call — triggers an interrupt on the server
for chunk in remote.stream({"query": "Analyse financials"}, cfg):
    if "__interrupt__" in chunk:
        question = chunk["__interrupt__"][0].value
        print("Server asks:", question)

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

When used as a node, `RemoteGraph` forwards the parent's `config` (including `thread_id`) automatically. The subgraph uses `checkpointer=True` semantics — it inherits the parent's checkpointing scope.

## Distributed tracing with LangSmith

Set `distributed_tracing=True` to propagate the LangSmith trace from your local process into the remote deployment, so all spans appear under a single trace:

```python
remote = RemoteGraph(
    "my_agent",
    url="https://my-deployment.langsmith.com",
    distributed_tracing=True,
)
```

Under the hood, `RemoteGraph` adds `X-Langsmith-Trace` headers to every request when a LangSmith `RunTree` is active.

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
    async for mode, data in remote.astream(
        {"messages": [("user", user_input)]},
        cfg,
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            msg, meta = data
            if msg.content:
                print(msg.content, end="", flush=True)
        elif mode == "updates" and "__interrupt__" in data:
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

- **`url` is required for auto-created clients.** If you pass neither `url` nor an explicit `client` / `sync_client`, every method call raises `ValueError`.
- **`assistant_id` is positional-only.** You cannot write `RemoteGraph(assistant_id="x", url=...)` — the first arg has no keyword form.
- **Thread-ID must come from the caller.** Unlike a local graph compiled with `InMemorySaver`, the remote server has its own checkpointer. Always set `config["configurable"]["thread_id"]`.
- **`checkpointer=True` semantics when used as a subgraph.** When wired into a parent `StateGraph`, the `RemoteGraph` uses the parent's `thread_id` to isolate its checkpoint namespace. The server-side checkpointer must be enabled on the deployment.
- **`name` defaults to `assistant_id`.** Always pass `name=` when adding as a node; the Mermaid diagram shows `assistant_id` otherwise, which is usually a UUID.
- **`astream_events` raises `NotImplementedError`.** Use `astream` with the appropriate `stream_mode` instead.
- **`distributed_tracing=True` requires an active LangSmith session.** Without a `LANGSMITH_API_KEY` and an active `RunTree`, the tracing headers are not added — no error is raised.
- **`params=` are passed as HTTP query parameters.** Useful for server-side feature flags or route selection, but their interpretation is deployment-specific.
- **`subgraphs=True` on `stream` exposes inner namespace paths.** The namespace tuples include the server-side task IDs, which change on each run. Filter by node name prefix, not the full tuple.

## Breaking changes

| Version | Change |
|---|---|
| 0.3.14 (sdk) | `RemoteGraph` uses the LangGraph SDK `SyncLangGraphClient` / `LangGraphClient` internally; `url` auto-creates both. |
| 1.2.1 | `bulk_update_state` / `abulk_update_state` added for multi-step state injection. `name=` kwarg respects custom labels in diagram. |
| 1.0 | `RemoteGraph` first stabilised; moved from `langgraph.pregel.remote` to top-level re-export from `langgraph.pregel`. |

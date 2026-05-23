---
title: "Chapter 5 — Memory & Persistence"
description: "Short-term state via checkpointers, long-term memory via Store, and cross-thread memory that survives across sessions."
framework: langgraph
language: python
sidebar:
  label: "5 · Memory & persistence"
  order: 5
---

# Chapter 5 — Memory & Persistence

**What you'll learn:** LangGraph's two memory tiers. **Checkpointers** save graph state at each step so you can resume after a failure. **Stores** provide durable key/value storage (and optional semantic search) that survives across threads and users. You'll also see the **cross-thread memory** pattern that lets one conversation learn from another.

Verified against **`langgraph==1.2.1`** (modules: `langgraph.checkpoint.memory`, `langgraph.store.memory`, `langgraph.store.base`).

**Time:** ~25 minutes.

> Prereqs: [Chapter 1 — Setup & core concepts](/langgraph-guide/python/chapter-01-setup-and-core-concepts/).

---

## Short-Term Memory: Checkpointers

Checkpointers save the full graph state at each step as a `Checkpoint`. They enable:

- **Resume after failure** — restart from the last saved step.
- **Human-in-the-loop** — `interrupt()` pauses the graph; `Command(resume=...)` continues it.
- **Time-travel debugging** — `get_state_history()` returns every historical snapshot.
- **Thread-scoped memory** — the same `thread_id` accumulates state across multiple `invoke` calls.

### `InMemorySaver` (development)

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# All state persists within this Python process only — ideal for tests and demos
```

### `SqliteSaver` (local persistence)

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based SQLite — survives process restarts
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Or use the convenience class method:
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Async variant (requires aiosqlite)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
async_checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")

graph = builder.compile(checkpointer=checkpointer)
```

### `PostgresSaver` (production)

```python
# pip install langgraph-checkpoint-postgres
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

conn_str = "postgresql://user:password@localhost/langgraph_db"

# Sync — run setup_tables() once before first use
with PostgresSaver.from_conn_string(conn_str) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

# Async variant
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(conn_str) as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

---

## Using Checkpoints

### Basic thread-scoped persistence

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator
from typing import Annotated

class CountState(TypedDict):
    count: Annotated[int, operator.add]  # reducer: accumulates across invocations

def bump(state: CountState) -> dict:
    return {"count": 1}

builder = StateGraph(CountState)
builder.add_node("bump", bump)
builder.add_edge(START, "bump")
builder.add_edge("bump", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "user-123"}}

# First invocation — count starts at 0 (or the initial value in invoke)
result1 = graph.invoke({"count": 0}, config=config)
print(result1["count"])   # 1

# Second invocation — count accumulated from checkpoint
result2 = graph.invoke({"count": 0}, config=config)
print(result2["count"])   # 2 (1 + 1, accumulated via reducer)

result3 = graph.invoke({"count": 0}, config=config)
print(result3["count"])   # 3
```

### Inspecting and time-travelling checkpoints

```python
# Get current state for a thread
current_state = graph.get_state(config)
print(f"Next node(s): {current_state.next}")
print(f"Values:       {current_state.values}")
print(f"Checkpoint:   {current_state.config['configurable']['checkpoint_id']}")

# Walk the full history for a thread (most recent first)
for i, snapshot in enumerate(graph.get_state_history(config)):
    cp_id = snapshot.config["configurable"]["checkpoint_id"]
    print(f"Step {i}: id={cp_id} values={snapshot.values}")

# Time-travel: resume execution from a specific past checkpoint
old_snapshot = list(graph.get_state_history(config))[1]   # second-most-recent
time_travel_config = old_snapshot.config

result = graph.invoke({"count": 0}, config=time_travel_config)
print("Resumed from checkpoint:", result)
```

### `update_state` — inject values as if a node ran

```python
# Manually set state between invocations (useful for testing or corrections)
graph.update_state(
    config,
    {"count": 100},          # values to write
    as_node="bump",          # pretend this update came from the "bump" node
)

result = graph.invoke({"count": 0}, config=config)
print(result["count"])   # 101 (100 from manual update + 1 from bump)
```

---

## Long-Term Memory: Store

A `Store` provides **cross-thread, cross-session** key/value storage with optional vector search. Unlike a checkpointer (which is thread-scoped), data in a store is shared across any number of conversation threads and graph runs.

### Data model

| Concept | Description |
|---|---|
| `namespace: tuple[str, ...]` | Hierarchical path, e.g. `("users", "alice", "prefs")` |
| `key: str` | Unique identifier within the namespace |
| `value: dict[str, Any]` | JSON-serializable payload |
| `Item` | Returned by `get`/`search`. Fields: `value`, `key`, `namespace`, `created_at`, `updated_at` |
| `SearchItem(Item)` | Returned by `search`. Adds `score: float | None` |

### `InMemoryStore` (development)

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# --- Put ---
store.put(("users", "alice"), "prefs", {"theme": "dark", "lang": "en"})
store.put(("users", "alice"), "profile", {"name": "Alice", "joined": "2025-01-15"})

# --- Get ---
item = store.get(("users", "alice"), "prefs")
if item:
    print(item.value)      # {'theme': 'dark', 'lang': 'en'}
    print(item.namespace)  # ('users', 'alice')
    print(item.key)        # 'prefs'

# --- Search (filter by field value) ---
hits = store.search(("users", "alice"), filter={"theme": "dark"})
for hit in hits:
    print(hit.key, hit.value)

# --- List namespaces ---
namespaces = store.list_namespaces(prefix=("users",))
print(namespaces)  # [('users', 'alice')]

# --- Delete ---
store.delete(("users", "alice"), "prefs")
```

### `InMemoryStore` with vector search

For semantic recall, pass an `index` configuration with a callable that returns embeddings:

```python
from langgraph.store.memory import InMemoryStore

# Any callable (list[str]) -> list[list[float]] works as the embed function.
# This example uses a toy embedding; in production use a real model.
def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic toy embeddings for testing — replace with a real model."""
    return [[len(t) / 100.0, hash(t) % 1000 / 1000.0] for t in texts]

store = InMemoryStore(
    index={
        "dims": 2,              # must match embedding dimension
        "embed": fake_embed,
        "fields": ["text"],     # which value fields to embed (default: ["$"] = whole value)
    }
)

store.put(("docs",), "doc1", {"text": "Python async concurrency guide"})
store.put(("docs",), "doc2", {"text": "TypeScript generics tutorial"})
store.put(("docs",), "doc3", {"text": "Python data classes and typing"})

# Semantic search — returns SearchItem objects with a `score` field
results = store.search(("docs",), query="python programming", limit=2)
for r in results:
    print(f"score={r.score:.3f} key={r.key}  text={r.value['text']}")
```

For production use with real embeddings:

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

store = InMemoryStore(index={"dims": 1536, "embed": embed_texts})
```

---

## Injecting Store into Nodes

Use the `runtime` parameter to access the store from inside any node:

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class UserContext:
    user_id: str


class ChatState(TypedDict):
    message: str
    response: str


store = InMemoryStore()


def personalization_node(state: ChatState, runtime: Runtime[UserContext]) -> dict:
    """Load per-user preferences from the store and write back after each turn."""
    user_id = runtime.context.user_id
    namespace = ("users", user_id, "prefs")

    # Load preferences
    prefs_item = runtime.store.get(namespace, "theme") if runtime.store else None
    prefs = prefs_item.value if prefs_item else {"theme": "default"}

    # Build a response using the preferences
    response = f"[theme={prefs['theme']}] Echo: {state['message']}"

    # Save updated interaction count
    if runtime.store:
        count_item = runtime.store.get(namespace, "count")
        count = count_item.value["n"] + 1 if count_item else 1
        runtime.store.put(namespace, "count", {"n": count})

    return {"response": response}


graph = (
    StateGraph(ChatState, context_schema=UserContext)
    .add_node("personalize", personalization_node)
    .add_edge(START, "personalize")
    .add_edge("personalize", END)
    .compile(checkpointer=InMemorySaver(), store=store)
)

# Pre-populate Alice's theme preference
store.put(("users", "alice", "prefs"), "theme", {"theme": "dark"})

cfg = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(
    {"message": "Hello!", "response": ""},
    cfg,
    context=UserContext(user_id="alice"),
)
print(result["response"])   # [theme=dark] Echo: Hello!
```

### Using `InjectedStore` in tools

When a tool needs store access, annotate the parameter with `InjectedStore` (requires `langchain-core >= 0.3.8`):

```python
from typing import Annotated, Any
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore


@tool
def remember(key: str, value: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Save a fact to long-term memory."""
    store.put(("memory",), key, {"value": value})
    return f"Remembered: {key} = {value}"


@tool
def recall(key: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Retrieve a previously stored fact."""
    item = store.get(("memory",), key)
    return item.value["value"] if item else f"Nothing stored for '{key}'"


# The `store` parameter is injected automatically — it is invisible to the LLM.
# Compile the graph with `store=` for injection to work:
#   graph = builder.compile(store=InMemoryStore())
```

---

## Complete Multi-Turn Memory Example

A full working graph that stores conversation summaries and user preferences in the store, and persists per-thread state in the checkpointer:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from dataclasses import dataclass


@dataclass
class AppContext:
    user_id: str


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_summary: str


store = InMemoryStore()
checkpointer = InMemorySaver()


def load_memories(state: AppState, runtime: Runtime[AppContext]) -> dict:
    """Pull per-user memories from the cross-thread store."""
    if not runtime.store:
        return {}
    uid = runtime.context.user_id
    mem_item = runtime.store.get(("users", uid), "memories")
    memories = mem_item.value.get("text", "") if mem_item else ""
    # Inject a synthetic "system" message with the user's memories
    if memories:
        return {
            "messages": [HumanMessage(content=f"[Your memories about me: {memories}]")]
        }
    return {}


def respond(state: AppState, runtime: Runtime[AppContext]) -> dict:
    """Generate a response (placeholder — replace with a real LLM call)."""
    last = state["messages"][-1]
    reply = f"You said: '{last.content}'. (Memory: {state.get('session_summary', 'none')})"
    return {"messages": [AIMessage(content=reply)]}


def save_memories(state: AppState, runtime: Runtime[AppContext]) -> dict:
    """Persist a summary of this turn to the cross-thread store."""
    if not runtime.store:
        return {}
    uid = runtime.context.user_id
    # Build a naive summary (replace with an LLM extraction step in production)
    user_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    summary = "; ".join(user_msgs[-3:])  # keep the last 3 user messages as memory
    runtime.store.put(("users", uid), "memories", {"text": summary})
    return {"session_summary": summary}


graph = (
    StateGraph(AppState, context_schema=AppContext)
    .add_node("load", load_memories)
    .add_node("respond", respond)
    .add_node("save", save_memories)
    .add_edge(START, "load")
    .add_edge("load", "respond")
    .add_edge("respond", "save")
    .add_edge("save", END)
    .compile(checkpointer=checkpointer, store=store)
)

ctx = AppContext(user_id="alice")

# Thread 1 — first conversation
cfg1 = {"configurable": {"thread_id": "thread-1"}}
r1 = graph.invoke({"messages": [HumanMessage("I love hiking!")], "session_summary": ""}, cfg1, context=ctx)
print(r1["messages"][-1].content)

# Thread 2 — completely different conversation, but Alice's memories carry over
cfg2 = {"configurable": {"thread_id": "thread-2"}}
r2 = graph.invoke({"messages": [HumanMessage("What do you know about me?")], "session_summary": ""}, cfg2, context=ctx)
print(r2["messages"][-1].content)  # The memory "I love hiking!" should appear
```

---

## Cross-Thread Memory Pattern

The core pattern for cross-thread memory is to use **different namespaces** for user-specific vs thread-specific data:

```python
# Thread-scoped data: live in the checkpointer (auto-managed)
# (no explicit store needed — the checkpointer handles this)

# Cross-thread data: use a store with a user-scoped namespace
def cross_thread_node(state: AppState, runtime: Runtime[AppContext]) -> dict:
    uid = runtime.context.user_id

    # User-scoped namespace — same for ALL threads belonging to this user
    user_ns = ("users", uid)

    # Thread-scoped namespace — different for each conversation
    # thread_id = runtime.execution_info.thread_id if runtime.execution_info else "unknown"
    # thread_ns = ("threads", uid, thread_id)

    if not runtime.store:
        return {}

    # Read cross-thread context
    profile = runtime.store.get(user_ns, "profile")
    user_name = profile.value.get("name", "User") if profile else "User"

    # Write back updated context (shared across threads)
    turn_count_item = runtime.store.get(user_ns, "turns")
    turns = (turn_count_item.value.get("n", 0) + 1) if turn_count_item else 1
    runtime.store.put(user_ns, "turns", {"n": turns})

    return {"response": f"Welcome back, {user_name}! Turn #{turns}."}
```

Key insight: the checkpointer manages **thread-local** state (per `thread_id`). The store manages **cross-thread** state (keyed by `user_id` or any other namespace that spans threads).

---

## Async Store Usage

All `BaseStore` methods have `async` variants prefixed with `a`:

```python
# Sync
item = store.get(namespace, key)
store.put(namespace, key, value)
results = store.search(namespace_prefix, query="...")

# Async
item = await store.aget(namespace, key)
await store.aput(namespace, key, value)
results = await store.asearch(namespace_prefix, query="...")

# Async node using await store operations
async def async_memory_node(state: AppState, runtime: Runtime[AppContext]) -> dict:
    if not runtime.store:
        return {}
    uid = runtime.context.user_id
    item = await runtime.store.aget(("users", uid), "prefs")
    prefs = item.value if item else {}
    # ...do async work...
    await runtime.store.aput(("users", uid), "prefs", {**prefs, "last_seen": "now"})
    return {}
```

---

## Store TTL (Time to Live)

Some store implementations support TTL per item. With `InMemoryStore`, pass `ttl=` (in minutes) per put operation:

```python
# Data expires after 60 minutes
store.put(("sessions",), "token_xyz", {"data": "..."}, ttl=60)

# Check if a store supports TTL
print(store.supports_ttl)   # True for InMemoryStore (with index or explicit ttl_config)
```

For `PostgresStore`, configure TTL at the store level and sweep expired rows with `abackground_tasks()`.

---

## Quick-Reference: Checkpointer vs Store

| Feature | Checkpointer | Store |
|---|---|---|
| Scope | Single `thread_id` | Any namespace (`user_id`, global, etc.) |
| Data shape | Full graph state snapshot | `{namespace, key, value}` |
| Access | Automatic — graph reads/writes it | Manual — node calls `store.get/put/search` |
| History | Yes — `get_state_history()` | No |
| Semantic search | No | Yes (with `index=` config) |
| Cross-thread | No | Yes |
| Required for `interrupt()` | Yes | No |

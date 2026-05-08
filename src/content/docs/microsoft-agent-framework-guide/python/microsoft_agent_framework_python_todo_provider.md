---
title: "Microsoft Agent Framework (Python) — TodoProvider"
description: "Give agents a built-in task list: TodoProvider, TodoItem, TodoSessionStore, and TodoFileStore. All verified against agent-framework-core 1.3.0."
framework: microsoft-agent-framework
language: python
---

# TodoProvider — Python

`TodoProvider` is a `ContextProvider` that equips an agent with a structured todo list it can manage across a session. Instead of the agent having to maintain its own ad-hoc task tracking in plain text, the provider injects five tools and a system-prompt section that teach the agent how to break work into trackable items, mark them complete, and clean up when topics change.

Verified against `agent-framework-core==1.3.0` (`agent_framework._harness._todo`). Marked `experimental` — API may evolve.

## Quick start

```python
import asyncio
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful project planning assistant.",
        context_providers=[TodoProvider()],
    )

    session = agent.create_session()

    # The agent can now create and manage todos as part of its responses.
    response = await agent.run(
        "I need to build a REST API with authentication, database access, and documentation.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

The provider adds a system-prompt block explaining the todo pattern, injects the current todo list as a user message before each turn, and registers five tools the model can call freely.

## The five tools the agent gets

| Tool | Arguments | What it does |
|---|---|---|
| `add_todos` | `todos: [{title, description?}]` | Add one or more items in one call |
| `complete_todos` | `ids: [int]` | Mark items complete by ID |
| `remove_todos` | `ids: [int]` | Delete items no longer needed |
| `get_remaining_todos` | — | Return only incomplete items |
| `get_all_todos` | — | Return all items including completed |

All five use `approval_mode="never_require"` — the agent calls them freely without pausing for human approval. If you want oversight on specific operations, wrap the provider with middleware (see [Middleware page](./microsoft_agent_framework_python_middleware/)).

## Backing stores

### `TodoSessionStore` (default)

Stores todo state directly in `AgentSession.state`. State lives as long as the session object does — in-process and in-memory. Good for single-process servers.

```python
from agent_framework import Agent, TodoProvider, TodoSessionStore
from agent_framework.openai import OpenAIChatClient

# TodoSessionStore is the default; you don't need to specify it.
agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a project planner.",
    context_providers=[TodoProvider(store=TodoSessionStore())],
)
```

Because the state lives inside `AgentSession.state`, it serializes with the session via `session.to_dict()` and restores via `AgentSession.from_dict(snapshot)`. This means you can persist or transfer sessions across process restarts and todos survive the round-trip.

### `TodoFileStore` — persistent JSON

`TodoFileStore` writes one JSON file per session and source ID to a directory tree. Useful when sessions span processes or need long-lived durability.

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoFileStore, TodoProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    store = TodoFileStore(
        base_path="./todo-storage",
        # owner_state_key links the session to a stable owner directory.
        # If omitted, all sessions share the same directory root.
        owner_state_key="user_id",
        kind="todos",             # subdirectory name under each owner
        state_filename="todos.json",  # file name per session
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a planning assistant.",
        context_providers=[TodoProvider(store=store)],
    )

    # session.state["user_id"] routes file I/O to ./todo-storage/<user_id>/todos/
    session = AgentSession(session_id="sess-001")
    session.state["user_id"] = "user-42"

    r1 = await agent.run("Plan my migration to cloud.", session=session)
    # Todos are now on disk under ./todo-storage/user-42/todos/sess-001/todos.json

    # Resume in a new process — recreate the session with the same IDs.
    resumed = AgentSession(session_id="sess-001")
    resumed.state["user_id"] = "user-42"
    r2 = await agent.run("What have I planned so far?", session=resumed)
    print(r2.text)


asyncio.run(main())
```

`TodoFileStore` writes atomically (write to a temp file, then `os.replace`) so a crash mid-write never leaves a corrupted state file.

## Multiple providers per agent

You can attach multiple `TodoProvider` instances with different `source_id` values — for example, one for immediate tasks and one for a longer-term backlog:

```python
from agent_framework import Agent, TodoFileStore, TodoProvider
from agent_framework.openai import OpenAIChatClient


backlog_store = TodoFileStore(base_path="./todos", owner_state_key="user_id", kind="backlog")
sprint_store = TodoFileStore(base_path="./todos", owner_state_key="user_id", kind="sprint")

agent = Agent(
    client=OpenAIChatClient(),
    instructions=(
        "You maintain two task lists: a sprint list for this week's work "
        "and a backlog for future items."
    ),
    context_providers=[
        TodoProvider(source_id="sprint", store=sprint_store),
        TodoProvider(source_id="backlog", store=backlog_store),
    ],
)
```

Tool names (`add_todos`, `complete_todos`, etc.) are the same regardless of `source_id`. The `source_id` controls only storage isolation — each value gets its own separate state bucket on disk, so `"sprint"` and `"backlog"` never share data even though their tools share the same names.

## Custom instructions

Override the default system-prompt block that explains todo management:

```python
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient

MINIMAL_INSTRUCTIONS = (
    "You have a todo list. Use `add_todos` to track work, "
    "`complete_todos` when done, and `get_remaining_todos` to check progress."
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a concise task tracker.",
    context_providers=[
        TodoProvider(instructions=MINIMAL_INSTRUCTIONS),
    ],
)
```

## Reading the todo state directly

The backing store exposes `load_items` for reading todos outside the agent loop — useful for dashboards, audit logs, or pre-populating items from an external system:

```python
import asyncio
from agent_framework import AgentSession, TodoFileStore, TodoProvider


async def show_todos(session: AgentSession) -> None:
    store = TodoFileStore(base_path="./todos", owner_state_key="user_id")
    items = await store.load_items(session, source_id="todo")
    for item in items:
        status = "✓" if item.is_complete else "○"
        print(f"  [{status}] #{item.id} {item.title}")
        if item.description:
            print(f"        {item.description}")
```

## Pre-populating todos before an agent run

Use `store.save_state` to seed the list from external ticket systems or onboarding flows:

```python
import asyncio
from agent_framework import AgentSession, TodoFileStore, TodoItem


async def seed_from_jira(session: AgentSession, jira_tickets: list[dict]) -> None:
    store = TodoFileStore(base_path="./todos", owner_state_key="user_id")

    items = [
        TodoItem(id=i + 1, title=t["summary"], description=t.get("description"))
        for i, t in enumerate(jira_tickets)
    ]
    await store.save_state(
        session,
        items,
        next_id=len(items) + 1,
        source_id="todo",
    )
```

## `TodoItem` and `TodoInput` reference

```python
from agent_framework import TodoItem, TodoInput

# TodoInput — what the agent passes to add_todos
todo_input = TodoInput(
    title="Write unit tests",            # required, non-empty
    description="Cover auth module",     # optional
)

# TodoItem — what comes back from the store
item = TodoItem(
    id=1,
    title="Write unit tests",
    description="Cover auth module",
    is_complete=False,
)

# Serialization round-trip
snapshot = item.to_dict()          # {"id": 1, "title": ..., ...}
restored = TodoItem.from_dict(snapshot)
assert restored == item
```

`TodoItem` slots: `id` (int), `title` (str), `description` (str | None), `is_complete` (bool).

## Combining with other providers

`TodoProvider` composes naturally with other context providers on the same agent. A common pairing is `MemoryContextProvider` (durable cross-session memory) plus `TodoProvider` (per-session task tracking):

```python
from agent_framework import Agent, MemoryContextProvider, MemoryFileStore, TodoFileStore, TodoProvider
from agent_framework.openai import OpenAIChatClient


memory_store = MemoryFileStore(base_path="./memory", owner_state_key="user_id")
todo_store   = TodoFileStore(base_path="./todos",  owner_state_key="user_id")

agent = Agent(
    client=OpenAIChatClient(),
    instructions=(
        "You are a long-running project assistant. "
        "Use your memory to recall past decisions and your todo list to track current work."
    ),
    context_providers=[
        MemoryContextProvider(store=memory_store, recent_turns=3),
        TodoProvider(store=todo_store),
    ],
)
```

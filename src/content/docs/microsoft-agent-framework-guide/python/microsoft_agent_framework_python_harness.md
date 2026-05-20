---
title: "Microsoft Agent Framework (Python) — HARNESS Providers"
description: "TodoProvider, AgentModeProvider, MemoryContextProvider — experimental context providers giving agents self-managed task lists, mode-switching, and long-term memory. Verified against agent-framework-core 1.5.0."
framework: microsoft-agent-framework
language: python
---

# HARNESS Providers — Python

> **Experimental.** `TodoProvider`, `AgentModeProvider`, `MemoryContextProvider`, and their backing stores are marked `ExperimentalFeature.HARNESS` in 1.5.0. The APIs are functional but may change between minor releases. Import warnings appear on first use.

HARNESS providers are a family of `ContextProvider` implementations that give agents durable, session-scoped capabilities beyond conversation history:

| Provider / Helper | What it adds to the agent |
|---|---|
| `TodoProvider` | Five tools (`add_todos`, `complete_todos`, `remove_todos`, `get_remaining_todos`, `get_all_todos`) + instructions for autonomous task tracking |
| `AgentModeProvider` | Two tools (`get_mode`, `set_mode`) + instructions for operating in distinct modes (plan / execute, or any custom set) |
| `get_agent_mode` / `set_agent_mode` | External helpers to read and flip the mode from application code without waiting for the agent to call `set_mode` |

Both providers plug in via `context_providers=` and store state inside `AgentSession.state` (or optionally on disk).

---

## `TodoProvider` — autonomous task tracking

`TodoProvider` injects a structured task list that the agent manages itself. The agent plans, tracks, and completes todo items across multiple turns within the same session.

### Quickstart — in-session todos

```python
import asyncio
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a project-planning assistant.",
    context_providers=[TodoProvider()],   # state lives in AgentSession by default
)


async def main() -> None:
    session = agent.create_session()

    # Turn 1 — the agent adds todos as it plans
    r1 = await agent.run(
        "Plan a three-day product launch: marketing, engineering, and support tasks.",
        session=session,
    )
    print("Plan:\n", r1.text)

    # Turn 2 — the agent checks get_remaining_todos and marks items done as it works
    r2 = await agent.run(
        "Draft the engineering checklist and mark those tasks complete.",
        session=session,
    )
    print("Progress:\n", r2.text)


asyncio.run(main())
```

The agent receives default instructions that explain all five tools. It manages the list autonomously — no application code needs to drive it.

### `TodoItem` dataclass

Every todo the agent creates becomes a `TodoItem`. The fields:

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Auto-assigned integer ID. Use this in `complete_todos([id])` and `remove_todos([id])`. |
| `title` | `str` | Short task description (non-empty, stripped). |
| `description` | `str \| None` | Optional longer description. |
| `is_complete` | `bool` | `True` after the agent calls `complete_todos`. |

```python
from agent_framework import TodoItem

item = TodoItem(id=1, title="Write unit tests", description="Cover the auth module", is_complete=False)
print(item)   # TodoItem(id=1, title='Write unit tests', ...)

# Serialise / deserialise (used by the stores internally)
d = item.to_dict()
restored = TodoItem.from_dict(d)
assert restored == item
```

### `TodoInput` — what the agent passes to `add_todos`

When the agent calls `add_todos`, it produces a list of `TodoInput` objects (or raw dicts with `title` and optional `description`). You rarely construct `TodoInput` yourself, but it is useful in tests where you want to pre-seed the store:

```python
from agent_framework import TodoInput

task = TodoInput(title="Review PR #42", description="Focus on error handling")
print(task.title)        # "Review PR #42"
print(task.to_dict())    # {"title": "Review PR #42", "description": "Review PR #42"}
```

### Inspecting todos from application code

Read the task list without going through the agent. This is useful for dashboards, webhook handlers, and CI status checks:

```python
import asyncio
from agent_framework import (
    Agent,
    AgentSession,
    TodoFileStore,
    TodoProvider,
    TodoSessionStore,
)
from agent_framework.openai import OpenAIChatClient


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a task assistant.",
    context_providers=[TodoProvider()],
)


async def main() -> None:
    session = agent.create_session(session_id="sprint-22")
    await agent.run("Add tasks: write tests, review PR, deploy to staging.", session=session)

    # Read from in-memory store — the same session object holds the state
    store = TodoSessionStore()
    items, _ = await store.load_state(session, source_id="todo")

    for item in items:
        status = "✓" if item.is_complete else "·"
        desc = f" — {item.description}" if item.description else ""
        print(f"  {status} [{item.id}] {item.title}{desc}")

    pending = [i for i in items if not i.is_complete]
    done    = [i for i in items if i.is_complete]
    print(f"\n{len(pending)} pending, {len(done)} complete")


asyncio.run(main())
```

### `TodoFileStore` — persist todos across process restarts

Swap `TodoSessionStore` (in-memory) for `TodoFileStore` when todos must survive restarts or span multiple processes:

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoFileStore, TodoProvider
from agent_framework.openai import OpenAIChatClient


# Todos written to ./todos/<session_id>/todos.json
store = TodoFileStore(base_path="./todos")

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a long-running task assistant.",
    context_providers=[TodoProvider(store=store)],
)


async def main() -> None:
    # First run — agent creates todos and persists them
    session = agent.create_session(session_id="project-launch-42")
    await agent.run("Break the product launch into 8 concrete tasks.", session=session)

    # Second run (new process, same session_id) — agent resumes with existing todos
    session2 = agent.create_session(session_id="project-launch-42")
    r = await agent.run("What tasks are still pending?", session=session2)
    print(r.text)


asyncio.run(main())
```

#### Multi-user / multi-tenant partitioning with `owner_state_key`

Pass `owner_state_key` to shard todos by user or tenant. The store derives a directory segment from `session.state[owner_state_key]`:

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoFileStore, TodoProvider
from agent_framework.openai import OpenAIChatClient


# Todos written to ./todos/<user_id>/<session_id>/todos.json
store = TodoFileStore(base_path="./todos", owner_state_key="user_id")

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a task assistant.",
    context_providers=[TodoProvider(store=store)],
)


async def main() -> None:
    # Alice's session
    alice_session = agent.create_session(session_id="s1")
    alice_session.state["user_id"] = "alice"
    await agent.run("I need to write a report.", session=alice_session)

    # Bob's session — completely separate directory
    bob_session = agent.create_session(session_id="s2")
    bob_session.state["user_id"] = "bob"
    await agent.run("Prepare the slides for Monday.", session=bob_session)

    # Load Alice's todos directly (without running the agent again)
    items, _ = await store.load_state(alice_session, source_id="todo")
    print(f"Alice has {len(items)} todo(s):", [i.title for i in items])


asyncio.run(main())
```

### Custom instructions

Override the default system-prompt block to tune the agent's behaviour — for example restricting which tools it may use or adjusting the tone:

```python
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient


focused_provider = TodoProvider(
    instructions=(
        "## Task List\n\n"
        "You have a task list. Break complex work into ≤5 tasks at a time.\n"
        "- Use `add_todos` when planning.\n"
        "- Use `complete_todos` when a task is done. Never skip this step.\n"
        "- Never remove tasks without the user's explicit instruction.\n"
    ),
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a disciplined sprint assistant.",
    context_providers=[focused_provider],
)
```

### `TodoProvider` constructor reference

```python
TodoProvider(
    source_id: str = "todo",          # key in session.state; change when stacking multiple providers
    *,
    instructions: str | None = None,  # None = use default instructions text
    store: TodoStore | None = None,   # None = TodoSessionStore (in-memory)
)
```

---

## `AgentModeProvider` — plan / execute and custom modes

`AgentModeProvider` lets an agent operate in distinct named modes — each with its own instructions injected at runtime. Two modes ship by default: **plan** (interactive, ask questions) and **execute** (autonomous, minimise interruptions).

### Quickstart — default plan / execute modes

```python
import asyncio
from agent_framework import Agent, AgentModeProvider
from agent_framework.openai import OpenAIChatClient


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a database migration assistant.",
    context_providers=[AgentModeProvider()],   # default modes: "plan" and "execute"
)


async def main() -> None:
    session = agent.create_session()

    # Phase 1 — planning: the agent asks questions and proposes a plan
    r1 = await agent.run(
        "I want to migrate our Postgres schema to add a new 'preferences' JSONB column to the users table.",
        session=session,
    )
    print("Plan phase:\n", r1.text)

    # Phase 2 — user approves and switches to execute; the agent works autonomously
    r2 = await agent.run(
        "The plan looks good. Switch to execute mode and run the migration.",
        session=session,
    )
    print("Execute phase:\n", r2.text)


asyncio.run(main())
```

In **plan** mode the agent is instructed to ask for clarification before acting. In **execute** mode it works autonomously and avoids unnecessary check-ins. Both instructions are injected automatically from the provider.

### Custom mode set

Define your own mode names and descriptions when the defaults don't fit. Mode names come from the `mode_descriptions` mapping keys:

```python
import asyncio
from agent_framework import Agent, AgentModeProvider
from agent_framework.openai import OpenAIChatClient


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a code-review assistant.",
    context_providers=[
        AgentModeProvider(
            default_mode="review",
            mode_descriptions={
                "review": (
                    "Read the diff and identify issues. Do not suggest fixes yet. "
                    "List every issue with a severity label: 🔴 critical, 🟡 minor, 🟢 nit."
                ),
                "suggest": (
                    "For each issue listed in the review, propose a concrete code fix. "
                    "Show the diff inline."
                ),
                "approve": (
                    "All issues are resolved. Write the final approval comment and call set_mode('done')."
                ),
                "done": "Review complete. Do not take any further actions.",
            },
        )
    ],
)


async def main() -> None:
    session = agent.create_session()

    diff = "--- a/auth.py\n+++ b/auth.py\n@@ -10,6 +10,7 @@ def login(username, password):\n+    user = db.query(username)  # SQL injection risk\n     ..."

    # Review phase — model annotates issues only, no fixes
    await agent.run(f"Review this diff:\n\n{diff}", session=session)

    # Suggest phase — model proposes fixes
    await agent.run("Switch to suggest mode and fix the issues.", session=session)

    # Approve phase — model writes the approval
    await agent.run("All fixes look good. Approve the PR.", session=session)


asyncio.run(main())
```

### Reading and setting mode from application code

`get_agent_mode` and `set_agent_mode` are standalone helpers that read/write the session state directly — no agent run needed. Use them to:
- Read current mode for a dashboard or audit log
- Trigger an automated mode transition from a CI gate or webhook

```python
import asyncio
from agent_framework import Agent, AgentModeProvider, AgentSession, get_agent_mode, set_agent_mode
from agent_framework.openai import OpenAIChatClient

MODES = ["review", "suggest", "approve", "done"]

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a code-review assistant.",
    context_providers=[AgentModeProvider(default_mode="review", mode_descriptions={m: m for m in MODES})],
)


async def main() -> None:
    session = agent.create_session(session_id="pr-review-88")

    # Start in review mode (default)
    current = get_agent_mode(session, default_mode="review", available_modes=MODES)
    print(f"Initial mode: {current}")   # → "review"

    # External event: CI passed — advance the mode to suggest automatically
    set_agent_mode(session, "suggest", available_modes=MODES)
    print(f"After CI gate: {get_agent_mode(session, available_modes=MODES)}")   # → "suggest"

    # The next agent.run() will see the injected mode-change notification in its context
    r = await agent.run("CI has passed. What should we do next?", session=session)
    print(r.text)


asyncio.run(main())
```

When you call `set_agent_mode` externally, the provider injects a `user`-role message on the next `run()` announcing the switch. This is necessary because system instructions alone may not override a model that has seen its own `set_mode` tool call earlier in the history.

### `AgentModeProvider` constructor reference

```python
AgentModeProvider(
    source_id: str = "agent_mode",   # session.state partition key
    *,
    default_mode: str | None = None, # first key in mode_descriptions if omitted
    mode_descriptions: Mapping[str, str] | None = None,  # None = {"plan": ..., "execute": ...}
    instructions: str | None = None, # override prompt block; supports {available_modes} and {current_mode}
)
```

---

## Combining `TodoProvider` and `AgentModeProvider`

The two providers compose naturally: the agent plans in **plan** mode (adding todos), switches to **execute** mode, then works through the list autonomously — completing todos as it goes.

```python
import asyncio
from agent_framework import Agent, AgentModeProvider, AgentSession, TodoProvider, set_agent_mode
from agent_framework.openai import OpenAIChatClient


agent = Agent(
    client=OpenAIChatClient(),
    instructions=(
        "You are an autonomous engineering assistant.\n"
        "In plan mode: break the user's request into concrete todo items before doing anything.\n"
        "In execute mode: work through your todo list, marking each item complete as you finish."
    ),
    context_providers=[
        TodoProvider(),
        AgentModeProvider(),   # default plan/execute
    ],
)


async def main() -> None:
    session = agent.create_session()

    # --- Planning turn ---
    r_plan = await agent.run(
        "I need to add JWT authentication to our FastAPI app.",
        session=session,
    )
    print("Planning:\n", r_plan.text)

    # --- Simulated external approval: advance to execute mode ---
    set_agent_mode(session, "execute")

    # --- Execution turn ---
    r_exec = await agent.run(
        "Good plan. Now execute it.",
        session=session,
    )
    print("Execution:\n", r_exec.text)


asyncio.run(main())
```

### Stacking with file persistence

Pair `TodoFileStore` with `AgentModeProvider` for workflows that span multiple process restarts — the todo state survives, and the mode is stored in `AgentSession.state` which you can also persist using `FileHistoryProvider`:

```python
import asyncio
from agent_framework import (
    Agent,
    AgentModeProvider,
    AgentSession,
    FileHistoryProvider,
    TodoFileStore,
    TodoProvider,
    get_agent_mode,
    set_agent_mode,
)
from agent_framework.openai import OpenAIChatClient


store = TodoFileStore(base_path="./workspace/todos")

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a long-running task assistant.",
    context_providers=[
        FileHistoryProvider(storage_path="./workspace/history"),  # persist conversation
        TodoProvider(store=store),                                 # persist todos
        AgentModeProvider(),                                       # mode stored in session.state
    ],
)


async def run_planning_session() -> None:
    """First process: plan and create todos."""
    session = agent.create_session(session_id="task-001")
    await agent.run(
        "I need to migrate our Redis cache from v5 to v7. Create a detailed plan.",
        session=session,
    )
    # Transition to execute and save state; mode is in session.state — serialise if needed
    set_agent_mode(session, "execute")
    print(f"Mode after planning: {get_agent_mode(session)}")


async def run_execution_session() -> None:
    """Second process (same session_id): execute the plan."""
    session = agent.create_session(session_id="task-001")
    # History loads from disk; todos load from TodoFileStore
    # Mode state must be restored from your own persistence layer (e.g. a DB or a JSON file)
    set_agent_mode(session, "execute", available_modes=["plan", "execute"])

    r = await agent.run("What tasks are remaining?", session=session)
    print(r.text)


asyncio.run(run_planning_session())
# In a new process:
asyncio.run(run_execution_session())
```

> **Note on mode persistence:** `AgentModeProvider` stores mode in `AgentSession.state`. If you need mode to survive across processes, persist `session.state` yourself — for example, write it to your database after each turn and reload it when you create the next session with `session.state.update(stored_state)`.

---

## Mode-triggered todo creation — advanced pattern

Use a custom `AgentModeProvider` instruction that ties todo creation to mode entry. The agent automatically creates todos when it enters **plan** mode and starts executing them when it switches to **execute**:

```python
import asyncio
from agent_framework import Agent, AgentModeProvider, AgentSession, TodoProvider, set_agent_mode
from agent_framework.openai import OpenAIChatClient


PLAN_INSTRUCTIONS = (
    "## Plan Mode\n\n"
    "You are in planning mode. Before doing anything else:\n"
    "1. Break the user's request into 3–7 concrete, actionable tasks.\n"
    "2. Add them with `add_todos` (include a description for each).\n"
    "3. Present the full list to the user for review.\n"
    "Do NOT execute any tasks. Ask the user to confirm before switching modes."
)

EXECUTE_INSTRUCTIONS = (
    "## Execute Mode\n\n"
    "You are in execution mode. Work autonomously:\n"
    "1. Call `get_remaining_todos` at the start of each turn.\n"
    "2. Pick the next incomplete task and execute it.\n"
    "3. Mark it done with `complete_todos([id])` immediately after.\n"
    "4. Do not ask the user questions. Make reasonable decisions independently.\n"
    "5. When all todos are complete, call `set_mode('review')`."
)

REVIEW_INSTRUCTIONS = (
    "## Review Mode\n\n"
    "All tasks are complete. Summarise what was done and ask the user if they want any changes."
)


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are an autonomous software engineering assistant.",
    context_providers=[
        TodoProvider(),
        AgentModeProvider(
            default_mode="plan",
            mode_descriptions={
                "plan":    PLAN_INSTRUCTIONS,
                "execute": EXECUTE_INSTRUCTIONS,
                "review":  REVIEW_INSTRUCTIONS,
            },
        ),
    ],
)


async def main() -> None:
    session = agent.create_session()

    # Trigger planning
    await agent.run(
        "Add role-based access control (RBAC) to the admin API.",
        session=session,
    )

    # User approves — advance to execute
    set_agent_mode(session, "execute")
    r = await agent.run("Plan approved. Go ahead.", session=session)
    print(r.text)


asyncio.run(main())
```

---

## Reference — `TodoStore` protocol

Both `TodoSessionStore` and `TodoFileStore` implement `TodoStore`:

```python
class TodoStore(ABC):
    async def load_state(self, session: AgentSession, *, source_id: str) -> tuple[list[TodoItem], int]:
        """Return (items, next_available_id)."""

    async def save_state(self, session: AgentSession, items: list[TodoItem], *, next_id: int, source_id: str) -> None:
        """Persist items and the next_id counter."""

    async def load_items(self, session: AgentSession, *, source_id: str) -> list[TodoItem]:
        """Convenience: load_state()[0]."""
```

Subclass `TodoStore` to use any backend — Redis, Cosmos DB, S3. The only I/O happens in `load_state` and `save_state`; the provider handles locking and tool wiring automatically.

```python
from agent_framework import AgentSession, TodoItem, TodoStore


class RedisTodoStore(TodoStore):
    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(redis_url)

    async def load_state(self, session: AgentSession, *, source_id: str) -> tuple[list[TodoItem], int]:
        key = f"todos:{source_id}:{session.session_id}"
        raw = await self._redis.get(key)
        if raw is None:
            return [], 1
        import json
        payload = json.loads(raw)
        items = [TodoItem.from_dict(i) for i in payload.get("items", [])]
        next_id = payload.get("next_id", 1)
        return items, next_id

    async def save_state(
        self, session: AgentSession, items: list[TodoItem], *, next_id: int, source_id: str
    ) -> None:
        import json
        key = f"todos:{source_id}:{session.session_id}"
        payload = json.dumps({
            "items": [i.to_dict(exclude_none=False) for i in items],
            "next_id": next_id,
        })
        await self._redis.set(key, payload)
```

---

## `MemoryContextProvider` — durable long-term memory

`MemoryContextProvider` gives agents cross-session recall. It automatically extracts durable facts from transcripts after each session and injects the most relevant topics at the start of future sessions — the agent accumulates knowledge without keeping entire conversation histories in the context window.

It is a `HistoryProvider` (subtype of `ContextProvider`) and handles both history persistence and memory extraction in a single provider.

### How it works

1. **Injection** — at `before_run`, the provider loads `MEMORY.md` (a keyword-indexed pointer list) and selects the most relevant topic files based on the current conversation. It prepends those as context along with a configurable number of recent turns.
2. **Extraction** — at `after_run`, the provider saves the new turn to a transcript file. When the extraction threshold is met an LLM summarises new durable facts into topic files and updates the index.
3. **Consolidation** — periodically (default: every 24 hours, after at least 5 sessions) a consolidation pass merges stale or duplicate topic files to keep the index compact.

### Quickstart

`MemoryFileStore` partitions memory by owner. Set `session.state[owner_state_key]` before the first `run()` call.

```python
import asyncio
from datetime import timedelta
from agent_framework import Agent, MemoryContextProvider, MemoryFileStore
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

store = MemoryFileStore(
    base_path="./memory",
    owner_state_key="user_id",   # session.state["user_id"] partitions per user
)

memory = MemoryContextProvider(
    store=store,
    recent_turns=2,               # inject the 2 most recent conversation turns
    selection_limit=3,            # load at most 3 topic files per session
    max_extractions=5,            # extract at most 5 facts per session
    consolidation_interval=timedelta(hours=24),
    consolidation_min_sessions=5,
)

agent = Agent(
    client=client,
    instructions="You are a personal assistant with long-term memory.",
    context_providers=[memory],
)


async def main() -> None:
    # Session 1 — agent learns a preference
    s1 = agent.create_session(session_id="user-42-s1")
    s1.state["user_id"] = "user-42"       # required — drives per-user file partitioning
    await agent.run("I prefer concise bullet-point answers.", session=s1)

    # Session 2 — preference is recalled from the topic store
    s2 = agent.create_session(session_id="user-42-s2")
    s2.state["user_id"] = "user-42"
    r = await agent.run("Summarise the benefits of asyncio.", session=s2)
    print(r.text)   # likely uses bullet points — remembered from session 1


asyncio.run(main())
```

### Multi-user isolation

Every distinct `owner_state_key` value gets its own subtree under `base_path`. Two users share a single `agent` and `store` instance with no cross-contamination:

```python
async def handle_request(user_id: str, message: str) -> str:
    session = agent.create_session()
    session.state["user_id"] = user_id    # → ./memory/<user_id>/memory/MEMORY.md
    return (await agent.run(message, session=session)).text
```

### Using a cheaper model for consolidation

The consolidation pass summarises and merges topics with an LLM call. Pass `consolidation_client=` to use a faster or cheaper model for that background task without affecting the main agent's model:

```python
from agent_framework.openai import OpenAIChatClient

main_client   = OpenAIChatClient(model="gpt-4o")
cheap_client  = OpenAIChatClient(model="gpt-4o-mini")

memory = MemoryContextProvider(
    store=store,
    consolidation_client=cheap_client,  # only used during topic consolidation
)

agent = Agent(client=main_client, context_providers=[memory])
```

### Filtering transcripts before save — `history_message_filter`

Pass a `history_message_filter` callback to redact or drop messages before they are written to the transcript store. The callback receives each `Message` and returns either a modified copy or `None` to drop the message entirely:

```python
from agent_framework import Message, MemoryContextProvider, MemoryFileStore
import re

PII_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")   # crude SSN pattern

def redact_pii(message: Message) -> Message | None:
    """Return message with SSNs redacted, or None to drop the message."""
    new_contents = []
    for content in message.contents:
        if isinstance(content, str):
            new_contents.append(PII_PATTERN.sub("[REDACTED]", content))
        else:
            new_contents.append(content)
    from dataclasses import replace
    return replace(message, contents=new_contents)

memory = MemoryContextProvider(
    store=MemoryFileStore(base_path="./memory", owner_state_key="user_id"),
    history_message_filter=redact_pii,
)
```

### Injecting recent turns alongside durable memory

`recent_turns` injects the most recent N turns from the transcript (not just the current session) as additional context. This gives the agent short-term coherence across sessions without blowing up the context window:

```python
memory = MemoryContextProvider(
    store=store,
    recent_turns=3,            # inject the 3 most recent turns
    load_tool_turns=False,     # skip tool-call turns in the recency window
)
```

Set `load_tool_turns=False` when tool-call rounds are noisy and you only want the user/assistant exchanges.

### Inspecting and managing memory files directly

`MemoryFileStore` methods are synchronous — call them from application code (dashboards, admin scripts) without running the agent:

```python
from agent_framework import AgentSession, MemoryFileStore

store = MemoryFileStore(base_path="./memory", owner_state_key="user_id")

session = AgentSession(session_id="admin-view")
session.state["user_id"] = "user-42"

# List all topic records for a user
topics = store.list_topics(session, source_id="memory")
for t in topics:
    print(t.topic, "—", t.summary[:60])

# Fetch a specific topic
record = store.get_topic(session, source_id="memory", topic="preferences")
print(record.to_markdown())

# Delete a stale topic
store.delete_topic(session, source_id="memory", topic="old-project-x")

# Rebuild the MEMORY.md index after manual edits
store.rebuild_index(session, source_id="memory")
```

### Custom `MemoryStore` backend

Subclass `MemoryStore` to back memory in any storage system — a database, blob storage, or vector DB. All abstract methods are **synchronous** (the provider wraps them in `asyncio.to_thread` internally):

```python
from agent_framework import AgentSession, MemoryFileStore, MemoryIndexEntry, MemoryStore, MemoryTopicRecord
from typing import Any
from collections.abc import Mapping
from pathlib import Path


class InMemoryMemoryStore(MemoryStore):
    """Ephemeral in-memory backend — useful for tests."""

    def __init__(self, owner_state_key: str) -> None:
        self._owner_state_key = owner_state_key
        self._topics: dict[str, dict[str, MemoryTopicRecord]] = {}  # owner → topic → record
        self._state: dict[str, dict[str, Any]] = {}

    def get_owner_id(self, session: AgentSession) -> str | None:
        return session.state.get(self._owner_state_key)

    def _owner(self, session: AgentSession) -> str:
        oid = self.get_owner_id(session)
        if oid is None:
            raise RuntimeError(f"session.state[{self._owner_state_key!r}] must be set")
        return str(oid)

    def export_provider_state(self, session: AgentSession) -> dict[str, Any]:
        return {}

    def import_provider_state(self, session: AgentSession, *, state: Mapping[str, Any]) -> None:
        pass

    def list_topics(self, session: AgentSession, *, source_id: str) -> list[MemoryTopicRecord]:
        return list(self._topics.get(self._owner(session), {}).values())

    def get_topic(self, session: AgentSession, *, source_id: str, topic: str) -> MemoryTopicRecord:
        owner_topics = self._topics.get(self._owner(session), {})
        if topic not in owner_topics:
            raise KeyError(f"Topic {topic!r} not found")
        return owner_topics[topic]

    def write_topic(self, session: AgentSession, record: MemoryTopicRecord, *, source_id: str) -> None:
        self._topics.setdefault(self._owner(session), {})[record.topic] = record

    def delete_topic(self, session: AgentSession, *, source_id: str, topic: str) -> None:
        self._topics.get(self._owner(session), {}).pop(topic, None)

    def rebuild_index(self, session: AgentSession, *, source_id: str, line_limit: int = 200, line_length: int = 150) -> list[MemoryIndexEntry]:
        return []  # index is implicit in list_topics for this backend

    def get_index_text(self, session: AgentSession, *, source_id: str, line_limit: int = 200, line_length: int = 150, index_entries=None) -> str:
        topics = self.list_topics(session, source_id=source_id)
        return "\n".join(f"- {t.topic}: {t.summary[:80]}" for t in topics)

    def read_state(self, session: AgentSession, *, source_id: str) -> dict[str, Any]:
        return self._state.get(self._owner(session), {})

    def write_state(self, session: AgentSession, state: Mapping[str, Any], *, source_id: str) -> None:
        self._state[self._owner(session)] = dict(state)

    def get_transcripts_directory(self, session: AgentSession, *, source_id: str) -> Path:
        raise NotImplementedError("In-memory store does not support transcript directories")

    def search_transcripts(
        self,
        session: AgentSession,
        *,
        source_id: str,
        query: str,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return []


# Wire up like MemoryFileStore
memory = MemoryContextProvider(store=InMemoryMemoryStore(owner_state_key="user_id"))
agent = Agent(client=client, context_providers=[memory])
```

### Tools injected into the agent

`MemoryContextProvider` injects 6 tools that the agent can call at any time to manage its own memory. Your application code does not need to wire these up — the provider handles it automatically:

| Tool name | Purpose |
|---|---|
| `list_memory_topics()` | List all known topic names and their summaries |
| `read_memory_topic(topic)` | Retrieve the full content of one topic file |
| `write_memory(topic, memory)` | Add or update a fact in a specific topic |
| `delete_memory_topic(topic)` | Remove a topic file entirely |
| `search_memory_transcripts(query, session_id, limit)` | Full-text search over saved session transcripts |
| `consolidate_memories()` | Manually trigger a consolidation pass now |

The agent calls these autonomously — for example, searching transcripts before answering a question about past decisions, or explicitly writing a memory when the user says "remember this for next time." No prompt engineering beyond the default `context_prompt` is needed to activate them.

### `MemoryContextProvider` constructor reference

```python
MemoryContextProvider(
    recent_turns: int = 0,               # inject last N turns from transcript into context
    load_tool_turns: bool = True,        # include tool-call turns in the recency window
    *,
    store: MemoryStore,                  # required — MemoryFileStore or custom
    source_id: str = "memory",           # partition key; change when stacking multiple providers
    context_prompt: str | None = None,   # override "## Memory\n..." header
    index_line_limit: int = 200,         # max pointer lines in MEMORY.md
    index_line_length: int = 150,        # max chars per pointer line
    selection_limit: int = 3,            # max topic files loaded per session
    max_extractions: int = 5,            # max memory items extracted per session
    consolidation_interval: timedelta = timedelta(hours=24),
    consolidation_min_sessions: int = 5, # require N sessions before consolidation runs
    extraction_prompt: str | None = None,        # override LLM extraction prompt
    consolidation_prompt: str | None = None,     # override LLM consolidation prompt
    consolidation_client: SupportsChatGetResponse | None = None,  # cheaper model for consolidation
    history_message_filter: Callable[[Message], Message | None] | None = None,
    history_dumps: Callable | None = None,   # custom JSON serialiser for transcripts
    history_loads: Callable | None = None,   # custom JSON deserialiser
)
```

### `MemoryFileStore` directory layout

```
base_path/
  <owner_id>/
    memory/
      MEMORY.md           ← keyword-indexed topic pointer list
      topics/
        preferences.md    ← one file per extracted topic
        coding-style.md
      transcripts/
        <session_id>.jsonl  ← conversation transcript per session
      state.json          ← maintenance state (last extraction, consolidation timestamps)
```

Customise directory names at construction time:

```python
store = MemoryFileStore(
    base_path="./data",
    owner_state_key="tenant_id",
    kind="agent-memory",                # → ./data/<tenant_id>/agent-memory/
    index_file_name="index.md",
    topics_directory_name="knowledge",
    transcripts_directory_name="logs",
    state_file_name="meta.json",
)
```

---

## Quick-reference table

| Class / helper | Import | Notes |
|---|---|---|
| `TodoProvider` | `agent_framework` | `ContextProvider`; injects 5 tools + instructions |
| `TodoItem` | `agent_framework` | Dataclass: `id`, `title`, `description`, `is_complete` |
| `TodoInput` | `agent_framework` | Tool argument shape for `add_todos` |
| `TodoSessionStore` | `agent_framework` | In-memory store; default for `TodoProvider` |
| `TodoFileStore` | `agent_framework` | File-backed; `owner_state_key` for multi-user |
| `AgentModeProvider` | `agent_framework` | `ContextProvider`; injects `get_mode`/`set_mode` + instructions |
| `get_agent_mode` | `agent_framework` | Read current mode from `AgentSession.state` |
| `set_agent_mode` | `agent_framework` | Write mode externally; triggers notification on next `run()` |
| `MemoryContextProvider` | `agent_framework` | `HistoryProvider`; long-term memory via extraction + injection |
| `MemoryFileStore` | `agent_framework` | File-backed memory store; `owner_state_key` for multi-user |
| `MemoryStore` | `agent_framework` | Abstract base — subclass for custom backends |
| `MemoryIndexEntry` | `agent_framework` | One line in `MEMORY.md`; `topic`, `slug`, `summary`, `updated_at` |
| `MemoryTopicRecord` | `agent_framework` | One topic file; `topic`, `memories` list, `summary`, `updated_at` |
| `DEFAULT_TODO_SOURCE_ID` | `agent_framework` | `"todo"` — the default `source_id` for `TodoProvider` |
| `DEFAULT_MODE_SOURCE_ID` | `agent_framework` | `"agent_mode"` — the default `source_id` for `AgentModeProvider` |
| `DEFAULT_MEMORY_SOURCE_ID` | `agent_framework` | `"memory"` — the default `source_id` for `MemoryContextProvider` |

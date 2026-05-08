---
title: "Microsoft Agent Framework (Python) — Memory"
description: "Persist durable facts across sessions with MemoryContextProvider, MemoryFileStore, and automated extraction/consolidation. Verified against agent-framework-core 1.3.0."
framework: microsoft-agent-framework
language: python
---

# Memory — Python

`MemoryContextProvider` gives an agent **durable, topic-based memory** that persists across sessions. At the end of each session, the framework extracts important facts from the transcript, organises them by topic, and writes them to Markdown files. On the next session, relevant topic files are loaded automatically — the agent remembers what matters without carrying the entire conversation history.

Verified against `agent-framework-core==1.3.0` (`agent_framework._harness._memory`). Marked `experimental` — API may evolve.

## How it works

The memory system operates in three layers:

1. **Index (`MEMORY.md`)** — a short list of topic pointers injected each turn. The model sees only topic names and one-line summaries, keeping initial context small.
2. **Topic files (`topics/<slug>.md`)** — per-topic Markdown files loaded on demand via the `read_memory_topic` tool.
3. **Transcripts** — conversation logs stored per session, used as the raw material for extraction and consolidation.

At the end of each turn the provider saves the full conversation to the transcript archive. Periodically (controlled by `consolidation_interval` and `consolidation_min_sessions`) a consolidation pass runs: it reads recent transcripts, calls the LLM to extract durable facts, and merges them into the topic files.

## Quick start

```python
import asyncio
from agent_framework import Agent, AgentSession, MemoryContextProvider, MemoryFileStore
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    store = MemoryFileStore(
        base_path="./agent-memory",
        owner_state_key="user_id",  # session.state key that holds the owner ID
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful long-term assistant.",
        context_providers=[MemoryContextProvider(store=store)],
    )

    # owner_state_key routes files to ./agent-memory/<encoded_source_id>/<encoded_user_id>/memory/
    session = AgentSession(session_id="session-1")
    session.state["user_id"] = "alice"

    r1 = await agent.run("My team uses Python and we deploy to Azure Container Apps.", session=session)
    r2 = await agent.run("What should I know about our stack?", session=session)
    print(r2.text)
    # → Remembers "Python", "Azure Container Apps" from the same session.

    # In a fresh session: relevant topic files are loaded from disk.
    new_session = AgentSession(session_id="session-2")
    new_session.state["user_id"] = "alice"
    r3 = await agent.run("What tech stack does my team use?", session=new_session)
    print(r3.text)
    # → Still knows the stack, loaded from the persisted topic files.


asyncio.run(main())
```

## `MemoryFileStore` — required backing store

All file paths are scoped to an owner, derived from `session.state[owner_state_key]`. This keeps each user's memory isolated.

```python
from agent_framework import MemoryFileStore

store = MemoryFileStore(
    base_path="./memory",           # root directory
    owner_state_key="user_id",      # session.state key → owner directory
    kind="memory",                  # subdirectory bucket name (default: "memory")
    owner_prefix="",                # optional prefix for owner directory names
    index_file_name="MEMORY.md",    # root index filename (default)
    topics_directory_name="topics", # topics subdirectory (default)
    transcripts_directory_name="transcripts",  # transcript archive (default)
    state_file_name="state.json",   # consolidation-state file (default)
)
```

The directory layout for owner `alice`:

```
./memory/
└── <source_id>/
    └── <alice_encoded>/
        └── memory/
            ├── MEMORY.md           ← topic index
            ├── state.json          ← consolidation metadata
            ├── topics/
            │   ├── tech-stack.md
            │   └── preferences.md
            └── transcripts/
                ├── session-1.jsonl
                └── session-2.jsonl
```

## `MemoryContextProvider` — configuration knobs

```python
from datetime import timedelta
from agent_framework import MemoryContextProvider, MemoryFileStore

store = MemoryFileStore(base_path="./memory", owner_state_key="user_id")

provider = MemoryContextProvider(
    store=store,

    # How many recent transcript turns to inject alongside durable memory.
    # 0 = only durable facts; >0 adds short-term context from recent turns.
    recent_turns=5,

    # Whether to include tool-call turns in the recent-turns window.
    load_tool_turns=True,

    # Max topic files auto-loaded per turn (the model can request more via tool).
    selection_limit=3,

    # Max memory bullets extracted per turn.
    max_extractions=5,

    # Minimum time between consolidation passes.
    consolidation_interval=timedelta(hours=24),

    # Number of sessions that must accumulate before consolidation runs.
    consolidation_min_sessions=5,

    # Use a cheaper model for consolidation to save cost.
    # Defaults to None, which reuses the agent's own client.
    consolidation_client=None,
)
```

### Short-term context via `recent_turns`

Set `recent_turns > 0` to inject the last N turns directly into context alongside durable memory. This bridges the gap between what the agent just heard and what it has consolidated long-term:

```python
provider = MemoryContextProvider(
    store=store,
    recent_turns=3,       # inject the last 3 conversation turns
    load_tool_turns=False, # skip tool-call turns in the short-term window
)
```

### Cheaper consolidation model

Consolidation runs a summarisation prompt that can be expensive. Route it to a cheaper model:

```python
from agent_framework.openai import OpenAIChatClient

consolidation_client = OpenAIChatClient(
    model="gpt-4o-mini",
    instructions="Extract and consolidate durable facts from agent transcripts.",
)

provider = MemoryContextProvider(
    store=store,
    consolidation_client=consolidation_client,
)
```

## Custom extraction and consolidation prompts

The default prompts instruct the model to extract durable facts and merge them by topic. Override either prompt when your domain needs domain-specific extraction rules:

```python
EXTRACTION_PROMPT = """You extract durable memory candidates from an agent transcript delta.

Return only JSON with this exact shape:
{"memories":[{"topic":"short topic name","memory":"durable fact"}]}

Rules:
- Record product names, architecture decisions, team preferences, and user goals.
- Do NOT record transient tool outputs, temporary counts, or one-off search results.
"""

provider = MemoryContextProvider(
    store=store,
    extraction_prompt=EXTRACTION_PROMPT,
)
```

## Tools the agent receives

The provider registers several tools the model uses to interact with its memory:

| Tool | Purpose |
|---|---|
| `read_memory_topic` | Load the full contents of one topic file |
| `search_memory` | Search topic summaries for relevant topics |
| `write_memory_topic` | Write or update a topic file directly |
| `delete_memory_topic` | Remove a topic that is no longer relevant |

The agent calls these automatically when the topic index indicates relevant knowledge exists.

## Combining with `TodoProvider`

Memory (durable cross-session facts) and todos (current session tasks) complement each other naturally:

```python
from agent_framework import (
    Agent,
    MemoryContextProvider,
    MemoryFileStore,
    TodoFileStore,
    TodoProvider,
)
from agent_framework.openai import OpenAIChatClient

memory_store = MemoryFileStore(base_path="./data/memory", owner_state_key="user_id")
todo_store   = TodoFileStore(base_path="./data/todos",   owner_state_key="user_id")

agent = Agent(
    client=OpenAIChatClient(),
    instructions=(
        "You are a long-running project assistant. "
        "Use your memory for past decisions and your todo list for current work."
    ),
    context_providers=[
        MemoryContextProvider(store=memory_store, recent_turns=3),
        TodoProvider(store=todo_store),
    ],
)
```

## Reading memory outside the agent loop

Inspect persisted topics or the index directly — useful for admin views or memory migration tools:

```python
import asyncio
from agent_framework import AgentSession, MemoryFileStore


async def print_memory_index(user_id: str) -> None:
    store = MemoryFileStore(base_path="./data/memory", owner_state_key="user_id")

    session = AgentSession()
    session.state["user_id"] = user_id

    # Load the topic index
    index_path = store.get_index_path(session, source_id="memory")
    if index_path.exists():
        print(index_path.read_text())
    else:
        print("No memory yet.")
```

## Filtering messages before transcript save

Pass a `history_message_filter` to rewrite or drop messages before they enter the transcript archive — useful for stripping PII or internal tool chatter:

```python
from agent_framework import Message

def drop_tool_results(messages: list[Message]) -> list[Message]:
    """Exclude raw tool result messages from the saved transcript."""
    return [m for m in messages if m.role != "tool"]

provider = MemoryContextProvider(
    store=store,
    history_message_filter=drop_tool_results,
)
```

## Resetting or migrating memory

Delete the owner's memory root to start fresh, or copy the directory to migrate a user's memory to a new deployment:

```python
import shutil
from agent_framework import AgentSession, MemoryFileStore

async def reset_user_memory(user_id: str, base_path: str = "./memory") -> None:
    """Delete all memory for one user — irreversible."""
    store = MemoryFileStore(base_path=base_path, owner_state_key="user_id")
    # Build a throwaway session so the store can resolve the correct
    # encoded path: base_path / encode(source_id) / encode(owner) / kind
    session = AgentSession()
    session.state["user_id"] = user_id
    # _get_memory_root is a private helper — it may change between releases.
    memory_root = store._get_memory_root(session, source_id="memory")
    if memory_root.exists():
        shutil.rmtree(memory_root)
```

> For production use, prefer soft-deletion (rename the directory) over hard deletion so you can restore accidentally deleted memory.

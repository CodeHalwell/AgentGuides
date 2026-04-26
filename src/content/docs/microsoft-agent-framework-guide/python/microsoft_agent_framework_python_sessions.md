---
title: "Microsoft Agent Framework (Python) — Sessions & Context Providers"
description: "AgentSession, ContextProvider, HistoryProvider, InMemoryHistoryProvider, and SessionContext — how state, conversation history, and per-run context flow through agent-framework-core 1.2.0."
framework: microsoft-agent-framework
language: python
---

# Sessions & Context Providers — Python

`agent_framework` separates three things that lots of frameworks blend together:

| Concept | Class | Lives where |
|---|---|---|
| **Per-conversation identity** | `AgentSession` | A handle you create with `agent.create_session()` and pass to every `agent.run(...)` |
| **Per-invocation state** | `SessionContext` | Created fresh per `run()`, passed through every `ContextProvider`, then thrown away |
| **Pluggable behaviour** | `ContextProvider` (and its `HistoryProvider` subclass) | Attached to an `Agent` via `context_providers=[...]` |

Imports below are stable in `agent-framework-core==1.2.0`. The full module is `agent_framework._sessions`, but everything you need is re-exported from `agent_framework`.

## The session, the context, and the providers

A picture before the code:

```
agent.run(...)
  ├─ creates SessionContext for THIS call
  ├─ for each ContextProvider in agent.context_providers:
  │     await provider.before_run(agent, session, context, state)
  ├─ context.context_messages + context.input_messages → model
  ├─ model returns AgentResponse → context.response
  └─ for each provider (reverse order):
        await provider.after_run(agent, session, context, state)
```

`AgentSession` is the long-lived thing. `SessionContext` is the short-lived bag of "what we're sending this turn / what came back this turn". Providers are how you mutate it.

## `AgentSession` — the long-lived handle

```python
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a friendly assistant.",
)

# Auto-generated session id (uuid4)
session = agent.create_session()
print(session.session_id)   # 'a3f4...'

# Or pin a specific id (useful for resuming external conversations)
session = agent.create_session(session_id="user-42-thread-1")

await agent.run("Hi!", session=session)
await agent.run("What was my first message?", session=session)
```

`session.state` is a plain dict — providers stash data there. Each provider gets a *scoped* sub-dict (keyed by `provider.source_id`) so providers don't trample each other.

### Persisting and restoring sessions

`AgentSession.to_dict()` and `from_dict()` round-trip the session through JSON. Anything you put in `session.state` that has `to_dict`/`from_dict` (or is a Pydantic model) is serialised automatically:

```python
import json

# Save
blob = json.dumps(session.to_dict())
# … later, possibly different process …
restored = AgentSession.from_dict(json.loads(blob))
```

For Pydantic models, the framework auto-registers the type on first serialisation. To be safe across cold starts (where the type hasn't been serialised yet in the new process), call `register_state_type` at import time:

```python
from pydantic import BaseModel
from agent_framework import register_state_type


class UserProfile(BaseModel):
    user_id: str
    plan: str = "free"


register_state_type(UserProfile)   # idempotent

session.state["profile"] = UserProfile(user_id="u-99", plan="pro")
blob = session.to_dict()           # 'profile' becomes {"type": "userprofile", "user_id": "u-99", ...}
```

Override `_get_type_identifier` to control the type id explicitly:

```python
class UserProfile(BaseModel):
    user_id: str

    @classmethod
    def _get_type_identifier(cls) -> str:
        return "myapp.user_profile"
```

## `SessionContext` — the per-run bag

`SessionContext` is created fresh for every `agent.run(...)` and handed to every provider. The interesting fields:

| Field | Type | Purpose |
|---|---|---|
| `session_id` | `str \| None` | The current `AgentSession.session_id` |
| `service_session_id` | `str \| None` | Server-side conversation id when the chat client manages history server-side |
| `input_messages` | `list[Message]` | What the **caller** is sending this turn |
| `context_messages` | `dict[source_id, list[Message]]` | What **providers** want to inject (e.g. retrieved docs, history) |
| `instructions` | `list[str]` | Extra instructions providers want prepended |
| `tools` | `list[Any]` | Run-scoped tool overrides providers want to add |
| `middleware` | `dict[source_id, list[MiddlewareTypes]]` | Chat/function middleware injected for this run |
| `response` | `AgentResponse \| None` | Populated by the framework before `after_run` |
| `options`, `metadata` | `dict` | Read-only options + free-form scratch space |

The mutation methods are the API — don't reach into the dicts directly:

```python
context.extend_messages(self, messages)        # 'self' → uses self.source_id and source_type
context.extend_messages("rag", messages)       # explicit source_id
context.extend_instructions(self.source_id, "Use UK English.")
context.extend_tools(self.source_id, [my_tool])
context.extend_middleware(self.source_id, my_middleware)
```

`extend_messages` *copies* every message, stamps `additional_properties["_attribution"] = {"source_id": ..., "source_type": ...}`, and stores them. So your originals are never mutated, and downstream code can filter by source.

### Filtering messages back out

`get_messages` is the consumer-side counterpart:

```python
# Everything except this provider's own injected messages
others = context.get_messages(exclude_sources={self.source_id})

# Only history + RAG, plus the new user message
combined = context.get_messages(
    sources={"in_memory", "rag"},
    include_input=True,
)
```

This is the same primitive `HistoryProvider.after_run` uses to decide what to persist — see below.

## `ContextProvider` — the base class

Subclass `ContextProvider` when you have something to inject before a run or to record after one. The two override points are intentionally tiny:

```python
from typing import Any
from agent_framework import (
    Agent,
    AgentSession,
    ContextProvider,
    Message,
    SessionContext,
)
from agent_framework.openai import OpenAIChatClient


class TimeContextProvider(ContextProvider):
    """Add today's date to the system prompt and remember the last question."""

    def __init__(self, source_id: str = "time_ctx") -> None:
        super().__init__(source_id)

    async def before_run(
        self,
        *,
        agent,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        context.extend_instructions(self.source_id, f"Today is {today}.")
        # Remember what the user asked, scoped to this provider:
        if context.input_messages:
            state.setdefault("last_user_text", context.input_messages[-1].text)

    async def after_run(
        self,
        *,
        agent,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        if context.response is not None:
            state["last_response_chars"] = len(context.response.text)


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[TimeContextProvider()],
)
session = agent.create_session()

await agent.run("What's the date?", session=session)
print(session.state["time_ctx"])   # {'last_user_text': "What's the date?", 'last_response_chars': 42}
```

Key details from the source:

- `state` here is the **provider-scoped** sub-dict. The full session state is at `session.state`. Two providers with different `source_id`s never see each other's `state` dicts.
- `before_run` may add **messages, instructions, tools, or middleware** to `context`. It cannot reach back into the agent's permanent middleware.
- `after_run` runs after the model call, in **reverse provider order**, so the last provider added is the first to observe the response. `context.response` is set by the framework, not by you.

### A retrieval provider

Inject snippets just before the model call, then no-op afterwards:

```python
class RagProvider(ContextProvider):
    def __init__(self, retriever, top_k: int = 3) -> None:
        super().__init__("rag")
        self._retriever = retriever
        self._top_k = top_k

    async def before_run(self, *, agent, session, context, state) -> None:
        if not context.input_messages:
            return
        query = context.input_messages[-1].text or ""
        snippets = await self._retriever.search(query, top_k=self._top_k)
        if not snippets:
            return
        context.extend_messages(
            self,                                    # passes source_id + source_type for attribution
            [
                Message(role="system", contents=[f"<doc>{s}</doc>"])
                for s in snippets
            ],
        )
```

Because `extend_messages(self, ...)` was used, every injected message carries `additional_properties["_attribution"] = {"source_id": "rag", "source_type": "RagProvider"}`. Other providers can include or exclude them with `context.get_messages(sources={"rag"})`.

## `HistoryProvider` — base class for storage backends

`HistoryProvider` is the `ContextProvider` you subclass when you want a real read/write history backend. It defines four orthogonal flags so the same class can act as a primary store, a write-only audit log, or a read-only replay source:

| Flag | Default | Effect |
|---|---|---|
| `load_messages` | `True` | Call `get_messages()` in `before_run` and inject results into `context` |
| `store_inputs` | `True` | Persist the caller's messages (`context.input_messages`) in `after_run` |
| `store_outputs` | `True` | Persist response messages |
| `store_context_messages` | `False` | Also persist messages other providers injected (filtered by `store_context_from`) |

Implementing your own backend is two methods:

```python
from collections.abc import Sequence
from agent_framework import HistoryProvider, Message


class RedisHistoryProvider(HistoryProvider):
    def __init__(self, redis_client, *, source_id: str = "redis_history", **flags) -> None:
        super().__init__(source_id=source_id, **flags)
        self._redis = redis_client

    async def get_messages(self, session_id, *, state=None, **kwargs) -> list[Message]:
        if not session_id:
            return []
        raw = await self._redis.get(f"history:{session_id}") or b"[]"
        return [Message.from_dict(d) for d in __import__("json").loads(raw)]

    async def save_messages(self, session_id, messages, *, state=None, **kwargs) -> None:
        if not session_id:
            return
        existing = await self.get_messages(session_id)
        combined = [*existing, *messages]
        await self._redis.set(
            f"history:{session_id}",
            __import__("json").dumps([m.to_dict() for m in combined]).encode("utf-8"),
        )
```

The default `before_run` and `after_run` from the parent class will:
1. Call `get_messages(session_id)` and `context.extend_messages(self, history)` if `load_messages`.
2. Compute the to-store batch from `store_inputs / store_outputs / store_context_messages` and call `save_messages(session_id, batch)`.

You only override the two abstract methods; the lifecycle is free. (See `agent_framework.redis.RedisHistoryProvider` for the production-grade version with pipelining and TTLs.)

## `InMemoryHistoryProvider` — the default

When you don't pass `context_providers=[...]` to an agent, the framework auto-installs an `InMemoryHistoryProvider` for local sessions. It stores messages inside `session.state["in_memory"]["messages"]`, so when you serialise the session, the history rides along for free.

```python
from agent_framework import Agent, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient

history = InMemoryHistoryProvider(
    source_id="my_history",         # default is 'in_memory'
    store_context_messages=True,    # also persist what RAG / Skills inject
    store_context_from={"rag"},     # but only from these source_ids
    skip_excluded=True,             # ignore messages the compaction provider excluded
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are helpful.",
    context_providers=[history],
)

session = agent.create_session()
await agent.run("Pick a fruit and remember it.", session=session)
await agent.run("What did you pick?", session=session)
```

`skip_excluded=True` plays nicely with `CompactionProvider` — the compaction provider marks groups `excluded=True` in stored history so the *next* turn loads a smaller transcript.

### Audit-log mode — write-only

```python
audit = InMemoryHistoryProvider(
    source_id="audit",
    load_messages=False,           # never inject anything on load
    store_inputs=True,
    store_outputs=True,
    store_context_messages=True,   # capture everything other providers added too
)
```

You'd typically pair this with a *real* history provider (e.g. another `InMemoryHistoryProvider` or `FileHistoryProvider`) so you have one read/write channel and one write-only channel.

### `FileHistoryProvider` — durable across restarts

```python
from agent_framework import FileHistoryProvider

history = FileHistoryProvider(
    storage_path="./sessions",
    skip_excluded=True,
)
```

One JSON-Lines file per session_id. From the source:

- `session_id` is sanitised — only `[A-Za-z0-9._-]` plus a few other safe characters are kept literally; anything else is base64url-encoded with a `~session-` prefix.
- Resolved file paths are validated against the storage root, so `session_id="../secrets"` raises `ValueError` rather than escaping the directory.
- Reads and writes are guarded by **64 striped locks** (one async, one threading) so concurrent reads in the same process don't deadlock and don't fight each other.

It is **not** safe for cross-process write contention — use the Redis or Cosmos providers for multi-replica deployments.

## Combining providers — order matters

Providers run in registration order on the way in, reverse on the way out:

```python
from agent_framework import (
    Agent, CompactionProvider, InMemoryHistoryProvider,
    SlidingWindowStrategy, ToolResultCompactionStrategy,
)

history    = InMemoryHistoryProvider(skip_excluded=True)
compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
    history_source_id=history.source_id,
)
rag        = RagProvider(retriever=my_retriever)

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[history, compaction, rag],   # before_run order
)
```

What happens per `agent.run(...)`:

1. **History → load** older messages from `session.state["in_memory"]`.
2. **Compaction → before_strategy** trims the just-loaded history.
3. **RAG → search** with the user's input and inject docs.
4. Model call.
5. **RAG → after_run** (no-op).
6. **Compaction → after_strategy** trims what's about to be stored.
7. **History → save** the final batch.

The reverse order on the way out matters: compaction has to act on the messages *before* the history provider commits them, so compaction must come *after* history in `context_providers=[...]`.

## Reading session state for debugging

Every provider's `state` lives under its `source_id` in the session:

```python
print(session.state.keys())             # dict_keys(['in_memory', 'compaction', 'rag', 'time_ctx'])
print(session.state["in_memory"]["messages"][:2])
print(session.state["compaction"])
print(session.state["time_ctx"]["last_user_text"])
```

This is the same dict that round-trips through `AgentSession.to_dict()` — anything you persist in `state` survives a serialise → deserialise → resume cycle as long as the type is registered (or is built-in).

## Common patterns

**Per-tenant prompt prefix.** A custom `ContextProvider` reads `context.kwargs.get("tenant_id")` and calls `context.extend_instructions(...)`. Pass the tenant id at run-time via `agent.run(..., tenant_id="acme")`.

**Hybrid memory.** Two providers — an `InMemoryHistoryProvider` as the read/write primary, plus a `FileHistoryProvider` configured with `load_messages=False` as a durable audit trail.

**Just-in-time tool injection.** A `ContextProvider` that queries a feature flag in `before_run` and calls `context.extend_tools(...)` only for users in the experiment.

**Per-run middleware.** A `ContextProvider` whose `before_run` calls `context.extend_middleware(self.source_id, MyChatMiddleware())` to inject middleware that should *only* be active when the provider is mounted (e.g. a redaction middleware specific to a regulated tenant).

`extend_middleware` accepts chat or function middleware only — agent middleware would form a circular pipeline and is rejected with a `MiddlewareException`.

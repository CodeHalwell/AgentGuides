---
title: "Microsoft Agent Framework (Python) — Sessions & history"
description: "AgentSession, HistoryProvider, InMemoryHistoryProvider, FileHistoryProvider, register_state_type, multi-provider audit logs, and serializing sessions across processes. APIs stable from agent-framework-core 1.3.0; verified against 1.3.0."
framework: microsoft-agent-framework
language: python
---

# Sessions & history — Python

A **session** is one logical conversation with an agent. The framework splits responsibilities cleanly:

| Object | Owns | Lifetime |
|---|---|---|
| `AgentSession` | `session_id`, optional `service_session_id`, mutable `state: dict` | Per conversation |
| `HistoryProvider` (subclass) | The actual messages — read/write to disk, Redis, in-memory, … | Process-long, attached to the agent |
| `Agent` | Orchestrates providers, threads `state` through them on every run | Process-long |

This page walks the moving parts in `agent_framework._sessions` and the patterns that fall out of them. The public surface (`AgentSession`, `HistoryProvider`, `InMemoryHistoryProvider`, `FileHistoryProvider`, `register_state_type`) is stable as of `agent-framework-core==1.3.0`, which matches what the rest of this guide targets unless a section explicitly says otherwise.

## TL;DR

```python
import asyncio
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[FileHistoryProvider(storage_path="./sessions")],
    )

    session = agent.create_session(session_id="user-42-conversation-1")

    # Turn 1
    r1 = await agent.run("Remember that my favourite colour is teal.", session=session)

    # Turn 2 — history is loaded automatically because the same provider is on the agent
    r2 = await agent.run("What is my favourite colour?", session=session)
    print(r2.text)


asyncio.run(main())
```

The provider lives on the agent. The session is just an id + scratchpad — pass it to every `run()` to opt into persistence for that conversation.

## `AgentSession` in one screen

```python
from agent_framework import AgentSession

# Auto-generated UUID
session = AgentSession()
print(session.session_id)             # → 'd4f0...e2'

# Stable ids let you correlate across services
session = AgentSession(session_id="customer-9281")

# `state` is a free-form dict shared with every provider for this session
session.state["preferred_currency"] = "EUR"
session.state["last_seen_at"] = "2026-04-30T08:42:00Z"
```

Two attributes you may set yourself:

- `session_id` — your stable correlation id (defaults to a UUID when omitted).
- `service_session_id` — a service-managed id (e.g. an OpenAI Responses thread id). Set it when the **provider** owns the conversation server-side and you only want to keep a pointer.

`session.state` is the cross-provider scratchpad. The framework already registers `Message` for round-trip serialisation; for your own classes, register them once at startup.

### Storing custom types in `state`

Built-in JSON types (str, int, float, bool, None, list, dict) survive `to_dict()`/`from_dict()` automatically. For custom types, either implement `to_dict`/`from_dict` (any class) or use a Pydantic `BaseModel` (auto-detected).

```python
from pydantic import BaseModel
from agent_framework import AgentSession, register_state_type


class UserProfile(BaseModel):
    user_id: str
    plan: str
    org_id: str | None = None


# Register once at startup so cold-start restores work even before the model
# has been serialised this process.
register_state_type(UserProfile)

session = AgentSession()
session.state["profile"] = UserProfile(user_id="u-42", plan="enterprise")

snapshot = session.to_dict()             # safe to JSON-encode
restored = AgentSession.from_dict(snapshot)
assert isinstance(restored.state["profile"], UserProfile)
```

If you need a different identifier than the lowercase class name, define a class method:

```python
class LegacyOrder:
    @classmethod
    def _get_type_identifier(cls) -> str:
        return "legacy.order.v2"

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "LegacyOrder": ...

register_state_type(LegacyOrder)
```

## `HistoryProvider` — the storage seam

Every persistence backend in the framework subclasses `HistoryProvider` and implements two coroutines:

```python
class HistoryProvider(ContextProvider):
    async def get_messages(self, session_id, *, state=None, **kwargs) -> list[Message]: ...
    async def save_messages(self, session_id, messages, *, state=None, **kwargs) -> None: ...
```

The base class wires `before_run`/`after_run` so subclasses do nothing else. The framework ships two:

| Class | Storage | Use when |
|---|---|---|
| `InMemoryHistoryProvider` | `session.state["messages"]` | Single-process bots, tests, ephemeral conversations |
| `FileHistoryProvider` | One JSONL file per `session_id` under a directory | Single-host deployments, durable across restarts |

Beta provider packages add Redis (`agent-framework-redis`), Cosmos DB (`agent-framework-azure-cosmos`), and Azure AI Search (`agent-framework-azure-ai-search`) backends — same `HistoryProvider` interface.

### Configuration flags shared by every history provider

These come straight from the constructor signature:

| Flag | Default | Effect |
|---|---|---|
| `load_messages` | `True` | If `False`, the provider never injects past messages. Use for write-only audit logs. |
| `store_inputs` | `True` | Whether new user messages get persisted. |
| `store_outputs` | `True` | Whether assistant responses get persisted. |
| `store_context_messages` | `False` | Whether messages added by **other** context providers (skills, RAG, …) get persisted. |
| `store_context_from` | `None` | When set, only persist context from these `source_id`s. |
| `skip_excluded` | `False` | When `True`, exclude messages flagged by compaction (`additional_properties["_excluded"] == True`). |

Pair `load_messages=False` + `store_outputs=True` for an audit log, `load_messages=True` + `store_outputs=False` for read-only replay, etc.

## `InMemoryHistoryProvider`

Default when you don't configure anything. Messages live in `session.state["messages"]` — so they automatically travel with `session.to_dict()` / `from_dict()`.

```python
import json
from agent_framework import Agent, AgentSession, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient


history = InMemoryHistoryProvider()

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[history],
)

session = AgentSession()
await agent.run("Remember the launch is on 2026-05-12.", session=session)

# Persist the *whole* session somewhere — Redis, a request envelope, anywhere.
blob = json.dumps(session.to_dict())
print(len(blob), "bytes")

# Later, in a different process:
restored = AgentSession.from_dict(json.loads(blob))
r = await agent.run("When is the launch?", session=restored)
print(r.text)
```

Because the messages live inside the session dict, you don't need a separate datastore — your existing request/session store handles it. This is the right default for serverless and stateless front-ends where you already have a session cookie or token.

### Skipping compacted messages on read

When you pair `InMemoryHistoryProvider` with a `CompactionProvider`, the compaction provider marks older messages as excluded **in stored history**. Tell the history provider to skip them on subsequent loads to honour that exclusion:

```python
from agent_framework import (
    Agent,
    CompactionProvider,
    InMemoryHistoryProvider,
    SlidingWindowStrategy,
)
from agent_framework.openai import OpenAIChatClient

history = InMemoryHistoryProvider(skip_excluded=True)
compaction = CompactionProvider(after_strategy=SlidingWindowStrategy(keep_last_groups=20))

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[history, compaction],
)
```

Without `skip_excluded=True`, the next turn re-loads everything compaction trimmed last turn, defeating the strategy.

## `FileHistoryProvider`

One JSONL file per `session_id` in a single directory. Append-only, single-line JSON per message — corruption of one line never destroys the whole conversation.

```python
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

history = FileHistoryProvider(
    storage_path="./conversations",     # directory; created automatically
    skip_excluded=True,                 # honour CompactionProvider exclusions
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[history],
)

session = agent.create_session(session_id="customer-9281")
await agent.run("Hi, I need help with order #4421.", session=session)
# Writes ./conversations/customer-9281.jsonl
```

### Path-traversal protection

`session_id` flows from your application — sometimes from URL routing, sometimes from headers. `FileHistoryProvider` resolves every `session_id` against the storage root and **rejects any id that would escape**:

- `../`, absolute paths, and Windows reserved stems (`CON`, `PRN`, …) are rewritten or rejected.
- The resolved path is checked against the storage root.

So `agent.create_session(session_id="../etc/passwd")` is safe — it lands inside `storage_path`, not at the OS path. You still need OS-level filesystem permissions (the contents are plaintext JSONL).

### Encrypted-at-rest sessions via `dumps`/`loads`

Inject your own JSON serialisers to add envelope encryption, schema migration, or PII redaction:

```python
import json
import os
from cryptography.fernet import Fernet
from agent_framework import Agent, FileHistoryProvider

key = os.environ["AGENT_HISTORY_FERNET_KEY"]   # 32-byte urlsafe-b64 key
cipher = Fernet(key)


def encrypt_dumps(payload: dict) -> str:
    # Fernet tokens are already URL-safe base64 — single line, no extra encoding required.
    plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return cipher.encrypt(plaintext).decode("ascii")


def decrypt_loads(line: str | bytes) -> dict:
    if isinstance(line, str):
        line = line.encode("ascii")
    return json.loads(cipher.decrypt(line).decode("utf-8"))


history = FileHistoryProvider(
    storage_path="./encrypted-conversations",
    dumps=encrypt_dumps,
    loads=decrypt_loads,
)
```

Two operational notes:

- `dumps` **must** return a single-line `str` or `bytes` (no `\n` / `\r`) — the provider validates this and raises if you violate it. The output does not have to be JSON; any single-line representation that round-trips through `loads` to a mapping is accepted, which is what makes this encrypted-token pattern work.
- Both callables must round-trip cleanly. Test with `loads(dumps(x)) == x` for a representative payload.

## Agent lifecycle helpers

```python
agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[FileHistoryProvider(storage_path="./conv")],
)

# 1. Create a fresh session — generates a UUID
session = agent.create_session()

# 2. Create with a stable id (e.g. correlated to your app's user id)
session = agent.create_session(session_id="user-42")

# 3. Bind a service-managed conversation thread (e.g. OpenAI Responses)
remote = agent.get_session(service_session_id="thread_abc123")
```

`get_session` is the right entry point when the **server side** owns the history (Foundry threads, OpenAI Responses, Anthropic conversations). The local `session_id` is just a correlation key your code uses; the actual messages stay server-side.

## Multi-provider patterns

A `HistoryProvider` is just a `ContextProvider`. Stack as many as you like — they run in order, share the session's `state`, and can each apply different storage policies.

### Primary store + write-only audit log

Persist everything to the primary provider; mirror only inputs to the audit log.

```python
from agent_framework import Agent, FileHistoryProvider

primary = FileHistoryProvider(
    storage_path="./conversations",
    skip_excluded=True,
)

audit = FileHistoryProvider(
    storage_path="./audit",
    source_id="audit",            # different id so it doesn't clash with primary
    load_messages=False,           # write-only — never re-injects past messages
    store_inputs=True,
    store_outputs=False,
)

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[primary, audit],
)
```

Now `./audit/` accumulates a tamper-evident record of every user prompt — separate from the primary conversation store.

### Logging only specific provider sources

When you have a RAG provider, a skills provider, and a history provider, you may want to keep RAG-injected context out of the persisted history (it's reconstituted each turn anyway):

```python
audit = FileHistoryProvider(
    storage_path="./audit",
    source_id="audit",
    load_messages=False,
    store_inputs=True,
    store_outputs=True,
    store_context_messages=True,
    store_context_from={"skills"},   # only persist context from the skills provider
)
```

Set `store_context_from` to whitelist source ids; leave `None` and set `store_context_messages=True` to persist every other source's contribution.

## Building your own `HistoryProvider`

Two coroutines, no inheritance gymnastics. Override the storage; the base class handles the load/store flags.

```python
import json
from collections.abc import Sequence
from typing import Any
from agent_framework import HistoryProvider, Message
import redis.asyncio as redis


class RedisHistoryProvider(HistoryProvider):
    DEFAULT_SOURCE_ID = "redis_history"

    def __init__(self, url: str, *, ttl_seconds: int | None = None, **kwargs: Any) -> None:
        super().__init__(source_id=self.DEFAULT_SOURCE_ID, **kwargs)
        self._client = redis.from_url(url)
        self._ttl = ttl_seconds

    def _key(self, session_id: str | None) -> str:
        return f"agent:history:{session_id or 'default'}"

    async def get_messages(self, session_id: str | None, *, state=None, **kwargs) -> list[Message]:
        raw = await self._client.lrange(self._key(session_id), 0, -1)
        return [Message.from_dict(json.loads(item)) for item in raw]

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state=None,
        **kwargs,
    ) -> None:
        if not messages:
            return
        key = self._key(session_id)
        pipe = self._client.pipeline()
        for m in messages:
            pipe.rpush(key, json.dumps(m.to_dict(), ensure_ascii=False))
        if self._ttl:
            pipe.expire(key, self._ttl)
        await pipe.execute()
```

That's it. `load_messages`, `store_inputs`, `store_outputs`, `store_context_messages` all work — the base class calls your two methods at the right moments.

> The official `agent-framework-redis` package ships a more sophisticated implementation (RedisVL-backed search, semantic recall) — use it for production. The example above is to show the contract.

## Sessions across processes — request handlers and queues

Stateless workers (Lambda, Cloud Run, Container Apps) need to pull the conversation in, run a turn, and push state back out. Two patterns work well.

### Pattern 1 — `InMemoryHistoryProvider` + your existing session store

Treat the agent like a pure function. Your web framework already has a session blob; round-trip it through `AgentSession.to_dict()` / `from_dict()`:

```python
import json
from fastapi import FastAPI
from agent_framework import Agent, AgentSession, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient

app = FastAPI()
agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[InMemoryHistoryProvider()],
)


@app.post("/chat")
async def chat(payload: dict) -> dict:
    session = (
        AgentSession.from_dict(json.loads(payload["session_blob"]))
        if payload.get("session_blob")
        else AgentSession()
    )
    response = await agent.run(payload["message"], session=session)
    return {
        "reply": response.text,
        "session_blob": json.dumps(session.to_dict()),
    }
```

The blob carries all messages plus any custom state you stored. Sign it (JWT) before returning to the client to make it tamper-evident.

### Pattern 2 — Server-side history, client-side ids only

Use `FileHistoryProvider` (or Redis/Cosmos) on the server; the client only sends a stable `session_id`:

```python
@app.post("/chat")
async def chat(payload: dict) -> dict:
    session = agent.get_session(
        service_session_id=None,
        session_id=payload["session_id"],
    )
    response = await agent.run(payload["message"], session=session)
    return {"reply": response.text}
```

Apply auth checks in the handler so users can only access sessions they own — `FileHistoryProvider` only protects you from path traversal, not authorisation.

## Common pitfalls

**Forgetting to pass `session=`.** Without a session, the agent runs stateless — no history, no `state`. Easy to miss when refactoring.

**Reusing one `AgentSession` across users.** Sessions are per-conversation. Mixing users into one `session_id` cross-contaminates history.

**Forgetting `register_state_type` for cold restarts.** A worker that restarts before the type has been serialised once cannot deserialise sessions that contain it. Register at module import.

**Compaction without `skip_excluded=True` on the history provider.** Compaction marks messages as excluded; if the history provider re-loads them on the next turn, you've gained nothing.

**Single-host JSONL across N replicas.** `FileHistoryProvider` uses per-process locks, not cross-process locks. Multiple replicas writing to the same NFS volume *will* race. Use Redis or Cosmos for multi-host deployments.

**`session.state` mutation outside an agent run.** Providers run in `before_run` / `after_run`. Mutating `state` while a run is in flight is undefined behaviour — do it before/after `agent.run(...)`.

## `SessionContext` — per-run pipeline state

`SessionContext` is created fresh for each `agent.run()` call and threaded through every `ContextProvider` in order. It is the channel through which providers inject context, instructions, tools, and middleware before the model is called. You never construct one yourself — the framework creates it and passes it to `before_run` / `after_run`.

The most useful methods when writing a custom provider:

| Method | Purpose |
|---|---|
| `extend_messages(source, messages)` | Inject context messages (e.g. RAG results, persona snippets) under a named source key |
| `extend_instructions(source_id, instructions)` | Append dynamic system-prompt fragments |
| `extend_tools(source_id, tools)` | Register extra tools for this run only |
| `extend_middleware(source_id, middleware)` | Add per-run chat or function middleware |
| `get_messages(*, sources=None, exclude_sources=None, include_input=False, include_response=False)` | Read back all accumulated messages |
| `get_middleware()` | Retrieve the flat list of middleware added by all providers |

### Writing a custom `ContextProvider`

Subclass `ContextProvider` when you need to inject context that doesn't map cleanly to history or skills — feature flags, A/B instructions, tenant-scoped prompts, live RAG results:

```python
import asyncio
from typing import Any
from agent_framework import Agent, ContextProvider, AgentSession, SessionContext
from agent_framework import Message, Content
from agent_framework.openai import OpenAIChatClient


class TenantContextProvider(ContextProvider):
    """Injects tenant-specific instructions and a live persona snippet."""

    DEFAULT_SOURCE_ID = "tenant_ctx"

    def __init__(self) -> None:
        super().__init__(source_id=self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        tenant_id: str = session.state.get("tenant_id", "default")

        # Append a dynamic system-prompt fragment
        context.extend_instructions(
            self.source_id,
            f"You are serving tenant '{tenant_id}'. Always address them formally.",
        )

        # Inject a context message (appears before the user's message in the model's view)
        persona = await self._fetch_persona(tenant_id)
        context.extend_messages(
            self.source_id,
            [Message(role="system", contents=[Content.from_text(persona)])],
        )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        # Inspect the response after the run completes.
        if context.response:
            session.state["last_response_len"] = len(context.response.text or "")

    async def _fetch_persona(self, tenant_id: str) -> str:
        # Replace with a real DB / cache call
        personas = {
            "acme": "ACME Corp persona: formal tone, no jargon.",
            "default": "Standard persona: helpful and concise.",
        }
        return personas.get(tenant_id, personas["default"])


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[TenantContextProvider()],
    )

    session = agent.create_session()
    session.state["tenant_id"] = "acme"

    response = await agent.run("Summarise our Q3 results.", session=session)
    print(response.text)
    print("response len stored:", session.state["last_response_len"])


asyncio.run(main())
```

### Injecting per-run tools from a provider

`extend_tools` adds tools dynamically based on per-request state — useful when the tool set varies per tenant, user role, or feature flag:

```python
from typing import Any
from agent_framework import Agent, ContextProvider, AgentSession, SessionContext, tool
from agent_framework.openai import OpenAIChatClient


@tool
def search_crm(query: str) -> str:
    """Search the internal CRM."""
    return f"CRM results for {query!r}"


@tool
def admin_wipe_data(user_id: str) -> str:
    """Wipe all data for a user (admin only)."""
    return f"Wiped data for user {user_id}"


class RoleBasedToolProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "role_tools"

    def __init__(self) -> None:
        super().__init__(source_id=self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        role = session.state.get("role", "user")
        tools = [search_crm]
        if role == "admin":
            tools.append(admin_wipe_data)
        context.extend_tools(self.source_id, tools)


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a CRM assistant.",
    context_providers=[RoleBasedToolProvider()],
)
```

### Inspecting accumulated context

Read back what all providers have contributed, useful for debugging or audit logging. `SessionContext.instructions`, `SessionContext.tools`, and `SessionContext.context_messages` are documented public attributes — not implementation details — and are safe to read in any provider's `after_run`:

```python
async def debug_context(agent, session):
    """Log every context message and instruction after the model run completes."""
    from agent_framework import ContextProvider, SessionContext, AgentSession
    from typing import Any

    class ContextDebugger(ContextProvider):
        def __init__(self):
            super().__init__(source_id="debugger")

        async def before_run(self, *, agent: Any, session: AgentSession, context: SessionContext, state: dict[str, Any]):
            pass  # other providers run first (order matters)

        async def after_run(self, *, agent: Any, session: AgentSession, context: SessionContext, state: dict[str, Any]):
            # Runs after all providers have completed before_run AND the model has responded.
            all_ctx = context.get_messages(include_input=True, include_response=True)
            print(f"  messages in context: {len(all_ctx)}")
            print(f"  instructions added: {len(context.instructions)}")
            print(f"  extra tools registered: {len(context.tools)}")
            for source, msgs in context.context_messages.items():
                print(f"    [{source}] → {len(msgs)} messages")
```

### `PerServiceCallHistoryPersistingMiddleware`

When `require_per_service_call_history_persistence=True` is set on an agent, the framework wraps each model call with a `PerServiceCallHistoryPersistingMiddleware` that flushes history after every model roundtrip — not just at the end of the outer `agent.run()`. This is useful for long-running agentic loops where a process crash mid-run would otherwise lose all intermediate turns:

```python
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

history = FileHistoryProvider(storage_path="./conversations")

# Flush history after EVERY model call inside the tool loop, not just after the outer run.
agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    context_providers=[history],
    require_per_service_call_history_persistence=True,
)

session = agent.create_session(session_id="long-research-session")
# Even if this crashes after 20 tool calls, the next run will resume with all 20 turns already persisted.
response = await agent.run(
    "Research the top 10 papers on diffusion models and summarise each one.",
    session=session,
)
```

The trade-off: every model call now incurs a history write. For short, single-turn agents this is wasted I/O; for multi-tool research agents it protects hours of work from a single process crash.

## See also

- [Compaction](./microsoft_agent_framework_python_compaction/) — pair a `CompactionStrategy` with the history provider for long-running conversations.
- [Middleware](./microsoft_agent_framework_python_middleware/) — read/write `AgentContext.session` from agent middleware.
- [Production guide](./microsoft_agent_framework_python_production_guide/) — deploying stateful sessions across multi-instance services.

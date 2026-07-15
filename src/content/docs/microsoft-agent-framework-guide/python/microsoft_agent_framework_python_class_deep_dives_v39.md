---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 39"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.11.0: CheckpointStorage+FileCheckpointStorage+WorkflowCheckpoint (checkpoint Protocol — 6 methods, hybrid JSON+pickle+base64 format, allowed_checkpoint_types security allowlist, previous_checkpoint_id chaining, graph_signature_hash compatibility guard); InProcRunnerContext (in-process workflow runner — lazy asyncio.Queue loop binding, send/drain messages, event streaming, add_request_info_event HITL correlation, set_runtime_checkpoint_storage per-run override); FunctionInvocationLayer (tool-execution loop MRO mixin — function_middleware pipeline, max_iterations/max_function_calls budget, per-call middleware injection, compaction_strategy/tokenizer overloads); AgentContext (agent middleware context object — agent/messages/session/tools/options/stream/metadata/result fields, stream_transform_hooks PII redaction, client_kwargs/function_invocation_kwargs passthrough); CopilotStudioAgent (Microsoft Copilot Studio bridge — COPILOTSTUDIOAGENT__ env prefix, PowerPlatformCloud/AgentType params, MSAL silent-then-interactive token flow, context_providers/middleware wiring); PurviewPolicyMiddleware+PurviewChatPolicyMiddleware+PurviewSettings (Microsoft Purview compliance — UPLOAD_TEXT pre-check, blocked_prompt_message/blocked_response_message, ignore_exceptions/ignore_payment_required, CacheProvider TTL, MiddlewareTermination short-circuit); DevServer (local OpenAI-compatible debug server — entities_dir scan, developer vs user mode, auth_enabled Bearer token, auto-generated loopback token, CORS allowlist, Host-header allowlist DNS rebinding guard); InMemoryHistoryProvider+FileHistoryProvider (history provider pair — skip_excluded compaction integration, store_context_messages/store_context_from scoping, 64-stripe threading.Lock file writes, Windows reserved name encoding, path-traversal guard); PerServiceCallHistoryPersistingMiddleware (per-model-call history persistence — service_stores_history flag, _prepare_service_call_context load path, _persist_service_call_response after each model call, sentinel conversation ID for local function loop); FunctionInvocationConfiguration+FunctionRequestResult+UserInputRequiredException (tool loop control TypedDict — max_iterations LLM roundtrips vs max_function_calls individual executions, terminate_on_unknown_calls, additional_tools hidden tools, FunctionRequestResult action/result_message/errors_in_a_row, UserInputRequiredException.contents propagation) — source-verified at agent-framework 1.11.0."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 62
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 39

Verified against **agent-framework 1.11.0** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`.

Sub-packages introspected:
`agent_framework._workflows._checkpoint`,
`agent_framework._workflows._runner_context`,
`agent_framework._tools`,
`agent_framework._middleware`,
`agent_framework.microsoft`,
`agent_framework.devui`,
`agent_framework._sessions`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) through [Vol. 38](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v38/) — 380+ classes covered.

This volume covers **ten class groups** across the checkpointing system, the in-process workflow runner, tool-loop internals, agent and chat middleware contexts, the Copilot Studio bridge, Purview compliance, the local debug server, history persistence, and tool-invocation control.

| # | Class / group | Module |
|---|---|---|
| 1 | `CheckpointStorage` · `FileCheckpointStorage` · `WorkflowCheckpoint` | `agent_framework._workflows._checkpoint` |
| 2 | `InProcRunnerContext` | `agent_framework._workflows._runner_context` |
| 3 | `FunctionInvocationLayer` | `agent_framework._tools` |
| 4 | `AgentContext` | `agent_framework._middleware` |
| 5 | `CopilotStudioAgent` | `agent_framework.microsoft` |
| 6 | `PurviewPolicyMiddleware` · `PurviewChatPolicyMiddleware` · `PurviewSettings` | `agent_framework.microsoft` |
| 7 | `DevServer` | `agent_framework.devui` |
| 8 | `InMemoryHistoryProvider` · `FileHistoryProvider` | `agent_framework._sessions` |
| 9 | `PerServiceCallHistoryPersistingMiddleware` | `agent_framework._sessions` |
| 10 | `FunctionInvocationConfiguration` · `FunctionRequestResult` · `UserInputRequiredException` | `agent_framework._tools` |

---

## 1 · Checkpoint System — `CheckpointStorage`, `FileCheckpointStorage`, `WorkflowCheckpoint`

**Module:** `agent_framework._workflows._checkpoint`

Workflow checkpointing lets you pause graph execution, persist all state (messages, variable store, pending HITL events), and resume from any saved point. Three classes form the complete checkpoint system.

### 1a · `CheckpointStorage` — Protocol

`CheckpointStorage` is a `@runtime_checkable` Protocol. Any object that implements its six methods is a valid backend — no inheritance required.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class CheckpointStorage(Protocol):
    async def save(self, checkpoint: WorkflowCheckpoint) -> CheckpointID: ...
    async def load(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint: ...
    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]: ...
    async def delete(self, checkpoint_id: CheckpointID) -> bool: ...
    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None: ...
    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[CheckpointID]: ...
```

| Method | Returns | Notes |
|---|---|---|
| `save(checkpoint)` | `CheckpointID` (str UUID) | Persist a checkpoint; returns its ID |
| `load(checkpoint_id)` | `WorkflowCheckpoint` | Raises `WorkflowCheckpointException` if not found |
| `list_checkpoints(workflow_name=)` | `list[WorkflowCheckpoint]` | All checkpoints for a workflow name |
| `delete(checkpoint_id)` | `bool` | `True` if deleted, `False` if not found |
| `get_latest(workflow_name=)` | `WorkflowCheckpoint \| None` | Newest checkpoint, or `None` |
| `list_checkpoint_ids(workflow_name=)` | `list[CheckpointID]` | IDs only (lighter than full list) |

**Custom backend example — SQLite:**

```python
import asyncio, json, sqlite3, threading
from agent_framework._workflows._checkpoint import CheckpointStorage, WorkflowCheckpoint

class SQLiteCheckpointStorage:
    """Minimal SQLite-backed checkpoint storage.

    Blocking sqlite3 calls are offloaded to a thread via asyncio.to_thread.
    A threading.Lock serialises all DB access so the shared connection is
    never used from two threads simultaneously (sqlite3 is not thread-safe
    for concurrent writes and can release the GIL during I/O).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS checkpoints
               (id TEXT PRIMARY KEY, workflow_name TEXT, data TEXT, ts TEXT)"""
        )
        self._db.commit()

    def _save_sync(self, checkpoint: WorkflowCheckpoint) -> str:
        with self._lock:
            data = json.dumps(checkpoint.to_dict(), default=str)
            self._db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
                (checkpoint.checkpoint_id, checkpoint.workflow_name, data, checkpoint.timestamp),
            )
            self._db.commit()
            return checkpoint.checkpoint_id

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        return await asyncio.to_thread(self._save_sync, checkpoint)

    def _load_sync(self, checkpoint_id: str) -> WorkflowCheckpoint:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM checkpoints WHERE id=?", (checkpoint_id,)
            ).fetchone()
        if not row:
            from agent_framework.exceptions import WorkflowCheckpointException
            raise WorkflowCheckpointException(f"Checkpoint {checkpoint_id} not found")
        return WorkflowCheckpoint.from_dict(json.loads(row[0]))

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint:
        return await asyncio.to_thread(self._load_sync, checkpoint_id)

    def _list_sync(self, workflow_name: str) -> list[WorkflowCheckpoint]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM checkpoints WHERE workflow_name=? ORDER BY ts",
                (workflow_name,),
            ).fetchall()
        return [WorkflowCheckpoint.from_dict(json.loads(r[0])) for r in rows]

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        return await asyncio.to_thread(self._list_sync, workflow_name)

    def _delete_sync(self, checkpoint_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM checkpoints WHERE id=?", (checkpoint_id,))
            self._db.commit()
        return cur.rowcount > 0

    async def delete(self, checkpoint_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, checkpoint_id)

    def _get_latest_sync(self, workflow_name: str) -> WorkflowCheckpoint | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM checkpoints WHERE workflow_name=? ORDER BY ts DESC LIMIT 1",
                (workflow_name,),
            ).fetchone()
        return WorkflowCheckpoint.from_dict(json.loads(row[0])) if row else None

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        return await asyncio.to_thread(self._get_latest_sync, workflow_name)

    def _list_ids_sync(self, workflow_name: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM checkpoints WHERE workflow_name=? ORDER BY ts",
                (workflow_name,),
            ).fetchall()
        return [r[0] for r in rows]

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]:
        return await asyncio.to_thread(self._list_ids_sync, workflow_name)

# isinstance works because it's a @runtime_checkable Protocol
storage: CheckpointStorage = SQLiteCheckpointStorage("agents.db")
assert isinstance(storage, CheckpointStorage)
```

### 1b · `FileCheckpointStorage` — File-Based Backend

```python
class FileCheckpointStorage:
    def __init__(
        self,
        storage_path: str | Path,
        *,
        allowed_checkpoint_types: list[str] | None = None,
    ) -> None: ...
```

| Parameter | Type | Description |
|---|---|---|
| `storage_path` | `str \| Path` | Directory where checkpoint files are stored (created if absent) |
| `allowed_checkpoint_types` | `list[str] \| None` | Extra types allowed during pickle deserialisation, in `"module:qualname"` format |

**Storage format:** Each checkpoint is a `.json` file. Primitive values are stored as JSON; complex Python objects (Pydantic models, dataclasses) are serialised with `pickle`, base64-encoded, and embedded as strings in the JSON. This makes files human-readable at the top level while still supporting arbitrary Python objects in workflow state.

**Security allowlist:** By default only built-in types, `datetime`, `uuid`, all `agent_framework.*` types, and `openai.types.*` are permitted during deserialisation. Add application types explicitly:

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._checkpoint import FileCheckpointStorage
from pathlib import Path

storage = FileCheckpointStorage(
    Path("/var/checkpoints/my_workflow"),
    allowed_checkpoint_types=[
        "my_app.models:PlanState",
        "my_app.models:ResearchResult",
    ],
)

# Attach to a workflow at build time
builder = WorkflowBuilder(name="research_pipeline", checkpoint_storage=storage)
```

**Runtime override — choose storage per run:**

```python
import asyncio
from agent_framework._workflows._checkpoint import FileCheckpointStorage, InMemoryCheckpointStorage

fast_storage = InMemoryCheckpointStorage()   # for tests
prod_storage = FileCheckpointStorage("/prod/checkpoints")

# Same workflow, different storage per run
workflow = builder.build()

async def run_with_storage(storage):
    result = await workflow.run("Analyse quarterly data", checkpoint_storage=storage)
    return result
```

### 1c · `WorkflowCheckpoint` — State Snapshot

```python
@dataclass(slots=True)
class WorkflowCheckpoint:
    workflow_name: str                              # logical workflow name for grouping
    graph_signature_hash: str                       # topology hash — prevents restore on incompatible graphs
    checkpoint_id: CheckpointID                     # UUID str, auto-generated
    previous_checkpoint_id: CheckpointID | None     # chain link for history
    timestamp: str                                  # ISO 8601 UTC
    messages: dict[str, list[WorkflowMessage]]      # inter-executor messages per source_id
    state: dict[str, Any]                           # committed workflow state + '_executor_state'
    pending_request_info_events: dict[str, WorkflowEvent[Any]]  # un-answered HITL events
    iteration_count: int                            # superstep count at time of checkpoint
    metadata: dict[str, Any]                        # framework metadata (graph signature etc.)
    version: str                                    # checkpoint format version ("1.0")
```

> **Key invariant:** `state` only contains *committed* values. Pending (not-yet-committed) `State` changes are never included in a checkpoint. The `_executor_state` reserved key stores per-executor sub-dicts.

**Checkpoint chaining — auditable history:**

```python
async def run_with_chain(storage: FileCheckpointStorage) -> None:
    workflow = builder.build()
    result = await workflow.run("Step 1: gather data", checkpoint_storage=storage)

    checkpoints = await storage.list_checkpoints(workflow_name="research_pipeline")
    latest = checkpoints[-1]

    print(f"Checkpoint: {latest.checkpoint_id}")
    print(f"Previous:   {latest.previous_checkpoint_id}")
    print(f"Iteration:  {latest.iteration_count}")
    print(f"Pending:    {list(latest.pending_request_info_events.keys())}")

    # Resume from latest checkpoint
    result2 = await workflow.run(
        "Step 2: summarise",
        checkpoint_id=latest.checkpoint_id,
        checkpoint_storage=storage,
    )
```

---

## 2 · `InProcRunnerContext` — In-Process Workflow Runner

**Module:** `agent_framework._workflows._runner_context`

`InProcRunnerContext` is the concrete execution context used when you call `workflow.run()` locally. It coordinates message passing between executors, streams `WorkflowEvent` objects, and integrates with the checkpoint system.

```python
class InProcRunnerContext:
    def __init__(self, checkpoint_storage: CheckpointStorage | None = None) -> None: ...
```

### Core subsystems

**Message routing — inter-executor communication:**

```python
# Executors call ctx.send_message() to pass WorkflowMessages to each other.
# The runner drains the buffer at each superstep boundary.
async def send_message(self, message: WorkflowMessage) -> None: ...
async def drain_messages(self) -> dict[str, list[WorkflowMessage]]: ...
async def has_messages(self) -> bool: ...
```

**Event streaming — real-time progress:**

The event queue uses **lazy asyncio loop binding**: the `Queue` is created on first use under the running event loop. This means a single `InProcRunnerContext` can be reused across multiple `asyncio.run()` calls without raising "bound to a different event loop" errors.

```python
async def add_event(self, event: WorkflowEvent) -> None: ...
async def drain_events(self) -> list[WorkflowEvent]: ...  # non-blocking, drains all queued
async def next_event(self) -> WorkflowEvent: ...           # blocking, waits for next
async def has_events(self) -> bool: ...
```

**HITL correlation — request/response pairing:**

```python
async def add_request_info_event(self, event: WorkflowEvent[Any]) -> None: ...
async def send_request_info_response(self, request_id: str, response: Any) -> None: ...
async def get_pending_request_info_events(self) -> dict[str, WorkflowEvent[Any]]: ...
```

`add_request_info_event` validates `event.type == "request_info"` and stores it in `_pending_request_info_events` keyed by `request_id`. `send_request_info_response` pops the pending event, type-validates the response if `event.response_type` was set, and sends a `MessageType.RESPONSE` `WorkflowMessage` back to the source executor.

**Per-run checkpoint override:**

```python
def set_runtime_checkpoint_storage(self, storage: CheckpointStorage) -> None: ...
def clear_runtime_checkpoint_storage(self) -> None: ...
```

`set_runtime_checkpoint_storage` takes precedence over the `checkpoint_storage` passed to `__init__`. The workflow runner calls `clear_runtime_checkpoint_storage()` automatically after each run completes, so the next run starts clean.

**Output classification:**

```python
def set_yield_output_classifier(self, classifier: YieldOutputClassifier) -> None: ...
def classify_yielded_output(self, executor_id: str) -> YieldOutputEventType | None: ...
```

The `YieldOutputClassifier` is a callable `(executor_id: str) -> "output" | "intermediate" | None`. Setting it before a run lets you route some executors to `intermediate_output_from` rather than the primary `output_from`.

**Example — consume events while a workflow runs:**

```python
import asyncio
from agent_framework._workflows._runner_context import InProcRunnerContext
from agent_framework._workflows._checkpoint import FileCheckpointStorage

async def run_with_events(workflow, prompt: str) -> None:
    ctx = InProcRunnerContext(
        checkpoint_storage=FileCheckpointStorage("/tmp/cp")
    )

    async def consume_events() -> None:
        while True:
            event = await ctx.next_event()
            if event.type == "started":
                print("Workflow started")
            elif event.type == "status":
                print(f"Status → {event.state}")
            elif event.type == "output":
                print(f"Output from {event.executor_id}: {event.data}")
            elif event.type in ("failed", "completed"):
                print(f"Done: {event.type}")
                break

    # Run workflow and consume events concurrently.
    # runner_context=ctx links ctx's event queue to this workflow run;
    # without it ctx.next_event() would block indefinitely.
    task = asyncio.create_task(consume_events())
    result = await workflow.run(prompt, runner_context=ctx)
    await task
```

---

## 3 · `FunctionInvocationLayer` — Tool-Execution Loop MRO Mixin

**Module:** `agent_framework._tools`

`FunctionInvocationLayer` is a generic MRO mixin that wraps any `BaseChatClient` subclass and adds the tool-execution loop. All concrete clients (`OpenAIChatClient`, `FoundryChatClient`, `AnthropicClient`, etc.) inherit from it.

```python
class FunctionInvocationLayer(Generic[OptionsCoT]):
    def __init__(
        self,
        *,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        **kwargs: Any,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `middleware` | List of `ChatMiddleware`, `FunctionMiddleware`, or callables. Mixed lists are categorised automatically: chat middleware goes to the pipeline layer, function middleware governs individual tool invocations. |
| `function_invocation_configuration` | TypedDict override for the tool loop (see §10). Merged with defaults at construction time. |

### Function middleware pipeline

`FunctionInvocationLayer` maintains its own `function_middleware` list separate from the chat middleware list. A `FunctionMiddlewarePipeline` is built lazily and cached by list identity:

```python
@property
def function_middleware(self) -> list[FunctionMiddlewareTypes]: ...
def _get_function_middleware_pipeline(
    self,
    middleware: Sequence[FunctionMiddlewareTypes],
) -> FunctionMiddlewarePipeline: ...
```

### `get_response()` — typed overloads

The layer provides three overloads of `get_response()`:

```python
# Non-streaming with typed response model
def get_response(messages, *, stream=False, options: ChatOptions[ResponseModelBoundT], ...) -> Awaitable[ChatResponse[ResponseModelBoundT]]: ...

# Non-streaming with plain options
def get_response(messages, *, stream=False, options: OptionsCoT | None = None, ...) -> Awaitable[ChatResponse[Any]]: ...

# Streaming
def get_response(messages, *, stream=True, ...) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...
```

### Per-call middleware injection

You can inject middleware for a single `get_response` call without touching the client's permanent middleware list:

```python
from agent_framework.openai import OpenAIChatClient
from agent_framework import Message
from agent_framework._middleware import ChatMiddleware, ChatContext

class RequestLoggingMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next):
        print(f"[request] {len(context.messages)} messages")
        await call_next()
        print(f"[response] {context.result}")

client = OpenAIChatClient(model="gpt-4o")

response = await client.get_response(
    [Message(role="user", contents=["Hello"])],
    middleware=[RequestLoggingMiddleware()],   # per-call only, not permanent
)
```

### Compaction and tokenizer injection

```python
from agent_framework._compaction import TruncationStrategy, CharacterEstimatorTokenizer

tokenizer = CharacterEstimatorTokenizer()
compaction = TruncationStrategy(max_n=8000, compact_to=6000, tokenizer=tokenizer)

response = await client.get_response(
    messages,
    compaction_strategy=compaction,
    tokenizer=tokenizer,
)
```

---

## 4 · `AgentContext` — Agent Middleware Context Object

**Module:** `agent_framework._middleware`

`AgentContext` is passed to every `AgentMiddleware.process()` call. It carries the full invocation state and lets middleware read, modify, or short-circuit the agent run.

```python
class AgentContext:
    def __init__(
        self,
        *,
        agent: SupportsAgentRun,
        messages: list[Message],
        session: AgentSession | None = None,
        tools: ToolTypes | Callable | Sequence[...] | None = None,
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        metadata: Mapping[str, Any] | None = None,
        result: AgentResponse | ResponseStream[...] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        stream_transform_hooks: Sequence[Callable] | None = None,
        stream_result_hooks: Sequence[Callable] | None = None,
        stream_cleanup_hooks: Sequence[Callable] | None = None,
    ) -> None: ...
```

### Key fields

| Field | Type | Usage |
|---|---|---|
| `agent` | `SupportsAgentRun` | The agent being invoked (read-only — mutating it doesn't affect execution) |
| `messages` | `list[Message]` | Mutable — prepend context, inject system messages, redact PII |
| `session` | `AgentSession \| None` | Access `session.state` for cross-turn data |
| `tools` | tool list | Per-run tool overrides |
| `options` | `dict` | Model options (model, temperature, etc.) |
| `stream` | `bool` | Whether this is a streaming invocation |
| `metadata` | `dict` | Shared dict between middleware — write timing, cost, token counts |
| `result` | `AgentResponse \| ResponseStream \| None` | Readable *after* `call_next()`; settable to short-circuit |
| `client_kwargs` | `dict` | Forwarded to the underlying chat client |
| `function_invocation_kwargs` | `dict` | Forwarded to the tool-invocation layer |

### Stream hook registration

```python
# stream_transform_hooks run on every AgentResponseUpdate chunk (before accumulation)
# stream_result_hooks run on the final AgentResponse (after streaming ends)
# stream_cleanup_hooks run last (even on cancellation)
context.stream_transform_hooks.append(my_redaction_fn)
context.stream_result_hooks.append(my_audit_fn)
context.stream_cleanup_hooks.append(my_teardown_fn)
```

### Usage patterns

**1 — Read-only timing middleware:**

```python
import time
from agent_framework import AgentMiddleware, AgentContext

class TimingMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        start = time.perf_counter()
        context.metadata["agent_name"] = context.agent.name
        await call_next()
        elapsed = time.perf_counter() - start
        context.metadata["elapsed_ms"] = round(elapsed * 1000, 2)
        print(f"Agent '{context.agent.name}' took {elapsed*1000:.0f} ms")
```

**2 — Message prepend before model call:**

```python
from agent_framework import AgentMiddleware, AgentContext, Message

class DateInjectionMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        from datetime import date
        today = Message(role="system", contents=[f"Today is {date.today().isoformat()}."])
        context.messages = [today, *context.messages]
        await call_next()
```

**3 — Short-circuit with cached result:**

```python
from agent_framework import AgentMiddleware, AgentContext, AgentResponse, Message
from agent_framework._middleware import MiddlewareTermination

class CachingMiddleware(AgentMiddleware):
    def __init__(self):
        self._cache: dict[str, AgentResponse] = {}

    async def process(self, context: AgentContext, call_next):
        key = str(context.messages[-1].contents) if context.messages else ""
        if key in self._cache:
            context.result = self._cache[key]
            raise MiddlewareTermination         # skip call_next entirely

        await call_next()
        if context.result:
            self._cache[key] = context.result
```

**4 — PII redaction on streaming chunks:**

```python
import re
from agent_framework import AgentMiddleware, AgentContext, AgentResponseUpdate, Content

class PIIRedactionMiddleware(AgentMiddleware):
    _CARD_RE = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")

    async def process(self, context: AgentContext, call_next):
        def redact(update: AgentResponseUpdate) -> AgentResponseUpdate:
            if not update.text:
                return update
            # Rebuild contents, redacting text within each Content item.
            # AgentResponseUpdate.contents is Sequence[Content]; pass Content
            # objects (not bare strings) to avoid a TypeError.
            new_contents = [
                Content.from_text(text=self._CARD_RE.sub("[CARD REDACTED]", item.text))
                if hasattr(item, "text") and item.text
                else item
                for item in (update.contents or [])
            ]
            return AgentResponseUpdate(contents=new_contents)

        context.stream_transform_hooks.append(redact)
        await call_next()
```

---

## 5 · `CopilotStudioAgent` — Microsoft Copilot Studio Bridge

**Module:** `agent_framework.microsoft`
**Install:** `pip install agent-framework-copilot-studio`

`CopilotStudioAgent` wraps a published Copilot Studio app as a framework-native agent. It handles authentication via MSAL, routes each turn as a DirectLine conversation, and supports the same `context_providers` and `middleware` as any other agent.

```python
class CopilotStudioAgent(BaseAgent):
    def __init__(
        self,
        client: CopilotClient | None = None,
        settings: ConnectionSettings | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: list[AgentMiddlewareTypes] | None = None,
        # Auth / connection (used only when client=None)
        environment_id: str | None = None,
        agent_identifier: str | None = None,
        client_id: str | None = None,
        tenant_id: str | None = None,
        token: str | None = None,
        cloud: PowerPlatformCloud | None = None,
        agent_type: AgentType | None = None,
        custom_power_platform_cloud: str | None = None,
        username: str | None = None,
        token_cache: Any | None = None,
        scopes: list[str] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### Environment variable prefix

All connection parameters can be set via environment variables with the `COPILOTSTUDIOAGENT__` prefix (double underscore):

| Env var | Constructor param |
|---|---|
| `COPILOTSTUDIOAGENT__ENVIRONMENTID` | `environment_id` |
| `COPILOTSTUDIOAGENT__SCHEMANAME` | `agent_identifier` |
| `COPILOTSTUDIOAGENT__AGENTAPPID` | `client_id` |
| `COPILOTSTUDIOAGENT__TENANTID` | `tenant_id` |

### Authentication flow

When `token=None`, the agent acquires a token via MSAL: silent (from `token_cache`) → interactive. Supply a `token_cache` (any MSAL-compatible cache object) to share tokens across agent instances.

### Basic usage

```python
import asyncio
from agent_framework.microsoft import CopilotStudioAgent

# Credentials from environment variables
agent = CopilotStudioAgent(
    name="HR Assistant",
    description="Routes HR questions to Copilot Studio",
)

async def main():
    response = await agent.run("How do I request annual leave?")
    print(response.text)

asyncio.run(main())
```

### With explicit params

```python
from agent_framework.microsoft import CopilotStudioAgent
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

copilot = CopilotStudioAgent(
    environment_id="00000000-0000-0000-0000-000000000001",
    agent_identifier="hr_assistant_v2",
    client_id="app-registration-client-id",
    tenant_id="your-tenant-id",
    cloud="Public",                         # PowerPlatformCloud literal
    agent_type="Agent",                     # or "Copilot"
)

# Combine with a local agent in a HandoffBuilder
from agent_framework.orchestrations import HandoffBuilder

router = Agent(
    client=OpenAIChatClient(model="gpt-4o-mini"),
    name="router",
    instructions="Route HR questions to copilot, technical questions answer directly.",
)

orchestration = (
    HandoffBuilder()
    .add_agent(router, handoffs=[copilot])
    .build()
)

async def main():
    result = await orchestration.run("I need to update my emergency contact.")
    print(result.final_message.text)
```

### Streaming turns

```python
async def stream_copilot():
    agent = CopilotStudioAgent(name="support")
    async for update in agent.run("Check order status", stream=True):
        print(update.text, end="", flush=True)
    print()
```

---

## 6 · Purview Compliance — `PurviewPolicyMiddleware`, `PurviewChatPolicyMiddleware`, `PurviewSettings`

**Module:** `agent_framework.microsoft`
**Install:** `pip install agent-framework-purview`

Microsoft Purview is a data-governance and compliance platform. `PurviewPolicyMiddleware` and `PurviewChatPolicyMiddleware` enforce Purview data-loss-prevention (DLP) policies on agent prompts and responses.

### `PurviewSettings` — TypedDict

```python
class PurviewSettings(TypedDict, total=False):
    app_name: str | None                     # identifies the calling application to Purview
    app_version: str | None                  # optional version string
    tenant_id: str | None                    # Azure tenant GUID
    purview_app_location: PurviewAppLocation | None  # enum: TEAMS, EXCHANGE, SHAREPOINT, etc.
    graph_base_uri: str | None               # Microsoft Graph base URI
    blocked_prompt_message: str | None       # custom message returned when a prompt is blocked
    blocked_response_message: str | None     # custom message returned when a response is blocked
    ignore_exceptions: bool | None           # log-but-continue on all Purview errors
    ignore_payment_required: bool | None     # log-but-continue on 402 payment errors
    cache_ttl_seconds: int | None            # policy cache TTL (default 14 400 s = 4 h)
    max_cache_size_bytes: int | None         # policy cache max size (default 200 MB)
```

### `PurviewPolicyMiddleware` — Agent-level enforcement

Intercepts at the **agent middleware** layer (wraps the full agent run):

```python
class PurviewPolicyMiddleware(AgentMiddleware):
    def __init__(
        self,
        credential: AzureCredentialTypes | AzureTokenProvider,
        settings: PurviewSettings,
        cache_provider: CacheProvider | None = None,
    ) -> None: ...
```

**Policy check flow:**
1. **Pre-check (prompt):** Calls Purview with `Activity.UPLOAD_TEXT`. If the policy blocks the prompt, sets `context.result` to a blocked message and raises `MiddlewareTermination`.
2. **Execute agent** via `await call_next()`.
3. **Post-check (response):** Calls Purview with `Activity.GENERATE_RESPONSE`. If the response is blocked, replaces it with `blocked_response_message`.

```python
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.microsoft import PurviewPolicyMiddleware, PurviewSettings
from agent_framework.openai import OpenAIChatClient

credential = DefaultAzureCredential()
purview_settings = PurviewSettings(
    app_name="ContosoAssistant",
    app_version="2.0.0",
    purview_app_location="Teams",
    blocked_prompt_message="Your message was blocked by company policy.",
    blocked_response_message="The response was blocked by company policy.",
    ignore_payment_required=True,           # don't crash if Purview billing is not set up
    cache_ttl_seconds=3600,                 # 1-hour policy cache
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="assistant",
    instructions="You are a helpful enterprise assistant.",
    middleware=[PurviewPolicyMiddleware(credential, purview_settings)],
)
```

### `PurviewChatPolicyMiddleware` — Chat-level enforcement

Intercepts at the **chat middleware** layer — useful when you want Purview enforcement on a chat client that is used directly rather than through an `Agent`:

```python
from agent_framework.microsoft import PurviewChatPolicyMiddleware

client = OpenAIChatClient(
    model="gpt-4o",
    middleware=[PurviewChatPolicyMiddleware(credential, purview_settings)],
)
```

### `CacheProvider` integration

Supply a `CacheProvider` to share the Purview policy cache across agent instances (e.g. in a multi-worker deployment):

```python
from agent_framework import CacheProvider     # Protocol
import redis.asyncio as aioredis             # async client — never blocks the event loop

class RedisCache(CacheProvider):
    def __init__(self, r: aioredis.Redis):
        self._r = r

    async def get(self, key: str) -> bytes | None:
        return await self._r.get(key)

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        await self._r.setex(key, ttl, value)

cache = RedisCache(aioredis.Redis())
policy = PurviewPolicyMiddleware(
    credential,
    PurviewSettings(app_name="ContosoAssistant", cache_ttl_seconds=3600),
    cache_provider=cache,
)
```

### Exception handling

| Exception | Condition | Default behaviour |
|---|---|---|
| `PurviewAuthenticationError` | MSAL auth failure | Raised unless `ignore_exceptions=True` |
| `PurviewPaymentRequiredError` | 402 from Purview | Raised unless `ignore_payment_required=True` |
| `PurviewRateLimitError` | 429 from Purview | Raised unless `ignore_exceptions=True` |
| `PurviewRequestError` | 4xx from Purview | Raised unless `ignore_exceptions=True` |
| `PurviewServiceError` | 5xx from Purview | Raised unless `ignore_exceptions=True` |

---

## 7 · `DevServer` — Local OpenAI-Compatible Debug Server

**Module:** `agent_framework.devui`
**Install:** `pip install agent-framework-devui`

`DevServer` launches an OpenAI-compatible HTTP API that fronts one or more agents. It is designed for local development and debugging — it is **not** intended for production. Use Azure Container Apps or Azure Functions for production deployments.

```python
class DevServer:
    def __init__(
        self,
        entities_dir: str | None = None,
        port: int = 8080,
        host: str = "127.0.0.1",
        cors_origins: list[str] | None = None,
        ui_enabled: bool = True,
        mode: str = "developer",
        auth_enabled: bool = True,
        auth_token: str | None = None,
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `entities_dir` | `None` | Directory to scan for `.agent.py` / `.workflow.py` files |
| `port` | `8080` | TCP port |
| `host` | `"127.0.0.1"` | Bind address. Non-loopback hosts **require** `auth_enabled=True` and an explicit token |
| `cors_origins` | `None` | Explicit CORS allowlist. `None` is treated as an empty allowlist — same-origin only, no wildcard |
| `ui_enabled` | `True` | Whether to serve the built-in DevUI web interface |
| `mode` | `"developer"` | `"developer"` = verbose error details; `"user"` = generic errors only |
| `auth_enabled` | `True` | Require `Authorization: Bearer <token>` on all `/v1/*` endpoints |
| `auth_token` | `None` | Token value. Falls back to `DEVUI_AUTH_TOKEN` env var. On loopback, auto-generates and logs a token |

### Security constraints

- Auth cannot be disabled on non-loopback hosts (`ValueError` at init time).
- CORS default is an empty allowlist — previous wildcard-on-localhost behaviour was removed to prevent cross-origin reads from pages visited in the same browser.
- Host-header enforcement is enabled on loopback binds to guard against DNS rebinding attacks.

### Quickstart

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import DevServer

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="assistant",
    instructions="You are a helpful assistant.",
)

server = DevServer(
    port=8080,
    host="127.0.0.1",
    auth_enabled=False,   # only safe on loopback
    ui_enabled=True,
)
server.set_pending_entities([agent])

asyncio.run(server.start())
```

The DevUI is then available at `http://127.0.0.1:8080` and the OpenAI-compatible API at `http://127.0.0.1:8080/v1/chat/completions`.

### Network-accessible server (team shared dev)

```python
import os
server = DevServer(
    port=8080,
    host="0.0.0.0",         # listen on all interfaces
    auth_enabled=True,
    auth_token=os.environ["DEVUI_AUTH_TOKEN"],   # mandatory for non-loopback
    cors_origins=["http://localhost:3000"],       # only allow the React dev server
    mode="user",            # hide internal error details from end users
)
```

### Register agents from a directory

Create a file `my_agents/research.agent.py`:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="research",
    instructions="You are a research assistant.",
)
```

Then point `entities_dir` at the folder:

```python
server = DevServer(entities_dir="./my_agents", port=8080)
```

All `.agent.py` files in the directory are discovered and registered automatically.

---

## 8 · History Providers — `InMemoryHistoryProvider`, `FileHistoryProvider`

**Module:** `agent_framework._sessions`

History providers control what messages are stored between agent runs and how they are retrieved. The two built-in providers cover the most common patterns; both share the same set of storage control flags.

### 8a · `InMemoryHistoryProvider`

Stores messages in `session.state["messages"]` (serialised via `AgentSession.to_dict()` / `from_dict()`). This is the default provider added automatically when no providers are configured.

```python
class InMemoryHistoryProvider(HistoryProvider):
    DEFAULT_SOURCE_ID: ClassVar[str] = "in_memory"

    def __init__(
        self,
        source_id: str | None = None,
        *,
        load_messages: bool = True,
        store_inputs: bool = True,
        store_context_messages: bool = False,
        store_context_from: set[str] | None = None,
        store_outputs: bool = True,
        skip_excluded: bool = False,
    ) -> None: ...
```

| Flag | Default | Effect |
|---|---|---|
| `load_messages` | `True` | Load stored messages before each run |
| `store_inputs` | `True` | Persist the user's input messages |
| `store_context_messages` | `False` | Persist context injected by other providers (e.g. memory, search results) |
| `store_context_from` | `None` | If set, only persist context from these `source_id`s |
| `store_outputs` | `True` | Persist the agent's response messages |
| `skip_excluded` | `False` | When `True`, omit compaction-excluded messages on load (so the loaded context reflects compacted state) |

**`skip_excluded` with compaction:**

```python
from agent_framework._sessions import InMemoryHistoryProvider
from agent_framework._compaction import CompactionProvider, TruncationStrategy, CharacterEstimatorTokenizer

history = InMemoryHistoryProvider(skip_excluded=True)  # excluded messages are not reloaded

compaction = CompactionProvider(
    before_strategy=TruncationStrategy(
        max_n=4000, compact_to=3000, tokenizer=CharacterEstimatorTokenizer()
    )
)

agent = Agent(
    client=client,
    name="assistant",
    context_providers=[history, compaction],
)
```

**Context scoping — store only memory provider context:**

```python
history = InMemoryHistoryProvider(
    store_context_messages=True,
    store_context_from={"memory"},          # only persist messages from the memory provider
)
```

### 8b · `FileHistoryProvider`

```python
@experimental(feature_id=ExperimentalFeature.FILE_HISTORY)
class FileHistoryProvider(HistoryProvider):
    DEFAULT_SOURCE_ID: ClassVar[str] = "file_history"

    def __init__(
        self,
        storage_path: str | Path,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        load_messages: bool = True,
        store_inputs: bool = True,
        store_context_messages: bool = False,
        store_context_from: set[str] | None = None,
        store_outputs: bool = True,
        skip_excluded: bool = False,
        dumps: JsonDumps | None = None,
        loads: JsonLoads | None = None,
    ) -> None: ...
```

`FileHistoryProvider` writes one **JSON Lines** (`.jsonl`) file per session. Each persisted message is one JSON object on a single line.

**Security posture:** Files are stored as plaintext on the local filesystem. Use OS-level file permissions on `storage_path`. The provider path-traversal-guards `session_id` (encodes characters that could escape the directory) and encodes Windows reserved names (`CON`, `NUL`, `COM1`…`COM9`, `LPT1`…`LPT9`).

**Concurrency:** File writes use a 64-slot `threading.Lock` stripe array. The stripe is selected by `hash(file_path) % 64`, so concurrent writes to different sessions rarely contend.

```python
from pathlib import Path
from agent_framework._sessions import FileHistoryProvider, InMemoryHistoryProvider

# Production: persist across restarts via FileHistoryProvider
file_history = FileHistoryProvider(
    storage_path=Path("/var/agent-history"),
    skip_excluded=True,           # respect compaction exclusions
    store_outputs=True,
    store_inputs=True,
)

# Development: in-memory only
dev_history = InMemoryHistoryProvider()

agent = Agent(
    client=client,
    name="assistant",
    context_providers=[file_history],       # swap dev_history for testing
)
```

**Custom JSON serialiser:**

```python
import orjson

history = FileHistoryProvider(
    "/var/agent-history",
    dumps=lambda obj: orjson.dumps(obj).decode(),   # faster serialisation
    loads=orjson.loads,
)
```

---

## 9 · `PerServiceCallHistoryPersistingMiddleware`

**Module:** `agent_framework._sessions`

This chat middleware persists history after **every individual model call** within a single agent run, rather than only at the end of the run. It is activated when `require_per_service_call_history_persistence=True` is set on the agent. Use it when you need durability against mid-run crashes.

```python
class PerServiceCallHistoryPersistingMiddleware(ChatMiddleware):
    def __init__(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        providers: Sequence[HistoryProvider],
        service_stores_history: bool = False,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `agent` | The agent that owns the history providers |
| `session` | The active session for the current run |
| `providers` | The history providers to persist after each model call |
| `service_stores_history` | When `True`, the chat client stores history server-side; the middleware is write-only and skips loading |

### How it works

**When `service_stores_history=False`** (local history management):
1. Before each model call, the middleware creates a `SessionContext` from the current messages.
2. It loads each provider that has `load_messages=True` into that context.
3. It uses a **local sentinel conversation ID** in `ChatContext.kwargs["conversation_id"]` so the function-invocation loop runs without forwarding that sentinel to the leaf client.
4. After the model call, it calls `provider.after_run()` for each provider (in reverse order), persisting the incremental response.

**When `service_stores_history=True`** (server-side history):
- Loading is skipped (the service owns it).
- The real `conversation_id` is preserved.
- The middleware is write-only: it only calls `after_run()` to sink responses.

### Example — crash-safe long-running agent

```python
from agent_framework import Agent
from agent_framework._sessions import FileHistoryProvider

history = FileHistoryProvider("/var/agent-history")

agent = Agent(
    client=client,
    name="long_runner",
    instructions="Complete multi-step research tasks.",
    context_providers=[history],
    # Activating this flag wires PerServiceCallHistoryPersistingMiddleware automatically:
    require_per_service_call_history_persistence=True,
)

session = AgentSession()
result = await agent.run(
    "Research quantum computing applications in cryptography. Write a 5-section report.",
    session=session,
)
# If the process crashes mid-run, history is already persisted up to the last model call.
```

---

## 10 · Tool Loop Control — `FunctionInvocationConfiguration`, `FunctionRequestResult`, `UserInputRequiredException`

**Module:** `agent_framework._tools`

### 10a · `FunctionInvocationConfiguration` — TypedDict

Controls the tool-execution loop that runs when the model requests function calls.

```python
class FunctionInvocationConfiguration(TypedDict, total=False):
    enabled: bool
    max_iterations: int
    max_function_calls: int | None
    max_consecutive_errors_per_request: int
    terminate_on_unknown_calls: bool
    additional_tools: Sequence[FunctionTool]
    include_detailed_errors: bool
```

| Key | Description |
|---|---|
| `enabled` | Master switch for the function-invocation loop (`True` by default) |
| `max_iterations` | Maximum number of **LLM round-trips**. Each trip may execute multiple tools in parallel, so this does not directly cap total tool calls |
| `max_function_calls` | Maximum **total individual tool invocations** across all iterations. Checked after each parallel batch — if 20 tools are called when the limit is 10, all 20 execute before the loop stops (best-effort) |
| `max_consecutive_errors_per_request` | How many back-to-back tool errors cause the loop to abort |
| `terminate_on_unknown_calls` | If `True`, raise an error when the model requests a tool that is not in the tool map |
| `additional_tools` | Extra `FunctionTool` objects available to the loop but **not advertised** in the tool list sent to the model |
| `include_detailed_errors` | If `True`, include exception details in the tool result returned to the model |

**Setting per-client:**

```python
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(model="gpt-4o")

# Cap to 3 LLM round-trips and 15 total tool executions
client.function_invocation_configuration["max_iterations"] = 3
client.function_invocation_configuration["max_function_calls"] = 15
client.function_invocation_configuration["terminate_on_unknown_calls"] = True
client.function_invocation_configuration["include_detailed_errors"] = True   # dev mode
```

**Why two caps?**

`max_iterations` and `max_function_calls` serve different purposes:

```
Single LLM round-trip (1 iteration) can execute N tools in parallel.
max_iterations=5 with max_function_calls=20:
  - Up to 5 round-trips
  - At most 20 individual tool calls total across all of them
  - Loop stops whichever limit is hit first
```

**Hidden tools via `additional_tools`:**

Hidden tools are available for the framework to call but are never listed in the tool schema given to the model. This is useful for infrastructure-level tools like internal logging, approval callbacks, or framework-generated tools:

```python
from agent_framework import tool, Agent
from agent_framework.openai import OpenAIChatClient

@tool(name="internal_log", approval_mode="never_require")
async def internal_log(message: str, level: str = "info") -> str:
    """Internal audit log — not shown to the model."""
    print(f"[{level.upper()}] {message}")
    return "logged"

client = OpenAIChatClient(model="gpt-4o")
# @tool already returns a FunctionTool — no wrapper needed
client.function_invocation_configuration["additional_tools"] = [internal_log]
```

### 10b · `FunctionRequestResult` — TypedDict

Returned by `FunctionMiddleware.process()` to control the next step of the tool loop.

```python
class FunctionRequestResult(TypedDict, total=False):
    action: Literal["return", "continue", "stop"]
    errors_in_a_row: int
    result_message: Message
    function_call_count: int
```

| Key | Description |
|---|---|
| `action` | `"return"` — keep the tool result and continue; `"continue"` — re-invoke the model (similar to return but resets error count); `"stop"` — terminate the loop immediately |
| `errors_in_a_row` | Running count of consecutive errors (used for `max_consecutive_errors_per_request`) |
| `result_message` | The tool result `Message` to add to the conversation |
| `function_call_count` | Total tool calls made so far in this request (for `max_function_calls` budget) |

**Custom function middleware that audits tool results:**

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext

class AuditMiddleware(FunctionMiddleware):
    async def process(
        self,
        context: FunctionInvocationContext,
        call_next,
    ):
        tool_name = context.function.name
        args = dict(context.arguments) if hasattr(context.arguments, "__iter__") else {}

        result: FunctionRequestResult = await call_next()

        # Inspect result and optionally override action
        if result.get("action") == "return" and tool_name == "delete_file":
            print(f"[AUDIT] delete_file called with {args}")
            # Force stop if a delete is called more than once per session
            result["action"] = "stop"
        return result
```

### 10c · `UserInputRequiredException` — Propagating HITL from Tools

```python
class UserInputRequiredException(ToolException):
    def __init__(
        self,
        contents: list[Any],
        message: str = "Tool requires user input to proceed.",
    ) -> None:
        super().__init__(message, log_level=None)
        self.contents = contents
```

When a `@tool` function wraps a sub-agent that enters a HITL state, the sub-agent's response may contain `oauth_consent_request` or `function_approval_request` content items. `UserInputRequiredException` is raised to propagate those content items back to the parent agent's response rather than swallowing them as a generic tool error.

```python
from agent_framework import tool, Agent, AgentSession
from agent_framework._tools import UserInputRequiredException

@tool(name="call_specialist_agent")
async def call_specialist_agent(question: str) -> str:
    """Delegate a question to the specialist agent."""
    specialist = Agent(client=specialist_client, name="specialist")
    session = AgentSession()
    response = await specialist.run(question, session=session)

    # Check if the specialist needs user input (e.g. OAuth consent, tool approval).
    # AgentResponse.messages is a list; the final message is messages[-1].
    user_input_items = [
        c for c in response.messages[-1].contents
        if hasattr(c, "type") and c.type in ("oauth_consent_request", "function_approval_request")
    ]
    if user_input_items:
        raise UserInputRequiredException(
            contents=user_input_items,
            message="The specialist agent requires user authorization.",
        )

    return response.text or ""
```

The `FunctionInvocationLayer` catches `UserInputRequiredException` and forwards `exc.contents` directly into the parent agent's response, allowing the UI layer to surface the approval request without losing the tool call context.

---

## Summary

| # | Class group | Key takeaway |
|---|---|---|
| 1 | `CheckpointStorage` + `FileCheckpointStorage` + `WorkflowCheckpoint` | Six-method Protocol; hybrid JSON+pickle format; `allowed_checkpoint_types` security allowlist; `previous_checkpoint_id` chain |
| 2 | `InProcRunnerContext` | Lazy asyncio.Queue loop binding; event streaming; HITL request/response correlation; per-run checkpoint override |
| 3 | `FunctionInvocationLayer` | MRO mixin; function middleware pipeline; per-call middleware injection; typed `get_response` overloads |
| 4 | `AgentContext` | Agent middleware context; mutable `messages`; `metadata` sharing; `stream_*_hooks`; `MiddlewareTermination` short-circuit |
| 5 | `CopilotStudioAgent` | `COPILOTSTUDIOAGENT__` env prefix; MSAL token flow; `PowerPlatformCloud` / `AgentType`; middleware/context_providers |
| 6 | `PurviewPolicyMiddleware` + settings | Pre/post DLP checks; `MiddlewareTermination` on block; error-resilience flags; `CacheProvider` for shared cache |
| 7 | `DevServer` | OpenAI-compatible local server; auth required on non-loopback; no wildcard CORS; developer vs user mode |
| 8 | `InMemoryHistoryProvider` + `FileHistoryProvider` | `skip_excluded` for compaction integration; `store_context_from` scoping; 64-stripe file locking; Windows name encoding |
| 9 | `PerServiceCallHistoryPersistingMiddleware` | Persist after every model call; `service_stores_history` mode; sentinel conversation ID |
| 10 | `FunctionInvocationConfiguration` + `FunctionRequestResult` + `UserInputRequiredException` | `max_iterations` vs `max_function_calls` semantics; hidden tools; `action` control from function middleware; HITL content propagation |

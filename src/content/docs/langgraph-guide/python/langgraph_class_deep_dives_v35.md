---
title: "LangGraph Class Deep-Dives Vol. 35"
description: "Source-verified deep dives into 10 class groups from langgraph==1.2.8 — EncryptedSerializer/CipherProtocol/SerializerProtocol (AES checkpoint encryption), SerdeEvent/register_serde_event_listener/emit_serde_event (thread-safe audit hooks), ReplayState (subgraph time-travel checkpoint hydration), UIMessage/RemoveUIMessage/push_ui_message/delete_ui_message/ui_message_reducer (UI streaming primitives), _RemoteGraphRunStream/_AsyncRemoteGraphRunStream/_ChannelProjection/_ProjectionRegistry (remote-graph run stream adapters), UntrackedValue (non-checkpointed per-step channel), HumanResponse + deprecated HumanInterrupt migration pattern, ManagedValue/ManagedValueSpec/ManagedValueMapping/IsLastStep/RemainingSteps (managed value injection), StreamToolCallHandler/ToolCallWriter/_tool_call_writer ContextVar (tool-call lifecycle streaming), and AsyncQueue/SyncQueue/Semaphore (internal FIFO concurrency primitives)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 35"
  order: 66
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.8` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `EncryptedSerializer` · `CipherProtocol` · `SerializerProtocol`

**Modules:** `langgraph.checkpoint.serde.encrypted` · `langgraph.checkpoint.serde.base`

`EncryptedSerializer` wraps any `SerializerProtocol` with a `CipherProtocol` to produce a checkpoint serializer that encrypts all bytes before they reach the store. The type tag emitted by `dumps_typed` gains a `+<ciphername>` suffix so `loads_typed` can transparently decrypt data that was written with encryption while still reading unencrypted legacy checkpoints (the `+` absence branch).

**Key source facts** (from `langgraph/checkpoint/serde/encrypted.py` and `serde/base.py`):

- `EncryptedSerializer.__init__(cipher, serde=JsonPlusSerializer())` — composable: any `CipherProtocol` × any `SerializerProtocol`.
- `dumps_typed(obj)` → `(f"{typ}+{ciphername}", ciphertext)` — encrypts after the inner serde serialises; the ciphername is returned by `cipher.encrypt()`.
- `loads_typed((enc_type, ciphertext))` — splits on the first `+`; if absent, falls through to inner serde for backward-compat reads of unencrypted checkpoints.
- `EncryptedSerializer.from_pycryptodome_aes(serde, **kwargs)` — classmethod factory; reads `LANGGRAPH_AES_KEY` env var (a plain string; `.encode()` must yield 16/24/32 bytes) or accepts an explicit `key=<bytes>` kwarg. Defaults mode to `AES.MODE_EAX` (authenticated encryption — nonce + tag + ciphertext layout).
- `CipherProtocol` — `Protocol` with `encrypt(plaintext) -> (cipher_name, ciphertext)` and `decrypt(cipher_name, ciphertext) -> plaintext`. Implement this to plug in any cipher (Fernet, ChaCha20, AWS KMS, etc.).
- `SerializerProtocol` — `runtime_checkable Protocol` with `dumps_typed(obj) -> (str, bytes)` and `loads_typed((str, bytes)) -> Any`. The inner serde is replaceable (e.g. `MsgPackSerializer`).

### Example 1 — AES-EAX checkpoint encryption via `from_pycryptodome_aes`

```python
import os
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# 32-byte AES-256 key (set via env var or passed directly)
key = os.urandom(32)
serde = EncryptedSerializer.from_pycryptodome_aes(key=key)

# Round-trip: any Python object
obj = {"messages": ["hello", "world"], "count": 42}
typ, ciphertext = serde.dumps_typed(obj)

print(f"type tag  : {typ!r}")        # 'dict+aes'
print(f"encrypted : {ciphertext[:20]}...")  # raw bytes

recovered = serde.loads_typed((typ, ciphertext))
print(f"recovered : {recovered}")    # {'messages': ['hello', 'world'], 'count': 42}
assert recovered == obj
```

### Example 2 — custom `CipherProtocol` (Fernet wrapper)

```python
from cryptography.fernet import Fernet
from langgraph.checkpoint.serde.base import CipherProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class FernetCipher(CipherProtocol):
    def __init__(self, key: bytes) -> None:
        self._f = Fernet(key)

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "fernet", self._f.encrypt(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "fernet", f"Unknown cipher: {ciphername}"
        return self._f.decrypt(ciphertext)


key = Fernet.generate_key()
serde = EncryptedSerializer(FernetCipher(key), JsonPlusSerializer())

typ, ct = serde.dumps_typed([1, 2, 3])
print(typ)                          # 'list+fernet'
print(serde.loads_typed((typ, ct))) # [1, 2, 3]
```

### Example 3 — backward-compat: reading unencrypted legacy checkpoints

```python
import os
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

key = os.urandom(32)
enc_serde = EncryptedSerializer.from_pycryptodome_aes(key=key)

# A checkpoint written *before* encryption was added (plain jsonplus)
plain_serde = JsonPlusSerializer()
plain_typ, plain_bytes = plain_serde.dumps_typed({"legacy": True})
print(f"plain type tag: {plain_typ!r}")   # 'dict' — no '+ciphername' suffix

# enc_serde.loads_typed falls through to inner serde when '+' is absent
recovered = enc_serde.loads_typed((plain_typ, plain_bytes))
print(recovered)   # {'legacy': True}  — no decryption attempted
```

---

## 2 · `SerdeEvent` · `register_serde_event_listener` · `emit_serde_event`

**Module:** `langgraph.checkpoint.serde.event_hooks`

The serde event-hook system provides a thread-safe, multi-listener publish/subscribe channel for `SerdeEvent` TypedDicts. When a new Python type is deserialised for the first time, `JsonPlusSerializer` emits a `SerdeEvent` here. Listeners can build allowlists, audit logs, or trigger alerts without modifying the core serialiser.

**Key source facts** (from `langgraph/checkpoint/serde/event_hooks.py`):

- `SerdeEvent` — `TypedDict` with `kind: str`, `module: str`, `name: str`, and `method: NotRequired[str]`. `kind` is currently `"serde-allowlist"` when the serialiser encounters an unknown type.
- `_listeners: list[SerdeEventListener]` — module-level list guarded by `_listeners_lock: threading.Lock`. Global scope means any listener registered in one thread is visible in all threads.
- `register_serde_event_listener(listener)` → callable — appends to `_listeners` under the lock and returns an unregister closure. Calling the returned closure removes the listener (swallowing `ValueError` if already removed).
- `emit_serde_event(event)` — takes a copy of the listener list under the lock (short critical section), then calls each listener with isolated failure handling: listener exceptions are logged as `WARNING` and never propagate.
- `SerdeEventListener = Callable[[SerdeEvent], None]` — plain callable; can be sync only (async listeners must schedule themselves).

### Example 1 — audit log: print every new type the serde encounters

```python
from langgraph.checkpoint.serde.event_hooks import (
    SerdeEvent,
    register_serde_event_listener,
    emit_serde_event,
)

log: list[SerdeEvent] = []

def audit_listener(event: SerdeEvent) -> None:
    log.append(event)
    print(f"[serde] {event['kind']} → {event['module']}.{event['name']}")

unregister = register_serde_event_listener(audit_listener)

# Simulate the serializer discovering a new type
emit_serde_event({
    "kind": "serde-allowlist",
    "module": "myapp.models",
    "name": "MyState",
})
# [serde] serde-allowlist → myapp.models.MyState

print(log)  # [{'kind': 'serde-allowlist', 'module': 'myapp.models', 'name': 'MyState'}]

# Cleanup
unregister()
```

### Example 2 — multiple listeners; isolated failure handling

```python
from langgraph.checkpoint.serde.event_hooks import (
    register_serde_event_listener,
    emit_serde_event,
)

results = []

def good_listener(event):
    results.append(f"good: {event['name']}")

def bad_listener(event):
    raise RuntimeError("listener crash — must not propagate!")

unreg_good = register_serde_event_listener(good_listener)
unreg_bad  = register_serde_event_listener(bad_listener)

# bad_listener raises but emit_serde_event swallows and logs it
emit_serde_event({"kind": "serde-allowlist", "module": "m", "name": "T"})

print(results)   # ['good: T']  — good_listener still ran
# No exception raised

unreg_good()
unreg_bad()
```

### Example 3 — build a runtime allowlist from serde events

```python
from threading import Lock
from langgraph.checkpoint.serde.event_hooks import (
    SerdeEvent,
    register_serde_event_listener,
    emit_serde_event,
)

class SerdeAllowlistMonitor:
    def __init__(self) -> None:
        self._lock = Lock()
        self._seen: set[tuple[str, str]] = set()
        self._unregister = register_serde_event_listener(self._on_event)

    def _on_event(self, event: SerdeEvent) -> None:
        if event.get("kind") == "serde-allowlist":
            key = (event["module"], event["name"])
            with self._lock:
                if key not in self._seen:
                    self._seen.add(key)
                    print(f"New type allowed: {key}")

    def close(self) -> None:
        self._unregister()

monitor = SerdeAllowlistMonitor()

emit_serde_event({"kind": "serde-allowlist", "module": "app", "name": "Order"})
# New type allowed: ('app', 'Order')
emit_serde_event({"kind": "serde-allowlist", "module": "app", "name": "Order"})
# (silent — already seen)
emit_serde_event({"kind": "serde-allowlist", "module": "app", "name": "Customer"})
# New type allowed: ('app', 'Customer')

monitor.close()
```

---

## 3 · `ReplayState`

**Module:** `langgraph._internal._replay`

`ReplayState` coordinates checkpoint loading for nested subgraphs during a parent time-travel replay. It ensures that on the **first visit** to each subgraph namespace (per replay invocation), the subgraph loads the checkpoint from *before* the replay point — rather than the latest checkpoint, which may be newer than the replayed parent. Subsequent visits to the same logical subgraph (e.g. in a loop) revert to normal latest-checkpoint loading.

**Key source facts** (from `langgraph/_internal/_replay.py`):

- `__slots__ = ("checkpoint_id", "_visited_ns")` — lightweight; `checkpoint_id` is the parent replay target; `_visited_ns: set[str]` tracks which subgraph namespaces have been loaded.
- `_is_first_visit(checkpoint_ns)` — strips the task-id suffix (`"sub_node:task_id"` → `"sub_node"`) using `NS_END` before checking `_visited_ns`, so the same logical subgraph is recognised across loop iterations with different task IDs.
- `get_checkpoint(checkpoint_ns, checkpointer, checkpoint_config)` — on first visit: calls `checkpointer.list(..., before={"configurable": {"checkpoint_id": self.checkpoint_id}}, limit=1)` to get the latest checkpoint *before* the replay point. On subsequent visits: delegates to `checkpointer.get_tuple(checkpoint_config)`.
- `aget_checkpoint(...)` — async twin; semantically identical but uses `alist` / `aget_tuple`.
- **Shared-by-reference pattern**: the single `ReplayState` instance is attached to the parent config and passed by reference into all derived child configs. Adding a namespace to `_visited_ns` in one call is immediately visible in all others.

### Example 1 — inspect `_is_first_visit` across loop iterations

```python
from langgraph._internal._replay import ReplayState

state = ReplayState(checkpoint_id="ckpt-abc-123")

# Subgraph namespace with task_id suffix (as produced by Pregel)
ns1 = "sub_node:task-0001"   # first loop iteration
ns2 = "sub_node:task-0002"   # second loop iteration

print(state._is_first_visit(ns1))  # True  — first encounter of "sub_node"
print(state._is_first_visit(ns2))  # False — "sub_node" already in _visited_ns

# Different top-level namespace is independent
ns3 = "other_agent:task-0001"
print(state._is_first_visit(ns3))  # True
```

### Example 2 — how `get_checkpoint` hydrates a subgraph during replay

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

def increment(state: State) -> State:
    return {"counter": state["counter"] + 1}

graph = StateGraph(State)
graph.add_node("inc", increment)
graph.add_edge(START, "inc")
graph.add_edge("inc", END)
compiled = graph.compile(checkpointer=InMemorySaver())

# Run three steps to create checkpoint history
config = {"configurable": {"thread_id": "replay-demo"}}
for i in range(3):
    compiled.invoke({"counter": i}, config=config)

# Inspect the checkpoint history
history = list(compiled.get_state_history(config))
print(f"Checkpoints: {len(history)}")

# Replay from the second checkpoint (counter=1 step)
replay_config = history[-2].config
replayed = compiled.invoke(None, config=replay_config)
print(f"Replayed state: {replayed}")
# ReplayState is used internally during this replay to ensure subgraphs
# load the correct pre-replay checkpoint on first visit
```

### Example 3 — manual `ReplayState` usage for custom checkpoint middleware

```python
from langgraph._internal._replay import ReplayState
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
replay_target_id = "ckpt-2026-07-07-001"

state = ReplayState(checkpoint_id=replay_target_id)

# Simulate what Pregel does internally: for each subgraph namespace
# encountered during replay, call state.get_checkpoint(...)
subgraph_ns_list = ["parser", "retriever", "synthesiser"]

for ns in subgraph_ns_list:
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ns}}
    # First visit: fetches checkpoint before replay_target_id
    result = state.get_checkpoint(ns, checkpointer, config)
    print(f"{ns}: first_visit → result={result}")

    # Simulated second visit in a loop — goes to get_tuple
    result2 = state.get_checkpoint(ns, checkpointer, config)
    print(f"{ns}: later_visit → result={result2}")
```

---

## 4 · `UIMessage` · `RemoveUIMessage` · `push_ui_message` · `delete_ui_message` · `ui_message_reducer`

**Module:** `langgraph.graph.ui`

LangGraph's UI primitives let graph nodes stream structured component-update events to a front-end. `push_ui_message` writes a `UIMessage` TypedDict both to the stream (so the client receives it in real time) and to the graph state (so it survives across checkpoints). `delete_ui_message` removes a component by ID. `ui_message_reducer` is the state-key reducer that applies both operations correctly.

**Key source facts** (from `langgraph/graph/ui.py`):

- `UIMessage` — TypedDict: `type="ui"`, `id: str`, `name: str`, `props: dict`, `metadata: dict`. When emitted by `push_ui_message`, `metadata` typically carries `run_id`, `tags`, and `name` from the LangChain callback config; `merge` and `message_id` are added when applicable. Direct dict construction (e.g. in examples) may supply a minimal or empty metadata.
- `RemoveUIMessage` — TypedDict: `type="remove-ui"`, `id: str`. Deleting a non-existent ID raises `ValueError`.
- `push_ui_message(name, props, *, id=None, metadata=None, message=None, state_key="ui", merge=False)` — generates a UUID when `id` is omitted; `merge=True` stores `{"merge": True}` in `metadata` so the reducer merges `props` with the existing entry instead of replacing it; calls `get_stream_writer()` and `CONFIG_KEY_SEND` to reach both the stream and state simultaneously.
- `delete_ui_message(id, *, state_key="ui")` — emits `RemoveUIMessage` to both stream and state.
- `ui_message_reducer(left, right)` — handles scalars and lists; applies `RemoveUIMessage` by building a `merged_by_id` index; respects `merge=True` in `metadata` for prop-level merges; raises on delete of unknown ID.
- `AnyUIMessage = UIMessage | RemoveUIMessage`.

### Example 1 — push a UI card, then update its props in place

```python
from langgraph.graph.ui import ui_message_reducer, UIMessage

# Initial state: one card
state: list[UIMessage] = [
    {"type": "ui", "id": "card-1", "name": "StatusCard",
     "props": {"title": "Processing…", "progress": 0},
     "metadata": {}}
]

# Update props via merge=True
update: UIMessage = {
    "type": "ui", "id": "card-1", "name": "StatusCard",
    "props": {"progress": 75},
    "metadata": {"merge": True},
}

merged = ui_message_reducer(state, update)
print(merged)
# [{'type': 'ui', 'id': 'card-1', 'name': 'StatusCard',
#   'props': {'title': 'Processing…', 'progress': 75}, 'metadata': {'merge': True}}]
```

### Example 2 — delete a UI component by ID

```python
from langgraph.graph.ui import ui_message_reducer, UIMessage, RemoveUIMessage

state: list[UIMessage] = [
    {"type": "ui", "id": "toast-1", "name": "Toast",
     "props": {"message": "Saved!"}, "metadata": {}},
    {"type": "ui", "id": "banner-1", "name": "Banner",
     "props": {"text": "Hello"}, "metadata": {}},
]

remove: RemoveUIMessage = {"type": "remove-ui", "id": "toast-1"}
result = ui_message_reducer(state, remove)

print(len(result))        # 1
print(result[0]["id"])    # 'banner-1'
```

### Example 3 — using `push_ui_message` inside a LangGraph node

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage, push_ui_message, ui_message_reducer

class AppState(TypedDict):
    query: str
    ui: Annotated[list[UIMessage], ui_message_reducer]

def search_node(state: AppState) -> dict:
    # Stream a loading spinner to the UI immediately
    push_ui_message(
        name="Spinner",
        props={"label": f"Searching for: {state['query']}"},
        id="search-spinner",
    )
    # … run the actual search …
    result_text = f"Found results for: {state['query']}"

    # Replace spinner with a result card: pass same id so the reducer
    # replaces the existing entry instead of appending a new one.
    push_ui_message(
        name="ResultCard",
        props={"text": result_text},
        id="search-spinner",
    )
    return {}

builder = StateGraph(AppState)
builder.add_node("search", search_node)
builder.add_edge(START, "search")
builder.add_edge("search", END)
# graph = builder.compile()
# result = graph.invoke({"query": "LangGraph UI streaming", "ui": []})
# result["ui"] contains ONE entry: the ResultCard replaced the Spinner
# (same id → reducer overwrites). Both were emitted on the stream.
```

---

## 5 · `_RemoteGraphRunStream` · `_AsyncRemoteGraphRunStream` · `_ChannelProjection` · `_ProjectionRegistry`

**Module:** `langgraph.pregel._remote_run_stream`

These four classes form the remote-graph streaming adapter layer — they map the `langgraph_sdk` wire protocol (`SyncThreadStream` / `AsyncThreadStream`) onto the same surface as a local `GraphRunStream` / `AsyncGraphRunStream`. This lets callers iterate `values`, `messages`, `updates`, `tasks`, `checkpoints`, `custom`, and named `extensions` channels using the same API regardless of whether the graph runs in-process or on a remote LangGraph server.

**Key source facts** (from `langgraph/pregel/_remote_run_stream.py`):

- `_translate_command_input(input)` — converts a local `Command(resume=...)` to its raw `resume` value before handing off to the v3 API. `Command.goto` / `Command.update` raise `NotImplementedError` because the v3 `run.start` path doesn't support them.
- `_ChannelProjection(sdk, channel)` — wraps an SDK stream subscription for channels the SDK doesn't type natively (`updates`, `checkpoints`, `tasks`, `custom`). Sync `__iter__` and async `__aiter__` both feed events through `DataDecoder(channel)` to yield `params.data` items — identical to what local transformers push.
- `_ProjectionRegistry` — a read-only `Mapping[str, Any]` backed by an SDK stream. Typed channels (`values`, `messages`, `tool_calls`, `subgraphs`) map to `getattr(sdk, name)`; decoded channels map to a `_ChannelProjection`; any other name resolves to `sdk.extensions[name]` (custom extension). Intentionally omits `lifecycle` and `debug`.
- `_RemoteGraphRunStream` — sync adapter: context-manager (`__enter__` calls `sdk_thread.__enter__()` then `run.start()`); exposes `.output`, `.interrupted`, `.interrupts`, `.values`, `.messages`, `.subgraphs`, `.tool_calls`, `.extensions`; `abort()` calls `client.runs.cancel(..., wait=False)` then `sdk.close()`; `interleave(*names)` delegates to `sdk.interleave_projections(names)`.
- `_AsyncRemoteGraphRunStream` — async twin. Key difference: `output` and `interrupted` are **async methods** (not properties) to match the local `AsyncGraphRunStream` surface. No `interleave()` because async callers compose with `asyncio.gather` / `asyncio.as_completed`.

### Example 1 — iterate the `updates` channel on a remote graph

```python
# Assumes a LangGraph server is running and langgraph_sdk is installed.
# This example shows the adapter surface without a live server.
from langgraph.pregel._remote_run_stream import (
    _ChannelProjection,
    _ProjectionRegistry,
    _translate_command_input,
)
from langgraph.types import Command

# Command(resume=...) translates to the raw resume value for the v3 API
cmd = Command(resume={"approved": True})
wire_input = _translate_command_input(cmd)
print(wire_input)   # {'approved': True}

# Command with goto raises NotImplementedError
try:
    _translate_command_input(Command(goto="some_node"))
except NotImplementedError as e:
    print(e)  # RemoteGraph v3 streaming supports Command(resume=...) only…
```

### Example 2 — `_ProjectionRegistry` channel routing logic

```python
from langgraph.pregel._remote_run_stream import _ProjectionRegistry

# Demonstrate the _TYPED vs _DECODED classification
print("Typed channels  :", _ProjectionRegistry._TYPED)
# ('values', 'messages', 'tool_calls', 'subgraphs')

print("Decoded channels:", _ProjectionRegistry._DECODED)
# ('updates', 'checkpoints', 'tasks', 'custom')

# The registry is a read-only Mapping; len() and iter() cover native channels
print("Native channels :", len(_ProjectionRegistry._NATIVE))
# 8  (4 typed + 4 decoded)
```

### Example 3 — sync remote run with Command(resume=...) and abort

```python
# Pattern for using _RemoteGraphRunStream with a real SDK client.
# Illustrates the context-manager and abort() lifecycle.

from langgraph.pregel._remote_run_stream import _RemoteGraphRunStream

def handle_remote_run(sync_client, thread_id: str, input_payload: dict) -> dict:
    """Drive a remote run, aborting on first interrupt."""
    sdk_thread = sync_client.runs.stream(
        thread_id, assistant_id="my-graph"
    )
    stream = _RemoteGraphRunStream(
        sync_client=sync_client,
        sdk_thread=sdk_thread,
        input=input_payload,
        config=None,
        metadata=None,
    )

    with stream as run:
        for snap in run.values:
            print("snapshot:", snap)
        if run.interrupted:
            print("Graph paused at interrupt — aborting")
            run.abort()
            return {}
        return run.output
```

---

## 6 · `UntrackedValue`

**Module:** `langgraph.channels.untracked_value`

`UntrackedValue` stores the last value written to a channel each superstep but **never checkpoints it**. `checkpoint()` always returns `MISSING`, and `from_checkpoint` always creates a fresh empty channel. This makes it ideal for per-step tool inputs, ephemeral scratch data, or secrets that must not persist in the checkpoint store.

**Key source facts** (from `langgraph/channels/untracked_value.py`):

- `__slots__ = ("value", "guard")` — `guard: bool` controls single-write-per-step enforcement.
- `update(values)` — raises `InvalidUpdateError` when `len(values) != 1` and `guard=True` (prevents silent last-write-wins in fan-out). Set `guard=False` when multiple writers may send to the same channel and you want the last one to win without error.
- `checkpoint() -> MISSING` — always; the saver skips this channel entirely.
- `from_checkpoint(checkpoint)` — ignores the checkpoint value and returns a fresh empty instance; `MISSING` is the effective reset every step.
- `is_available()` — returns `False` until `update` has been called at least once in the current superstep.
- Unlike `EphemeralValue` (which also auto-clears), `UntrackedValue` does **not** clear after the step — its value persists in memory until the next `update`. The key distinction is the checkpoint behaviour.

### Example 1 — ephemeral tool-input channel that never pollutes the checkpoint

```python
from typing_extensions import TypedDict, Annotated
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph import StateGraph, START, END

# UntrackedValue is registered via Annotated + channel class
class State(TypedDict):
    result: str
    # tool_input is never persisted in the checkpoint
    tool_input: Annotated[str, UntrackedValue(str)]

def tool_node(state: State) -> dict:
    # tool_input is available in-memory during this superstep
    raw_input = state.get("tool_input", "")
    return {"result": f"Processed: {raw_input}"}

# Without a checkpointer, channels function normally
builder = StateGraph(State)
builder.add_node("tool", tool_node)
builder.add_edge(START, "tool")
builder.add_edge("tool", END)
graph = builder.compile()

result = graph.invoke({"result": "", "tool_input": "search query"})
print(result["result"])   # 'Processed: search query'
```

### Example 2 — guard=False for multi-writer fan-out

```python
from langgraph.channels.untracked_value import UntrackedValue

# guard=True (default): one writer per step
strict_ch: UntrackedValue[int] = UntrackedValue(int, guard=True)
strict_ch.update([42])           # OK

try:
    strict_ch.update([1, 2])     # Raises: two values in same step
except Exception as e:
    print(f"guard=True error: {e}")

# guard=False: last value wins, no error
lax_ch: UntrackedValue[int] = UntrackedValue(int, guard=False)
lax_ch.update([1, 2, 3])        # Takes last: 3
print(lax_ch.get())             # 3
```

### Example 3 — confirm `checkpoint()` always returns MISSING

```python
from langgraph.channels.untracked_value import UntrackedValue
from langgraph._internal._typing import MISSING

ch: UntrackedValue[str] = UntrackedValue(str)
ch.update(["secret-api-key"])

# Value is readable in-memory
print(ch.get())          # 'secret-api-key'

# But checkpoint returns MISSING — the saver will skip this channel
ckpt = ch.checkpoint()
print(ckpt is MISSING)   # True

# from_checkpoint ignores the persisted value and resets to empty.
# Pregel always passes MISSING here (since checkpoint() returns MISSING).
restored = ch.from_checkpoint(MISSING)
print(restored.is_available())  # False — fresh empty state
```

---

## 7 · `HumanResponse` · Deprecated `HumanInterrupt` migration pattern

**Module:** `langgraph.prebuilt.interrupt`

`HumanResponse` is the TypedDict returned when an interrupted graph resumes after human review. The three *request* types (`HumanInterruptConfig`, `ActionRequest`, `HumanInterrupt`) were deprecated in LangGraph v1.0 and moved to `langchain.agents.interrupt`. Existing code still runs but emits `LangGraphDeprecatedSinceV10` warnings. `HumanResponse` itself remains stable.

**Key source facts** (from `langgraph/prebuilt/interrupt.py`):

- `HumanResponse` — TypedDict: `type: Literal["accept", "ignore", "response", "edit"]`; `args: None | str | ActionRequest`.
  - `"accept"` / `"ignore"` → `args=None`.
  - `"response"` → `args=str` (free-text feedback).
  - `"edit"` → `args=ActionRequest` with updated `action`/`args` fields.
- `HumanInterruptConfig`, `ActionRequest`, `HumanInterrupt` — all `@deprecated(category=LangGraphDeprecatedSinceV10)`. New import path: `from langchain.agents.interrupt import HumanInterruptConfig, ActionRequest, HumanInterrupt`.
- The HITL pattern: `interrupt([HumanInterrupt(...)])` pauses execution; the graph resumes with `Command(resume=[HumanResponse(...)])` from the caller.

### Example 1 — accept/ignore response patterns

```python
from langgraph.prebuilt.interrupt import HumanResponse

# Human approves without changes
accept_resp: HumanResponse = {"type": "accept", "args": None}

# Human skips the current step
ignore_resp: HumanResponse = {"type": "ignore", "args": None}

# Human provides feedback text
feedback_resp: HumanResponse = {
    "type": "response",
    "args": "Please use a more formal tone.",
}

for r in [accept_resp, ignore_resp, feedback_resp]:
    print(r["type"], "→", r["args"])
```

### Example 2 — full HITL loop using `interrupt()` and `Command(resume=...)`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.interrupt import HumanResponse

class ReviewState(TypedDict):
    draft: str
    approved: bool

def write_draft(state: ReviewState) -> ReviewState:
    return {"draft": "Draft email: Hello, please find attached…"}

def review_node(state: ReviewState) -> ReviewState:
    # Pause for human review
    response: list[HumanResponse] = interrupt([
        {
            "action_request": {"action": "approve_email", "args": {"draft": state["draft"]}},
            "description": "Approve this draft before sending?",
        }
    ])
    resp = response[0]
    if resp["type"] == "accept":
        return {"approved": True}
    elif resp["type"] == "edit":
        # Human edited the draft
        return {"draft": resp["args"]["args"].get("draft", state["draft"]), "approved": True}
    return {"approved": False}

graph = StateGraph(ReviewState)
graph.add_node("write", write_draft)
graph.add_node("review", review_node)
graph.add_edge(START, "write")
graph.add_edge("write", "review")
graph.add_edge("review", END)

checkpointer = InMemorySaver()
compiled = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "review-001"}}

# Step 1: run until interrupt
result = compiled.invoke({"draft": "", "approved": False}, config=config)
print("Paused:", compiled.get_state(config).next)

# Step 2: resume with human approval
final = compiled.invoke(
    Command(resume=[{"type": "accept", "args": None}]),
    config=config,
)
print("Approved:", final["approved"])
```

### Example 3 — migration from deprecated `HumanInterrupt` to `langchain.agents.interrupt`

```python
# BEFORE (deprecated — emits LangGraphDeprecatedSinceV10 warning):
# from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig, ActionRequest

# AFTER (correct import path from v1.0 onwards):
from langchain.agents.interrupt import HumanInterrupt, HumanInterruptConfig, ActionRequest

request = HumanInterrupt(
    action_request=ActionRequest(
        action="approve_deletion",
        args={"resource_id": "file-abc-123"},
    ),
    config=HumanInterruptConfig(
        allow_ignore=True,
        allow_respond=False,
        allow_edit=False,
        allow_accept=True,
    ),
    description="Permanently delete file-abc-123?",
)

from langgraph.types import interrupt
# response = interrupt([request])  # pauses graph; returns list[HumanResponse]
print("action:", request["action_request"]["action"])
# approve_deletion
```

---

## 8 · `ManagedValue` · `ManagedValueSpec` · `ManagedValueMapping` · `is_managed_value` · `IsLastStep` · `RemainingSteps`

**Modules:** `langgraph.managed.base` · `langgraph.managed.is_last_step`

`ManagedValue` is the base class for dependency-injection values that Pregel computes from the `PregelScratchpad` and injects into nodes at call time. Unlike channel values, managed values are **not** stored in the state dict — they are derived each superstep from execution metadata. `IsLastStep` and `RemainingSteps` are the two built-in implementations.

**Key source facts** (from `langgraph/managed/base.py` and `managed/is_last_step.py`):

- `ManagedValue` — `ABC`; single abstract `@staticmethod get(scratchpad: PregelScratchpad) -> V`. Must be a `@staticmethod` so Pregel can call `ManagedValueSpec.get(scratchpad)` without instantiation.
- `ManagedValueSpec = type[ManagedValue]` — the class itself (not an instance) is the spec.
- `is_managed_value(value)` — `TypeGuard[ManagedValueSpec]`: returns `True` when `value` is a class that is a subclass of `ManagedValue`. Used by Pregel to distinguish managed keys from channel keys in a state TypedDict.
- `ManagedValueMapping = dict[str, ManagedValueSpec]` — the mapping Pregel maintains internally: key → manager class.
- `PregelScratchpad` (from `langgraph._internal._scratchpad`) — dataclass providing: `step: int`, `stop: int`, `call_counter`, `interrupt_counter`, `get_null_resume`, `resume: list[Any]`, `subgraph_counter`.
- `IsLastStep = Annotated[bool, IsLastStepManager]` — `IsLastStepManager.get(s)` returns `s.step == s.stop - 1`. Annotate a node argument with `IsLastStep` to receive `True` on the final superstep.
- `RemainingSteps = Annotated[int, RemainingStepsManager]` — `RemainingStepsManager.get(s)` returns `s.stop - s.step`. Counts steps remaining including the current one.

### Example 1 — using `IsLastStep` and `RemainingSteps` in a node

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps

class State(TypedDict):
    value: int
    is_last_step: IsLastStep      # managed: injected as bool each superstep
    remaining_steps: RemainingSteps  # managed: injected as int each superstep

def guarded_node(state: State) -> dict:
    print(f"  step → is_last={state['is_last_step']}, remaining={state['remaining_steps']}")
    if state["is_last_step"]:
        print("  Final step — stopping recursion")
        return {"value": state["value"]}
    return {"value": state["value"] + 1}

builder = StateGraph(State)
builder.add_node("work", guarded_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()
result = graph.invoke({"value": 0}, config={"recursion_limit": 4})
print("final value:", result["value"])
```

### Example 2 — `is_managed_value` distinguishes managed keys from channel keys

```python
from langgraph.managed.base import is_managed_value, ManagedValue
from langgraph.managed.is_last_step import IsLastStepManager, RemainingStepsManager
from langgraph.channels.last_value import LastValue

# Managed value specs (the class itself)
print(is_managed_value(IsLastStepManager))    # True
print(is_managed_value(RemainingStepsManager)) # True

# Regular channel types are not managed values
print(is_managed_value(LastValue))            # False
print(is_managed_value(str))                  # False
print(is_managed_value(42))                   # False
```

### Example 3 — custom `ManagedValue`: inject call count into every node

```python
from typing_extensions import TypedDict, Annotated
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.managed.base import ManagedValue
from langgraph.graph import StateGraph, START, END

class CallCountManager(ManagedValue[int]):
    """Injects the number of node calls made so far in this superstep."""
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.call_counter()

# Annotated type alias for ergonomic use in node signatures
CallCount = Annotated[int, CallCountManager]

class State(TypedDict):
    log: list[str]
    call_count: CallCount  # managed: injected as int each superstep

def logging_node(state: State) -> State:
    entry = f"call_count={state['call_count']}"
    print(entry)
    return {"log": state["log"] + [entry]}

builder = StateGraph(State)
builder.add_node("log", logging_node)
builder.add_edge(START, "log")
builder.add_edge("log", END)

graph = builder.compile()
result = graph.invoke({"log": []})
print("log:", result["log"])
```

---

## 9 · `StreamToolCallHandler` · `ToolCallWriter` · `_tool_call_writer` ContextVar

**Module:** `langgraph.pregel._tools`

`StreamToolCallHandler` is a LangChain callback handler that fires on `on_tool_start` / `on_tool_end` / `on_tool_error` to push `tool-started`, `tool-output-delta`, `tool-finished`, and `tool-error` events onto the `tools` stream channel. It installs a per-call `ToolCallWriter` closure in the `_tool_call_writer` ContextVar, which `ToolRuntime.emit_output_delta()` reads to stream partial output without requiring explicit plumbing through tool signatures.

**Key source facts** (from `langgraph/pregel/_tools.py`):

- `ToolCallWriter = Callable[[Any], None]` — type alias for the per-call closure stored in `_tool_call_writer`.
- `_tool_call_writer: ContextVar[ToolCallWriter | None]` — module-level ContextVar; default `None`. Installed by `on_tool_start`, reset by `on_tool_end`/`on_tool_error`. Resetting uses a `Token` from `_tool_call_writer.set(writer)`, with `ValueError` swallowed if the reset context differs (async hand-off).
- `StreamToolCallHandler.__init__(stream, subgraphs, *, parent_ns=None)` — `stream` is `Callable[[StreamChunk], None]`; `run_inline = True` ensures callbacks run synchronously in the calling thread, preserving event order.
- `_ns_for_emit(metadata, tags)` — derives the emitting namespace from `langgraph_checkpoint_ns` (drops the trailing `node_name:task_id` segment). Returns `None` to silently suppress when `TAG_NOSTREAM` is present, `metadata` is missing, or `subgraphs=False` and the tool runs in a subgraph that doesn't match `parent_ns`.
- `_run_to_call: dict[UUID, (ns, tool_call_id, token)]` — correlates `run_id` across `on_tool_start` → `on_tool_end` / `on_tool_error`, because `on_tool_end` does not receive `tool_call_id` in kwargs.
- `tap_output_aiter` / `tap_output_iter` — required by `_StreamingCallbackHandler` protocol; both are pass-throughs.

### Example 1 — inspect which events `StreamToolCallHandler` emits

```python
from langgraph.pregel._tools import StreamToolCallHandler
from langgraph.pregel.protocol import StreamChunk
from uuid import uuid4

events: list[StreamChunk] = []

def collect(chunk: StreamChunk) -> None:
    events.append(chunk)

handler = StreamToolCallHandler(stream=collect, subgraphs=False)

tool_run_id = uuid4()
tool_call_id = "call-abc-001"

# Simulate on_tool_start
handler.on_tool_start(
    serialized={"name": "web_search"},
    input_str='{"query": "LangGraph"}',
    run_id=tool_run_id,
    metadata={"langgraph_checkpoint_ns": "agent:task-001"},
    inputs={"query": "LangGraph"},
    tool_call_id=tool_call_id,
)

print("After start:", events[-1])
# ((), 'tools', {'event': 'tool-started', 'tool_call_id': 'call-abc-001', ...})

# Simulate on_tool_end
handler.on_tool_end(
    output="LangGraph is a multi-actor framework.",
    run_id=tool_run_id,
)
print("After end:", events[-1])
# ((), 'tools', {'event': 'tool-finished', 'tool_call_id': 'call-abc-001', ...})
```

### Example 2 — `_tool_call_writer` ContextVar enables in-tool delta streaming

```python
from langgraph.pregel._tools import _tool_call_writer

# Outside a tool call the ContextVar is None
print(_tool_call_writer.get())   # None

# Inside a tool call StreamToolCallHandler installs a writer:
def fake_writer(delta):
    print(f"delta: {delta}")

token = _tool_call_writer.set(fake_writer)

# ToolRuntime.emit_output_delta() does exactly this:
writer = _tool_call_writer.get()
if writer is not None:
    writer({"text": "partial result …"})
# delta: {'text': 'partial result …'}

_tool_call_writer.reset(token)
print(_tool_call_writer.get())   # None  — reset complete
```

### Example 3 — attach handler to a graph with `stream_mode="tools"`

```python
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def echo(text: str) -> str:
    """Echo the input text."""
    return f"Echo: {text}"

tools = [echo]

class State(MessagesState):
    pass

def call_model(state: State):
    from langchain_core.messages import AIMessage, ToolCall
    tool_call = ToolCall(name="echo", args={"text": "hello"}, id="call-1")
    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

tool_node = ToolNode(tools)

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", "tools")
builder.add_edge("tools", END)

graph = builder.compile()

# stream_mode="tools" activates StreamToolCallHandler automatically
tool_events = []
for chunk in graph.stream(
    {"messages": [HumanMessage(content="say hello")]},
    stream_mode="tools",
):
    tool_events.append(chunk)
    print(chunk)
# Emits tool-started and tool-finished events for the "echo" call
```

---

## 10 · `AsyncQueue` · `SyncQueue` · `Semaphore`

**Module:** `langgraph._internal._queue`

These three classes are LangGraph's internal concurrency primitives for the Pregel task execution loop. `AsyncQueue` adds a non-consuming `wait()` to `asyncio.Queue` for peek semantics. `SyncQueue` is a pure-Python unbounded FIFO with optional blocking `get()` and a non-consuming `wait()`. `Semaphore` adds the same `wait()` semantics to `threading.Semaphore` to avoid acquiring the semaphore just to check availability.

**Key source facts** (from `langgraph/_internal/_queue.py`):

- `AsyncQueue(asyncio.Queue)` — `wait()` mirrors the internal `asyncio.Queue.get()` except the item is **not consumed**. Multiple tasks can `await wait()` concurrently; the item remains in the queue for the consumer that calls `get_nowait()`.
- `Semaphore(threading.Semaphore)` — `wait(blocking, timeout)` acquires the underlying condition lock and waits until `_value > 0`, then exits **without decrementing**. Used to signal that a slot is available without claiming it.
- `SyncQueue` — pure Python (no `queue.SimpleQueue` subclass): backed by a `deque` + a `Semaphore(0)`. `put()` appends and calls `_count.release()`; `get(block, timeout)` acquires the semaphore (with optional timeout → `queue.Empty`) then `popleft()`; `wait(block, timeout)` delegates to `Semaphore.wait()` for non-consuming peek; `empty()` and `qsize()` are best-effort (not thread-safe between calls).
- `__class_getitem__` on `SyncQueue` uses `types.GenericAlias` for `SyncQueue[int]` type-hint syntax support.

### Example 1 — `AsyncQueue.wait()` peek without consuming

```python
import asyncio
from langgraph._internal._queue import AsyncQueue

async def demo():
    q: AsyncQueue[int] = AsyncQueue()
    await q.put(42)

    # wait() returns when an item is available but does NOT consume it
    await q.wait()
    print("item visible, size:", q.qsize())  # 1

    # get_nowait() consumes the item
    item = q.get_nowait()
    print("item:", item)                     # 42
    print("size after get:", q.qsize())      # 0

asyncio.run(demo())
```

### Example 2 — `SyncQueue` producer/consumer in separate threads

```python
import threading
import time
from langgraph._internal._queue import SyncQueue

q: SyncQueue[str] = SyncQueue()
results: list[str] = []

def producer():
    for i in range(3):
        time.sleep(0.01)
        q.put(f"task-{i}")

def consumer():
    for _ in range(3):
        q.wait()            # peek: block until something is available
        item = q.get()      # consume
        results.append(item)

t_prod = threading.Thread(target=producer)
t_cons = threading.Thread(target=consumer)
t_cons.start()
t_prod.start()
t_prod.join()
t_cons.join()

print(results)  # ['task-0', 'task-1', 'task-2']
```

### Example 3 — `SyncQueue` with timeout and `queue.Empty`

```python
import queue
from langgraph._internal._queue import SyncQueue

q: SyncQueue[int] = SyncQueue()

# Non-blocking get on empty queue → Empty
try:
    q.get(block=False)
except queue.Empty:
    print("Empty (non-blocking)")

# Blocking get with short timeout → Empty
try:
    q.get(block=True, timeout=0.01)
except queue.Empty:
    print("Timed out (blocking, 10ms)")

# Put then get
q.put(99)
print("got:", q.get())  # 99

# Generic alias syntax
q2: SyncQueue[str] = SyncQueue()
print(type(q2))  # <class 'langgraph._internal._queue.SyncQueue'>
```

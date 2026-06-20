---
title: "Class deep-dives Vol. 21 — channels, serialization & graph protocols (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented or under-documented classes in LangGraph 1.2.6: BaseChannel lifecycle contract (custom channel creation), BinaryOperatorAggregate (the channel behind Annotated reducers + Overwrite sentinel), UntrackedValue (in-memory-only channels with guard semantics), register_serde_event_listener+SerdeEvent (serde observability hooks), SerializerProtocol+CipherProtocol+SerializerCompat (custom serializer contracts), AsyncBatchedBaseStore (background-queue coalescing + deadlock prevention), UIMessage+RemoveUIMessage+ui_message_reducer (full UI streaming protocol), ManagedValue+IsLastStepManager+RemainingStepsManager (custom managed values from PregelScratchpad), get_field_default+get_cached_annotated_keys (TypedDict/Pydantic schema internals), and PregelProtocol+StreamProtocol (the graph executor interface). All signatures and behaviours verified from installed package source at /usr/local/lib/python3.11/dist-packages/langgraph/."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 21"
  order: 52
---

# Class deep-dives Vol. 21 — channels, serialization & graph protocols (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not from documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `BaseChannel` — channel lifecycle contract | `langgraph.channels.base` |
| 2 | `BinaryOperatorAggregate` — reducer channels & `Overwrite` | `langgraph.channels.binop` |
| 3 | `UntrackedValue` — in-memory-only channels | `langgraph.channels.untracked_value` |
| 4 | `register_serde_event_listener` + `SerdeEvent` | `langgraph.checkpoint.serde.event_hooks` |
| 5 | `SerializerProtocol` + `CipherProtocol` + `SerializerCompat` | `langgraph.checkpoint.serde.base` |
| 6 | `AsyncBatchedBaseStore` — background-queue coalescing | `langgraph.store.base.batch` |
| 7 | `UIMessage` + `RemoveUIMessage` + `ui_message_reducer` | `langgraph.graph.ui` |
| 8 | `ManagedValue` + `IsLastStepManager` + `RemainingStepsManager` | `langgraph.managed.base` · `langgraph.managed.is_last_step` |
| 9 | `get_field_default` + `get_cached_annotated_keys` + `get_enhanced_type_hints` | `langgraph._internal._fields` |
| 10 | `PregelProtocol` + `StreamProtocol` | `langgraph.pregel.protocol` |

---

## 1 · `BaseChannel` — channel lifecycle contract

**Module:** `langgraph.channels.base`

`BaseChannel` is the abstract base for every channel in LangGraph — `LastValue`, `BinaryOperatorAggregate`, `Topic`, `EphemeralValue`, `NamedBarrierValue`, `UntrackedValue`, and `DeltaChannel` all inherit from it. Understanding the contract lets you build custom channels that integrate seamlessly with Pregel's execution engine.

### Source signature

```python
class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None: ...

    # abstract — must override
    @property @abstractmethod
    def ValueType(self) -> Any: ...       # The type stored in the channel

    @property @abstractmethod
    def UpdateType(self) -> Any: ...      # The type received per super-step

    @abstractmethod
    def from_checkpoint(self, checkpoint: Checkpoint | Any) -> Self: ...
    @abstractmethod
    def get(self) -> Value: ...           # Raises EmptyChannelError when empty
    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool: ...  # Return True if changed

    # concrete with sensible defaults — can override
    def checkpoint(self) -> Checkpoint | Any: ...  # Default: returns self.get()
    def copy(self) -> Self: ...           # Default: checkpoint → from_checkpoint
    def consume(self) -> bool: ...        # Default: returns False (persistent)
    def is_available(self) -> bool: ...   # Default: try/except get()
```

### Key behaviours

| Method | Guarantee |
|--------|-----------|
| `update(values)` | Called every super-step with all pending writes; empty list means no writes |
| `update(values)` return | `True` triggers downstream nodes; `False` means no change |
| `get()` | Raises `EmptyChannelError` when the channel has never been updated |
| `checkpoint()` | Default calls `get()` — return `MISSING` to skip storing a blob |
| `from_checkpoint(c)` | Must return a **new** channel instance; never mutate `self` |
| `consume()` | Return `True` to self-destruct after one read (single-use channels) |
| `is_available()` | Default try/catches `EmptyChannelError` — override for efficiency |

### Example 1 — custom counter channel

```python
from __future__ import annotations
from collections.abc import Sequence
from typing_extensions import Self
from langgraph.channels.base import BaseChannel
from langgraph.errors import EmptyChannelError

class CounterChannel(BaseChannel[int, int, int]):
    """Counts the number of times any value was written this thread."""

    __slots__ = ("_count",)

    def __init__(self) -> None:
        super().__init__(int)
        self._count = 0

    @property
    def ValueType(self) -> type[int]:
        return int

    @property
    def UpdateType(self) -> type[int]:
        return int

    def from_checkpoint(self, checkpoint: int) -> Self:
        ch = self.__class__()
        ch.key = self.key
        ch._count = checkpoint if checkpoint is not None else 0
        return ch

    def update(self, values: Sequence[int]) -> bool:
        if not values:
            return False
        self._count += len(values)
        return True

    def get(self) -> int:
        return self._count

    def is_available(self) -> bool:
        return True  # always readable

    def checkpoint(self) -> int:
        return self._count


# Wire the custom channel via Annotated
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str
    call_count: Annotated[int, CounterChannel()]

def node_a(state: State) -> dict:
    return {"message": "hello", "call_count": 1}

def node_b(state: State) -> dict:
    return {"message": "world", "call_count": 1}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile()

result = graph.invoke({"message": "start", "call_count": 0})
print(result["call_count"])  # 2 — both nodes incremented
```

### Example 2 — single-use trigger channel with `consume()`

```python
from __future__ import annotations
from collections.abc import Sequence
from typing_extensions import Self
from langgraph.channels.base import BaseChannel
from langgraph.errors import EmptyChannelError
from langgraph._internal._typing import MISSING

class TriggerChannel(BaseChannel[str, str, str]):
    """Stores one string; consumed (cleared) after it is read once."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        super().__init__(str)
        self._value = MISSING

    @property
    def ValueType(self) -> type[str]:
        return str

    @property
    def UpdateType(self) -> type[str]:
        return str

    def from_checkpoint(self, checkpoint: str) -> Self:
        ch = self.__class__()
        ch.key = self.key
        if checkpoint is not MISSING:
            ch._value = checkpoint
        return ch

    def update(self, values: Sequence[str]) -> bool:
        if not values:
            return False
        self._value = values[-1]
        return True

    def get(self) -> str:
        if self._value is MISSING:
            raise EmptyChannelError()
        return self._value

    def is_available(self) -> bool:
        return self._value is not MISSING

    def consume(self) -> bool:
        """Return True → Pregel clears the channel after reading."""
        if self._value is not MISSING:
            self._value = MISSING
            return True
        return False

    def checkpoint(self) -> str:
        return self._value
```

### Example 3 — inspecting channel internals at runtime

```python
from langgraph.channels.last_value import LastValue
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.topic import Topic
import operator

# Introspect any channel object
channels = {
    "name": LastValue(str),
    "total": BinaryOperatorAggregate(int, operator.add),
    "log": Topic(str, accumulate=True),
}

for key, ch in channels.items():
    ch.key = key
    print(f"{key}: ValueType={ch.ValueType}, UpdateType={ch.UpdateType}")

# Checkpoint round-trip
ch = BinaryOperatorAggregate(int, operator.add)
ch.key = "counter"
ch.update([1, 2, 3])
blob = ch.checkpoint()
print(f"blob={blob}")  # 6

restored = ch.from_checkpoint(blob)
restored.update([10])
print(f"restored.get()={restored.get()}")  # 16
```

---

## 2 · `BinaryOperatorAggregate` — reducer channels and `Overwrite`

**Module:** `langgraph.channels.binop`

`BinaryOperatorAggregate` is the channel created when you annotate a state field with a two-argument function: `Annotated[int, operator.add]`, `Annotated[list, operator.add]`, `Annotated[list, lambda a, b: a + b]`. It stores state by folding each batch of writes through the `operator` function.

### Source signature

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]): ...
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| Initial value | `typ()` — zero-initialised; falls back to `MISSING` if `typ()` raises |
| `update()` first write | Sets `self.value = values[0]`, then folds remaining through `operator` |
| `Overwrite` detection | Checks `isinstance(v, Overwrite)` **AND** `isinstance(v, dict) and OVERWRITE in v` |
| Multiple `Overwrite` in one batch | Raises `InvalidUpdateError(ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE)` |
| `Overwrite(None)` | Resets to `typ()` — same as `Overwrite(value=None)` |
| `_operators_equal` | Lambda names are all `"<lambda>"` — any lambda pair is treated as equal |
| ABC type normalisation | `Sequence/MutableSequence → list`, `Set/MutableSet → set`, `Mapping/MutableMapping → dict` |

### Example 1 — basic accumulation and `Overwrite` reset

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class State(TypedDict):
    total: Annotated[int, operator.add]
    tags: Annotated[list[str], operator.add]

def accumulate(state: State) -> dict:
    return {"total": 10, "tags": ["a", "b"]}

def reset_total(state: State) -> dict:
    # Overwrite bypasses the reducer — sets total directly to 0
    return {"total": Overwrite(0), "tags": ["c"]}

builder = StateGraph(State)
builder.add_node("accumulate", accumulate)
builder.add_node("reset", reset_total)
builder.add_edge(START, "accumulate")
builder.add_edge("accumulate", "reset")
builder.add_edge("reset", END)
graph = builder.compile()

result = graph.invoke({"total": 5, "tags": []})
print(result["total"])   # 0  — Overwrite replaced the accumulated value
print(result["tags"])    # ['a', 'b', 'c']  — normal accumulation
```

### Example 2 — understanding `_operators_equal` and the lambda gotcha

```python
import operator
from langgraph.channels.binop import BinaryOperatorAggregate, _operators_equal

# Named functions: strict identity check
ch1 = BinaryOperatorAggregate(int, operator.add)
ch2 = BinaryOperatorAggregate(int, operator.add)
print(ch1 == ch2)  # True — same object (operator.add is a singleton)

ch3 = BinaryOperatorAggregate(int, operator.mul)
print(ch1 == ch3)  # False — different operators

# Lambdas: all have name '<lambda>' so any pair is treated as EQUAL
add_lambda  = lambda a, b: a + b  # noqa: E731
mul_lambda  = lambda a, b: a * b  # noqa: E731
print(_operators_equal(add_lambda, mul_lambda))  # True — BOTH are lambdas

ch4 = BinaryOperatorAggregate(int, add_lambda)
ch5 = BinaryOperatorAggregate(int, mul_lambda)
print(ch4 == ch5)  # True — equality is based on lambda safety heuristic

# Implication: if you recompile a graph and the channel annotation uses
# a lambda, LangGraph treats it as the same channel regardless of content.
# Use named functions for clarity and deterministic equality checks.
```

### Example 3 — custom set-union reducer with `Overwrite` reset

```python
import operator
from typing import Annotated, FrozenSet
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite
from langgraph.checkpoint.memory import InMemorySaver

def set_union(a: frozenset, b: frozenset) -> frozenset:
    return a | b

class State(TypedDict):
    seen_ids: Annotated[frozenset, set_union]

def collector(state: State) -> dict:
    return {"seen_ids": frozenset(["id-1", "id-2"])}

def extra(state: State) -> dict:
    return {"seen_ids": frozenset(["id-3"])}

def reset(state: State) -> dict:
    # Clear all seen IDs for a new run cycle
    return {"seen_ids": Overwrite(frozenset())}

builder = StateGraph(State)
builder.add_node("collect", collector)
builder.add_node("extra", extra)
builder.add_node("reset", reset)
builder.add_edge(START, "collect")
builder.add_edge("collect", "extra")
builder.add_edge("extra", "reset")
builder.add_edge("reset", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "1"}}

result = graph.invoke({"seen_ids": frozenset()}, config=config)
print(result["seen_ids"])  # frozenset() — reset cleared everything
```

---

## 3 · `UntrackedValue` — in-memory-only channels

**Module:** `langgraph.channels.untracked_value`

`UntrackedValue` stores the last value received — but **never writes a checkpoint blob**. Its `checkpoint()` method always returns `MISSING`, so the Pregel loop omits the channel from `channel_values` entirely. When a thread is resumed from a checkpoint, `from_checkpoint` always returns a fresh empty channel. The value is local to the current invocation.

### Source signature

```python
class UntrackedValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "guard")

    guard: bool
    value: Value | Any

    def __init__(self, typ: type[Value], guard: bool = True) -> None: ...

    def checkpoint(self) -> Value | Any:
        return MISSING   # Never stored

    def from_checkpoint(self, checkpoint: Value) -> Self:
        # Always ignores checkpoint — returns empty channel
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        return empty
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `guard=True` (default) | Raises `InvalidUpdateError` if more than one write arrives in a super-step |
| `guard=False` | Silently takes `values[-1]` — last writer wins, no error |
| `checkpoint()` | Always `MISSING` — no blob stored, no persistence overhead |
| `from_checkpoint` | Ignores the checkpoint argument; always returns empty channel |
| `__eq__` | Two `UntrackedValue` channels are equal if `guard` matches — operator doesn't factor in |
| Use case | Volatile per-invocation scratchpad; computed caches; non-persistent counters |

### Example 1 — scratchpad that vanishes on resume

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    message: str
    # This field exists only in memory — never survives a checkpoint
    temp_scratch: Annotated[str, UntrackedValue(str)]

def step1(state: State) -> dict:
    return {"message": "persisted", "temp_scratch": "volatile data"}

def step2(state: State) -> dict:
    print(f"scratch in step2: {state['temp_scratch']!r}")
    return {}

builder = StateGraph(State)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "scratch-demo"}}

# First run
graph.invoke({"message": "start", "temp_scratch": ""}, config=config)
# step2 sees: "volatile data"

# Get state from checkpoint — temp_scratch is absent
saved = graph.get_state(config)
print(saved.values.get("temp_scratch"))  # None — not in checkpoint
```

### Example 2 — `guard=False` for last-writer-wins in parallel branches

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Send

class State(TypedDict):
    items: Annotated[list[str], operator.add]
    # One of the parallel branches gets to set this; no ordering guarantee
    winner: Annotated[str, UntrackedValue(str, guard=False)]

def branch(state: State, item: str) -> dict:
    return {"items": [item], "winner": item}

def fan_out(state: State):
    return [Send("branch", "alpha"), Send("branch", "beta")]

builder = StateGraph(State)
builder.add_node("branch", branch)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("branch", END)
graph = builder.compile()

result = graph.invoke({"items": [], "winner": ""})
print(result["items"])   # ['alpha', 'beta'] or ['beta', 'alpha']
print(result["winner"])  # either 'alpha' or 'beta' — last writer wins
```

### Example 3 — using `UntrackedValue` for a per-run token counter

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.untracked_value import UntrackedValue

class TokenCounter:
    """Accumulates token counts but never persists them."""
    def __init__(self):
        self.input = 0
        self.output = 0

    def __repr__(self):
        return f"TokenCounter(in={self.input}, out={self.output})"

def _counter_channel():
    return UntrackedValue(TokenCounter)

class State(TypedDict):
    response: str
    # Tracks tokens this invocation only — cleared on resume
    token_usage: Annotated[TokenCounter | None, _counter_channel()]

def llm_call(state: State) -> dict:
    counter = TokenCounter()
    counter.input = 128
    counter.output = 256
    return {"response": "The answer is 42", "token_usage": counter}

def log_tokens(state: State) -> dict:
    usage = state.get("token_usage")
    if usage:
        print(f"This invocation used: {usage}")
    return {}

builder = StateGraph(State)
builder.add_node("llm_call", llm_call)
builder.add_node("log_tokens", log_tokens)
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", "log_tokens")
builder.add_edge("log_tokens", END)
graph = builder.compile()
graph.invoke({"response": "", "token_usage": None})
# logs: "This invocation used: TokenCounter(in=128, out=256)"
```

---

## 4 · `register_serde_event_listener` + `SerdeEvent`

**Module:** `langgraph.checkpoint.serde.event_hooks`

**First dedicated coverage.** This module provides observability hooks into the checkpoint serialization subsystem. When `JsonPlusSerializer` encounters a type that isn't in the msgpack allowlist — or that is explicitly blocked — it calls `emit_serde_event`. You can subscribe with `register_serde_event_listener` to receive those events, which is useful for audit logging, building custom allowlists, or security monitoring.

### Source signature

```python
class SerdeEvent(TypedDict):
    kind: str           # e.g. "unregistered", "blocked"
    module: str         # e.g. "my_module"
    name: str           # e.g. "MyClass"
    method: NotRequired[str]  # e.g. "dumps" or "loads"

SerdeEventListener = Callable[[SerdeEvent], None]

def register_serde_event_listener(
    listener: SerdeEventListener
) -> Callable[[], None]:
    """Returns an unregister callable."""

def emit_serde_event(event: SerdeEvent) -> None:
    """Emit to all registered listeners; failures are isolated + logged."""
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `_listeners` | Plain `list[SerdeEventListener]` protected by `_listeners_lock = Lock()` |
| Thread safety | All mutations and iteration of `_listeners` hold `_listeners_lock` |
| Listener failures | Caught per-listener; logged via `logger.warning`; other listeners still run |
| Unregister | Returned callable removes the listener; swallows `ValueError` if already removed |
| Snapshot | `emit_serde_event` snapshots `_listeners` under lock, then releases before calling |
| `_MAX_WARNED_TYPES` | Cap of 1000 unique `(module, name)` pairs; beyond that, warnings are silently dropped |

### Example 1 — audit log for unknown checkpoint types

```python
from langgraph.checkpoint.serde.event_hooks import (
    register_serde_event_listener,
    SerdeEvent,
)

audit_log: list[SerdeEvent] = []

def my_listener(event: SerdeEvent) -> None:
    audit_log.append(event)
    print(f"[SERDE] kind={event['kind']} "
          f"type={event['module']}.{event['name']} "
          f"method={event.get('method', 'N/A')}")

# Register and capture the unregister callable
unregister = register_serde_event_listener(my_listener)

# Now any serialization that touches an unregistered / blocked type
# will fire my_listener. When done:
unregister()  # Remove the listener cleanly
```

### Example 2 — allowlist builder: discover unknown types in your checkpoint store

```python
from collections import defaultdict
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener, SerdeEvent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class MyData:
    """A custom class that JsonPlusSerializer doesn't know about by default."""
    def __init__(self, v: int):
        self.v = v

type_report: dict[str, list[str]] = defaultdict(list)

def collector(event: SerdeEvent) -> None:
    key = f"{event['module']}.{event['name']}"
    type_report[event["kind"]].append(key)

unregister = register_serde_event_listener(collector)

# Trigger serialization of a MyData object
serde = JsonPlusSerializer()
try:
    serde.dumps_typed(MyData(42))
except Exception:
    pass  # May fail if type is blocked

unregister()

print("Unregistered types encountered:")
for t in type_report.get("unregistered", []):
    print(f"  {t}")
```

### Example 3 — per-thread listener with context isolation

```python
import threading
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener, SerdeEvent

# Track events per-thread in a thread-local
_local = threading.local()

def thread_listener(event: SerdeEvent) -> None:
    if not hasattr(_local, "events"):
        _local.events = []
    _local.events.append(event)

unregister = register_serde_event_listener(thread_listener)

def worker():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    serde = JsonPlusSerializer()
    try:
        serde.dumps_typed(object())
    except Exception:
        pass
    events = getattr(_local, "events", [])
    print(f"Thread {threading.current_thread().name} saw {len(events)} serde event(s)")

threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

unregister()
```

---

## 5 · `SerializerProtocol` + `CipherProtocol` + `SerializerCompat`

**Module:** `langgraph.checkpoint.serde.base`

**First dedicated coverage.** These are the structural protocol contracts that govern how LangGraph stores and retrieves checkpoint blobs. Any class that implements `dumps_typed` / `loads_typed` satisfies `SerializerProtocol` — no inheritance required. `CipherProtocol` pairs with `EncryptedSerializer` (from `langgraph.checkpoint.serde.encrypted`) for transparent encryption.

### Source signatures

```python
@runtime_checkable
class SerializerProtocol(Protocol):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...
    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...

class UntypedSerializerProtocol(Protocol):
    def dumps(self, obj: Any) -> bytes: ...
    def loads(self, data: bytes) -> Any: ...

class SerializerCompat(SerializerProtocol):
    """Wraps an untyped serializer; uses type(obj).__name__ as the type tag."""
    def __init__(self, serde: UntypedSerializerProtocol) -> None: ...
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return type(obj).__name__, self.serde.dumps(obj)
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.serde.loads(data[1])  # ignores the type tag on load

class CipherProtocol(Protocol):
    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]: ...
    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes: ...

def maybe_add_typed_methods(
    serde: SerializerProtocol | UntypedSerializerProtocol,
) -> SerializerProtocol:
    """Wrap old-style serde in SerializerCompat for backwards compat."""
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `SerializerProtocol` | `@runtime_checkable` — `isinstance(x, SerializerProtocol)` works at runtime |
| `dumps_typed` return | `(type_str, bytes)` — `type_str` stored in `channel_versions` as the blob type discriminator |
| `SerializerCompat.loads_typed` | Ignores the type tag; passes raw `data[1]` to `serde.loads` — type-blind restore |
| `maybe_add_typed_methods` | No-op if already `SerializerProtocol`; wraps otherwise |
| `CipherProtocol.encrypt` return | `(cipher_name, ciphertext)` — cipher name is prepended to the type tag with `+` in `EncryptedSerializer` |

### Example 1 — custom JSON serializer that satisfies `SerializerProtocol`

```python
import json
from typing import Any
from langgraph.checkpoint.serde.base import SerializerProtocol

class JsonSerializer:
    """A simple JSON-only checkpoint serializer."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        type_name = type(obj).__name__
        data = json.dumps(obj, default=str).encode()
        return type_name, data

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        _, raw = data
        return json.loads(raw)

# Verify it satisfies the protocol at runtime
serde = JsonSerializer()
assert isinstance(serde, SerializerProtocol), "must satisfy protocol"

# Use it with any BaseCheckpointSaver that accepts a custom serde
blob_type, blob = serde.dumps_typed({"key": "value", "count": 42})
print(f"type={blob_type!r}, bytes={blob}")

restored = serde.loads_typed((blob_type, blob))
print(restored)  # {'key': 'value', 'count': 42}
```

### Example 2 — `SerializerCompat` to wrap an old-style `dumps/loads` serde

```python
import pickle
from langgraph.checkpoint.serde.base import SerializerCompat, maybe_add_typed_methods

# Old-style serde with only dumps/loads
class PickleSerde:
    def dumps(self, obj) -> bytes:
        return pickle.dumps(obj)
    def loads(self, data: bytes):
        return pickle.loads(data)  # noqa: S301

old_serde = PickleSerde()

# Upgrade via maybe_add_typed_methods
compat = maybe_add_typed_methods(old_serde)
print(type(compat))  # <class 'langgraph.checkpoint.serde.base.SerializerCompat'>

# dumps_typed uses type(obj).__name__ as the type tag
type_tag, data = compat.dumps_typed({"x": 1})
print(f"type_tag={type_tag!r}")   # 'dict'

restored = compat.loads_typed((type_tag, data))
print(restored)   # {'x': 1}
```

### Example 3 — bring-your-own cipher with `CipherProtocol` + `EncryptedSerializer`

```python
import os
import base64
from typing import Any
from langgraph.checkpoint.serde.base import CipherProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

class XorCipher(CipherProtocol):
    """Toy XOR cipher — DO NOT use in production."""

    def __init__(self, key: int = 0x42):
        self.key = key

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        ciphertext = bytes(b ^ self.key for b in plaintext)
        return "xor", ciphertext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "xor"
        return bytes(b ^ self.key for b in ciphertext)

# Compose with EncryptedSerializer
cipher = XorCipher(key=0x5A)
serde = EncryptedSerializer(cipher=cipher)

obj = {"secret": "my_api_key_abc123", "count": 7}
type_tag, blob = serde.dumps_typed(obj)
print(f"type_tag={type_tag!r}")   # e.g. 'msgpack+xor'
print(f"blob is encrypted bytes: {blob[:8].hex()}")

restored = serde.loads_typed((type_tag, blob))
print(restored)  # {'secret': 'my_api_key_abc123', 'count': 7}
```

---

## 6 · `AsyncBatchedBaseStore` — background-queue coalescing

**Module:** `langgraph.store.base.batch`

`AsyncBatchedBaseStore` is the foundation for any async store implementation that wants to batch operations for efficiency. It maintains an `asyncio.Queue` and a background task that drains it, coalescing multiple concurrent `aget`/`aput`/`asearch` calls into fewer round-trips to the underlying backend (e.g., a Postgres table or Redis cluster).

### Source signature

```python
class AsyncBatchedBaseStore(BaseStore):
    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self._aqueue: asyncio.Queue[tuple[asyncio.Future, Op]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ensure_task()

    def _ensure_task(self) -> None:
        """Recreate the background drainer if it was cancelled or errored."""
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(
                _run(self._aqueue, weakref.ref(self))
            )

    def __del__(self) -> None:
        if self._task:
            self._task.cancel()
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `_check_loop` decorator | Raises `asyncio.InvalidStateError` if the calling loop is the same as `store._loop` |
| Deadlock prevention | Sync methods (`get`, `put`, etc.) must NOT be called from the same event loop — they block waiting for the background task, which lives IN that loop |
| `weakref.ref(self)` | Background drainer holds a weak reference to the store; when the store is GC'd, the drainer exits cleanly |
| `_ensure_task()` | Called at construction AND before every `aget*` to recreate if cancelled |
| `__del__` | Cancels the background task — no background work survives the store's lifetime |
| `aqueue` item | `(asyncio.Future, Op)` — future is resolved by the drainer when the batch completes |

### Example 1 — subclassing `AsyncBatchedBaseStore` for a batched backend

```python
import asyncio
from typing import Any, Iterator
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.base import (
    BaseStore, GetOp, PutOp, SearchOp, ListNamespacesOp,
    Item, SearchItem, Op, Result,
)

class DictStore(AsyncBatchedBaseStore):
    """In-memory store that uses AsyncBatchedBaseStore's batching machinery."""

    def __init__(self) -> None:
        super().__init__()
        self._data: dict[tuple, dict[str, Item]] = {}

    def batch(self, ops: Iterator[Op]) -> list[Result]:
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                ns_data = self._data.get(op.namespace, {})
                results.append(ns_data.get(op.key))
            elif isinstance(op, PutOp):
                if op.value is None:
                    self._data.get(op.namespace, {}).pop(op.key, None)
                else:
                    ns = self._data.setdefault(op.namespace, {})
                    ns[op.key] = Item(
                        value=op.value,
                        key=op.key,
                        namespace=op.namespace,
                        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                        updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                    )
                results.append(None)
            elif isinstance(op, SearchOp):
                ns_data = self._data.get(op.namespace_prefix, {})
                results.append(list(ns_data.values()))
            elif isinstance(op, ListNamespacesOp):
                results.append(list(self._data.keys()))
        return results

    async def abatch(self, ops: Iterator[Op]) -> list[Result]:
        return self.batch(ops)


async def main():
    store = DictStore()
    await store.aput(("users",), "alice", {"role": "admin"})
    item = await store.aget(("users",), "alice")
    print(item.value)  # {'role': 'admin'}

asyncio.run(main())
```

### Example 2 — `_check_loop` deadlock trap and how to avoid it

```python
import asyncio
from langgraph.store.memory import InMemoryStore

async def safe_pattern():
    """Always use async methods inside the event loop."""
    store = InMemoryStore()
    await store.aput(("ns",), "key1", {"data": 1})

    # CORRECT: async method in async context
    item = await store.aget(("ns",), "key1")
    print(item.value)

    # WRONG (would deadlock or raise InvalidStateError):
    # item = store.get(("ns",), "key1")  # sync call from same event loop

asyncio.run(safe_pattern())
```

### Example 3 — `_ensure_task` and task restart on cancellation

```python
import asyncio
from langgraph.store.memory import InMemoryStore

async def demonstrate_task_restart():
    store = InMemoryStore()

    # The background task starts automatically on __init__
    print(f"task running: {not store._task.done()}")  # True

    # Cancel the background drainer
    store._task.cancel()
    await asyncio.sleep(0)  # yield to let cancellation propagate
    print(f"task cancelled: {store._task.done()}")  # True

    # _ensure_task is called before any aget — restarts the drainer
    await store.aput(("ns",), "k", {"v": 1})
    print(f"task restarted: {not store._task.done()}")  # True

    item = await store.aget(("ns",), "k")
    print(item.value)  # {'v': 1}

asyncio.run(demonstrate_task_restart())
```

---

## 7 · `UIMessage` + `RemoveUIMessage` + `ui_message_reducer`

**Module:** `langgraph.graph.ui`

LangGraph's UI streaming protocol lets nodes push structured component events to the frontend in real-time. `push_ui_message` emits a `UIMessage` to the stream and writes it into a designated state key. `delete_ui_message` sends a `RemoveUIMessage` tombstone. `ui_message_reducer` is the reducer function that merges these into a list with correct deduplication, merge, and deletion semantics.

### Source signatures

```python
class UIMessage(TypedDict):
    type: Literal["ui"]
    id: str
    name: str
    props: dict[str, Any]
    metadata: dict[str, Any]  # includes: merge, run_id, tags, name, message_id

class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]
    id: str

AnyUIMessage = UIMessage | RemoveUIMessage

def push_ui_message(
    name: str, props: dict[str, Any], *,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    message: AnyMessage | None = None,
    state_key: str | None = "ui",
    merge: bool = False,
) -> UIMessage: ...

def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage: ...

def ui_message_reducer(
    left: list[AnyUIMessage] | AnyUIMessage,
    right: list[AnyUIMessage] | AnyUIMessage,
) -> list[AnyUIMessage]: ...
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `push_ui_message` side effects | Calls `get_stream_writer()(evt)` AND `config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])` |
| `state_key=None` | Stream-only emission — nothing written to graph state |
| `merge=True` in metadata | Reducer merges `{**prev_msg['props'], **new_msg['props']}` instead of replacing |
| `remove-ui` for existing ID | Adds ID to `ids_to_remove` set; message is filtered at end |
| `remove-ui` for unknown ID | Raises `ValueError` — IDs must exist before they can be removed |
| `merge=True` dedup | If same ID already in list, updates it in-place (no append); `merge` flag in metadata triggers prop merge |
| `id` default | `str(uuid4())` when not provided |

### Example 1 — streaming a progress indicator component

```python
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import push_ui_message, delete_ui_message, ui_message_reducer, AnyUIMessage

class State(TypedDict):
    result: str
    ui: Annotated[list[AnyUIMessage], ui_message_reducer]

def long_running_task(state: State) -> dict:
    # Emit a spinner to the stream
    msg = push_ui_message(
        name="Spinner",
        props={"label": "Processing..."},
        id="progress-1",
    )
    # ... do actual work ...
    time.sleep(0.1)

    # Update the spinner with progress (merge=True keeps other props)
    push_ui_message(
        name="Spinner",
        props={"label": "Almost done..."},
        id="progress-1",
        merge=True,
    )
    return {"result": "done"}

def finish(state: State) -> dict:
    # Remove the spinner now that we're done
    delete_ui_message("progress-1")
    return {}

builder = StateGraph(State)
builder.add_node("task", long_running_task)
builder.add_node("finish", finish)
builder.add_edge(START, "task")
builder.add_edge("task", "finish")
builder.add_edge("finish", END)
graph = builder.compile()

result = graph.invoke({"result": "", "ui": []})
print(result["ui"])  # [] — spinner was removed
```

### Example 2 — `ui_message_reducer` merge semantics

```python
from langgraph.graph.ui import ui_message_reducer

# Start with one message
state: list = [
    {"type": "ui", "id": "card-1", "name": "Card",
     "props": {"title": "Hello", "body": "World"}, "metadata": {}}
]

# Update only the title via merge=True
update: list = [
    {"type": "ui", "id": "card-1", "name": "Card",
     "props": {"title": "Updated Title"}, "metadata": {"merge": True}}
]
merged = ui_message_reducer(state, update)
print(merged[0]["props"])
# {'title': 'Updated Title', 'body': 'World'}  — body preserved

# Full replacement via merge=False (default)
replace: list = [
    {"type": "ui", "id": "card-1", "name": "Card",
     "props": {"title": "Replaced"}, "metadata": {"merge": False}}
]
replaced = ui_message_reducer(merged, replace)
print(replaced[0]["props"])
# {'title': 'Replaced'}  — body is gone

# Remove
removed = ui_message_reducer(replaced, [{"type": "remove-ui", "id": "card-1"}])
print(removed)  # []

# remove-ui for non-existent ID raises ValueError
try:
    ui_message_reducer([], [{"type": "remove-ui", "id": "does-not-exist"}])
except ValueError as e:
    print(e)  # "Attempting to delete an UI message with an ID that doesn't exist ('does-not-exist')"
```

### Example 3 — stream-only UI events with `state_key=None`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import push_ui_message

class State(TypedDict):
    answer: str
    # No 'ui' key — all events are stream-only

def generate(state: State) -> dict:
    # Emit thinking indicator to stream but don't persist in state
    push_ui_message(
        name="ThinkingIndicator",
        props={"active": True},
        state_key=None,  # stream-only
    )

    result = "42"  # ... actual computation ...

    push_ui_message(
        name="ThinkingIndicator",
        props={"active": False},
        state_key=None,
    )
    return {"answer": result}

builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_edge(START, "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# Observe stream-only UI events without state pollution
for chunk in graph.stream({"answer": ""}, stream_mode="custom"):
    if isinstance(chunk, dict) and chunk.get("type") == "ui":
        print(f"UI event: {chunk['name']} active={chunk['props']['active']}")
```

---

## 8 · `ManagedValue` + `IsLastStepManager` + `RemainingStepsManager`

**Module:** `langgraph.managed.base` · `langgraph.managed.is_last_step`

Managed values are type annotations that LangGraph resolves from the `PregelScratchpad` at the start of each node invocation, injecting a computed value into the state dict. They look like regular state fields but are never stored or updated by nodes. `IsLastStep` and `RemainingSteps` are the two built-in managed values; you can create custom ones.

### Source signatures

```python
# langgraph.managed.base
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]

def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)

ManagedValueMapping = dict[str, ManagedValueSpec]

# langgraph.managed.is_last_step
class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

IsLastStep = Annotated[bool, IsLastStepManager]

class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `get(scratchpad)` | `@staticmethod` — no instance created; the class itself is stored in `ManagedValueMapping` |
| `IsLastStep` truth | `scratchpad.step == scratchpad.stop - 1` — true only on the final recursion step |
| `RemainingSteps` value | `scratchpad.stop - scratchpad.step` — counts down from `recursion_limit` |
| `scratchpad.stop` | Set from `config["recursion_limit"]` (default 25); see `PregelScratchpad` source |
| `is_managed_value` | Checks `isclass(value) and issubclass(value, ManagedValue)` — type annotation scanning |
| No checkpointing | Managed values are never stored — they are always recomputed from the scratchpad |

### Example 1 — using built-in `IsLastStep` and `RemainingSteps`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps

class State(TypedDict):
    messages: list[str]
    is_last: IsLastStep
    remaining: RemainingSteps

def agent(state: State) -> dict:
    print(f"is_last={state['is_last']}, remaining={state['remaining']}")
    if state["is_last"]:
        return {"messages": [*state["messages"], "FINAL"]}
    return {"messages": [*state["messages"], "step"]}

def router(state: State) -> str:
    if "FINAL" in state["messages"]:
        return END
    return "agent"

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router)
graph = builder.compile()

result = graph.invoke({"messages": []}, config={"recursion_limit": 3})
print(result["messages"])
# ['step', 'step', 'FINAL']  — last step triggers FINAL
```

### Example 2 — custom `ManagedValue` that reads the scratchpad step counter

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.graph import StateGraph, START, END

class StepIndexManager(ManagedValue[int]):
    """Inject the current zero-based step index into the node state."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.step

StepIndex = Annotated[int, StepIndexManager]

class State(TypedDict):
    log: list[str]
    step_index: StepIndex

def node(state: State) -> dict:
    return {"log": [*state["log"], f"step-{state['step_index']}"]}

def router(state: State) -> str:
    if len(state["log"]) >= 3:
        return END
    return "node"

builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")
builder.add_conditional_edges("node", router)
graph = builder.compile()

result = graph.invoke({"log": []})
print(result["log"])  # ['step-0', 'step-1', 'step-2']
```

### Example 3 — `is_managed_value` introspection to separate channels from managed values

```python
from typing import Annotated, get_args, get_origin
from typing_extensions import TypedDict
from langgraph.managed.base import is_managed_value
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
import operator

class State(TypedDict):
    count: Annotated[int, operator.add]
    messages: list[str]
    is_last: IsLastStep
    remaining: RemainingSteps

# Walk type hints and classify each field
import typing
hints = typing.get_type_hints(State, include_extras=True)

for name, hint in hints.items():
    origin = get_origin(hint)
    if origin is Annotated:
        args = get_args(hint)
        metadata = args[1:]
        for m in metadata:
            if is_managed_value(m):
                print(f"  {name}: MANAGED VALUE ({m.__name__})")
                break
        else:
            print(f"  {name}: REDUCER channel (metadata={metadata})")
    else:
        print(f"  {name}: plain field ({hint})")

# Output:
#   count: REDUCER channel (metadata=(<built-in function add>,))
#   messages: plain field (list[str])
#   is_last: MANAGED VALUE (IsLastStepManager)
#   remaining: MANAGED VALUE (RemainingStepsManager)
```

---

## 9 · `get_field_default` + `get_cached_annotated_keys` + `get_enhanced_type_hints`

**Module:** `langgraph._internal._fields`

**First dedicated coverage.** This module contains the low-level introspection utilities LangGraph uses to resolve default values for TypedDict, dataclass, and Pydantic state schemas at graph compile time. `get_cached_annotated_keys` is particularly important: it walks the full MRO to collect annotation keys in definition order, and caches the result in a `WeakKeyDictionary` to avoid repeated traversal.

### Source signatures (condensed)

```python
def get_field_default(name: str, type_: Any, schema: type[Any]) -> Any:
    """Return None (optional field), ... (required), or the dataclass default."""

def get_enhanced_type_hints(type: type[Any]) -> Generator[
    tuple[str, Any, Any, str | None], None, None
]:
    """Yield (name, type, default, description) for config schema generation."""

ANNOTATED_KEYS_CACHE: WeakKeyDictionary[type[Any], tuple[str, ...]] = WeakKeyDictionary()

def get_cached_annotated_keys(obj: type[Any]) -> tuple[str, ...]:
    """Return MRO-ordered annotation keys, cached per class."""

def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Filter Pydantic model updates to only changed/non-default fields."""
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `get_field_default` priority | 1. `total=False` or `NotRequired` → `None`; 2. `Required` → `...`; 3. dataclass default/factory; 4. `Optional[T]` → `None`; else → `...` |
| `ANNOTATED_KEYS_CACHE` | `WeakKeyDictionary` — entries are garbage-collected with the class they describe |
| MRO order | `get_cached_annotated_keys` iterates `reversed(obj.__mro__)` so base-class keys appear before subclass overrides |
| `get_update_as_tuples` Pydantic | Only returns fields that are in `model_fields_set` OR differ from default — prevents spurious reducer updates |
| Python 3.14 compat | Falls back to `getattr(base, "__annotations__")` if `__dict__.get` returns `GetSetDescriptorType` (Pydantic v3 behaviour) |

### Example 1 — inspecting default values for a mixed TypedDict

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict, NotRequired, Required
import operator
from langgraph._internal._fields import get_field_default

class AppState(TypedDict, total=False):
    name: Required[str]     # explicitly required even in total=False dict
    count: Annotated[int, operator.add]
    tags: list[str]
    description: Optional[str]

hints = {
    "name": Required[str],
    "count": Annotated[int, operator.add],
    "tags": list[str],
    "description": Optional[str],
}

for field, typ in hints.items():
    default = get_field_default(field, typ, AppState)
    label = "required (Ellipsis)" if default is ... else f"default={default!r}"
    print(f"  {field}: {label}")

# name: required (Ellipsis)  — Required overrides total=False
# count: required (Ellipsis)  — int is not Optional
# tags:  required (Ellipsis)  — list is not Optional
# description: default=None  — Optional[str] gives None default
```

### Example 2 — `get_cached_annotated_keys` with multiple inheritance

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph._internal._fields import get_cached_annotated_keys, ANNOTATED_KEYS_CACHE

class BaseState(TypedDict):
    base_field: str
    shared: int

class ExtendedState(BaseState):
    extra_field: Annotated[list[str], operator.add]
    shared: Annotated[int, operator.add]  # override with reducer

# First call: traverses MRO and caches
keys = get_cached_annotated_keys(ExtendedState)
print(keys)
# ('base_field', 'shared', 'extra_field', 'shared')
# Note: 'shared' appears twice — from BaseState and ExtendedState
# The second 'shared' (with the reducer) wins in LangGraph's channel resolution

# Cached on second call — no MRO traversal
keys_again = get_cached_annotated_keys(ExtendedState)
assert ExtendedState in ANNOTATED_KEYS_CACHE

# WeakKeyDictionary: if class is deleted, cache entry is auto-removed
import gc
del ExtendedState
gc.collect()
print(f"cache size after deletion: {len(ANNOTATED_KEYS_CACHE)}")
```

### Example 3 — `get_update_as_tuples` to see which Pydantic fields LangGraph will update

```python
from pydantic import BaseModel, Field
from langgraph._internal._fields import get_update_as_tuples

class AgentOutput(BaseModel):
    message: str = ""
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)

# Simulate a node that only sets 'message'
output = AgentOutput(message="Hello world")
print(f"model_fields_set: {output.model_fields_set}")
# {'message'}

# get_update_as_tuples returns only the changed fields
keys = list(AgentOutput.model_fields.keys())
updates = get_update_as_tuples(output, keys)
print(updates)
# [('message', 'Hello world')]
# 'score' and 'metadata' are at default — omitted from updates
# This prevents unnecessary reducer calls for unchanged fields
```

---

## 10 · `PregelProtocol` + `StreamProtocol`

**Module:** `langgraph.pregel.protocol`

**First dedicated coverage.** `PregelProtocol` is the abstract interface that every LangGraph graph executor implements — both the local `Pregel` class and the remote `RemoteGraph` satisfy it. Any code that accepts a `PregelProtocol` works equally with local and deployed graphs. `StreamProtocol` is the slim two-slot object that maps stream modes to their dispatch callable.

### Source signatures

```python
class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
    # Graph structure
    @abstractmethod
    def get_graph(self, config, *, xray: int | bool = False) -> DrawableGraph: ...
    async def aget_graph(self, ...) -> DrawableGraph: ...

    # State access
    @abstractmethod
    def get_state(self, config, *, subgraphs: bool = False) -> StateSnapshot: ...
    async def aget_state(self, ...) -> StateSnapshot: ...

    @abstractmethod
    def get_state_history(self, config, *, filter, before, limit) -> Iterator[StateSnapshot]: ...
    def aget_state_history(self, ...) -> AsyncIterator[StateSnapshot]: ...

    # State mutation
    @abstractmethod
    def update_state(self, config, values, as_node=None) -> RunnableConfig: ...
    async def aupdate_state(self, ...) -> RunnableConfig: ...

    @abstractmethod
    def bulk_update_state(self, config, updates) -> RunnableConfig: ...
    async def abulk_update_state(self, ...) -> RunnableConfig: ...

    # Execution — v1 and v2 overloads
    @abstractmethod
    def stream(self, input, config=None, *, version: Literal["v1","v2"] = "v1", ...) -> Iterator: ...
    @abstractmethod
    def astream(self, ...) -> AsyncIterator: ...
    @abstractmethod
    def invoke(self, ...) -> dict | GraphOutput: ...
    @abstractmethod
    async def ainvoke(self, ...) -> dict | GraphOutput: ...


StreamChunk = tuple[tuple[str, ...], str, Any]  # (namespace, mode, payload)

class StreamProtocol:
    __slots__ = ("modes", "__call__")

    modes: set[StreamMode]
    __call__: Callable[[StreamChunk], None]

    def __init__(
        self, __call__: Callable[[StreamChunk], None], modes: set[StreamMode]
    ) -> None: ...
```

### Key source-verified facts

| Detail | Value |
|--------|-------|
| `PregelProtocol` generics | `Generic[StateT, ContextT, InputT, OutputT]` — four type parameters for type-safe graphs |
| `stream` v1 return | `Iterator[dict[str, Any] | Any]` — untyped chunks |
| `stream` v2 return | `Iterator[StreamPart[StateT, OutputT]]` — typed `StreamPart` union |
| `StreamChunk` tuple | `(namespace, mode, payload)` — `namespace` is `tuple[str, ...]` for subgraph path |
| `StreamProtocol.__slots__` | `("modes", "__call__")` — ultra-lightweight; no `__dict__` |
| `modes` attribute | `set[StreamMode]` — used by Pregel to decide which transformers to activate |
| Dual implementation | Both `Pregel` (local) and `RemoteGraph` (SDK) implement `PregelProtocol` |

### Example 1 — writing graph-agnostic code with `PregelProtocol`

```python
from typing import Any, Iterator
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import StateSnapshot

def replay_last_n_states(
    graph: PregelProtocol,
    config: RunnableConfig,
    n: int = 3,
) -> list[StateSnapshot]:
    """Works with any PregelProtocol implementor — local or remote."""
    history = list(graph.get_state_history(config, limit=n))
    return history

# Works with local graph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

class S(TypedDict):
    v: int

builder = StateGraph(S)
builder.add_node("inc", lambda s: {"v": s["v"] + 1})
builder.add_edge(START, "inc")
builder.add_edge("inc", END)
saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
cfg = {"configurable": {"thread_id": "t1"}}
graph.invoke({"v": 0}, cfg)
graph.invoke({"v": 0}, cfg)

states = replay_last_n_states(graph, cfg, n=2)
print(f"Replayed {len(states)} snapshots, latest v={states[0].values['v']}")
```

### Example 2 — `StreamProtocol` internals: modes filtering

```python
from langgraph.pregel.protocol import StreamProtocol, StreamChunk

collected: list[StreamChunk] = []

def my_handler(chunk: StreamChunk) -> None:
    collected.append(chunk)

# Create a StreamProtocol that only handles "values" and "updates"
sp = StreamProtocol(my_handler, modes={"values", "updates"})

# Check which modes this protocol handles
print(sp.modes)  # {'values', 'updates'}

# The __call__ dispatches chunks
chunk: StreamChunk = (("root",), "values", {"v": 42})
sp(chunk)

print(collected)
# [(('root',), 'values', {'v': 42})]

# Mode filtering happens at the Pregel level before calling StreamProtocol;
# StreamProtocol itself passes all incoming chunks to __call__ without filtering.
print(f"'debug' in sp.modes: {'debug' in sp.modes}")  # False
```

### Example 3 — `isinstance` check against `PregelProtocol` for runtime dispatch

```python
from langgraph.pregel.protocol import PregelProtocol
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    x: int

builder = StateGraph(S)
builder.add_node("n", lambda s: {"x": s["x"] + 1})
builder.add_edge(START, "n")
builder.add_edge("n", END)
local_graph = builder.compile()

# isinstance check does not work because PregelProtocol is a Protocol
# without @runtime_checkable — use type() or duck-typing instead
print(type(local_graph).__name__)    # CompiledStateGraph
print(hasattr(local_graph, "get_state"))  # True
print(hasattr(local_graph, "get_state_history"))  # True
print(hasattr(local_graph, "update_state"))  # True

# Check all required PregelProtocol methods exist
required = [
    "get_state", "aget_state",
    "get_state_history", "aget_state_history",
    "update_state", "aupdate_state",
    "bulk_update_state", "abulk_update_state",
    "stream", "astream", "invoke", "ainvoke",
    "get_graph", "aget_graph",
]
for method in required:
    assert hasattr(local_graph, method), f"Missing: {method}"
print("All PregelProtocol methods present.")
```

---

## Summary

| # | Class / symbol | Module | First coverage? |
|---|---------------|--------|-----------------|
| 1 | `BaseChannel` | `langgraph.channels.base` | First dedicated deep-dive |
| 2 | `BinaryOperatorAggregate` | `langgraph.channels.binop` | First dedicated deep-dive |
| 3 | `UntrackedValue` | `langgraph.channels.untracked_value` | Deeper than Vol. 5 brief mention |
| 4 | `register_serde_event_listener` + `SerdeEvent` | `langgraph.checkpoint.serde.event_hooks` | **First ever** |
| 5 | `SerializerProtocol` + `CipherProtocol` + `SerializerCompat` | `langgraph.checkpoint.serde.base` | First dedicated deep-dive |
| 6 | `AsyncBatchedBaseStore` | `langgraph.store.base.batch` | Deeper than Vol. 15 brief mention |
| 7 | `UIMessage` + `RemoveUIMessage` + `ui_message_reducer` | `langgraph.graph.ui` | Deeper — full reducer semantics |
| 8 | `ManagedValue` + `IsLastStepManager` + `RemainingStepsManager` | `langgraph.managed` | Deeper — `PregelScratchpad` linkage |
| 9 | `get_field_default` + `get_cached_annotated_keys` + `get_enhanced_type_hints` | `langgraph._internal._fields` | **First ever** |
| 10 | `PregelProtocol` + `StreamProtocol` | `langgraph.pregel.protocol` | **First ever** |

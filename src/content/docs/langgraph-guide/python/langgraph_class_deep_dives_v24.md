---
title: "Class deep-dives Vol. 24 — debug payloads, injection internals, task IDs, deprecation hierarchy & serde safety (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: _DebugCheckpointPayload+_DebugTaskPayload+_DebugTaskResultPayload+DebugPayload (typed debug-stream envelopes — discriminating on 'type', walking the debug event log programmatically), _InjectedArgs+_DirectlyInjectedToolArg (ToolNode injection map built at __init__ — state-field narrowing, store injection, runtime injection, InjectedToolArg sentinel), _TaskIDFn+_uuid5_str+_xxhash_str+task_path_str (task ID generation protocol — SHA-1 UUID5 vs XXH3-128 content-addressed IDs, is_xxh3_128_hexdigest detector), LangGraphDeprecationWarning+LangGraphDeprecatedSinceV05/V10/V11 (full deprecation hierarchy — since/expected_removal fields, __str__ format, V11 new in 1.2.6), DeprecatedKwargs (TypedDict sentinel for deprecated kwargs — enables static-analysis warnings on deprecated keyword arguments in add_node/compile), InvalidModuleError (strict-msgpack allowlist violation — LANGGRAPH_STRICT_MSGPACK env var, register_serde_event_listener audit trail, per-class explicit allowlist construction), _TasksLifecycleBase (shared template-method base for LifecycleTransformer+SubgraphTransformer — _should_track/_on_started/_on_terminal hooks, tasks-channel suppression), _DeltaSnapshot (checkpoint blob for DeltaChannel finite snapshot_frequency — EXT_DELTA_SNAPSHOT msgpack ext code, ancestor walk termination, from_checkpoint shortcut), CacheKey (namespace+key+ttl NamedTuple for cache entries — how CachePolicy drives the key, how InMemoryCache uses the namespace tuple), and _ToolCallRequestOverrides (TypedDict for ToolCallRequest.override() — tool_call/tool/state keys, immutable-update pattern for ToolNode interceptors)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 24"
  order: 55
---

# Class deep-dives Vol. 24 — debug payloads, injection internals, task IDs, deprecation hierarchy & serde safety (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `_DebugCheckpointPayload` + `_DebugTaskPayload` + `_DebugTaskResultPayload` + `DebugPayload` | `langgraph.types` |
| 2 | `_InjectedArgs` + `_DirectlyInjectedToolArg` | `langgraph.prebuilt.tool_node` |
| 3 | `_TaskIDFn` + `_uuid5_str` + `_xxhash_str` + `task_path_str` | `langgraph.pregel._algo` |
| 4 | `LangGraphDeprecationWarning` + `LangGraphDeprecatedSinceV05/V10/V11` | `langgraph.warnings` |
| 5 | `DeprecatedKwargs` | `langgraph._internal._typing` |
| 6 | `InvalidModuleError` | `langgraph.checkpoint.serde.jsonplus` |
| 7 | `_TasksLifecycleBase` | `langgraph.stream.transformers` |
| 8 | `_DeltaSnapshot` | `langgraph.checkpoint.serde.types` |
| 9 | `CacheKey` | `langgraph.types` |
| 10 | `_ToolCallRequestOverrides` | `langgraph.prebuilt.tool_node` |

---

## 1 · `_DebugCheckpointPayload` + `_DebugTaskPayload` + `_DebugTaskResultPayload` + `DebugPayload`

**Module**: `langgraph.types`  
**First dedicated coverage.**

`stream_mode="debug"` emits a sequence of wrapped `DebugPayload` events — one per task start (`"task"`), one per task finish (`"task_result"`), and one per checkpoint (`"checkpoint"`). Each raw dict has a `"type"` discriminator; the three typed wrappers let you exhaustively narrow at static-analysis time.

```python
class _DebugTaskPayload(TypedDict):
    step: int        # 0-indexed superstep number
    timestamp: str   # ISO 8601 wall clock
    type: Literal["task"]
    payload: TaskPayload  # id, name, input, triggers, metadata

class _DebugTaskResultPayload(TypedDict):
    step: int
    timestamp: str
    type: Literal["task_result"]
    payload: TaskResultPayload  # id, name, error, result (list of writes)

class _DebugCheckpointPayload(TypedDict, Generic[StateT]):
    step: int
    timestamp: str
    type: Literal["checkpoint"]
    payload: CheckpointPayload[StateT]  # config, values, metadata, next, tasks, interrupts

DebugPayload = (
    _DebugCheckpointPayload[StateT] | _DebugTaskPayload | _DebugTaskResultPayload
)
```

### Consuming `stream_mode="debug"`

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools([multiply])

def agent(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

from langchain_core.messages import HumanMessage

for event in graph.stream(
    {"messages": [HumanMessage(content="What is 7 * 6?")]},
    stream_mode="debug",
):
    kind = event["type"]
    step = event["step"]
    if kind == "task":
        payload = event["payload"]
        print(f"[step {step}] task STARTED: {payload['name']} (id={payload['id'][:8]})")
    elif kind == "task_result":
        payload = event["payload"]
        err = payload.get("error")
        writes = payload.get("result", [])
        print(f"[step {step}] task FINISHED: {payload['name']} — writes={[w[0] for w in writes]}"
              + (f" ERROR: {err}" if err else ""))
    elif kind == "checkpoint":
        payload = event["payload"]
        next_nodes = payload.get("next", [])
        print(f"[step {step}] CHECKPOINT — next={next_nodes}")
```

### Typed narrowing with `isinstance` / `match`

```python
from langgraph.types import (
    _DebugCheckpointPayload, _DebugTaskPayload, _DebugTaskResultPayload,
    TaskPayload, TaskResultPayload, CheckpointPayload,
)
from typing import cast

def process_debug_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "task":
        typed = cast(_DebugTaskPayload, event)
        p: TaskPayload = typed["payload"]
        print(f"  Task {p['name']} triggered by {p['triggers']}")
    elif kind == "task_result":
        typed = cast(_DebugTaskResultPayload, event)
        p2: TaskResultPayload = typed["payload"]
        if p2.get("error"):
            print(f"  Task {p2['name']} FAILED: {p2['error']}")
        else:
            print(f"  Task {p2['name']} wrote {len(p2.get('result', []))} channels")
    elif kind == "checkpoint":
        typed = cast(_DebugCheckpointPayload, event)
        p3: CheckpointPayload = typed["payload"]
        print(f"  State keys: {list(p3['values'].keys())}")

for event in graph.stream(
    {"messages": [HumanMessage(content="What is 3 * 9?")]},
    stream_mode="debug",
):
    print(f"--- {event['type']} step={event['step']} ---")
    process_debug_event(event)
```

### Async `stream_mode="debug"` with filtering

```python
import asyncio
from langchain_core.messages import HumanMessage

async def only_task_events():
    events = []
    async for event in graph.astream(
        {"messages": [HumanMessage(content="2 * 21?")]},
        stream_mode="debug",
    ):
        if event["type"] in ("task", "task_result"):
            events.append(event)
    return events

events = asyncio.run(only_task_events())
for e in events:
    print(f"{e['type']}: {e['payload']['name']}")
```

### Combining `stream_mode=["debug", "values"]`

```python
for (mode, event) in graph.stream(
    {"messages": [HumanMessage(content="7 times 8?")]},
    stream_mode=["debug", "values"],
):
    if mode == "debug":
        print(f"debug/{event['type']} step={event['step']}")
    else:
        last_msg = event["messages"][-1]
        print(f"values: last msg type={last_msg.type}")
```

---

## 2 · `_InjectedArgs` + `_DirectlyInjectedToolArg`

**Module**: `langgraph.prebuilt.tool_node`  
**First dedicated coverage.**

`_InjectedArgs` is the internal data structure that `ToolNode.__init__` builds once per tool by analysing the tool's parameter annotations. During node execution it is consulted — not rebuilt — for each call, avoiding repeated `inspect.signature` overhead.

```python
class _InjectedArgs:
    state: dict[str, str | None]
    # { param_name -> state_field_name_or_None }
    # None means "inject the entire state dict"

    store: str | None
    # param name for BaseStore injection, or None

    runtime: str | None
    # param name for ToolRuntime injection, or None
```

`_DirectlyInjectedToolArg` is a sentinel NamedTuple used when a parameter is annotated with `InjectedToolArg` (not `InjectedState`) — it marks the field as "provided by the framework, not the model".

```python
class _DirectlyInjectedToolArg(NamedTuple):
    name: str  # tool parameter name
```

### How `ToolNode` builds `_InjectedArgs`

During `ToolNode.__init__`, the node calls `_get_injected_args(tool)` for each tool. That helper:
1. Walks `tool.args_schema.model_fields` (for Pydantic-backed tools) or `inspect.signature(tool.func)`.
2. Checks each parameter annotation for `InjectedState`, `InjectedStore`, or `ToolRuntime`.
3. Returns an `_InjectedArgs` instance with the injection map.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from typing import Annotated
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: list
    user_id: str

# State injection — ToolNode maps "user_id" param → state["user_id"]
@tool
def get_user_prefs(
    category: str,
    user_id: Annotated[str, InjectedState("user_id")],
) -> str:
    """Get user preferences. user_id is injected from state."""
    return f"Prefs for {user_id} in {category}"

# Full state injection — the entire state dict is injected
@tool
def count_messages(
    state: Annotated[AgentState, InjectedState()],
) -> int:
    """Return message count from state."""
    return len(state.get("messages", []))

# Store injection
@tool
def lookup_memory(
    query: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> list:
    """Search long-term memory."""
    return store.search(("memory",), query=query)

node = ToolNode([get_user_prefs, count_messages, lookup_memory])

# Inspect the built injection maps (internal, for debugging)
for tool_obj in node.tools_by_name.values():
    injected = node._injected_args.get(tool_obj.name)
    if injected:
        print(f"{tool_obj.name}: state={injected.state}, store={injected.store}, runtime={injected.runtime}")
```

### `InjectedToolArg` vs `InjectedState`

`InjectedToolArg` is the base annotation — it signals "this parameter is injected by the framework, not the model". `InjectedState` and `InjectedStore` are subclasses that carry the specific injection source. Both produce `_DirectlyInjectedToolArg` entries in the injection map, causing `ToolNode` to:
1. Strip the parameter from the tool's advertised JSON schema (so the model doesn't try to fill it).
2. Fill it from the graph state / store at execution time.

```python
from langgraph.prebuilt.tool_node import InjectedToolArg
from typing import Annotated

# Raw InjectedToolArg — inject a constant config value
@tool
def api_call(
    endpoint: str,
    api_key: Annotated[str, InjectedToolArg],  # injected, not model-supplied
) -> str:
    """Call an external API."""
    return f"calling {endpoint} with key={api_key[:4]}..."
```

---

## 3 · `_TaskIDFn` + `_uuid5_str` + `_xxhash_str` + `task_path_str`

**Module**: `langgraph.pregel._algo`  
**First dedicated coverage.**

Every `PregelExecutableTask` carries a deterministic `id` string derived from the checkpoint namespace, node name, and a counter or content hash. Two algorithms exist: SHA-1 UUID5 (portable, secure) and XXH3-128 (fast, non-cryptographic). The Pregel loop picks the algorithm based on `is_xxh3_128_hexdigest`.

```python
class _TaskIDFn(Protocol):
    def __call__(self, namespace: bytes, *parts: str | bytes) -> str: ...
```

### `_uuid5_str` — SHA-1 UUID5 (default for most deployments)

```python
def _uuid5_str(namespace: bytes, *parts: str | bytes) -> str:
    sha = sha1(namespace, usedforsecurity=False)
    sha.update(b"".join(p.encode() if isinstance(p, str) else p for p in parts))
    hex = sha.hexdigest()
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"
```

The result is a UUID-formatted string (but not a standards-compliant UUID5 — no version bits are set). The namespace bytes are the checkpoint namespace encoded as bytes; `parts` are `(node_name, str(task_counter))`.

### `_xxhash_str` — XXH3-128 (faster, content-addressed)

```python
def _xxhash_str(namespace: bytes, *parts: str | bytes) -> str:
    hex = xxh3_128_hexdigest(
        namespace + b"".join(p.encode() if isinstance(p, str) else p for p in parts)
    )
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"
```

XXH3-128 produces a 128-bit non-cryptographic hash in UUID format. It is faster than SHA-1 for high-throughput workloads. The Pregel loop detects which algorithm an existing task ID was produced by using `is_xxh3_128_hexdigest` — if a checkpoint was written by a different runtime, the right algorithm is used for continuation.

### `task_path_str` — task path to string

```python
def task_path_str(tup: str | int | tuple) -> str:
    return (
        f"~{', '.join(task_path_str(x) for x in tup)}"  # tuple → "~part1, part2"
        if isinstance(tup, (tuple, list))
        else f"{tup:010d}"                               # int → zero-padded 10 digits
        if isinstance(tup, int)
        else str(tup)                                    # str → pass-through
    )
```

`task_path_str` converts a task `path` tuple (which appears in `PregelExecutableTask.path` and `PregelTaskWrites.path`) to a sortable string. The zero-padding on integers ensures that `ORDER BY task_id, idx` sorts correctly in checkpointer SQL queries.

### Practical: inspecting task IDs

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def chat(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
compiled = graph.compile()

# Debug-stream reveals task IDs
for event in compiled.stream(
    {"messages": [HumanMessage(content="hello")]},
    stream_mode="debug",
):
    if event["type"] == "task":
        task_id = event["payload"]["id"]
        print(f"task id={task_id} (len={len(task_id.replace('-',''))})")
        # UUID-formatted: 8 hex chars - 4 - 4 - 4 - 12
```

---

## 4 · `LangGraphDeprecationWarning` + `LangGraphDeprecatedSinceV05/V10/V11`

**Module**: `langgraph.warnings`  
**First dedicated coverage (including `LangGraphDeprecatedSinceV11`, new in 1.2.6).**

LangGraph uses a structured deprecation hierarchy that carries machine-readable version metadata. This allows CI pipelines to detect when they are using deprecated APIs and how long they have before forced removal.

```python
class LangGraphDeprecationWarning(DeprecationWarning):
    message: str
    since: tuple[int, int]           # (major, minor) when deprecated
    expected_removal: tuple[int, int] # (major, minor) when removed

    def __str__(self) -> str:
        return (
            f"{self.message}. Deprecated in LangGraph "
            f"V{self.since[0]}.{self.since[1]} to be removed in "
            f"V{self.expected_removal[0]}.{self.expected_removal[1]}."
        )

class LangGraphDeprecatedSinceV05(LangGraphDeprecationWarning):
    # since=(0,5), expected_removal=(2,0)  — removal at major v2

class LangGraphDeprecatedSinceV10(LangGraphDeprecationWarning):
    # since=(1,0), expected_removal=(2,0)  — removal at major v2

class LangGraphDeprecatedSinceV11(LangGraphDeprecationWarning):  # NEW in 1.2.6
    # since=(1,1), expected_removal=(3,0)  — removal at major v3
```

The `V11` sentinel is new in 1.2.6. It carries `expected_removal=(3, 0)` — removal is deferred until v3.0, giving a longer deprecation window than `V05`/`V10` (both target removal at v2.0).

### Detecting deprecated APIs in CI

```python
import warnings
from langgraph.warnings import LangGraphDeprecationWarning

# Convert all LangGraph deprecations to errors in tests
with warnings.catch_warnings():
    warnings.simplefilter("error", LangGraphDeprecationWarning)
    try:
        from langgraph.prebuilt import ValidationNode
        node = ValidationNode([])   # ← deprecated since V1.0
    except LangGraphDeprecationWarning as e:
        print(f"Deprecated API: {e}")
        print(f"  since: {e.since}")
        print(f"  must fix before: {e.expected_removal}")
```

### Filtering by removal version

```python
import warnings
from langgraph.warnings import LangGraphDeprecationWarning

def check_removal_urgency(w: warnings.WarningMessage) -> str:
    exc = w.category
    if issubclass(exc, LangGraphDeprecationWarning):
        expected = getattr(w.message, "expected_removal", None)
        if expected and expected <= (2, 0):
            return "URGENT"
        return "DEFERRED"
    return "OTHER"

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", LangGraphDeprecationWarning)
    # Run your application code here...
    from langgraph.prebuilt import ValidationNode  # triggers a V10 warning
    node = ValidationNode([])

for w in caught:
    urgency = check_removal_urgency(w)
    msg = str(w.message)
    print(f"[{urgency}] {msg}")
```

### Where V11 is used in 1.2.6

`LangGraphDeprecatedSinceV11` is currently used to warn when `AgentState` / `AgentStatePydantic` are imported from `langgraph.prebuilt`. These were the built-in state types before custom `TypedDict` schemas became the recommended approach:

```python
# This triggers LangGraphDeprecatedSinceV11:
# from langgraph.prebuilt import AgentState  # deprecated

# Correct replacement:
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
```

---

## 5 · `DeprecatedKwargs`

**Module**: `langgraph._internal._typing`  
**First dedicated coverage.**

`DeprecatedKwargs` is a `TypedDict` sentinel with no keys. It is used as the base class for a `**kwargs`-typed parameter that carries deprecated keyword arguments. By giving the kwargs dict a TypedDict type, static analysers (mypy, pyright) can emit warnings when callers pass any key.

```python
class DeprecatedKwargs(TypedDict):
    """TypedDict to use for extra keyword arguments, enabling
    type checking warnings for deprecated arguments."""
```

### How it's used in LangGraph internals

In `StateGraph.add_node()` and `CompiledStateGraph.compile()`, optional legacy parameters are typed as `**kwargs: Unpack[DeprecatedKwargs]`. Since `DeprecatedKwargs` has no keys, any keyword argument passed to those positions will produce a `TypedDict` incompatibility error in strict type-checking mode.

```python
from langgraph._internal._typing import DeprecatedKwargs
from typing_extensions import Unpack

# Pattern used inside LangGraph (illustration):
def add_node(
    self,
    node: str | None = None,
    action: Any | None = None,
    *,
    metadata: dict | None = None,
    **kwargs: Unpack[DeprecatedKwargs],  # <-- static warning if any key is passed
) -> None: ...
```

### Using the pattern in your own APIs

If you build LangGraph extensions or wrappers that need to sunset keyword arguments, `DeprecatedKwargs` is a clean way to surface deprecations in IDEs and CI:

```python
from typing_extensions import TypedDict, Unpack
import warnings

class _LegacyKwargs(TypedDict, total=False):
    use_async: bool       # deprecated; always True now
    timeout_sec: float    # deprecated; use TimeoutPolicy instead

def compile_graph(
    graph: object,
    *,
    checkpointer: object | None = None,
    **kwargs: Unpack[_LegacyKwargs],
) -> object:
    if "use_async" in kwargs:
        warnings.warn(
            "use_async is deprecated and has no effect; async is always enabled.",
            DeprecationWarning,
            stacklevel=2,
        )
    if "timeout_sec" in kwargs:
        warnings.warn(
            "timeout_sec is deprecated; use TimeoutPolicy(run_timeout=...) on add_node.",
            DeprecationWarning,
            stacklevel=2,
        )
    return graph  # actual compilation elided
```

---

## 6 · `InvalidModuleError`

**Module**: `langgraph.checkpoint.serde.jsonplus`  
**First dedicated coverage.**

`InvalidModuleError` is raised when `JsonPlusSerializer` tries to deserialise a msgpack value whose type is not in the allowlist and `LANGGRAPH_STRICT_MSGPACK=true` is set (or an explicit `allowed_msgpack_modules` was passed).

```python
class InvalidModuleError(Exception):
    def __init__(self, message: str):
        self.message = message
```

### When it is raised

The serializer calls `emit_serde_event()` every time it encounters a type it would decode. Under strict mode, if the type's `(module, name)` tuple is not in `allowed_msgpack_modules`, the ext-hook raises `InvalidModuleError`.

```python
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener

# Audit every type that passes through the serializer
def audit_listener(event):
    print(f"serde: module={event['module']}, name={event['name']}, kind={event['kind']}")

unregister = register_serde_event_listener(audit_listener)

serde = JsonPlusSerializer()

# Demonstrate: roundtrip a plain dict — no exotic types, no error
data = {"key": "value", "count": 42}
typ, raw = serde.dumps_typed(data)
print(f"encoded as: {typ}")
result = serde.loads_typed((typ, raw))
print(f"decoded: {result}")

unregister()
```

### Building an explicit allowlist

```python
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.event_hooks import (
    register_serde_event_listener,
    SerdeEvent,
)

# First: discover all types you actually use in your state
discovered: set[tuple[str, str]] = set()

def discover_listener(event: SerdeEvent) -> None:
    discovered.add((event["module"], event["name"]))

unregister = register_serde_event_listener(discover_listener)

# Run your graph once in a dev environment to populate `discovered`
# ... graph.invoke(...) ...

unregister()

# Then hardcode the allowlist in production:
explicit_allowlist = list(discovered)
serde = JsonPlusSerializer(allowed_msgpack_modules=explicit_allowlist)

from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver(serde=serde)
```

### Catching `InvalidModuleError` at graph startup

```python
from langgraph.checkpoint.serde.jsonplus import InvalidModuleError, JsonPlusSerializer
import os

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

serde = JsonPlusSerializer()

try:
    # This would fail if the checkpoint contains a type not in the allowlist:
    serde.loads_typed(("msgpack", b"\x81\xc7\x18some_ext_code"))
except InvalidModuleError as e:
    print(f"Blocked type: {e.message}")
    # Log and fallback or raise a startup error
except Exception:
    pass  # other decoding errors
```

---

## 7 · `_TasksLifecycleBase`

**Module**: `langgraph.stream.transformers`  
**First dedicated coverage.**

`_TasksLifecycleBase` is the shared base class for `LifecycleTransformer` and `SubgraphTransformer`. Both need to discover subgraphs by watching `tasks` events — `_TasksLifecycleBase` centralises the detection logic so the two surfaces stay in sync.

```python
class _TasksLifecycleBase(StreamTransformer):
    # Template-method hooks for subclasses:
    def _should_track(self, ns: tuple[str, ...]) -> bool: ...
    def _on_started(self, ns: tuple[str, ...], graph_name: str | None,
                    trigger_call_id: str | None) -> None: ...
    def _on_terminal(self, ns: tuple[str, ...],
                     status: Literal["done", "error", "interrupt"],
                     error: str | None) -> None: ...

    def process(self, event: ProtocolEvent) -> bool:
        # Watches "tasks" channel events:
        # - "started" → call _on_started for the first new namespace
        # - "TaskResultPayload" from parent → call _on_terminal
        # Returns False to suppress tasks events from the main event log
        ...
```

### How `LifecycleTransformer` and `SubgraphTransformer` differ

| Aspect | `LifecycleTransformer` | `SubgraphTransformer` |
|---|---|---|
| `_should_track` | All namespaces with depth > current | Only direct children |
| `_on_started` | Pushes `LifecyclePayload(status="running")` to `run.lifecycle` | Builds a `SubgraphRunStream` handle, pushes to `run.subgraphs` |
| `_on_terminal` | Pushes `LifecyclePayload(status="done/error/interrupt")` | Updates `SubgraphRunStream.status`, closes its channels |

### Writing a custom `_TasksLifecycleBase` subclass

You can subclass `_TasksLifecycleBase` to build your own lifecycle tracking transformer — for example, to record execution metrics per subgraph invocation:

```python
from langgraph.stream.transformers import _TasksLifecycleBase
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel
from typing import Any, Literal

class MetricsTransformer(_TasksLifecycleBase):
    """Track how long each subgraph namespace takes."""

    required_stream_modes = ("tasks",)
    _native = False  # expose via run.extensions["metrics"]

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._started_at: dict[tuple[str, ...], str] = {}
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"metrics": self._log}

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        # Track all sub-namespaces (not the root itself)
        return len(ns) > len(self.scope)

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        import datetime
        self._started_at[ns] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: Literal["done", "error", "interrupt"],
        error: str | None,
    ) -> None:
        import datetime
        started = self._started_at.pop(ns, None)
        entry = {
            "namespace": "/".join(ns),
            "status": status,
            "started_at": started,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": error,
        }
        self._log.push(entry)

    def finalize(self) -> None:
        self._log.close()

    def fail(self, err: BaseException) -> None:
        self._log.fail(err)
```

---

## 8 · `_DeltaSnapshot`

**Module**: `langgraph.checkpoint.serde.types`  
**First dedicated coverage.**

`_DeltaSnapshot` is a `NamedTuple` used when `DeltaChannel` is configured with a finite `snapshot_frequency`. It acts as a "terminate the ancestor walk" marker stored in `checkpoint_blobs`.

```python
class _DeltaSnapshot(NamedTuple):
    """Snapshot blob for a DeltaChannel with finite snapshot_frequency.

    Stored in checkpoint_blobs via the EXT_DELTA_SNAPSHOT msgpack ext code.
    The ancestor walk in BaseCheckpointSaver.get_delta_channel_history terminates
    when it encounters this type — it IS the accumulated state.
    """
    value: Any
```

### Background: how `DeltaChannel` and snapshots work

A `DeltaChannel` stores only the writes from each superstep (deltas), not the full accumulated value. To reconstruct the full value at any checkpoint you walk backwards through ancestor checkpoints and replay deltas. With large histories this walk becomes expensive.

Setting `snapshot_frequency=N` causes `DeltaChannel` to write a `_DeltaSnapshot` blob every `N` steps. When the checkpointer walks ancestors to reconstruct a `DeltaChannel` value it stops as soon as it sees a `_DeltaSnapshot` — that blob contains the fully accumulated value at that point.

> **Note on `StateGraph`:** `StateGraph` backs `Annotated[Type, reducer]` fields with `BinaryOperatorAggregate`, not `DeltaChannel`. `DeltaChannel` is an internal channel type used in low-level `Pregel` instances; it does not appear automatically when you use `StateGraph`. The serialiser support for `_DeltaSnapshot` is demonstrated below using the serialiser directly and via `get_state_history` for a regular `StateGraph`.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict

def append_reducer(existing: list, new: list) -> list:
    return (existing or []) + new

class EventState(TypedDict):
    # Backed by BinaryOperatorAggregate in StateGraph (not DeltaChannel)
    events: Annotated[list[str], append_reducer]

def add_event(state: EventState) -> dict:
    return {"events": [f"event_{len(state['events']) + 1}"]}

builder = StateGraph(EventState)
builder.add_node("add", add_event)
builder.add_edge(START, "add")
builder.add_edge("add", END)

mem = MemorySaver()
graph = builder.compile(checkpointer=mem)

config = {"configurable": {"thread_id": "delta_t1"}}
for _ in range(5):
    graph.invoke({"events": []}, config=config)

# walk checkpoint history via the standard StateGraph API
history = list(graph.get_state_history(config))
print(f"Checkpoint count: {len(history)}")
for snap in history[:3]:
    step = snap.metadata.get("step", "?")
    print(f"  step {step}: events={snap.values.get('events', [])}")
```

### Inspecting `_DeltaSnapshot` in checkpoint blobs

```python
from langgraph.checkpoint.serde.types import _DeltaSnapshot

# Simulate a snapshot:
snap = _DeltaSnapshot(value=["event_1", "event_2", "event_3"])
print(f"snapshot value: {snap.value}")
print(f"is NamedTuple: {isinstance(snap, tuple)}")

# The serializer uses EXT_DELTA_SNAPSHOT (ext code 4) for msgpack encoding.
# You can inspect raw checkpoint blobs:
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde = JsonPlusSerializer()
typ, data = serde.dumps_typed(snap)
print(f"serialized type tag: {typ}")  # "msgpack"
restored = serde.loads_typed((typ, data))
print(f"restored: {restored}, is _DeltaSnapshot: {isinstance(restored, _DeltaSnapshot)}")
```

---

## 9 · `CacheKey`

**Module**: `langgraph.types`  
**First dedicated coverage (mentioned 15x in prior volumes, never the primary subject).**

`CacheKey` is the `NamedTuple` that identifies a single entry in a `BaseCache`. It carries the full context needed to retrieve or store a cached task result: the namespace path, the content key, and an optional TTL.

```python
class CacheKey(NamedTuple):
    ns: tuple[str, ...]   # namespace — mirrors the checkpoint namespace hierarchy
    key: str              # content hash or custom key string
    ttl: int | None       # time-to-live in seconds; None means no expiry
```

### How `CachePolicy` produces `CacheKey`

When `CachePolicy` is set on a node or `@task`, the Pregel loop calls `CachePolicy.key_func(task_input)` (default: `default_cache_key`) to produce a `str | bytes` key. That key is then wrapped in a `CacheKey(ns=checkpoint_ns, key=..., ttl=policy.ttl)`.

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    query: str
    result: str

def expensive_lookup(state: State) -> dict:
    import time
    time.sleep(0.1)  # simulate work
    return {"result": f"answer for '{state['query']}'"}

cache = InMemoryCache()

builder = StateGraph(State)
builder.add_node(
    "lookup",
    expensive_lookup,
    cache_policy=CachePolicy(ttl=300),  # → CacheKey(ns=..., key=hash, ttl=300)
)
builder.add_edge(START, "lookup")
builder.add_edge("lookup", END)
graph = builder.compile(cache=cache)

import time

# First call: cache miss
t0 = time.perf_counter()
r1 = graph.invoke({"query": "LangGraph", "result": ""})
t1 = time.perf_counter()
print(f"First call: {r1['result']} ({(t1-t0)*1000:.0f}ms)")

# Second call: cache hit
t2 = time.perf_counter()
r2 = graph.invoke({"query": "LangGraph", "result": ""})
t3 = time.perf_counter()
print(f"Second call: {r2['result']} ({(t3-t2)*1000:.0f}ms) — should be faster")
```

### Custom `key_func` and `CacheKey` composition

```python
from langgraph.types import CachePolicy

def semantic_key(state: dict) -> str:
    # Normalise query for cache hits on equivalent phrasings
    q = state.get("query", "").lower().strip().replace("  ", " ")
    return f"sem:{q}"

builder2 = StateGraph(State)
builder2.add_node(
    "lookup",
    expensive_lookup,
    cache_policy=CachePolicy(ttl=60, key_func=semantic_key),
)
builder2.add_edge(START, "lookup")
builder2.add_edge("lookup", END)
graph2 = builder2.compile(cache=cache)

# "LangGraph" and "langgraph" now share the same cache entry
graph2.invoke({"query": "LangGraph", "result": ""})
r = graph2.invoke({"query": "langgraph", "result": ""})  # cache hit
print(r["result"])
```

### `CacheKey` namespace scoping

The `ns` field mirrors the checkpoint namespace. For a flat graph, `ns = ("",)`. For a subgraph, it is the full nested path, e.g. `("", "sub_agent", "0")`. This means a cached task result from step 1 of a subgraph is never accidentally used in step 2 or a sibling subgraph:

```python
from langgraph.types import CacheKey

# Two tasks from different namespaces — distinct CacheKeys
k1 = CacheKey(ns=("", "sub1", "0"), key="abc123", ttl=60)
k2 = CacheKey(ns=("", "sub2", "0"), key="abc123", ttl=60)
print(k1 == k2)   # False — different ns despite same key
```

---

## 10 · `_ToolCallRequestOverrides`

**Module**: `langgraph.prebuilt.tool_node`  
**First dedicated coverage as a standalone class.**

`_ToolCallRequestOverrides` is the `TypedDict` that describes the set of fields that `ToolCallRequest.override()` accepts. It has `total=False` — all three keys are optional:

```python
class _ToolCallRequestOverrides(TypedDict, total=False):
    tool_call: ToolCall   # replace the model's tool call dict (args, name, id)
    tool: BaseTool        # swap in a different tool object entirely
    state: Any            # override the state injected into the tool
```

### Using `ToolCallRequest.override()` in a `ToolNode` interceptor

The `wrap_tool_call` parameter of `ToolNode` accepts a `ToolCallWrapper` — a callable with signature `(request: ToolCallRequest, execute: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command`. The wrapper receives the request *before* execution and must call `execute(req)` (or `execute(req.override(...))`) to complete the tool call. This is the correct way to sanitise or transform tool calls before execution:

```python
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.tools import tool
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.types import Command
from typing import Callable

@tool
def safe_eval(expression: str) -> float:
    """Evaluate a simple arithmetic expression."""
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def _eval(node):
        if isinstance(node, ast.Constant): return float(node.value)
        if isinstance(node, ast.BinOp): return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"Unsupported: {node}")
    return _eval(ast.parse(expression, mode="eval").body)

def sanitise_interceptor(
    req: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Strip unsafe characters from expression before evaluation."""
    if req.tool_call["name"] == "safe_eval":
        expr = req.tool_call.get("args", {}).get("expression", "")
        clean = "".join(c for c in expr if c in "0123456789+-*/.(). ")
        if clean != expr:
            new_tc = ToolCall(
                name=req.tool_call["name"],
                args={"expression": clean},
                id=req.tool_call["id"],
            )
            return execute(req.override(tool_call=new_tc))
    return execute(req)

node = ToolNode([safe_eval], wrap_tool_call=sanitise_interceptor)
```

### Overriding the tool itself

`_ToolCallRequestOverrides.tool` lets an interceptor swap in a completely different `BaseTool` instance. This is useful for A/B testing tool implementations or routing to stubs in tests:

```python
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

def mock_interceptor(
    req: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Redirect all tool calls to a mock during tests."""
    import os
    if os.environ.get("TESTING"):
        stub = StructuredTool.from_function(
            func=lambda **kwargs: "mock_result",
            name=req.tool_call["name"],
        )
        return execute(req.override(tool=stub))
    return execute(req)
```

### Overriding the state

`_ToolCallRequestOverrides.state` allows an interceptor to inject a different state dict into `InjectedState` parameters — useful when you want to enrich state with data computed during the intercept:

```python
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

def enrich_state_interceptor(
    req: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Add runtime context to the state seen by tools."""
    import datetime
    enriched = {
        **(req.state or {}),
        "request_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_id": "req-12345",
    }
    return execute(req.override(state=enriched))
```

---

## Summary table

| Class | Module | Why it matters |
|---|---|---|
| `_DebugCheckpointPayload` | `langgraph.types` | Typed `"checkpoint"` envelope for `stream_mode="debug"` |
| `_DebugTaskPayload` | `langgraph.types` | Typed `"task"` envelope (task start) for debug stream |
| `_DebugTaskResultPayload` | `langgraph.types` | Typed `"task_result"` envelope (task finish) for debug stream |
| `DebugPayload` | `langgraph.types` | Union type alias for all three debug envelope variants |
| `_InjectedArgs` | `langgraph.prebuilt.tool_node` | Pre-computed injection map built at `ToolNode.__init__` |
| `_DirectlyInjectedToolArg` | `langgraph.prebuilt.tool_node` | Sentinel NamedTuple marking framework-injected tool params |
| `_TaskIDFn` | `langgraph.pregel._algo` | Protocol for task ID generation functions |
| `_uuid5_str` | `langgraph.pregel._algo` | SHA-1 UUID5 task ID generation (default) |
| `_xxhash_str` | `langgraph.pregel._algo` | XXH3-128 fast task ID generation |
| `task_path_str` | `langgraph.pregel._algo` | Task path tuple → sortable string for checkpoint ordering |
| `LangGraphDeprecationWarning` | `langgraph.warnings` | Base deprecation warning with `since`/`expected_removal` |
| `LangGraphDeprecatedSinceV11` | `langgraph.warnings` | New in 1.2.6 — longer window (`expected_removal=(3,0)`) |
| `DeprecatedKwargs` | `langgraph._internal._typing` | Empty TypedDict enabling static warnings on deprecated kwargs |
| `InvalidModuleError` | `langgraph.checkpoint.serde.jsonplus` | Raised when strict-msgpack blocks an unknown type |
| `_TasksLifecycleBase` | `langgraph.stream.transformers` | Shared template for subgraph lifecycle detection |
| `_DeltaSnapshot` | `langgraph.checkpoint.serde.types` | Snapshot marker that terminates the delta ancestor walk |
| `CacheKey` | `langgraph.types` | `(ns, key, ttl)` NamedTuple for cache namespace+key+TTL |
| `_ToolCallRequestOverrides` | `langgraph.prebuilt.tool_node` | TypedDict for `ToolCallRequest.override()` fields |

---
title: "LangGraph Class Deep-Dives Vol. 31"
description: "Source-verified deep dives into 10 previously undocumented class groups — _is_optional_type/_is_required_type/_is_readonly_type annotation predicates, get_update_as_tuples Pydantic partial-update filter, node Protocol variants and KWARGS_CONFIG_KEYS dispatch table, StrEnum/set_config_context/create_task_in_config_context, _create_root_model/_remap_field_definitions/create_model Pydantic internals, is_supported_by_pydantic/get_fields, map_output_values/map_output_updates output projection, PregelNode.copy()/flat_writers/node pipeline assembly, _ensure_future/_convert_future_exc async internals, and _copy_future_state/_set_concurrent_future_state cross-executor bridging — verified against langgraph==1.2.7."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 31"
  order: 62
---

Source-verified deep dives into **10 previously undocumented class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.7` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `_is_optional_type` · `_is_required_type` · `_is_readonly_type`

These three predicate functions in `langgraph._internal._fields` power how LangGraph decides whether a state key is required, optional, or write-protected at schema build time.

**Key source facts** (from `langgraph/_internal/_fields.py`):

- `_is_optional_type` handles three flavours of optional: PEP 604 `str | None` (`types.UnionType`), `Optional[X]` (origin is `Optional`), `Union[X, None]` (any arg is `None`), and `Annotated[Optional[X], ...]` — it recurses through `Annotated` wrappers.
- `_is_required_type` returns `True` if `Required[X]`, `False` if `NotRequired[X]`, and `None` if neither marker is present. It also recurses through `Annotated` and through `Union.__args__`.
- `_is_readonly_type` checks `ReadOnly` from `typing_extensions`. LangGraph notes in a code comment that it *ignores* read-only at runtime ("we don't care if you mutate the state in your node") — but the predicate is there so schema introspection tools can see it.
- All three functions are recursive: they strip `Annotated[X, ...]` by looking at `__args__[0]` before delegating.
- `get_field_default` composes these predicates: it checks `__optional_keys__` → `_is_required_type` → dataclass field defaults → `_is_optional_type`, returning `...` (required sentinel) or `None` (optional default).

**Example 1 — query annotation predicates directly**

```python
from langgraph._internal._fields import (
    _is_optional_type,
    _is_required_type,
    _is_readonly_type,
)
from typing import Optional, Union
from typing_extensions import Annotated, NotRequired, ReadOnly, Required

# Optional detection
print(_is_optional_type(Optional[str]))      # True
print(_is_optional_type(str | None))         # True (PEP 604)
print(_is_optional_type(str))                # False
print(_is_optional_type(Union[str, None]))   # True

# Required/NotRequired detection
print(_is_required_type(Required[str]))      # True
print(_is_required_type(NotRequired[str]))   # False
print(_is_required_type(str))               # None

# ReadOnly detection
print(_is_readonly_type(ReadOnly[str]))      # True
print(_is_readonly_type(Annotated[ReadOnly[str], "doc"]))  # True
print(_is_readonly_type(str))               # False
```

**Example 2 — trace how get_field_default uses these predicates**

```python
from typing_extensions import TypedDict, NotRequired, Required
from langgraph._internal._fields import get_field_default

class MyState(TypedDict, total=True):
    required_field: str
    optional_field: NotRequired[str]
    nullable_field: str | None

# required_field: Required by total=True, no Optional → returns ... (required sentinel)
print(get_field_default("required_field", str, MyState))              # Ellipsis

# optional_field: NotRequired → returns None (optional default)
print(get_field_default("optional_field", NotRequired[str], MyState)) # None

# nullable_field: _is_optional_type(str | None) = True → returns None
print(get_field_default("nullable_field", str | None, MyState))       # None
```

**Example 3 — annotated wrappers are transparently unwrapped**

```python
from typing import Optional
from typing_extensions import Annotated, NotRequired, ReadOnly
from langgraph._internal._fields import (
    _is_optional_type,
    _is_required_type,
    _is_readonly_type,
)

# All three handle multi-level Annotated wrappers
deep = Annotated[Annotated[Optional[str], "meta1"], "meta2"]
print(_is_optional_type(deep))   # True

ro = Annotated[ReadOnly[Annotated[str, "doc"]], "more"]
print(_is_readonly_type(ro))     # True

nr = Annotated[NotRequired[str], "hint"]
print(_is_required_type(nr))     # False
```

---

## 2 · `get_update_as_tuples`

`get_update_as_tuples` in `langgraph._internal._fields` converts a state update object into a list of `(key, value)` pairs consumed by the channel write subsystem. The inclusion rule differs between Pydantic and non-Pydantic inputs.

**Key source facts** (from `langgraph/_internal/_fields.py`):

- When `input` is a `BaseModel`, it reads `model_fields_set` and the `model_fields` defaults.
- A field is **included** when `value is not None` **OR** the field's default is not `None` **OR** the field is in `model_fields_set`. Non-None values always pass through — there is **no** `model_fields_set` filter on them.
- A field is **excluded** only in one case: `value is None`, the field default is also `None`, and the field was **not** explicitly assigned. This backwards-compat rule prevents un-set optional fields from wiping existing state.
- For non-Pydantic inputs (dataclasses), `keep=None` and `defaults={}`. Because `defaults.get(k, MISSING) is not None` evaluates to `True` for any key not in the empty dict, all non-MISSING `getattr` values pass through — including `None`.
- `TypedDict` annotations create plain `dict` objects at runtime; `getattr(dict, key, MISSING)` always returns `MISSING`, so passing a TypedDict instance returns `[]`. Use a dataclass for non-Pydantic examples.
- The function returns a `list[tuple[str, Any]]` consumed by the channel write subsystem.

**Example 1 — non-None values always pass through; None with None-default is excluded**

```python
from pydantic import BaseModel
from langgraph._internal._fields import get_update_as_tuples

class AgentOutput(BaseModel):
    answer: str = ""
    confidence: float | None = None   # default is None
    citations: list[str] = []         # default is non-None ([])

# Only 'answer' was explicitly set; 'citations' has a non-None default value
output = AgentOutput(answer="Paris")
print(output.model_fields_set)  # {'answer'}
tuples = get_update_as_tuples(output, ["answer", "confidence", "citations"])
print(tuples)  # [('answer', 'Paris'), ('citations', [])]
# 'confidence' is excluded: value is None, default is None, not in model_fields_set
# 'citations' passes through: its value [] is non-None (even though not explicitly set)
```

**Example 2 — explicit None clears a field that had a non-None default**

```python
from pydantic import BaseModel
from langgraph._internal._fields import get_update_as_tuples

class State(BaseModel):
    error: str | None = "initial"

# Explicitly set error=None — LangGraph will write None to clear the field
s = State(error=None)
print(s.model_fields_set)  # {'error'}
tuples = get_update_as_tuples(s, ["error"])
print(tuples)  # [('error', None)]
```

**Example 3 — dataclass passes all keys including None (keep=None path)**

```python
from dataclasses import dataclass
from langgraph._internal._fields import get_update_as_tuples

@dataclass
class SimpleState:
    x: int
    y: int
    z: str | None = None

# dataclass — not a BaseModel → keep=None; defaults={} so all getattr values pass through
d = SimpleState(x=1, y=0, z=None)
tuples = get_update_as_tuples(d, ["x", "y", "z"])
print(tuples)  # [('x', 1), ('y', 0), ('z', None)]
# Note: TypedDict creates plain dicts at runtime — getattr(dict, key) always returns
# MISSING, so passing a TypedDict instance would return []. Use dataclass instead.
```

---

## 3 · Node Protocol variants · `KWARGS_CONFIG_KEYS` dispatch

LangGraph accepts six flavours of node callable (beyond plain functions) defined as `Protocol` types in `langgraph._internal._runnable`, all collected into the `RunnableLike` union. At `__init__`, `RunnableCallable` inspects the function signature and builds `func_accepts` — a dict mapping kwarg names to `(runtime_key, default)` pairs.

**Key source facts** (from `langgraph/_internal/_runnable.py`):

- `KWARGS_CONFIG_KEYS` is a `tuple[tuple[str, tuple[Any, ...], str, Any], ...]`: each entry is `(kwarg_name, accepted_types, runtime_key, default)`.
- Supported kwarg names are `"config"`, `"writer"`, `"store"`, `"previous"`, `"runtime"`, `"error"`.
- `config` must be typed as `RunnableConfig` or `RunnableConfig | None`; any other annotation emits a `UserWarning`.
- `store` accepts both `BaseStore` (required) and `Optional[BaseStore]` (with a `None` default); the two entries in `KWARGS_CONFIG_KEYS` differ only in accepted types and default.
- `previous` and `runtime` accept `ANY_TYPE` — any annotation matches, no type restriction.
- At invoke time, `func_accepts` is iterated and values are read from `runtime.{runtime_key}` (e.g. `runtime.stream_writer` for `writer`, `runtime.store` for `store`).
- `_RunnableWithConfigStore`, `_RunnableWithConfigWriter`, `_RunnableWithConfigWriterStore` are `Protocol` types, not base classes — you never subclass them; just match the signature.

**Example 1 — node that auto-receives config and writer**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    value: int

def node_with_config(state: State, *, config: RunnableConfig, writer: StreamWriter) -> dict:
    # LangGraph auto-injects config and writer because the parameters match KWARGS_CONFIG_KEYS
    writer({"progress": "halfway"})
    run_name = config.get("run_name", "unknown")
    return {"value": state["value"] + 1}

builder = StateGraph(State)
builder.add_node("step", node_with_config)
builder.add_edge(START, "step")
builder.add_edge("step", END)
graph = builder.compile()

for chunk in graph.stream({"value": 0}, stream_mode=["updates", "custom"]):
    print(chunk)
```

**Example 2 — inspect func_accepts to see which kwargs were detected**

```python
from langgraph._internal._runnable import RunnableCallable
from langgraph.types import StreamWriter
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

def my_node(state: dict, *, config: RunnableConfig, store: BaseStore | None = None) -> dict:
    return state

rc = RunnableCallable(my_node)
# func_accepts tells you exactly which kwargs will be injected at runtime
print(rc.func_accepts)
# {'config': ('N/A', <Parameter.empty>), 'store': ('store', None)}
```

**Example 3 — Optional[BaseStore] gets None default while BaseStore raises if missing**

```python
from typing import Optional
from langgraph._internal._runnable import RunnableCallable
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

def node_required_store(state: dict, *, store: BaseStore) -> dict:
    return state

def node_optional_store(state: dict, *, store: Optional[BaseStore] = None) -> dict:
    return state

rc_req = RunnableCallable(node_required_store)
rc_opt = RunnableCallable(node_optional_store)

# Required store: default is Parameter.empty → raises ValueError if store absent
print(rc_req.func_accepts["store"])  # ('store', <Parameter.empty>)

# Optional store: default is None → gracefully skips injection if store absent
print(rc_opt.func_accepts["store"])  # ('store', None)
```

---

## 4 · `StrEnum` · `set_config_context` · `create_task_in_config_context`

Three utilities in `langgraph._internal._runnable` that handle Python version compatibility and asyncio config propagation.

**Key source facts** (from `langgraph/_internal/_runnable.py`):

- `StrEnum` is defined as `class StrEnum(str, enum.Enum)` — a pre-Python 3.11 backport of `enum.StrEnum`. LangGraph uses it internally for string-valued enums (e.g. node type tags) without requiring ≥ 3.11.
- `set_config_context` is a `@contextmanager` that calls `copy_context()` to snapshot the current `contextvars.Context`, then runs `_set_config_context` inside that snapshot to set `var_child_runnable_config`. The context manager yields the copied context so callers can `ctx.run(func)`. The token is reset in `finally`.
- `create_task_in_config_context` wraps `asyncio.create_task` inside a `set_config_context` block, so the spawned task inherits the config ContextVar from its parent. Without this, `asyncio.create_task` snapshots the current context — which may already be correct, but the explicit `context.run` guarantees it even if the caller mutates the context after task creation.
- Both functions accept an optional `run` argument for LangSmith tracing context propagation (via `_set_tracing_context`).

**Example 1 — StrEnum as a string-valued enum**

```python
from langgraph._internal._runnable import StrEnum

class NodeKind(StrEnum):
    TOOL = "tool"
    AGENT = "agent"
    ROUTER = "router"

# Values compare equal to plain strings
print(NodeKind.TOOL == "tool")       # True
print(f"kind={NodeKind.AGENT}")      # kind=agent
print(isinstance(NodeKind.ROUTER, str))  # True

# Still an enum — supports membership tests
print(NodeKind("tool"))              # NodeKind.TOOL
```

**Example 2 — set_config_context propagates config to a synchronous nested call**

```python
import asyncio
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._runnable import set_config_context

config: RunnableConfig = {"run_name": "my-run", "configurable": {"thread_id": "t1"}}

def sync_reader():
    # ctx.run() works with synchronous callables: the body executes entirely inside ctx
    current = var_child_runnable_config.get()
    return current.get("run_name") if current else None

async def main():
    with set_config_context(config) as ctx:
        # ctx.run(sync_fn) runs the function body inside the copied context
        result = ctx.run(sync_reader)
    return result

print(asyncio.run(main()))  # my-run
# Note: ctx.run(async_fn) only creates the coroutine object inside ctx — the async
# body executes later when awaited, in the caller's context, NOT in ctx.
# For async propagation use create_task_in_config_context (see Example 3).
```

**Example 3 — create_task_in_config_context propagates config to an asyncio task**

```python
import asyncio
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._runnable import create_task_in_config_context

config: RunnableConfig = {"run_name": "child-task", "configurable": {"thread_id": "t2"}}

async def child_task():
    cfg = var_child_runnable_config.get()
    return cfg.get("run_name") if cfg else "no config"

async def main():
    task = create_task_in_config_context(child_task, config)
    result = await task
    print(result)  # child-task

asyncio.run(main())
```

---

## 5 · `_create_root_model` · `_remap_field_definitions` · `create_model`

These three functions in `langgraph._internal._pydantic` are how LangGraph converts type annotations and TypedDicts into Pydantic models for JSON schema generation and validation.

**Key source facts** (from `langgraph/_internal/_pydantic.py`):

- `_create_root_model(name, type_, ...)` creates a Pydantic `RootModel` subclass dynamically; it overrides both `schema()` (v1 compat) and `model_json_schema()` (v2) to inject the caller-supplied `name` as the schema `"title"`. If `default_` is provided, it's set as the root field default.
- `_create_root_model_cached` is an `@lru_cache(maxsize=256)` wrapper — the same `(name, type_)` pair always returns the same class object, which matters for Pydantic forward-reference resolution.
- `_remap_field_definitions` remaps field definitions whose names start with `_` or match a Pydantic-reserved name (e.g. `model_dump`, `dict`, `schema`, `validate`). It prefixes the attribute as `private_{name}` and adds `alias=name` + `serialization_alias=name` so wire format is unchanged. Passing a `FieldInfo` for a reserved name raises `NotImplementedError`.
- `_RESERVED_NAMES` is built once as `{key for key in dir(BaseModel) if not key.startswith("_")}` — it stays up-to-date as Pydantic adds new methods.
- `create_model(model_name, *, field_definitions=None, root=None)` is the top-level factory: `root=` routes to `_create_root_model_cached`; `field_definitions=` routes to `_create_model_cached` (which calls `_remap_field_definitions` first).

**Example 1 — create_model with root type (used for typed I/O schemas)**

```python
from langgraph._internal._pydantic import create_model

# LangGraph uses this to wrap typed output schemas
StringModel = create_model("MyString", root=str)
IntModel = create_model("Count", root=(int, 0))  # (type, default)

m = StringModel("hello")  # type: ignore
print(m.root)           # hello
print(StringModel.model_json_schema()["title"])  # MyString

m2 = IntModel()
print(m2.root)          # 0  (default applied)
```

**Example 2 — _remap_field_definitions escapes reserved Pydantic names**

```python
from langgraph._internal._pydantic import _remap_field_definitions, create_model

# 'dict' and 'schema' are reserved Pydantic method names
field_defs = {
    "name": (str, "Alice"),
    "dict": (str, "some dict field"),  # conflicts with BaseModel.dict()
}
remapped = _remap_field_definitions(field_defs)
print(remapped)
# {'name': ('str', 'Alice'), 'private_dict': (str, FieldInfo(alias='dict', ...))}

# Create model and verify alias round-trip
M = create_model("M", field_definitions={"name": (str, ""), "dict": (str, "data")})
instance = M(name="test", **{"dict": "payload"})
print(instance.model_dump(by_alias=True))  # {'name': 'test', 'dict': 'payload'}
```

**Example 3 — LRU cache ensures identical class objects for the same signature**

```python
from langgraph._internal._pydantic import (
    _create_root_model_cached,
    _create_model_cached,
)

# Same (name, type) always returns the same class — important for Pydantic
A = _create_root_model_cached("Score", float)
B = _create_root_model_cached("Score", float)
print(A is B)  # True

# Field-definition model is also cached
C = _create_model_cached("Point", x=(float, 0.0), y=(float, 0.0))
D = _create_model_cached("Point", x=(float, 0.0), y=(float, 0.0))
print(C is D)  # True
print(C(x=1.0, y=2.0).model_dump())  # {'x': 1.0, 'y': 2.0}
```

---

## 6 · `is_supported_by_pydantic` · `get_fields`

Two utility functions in `langgraph._internal._pydantic` that deal with Pydantic compatibility checks and cross-version field access.

**Key source facts** (from `langgraph/_internal/_pydantic.py`):

- `is_supported_by_pydantic(type_)` returns `True` for dataclasses (`is_dataclass`), Pydantic `BaseModel` subclasses, and `TypedDict` subclasses. For `TypedDict`, it checks `__orig_bases__` for `TypedDict` from `typing_extensions` (always) or from `typing` (only Python ≥ 3.12, since Pydantic supports `typing.TypedDict` only from 3.12 onwards).
- The function returns `False` for primitives (`int`, `str`, `bool`) — the intent is to detect *container* types that benefit from Pydantic schema generation.
- `get_fields(model)` is a shim supporting both Pydantic v2 (`.model_fields` dict of `FieldInfo`) and the legacy Pydantic v1 API (`.___fields__`). It raises `TypeError` if neither attribute is present.
- LangGraph calls `is_supported_by_pydantic` during state schema validation to decide whether to generate a JSON schema for the state type — primitive types skip schema generation.

**Example 1 — is_supported_by_pydantic across type categories**

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph._internal._pydantic import is_supported_by_pydantic

@dataclass
class DC:
    x: int

class PModel(BaseModel):
    x: int

class TD(TypedDict):
    x: int

# Supported container types
print(is_supported_by_pydantic(DC))      # True
print(is_supported_by_pydantic(PModel))  # True
print(is_supported_by_pydantic(TD))      # True

# Primitives are NOT supported (schema generation is skipped)
print(is_supported_by_pydantic(int))     # False
print(is_supported_by_pydantic(str))     # False
print(is_supported_by_pydantic(list))    # False
```

**Example 2 — get_fields works with both Pydantic v1 and v2**

```python
from pydantic import BaseModel, Field
from langgraph._internal._pydantic import get_fields

class Config(BaseModel):
    host: str = "localhost"
    port: int = Field(8080, description="Server port")
    debug: bool = False

fields = get_fields(Config)
print(list(fields.keys()))           # ['host', 'port', 'debug']
print(fields["port"].description)    # Server port
print(fields["host"].default)        # localhost

# Also works on instances (not just classes)
cfg = Config()
instance_fields = get_fields(cfg)
print(list(instance_fields.keys()))  # ['host', 'port', 'debug']
```

**Example 3 — using is_supported_by_pydantic before schema generation**

```python
from typing import Any
from typing_extensions import TypedDict
from langgraph._internal._pydantic import is_supported_by_pydantic, create_model

def maybe_build_schema(state_type: type) -> dict[str, Any] | None:
    """Only generate a JSON schema for container types."""
    if not is_supported_by_pydantic(state_type):
        return None
    # Wrap the state type in a RootModel and extract the schema
    RootM = create_model("StateSchema", root=state_type)
    return RootM.model_json_schema()

class AgentState(TypedDict):
    messages: list[str]
    step: int

schema = maybe_build_schema(AgentState)
print(schema is not None)     # True
print(maybe_build_schema(int))  # None — primitives skip schema generation
```

---

## 7 · `map_output_values` · `map_output_updates`

These two functions in `langgraph.pregel._io` are the output projection layer — they translate internal pending-writes lists into the chunks emitted to callers using `stream_mode="values"` and `stream_mode="updates"` respectively.

**Key source facts** (from `langgraph/pregel/_io.py`):

- `map_output_values(output_channels, pending_writes, channels)` yields a single snapshot for the whole graph step. If `output_channels` is a string: yields `read_channel(channels, output_channels)` whenever that channel appears in `pending_writes` (or when `pending_writes is True` — the "emit unconditionally" sentinel). If a list: yields the dict of all output channels whenever any of them appears in pending writes.
- `map_output_updates(output_channels, tasks, cached=False)` groups by task. It first filters out tasks tagged with `TAG_HIDDEN` and tasks whose first write is to `ERROR` or `INTERRUPT` channels. Then, per task: if a `RETURN` sentinel is in the writes, that value is used instead of the channel write. Multi-write tasks (same channel written twice) are split into individual `{channel: value}` dicts.
- The `cached=False` parameter is reserved for future use to distinguish live vs cached task outputs.
- `map_output_updates` yields `dict[str, Any | dict[str, Any]]` — the outer key is the task (node) name; the inner value is either the channel value (single-channel output), a `{channel: value}` dict (multi-channel single-write), or a **list of `{channel: value}` dicts** when the same channel is written more than once in one step (e.g. `{"node": [{"x": 1}, {"x": 2}]}`). One-element lists are unwrapped to a single value; empty entries become `None`.

**Example 1 — stream_mode="values" vs stream_mode="updates" output shape**

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    x: int
    y: int

def add_x(state: State) -> dict:
    return {"x": state["x"] + 10}

def add_y(state: State) -> dict:
    return {"y": state["y"] + 20}

builder = StateGraph(State)
builder.add_node("add_x", add_x)
builder.add_node("add_y", add_y)
builder.add_edge(START, "add_x")
builder.add_edge("add_x", "add_y")
builder.add_edge("add_y", END)
graph = builder.compile()

print("=== values mode ===")
for chunk in graph.stream({"x": 0, "y": 0}, stream_mode="values"):
    print(chunk)  # full state snapshot after each step

print("=== updates mode ===")
for chunk in graph.stream({"x": 0, "y": 0}, stream_mode="updates"):
    print(chunk)  # per-node delta: {"add_x": {"x": 10}} then {"add_y": {"y": 20}}
```

**Example 2 — TAG_HIDDEN suppresses a node from updates output**

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import TAG_HIDDEN

class State(TypedDict):
    value: int
    internal: int

def visible_node(state: State) -> dict:
    return {"value": state["value"] + 1, "internal": 99}

def hidden_node(state: State) -> dict:
    return {"internal": 0}

builder = StateGraph(State)
builder.add_node("visible", visible_node)
# TAG_HIDDEN prevents this node from appearing in stream_mode="updates" output
builder.add_node("hidden", hidden_node, tags=[TAG_HIDDEN])
builder.add_edge(START, "visible")
builder.add_edge("visible", "hidden")
builder.add_edge("hidden", END)
graph = builder.compile()

for chunk in graph.stream({"value": 0, "internal": 0}, stream_mode="updates"):
    print(chunk)  # only {"visible": {...}} — "hidden" is suppressed
```

**Example 3 — RETURN sentinel lets a task return a value outside output_channels**

```python
# map_output_updates uses the RETURN sentinel to collect values that
# nodes return via Command(update={...}) with a top-level return value.
# Demonstrated using the functional API where @task returns a direct value:
import asyncio
from typing_extensions import TypedDict
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

class Report(TypedDict):
    title: str
    body: str

@task
async def draft_report(title: str) -> Report:
    return {"title": title, "body": f"Draft of {title}"}

@entrypoint(checkpointer=InMemorySaver())
async def pipeline(topic: str) -> Report:
    result = await draft_report(topic)
    return result

async def main():
    config = {"configurable": {"thread_id": "r1"}}
    async for chunk in pipeline.astream("AI trends", config=config, stream_mode="updates"):
        print(chunk)

asyncio.run(main())
```

---

## 8 · `PregelNode.copy()` · `PregelNode.flat_writers` · `PregelNode.node`

`PregelNode` in `langgraph.pregel._read` is the internal container that `add_node()` ultimately creates. Three cached properties and `copy()` control how the node pipeline is assembled at compile time.

**Key source facts** (from `langgraph/pregel/_read.py`):

- `PregelNode.copy(update: dict)` creates a shallow copy, merging `update` into `__dict__`. Critically, it pops `"flat_writers"`, `"node"`, and `"input_cache_key"` from `attrs` before calling `__init__` — this clears the `@cached_property` values so the new node recomputes them from the updated config.
- `flat_writers` (`@cached_property`) deduplicates consecutive `ChannelWrite` instances at the tail of the writers list. It merges them by combining their `writes` lists, reducing the number of channel-write Runnable invocations in the pipeline.
- `node` (`@cached_property`) assembles the full pipeline as `RunnableSeq(bound, *writers)`. Edge cases: if `bound is DEFAULT_BOUND` (the identity lambda) and there are no writers, `node` returns `None`; if there is exactly one writer and `bound` is the default, it returns the writer directly (no seq overhead).
- `input_cache_key` (`@cached_property`) is a `(mapper, tuple[str, ...])` tuple used to avoid reading the same channels multiple times for nodes that share a channel subscription.
- `invoke` / `ainvoke` pass `merge_configs({"metadata": self.metadata, "tags": self.tags}, config)` so node-level tracing metadata is always injected.

**Example 1 — inspect PregelNode after compilation**

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import PregelNode

class State(TypedDict):
    count: int

def my_node(state: State) -> dict:
    return {"count": state["count"] + 1}

builder = StateGraph(State)
builder.add_node("counter", my_node)
builder.add_edge(START, "counter")
builder.add_edge("counter", END)
graph = builder.compile()

# After compilation, inspect the PregelNode
pn: PregelNode = graph.nodes["counter"]
print("channels:", pn.channels)       # ['count'] or '__root__' depending on schema
print("triggers:", pn.triggers)       # ['branch:to:counter']  ← branch-scheduling channel, not the node name
print("writers:", pn.writers)         # [ChannelWrite(...)]
print("node type:", type(pn.node).__name__)   # RunnableSeq
```

**Example 2 — flat_writers deduplicates consecutive ChannelWrites**

```python
from langgraph.pregel._read import PregelNode, DEFAULT_BOUND
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry

# Simulate two consecutive ChannelWrites
w1 = ChannelWrite(writes=[ChannelWriteEntry(channel="a", value=None, skip_none=False)])
w2 = ChannelWrite(writes=[ChannelWriteEntry(channel="b", value=None, skip_none=False)])

pn = PregelNode(
    channels=["a"],
    triggers=["a"],
    writers=[w1, w2],  # two consecutive writes
)

# flat_writers merges consecutive ChannelWrites into one
flat = pn.flat_writers
print(len(flat))                 # 1  (merged)
print(len(flat[0].writes))       # 2  (both entries combined)
```

**Example 3 — copy() clears cached properties on modified nodes**

```python
from langgraph.pregel._read import PregelNode, DEFAULT_BOUND
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.types import RetryPolicy

pn = PregelNode(channels=["state"], triggers=["state"], writers=[])

# Access cached property to populate the cache
_ = pn.node  # now cached

# copy() with a retry policy drops the cached node so it recomputes
retry = RetryPolicy(max_attempts=3)
pn2 = pn.copy({"retry_policy": [retry]})

print(pn2.retry_policy)              # [RetryPolicy(max_attempts=3)]
print("node" not in pn2.__dict__)    # True — cleared, will recompute on next access
```

---

## 9 · `_ensure_future` · `_convert_future_exc`

`_ensure_future` and `_convert_future_exc` in `langgraph._internal._future` are the building blocks for scheduling coroutines across Python versions and converting between `concurrent.futures` and `asyncio` exception types.

**Key source facts** (from `langgraph/_internal/_future.py`):

- Two module-level constants gate behaviour: `CONTEXT_NOT_SUPPORTED = sys.version_info < (3, 11)` (no `context=` kwarg on `create_task`) and `EAGER_NOT_SUPPORTED = sys.version_info < (3, 12)` (no `asyncio.eager_task_factory`).
- `_ensure_future(coro_or_future, *, loop, name=None, context=None, lazy=True)`: on < 3.11 calls `loop.create_task(coro, name=name)` (no context); on ≥ 3.11 passes `context=context`; on ≥ 3.12 with `lazy=False` uses `asyncio.eager_task_factory` which starts the coroutine synchronously until the first `await`, avoiding a scheduler round-trip.
- Non-coroutine awaitables are wrapped via `_wrap_awaitable` (a `@types.coroutine` generator) so `create_task` accepts them. If wrapping fails, the original coroutine is `.close()`d to prevent a RuntimeWarning.
- `_convert_future_exc` is a type-only translation: `concurrent.futures.CancelledError → asyncio.CancelledError`, `TimeoutError → asyncio.TimeoutError`, `InvalidStateError → asyncio.InvalidStateError`, everything else passes through unchanged.

**Example 1 — _ensure_future schedules a coroutine on the current loop**

```python
import asyncio
from langgraph._internal._future import _ensure_future

async def compute(n: int) -> int:
    await asyncio.sleep(0)
    return n * 2

async def main():
    loop = asyncio.get_running_loop()
    task = _ensure_future(compute(21), loop=loop, name="compute-21", lazy=True)
    result = await task
    print(result)  # 42

asyncio.run(main())
```

**Example 2 — _convert_future_exc maps concurrent.futures exceptions to asyncio**

```python
import concurrent.futures
from langgraph._internal._future import _convert_future_exc

# CancelledError from a thread pool becomes asyncio.CancelledError
exc = concurrent.futures.CancelledError()
converted = _convert_future_exc(exc)
print(type(converted).__name__)   # CancelledError (asyncio version)

# TimeoutError becomes asyncio.TimeoutError
exc2 = concurrent.futures.TimeoutError()
converted2 = _convert_future_exc(exc2)
print(type(converted2).__name__)  # TimeoutError (asyncio version)

# Other exceptions pass through unchanged
exc3 = ValueError("custom error")
print(_convert_future_exc(exc3) is exc3)  # True
```

**Example 3 — eager_task_factory path (Python ≥ 3.12) avoids scheduler round-trip**

```python
import asyncio
import sys
from langgraph._internal._future import _ensure_future, EAGER_NOT_SUPPORTED

async def sync_until_io() -> int:
    # No await before return — runs entirely synchronously with eager_task_factory
    return 99

async def main():
    loop = asyncio.get_running_loop()

    if EAGER_NOT_SUPPORTED:
        print("Eager tasks not available on Python < 3.12; using lazy path")
        task = _ensure_future(sync_until_io(), loop=loop, lazy=True)
    else:
        # lazy=False: starts the coroutine synchronously on Python 3.12+
        task = _ensure_future(sync_until_io(), loop=loop, lazy=False)
        # For a coroutine with no await, the task may already be done at this point
        print("Already done:", task.done())

    result = await task
    print(result)  # 99

asyncio.run(main())
```

---

## 10 · `_copy_future_state` · `_set_concurrent_future_state`

The final two bridge utilities in `langgraph._internal._future` handle bidirectional result propagation between `asyncio.Future` and `concurrent.futures.Future` — used when Pregel runs coroutines on a background thread's event loop.

**Key source facts** (from `langgraph/_internal/_future.py`):

- `_copy_future_state(source, dest)` is the `AnyFuture → asyncio.Future` bridge. It is a no-op if `dest` is already done (`if dest.done(): return`). If `source` is cancelled, it calls `dest.cancel()`. Otherwise it reads `source.exception()`, applies `_convert_future_exc` if the exception came from a `concurrent.futures.Future`, and calls `dest.set_exception` or `dest.set_result`.
- `_set_concurrent_future_state(concurrent_future, source)` is the `AnyFuture → concurrent.futures.Future` bridge. It calls `concurrent.cancel()` if `source` is cancelled and then `set_running_or_notify_cancel()` — if that returns `False` the future was already cancelled and the function returns early. It also applies `_convert_future_exc` on the exception path.
- Both functions are called by `_chain_future`'s internal callbacks (`_call_set_state`, `_call_check_cancel`) — neither is typically called directly by user code.
- `_copy_future_state` is called from the *destination* loop via `dest_loop.call_soon_threadsafe(...)`, ensuring thread-safety even when source and destination are on different loops.

**Example 1 — _copy_future_state bridges a completed concurrent.futures.Future**

```python
import asyncio
import concurrent.futures
from langgraph._internal._future import _copy_future_state

async def main():
    loop = asyncio.get_running_loop()

    # A resolved thread-pool future
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        cf = pool.submit(lambda: 42)
        cf.result()  # Wait for it

        # Bridge to an asyncio.Future
        af = loop.create_future()
        _copy_future_state(cf, af)
        print(af.result())  # 42

asyncio.run(main())
```

**Example 2 — _copy_future_state is a no-op if destination is already done**

```python
import asyncio
import concurrent.futures
from langgraph._internal._future import _copy_future_state

async def main():
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        cf = pool.submit(lambda: 99)
        cf.result()

        af = loop.create_future()
        af.set_result(1)  # Already resolved

        # _copy_future_state is a no-op when dest is done
        _copy_future_state(cf, af)
        print(af.result())  # 1 — unchanged

asyncio.run(main())
```

**Example 3 — cancellation propagates from source to destination**

```python
import asyncio
import concurrent.futures
from langgraph._internal._future import _copy_future_state

async def main():
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        cf = pool.submit(lambda: 0)
        cf.result()

        # Manually cancel the source after it ran — simulate "already cancelled" state
        cancelled_cf = concurrent.futures.Future()
        cancelled_cf.cancel()  # sets the cancelled state

        af = loop.create_future()
        _copy_future_state(cancelled_cf, af)
        print(af.cancelled())  # True — cancellation bridged across executor boundary

asyncio.run(main())
```

---

## Summary

| # | Class / Symbol | Module | Key insight |
|---|---------------|--------|-------------|
| 1 | `_is_optional_type` · `_is_required_type` · `_is_readonly_type` | `langgraph._internal._fields` | TypedDict annotation predicates; all recurse through `Annotated` wrappers |
| 2 | `get_update_as_tuples` | `langgraph._internal._fields` | Pydantic state-update projection; non-`None` values always pass through; `None` excluded only when default is also `None` and field not explicitly set |
| 3 | `_RunnableWithConfigStore` · `KWARGS_CONFIG_KEYS` | `langgraph._internal._runnable` | Six Protocol variants + dispatch table auto-injecting `config`/`writer`/`store`/`runtime` into node functions |
| 4 | `StrEnum` · `set_config_context` · `create_task_in_config_context` | `langgraph._internal._runnable` | Pre-3.11 enum backport; config ContextVar propagation into copied contexts and asyncio tasks |
| 5 | `_create_root_model` · `_remap_field_definitions` · `create_model` | `langgraph._internal._pydantic` | RootModel factory; reserved-name escaping with `private_` prefix + alias; LRU-cached model creation |
| 6 | `is_supported_by_pydantic` · `get_fields` | `langgraph._internal._pydantic` | Container type detection; v1/v2 Pydantic field access shim |
| 7 | `map_output_values` · `map_output_updates` | `langgraph.pregel._io` | Output projection layer: full-snapshot for `"values"` vs per-task deltas for `"updates"`; TAG_HIDDEN suppression; RETURN sentinel |
| 8 | `PregelNode.copy()` · `flat_writers` · `node` | `langgraph.pregel._read` | Node container assembly: `copy()` clears cached properties; `flat_writers` merges consecutive writes; `node` builds final `RunnableSeq` pipeline |
| 9 | `_ensure_future` · `_convert_future_exc` | `langgraph._internal._future` | Python version-gated task factory (lazy vs eager, context vs no-context); `concurrent.futures` → `asyncio` exception mapping |
| 10 | `_copy_future_state` · `_set_concurrent_future_state` | `langgraph._internal._future` | Cross-executor result bridging; no-op guard on already-done destination; cancellation propagation |

**Cross-references:** The dispatch table in group 3 interacts with `RunnableCallable` (Vol. 23 §1) and the runtime injection system in `ToolRuntime` (Vol. 2 §9). The output projection layer in group 7 feeds the stream transformers covered in Vol. 16 and Vol. 22. `PregelNode` in group 8 is the compiled form of `StateNodeSpec` (Vol. 25 §2). The future bridging in groups 9–10 underlies the cross-loop scheduling in `BackgroundExecutor` (Vol. 7 §2) and `run_coroutine_threadsafe` (Vol. 8 §9).

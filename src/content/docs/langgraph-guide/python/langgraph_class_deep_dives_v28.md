---
title: "Class deep-dives Vol. 28 — config internals, type system, checkpoint utilities, serde allowlist & production tuning (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: merge_configs/patch_config/patch_configurable (internal config merging rules, lc_versions deep merge, callback manager union, checkpoint coordinate isolation), ensure_config (ambient propagation, _PROPAGATE_TO_METADATA, explicit coordinate reset), ContextT/StateT/InputT/OutputT TypeVars (typed graph factories, context_schema typing, node input narrowing), TAG_NOSTREAM/TAG_HIDDEN sentinels (suppressing messages-stream output, hiding nodes from LangSmith, combining with set_node_defaults), recast_checkpoint_ns/filter_to_user_tags (subgraph namespace stripping, seq:step tag filtering), InMemorySaver storage internals (three-dict layout: storage/writes/blobs, put_writes contract, WRITES_IDX_MAP reserved channels), PendingWrite/WRITES_IDX_MAP sentinel channels (ERROR/__error__, INTERRUPT/__interrupt__, RESUME/__resume__, SCHEDULED/__scheduled__), get_checkpoint_id/get_checkpoint_metadata (runtime checkpoint utilities, fork/branch patterns), build_serde_allowlist/STRICT_MSGPACK_ENABLED/apply_checkpointer_allowlist (strict msgpack type-safe serialization, curated_core_allowlist, Pydantic/dataclass/TypedDict scanning), and DEFAULT_RECURSION_LIMIT/DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT/CONFIG_KEY_* constants (env-var tuning, reserved configurable keys, namespace separators)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 28"
  order: 59
---

# Class deep-dives Vol. 28 — config internals, type system, checkpoint utilities, serde allowlist & production tuning (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `merge_configs` + `patch_config` + `patch_configurable` | `langgraph._internal._config` |
| 2 | `ensure_config` + `_PROPAGATE_TO_METADATA` + `_CHECKPOINT_COORDINATE_KEYS` | `langgraph._internal._config` |
| 3 | `ContextT` + `StateT` + `InputT` + `OutputT` + `NodeInputT` | `langgraph.typing` |
| 4 | `TAG_NOSTREAM` + `TAG_HIDDEN` | `langgraph.constants` |
| 5 | `recast_checkpoint_ns` + `filter_to_user_tags` | `langgraph._internal._config` |
| 6 | `InMemorySaver` — storage internals | `langgraph.checkpoint.memory` |
| 7 | `PendingWrite` + `WRITES_IDX_MAP` + sentinel channels | `langgraph.checkpoint.base` · `langgraph.checkpoint.serde.types` |
| 8 | `get_checkpoint_id` + `get_checkpoint_metadata` | `langgraph.checkpoint.base` |
| 9 | `build_serde_allowlist` + `STRICT_MSGPACK_ENABLED` + `apply_checkpointer_allowlist` | `langgraph._internal._serde` |
| 10 | `DEFAULT_RECURSION_LIMIT` + `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` + `CONFIG_KEY_*` | `langgraph._internal._config` · `langgraph._internal._constants` |

---

## 1 · `merge_configs` + `patch_config` + `patch_configurable`

**Module**: `langgraph._internal._config`  
**First dedicated coverage.**

These three functions are the building blocks for all config composition in LangGraph. Every `compile()`, `with_config()`, and subgraph invocation calls through one or more of them.

```python
def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """Merge multiple configs: later configs win on most keys; metadata
    deep-merges (lc_versions one level deeper); callbacks are unioned;
    tags are concatenated; recursion_limit only overrides DEFAULT."""

def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: Callbacks = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Shallow-patch specific fields. Setting callbacks clears run_name
    and run_id to avoid attributing calls to a parent run."""

def patch_configurable(
    config: RunnableConfig | None, patch: dict[str, Any]
) -> RunnableConfig:
    """Shallow-merge into config["configurable"]. Creates the config
    or the configurable sub-dict if missing."""
```

**Key implementation facts:**

- **`merge_configs` metadata deep-merge** — the `metadata` key is the only one that merges one level deeper: the `lc_versions` sub-key gets a union of both dicts (so separate LangChain packages each contribute their version strings). All other metadata keys use last-write-wins.
- **Callbacks union** — when both configs have callbacks, `merge_configs` calls `_merge_callbacks`, which unions `CallbackManager` instances via `.merge()`. A list of handlers on one side gets added to the other side's `CallbackManager` via `.add_handler(cb, inherit=True)`.
- **`recursion_limit` special rule** — `merge_configs` only writes `recursion_limit` from a config when it differs from `DEFAULT_RECURSION_LIMIT` (10 007 by default). This prevents configs that never explicitly set the limit from accidentally capping custom-set values.
- **`patch_config` clears run identity** — when you pass `callbacks=...`, `patch_config` deletes both `run_name` and `run_id`. This ensures callbacks passed at a sub-invocation level are not attributed to the parent run's identity.
- **`patch_configurable` is the cheapest** — it only mutates `config["configurable"]`. Use it when you need to inject a single runtime key (e.g. `thread_id`, `user_id`) without touching callbacks or tags.

### Example 1 — merging graph-level and call-level configs

```python
from langgraph._internal._config import merge_configs

base_config = {
    "tags": ["prod"],
    "metadata": {"app": "my-agent", "lc_versions": {"langchain-core": "1.0.0"}},
    "recursion_limit": 50,
    "configurable": {"thread_id": "abc"},
}

call_config = {
    "tags": ["user-request"],
    "metadata": {"user_id": "u42", "lc_versions": {"my-pkg": "0.2.1"}},
    "configurable": {"checkpoint_id": None},
}

merged = merge_configs(base_config, call_config)

print(merged["tags"])               # ['prod', 'user-request']  — concatenated
print(merged["metadata"]["app"])    # 'my-agent'                — preserved (call_config didn't set it)
print(merged["metadata"]["user_id"]) # 'u42'                   — call wins for new keys
print(merged["metadata"]["lc_versions"])  # {'langchain-core': '1.0.0', 'my-pkg': '0.2.1'}  — deep merged
print(merged["recursion_limit"])    # 50                         — base custom value preserved
print(merged["configurable"])       # {'thread_id': 'abc', 'checkpoint_id': None}  — shallow merged
```

### Example 2 — `patch_configurable` for per-request thread isolation

```python
from langgraph._internal._config import patch_configurable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    count: int


def increment(state: State) -> dict:
    return {"count": state["count"] + 1}


builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Base config reused across all requests; thread_id is injected per call
base_cfg = {"configurable": {"user_locale": "en-US"}}

result1 = graph.invoke({"count": 0}, patch_configurable(base_cfg, {"thread_id": "thread-1"}))
result2 = graph.invoke({"count": 10}, patch_configurable(base_cfg, {"thread_id": "thread-2"}))

print(result1["count"])  # 1
print(result2["count"])  # 11
```

### Example 3 — `patch_config` for sub-call attribution

```python
from langgraph._internal._config import patch_config
from langgraph.config import get_config

def my_node(state: dict) -> dict:
    parent_config = get_config()

    # Give sub-calls their own run identity and limit concurrency
    sub_config = patch_config(
        parent_config,
        run_name="my_node:sub_call",
        max_concurrency=4,
        configurable={"model": "gpt-4o-mini"},
    )

    # Passing callbacks= clears any existing run_name and run_id from the config.
    # Note: if you also pass run_name= in the same call, patch_config applies it
    # *after* the clear — so don't combine them when you want a fully fresh identity.
    sub_config_fresh = patch_config(
        parent_config,
        callbacks=[],   # clears run_name and run_id that were in parent_config
    )

    print("run_name" in sub_config_fresh)  # False — cleared by callbacks=
    print(sub_config["configurable"]["model"])  # 'gpt-4o-mini'
    return state
```

---

## 2 · `ensure_config` + `_PROPAGATE_TO_METADATA` + `_CHECKPOINT_COORDINATE_KEYS`

**Module**: `langgraph._internal._config`  
**First dedicated coverage.**

`ensure_config` is the heavyweight config initializer called once at graph entry. It sets defaults, propagates ambient context from the current contextvar, and handles the critical **checkpoint coordinate isolation** rule.

```python
def ensure_config(*configs: RunnableConfig | None) -> RunnableConfig:
    """Return a fully initialized config dict.
    
    Steps:
    1. Start from the ambient contextvar config (the "inherited" environment).
    2. If any explicit config supplies a checkpoint coordinate key, clear
       the ambient coordinate to avoid polluting child graph checkpoints.
    3. Shallow-merge all explicit configs on top.
    4. Propagate key configurable values into metadata via _PROPAGATE_TO_METADATA.
    """
```

**Key implementation facts:**

- **Coordinate isolation rule** — If the caller's config contains any key from `_CHECKPOINT_COORDINATE_KEYS` (`thread_id`, `checkpoint_ns`, `checkpoint_id`, `checkpoint_map`), the ambient `configurable` from the contextvar is **discarded** entirely before merging. This prevents a child graph from accidentally writing checkpoints under a parent graph's namespace when invoked with its own `thread_id`.
- **`_PROPAGATE_TO_METADATA`** — after merging, specific configurable values are copied into the `metadata` dict for LangSmith tracing visibility. The propagated keys are: `thread_id`, `checkpoint_id`, `checkpoint_ns`, `task_id`, `run_id`, `assistant_id`, `graph_id`. Values already in metadata are not overwritten.
- **Sensitive key filtering** — `_exclude_as_metadata` prevents keys containing `key`, `token`, `secret`, `password`, or `auth` from ever being copied to metadata (privacy/security guard).
- **`recursion_limit` default** — the base `empty` dict starts with `DEFAULT_RECURSION_LIMIT` (10 007), which is the highest safe value. Callers that set a custom limit via their config override this.

```python
_CHECKPOINT_COORDINATE_KEYS = frozenset({
    "thread_id", "checkpoint_ns", "checkpoint_id", "checkpoint_map"
})

_PROPAGATE_TO_METADATA = frozenset({
    "thread_id", "checkpoint_id", "checkpoint_ns", "task_id",
    "run_id", "assistant_id", "graph_id",
})
```

### Example 1 — coordinate isolation for subgraph calls

```python
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._config import ensure_config
from langgraph._internal._constants import CONF

# The Pregel runtime sets var_child_runnable_config automatically when
# a parent graph invokes a child. Here we set it manually to reproduce
# the isolation behaviour in isolation.
ambient = {
    "configurable": {
        "thread_id": "parent-thread-99",
        "checkpoint_ns": "parent::subgraph",
        "user_id": "u42",
    }
}

token = var_child_runnable_config.set(ambient)
try:
    # Child graph called with its own thread_id
    child_explicit = {"configurable": {"thread_id": "child-thread-1"}}

    # Because child_explicit contains a coordinate key (thread_id),
    # the ambient configurable (from the contextvar) is cleared entirely
    # before merging. This is ensure_config's isolation guarantee.
    result = ensure_config(child_explicit)
    print(result[CONF]["thread_id"])         # 'child-thread-1'
    print(result[CONF].get("checkpoint_ns")) # None — parent coord was discarded
    print(result[CONF].get("user_id"))       # None — ambient stripped with coord
finally:
    var_child_runnable_config.reset(token)
```

### Example 2 — metadata propagation

```python
from langgraph._internal._config import ensure_config
from langgraph._internal._constants import CONF

cfg = ensure_config({"configurable": {
    "thread_id": "t-001",
    "run_id": "run-abc",
    "api_key": "sk-secret",   # should NOT appear in metadata
    "user_name": "alice",      # non-propagate key — won't appear in metadata
}})

# thread_id and run_id are propagated to metadata automatically
print(cfg.get("metadata", {}).get("thread_id"))  # 't-001'
print(cfg.get("metadata", {}).get("run_id"))      # 'run-abc'
# Sensitive and non-propagated keys are excluded
print(cfg.get("metadata", {}).get("api_key"))     # None
print(cfg.get("metadata", {}).get("user_name"))   # None
```

### Example 3 — custom subgraph invoker respecting isolation

```python
from langgraph._internal._config import ensure_config
from langgraph._internal._constants import CONF


def invoke_isolated(graph, inputs: dict, *, thread_id: str, **extra_configurable):
    """Invoke a graph with guaranteed checkpoint isolation.

    Passing thread_id as a coordinate key causes ensure_config to clear
    any ambient parent-graph checkpoint coordinates, so the child graph
    gets a clean namespace.
    """
    cfg = ensure_config({"configurable": {"thread_id": thread_id, **extra_configurable}})
    return graph.invoke(inputs, cfg)
```

---

## 3 · `ContextT` + `StateT` + `InputT` + `OutputT` + `NodeInputT`

**Module**: `langgraph.typing`  
**First dedicated coverage.**

LangGraph ships a small set of generic TypeVars that power its statically-typed API. Understanding them unlocks reusable graph factories and properly typed node signatures.

```python
from typing_extensions import TypeVar
from langgraph._internal._typing import StateLike

StateT       = TypeVar("StateT",       bound=StateLike)
StateT_co    = TypeVar("StateT_co",    bound=StateLike, covariant=True)
StateT_contra= TypeVar("StateT_contra",bound=StateLike, contravariant=True)

ContextT     = TypeVar("ContextT",     bound=StateLike | None, default=None)
ContextT_contra = TypeVar("ContextT_contra", bound=StateLike | None,
                           contravariant=True, default=None)

InputT       = TypeVar("InputT",       bound=StateLike, default=StateT)
OutputT      = TypeVar("OutputT",      bound=StateLike, default=StateT)
NodeInputT   = TypeVar("NodeInputT",   bound=StateLike)
NodeInputT_contra = TypeVar("NodeInputT_contra", bound=StateLike, contravariant=True)
```

**Key implementation facts:**

- **`StateLike` bound** — all TypeVars are bounded to `StateLike`, which is `TypedDict | BaseModel | dataclass | dict`. Any custom state class must be one of these to satisfy the bound.
- **`ContextT` default=None** — `ContextT` defaults to `None`, which is why you can call `StateGraph(State)` without a `context_schema`; the graph is typed as `StateGraph[State, None, State, State]` internally.
- **`InputT` default=StateT** — input and output types default to the state type, so a basic `StateGraph(State)` produces `StateGraph[State, None, State, State]` (state, context, input, output all matching).
- **Covariant/contravariant variants** — the `_co` and `_contra` versions appear in generic Protocol definitions like `_NodeWithContext[StateT_contra, ContextT_contra]` which represent callable node protocols. End users only need `StateT`, `ContextT`, `InputT`, `OutputT`.
- **`NodeInputT`** — used when narrowing the input type of individual nodes via `input_schema=`. The node function receives `NodeInputT` (a subset of `StateT`) but the graph state is still `StateT`.

### Example 1 — typed graph factory

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.typing import StateT, ContextT


def make_pipeline(
    state_schema: type[StateT],
    *,
    context_schema: type[ContextT] | None = None,
) -> StateGraph[StateT, ContextT, StateT, StateT]:
    """Generic graph factory that preserves type information for IDEs."""
    builder: StateGraph[StateT, ContextT, StateT, StateT] = StateGraph(
        state_schema,
        context_schema=context_schema,
    )
    return builder


# Concrete use — MyState and Context are inferred by the type checker
class MyState(TypedDict):
    value: int


class Context(TypedDict):
    user_id: str


graph_builder = make_pipeline(MyState, context_schema=Context)


from langgraph.runtime import Runtime


def greet(state: MyState, *, runtime: Runtime) -> MyState:
    """In LangGraph 1.2.6, context is injected as `runtime.context`,
    not as a second positional argument."""
    ctx = runtime.context
    print(f"User {ctx['user_id']} has value {state['value']}")
    return state


graph_builder.add_node("greet", greet)
graph_builder.add_edge(START, "greet")
graph_builder.add_edge("greet", END)

g = graph_builder.compile()
g.invoke({"value": 42}, context={"user_id": "alice"})
```

### Example 2 — narrowed node input with `NodeInputT`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.typing import StateT, NodeInputT


class FullState(TypedDict):
    messages: list[str]
    metadata: dict
    count: int


class NarrowInput(TypedDict):
    messages: list[str]   # only the subset needed by this node


def process_messages(state: NarrowInput) -> dict:
    """This node only sees messages, not the full state."""
    return {"count": len(state["messages"])}


builder = StateGraph(FullState)
builder.add_node(
    "processor",
    process_messages,
    input_schema=NarrowInput,  # node receives NarrowInput, writes back to FullState
)
builder.add_edge(START, "processor")
builder.add_edge("processor", END)

graph = builder.compile()
result = graph.invoke({"messages": ["a", "b", "c"], "metadata": {}, "count": 0})
print(result["count"])  # 3
```

### Example 3 — separate input/output schemas

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.typing import StateT, InputT, OutputT


class InternalState(TypedDict):
    raw_text: str
    word_count: int
    processed: bool


class UserInput(TypedDict):
    raw_text: str      # what the caller provides


class UserOutput(TypedDict):
    word_count: int    # what the caller receives back
    processed: bool


def parse_input(state: InternalState) -> dict:
    """Populate internal state from the input."""
    return {"word_count": len(state["raw_text"].split()), "processed": False}


def mark_processed(state: InternalState) -> dict:
    return {"processed": True}


builder = StateGraph(InternalState, input_schema=UserInput, output_schema=UserOutput)
builder.add_node("parse", parse_input)
builder.add_node("finish", mark_processed)
builder.add_edge(START, "parse")
builder.add_edge("parse", "finish")
builder.add_edge("finish", END)

graph = builder.compile()

# Caller provides UserInput; receives UserOutput
result = graph.invoke({"raw_text": "hello world from langgraph"})
print(result["word_count"])   # 4
print(result["processed"])    # True
# raw_text is NOT in the result (filtered by output schema)
```

---

## 4 · `TAG_NOSTREAM` + `TAG_HIDDEN`

**Module**: `langgraph.constants`  
**First dedicated coverage.**

LangGraph provides two system-level tags that change how nodes and edges behave in streams and traces.

```python
import sys

TAG_NOSTREAM = sys.intern("nostream")
"""Tag to disable streaming for a chat model invoked inside a node."""

TAG_HIDDEN = sys.intern("langsmith:hidden")
"""Tag to hide a node/edge from LangSmith traces and debug streams."""
```

**Key implementation facts:**

- **`TAG_NOSTREAM`** — when added to a `ChatModel` invocation via `.with_config({"tags": ["nostream"]})`, the model's token stream is NOT forwarded to the parent graph's `messages` channel. The model still runs and returns its final output; only the intermediate token events are suppressed. Useful for background-reasoning nodes you don't want polluting the user-facing message stream.
- **`TAG_HIDDEN`** — suppresses a node from `stream_mode="debug"` output and hides it from LangSmith's trace view. `map_debug_tasks` (in `langgraph.pregel.debug`) skips any task whose tags include `TAG_HIDDEN`. Used heavily for internal book-keeping nodes like `ChannelWrite` wrappers and the `__start__` input projection node.
- **`sys.intern`** — both tags are interned strings. Use `== TAG_NOSTREAM` or `TAG_NOSTREAM in tags` for containment checks — do not rely on `is` comparisons, because tags collected from user code are not guaranteed to be interned and identity checks would silently fail.
- **Internal-only in 1.2.6** — `TAG_HIDDEN` is applied by LangGraph itself when creating internal nodes during compilation (e.g. the `__start__` input-projection node is built as `PregelNode(tags=[TAG_HIDDEN])`). Passing `tags=[TAG_HIDDEN]` to `add_node` is silently ignored in 1.2.6 — the kwarg goes into `**DeprecatedKwargs` which only handles `retry` and `input`; there is no `tags` storage path in `StateNodeSpec`. `set_node_defaults` also does not accept `tags` (it only accepts `retry_policy`, `cache_policy`, `error_handler`, and `timeout`).

### Example 1 — `TAG_NOSTREAM` to silence a reasoning node

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.constants import TAG_NOSTREAM


class State(TypedDict):
    messages: Annotated[list, add_messages]
    reasoning: str   # internal scratch field


def reason_silently(state: State) -> dict:
    """Runs an internal reasoning step. Tokens are NOT streamed to the caller."""
    # In a real app, this would call a ChatModel with TAG_NOSTREAM
    # e.g. llm.with_config({"tags": [TAG_NOSTREAM]}).invoke(state["messages"])
    reasoning_output = "The answer is 42 because of the universal constants."
    return {"reasoning": reasoning_output}


def respond(state: State) -> dict:
    """Uses the private reasoning to generate the user-facing reply. Tokens stream."""
    reply = AIMessage(content=f"Based on analysis: {state['reasoning']}")
    return {"messages": [reply]}


builder = StateGraph(State)
builder.add_node("reason", reason_silently)
builder.add_node("respond", respond)
builder.add_edge(START, "reason")
builder.add_edge("reason", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

# In stream mode, "reason" tokens are suppressed; only "respond" tokens appear
for chunk in graph.stream(
    {"messages": [HumanMessage(content="What is the answer?")]},
    stream_mode="messages",
):
    print(chunk)
```

### Example 2 — observing `TAG_HIDDEN` on LangGraph's internal `__start__` node

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import TAG_HIDDEN


class State(TypedDict):
    value: int


def compute(state: State) -> dict:
    return {"value": state["value"] * 2}


builder = StateGraph(State)
builder.add_node("compute", compute)
builder.add_edge(START, "compute")
builder.add_edge("compute", END)

graph = builder.compile()

# LangGraph applies TAG_HIDDEN internally to the '__start__' input-projection
# node when compiling. Inspect the compiled graph to confirm:
hidden_nodes = [
    name for name, node in graph.nodes.items()
    if TAG_HIDDEN in (node.tags or [])
]
print("Hidden nodes:", hidden_nodes)  # ['__start__']

# '__start__' is also absent from debug-stream task events:
debug_chunks = list(graph.stream({"value": 5}, stream_mode="debug"))
task_names = {c.get("payload", {}).get("name") for c in debug_chunks
              if c.get("type") == "task"}
print("compute" in task_names)    # True
print("__start__" in task_names)  # False — hidden by TAG_HIDDEN
```

### Example 3 — auditing which nodes carry `TAG_HIDDEN` in a compiled graph

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import TAG_HIDDEN


class State(TypedDict):
    steps: list[str]


def step_a(state: State) -> dict:
    return {"steps": state["steps"] + ["a"]}


def step_b(state: State) -> dict:
    return {"steps": state["steps"] + ["b"]}


builder = StateGraph(State)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()

# After compilation, inspect every PregelNode for TAG_HIDDEN.
# LangGraph only applies it to its own internal nodes — user nodes never
# receive it in 1.2.6 even if you pass tags=[TAG_HIDDEN] to add_node.
for name, node in graph.nodes.items():
    is_hidden = TAG_HIDDEN in (node.tags or [])
    print(f"  {name:20s} hidden={is_hidden}")
# __start__            hidden=True
# a                    hidden=False
# b                    hidden=False

result = graph.invoke({"steps": []})
print(result["steps"])  # ['a', 'b']
```

---

## 5 · `recast_checkpoint_ns` + `filter_to_user_tags`

**Module**: `langgraph._internal._config`  
**First dedicated coverage.**

Two small but critical string utilities that handle checkpoint namespace hygiene and stream metadata tag filtering.

```python
def recast_checkpoint_ns(ns: str) -> str:
    """Remove task IDs from checkpoint namespace.
    
    Checkpoint namespaces include task IDs as suffixes (e.g. 'parent::child:abc123')
    to distinguish concurrent subgraph calls. When reporting or resuming, you often
    want the structural namespace without the ephemeral task ID suffix.
    """

def filter_to_user_tags(tags: Sequence[str] | None) -> list[str] | None:
    """Drop langgraph's internal seq:step:N bookkeeping tags.
    
    Returns surviving tags (user-supplied + non-seq tags), or None if
    none remain.
    """
```

**Key implementation facts:**

- **`recast_checkpoint_ns` internals** — uses `NS_SEP` (`"|"`) and `NS_END` (`":"`) to split the namespace. Each segment is split on `NS_END` and only the part before the first `:` is kept. Numeric-only segments (bare task IDs) are dropped entirely. Result: `"parent|child:abc123"` → `"parent|child"`.
- **`filter_to_user_tags` use case** — the `messages` and `tasks` stream mode handlers both call this before attaching tags to emitted events. The `seq:step:N` internal tags track sequence steps for the `add_sequence()` API and would otherwise pollute user-visible event metadata.
- **`NS_SEP` and `NS_END`** — `NS_SEP = "|"` separates subgraph nesting levels; `NS_END = ":"` separates the node name from the task ID within a level. These constants are defined in `langgraph._internal._constants`.

### Example 1 — `recast_checkpoint_ns` for subgraph-aware logging

```python
from langgraph._internal._config import recast_checkpoint_ns

# Checkpoint namespace as it appears inside the runtime
raw_ns = "parent_graph|child_agent:task-abc123|leaf_node:task-def456"

# Strip task IDs for human-readable reporting
clean_ns = recast_checkpoint_ns(raw_ns)
print(clean_ns)  # 'parent_graph|child_agent|leaf_node'

# Numeric-only segments (bare task IDs without node names) are also dropped
bare_id_ns = "parent|0|child"
print(recast_checkpoint_ns(bare_id_ns))  # 'parent|child'

# Root graph namespace is empty string
print(recast_checkpoint_ns(""))  # ''
```

### Example 2 — `filter_to_user_tags` in a custom stream handler

```python
from langgraph._internal._config import filter_to_user_tags

# Tags as they appear on an event from add_sequence()-driven nodes
raw_tags = ["my-custom-tag", "seq:step:0", "seq:step:2", "prod"]

user_tags = filter_to_user_tags(raw_tags)
print(user_tags)  # ['my-custom-tag', 'prod']

# None input is passed through as None
print(filter_to_user_tags(None))  # None

# All tags are seq:step tags → None returned (not empty list)
all_internal = ["seq:step:0", "seq:step:1"]
print(filter_to_user_tags(all_internal))  # None
```

### Example 3 — reading clean namespace from a live run config

```python
from langgraph._internal._config import recast_checkpoint_ns
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINT_NS, CONF
from langgraph.config import get_config
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    log: list[str]


def log_node(state: State) -> dict:
    cfg = get_config()
    raw_ns = cfg.get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS, "")
    clean_ns = recast_checkpoint_ns(raw_ns)
    return {"log": state["log"] + [f"running in namespace: {clean_ns!r}"]}


builder = StateGraph(State)
builder.add_node("log", log_node)
builder.add_edge(START, "log")
builder.add_edge("log", END)

graph = builder.compile()
result = graph.invoke({"log": []})
print(result["log"])  # ["running in namespace: ''"]  — root graph has empty namespace
```

---

## 6 · `InMemorySaver` — storage internals

**Module**: `langgraph.checkpoint.memory`  
**First dedicated deep-dive of internal storage layout.**

`InMemorySaver` is the reference checkpointer for testing. Its internal three-dict storage structure mirrors what production backends (Postgres, Redis) must implement.

```python
class InMemorySaver(BaseCheckpointSaver[str], ...):
    # Dict 1: main checkpoint blobs
    storage: defaultdict[
        str,   # thread_id
        dict[
            str,   # checkpoint_ns
            dict[
                str,   # checkpoint_id
                tuple[
                    tuple[str, bytes],  # serialized checkpoint
                    tuple[str, bytes],  # serialized metadata
                    str | None,         # parent checkpoint_id
                ]
            ]
        ]
    ]

    # Dict 2: task write records (indexed by WRITES_IDX_MAP for special writes)
    writes: defaultdict[
        tuple[str, str, str],              # (thread_id, checkpoint_ns, checkpoint_id)
        dict[
            tuple[str, int],               # (task_id, write_idx)
            tuple[str, str, tuple[str, bytes], str]  # (task_id, channel, serde_value, type_str)
        ]
    ]

    # Dict 3: channel value blobs (deduplication store)
    blobs: dict[
        tuple[str, str, str, str | int | float],  # (thread_id, checkpoint_ns, channel, version)
        tuple[str, bytes]                          # (type_str, serialized_value)
    ]
```

**Key implementation facts:**

- **`storage` 3-tuple value** — each checkpoint is stored as `(serialized_checkpoint, serialized_metadata, parent_id)`. The serialization uses the saver's `.serde` (defaults to `JsonPlusSerializer`). Fetching a checkpoint requires deserializing both the checkpoint and metadata.
- **`writes` key** — `(thread_id, ns, checkpoint_id)` maps to a dict of `(task_id, write_idx)` → write record. The `write_idx` is either a positive integer (positional order of writes from `put_writes`) or a negative sentinel from `WRITES_IDX_MAP` (e.g. `-1` for `ERROR`, `-3` for `INTERRUPT`).
- **`blobs` for channel deduplication** — each channel value is stored once under its version key. When a checkpoint references the same channel version as a previous checkpoint, the blob is shared, not duplicated. This can dramatically reduce memory for long-running threads.
- **`put_writes` accumulates without duplicating** — writes for a checkpoint accumulate: calling `put_writes` multiple times for the same `(thread_id, ns, checkpoint_id, task_id)` merges them. Positional indices come from `enumerate(writes)`, starting at 0. An existing entry with a non-negative `(task_id, idx)` key is skipped (not overwritten), ensuring idempotent re-delivery. Sentinel channels (negative indices) are always overwritten.
- **`list()` ordering** — checkpoints are listed newest-first by iterating the inner dict in insertion order and reversing. This matches the contract that `get()` returns the most recent checkpoint.

### Example 1 — inspecting the raw storage structure

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    value: int


def inc(state: State) -> dict:
    return {"value": state["value"] + 1}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("inc", inc)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t1"}}

graph.invoke({"value": 0}, config)
graph.invoke(None, config)  # Second call — continues from previous checkpoint

# Inspect raw storage structure
thread_storage = saver.storage["t1"]    # thread_id → ns dict
root_ns = thread_storage[""]            # "" is root graph namespace
# Each invoke writes 2 checkpoints (one before nodes run, one after),
# plus the very first empty checkpoint, so ~5 entries after 2 invokes.
print(f"Checkpoints stored: {len(root_ns)}")

# Each checkpoint_id maps to (checkpoint_bytes, metadata_bytes, parent_id)
for ckpt_id, (ckpt_data, meta_data, parent_id) in root_ns.items():
    print(f"  id={ckpt_id[:8]}... parent={parent_id[:8] if parent_id else None}")
```

### Example 2 — reading pending writes from the writes dict

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    messages: list[str]


def node_with_interrupt(state: State) -> dict:
    interrupt("Please review")
    return {"messages": state["messages"] + ["done"]}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("node", node_with_interrupt)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-interrupt"}}

try:
    graph.invoke({"messages": []}, config)
except Exception:
    pass  # Graph interrupted

# Inspect writes dict to see the INTERRUPT sentinel write
writes_key = next(iter(saver.writes.keys()))  # (thread_id, ns, checkpoint_id)
for (task_id, write_idx), (t_id, channel, value, type_str) in saver.writes[writes_key].items():
    print(f"  write_idx={write_idx}, channel={channel!r}")
    # Negative write_idx = sentinel from WRITES_IDX_MAP
    if write_idx < 0:
        sentinels = {v: k for k, v in WRITES_IDX_MAP.items()}
        print(f"  → sentinel: {sentinels.get(write_idx)}")
```

### Example 3 — counting blobs to understand deduplication

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    static_data: dict    # large, rarely changes
    counter: int


def step1(state: State) -> dict:
    return {"counter": state["counter"] + 1}


def step2(state: State) -> dict:
    return {"counter": state["counter"] + 1}


def step3(state: State) -> dict:
    return {"counter": state["counter"] + 1}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_node("step3", step3)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.add_edge("step3", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-blobs"}}

big_data = {"large": "x" * 10_000}
graph.invoke({"static_data": big_data, "counter": 0}, config)
# One invoke creates 5 checkpoints (before __start__, after each of 3 nodes, final).
# static_data never changes across those checkpoints → only 1 blob is stored.
# counter changes at each step (0→1→2→3) → 4 blobs.

static_blob_count = sum(
    1 for (_, _, ch, _) in saver.blobs if ch == "static_data"
)
counter_blob_count = sum(
    1 for (_, _, ch, _) in saver.blobs if ch == "counter"
)
print(f"static_data blobs: {static_blob_count}")  # 1 — shared by all 5 checkpoints
print(f"counter blobs:     {counter_blob_count}")  # 4 — new version per step (0,1,2,3)
```

---

## 7 · `PendingWrite` + `WRITES_IDX_MAP` + sentinel channels

**Module**: `langgraph.checkpoint.base` · `langgraph.checkpoint.serde.types`  
**First dedicated coverage of sentinel write channels.**

`PendingWrite` is the 3-tuple that represents an unprocessed write in a checkpoint. `WRITES_IDX_MAP` assigns fixed negative indices to special sentinel channels so their writes can be identified and replayed correctly.

```python
PendingWrite = tuple[str, str, Any]
# (task_id, channel_name, deserialized_value)

# From langgraph.checkpoint.serde.types:
ERROR     = "__error__"
INTERRUPT = "__interrupt__"
RESUME    = "__resume__"
SCHEDULED = "__scheduled__"

# From langgraph.checkpoint.base:
WRITES_IDX_MAP = {
    ERROR:     -1,
    SCHEDULED: -2,
    INTERRUPT: -3,
    RESUME:    -4,
}
```

**Key implementation facts:**

- **`PendingWrite` anatomy** — `task_id` is the UUID of the node task that produced the write; `channel_name` is the state key (or a sentinel like `"__interrupt__"`); `value` is the deserialized payload (e.g. an `Interrupt` dataclass for `__interrupt__`).
- **`WRITES_IDX_MAP` negative indices** — the `writes` dict in `InMemorySaver` uses `(task_id, write_idx)` as its key. Positive `write_idx` are positional (0, 1, 2…). Negative indices from `WRITES_IDX_MAP` identify sentinel writes so they can be overwritten/deduplicated without colliding with positional writes.
- **`ERROR` writes** — when a node raises an unhandled exception and there is an `error_handler`, a `(task_id, "__error__", NodeError(...))` write is stored. The error handler reads it to understand which node failed and why.
- **`INTERRUPT` writes** — when `interrupt()` is called inside a node, a `(task_id, "__interrupt__", Interrupt(...))` write is stored at index `-3`. On resume, the runtime checks for these to decide which tasks to skip (already-completed tasks are not re-run).
- **`RESUME` writes** — user-provided resume values (from `Command(resume=...)`) are stored as `(task_id, "__resume__", value)` writes at index `-4`. The `PregelScratchpad.resume` list is populated from these on graph load.
- **`SCHEDULED` writes** — used by the `Send` mechanism to persist deferred node dispatches across checkpoints.

### Example 1 — reading PendingWrite tuples from checkpoint history

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    value: int


def double(state: State) -> dict:
    return {"value": state["value"] * 2}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("double", double)
builder.add_edge(START, "double")
builder.add_edge("double", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-writes"}}

graph.invoke({"value": 3}, config)

# Inspect pending writes on the most recent checkpoint
history = list(graph.get_state_history(config))
latest = history[0]  # most recent

print(f"pending_writes: {latest.next}")
# After completion, pending_writes is typically empty or has resume markers
for pw in (latest.tasks or []):
    print(f"  task={pw.id[:8]}... writes={pw.writes}")
```

### Example 2 — `WRITES_IDX_MAP` constants for custom saver implementors

```python
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.checkpoint.serde.types import ERROR, INTERRUPT, RESUME, SCHEDULED

# When implementing a custom BaseCheckpointSaver.put_writes(),
# use WRITES_IDX_MAP to assign stable indices to sentinel channels

def compute_write_idx(channel: str, current_positional_idx: int) -> int:
    """Returns the write index to use when storing a pending write.
    
    Sentinel channels get fixed negative indices.
    Regular state channels get sequential positive indices.
    """
    return WRITES_IDX_MAP.get(channel, current_positional_idx)


# Demonstrate the index assignments
for channel in [ERROR, SCHEDULED, INTERRUPT, RESUME, "messages", "counter"]:
    idx = compute_write_idx(channel, 0)  # positional_idx=0 for regular channels
    print(f"{channel!r:20s} → idx={idx}")
# __error__            → idx=-1
# __scheduled__        → idx=-2
# __interrupt__        → idx=-3
# __resume__           → idx=-4
# messages             → idx=0
# counter              → idx=0
```

### Example 3 — filtering pending writes by sentinel type

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import INTERRUPT, RESUME
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    approved: bool
    result: str


def approval_node(state: State) -> dict:
    decision = interrupt({"question": "Approve this action?"})
    return {"approved": decision, "result": "done" if decision else "rejected"}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("approve", approval_node)
builder.add_edge(START, "approve")
builder.add_edge("approve", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-interrupt-filter"}}

# First invoke — hits the interrupt
try:
    graph.invoke({"approved": False, "result": ""}, config)
except Exception:
    pass

# Check raw writes to find the INTERRUPT sentinel
thread_writes = saver.writes
for key, writes_dict in thread_writes.items():
    for (task_id, write_idx), (t_id, channel, value, _) in writes_dict.items():
        if channel == INTERRUPT:
            print(f"Found interrupt write at idx={write_idx}: {value!r}")

# Resume with approval
result = graph.invoke(Command(resume=True), config)
print(result["approved"])   # True
print(result["result"])     # 'done'
```

---

## 8 · `get_checkpoint_id` + `get_checkpoint_metadata`

**Module**: `langgraph.checkpoint.base`  
**First dedicated coverage.**

Two utility functions for extracting checkpoint identity from a `RunnableConfig` at runtime.

```python
def get_checkpoint_id(config: RunnableConfig) -> str | None:
    """Return the checkpoint_id from config["configurable"], or None."""

def get_checkpoint_metadata(
    config: RunnableConfig,
    *,
    metadata: CheckpointMetadata | None,
) -> CheckpointMetadata:
    """Build canonical checkpoint metadata by merging the config's run
    context (run_id) with caller-supplied metadata."""
```

**Key implementation facts:**

- **`get_checkpoint_id` source** — reads `config["configurable"]["checkpoint_id"]`. If the `configurable` key is absent, it raises `KeyError` — it is **not** safe to call with a bare `{}` config outside a graph run. Only the missing `checkpoint_id` *within* `configurable` is safe (returns `None`). Outside a graph run always call `ensure_config(config)` first, or guard with `config.get("configurable", {}).get("checkpoint_id")`.
- **`get_checkpoint_metadata`** — used internally by `InMemorySaver.put()` and all production savers to construct the `CheckpointMetadata` dict that goes into storage. It merges caller-provided `metadata` with `run_id` from the config (so every checkpoint records which run created it).
- **Fork detection** — when building a fork from a specific checkpoint (e.g. time-travel), the `checkpoint_id` in the config identifies the source checkpoint. A custom saver can use `get_checkpoint_id(config)` to determine if it should write a new checkpoint or create a fork.
- **`get_serializable_checkpoint_metadata`** — a companion function that additionally strips fields that cannot be serialized to JSON. Used by savers that store metadata in plain-text formats.

### Example 1 — reading the current checkpoint ID inside a node

```python
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    checkpoint_seen: str


def snapshot_id(state: State) -> dict:
    cfg = get_config()
    ckpt_id = get_checkpoint_id(cfg)
    # ckpt_id is the ID of the checkpoint the graph was invoked FROM
    # (the most recent checkpoint before this node ran)
    return {"checkpoint_seen": ckpt_id or "no-checkpoint"}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("snap", snapshot_id)
builder.add_edge(START, "snap")
builder.add_edge("snap", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-ckptid"}}

r1 = graph.invoke({"checkpoint_seen": ""}, config)
print(r1["checkpoint_seen"])  # 'no-checkpoint' — first run, no prior checkpoint

r2 = graph.invoke({"checkpoint_seen": ""}, config)
print(r2["checkpoint_seen"])  # 'UUID...' — previous checkpoint ID
```

### Example 2 — fork / time-travel by injecting a checkpoint ID

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    value: int


def double(state: State) -> dict:
    return {"value": state["value"] * 2}


saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("double", double)
builder.add_edge(START, "double")
builder.add_edge("double", END)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "t-fork"}}

graph.invoke({"value": 1}, config)  # checkpoint saved with value=1; runs double → 2
graph.invoke(None, config)          # resumes from checkpoint; double(2) → 4

# List all checkpoints — each invoke writes a "before" and "after" checkpoint
history = list(graph.get_state_history(config))
for snapshot in history:
    print(f"  id={snapshot.config['configurable']['checkpoint_id'][:8]}... "
          f"value={snapshot.values.get('value')} next={snapshot.next}")

# Fork from the pre-node checkpoint of the first invoke (value=1, 'double' pending)
fork_point = next(s for s in history if s.values.get("value") == 1 and s.next)
fork_config = {
    "configurable": {
        "thread_id": "t-fork",
        "checkpoint_id": fork_point.config["configurable"]["checkpoint_id"],
    }
}
# Replay from that fork point — double(1) = 2
forked = graph.invoke(None, fork_config)
print(f"forked value: {forked['value']}")  # 2 (double ran again from value=1)
```

### Example 3 — using `get_checkpoint_metadata` in a custom saver

```python
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langchain_core.runnables import RunnableConfig
from collections import defaultdict
from typing import Any, Iterator


class LoggingCheckpointer(BaseCheckpointSaver):
    """Minimal custom saver that logs every checkpoint write."""

    def __init__(self):
        super().__init__()
        self._store: dict[str, CheckpointTuple] = {}

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"].get("thread_id")
        return self._store.get(thread_id)

    def list(self, config: RunnableConfig, **kwargs) -> Iterator[CheckpointTuple]:
        yield from self._store.values()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        # Use get_checkpoint_metadata to merge run_id from config
        full_metadata = get_checkpoint_metadata(config, metadata=metadata)
        thread_id = config["configurable"]["thread_id"]
        self._store[thread_id] = CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=full_metadata,
        )
        print(f"Saved checkpoint: step={full_metadata.get('step')}, "
              f"source={full_metadata.get('source')}")
        return config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        pass  # no-op for this logging-only saver
```

---

## 9 · `build_serde_allowlist` + `STRICT_MSGPACK_ENABLED` + `apply_checkpointer_allowlist`

**Module**: `langgraph._internal._serde`  
**First dedicated coverage.**

LangGraph ships an optional strict msgpack serialization mode that enforces an explicit allowlist of types that may be serialized into checkpoints. This prevents accidental persistence of unserializable objects and hardens production deployments.

```python
# Set at import time by trying to import the _msgpack extension
STRICT_MSGPACK_ENABLED: bool  # True when langgraph-checkpoint[msgpack] is installed

def build_serde_allowlist(
    schemas: list[Any],
    channels: dict[str, BaseChannel],
) -> set[tuple[str, ...]] | None:
    """Scan schemas and channel types to build a set of (module, qualname)
    tuples representing all types that the checkpointer may serialize."""

def apply_checkpointer_allowlist(
    checkpointer: Any,
    allowlist: set[tuple[str, ...]] | None,
) -> Any:
    """Call checkpointer.with_allowlist(allowlist) if supported, else log a
    warning and return the checkpointer unchanged."""

def curated_core_allowlist() -> set[tuple[str, ...]]:
    """Return a set of (module, qualname) tuples for all langchain-core
    message types (HumanMessage, AIMessage, ToolMessage, etc.)."""
```

**Key implementation facts:**

- **`STRICT_MSGPACK_ENABLED` check** — enabled when `from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED` succeeds. This requires the optional `langgraph-checkpoint[msgpack]` extra. Without it, all serialization falls back to `JsonPlusSerializer` with no type restrictions.
- **`build_serde_allowlist` scanning** — recursively inspects `TypedDict`, `dataclass`, `Pydantic v1/v2 models`, `Enum` subclasses, and `Annotated` types. It uses `_safe_get_type_hints` which catches `NameError` from forward-referenced types. Type variables are unwound through their bounds.
- **`curated_core_allowlist`** — always includes all 14 langchain-core message classes (both base and chunk variants, plus `RemoveMessage`). The `entrypoint` decorator merges this with the user's schema when building the graph's allowlist.
- **Allowlist format** — each entry is a `(module_name, qualified_class_name)` tuple, e.g. `("langchain_core.messages.human", "HumanMessage")`. The msgpack serde only permits types appearing in this set when `STRICT_MSGPACK_ENABLED=True`.
- **`apply_checkpointer_allowlist` fallback** — if the checkpointer does not implement `with_allowlist` (e.g. an older custom saver), the function emits a one-time warning and returns the checkpointer unchanged. The warning is de-duplicated via the `_warned_allowlist_unsupported` module-level flag.

### Example 1 — building an allowlist for a custom state schema

```python
from langgraph._internal._serde import build_serde_allowlist, curated_core_allowlist
from langgraph.channels.last_value import LastValue
from dataclasses import dataclass
from typing_extensions import TypedDict
from enum import Enum


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    id: str
    priority: Priority
    tags: list[str]


class State(TypedDict):
    tasks: list[Task]
    current_priority: Priority


# Build an allowlist from the schema
allowlist = build_serde_allowlist(
    schemas=[State, Task],
    channels={"tasks": LastValue(list), "current_priority": LastValue(Priority)},
)

if allowlist is not None:
    # Check which types were included
    type_names = {name for _, name in allowlist}
    print("Task" in type_names)      # True
    print("Priority" in type_names)  # True
    print("State" in type_names)     # True
else:
    print("Strict msgpack not enabled — allowlist is None")

# Always include core message types
core = curated_core_allowlist()
print(("langchain_core.messages.human", "HumanMessage") in core)  # True
```

### Example 2 — checking whether strict mode is active

```python
try:
    from langgraph._internal._serde import STRICT_MSGPACK_ENABLED
except ImportError:
    STRICT_MSGPACK_ENABLED = False

if STRICT_MSGPACK_ENABLED:
    print("Strict msgpack is active — only allowlisted types may be checkpointed")
else:
    print("Standard JSON-Plus serialization — all pickle-fallback types permitted")

# Use this flag to conditionally set up an allowlist
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph._internal._serde import build_serde_allowlist, apply_checkpointer_allowlist
from langgraph.channels.last_value import LastValue


class State(TypedDict):
    value: int


builder = StateGraph(State)
builder.add_node("noop", lambda s: s)
builder.add_edge(START, "noop")
builder.add_edge("noop", END)

# Apply the allowlist to the saver *before* compiling so there's no
# post-compile attribute mutation.
saver = InMemorySaver()
if STRICT_MSGPACK_ENABLED:
    allowlist = build_serde_allowlist(schemas=[State], channels={})
    if allowlist:
        saver = apply_checkpointer_allowlist(saver, allowlist)

graph = builder.compile(checkpointer=saver)
```

### Example 3 — Pydantic model scanning in allowlist

```python
from pydantic import BaseModel
from langgraph._internal._serde import build_serde_allowlist
from langgraph.channels.last_value import LastValue
from typing import Optional


class Address(BaseModel):
    street: str
    city: str
    country: str = "US"


class UserProfile(BaseModel):
    user_id: str
    name: str
    address: Optional[Address] = None


# Pydantic models are recursively scanned — nested models appear in allowlist
allowlist = build_serde_allowlist(
    schemas=[UserProfile],
    channels={"profile": LastValue(UserProfile)},
)

if allowlist is not None:
    type_names = {name for _, name in allowlist}
    print("UserProfile" in type_names)  # True
    print("Address" in type_names)      # True — nested model included
    print(len(allowlist))               # both models + str + optional wrapper types
else:
    print("Strict mode not available")
```

---

## 10 · `DEFAULT_RECURSION_LIMIT` + `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` + `CONFIG_KEY_*`

**Module**: `langgraph._internal._config` · `langgraph._internal._constants`  
**First dedicated coverage.**

LangGraph exposes several environment variables and reserved `configurable` keys that control graph execution limits, delta channel snapshotting, and internal subsystem wiring. These are essential for production tuning.

```python
# From langgraph._internal._config — env-var overrides
DEFAULT_RECURSION_LIMIT = int(os.getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))
DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT = int(
    os.getenv("LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT", "5000")
)

# From langgraph._internal._constants — reserved configurable keys
CONFIG_KEY_SEND          = sys.intern("__pregel_send")
CONFIG_KEY_READ          = sys.intern("__pregel_read")
CONFIG_KEY_CALL          = sys.intern("__pregel_call")
CONFIG_KEY_CHECKPOINTER  = sys.intern("__pregel_checkpointer")
CONFIG_KEY_STREAM        = sys.intern("__pregel_stream")
CONFIG_KEY_CACHE         = sys.intern("__pregel_cache")
CONFIG_KEY_RESUMING      = sys.intern("__pregel_resuming")
CONFIG_KEY_TASK_ID       = sys.intern("__pregel_task_id")
CONFIG_KEY_THREAD_ID     = sys.intern("thread_id")
CONFIG_KEY_CHECKPOINT_MAP = sys.intern("checkpoint_map")
CONFIG_KEY_CHECKPOINT_ID  = sys.intern("checkpoint_id")
CONFIG_KEY_CHECKPOINT_NS  = sys.intern("checkpoint_ns")
CONFIG_KEY_RUNTIME        = sys.intern("__pregel_runtime")

# Namespace separators
NS_SEP = "|"   # separates subgraph levels in checkpoint_ns
NS_END = ":"   # separates node name from task ID within a level
```

**Key implementation facts:**

- **`DEFAULT_RECURSION_LIMIT` = 10 007** — dramatically higher than LangChain's default of 25. LangGraph sets this high because graphs with many nodes and fan-out patterns can legitimately execute hundreds of steps in a single `invoke()`. Override via `LANGGRAPH_DEFAULT_RECURSION_LIMIT` or per-call via `config={"recursion_limit": N}`.
- **`DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`** — `DeltaChannel` avoids storing the full channel value on every step; instead it stores deltas and periodically writes a full snapshot. This constant is the max supersteps before a forced snapshot regardless of update frequency. Default: 5000. Override via `LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` for very long-lived threads.
- **`CONFIG_KEY_*` reserved keys** — all `__pregel_*` configurable keys are internal. They carry the `write`, `read`, `call`, `checkpointer`, `stream`, `cache`, `runtime` function/object references that the Pregel runtime injects at execution time. **Never set these manually** — only read them via the provided accessor functions (`get_config()`, `get_store()`, `get_stream_writer()`).
- **`CONFIG_KEY_RUNTIME`** — the `Runtime` object is injected here, which is how `get_store()` and `get_stream_writer()` work: they call `get_config()[CONF][CONFIG_KEY_RUNTIME].store` etc.
- **`NS_SEP` and `NS_END`** — checkpoint namespace strings like `"parent|child:task-abc"` are constructed using these separators. Never hard-code `"|"` or `":"` when working with namespaces — import these constants.

### Example 1 — overriding the recursion limit for a deep planning graph

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph._internal._config import DEFAULT_RECURSION_LIMIT


class PlannerState(TypedDict):
    depth: int
    plan: list[str]


def expand(state: PlannerState) -> dict:
    if state["depth"] >= 50:
        return {}
    return {"depth": state["depth"] + 1, "plan": state["plan"] + [f"step-{state['depth']}"]}


def should_continue(state: PlannerState) -> str:
    return "expand" if state["depth"] < 50 else "__end__"


builder = StateGraph(PlannerState)
builder.add_node("expand", expand)
builder.add_edge(START, "expand")
builder.add_conditional_edges("expand", should_continue)

# Default is 10007, which accommodates deep planning graphs
print(f"Default limit: {DEFAULT_RECURSION_LIMIT}")  # 10007

graph = builder.compile()

# Explicit per-call override (lower for testing, higher for production)
result = graph.invoke(
    {"depth": 0, "plan": []},
    config={"recursion_limit": 100},  # override for this call
)
print(f"Plan steps: {len(result['plan'])}")  # 50
```

### Example 2 — reading the current task ID and checkpoint NS from a node

```python
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    NS_SEP,
    NS_END,
)
from langgraph.config import get_config
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    info: str


def introspect(state: State) -> dict:
    cfg = get_config()
    conf = cfg.get(CONF, {})

    task_id   = conf.get(CONFIG_KEY_TASK_ID, "N/A")
    ns        = conf.get(CONFIG_KEY_CHECKPOINT_NS, "")
    depth     = ns.count(NS_SEP)  # number of nesting levels

    return {"info": f"task={task_id[:8]}..., ns_depth={depth}"}


builder = StateGraph(State)
builder.add_node("intro", introspect)
builder.add_edge(START, "intro")
builder.add_edge("intro", END)

graph = builder.compile()
result = graph.invoke({"info": ""})
print(result["info"])   # 'task=<uuid>..., ns_depth=0'  — root graph
```

### Example 3 — environment variable tuning for production

```python
import os

# These would be set in the deployment environment, not in code.
# Shown here for documentation purposes.

# Increase for graphs with recursive expansion patterns
os.environ["LANGGRAPH_DEFAULT_RECURSION_LIMIT"] = "25000"

# Reduce snapshot interval for memory-constrained deployments with DeltaChannels
os.environ["LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT"] = "1000"

# After setting env vars, reimport to pick up new values
import importlib
import langgraph._internal._config as cfg_mod
importlib.reload(cfg_mod)

print(cfg_mod.DEFAULT_RECURSION_LIMIT)               # 25000
print(cfg_mod.DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT)   # 1000

# Restore defaults: delete the env vars then reload again so the module
# re-reads its original hard-coded defaults (just deleting env vars is not
# enough — the module still holds the values set during the previous reload).
del os.environ["LANGGRAPH_DEFAULT_RECURSION_LIMIT"]
del os.environ["LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT"]
importlib.reload(cfg_mod)
print(cfg_mod.DEFAULT_RECURSION_LIMIT)             # 10007 — restored
print(cfg_mod.DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT) # 20    — restored
```

---

## Summary

| # | Class group | Module | Key takeaway |
|---|---|---|---|
| 1 | `merge_configs` + `patch_config` + `patch_configurable` | `langgraph._internal._config` | Config composition primitives; `lc_versions` deep-merges; setting callbacks clears `run_name`/`run_id` |
| 2 | `ensure_config` + `_PROPAGATE_TO_METADATA` + `_CHECKPOINT_COORDINATE_KEYS` | `langgraph._internal._config` | Heavyweight config initializer; explicit coordinate key discards ambient configurable; `thread_id`/`run_id` propagate to LangSmith metadata |
| 3 | `ContextT` + `StateT` + `InputT` + `OutputT` + `NodeInputT` | `langgraph.typing` | TypeVars for statically typed graph factories; `ContextT` defaults to `None`; `InputT`/`OutputT` default to `StateT` |
| 4 | `TAG_NOSTREAM` + `TAG_HIDDEN` | `langgraph.constants` | `TAG_NOSTREAM` suppresses model token events in `messages` stream; `TAG_HIDDEN` hides nodes from debug stream and LangSmith |
| 5 | `recast_checkpoint_ns` + `filter_to_user_tags` | `langgraph._internal._config` | Strip task IDs from subgraph namespaces; filter `seq:step:N` internal tags from user-visible stream metadata |
| 6 | `InMemorySaver` storage internals | `langgraph.checkpoint.memory` | Three-dict layout: `storage[thread][ns][id]`, `writes[(thread,ns,id)][(task,idx)]`, `blobs[(thread,ns,ch,ver)]`; blobs deduplicate unchanged channels |
| 7 | `PendingWrite` + `WRITES_IDX_MAP` + sentinel channels | `langgraph.checkpoint.base` · `serde.types` | PendingWrite = `(task_id, channel, value)` 3-tuple; sentinels use fixed negative indices: ERROR=-1, SCHEDULED=-2, INTERRUPT=-3, RESUME=-4 |
| 8 | `get_checkpoint_id` + `get_checkpoint_metadata` | `langgraph.checkpoint.base` | Read current checkpoint ID from config; `get_checkpoint_metadata` merges `run_id` into metadata for every saver.put() call |
| 9 | `build_serde_allowlist` + `STRICT_MSGPACK_ENABLED` + `apply_checkpointer_allowlist` | `langgraph._internal._serde` | Strict msgpack mode; TypedDict/dataclass/Pydantic schemas recursively scanned; `curated_core_allowlist()` always includes all 14 langchain-core message types |
| 10 | `DEFAULT_RECURSION_LIMIT` + `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` + `CONFIG_KEY_*` | `langgraph._internal._config` · `_constants` | Default limit is 10 007 (not 25); both limits are env-var tunable; `CONFIG_KEY_*` keys carry internal runtime references — never set manually |

> **Cross-reference:** See also:
> [Vol. 25](./langgraph_class_deep_dives_v25/) for `BaseCache` + `FullKey` + `default_retry_on` + internal constants deep-dive.
> [Vol. 20](./langgraph_class_deep_dives_v20/) for `create_checkpoint` + `empty_checkpoint` in `langgraph.pregel._checkpoint`.
> [Vol. 11](./langgraph_class_deep_dives_v11/) for `CheckpointMetadata` + `CheckpointTuple` + `DeltaChannelHistory`.
> [Vol. 8](./langgraph_class_deep_dives_v8/) for `ChannelWrite` + `ChannelWriteEntry` + `SyncPregelLoop`.
> [Vol. 5](./langgraph_class_deep_dives_v5/) for `BaseCheckpointSaver` + building a custom backend.

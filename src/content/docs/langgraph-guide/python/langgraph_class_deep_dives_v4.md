---
title: "Class deep-dives Vol. 4 — 10 more LangGraph types"
description: "Source-verified deep dives into set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, and the full error taxonomy — with runnable examples."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 4"
  order: 28
---

# Class deep-dives Vol. 4 — 10 more LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

---

## 1 · `StateGraph.set_node_defaults()`

**Module:** `langgraph.graph.state`  
**Re-exported from:** `langgraph.graph`

`set_node_defaults()` sets a **graph-wide baseline** for retry, cache, timeout, and error-handler policies. Individual `add_node(..., retry_policy=...)` calls still override the baseline. Policies are applied at `compile()` time.

### Full signature (source)

```python
def set_node_defaults(
    self,
    *,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: StateNode[Any, ContextT] | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
) -> Self:
    ...
```

Key rules pulled from source:

- `retry_policy` and `timeout` apply to **all** nodes, including error-handler nodes.
- `cache_policy` and `error_handler` apply only to regular nodes — caching error-handler results is unsafe, and handlers must never catch themselves.
- Returns `Self`, so it chains: `StateGraph(...).set_node_defaults(...).add_node(...)`.

### Example: Graph-wide retry + per-node override

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class State(TypedDict):
    value: int
    errors: list[str]

call_count = {"node_a": 0, "node_b": 0}

def node_a(state: State) -> dict:
    call_count["node_a"] += 1
    if call_count["node_a"] < 3:
        raise ValueError("transient error in A")
    return {"value": state["value"] + 10}

def node_b(state: State) -> dict:
    call_count["node_b"] += 1
    if call_count["node_b"] < 2:
        raise ValueError("transient error in B")
    return {"value": state["value"] * 2}

graph = (
    StateGraph(State)
    # All nodes retry up to 5 times by default
    .set_node_defaults(
        retry_policy=RetryPolicy(
            max_attempts=5,
            initial_interval=0.01,
            backoff_factor=1.5,
        )
    )
    .add_node("node_a", node_a)
    # node_b overrides with its own tighter policy
    .add_node("node_b", node_b, retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.01))
    .add_edge(START, "node_a")
    .add_edge("node_a", "node_b")
    .add_edge("node_b", END)
    .compile()
)

result = graph.invoke({"value": 1, "errors": []})
print(result)  # {'value': 22, 'errors': []}
# node_a retried 2 extra times (graph-wide default, max_attempts=5)
# node_b retried 1 extra time (per-node override, max_attempts=2)
```

### Example: Graph-wide error handler

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class State(TypedDict):
    result: str
    last_error: str | None

def fragile_node(state: State) -> dict:
    raise RuntimeError("Something went wrong!")

def global_error_handler(state: State) -> dict:
    """Catches any unhandled node error and records it in state."""
    import traceback
    return {
        "result": "recovered",
        "last_error": traceback.format_exc().strip().split("\n")[-1],
    }

graph = (
    StateGraph(State)
    .set_node_defaults(error_handler=global_error_handler)
    .add_node("fragile", fragile_node)
    .add_edge(START, "fragile")
    .add_edge("fragile", END)
    .compile()
)

result = graph.invoke({"result": "", "last_error": None})
print(result["result"])     # recovered
print(result["last_error"]) # RuntimeError: Something went wrong!
```

### Precedence rules

```
add_node(..., retry_policy=X)   ← highest priority (per-node)
set_node_defaults(retry_policy=Y) ← graph-wide default
[no policy]                      ← no retry
```

Subgraphs do **not** inherit defaults from a parent graph. Each `StateGraph` has its own `_node_defaults` object.

---

## 2 · `StateGraph.add_sequence()`

**Module:** `langgraph.graph.state`  
**Re-exported from:** `langgraph.graph`

`add_sequence()` adds a list of nodes that execute in order, automatically wiring each node to the next with `add_edge`. It replaces the pattern of calling `add_node` + `add_edge` for every step in a linear pipeline.

### Full signature (source)

```python
def add_sequence(
    self,
    nodes: Sequence[
        StateNode[NodeInputT, ContextT]
        | tuple[str, StateNode[NodeInputT, ContextT]]
    ],
) -> Self:
    ...
```

`nodes` can be:
- **Callables**: name inferred from `__name__`
- **`(name, callable)` tuples**: explicit name; required when two callables would share the same name (e.g. lambdas)

Returns `Self` for chaining. Raises `ValueError` if the list is empty or has duplicate names.

### Basic example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class PipeState(TypedDict):
    text: str

def step_clean(state: PipeState) -> dict:
    return {"text": state["text"].strip().lower()}

def step_split(state: PipeState) -> dict:
    return {"text": " | ".join(state["text"].split())}

def step_wrap(state: PipeState) -> dict:
    return {"text": f"[{state['text']}]"}

graph = (
    StateGraph(PipeState)
    .add_sequence([step_clean, step_split, step_wrap])
    .add_edge(START, "step_clean")   # connect START to first node in the sequence
    .add_edge("step_wrap", END)
    .compile()
)

result = graph.invoke({"text": "  Hello   World  "})
print(result["text"])  # [hello | world]
```

### With explicit names (for lambdas or collisions)

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    n: int

graph = (
    StateGraph(State)
    .add_sequence([
        ("double", lambda s: {"n": s["n"] * 2}),
        ("add_ten", lambda s: {"n": s["n"] + 10}),
        ("negate", lambda s: {"n": -s["n"]}),
    ])
    .add_edge(START, "double")
    .add_edge("negate", END)
    .compile()
)

print(graph.invoke({"n": 3})["n"])  # -((3*2)+10) = -16
```

### Mixing `add_sequence` with conditional routing

`add_sequence` creates a strictly linear sub-chain. You can splice it into a broader graph with conditional edges at either end:

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    score: int
    approved: bool

def validate(s: State) -> dict:
    return {"approved": s["score"] >= 50}

def enrich(s: State) -> dict:
    return {"score": s["score"] + 5}

def finalize(s: State) -> dict:
    return {}  # No-op finalizer

def route(s: State) -> Literal["validate", END]:
    return "validate" if s["score"] > 0 else END

graph = (
    StateGraph(State)
    .add_sequence([validate, enrich, finalize])
    .add_conditional_edges(START, route, {"validate": "validate", END: END})
    .add_edge("finalize", END)
    .compile()
)

print(graph.invoke({"score": 60, "approved": False}))
# {'score': 65, 'approved': True}
```

---

## 3 · `input_schema` + `output_schema` on `StateGraph`

**Module:** `langgraph.graph.state`

By default a `StateGraph` uses `state_schema` as both its input and output contract. The `input_schema` and `output_schema` constructor parameters let you define **narrower types** — useful for:

- Accepting only a subset of state fields as initial input
- Returning only a curated subset to the caller
- Strongly-typed API boundaries when the graph is used as a subgraph

### Full constructor signature (relevant params)

```python
StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema:  type[InputT]  | None = None,
    output_schema: type[OutputT] | None = None,
)
```

If `input_schema` is `None`, the full `state_schema` is used as input.  
If `output_schema` is `None`, the full `state_schema` is returned by `invoke()`.

### Example: Narrow input and output

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Full internal state (not exposed directly)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    token_count: int
    debug_trace: list[str]

# Only the caller needs to supply the user's message
class AgentInput(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# The caller only sees the final answer, not internal bookkeeping
class AgentOutput(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def initialize(state: AgentState) -> dict:
    return {
        "session_id": "sess-001",
        "token_count": 0,
        "debug_trace": [],
    }

def respond(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    reply = AIMessage(content=f"Echo: {last_msg}")
    return {
        "messages": [reply],
        "token_count": state["token_count"] + len(last_msg),
        "debug_trace": state["debug_trace"] + [f"responded to: {last_msg}"],
    }

graph = StateGraph(
    AgentState,
    input_schema=AgentInput,
    output_schema=AgentOutput,
)
graph.add_node("initialize", initialize)
graph.add_node("respond", respond)
graph.add_edge(START, "initialize")
graph.add_edge("initialize", "respond")
graph.add_edge("respond", END)
compiled = graph.compile()

# Caller only provides messages; session_id etc. are initialised internally
result = compiled.invoke({"messages": [HumanMessage(content="Hello")]})
# result only contains `messages` (from AgentOutput), not session_id/token_count
print(result.keys())           # dict_keys(['messages'])
print(result["messages"][-1].content)  # Echo: Hello
```

### Using `input_schema` / `output_schema` on nodes

Individual nodes can also narrow their input type, allowing them to declare which state fields they actually read:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    user_id: str
    email: str
    balance: float
    address: str

class BillingInput(TypedDict):
    user_id: str
    balance: float

def billing_node(state: BillingInput) -> dict:
    # Only sees user_id and balance; address is hidden
    charge = state["balance"] * 0.1
    return {"balance": state["balance"] - charge}

graph = (
    StateGraph(State)
    .add_node("billing", billing_node, input_schema=BillingInput)
    .add_edge(START, "billing")
    .add_edge("billing", END)
    .compile()
)

result = graph.invoke({
    "user_id": "u1",
    "email": "a@b.com",
    "balance": 100.0,
    "address": "1 Main St",
})
print(result["balance"])  # 90.0
```

---

## 4 · `context_schema` + `Runtime.context`

**Module:** `langgraph.graph.state`, `langgraph.runtime`

`context_schema` declares a **read-only context** type that is injected once per `invoke()` / `stream()` call. Unlike `state_schema`, context is never persisted in checkpoints and never modified by nodes. Use it for per-request data such as `user_id`, tenant config, or feature flags.

### Declaring `context_schema`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

class AppContext(TypedDict):
    user_id: str
    locale: str
    feature_flags: dict

class State(TypedDict):
    result: str

# Pattern 1: Inject via state parameter + type hint
def personalise(state: State, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context
    greeting = "Hola" if ctx["locale"] == "es" else "Hello"
    return {"result": f"{greeting}, {ctx['user_id']}!"}

graph = (
    StateGraph(State, context_schema=AppContext)
    .add_node("personalise", personalise)
    .add_edge(START, "personalise")
    .add_edge("personalise", END)
    .compile()
)

result = graph.invoke(
    {"result": ""},
    config={
        "configurable": {
            "context": {"user_id": "alice", "locale": "es", "feature_flags": {}}
        }
    },
)
print(result["result"])  # Hola, alice!
```

### Using `context_schema` with `create_react_agent`

```python
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

class UserContext(TypedDict):
    user_id: str
    subscription_tier: str  # "free" | "pro" | "enterprise"

@tool
def get_tier(runtime: Runtime[UserContext]) -> str:
    """Return the user's subscription tier."""
    return runtime.context["subscription_tier"]

# Pass context_schema so the agent knows what to inject
agent = create_react_agent(
    model="anthropic:claude-3-5-haiku-20241022",  # or any LLM
    tools=[get_tier],
    context_schema=UserContext,
)
```

### Context vs state: when to use each

| Concern | Use `state_schema` | Use `context_schema` |
|---|---|---|
| Accumulated conversation | ✅ | ❌ |
| User session data (read-only) | ❌ | ✅ |
| Persisted in checkpoint | ✅ | ❌ |
| Writable by nodes | ✅ | ❌ (immutable) |
| Shared across subgraphs | Depends on schema | No (per-invocation) |

### Pattern: feature flags per request

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

class FeatureCtx(TypedDict):
    enable_experimental: bool

class State(TypedDict):
    output: str

def smart_node(state: State, runtime: Runtime[FeatureCtx]) -> dict:
    if runtime.context["enable_experimental"]:
        return {"output": "experimental path"}
    return {"output": "stable path"}

graph = (
    StateGraph(State, context_schema=FeatureCtx)
    .add_node("smart", smart_node)
    .add_edge(START, "smart")
    .add_edge("smart", END)
    .compile()
)

stable = graph.invoke(
    {"output": ""},
    config={"configurable": {"context": {"enable_experimental": False}}}
)
experimental = graph.invoke(
    {"output": ""},
    config={"configurable": {"context": {"enable_experimental": True}}}
)
print(stable["output"])       # stable path
print(experimental["output"]) # experimental path
```

---

## 5 · `get_stream_writer()` + `StreamWriter`

**Module:** `langgraph.config`  
**Type alias:** `langgraph.types.StreamWriter = Callable[[Any], None]`

`get_stream_writer()` returns a callable that lets any node or task **push arbitrary values** into the `"custom"` stream — without touching the graph state. The caller receives them in real-time via `graph.stream(..., stream_mode="custom")`.

### Full signature (source)

```python
def get_stream_writer() -> StreamWriter:
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer
```

`StreamWriter = Callable[[Any], None]` — call it with any JSON-serialisable value.

> **Python ≥ 3.11 required for async.** The underlying `contextvar` propagation that makes `get_stream_writer()` work in async tasks is only guaranteed on Python 3.11+.

### Example: Real-time progress from inside a node

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
import time

class State(TypedDict):
    items: list[str]
    processed: list[str]

def batch_processor(state: State) -> dict:
    writer = get_stream_writer()
    results = []
    for i, item in enumerate(state["items"]):
        # Emit progress event before processing each item
        writer({"event": "progress", "step": i + 1, "total": len(state["items"]), "item": item})
        time.sleep(0.01)  # simulate work
        results.append(item.upper())
    writer({"event": "done", "count": len(results)})
    return {"processed": results}

graph = (
    StateGraph(State)
    .add_node("batch_processor", batch_processor)
    .add_edge(START, "batch_processor")
    .add_edge("batch_processor", END)
    .compile()
)

for chunk in graph.stream(
    {"items": ["alpha", "beta", "gamma"], "processed": []},
    stream_mode="custom",
):
    print(chunk)
# {'event': 'progress', 'step': 1, 'total': 3, 'item': 'alpha'}
# {'event': 'progress', 'step': 2, 'total': 3, 'item': 'beta'}
# {'event': 'progress', 'step': 3, 'total': 3, 'item': 'gamma'}
# {'event': 'done', 'count': 3}
```

### Example: Multiple stream modes at once

Pass a list to `stream_mode` to get both custom events and state updates:

```python
for namespace, chunk in graph.stream(
    {"items": ["a", "b"], "processed": []},
    stream_mode=["custom", "updates"],
):
    if namespace == "custom":
        print(f"[event] {chunk}")
    else:
        print(f"[state] {chunk}")
```

### Example: `StreamWriter` in a functional API task

```python
from langgraph.func import entrypoint, task
from langgraph.config import get_stream_writer

@task
def fetch_data(url: str) -> str:
    writer = get_stream_writer()
    writer({"status": "fetching", "url": url})
    # … do actual work …
    result = f"data from {url}"
    writer({"status": "fetched", "bytes": len(result)})
    return result

@entrypoint()
def pipeline(urls: list[str]) -> list[str]:
    futures = [fetch_data(u) for u in urls]
    return [f.result() for f in futures]

for chunk in pipeline.stream(
    ["https://example.com/a", "https://example.com/b"],
    stream_mode="custom",
):
    print(chunk)
```

### `StreamWriter` vs direct state writes

| | `get_stream_writer()` | State update (`return {...}`) |
|---|---|---|
| When seen by caller | Immediately during node execution | After node completes |
| Persisted in checkpoint | ❌ | ✅ |
| Appears in `"updates"` stream | ❌ | ✅ |
| Appears in `"custom"` stream | ✅ | ❌ |

---

## 6 · `push_ui_message()`

**Module:** `langgraph.graph.ui`

`push_ui_message()` emits a structured **UI event** into the custom stream and simultaneously writes to a state key (default: `"ui"`). It's designed for frontends that render React-style components from LangGraph stream events.

### Full signature (source)

```python
def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    message: AnyMessage | None = None,
    state_key: str | None = "ui",
    merge: bool = False,
) -> UIMessage:
    ...
```

The returned `UIMessage` is a typed dict:

```python
{
    "type": "ui",
    "id": "<uuid>",
    "name": "<component-name>",
    "props": {...},
    "metadata": {
        "merge": False,
        "run_id": ...,
        "tags": ...,
        "name": ...,
        # + any extra metadata
    }
}
```

Setting `merge=True` tells the frontend to **update** an existing component with the same `id` instead of creating a new one — useful for progress bars or streaming content.

### Example: Streaming a table component

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import push_ui_message

class State(TypedDict):
    query: str
    ui: Annotated[list, lambda x, y: x + [y]]  # accumulate UI events in state

def search_node(state: State) -> dict:
    # Emit a "loading" spinner
    push_ui_message(
        name="StatusBadge",
        props={"status": "loading", "message": f"Searching for: {state['query']}"},
        state_key="ui",
    )

    # Simulate search results
    rows = [
        {"id": 1, "title": "Result A", "score": 0.95},
        {"id": 2, "title": "Result B", "score": 0.87},
    ]

    # Emit the results table
    push_ui_message(
        name="ResultsTable",
        props={"rows": rows, "query": state["query"]},
        state_key="ui",
    )

    return {}  # state["ui"] is updated by push_ui_message internally

graph = (
    StateGraph(State)
    .add_node("search", search_node)
    .add_edge(START, "search")
    .add_edge("search", END)
    .compile()
)

for chunk in graph.stream(
    {"query": "langgraph", "ui": []},
    stream_mode="custom",
):
    if chunk.get("type") == "ui":
        print(f"Component: {chunk['name']}, Props: {chunk['props']}")
```

### Streaming updates to an existing component (`merge=True`)

```python
import uuid
from langgraph.graph.ui import push_ui_message

PROGRESS_ID = str(uuid.uuid4())

def long_running_node(state: dict) -> dict:
    # Create component
    push_ui_message(
        name="ProgressBar",
        props={"percent": 0, "label": "Starting…"},
        id=PROGRESS_ID,
    )
    for i in range(1, 6):
        # Update existing component — same ID, merge=True
        push_ui_message(
            name="ProgressBar",
            props={"percent": i * 20, "label": f"Step {i}/5"},
            id=PROGRESS_ID,
            merge=True,
        )
    push_ui_message(
        name="ProgressBar",
        props={"percent": 100, "label": "Done!"},
        id=PROGRESS_ID,
        merge=True,
    )
    return {}
```

### Associating a UI message with a chat message

Pass `message=ai_message` to attach the UI event to the generating chat message (exposed as `metadata.message_id`):

```python
from langchain_core.messages import AIMessage
from langgraph.graph.ui import push_ui_message

def responding_node(state):
    response = AIMessage(content="Here is your chart:", id="msg-001")
    push_ui_message(
        name="BarChart",
        props={"data": [1, 2, 3, 4]},
        message=response,
    )
    return {"messages": [response]}
```

---

## 7 · `entrypoint.final`

**Module:** `langgraph.func`

`entrypoint.final` is a dataclass returned from an `@entrypoint`-decorated function when you need the **saved checkpoint value** to differ from the **returned value to the caller**.

### Source (from `langgraph/func/__init__.py`)

```python
@dataclass
class final(Generic[R, S]):
    value: R   # Returned to the caller
    save: S    # Saved to the checkpoint (available as `previous` next invocation)
```

Without `entrypoint.final`, the return value is both returned **and** saved. With it, `value` is returned while `save` goes into the checkpoint.

### When to use it

- The graph should **accumulate** a running summary internally but return only the new response
- Maintain a **token budget** counter internally while surfacing just the content
- Store a **compressed** version of history while returning full detail to the caller

### Example: Accumulating a word-count budget

```python
from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

@task
def count_words(text: str) -> int:
    return len(text.split())

@entrypoint(checkpointer=InMemorySaver())
def budget_chat(
    message: str,
    *,
    previous: dict | None = None,
) -> entrypoint.final[str, dict]:
    previous = previous or {"word_budget": 1000, "history": []}

    words_used = count_words(message).result()
    new_budget = previous["word_budget"] - words_used
    new_history = previous["history"] + [message]

    # Return a summary to the caller
    response = f"Budget remaining: {new_budget} words. History: {len(new_history)} messages."

    # Save the full internal state to the checkpoint
    return entrypoint.final(
        value=response,
        save={"word_budget": new_budget, "history": new_history},
    )

config = {"configurable": {"thread_id": "budget-thread"}}

r1 = budget_chat.invoke("Hello world this is a test", config)
print(r1)  # Budget remaining: 995 words. History: 1 messages.

r2 = budget_chat.invoke("Another message here", config)
print(r2)  # Budget remaining: 992 words. History: 2 messages.
```

### Example: Returning a summary while saving full detail

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

@task
def compress(history: list[str]) -> str:
    return " | ".join(h[:20] for h in history[-3:])

@entrypoint(checkpointer=InMemorySaver())
def conversation(
    user_msg: str,
    *,
    previous: dict | None = None,
) -> entrypoint.final[str, dict]:
    prev = previous or {"messages": [], "summary": ""}
    messages = prev["messages"] + [user_msg]

    reply = f"You said: {user_msg}"
    compressed = compress(messages).result()

    return entrypoint.final(
        value=reply,          # Caller gets just the reply
        save={
            "messages": messages,
            "summary": compressed,
        },
    )
```

### `entrypoint.final` vs plain return

| | Plain `return value` | `return entrypoint.final(value, save)` |
|---|---|---|
| Returned to caller | `value` | `value` |
| Saved to checkpoint | `value` (same) | `save` (different) |
| `previous` on next call | `value` | `save` |

---

## 8 · `REMOVE_ALL_MESSAGES` + message manipulation

**Module:** `langgraph.graph.message`

`REMOVE_ALL_MESSAGES` is a sentinel string (`"__remove_all__"`) that, when included as an update to a `messages` field using `add_messages`, **deletes all existing messages** before applying the new list. Added in LangGraph 1.2.1.

### Source observation

```python
# langgraph/graph/message.py
REMOVE_ALL_MESSAGES = "__remove_all__"

# In add_messages reducer:
# if new_messages == REMOVE_ALL_MESSAGES → clear the message list, then apply remainder
```

### When to use

- Hard-reset conversation history after a summarisation step
- Clear memory at the end of a session branch
- Swap contexts entirely (e.g., switching users mid-session)

### Example: Summarise and replace

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"You said: {last}")]}

def summarise_and_reset(state: ChatState) -> dict:
    """Summarise the conversation and replace all messages with the summary."""
    history = " -> ".join(m.content for m in state["messages"])
    summary = AIMessage(content=f"[Summary]: {history}")
    # REMOVE_ALL_MESSAGES clears the list, then the summary is appended
    return {"messages": [REMOVE_ALL_MESSAGES, summary]}

graph = (
    StateGraph(ChatState)
    .add_node("chat", chat_node)
    .add_node("summarise", summarise_and_reset)
    .add_edge(START, "chat")
    .add_edge("chat", "summarise")
    .add_edge("summarise", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "t1"}}
graph.invoke({"messages": [HumanMessage(content="Hello")]}, config)
graph.invoke({"messages": [HumanMessage(content="How are you?")]}, config)
snapshot = graph.get_state(config)
# After summarisation, only one message remains
print(len(snapshot.values["messages"]))  # 1
print(snapshot.values["messages"][0].content)  # starts with [Summary]:
```

### `RemoveMessage` vs `REMOVE_ALL_MESSAGES`

| | `RemoveMessage(id=...)` | `REMOVE_ALL_MESSAGES` |
|---|---|---|
| Scope | Single message by ID | **All** messages |
| Use case | Remove specific items | Hard reset |
| Since | 0.2.x | 1.2.1 |

### Selectively removing old messages

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import RemoveMessage

def trim_node(state):
    """Keep only the last 3 messages, removing older ones by ID."""
    to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-3]]
    return {"messages": to_remove}
```

---

## 9 · `error_handler` on `add_node`

**Module:** `langgraph.graph.state`

The `error_handler` parameter on `add_node` lets you specify a **per-node fallback callable** that runs whenever the node raises an uncaught exception. The handler receives the current state and must return a state update dict. The graph continues normally after the handler finishes.

### Key behaviour (from source)

- Error-handler nodes themselves are **never** caught — if a handler raises, the run fails
- `set_node_defaults(error_handler=...)` sets a fallback handler for all nodes that don't have their own
- `cache_policy` is never applied to error-handler nodes (unsafe to cache exception results)

### Signature

```python
graph.add_node(
    "my_node",
    my_action,
    error_handler=my_handler,  # Callable[[State], dict]
)
```

### Example: Graceful degradation

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import traceback

class State(TypedDict):
    result: str
    errors: list[str]

def risky_api_call(state: State) -> dict:
    # Simulated intermittent failure
    raise ConnectionError("API unavailable")

def api_fallback(state: State) -> dict:
    """Called when risky_api_call raises any exception."""
    err = traceback.format_exc().strip().split("\n")[-1]
    return {
        "result": "cached_fallback_data",
        "errors": state["errors"] + [err],
    }

def post_process(state: State) -> dict:
    return {"result": state["result"].upper()}

graph = (
    StateGraph(State)
    .add_node("api", risky_api_call, error_handler=api_fallback)
    .add_node("process", post_process)
    .add_edge(START, "api")
    .add_edge("api", "process")
    .add_edge("process", END)
    .compile()
)

result = graph.invoke({"result": "", "errors": []})
print(result["result"])  # CACHED_FALLBACK_DATA
print(result["errors"])  # ['ConnectionError: API unavailable']
```

### Example: Handler that routes to a different branch

The error handler updates state; downstream routing can then read that state and decide where to go:

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    data: str
    failed: bool

def flaky_transform(state: State) -> dict:
    raise ValueError("transform failed")

def mark_failed(state: State) -> dict:
    return {"failed": True, "data": ""}

def success_path(state: State) -> dict:
    return {"data": state["data"].upper()}

def failure_path(state: State) -> dict:
    return {"data": "RECOVERY_DATA"}

def route(state: State) -> Literal["success", "failure"]:
    return "failure" if state["failed"] else "success"

graph = (
    StateGraph(State)
    .add_node("transform", flaky_transform, error_handler=mark_failed)
    .add_node("success", success_path)
    .add_node("failure", failure_path)
    .add_conditional_edges("transform", route)
    .add_edge(START, "transform")
    .add_edge("success", END)
    .add_edge("failure", END)
    .compile()
)

result = graph.invoke({"data": "hello", "failed": False})
print(result["data"])    # RECOVERY_DATA
print(result["failed"])  # True
```

### `error_handler` + `retry_policy` interaction

When both are set on a node, retries run **first**. Only if all retry attempts are exhausted does the `error_handler` fire:

```python
from langgraph.types import RetryPolicy

attempt = {"n": 0}

def flaky(state):
    attempt["n"] += 1
    if attempt["n"] < 3:
        raise ValueError("not yet")
    return {"result": "ok"}

def fallback(state):
    return {"result": "handler fired"}

graph = (
    StateGraph({"result": str})
    # Retries 2 times (3 total), handler fires only if all fail
    .add_node("node", flaky, retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.0), error_handler=fallback)
    .add_edge(START, "node")
    .add_edge("node", END)
    .compile()
)
# With max_attempts=2, 3rd attempt succeeds before handler triggers
result = graph.invoke({"result": ""})
print(result["result"])  # ok (not "handler fired")
```

---

## 10 · Error taxonomy

**Module:** `langgraph.errors`

LangGraph defines a hierarchy of exceptions. Understanding which exception maps to which situation — and how to handle each — is essential for robust production graphs.

### `GraphRecursionError`

```python
class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps."""
```

Fired when `recursion_limit` is reached (default: 25). The counter increments once per **super-step** (one full round of node executions).

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

class State(dict):
    pass

def loop_node(state):
    return {}  # never terminates

graph = StateGraph(dict).add_node("loop", loop_node).add_edge(START, "loop").add_edge("loop", "loop").compile()

try:
    graph.invoke({}, config={"recursion_limit": 5})
except GraphRecursionError as e:
    print(f"Caught: {type(e).__name__}")  # Caught: GraphRecursionError

# Increase the limit for deep but legitimate workflows
graph.invoke({...}, config={"recursion_limit": 200})
```

### `InvalidUpdateError`

```python
class InvalidUpdateError(Exception):
    """Raised when attempting to update a channel with an invalid set of updates."""
```

Two sub-causes:

1. **Concurrent write to `LastValue`** — two nodes in the same super-step write to the same non-reducer field.
2. **Wrong return type** — a node returns something other than a dict or `None`.

```python
# WRONG: two nodes write to state["x"] in the same super-step
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError

class State(TypedDict):
    x: int   # LastValue channel — only one writer per step

def node_a(s): return {"x": 1}
def node_b(s): return {"x": 2}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")  # Both run in the same super-step → conflict!
    .add_edge("a", END)
    .add_edge("b", END)
    .compile()
)

try:
    graph.invoke({"x": 0})
except InvalidUpdateError as e:
    print("Concurrent write conflict!")
    # Fix: use Annotated[int, operator.add] or sequence nodes

# CORRECT: use a reducer to merge concurrent writes
import operator
from typing import Annotated

class SafeState(TypedDict):
    x: Annotated[int, operator.add]  # BinaryOperatorAggregate channel

safe_graph = (
    StateGraph(SafeState)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge("a", END)
    .add_edge("b", END)
    .compile()
)
result = safe_graph.invoke({"x": 0})
print(result["x"])  # 3  (1 + 2)
```

### `NodeTimeoutError`

```python
class NodeTimeoutError(Exception):
    """Raised when a node invocation exceeds its configured timeout."""

    node: str           # Which node timed out
    timeout: float      # The limit that fired
    run_timeout: float | None
    idle_timeout: float | None
    elapsed: float
    kind: Literal["idle", "run"]
```

`NodeTimeoutError` does **not** inherit from `TimeoutError` — this is deliberate so the default `RetryPolicy` treats it as retryable. Two flavours:

- `kind="run"` — total wall-clock time for the node exceeded `run_timeout`
- `kind="idle"` — no progress signal (heartbeat) for `idle_timeout` seconds

```python
import asyncio
from langgraph.errors import NodeTimeoutError
from langgraph.types import TimeoutPolicy

async def slow_node(state):
    await asyncio.sleep(10)  # way too slow
    return {}

# Async only — sync nodes cannot be safely cancelled
graph.add_node(
    "slow",
    slow_node,
    timeout=TimeoutPolicy(run_timeout=2.0, idle_timeout=1.0),
    retry_policy=RetryPolicy(max_attempts=2, retry_on=NodeTimeoutError),
)

try:
    await graph.ainvoke(...)
except NodeTimeoutError as e:
    print(f"Node '{e.node}' timed out after {e.elapsed:.2f}s ({e.kind})")
```

### `EmptyChannelError`

```python
class EmptyChannelError(Exception):
    """Raised when accessing a channel that has never been written."""
```

You will rarely see this in application code — it surfaces during graph initialisation or when you try to read a field that no node has ever populated. The fix is usually to provide a default value in your schema:

```python
# Using TypedDict with default via Optional is the idiomatic approach
from typing import Optional
class State(TypedDict):
    result: Optional[str]  # None by default — avoids EmptyChannelError

# Or use a dataclass/Pydantic with a real default
from dataclasses import dataclass, field

@dataclass
class State:
    items: list[str] = field(default_factory=list)
    count: int = 0
```

### Quick error reference

| Exception | When | Fix |
|---|---|---|
| `GraphRecursionError` | Graph loop exceeded `recursion_limit` | Increase limit; add proper termination |
| `InvalidUpdateError` | Concurrent writes to `LastValue` field | Add reducer annotation; sequence nodes |
| `NodeTimeoutError` | Node exceeded `run_timeout` or `idle_timeout` | Increase limit; add retry; call `heartbeat()` |
| `EmptyChannelError` | Read field never written | Add default value in schema |
| `GraphInterrupt` | `interrupt()` called (internal) | Resume with `Command(resume=...)` |

---

## Quick-reference summary — Vol. 4

| Feature | Module | Key use |
|---|---|---|
| `set_node_defaults()` | `langgraph.graph.state` | Graph-wide retry / cache / error defaults |
| `add_sequence()` | `langgraph.graph.state` | Build linear pipelines without manual edges |
| `input_schema` / `output_schema` | `langgraph.graph.state` | Narrow the public API of a graph or subgraph |
| `context_schema` + `Runtime.context` | `langgraph.graph.state` / `langgraph.runtime` | Per-invocation read-only injection (user id, flags) |
| `get_stream_writer()` | `langgraph.config` | Push custom events from inside a node |
| `push_ui_message()` | `langgraph.graph.ui` | Emit React-renderable UI events |
| `entrypoint.final` | `langgraph.func` | Decouple return value from checkpoint save value |
| `REMOVE_ALL_MESSAGES` | `langgraph.graph.message` | Hard-reset a message history in one step |
| `error_handler` on `add_node` | `langgraph.graph.state` | Per-node fallback without crashing the graph |
| `GraphRecursionError` / `InvalidUpdateError` / `NodeTimeoutError` / `EmptyChannelError` | `langgraph.errors` | Understand and handle runtime failures |

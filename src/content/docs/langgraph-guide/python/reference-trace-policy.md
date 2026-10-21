---
title: "TracePolicy — per-node LangSmith tracing API reference"
description: "Control how individual graph nodes are traced in LangSmith with TracePolicy, omit_payload, and process_inputs/process_outputs processors — source-verified."
framework: "langgraph"
language: "python"
sidebar:
  label: "Ref · TracePolicy"
  order: 41
---

Verified against **`langgraph==%%version=langgraph%%`** (module: `langgraph.types`).

`TracePolicy` lets you control what each node records in LangSmith at a fine-grained level. Rather than blanket-hiding inputs/outputs across all runs, you can selectively truncate, redact, or silence individual nodes while leaving the rest fully traced.

---

## Class definition

```python
# langgraph.types
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(kw_only=True, slots=True, frozen=True)
class TracePolicy:
    """Configuration for how a node's run is traced."""

    process_inputs: Callable[[Any], Any] | None = None
    """Transform the node's input before recording it on the trace run."""

    process_outputs: Callable[[Any], Any] | None = None
    """Transform the node's output before recording it on the trace run."""
```

### `omit_payload` helper

```python
# langgraph.types
from typing import Any

def omit_payload(_value: Any) -> dict[str, Any]:
    """Record an empty payload, dropping the value entirely.

    Use as process_inputs and/or process_outputs on a TracePolicy to keep a
    node's span and its timing while omitting its inputs/outputs from the trace.
    """
    return {}
```

---

## Imports

```python
from langgraph.types import TracePolicy, omit_payload
```

---

## Scope and limitations

`TracePolicy` applies **only to the node's own run record**. It does not affect:
- Child runs created by a `bound` runnable traced inside the node
- The root graph run
- Sibling nodes

To redact across all runs and children, use the LangSmith client's `hide_inputs` / `hide_outputs` / `anonymizer` instead.

The filtered record is the node-level run (the run named after the node, which wraps the node callable and its state writers). Plain function nodes get this run too: `add_node` coerces the function with `trace=False`, which only suppresses an extra nested run for the callable itself. So `TracePolicy` works on a plain `def` node; wrapping it in `RunnableLambda` (as some examples below do) is optional. The root graph run still records the original, unredacted invocation input and output.

---

## Attaching a `TracePolicy` to a node

Pass `trace_policy=` when adding a node:

```python
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import TracePolicy, omit_payload

builder = StateGraph(dict)

def _my_node(state: dict) -> dict:
    return state

# RunnableLambda is optional here; a plain function node is traced too.
my_node = RunnableLambda(_my_node)

builder.add_node(
    "my_node",
    my_node,
    trace_policy=TracePolicy(
        process_inputs=omit_payload,   # record empty {} instead of inputs
        process_outputs=omit_payload,  # record empty {} instead of outputs
    ),
)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()
```

---

## Patterns

### Pattern 1: Silence a node completely

Drop both inputs and outputs from the trace while keeping the span timing:

```python
from langchain_core.runnables import RunnableLambda
from langgraph.types import TracePolicy, omit_payload

# RunnableLambda is optional here; a plain function node is traced too.
internal_router = RunnableLambda(router_fn)

builder.add_node(
    "internal_router",
    internal_router,
    trace_policy=TracePolicy(
        process_inputs=omit_payload,
        process_outputs=omit_payload,
    ),
)
```

### Pattern 2: Truncate large message history

Large `messages` lists inflate trace storage. Record only the last two messages:

```python
from typing import Any
from langchain_core.runnables import RunnableLambda
from langgraph.types import TracePolicy


def truncate_messages(value: Any) -> Any:
    """Keep only the last 2 messages for tracing; leave everything else as-is."""
    if isinstance(value, dict) and "messages" in value:
        return {**value, "messages": value["messages"][-2:]}
    return value


# RunnableLambda is optional here; a plain function node is traced too.
call_model = RunnableLambda(call_model_fn)
builder.add_node(
    "call_model",
    call_model,
    trace_policy=TracePolicy(
        process_inputs=truncate_messages,
        process_outputs=truncate_messages,
    ),
)
```

### Pattern 3: Redact sensitive fields

Strip PII or credentials from a node's inputs before they hit LangSmith:

```python
from typing import Any
from langchain_core.runnables import RunnableLambda
from langgraph.types import TracePolicy

SENSITIVE_KEYS = {"api_key", "password", "token", "ssn", "credit_card"}


def redact_sensitive(value: Any) -> Any:
    """Replace sensitive fields with [REDACTED]."""
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if k in SENSITIVE_KEYS else v
            for k, v in value.items()
        }
    return value


# RunnableLambda is optional here; a plain function node is traced too.
auth_node = RunnableLambda(auth_fn)
builder.add_node(
    "auth_node",
    auth_node,
    trace_policy=TracePolicy(process_inputs=redact_sensitive),
)
```

### Pattern 4: Summarize outputs for large embeddings

Embedding nodes produce large float arrays. Record metadata only:

```python
from typing import Any
from langchain_core.runnables import RunnableLambda
from langgraph.types import TracePolicy


def summarize_embedding_output(value: Any) -> Any:
    """Replace embedding vectors with their shape for trace readability."""
    if isinstance(value, dict) and "embedding" in value:
        vec = value["embedding"]
        return {**value, "embedding": f"<vector len={len(vec)}>"}
    return value


# RunnableLambda is optional here; a plain function node is traced too.
embed_node = RunnableLambda(embed_fn)
builder.add_node(
    "embed",
    embed_node,
    trace_policy=TracePolicy(process_outputs=summarize_embedding_output),
)
```

### Pattern 5: A shared default policy with per-node overrides

`set_node_defaults` does not take a `trace_policy` (in %%version=langgraph%% it accepts `retry_policy`, `cache_policy`, `error_handler` and `timeout` only), so apply a shared policy object to each node and override where needed:

```python
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import TracePolicy, omit_payload

builder = StateGraph(dict)

# Default: silence nodes that use this policy
SILENT = TracePolicy(process_inputs=omit_payload, process_outputs=omit_payload)
builder.add_node("fetch", fetch_fn, trace_policy=SILENT)
builder.add_node("transform", transform_fn, trace_policy=SILENT)

def verbose_trace(v):
    return v  # pass through — records everything

# Override for one critical node that you DO want traced
decision_node = RunnableLambda(decision_fn)
builder.add_node(
    "decision",
    decision_node,
    trace_policy=TracePolicy(
        process_inputs=verbose_trace,
        process_outputs=verbose_trace,
    ),
)
```

### Pattern 6: Different input vs output policy

Record what went in (for debugging) but not what came out (for compliance):

```python
from langchain_core.runnables import RunnableLambda
from langgraph.types import TracePolicy, omit_payload


def strip_pii(value):
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "user_data"}
    return value


# RunnableLambda is optional here; a plain function node is traced too.
pii_processor = RunnableLambda(pii_fn)
builder.add_node(
    "pii_processor",
    pii_processor,
    trace_policy=TracePolicy(
        process_inputs=strip_pii,    # record input minus PII
        process_outputs=omit_payload, # never record output
    ),
)
```

---

## Full working example

```python
from typing import Annotated, Any
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import TracePolicy, omit_payload
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    secret_key: str
    embedding: list[float]


def _sensitive_fn(state: State) -> dict:
    """Returns embeddings; secret_key is redacted in this node's span via TracePolicy."""
    return {"embedding": [0.1, 0.2, 0.3, 0.4]}


# RunnableLambda is optional here; a plain function node is traced too.
sensitive_runnable = RunnableLambda(_sensitive_fn)


def summarize_secret(value: Any) -> Any:
    """Redact secret_key from trace."""
    if isinstance(value, dict):
        return {k: ("***" if k == "secret_key" else v) for k, v in value.items()}
    return value


def summarize_embedding(value: Any) -> Any:
    """Summarize the embedding vector."""
    if isinstance(value, dict) and "embedding" in value:
        vec = value["embedding"]
        return {**value, "embedding": f"<float[{len(vec)}]>"}
    return value


builder = StateGraph(State)
builder.add_node(
    "sensitive",
    sensitive_runnable,
    trace_policy=TracePolicy(
        process_inputs=summarize_secret,
        process_outputs=summarize_embedding,
    ),
)
builder.add_edge(START, "sensitive")
builder.add_edge("sensitive", END)
graph = builder.compile()

result = graph.invoke(
    {
        "messages": [HumanMessage(content="hello")],
        "secret_key": "sk-actual-secret",
        "embedding": [],
    }
)
print(result["embedding"])  # [0.1, 0.2, 0.3, 0.4]
# In LangSmith: the "sensitive" node span shows secret_key=*** and embedding=<float[4]>.
# NOTE: TracePolicy only affects the node's own span. The root graph run still records
# the full invocation input, including "secret_key". To hide credentials from ALL traces
# (root run and children), use LangSmith-wide hide_inputs/anonymizer, or keep credentials
# out of graph state entirely.
```

---

## `TracePolicy` field reference

| Field | Type | Default | Description |
|---|---|---|---|
| `process_inputs` | `Callable[[Any], Any] \| None` | `None` | Transform the node's raw input before recording. Return value is what LangSmith records. |
| `process_outputs` | `Callable[[Any], Any] \| None` | `None` | Transform the node's raw output before recording. Return value is what LangSmith records. |

### `omit_payload` reference

```text
omit_payload(value: Any) -> dict[str, Any]
# Always returns {}
# Use as process_inputs or process_outputs to record nothing
```

---

## `TracePolicy` vs LangSmith client hide_inputs

| Feature | `TracePolicy` | LangSmith `hide_inputs`/`hide_outputs` |
|---|---|---|
| Scope | Per-node, node's own span only | All runs and children |
| Granularity | Field-level transform | All or nothing |
| Requires LangSmith SDK | No | Yes |
| Applies to child runs | No | Yes |
| Selectively blank one field | Yes (`process_inputs`) | No |

---

## Gotchas

- **Processors must not mutate their argument in place.** They receive the node's raw input/output by reference. Return a new object rather than modifying the existing one to avoid side effects on the actual execution.
- **`process_inputs` does not change what the node receives.** It only changes what is recorded in the trace. The node always gets the real input.
- **`process_outputs` does not change what the graph writes to state.** It only changes what is recorded in the trace.
- **`None` processor means "record as-is".** Passing `process_inputs=None` is the same as not setting it — the full value is recorded.
- **It filters the node-level run only.** Plain function nodes have one (verified: a `process_inputs` redaction shows up on the node run of a plain `def` node), but the root graph run and any child runs created inside the node keep the raw payloads.

---

## Version history

| Version | Change |
|---|---|
| 1.2.11 | `TracePolicy`, `omit_payload` production-stable |
| 1.2.0 | (An earlier revision claimed `set_node_defaults(trace_policy=...)` support here; `set_node_defaults` has no `trace_policy` parameter.) |
| 1.1.0 | `TracePolicy` first introduced |

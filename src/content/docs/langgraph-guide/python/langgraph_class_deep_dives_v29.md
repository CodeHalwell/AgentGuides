---
title: "Class deep-dives Vol. 29 — graph visualization, compiled-graph APIs, deferred nodes, delta-message reducer & add_messages internals (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: Edge/TriggerEdge/draw_graph (dry-run simulation that discovers conditional edges), get_graph(xray=)/draw_mermaid()/draw_mermaid_png() (visualization API with subgraph depth control), Pregel.as_tool() beta (graph-to-StructuredTool conversion), get_subgraphs()/aget_subgraphs() (recursive namespace traversal), get_input_jsonschema()/get_output_jsonschema()/get_context_jsonschema() (schema introspection for API integration), clear_cache()/aclear_cache() (per-namespace TTL invalidation on compiled graphs), _messages_delta_reducer (batch-safe DeltaChannel message accumulation, batching invariant), _add_messages_wrapper (partial-application decorator enabling format= specialisation), add_node(defer=True) (deferred super-step execution pattern), REMOVE_ALL_MESSAGES sentinel and bulk message deletion patterns."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 29"
  order: 60
---

# Class deep-dives Vol. 29 — graph visualization, compiled-graph APIs, deferred nodes, delta-message reducer & `add_messages` internals (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `Edge` + `TriggerEdge` + `draw_graph()` | `langgraph.pregel._draw` |
| 2 | `Pregel.get_graph(xray=)` + `Graph.draw_mermaid()` + `draw_mermaid_png()` | `langgraph.pregel.main` · `langchain_core.runnables.graph` |
| 3 | `Pregel.as_tool()` (beta) | `langgraph.pregel.main` |
| 4 | `Pregel.get_subgraphs()` + `aget_subgraphs()` | `langgraph.pregel.main` |
| 5 | `Pregel.get_input_jsonschema()` + `get_output_jsonschema()` + `get_context_jsonschema()` | `langgraph.pregel.main` |
| 6 | `Pregel.clear_cache()` + `aclear_cache()` | `langgraph.pregel.main` |
| 7 | `_messages_delta_reducer` | `langgraph.graph.message` |
| 8 | `_add_messages_wrapper` + partial `add_messages` | `langgraph.graph.message` |
| 9 | `StateGraph.add_node(defer=True)` | `langgraph.graph.state` |
| 10 | `REMOVE_ALL_MESSAGES` + bulk message deletion | `langgraph.graph.message` |

---

## 1 · `Edge` + `TriggerEdge` + `draw_graph()`

**Module**: `langgraph.pregel._draw`  
**First dedicated coverage.**

`draw_graph()` does not read static metadata — it **simulates a full dry-run** of the graph from an empty checkpoint to discover which edges are conditional and which are always-active. It returns a `langchain_core.runnables.graph.Graph` object (not a LangGraph graph) that is then used by every visualization method.

```python
class Edge(NamedTuple):
    source: str
    target: str
    conditional: bool      # True when the edge comes from add_conditional_edges
    data: str | None       # label string (e.g. the routing function name)

class TriggerEdge(NamedTuple):
    source: str            # which channel triggers the node
    conditional: bool
    data: str | None
```

Key implementation facts:
- `draw_graph()` accepts the raw `nodes`, `specs`, `input_channels`, `trigger_to_nodes`, and `interrupt_*_nodes` dicts directly from `Pregel` — it is not a method of any class.
- The simulation calls `prepare_next_tasks()` in a loop starting from an empty `Checkpoint`, recording every `ChannelWrite` it observes. Conditional edges produce `Edge(conditional=True)`.
- `interrupt_before_nodes` and `interrupt_after_nodes` become visual annotations, not separate nodes.
- The returned `Graph` object's `.nodes` is `dict[str, Node]` and `.edges` is `list[Edge]` — these are **langchain-core** types, not LangGraph types.
- The `limit=250` parameter caps simulation iterations to prevent infinite loops in cyclic graphs.

```python
# Example 1: inspect edges directly from a compiled graph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int
    step: str

def router(state: State) -> str:
    return "double" if state["value"] < 10 else END

graph = StateGraph(State)
graph.add_node("increment", lambda s: {"value": s["value"] + 1, "step": "inc"})
graph.add_node("double", lambda s: {"value": s["value"] * 2, "step": "dbl"})
graph.add_edge(START, "increment")
graph.add_conditional_edges("increment", router, ["double", END])
graph.add_edge("double", END)

compiled = graph.compile()
g = compiled.get_graph()

for edge in g.edges:
    print(f"{edge.source} -> {edge.target}  conditional={edge.conditional}  label={edge.data!r}")
# __start__ -> increment  conditional=False  label=None
# increment -> double     conditional=True   label=None
# increment -> __end__    conditional=True   label=None
# double -> __end__       conditional=False  label=None
```

```python
# Example 2: counting conditional vs static edges
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    x: int

def route(s): return "b" if s["x"] % 2 == 0 else "c"

g = StateGraph(S)
g.add_node("a", lambda s: {"x": s["x"] + 1})
g.add_node("b", lambda s: {"x": s["x"] * 2})
g.add_node("c", lambda s: {"x": s["x"] * 3})
g.add_edge(START, "a")
g.add_conditional_edges("a", route, ["b", "c"])
g.add_edge("b", END)
g.add_edge("c", END)
compiled = g.compile()

graph_repr = compiled.get_graph()
static    = [e for e in graph_repr.edges if not e.conditional]
cond      = [e for e in graph_repr.edges if e.conditional]
print(f"static={len(static)}  conditional={len(cond)}")
# static=3  conditional=2
```

```python
# Example 3: using graph metadata to build a runtime routing table
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    phase: str

def pick(s): return s["phase"]

g = StateGraph(S)
for name in ["extract", "transform", "load"]:
    g.add_node(name, lambda s, n=name: {"phase": n})
g.add_edge(START, "extract")
g.add_conditional_edges("extract", pick, ["transform", "load"])
g.add_edge("transform", "load")
g.add_edge("load", END)
compiled = g.compile()

# Build a routing table from the graph metadata
routing_table: dict[str, list[str]] = {}
for edge in compiled.get_graph().edges:
    routing_table.setdefault(edge.source, []).append(edge.target)

print(routing_table)
# {'__start__': ['extract'], 'extract': ['transform', 'load'],
#  'transform': ['load'], 'load': ['__end__']}
```

---

## 2 · `Pregel.get_graph(xray=)` + `Graph.draw_mermaid()` + `draw_mermaid_png()`

**Module**: `langgraph.pregel.main` · `langchain_core.runnables.graph`  
**First dedicated coverage.**

`get_graph(xray=False)` returns a shallow `Graph` with compiled subgraphs shown as single nodes. Passing `xray=True` (or an integer depth) **expands subgraphs inline** — each subgraph's internal nodes are prefixed with `<parent_node>:` and wired to the surrounding graph's edges.

```python
def get_graph(
    self,
    config: RunnableConfig | None = None,
    *,
    xray: int | bool = False,
) -> Graph:
    """Return a langchain_core Graph object suitable for visualization."""
```

- `xray=True` is equivalent to `xray=1` which recurses one level; `xray=2` recurses two levels, etc.
- `graph.draw_mermaid()` returns a Mermaid markdown string. The `classDef first/last/default` blocks and `flowchart TD` header are always included.
- `graph.draw_mermaid_png()` requires `pillow` and `cairosvg` (or a `MermaidDrawMethod` — currently defaults to `api` which calls the Mermaid.ink API over HTTPS).
- The `Graph` object is from `langchain_core`, not LangGraph. Its `.nodes` dict maps node IDs to `Node(id, data, metadata)` instances.

```python
# Example 1: draw_mermaid for docs / README
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def inc(s): return {"count": s["count"] + 1}
def check(s): return END if s["count"] >= 3 else "increment"

g = StateGraph(State)
g.add_node("increment", inc)
g.add_edge(START, "increment")
g.add_conditional_edges("increment", check, ["increment", END])
compiled = g.compile()

mermaid_md = compiled.get_graph().draw_mermaid()
print(mermaid_md)
# ---
# config:
#   flowchart:
#     curve: linear
# ---
# graph TD;
#     __start__([<p>__start__</p>]):::first
#     increment(increment)
#     __end__([<p>__end__</p>]):::last
#     __start__ --> increment;
#     increment --> increment;
#     increment --> __end__;
#     ...
```

```python
# Example 2: shallow vs deep (xray) view of subgraphs
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    x: int

# Build a reusable subgraph
sub = StateGraph(S)
sub.add_node("sub_step", lambda s: {"x": s["x"] * 2})
sub.set_entry_point("sub_step")
sub.set_finish_point("sub_step")
compiled_sub = sub.compile()

# Main graph uses the subgraph as a node
main = StateGraph(S)
main.add_node("subgraph", compiled_sub)
main.add_node("post", lambda s: {"x": s["x"] + 100})
main.add_edge(START, "subgraph")
main.add_edge("subgraph", "post")
main.add_edge("post", END)
compiled_main = main.compile()

shallow = compiled_main.get_graph()
deep    = compiled_main.get_graph(xray=True)

print("Shallow nodes:", list(shallow.nodes.keys()))
# ['__start__', 'subgraph', 'post', '__end__']

print("Deep nodes:", list(deep.nodes.keys()))
# ['__start__', '__end__', 'subgraph:sub_step', 'post']
```

```python
# Example 3: save Mermaid markdown to file for CI documentation
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import pathlib

class PipelineState(TypedDict):
    raw: str
    processed: str

g = StateGraph(PipelineState)
g.add_node("extract", lambda s: {"raw": s["raw"].strip()})
g.add_node("transform", lambda s: {"processed": s["raw"].upper()})
g.add_node("load", lambda s: s)  # passthrough
g.add_edge(START, "extract")
g.add_edge("extract", "transform")
g.add_edge("transform", "load")
g.add_edge("load", END)
compiled = g.compile()

mermaid = compiled.get_graph().draw_mermaid()
pathlib.Path("/tmp/pipeline.mmd").write_text(mermaid)
print("Saved Mermaid diagram")
# Convert to PNG: `mmdc -i /tmp/pipeline.mmd -o /tmp/pipeline.png`
```

---

## 3 · `Pregel.as_tool()` (beta)

**Module**: `langgraph.pregel.main`  
**First dedicated coverage.** Decorated `@beta` — API may change.

`as_tool()` wraps a compiled graph in a LangChain `StructuredTool` so it can be passed to any tool-calling model or included in a `ToolNode`. The tool's `args_schema` is inferred from the graph's input schema, but can be overridden.

```python
@beta_decorator.beta(message="This API is in beta and may change in the future.")
def as_tool(
    self,
    args_schema: type[BaseModel] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool:
```

- Internally delegates to `langchain_core.tools.convert_runnable_to_tool()`.
- The `name` defaults to the graph's `name` attribute (set automatically from the function name for `@entrypoint` graphs).
- `arg_types` lets you specify only the relevant keys of a dict-typed input — the rest are ignored by the schema.
- Calling the returned tool calls `compiled.invoke()` with the tool arguments. The tool is synchronous; use `atool.ainvoke()` for async.

```python
# Example 1: expose a compiled subgraph as a tool for a parent agent
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

class SubState(TypedDict):
    query: str
    result: str

def search(s: SubState) -> dict:
    # simulate a search tool
    return {"result": f"Results for: {s['query']}"}

sub = StateGraph(SubState)
sub.add_node("search", search)
sub.add_edge(START, "search")
sub.add_edge("search", END)
compiled_sub = sub.compile()

search_tool = compiled_sub.as_tool(
    name="web_search",
    description="Search the web and return relevant results",
)
print("Tool name:", search_tool.name)
print("Tool schema:", search_tool.args_schema.model_fields.keys())
# Tool name: web_search
# Tool schema: dict_keys(['query', 'result'])
```

```python
# Example 2: as_tool with explicit args_schema to expose only the input key
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class WorkerState(TypedDict):
    input_text: str
    tokens: list[str]
    summary: str

class WorkerInput(BaseModel):
    """Tokenize and summarize text."""
    input_text: str = Field(description="The text to process")

def tokenize(s): return {"tokens": s["input_text"].split()}
def summarize(s): return {"summary": f"{len(s['tokens'])} tokens"}

g = StateGraph(WorkerState)
g.add_node("tokenize", tokenize)
g.add_node("summarize", summarize)
g.add_edge(START, "tokenize")
g.add_edge("tokenize", "summarize")
g.add_edge("summarize", END)
compiled = g.compile()

tool = compiled.as_tool(
    args_schema=WorkerInput,
    name="text_processor",
    description="Tokenize and summarize text, return token count",
)
result = tool.invoke({"input_text": "LangGraph is a powerful framework"})
print("Summary:", result["summary"])
# Summary: 6 tokens
```

```python
# Example 3: embed a graph tool into a ToolNode for multi-agent use
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode
import operator

class PricingState(TypedDict):
    product_id: str
    price: float

def lookup_price(s: PricingState) -> dict:
    prices = {"widget": 9.99, "gadget": 24.99, "doohickey": 4.49}
    return {"price": prices.get(s["product_id"], 0.0)}

price_graph = StateGraph(PricingState)
price_graph.add_node("lookup", lookup_price)
price_graph.add_edge(START, "lookup")
price_graph.add_edge("lookup", END)
compiled_price = price_graph.compile()

price_tool = compiled_price.as_tool(
    name="get_price",
    description="Get the price for a product by ID",
)

# The tool can now be passed to ToolNode, create_react_agent, etc.
tool_node = ToolNode([price_tool])
print("ToolNode tools:", [t.name for t in tool_node.tools_by_name.values()])
# ToolNode tools: ['get_price']
```

---

## 4 · `Pregel.get_subgraphs()` + `aget_subgraphs()`

**Module**: `langgraph.pregel.main`  
**First dedicated coverage.**

`get_subgraphs()` iterates every node and yields `(namespace, subgraph)` pairs for nodes whose `node.subgraphs` list is non-empty. With `recurse=True` it descends transitively, prefixing namespaces with `NS_SEP` (`:`) to form hierarchical paths like `"outer:inner:leaf"`.

```python
def get_subgraphs(
    self,
    *,
    namespace: str | None = None,
    recurse: bool = False,
) -> Iterator[tuple[str, PregelProtocol]]:
```

- `namespace=None` (default) yields all immediate subgraphs.
- `namespace="some_node"` returns only that named subgraph (and exits early via `return`).
- `recurse=True` calls `graph.get_subgraphs()` on each discovered subgraph, prepending the parent name.
- The yielded type is `PregelProtocol` — in practice always a `CompiledStateGraph` or `CompiledGraph`, which exposes `invoke`, `stream`, `get_state`, etc.
- `aget_subgraphs()` is the `async def` mirror; it delegates to `get_subgraphs()` via `yield from`.

```python
# Example 1: list all immediate subgraphs of a multi-agent graph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    x: int

def make_subgraph(multiplier: int):
    g = StateGraph(S)
    g.add_node("op", lambda s, m=multiplier: {"x": s["x"] * m})
    g.set_entry_point("op")
    g.set_finish_point("op")
    return g.compile()

main = StateGraph(S)
main.add_node("double",  make_subgraph(2))
main.add_node("triple",  make_subgraph(3))
main.add_edge(START, "double")
main.add_edge("double", "triple")
main.add_edge("triple", END)
compiled = main.compile()

for name, sg in compiled.get_subgraphs():
    result = sg.invoke({"x": 4})
    print(f"  subgraph '{name}': invoke(x=4) → {result}")
# subgraph 'double': invoke(x=4) → {'x': 8}
# subgraph 'triple': invoke(x=4) → {'x': 12}
```

```python
# Example 2: recurse=True to navigate deeply nested subgraphs
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    x: int

leaf = StateGraph(S)
leaf.add_node("leaf_op", lambda s: {"x": s["x"] + 1000})
leaf.set_entry_point("leaf_op")
leaf.set_finish_point("leaf_op")
compiled_leaf = leaf.compile()

mid = StateGraph(S)
mid.add_node("leaf_wrapper", compiled_leaf)
mid.set_entry_point("leaf_wrapper")
mid.set_finish_point("leaf_wrapper")
compiled_mid = mid.compile()

top = StateGraph(S)
top.add_node("mid_wrapper", compiled_mid)
top.set_entry_point("mid_wrapper")
top.set_finish_point("mid_wrapper")
compiled_top = top.compile()

print("Shallow subgraphs:")
for ns, _ in compiled_top.get_subgraphs():
    print(f"  {ns!r}")

print("Deep subgraphs (recurse=True):")
for ns, _ in compiled_top.get_subgraphs(recurse=True):
    print(f"  {ns!r}")
# Shallow subgraphs:
#   'mid_wrapper'
# Deep subgraphs (recurse=True):
#   'mid_wrapper'
#   'mid_wrapper:leaf_wrapper'
```

```python
# Example 3: fetch a specific named subgraph by namespace
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    value: int

def make_node(n: int):
    g = StateGraph(S)
    g.add_node("work", lambda s, n=n: {"value": s["value"] + n})
    g.set_entry_point("work")
    g.set_finish_point("work")
    return g.compile()

main = StateGraph(S)
main.add_node("add_10", make_node(10))
main.add_node("add_20", make_node(20))
main.add_edge(START, "add_10")
main.add_edge("add_10", "add_20")
main.add_edge("add_20", END)
compiled = main.compile()

# Retrieve only the "add_20" subgraph
pairs = list(compiled.get_subgraphs(namespace="add_20"))
if pairs:
    ns, sg = pairs[0]
    print(f"Found '{ns}':", sg.invoke({"value": 0}))
# Found 'add_20': {'value': 20}
```

---

## 5 · `Pregel.get_input_jsonschema()` + `get_output_jsonschema()` + `get_context_jsonschema()`

**Module**: `langgraph.pregel.main`  
**First dedicated coverage.**

These three methods return standard JSON Schema `dict` objects derived from the graph's declared schemas. They are the correct integration points for OpenAPI/FastAPI route generation, form auto-rendering, and API gateway validation.

```python
def get_input_jsonschema(self) -> dict[str, Any]:   # from Runnable base
def get_output_jsonschema(self) -> dict[str, Any]:  # from Runnable base
def get_context_jsonschema(self) -> dict[str, Any]: # LangGraph-specific
# DEPRECATED: get_config_jsonschema() — use get_context_jsonschema()
```

- `get_input_jsonschema()` delegates to `get_input_schema().model_json_schema()`, which reflects the `state_schema` (or `input_schema` if set).
- `get_output_jsonschema()` reflects the `state_schema` (or `output_schema` if set).
- `get_context_jsonschema()` reflects the `context_schema` passed to `StateGraph(context_schema=...)`.
- `get_config_jsonschema()` is deprecated since v1.0 — it returns a schema wrapping context in `configurable`, which was the pre-v1.0 mechanism. Use `get_context_jsonschema()` instead.
- All three return `{}` (empty schema) when the corresponding schema is `Any` or not declared.

```python
# Example 1: generate OpenAPI-compatible schemas from a graph
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import json

class InvoiceState(TypedDict):
    vendor: str
    amount: float
    approved: bool

class ReviewContext(BaseModel):
    reviewer_id: str = Field(description="ID of the reviewing user")
    budget_limit: float = Field(description="Maximum auto-approval amount")

class InvoiceInput(TypedDict):
    vendor: str
    amount: float

g = StateGraph(
    InvoiceState,
    input_schema=InvoiceInput,
    context_schema=ReviewContext,
)
g.add_node("review", lambda s: {"approved": s["amount"] < 1000})
g.set_entry_point("review")
g.set_finish_point("review")
compiled = g.compile()

print("Input schema:")
print(json.dumps(compiled.get_input_jsonschema(), indent=2))
# {"properties": {"vendor": {"title": "Vendor", "type": "string"},
#                 "amount": {"title": "Amount", "type": "number"}}, ...}

print("\nContext schema:")
print(json.dumps(compiled.get_context_jsonschema(), indent=2))
# {"properties": {"reviewer_id": {...}, "budget_limit": {...}}, ...}
```

```python
# Example 2: use schemas to validate inputs before invoking
from typing_extensions import TypedDict
from pydantic import BaseModel, ValidationError, create_model
from langgraph.graph import StateGraph, START, END
import json

class OrderState(TypedDict):
    item_id: str
    quantity: int
    total: float

g = StateGraph(OrderState)
g.add_node("price", lambda s: {"total": s["quantity"] * 9.99})
g.set_entry_point("price")
g.set_finish_point("price")
compiled = g.compile()

schema = compiled.get_input_jsonschema()
InputModel = create_model("InputModel", **{
    k: (v.get("type", "string"), ...) for k, v in schema.get("properties", {}).items()
})

try:
    raw_input = {"item_id": "abc", "quantity": 3, "total": 0.0}
    validated = InputModel(**raw_input)
    result = compiled.invoke(dict(validated))
    print("Result:", result)
except ValidationError as e:
    print("Validation failed:", e)
# Result: {'item_id': 'abc', 'quantity': 3, 'total': 29.97}
```

```python
# Example 3: auto-register graph schemas with a FastAPI app
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

class TranslationState(TypedDict):
    text: str
    language: str
    translated: str

class TranslationContext(BaseModel):
    api_key: str
    model: str = "gpt-4o"

g = StateGraph(TranslationState, context_schema=TranslationContext)
g.add_node("translate", lambda s: {"translated": f"[{s['language']}] {s['text']}"})
g.set_entry_point("translate")
g.set_finish_point("translate")
compiled = g.compile()

# Schema objects you'd pass to FastAPI route generation
schemas = {
    "input":   compiled.get_input_jsonschema(),
    "output":  compiled.get_output_jsonschema(),
    "context": compiled.get_context_jsonschema(),
}
for name, schema in schemas.items():
    props = list(schema.get("properties", {}).keys())
    print(f"{name}: {props}")
# input:   ['text', 'language', 'translated']
# output:  ['text', 'language', 'translated']
# context: ['api_key', 'model']
```

---

## 6 · `Pregel.clear_cache()` + `aclear_cache()`

**Module**: `langgraph.pregel.main`  
**First dedicated coverage.**

`clear_cache()` invalidates **all** cached task and entrypoint results stored in the graph's `InMemoryCache` (or any `BaseCache` implementation). Passing `nodes=["node_name"]` (or using the per-task `task.clear_cache()`) allows **namespace-scoped** clearing.

```python
def clear_cache(self, nodes: Sequence[str] | None = None) -> None:
    """Clear the cache for the graph, or for specific nodes."""
async def aclear_cache(self, nodes: Sequence[str] | None = None) -> None:
    """Async version of clear_cache."""
```

- Internally calls `cache.clear(namespaces)` where `namespaces` is a list of `(CACHE_NS_WRITES, node_identifier)` tuples.
- When `nodes=None`, **all** namespaces registered to this graph are cleared, not the entire `BaseCache` object (other graphs sharing the cache are unaffected).
- For `@task`-decorated functions, use `task_func.clear_cache(cache)` or `await task_func.aclear_cache(cache)` — these are convenience wrappers for the same underlying mechanism.
- `clear_cache()` is a no-op if no `cache=` was passed at compile time or if all entries have already expired via TTL.

```python
# Example 1: clear cache between test runs to force re-execution
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

call_count = 0

class S(TypedDict):
    n: int
    result: int

def expensive_node(s: S) -> dict:
    global call_count
    call_count += 1
    return {"result": s["n"] ** 2}

cache = InMemoryCache()
g = StateGraph(S)
g.add_node("compute", expensive_node, cache_policy=CachePolicy())
g.set_entry_point("compute")
g.set_finish_point("compute")
compiled = g.compile(cache=cache)

compiled.invoke({"n": 7, "result": 0})
print(f"After 1st call: call_count={call_count}")  # 1

compiled.invoke({"n": 7, "result": 0})
print(f"After 2nd call (cached): call_count={call_count}")  # still 1

compiled.clear_cache()
compiled.invoke({"n": 7, "result": 0})
print(f"After clear + 3rd call: call_count={call_count}")  # 2
```

```python
# Example 2: clear only a specific node's cache while preserving others
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

fetch_count = 0
transform_count = 0

class S(TypedDict):
    raw: str
    cleaned: str
    summary: str

def fetch(s):
    global fetch_count; fetch_count += 1
    return {"raw": "data from API"}

def transform(s):
    global transform_count; transform_count += 1
    return {"cleaned": s["raw"].upper(), "summary": f"len={len(s['raw'])}"}

cache = InMemoryCache()
g = StateGraph(S)
g.add_node("fetch",     fetch,     cache_policy=CachePolicy())
g.add_node("transform", transform, cache_policy=CachePolicy())
g.add_edge(START, "fetch")
g.add_edge("fetch", "transform")
g.add_edge("transform", END)
compiled = g.compile(cache=cache)

compiled.invoke({"raw": "", "cleaned": "", "summary": ""})
print(f"1st: fetch={fetch_count} transform={transform_count}")  # 1 1

compiled.invoke({"raw": "", "cleaned": "", "summary": ""})
print(f"2nd: fetch={fetch_count} transform={transform_count}")  # 1 1

# Clear only the transform node
compiled.clear_cache(nodes=["transform"])
compiled.invoke({"raw": "", "cleaned": "", "summary": ""})
print(f"3rd: fetch={fetch_count} transform={transform_count}")  # 1 2
```

```python
# Example 3: async cache clearing in an async workflow
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

class S(TypedDict):
    query: str
    result: str

hits = 0

async def async_lookup(s: S) -> dict:
    global hits; hits += 1
    await asyncio.sleep(0.01)  # simulate I/O
    return {"result": f"answer to '{s['query']}'"}

cache = InMemoryCache()
g = StateGraph(S)
g.add_node("lookup", async_lookup, cache_policy=CachePolicy(ttl=60))
g.set_entry_point("lookup")
g.set_finish_point("lookup")
compiled = g.compile(cache=cache)

async def main():
    await compiled.ainvoke({"query": "weather", "result": ""})
    await compiled.ainvoke({"query": "weather", "result": ""})
    print(f"Hits before clear: {hits}")  # 1
    await compiled.aclear_cache()
    await compiled.ainvoke({"query": "weather", "result": ""})
    print(f"Hits after clear: {hits}")   # 2

asyncio.run(main())
```

---

## 7 · `_messages_delta_reducer`

**Module**: `langgraph.graph.message`  
**First dedicated coverage.** Marked **Experimental** in the source.

`_messages_delta_reducer` is a **batch-safe** message accumulation function designed for use with `DeltaChannel`. Unlike `add_messages`, which processes one write at a time, `_messages_delta_reducer` receives the full current state **plus a list of all writes** from a super-step in a single call — making it **batching-invariant**.

```python
def _messages_delta_reducer(
    state: list[AnyMessage],
    writes: list[list[AnyMessage]],
) -> list[AnyMessage]:
```

The **batching invariant** means:  
`reducer(reducer(state, xs), ys) == reducer(state, xs + ys)`

This is required by `DeltaChannel`, which may coalesce writes from multiple nodes before calling the reducer. Standard `add_messages` does NOT satisfy this invariant.

Key differences from `add_messages`:
- Does NOT handle `REMOVE_ALL_MESSAGES` — use `add_messages` for that.
- Does NOT auto-assign UUIDs to messages without IDs.
- Does NOT convert `BaseMessageChunk` via `message_chunk_to_message`.
- DOES coerce raw dicts/strings/tuples via `convert_to_messages`.
- Uses a fast path when `state[0]` is already a `BaseMessage` (skips re-conversion).

```python
# Example 1: use _messages_delta_reducer with DeltaChannel for batching
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.graph.message import _messages_delta_reducer
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]

def node_a(s): return {"messages": [AIMessage(content="Hello from A", id="a1")]}
def node_b(s): return {"messages": [AIMessage(content="Hello from B", id="b1")]}

g = StateGraph(State)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)
compiled = g.compile()

state = compiled.invoke({
    "messages": [HumanMessage(content="start", id="h1")]
})
print([m.content for m in state["messages"]])
# ['start', 'Hello from A', 'Hello from B']
```

```python
# Example 2: demonstrate the batching invariant directly
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import _messages_delta_reducer

state = [HumanMessage(content="q", id="1")]
xs = [[AIMessage(content="a1", id="2")]]
ys = [[AIMessage(content="a2", id="3")]]

# Two separate calls
step1 = _messages_delta_reducer(state, xs)
step2 = _messages_delta_reducer(step1, ys)

# One batched call
combined = _messages_delta_reducer(state, xs + ys)

print("Sequential:", [m.content for m in step2])
print("Batched:   ", [m.content for m in combined])
# Both: ['q', 'a1', 'a2'] — batching invariant holds
```

```python
# Example 3: in-place message updates via ID matching
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import _messages_delta_reducer

initial = [
    HumanMessage(content="original", id="msg-1"),
    AIMessage(content="first reply", id="msg-2"),
]

# Update msg-1 in place, add msg-3
writes = [[
    HumanMessage(content="edited question", id="msg-1"),
    AIMessage(content="follow-up", id="msg-3"),
]]

result = _messages_delta_reducer(initial, writes)
for m in result:
    print(f"{m.id}: {m.content}")
# msg-1: edited question   (updated in place)
# msg-2: first reply       (unchanged)
# msg-3: follow-up         (appended)
```

---

## 8 · `_add_messages_wrapper` + partial `add_messages`

**Module**: `langgraph.graph.message`  
**First dedicated coverage.**

`_add_messages_wrapper` is the internal decorator that transforms `add_messages` from a plain two-argument function into a **curriable reducer**. When called with both `left` and `right` it behaves normally; when called with neither argument (or with keyword-only `format=`) it returns a `functools.partial` — making it usable as an `Annotated` reducer that carries configuration.

```python
def _add_messages_wrapper(func: Callable) -> Callable[[Messages, Messages], Messages]:
    def _add_messages(
        left: Messages | None = None,
        right: Messages | None = None,
        **kwargs: Any,
    ) -> Messages | Callable[[Messages, Messages], Messages]:
        if left is not None and right is not None:
            return func(left, right, **kwargs)       # normal reduction call
        elif left is not None or right is not None:
            raise ValueError(...)                    # exactly one arg: error
        else:
            return partial(func, **kwargs)           # zero args: return partial
```

- `add_messages` (the public symbol) is the result of `_add_messages_wrapper(original_add_messages)`.
- `add_messages(format="langchain-openai")` returns a `functools.partial` bound to `format="langchain-openai"`.
- That partial is a valid reducer function and can be used directly in `Annotated[list, add_messages(format="langchain-openai")]`.
- Calling `add_messages()` (no arguments) also returns a `partial` with no extra bindings — equivalent to `add_messages` itself.

```python
# Example 1: use format="langchain-openai" as a TypedDict reducer
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# The reducer is a partial that normalises content blocks
class ChatState(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]

def chatbot(s: ChatState) -> dict:
    return {"messages": [AIMessage(content="Hi!", id="r1")]}

g = StateGraph(ChatState)
g.add_node("bot", chatbot)
g.set_entry_point("bot")
g.set_finish_point("bot")
compiled = g.compile()

result = compiled.invoke({
    "messages": [HumanMessage(content="Hello", id="u1")]
})
print([type(m).__name__ for m in result["messages"]])
# ['HumanMessage', 'AIMessage']
```

```python
# Example 2: verify the partial type at runtime
from langgraph.graph.message import add_messages
import functools

full_call = add_messages(
    [{"role": "user", "content": "hi", "id": "1"}],
    [{"role": "assistant", "content": "hello", "id": "2"}],
)
print("Full call type:", type(full_call))  # list

partial_reducer = add_messages(format="langchain-openai")
print("Partial type:", type(partial_reducer))  # functools.partial

no_arg_partial = add_messages()
print("No-arg type:", type(no_arg_partial))  # functools.partial
```

```python
# Example 3: store two differently-configured reducers in one graph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# Plain reducer (no format conversion)
plain_reducer = add_messages()

# OpenAI-format reducer
oai_reducer = add_messages(format="langchain-openai")

class DualState(TypedDict):
    raw_messages: Annotated[list, plain_reducer]
    oai_messages: Annotated[list, oai_reducer]

def echo(s: DualState) -> dict:
    msg = AIMessage(content="Echo", id="e1")
    return {"raw_messages": [msg], "oai_messages": [msg]}

g = StateGraph(DualState)
g.add_node("echo", echo)
g.set_entry_point("echo")
g.set_finish_point("echo")
compiled = g.compile()

result = compiled.invoke({"raw_messages": [], "oai_messages": []})
print("Raw count:", len(result["raw_messages"]))    # 1
print("OAI count:", len(result["oai_messages"]))    # 1
```

---

## 9 · `StateGraph.add_node(defer=True)`

**Module**: `langgraph.graph.state`  
**First dedicated coverage.**

Passing `defer=True` to `add_node()` marks the node to run **after all non-deferred nodes have completed their current super-step**. This is a coordination primitive for cleanup, notification, or aggregation nodes that must run last — without requiring manual edge wiring to every predecessor.

```python
def add_node(
    self,
    node: ...,
    *,
    defer: bool = False,     # <-- new in 1.x
    ...
) -> Self:
```

Implementation facts drawn from `langgraph.graph.state`:
- Deferred nodes are stored in `self._deferred_nodes` (a set of node names) and only compiled into the Pregel graph after all non-deferred nodes have been added.
- A deferred node still requires edges — it just guarantees that all non-deferred nodes in the same super-step have finished before it runs.
- Deferred nodes are commonly used with `BinaryOperatorAggregate` state keys: all parallel workers write to the same list, the deferred node reads the fully-aggregated list.
- If a deferred node has no incoming edges it will never execute — the `defer=True` flag does not automatically connect it.
- Circular edges among deferred nodes are supported; the super-step guarantee applies to non-deferred nodes only.

```python
# Example 1: deferred notification node that runs after all workers
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class PipelineState(TypedDict):
    results: Annotated[list[str], operator.add]
    report: str

def worker_a(s): return {"results": ["result-A"]}
def worker_b(s): return {"results": ["result-B"]}
def worker_c(s): return {"results": ["result-C"]}

def notify(s: PipelineState) -> dict:
    # Runs after all workers — sees the fully-accumulated list
    return {"report": f"Pipeline complete: {s['results']}"}

g = StateGraph(PipelineState)
g.add_node("worker_a", worker_a)
g.add_node("worker_b", worker_b)
g.add_node("worker_c", worker_c)
g.add_node("notify",   notify, defer=True)

g.add_edge(START, "worker_a")
g.add_edge(START, "worker_b")
g.add_edge(START, "worker_c")
g.add_edge("worker_a", "notify")
g.add_edge("worker_b", "notify")
g.add_edge("worker_c", "notify")
g.add_edge("notify", END)

compiled = g.compile()
result = compiled.invoke({"results": [], "report": ""})
print("Report:", result["report"])
# Report: Pipeline complete: ['result-A', 'result-B', 'result-C']
```

```python
# Example 2: deferred cleanup node in a data processing pipeline
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    data: list[str]
    cleaned: list[str]
    audit_log: str

def ingest(s): return {"data": ["raw1", "  raw2  ", "raw3"]}
def clean(s):  return {"cleaned": [x.strip() for x in s["data"]]}

def audit(s: S) -> dict:
    # defer=True ensures this runs after both ingest and clean
    return {"audit_log": f"Processed {len(s['cleaned'])} items from {len(s['data'])} raw"}

g = StateGraph(S)
g.add_node("ingest", ingest)
g.add_node("clean",  clean)
g.add_node("audit",  audit, defer=True)

g.add_edge(START, "ingest")
g.add_edge("ingest", "clean")
g.add_edge("clean", "audit")
g.add_edge("audit", END)

compiled = g.compile()
result = compiled.invoke({"data": [], "cleaned": [], "audit_log": ""})
print("Audit:", result["audit_log"])
# Audit: Processed 3 items from 3 raw
```

```python
# Example 3: defer=True for fan-in aggregation without explicit barriers
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class MapReduceState(TypedDict):
    inputs: list[str]
    word_counts: Annotated[list[int], operator.add]
    total: int

def dispatch(s: MapReduceState):
    return [Send("count_words", {"text": t, "inputs": [], "word_counts": [], "total": 0})
            for t in s["inputs"]]

def count_words(s: MapReduceState) -> dict:
    return {"word_counts": [len(s.get("text", "").split())]}  # type: ignore[arg-type]

def aggregate(s: MapReduceState) -> dict:
    # Deferred: sees the fully-reduced word_counts list
    return {"total": sum(s["word_counts"])}

g = StateGraph(MapReduceState)
g.add_node("count_words", count_words)
g.add_node("aggregate",   aggregate,   defer=True)
g.add_conditional_edges(START, dispatch, ["count_words"])
g.add_edge("count_words", "aggregate")
g.add_edge("aggregate",   END)

compiled = g.compile()
result = compiled.invoke({
    "inputs": ["hello world", "foo bar baz", "one"],
    "word_counts": [], "total": 0,
})
print("Total words:", result["total"])  # 2 + 3 + 1 = 6
```

---

## 10 · `REMOVE_ALL_MESSAGES` + bulk message deletion in `add_messages`

**Module**: `langgraph.graph.message`  
**First dedicated coverage.**

`REMOVE_ALL_MESSAGES = "__remove_all__"` is a sentinel string constant. When `add_messages` encounters a `RemoveMessage` whose `id` equals `REMOVE_ALL_MESSAGES`, it **discards all messages accumulated so far** and returns only the messages that follow the sentinel in the right-hand list.

```python
REMOVE_ALL_MESSAGES = "__remove_all__"

# In add_messages():
for idx, m in enumerate(right):
    if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
        remove_all_idx = idx   # record position

if remove_all_idx is not None:
    return right[remove_all_idx + 1:]   # return ONLY what comes after sentinel
```

Key facts:
- Everything in `left` (the existing state) is discarded when `REMOVE_ALL_MESSAGES` is encountered.
- Everything in `right` **before** the sentinel is also discarded.
- Messages in `right` **after** the sentinel are returned as the new state (including any that follow immediately).
- A `RemoveMessage(id=REMOVE_ALL_MESSAGES)` with `content=""` is the standard construction.
- This is distinct from `RemoveMessage(id="some-id")` which removes only that specific message.
- `_messages_delta_reducer` does **not** implement `REMOVE_ALL_MESSAGES` handling — it is `add_messages`-only.

```python
# Example 1: reset conversation history on topic change
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.graph import StateGraph, START, END

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]

def reset_and_respond(s: ConversationState) -> dict:
    new_topic_msg = HumanMessage(content="Let's talk about something new", id="new-1")
    ai_response   = AIMessage(content="Sure! What would you like to discuss?", id="new-2")
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # wipe history
            new_topic_msg,
            ai_response,
        ]
    }

g = StateGraph(ConversationState)
g.add_node("reset", reset_and_respond)
g.set_entry_point("reset")
g.set_finish_point("reset")
compiled = g.compile()

result = compiled.invoke({
    "messages": [
        HumanMessage(content="Tell me about cats", id="old-1"),
        AIMessage(content="Cats are great...", id="old-2"),
        HumanMessage(content="And dogs?", id="old-3"),
    ]
})
print("Message count after reset:", len(result["messages"]))     # 2
print("First message:", result["messages"][0].content)
# 2
# Let's talk about something new
```

```python
# Example 2: REMOVE_ALL_MESSAGES discards everything before it in the right list
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES

left = [
    HumanMessage(content="old-1", id="o1"),
    AIMessage(content="old-2", id="o2"),
]
right = [
    AIMessage(content="this will be discarded too", id="d1"),  # before sentinel
    RemoveMessage(id=REMOVE_ALL_MESSAGES),                      # sentinel
    HumanMessage(content="fresh start", id="n1"),               # after sentinel — kept
]

result = add_messages(left, right)
print("Result count:", len(result))             # 1
print("Result content:", result[0].content)     # fresh start
# Proof: "d1" (before sentinel) and all of `left` are discarded
```

```python
# Example 3: implement a context-window trimmer using REMOVE_ALL_MESSAGES
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.graph import StateGraph, START, END

MAX_HISTORY = 4  # keep at most 4 messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def trim_and_respond(s: State) -> dict:
    history = s["messages"]
    ai_reply = AIMessage(content=f"Reply #{len(history) + 1}", id=f"r{len(history)}")

    # If we would exceed the limit, reset to a rolling window
    if len(history) >= MAX_HISTORY:
        kept = history[-(MAX_HISTORY - 1):]  # keep last N-1 messages + new reply
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + kept + [ai_reply]
        }

    return {"messages": [ai_reply]}

g = StateGraph(State)
g.add_node("bot", trim_and_respond)
g.set_entry_point("bot")
g.set_finish_point("bot")
compiled = g.compile()

# Simulate a growing conversation
state: dict = {"messages": [HumanMessage(content="Hello", id="u0")]}
for i in range(6):
    state = compiled.invoke(state)
    print(f"Turn {i+1}: {len(state['messages'])} messages")
# Turn 1: 2 messages  (u0 + r0)
# Turn 2: 3 messages  ...
# Turn 3: 4 messages  (at limit)
# Turn 4: 4 messages  (trimmed: kept last 3 + new)
# Turn 5: 4 messages
# Turn 6: 4 messages
```

---

## Summary

| # | Class / symbol | Key source fact |
|---|---|---|
| 1 | `Edge` + `TriggerEdge` + `draw_graph()` | `draw_graph()` dry-runs the graph from an empty checkpoint; `Edge.conditional=True` marks `add_conditional_edges` origins |
| 2 | `get_graph(xray=)` + `draw_mermaid()` | `xray=True` ≡ `xray=1`; subgraph nodes are prefixed `parent:child`; `draw_mermaid()` returns Mermaid markdown including `classDef` blocks |
| 3 | `as_tool()` (beta) | Delegates to `langchain_core.tools.convert_runnable_to_tool`; returns `StructuredTool`; schema inferred from `get_input_schema()` |
| 4 | `get_subgraphs()` + `aget_subgraphs()` | `namespace=` exits early via `return` after finding the match; `recurse=True` prefixes paths with `NS_SEP` (`:`) |
| 5 | `get_input_jsonschema()` / `get_output_jsonschema()` / `get_context_jsonschema()` | `get_config_jsonschema()` is deprecated since v1.0; returns `{}` when schema is `Any` |
| 6 | `clear_cache()` + `aclear_cache()` | Clears only this graph's namespaces — not the entire `BaseCache`; `nodes=["n"]` for partial clearing |
| 7 | `_messages_delta_reducer` | Batching-invariant; no UUID auto-assignment; no `REMOVE_ALL_MESSAGES`; fast path skips `convert_to_messages` when state is already typed |
| 8 | `_add_messages_wrapper` + partial `add_messages` | Zero-arg call returns `functools.partial`; `format=` binds to partial; used as `Annotated[list, add_messages(format="langchain-openai")]` |
| 9 | `add_node(defer=True)` | Deferred nodes run after all non-deferred nodes in the super-step; still require explicit edges; work naturally with `BinaryOperatorAggregate` fan-in |
| 10 | `REMOVE_ALL_MESSAGES` + bulk deletion | Sentinel processed in `add_messages` only (not `_messages_delta_reducer`); everything in `left` AND in `right` before the sentinel is discarded |

### Cross-references

- Vol. 1: `add_messages` base behaviour, ID-based dedup
- Vol. 5: `BinaryOperatorAggregate` — reducer channels (relevant to `defer=True` fan-in)
- Vol. 8: `InMemorySaver` storage internals; `InMemoryCache` (used by `clear_cache()`)
- Vol. 11: `Topic` + `EphemeralValue` channel types
- Vol. 20: `DeltaChannel` — the channel type `_messages_delta_reducer` targets
- Vol. 25: `RetryPolicy` + `CachePolicy` + `TimeoutPolicy` — `add_node` policy parameters
- Vol. 27: `_TimedAttemptScope` + `_AttemptContext` — timeout internals
- Vol. 28: `merge_configs` / `ensure_config` — config internals; `TAG_NOSTREAM` / `TAG_HIDDEN` — stream suppression

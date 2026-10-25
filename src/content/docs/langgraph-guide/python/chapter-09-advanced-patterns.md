---
title: "Chapter 9 — Advanced Patterns"
description: "RetryPolicy, CachePolicy, TimeoutPolicy, Runtime context injection, map-reduce with Send, add_sequence, Overwrite, GraphOutput v2, set_node_defaults, and the Functional API — source-verified patterns."
framework: "langgraph"
language: "python"
sidebar:
  label: "9 · Advanced patterns"
  order: 9
---

**What you'll learn:** the patterns you reach for when simple graphs aren't enough — `RetryPolicy` with custom callables and sequences, built-in `CachePolicy` with `InMemoryCache`, `TimeoutPolicy` with idle/heartbeat semantics, `Runtime[Context]` for type-safe run-scoped data, map-reduce fan-out with `Send` (including per-send timeouts), `add_sequence()` for concise linear pipelines, `Overwrite` for bypassing reducers, `GraphOutput` with the `version="v2"` invoke API, `set_node_defaults()` for graph-wide policies, plus the Functional API `@entrypoint`/`@task` with async, timeout, context and more.

Verified against **`langgraph==%%version=langgraph%%`** (modules: `langgraph.types`, `langgraph.runtime`, `langgraph.cache.memory`, `langgraph.func`, `langgraph.graph.state`).

**Time:** ~50 minutes. Most of this is reference — skim for patterns you need.

> Prereqs: [Chapter 3 — Multi-agent systems](/langgraph-guide/python/chapter-03-multi-agent/) and [Chapter 4 — Tools](/langgraph-guide/python/chapter-04-tools/).

## Advanced Patterns

### Pattern 1: ReAct (Reasoning + Acting) with native LangGraph

Build a ReAct-style agent entirely in LangGraph using `ToolNode`, `tools_condition`, and `MessagesState`. No external agent executor needed.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    # Replace with a real search API in production
    return f"Search results for '{query}': Found 3 relevant articles."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A simple arithmetic expression like '3.7 * 13_960_000 * 0.15'
    """
    import ast, operator as op
    # Minimal safe eval for arithmetic only
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
               ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


tools = [search_web, calculator]

# ── LLM (bind tools so it knows how to call them) ───────────────────────────

# Swap ChatAnthropic / ChatOpenAI / any chat model
from langchain_anthropic import ChatAnthropic   # pip install langchain-anthropic

llm = ChatAnthropic(model="%%model=anthropic-flagship%%").bind_tools(tools)


# ── Graph nodes ──────────────────────────────────────────────────────────────

def agent(state: MessagesState) -> dict:
    """Call the LLM. It decides whether to use tools or finish."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── Build the ReAct graph ────────────────────────────────────────────────────

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
# tools_condition routes to "tools" if the last message has tool_calls, else END
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")   # after tool execution, return to the agent

graph = builder.compile(checkpointer=InMemorySaver())

# ── Run ──────────────────────────────────────────────────────────────────────

config = {"configurable": {"thread_id": "react-1"}}

for event in graph.stream(
    {"messages": [HumanMessage("What is 15% of Tokyo's population? Search for the population first, then calculate.")]},
    config,
    stream_mode="updates",
):
    for node, updates in event.items():
        print(f"── {node} ──")
        for msg in updates.get("messages", []):
            print(f"  {type(msg).__name__}: {msg.content[:120] if msg.content else '(tool call)'}")
```

**How it works:**

1. `agent` calls the LLM, which emits either an `AIMessage` with `tool_calls` or a plain text reply.
2. `tools_condition` inspects the last message: if it has tool calls, go to `"tools"`; otherwise go to `END`.
3. `ToolNode` executes all pending tool calls in parallel and returns `ToolMessage` results.
4. Execution loops back to `agent`, which sees the tool results and decides the next action.
5. When the LLM is satisfied, it returns a plain `AIMessage` and the graph exits.

The checkpointer persists the full message history per `thread_id`, enabling multi-turn conversations across `invoke` / `stream` calls.

### Pattern 2: Tree-of-Thoughts

Explore several reasoning paths in parallel, keep the best, and answer from it. The fan-out uses `Send`. Return a list of `Send` objects from a **routing function** (passed to `add_conditional_edges`), or from a node wrapped in a `Command`: `return Command(goto=[Send(...), ...])`. A node that returns a bare list of `Send` objects fails with `InvalidUpdateError: Expected dict`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

model = init_chat_model("anthropic:%%model=anthropic%%")


class ThoughtState(TypedDict, total=False):
    question: str
    thoughts: Annotated[list[dict], operator.add]   # parallel branches append here
    best_thought: dict
    final_answer: str


class PathInput(TypedDict):
    question: str
    path_number: int


def fan_out(state: ThoughtState) -> list[Send]:
    """Routing function: one Send per reasoning path."""
    return [Send("explore", {"question": state["question"], "path_number": i}) for i in range(3)]


def explore(state: PathInput) -> dict:
    response = model.invoke(
        f"Question: {state['question']}\nGive approach #{state['path_number'] + 1} in two sentences."
    )
    score = len(response.content)            # stand-in for a real evaluator
    return {"thoughts": [{"path": state["path_number"], "reasoning": response.content, "score": score}]}


def select_best(state: ThoughtState) -> dict:
    return {"best_thought": max(state["thoughts"], key=lambda t: t["score"])}


def synthesize(state: ThoughtState) -> dict:
    response = model.invoke(f"Using this reasoning, answer briefly:\n{state['best_thought']['reasoning']}")
    return {"final_answer": response.content}


tot_graph = (
    StateGraph(ThoughtState)
    .add_node("explore", explore)
    .add_node("select", select_best)
    .add_node("synthesize", synthesize)
    .add_conditional_edges(START, fan_out, ["explore"])   # Send objects come from a routing function
    .add_edge("explore", "select")                         # select runs once, after all three paths
    .add_edge("select", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)

result = tot_graph.invoke({"question": "How should a small team reduce cloud costs?"})
print(len(result["thoughts"]), "paths explored")   # 3 paths explored
print("Final answer:", result["final_answer"])
```

### Pattern 3: Self-Reflection

The agent drafts an answer, critiques it, and revises until the critique finds nothing to fix or a revision cap is reached. Each critique looks at the latest draft, not the first one. This continues from Pattern 2 (it reuses `model`):

```python
class ReflectionState(TypedDict, total=False):
    question: str
    initial_response: str
    critique: str
    refined_response: str
    reflection_count: int

def latest_draft(state: ReflectionState) -> str:
    return state.get("refined_response") or state["initial_response"]

def generate_response(state: ReflectionState) -> dict:
    """Generate initial response."""
    
    response = model.invoke(state["question"])
    
    return {
        "initial_response": response.content,
        "reflection_count": 0
    }

def self_critique(state: ReflectionState) -> dict:
    """Critique the response."""
    
    prompt = f"""
    Question: {state['question']}
    Response: {latest_draft(state)}

    Critique this response. What could be improved?
    """
    
    critique = model.invoke(prompt)
    
    return {"critique": critique.content}

def should_refine(state: ReflectionState) -> str:
    """Decide if response needs refinement."""
    
    if state["reflection_count"] >= 2:
        return "done"
    
    # Check critique for issues
    if any(word in state["critique"].lower() 
           for word in ["incorrect", "missing", "unclear", "incomplete"]):
        return "refine"
    
    return "done"

def refine_response(state: ReflectionState) -> dict:
    """Create refined response based on critique."""
    
    prompt = f"""
    Original question: {state['question']}
    Your response: {latest_draft(state)}
    Critique: {state['critique']}

    Provide an improved response addressing the critique.
    """
    
    refined = model.invoke(prompt)
    
    return {
        "refined_response": refined.content,
        "reflection_count": state["reflection_count"] + 1
    }

# Build reflection loop
builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_response)
builder.add_node("critique", self_critique)
builder.add_node("refine", refine_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")

builder.add_conditional_edges(
    "critique",
    should_refine,
    {"refine": "refine", "done": END}
)

builder.add_edge("refine", "critique")  # Loop back for re-critique

reflection_graph = builder.compile()

# Use it
result = reflection_graph.invoke({
    "question": "Explain quantum computing to a child"
})

print("Initial:", result["initial_response"])
print("Refined:", result.get("refined_response", "No refinement needed"))
print("Reflection iterations:", result["reflection_count"])
```

### Pattern 4: Structured Output with Validation

`with_structured_output` asks the model for output that matches a Pydantic model. Validation can still fail, so loop back with the error as a hint, and cap the attempts: a validation loop without a cap runs until the recursion limit.

```python
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

model = init_chat_model("anthropic:%%model=anthropic%%")
MAX_ATTEMPTS = 3


class ResearchOutput(BaseModel):
    topic: str
    key_findings: list[str]
    confidence_score: float = Field(ge=0, le=1)


class StructuredState(TypedDict, total=False):
    topic: str
    output: ResearchOutput
    attempts: int
    errors: list[str]


structured_model = model.with_structured_output(ResearchOutput)


def research(state: StructuredState) -> dict:
    hint = f"\nThe previous attempt failed: {state['errors'][-1]}" if state.get("errors") else ""
    try:
        output = structured_model.invoke(f"Summarise current research on {state['topic']}.{hint}")
        return {"output": output, "attempts": state.get("attempts", 0) + 1}
    except (ValidationError, ValueError) as exc:          # bad or missing structured output
        return {"errors": [*state.get("errors", []), str(exc)[:200]],
                "attempts": state.get("attempts", 0) + 1}


def route(state: StructuredState) -> str:
    if state.get("output") is not None or state["attempts"] >= MAX_ATTEMPTS:
        return END
    return "research"                                      # try again, with the error as a hint


graph = (
    StateGraph(StructuredState)
    .add_node("research", research)
    .add_edge(START, "research")
    .add_conditional_edges("research", route, ["research", END])
    .compile()
)

result = graph.invoke({"topic": "AI safety"})
if result.get("output"):
    print(result["output"].key_findings)
else:
    print(f"gave up after {result['attempts']} attempts")
```

For `create_agent` agents, pass `response_format=ResearchOutput` instead; the result is in `result["structured_response"]`.

---

### Patterns 5–7: caching, retries and timeouts

Node caching (`CachePolicy`), retries (`RetryPolicy`), timeouts (`TimeoutPolicy`) and error handlers are per-node options on `add_node`, with graph-wide defaults from `set_node_defaults()`. They are documented, with a runnable example that uses all of them, in [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies); [Chapter 10](/langgraph-guide/python/chapter-10-production/#containing-loops-and-failures) covers when to use them.

---

### Pattern 8: `Runtime[Context]` — type-safe run-scoped data

`Runtime` bundles per-run context (user ID, tenant ID, feature flags) separate from graph state. Declare a `context_schema` on `StateGraph`, then inject `Runtime[Ctx]` into any node.

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, ExecutionInfo
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class RequestContext:
    user_id: str
    tenant_id: str
    is_premium: bool = False


class QueryState(TypedDict):
    question: str
    answer: str
    attempt: int


def answer_node(state: QueryState, runtime: Runtime[RequestContext]) -> dict:
    ctx = runtime.context                         # type: RequestContext

    # Use context for access control
    model = "%%model=anthropic-flagship%%" if ctx.is_premium else "%%model=anthropic-small%%"

    # Use store for long-term memory keyed by user
    if runtime.store:
        history = runtime.store.search(
            ("history", ctx.user_id),
            query=state["question"],
            limit=3,
        )
        prior = " | ".join(h.value.get("answer", "") for h in history)
    else:
        prior = ""

    answer = f"[{model}] Answer for {ctx.user_id}: {state['question']} (prior: {prior})"

    # Write this answer to long-term memory
    if runtime.store:
        runtime.store.put(
            ("history", ctx.user_id),
            f"q-{len(state['question'])}",
            {"question": state["question"], "answer": answer},
        )

    # ExecutionInfo gives checkpoint/run metadata
    exec_info: ExecutionInfo | None = runtime.execution_info
    if exec_info:
        print(f"attempt={exec_info.node_attempt}, thread={exec_info.thread_id}")

    return {"answer": answer}


store = InMemoryStore()

builder = StateGraph(QueryState, context_schema=RequestContext)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile(checkpointer=InMemorySaver(), store=store)

# Pass context at invoke time — not part of state
result = graph.invoke(
    {"question": "What is LangGraph?", "answer": "", "attempt": 0},
    {"configurable": {"thread_id": "session-1"}},
    context=RequestContext(user_id="alice", tenant_id="acme", is_premium=True),
)
print(result["answer"])
```

`Runtime` fields:

| Field | Type | Notes |
|---|---|---|
| `context` | `ContextT` | What you passed as `context=` at invoke time |
| `store` | `BaseStore \| None` | What you passed to `compile(store=...)` |
| `stream_writer` | `(Any) -> None` | Write to `stream_mode="custom"` |
| `heartbeat` | `() -> None` | Refresh `TimeoutPolicy(idle_timeout=...)` |
| `previous` | `Any` | Functional API only: last return value for this thread |
| `execution_info` | `ExecutionInfo \| None` | `checkpoint_id`, `thread_id`, `run_id`, `node_attempt` |
| `server_info` | `ServerInfo \| None` | Agent Server only |

---

### Pattern 9: Map-reduce fan-out with `Send`

`Send` dispatches a named node with a custom state snapshot — each item in a list gets its own parallel execution. A reducer on the downstream channel collects all results.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class Pipeline(TypedDict):
    items: list[str]
    scores: Annotated[list[float], operator.add]   # reducer accumulates results


class WorkerInput(TypedDict):
    item: str


def dispatch(state: Pipeline) -> list[Send]:
    # add_conditional_edges accepts a path function that returns list[Send],
    # not just string node names — this is what enables map-reduce fan-out.
    return [Send("score_item", WorkerInput(item=i)) for i in state["items"]]


def score_item(state: WorkerInput) -> dict:
    """Runs in parallel for every item sent by dispatch."""
    return {"scores": [len(state["item"]) / 10.0]}


def summarize(state: Pipeline) -> dict:
    avg = sum(state["scores"]) / len(state["scores"]) if state["scores"] else 0.0
    return {"items": [f"avg_score={avg:.2f}"]}


builder = StateGraph(Pipeline)
builder.add_node("score_item", score_item)
builder.add_node("summarize", summarize)

# Conditional edge from START fans out to N parallel score_item runs
builder.add_conditional_edges(START, dispatch)
# All score_item tasks drain before summarize starts (barrier edge)
builder.add_edge("score_item", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
result = graph.invoke({"items": ["hello", "hi", "hey there"], "scores": []})
print(result["scores"])   # [0.5, 0.2, 0.9] (order may vary)
print(result["items"])    # ['avg_score=0.53']
```

---

### Pattern 10: `Send` with `timeout` parameter

`Send` accepts an optional `timeout` keyword argument (since LangGraph 1.2.0). Each fan-out task then carries its own deadline, independent of the node-level timeout. As with node timeouts, the target node must be `async def`; with a sync node, `invoke` raises `ValueError: Node timeouts are only supported for async nodes`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import asyncio
from langgraph.types import Send, TimeoutPolicy


class CrawlState(TypedDict):
    urls: list[str]
    results: Annotated[list[str], operator.add]


class PageInput(TypedDict):
    url: str


def fan_out(state: CrawlState) -> list[Send]:
    """Each URL gets its own Send with a 30-second deadline."""
    return [
        Send(
            "fetch_page",
            PageInput(url=url),
            timeout=30.0,           # float: seconds until this branch is cancelled
        )
        for url in state["urls"]
    ]


def fan_out_with_policy(state: CrawlState) -> list[Send]:
    """Use a full TimeoutPolicy for fine-grained idle + wall-clock control."""
    return [
        Send(
            "fetch_page",
            PageInput(url=url),
            timeout=TimeoutPolicy(
                run_timeout=30.0,   # hard cap
                idle_timeout=10.0,  # cancel if idle for 10 s between progress signals
                refresh_on="auto",
            ),
        )
        for url in state["urls"]
    ]


async def fetch_page(state: PageInput) -> dict:
    await asyncio.sleep(0.1)   # simulate network
    return {"results": [f"content:{state['url']}"]}


builder = StateGraph(CrawlState)
builder.add_node("fetch_page", fetch_page)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("fetch_page", END)

graph = builder.compile()
result = asyncio.run(graph.ainvoke({"urls": ["a.com", "b.com", "c.com"], "results": []}))
print(result["results"])   # ['content:a.com', 'content:b.com', 'content:c.com']
```

**When to use per-`Send` timeout vs. node-level `TimeoutPolicy`:**

| Scenario | Recommendation |
|---|---|
| All fan-out branches share the same deadline | Node-level `TimeoutPolicy` on `add_node` |
| Different branches need different deadlines | Per-`Send` `timeout=` |
| You need idle detection per-branch | Per-`Send` `timeout=TimeoutPolicy(idle_timeout=...)` |

---

### Pattern 11: `add_sequence()` — concise linear pipelines

`StateGraph.add_sequence()` is a convenience method. It registers a list of nodes and automatically wires them with edges — no separate `add_node` + `add_edge` calls needed. Each element is either a callable (name is derived from `__name__`) or a `("name", callable)` tuple.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class DocState(TypedDict):
    raw: str
    cleaned: str
    summary: str
    translated: str


# ── Step functions ────────────────────────────────────────────────────────────

def clean(state: DocState) -> dict:
    return {"cleaned": state["raw"].strip().lower()}


def summarize(state: DocState) -> dict:
    # In production: call an LLM here
    return {"summary": state["cleaned"][:80] + "..."}


def translate(state: DocState) -> dict:
    return {"translated": f"[ES] {state['summary']}"}


# ── Build with add_sequence ───────────────────────────────────────────────────

builder = StateGraph(DocState)

# Registers "clean" → "summarize" → "translate" nodes and connects them in order.
# Equivalent to three add_node() calls plus two add_edge() calls.
builder.add_sequence([clean, summarize, translate])

# add_sequence does NOT wire START or END — do those explicitly:
builder.add_edge(START, "clean")
builder.add_edge("translate", END)

graph = builder.compile()

result = graph.invoke({"raw": "  Hello World  ", "cleaned": "", "summary": "", "translated": ""})
print(result["translated"])   # '[ES] hello world...'
```

**Mixed name forms** — pass a `("custom_name", fn)` tuple when you need a specific node name (e.g. to attach conditional edges or retry policies later):

```python
builder2 = StateGraph(DocState)
builder2.add_sequence([
    clean,                          # name inferred: "clean"
    ("summarize_v2", summarize),    # explicit name
    translate,                      # name inferred: "translate"
])
builder2.add_edge(START, "clean").add_edge("translate", END)
print(list(builder2.compile().get_graph().nodes))
# ['__start__', 'clean', 'summarize_v2', 'translate', '__end__']
```

`add_sequence` takes no policy options, and adding the same node name again with `add_node` raises `ValueError: Node names must be unique`. If one step needs a retry or cache policy, register that step with `add_node(..., retry_policy=...)` and connect it with `add_edge`, or set graph-wide defaults with `set_node_defaults()` before calling `add_sequence`.

> **Note:** `add_sequence` returns the `StateGraph` instance, so calls can be chained: `builder.add_sequence([...]).add_edge(START, "clean")`.

---

### Pattern 12: `Overwrite` — bypass reducers for one-shot resets

When a state channel uses a reducer (e.g. `Annotated[list, operator.add]`), every node update *appends* rather than replaces. `Overwrite` is a wrapper that signals LangGraph to skip the reducer and write the value directly.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite


class LogState(TypedDict):
    # All node updates normally *append* to this list via operator.add
    events: Annotated[list[str], operator.add]
    run_id: str


def append_events(state: LogState) -> dict:
    """Normal update — appends to the existing list."""
    return {"events": ["event_a", "event_b"]}


def reset_events(state: LogState) -> dict:
    """Overwrite bypasses operator.add and replaces the list entirely."""
    return {"events": Overwrite(["session_reset"])}


builder = StateGraph(LogState)
builder.add_node("append", append_events)
builder.add_node("reset", reset_events)
builder.add_edge(START, "append")
builder.add_edge("append", "reset")
builder.add_edge("reset", END)

graph = builder.compile()

result = graph.invoke({"events": ["boot"], "run_id": "r1"})
print(result["events"])
# ["session_reset"]  — not ["boot", "event_a", "event_b", "session_reset"]
```

**Why this matters:** Without `Overwrite`, every fan-in from parallel `Send` branches accumulates results in the reducer. Use `Overwrite` when a later node needs to authoritatively set the channel to a clean value regardless of what was collected earlier.

A fan-in node for the crawler in Pattern 10 could deduplicate what the branches collected:

```python
from langgraph.types import Overwrite

def consolidate(state: CrawlState) -> dict:
    # Deduplicate and canonicalise — then overwrite so nothing else appends
    canonical = list(dict.fromkeys(state["results"]))
    return {"results": Overwrite(canonical)}
```

---

### Pattern 13: `GraphOutput` with `version="v2"`

The `invoke` and `stream` APIs accept an optional `version` keyword argument. When `version="v2"`, the return type changes from a raw state dict to a `GraphOutput`, a frozen dataclass that bundles the output value with any `Interrupt` objects raised during the run.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from typing_extensions import TypedDict


class ReviewState(TypedDict):
    document: str
    approved: bool


def review_node(state: ReviewState) -> dict:
    decision = interrupt({"prompt": "Approve this document?", "doc": state["document"]})
    return {"approved": decision == "yes"}


builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "doc-review-1"}}

# ── version="v2" invoke ───────────────────────────────────────────────────────

output = graph.invoke(
    {"document": "Quarterly earnings report", "approved": False},
    cfg,
    version="v2",           # opt-in to GraphOutput return type
)

# GraphOutput exposes .value and .interrupts
print(type(output))         # <class 'langgraph.types.GraphOutput'>
print(output.value)         # state dict up to the interrupt point
print(output.interrupts)    # tuple of Interrupt objects

# Inspect each interrupt
for interrupt_obj in output.interrupts:
    print(interrupt_obj.value)   # {"prompt": "Approve this document?", "doc": "..."}

# Resume with a human decision
from langgraph.types import Command

final = graph.invoke(Command(resume="yes"), cfg, version="v2")
print(final.value["approved"])   # True
print(final.interrupts)          # ()  — empty tuple, run completed
```

**`GraphOutput` fields:**

| Field | Type | Description |
|---|---|---|
| `value` | `dict` | The graph's state at the point of return or interrupt |
| `interrupts` | `tuple[Interrupt, ...]` | All interrupts raised during this invoke call |

**Streaming with `version="v2"`** — each item is a `StreamPart` dict with `type`, `ns`, `data` and (for `values`) `interrupts` keys, not a `GraphOutput`:

```python
for chunk in graph.stream(
    {"document": "Board minutes", "approved": False},
    cfg,
    stream_mode="values",
    version="v2",
):
    print(chunk["data"], chunk.get("interrupts"))
```

> **Migration note:** `version="v2"` is opt-in and backward-compatible. Existing code that omits `version=` continues to receive plain dicts.

---

### Pattern 14: `context_schema` and `Runtime[Context]` — complete injection example

This pattern expands on Pattern 8 to show the minimal wiring needed for context injection from scratch.

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


# ── 1. Define an immutable context dataclass ──────────────────────────────────

@dataclass
class AppContext:
    user_id: str
    db_url: str
    feature_flags: tuple[str, ...] = ()


# ── 2. Define graph state ─────────────────────────────────────────────────────

class AppState(TypedDict):
    query: str
    result: str


# ── 3. Nodes receive Runtime[AppContext] as a second argument ─────────────────

def lookup_node(state: AppState, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context               # fully typed as AppContext
    user_id = ctx.user_id               # IDE-aware, no dict key typos
    db_url = ctx.db_url

    # Simulate a DB call using the injected URL
    result = f"DB({db_url}) answered '{state['query']}' for user '{user_id}'"

    if "beta_feature" in ctx.feature_flags:
        result += " [beta mode]"

    return {"result": result}


def format_node(state: AppState, runtime: Runtime[AppContext]) -> dict:
    prefix = f"[{runtime.context.user_id}]"
    return {"result": f"{prefix} {state['result']}"}


# ── 4. Declare context_schema on StateGraph ───────────────────────────────────

builder = StateGraph(AppState, context_schema=AppContext)
builder.add_node("lookup", lookup_node)
builder.add_node("format", format_node)
builder.add_edge(START, "lookup")
builder.add_edge("lookup", "format")
builder.add_edge("format", END)

graph = builder.compile(checkpointer=InMemorySaver())

# ── 5. Pass context at runtime via the `context=` keyword ────────────────────

ctx = AppContext(
    user_id="u42",
    db_url="postgresql://prod/mydb",
    feature_flags=("beta_feature",),
)

result = graph.invoke(
    {"query": "What is my account balance?", "result": ""},
    {"configurable": {"thread_id": "sess-42"}},
    context=ctx,
)
print(result["result"])
# [u42] DB(postgresql://prod/mydb) answered 'What is my account balance?' for user 'u42' [beta mode]
```

**Key rules for `context_schema`:**

- The dataclass must be passed as `context=` at `invoke`/`stream` time, not embedded in state.
- Context is immutable during a run — nodes can read it but cannot write back to it.
- Any node that declares `runtime: Runtime[YourContext]` as a second parameter receives the injected context. Nodes that omit the second parameter are unaffected.

---

## Functional API (`@entrypoint` / `@task`)

The Functional API is the imperative alternative to `StateGraph`. The result is still a `Pregel` graph with the same `invoke`/`stream`/`get_state` surface.

### Basic parallel fan-out

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver


@task
def fetch_page(url: str) -> str:
    return f"content:{url}"     # replace with real I/O


@task
def summarize_page(content: str) -> str:
    return f"summary:{content[:20]}"


@entrypoint(checkpointer=InMemorySaver())
def pipeline(urls: list[str]) -> list[str]:
    # All fetches launch in parallel; .result() blocks until done
    pages = [fetch_page(u) for u in urls]
    summaries = [summarize_page(p.result()) for p in pages]
    return [s.result() for s in summaries]


cfg = {"configurable": {"thread_id": "run-1"}}
print(pipeline.invoke(["a.html", "b.html"], cfg))
# ['summary:content:a.html', 'summary:content:b.html']
```

### `entrypoint.final` — return one value, save another

Use `entrypoint.final` when the value you want to return to the caller differs from what you want the checkpointer to remember for `previous`.

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@entrypoint(checkpointer=InMemorySaver())
def accumulator(n: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    total = (previous or 0) + n
    # Return `total` to the caller; save `total` for the next call's `previous`.
    return entrypoint.final(value=total, save=total)


cfg = {"configurable": {"thread_id": "acc"}}
print(accumulator.invoke(5, cfg))   # 5
print(accumulator.invoke(3, cfg))   # 8
print(accumulator.invoke(2, cfg))   # 10
```

### Tasks with `RetryPolicy` and `CachePolicy`

`@task` accepts the same `retry_policy` and `cache_policy` kwargs as `StateGraph.add_node`. Pass a `BaseCache` to `@entrypoint(cache=...)`.

```python
import httpx
from langgraph.func import entrypoint, task
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy, CachePolicy


@task(
    retry_policy=RetryPolicy(max_attempts=4, retry_on=httpx.TransportError),
    cache_policy=CachePolicy(ttl=300),
)
async def fetch(url: str) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.get(url, timeout=10)
        r.raise_for_status()
        return r.text[:500]


cache = InMemoryCache()


@entrypoint(checkpointer=InMemorySaver(), cache=cache)
async def crawl(urls: list[str]) -> list[str]:
    futures = [fetch(u) for u in urls]      # all fetches start at once
    return [await f for f in futures]       # await task futures in async code


# asyncio.run(crawl.ainvoke(["https://example.com"], {"configurable": {"thread_id": "c1"}}))
```

The cache key is the task's arguments, and the cache is shared across threads: a second run on another `thread_id` with the same URL within 300 seconds returns the cached result without calling `fetch` again. In async code, `await` the futures; `.result()` is for sync entrypoints.

### Resuming after `interrupt` in a task workflow

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


@task
def draft_content(topic: str) -> str:
    return f"Draft about {topic}"


@entrypoint(checkpointer=InMemorySaver())
def review_flow(topic: str) -> dict:
    draft = draft_content(topic).result()   # cached on resume — not re-run
    edit = interrupt({"question": "Edit this draft?", "draft": draft})
    return {"draft": draft, "edit": edit}


cfg = {"configurable": {"thread_id": "review-1"}}

# First pass: pauses at interrupt
for ev in review_flow.stream("climate change", cfg):
    print(ev)

# Resume with the human's edit
for ev in review_flow.stream(Command(resume="Make it shorter"), cfg):
    print(ev)
# {'review_flow': {'draft': '...', 'edit': 'Make it shorter'}}
```

### Async tasks with timeout

`@task` supports an optional `timeout` parameter for async functions. When the deadline fires, `NodeTimeoutError` is raised (the retry policy decides whether to retry).

```python
import asyncio
from datetime import timedelta
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy, TimeoutPolicy


@task(
    timeout=TimeoutPolicy(run_timeout=5.0, idle_timeout=2.0),
    retry_policy=RetryPolicy(max_attempts=3),
)
async def fetch_url(url: str) -> str:
    """Fetch a URL with a 5-second hard cap and 2-second idle cap."""
    await asyncio.sleep(0.1)   # simulate network I/O
    return f"content:{url}"


@entrypoint(checkpointer=InMemorySaver())
async def crawl(urls: list[str]) -> list[str]:
    futures = [fetch_url(u) for u in urls]      # all three start at once
    return list(await asyncio.gather(*futures))  # task futures are awaitable


cfg = {"configurable": {"thread_id": "crawl-1"}}
results = asyncio.run(crawl.ainvoke(["a.com", "b.com", "c.com"], cfg))
print(results)
```

### `@task` with explicit `name`

Use `name=` when the decorated function's `__name__` is not descriptive enough for traces, or when you want stable names across refactors:

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache


@task(name="embed_document", cache_policy=CachePolicy(ttl=3600))
def embed(text: str) -> list[float]:
    """Expensive embedding — cached for 1 hour under the name 'embed_document'."""
    return [len(text) * 0.01, 0.5]


cache = InMemoryCache()


@entrypoint(checkpointer=InMemorySaver(), cache=cache)
def index_pipeline(docs: list[str]) -> list[list[float]]:
    futures = [embed(doc) for doc in docs]
    return [f.result() for f in futures]


cfg = {"configurable": {"thread_id": "index-1"}}
print(index_pipeline.invoke(["hello world", "foo bar"], cfg))
```

### `@entrypoint` with `context_schema` and `runtime` injection

An `@entrypoint` can declare a `context_schema` — the `runtime` parameter is then injected automatically, giving access to `context`, `store`, `stream_writer`, `previous`, and `execution_info`:

```python
from dataclasses import dataclass
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


@dataclass
class UserCtx:
    user_id: str
    is_premium: bool = False


@task
def analyse(doc: str) -> str:
    tier = "detailed" if len(doc) > 100 else "brief"
    return f"{tier} analysis of: {doc[:40]}"


store = InMemoryStore()

@entrypoint(
    checkpointer=InMemorySaver(),
    store=store,
    context_schema=UserCtx,
)
def analyse_flow(docs: list[str], *, runtime: Runtime[UserCtx]) -> dict:
    """Analyse documents, personalise output, and persist a summary per user."""
    ctx = runtime.context                        # fully typed as UserCtx
    limit = 10 if ctx.is_premium else 3
    docs_to_process = docs[:limit]

    futures = [analyse(d) for d in docs_to_process]
    analyses = [f.result() for f in futures]

    # Persist summary to long-term store
    if runtime.store:
        runtime.store.put(
            ("analyses", ctx.user_id),
            "latest",
            {"results": analyses},
        )

    return {"user": ctx.user_id, "analyses": analyses}


cfg = {"configurable": {"thread_id": "flow-1"}}
result = analyse_flow.invoke(
    ["Doc about Python.", "Doc about TypeScript." * 10, "Short doc."],
    cfg,
    context=UserCtx(user_id="alice", is_premium=True),
)
print(result["user"])       # alice
print(len(result["analyses"]))  # up to 10 (premium)
```

### `previous` parameter — accumulate state across calls

The `previous` parameter receives the return value of the last call on the same thread. Use `entrypoint.final` when you want to return something different from what you save:

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@entrypoint(checkpointer=InMemorySaver())
def running_stats(
    new_values: list[float],
    *,
    previous: dict[str, Any] | None = None,
) -> entrypoint.final[dict, dict]:
    """Return stats to the caller while saving a compact state to the checkpoint."""
    prev = previous or {"n": 0, "total": 0.0, "min": float("inf"), "max": float("-inf")}

    if not new_values:
        mean = prev["total"] / prev["n"] if prev["n"] else 0.0
        return entrypoint.final(
            value={"n": prev["n"], "mean": mean, "min": prev["min"], "max": prev["max"]},
            save=prev,
        )

    n = prev["n"] + len(new_values)
    total = prev["total"] + sum(new_values)
    lo = min(prev["min"], *new_values)
    hi = max(prev["max"], *new_values)

    # What the caller gets: full stats
    output = {"n": n, "mean": total / n, "min": lo, "max": hi}
    # What the checkpoint saves: compact state for next call
    save = {"n": n, "total": total, "min": lo, "max": hi}

    return entrypoint.final(value=output, save=save)


cfg = {"configurable": {"thread_id": "stats-1"}}
print(running_stats.invoke([1.0, 2.0, 3.0], cfg))
# {'n': 3, 'mean': 2.0, 'min': 1.0, 'max': 3.0}
print(running_stats.invoke([4.0, 5.0], cfg))
# {'n': 5, 'mean': 3.0, 'min': 1.0, 'max': 5.0}
```


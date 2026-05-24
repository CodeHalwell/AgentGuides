---
title: "Chapter 9 — Advanced Patterns"
description: "RetryPolicy, CachePolicy, TimeoutPolicy, Runtime context injection, map-reduce with Send, add_sequence, Overwrite, GraphOutput v2, and the Functional API — source-verified patterns for LangGraph 1.2.1."
framework: langgraph
language: python
sidebar:
  label: "9 · Advanced patterns"
  order: 9
---

# Chapter 9 — Advanced Patterns

**What you'll learn:** the patterns you reach for when simple graphs aren't enough — `RetryPolicy` with custom callables and sequences, built-in `CachePolicy` with `InMemoryCache`, `TimeoutPolicy` with idle/heartbeat semantics, `Runtime[Context]` for type-safe run-scoped data, map-reduce fan-out with `Send` (including per-send timeouts), `add_sequence()` for concise linear pipelines, `Overwrite` for bypassing reducers, `GraphOutput` with the `version="v2"` invoke API, plus the Functional API `@entrypoint`/`@task`.

Verified against **`langgraph==1.2.1`** (modules: `langgraph.types`, `langgraph.runtime`, `langgraph.cache.memory`, `langgraph.func`).

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

llm = ChatAnthropic(model="claude-opus-4-7").bind_tools(tools)


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

Explore multiple reasoning paths:


```python
from langgraph.types import Send

class ThoughtState(TypedDict):
    question: str
    thoughts: Annotated[list[dict], lambda x, y: x + y]
    best_thought: dict
    final_answer: str

def generate_thoughts(state: ThoughtState) -> list[Send]:
    """Generate multiple solution approaches."""
    
    num_paths = 3
    returns = []
    
    for i in range(num_paths):
        returns.append(
            Send("explore_thought", {
                "question": state["question"],
                "path_number": i
            })
        )
    
    return returns

def explore_thought(state: ThoughtState) -> dict:
    """Explore one reasoning path."""
    
    prompt = f"""
    Question: {state['question']}
    Path #{state.get('path_number', 0)}
    
    Provide your reasoning for this specific approach.
    """
    
    response = model.invoke(prompt)
    
    return {
        "thoughts": [{
            "path": state.get("path_number"),
            "reasoning": response.content,
            "quality_score": 0.8  # Could be evaluated
        }]
    }

def select_best(state: ThoughtState) -> dict:
    """Select the best thought."""
    
    if not state["thoughts"]:
        return {"best_thought": {}}
    
    best = max(state["thoughts"], key=lambda x: x.get("quality_score", 0))
    
    return {"best_thought": best}

def synthesize(state: ThoughtState) -> dict:
    """Synthesize best thought into answer."""
    
    best_reasoning = state["best_thought"].get("reasoning", "")
    
    prompt = f"""
    Best reasoning: {best_reasoning}
    
    Provide a final answer based on this reasoning.
    """
    
    response = model.invoke(prompt)
    
    return {"final_answer": response.content}

# Build tree-of-thoughts
builder = StateGraph(ThoughtState)
builder.add_node("generate", generate_thoughts)
builder.add_node("explore", explore_thought)
builder.add_node("select", select_best)
builder.add_node("synthesize", synthesize)

builder.add_conditional_edges(
    START,
    lambda _: "generate"
)
builder.add_conditional_edges(
    "generate",
    lambda _: ["explore"],
    ["explore"]
)
builder.add_edge("explore", "select")
builder.add_edge("select", "synthesize")
builder.add_edge("synthesize", END)

tot_graph = builder.compile()

# Use it
result = tot_graph.invoke({
    "question": "How should we approach climate change?"
})

print("Best thought:", result["best_thought"]["reasoning"])
print("Final answer:", result["final_answer"])
```


### Pattern 3: Self-Reflection

Agent critiques its own output:

```python
class ReflectionState(TypedDict):
    question: str
    initial_response: str
    critique: str
    refined_response: str
    reflection_count: int

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
    Response: {state['initial_response']}
    
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
    Your response: {state['initial_response']}
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

```python
from pydantic import BaseModel, field_validator

class ResearchOutput(BaseModel):
    """Structured research output."""
    topic: str
    key_findings: list[str]
    sources: list[str]
    confidence_score: float
    
    @field_validator('confidence_score')
    def score_in_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Must be between 0 and 1')
        return v

class StructuredState(TypedDict):
    topic: str
    raw_research: str
    structured_output: ResearchOutput
    validation_passed: bool
    errors: list[str]

def research_node(state: StructuredState) -> dict:
    """Conduct research."""
    
    result = model.invoke(f"Research: {state['topic']}")
    
    return {"raw_research": result.content}

def structure_output(state: StructuredState) -> dict:
    """Parse into structured format."""
    
    prompt = f"""
    Research content: {state['raw_research']}
    
    Extract into JSON with fields:
    - topic
    - key_findings (list)
    - sources (list)
    - confidence_score (0-1)
    """
    
    response = model.invoke(prompt)
    
    try:
        import json
        parsed = json.loads(response.content)
        output = ResearchOutput(**parsed)
        return {
            "structured_output": output,
            "validation_passed": True,
            "errors": []
        }
    except Exception as e:
        return {
            "validation_passed": False,
            "errors": [str(e)]
        }

def decide_next(state: StructuredState) -> str:
    """Route based on validation."""
    if state["validation_passed"]:
        return "success"
    else:
        return "retry"

def retry_node(state: StructuredState) -> dict:
    """Re-attempt with error context."""
    
    prompt = f"""
    Previous errors: {', '.join(state['errors'])}
    Retry research on: {state['topic']}
    """
    
    result = model.invoke(prompt)
    
    return {"raw_research": result.content}

# Build validation graph
builder = StateGraph(StructuredState)
builder.add_node("research", research_node)
builder.add_node("structure", structure_output)
builder.add_node("retry", retry_node)

builder.add_edge(START, "research")
builder.add_edge("research", "structure")

builder.add_conditional_edges(
    "structure",
    decide_next,
    {"success": END, "retry": "retry"}
)

builder.add_edge("retry", "structure")  # Loop back

validation_graph = builder.compile()

# Use it
result = validation_graph.invoke({
    "topic": "AI safety"
})

if result["validation_passed"]:
    output = result["structured_output"]
    print(f"Topic: {output.topic}")
    print(f"Confidence: {output.confidence_score}")
    print(f"Findings: {output.key_findings}")
```

### Pattern 5: Node caching with `CachePolicy` and `InMemoryCache`

`CachePolicy` memoizes a node's output by its input hash. The first call executes the node; subsequent calls with the same input skip it and return the cached result. Wire a `BaseCache` backend to `compile(cache=...)`.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


class EmbedState(TypedDict):
    text: str
    embedding: list[float]


def embed_text(state: EmbedState) -> dict:
    """Expensive embedding — cached after first call for the same input."""
    print(f"[embed] computing embedding for: {state['text'][:30]}...")
    # Simulate embedding (replace with a real model call)
    return {"embedding": [len(state["text"]) * 0.01, 0.5, 0.3]}


cache = InMemoryCache()

builder = StateGraph(EmbedState)
builder.add_node(
    "embed",
    embed_text,
    cache_policy=CachePolicy(ttl=3600),   # cache for 1 hour
)
builder.add_edge(START, "embed")
builder.add_edge("embed", END)

graph = builder.compile(
    cache=cache,
    checkpointer=InMemorySaver(),
)

cfg1 = {"configurable": {"thread_id": "t1"}}
cfg2 = {"configurable": {"thread_id": "t2"}}

# First call: executes `embed_text` and stores the result in `cache`
result1 = graph.invoke({"text": "Hello world", "embedding": []}, cfg1)
print(result1["embedding"])   # computed

# Second call with identical text (different thread): hits the cache, no print
result2 = graph.invoke({"text": "Hello world", "embedding": []}, cfg2)
print(result2["embedding"])   # same value, from cache
```

**Custom `key_func`** — override the cache key when you need a deterministic, human-readable key:

```python
from langgraph.types import CachePolicy

def text_key(state: EmbedState) -> str:
    return state["text"].strip().lower()

builder.add_node("embed", embed_text, cache_policy=CachePolicy(key_func=text_key, ttl=600))
```

**Clearing the cache** — call `cache.clear()` to wipe all entries, or `cache.clear(namespaces=[...])` for targeted eviction.

---

### Pattern 6: `RetryPolicy` — custom predicates and layered strategies

`RetryPolicy` is a `NamedTuple` applied per node (or per `@task`). Beyond simple exception types, you can pass a callable that returns `True` to trigger a retry.

```python
import httpx
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class FetchState(TypedDict):
    url: str
    body: str


# ── Predicate-based retry ────────────────────────────────────────────────────
def should_retry(exc: Exception) -> bool:
    """Retry 5xx and network errors, but not 4xx client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    if isinstance(exc, httpx.TransportError):
        return True
    return False


def fetch_node(state: FetchState) -> dict:
    resp = httpx.get(state["url"], timeout=10)
    resp.raise_for_status()
    return {"body": resp.text[:200]}


# ── Layered retry sequence ───────────────────────────────────────────────────
# First policy handles transient HTTP errors with fast retries.
# Second policy catches anything else with a slower, longer-lived strategy.
fast_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    retry_on=should_retry,
)
slow_retry = RetryPolicy(
    max_attempts=5,
    initial_interval=2.0,
    backoff_factor=1.5,
    max_interval=30.0,
    retry_on=Exception,      # fallback: any exception
)

builder = StateGraph(FetchState)
builder.add_node(
    "fetch",
    fetch_node,
    retry_policy=[fast_retry, slow_retry],   # first matching policy wins
)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)

graph = builder.compile()
```

Key `RetryPolicy` fields (all have defaults):

| Field | Default | Effect |
|---|---|---|
| `max_attempts` | `3` | Total attempts including first |
| `initial_interval` | `0.5` | Seconds before first retry |
| `backoff_factor` | `2.0` | Multiplier per retry |
| `max_interval` | `128.0` | Cap on interval seconds |
| `jitter` | `True` | Random noise added to interval |
| `retry_on` | transient HTTP/network | Exception type(s) or `(exc) -> bool` |

---

### Pattern 7: `TimeoutPolicy` — wall-clock and idle timeouts with heartbeat

`TimeoutPolicy` applies to async nodes and `@task`s (sync tasks cannot be cancelled in-process). Set `run_timeout` for a hard wall-clock cap, `idle_timeout` for a progress-based cap, and call `runtime.heartbeat()` to refresh the idle timer from inside slow work.

```python
import asyncio
from datetime import timedelta
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import TimeoutPolicy


class ScrapeState(TypedDict):
    urls: list[str]
    results: list[str]


async def slow_scrape(state: ScrapeState, runtime: Runtime) -> dict:
    """Scrapes several URLs; heartbeat refreshes the idle timer between pages."""
    collected = []
    for url in state["urls"]:
        await asyncio.sleep(1)          # simulate network I/O
        collected.append(f"content:{url}")
        runtime.heartbeat()             # refresh idle_timeout after each page
    return {"results": collected}


builder = StateGraph(ScrapeState)
builder.add_node(
    "scrape",
    slow_scrape,
    # Hard cap: whole node cannot run longer than 30 seconds total.
    # Idle cap: if no heartbeat is received within 5 seconds, cancel.
    # refresh_on="heartbeat" means only explicit heartbeat() calls reset the idle timer.
    timeout=TimeoutPolicy(
        run_timeout=30.0,
        idle_timeout=5.0,
        refresh_on="heartbeat",
    ),
)
builder.add_edge(START, "scrape")
builder.add_edge("scrape", END)

graph = builder.compile()
```

`TimeoutPolicy` also accepts `timedelta` for both timeout fields:

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

# Using timedelta for more readable durations
timeout = TimeoutPolicy(
    run_timeout=timedelta(minutes=2),   # 2-minute hard cap
    idle_timeout=timedelta(seconds=15), # 15-second idle cap
    refresh_on="auto",                  # default: any progress resets the idle timer
)
```

`TimeoutPolicy` dataclass fields:

| Field | Type | Default | Effect |
|---|---|---|---|
| `run_timeout` | `float \| timedelta \| None` | `None` | Hard wall-clock cap on total node runtime |
| `idle_timeout` | `float \| timedelta \| None` | `None` | Max time allowed without a progress signal |
| `refresh_on` | `"auto" \| "heartbeat"` | `"auto"` | What resets `idle_timeout` |

`refresh_on` values:

| Value | What resets `idle_timeout` |
|---|---|
| `"auto"` (default) | LangGraph progress signals **and** `runtime.heartbeat()` |
| `"heartbeat"` | Only explicit `runtime.heartbeat()` calls |

When the timeout fires, `NodeTimeoutError` is raised inside the task. If a `retry_policy` is also set, the retry machinery decides whether to retry.

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
    model = "claude-opus" if ctx.is_premium else "claude-haiku"

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
| `server_info` | `ServerInfo \| None` | LangGraph Platform only |

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

Since LangGraph 1.2.1, `Send` accepts an optional `timeout` keyword argument. This lets each individual fan-out branch carry its own deadline independently of the node-level `TimeoutPolicy`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
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


def fetch_page(state: PageInput) -> dict:
    import time
    time.sleep(0.1)   # simulate network
    return {"results": [f"content:{state['url']}"]}


builder = StateGraph(CrawlState)
builder.add_node("fetch_page", fetch_page)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("fetch_page", END)

graph = builder.compile()
result = graph.invoke({"urls": ["a.com", "b.com", "c.com"], "results": []})
print(result["results"])
```

**When to use per-`Send` timeout vs. node-level `TimeoutPolicy`:**

| Scenario | Recommendation |
|---|---|
| All fan-out branches share the same deadline | Node-level `TimeoutPolicy` on `add_node` |
| Different branches need different deadlines | Per-`Send` `timeout=` |
| You need idle detection per-branch | Per-`Send` `timeout=TimeoutPolicy(idle_timeout=...)` |

---

### Pattern 11: `add_sequence()` — concise linear pipelines

`StateGraph.add_sequence()` is a convenience method introduced in LangGraph 1.2.1. It registers a list of nodes and automatically wires them with edges — no separate `add_node` + `add_edge` calls needed. Each element is either a callable (name is derived from `__name__`) or a `("name", callable)` tuple.

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
builder.add_sequence([
    clean,                          # name inferred: "clean"
    ("summarize_v2", summarize),    # explicit name
    translate,                      # name inferred: "translate"
])

# You can still attach policies by name after the sequence is registered:
builder.add_node(
    "summarize_v2",
    summarize,
    retry_policy=RetryPolicy(max_attempts=3),
    cache_policy=CachePolicy(ttl=300),
)
```

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

```python
from langgraph.types import Overwrite

def consolidate(state: PipelineState) -> dict:
    # Deduplicate and canonicalise — then overwrite so nothing else appends
    canonical = list(dict.fromkeys(state["results"]))
    return {"results": Overwrite(canonical)}
```

---

### Pattern 13: `GraphOutput` with `version="v2"`

The `invoke` and `stream` APIs accept an optional `version` keyword argument. When `version="v2"`, the return type changes from a raw state dict to a `GraphOutput` named-tuple that bundles the output value with any `Interrupt` objects raised during the run.

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

**Streaming with `version="v2"`** — each emitted event in `stream_mode="values"` is also a `GraphOutput`:

```python
for chunk in graph.stream(
    {"document": "Board minutes", "approved": False},
    cfg,
    stream_mode="values",
    version="v2",
):
    print(chunk.value, chunk.interrupts)
```

> **Migration note:** `version="v2"` is opt-in and backward-compatible. Existing code that omits `version=` continues to receive plain dicts.

---

### Pattern 14: `context_schema` and `Runtime[Context]` — complete injection example

This pattern expands on Pattern 8 to show the minimal wiring needed for context injection from scratch, including the `configurable` key used by LangGraph Platform deployments.

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
- On LangGraph Platform, context can also arrive via `{"configurable": {"context": {...}}}` in the run config (the platform serialises/deserialises the dataclass automatically).
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
    futures = [fetch(u) for u in urls]
    return [f.result() for f in futures]
```

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

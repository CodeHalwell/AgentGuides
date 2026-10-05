---
title: "Class Deep Dives Vol. 1 — 10 API Classes (v1.2.12)"
description: "Source-verified deep dives for Send, Command, StateSnapshot, Interrupt, TracePolicy, InMemoryStore, InjectedState, InjectedStore, RetryPolicy, and CachePolicy — each with multiple runnable code examples."
---

# Class Deep Dives Vol. 1 — 10 API Classes (v1.2.12)

> **Source-verified against `langgraph==1.2.12`** — October 2026.

This reference dives into ten classes that appear throughout LangGraph but are often
used only in their simplest form. Each section shows the complete constructor signature,
explains every field from the source, and gives **multiple runnable examples** covering
the broad range of real-world usage patterns.

---

## 1. `Send` — Dynamic Fan-Out

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import Send`

```python
class Send:
    node: str              # target node name
    arg:  Any              # state dict (or any value) to pass to that node
    timeout: float | timedelta | TimeoutPolicy | None = None
```

`Send` is returned from a conditional edge to spawn one or more parallel branches,
each carrying its own isolated slice of state. LangGraph waits for every branch to
finish before moving on.

### Example 1a — Basic map-reduce (parallel fan-out + aggregation)

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

# ── State ──────────────────────────────────────────────────────────────────────
class OverallState(TypedDict):
    topics: list[str]
    # Annotated with operator.add so each branch's result is appended.
    summaries: Annotated[list[str], operator.add]

class WorkerState(TypedDict):
    topic: str

# ── Nodes ──────────────────────────────────────────────────────────────────────
def summarise(state: WorkerState) -> dict:
    # In production: call an LLM here.
    return {"summaries": [f"Summary of '{state['topic']}'"]}

def fan_out(state: OverallState) -> list[Send]:
    # Return one Send per topic — LangGraph runs them in parallel.
    return [Send("summarise", {"topic": t}) for t in state["topics"]]

# ── Graph ──────────────────────────────────────────────────────────────────────
builder = StateGraph(OverallState)
builder.add_node("summarise", summarise)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("summarise", END)
graph = builder.compile()

result = graph.invoke({"topics": ["LangGraph", "RAG", "Tool Calling"]})
print(result["summaries"])
# ["Summary of 'LangGraph'", "Summary of 'RAG'", "Summary of 'Tool Calling'"]
```

### Example 1b — Per-task timeout via `Send.timeout`

Set a deadline on individual parallel branches without touching the graph-wide policy.
Timed-out branches raise `NodeTimeoutError`, which can be caught by an error handler.

```python
import asyncio
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send, TimeoutPolicy

class State(TypedDict):
    urls: list[str]
    results: Annotated[list[str], operator.add]

class FetchState(TypedDict):
    url: str

async def fetch(state: FetchState) -> dict:
    # Simulate a slow fetch — TimeoutPolicy cancellation requires async execution.
    await asyncio.sleep(0.1)
    return {"results": [f"content:{state['url']}"]}

def start_fetches(state: State) -> list[Send]:
    return [
        Send(
            "fetch",
            {"url": url},
            # Hard cap of 5 seconds per fetch task.
            timeout=TimeoutPolicy(run_timeout=5.0),
        )
        for url in state["urls"]
    ]

builder = StateGraph(State)
builder.add_node("fetch", fetch)
builder.add_conditional_edges(START, start_fetches)
builder.add_edge("fetch", END)

graph = builder.compile()
out = asyncio.run(graph.ainvoke({"urls": ["https://a.example", "https://b.example"]}))
print(out["results"])
```

### Example 1c — Nested fan-out (two-level parallelism via Command)

A `Send` branch can itself fan out further by returning a `Command` with a
`goto` list of inner `Send` objects. Conditional edge functions receive the
**root graph state**, not the branch-local `Send` payload, so the secondary
fan-out logic must live inside the node itself.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, Send

class RootState(TypedDict):
    sections: list[str]
    chunks: Annotated[list[str], operator.add]

class SectionState(TypedDict):
    section: str
    chunks: Annotated[list[str], operator.add]

class ChunkState(TypedDict):
    chunk: str

# Level-1 conditional edge: fan out sections from root state
def expand_sections(state: RootState) -> list[Send]:
    return [Send("section_entry", {"section": s, "chunks": []}) for s in state["sections"]]

# Level-1 node: receives the section-local payload and fans out chunks via Command.
# The node (not a conditional edge) performs the inner fan-out so it can access
# the branch-local `section` value from the Send payload.
def section_entry(state: SectionState) -> Command:
    words = state["section"].split()
    mid = max(1, len(words) // 2)
    parts = [" ".join(words[:mid]), " ".join(words[mid:])]
    return Command(goto=[Send("process_chunk", {"chunk": p}) for p in parts if p])

# Level-2 node: process each chunk independently
def process_chunk(state: ChunkState) -> dict:
    return {"chunks": [state["chunk"].upper()]}

builder = StateGraph(RootState)
builder.add_node("section_entry", section_entry)
builder.add_node("process_chunk", process_chunk)
builder.add_conditional_edges(START, expand_sections)
builder.add_edge("process_chunk", END)

graph = builder.compile()
result = graph.invoke({"sections": ["hello world", "foo bar baz"], "chunks": []})
print(result["chunks"])  # ['HELLO', 'WORLD', 'FOO BAR', 'BAZ'] (order may vary)
```

---

## 2. `Command` — Update State and Route in One Step

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import Command`

```python
@dataclass
class Command:
    graph:  str | None = None          # None = current graph; Command.PARENT = parent
    update: Any | None = None          # state patch applied before routing
    resume: dict | Any | None = None   # value(s) to resume pending interrupt(s)
    goto:   Send | Sequence[Send | str] | str = ()
```

A `Command` lets a node (or tool) do two things simultaneously: **patch the graph
state** and **direct the next routing**. This eliminates the need for separate routing
functions when the decision depends on data only the node itself computed.

### Example 2a — Combined `update` + `goto` in one node

```python
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command

class State(TypedDict):
    score: int
    verdict: str

def evaluate(state: State) -> Command:
    score = state["score"]
    if score >= 80:
        return Command(update={"verdict": "pass"}, goto="approve")
    elif score >= 50:
        return Command(update={"verdict": "review"}, goto="manual_review")
    else:
        return Command(update={"verdict": "fail"}, goto="reject")

def approve(state: State) -> dict:
    print(f"Approved with score {state['score']}")
    return {}

def manual_review(state: State) -> dict:
    print(f"Needs manual review — score {state['score']}")
    return {}

def reject(state: State) -> dict:
    print(f"Rejected — score {state['score']}")
    return {}

builder = StateGraph(State)
builder.add_node("evaluate", evaluate)
builder.add_node("approve", approve)
builder.add_node("manual_review", manual_review)
builder.add_node("reject", reject)
builder.add_edge(START, "evaluate")
for node in ("approve", "manual_review", "reject"):
    builder.add_edge(node, END)

graph = builder.compile()
graph.invoke({"score": 85, "verdict": ""})  # → "Approved with score 85"
graph.invoke({"score": 30, "verdict": ""})  # → "Rejected"
```

### Example 2b — Cross-subgraph routing with `Command.PARENT`

A subgraph node can escalate control to its parent by using `graph=Command.PARENT`.
The parent receives the state update and routes from there.

```python
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command

# ── Inner subgraph ─────────────────────────────────────────────────────────────
class InnerState(TypedDict):
    task: str
    escalate: bool

def inner_worker(state: InnerState) -> Command | dict:
    if state["escalate"]:
        # Bubble up to the parent graph and route there.
        return Command(
            graph=Command.PARENT,
            update={"status": "escalated"},
            goto="escalation_handler",
        )
    return {"task": state["task"] + " [done]"}

inner = StateGraph(InnerState)
inner.add_node("worker", inner_worker)
inner.add_edge(START, "worker")
inner.add_edge("worker", END)
inner_graph = inner.compile()

# ── Outer graph ────────────────────────────────────────────────────────────────
class OuterState(TypedDict):
    task: str
    escalate: bool
    status: str

def escalation_handler(state: OuterState) -> dict:
    print(f"Escalation received for task: {state['task']}")
    return {"status": "handled"}

outer = StateGraph(OuterState)
outer.add_node("subgraph", inner_graph)
outer.add_node("escalation_handler", escalation_handler)
outer.add_edge(START, "subgraph")
outer.add_edge("subgraph", END)
outer.add_edge("escalation_handler", END)

graph = outer.compile()

# Normal run — no escalation
result = graph.invoke({"task": "report", "escalate": False, "status": ""})
print(result)

# Escalated run — inner node hands control to parent
result = graph.invoke({"task": "report", "escalate": True, "status": ""})
print(result)  # status == "handled"
```

### Example 2c — `Command` returned from a `@tool` (tool-driven routing)

When `ToolNode` executes a tool that returns a `Command`, it unwraps the `Command`
into state updates and routing signals. This lets tools drive agent hand-offs.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str

@tool
def escalate_to_specialist(
    department: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Escalate this conversation to a specialist department."""
    # When a tool returns Command, ToolNode requires a ToolMessage in the update
    # so the message history stays valid (one ToolMessage per tool_call_id).
    return Command(
        update={
            "active_agent": department,
            "messages": [ToolMessage(content=f"Escalating to {department}.", tool_call_id=tool_call_id)],
        },
        goto=department,
    )

tool_node = ToolNode([escalate_to_specialist])

def billing_agent(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="Billing agent handling your request.")]}

def technical_agent(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="Technical agent here.")]}

# Simulate: triage LLM decides to escalate
def triage(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "escalate_to_specialist",
                    "args": {"department": "billing"},
                    "id": "call_1",
                    "type": "tool_call",
                }],
            )
        ]
    }

builder = StateGraph(AgentState)
builder.add_node("triage", triage)
builder.add_node("tools", tool_node)
builder.add_node("billing", billing_agent)
builder.add_node("technical", technical_agent)
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", tools_condition)
builder.add_edge("billing", END)
builder.add_edge("technical", END)

graph = builder.compile()
result = graph.invoke({
    "messages": [("user", "I have a billing question")],
    "active_agent": "triage",
})
print(result["active_agent"])  # "billing"
```

---

## 3. `StateSnapshot` — Inspecting Graph History

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import StateSnapshot`

```python
class StateSnapshot(NamedTuple):
    values:        dict[str, Any] | Any   # current channel values
    next:          tuple[str, ...]         # node(s) scheduled to run next
    config:        RunnableConfig          # config that produced this snapshot
    metadata:      CheckpointMetadata | None
    created_at:    str | None              # ISO timestamp
    parent_config: RunnableConfig | None   # config of the preceding snapshot
    tasks:         tuple[PregelTask, ...]  # tasks that ran (or will run)
    interrupts:    tuple[Interrupt, ...]   # pending interrupts
```

`graph.get_state()` returns a `StateSnapshot` for the current checkpoint.
`graph.get_state_history()` yields every snapshot from newest to oldest.

### Example 3a — Inspecting all snapshot fields

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    count: int
    messages: list[str]

def step_a(state: State) -> dict:
    return {"count": state["count"] + 1, "messages": state["messages"] + ["a ran"]}

def step_b(state: State) -> dict:
    return {"count": state["count"] + 10, "messages": state["messages"] + ["b ran"]}

builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo"}}
graph.invoke({"count": 0, "messages": []}, config=config)

# ── Current snapshot ───────────────────────────────────────────────────────────
snap = graph.get_state(config)

print("values:     ", snap.values)           # {'count': 11, 'messages': ['a ran', 'b ran']}
print("next:       ", snap.next)             # () — graph is finished
print("created_at: ", snap.created_at)       # ISO timestamp string
print("metadata:   ", snap.metadata)         # step count, source, writes, parents
print("tasks:      ", snap.tasks)            # tasks that completed at this step
print("interrupts: ", snap.interrupts)       # () — no pending interrupts

# Access metadata sub-fields
if snap.metadata:
    print("step:   ", snap.metadata.get("step"))
    print("source: ", snap.metadata.get("source"))  # "loop", "input", etc.
```

### Example 3b — Time-travel: fork from a historical snapshot

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    value: int

def add_ten(state: State) -> dict:
    return {"value": state["value"] + 10}

def double(state: State) -> dict:
    return {"value": state["value"] * 2}

checkpointer = MemorySaver()
builder = StateGraph(State)
builder.add_node("add_ten", add_ten)
builder.add_node("double", double)
builder.add_edge(START, "add_ten")
builder.add_edge("add_ten", "double")
builder.add_edge("double", END)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "time-travel-demo"}}
graph.invoke({"value": 5}, config=config)
# → value == 30  (5 + 10 = 15, 15 * 2 = 30)

# Walk history — snapshots are newest → oldest
history = list(graph.get_state_history(config))
print(f"{len(history)} snapshots recorded")
for snap in history:
    print(f"  step {snap.metadata['step'] if snap.metadata else '?'}:"
          f" value={snap.values.get('value')!r}  next={snap.next}")

# Fork from the snapshot where `add_ten` just finished (value=15)
# Find the snapshot where `double` is about to run
fork_snap = next(s for s in history if "double" in s.next)

# Inject a modified value and re-run from that point.
# update_state returns a new RunnableConfig pointing at the fork checkpoint.
new_config = graph.update_state(fork_snap.config, {"value": 100})
result = graph.invoke(None, config=new_config)
print(result)  # {'value': 200}  (100 * 2)
```

---

## 4. `Interrupt` — Typed Human-in-the-Loop

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import interrupt, Interrupt`

```python
@dataclass(slots=True)
class Interrupt:
    value:           Any              # payload shown to the human
    id:              str              # stable hash-based ID
    response_schema: type | dict | None = None  # JSON Schema for the expected resume value
```

`interrupt(value)` raises an `Interrupt` mid-node. The graph pauses and surfaces
the interrupt in the snapshot's `.interrupts` tuple. Resume with
`Command(resume=answer)` or `Command(resume={id: answer})` for per-id targeting.

### Example 4a — Typed interrupt with `response_schema`

`response_schema` accepts either a **JSON Schema dict** (pure metadata — no
validation) or a **Python type** such as a Pydantic model, `TypedDict`, or
dataclass (LangGraph validates the resume value and raises
`pydantic.ValidationError` on mismatch, then returns the constructed model
instance). Pass a JSON Schema dict when you want the schema surfaced to callers
without enforcing it, or when using a persistent checkpointer that cannot
serialize Python class objects.

The `Interrupt` stored in the snapshot always carries `response_schema` as a
**JSON Schema dict** — even when you passed a Python type, it is converted
before storage.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

# A plain JSON Schema dict — metadata-only, no validation at resume time.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["approve", "reject", "escalate"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

class State(TypedDict):
    proposal: str
    decision: dict | None

def review(state: State) -> dict:
    # Interrupt with a schema hint — callers know what shape to resume with.
    decision = interrupt(
        {"proposal": state["proposal"], "message": "Please review and decide."},
        response_schema=DECISION_SCHEMA,
    )
    return {"decision": decision}

builder = StateGraph(State)
builder.add_node("review", review)
builder.add_edge(START, "review")
builder.add_edge("review", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "typed-interrupt"}}

# First run — pauses at interrupt
result = list(graph.stream({"proposal": "Deploy to prod", "decision": None}, config))
snap = graph.get_state(config)

# Inspect the interrupt — response_schema is always stored as a JSON Schema dict
intr = snap.interrupts[0]
print("interrupt id:     ", intr.id)
print("interrupt value:  ", intr.value)
print("response_schema:  ", intr.response_schema)  # {'type': 'object', ...}

# Resume with the structured decision — returned as-is (no validation with dict schema)
answer = {"action": "approve", "reason": "All checks passed."}
result = graph.invoke(Command(resume=answer), config=config)
print(result["decision"])  # {'action': 'approve', 'reason': 'All checks passed.'}
```

### Example 4b — Sequential resume by interrupt id

When a node calls `interrupt()` more than once (each call pauses the graph
and requires a separate resume), each interrupt gets a unique stable ID.
Use `Command(resume={id: value})` to answer one interrupt at a time and
advance through the sequence.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    legal_ok: bool
    finance_ok: bool

def dual_approval(state: State) -> dict:
    legal   = interrupt({"department": "legal",   "question": "Approve the contract?"})
    finance = interrupt({"department": "finance",  "question": "Approve the budget?"})
    return {"legal_ok": legal == "yes", "finance_ok": finance == "yes"}

builder = StateGraph(State)
builder.add_node("dual_approval", dual_approval)
builder.add_edge(START, "dual_approval")
builder.add_edge("dual_approval", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "dual"}}

# Run 1 — graph pauses at the legal interrupt
list(graph.stream({"legal_ok": False, "finance_ok": False}, config))
snap = graph.get_state(config)
legal_id = snap.interrupts[0].id
print("Legal interrupt id:", legal_id)

# Resume with the legal answer → graph pauses at the finance interrupt
list(graph.stream(Command(resume={legal_id: "yes"}), config))
snap2 = graph.get_state(config)
finance_id = snap2.interrupts[0].id
print("Finance interrupt id:", finance_id)

# Resume with the finance answer → graph completes
result = graph.invoke(Command(resume={finance_id: "yes"}), config)
print(result)  # {'legal_ok': True, 'finance_ok': True}
```

### Example 4c — Collecting retained interrupts across snapshots

> **Note:** Checkpoint history is **not** a complete interrupt audit log.
> Multiple `interrupt()` calls in the same node reuse the same task/checkpoint
> slot, so earlier interrupt writes are overwritten by later ones in the
> snapshot. This helper yields only the **interrupts retained** in each
> snapshot — not every interrupt that ever fired. Capture streamed interrupt
> events during execution when a complete audit trail is required.

```python
# After a graph has run with interrupts, inspect the retained interrupt snapshots:
def collect_interrupts(graph, config):
    """Yield Interrupt objects retained in checkpoint history (not a complete log)."""
    for snapshot in graph.get_state_history(config):
        for intr in snapshot.interrupts:
            yield intr

# Usage after running the dual-approval graph above:
for intr in collect_interrupts(graph, config):
    print(f"id={intr.id[:8]}…  value={intr.value}")
```

---

## 5. `TracePolicy` — Controlling What Gets Recorded

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import TracePolicy`

```python
@dataclass
class TracePolicy:
    process_inputs:  Callable[[Any], Any] | None = None
    process_outputs: Callable[[Any], Any] | None = None
```

`TracePolicy` transforms what a **node's own trace run** records in LangSmith (or
any tracer). It does **not** affect the value passed to or returned by the node —
only the recorded span. Attach it per node via `add_node(..., trace_policy=...)`.
(`set_node_defaults` does **not** accept `trace_policy` — use `add_node` for each node.)

> **Scope:** only the node's own run span is filtered. The root graph trace span
> still records the original, unredacted inputs/outputs. Child runs created by
> traced runnables inside the node are also unaffected.

### Example 5a — Redacting PII from inputs before tracing

```python
import copy
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import TracePolicy

PII_FIELDS = {"email", "phone", "ssn", "password", "credit_card"}

def redact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip any message content that looks like PII field names."""
    cleaned = []
    for msg in messages:
        if isinstance(msg.content, str):
            content = msg.content
            for field in PII_FIELDS:
                if field in content.lower():
                    content = "[REDACTED]"
                    break
            # Use copy() to preserve message id, tool_call_id, and other fields.
            cleaned.append(msg.copy(update={"content": content}))
        else:
            cleaned.append(msg)
    return cleaned

def redact_state(state):
    """Return a copy of the state with messages redacted."""
    if not isinstance(state, dict) or "messages" not in state:
        return state
    result = dict(state)
    result["messages"] = redact_messages(state["messages"])
    return result

pii_policy = TracePolicy(
    process_inputs=redact_state,
    process_outputs=redact_state,
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    # In production: call an LLM
    return {"messages": [AIMessage(content="I cannot share personal data.")]}

builder = StateGraph(ChatState)
builder.add_node(
    "chat",
    chat_node,
    trace_policy=pii_policy,
)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

graph.invoke({
    "messages": [HumanMessage(content="My email is alice@example.com")]
})
# LangSmith records "[REDACTED]" instead of the email address.
```

### Example 5b — Summarising a long message history before recording

Avoid recording thousands of tokens by keeping only the last few messages in the
trace span while passing the full history to the node.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import TracePolicy

def keep_last_n(n: int):
    """Return a TracePolicy that records only the last n messages."""
    def _trim(state):
        if isinstance(state, dict) and "messages" in state:
            return {**state, "messages": state["messages"][-n:]}
        return state
    return TracePolicy(process_inputs=_trim, process_outputs=_trim)

class LongConvState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def summarise_node(state: LongConvState) -> dict:
    # Works with the full history — trace only sees the last 5 messages.
    return {}

builder = StateGraph(LongConvState)
builder.add_node(
    "summarise",
    summarise_node,
    trace_policy=keep_last_n(5),
)
builder.add_edge(START, "summarise")
builder.add_edge("summarise", END)
graph = builder.compile()
```

### Example 5c — Suppressing node outputs from traces

```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import TracePolicy

def drop_outputs(_):
    return None  # Record nothing for outputs

builder = StateGraph(dict)
# Per-node attachment via trace_policy= kwarg:
builder.add_node(
    "my_node",
    lambda s: s,
    trace_policy=TracePolicy(process_outputs=drop_outputs),
)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()
```

---

## 6. `InMemoryStore` — Long-Term Cross-Thread Memory

**Module:** `langgraph.store.memory`  
**Import:** `from langgraph.store.memory import InMemoryStore`

```python
class InMemoryStore(BaseStore):
    def __init__(self, *, index: IndexConfig | None = None) -> None
```

`InMemoryStore` is a dictionary-backed store that supports key-value storage, field
filtering, and (with an embedding function) vector similarity search. Data is
namespaced as tuples of strings.

Key methods:

| Method | Description |
|--------|-------------|
| `put(ns, key, value, *, index=…)` | Store or update an item |
| `get(ns, key)` | Retrieve a single item (or `None`) |
| `search(ns, *, query=…, filter=…, limit=…, offset=…)` | Search items |
| `list_namespaces(*, prefix=…, suffix=…, max_depth=…)` | List namespaces |
| `delete(ns, key)` | Remove an item |
| `batch(ops)` | Execute mixed operations in one call (no transactional guarantees) |

### Example 6a — CRUD operations and batch

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import GetOp, PutOp

store = InMemoryStore()

# ── Put ────────────────────────────────────────────────────────────────────────
store.put(("users", "alice"), "prefs", {"theme": "dark", "lang": "en"})
store.put(("users", "alice"), "history", {"last_login": "2026-10-01"})
store.put(("users", "bob"),   "prefs", {"theme": "light", "lang": "fr"})

# ── Get ────────────────────────────────────────────────────────────────────────
item = store.get(("users", "alice"), "prefs")
print(item.value)         # {'theme': 'dark', 'lang': 'en'}
print(item.namespace)     # ('users', 'alice')
print(item.created_at)    # datetime

# ── Search by field filter ─────────────────────────────────────────────────────
results = store.search(
    ("users",),              # namespace prefix to search under
    filter={"theme": "dark"},
    limit=10,
)
for r in results:
    print(r.namespace, r.key, r.value)

# ── List namespaces ────────────────────────────────────────────────────────────
namespaces = store.list_namespaces(prefix=("users",), max_depth=2)
print(namespaces)  # [('users', 'alice'), ('users', 'bob')]

# ── Batch: mix put + get in one call ──────────────────────────────────────────
from langgraph.store.base import GetOp, PutOp
results = store.batch([
    PutOp(namespace=("cache",), key="k1", value={"data": "hello"}),
    GetOp(namespace=("users", "alice"), key="prefs"),
])
print(results[1].value)   # {'theme': 'dark', 'lang': 'en'}

# ── Delete ────────────────────────────────────────────────────────────────────
store.delete(("users", "alice"), "history")
print(store.get(("users", "alice"), "history"))  # None
```

### Example 6b — Vector search with a custom embedding function

```python
import math
from langgraph.store.memory import InMemoryStore

# Toy embedding: each word maps to a unique dimension.
VOCAB = ["python", "typescript", "machine", "learning", "web", "api", "data"]
def embed(texts: list[str]) -> list[list[float]]:
    def _enc(text: str) -> list[float]:
        words = text.lower().split()
        return [float(word in words) for word in VOCAB]
    return [_enc(t) for t in texts]

store = InMemoryStore(index={
    "dims": len(VOCAB),
    "embed": embed,
    "fields": ["text"],   # which field to embed
})

# Store documents
docs = [
    ("python tutorial",    {"text": "python api web"}),
    ("ml guide",           {"text": "machine learning data"}),
    ("ts reference",       {"text": "typescript api web"}),
]
for key, value in docs:
    store.put(("docs",), key, value)

# Similarity search — "python web" should surface the python and ts docs
results = store.search(("docs",), query="python web", limit=3)
for r in results:
    print(f"{r.key:20s}  score={r.score:.3f}  value={r.value}")
```

### Example 6c — Multi-tenant namespace pattern

Use the user ID (or tenant ID) as a namespace segment to isolate data naturally.

```python
import uuid
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

store = InMemoryStore()
checkpointer = MemorySaver()

class State(TypedDict):
    user_id: str
    message: str
    recall: str | None

def remember(state: State, *, store: BaseStore) -> dict:
    """Write the current message to the user's memory namespace."""
    ns = ("memories", state["user_id"])
    key = uuid.uuid4().hex  # unique key avoids collisions on concurrent saves
    store.put(ns, key, {"text": state["message"]})
    return {}

def recall_node(state: State, *, store: BaseStore) -> dict:
    """Retrieve all stored memories for this user."""
    ns = ("memories", state["user_id"])
    items = store.search(ns, limit=50)
    recall_text = "; ".join(i.value["text"] for i in items)
    return {"recall": recall_text or "No memories yet."}

builder = StateGraph(State)
builder.add_node("remember", remember)
builder.add_node("recall_node", recall_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", "recall_node")
builder.add_edge("recall_node", END)

graph = builder.compile(checkpointer=checkpointer, store=store)

# Thread 1 — user alice
cfg1 = {"configurable": {"thread_id": "t1"}}
graph.invoke({"user_id": "alice", "message": "I like dark mode", "recall": None}, cfg1)

# Thread 2 — different conversation, same user
cfg2 = {"configurable": {"thread_id": "t2"}}
result = graph.invoke({"user_id": "alice", "message": "Remind me of my prefs", "recall": None}, cfg2)
print(result["recall"])  # "I like dark mode; Remind me of my prefs"

# Bob's thread — isolated namespace
cfg3 = {"configurable": {"thread_id": "t3"}}
result = graph.invoke({"user_id": "bob", "message": "Hello", "recall": None}, cfg3)
print(result["recall"])  # "Hello" (only bob's messages)
```

---

## 7. `InjectedState` — Injecting Graph State into Tools

**Module:** `langgraph.prebuilt.tool_node`  
**Import:** `from langgraph.prebuilt import InjectedState`

```python
class InjectedState(InjectedToolArg):
    def __init__(self, field: str | None = None) -> None
```

Annotate a tool parameter with `InjectedState` and `ToolNode` fills it with the
graph state automatically. The parameter is **invisible to the LLM** — it does not
appear in the tool's JSON schema.

- `InjectedState()` or `InjectedState(None)` — injects the entire state dict.
- `InjectedState("field_name")` — injects only `state["field_name"]`.

### Example 7a — Full state vs field-specific injection

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition

class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_count: int

# Full-state injection — tool sees the entire state dict
@tool
def get_session_info(state: Annotated[dict, InjectedState()]) -> str:
    """Return info about the current session."""
    return (
        f"User: {state['user_id']}, "
        f"Sessions: {state['session_count']}, "
        f"Messages so far: {len(state['messages'])}"
    )

# Field-specific injection — cleaner, less coupling
@tool
def personalise_greeting(user_id: Annotated[str, InjectedState("user_id")]) -> str:
    """Generate a personalised greeting using the user ID from state."""
    return f"Hello, {user_id}! Welcome back."

@tool
def count_messages(
    session_count: Annotated[int, InjectedState("session_count")],
) -> str:
    """Report how many sessions the user has had."""
    return f"You have had {session_count} session(s) with us."

# Build the graph
def agent(state: AppState) -> dict:
    # Simulate LLM choosing to call get_session_info
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_session_info", "args": {}, "id": "c1", "type": "tool_call"},
                    {"name": "personalise_greeting", "args": {}, "id": "c2", "type": "tool_call"},
                ],
            )
        ]
    }

tool_node = ToolNode([get_session_info, personalise_greeting, count_messages])

builder = StateGraph(AppState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()
result = graph.invoke({
    "messages": [("user", "Tell me about my session")],
    "user_id": "alice",
    "session_count": 7,
})
for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(msg.content)
```

### Example 7b — Combining `InjectedState` with regular tool arguments

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition

class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_user: str
    permissions: list[str]

@tool
def execute_action(
    action: str,                                                    # LLM provides this
    resource: str,                                                  # LLM provides this
    user: Annotated[str, InjectedState("current_user")],           # injected
    permissions: Annotated[list[str], InjectedState("permissions")], # injected
) -> str:
    """Execute an action on a resource if the user has permission."""
    if action not in permissions:
        return f"User '{user}' lacks permission for action '{action}'."
    return f"User '{user}' executed '{action}' on '{resource}' successfully."

def agent_node(state: WorkflowState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "execute_action",
                    # LLM only needs to supply the non-injected args
                    "args": {"action": "delete", "resource": "report.pdf"},
                    "id": "c1",
                    "type": "tool_call",
                }],
            )
        ]
    }

tool_node = ToolNode([execute_action])
builder = StateGraph(WorkflowState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()

# User with delete permission
r = graph.invoke({
    "messages": [("user", "Delete the report")],
    "current_user": "alice",
    "permissions": ["read", "delete"],
})
print(r["messages"][-1].content)  # "User 'alice' executed 'delete' on 'report.pdf' successfully."

# User without delete permission
r = graph.invoke({
    "messages": [("user", "Delete the report")],
    "current_user": "bob",
    "permissions": ["read"],
})
print(r["messages"][-1].content)  # "User 'bob' lacks permission for action 'delete'."
```

---

## 8. `InjectedStore` — Injecting the Persistent Store into Tools

**Module:** `langgraph.prebuilt.tool_node`  
**Import:** `from langgraph.prebuilt import InjectedStore`

```python
class InjectedStore(InjectedToolArg):
    pass  # no constructor args — always injects the full store
```

Annotate a tool parameter with `InjectedStore` and `ToolNode` passes whatever
store was supplied to `graph.compile(store=...)`. Like `InjectedState`, it is
**hidden from the LLM** and does not appear in the tool schema.

The injected value is a `BaseStore` instance — use `put`, `get`, `search`,
`delete`, and `batch` on it.

### Example 8a — Persistent memory tool (read + write)

```python
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

class MemoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

@tool
def save_memory(
    content: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Save a fact about the user to long-term memory."""
    ns = ("memories", user_id)
    key = uuid.uuid4().hex  # unique key avoids collisions on concurrent saves
    store.put(ns, key, {"content": content})
    return f"Saved: '{content}'"

@tool
def recall_memories(
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Recall all stored facts about the user."""
    ns = ("memories", user_id)
    items = store.search(ns, limit=100)
    if not items:
        return "No memories stored yet."
    return "\n".join(f"- {i.value['content']}" for i in items)

def agent_node(state: MemoryState) -> dict:
    from langchain_core.messages import ToolMessage
    msgs = state["messages"]
    tool_names = [m.name for m in msgs if isinstance(m, ToolMessage)]
    if "recall_memories" in tool_names:
        # Both tools done — emit a plain response so tools_condition exits to END.
        return {"messages": [AIMessage(content="Done.")]}
    if "save_memory" in tool_names:
        # Save is done — now recall.
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "recall_memories", "args": {}, "id": "c2", "type": "tool_call"}],
                )
            ]
        }
    # First turn — save the preference.
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "save_memory", "args": {"content": "User prefers Python over Java"}, "id": "c1", "type": "tool_call"}
                ],
            )
        ]
    }

long_term_store = InMemoryStore()
tool_node = ToolNode([save_memory, recall_memories])

builder = StateGraph(MemoryState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
# Loop back so agent can emit a second tool call after the save completes.
# tools_condition routes to END when the last agent message has no tool_calls.
builder.add_edge("tools", "agent")

graph = builder.compile(store=long_term_store)
result = graph.invoke({
    "messages": [("user", "Remember my preference")],
    "user_id": "alice",
})
# ToolMessage from recall_memories:
for msg in result["messages"]:
    if hasattr(msg, "content") and "User prefers" in str(msg.content):
        print(msg.content)
```

### Example 8b — RAG tool: search the store and answer

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Toy embedding
VOCAB = ["python", "javascript", "machine", "learning", "web", "api"]
def embed(texts):
    return [[float(w in t.lower().split()) for w in VOCAB] for t in texts]

@tool
def search_knowledge_base(
    query: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Search the knowledge base and return the top relevant documents."""
    results = store.search(("knowledge",), query=query, limit=3)
    if not results:
        return "No relevant documents found."
    return "\n".join(
        f"[score={r.score:.2f}] {r.value.get('text', '')}"
        for r in results
    )

# Pre-populate the knowledge base
kb_store = InMemoryStore(index={"dims": len(VOCAB), "embed": embed, "fields": ["text"]})
docs = [
    ("doc1", {"text": "python api web framework tutorial"}),
    ("doc2", {"text": "machine learning python data science"}),
    ("doc3", {"text": "javascript web frontend api"}),
]
for key, value in docs:
    kb_store.put(("knowledge",), key, value)

def agent_node(state: RAGState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_knowledge_base",
                    "args": {"query": "python web"},
                    "id": "c1",
                    "type": "tool_call",
                }],
            )
        ]
    }

tool_node = ToolNode([search_knowledge_base])
builder = StateGraph(RAGState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile(store=kb_store)
result = graph.invoke({"messages": [("user", "What do you know about python web?")]})
for msg in result["messages"]:
    if hasattr(msg, "content") and "[score=" in str(msg.content):
        print(msg.content)
```

---

## 9. `RetryPolicy` — Configuring Node Retries

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import RetryPolicy`

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5        # seconds before first retry
    backoff_factor:   float = 2.0        # multiplier applied each retry
    max_interval:     float = 128.0      # cap on inter-retry interval
    max_attempts:     int   = 3          # total attempts (including the first)
    jitter:           bool  = True       # add randomness to avoid thundering herd
    retry_on: type[Exception]
              | Sequence[type[Exception]]
              | Callable[[Exception], bool] = <built-in predicate>
```

The built-in `retry_on` predicate retries on `ConnectionError`, on `httpx` /
`requests` 5xx HTTP errors, and on most other exceptions — but **not** on
`OSError` subclasses (including the built-in `TimeoutError`), not on 4xx
errors (including HTTP 429 rate limits), and not on `ValueError`, `TypeError`,
`RuntimeError`, or other programmer errors. Override `retry_on` with a custom
predicate whenever the default coverage doesn't match your use case.

### Example 9a — Custom `retry_on` predicate

```python
import random
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy

attempt_count = 0

def is_transient(exc: Exception) -> bool:
    """Retry only on transient network errors, not on permanent failures."""
    # ConnectionError covers refused/reset connections.
    # TimeoutError covers timed-out socket operations.
    # OSError is excluded: it includes FileNotFoundError, PermissionError, etc.
    return isinstance(exc, (ConnectionError, TimeoutError))

class State(TypedDict):
    result: str

def flaky_api_call(state: State) -> dict:
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"Simulated network error (attempt {attempt_count})")
    return {"result": f"success on attempt {attempt_count}"}

builder = StateGraph(State)
builder.add_node(
    "call_api",
    flaky_api_call,
    retry_policy=RetryPolicy(
        initial_interval=0.01,   # fast for demos
        backoff_factor=2.0,
        max_attempts=5,
        retry_on=is_transient,   # only retry on transient errors
    ),
)
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

graph = builder.compile()
attempt_count = 0
result = graph.invoke({"result": ""})
print(result)  # {'result': 'success on attempt 3'}
```

### Example 9b — Per-node retry vs graph-wide default

```python
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy

class State(TypedDict):
    data: str

def critical_node(state: State) -> dict:
    # High-value node — retry aggressively
    return {"data": state["data"] + ":critical"}

def best_effort_node(state: State) -> dict:
    # Low-value node — fewer retries, shorter wait
    return {"data": state["data"] + ":best_effort"}

AGGRESSIVE = RetryPolicy(
    initial_interval=0.1,
    backoff_factor=3.0,
    max_attempts=10,
    jitter=True,
)
LENIENT = RetryPolicy(
    initial_interval=0.5,
    max_attempts=2,
    jitter=False,
)
GRAPH_DEFAULT = RetryPolicy(
    initial_interval=0.25,
    max_attempts=3,
)

builder = StateGraph(State)
builder.set_node_defaults(retry_policy=GRAPH_DEFAULT)          # applies to all nodes
builder.add_node("critical", critical_node, retry_policy=AGGRESSIVE)  # overrides default
builder.add_node("best_effort", best_effort_node, retry_policy=LENIENT)  # overrides default
builder.add_edge(START, "critical")
builder.add_edge("critical", "best_effort")
builder.add_edge("best_effort", END)

graph = builder.compile()
result = graph.invoke({"data": "start"})
print(result)  # {'data': 'start:critical:best_effort'}
```

### Example 9c — Policy list: first-match-wins per exception type

When you pass a **list** of `RetryPolicy` objects, LangGraph picks the **first
policy whose `retry_on` predicate matches the raised exception**. The policies
do not chain or waterfall — only the single matching policy's `max_attempts` and
backoff apply. Use distinct predicates to give different exception classes
different retry budgets.

```python
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy

class State(TypedDict):
    value: int

attempt_count = 0

def unreliable_node(state: State) -> dict:
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"transient network error (attempt {attempt_count})")
    return {"value": state["value"] + 1}

# Policy for transient network errors — quick retries
NETWORK_POLICY = RetryPolicy(
    initial_interval=0.01,
    max_attempts=5,
    retry_on=lambda e: isinstance(e, ConnectionError),
)

# Policy for value errors — fewer retries, slower
VALUE_POLICY = RetryPolicy(
    initial_interval=0.5,
    max_attempts=2,
    retry_on=lambda e: isinstance(e, ValueError),
)

builder = StateGraph(State)
builder.add_node(
    "unreliable",
    unreliable_node,
    # NETWORK_POLICY matches ConnectionError, VALUE_POLICY matches ValueError.
    # Only the first matching policy is used — they do not combine.
    retry_policy=[NETWORK_POLICY, VALUE_POLICY],
)
builder.add_edge(START, "unreliable")
builder.add_edge("unreliable", END)

graph = builder.compile()
attempt_count = 0
result = graph.invoke({"value": 0})
print(result)  # {'value': 1}  — succeeded on attempt 3
```

---

## 10. `CachePolicy` — Memoising Deterministic Nodes

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import CachePolicy`

```python
@dataclass
class CachePolicy:
    key_func: Callable[[Any], str] = default_cache_key  # hashes the input
    ttl:      int | None = None                          # seconds; None = forever
```

Attach a `CachePolicy` to a node via `add_node(..., cache_policy=...)` or
`set_node_defaults(cache_policy=...)`. On a cache hit LangGraph returns the stored
output without executing the node. The cache backend is passed to `compile(cache=...)`.

### Example 10a — Caching an expensive node with default key

```python
import time
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import START, END, StateGraph
from langgraph.types import CachePolicy

call_count = 0

class State(TypedDict):
    query: str
    result: str

def expensive_node(state: State) -> dict:
    global call_count
    call_count += 1
    time.sleep(0.01)  # simulate expensive work
    return {"result": f"answer to: {state['query']}"}

cache = InMemoryCache()

builder = StateGraph(State)
builder.add_node(
    "expensive",
    expensive_node,
    cache_policy=CachePolicy(),   # default key = hash of the full input
)
builder.add_edge(START, "expensive")
builder.add_edge("expensive", END)

graph = builder.compile(checkpointer=MemorySaver(), cache=cache)

cfg1 = {"configurable": {"thread_id": "c1"}}
cfg2 = {"configurable": {"thread_id": "c2"}}

# First call — cache miss
call_count = 0
r1 = graph.invoke({"query": "What is LangGraph?", "result": ""}, config=cfg1)
print(f"call_count={call_count}, result={r1['result']}")  # call_count=1

# Second call with the same query — cache hit, node not executed
r2 = graph.invoke({"query": "What is LangGraph?", "result": ""}, config=cfg2)
print(f"call_count={call_count}, result={r2['result']}")  # call_count=1 (cached!)

# Different query — cache miss
r3 = graph.invoke({"query": "What is RAG?", "result": ""}, config=cfg1)
print(f"call_count={call_count}")  # call_count=2
```

### Example 10b — Custom `key_func` for semantic-aware caching

Use a custom key function to cache based only on the semantically relevant parts
of the input, ignoring ephemeral fields like timestamps or thread IDs.

```python
import hashlib, json
from typing_extensions import TypedDict
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import START, END, StateGraph
from langgraph.types import CachePolicy

class State(TypedDict):
    query: str
    user_id: str   # ephemeral — should NOT affect the cache key
    timestamp: str  # ephemeral — should NOT affect the cache key
    result: str

def query_only_key(state: dict) -> str:
    """Cache key based only on the query — ignore user_id and timestamp."""
    canonical = json.dumps({"query": state.get("query", "")}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

call_count = 0
def research_node(state: State) -> dict:
    global call_count
    call_count += 1
    return {"result": f"research result for: {state['query']}"}

cache = InMemoryCache()
builder = StateGraph(State)
builder.add_node(
    "research",
    research_node,
    cache_policy=CachePolicy(key_func=query_only_key),
)
builder.add_edge(START, "research")
builder.add_edge("research", END)

graph = builder.compile(cache=cache)

call_count = 0
# Different users, same query — should share the cache
for user in ("alice", "bob", "carol"):
    state = {"query": "Explain transformers", "user_id": user, "timestamp": "2026-10-05", "result": ""}
    r = graph.invoke(state)
    print(f"{user}: {r['result']} (call_count={call_count})")
# alice: … (call_count=1)
# bob:   … (call_count=1)   ← cache hit
# carol: … (call_count=1)   ← cache hit
```

### Example 10c — TTL-based cache and graph-wide default

```python
from typing_extensions import TypedDict
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import START, END, StateGraph
from langgraph.types import CachePolicy

class State(TypedDict):
    input: str
    output: str

def slow_node(state: State) -> dict:
    return {"output": state["input"].upper()}

def fast_node(state: State) -> dict:
    return {"output": state["output"] + "!"}

cache = InMemoryCache()
builder = StateGraph(State)

# Graph-wide default: cache for 5 minutes
builder.set_node_defaults(cache_policy=CachePolicy(ttl=300))

# slow_node uses the graph default (5 min TTL)
builder.add_node("slow_node", slow_node)

# fast_node overrides: cache for 60 seconds only
builder.add_node("fast_node", fast_node, cache_policy=CachePolicy(ttl=60))

builder.add_edge(START, "slow_node")
builder.add_edge("slow_node", "fast_node")
builder.add_edge("fast_node", END)

graph = builder.compile(cache=cache)
result = graph.invoke({"input": "hello", "output": ""})
print(result["output"])  # "HELLO!"
```

---

## Summary Table

| Class | Module | Key Use Case |
|-------|--------|--------------|
| `Send` | `langgraph.types` | Parallel fan-out with per-task state and optional timeout |
| `Command` | `langgraph.types` | Combine state update + routing; cross-subgraph escalation |
| `StateSnapshot` | `langgraph.types` | Inspect checkpoint values, tasks, interrupts; time-travel |
| `Interrupt` | `langgraph.types` | Typed human-in-the-loop with `response_schema` and per-id resume |
| `TracePolicy` | `langgraph.types` | Redact PII or summarize payloads before LangSmith records them |
| `InMemoryStore` | `langgraph.store.memory` | Cross-thread KV + vector search; multi-tenant namespacing |
| `InjectedState` | `langgraph.prebuilt` | Pass graph state (or a single field) into a tool invisibly |
| `InjectedStore` | `langgraph.prebuilt` | Pass the persistent store into a tool for RAG or memory writes |
| `RetryPolicy` | `langgraph.types` | Custom retry predicates, first-match policy lists, per-node overrides |
| `CachePolicy` | `langgraph.types` | Memoise deterministic nodes with custom keys and TTL |

> All examples verified against **`langgraph==1.2.12`** — October 2026.

---
title: "Chapter 3 — Multi-Agent Systems"
description: "Supervisor routing, parallel fan-out/fan-in, Command-based hand-off, and subgraph composition for coordinating multiple specialist agents in LangGraph 1.2."
framework: langgraph
language: python
sidebar:
  label: "3 · Multi-agent systems"
  order: 3
---

# Chapter 3 — Multi-Agent Systems

**What you'll learn:** four canonical multi-agent topologies — a Command-based supervisor routing to specialists, parallel workers with fan-out/fan-in, direct hand-off between agents using `Command`, and subgraph composition for nested teams.

**Time:** ~30 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

---

## Overview of multi-agent topologies

| Topology | Key mechanism | Best when |
|---|---|---|
| **Supervisor** | One coordinator node returns `Command(goto=...)` to route to specialists | Tasks require classification or sequential delegation |
| **Parallel fan-out / fan-in** | `Send` from a conditional edge launches N workers in parallel | Independent sub-tasks that can run concurrently |
| **Direct hand-off** | A node or tool returns `Command(goto=..., update=...)` | Specialist agents can escalate to each other without a central controller |
| **Subgraph composition** | A compiled `StateGraph` added as a node in a parent graph | Encapsulated teams with their own state and tools |

---

## Example 1: Command-based Supervisor Pattern

A supervisor node inspects state and returns `Command(goto=...)` to route to specialists. No explicit edges from the supervisor to workers are needed — the Command carries the routing intent.

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# --- Specialist tools -------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information."""
    # In production, call a real search API here.
    return f"Search results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely using AST parsing."""
    import ast, operator as op

    OPS = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


research_tools = [search_web]
math_tools = [calculate]


# --- State ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    active_specialist: str


# --- Specialist agent nodes -------------------------------------------------
# Each specialist has its own LLM bound to its tools and its own ToolNode.
# In a real app swap the stub LLM below for ChatOpenAI(...).bind_tools(tools).

try:
    from langchain_openai import ChatOpenAI
    research_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(research_tools)
    math_llm     = ChatOpenAI(model="gpt-4o-mini").bind_tools(math_tools)
    _have_llm = True
except Exception:
    _have_llm = False  # running without API key — demo only


def research_agent(state: AgentState) -> dict:
    if _have_llm:
        response = research_llm.invoke(state["messages"])
    else:
        response = AIMessage(content="[research stub] results found")
    return {"messages": [response], "active_specialist": "research"}


def math_agent(state: AgentState) -> dict:
    if _have_llm:
        response = math_llm.invoke(state["messages"])
    else:
        response = AIMessage(content="[math stub] 42")
    return {"messages": [response], "active_specialist": "math"}


# --- Supervisor node ---------------------------------------------------------
# Returns Command(goto=...) to route to a specialist or END.
# No add_edge calls needed from supervisor to the specialists.

def supervisor(state: AgentState) -> Command[Literal["research_agent", "math_agent", "__end__"]]:
    last = state["messages"][-1]

    # If the most recent message is from a specialist, we're done.
    if isinstance(last, AIMessage) and last.name in ("research_agent", "math_agent"):
        return Command(goto=END)

    content = last.content.lower() if hasattr(last, "content") else ""
    if any(w in content for w in ("search", "research", "find", "who", "what", "when")):
        return Command(goto="research_agent", update={"active_specialist": "research"})
    if any(w in content for w in ("calculate", "compute", "how much", "+", "-", "*", "/")):
        return Command(goto="math_agent", update={"active_specialist": "math"})

    return Command(goto=END)


# --- Tool nodes for each specialist -----------------------------------------

research_tool_node = ToolNode(research_tools)
math_tool_node     = ToolNode(math_tools)


# --- Build the graph ---------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("supervisor",          supervisor)
builder.add_node("research_agent",      research_agent)
builder.add_node("math_agent",          math_agent)
builder.add_node("research_tools",      research_tool_node)
builder.add_node("math_tools",          math_tool_node)

builder.add_edge(START, "supervisor")

# Specialists loop back to the supervisor after finishing.
builder.add_conditional_edges("research_agent", tools_condition,
                               {"tools": "research_tools", "__end__": "supervisor"})
builder.add_edge("research_tools", "supervisor")

builder.add_conditional_edges("math_agent", tools_condition,
                               {"tools": "math_tools", "__end__": "supervisor"})
builder.add_edge("math_tools", "supervisor")

# `supervisor` uses Command(goto=...) — no outgoing edges needed.
# Add `destinations=` only for diagram accuracy:
builder.add_node.__self__  # no-op; destinations set via add_node kwargs above

supervisor_graph = builder.compile(checkpointer=InMemorySaver())

# --- Test -------------------------------------------------------------------

config = {"configurable": {"thread_id": "supervisor-demo-1"}}

result = supervisor_graph.invoke(
    {"messages": [HumanMessage(content="Search for recent LangGraph news")],
     "active_specialist": ""},
    config=config,
)
print("Specialist used:", result["active_specialist"])
print("Last message:", result["messages"][-1].content)
```

Key design decisions:

- `supervisor` returns `Command(goto=...)` — no `add_edge` from `supervisor` to the workers needed.
- Specialists loop back through their `ToolNode` when they emit tool calls, then return to `supervisor`.
- Adding `destinations=("research_agent", "math_agent", END)` to `add_node("supervisor", ...)` makes the Mermaid diagram accurate without affecting execution.

---

## Example 2: Parallel Worker Pattern (Fan-out / Fan-in)

Dispatch N tasks in parallel using `Send`, collect results with a reducer, then aggregate:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# --- State ------------------------------------------------------------------

class WorkflowState(TypedDict):
    tasks: list[dict]
    # `operator.add` merges lists from parallel workers
    results: Annotated[list[dict], operator.add]


# The payload delivered to each worker via Send — workers see only these fields.
class WorkerPayload(TypedDict):
    task_id: str
    task_data: str


# --- Nodes ------------------------------------------------------------------

def dispatch(state: WorkflowState) -> list[Send]:
    """Fan-out: one Send per task launches that many parallel worker copies."""
    return [
        Send("worker", {"task_id": t["id"], "task_data": t["data"], "results": []})
        for t in state["tasks"]
    ]


def worker_node(payload: WorkerPayload) -> dict:
    """Process one task. Returns a single-element list; reducer appends it."""
    return {"results": [{"id": payload["task_id"], "output": payload["task_data"].upper()}]}


def aggregate(state: WorkflowState) -> dict:
    """Fan-in: runs once after ALL parallel workers complete."""
    summary = f"Processed {len(state['results'])} tasks"
    return {"results": [{"id": "summary", "output": summary}]}


# --- Build ------------------------------------------------------------------

builder = StateGraph(WorkflowState)
builder.add_node("worker",    worker_node)
builder.add_node("aggregate", aggregate)

# Conditional edge returning list[Send] triggers parallel fan-out.
builder.add_conditional_edges(START, dispatch, ["worker"])

# Every worker writes to "aggregate"; LangGraph waits for all branches.
builder.add_edge("worker",    "aggregate")
builder.add_edge("aggregate", END)

parallel_graph = builder.compile()

# --- Test -------------------------------------------------------------------

result = parallel_graph.invoke({
    "tasks": [
        {"id": "t1", "data": "hello"},
        {"id": "t2", "data": "world"},
        {"id": "t3", "data": "langgraph"},
    ],
    "results": [],
})

for r in result["results"]:
    print(r["id"], "→", r["output"])
```

Notes:

- `Annotated[list[dict], operator.add]` is a `BinaryOperatorAggregate` channel — each worker appends its list, and `operator.add` concatenates them all.
- `Send("worker", {...})` delivers a custom snapshot to the worker node; workers never see `WorkflowState` directly — only their payload.
- The barrier edge `add_edge(["worker"], "aggregate")` (implicit when every Send targets the same downstream node) ensures `aggregate` runs **once** after all workers finish.

---

## Example 3: Direct Hand-off with `Command`

Agents hand off directly to each other by returning `Command(goto=..., update=...)` — no supervisor required. This is the recommended pattern for peer-to-peer escalation.

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class HandoffState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    handled_by: str
    escalation_reason: str


def tier1_agent(state: HandoffState) -> Command[Literal["tier2_agent", "__end__"]]:
    """First-line support. Escalates complex issues to tier-2."""
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else ""

    if "complex" in content.lower() or "escalate" in content.lower():
        return Command(
            goto="tier2_agent",
            update={
                "messages": [AIMessage(content="Escalating to tier-2 specialist.")],
                "handled_by": "tier2",
                "escalation_reason": "Complex query detected",
            },
        )

    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=f"Tier-1 handled: {content}")],
            "handled_by": "tier1",
        },
    )


def tier2_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    """Specialist support. Always resolves the issue."""
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "your request",
    )
    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=f"Tier-2 resolution for: {last_user_msg}")],
            "handled_by": "tier2",
        },
    )


# --- Build ------------------------------------------------------------------
# Both agents use Command, so no outgoing edges needed from them.

builder = StateGraph(HandoffState)
builder.add_node("tier1_agent", tier1_agent,
                 destinations=("tier2_agent", END))
builder.add_node("tier2_agent", tier2_agent,
                 destinations=(END,))
builder.add_edge(START, "tier1_agent")

handoff_graph = builder.compile(checkpointer=InMemorySaver())

# --- Test -------------------------------------------------------------------

config = {"configurable": {"thread_id": "handoff-demo"}}

# Simple query — tier-1 resolves it
result = handoff_graph.invoke(
    {"messages": [HumanMessage(content="How do I reset my password?")],
     "handled_by": "", "escalation_reason": ""},
    config=config,
)
print("Handler:", result["handled_by"])
print("Response:", result["messages"][-1].content)

# Complex query — tier-1 hands off to tier-2
config2 = {"configurable": {"thread_id": "handoff-demo-2"}}
result2 = handoff_graph.invoke(
    {"messages": [HumanMessage(content="This is a complex billing dispute — escalate please")],
     "handled_by": "", "escalation_reason": ""},
    config=config2,
)
print("Handler:", result2["handled_by"])
print("Escalation reason:", result2["escalation_reason"])
```

Key differences from Example 1:

- `tier1_agent` and `tier2_agent` **both** use `Command(goto=...)` — no routing function, no conditional edges.
- `destinations=` on `add_node` is a diagram hint only; execution is driven by `Command.goto`.
- The `update=` field inside `Command` replaces returning a plain dict — same reducers apply.

---

## Example 4: Subgraph Composition

Encapsulate a team of agents as a compiled subgraph and wire it into a parent orchestrator:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# --- Inner subgraph: a two-step research team --------------------------------

class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    findings: str


def fetch_data(state: ResearchState) -> dict:
    topic = state["messages"][-1].content if state["messages"] else "unknown"
    return {"findings": f"Data about: {topic}"}


def analyse_data(state: ResearchState) -> dict:
    return {
        "messages": [AIMessage(content=f"Analysis: {state['findings']}")],
    }


research_builder = StateGraph(ResearchState)
research_builder.add_node("fetch",   fetch_data)
research_builder.add_node("analyse", analyse_data)
research_builder.add_edge(START,     "fetch")
research_builder.add_edge("fetch",   "analyse")
research_builder.add_edge("analyse", END)

# Compile with checkpointer=True so it inherits the parent's checkpointer.
research_subgraph = research_builder.compile(checkpointer=True)


# --- Outer orchestrator -------------------------------------------------------

class OrchestratorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task_type: str


def classify(state: OrchestratorState) -> dict:
    content = state["messages"][-1].content.lower() if state["messages"] else ""
    return {"task_type": "research" if "research" in content else "chat"}


def router(state: OrchestratorState) -> str:
    return "research_team" if state["task_type"] == "research" else "chat_node"


def chat_node(state: OrchestratorState) -> dict:
    return {"messages": [AIMessage(content="I can help with that directly!")]}


orch_builder = StateGraph(OrchestratorState)
orch_builder.add_node("classify",      classify)
orch_builder.add_node("research_team", research_subgraph)   # compiled subgraph as node
orch_builder.add_node("chat_node",     chat_node)

orch_builder.add_edge(START, "classify")
orch_builder.add_conditional_edges("classify", router,
                                   {"research_team": "research_team", "chat_node": "chat_node"})
orch_builder.add_edge("research_team", END)
orch_builder.add_edge("chat_node",     END)

orchestrator = orch_builder.compile(checkpointer=InMemorySaver())

# --- Test -------------------------------------------------------------------

config = {"configurable": {"thread_id": "orch-demo-1"}}
result = orchestrator.invoke(
    {"messages": [HumanMessage(content="Research the history of LangGraph")],
     "task_type": ""},
    config=config,
)
print("Task type:", result["task_type"])
print("Last message:", result["messages"][-1].content)
```

Key subgraph rules:

- `checkpointer=True` tells the subgraph to inherit the parent's checkpointer instead of running unchecked. Use `checkpointer=False` to disable checkpointing for a subgraph even when the parent has one.
- The subgraph's state schema (`ResearchState`) is separate from the parent's (`OrchestratorState`). State is **not** shared between levels automatically — the parent passes its input to the subgraph node and receives its output.
- Streaming with `subgraphs=True` on the parent graph exposes events emitted inside the subgraph with their namespace path.

---

## Streaming multi-agent events

```python
from langgraph.checkpoint.memory import InMemorySaver

config = {"configurable": {"thread_id": "stream-demo"}}

# Stream updates from every node, including subgraphs
for ns, chunk in orchestrator.stream(
    {"messages": [HumanMessage(content="Research AI safety")], "task_type": ""},
    config,
    stream_mode="updates",
    subgraphs=True,
):
    node_path = " > ".join(n.split(":")[0] for n in ns) if ns else "root"
    print(f"[{node_path}]", chunk)
```

`ns` is a tuple of namespace segments. The `:task_id` suffix is unique per run; split on `":"` to get the stable node names.

---

## Example 5: `graph.as_tool()` — use a compiled graph as a LangChain tool

`CompiledStateGraph.as_tool()` (beta) converts a compiled graph into a `BaseTool` that any LangChain agent or `ToolNode` can call. This lets you compose graphs at the tool-call level: a parent agent calls a sub-agent via a regular tool invocation, and the sub-agent's result is returned as a `ToolMessage`.

```python
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


# ── Specialist sub-agent: research graph ─────────────────────────────────────

class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    topic: str
    summary: str


model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def research_and_summarize(state: ResearchState) -> dict:
    response = model.invoke([
        HumanMessage(
            f"Research this topic and provide a concise 2-paragraph summary: {state['topic']}"
        )
    ])
    return {"summary": response.content, "messages": [response]}


research_builder = StateGraph(ResearchState)
research_builder.add_node("research", research_and_summarize)
research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)

research_graph = research_builder.compile()

# ── Convert the graph to a tool ───────────────────────────────────────────────

class ResearchInput(BaseModel):
    topic: str = Field(description="The topic to research and summarize.")


# as_tool() is beta — API may change in future versions
research_tool = research_graph.as_tool(
    args_schema=ResearchInput,
    name="research_topic",
    description=(
        "Research a topic thoroughly and return a concise 2-paragraph summary. "
        "Use this when you need background information or fact-checking on any subject."
    ),
)

# ── Parent orchestrator that calls research_tool ──────────────────────────────

class OrchestratorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tools = [research_tool]
orchestrator_model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools(tools)
tool_node = ToolNode(tools)


def orchestrator_agent(state: OrchestratorState) -> dict:
    response = orchestrator_model.invoke(state["messages"])
    return {"messages": [response]}


orch_builder = StateGraph(OrchestratorState)
orch_builder.add_node("agent", orchestrator_agent)
orch_builder.add_node("tools", tool_node)
orch_builder.add_edge(START, "agent")
orch_builder.add_conditional_edges("agent", tools_condition)
orch_builder.add_edge("tools", "agent")

orchestrator = orch_builder.compile(checkpointer=InMemorySaver())

# ── Run ───────────────────────────────────────────────────────────────────────

config = {"configurable": {"thread_id": "as-tool-demo-1"}}
result = orchestrator.invoke(
    {"messages": [HumanMessage("Tell me about the history of LangGraph.")]},
    config,
)
print(result["messages"][-1].content)
```

Key points about `as_tool()`:

- The graph's **input schema** is inferred from its state `TypedDict` if you don't pass `args_schema`. Pass an explicit Pydantic model to control exactly which fields the LLM sees.
- The tool returns the **final state** as a dict. Wrap the result if you want a string output for the `ToolMessage`.
- `as_tool()` is marked **beta** — the API surface may change in future LangGraph versions.
- For nested graphs that need human-in-the-loop, use a subgraph node (Example 4) instead, since tool calls don't support mid-execution interrupts.

---

## Example 6: `Command` with `Command.PARENT` — child-to-parent escalation

A subgraph node can send updates **up to the parent graph** by returning `Command(graph=Command.PARENT, ...)`. This enables a specialist subgraph to escalate to the parent (e.g. signal that it needs more resources, or surface a final result into the parent's state).

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


# ── Shared state keys (must exist in BOTH parent and child schemas) ───────────

class ChildState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    escalate: bool   # child sets this; parent reads it


class ParentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    escalate: bool   # parent receives this update from the child
    final_answer: str


# ── Child graph ───────────────────────────────────────────────────────────────

def child_worker(state: ChildState) -> Command[str]:
    """A specialist that can either resolve a task or escalate to the parent."""
    last_msg = state["messages"][-1].content if state["messages"] else ""

    if "complex" in last_msg.lower():
        # Escalate: update parent state and navigate parent to "escalation_handler"
        return Command(
            graph=Command.PARENT,           # target the closest parent graph
            update={"escalate": True},      # write to parent's state key
            goto="escalation_handler",      # navigate parent to this node
        )

    # Happy path: return a result normally
    return {"messages": [AIMessage("Task completed by specialist.")]}


child_builder = StateGraph(ChildState)
child_builder.add_node("worker", child_worker, destinations={"escalation_handler"})
child_builder.add_edge(START, "worker")
child_builder.add_edge("worker", END)

child_graph = child_builder.compile()


# ── Parent graph ──────────────────────────────────────────────────────────────

def supervisor(state: ParentState) -> dict:
    return {"final_answer": "Delegating to specialist..."}


def escalation_handler(state: ParentState) -> dict:
    """Handles tasks the specialist couldn't complete."""
    return {
        "final_answer": "Escalated task handled at supervisor level.",
        "escalate": False,
    }


parent_builder = StateGraph(ParentState)
parent_builder.add_node("supervisor", supervisor)
parent_builder.add_node("specialist", child_graph)      # child graph as a node
parent_builder.add_node("escalation_handler", escalation_handler)

parent_builder.add_edge(START, "supervisor")
parent_builder.add_edge("supervisor", "specialist")
parent_builder.add_edge("specialist", END)
parent_builder.add_edge("escalation_handler", END)

parent_graph = parent_builder.compile(checkpointer=InMemorySaver())

# ── Test ──────────────────────────────────────────────────────────────────────

config = {"configurable": {"thread_id": "parent-cmd-demo"}}

# Simple task — no escalation
r1 = parent_graph.invoke(
    {"messages": [HumanMessage("simple task")], "escalate": False, "final_answer": ""},
    config,
)
print(r1["final_answer"])   # "Task completed by specialist."

# Complex task — child escalates to parent
config2 = {"configurable": {"thread_id": "parent-cmd-demo-2"}}
r2 = parent_graph.invoke(
    {"messages": [HumanMessage("complex task that needs escalation")],
     "escalate": False, "final_answer": ""},
    config2,
)
print(r2["final_answer"])   # "Escalated task handled at supervisor level."
```

Rules for `Command.PARENT`:

- The state keys in `Command(update=...)` must exist in the **parent's** state schema. Writing to a key the parent doesn't have raises `InvalidUpdateError`.
- `Command.PARENT` navigates the **closest** parent — if graphs are nested 3 levels deep, it targets level 2, not level 1.
- Use `destinations={"node_name"}` on the child node's `add_node` call to make the Mermaid diagram show the edge correctly.

---

## Gotchas

- **`Command(goto=END)` vs `add_edge(node, END)`:** A node that returns `Command(goto=...)` bypasses all explicit edges from that node. Pick one style per node — never mix `add_edge` and `Command` routing for the same node.
- **Reducers are required for parallel writes.** When multiple workers write to the same state key in the same super-step, the key must have a reducer (e.g., `Annotated[list, operator.add]`). Without one, LangGraph raises `InvalidUpdateError`.
- **`Send` arg is the entire snapshot for the target node.** Workers see only what you passed in `Send("worker", {...})`, not the full `WorkflowState`. Include every key the worker reads.
- **`checkpointer=True` is only for subgraphs.** Passing `checkpointer=True` to a root graph raises `RuntimeError`. Only compiled subgraphs can inherit a parent's checkpointer.
- **`destinations=` is diagram-only.** It does not change execution — its sole purpose is to make the Mermaid visualization show the correct edges for nodes that use `Command`.
- **`as_tool()` is beta.** The API is functional but may change in future LangGraph releases. Pin your `langgraph` version if relying on it in production.

---
title: "Chapter 10 — Production & Troubleshooting"
description: "Taking a LangGraph app to production: choosing a checkpointer and store, durability modes, loop limits and node policies, testing with scripted models, deploying with the langgraph CLI, and fixes for common errors."
framework: "langgraph"
language: "python"
sidebar:
  label: "10 · Production & troubleshooting"
  order: 10
---

**What you'll learn:** what changes when a graph leaves your laptop: where state is stored, how often it is written, how runaway loops and flaky calls are contained, how to test without calling a model, and how to deploy with the `langgraph` CLI.

Checked against `langgraph` %%version=langgraph%%, `langchain` %%version=langchain%% and `langgraph-cli` %%version=langgraph-cli%%. **Time:** ~30 minutes.

> The [production guide](/langgraph-guide/python/langgraph_production_guide/) is the deployment reference (`langgraph.json`, LangSmith Deployment, self-hosting). This chapter is the tutorial path to it.

## Checklist

- A durable checkpointer (Postgres) in place of `InMemorySaver`, and a `thread_id` per conversation.
- A `PostgresStore` if you keep long-term memory, with a TTL if the data should expire.
- `recursion_limit` set on every agent run.
- Retry and timeout policies on the nodes that call models and external services.
- Tests that run the graph with a scripted model, including interrupt and resume.
- LangSmith tracing turned on (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`); see [Observability](/langgraph-guide/python/langgraph_observability_python/).

## Choosing a checkpointer and store

| Saver | Package | Use it for |
|---|---|---|
| `InMemorySaver` | `langgraph` | Tests and notebooks. State is lost when the process exits. |
| `SqliteSaver` / `AsyncSqliteSaver` | `langgraph-checkpoint-sqlite` | A single process on one machine; local tools |
| `PostgresSaver` / `AsyncPostgresSaver` | `langgraph-checkpoint-postgres` | Production, several workers sharing threads |
| Provided by the server | — | LangSmith Deployment and `langgraph dev`/`up` supply a checkpointer and store; do not pass your own to `compile()` there |

For long-term memory, `InMemoryStore` is for development (it has no TTL support); `PostgresStore`/`AsyncPostgresStore` (in `langgraph-checkpoint-postgres`) is for production and supports `ttl=TTLConfig(...)`.

`from_conn_string(...)` returns a context manager, and `setup()` creates the tables (run it once, for example at deploy time):

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://user:pass@localhost:5432/app?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as saver, \
     PostgresStore.from_conn_string(DB_URI, ttl={"default_ttl": 60 * 24 * 30}) as store:   # minutes
    saver.setup()
    store.setup()
    graph = builder.compile(checkpointer=saver, store=store)
    graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "user-42"}})
```

In a web server, open an `AsyncPostgresSaver` once at start-up and share it; the [production guide](/langgraph-guide/python/langgraph_production_guide/#self-hosting-in-your-own-service) shows this with a FastAPI lifespan. Feature details, encryption (`EncryptedSerializer`) and the other savers are in the [checkpointers reference](/langgraph-guide/python/reference-checkpointers/).

## Durability: how often state is written

Pass `durability=` to `invoke`, `stream`, `ainvoke` or `astream`. It is not a `compile()` argument, and it has no effect without a checkpointer.

| Mode | When the checkpoint is written | Trade-off |
|---|---|---|
| `"async"` (default) | In the background while the next step runs | Fast; a crash can lose the latest step |
| `"sync"` | Before the next step starts | Slowest; nothing is lost on a crash |
| `"exit"` | Only when the run finishes, fails or is interrupted | Fewest writes; no mid-run recovery or step-by-step history |

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    amount: int
    approved: bool


def approve(state: State) -> dict:
    return {"approved": interrupt(f"Approve {state['amount']}?") == "yes"}


graph = (
    StateGraph(State)
    .add_node("approve", approve)
    .add_edge(START, "approve")
    .compile(checkpointer=InMemorySaver())
)
cfg = {"configurable": {"thread_id": "order-1"}}

graph.invoke({"amount": 500}, cfg, durability="exit")            # pauses at interrupt()
print(graph.invoke(Command(resume="yes"), cfg, durability="exit"))   # {'amount': 500, 'approved': True}
```

Interrupts work under all three modes, because the checkpoint is saved when the run pauses. Use `"sync"` when a lost step would be expensive to redo (a payment, an e-mail), `"exit"` for short batch runs, and the default otherwise.

## Containing loops and failures

**Recursion limit.** Each super-step (a model call, a tools step, a node) counts towards `recursion_limit`. The default is **10007**, so a model that keeps calling tools will run for a long time before `GraphRecursionError` stops it. Set a limit that fits the task:

```python
agent.invoke({"messages": [("user", "Plan my trip")]}, {"recursion_limit": 25, "configurable": {"thread_id": "t1"}})
```

To finish cleanly instead of raising, read the `RemainingSteps` managed value in the node and wrap up when it is low:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps     # filled in by LangGraph; read-only


def work(state: State) -> dict:
    if state["remaining_steps"] <= 2:
        return {"messages": [("ai", "Stopping here; this is what I have so far.")]}
    return {"messages": [("ai", "still working")]}


def route(state: State) -> str:
    return "__end__" if state["messages"][-1].content.startswith("Stopping") else "work"


graph = (
    StateGraph(State)
    .add_node("work", work)
    .add_edge(START, "work")
    .add_conditional_edges("work", route)
    .compile()
)
out = graph.invoke({"messages": [("user", "go")]}, {"recursion_limit": 10})
print(len(out["messages"]), out["messages"][-1].content)   # 9 Stopping here; ...
```

`create_agent` agents can use `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` instead ([Chapter 8](/langgraph-guide/python/chapter-08-middleware-hooks/#built-in-middleware)).

**Node policies.** Put a `RetryPolicy` and a timeout on the nodes that call models or services, and an error handler where a failure should be recorded rather than end the run. Timeouts need `async def` nodes. The options, defaults and a complete example are in one place: [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies). `Send(..., timeout=...)` sets a timeout for one fan-out task; see the [Command and Send reference](/langgraph-guide/python/reference-command-and-send/).

**Shutdown.** To stop at a step boundary on `SIGTERM` (so the next start resumes from a checkpoint), pass a `RunControl` and call `request_drain()`; see [graceful shutdown](/langgraph-guide/python/reference-graceful-shutdown/).

## Testing with scripted models

Tests should not call a real model. `GenericFakeChatModel` from `langchain_core` replays the messages you give it; a two-line subclass adds the `bind_tools` method that `create_agent` needs. With an `InMemorySaver` per test you can also check interrupts and resumes.

```python
# test_agent.py — run with: pytest test_agent.py
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted messages; accepts bind_tools so create_agent can use it."""

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


def scripted(*messages: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


def test_agent_calls_tool_then_answers():
    model = scripted(
        AIMessage("", tool_calls=[{"name": "refund", "args": {"order_id": "42"}, "id": "c1"}]),
        AIMessage("Order 42 has been refunded."),
    )
    agent = create_agent(model, tools=[refund])
    result = agent.invoke({"messages": [{"role": "user", "content": "Refund order 42"}]})
    tool_msg = result["messages"][2]
    assert tool_msg.content == "refunded 42"
    assert result["messages"][-1].content == "Order 42 has been refunded."


def test_refund_needs_approval():
    model = scripted(
        AIMessage("", tool_calls=[{"name": "refund", "args": {"order_id": "42"}, "id": "c1"}]),
        AIMessage("Refund cancelled."),
    )
    agent = create_agent(
        model, tools=[refund], checkpointer=InMemorySaver(),
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"refund": True})],
    )
    cfg = {"configurable": {"thread_id": "test-approval"}}
    paused = agent.invoke({"messages": [{"role": "user", "content": "Refund order 42"}]}, cfg)
    assert paused["__interrupt__"][0].value["action_requests"][0]["name"] == "refund"

    done = agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "No"}]}), cfg)
    assert "refunded" not in [m.content for m in done["messages"]]


class State(TypedDict, total=False):
    amount: int
    approved: bool


def build_graph():
    def check(state: State) -> dict:
        return {"approved": interrupt({"amount": state["amount"]}) == "yes"}

    return StateGraph(State).add_node("check", check).add_edge(START, "check").add_edge("check", END)


def test_graph_interrupt_and_resume():
    graph = build_graph().compile(checkpointer=InMemorySaver())   # fresh checkpointer per test
    cfg = {"configurable": {"thread_id": "t1"}}
    first = graph.invoke({"amount": 500}, cfg)
    assert first["__interrupt__"][0].value == {"amount": 500}
    assert graph.get_state(cfg).next == ("check",)
    assert graph.invoke(Command(resume="yes"), cfg)["approved"] is True


def test_single_node():
    graph = build_graph().compile()
    # graph.nodes gives direct access to a node; no checkpointer is involved
    assert "check" in graph.nodes
```

`GenericFakeChatModel` streams by splitting the message content, so it suits tests that call `invoke`. Under `stream(..., stream_mode="messages")` a scripted message with empty content and only `tool_calls` fails with `ValueError: No generations found in stream`; with non-empty content it runs, but the streamed chunks carry no tool calls, so the agent stops after the first turn. Test streaming of tool-calling agents with a fake that implements `_stream` and yields `AIMessageChunk`s with `tool_call_chunks`.

Other useful techniques (from the [LangGraph testing docs](https://docs.langchain.com/oss/python/langgraph/test)):

- `graph.nodes["name"].invoke(state)` runs one node on its own.
- `graph.update_state(cfg, values, as_node="node1")` followed by `graph.invoke(None, cfg, interrupt_after="node3")` runs only part of a graph.
- For model quality rather than code paths, use LangSmith datasets and evaluators (`langsmith.evaluate`) instead of unit tests.

## Deploying with the `langgraph` CLI

The CLI (`pip install -U "langgraph-cli[inmem]"`) runs your graphs in the LangGraph Agent Server, which adds threads, background runs, streaming endpoints, a task queue, and built-in Postgres persistence. You describe the app in `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.12"
}
```

`graph` is a compiled graph (or a function that returns one), compiled **without** a checkpointer or store; the server provides both.

| Command | What it does |
|---|---|
| `langgraph new --template new-langgraph-project-python my-app` | Start a project from a template (others: `agent-python`, `deep-agent-python` and the `-js` variants) |
| `langgraph dev` | Run the server in-process with hot reload (no Docker) on port 2024 and open LangSmith Studio |
| `langgraph validate` | Check `langgraph.json` |
| `langgraph build -t my-agent` | Build a Docker image of the server and your code |
| `langgraph up` | Run the image locally with Postgres and Redis (needs Docker and a LangSmith licence or API key) |
| `langgraph dockerfile Dockerfile` | Write the Dockerfile instead of building |
| `langgraph deploy` (beta) | Build and deploy to LangSmith Deployment; `deploy list`, `deploy logs`, `deploy revisions list` and `deploy delete` manage it |

Once deployed, call the graph with the `langgraph-sdk` client, or with [`RemoteGraph`](/langgraph-guide/python/reference-remote-graph/), which has the same `invoke`/`stream`/`get_state` interface as a local graph. The [production guide](/langgraph-guide/python/langgraph_production_guide/) covers the other `langgraph.json` keys (TTL, auth, HTTP), self-hosting, and running a graph inside your own service.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Cannot use Command(resume=...) without checkpointer` | The graph was compiled without a checkpointer | Compile with a checkpointer |
| `Command(resume=...)` starts a fresh run or finds nothing to resume | A different `thread_id` from the paused run | Reuse the same config |
| `RuntimeError: When there are multiple pending interrupts, you must specify the interrupt id when resuming.` | Parallel nodes are paused at once | `Command(resume={interrupt.id: value, ...})` ([Chapter 7](/langgraph-guide/python/chapter-07-human-in-the-loop/#example-3-resume-by-interrupt-id)) |
| `ValueError: Checkpointer requires ... thread_id` | Compiled with a checkpointer, invoked without a thread | Add `{"configurable": {"thread_id": ...}}` |
| `InvalidUpdateError: At key 'x': Can receive only one value per step` | Two parallel nodes wrote a key that has no reducer | `Annotated[list, operator.add]` (or another reducer) |
| A list key is replaced, not appended | No reducer on the key | Add the reducer to the type: `messages: Annotated[list, add_messages]` |
| The model never calls tools | The model in your node is not bound to them | `model.bind_tools(tools)`, or use `create_agent` |
| `GraphRecursionError` | A loop with no exit, or a limit too low for the task | Fix the routing; set `recursion_limit` |
| `ValueError: Node timeouts are only supported for async nodes` | At `compile()`: a timeout (possibly from `set_node_defaults`) on a sync node or handler. At run time: `Send(..., timeout=...)` to a sync node | Make it `async def` |
| `LangGraphDeprecatedSinceV10` warning for `create_react_agent` | Deprecated factory | `langchain.agents.create_agent` ([Chapter 2](/langgraph-guide/python/chapter-02-simple-agents/)) |

## Next steps

- [Recipes](/langgraph-guide/python/langgraph_recipes/) for complete applications.
- [Production guide](/langgraph-guide/python/langgraph_production_guide/) for deployment.
- [Performance](/langgraph-guide/python/langgraph_performance_optimization/) and [Observability](/langgraph-guide/python/langgraph_observability_python/).

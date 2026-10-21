---
title: "LangGraph (Python): API map and quick reference"
description: "One-page map of the LangGraph Python API: which package each symbol comes from, the defaults that matter, deprecations, and where each topic is covered in the tutorial and reference pages."
framework: "langgraph"
language: "python"
sidebar:
  label: "API map & quick reference"
  order: 20
---

This page is a map, not a second tutorial. It lists the packages and symbols you use with LangGraph, the defaults worth knowing, and the page where each topic is covered in full. Checked against `langgraph` %%version=langgraph%%, `langgraph-prebuilt` %%version=langgraph-prebuilt%%, `langgraph-checkpoint` %%version=langgraph-checkpoint%% and `langchain` %%version=langchain%% (%%versions-checked%%).

- Learning LangGraph from scratch: start at [Chapter 1](/langgraph-guide/python/chapter-01-setup-and-core-concepts/).
- Looking up a signature: use the reference pages linked in the tables below.

## Packages

| Package | What it gives you | Install |
|---|---|---|
| `langgraph` | `StateGraph`, `Command`, `Send`, `interrupt`, the Functional API (`entrypoint`, `task`), policies, `InMemorySaver`, `InMemoryStore` | `pip install langgraph` |
| `langgraph-prebuilt` | `ToolNode`, `tools_condition`, `InjectedState`, `InjectedStore`, `ToolRuntime`. Installed with `langgraph`. | (dependency) |
| `langchain` | `create_agent` and agent middleware (`AgentMiddleware`, `HumanInTheLoopMiddleware`, `SummarizationMiddleware`, …). This is the recommended way to build a tool-calling agent; it compiles to a LangGraph graph. | `pip install langchain` |
| `langchain-anthropic`, `langchain-openai`, … | Chat model integrations | `pip install langchain-anthropic` |
| `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres` | Durable checkpointers (and `PostgresStore`) | `pip install langgraph-checkpoint-postgres` |
| `langgraph-sdk` | Client for a LangGraph server (`get_client`), used by `RemoteGraph` | `pip install langgraph-sdk` |
| `langgraph-cli` | `langgraph dev`, `build`, `up`, `deploy` for LangGraph servers and LangSmith Deployment | `pip install "langgraph-cli[inmem]"` |

LangGraph needs Python 3.10 or later.

## Two ways in

**A tool-calling agent.** Use `create_agent` from `langchain`. You get the model-and-tools loop, structured output, and middleware hooks without writing the graph yourself. The result is a compiled LangGraph graph, so checkpointers, streaming, interrupts and LangSmith Deployment work as they do for any graph.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"It is sunny in {city}."


agent = create_agent(
    "anthropic:%%model=anthropic%%",
    tools=[get_weather],
    system_prompt="You are a concise assistant.",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "user-1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]}, config)
print(result["messages"][-1].content)
```

`langgraph.prebuilt.create_react_agent` still works but is deprecated (`LangGraphDeprecatedSinceV10`, removal planned for 2.0). Its `pre_model_hook`/`post_model_hook` arguments map to `before_model`/`after_model` middleware. See [Chapter 8](/langgraph-guide/python/chapter-08-middleware-hooks/) for middleware and the migration table.

**Your own graph.** Use `StateGraph` when you want to control the steps: routing, parallel branches, multi-agent handoffs, approval gates. [Chapter 1](/langgraph-guide/python/chapter-01-setup-and-core-concepts/) builds one from scratch; the [StateGraph reference](/langgraph-guide/python/reference-state-graph/) has every builder method.

## Where each topic lives

| Topic | Tutorial | Reference |
|---|---|---|
| State, reducers, nodes, edges, `compile()` | [Ch. 1](/langgraph-guide/python/chapter-01-setup-and-core-concepts/) | [StateGraph](/langgraph-guide/python/reference-state-graph/), [Channels](/langgraph-guide/python/reference-channels/) |
| `create_agent`, routing, loops | [Ch. 2](/langgraph-guide/python/chapter-02-simple-agents/) | [Prebuilt nodes](/langgraph-guide/python/reference-prebuilt-nodes/) |
| Supervisor, parallel workers, handoffs, subgraphs | [Ch. 3](/langgraph-guide/python/chapter-03-multi-agent/) | [Command and Send](/langgraph-guide/python/reference-command-and-send/) |
| Tools, `ToolNode`, tool errors, injected state/store | [Ch. 4](/langgraph-guide/python/chapter-04-tools/) | [Prebuilt nodes](/langgraph-guide/python/reference-prebuilt-nodes/), [Runtime and managed values](/langgraph-guide/python/reference-runtime-and-managed-values/) |
| Checkpointers, threads, time travel, Store | [Ch. 5](/langgraph-guide/python/chapter-05-memory/) | [Checkpointers](/langgraph-guide/python/reference-checkpointers/), [Store](/langgraph-guide/python/reference-store/) |
| Streaming, v2 typed output, debugging | [Ch. 6](/langgraph-guide/python/chapter-06-streaming-and-debugging/) | [Streaming modes](/langgraph-guide/python/reference-streaming-modes/) |
| `interrupt()`, `Command(resume=...)`, approvals | [Ch. 7](/langgraph-guide/python/chapter-07-human-in-the-loop/) | [Command and Send](/langgraph-guide/python/reference-command-and-send/) |
| Agent middleware, error handlers | [Ch. 8](/langgraph-guide/python/chapter-08-middleware-hooks/) | — |
| Retry, timeout and cache policies, `set_node_defaults` | [Ch. 10](/langgraph-guide/python/chapter-10-production/) (when to use them) | [StateGraph: node policies](/langgraph-guide/python/reference-state-graph/#node-policies) (canonical) |
| Map-reduce, reflection, Functional API | [Ch. 9](/langgraph-guide/python/chapter-09-advanced-patterns/) | [Functional API](/langgraph-guide/python/reference-functional-api/) |
| Deployment, durability, testing | [Ch. 10](/langgraph-guide/python/chapter-10-production/) | [Production guide](/langgraph-guide/python/langgraph_production_guide/), [RemoteGraph](/langgraph-guide/python/reference-remote-graph/), [Graceful shutdown](/langgraph-guide/python/reference-graceful-shutdown/) |
| Tracing and redaction | [Observability](/langgraph-guide/python/langgraph_observability_python/) | [TracePolicy](/langgraph-guide/python/reference-trace-policy/) |
| Node caching backends | — | [Cache backends](/langgraph-guide/python/reference-cache-backends/) |

## Symbols by module

| Import | Symbols |
|---|---|
| `langgraph.graph` | `StateGraph`, `START`, `END`, `MessagesState`, `add_messages` |
| `langgraph.graph.message` | `add_messages`, `REMOVE_ALL_MESSAGES` |
| `langgraph.types` | `Command`, `Send`, `interrupt`, `Interrupt`, `RetryPolicy`, `TimeoutPolicy`, `CachePolicy`, `Durability`, `StreamMode`, `StreamPart`, `GraphOutput`, `StateSnapshot`, `Overwrite`, `StreamWriter` |
| `langgraph.errors` | `GraphRecursionError`, `NodeError` (error-handler argument), `NodeTimeoutError`, `GraphInterrupt`, `InvalidUpdateError` |
| `langgraph.func` | `entrypoint`, `task` |
| `langgraph.runtime` | `Runtime`, `get_runtime` |
| `langgraph.managed` | `IsLastStep`, `RemainingSteps` |
| `langgraph.config` | `get_config`, `get_store`, `get_stream_writer` |
| `langgraph.checkpoint.memory` | `InMemorySaver` (`MemorySaver` is an older alias of the same class) |
| `langgraph.checkpoint.sqlite`, `langgraph.checkpoint.postgres` | `SqliteSaver`, `AsyncSqliteSaver`, `PostgresSaver`, `AsyncPostgresSaver` |
| `langgraph.store.memory`, `langgraph.store.postgres` | `InMemoryStore`, `PostgresStore`, `AsyncPostgresStore` |
| `langgraph.cache.memory` | `InMemoryCache` |
| `langgraph.prebuilt` | `ToolNode`, `tools_condition`, `InjectedState`, `InjectedStore`, `ToolRuntime` |
| `langgraph.pregel.remote` | `RemoteGraph` |
| `langchain.agents` | `create_agent`, `AgentState` |
| `langchain.agents.middleware` | `AgentMiddleware`, `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`, `dynamic_prompt`, and the built-in middleware classes |

## Defaults and limits worth knowing

These are the values in the installed source. Several older pages and blog posts give different numbers.

| Setting | Value |
|---|---|
| Default `recursion_limit` | **10007** super-steps (override with the `LANGGRAPH_DEFAULT_RECURSION_LIMIT` environment variable, or per run with `config={"recursion_limit": 25}`). Older releases used 25. Set it explicitly for agent loops: the default is not a useful guard against a runaway loop. |
| Default `durability` | `"async"`. Passed to `invoke`/`stream`/`ainvoke`/`astream`, not to `compile()`. Options: `"sync"`, `"async"`, `"exit"`. It has no effect without a checkpointer. |
| Stream modes | Seven typed values in `StreamMode`: `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages`, `custom`. `"tools"` is also accepted at runtime but is not in the `Literal`. |
| `invoke` default `stream_mode` | `"values"` (returns the final state). |
| Node timeouts | `add_node(..., timeout=...)` and `TimeoutPolicy` work on **async** nodes only. A sync node with a timeout is accepted by `add_node` and rejected by `compile()` with `ValueError`. |
| `ToolNode` error handling | Default catches only argument-validation errors (`ToolInvocationError`) and returns them to the model; other exceptions propagate. Pass `handle_tool_errors=True` to catch everything. |
| `interrupt()` without a checkpointer | `invoke` returns with an `__interrupt__` key, but the run cannot be resumed. Compile with a checkpointer and pass a `thread_id`. |
| `InMemoryStore` TTL | Not supported (`supports_ttl` is `False`). Use `PostgresStore` with `TTLConfig`. |
| `checkpointer=True` on `compile()` | Only valid for a subgraph (inherit the parent's checkpointer). |

## Compiled-graph utilities

A compiled graph has a few helpers that are easy to miss.

```python
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


class State(TypedDict):
    text: str
    word_count: int


inner = (
    StateGraph(State)
    .add_node("count", lambda s: {"word_count": len(s["text"].split())},
              cache_policy=CachePolicy(ttl=60))
    .add_edge(START, "count")
    .compile()
)
graph = (
    StateGraph(State)
    .add_node("inner", inner)
    .add_edge(START, "inner")
    .add_edge("inner", END)
    .compile(cache=InMemoryCache())
)

# Diagrams: xray=True expands subgraphs inline
print(sorted(graph.get_graph(xray=True).nodes))   # ['__end__', '__start__', 'inner:count']
print(graph.get_graph().draw_mermaid()[:40])

# Subgraphs and JSON schemas
print([name for name, _ in graph.get_subgraphs()])  # ['inner']
print(sorted(graph.get_input_jsonschema()["properties"]))  # ['text', 'word_count']

# Invalidate node caches
graph.clear_cache()


# Expose a graph as a LangChain tool (beta)
class CountInput(BaseModel):
    text: str = Field(description="Text to count")


count_tool = inner.as_tool(args_schema=CountInput, name="word_count",
                           description="Count words in a text")
print(count_tool.invoke({"text": "LangGraph builds agents"})["word_count"])  # 3
```

## Deprecations

| Deprecated | Use instead |
|---|---|
| `langgraph.prebuilt.create_react_agent` (and `pre_model_hook`/`post_model_hook`) | `langchain.agents.create_agent` with middleware |
| `AgentState`, `AgentStatePydantic` in `langgraph.prebuilt.chat_agent_executor` | `langchain.agents.AgentState` |
| `HumanInterrupt`, `HumanInterruptConfig`, `ActionRequest` (`langgraph.prebuilt.interrupt`) | `HumanInTheLoopMiddleware` and its `HITLRequest`/`HITLResponse` types in `langchain.agents.middleware.human_in_the_loop` |
| `ValidationNode` | `ToolNode` with a Pydantic `args_schema` on each tool, or your own validation node |
| `MessageGraph` | `StateGraph(MessagesState)` |
| `add_node(..., retry=...)`, `add_node(..., cache=...)` | `retry_policy=`, `cache_policy=` |
| `@entrypoint(config_schema=...)`, `get_config_jsonschema()` | `context_schema=`, `get_context_jsonschema()` |
| `checkpoint_during=` on `invoke`/`stream` | `durability=` |
| `ShallowPostgresSaver` | `PostgresSaver` with `durability="exit"` |

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Checkpointer requires ... thread_id` | Graph compiled with a checkpointer, invoked without a thread | Pass `{"configurable": {"thread_id": "..."}}` |
| State does not carry over between calls | No checkpointer, or a different `thread_id` each time | Compile with a checkpointer and reuse the thread ID |
| A list field is overwritten instead of appended | No reducer on the field | `Annotated[list, operator.add]` or `add_messages` |
| `InvalidUpdateError: Expected dict, got [Send(...)]` | A function returning `Send` objects was registered as a node | Return `Send` from a routing function passed to `add_conditional_edges` |
| `ValueError: Node timeouts are only supported for async nodes` | `timeout=` (or a `set_node_defaults` timeout) reaches a sync node, including a sync error handler | Make the node `async def` |
| Error handler raises `TypeError: missing 1 required positional argument` | Handler parameter not annotated as `NodeError` | `def handler(state, error: NodeError)` |
| `GraphRecursionError` after many steps | A loop with no exit condition | Fix the routing; set `recursion_limit` |

## Resources

- LangGraph docs: [docs.langchain.com/oss/python/langgraph](https://docs.langchain.com/oss/python/langgraph/overview)
- Source: [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- LangSmith Deployment: [docs.langchain.com/langsmith/deployments](https://docs.langchain.com/langsmith/deployments)

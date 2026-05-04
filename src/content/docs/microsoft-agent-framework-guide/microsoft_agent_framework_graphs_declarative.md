---
title: "Microsoft Agent Framework — Workflows & Declarative Agents"
description: "Real WorkflowBuilder + Edge model in agent-framework-core, and YAML declarative agents/workflows via agent-framework-declarative."
framework: microsoft-agent-framework
---

# Workflows & Declarative Agents

> **Errata (April 2026).** An earlier draft of this page described an `agent_framework.graphs` module with classes like `AgentGraph`, `GraphNode`, `GraphEdge`, `ConditionalEdge`, `CheckpointConfig`, `HITLConfig`, `TimeTravelConfig`, `ErrorPolicy`, and `ObservabilityConfig`. **None of those classes exist** in the `agent-framework` package. This page was rewritten after direct introspection of `agent-framework-core==1.1.0` and `agent-framework-declarative==1.0.0b260421`.

Two real capabilities ship under this umbrella:

1. **Imperative workflows** — `WorkflowBuilder` in `agent-framework-core`. Composable graphs with fan-in/fan-out/switch-case routing, checkpointing, and `WorkflowAgent` wrapping.
2. **Declarative agents/workflows** — `agent-framework-declarative` package with `AgentFactory` and `WorkflowFactory` that load agents and workflows from YAML.

## Imperative workflows — the real API

### Install

```bash
pip install agent-framework
```

The workflow types are exported from the top-level `agent_framework` package.

### Primitives (verified against `agent-framework-core==1.2.2`)

| Class / function | Purpose |
|---|---|
| `Workflow` | Compiled, runnable workflow. |
| `WorkflowBuilder(start_executor=..., name=..., description=..., max_iterations=100, checkpoint_storage=None, output_executors=None)` | Fluent builder that returns a `Workflow` from `.build()`. |
| `Executor(id, ...)` | Base class for workflow nodes. Subclass or use `FunctionExecutor` to wrap a plain function. |
| `FunctionExecutor(func, id=None, *, input=None, output=None, workflow_output=None)` | Wraps a sync or async Python callable. |
| `WorkflowAgent(workflow, *, id, name, description, context_providers, ...)` | Exposes a compiled `Workflow` as an `Agent` — so workflows can be used interchangeably with single agents. |
| `Edge`, `EdgeCondition`, `FanInEdgeGroup`, `FanOutEdgeGroup`, `SingleEdgeGroup`, `SwitchCaseEdgeGroup`, `Case`, `Default` | Edge types — typically you don't instantiate these directly; `WorkflowBuilder` does it for you. |
| `CheckpointStorage` (protocol), `InMemoryCheckpointStorage`, `FileCheckpointStorage(storage_path, *, allowed_checkpoint_types=None)` | Checkpointing. Pass via `checkpoint_storage=` on the builder. |
| `WorkflowContext`, `WorkflowEvent`, `WorkflowEventSource`, `WorkflowEventType`, `WorkflowRunResult`, `WorkflowRunState`, `WorkflowMessage` | Runtime types for events, results, and messages. |
| `WorkflowViz` | Visualisation helper. |
| `validate_workflow_graph` | Top-level function that checks edge consistency before `.build()` if you want early validation. |

### Verified signature

```python
WorkflowBuilder(
    max_iterations: int = 100,
    name: str | None = None,
    description: str | None = None,
    *,
    start_executor: Executor | SupportsAgentRun,
    checkpoint_storage: CheckpointStorage | None = None,
    output_executors: list[Executor | SupportsAgentRun] | None = None,
)
```

Builder methods (all return `Self` for chaining):

- `.add_edge(source, target, condition=None)` — fixed or conditional edge.
- `.add_chain(executors)` — shorthand for a linear `A → B → C`.
- `.add_fan_out_edges(source, targets)` — one-to-many fan-out.
- `.add_fan_in_edges(sources, target)` — many-to-one fan-in (target waits for all sources).
- `.add_multi_selection_edge_group(source, targets, selection_func)` — dynamic fan-out where a selector function returns the subset of target IDs to run.
- `.add_switch_case_edge_group(source, cases)` — switch/case routing on the payload.
- `.build() → Workflow`.

### Minimum viable workflow

```python
from agent_framework import WorkflowBuilder, FunctionExecutor, InMemoryCheckpointStorage

def research(query: str) -> dict:
    return {"findings": f"Results for: {query}"}

def summarize(findings: dict) -> str:
    return f"Summary: {findings['findings']}"

research_node = FunctionExecutor(research, id="research")
summarize_node = FunctionExecutor(summarize, id="summarize")

workflow = (
    WorkflowBuilder(
        start_executor=research_node,
        name="ContentPipeline",
        checkpoint_storage=InMemoryCheckpointStorage(),
    )
    .add_edge(research_node, summarize_node)
    .build()
)
```

### Conditional routing (switch/case)

```python
from agent_framework import WorkflowBuilder, FunctionExecutor, Case, Default

def route(payload: dict) -> str:
    return payload.get("kind", "unknown")

def billing(payload: dict) -> str: return "handled by billing"
def refund(payload: dict) -> str: return "handled by refund"
def fallback(payload: dict) -> str: return "general support"

classify = FunctionExecutor(route, id="classify")
b = FunctionExecutor(billing, id="billing")
r = FunctionExecutor(refund, id="refund")
f = FunctionExecutor(fallback, id="fallback")

workflow = (
    WorkflowBuilder(start_executor=classify)
    .add_switch_case_edge_group(
        classify,
        cases=[
            Case(condition=lambda out: out == "billing", target=b),
            Case(condition=lambda out: out == "refund", target=r),
            Default(target=f),
        ],
    )
    .build()
)
```

### Fan-out / fan-in (parallel workers)

```python
def split(task: dict) -> list[dict]: ...
def work(item: dict) -> dict: ...
def merge(results: list[dict]) -> dict: ...

splitter  = FunctionExecutor(split,  id="split")
worker_a  = FunctionExecutor(work,   id="worker_a")
worker_b  = FunctionExecutor(work,   id="worker_b")
worker_c  = FunctionExecutor(work,   id="worker_c")
collector = FunctionExecutor(merge,  id="collect")

workflow = (
    WorkflowBuilder(start_executor=splitter)
    .add_fan_out_edges(splitter, [worker_a, worker_b, worker_c])
    .add_fan_in_edges([worker_a, worker_b, worker_c], collector)
    .build()
)
```

### Exposing a workflow as an agent

```python
from agent_framework import WorkflowAgent

agent = WorkflowAgent(
    workflow=workflow,
    name="ContentPipelineAgent",
    description="Runs research → summarize",
)

response = await agent.run("Latest AI research")
print(response.text)
```

### Checkpointing and resume

```python
from agent_framework import FileCheckpointStorage

storage = FileCheckpointStorage(storage_path="./checkpoints")

workflow = (
    WorkflowBuilder(start_executor=first, checkpoint_storage=storage)
    .add_edge(first, second)
    .build()
)
```

The `FileCheckpointStorage` API exposes `.save`, `.load`, `.get_latest`, `.list_checkpoint_ids`, `.list_checkpoints`, and `.delete`. There's no separate `HITLConfig` or `TimeTravelConfig` class — navigation through checkpoint history is done via the storage API directly.

## Declarative agents & workflows — the real API

### Install

```bash
pip install agent-framework-declarative --pre
```

### Primitives (verified against `agent-framework-declarative==1.0.0b260429`)

| Class | Purpose |
|---|---|
| `AgentFactory` | Load a single agent from YAML. |
| `WorkflowFactory` | Load a multi-agent workflow from YAML; register agents and tools before creating. |
| `ExternalInputRequest`, `ExternalInputResponse`, `AgentExternalInputRequest`, `AgentExternalInputResponse`, `WorkflowState` | Runtime types for HITL/external-input actions. |
| `DeclarativeLoaderError`, `DeclarativeWorkflowError`, `ProviderLookupError`, `ProviderTypeMapping` | Errors + provider configuration. |

### Loading an agent from YAML

```python
from agent_framework.declarative import AgentFactory

agent = AgentFactory.create_from_yaml_path("agent.yaml")
result = await agent.run("Hello")
```

### Loading a workflow with registered agents

```python
from agent_framework.declarative import WorkflowFactory

factory = WorkflowFactory()
factory.register_agent("ResearcherAgent", researcher_agent)
factory.register_agent("WriterAgent",      writer_agent)
factory.register_tool("get_weather",       get_weather_fn)

workflow = factory.create_workflow_from_yaml_path("pipeline.yaml")
result = await workflow.run({"topic": "AI in healthcare"})
```

### YAML action reference

Declarative workflows are a dialect of Power Fx expressions + a fixed set of `kind:` actions. The canonical reference is the [official docs](https://learn.microsoft.com/agent-framework/workflows/declarative/) — summary of action kinds:

| Category | Actions |
|---|---|
| Variable | `SetVariable`, `SetMultipleVariables`, `AppendValue`, `ResetVariable`, `SetValue` (Python alias for `SetVariable`) |
| Control flow | `If`, `ConditionGroup`, `Foreach`, `RepeatUntil`, `BreakLoop`, `ContinueLoop`, `GotoAction` |
| Output | `SendActivity`, `EmitEvent` |
| Agents & tools | `InvokeAzureAgent`, `InvokeFunctionTool` |
| Human-in-the-loop | `Question`, `Confirmation`, `RequestExternalInput`, `WaitForInput` |
| Workflow control | `EndWorkflow`, `EndConversation`, `CreateConversation` |

Example greeting workflow:

```yaml
kind: AdaptiveDialog
schema: v1

actions:
  - kind: SetVariable
    id: greet
    variable: Local.name
    value: =Workflow.Inputs.name

  - kind: SendActivity
    activity:
      text: =Concat("Hello, ", Local.name, "!")
```

## Workflow visualisation — `WorkflowViz`

`WorkflowViz` generates Mermaid flowcharts and Graphviz diagrams from any compiled `Workflow`. Useful for documentation, debugging, and design reviews.

```python
from agent_framework import WorkflowViz

viz = WorkflowViz(workflow)
```

### Mermaid (no extra dependencies)

```python
mermaid_src = viz.to_mermaid()
print(mermaid_src)
# flowchart TD
#   research["research"]
#   summarize["summarize"]
#   research --> summarize
```

Paste the output directly into GitHub Markdown, Notion, or any Mermaid-compatible renderer:

````markdown
```mermaid
flowchart TD
  research["research"]
  summarize["summarize"]
  research --> summarize
```
````

### Graphviz export (SVG, PNG, PDF, DOT)

Requires the `graphviz` Python package (`pip install graphviz`) and the system Graphviz binary:

```python
# Return the DOT source without saving
dot_src = viz.to_digraph()

# Save to SVG (returns the file path)
path = viz.save_svg("workflow.svg")
print(path)          # ./workflow.svg

# Save to PNG
viz.save_png("workflow.png")

# Save to PDF
viz.save_pdf("workflow.pdf")

# Generic export — format can be 'svg', 'png', 'pdf', or 'dot'
viz.export(format="svg", filename="pipeline.svg")
```

### Including internal executors

By default, framework-internal executor wrappers (e.g. the agents-as-executors injected by `WorkflowAgent`) are hidden. Show them with `include_internal_executors=True`:

```python
mermaid_src = viz.to_mermaid(include_internal_executors=True)
svg_path = viz.save_svg("debug.svg", include_internal_executors=True)
```

### Full visualisation example

```python
import asyncio
from agent_framework import (
    Agent,
    Case,
    Default,
    FileCheckpointStorage,
    FunctionExecutor,
    WorkflowBuilder,
    WorkflowViz,
)
from agent_framework.openai import OpenAIChatClient


def classify(text: str) -> str:
    words = text.lower().split()
    if any(w in words for w in ("urgent", "critical", "asap")):
        return "urgent"
    return "normal"


def handle_urgent(text: str) -> str:
    return f"URGENT escalation: {text}"


def handle_normal(text: str) -> str:
    return f"Queued: {text}"


classify_node = FunctionExecutor(classify, id="classify")
urgent_node = FunctionExecutor(handle_urgent, id="urgent")
normal_node = FunctionExecutor(handle_normal, id="normal")

workflow = (
    WorkflowBuilder(start_executor=classify_node, name="ticket-router")
    .add_switch_case_edge_group(
        classify_node,
        cases=[
            Case(condition=lambda out: out == "urgent", target=urgent_node),
            Default(target=normal_node),
        ],
    )
    .build()
)

# Visualise the compiled workflow
viz = WorkflowViz(workflow)
print(viz.to_mermaid())
viz.save_svg("ticket-router.svg")
```

---

## `add_chain` — linear shorthand

`.add_chain(executors)` connects a sequence `A → B → C → …` in a single call. Equivalent to multiple `.add_edge` calls but more readable for long pipelines:

```python
from agent_framework import FunctionExecutor, WorkflowBuilder

ingest  = FunctionExecutor(lambda raw: raw.strip(),            id="ingest")
enrich  = FunctionExecutor(lambda text: {"text": text},        id="enrich")
score   = FunctionExecutor(lambda d: {**d, "score": 0.9},      id="score")
publish = FunctionExecutor(lambda d: f"published: {d['text']}", id="publish")

workflow = (
    WorkflowBuilder(start_executor=ingest)
    .add_chain([ingest, enrich, score, publish])
    .build()
)
```

`add_chain` calls `.add_edge` between consecutive executors and automatically sets the first one as the start executor if no `start_executor=` was provided. The chain method returns `Self` so it composes with other builder calls:

```python
workflow = (
    WorkflowBuilder(start_executor=ingest)
    .add_chain([ingest, enrich, score])
    .add_edge(score, publish, condition=lambda d: d["score"] > 0.5)
    .build()
)
```

---

## `add_multi_selection_edge_group` — dynamic fan-out

Fan-out to a runtime-selected subset of targets. A `selection_func` receives the outgoing message and the list of all target IDs, and returns which IDs should receive the message this turn:

```python
from agent_framework import FunctionExecutor, WorkflowBuilder


def dispatch(payload: dict) -> dict:
    """Enrich the payload so the selector can inspect it."""
    return payload


def run_translation(payload: dict) -> str:
    return f"translated ({payload.get('lang', 'en')}): {payload.get('text', '')}"


def run_moderation(payload: dict) -> str:
    return f"moderated: {payload.get('text', '')}"


def run_summarisation(payload: dict) -> str:
    return f"summary: {payload.get('text', '')}"


def select_services(payload: dict, available: list[str]) -> list[str]:
    """Pick which services to run based on payload flags."""
    selected = []
    if payload.get("needs_translation"):
        selected.append("translate")
    if payload.get("needs_moderation"):
        selected.append("moderate")
    if payload.get("needs_summary"):
        selected.append("summarise")
    # Fallback — always run moderation if nothing selected
    return selected or ["moderate"]


dispatcher   = FunctionExecutor(dispatch,          id="dispatch")
translator   = FunctionExecutor(run_translation,   id="translate")
moderator    = FunctionExecutor(run_moderation,    id="moderate")
summariser   = FunctionExecutor(run_summarisation, id="summarise")

workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_multi_selection_edge_group(
        dispatcher,
        targets=[translator, moderator, summariser],
        selection_func=select_services,
    )
    .build()
)

import asyncio
result = asyncio.run(workflow.run({
    "text": "Breaking news story",
    "needs_translation": True,
    "needs_summary": True,
}))
print(result.get_outputs())
```

`selection_func(payload, available: list[str]) -> list[str]` — the second argument is a list of target executor IDs from which you return a subset. Return an empty list to route to none of them (the run still completes).

---

## `@executor` decorator — lightweight function executors

The `@executor` decorator converts a standalone module-level function into a `FunctionExecutor` without an intermediate variable. Use it when you prefer a declarative registration style:

```python
from agent_framework import WorkflowBuilder, WorkflowContext, executor


@executor(id="normalise", output=str)
def normalise(text: str) -> str:
    """Strip and lower-case the input."""
    return text.strip().lower()


@executor(id="word-count", input=str, output=dict)
def word_count(text: str) -> dict:
    return {"text": text, "words": len(text.split())}


@executor(id="report", input=dict, workflow_output=str)
def build_report(data: dict) -> str:
    return f"'{data['text']}' has {data['words']} words."


workflow = (
    WorkflowBuilder(start_executor=normalise)
    .add_chain([normalise, word_count, build_report])
    .build()
)
```

`@executor` parameters:

| Parameter | Purpose |
|---|---|
| `id` | Executor ID — used in edge wiring and checkpoint keys. Defaults to the function name. |
| `input` | Expected input type — used for graph-level type validation. |
| `output` | Type of messages sent to downstream executors via `send_message`. |
| `workflow_output` | Type of values emitted as workflow-level output via `yield_output`. |

> **Note:** Use `@executor` for module-level functions only. For class-based executors with per-instance state or `@response_handler` (HITL), subclass `Executor` directly.

---

## `Agent.as_tool()` — wrapping an agent as a callable tool

Any `Agent` (or `BaseAgent` subclass) can be converted into a `FunctionTool` so it can be called by a parent agent or used inside a workflow node:

```python
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Specialised sub-agent
analyst = Agent(
    client=client,
    name="market-analyst",
    description="Performs in-depth market analysis for a given sector.",
    instructions="Return a concise market analysis report.",
)

# Wrap as a FunctionTool
analyst_tool = analyst.as_tool(
    name="analyse_market",                           # tool name exposed to the model
    description="Run a market analysis for a given sector and return a report.",
    arg_name="sector",                               # name of the single string argument
    arg_description="The market sector to analyse (e.g. 'cloud storage')",
    approval_mode="never_require",                   # or "always_require" for human gating
)

# The supervisor receives analyst_tool and can call it just like @tool functions
supervisor = Agent(
    client=client,
    instructions="You coordinate research. Use analyse_market when sector analysis is needed.",
    tools=[analyst_tool],
)

import asyncio
response = asyncio.run(supervisor.run("Give me a market analysis of enterprise AI infrastructure."))
print(response.text)
```

### `as_tool()` parameters

| Parameter | Default | Purpose |
|---|---|---|
| `name` | agent's `name` | Tool name exposed to the model |
| `description` | agent's `description` | Tool description in the schema |
| `arg_name` | `"task"` | Name of the input argument |
| `arg_description` | `f"Task for {name}"` | Description of the input argument |
| `approval_mode` | `"never_require"` | Set `"always_require"` for human gating on every call |
| `stream_callback` | `None` | Async callback receiving `AgentResponseUpdate` objects as the sub-agent streams |
| `propagate_session` | `False` | When `True`, the parent's session is passed through so the sub-agent shares conversation history |

### Streaming from a sub-agent tool

When `stream_callback` is set, the tool uses `run(..., stream=True)` internally and calls your callback on every update. Use this to relay partial output from the sub-agent into a broader streaming UI:

```python
from agent_framework import AgentResponseUpdate


async def relay(update: AgentResponseUpdate) -> None:
    if update.text:
        print(f"[analyst] {update.text}", end="", flush=True)


analyst_tool = analyst.as_tool(
    name="analyse_market",
    stream_callback=relay,
)
```

### Session propagation

By default the sub-agent runs stateless on each call. Set `propagate_session=True` when the parent and sub-agent should share conversation history — useful when the sub-agent needs context from earlier turns:

```python
analyst_tool = analyst.as_tool(
    name="analyse_market",
    propagate_session=True,
)
```

---

## Functional workflows (experimental)

Beyond graph-based `WorkflowBuilder`, the framework ships `@workflow` and `@step` — a decorator-based pattern for writing workflows as plain `async` Python functions. See the dedicated page for full coverage:

**[Functional Workflows →](./python/microsoft_agent_framework_python_functional_workflows/)**

One-glance example:

```python
from agent_framework import Agent, step, workflow
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="Return bullet facts.")
writer = Agent(client=client, name="writer", instructions="Expand bullets into a paragraph.")


@step
async def research(topic: str) -> str:
    return (await researcher.run(topic)).text


@step
async def write(facts: str) -> str:
    return (await writer.run(f"Expand:\n{facts}")).text


@workflow(name="simple-pipeline")
async def pipeline(topic: str) -> str:
    facts = await research(topic)
    return await write(facts)


import asyncio
result = asyncio.run(pipeline.run("quantum networking"))
print(result.get_outputs()[-1])
```

---

## What was removed from this page

All of the following appeared in earlier drafts and are **not** real Python APIs in `agent-framework`:

- `from agent_framework.graphs import AgentGraph, GraphNode, GraphEdge, ConditionalEdge` — no `agent_framework.graphs` module.
- `AgentGraph(name="...")`, `graph.add_node(...)`, `graph.add_edge(...)` — real API is `WorkflowBuilder(start_executor=...).add_edge(...).build()`.
- `CheckpointConfig(...)`, `HITLConfig(...)`, `TimeTravelConfig(...)`, `ObservabilityConfig(...)`, `ErrorPolicy(...)` — no such classes. Checkpointing is via `FileCheckpointStorage` / `InMemoryCheckpointStorage`.
- `DeclarativeAgentLoader` — real class is `AgentFactory` (single agent) or `WorkflowFactory` (multi-agent).

## Further reading

- [WorkflowBuilder reference on Microsoft Learn](https://learn.microsoft.com/agent-framework/workflows/) — code-first workflow APIs.
- [Declarative workflows reference](https://learn.microsoft.com/agent-framework/workflows/declarative/) — full YAML action catalogue.
- Source: <https://github.com/microsoft/agent-framework>.

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

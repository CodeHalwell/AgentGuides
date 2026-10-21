---
title: "Microsoft Agent Framework — Workflows & Declarative Agents"
description: "Graph workflows with WorkflowBuilder and WorkflowContext, workflow visualisation, and YAML-defined agents and workflows with agent-framework-declarative."
framework: "microsoft-agent-framework"
sidebar:
  label: "Workflows & declarative agents"
---

This page covers two related capabilities:

1. **Graph workflows**: `WorkflowBuilder` in `agent-framework-core` (Python) and `Microsoft.Agents.AI.Workflows` (.NET) wires executors and agents into a directed graph with fan-out, fan-in, switch-case and conditional routing, checkpointing and human-in-the-loop pauses.
2. **Declarative agents and workflows**: `agent-framework-declarative` loads agents and workflows from YAML.

The examples are Python. For .NET, see the workflow and declarative sections of the [.NET feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/).

## Graph workflows

The workflow types are exported from `agent_framework`.

| Class / function | Purpose |
|---|---|
| `WorkflowBuilder(max_iterations=100, name=None, description=None, *, start_executor, checkpoint_storage=None, output_from=..., intermediate_output_from=...)` | Fluent builder; `.build()` returns a `Workflow`. |
| `Workflow` | The compiled graph. `run(message)`, `run(message, stream=True)`, `run(responses=...)`, `run(checkpoint_id=...)`, `as_agent()`. |
| `Executor` + `@handler` | Class-based node with one handler per input type. |
| `@executor` / `FunctionExecutor` | A node from a single function. |
| `AgentExecutor` | Wraps an agent; the builder does this for you when you pass an agent. |
| `WorkflowExecutor` | Wraps a whole workflow as one node. |
| `Case`, `Default` | Branches for switch-case routing. |
| `InMemoryCheckpointStorage`, `FileCheckpointStorage` | Checkpoint stores, passed as `checkpoint_storage=`. |
| `WorkflowContext`, `WorkflowEvent`, `WorkflowRunResult`, `WorkflowRunState` | Runtime types. |
| `WorkflowViz` | Mermaid and Graphviz rendering. |

Builder methods, all returning the builder for chaining:

- `.add_edge(source, target, condition=None)`: a fixed or conditional edge.
- `.add_chain([a, b, c])`: shorthand for consecutive edges.
- `.add_fan_out_edges(source, targets)` and `.add_fan_in_edges(sources, target)`: parallel branches and their join (the target receives a list).
- `.add_switch_case_edge_group(source, [Case(...), ..., Default(...)])`: first matching branch.
- `.add_multi_selection_edge_group(source, targets, selection_func)`: a function picks the subset of targets.
- `.build()`: validates the graph (types, connectivity, duplicate ids) and returns the `Workflow`.

### How data moves between executors

Two rules catch most people out:

- **Return values are discarded.** An executor passes data on only by calling `await ctx.send_message(...)`, and produces workflow output only with `await ctx.yield_output(...)`. A function executor that just returns a value sends nothing, and the workflow finishes with no output.
- **Types are checked at `build()`.** Function executors need annotations (or explicit `input=` / `output=` / `workflow_output=` arguments), so lambdas can't be used. Each edge's target must accept a type the source sends; an agent node accepts `str`, `Message`, a list of these or an upstream agent's response.

```python
import asyncio
from typing_extensions import Never

from agent_framework import WorkflowBuilder, WorkflowContext, executor


@executor(id="score")
async def score(text: str, ctx: WorkflowContext[dict]) -> None:
    await ctx.send_message({"text": text, "score": len(text.split()) / 10})


@executor(id="accept")
async def accept(payload: dict, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"Accepted: {payload['text']}")


@executor(id="reject")
async def reject(payload: dict, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"Rejected: {payload['text']}")


workflow = (
    WorkflowBuilder(start_executor=score, name="review")
    .add_edge(score, accept, condition=lambda p: p["score"] >= 0.5)
    .add_edge(score, reject, condition=lambda p: p["score"] < 0.5)
    .build()
)

print(asyncio.run(workflow.run("two words")).get_outputs())   # ['Rejected: two words']
```

Runnable examples of every routing shape, output filtering with `output_from`, nesting, events, checkpointing and `as_agent()` are in the [Python comprehensive guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#workflows). The prebuilt multi-agent patterns (sequential, concurrent, handoff, group chat, Magentic) are on the [orchestration page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/), and workflows written as plain async functions with `@workflow` / `@step` are on the [functional workflows page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_functional_workflows/).

### Checkpointing

```python
from agent_framework import FileCheckpointStorage, WorkflowBuilder

storage = FileCheckpointStorage("./checkpoints")
workflow = (
    WorkflowBuilder(start_executor=score, name="review", checkpoint_storage=storage)
    .add_edge(score, accept, condition=lambda p: p["score"] >= 0.5)
    .add_edge(score, reject, condition=lambda p: p["score"] < 0.5)
    .build()
)
```

The store saves one checkpoint per superstep and exposes `save`, `load`, `get_latest`, `list_checkpoints`, `list_checkpoint_ids` and `delete`. Resume with `workflow.run(checkpoint_id=...)`. Details and custom backends are on the [checkpointing page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_checkpointing/).

## `WorkflowContext` reference

Every handler takes the message plus a `WorkflowContext`. Its type parameters declare what the executor may send and yield:

| Annotation | Allows |
|---|---|
| `WorkflowContext` | Side effects only. |
| `WorkflowContext[OutT]` | `ctx.send_message(value: OutT)` to downstream executors. |
| `WorkflowContext[OutT, W_OutT]` | `send_message` plus `ctx.yield_output(value: W_OutT)` as workflow output. |
| `WorkflowContext[Never, W_OutT]` | Workflow output only. |
| `WorkflowContext[str \| int, bool]` | Unions on either side. |

| Member | Purpose |
|---|---|
| `await send_message(message, target_id=None)` | Send downstream; `target_id` delivers to one connected executor instead of all of them. |
| `await yield_output(output)` | Emit a workflow output (`result.get_outputs()`). |
| `await add_event(event)` | Emit a custom `WorkflowEvent`. |
| `await request_info(request_data, response_type, *, request_id=None)` | Pause for an external response, received by a `@response_handler` on the same executor. |
| `get_state(key, default=None)` / `set_state(key, value)` | Run-scoped state shared by all executors. |
| `source_executor_ids` / `get_source_executor_id()` | Who sent the current message; the method raises when there are several (fan-in). |
| `is_streaming()` | Whether the run was started with `stream=True`. |

Targeted delivery and shared state together:

```python
import asyncio
from typing_extensions import Never

from agent_framework import WorkflowBuilder, WorkflowContext, executor


@executor(id="intake")
async def intake(ticket: dict, ctx: WorkflowContext[str]) -> None:
    ctx.set_state("ticket_id", ticket["id"])
    target = "premium" if ticket.get("tier") == "pro" else "standard"
    await ctx.send_message(ticket["text"], target_id=target)


@executor(id="premium")
async def premium(text: str, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"[{ctx.get_state('ticket_id')}] priority queue: {text}")


@executor(id="standard")
async def standard(text: str, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"[{ctx.get_state('ticket_id')}] standard queue: {text}")


workflow = (
    WorkflowBuilder(start_executor=intake)
    .add_edge(intake, premium)
    .add_edge(intake, standard)
    .build()
)

result = asyncio.run(workflow.run({"id": "T-1", "tier": "pro", "text": "Login fails"}))
print(result.get_outputs())   # ['[T-1] priority queue: Login fails']
```

## Visualising workflows

`WorkflowViz(workflow)` renders any built workflow. Mermaid and DOT need no extra dependencies; SVG, PNG and PDF need the `graphviz` package and the Graphviz `dot` binary.

```python
from agent_framework import WorkflowViz

viz = WorkflowViz(workflow)
print(viz.to_mermaid())
print(viz.to_digraph())                    # DOT source
# viz.save_svg("workflow.svg")             # also save_png, save_pdf, export(format=...)
```

For the conditional workflow in the first example, `to_mermaid()` produces:

```text
flowchart TD
  score["score (Start)"];
  accept["accept"];
  reject["reject"];
  score -. conditional .-> accept;
  score -. conditional .-> reject;
```

Conditional edges are drawn dashed and labelled, nested workflows appear as subgraphs, and `include_internal_executors=True` adds the framework's internal nodes.

## Declarative agents and workflows

```bash
pip install agent-framework-declarative
```

`agent-framework-declarative` is stable and needs no `--pre`.

| Class | Purpose |
|---|---|
| `AgentFactory` | Builds an agent from YAML: `create_agent_from_yaml_path`, `create_agent_from_yaml`, `create_agent_from_dict` (each with an `_async` twin). |
| `WorkflowFactory` | Builds a workflow from YAML after `register_agent`, `register_tool` and `register_binding` calls: `create_workflow_from_yaml_path`, `create_workflow_from_yaml`, `create_workflow_from_definition`. |
| `ExternalInputRequest`, `ExternalInputResponse`, `AgentExternalInputRequest`, `AgentExternalInputResponse` | Human-in-the-loop request and response types. |
| `DefaultHttpRequestHandler`, `DefaultMCPToolHandler` | Handlers for the HTTP-request and MCP-tool actions. |
| `DeclarativeLoaderError`, `DeclarativeWorkflowError`, `ProviderLookupError` | Errors. |

### An agent from YAML

```python
from agent_framework.declarative import AgentFactory

agent = AgentFactory().create_agent_from_yaml_path("agent.yaml")
result = await agent.run("Hello")
```

`AgentFactory(safe_mode=True)` (the default) makes `=Env.*` expressions in the YAML resolve to empty values, so a YAML file from an untrusted source can't read your environment variables. It does not check file paths, so validate any path you pass to `create_agent_from_yaml_path`.

### A workflow from YAML

```python
from agent_framework.declarative import WorkflowFactory

factory = WorkflowFactory()
factory.register_agent("ResearcherAgent", researcher_agent)
factory.register_agent("WriterAgent", writer_agent)
factory.register_tool("get_weather", get_weather)

workflow = factory.create_workflow_from_yaml_path("pipeline.yaml")
result = await workflow.run({"topic": "AI in healthcare"})
```

A workflow that uses the HTTP-request or MCP-tool actions needs `http_request_handler=` or `mcp_tool_handler=` on the factory; without them, building the workflow raises `DeclarativeWorkflowError`.

### YAML actions

Declarative workflows use Power Fx expressions and a fixed set of `kind:` actions. The full reference is in [Microsoft's documentation](https://learn.microsoft.com/agent-framework/workflows/declarative/).

| Category | Actions |
|---|---|
| Variables | `SetVariable`, `SetMultipleVariables`, `AppendValue`, `ResetVariable`, `SetValue` |
| Control flow | `If`, `ConditionGroup`, `Foreach`, `RepeatUntil`, `BreakLoop`, `ContinueLoop`, `GotoAction` |
| Output | `SendActivity`, `EmitEvent` |
| Agents and tools | `InvokeAzureAgent`, `InvokeFunctionTool` |
| Human in the loop | `Question`, `Confirmation`, `RequestExternalInput`, `WaitForInput` |
| Workflow control | `EndWorkflow`, `EndConversation`, `CreateConversation` |

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

## Further reading

- [Python comprehensive guide: workflows](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#workflows)
- [Multi-agent orchestration](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/)
- [Functional workflows](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_functional_workflows/)
- [Agents as tools (`as_tool()`)](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/#wrapping-an-agent-as-a-tool--as_tool)
- [Declarative workflows on Microsoft Learn](https://learn.microsoft.com/agent-framework/workflows/declarative/)

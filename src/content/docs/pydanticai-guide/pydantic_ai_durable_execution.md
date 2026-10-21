---
title: "PydanticAI: Durable Execution"
description: "Make Pydantic AI agents survive crashes, restarts and long waits with Temporal, DBOS or Prefect, using the TemporalDurability, DBOSDurability and PrefectDurability capabilities."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Durable execution"
  order: 16
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%) — source package: `pydantic_ai.durable_exec`. The Temporal, DBOS and Prefect examples below were run against temporalio 1.34, dbos 3.3 and prefect 3.9.

## What durable execution gives you

An agent run is a loop of model requests and tool calls. If the process dies half-way through, a plain `agent.run()` starts again from scratch — re-paying for every model call and re-running every side effect. A durable execution engine records each step's result in a journal; after a crash, the run is replayed from the journal and continues from the first step that has no recorded result.

Pydantic AI integrates with three engines. In each case the integration routes the agent's I/O — model requests, MCP communication and (depending on the engine) tool calls and event handling — through the engine's durable unit, while your agent code stays an ordinary `Agent`:

| Engine | Durable unit | Container | Capability | Extra |
|---|---|---|---|---|
| [Temporal](https://temporal.io) | activity | workflow | `TemporalDurability` | `pydantic-ai[temporal]` |
| [DBOS](https://dbos.dev) | step | workflow | `DBOSDurability` | `pydantic-ai[dbos]` |
| [Prefect](https://prefect.io) | task | flow | `PrefectDurability` | `pydantic-ai[prefect]` |

None of the three extras is in the default `pydantic-ai` install, so add the one you use.

### Capability or wrapper?

Since 2.14.0 durability is a **capability** you attach to a normal agent:

```python
agent = Agent('openai:%%model=openai%%', name='support', capabilities=[TemporalDurability()])
```

The older wrapper classes — `TemporalAgent`, `DBOSAgent` and `PrefectAgent` — still work but emit `PydanticAIDeprecationWarning` and will be removed in V3. Their deprecation message lists the mapping for every constructor argument; the [migration table](#migrating-from-the-wrapper-classes) below summarises it.

Outside a workflow or flow, each capability is transparent: the agent behaves like a plain agent, which keeps unit tests simple.

---

## Temporal

```bash
pip install "pydantic-ai[temporal]"
temporal server start-dev        # local dev server on localhost:7233
```

Temporal splits code into **workflows** (deterministic orchestration, replayed from history) and **activities** (anything that does I/O). `TemporalDurability` turns the agent's model requests, MCP calls and tool calls into activities, so the workflow itself stays deterministic.

```python
import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    PydanticAIPlugin,
    PydanticAIWorkflow,
    TemporalDurability,
)

agent = Agent(
    'openai:%%model=openai%%',
    name='geography',                     # stable: it is part of every activity name
    instructions='Answer concisely.',
    capabilities=[
        TemporalDurability(
            activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
        )
    ],
)


@agent.tool_plain
async def population(city: str) -> int:
    """Look up a city's population."""
    return 9_200_000  # real code would call an API — this runs as an activity


@workflow.defn
class GeographyWorkflow(PydanticAIWorkflow):
    # PydanticAIPlugin reads this list and registers each agent's activities on the worker
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await agent.run(prompt)
        return result.output


async def main():
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='geography', workflows=[GeographyWorkflow]):
        output = await client.execute_workflow(
            GeographyWorkflow.run,
            args=['What is the capital of Mexico, and how many people live there?'],
            id='geography-1',
            task_queue='geography',
        )
        print(output)


asyncio.run(main())
```

Points worth knowing:

- **`PydanticAIPlugin`** installs the Pydantic payload converter (so messages, deps and outputs serialise correctly) and registers the activities of every agent listed in `__pydantic_ai_agents__`. `AgentPlugin(agent)` registers a single agent when you are not using `PydanticAIWorkflow`.
- **Activity configuration** is layered: `activity_config` (base, default 60-second `start_to_close_timeout`) → `model_activity_config` → `toolset_activity_config={toolset_id: ...}`. Per-tool configuration moved to tool metadata (`metadata={'temporal': ...}` on the tool, or a `SetToolMetadata` capability).
- **Models chosen at run time** must be registered: pass `models={'fast': some_model}` to `TemporalDurability` and select them by key. Model-name strings need no registration — they are rebuilt on the worker. For per-tenant credentials use a `ResolveModelId` capability rather than the old `provider_factory=`.
- **Deps must be serialisable.** They cross the activity boundary; `TemporalRunContext` carries only the JSON-safe part of `RunContext`.
- **Streaming**: `event_stream_handler=` on the capability runs inside activities; `event_stream_topic=` publishes events to a Temporal Workflow Stream that a client outside the workflow can read with `TemporalDurability.stream_agent_events()`.
- **Logfire**: add `LogfirePlugin()` next to `PydanticAIPlugin()` on the client for workflow and activity spans. It does not replace `PydanticAIPlugin`.

---

## DBOS

```bash
pip install "pydantic-ai[dbos]"
```

DBOS checkpoints workflows into a database (SQLite for development, Postgres in production). You write the workflow yourself with `@DBOS.workflow()` and call `agent.run()` inside it; `DBOSDurability` makes model requests and MCP I/O into checkpointed steps.

```python
import asyncio

from dbos import DBOS, DBOSConfig
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSDurability

agent = Agent(
    'openai:%%model=openai%%',
    name='support',
    instructions='Be concise.',
    capabilities=[DBOSDurability(model_step_config={'max_attempts': 3, 'backoff_rate': 2.0})],
)


@agent.tool_plain
@DBOS.step()          # function tools are NOT wrapped automatically — make side effects steps
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f'order {order_id}: shipped'


@DBOS.workflow()
async def answer(prompt: str) -> str:
    result = await agent.run(prompt)
    return result.output


async def main():
    config: DBOSConfig = {'name': 'support-app', 'system_database_url': 'sqlite:///dbos.sqlite'}
    DBOS(config=config)
    DBOS.launch()
    print(await answer('Where is order 42?'))


asyncio.run(main())
```

- `StepConfig` keys are `retries_allowed`, `interval_seconds`, `max_attempts` and `backoff_rate`; set them separately for model requests (`model_step_config`), MCP (`mcp_step_config`) and event handling (`event_stream_handler_step_config`).
- `parallel_execution_mode` accepts only `'parallel_ordered_events'` (default) and `'sequential'`: DBOS needs a deterministic order on replay, so the fully unordered `'parallel'` mode is not offered.
- If a previous deployment used `DBOSAgent`, `register_legacy_workflows=True` keeps its workflow names resolvable while in-flight runs finish.

---

## Prefect

```bash
pip install "pydantic-ai[prefect]"
```

Prefect runs the agent inside a **flow**; `PrefectDurability` makes model requests, tool calls and MCP calls into **tasks** with retries and result caching. A flow retry reuses the persisted results of tasks that already succeeded.

```python
import asyncio

from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectDurability, TaskConfig

agent = Agent(
    'openai:%%model=openai%%',
    name='research',
    capabilities=[
        PrefectDurability(
            model_task_config=TaskConfig(retries=3, retry_delay_seconds=[1, 5, 30]),
            tool_task_config=TaskConfig(retries=2, timeout_seconds=60),
        )
    ],
)


@flow(name='research-flow')
async def research_flow(topic: str) -> str:
    result = await agent.run(f'Research: {topic}')
    return result.output


if __name__ == '__main__':
    print(asyncio.run(research_flow('durable agents')))
```

`TaskConfig` is a `TypedDict` with `retries`, `retry_delay_seconds`, `retry_condition_fn`, `timeout_seconds`, `cache_policy`, `persist_result`, `result_storage` and `log_prints`. The default cache policy (`DEFAULT_PYDANTIC_AI_CACHE_POLICY`) ignores run-specific fields such as timestamps and run IDs, and is scoped to the flow run, so results are reused on a retry but not across unrelated runs.

---

## Choosing an engine

| You want… | Reach for |
|---|---|
| Runs that last hours or days, wait on humans or timers, and run on a fleet of workers | Temporal |
| Durability inside an existing Python service, with state in the Postgres you already run | DBOS |
| Agents as steps in data pipelines, with Prefect's scheduling, UI and caching | Prefect |

All three handle the same agent features: tools, MCP toolsets, structured output, deferred tools (human-in-the-loop) and streaming through an event handler.

---

## Migrating from the wrapper classes

| Wrapper argument (`TemporalAgent` / `DBOSAgent` / `PrefectAgent`) | Capability equivalent |
|---|---|
| `wrapped=agent` | Put the capability on the agent: `Agent(..., capabilities=[XDurability(...)])` |
| `name=` | `name=` on `Agent` (or on the capability) |
| `models=` | `models=` on the capability |
| `provider_factory=` (Temporal) | A deps-aware `ResolveModelId` capability |
| `activity_config=`, `model_activity_config=`, `toolset_activity_config=` (Temporal) | Same names on `TemporalDurability` |
| `tool_activity_config=` (Temporal) | Per-tool `metadata={'temporal': ...}` or `SetToolMetadata` |
| `model_step_config=`, `mcp_step_config=`, `parallel_execution_mode=` (DBOS) | Same names on `DBOSDurability` |
| `model_task_config=`, `mcp_task_config=`, `tool_task_config=` (Prefect) | Same names on `PrefectDurability` |
| DBOS / Prefect wrappers started the workflow/flow for you | Call `agent.run()` inside your own `@DBOS.workflow()` / `@flow` |

Temporal workflows started under `TemporalAgent` replay correctly after migrating as long as the agent name, toolset IDs and model registry keys stay the same.

---

## What durable execution does not cover

- **Graph state persistence.** `pydantic_graph.persistence` (`FullStatePersistence`, `FileStatePersistence`) was removed in 2.0, and the `GraphBuilder` API does not snapshot state. To save, resume and fork *agent run* state outside a durable engine, the separate [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) package provides a `StepPersistence` capability.
- **Exactly-once side effects.** A durable unit can run more than once if the process dies after the side effect but before the checkpoint commits. Make tools idempotent (idempotency keys, upserts) or use the engine's own at-most-once options.
- **Toolsets passed per run.** Toolsets given to `agent.run(toolsets=...)` arrive after the durable wrapping and are not checkpointed; engines reject the kinds they cannot support. Register durable toolsets on the agent itself.

---

## Gotchas

- **Give every durable agent a stable `name`.** It is baked into activity / step / task names, which are persisted in the journal; renaming breaks replay of in-flight runs.
- **Keep workflow code deterministic.** No `datetime.now()`, random numbers or direct network calls in the workflow body — put them in tools, which run as durable units.
- **Async tools under Temporal.** A tool you exclude from activity wrapping must be `async`; sync tools run in threads, which is non-deterministic inside a workflow.
- **Test without the engine.** Outside a workflow/flow the capability does nothing, so `agent.override(model=TestModel())` works in ordinary unit tests.

## Reference

- `TemporalDurability`, `PydanticAIPlugin`, `AgentPlugin`, `PydanticAIWorkflow`, `LogfirePlugin`, `TemporalRunContext` — `pydantic_ai/durable_exec/temporal/`
- `DBOSDurability`, `StepConfig`, `DBOSParallelExecutionMode` — `pydantic_ai/durable_exec/dbos/`
- `PrefectDurability`, `TaskConfig`, `DEFAULT_PYDANTIC_AI_CACHE_POLICY` — `pydantic_ai/durable_exec/prefect/`
- Deprecated wrappers: `TemporalAgent`, `DBOSAgent`, `PrefectAgent` (same packages)
- Engine-neutral machinery for building new integrations (`BaseDurabilityCapability`, `CallableOperationBackend`, `RegisteredOperationBackend`) — private modules under `pydantic_ai/durable_exec/`; treat as advanced and subject to change

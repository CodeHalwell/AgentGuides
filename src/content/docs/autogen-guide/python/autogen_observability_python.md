---
title: "AutoGen Observability & Recovery (Python)"
description: "Logging, OpenTelemetry tracing, token accounting, cancellation and state checkpoints for Microsoft AutoGen 0.7 teams."
framework: "autogen"
language: "python"
---

> **Microsoft AutoGen** (`autogen-agentchat` %%version=autogen-agentchat%%, checked %%versions-checked%%) — not AG2. The project is archived; prefer [Microsoft Agent Framework](/microsoft-agent-framework-guide/) for new deployments.

## Logging

AutoGen logs through the standard `logging` module under well-known logger names:

```python
import logging

from autogen_agentchat import EVENT_LOGGER_NAME as AGENTCHAT_EVENTS, TRACE_LOGGER_NAME as AGENTCHAT_TRACE
from autogen_core import EVENT_LOGGER_NAME as CORE_EVENTS

logging.basicConfig(level=logging.WARNING)
logging.getLogger(AGENTCHAT_TRACE).setLevel(logging.DEBUG)     # human-readable debug trace
logging.getLogger(AGENTCHAT_EVENTS).setLevel(logging.INFO)     # structured agent/team events
logging.getLogger(CORE_EVENTS).setLevel(logging.INFO)          # LLM calls, tool calls, message delivery
```

Event loggers emit structured records (for example LLM call events with prompt and completion token counts), suitable for shipping to a log pipeline.

## OpenTelemetry tracing

The Core runtime emits OpenTelemetry spans for message delivery, agent handlers and tool execution. Give it a tracer provider and let the team use that runtime:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import SingleThreadedAgentRuntime

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))  # swap for an OTLP exporter
trace.set_tracer_provider(provider)


def traced_team(agents, termination) -> tuple[RoundRobinGroupChat, SingleThreadedAgentRuntime]:
    runtime = SingleThreadedAgentRuntime(tracer_provider=provider)
    team = RoundRobinGroupChat(agents, termination_condition=termination, runtime=runtime)
    return team, runtime

# runtime.start(); await team.run(task=...); await runtime.stop()
```

Set `AUTOGEN_DISABLE_RUNTIME_TRACING=true` to turn runtime spans off.

## Token usage

Every `TaskResult.messages` entry from a model carries `models_usage` (`prompt_tokens`, `completion_tokens`). Sum them per run, or stop runaway teams with `TokenUsageTermination(max_total_token=...)`.

```python
def total_tokens(result) -> int:
    return sum(
        (m.models_usage.prompt_tokens + m.models_usage.completion_tokens)
        for m in result.messages
        if getattr(m, "models_usage", None)
    )
```

## Termination and recovery

- **Bound every team**: combine a goal condition with a hard cap, e.g. `TextMentionTermination("APPROVE") | MaxMessageTermination(20) | TimeoutTermination(300)`.
- **Cancel from outside**: pass a `CancellationToken` to `run()` / `run_stream()` and call `token.cancel()`; or attach `ExternalTermination()` and call `.set()` to stop after the current speaker finishes.
- **Checkpoint**: `state = await team.save_state()` after each run and persist the JSON; `await team.load_state(state)` restores a fresh team in a new process. The same works per agent.
- **Retry transient model errors** at the model-client level (the OpenAI clients accept `max_retries=`), not by re-running whole teams.
- **Escalate to a human** with `HandoffTermination(target="user")` in a `Swarm`, or a `UserProxyAgent` participant.

---
title: "Haystack Observability & Error Recovery (Python)"
description: "Tracing pipelines and agents, logging, retries, fallbacks and dead letters in Haystack 3."
framework: "haystack"
language: "python"
sidebar:
  label: "Observability"
  order: 6
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). Upstream: [release notes](https://github.com/deepset-ai/haystack/releases) · [tracing docs](https://docs.haystack.deepset.ai/docs/tracing)

Haystack 3 changed observability in three ways that matter when upgrading:

1. **No tracing backend ships in the core.** `OpenTelemetryTracer` moved to `opentelemetry-haystack` and `DatadogTracer` to `datadog-haystack`.
2. **Nothing is auto-enabled.** Installing `opentelemetry-sdk` or `ddtrace` no longer switches tracing on; `HAYSTACK_AUTO_TRACE_ENABLED` has no effect. Enable a tracer explicitly or add a connector component.
3. **Agent spans were reshaped.** Each loop iteration emits a `haystack.agent.step` span with children `haystack.agent.step.llm` and one `haystack.agent.step.tool` span per tool call.

## Tracing

### Quick local tracing

`LoggingTracer` writes spans and their tags to the Python log, which is enough for development:

```python
import logging

from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)

tracing.tracer.is_content_tracing_enabled = True   # include inputs/outputs; avoid in production
tracing.enable_tracing(LoggingTracer())
```

### OpenTelemetry

```bash
pip install opentelemetry-haystack opentelemetry-exporter-otlp
```

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from haystack import tracing
from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer

provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "haystack-api"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")))
trace.set_tracer_provider(provider)

tracing.enable_tracing(OpenTelemetryTracer(trace.get_tracer("haystack-api")))
```

Alternatively add `OpenTelemetryConnector()` (from `haystack_integrations.components.connectors.opentelemetry`) to a pipeline.

### Langfuse

```python
import os
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"   # set before importing haystack

from haystack_integrations.components.connectors.langfuse import LangfuseConnector  # pip install langfuse-haystack

# Instantiating the connector registers the tracer globally, so standalone Agent runs are traced too.
LangfuseConnector("support-agent")
```

Reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and optionally `LANGFUSE_HOST`. Adding the connector to a pipeline also exposes a `trace_url` output.

### What gets traced

| Span | Emitted for |
|------|-------------|
| `haystack.pipeline.run` | Every pipeline run; tag `haystack.pipeline.execution_mode` is `sync` or `async` |
| `haystack.component.run` | Every component run inside a pipeline |
| `haystack.agent.run` | An agent run |
| `haystack.agent.step` | One loop iteration |
| `haystack.agent.step.llm` | The chat-generator call in that step |
| `haystack.agent.step.tool` | One per tool call, tagged `haystack.tool.name` |

Inputs and outputs are attached as tags only when content tracing is enabled (`HAYSTACK_CONTENT_TRACING_ENABLED=true`). Leave it off where prompts may contain personal data, or redact in your exporter.

### Recording spans in tests

A tracer is a small abstract class: `Tracer.trace()` is a context manager that yields a `Span`, and `Span.set_tag()` receives the tags. Writing one that keeps spans in a list lets a test assert on what a pipeline or agent actually did (which tools ran, how many steps) without any external backend. The model is scripted with `MockChatGenerator`, so this runs offline:

```python
import contextlib
from collections.abc import Iterator
from typing import Any

from haystack import Pipeline, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool
from haystack.tracing import Span, Tracer


class RecordedSpan(Span):
    def __init__(self, name: str, tags: dict[str, Any] | None):
        self.name = name
        self.tags = dict(tags or {})

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


class RecordingTracer(Tracer):
    """Keeps every span in a list, so a test can assert on what ran."""

    def __init__(self):
        self.spans: list[RecordedSpan] = []

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        span = RecordedSpan(operation_name, tags)
        self.spans.append(span)
        yield span

    def current_span(self) -> Span | None:
        return None      # only used to correlate log records with spans


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


llm = MockChatGenerator(responses=[
    ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Paris"})]),
    "It is sunny in Paris.",
])
pipeline = Pipeline()
pipeline.add_component("agent", Agent(chat_generator=llm, tools=[get_weather]))

recorder = RecordingTracer()
tracing.enable_tracing(recorder)
try:
    pipeline.run({"agent": {"messages": [ChatMessage.from_user("Weather in Paris?")]}})
finally:
    tracing.disable_tracing()

for span in recorder.spans:
    print(span.name, span.tags.get("haystack.tool.name", ""))
# haystack.pipeline.run
# haystack.component.run
# haystack.agent.run
# haystack.agent.step
# haystack.agent.step.llm
# haystack.agent.step.tool get_weather
# haystack.agent.step
# haystack.agent.step.llm
```

Two steps ran: the first called `get_weather`, the second produced the final text. Use `tracing.disable_tracing()` in a `finally` block or fixture so one test's tracer does not leak into the next. Real backends implement the same two classes; `OpenTelemetryTracer` and the Langfuse tracer are examples.

## Logging

Importing Haystack no longer reconfigures the root logger. Its handler is attached to the `haystack`, `haystack_integrations` and `haystack_experimental` loggers only.

```python
from haystack import logging as haystack_logging

haystack_logging.configure_logging(use_json=True)          # structured JSON for Haystack's loggers
haystack_logging.configure_logging(propagate=False)        # stop duplicate lines if your root logger has a handler
haystack_logging.configure_logging(logger_name="")         # 2.x behaviour: format every record in the process
```

Correlate logs with traces by logging the trace or request ID in your API layer. With `use_json=True`, each record also carries the current trace and span IDs when the active tracer provides them; the OpenTelemetry and Langfuse tracers do, `LoggingTracer` does not.

## Error recovery

### Retries and fallbacks

This is the guide's one place for resilience around the model and the agent loop. Several layers help:

1. **Inside tools.** Return error strings the model can act on. Tool exceptions are reported to the model as error results unless the Agent has `raise_on_tool_invocation_failure=True`.
2. **Chat generator.** `timeout` and `max_retries` on OpenAI-family generators retry transient API errors inside the client.
3. **Provider fallback.** `FallbackChatGenerator` tries generators in order until one succeeds.
4. **Loop bounds.** `max_agent_steps` and [`TokenBudgetHook`](/haystack-guide/haystack_advanced_agents_python/#token-budgets) cap cost; `exit_reason` in the result says why the run stopped, and can drive a `ConditionalRouter` downstream.
5. **Around the run.** Retry with back-off for errors that escape the agent.

```python
import logging
import time

from haystack.components.agents import Agent
from haystack.components.generators.chat import FallbackChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator  # pip install anthropic-haystack

logger = logging.getLogger(__name__)

resilient_llm = FallbackChatGenerator(chat_generators=[
    OpenAIChatGenerator(model="%%model=openai-small%%", timeout=30, max_retries=2),
    AnthropicChatGenerator(model="%%model=anthropic%%", timeout=30, max_retries=2),
])

agent = Agent(chat_generator=resilient_llm, max_agent_steps=10)


def run_with_retry(agent: Agent, text: str, attempts: int = 3, backoff: float = 2.0) -> dict:
    for attempt in range(1, attempts + 1):
        try:
            result = agent.run(messages=[ChatMessage.from_user(text)])
            # exit_reason is "text", "length", "content_filter", "max_agent_steps",
            # the name of an exit-condition tool, or a reason set by a hook (stop_run)
            if result["exit_reason"] == "max_agent_steps":
                logger.warning("Agent hit max_agent_steps; the answer may be incomplete")
            return result
        except Exception:
            logger.exception("Agent run failed (attempt %d/%d)", attempt, attempts)
            if attempt == attempts:
                raise
            time.sleep(backoff ** attempt)
```

### Pipeline errors

When a component raises inside `Pipeline.run`, Haystack wraps it in `PipelineRuntimeError`, which carries the component name. Catch it at the boundary and route the failure to a dead-letter store with enough context to replay:

```python
import json
import logging
import time
import uuid

from haystack.core.errors import PipelineRuntimeError

logger = logging.getLogger(__name__)


def run_with_dead_letter(pipeline, data: dict, attempts: int = 3) -> dict | None:
    run_id = str(uuid.uuid4())
    for attempt in range(1, attempts + 1):
        try:
            return pipeline.run(data)
        except PipelineRuntimeError as exc:
            logger.warning("run %s failed in %s (attempt %d): %s", run_id, exc.component_name, attempt, exc)
            if attempt < attempts:
                time.sleep(2 ** attempt)
            else:
                with open("dead_letters.jsonl", "a") as fp:   # use a queue or table in production
                    fp.write(json.dumps({"run_id": run_id, "component": exc.component_name,
                                         "error": str(exc), "data": data}, default=str) + "\n")
    return None
```

### Pipeline breakpoints

For debugging a misbehaving pipeline, `Pipeline.run(..., break_point=Breakpoint(component_name="llm", visit_count=0, snapshot_file_path="snapshots/"))` stops before a component and writes a `PipelineSnapshot` you can inspect and resume with `pipeline_snapshot=`. Breakpoints are supported by synchronous `run` only, and the agent-internal breakpoints of 2.x (`AgentBreakpoint`, `ToolBreakpoint`) were removed in 3.0; use tracing to inspect agent steps.

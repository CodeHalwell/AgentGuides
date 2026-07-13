---
title: "Microsoft Agent Framework (Python) — Observability & Telemetry"
description: "OpenTelemetry traces, metrics, and logs emitted by agent-framework-core 1.6.0. Enable instrumentation, ship to Azure Monitor / OTLP / console, or the VS Code AI Toolkit."
framework: microsoft-agent-framework
language: python
---

# Observability & Telemetry — Python

`agent-framework-core` emits OpenTelemetry signals following the **GenAI semantic conventions**. Every agent run, model call, tool invocation, workflow executor, and edge group is a span; every chat completion emits a duration histogram and a token-usage histogram. Nothing is exported by default — you opt in either by calling a helper, setting one env var, or wiring your own OTel providers.

Verified against `agent-framework-core==1.6.0` (`agent_framework.observability`).

## Three ways to turn it on

Pick exactly one. Mixing them causes duplicate providers.

### 1. `configure_otel_providers()` — batteries included

Auto-wires OTLP exporters from standard OTel env vars and enables instrumentation in one call.

```python
from agent_framework.observability import configure_otel_providers

configure_otel_providers()
```

Set the usual OTel env vars before calling:

```bash
export ENABLE_INSTRUMENTATION=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_SERVICE_NAME=my-agent
```

For local debugging, route everything to stdout:

```python
configure_otel_providers(enable_console_exporters=True)
```

### 2. Bring your own providers — just enable instrumentation

When your app already owns OTel setup (Azure Monitor, Datadog, Honeycomb agents), skip `configure_otel_providers` and call `enable_instrumentation()`:

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework.observability import enable_instrumentation

configure_azure_monitor(connection_string=AZURE_MONITOR_CONNECTION)
enable_instrumentation()
```

### 3. Custom exporters, keeping the framework's providers

Pass your own exporters and optional `View`s; framework-managed providers are still created:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.metrics.view import View
from agent_framework.observability import configure_otel_providers, create_metric_views

configure_otel_providers(
    exporters=[
        OTLPSpanExporter(endpoint="http://otel-collector:4317"),
        OTLPLogExporter(endpoint="http://otel-collector:4317"),
    ],
    views=create_metric_views(),  # default GenAI + agent_framework views
)
```

## What you get

### Spans

| Span name | Source | Key attributes |
|---|---|---|
| `invoke_agent` | `Agent.run()` | `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.conversation.id` |
| `chat` | Every chat-client call | `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons` |
| `execute_tool` | Each tool invocation | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |
| `workflow.build` / `workflow.run` | Workflow lifecycle | `workflow.name`, `workflow.id` |
| `executor.process` | Each executor run | `executor.id`, `executor.type` |
| `edge_group.process` | Each edge traversal | `edge_group.id`, `edge_group.type`, `edge_group.delivery_status` |
| `message.send` | Cross-executor messaging | `message.source_id`, `message.target_id`, `message.type` |

All attribute constants live in `agent_framework.observability.OtelAttr` — use them to build span processors and samplers rather than typing strings.

### Metrics

Two histograms emit automatically once instrumentation is on:

| Metric | Unit | Dimensions |
|---|---|---|
| `gen_ai.client.operation.duration` | seconds | `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.provider.name`, `error.type` |
| `gen_ai.client.token.usage` | tokens | Same + `gen_ai.token.type` (`input` \| `output`) |

Plus `agent_framework.function.invocation.duration` per tool call.

### Logs

Agent-framework forwards to the standard Python `logging` module. Configure via `dictConfig`:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("agent_framework").setLevel(logging.DEBUG)
```

When instrumentation is enabled, logs flow through OTel's `LoggingHandler` and ship alongside traces/metrics.

### `MessageListTimestampFilter` — deconflicting log timestamps

When a single `chat` span emits a burst of message logs, they can share the same timestamp and appear out of order in some log UIs. `MessageListTimestampFilter` fixes this by nudging each log record's `created` time forward by `index × 1 µs`, so message logs sort stably:

```python
import logging
from agent_framework.observability import MessageListTimestampFilter

handler = logging.StreamHandler()
handler.addFilter(MessageListTimestampFilter())
logging.getLogger("agent_framework").addHandler(handler)
```

The filter checks for the `chat_message_index` attribute on each `LogRecord` (`MessageListTimestampFilter.INDEX_KEY`). Records without it are passed through unchanged, so wiring it globally is safe.

## Sensitive-data events

Prompts and completions are redacted by default. Flip `enable_sensitive_data=True` to include full message bodies as span events — useful in dev, **avoid in production**.

```python
configure_otel_providers(enable_sensitive_data=True)
# or: export ENABLE_SENSITIVE_DATA=true
```

## VS Code AI Toolkit / Foundry extension

The AI Toolkit and Azure AI Foundry VS Code extensions listen on a local port for live traces. Point the framework at it:

```python
configure_otel_providers(vs_code_extension_port=4317)
# or: export VS_CODE_EXTENSION_PORT=4317
```

Open the "AI Toolkit: Traces" view — every agent run, tool call, and workflow edge shows up with inputs, outputs, and timing.

## Provider telemetry layers — `ChatTelemetryLayer` and `EmbeddingTelemetryLayer`

Every built-in chat and embedding client inherits from one of these mix-in classes via Python's MRO. You do not instantiate them directly; they slot silently into the class hierarchy and gate span/histogram emission on `OBSERVABILITY_SETTINGS.ENABLED`.

```
OpenAIChatClient → FunctionInvocationLayer → ChatMiddlewareLayer → ChatTelemetryLayer → RawOpenAIChatClient
OpenAIEmbeddingClient → EmbeddingTelemetryLayer → RawOpenAIEmbeddingClient
```

**Writing a custom provider** that participates in the same telemetry pipeline inherits from `ChatTelemetryLayer` (or `EmbeddingTelemetryLayer`) plus your raw client:

```python
from agent_framework.observability import ChatTelemetryLayer
from agent_framework import BaseChatClient


class RawMyModelClient(BaseChatClient):
    """Bare client — no tracing."""

    OTEL_PROVIDER_NAME = "my_model"

    async def _get_response_impl(self, messages, *, options=None, **kwargs):
        # ... call your model endpoint ...
        pass


class MyModelClient(ChatTelemetryLayer, RawMyModelClient):
    """Production client — OTel tracing included via mix-in."""
    pass


# MyModelClient now emits `chat` spans with gen_ai.provider.name="my_model"
# whenever OBSERVABILITY_SETTINGS.ENABLED is True.
```

`ChatTelemetryLayer.__init__` accepts an optional `otel_provider_name` kwarg that overrides `OTEL_PROVIDER_NAME` at construction time:

```python
client = MyModelClient(otel_provider_name="my_model_prod")
```

`EmbeddingTelemetryLayer` follows the same pattern for embedding clients — wrap `RawMyEmbeddingClient` to get `embeddings` spans with token-usage histograms.

## Custom spans around business logic

You can still wrap your own code in spans; they slot neatly into the framework's tree.

```python
from opentelemetry import trace

tracer = trace.get_tracer("myapp.customer")

with tracer.start_as_current_span("process_order", attributes={"order.id": order_id}):
    response = await agent.run(f"Draft reply for order {order_id}")
```

For agent-level wrapping use `AgentMiddleware` instead — see the [middleware page](./microsoft_agent_framework_python_middleware/#emitting-opentelemetry-spans).

### Use the framework's tracer / meter helpers

`agent_framework.observability.get_tracer()` and `get_meter()` return providers that share the same configuration the framework uses internally. Calling them ensures your custom spans land alongside `invoke_agent` / `chat` / `execute_tool` and inherit the same resource attributes (service name, version):

```python
from agent_framework.observability import get_tracer, get_meter

tracer = get_tracer("myapp.pricing")
meter = get_meter("myapp.pricing")

cache_hits = meter.create_counter(
    "myapp.pricing.cache_hits",
    description="Pricing cache hit count",
    unit="{request}",
)


async def price_quote(sku: str, *, cache, agent):
    with tracer.start_as_current_span("price_quote", attributes={"sku": sku}):
        if cached := cache.get(sku):
            cache_hits.add(1, {"sku.tier": cached.tier})
            return cached
        return await agent.run(f"Price quote for {sku}")
```

## Reading framework attributes with `OtelAttr`

`OtelAttr` is the single source of truth for every span attribute the framework emits. Use it to filter, sample, or post-process spans without typing string keys:

```python
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SpanProcessor
from agent_framework.observability import OtelAttr


class TokenSpendProcessor(SpanProcessor):
    """Side-car processor that counts model spend per agent."""

    def __init__(self) -> None:
        self.totals: dict[str, int] = {}

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        # No-op — counting happens once spans are finalised in `on_end`.
        return

    def on_end(self, span: ReadableSpan) -> None:
        if span.name != OtelAttr.CHAT_COMPLETION_OPERATION.value:
            return
        attrs = dict(span.attributes or {})
        agent = attrs.get(OtelAttr.AGENT_NAME.value, "unknown")
        in_tok = attrs.get(OtelAttr.INPUT_TOKENS.value, 0)
        out_tok = attrs.get(OtelAttr.OUTPUT_TOKENS.value, 0)
        self.totals[agent] = self.totals.get(agent, 0) + in_tok + out_tok

    def shutdown(self) -> None: ...
    def force_flush(self, timeout_millis: int = 30_000) -> bool: return True


# Register alongside the framework's exporters.
from opentelemetry import trace
trace.get_tracer_provider().add_span_processor(TokenSpendProcessor())
```

Common attribute keys you'll reach for, organised by domain:

**Agent & conversation**

| Attribute | String value |
|---|---|
| `OtelAttr.AGENT_NAME` | `gen_ai.agent.name` |
| `OtelAttr.AGENT_ID` | `gen_ai.agent.id` |
| `OtelAttr.AGENT_DESCRIPTION` | `gen_ai.agent.description` |
| `OtelAttr.CONVERSATION_ID` | `gen_ai.conversation.id` |
| `OtelAttr.SYSTEM_INSTRUCTIONS` | `gen_ai.system_instructions` |

**Model request / response**

| Attribute | String value |
|---|---|
| `OtelAttr.REQUEST_MODEL` | `gen_ai.request.model` |
| `OtelAttr.RESPONSE_MODEL` | `gen_ai.response.model` |
| `OtelAttr.PROVIDER_NAME` | `gen_ai.provider.name` |
| `OtelAttr.OPERATION` | `gen_ai.operation.name` |
| `OtelAttr.FINISH_REASONS` | `gen_ai.response.finish_reasons` |
| `OtelAttr.REQUEST_MAX_TOKENS` | `gen_ai.request.max_tokens` |
| `OtelAttr.REQUEST_TEMPERATURE` | `gen_ai.request.temperature` |
| `OtelAttr.REQUEST_TOP_P` | `gen_ai.request.top_p` |

**Token usage**

| Attribute | String value |
|---|---|
| `OtelAttr.INPUT_TOKENS` | `gen_ai.usage.input_tokens` |
| `OtelAttr.OUTPUT_TOKENS` | `gen_ai.usage.output_tokens` |
| `OtelAttr.CACHE_READ_INPUT_TOKENS` | `gen_ai.usage.cache_read.input_tokens` |
| `OtelAttr.CACHE_CREATION_INPUT_TOKENS` | `gen_ai.usage.cache_creation.input_tokens` |
| `OtelAttr.REASONING_OUTPUT_TOKENS` | `gen_ai.usage.reasoning.output_tokens` |

**Tools**

| Attribute | String value |
|---|---|
| `OtelAttr.TOOL_NAME` | `gen_ai.tool.name` |
| `OtelAttr.TOOL_TYPE` | `gen_ai.tool.type` |
| `OtelAttr.TOOL_CALL_ID` | `gen_ai.tool.call.id` |
| `OtelAttr.TOOL_ARGUMENTS` | `gen_ai.tool.call.arguments` |
| `OtelAttr.TOOL_RESULT` | `gen_ai.tool.call.result` |
| `OtelAttr.TOOL_DESCRIPTION` | `gen_ai.tool.description` |
| `OtelAttr.TOOL_DEFINITIONS` | `gen_ai.tool.definitions` |

**Workflow / executor / edge**

| Attribute | String value |
|---|---|
| `OtelAttr.WORKFLOW_NAME` | `workflow.name` |
| `OtelAttr.WORKFLOW_ID` | `workflow.id` |
| `OtelAttr.EXECUTOR_ID` | `executor.id` |
| `OtelAttr.EXECUTOR_TYPE` | `executor.type` |
| `OtelAttr.EDGE_GROUP_ID` | `edge_group.id` |
| `OtelAttr.EDGE_GROUP_TYPE` | `edge_group.type` |
| `OtelAttr.EDGE_GROUP_DELIVERY_STATUS` | `edge_group.delivery_status` |

**Span names** (compare `span.name` against these)

| Attribute | String value |
|---|---|
| `OtelAttr.CHAT_COMPLETION_OPERATION` | `chat` |
| `OtelAttr.EMBEDDING_OPERATION` | `embeddings` |
| `OtelAttr.TOOL_EXECUTION_OPERATION` | `execute_tool` |
| `OtelAttr.AGENT_INVOKE_OPERATION` | `invoke_agent` |
| `OtelAttr.WORKFLOW_BUILD_SPAN` | `workflow.build` |
| `OtelAttr.WORKFLOW_RUN_SPAN` | `workflow.run` |
| `OtelAttr.EXECUTOR_PROCESS_SPAN` | `executor.process` |
| `OtelAttr.EDGE_GROUP_PROCESS_SPAN` | `edge_group.process` |
| `OtelAttr.MESSAGE_SEND_SPAN` | `message.send` |

**Metrics**

| Attribute | String value |
|---|---|
| `OtelAttr.LLM_OPERATION_DURATION` | `gen_ai.client.operation.duration` |
| `OtelAttr.LLM_TOKEN_USAGE` | `gen_ai.client.token.usage` |
| `OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION` | `agent_framework.function.invocation.duration` |

## Sampling — keep traces affordable

OTel samplers run before exporters; pair them with `configure_otel_providers(...)` so the framework spans inherit your sampler:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from agent_framework.observability import enable_instrumentation

# 5% head-based sampling for high-volume production; always trace if upstream did.
sampler = ParentBased(root=TraceIdRatioBased(0.05))
provider = TracerProvider(sampler=sampler)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

enable_instrumentation()        # do NOT call configure_otel_providers — your provider wins
```

For tail-based sampling (keep every trace that errored or exceeded a latency budget), use the OTel Collector's `tail_sampling` processor — agent-framework spans expose `error.type` and the duration histograms as standard inputs.

## Filtering noisy spans

A custom `SpanProcessor` can drop spans the exporter never sees — useful when you want to capture model calls but not every internal `edge_group.process`:

```python
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter
from agent_framework.observability import OtelAttr


class DropInternalEdgeSpans(SpanExporter):
    """Wrap any exporter to drop edge-group spans below a configurable threshold."""

    def __init__(self, inner: SpanExporter, *, min_duration_ms: float = 50.0) -> None:
        self._inner = inner
        self._min_ns = min_duration_ms * 1_000_000

    def export(self, spans):
        keep = [
            s for s in spans
            if s.name != OtelAttr.EDGE_GROUP_PROCESS_SPAN.value
            or (s.end_time - s.start_time) > self._min_ns
        ]
        return self._inner.export(keep) if keep else SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # Forward flush so graceful shutdown drains the wrapped exporter.
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()
```

Wrap your real exporter in `BatchSpanProcessor(DropInternalEdgeSpans(real_exporter))`.

## Capping metric cardinality

`create_metric_views()` returns the framework's default views. Append your own to drop attributes you don't query on — high-cardinality dimensions (per-user IDs, request IDs) blow up storage costs:

```python
from opentelemetry.sdk.metrics.view import View
from agent_framework.observability import (
    configure_otel_providers,
    create_metric_views,
    OtelAttr,
)

views = create_metric_views() + [
    # Aggregate token usage without conversation_id — keep model + agent dimensions only.
    View(
        instrument_name=OtelAttr.LLM_TOKEN_USAGE.value,
        attribute_keys={
            OtelAttr.RESPONSE_MODEL.value,
            OtelAttr.AGENT_NAME.value,
            OtelAttr.T_TYPE.value,
        },
    ),
]

configure_otel_providers(views=views)
```

## Stamping your own context onto every span

Two ways to attach business attributes — pick the one that matches lifetime:

1. **Per-`agent.run(...)` call** — pass `additional_properties={"gen_ai.conversation.id": req_id}` so the conversation correlator uses your upstream request ID.
2. **Per-process resource attributes** — set `OTEL_RESOURCE_ATTRIBUTES=team=payments,env=prod,deployment.version=1.4.2` before launching your service. The framework attaches them to every span/metric/log automatically.

```python
import os
os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "team=payments,env=prod,deployment.version=1.4.2"
os.environ["OTEL_SERVICE_NAME"] = "checkout-agent"

from agent_framework.observability import configure_otel_providers
configure_otel_providers()
```

For request-scoped attributes use a span processor that reads from a `contextvars.ContextVar` you set at the API boundary — that way every framework-emitted span inherits them without changing call sites.

## Correlating conversations

The framework stamps `gen_ai.conversation.id` on every span in a single `agent.run(...)` call and across `session`-scoped runs. In your tracing UI, filter by that attribute to see the full conversation on one trace.

Pass a custom ID through `additional_properties` on `Agent(...)` to use your own correlation key (e.g. a request ID from upstream).

## Azure Monitor

One-liner for Application Insights:

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework.observability import enable_instrumentation

configure_azure_monitor(connection_string="InstrumentationKey=...;IngestionEndpoint=...")
enable_instrumentation()
```

Then query in Application Insights:

- **All agent runs** — `dependencies | where name == "invoke_agent"`
- **Slow chat calls** — `dependencies | where name == "chat" | where duration > 3000 | summarize count() by customDimensions.["gen_ai.response.model"]`
- **Tool-call volume** — `dependencies | where name == "execute_tool" | summarize count() by customDimensions.["gen_ai.tool.name"]`

## Production checklist

- Set `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` (team, env, version).
- Leave `ENABLE_SENSITIVE_DATA` **off** unless you're scrubbing downstream.
- Use OTLP gRPC (`OTEL_EXPORTER_OTLP_PROTOCOL=grpc`) to an in-cluster collector; batch there rather than from each pod.
- Wire a `LogRecordExporter` too — tool stack traces and middleware failures land in logs, not spans.
- Filter noise with `create_metric_views()` and pass extra `View(instrument_name="...", aggregation=...)` entries to cap cardinality on high-traffic deployments.
- Disable the `user-agent` telemetry banner with `AGENT_FRAMEWORK_USER_AGENT_TELEMETRY_DISABLED=true` if corporate policy forbids it.

## `ObservabilitySettings` — the runtime toggle singleton

`OBSERVABILITY_SETTINGS` is the process-wide singleton that all framework code checks before emitting any signal. You rarely need to touch it directly — `enable_instrumentation()`, `disable_instrumentation()`, and `configure_otel_providers()` are the public API — but knowing its properties helps when debugging:

```python
from agent_framework.observability import OBSERVABILITY_SETTINGS

# True when instrumentation is on and no sticky-disable is in effect.
print(OBSERVABILITY_SETTINGS.ENABLED)

# True when sensitive-data capture is also enabled.
print(OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED)

# True after disable_instrumentation() is called; False until then.
print(OBSERVABILITY_SETTINGS.is_user_disabled)

# True after _configure() has run (i.e. OTel providers are set up).
print(OBSERVABILITY_SETTINGS.is_setup)
```

You can create a fresh `ObservabilitySettings` with constructor overrides (useful in tests):

```python
from agent_framework.observability import ObservabilitySettings

settings = ObservabilitySettings(
    enable_instrumentation=True,
    enable_sensitive_data=False,
    enable_console_exporters=True,
)
```

Or load from a `.env` file:

```python
settings = ObservabilitySettings(env_file_path=".env.test", env_file_encoding="utf-8")
```

## Disabling everything

`ENABLE_INSTRUMENTATION` defaults to `True` in code (the env var `ENABLE_INSTRUMENTATION=false` overrides it). Call `disable_instrumentation()` to apply a **sticky disable** that survives any subsequent `enable_instrumentation()` call, library auto-setup, or direct property write — nothing re-enables it until you pass `force=True`:

```python
from agent_framework.observability import disable_instrumentation, enable_instrumentation

# Hard-disable — sticky, survives auto-setup and library integrations.
disable_instrumentation()

# Any subsequent call to enable_instrumentation() without force=True is a no-op.
enable_instrumentation()                    # ignored — logs a warning
print(OBSERVABILITY_SETTINGS.ENABLED)       # False

# To re-enable after a sticky disable, use force=True.
enable_instrumentation(force=True)
print(OBSERVABILITY_SETTINGS.ENABLED)       # True
```

`disable_instrumentation()` does **not** tear down already-configured OTel providers or in-flight spans; it only gates future span/metric/log emission from agent-framework code paths. Third-party instrumentations continue unaffected.

Use `enable_instrumentation(enable_sensitive_data=True)` to switch on both at once (equivalent to setting both env vars):

```python
enable_instrumentation(enable_sensitive_data=True)
```

### Scoped disable in tests

A common test pattern: disable globally, run the agent, re-enable.

```python
import pytest
from agent_framework.observability import disable_instrumentation, enable_instrumentation

@pytest.fixture(autouse=True)
def no_telemetry():
    disable_instrumentation()
    yield
    enable_instrumentation(force=True)
```

---
title: "Microsoft Agent Framework (Python) — Observability & Telemetry"
description: "OpenTelemetry traces, metrics, and logs emitted by agent-framework-core 1.1.0. Enable instrumentation, ship to Azure Monitor / OTLP / console, or the VS Code AI Toolkit."
framework: microsoft-agent-framework
language: python
---

# Observability & Telemetry — Python

`agent-framework-core` emits OpenTelemetry signals following the **GenAI semantic conventions**. Every agent run, model call, tool invocation, workflow executor, and edge group is a span; every chat completion emits a duration histogram and a token-usage histogram. Nothing is exported by default — you opt in either by calling a helper, setting one env var, or wiring your own OTel providers.

Verified against `agent-framework-core==1.1.0` (`agent_framework.observability`).

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

## Custom spans around business logic

You can still wrap your own code in spans; they slot neatly into the framework's tree.

```python
from opentelemetry import trace

tracer = trace.get_tracer("myapp.customer")

with tracer.start_as_current_span("process_order", attributes={"order.id": order_id}):
    response = await agent.run(f"Draft reply for order {order_id}")
```

For agent-level wrapping use `AgentMiddleware` instead — see the [middleware page](./microsoft_agent_framework_python_middleware/#emitting-opentelemetry-spans).

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

## Disabling everything

`ENABLE_INSTRUMENTATION` defaults to `false`. Without either env var or `enable_instrumentation()`, no signals are produced — even if OTel SDK is configured elsewhere.

To hard-disable mid-run, unset `OBSERVABILITY_SETTINGS.enable_instrumentation = False` from the same module — but prefer controlling it at startup.

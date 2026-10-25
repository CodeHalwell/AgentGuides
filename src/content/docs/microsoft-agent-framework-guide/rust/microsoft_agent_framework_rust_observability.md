---
title: "Microsoft Agent Framework (Rust) — Observability"
description: "OpenTelemetry for agent-framework-rs: GenAI-convention tracing spans for agents, model calls and tools, their attributes, sensitive-data capture and semantic-convention versions, the otel-metrics histograms, the otel-export OTLP pipeline, composing with your own tracing subscriber, and wiring to an OpenTelemetry Collector."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Observability"
  order: 13
---

The crate instruments agents, model calls and tool calls with [`tracing`](https://docs.rs/tracing) spans named and attributed according to the OpenTelemetry GenAI semantic conventions, and can record the GenAI metrics through the `opentelemetry` API. Nothing is exported by default: spans cost almost nothing until a subscriber is installed, and the metrics are no-ops until a meter provider is. That is why there is no switch like Python's `ENABLE_OTEL`.

| Feature | Adds | Pulls in |
| --- | --- | --- |
| (none) | Spans, visible to any `tracing` subscriber | `tracing` only |
| `otel-metrics` | Three GenAI histograms | The `opentelemetry` API crate, no SDK |
| `otel-export` | `observability::export::OtelExport`, a ready OTLP/HTTP pipeline for spans and metrics; implies `otel-metrics` | `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`; building it needs a C toolchain and `cmake` (for `aws-lc-rs`) |

The examples were compiled against %%version=agent-framework-rs%% with `otel-export` and run against a stand-in for the OpenAI API, which also accepted the OTLP requests.

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["otel-export"] }
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

## Spans

Three kinds of span are emitted, each with the human-readable name in `otel.name` (the `tracing` name is the bare operation, because `tracing` needs a literal):

| Span | `otel.name` | Emitted by |
| --- | --- | --- |
| `invoke_agent` | `invoke_agent {agent name or id}` | `Agent::run`, and `run_stream` when the run goes through agent middleware |
| `chat` | `chat {model}` | `ObservableChatClient`, once per model call |
| `execute_tool` | `execute_tool {tool}` | The agent's tool loop, once per local tool call |

The agent and tool spans need no setup. The `chat` span needs the chat client to be wrapped: `ObservableChatClient::new(client, "openai")`, where the second argument is the provider name reported on the span. The builder puts the agent's tool loop around the client you give it, so a wrapped client produces one `chat` span per model call inside the run, and a run with one tool call has the shape `invoke_agent` > `chat`, `execute_tool`, `chat`:

```rust
use agent_framework::observability::{ObservabilityConfig, ObservableChatClient};
use agent_framework::prelude::*;
use serde_json::json;
use tracing_subscriber::fmt::format::FmtSpan;

#[tokio::main]
async fn main() -> Result<()> {
    // Any tracing subscriber shows the spans; this one prints each on close.
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .with_ansi(false)
        .init();

    // Content capture is off unless asked for. Use one config for the chat
    // spans and the agent's tool spans, so a trace reports one convention.
    let config = ObservabilityConfig { enable_sensitive_data: true, ..ObservabilityConfig::from_env() };
    let client = ObservableChatClient::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?, "openai")
        .with_observability_config(config.clone());

    let weather = FunctionTool::new(
        "get_weather",
        "Current weather for a city.",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        |args| async move { Ok(json!({ "city": args["city"], "sky": "clear" })) },
    );
    let agent = Agent::builder(client)
        .name("forecaster")
        .tool(weather.into_definition())
        .observability_config(config)
        .build();
    println!("{}", agent.run_once("Weather in Paris?").await?.text());
    Ok(())
}
```

A plain `run_stream` with no agent middleware does not create the `invoke_agent` span; its model calls are still observed by `ObservableChatClient`, whose span stays open until the stream is drained and then records the finish reason and usage. Calls to other agents (A2A, Copilot Studio), embedding clients, workflows and executors have no spans of their own: workflow telemetry is not yet ported, and there is an `embeddings` operation name in `observability::op` but nothing emits it.

| Span | Attributes |
| --- | --- |
| `chat` | `gen_ai.operation.name`, `gen_ai.provider.name` (or `gen_ai.system`), `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.reasoning.output_tokens`, `gen_ai.request.temperature`, `top_p`, `max_tokens`, `seed`, `frequency_penalty`, `presence_penalty`, `stop_sequences`, `gen_ai.conversation.id`, `server.address`, `server.port`; on failure `error.type`, `otel.status_code`, `otel.status_message` |
| `chat`, with content capture | `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions` |
| `invoke_agent` | `gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `error.type` |
| `execute_tool` | `gen_ai.operation.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.description`, `gen_ai.tool.type` (always `function`), `error.type` |
| `execute_tool`, with content capture | `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |

Request parameters and usage are recorded only when present. The attribute names are constants in `observability::attr`, and the span constructors (`chat_span`, `agent_span`, `tool_span_ex`) and recorders (`record_request`, `record_response`, `record_error`) are public, for a custom chat client or agent that should report the same way. `error_type(&err)` gives the `error.type` value for an `Error`.

### Checking spans in tests

The spans are ordinary `tracing` data, so a small `Layer` can collect them for assertions without any OpenTelemetry machinery. This one records every span's fields and prints the names that were set:

```rust
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use agent_framework::observability::ObservableChatClient;
use agent_framework::prelude::*;
use serde_json::json;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

type Fields = BTreeMap<String, String>;

/// Collects every span's recorded fields, for assertions in tests.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<(String, Fields)>>>);

struct Collect<'a>(&'a mut Fields);
impl Visit for Collect<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_string(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = Fields::new();
        attrs.record(&mut Collect(&mut fields));
        ctx.span(id).unwrap().extensions_mut().insert(fields);
    }
    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(fields) = ctx.span(id).unwrap().extensions_mut().get_mut::<Fields>() {
            values.record(&mut Collect(fields));
        }
    }
    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        let fields = span.extensions().get::<Fields>().cloned().unwrap_or_default();
        self.0.lock().unwrap().push((span.name().to_string(), fields));
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let capture = Capture::default();
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));

    let client = ObservableChatClient::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?, "openai");
    let tool = FunctionTool::new(
        "lookup",
        "Look something up.",
        json!({ "type": "object", "properties": { "query": { "type": "string" } } }),
        |_| async move { Ok(json!("found it")) },
    );
    let agent = Agent::builder(client).name("helper").tool(tool.into_definition()).build();
    agent.run_once("Find the returns policy").await?;

    for (name, fields) in capture.0.lock().unwrap().iter() {
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        println!("{name}: {}", keys.join(", "));
    }
    Ok(())
}
```

`tracing::subscriber::set_default` scopes the subscriber to the current thread, which suits a test; a multi-threaded runtime may poll the agent on other threads, so use a current-thread runtime (the `#[tokio::test]` default) or `set_global_default` when capturing in tests.

## Sensitive data and convention versions

Two environment variables, read by `ObservabilityConfig::from_env()`, control what is recorded:

| Variable | Default | Effect |
| --- | --- | --- |
| `ENABLE_SENSITIVE_DATA` | off | Content capture: messages, system instructions, tool definitions, tool arguments and results. `1`, `true`, `yes`, `on` (any case) enable it |
| `OTEL_SEMCONV_STABILITY_OPT_IN` | unset, meaning the latest experimental conventions | A comma-separated list; when it is set and does not contain `gen_ai_latest_experimental`, the v1.36.0 baseline is used |
| `OTEL_SERVICE_NAME` | `agent_framework` | Read by `ObservabilityConfig::otel_service_name()` for your own resource; `OtelExport` takes the service name as an argument instead |

The baseline differs in two ways: the provider is reported as `gen_ai.system` rather than `gen_ai.provider.name`, and the attributes added since (cache and reasoning token counts, `gen_ai.tool.definitions`, and the tool call arguments and result) are not emitted. Leaving the variable unset opts in to the newer conventions, which, as upstream documents, differs from OpenTelemetry's own default.

The two places that capture content read the environment differently, which is easy to trip over:

- The agent's tool loop reads `ENABLE_SENSITIVE_DATA` itself, unless you pass `AgentBuilder::observability_config(config)`. The `invoke_agent` span follows the same config, though it records only error text from it.
- `ObservableChatClient::new` always starts with content capture **off**, whatever the environment says (it does read the convention version). Use `ObservableChatClient::from_env(client, name)` to honour `ENABLE_SENSITIVE_DATA`, `with_content_capture(true)` to force it, or `with_observability_config(config)` to set both values.

So with only `ENABLE_SENSITIVE_DATA=true` set and `new`, tool arguments and results appear on `execute_tool` spans but no messages appear on `chat` spans. Passing one `ObservabilityConfig` to both, as the first example does, keeps one trace consistent. Captured content includes whatever users and tools put in it, so enable it in development, or where the trace store is as protected as the data.

## Metrics

With `otel-metrics`, three histograms are recorded through `opentelemetry::global::meter("agent_framework")`:

| Name | Unit | Recorded by | Attributes |
| --- | --- | --- | --- |
| `gen_ai.client.token.usage` | `tokens` | `ObservableChatClient`, once per token type on a successful call | `gen_ai.operation.name`, provider, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.token.type` (`input` or `output`) |
| `gen_ai.client.operation.duration` | `s` | `ObservableChatClient`, every call, failures included | as above without the token type; `error.type` on failure |
| `agent_framework.function.invocation.duration` | `s` | The agent's tool loop, every tool call | `agent_framework.function.name`, `error.type` on failure |

The bucket boundaries match upstream's (`observability::metrics::TOKEN_USAGE_BUCKET_BOUNDARIES`, powers of four from 1 to 67,108,864; `OPERATION_DURATION_BUCKET_BOUNDARIES`, doubling from 0.01 s to 81.92 s). Recording a failed call's duration matters: without it the histogram has no error latency and no error rate, and the slowest observations, timeouts, would be missing from the tail.

The instruments are created once, on first use, against whichever meter provider is installed at that moment. Install your provider before any instrumented call, or the metrics stay on the no-op provider for the life of the process, silently. To use your own SDK setup instead of `otel-export`, add `opentelemetry` and `opentelemetry_sdk` at the versions the crate uses (0.32), build an `SdkMeterProvider` with your reader and exporter, and call `opentelemetry::global::set_meter_provider` at startup; the crate's `otel_metrics` example does this with an in-memory exporter. That route needs those two dependencies, which is why it is not shown as a compiled block here.

## Exporting with `otel-export`

`OtelExport::new(service_name)` builds an OTLP exporter, a tracer provider, a meter provider (installed globally by `build()`) and the `tracing` bridge, wired to the conventions above:

```rust
use std::time::Duration;

use agent_framework::observability::export::OtelExport;
use agent_framework::observability::ObservableChatClient;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Build first: the GenAI metric instruments bind to whichever meter
    // provider exists the first time they are used.
    let pipeline = OtelExport::new("travel-agent") // service.name
        // Defaults to OTEL_EXPORTER_OTLP_ENDPOINT, then http://localhost:4318.
        .with_endpoint(std::env::var("OTLP_ENDPOINT").unwrap_or_else(|_| "http://localhost:4318".into()))
        .with_metric_interval(Duration::from_secs(10))
        .build()?;
    pipeline.install()?; // claims the global tracing subscriber; RUST_LOG filters

    let client = ObservableChatClient::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?, "openai");
    let agent = Agent::builder(client).name("assistant").build();
    for question in ["Hello", "What can you do?"] {
        println!("{}", agent.run_once(question).await?.text());
    }

    // Flush both providers. A failure means queued telemetry was dropped.
    pipeline.shutdown()?;
    Ok(())
}
```

| Method | Default | Effect |
| --- | --- | --- |
| `with_endpoint(url)` | `OTEL_EXPORTER_OTLP_ENDPOINT`, else `http://localhost:4318` | The OTLP/HTTP base URL; `/v1/traces` and `/v1/metrics` are appended. A URL that already ends in a signal path, or has a query string, is handled |
| `with_traces(bool)` | `true` | Export spans; with `false`, `install()` does nothing, leaving the global subscriber free |
| `with_metrics(bool)` | `true` | Export the histograms |
| `with_metric_interval(d)` | 60 s | Push interval for metrics |

`build()` returns an `OtelPipeline`. `install()` claims the global `tracing` subscriber with an `EnvFilter` from `RUST_LOG` (default `info`) and fails if one is already set; `tracing_layer()` gives the layer to add to a subscriber you build yourself; `shutdown()` flushes and stops both providers and returns any error rather than logging it, since the logging may be what is shutting down. Call `shutdown()` before exiting: spans are batched, and dropping the pipeline loses whatever is queued. Keep the pipeline alive as long as its layer is installed.

The transport is OTLP over HTTP with protobuf, not gRPC, on a blocking HTTP client in the exporters' background threads (an async client would need a tokio reactor there). Only `OTEL_EXPORTER_OTLP_ENDPOINT` is read; the signal-specific endpoint variables, headers, sampling and batching settings are not, so a collector that needs an authentication header has to be reached through a local collector, or you build the pipeline yourself.

### Composing with your own subscriber

When the application already has a subscriber stack (log formatting, filters, other layers), build the pipeline and add its layer instead of calling `install()`:

```rust
use agent_framework::observability::export::OtelExport;
use agent_framework::observability::ObservableChatClient;
use agent_framework::prelude::*;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let pipeline = OtelExport::new("travel-agent").build()?;

    // Your own subscriber stack, with the pipeline's layer added to it.
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_ansi(false).compact())
        .with(pipeline.tracing_layer())
        .try_init()?;

    tracing::info!("service starting");
    let client = ObservableChatClient::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?, "openai");
    let agent = Agent::builder(client).name("assistant").build();
    println!("{}", agent.run_once("Hello").await?.text());

    // Keep the pipeline alive while the subscriber is installed; shut down last.
    pipeline.shutdown()?;
    Ok(())
}
```

## Wiring to a collector

Point the pipeline at an [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) and let the collector fan out to backends. There is no Rust exporter for Azure Monitor or Application Insights, so the collector's Azure Monitor exporter is the supported route to them. A minimal collector configuration that receives OTLP/HTTP on the default port and sends traces and metrics to Azure Monitor and to the console:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  azuremonitor:
    connection_string: ${env:APPLICATIONINSIGHTS_CONNECTION_STRING}
  debug:
    verbosity: basic
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [azuremonitor, debug]
    metrics:
      receivers: [otlp]
      exporters: [azuremonitor, debug]
```

The Azure Monitor exporter is in the collector's contrib distribution (`otel/opentelemetry-collector-contrib`). For a local trace viewer, Jaeger and the Aspire dashboard both accept OTLP directly, so `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` (Jaeger) or the dashboard's OTLP/HTTP port is enough. The Azure SDK's own client spans (token requests from the Azure credentials, for example) do not join this pipeline yet: the bridge crate for them targets an older `opentelemetry` release, and mixing the two would install into a second, unconnected global provider.

## Compared with Python

Python's `configure_otel_providers` and `ObservabilitySettings` correspond to `OtelExport` and `ObservabilityConfig`; `ENABLE_OTEL` has no equivalent, for the reason given at the top. The span and metric names and attributes match upstream, so dashboards built for the Python SDK read Rust services too. Not ported: workflow and executor spans, embedding spans, the Azure Monitor exporter, and Python's environment-driven exporter selection (such as `OTEL_EXPORTER_OTLP_PROTOCOL`).

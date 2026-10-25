---
title: "Microsoft Agent Framework (Go) — Observability"
description: "OpenTelemetry tracing and metrics for Go agents and workflows: the otelprovider middleware and its GenAI spans and metrics, OTLP export, workflow telemetry, embedding telemetry, structured logging with slog, custom tracers, testing telemetry and the User-Agent feature mask."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Observability"
  order: 13
---

The Go SDK has three observability hooks, and they are independent of each other:

| What | How | Emits |
| --- | --- | --- |
| Agents, tools and model calls | `otelprovider.NewMiddleware` in `agent.Config.Middlewares` | Spans and metrics following the OpenTelemetry GenAI semantic conventions |
| Workflows | `Builder.WithTelemetry(opentelemetry.New(cfg), workflow.TelemetryOptions{})` | Spans for building, running, executors, edges and messages |
| Embedding generators | `otelprovider.WrapEmbeddingGenerator` | GenAI embedding spans and metrics |
| Logs | `agent.Config.Logger` (`*slog.Logger`) and `toolautocall.Config.Logger` | Debug-level run and tool-call records |

All of them use the **global** OpenTelemetry providers (`otel.GetTracerProvider()`, `otel.GetMeterProvider()`). Set those up once at start-up and shut them down on exit; the SDK never creates or closes them. Nothing is exported until you install an SDK provider, so the instrumentation costs almost nothing when telemetry is off.

Every program here was compiled with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%% and run. The agents call a local stand-in for the OpenAI API (set `OPENAI_BASE_URL` and `OPENAI_API_KEY`), so model names, token counts and replies come from the stand-in.

## Agent tracing and metrics

`otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: ...})` returns an `agent.Middleware`. Put it in `agent.Config.Middlewares` and every run of that agent produces:

- an `invoke_agent <name>` span for the run;
- a `chat <model>` client span for each model call (OpenAI, Foundry Responses, Anthropic and Gemini adapters), as children of the agent span;
- an `execute_tool <tool>` span for each function-tool call, also a child of the agent span;
- the `gen_ai.client.operation.duration` and `gen_ai.client.token.usage` histograms.

The program below records into in-memory exporters and prints what it got:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/provider/otelprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type weatherIn struct {
	City string `json:"city" jsonschema:"the city name"`
}

func main() {
	ctx := context.Background()

	spans := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spans))
	defer func() { _ = tp.Shutdown(ctx) }()
	otel.SetTracerProvider(tp)

	metrics := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(metrics))
	defer func() { _ = mp.Shutdown(ctx) }()
	otel.SetMeterProvider(mp)

	weather := functool.MustNew(functool.Config{Name: "get_weather", Description: "Current weather for a city."},
		func(_ context.Context, in weatherIn) (string, error) { return "14°C in " + in.City, nil })

	forecaster := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai-small%%",
		Config: agent.Config{
			Name:  "Forecaster",
			Tools: []tool.Tool{weather},
			Middlewares: []agent.Middleware{
				otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: "weather-service"}),
			},
		},
	})
	if _, err := forecaster.RunText(ctx, "What's the weather in Leeds?").Collect(); err != nil {
		log.Fatal(err)
	}

	// Spans end child-first, so the agent span is last.
	for _, s := range spans.GetSpans() {
		var keys []string
		for _, kv := range s.Attributes {
			keys = append(keys, string(kv.Key))
		}
		sort.Strings(keys)
		parent := "root"
		if s.Parent.IsValid() {
			parent = "child"
		}
		fmt.Printf("%-24s %-8s %-5s scope=%s\n    %s\n", s.Name, s.SpanKind, parent,
			s.InstrumentationScope.Name, strings.Join(keys, " "))
	}

	var rm metricdata.ResourceMetrics
	if err := metrics.Collect(ctx, &rm); err != nil {
		log.Fatal(err)
	}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			fmt.Printf("metric %s (%s)\n", m.Name, m.Unit)
		}
	}
}
```

Output:

```text
chat %%model=openai-small%%          client   child scope=weather-service
    gen_ai.operation.name gen_ai.provider.name gen_ai.request.model gen_ai.response.finish_reasons gen_ai.response.id gen_ai.response.model gen_ai.usage.input_tokens gen_ai.usage.output_tokens
execute_tool get_weather internal child scope=weather-service
    gen_ai.operation.name gen_ai.tool.call.id gen_ai.tool.description gen_ai.tool.name gen_ai.tool.type
chat %%model=openai-small%%          client   child scope=weather-service
    gen_ai.operation.name gen_ai.provider.name gen_ai.request.model gen_ai.response.finish_reasons gen_ai.response.id gen_ai.response.model gen_ai.usage.input_tokens gen_ai.usage.output_tokens
invoke_agent Forecaster  internal root  scope=weather-service
    gen_ai.agent.id gen_ai.agent.name gen_ai.operation.name gen_ai.provider.name gen_ai.response.finish_reasons gen_ai.response.id gen_ai.usage.input_tokens gen_ai.usage.output_tokens
metric gen_ai.client.operation.duration (s)
metric gen_ai.client.token.usage ({token})
```

A tool round trip is two model calls with a tool call between them, so the trace has three children under one `invoke_agent` span. The agent span's token counts are the sum over its model calls.

| Span | Kind | Notable attributes |
| --- | --- | --- |
| `invoke_agent <agent name>` | internal | `gen_ai.operation.name=invoke_agent`, `gen_ai.provider.name`, `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.agent.description` (when set), `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens` / `output_tokens` |
| `chat <model>` | client | `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`, finish reasons, token usage (including `gen_ai.usage.cache_read.input_tokens` when the provider reports it) |
| `execute_tool <tool name>` | internal | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type=function`, `gen_ai.tool.description` |

| Metric | Unit | Attributes |
| --- | --- | --- |
| `gen_ai.client.operation.duration` | seconds | `gen_ai.operation.name` (`chat`, `invoke_agent`), provider, request and response model, `error.type` on failure |
| `gen_ai.client.token.usage` | `{token}` | As above plus `gen_ai.token.type` (`input` or `output`) |

Behaviour worth knowing:

- **No content is recorded.** Prompts, completions, tool arguments and tool results never appear on these spans, and there is no switch to turn them on in v%%version=github.com/microsoft/agent-framework-go%%. (The workflow and embedding instrumentation do have opt-in switches; see below.)
- **Errors** set the span status to error and record the error. The original error is returned to you unchanged.
- **`MiddlewareConfig.DisableModelTracing`** drops the `chat` spans and their metrics but keeps the agent and tool spans, useful when the provider SDK already emits its own HTTP or model spans.
- **`SourceName`** is the instrumentation scope for both the tracer and the meter. An empty one falls back to the SDK's default scope; set it to your service or component name.
- **Nested agents** (an agent used as a tool through `agenttool`, or agents inside a workflow) produce spans only if they have the middleware themselves; they do not inherit it from the caller. Their spans nest under the caller's span through the context.
- **Retries** inside the provider SDK stay within one `chat` span.
- **Position.** `agent.Config.Middlewares` wrap the whole run, outside history, context providers and the provider's automatic tool calling, so the tool and model spans are captured whatever else is in the list.

## Exporting with OTLP

In a real service you export to a collector, Azure Monitor, Jaeger or similar. The usual set-up uses the OTLP exporters, which read the standard `OTEL_EXPORTER_OTLP_*` environment variables, plus a resource that names the service:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/provider/otelprovider"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
)

// setupTelemetry installs global tracer and meter providers that export over
// OTLP/HTTP (OTEL_EXPORTER_OTLP_ENDPOINT, default http://localhost:4318).
func setupTelemetry(ctx context.Context, service, version string) (func(context.Context) error, error) {
	res, err := resource.New(ctx,
		resource.WithAttributes(semconv.ServiceName(service), semconv.ServiceVersion(version)),
		resource.WithFromEnv(), // OTEL_RESOURCE_ATTRIBUTES, OTEL_SERVICE_NAME
		resource.WithHost(),
	)
	if err != nil {
		return nil, err
	}

	traceExp, err := otlptracehttp.New(ctx)
	if err != nil {
		return nil, err
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithBatcher(traceExp),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(1.0))),
	)

	metricExp, err := otlpmetrichttp.New(ctx)
	if err != nil {
		return nil, err
	}
	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp, sdkmetric.WithInterval(30*time.Second))),
	)

	otel.SetTracerProvider(tp)
	otel.SetMeterProvider(mp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

	return func(ctx context.Context) error {
		// Shutdown flushes anything still buffered.
		return errors.Join(tp.Shutdown(ctx), mp.Shutdown(ctx))
	}, nil
}

func main() {
	ctx := context.Background()
	shutdown, err := setupTelemetry(ctx, "support-bot", "1.4.0")
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := shutdown(ctx); err != nil {
			log.Printf("telemetry shutdown: %v", err)
		}
	}()

	bot := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai-small%%",
		Config: agent.Config{
			Name:        "SupportBot",
			Middlewares: []agent.Middleware{otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: "support-bot"})},
		},
	})
	resp, err := bot.RunText(ctx, "How do I reset my password?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Run it with `OTEL_EXPORTER_OTLP_ENDPOINT` pointing at your collector. Against a local endpoint that accepts OTLP/HTTP, the program printed the agent's reply and the endpoint received one `/v1/traces` and one `/v1/metrics` request at shutdown.

For Azure Monitor (Application Insights), send OTLP to an OpenTelemetry Collector or the Azure Monitor agent, which forward to your workspace; there is no Azure Monitor exporter in the Go SDK itself.

**Trace propagation.** Agent spans join whatever span is in the `context.Context` you pass to `Run`. In an HTTP service, wrap the handler with `otelhttp.NewHandler` (from `go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp`) so incoming requests carry the caller's trace, and the agent's spans become part of it. For outgoing model calls to show up as HTTP spans too, give the provider client an instrumented transport, for example `option.WithHTTPClient(&http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)})` for the OpenAI client.

## Workflow telemetry

Workflows are instrumented separately, per workflow, on the builder. `opentelemetry.New` (package `workflow/observability/opentelemetry`) creates a tracer from the global provider; `workflow.TelemetryOptions` chooses what to record:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	workflowotel "github.com/microsoft/agent-framework-go/workflow/observability/opentelemetry"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func main() {
	ctx := context.Background()
	spans := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spans))
	defer func() { _ = tp.Shutdown(ctx) }()
	otel.SetTracerProvider(tp)

	upper := workflow.NewExecutor("upper", strings.ToUpper).Bind()
	exclaim := workflow.NewExecutor("exclaim", func(s string) string { return s + "!" }).Bind()
	wf, err := workflow.NewBuilder(upper).AddEdge(upper, exclaim).WithOutputFrom(exclaim).
		WithName("shout").
		WithTelemetry(workflowotel.New(workflowotel.Config{SourceName: "shout-service"}),
			workflow.TelemetryOptions{EnableSensitiveData: true}). // records inputs and outputs
		Build()
	if err != nil {
		log.Fatal(err)
	}

	run, err := inproc.Default.Run(ctx, wf, "hello")
	if err != nil {
		log.Fatal(err)
	}
	for range run.NewEvents() {
	}
	_ = run.Close(ctx)

	for _, s := range spans.GetSpans() {
		var attrs []string
		for _, kv := range s.Attributes {
			switch kv.Key {
			case "executor.input", "executor.output", "message.content", "executor.id", "message.source_id", "message.target_id":
				attrs = append(attrs, fmt.Sprintf("%s=%s", kv.Key, kv.Value.Emit()))
			}
		}
		var events []string
		for _, e := range s.Events {
			events = append(events, e.Name)
		}
		fmt.Printf("%-24s %s %v\n", s.Name, strings.Join(attrs, " "), events)
	}
}
```

Output:

```text
workflow.build            [build.started build.validation_completed build.completed]
edge_group.process       message.target_id=upper[] []
edge_group.process       message.source_id=upper message.target_id=exclaim []
message.send             message.source_id=upper message.content="HELLO" []
executor.process upper   executor.id=upper executor.input="hello" executor.output="HELLO" []
message.send             message.source_id=exclaim message.content="HELLO!" []
executor.process exclaim executor.id=exclaim executor.input="HELLO" executor.output="HELLO!" []
workflow_invoke           [workflow.started workflow.completed]
workflow.session          [session.started session.completed]
```

| Span | Recorded | Turn off with |
| --- | --- | --- |
| `workflow.build` | When `Build` runs; `workflow.id`, `workflow.name`, `workflow.description`, and the whole graph as JSON in `workflow.definition` | `DisableWorkflowBuild` |
| `workflow.session` | One per run session; `session.id` | |
| `workflow_invoke` | One per input-to-halt cycle, with `workflow.started` / `workflow.completed` / `workflow.error` events | `DisableWorkflowRun` |
| `executor.process <id>` | Each handler call; `executor.id`, `executor.type`, `message.type` | `DisableExecutorProcess` |
| `edge_group.process` | Each delivery decision; `edge_group.type`, source and target, `edge_group.delivered`, `edge_group.delivery_status` | `DisableEdgeGroupProcess` |
| `message.send` (producer) | Each sent message; `message.source_id` | `DisableMessageSend` |

`EnableSensitiveData` adds `executor.input`, `executor.output` and `message.content`, serialised as JSON. It is off by default. Turn it on only where the data may be stored in your tracing backend: agent messages in a workflow often contain user data.

`workflow.build` is recorded when the workflow is built, before the first run, and is not a parent of the run spans. Agents inside a workflow emit their own spans if they have the `otelprovider` middleware; they nest under the `executor.process` span of the executor that runs them. A workflow wrapped with `agentworkflow.NewAgent` can also take the middleware in its `AgentConfig.Config`, which adds an `invoke_agent` span around each turn.

### A custom tracer

The workflow engine depends only on the small `workflow/observability.Tracer` interface (`Start(ctx, name, options)` returning a `Span`, and `ExtractTraceContext`). Implement it to send workflow activity somewhere other than OpenTelemetry, for example to structured logs:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	"github.com/microsoft/agent-framework-go/workflow/observability"
)

type logTracer struct{}

type logSpan struct {
	name  string
	start time.Time
}

func (logTracer) Start(ctx context.Context, name string, _ observability.SpanOptions) (context.Context, observability.Span) {
	return ctx, &logSpan{name: name, start: time.Now()}
}

func (logTracer) ExtractTraceContext(context.Context) map[string]string { return nil }

func (s *logSpan) End() {
	if strings.HasPrefix(s.name, "executor.process") {
		fmt.Printf("%s finished\n", s.name)
	}
}
func (s *logSpan) AddEvent(string, ...observability.Attribute) {}
func (s *logSpan) SetAttributes(...observability.Attribute)    {}
func (s *logSpan) RecordError(err error)                       { fmt.Printf("%s failed: %v\n", s.name, err) }
func (s *logSpan) SetError(msg string)                         { fmt.Printf("%s error: %s\n", s.name, msg) }

func main() {
	a := workflow.NewExecutor("parse", strings.TrimSpace).Bind()
	b := workflow.NewExecutor("store", strings.ToLower).Bind()
	wf, err := workflow.NewBuilder(a).AddEdge(a, b).WithOutputFrom(b).
		WithTelemetry(logTracer{}, workflow.TelemetryOptions{}).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	run, err := inproc.Default.Run(context.Background(), wf, "  Hello ")
	if err != nil {
		log.Fatal(err)
	}
	for range run.NewEvents() {
	}
}
```

Output:

```text
executor.process parse finished
executor.process store finished
```

## Embedding telemetry

`otelprovider.WrapEmbeddingGenerator(inner, cfg)` wraps any `embedding.Generator` with `embeddings` client spans plus duration and input-token metrics:

```go
gen := otelprovider.WrapEmbeddingGenerator(
	openaiprovider.NewEmbeddingGenerator(openai.NewClient(), openaiprovider.EmbeddingGeneratorConfig{Model: "text-embedding-3-small"}),
	otelprovider.EmbeddingGeneratorConfig{
		SourceName:     "search-indexer",
		ProviderName:   "openai",
		ServerAddress:  "api.openai.com",
		ServerPort:     443,
		DefaultModelID: "text-embedding-3-small",
	},
)
```

| Field | Effect |
| --- | --- |
| `SourceName` | Instrumentation scope; empty means `github.com/microsoft/agent-framework-go` |
| `ProviderName`, `ServerAddress`, `ServerPort` | Span attributes only; they do not change routing |
| `DefaultModelID`, `DefaultModelDimensions` | Used when a request does not name a model or dimension count |
| `EnableSensitiveData` | Adds extra request and response properties (never raw inputs or vectors). Nil reads `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` once, when the wrapper is built. |
| `Logger` | Receives failures as warnings |

The wrapper keeps `io.Closer` only if the inner generator implements it, and never shuts down the global providers.

## Logging

`agent.Config.Logger` takes an `*slog.Logger` and logs each run at **debug** level: `run invoked`, `run completed` with its duration, and the agent's ID and name. The automatic tool-calling middleware has its own `Logger` in `toolautocall.Config` (set it through the provider config's `ToolAutoCall` field) and logs `calling function` and `function call completed`, with durations.

`LogSensitiveData: true`, on either, adds the input messages, each streamed update, tool arguments and tool results. That is useful when debugging locally, but it puts prompts and user data into your logs, so leave it off in production. `DisableRunLogs` switches off the agent's run records while keeping the logger for other components. With `LogSensitiveData` on, a run against the stand-in logs this (time stamps and agent IDs removed):

```text
level=DEBUG msg="run invoked" messages="[weather in Leeds]" opts="[{Tool:0x3f7c09032b40} {Session:0x3f7c09002480} true {}]" agentName=Forecaster middleware=logger
level=DEBUG msg="calling function" funcName=get_weather arguments="{\"city\": \"Paris\"}" middleware=autocall
level=DEBUG msg="function call completed" funcName=get_weather duration=19.222µs result=14C middleware=autocall
level=DEBUG msg="run received update" update="" agentName=Forecaster middleware=logger
level=DEBUG msg="run received update" update="" agentName=Forecaster middleware=logger
level=DEBUG msg="run received update" update="Here is what the tool returned: 14C" agentName=Forecaster middleware=logger
level=DEBUG msg="run completed" duration=2.295867ms agentName=Forecaster middleware=logger
```

Without it, only `run invoked`, `calling function`, `function call completed` and `run completed` appear, with names and durations. Because everything is at debug level, a handler at the default info level shows nothing.

The workflow builder logs graph warnings (dead-end executors, self-loops) through `slog.Default()`, and the a2a-go server logs through `a2asrv.WithLogger`. The AG-UI SSE client logs each connection at info level through logrus; lower it with `logrus.SetLevel(logrus.WarnLevel)`.

To correlate logs with traces, use a `slog.Handler` that adds the trace and span IDs from the context, such as the OpenTelemetry slog bridge (`go.opentelemetry.io/contrib/bridges/otelslog`), which sends log records through the OpenTelemetry logs pipeline.

## Testing your telemetry

`tracetest.NewInMemoryExporter` and `sdkmetric.NewManualReader`, as in the first example, make telemetry assertable in an ordinary `go test`. Combined with the fake provider on the [testing page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_testing/#asserting-on-telemetry), you can check that a run produced an `invoke_agent` span with the right name, that tool spans exist, and that no attribute contains user text.

## Usage telemetry in the User-Agent header

Separately from OpenTelemetry, the OpenAI and Foundry adapters add the framework's identity and a feature-usage bitmask to the `User-Agent` header, for example `agent-framework-go/v%%version=github.com/microsoft/agent-framework-go%% ... (feat=v1.15)`. The mask records which framework features have been used in the process; it holds no identifiers, prompts or data. It is sent only on HTTPS requests to approved Azure hosts (`openai.azure.com`, `cognitiveservices.azure.com`, `services.ai.azure.com`, `inference.ai.azure.com`), never to api.openai.com, other providers or custom gateways. Two environment variables, read once per process, turn it off:

| Variable | Effect |
| --- | --- |
| `AGENT_FRAMEWORK_FEATURE_MASK_DISABLED=1` | Drop the feature mask, keep the SDK identity |
| `AGENT_FRAMEWORK_USER_AGENT_DISABLED=1` | Drop the framework's whole `User-Agent` contribution |

## Compared with Python and .NET

| Python / .NET | Go |
| --- | --- |
| `configure_otel_providers()` / `UseOpenTelemetry()` on the agent builder | Configure the global providers yourself; add `otelprovider.NewMiddleware` per agent |
| `enable_sensitive_data` for agent spans | Not available; agent, model and tool spans never carry content |
| Workflow spans | `Builder.WithTelemetry`, with per-span switches and `EnableSensitiveData` |
| Azure Monitor helpers | Not available; export OTLP through a collector |
| DevUI trace viewer | Not available |

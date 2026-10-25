---
title: "Microsoft Agent Framework (.NET) — Observability"
description: "OpenTelemetry for .NET agents and workflows: OpenTelemetryAgent and UseOpenTelemetry, the GenAI-convention spans and attributes, sensitive data, chat-client instrumentation, gen_ai metrics, workflow spans and WorkflowTelemetryOptions, logging with UseLogging, OTLP, Aspire dashboard and Azure Monitor export, and your own spans and metrics."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Observability"
  order: 13
---

The framework instruments agents, model calls, tool calls and workflows with `System.Diagnostics` activities and meters named after the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (version 1.37 at the time of writing). Nothing is recorded until you opt in on two sides: wrap the agent (or build the workflow) with OpenTelemetry, and subscribe to its source name in your tracer and meter providers. With either missing you get no telemetry and no error.

Every C# block on this page compiles against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%) with OpenTelemetry 1.20, and was run. The span listings below were captured with the in-memory exporter from runs against a local stand-in for the OpenAI API; the ASP.NET Core example was run with its OTLP exporter pointed at the same stand-in to check that traces and metrics are sent.

```bash
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%
dotnet add package OpenTelemetry.Extensions.Hosting
dotnet add package OpenTelemetry.Exporter.OpenTelemetryProtocol
dotnet add package Azure.Monitor.OpenTelemetry.Exporter       # or Azure.Monitor.OpenTelemetry.AspNetCore
```

## How it differs from Python

Python enables telemetry with `configure_otel_providers()` or environment variables (`ENABLE_INSTRUMENTATION`, `ENABLE_SENSITIVE_DATA`) and instruments every agent once switched on. .NET has no global switch: you wrap each agent with `UseOpenTelemetry`, call `WithOpenTelemetry` on each workflow, and configure providers with the standard OpenTelemetry .NET SDK, which also handles exporters, sampling and resource attributes. The span names and `gen_ai.*` attributes are the same in both, so dashboards built for one work for the other.

## Tracing an agent

`agent.AsBuilder().UseOpenTelemetry(sourceName, configure)` wraps an agent in an `OpenTelemetryAgent`. Subscribe to the same name with `AddSource` and `AddMeter`. If you leave the name out it is `OpenTelemetryAgent.DefaultSourceName`, currently `Experimental.Microsoft.Agents.AI`; pass your own so that a rename in a later release does not silently stop your traces.

```csharp
using System.ComponentModel;
using System.Diagnostics;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Trace;

public static class WeatherTools
{
    [Description("Get the weather for a city.")]
    public static string GetWeather([Description("The city")] string city) => $"{city}: 18°C, light rain.";
}

public static class AgentTracing
{
    public const string SourceName = "Contoso.Agents";

    public static async Task RunAsync()
    {
        // Collect in memory so the example can print what was recorded; in an app, use an exporter instead.
        List<Activity> spans = [];
        List<Metric> metrics = [];
        using TracerProvider tracing = Sdk.CreateTracerProviderBuilder()
            .AddSource(SourceName)
            .AddInMemoryExporter(spans)
            .Build();
        using MeterProvider metering = Sdk.CreateMeterProviderBuilder()
            .AddMeter(SourceName)
            .AddInMemoryExporter(metrics)
            .Build();

        AIAgent agent = new ChatClientAgent(Models.ChatClient(), "You report the weather.", name: "weather",
                tools: [AIFunctionFactory.Create(WeatherTools.GetWeather)])
            .AsBuilder()
            .UseOpenTelemetry(SourceName, otel => otel.EnableSensitiveData = false)   // the default: no message content
            .Build();

        await agent.RunAsync("What's the weather in Leeds?");

        tracing.ForceFlush();
        foreach (Activity span in spans)
        {
            Console.WriteLine($"{span.DisplayName,-30} parent={(span.Parent?.DisplayName ?? span.ParentSpanId.ToString()),-20} " +
                string.Join(" ", span.TagObjects.Where(t => t.Key.StartsWith("gen_ai.") && t.Key != "gen_ai.tool.definitions").Select(t => $"{t.Key}={t.Value}")));
        }

        metering.ForceFlush();
        Console.WriteLine("metrics: " + string.Join(", ", metrics.Select(m => m.Name).Distinct()));
    }
}
```

That run (an agent with one tool, so two model calls) records:

```text
chat gpt-...                  parent=invoke_agent weather(...)  gen_ai.operation.name=chat gen_ai.provider.name=openai gen_ai.response.finish_reasons=["tool_calls"] gen_ai.usage.input_tokens=12 gen_ai.usage.output_tokens=8
execute_tool GetWeather       parent=invoke_agent weather(...)  gen_ai.operation.name=execute_tool gen_ai.tool.name=GetWeather gen_ai.tool.call.id=call_1 gen_ai.tool.type=function
chat gpt-...                  parent=invoke_agent weather(...)  gen_ai.response.finish_reasons=["stop"] gen_ai.usage.input_tokens=12 gen_ai.usage.output_tokens=8
invoke_agent weather(...)     (root)                            gen_ai.agent.name=weather gen_ai.agent.id=... gen_ai.usage.input_tokens=24 gen_ai.usage.output_tokens=16
metrics: gen_ai.client.token.usage, gen_ai.client.operation.duration
```

| Span | Created by | Key attributes |
| --- | --- | --- |
| `invoke_agent {name}({id})` | `OpenTelemetryAgent` | `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`, token totals for the whole run |
| `chat {model}` | `OpenTelemetryChatClient` (Microsoft.Extensions.AI) | `gen_ai.request.model`, `gen_ai.response.id`, `gen_ai.usage.input_tokens` / `output_tokens` for that call, `gen_ai.tool.definitions` (names only), `server.address` |
| `execute_tool {tool}` | The function-invocation loop | `gen_ai.tool.name`, `gen_ai.tool.description`, `gen_ai.tool.call.id`, `gen_ai.tool.type` |

The `chat` and `execute_tool` spans appear without any extra code because `OpenTelemetryAgent` wraps a `ChatClientAgent`'s chat client for each call (`autoWireChatClient`, on by default). If the chat client is already instrumented, as in the ASP.NET Core example below, it does not wrap it twice. For agents that are not `ChatClientAgent`s (A2A, Copilot Studio, a workflow as an agent), only the `invoke_agent` span is created; instrument whatever they call separately.

The metrics are the GenAI client histograms `gen_ai.client.operation.duration` (seconds) and `gen_ai.client.token.usage` (tokens, tagged by `gen_ai.token.type`), recorded under the same source name.

### Sensitive data

By default the spans hold metadata only: names, IDs, models, token counts and durations. `EnableSensitiveData = true` adds the content as well:

| Span | Added attributes |
| --- | --- |
| `invoke_agent`, `chat` | `gen_ai.input.messages`, `gen_ai.output.messages`; `chat` also gets `gen_ai.system_instructions` and full tool schemas in `gen_ai.tool.definitions` |
| `execute_tool` | `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |

That is prompts, replies, tool arguments and tool results in your trace store, so treat the backend as holding personal data. Turn it on per environment rather than in code, for example `otel.EnableSensitiveData = builder.Environment.IsDevelopment()`, and remember that it can be set on the chat client's `OpenTelemetryChatClient` separately.

## Tracing workflows

`WorkflowBuilder.WithOpenTelemetry(configure, activitySource)` instruments a workflow. Without an `ActivitySource`, spans go to `Microsoft.Agents.AI.Workflows`, so subscribe to that name (or to your own source if you pass one).

```csharp
using System.Diagnostics;
using Microsoft.Agents.AI.Workflows;
using OpenTelemetry;
using OpenTelemetry.Trace;

public static class WorkflowTracing
{
    private static readonly ActivitySource Source = new("Contoso.Workflows");

    public static async Task RunAsync()
    {
        List<Activity> spans = [];
        using TracerProvider tracing = Sdk.CreateTracerProviderBuilder()
            .AddSource(Source.Name)
            .AddInMemoryExporter(spans)
            .Build();

        ExecutorBinding parse = ((Func<string, int>)int.Parse).BindAsExecutor("parse");
        ExecutorBinding square = ((Func<int, int>)(n => n * n)).BindAsExecutor("square");

        Workflow workflow = new WorkflowBuilder(parse)
            .AddEdge(parse, square)
            .WithOutputFrom(square)
            .WithName("square-numbers")
            // Without an ActivitySource the spans go to "Microsoft.Agents.AI.Workflows".
            .WithOpenTelemetry(options => options.EnableSensitiveData = true, Source)   // records executor inputs and outputs
            .Build();

        await using (Run run = await InProcessExecution.RunAsync(workflow, "12")) { }

        tracing.ForceFlush();
        foreach (Activity span in spans)
        {
            Console.WriteLine($"{span.DisplayName,-26} " + string.Join(" ", span.Tags.Select(t => $"{t.Key}={t.Value}")));
        }
    }
}
```

| Span | One per | Attributes |
| --- | --- | --- |
| `workflow.build` | `Build()` call | `workflow.id`, `workflow.name`, `workflow.definition` (the graph as JSON) |
| `workflow.session` | Run | `workflow.id`, `session.id` |
| `workflow_invoke` | Execution of the run's supersteps | `workflow.id`, `session.id` |
| `executor.process {id}` | Message handled | `executor.id`, `executor.type`, `message.type`; with sensitive data, `executor.input` and `executor.output` |
| `message.send` | Message sent | `message.source_id`; with sensitive data, `message.content` |
| `edge_group.process` | Edge group delivery | `edge_group.type`, `message.source_id`, `message.target_id`, `edge_group.delivery_status` |

Errors set `error.type` and `error.message` on the span that failed. `WorkflowTelemetryOptions` turns individual span kinds off (`DisableWorkflowBuild`, `DisableWorkflowRun`, `DisableExecutorProcess`, `DisableEdgeGroupProcess`, `DisableMessageSend`), which is worth doing for busy fan-out graphs where `message.send` and `edge_group.process` dominate the trace. Agents inside a workflow are traced only if you wrap them with `UseOpenTelemetry` before adding them; their `invoke_agent` spans then nest under `executor.process`. Declarative workflows take the same settings through `DeclarativeWorkflowOptions.ConfigureTelemetry` and `TelemetryActivitySource`.

## ASP.NET Core and OTLP

In an app, configure everything through `AddOpenTelemetry()`. `UseOtlpExporter()` sends traces, metrics and logs to whatever `OTEL_EXPORTER_OTLP_ENDPOINT` names: the .NET Aspire dashboard, Jaeger, Grafana, or an OpenTelemetry Collector in front of anything else.

```csharp
// Program.cs: OpenTelemetry for an ASP.NET Core agent service, exported over OTLP (Aspire dashboard, Jaeger, a collector ...).
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using OpenTelemetry;
using OpenTelemetry.Logs;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

const string SourceName = "Contoso.Agents";

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenTelemetry()
    .ConfigureResource(resource => resource.AddService("support-agents"))
    .WithTracing(tracing => tracing
        .AddSource(SourceName)                          // agent and chat spans
        .AddSource("Microsoft.Agents.AI.Workflows")     // workflow spans, if you run workflows
        .AddAspNetCoreInstrumentation()                 // the incoming HTTP request is the root span
        .AddHttpClientInstrumentation())                // the outgoing call to the model
    .WithMetrics(metrics => metrics
        .AddMeter(SourceName)                           // gen_ai.client.operation.duration, gen_ai.client.token.usage
        .AddAspNetCoreInstrumentation())
    .WithLogging()
    .UseOtlpExporter();                                 // OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_PROTOCOL

// Instrument the chat client too, so model calls get their own spans with token counts.
builder.Services.AddChatClient(Models.ChatClient())
    .UseOpenTelemetry(sourceName: SourceName);

builder.AddAIAgent("support", (sp, name) =>
    new ChatClientAgent(sp.GetRequiredService<IChatClient>(), "You are a support assistant.", name: name)
        .AsBuilder()
        .UseOpenTelemetry(SourceName)
        .UseLogging(sp.GetRequiredService<ILoggerFactory>())
        .Build());

var app = builder.Build();

app.MapPost("/ask", async (AskRequest request, [FromKeyedServices("support")] AIAgent agent, CancellationToken ct) =>
    (await agent.RunAsync(request.Question, cancellationToken: ct)).Text);

app.Run();

public record AskRequest(string Question);
```

To try it locally, run the Aspire dashboard (`docker run --rm -it -p 18888:18888 -p 4317:18889 mcr.microsoft.com/dotnet/aspire-dashboard`), set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317`, and open `http://localhost:18888`. A request then appears as one trace: the ASP.NET Core request span, `invoke_agent support(...)`, the `chat` spans with the outgoing HTTP call beneath them, and any `execute_tool` spans. The default OTLP protocol is gRPC; set `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` for collectors that only accept HTTP.

The [hosting](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/) endpoints run inside your ASP.NET Core pipeline, so an agent served over the OpenAI, A2A or AG-UI endpoints is traced the same way, under the incoming request.

## Azure Monitor

For Application Insights, add the Azure Monitor exporters to the same providers. In ASP.NET Core, `Azure.Monitor.OpenTelemetry.AspNetCore`'s `builder.Services.AddOpenTelemetry().UseAzureMonitor()` replaces the explicit exporters; keep the `AddSource` and `AddMeter` calls for the agent and workflow sources, because the distro does not know about them.

```csharp
using Azure.Monitor.OpenTelemetry.Exporter;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

// Console apps and workers: build the providers yourself and export to Application Insights.
public static class AzureMonitorTelemetry
{
    public static (TracerProvider Tracing, MeterProvider Metrics) Start(string sourceName)
    {
        string connectionString = Environment.GetEnvironmentVariable("APPLICATIONINSIGHTS_CONNECTION_STRING")!;
        ResourceBuilder resource = ResourceBuilder.CreateDefault().AddService("batch-agents");

        TracerProvider tracing = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(resource)
            .AddSource(sourceName, "Microsoft.Agents.AI.Workflows")
            .AddAzureMonitorTraceExporter(o => o.ConnectionString = connectionString)
            .Build();

        MeterProvider metrics = Sdk.CreateMeterProviderBuilder()
            .SetResourceBuilder(resource)
            .AddMeter(sourceName)
            .AddAzureMonitorMetricExporter(o => o.ConnectionString = connectionString)
            .Build();

        return (tracing, metrics);   // dispose both on shutdown to flush
    }
}
// In ASP.NET Core, the Azure.Monitor.OpenTelemetry.AspNetCore package does the same with
// builder.Services.AddOpenTelemetry().UseAzureMonitor() plus the AddSource/AddMeter calls above.
```

The spans then appear in Application Insights' transaction search with their `gen_ai.*` attributes as custom properties. If the exporter authenticates with Entra ID (`o.Credential = new DefaultAzureCredential()`) instead of a connection-string key, the app's identity needs the Monitoring Metrics Publisher role on the Application Insights resource.

## Logging

`UseLogging(loggerFactory)` wraps an agent in a `LoggingAgent` that logs each run at `Debug` (start and completion) and, at `Trace`, the messages, options and responses serialised with its `JsonSerializerOptions`. Because `Trace` includes content, keep that level off in production. With `WithLogging()` in the OpenTelemetry set-up, these logs are exported with the trace and span IDs of the run, so they line up with the spans. The chat client has the matching `UseLogging()` on `ChatClientBuilder`.

## Your own spans and metrics

Business-level telemetry is ordinary `System.Diagnostics`. A span you start around an agent call becomes the parent of the framework's spans, so a dashboard can group runs by ticket, tenant or customer:

```csharp
using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Agents.AI;

// Your own spans and metrics are plain System.Diagnostics. Token usage is on every AgentResponse.
public static class AgentMetrics
{
    public const string Name = "Contoso.Agents.Business";
    private static readonly ActivitySource Source = new(Name);
    private static readonly Meter Meter = new(Name);
    private static readonly Counter<long> Tokens = Meter.CreateCounter<long>("contoso.agent.tokens", unit: "{token}");
    private static readonly Histogram<double> Latency = Meter.CreateHistogram<double>("contoso.agent.latency", unit: "ms");

    public static async Task<AgentResponse> RunForTenantAsync(AIAgent agent, string input, string tenant, CancellationToken ct = default)
    {
        // A parent span that groups the agent's own spans under a business operation.
        using Activity? activity = Source.StartActivity("answer-ticket");
        activity?.SetTag("contoso.tenant", tenant);

        var tags = new TagList { { "agent", agent.Name }, { "tenant", tenant } };
        long start = Stopwatch.GetTimestamp();
        AgentResponse response = await agent.RunAsync(input, cancellationToken: ct);

        Latency.Record(Stopwatch.GetElapsedTime(start).TotalMilliseconds, tags);
        Tokens.Add(response.Usage?.TotalTokenCount ?? 0, tags);
        activity?.SetTag("contoso.tokens", response.Usage?.TotalTokenCount);
        return response;
    }
}
// Subscribe with .AddSource(AgentMetrics.Name) and .AddMeter(AgentMetrics.Name).
```

Keep tag values low in cardinality (tenant and agent name, not user ID or prompt text) so that metric series stay bounded.

## Checking telemetry in tests

The in-memory exporter (package `OpenTelemetry.Exporter.InMemory`) used in the first example is also the simplest way to assert on telemetry in a unit test: build a `TracerProvider` with `AddInMemoryExporter(list)`, run the agent against a fake chat client, and check the spans' names and tags. The [testing page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/#asserting-on-telemetry) has a complete test.

## Production checklist

- Use your own source names, subscribe to them and to `Microsoft.Agents.AI.Workflows`, and check in a test that spans arrive.
- Leave `EnableSensitiveData` off in production, or send those traces to a store with access controls and a retention limit.
- Instrument the chat client in DI (`AddChatClient(...).UseOpenTelemetry()`) so model calls made outside agents are traced too.
- Set sampling on the tracer provider (`SetSampler(new ParentBasedSampler(new TraceIdRatioBasedSampler(0.1)))`) once volume grows; metrics are not sampled, so token totals stay accurate.
- Alert on `gen_ai.client.token.usage` per agent and on `error.type` counts, not only on HTTP errors.

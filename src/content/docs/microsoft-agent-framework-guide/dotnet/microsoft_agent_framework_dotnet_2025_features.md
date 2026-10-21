---
title: "Microsoft Agent Framework - .NET Feature Tour"
description: "Tour of the .NET A2A, workflow, checkpointing, human-in-the-loop, declarative agent, OpenTelemetry, content-safety and dependency-injection features, with compiled examples."
framework: microsoft-agent-framework
language: dotnet
---

# Microsoft Agent Framework - .NET Feature Tour

**First written:** November 2025, for the October 2025 public preview
**Verified against:** `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%). Every C# block on this page was compiled against that release.
**Target audience:** .NET/C# developers
**.NET version:** 8.0+

The preview-era version of this page described `Microsoft.Agents.AI.Graphs`, `.Config`, `.Telemetry` and `.Safety` namespaces (`AgentGraph`, `HITLConfig`, `A2AProtocolAdapter`, `AgentTelemetry`, `ContentSafetyConfig`). They never shipped. The shipped equivalents are:

| Feature | Package / namespace | Key types |
| --- | --- | --- |
| A2A client | `Microsoft.Agents.AI.A2A` (preview) | `A2ACardResolver.GetAIAgentAsync()`, `A2AAgent` |
| A2A server | `Microsoft.Agents.AI.Hosting.A2A.AspNetCore` (preview) | `AddA2AServer`, `MapA2AJsonRpc` |
| Graph workflows | `Microsoft.Agents.AI.Workflows` | `WorkflowBuilder`, `Executor<TIn,TOut>`, `InProcessExecution` |
| Checkpointing | `Microsoft.Agents.AI.Workflows` | `CheckpointManager`, `CheckpointInfo` |
| Human-in-the-loop | `Microsoft.Agents.AI.Workflows` | `RequestPort`, `RequestInfoEvent` |
| Declarative agents | `Microsoft.Agents.AI.Declarative` (rc) | `ChatClientPromptAgentFactory`, `CreateFromYamlAsync` |
| Declarative workflows | `Microsoft.Agents.AI.Workflows.Declarative` | `DeclarativeWorkflowBuilder` |
| OpenTelemetry | `Microsoft.Agents.AI` | `.AsBuilder().UseOpenTelemetry()` |
| Content safety | your middleware over `Azure.AI.ContentSafety` | `.AsBuilder().Use(...)` |
| Hosting / DI | `Microsoft.Agents.AI.Hosting` (preview) | `AddAIAgent`, `AddWorkflow` |

The examples share one helper for creating a chat client:

```csharp
// Models.cs
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Extensions.AI;

public static class Models
{
    public static IChatClient ChatClient() =>
        new AzureOpenAIClient(
                new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!),
                new DefaultAzureCredential())
            .GetChatClient(Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "gpt-4o-mini")
            .AsIChatClient();
}
```

## Table of Contents

1. [Agent2Agent (A2A) Protocol - .NET](#agent2agent-a2a-protocol---net)
2. [Graph-Based Workflows - .NET](#graph-based-workflows---net)
3. [Declarative Agents with .NET](#declarative-agents-with-net)
4. [OpenTelemetry Integration - .NET](#opentelemetry-integration---net)
5. [Content Safety - .NET](#content-safety---net)
6. [Dependency Injection Patterns](#dependency-injection-patterns)
7. [Best Practices](#best-practices)

---

## Agent2Agent (A2A) Protocol - .NET

[A2A](https://a2a-protocol.org/) lets agents on different frameworks call each other over HTTP. In the framework a remote A2A agent is just another `AIAgent`, and any `AIAgent` can be published as an A2A endpoint.

```bash
dotnet add package Microsoft.Agents.AI.A2A --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.A2A.AspNetCore --prerelease
```

### Calling a remote A2A agent

The client resolves the remote agent's card (`/.well-known/agent-card.json`) and wraps it as an `AIAgent`. Authentication is whatever the remote agent's card asks for; supply it through the `HttpClient`.

```csharp
using A2A;
using Microsoft.Agents.AI;

public static class A2AClientExample
{
    public static async Task CallRemoteAgentAsync()
    {
        using var http = new HttpClient();
        // e.g. http.DefaultRequestHeaders.Authorization = new("Bearer", token);

        var resolver = new A2ACardResolver(new Uri("https://api.partner.com/agents/analytics/"), http);
        AIAgent analytics = await resolver.GetAIAgentAsync(http);

        AgentSession session = await analytics.CreateSessionAsync();   // maps to an A2A context id
        AgentResponse response = await analytics.RunAsync("Analyse Q3 sales trends", session);
        Console.WriteLine(response.Text);
    }
}
```

Because it is an `AIAgent`, a remote A2A agent can sit in a workflow, be wrapped with middleware, or be handed to another agent as a tool with `analytics.AsAIFunction()`.

### Exposing an agent over A2A

```csharp
// Program.cs (ASP.NET Core)
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

AIAgent salesAgent = new ChatClientAgent(
    Models.ChatClient(),
    instructions: "You are a helpful sales data agent.",
    name: "SalesAgent",
    description: "Answers questions about sales figures.");

builder.AddA2AServer(salesAgent);

var app = builder.Build();
app.MapA2AJsonRpc(salesAgent, "/agents/sales");   // JSON-RPC binding; MapA2AHttpJson for HTTP+JSON
app.Run();
```

Put the endpoint behind your normal ASP.NET Core authentication (`.RequireAuthorization()`).

### Type-safe exchanges

A2A carries messages, not typed RPC. For a typed contract, serialise the request into the message and ask the remote side for JSON; locally, `RunAsync<T>` gives you the same safety against an in-process agent.

```csharp
using System.Text.Json;
using Microsoft.Agents.AI;

public record SalesQuery(string QueryType, string DateRange, string? Region = null, string Currency = "USD");

public record SalesResponse(decimal TotalSales, string Period, string Region, string Currency, Dictionary<string, decimal> Breakdown);

public static class TypedSales
{
    public static async Task<SalesResponse> QueryAsync(AIAgent salesAgent, SalesQuery query)
    {
        AgentResponse<SalesResponse> response = await salesAgent.RunAsync<SalesResponse>(
            $"Answer this query as JSON: {JsonSerializer.Serialize(query)}");
        return response.Result;
    }
}
```

---

## Graph-Based Workflows - .NET

Workflows are directed graphs of **executors**. An executor is a class deriving from `Executor<TInput, TOutput>` (or `Executor<TInput>`), a plain function bound with `BindAsExecutor`, or an `AIAgent`. Edges carry typed messages between them; the workflow runs in supersteps and can checkpoint after each one.

```bash
dotnet add package Microsoft.Agents.AI.Workflows --version %%version=Microsoft.Agents.AI%%
```

### Building a workflow

```csharp
using Microsoft.Agents.AI.Workflows;

public record Document(string Title, string Body);

public sealed class DraftExecutor() : Executor<string, Document>("draft")
{
    public override ValueTask<Document> HandleAsync(string topic, IWorkflowContext context, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(new Document(topic, $"A first draft about {topic}."));
}

public sealed class ReviewExecutor() : Executor<Document, string>("review")
{
    public override async ValueTask<string> HandleAsync(Document doc, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        await context.QueueStateUpdateAsync("reviewed-title", doc.Title, cancellationToken: cancellationToken);
        return $"{doc.Title}: {doc.Body} (reviewed)";
    }
}

public static class ContentPipeline
{
    public static Workflow Build()
    {
        var draft = new DraftExecutor();
        var review = new ReviewExecutor();

        return new WorkflowBuilder(draft)
            .AddEdge(draft, review)
            .WithOutputFrom(review)
            .WithName("ContentPipeline")
            .Build();
    }

    public static async Task RunAsync()
    {
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(Build(), "agent frameworks");
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case ExecutorCompletedEvent done:
                    Console.WriteLine($"[{done.ExecutorId}] completed");
                    break;
                case WorkflowOutputEvent output:
                    Console.WriteLine(output.As<string>());
                    break;
            }
        }
    }
}
```

Agents slot in the same way: `new WorkflowBuilder(researcher).AddEdge(researcher, writer)` where both are `AIAgent`s. Send a `TurnToken` after starting the run so agent executors begin their turn (see the [.NET recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-8-explicit-multi-agent-graph-with-workflowbuilder)).

### Conditional routing

`AddEdge<T>(source, target, condition)` only delivers a message when the predicate holds. For many branches, `AddSwitch` reads more clearly.

```csharp
using Microsoft.Agents.AI.Workflows;

public record Claim(string Id, decimal Amount);

public static class ClaimsRouting
{
    public static Workflow Build()
    {
        var intake = ((Func<string, Claim>)(text => new Claim(Guid.NewGuid().ToString("N")[..8], decimal.Parse(text))))
            .BindAsExecutor("intake");
        var autoApprove = ((Func<Claim, string>)(c => $"Claim {c.Id} auto-approved")).BindAsExecutor("auto-approve");
        var manualReview = ((Func<Claim, string>)(c => $"Claim {c.Id} sent for manual review")).BindAsExecutor("manual-review");

        return new WorkflowBuilder(intake)
            .AddSwitch(intake, s => s
                .AddCase<Claim>(c => c!.Amount < 1_000m, autoApprove)
                .WithDefault(manualReview))
            .WithOutputFrom(autoApprove, manualReview)
            .Build();
    }
}
```

### Parallel execution

Fan-out sends one message to several executors in the same superstep; a fan-in barrier waits for all of them. For agents only, `AgentWorkflowBuilder.BuildConcurrent(agents)` does this for you.

```csharp
using Microsoft.Agents.AI.Workflows;

public static class ParallelChecks
{
    public static Workflow Build()
    {
        var start = ((Func<string, string>)(text => text)).BindAsExecutor("start");
        var spelling = ((Func<string, string>)(t => $"spelling: ok ({t.Length} chars)")).BindAsExecutor("spelling");
        var tone = ((Func<string, string>)(t => "tone: neutral")).BindAsExecutor("tone");

        // Collects one result per branch, then emits the combined report.
        var report = new AggregatingExecutor<string, string>(
            "report", (acc, next) => acc is null ? next : $"{acc}; {next}");

        return new WorkflowBuilder(start)
            .AddFanOutEdge(start, [spelling, tone])
            .AddFanInBarrierEdge([spelling, tone], report)
            .WithOutputFrom(report)
            .Build();
    }
}
```

### Checkpointing and recovery

Pass a `CheckpointManager` when you start the run and the workflow checkpoints at the end of every superstep. `CheckpointManager.CreateInMemory()` is for tests; `CheckpointManager.CreateJson(new FileSystemJsonCheckpointStore(dir))` or the Cosmos DB store (`Microsoft.Agents.AI.CosmosNoSql`) persist across processes.

```csharp
using Microsoft.Agents.AI.Workflows;
using Microsoft.Agents.AI.Workflows.Checkpointing;

public static class CheckpointDemo
{
    public static async Task RunAsync()
    {
        var store = new FileSystemJsonCheckpointStore(new DirectoryInfo("./checkpoints"));
        CheckpointManager checkpoints = CheckpointManager.CreateJson(store);
        Workflow workflow = ContentPipeline.Build();

        CheckpointInfo? last = null;
        await using (StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, "durable workflows", checkpoints))
        {
            await foreach (WorkflowEvent evt in run.WatchStreamAsync())
            {
                if (evt is SuperStepCompletedEvent step && step.CompletionInfo?.Checkpoint is { } cp)
                {
                    last = cp;   // persist cp.SessionId / cp.CheckpointId somewhere durable
                }
            }
        }

        // Later, possibly in another process: resume from the saved checkpoint.
        if (last is not null)
        {
            await using StreamingRun resumed = await InProcessExecution.ResumeStreamingAsync(workflow, last, checkpoints);
            await foreach (WorkflowEvent evt in resumed.WatchStreamAsync())
            {
                if (evt is WorkflowOutputEvent output) Console.WriteLine(output.As<string>());
            }
        }
    }
}
```

### Human-in-the-loop

A `RequestPort` is a node that leaves the workflow: messages sent to it surface as `RequestInfoEvent`s, and the workflow waits until you send back a response. Combined with checkpointing, the wait can last as long as you like.

```csharp
using Microsoft.Agents.AI.Workflows;

public record ApprovalRequest(string Content);
public record ApprovalDecision(bool Approved, string? Comment = null);

public static class ApprovalWorkflow
{
    public static Workflow Build()
    {
        var draft = ((Func<string, ApprovalRequest>)(topic => new ApprovalRequest($"Draft post about {topic}")))
            .BindAsExecutor("draft");
        var approval = RequestPort.Create<ApprovalRequest, ApprovalDecision>("human-approval");
        var publish = ((Func<ApprovalDecision, string>)(d => d.Approved ? "Published." : $"Rejected: {d.Comment}"))
            .BindAsExecutor("publish");

        return new WorkflowBuilder(draft)
            .AddEdge(draft, approval)
            .AddEdge(approval, publish)
            .WithOutputFrom(publish)
            .Build();
    }

    public static async Task RunAsync()
    {
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(Build(), "release notes");
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case RequestInfoEvent request when request.Request.TryGetDataAs<ApprovalRequest>(out var ask):
                    Console.Write($"Approve '{ask.Content}'? [y/N] ");
                    bool ok = Console.ReadLine()?.Trim().ToLowerInvariant() == "y";
                    await run.SendResponseAsync(request.Request.CreateResponse(new ApprovalDecision(ok, ok ? null : "Not now")));
                    break;
                case WorkflowOutputEvent output:
                    Console.WriteLine(output.As<string>());
                    break;
            }
        }
    }
}
```

For tool-level approval inside a single agent (rather than a workflow step), wrap the tool in `ApprovalRequiredAIFunction`; see the [.NET comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#tool-approval).

---

## Declarative Agents with .NET

`Microsoft.Agents.AI.Declarative` (release candidate) builds agents from a YAML definition, so instructions, model options and output schema can live in configuration rather than code.

```bash
dotnet add package Microsoft.Agents.AI.Declarative --prerelease
```

### YAML definition

```yaml
# agents/customer-service.yaml
kind: Prompt
name: CustomerServiceAgent
description: Handles customer enquiries
instructions: |
  You are a helpful customer service agent.
  Always be polite and professional.
model:
  options:
    temperature: 0.7
    topP: 0.95
outputSchema:
  properties:
    category:
      type: string
      required: true
      description: billing, technical or general
    answer:
      type: string
      required: true
      description: The reply to the customer.
```

### Loading the definition

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class DeclarativeAgents
{
    public static async Task<AIAgent> LoadAsync(IChatClient chatClient, string path)
    {
        var factory = new ChatClientPromptAgentFactory(chatClient);
        string yaml = await File.ReadAllTextAsync(path);
        return await factory.CreateFromYamlAsync(yaml);
    }

    public static async Task RunAsync()
    {
        AIAgent agent = await LoadAsync(Models.ChatClient(), "agents/customer-service.yaml");
        Console.WriteLine(await agent.RunAsync("My invoice is wrong."));
    }
}
```

The model is the chat client you pass in; the YAML sets the options. Workflows have a declarative form too (`DeclarativeWorkflowBuilder.Build<T>("workflow.yaml", options)` in `Microsoft.Agents.AI.Workflows.Declarative`), covered on the [graph workflows and declarative agents page](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/).

### Strongly-typed configuration with the options pattern

When you would rather keep agent settings in `appsettings.json`, bind them with the ordinary .NET options pattern and build the agent from them. Nothing here is framework-specific, which is the point.

```json
{
  "Agents": {
    "Support": {
      "Name": "support",
      "Instructions": "You are a helpful customer service agent.",
      "Temperature": 0.7,
      "MaxOutputTokens": 800
    }
  }
}
```

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

public sealed class AgentSettings
{
    public required string Name { get; init; }
    public required string Instructions { get; init; }
    public float Temperature { get; init; } = 0.7f;
    public int MaxOutputTokens { get; init; } = 1000;
}

public static class ConfiguredAgentRegistration
{
    public static void AddConfiguredSupportAgent(this IHostApplicationBuilder builder)
    {
        builder.Services.AddOptions<AgentSettings>()
            .Bind(builder.Configuration.GetSection("Agents:Support"))
            .ValidateDataAnnotations()
            .ValidateOnStart();

        builder.AddAIAgent("support", (sp, name) =>
        {
            AgentSettings settings = sp.GetRequiredService<IOptions<AgentSettings>>().Value;
            return new ChatClientAgent(sp.GetRequiredService<IChatClient>(), new ChatClientAgentOptions
            {
                Name = name,
                ChatOptions = new ChatOptions
                {
                    Instructions = settings.Instructions,
                    Temperature = settings.Temperature,
                    MaxOutputTokens = settings.MaxOutputTokens,
                },
            });
        });
    }
}
```

---

## OpenTelemetry Integration - .NET

### Observability set-up

Wrap the agent with `UseOpenTelemetry(sourceName)` and subscribe to that source. Workflows have the matching `WorkflowBuilder.WithOpenTelemetry()`. The spans follow the OpenTelemetry GenAI conventions (`invoke_agent`, `chat`, `execute_tool`).

```csharp
using Azure.Monitor.OpenTelemetry.Exporter;
using Microsoft.Agents.AI;
using OpenTelemetry;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

public static class Observability
{
    public const string SourceName = "Contoso.Agents";

    public static TracerProvider StartTracing() =>
        Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(ResourceBuilder.CreateDefault().AddService("enterprise-agents"))
            .AddSource(SourceName)
            .AddOtlpExporter()                                   // OTEL_EXPORTER_OTLP_ENDPOINT
            .AddAzureMonitorTraceExporter(o =>
                o.ConnectionString = Environment.GetEnvironmentVariable("APPLICATIONINSIGHTS_CONNECTION_STRING"))
            .Build();

    public static AIAgent Instrument(AIAgent agent) =>
        agent.AsBuilder()
            .UseOpenTelemetry(SourceName, o => o.EnableSensitiveData = false)   // no prompt text in spans
            .Build();
}
```

In an ASP.NET Core app use `builder.Services.AddOpenTelemetry().WithTracing(t => t.AddSource(Observability.SourceName))` from `OpenTelemetry.Extensions.Hosting` instead of building the provider by hand.

### Custom metrics

Your own metrics are plain `System.Diagnostics.Metrics`. Token usage is on every `AgentResponse`.

```csharp
using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Agents.AI;

public sealed class AgentMetrics
{
    private static readonly Meter Meter = new("Contoso.Agents.Business");
    private static readonly Counter<long> Requests = Meter.CreateCounter<long>("agent.requests");
    private static readonly Histogram<double> Latency = Meter.CreateHistogram<double>("agent.latency", unit: "ms");
    private static readonly Counter<long> Tokens = Meter.CreateCounter<long>("agent.tokens");

    public static async Task<AgentResponse> RunMeasuredAsync(AIAgent agent, string input, string tenant)
    {
        var tags = new TagList { { "agent", agent.Name }, { "tenant", tenant } };
        var stopwatch = Stopwatch.StartNew();

        AgentResponse response = await agent.RunAsync(input);

        Requests.Add(1, tags);
        Latency.Record(stopwatch.Elapsed.TotalMilliseconds, tags);
        Tokens.Add(response.Usage?.TotalTokenCount ?? 0, tags);
        return response;
    }
}

// Export it with .AddMeter("Contoso.Agents.Business") on your MeterProvider.
```

---

## Content Safety - .NET

There is no `Microsoft.Agents.AI.Safety` namespace and no `contentSafety:` option on `ChatClientAgent`. Content safety is middleware you write over `Azure.AI.ContentSafety`, attached with `agent.AsBuilder().Use(...)`. The full input-and-output screening example is on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#net-content-safety-integration).

### A reusable safety wrapper

The middleware can also be written as a class deriving from `DelegatingAIAgent`, which is easier to test and to register in DI.

```csharp
using Azure.AI.ContentSafety;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

public sealed class ContentSafetyAgent(AIAgent inner, ContentSafetyClient safety, int maxSeverity = 2)
    : DelegatingAIAgent(inner)
{
    protected override async Task<AgentResponse> RunCoreAsync(
        IEnumerable<ChatMessage> messages, AgentSession? session, AgentRunOptions? options, CancellationToken cancellationToken)
    {
        var text = string.Join("\n", messages.Where(m => m.Role == ChatRole.User).Select(m => m.Text));
        var analysis = await safety.AnalyzeTextAsync(new AnalyzeTextOptions(text), cancellationToken);
        if (analysis.Value.CategoriesAnalysis.Any(c => c.Severity > maxSeverity))
        {
            return new AgentResponse(new ChatMessage(ChatRole.Assistant, "I can't help with that request."));
        }

        return await base.RunCoreAsync(messages, session, options, cancellationToken);
    }
}

// AIAgent safe = new ContentSafetyAgent(agent, new ContentSafetyClient(endpoint, credential));
```

Override `RunCoreStreamingAsync` as well if callers stream; otherwise streaming calls bypass the check.

### PII detection

Azure AI Language (`Azure.AI.TextAnalytics`) redacts PII; run it on user messages before the model sees them.

```csharp
using Azure.AI.TextAnalytics;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

public static class PiiRedaction
{
    public static AIAgent WithPiiRedaction(this AIAgent agent, TextAnalyticsClient language) =>
        agent.AsBuilder()
            .Use(
                runFunc: async (messages, session, options, inner, ct) =>
                {
                    var redacted = new List<ChatMessage>();
                    foreach (ChatMessage message in messages)
                    {
                        if (message.Role == ChatRole.User && !string.IsNullOrWhiteSpace(message.Text))
                        {
                            PiiEntityCollection entities = await language.RecognizePiiEntitiesAsync(message.Text, cancellationToken: ct);
                            redacted.Add(new ChatMessage(ChatRole.User, entities.RedactedText));
                        }
                        else
                        {
                            redacted.Add(message);
                        }
                    }
                    return await inner.RunAsync(redacted, session, options, ct);
                },
                runStreamingFunc: null)
            .Build();
}
```

---

## Dependency Injection Patterns

### Registering agents and workflows

`Microsoft.Agents.AI.Hosting` registers agents (and workflows) as keyed `AIAgent` services. A workflow registered with `AddWorkflow(...).AddAsAIAgent()` is callable exactly like an agent, and both can then be mapped to OpenAI-compatible, A2A or AG-UI endpoints.

```csharp
// Program.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Workflows;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddChatClient(Models.ChatClient());

builder.AddAIAgent("researcher", "You research topics thoroughly.");
builder.AddAIAgent("writer", "You turn research into a short, clear brief.");

builder.AddWorkflow("research-brief", (sp, name) =>
        AgentWorkflowBuilder.BuildSequential(
            name,
            [sp.GetRequiredKeyedService<AIAgent>("researcher"), sp.GetRequiredKeyedService<AIAgent>("writer")]))
    .AddAsAIAgent();

builder.AddOpenAIResponses();

var app = builder.Build();

app.MapPost("/brief", async (BriefRequest request, [FromKeyedServices("research-brief")] AIAgent workflow) =>
    Results.Ok(new { brief = (await workflow.RunAsync(request.Topic)).Text }));

app.MapOpenAIResponses();   // POST /v1/responses with "model": "researcher" | "writer" | "research-brief"
app.Run();

public record BriefRequest(string Topic);
```

### Lifetimes

- Register the `IChatClient` as a singleton; it owns the HTTP connection pool.
- Agents are safe to share as singletons (the default for `AddAIAgent`); keep per-user state in sessions.
- Use `ServiceLifetime.Scoped` on `AddAIAgent` only when the agent depends on scoped services (for example a per-request `DbContext` used by its tools).

---

## Best Practices

### 1. Async all the way
- Use `async`/`await` throughout; never block on `.Result` or `.Wait()`.
- Flow `CancellationToken`s from the request into `RunAsync`.

### 2. Dependency injection
- One `IChatClient` per model deployment, registered once.
- Resolve agents as `AIAgent` (keyed by name) so implementations can change without touching callers.
- Bind configuration with `IOptions<T>` and validate it on start-up.

### 3. Type safety
- Use records for workflow messages and structured output.
- Prefer `RunAsync<T>` over parsing JSON out of `response.Text`.

### 4. Error handling
- Catch `ClientResultException` (OpenAI SDK) or `RequestFailedException` (Azure SDK) around model calls; add retries with `Microsoft.Extensions.Http.Resilience` on the `HttpClient`.
- Watch for `WorkflowErrorEvent` and `ExecutorFailedEvent` in workflow streams.

### 5. Performance
- Stream (`RunStreamingAsync`) for user-facing latency.
- Use `AgentWorkflowBuilder.BuildConcurrent` or fan-out edges for independent work.
- Keep histories bounded with compaction or a chat reducer.

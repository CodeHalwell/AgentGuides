---
title: "Microsoft Agent Framework (.NET) — Feature Tour"
description: "A short tour of the .NET framework's main features beyond the single agent, with links to the pages that cover each: A2A, graph workflows, checkpointing, human-in-the-loop, orchestrations, hosting, OpenTelemetry and evaluation, plus declarative YAML agents, content-safety middleware and dependency-injection patterns with compiled examples."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Feature tour"
  order: 17
---

**First written:** November 2025, for the October 2025 public preview
**Verified against:** `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%). Every C# block on this page was compiled against that release, and the declarative agent was run against a local stand-in for the OpenAI API.
**Target audience:** .NET/C# developers
**.NET version:** 8.0+

This page started as a tour of the preview's headline features. Most of them now have their own pages; the sections below summarise each and link to the full treatment, and keep the code only for the features that are not covered elsewhere (declarative agents, content-safety middleware and dependency-injection patterns).

The preview-era version of this page described `Microsoft.Agents.AI.Graphs`, `.Config`, `.Telemetry` and `.Safety` namespaces (`AgentGraph`, `HITLConfig`, `A2AProtocolAdapter`, `AgentTelemetry`, `ContentSafetyConfig`). They never shipped. The shipped equivalents are:

| Feature | Package / namespace | Key types | Covered on |
| --- | --- | --- | --- |
| A2A client | `Microsoft.Agents.AI.A2A` (preview) | `A2ACardResolver.GetAIAgentAsync()`, `A2AAgent` | [Hosting: A2A](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#a2a) |
| A2A server | `Microsoft.Agents.AI.Hosting.A2A.AspNetCore` (preview) | `AddA2AServer`, `MapA2AJsonRpc`, `MapA2AHttpJson` | [Hosting: A2A](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#a2a) |
| Graph workflows | `Microsoft.Agents.AI.Workflows` | `WorkflowBuilder`, `Executor<TIn,TOut>`, `InProcessExecution` | [Workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) |
| Checkpointing | `Microsoft.Agents.AI.Workflows` | `CheckpointManager`, `CheckpointInfo` | [Workflows: checkpointing](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#checkpointing) |
| Human-in-the-loop | `Microsoft.Agents.AI.Workflows` | `RequestPort`, `RequestInfoEvent` | [Workflows: request ports](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#human-input-with-request-ports) |
| Orchestrations | `Microsoft.Agents.AI.Workflows` | `AgentWorkflowBuilder` | [Orchestrations](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/) |
| Declarative agents | `Microsoft.Agents.AI.Declarative` (rc) | `ChatClientPromptAgentFactory`, `CreateFromYamlAsync` | [Below](#declarative-agents-with-net) |
| Declarative workflows | `Microsoft.Agents.AI.Workflows.Declarative` | `DeclarativeWorkflowBuilder` | [Workflows: declarative](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#declarative-workflows) |
| OpenTelemetry | `Microsoft.Agents.AI`, `Microsoft.Agents.AI.Workflows` | `.AsBuilder().UseOpenTelemetry()`, `WithOpenTelemetry()` | [Observability](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/) |
| Evaluation | `Microsoft.Agents.AI` | `LocalEvaluator`, `EvalChecks`, `EvaluateAsync` | [Testing and evaluation](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/) |
| Content safety | your middleware over `Azure.AI.ContentSafety` | `.AsBuilder().Use(...)`, `DelegatingAIAgent` | [Below](#content-safety---net) |
| Hosting / DI | `Microsoft.Agents.AI.Hosting` (preview) | `AddAIAgent`, `AddWorkflow` | [Hosting](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/), and [below](#dependency-injection-patterns) |

The examples on this page share one helper for creating a chat client:

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
            .GetChatClient(Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai-small%%")
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

[A2A](https://a2a-protocol.org/) lets agents on different frameworks call each other over HTTP. In the framework a remote A2A agent is just another `AIAgent` (`await new A2ACardResolver(url, http).GetAIAgentAsync(http)`), so it can sit in a workflow, be wrapped with middleware, or become a tool with `AsAIFunction()`. Any `AIAgent` can be published with `AddA2AServer` and `MapA2AJsonRpc` / `MapA2AHttpJson`; publish its agent card as well with `MapWellKnownAgentCard`, because the server does not do it for you and clients discover agents from the card. A complete server and client, both run, are on the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#a2a). Earlier versions of this page showed a server without a card, which a card-resolving client cannot find.

For a typed contract over A2A, which carries messages rather than typed calls, serialise the request into the message and ask the remote side for JSON; against a local agent, `RunAsync<T>` gives you the same safety directly.

---

## Graph-Based Workflows - .NET

Workflows are typed graphs of executors joined by edges, run in supersteps. The [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) covers them in full, with run examples for each feature:

- [Executors](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#executors): `Executor<TIn, TOut>`, function executors, multi-handler executors and the `[MessageHandler]` source generator.
- [Edges and routing](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#edges-and-routing): conditions, switch, fan-out and fan-in, multi-selection.
- [Shared state](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#shared-state), [events and errors](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#custom-events-halting-and-errors).
- [Human input with request ports](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#human-input-with-request-ports) and [checkpointing](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#checkpointing), including resuming in another process.
- [Sub-workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#sub-workflows), [agents in workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#agents-in-workflows) and [workflows as agents](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#workflows-as-agents).

### Human-in-the-loop

A `RequestPort` is a node whose input leaves the workflow as a `RequestInfoEvent`; the caller answers with `request.CreateResponse(value)` and the run continues from the port. With a durable checkpoint store the answer can arrive days later in another process, and agent tool approvals inside workflows surface the same way. See [human input with request ports](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#human-input-with-request-ports), and [HostedWorkflowState](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#human-in-the-loop-workflows-over-http) for doing it over HTTP. For tool-level approval inside a single agent, wrap the tool in `ApprovalRequiredAIFunction`; see the [comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#tool-approval).

The prebuilt multi-agent patterns (sequential, concurrent, group chat, handoff, Magentic) are on the [orchestrations page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/). Two behaviours that catch people coming from Python: there is no superstep limit, so every cycle needs its own exit, and agent nodes only start when the run receives a `TurnToken`.

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

The model is the chat client you pass in; the YAML sets the options. With the definition above, the request carries `temperature` 0.7, `top_p` 0.95, the instructions as the system prompt and a JSON schema built from `outputSchema`, so the reply is a JSON object with `category` and `answer`. Workflows have a declarative form too, covered on the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#declarative-workflows).

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

Wrap an agent with `.AsBuilder().UseOpenTelemetry(sourceName)` and subscribe to that source with `AddSource` and `AddMeter`; workflows have `WorkflowBuilder.WithOpenTelemetry()`, whose spans go to `Microsoft.Agents.AI.Workflows` unless you pass your own `ActivitySource`. The spans follow the GenAI conventions (`invoke_agent`, `chat`, `execute_tool`) and carry token usage, and the meters record `gen_ai.client.operation.duration` and `gen_ai.client.token.usage`. Prompt text is excluded unless you set `EnableSensitiveData`. The [observability page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/) lists every span and attribute and has set-ups for OTLP, the Aspire dashboard and Azure Monitor, and your own business metrics.

---

## Content Safety - .NET

There is no `Microsoft.Agents.AI.Safety` namespace and no `contentSafety:` option on `ChatClientAgent`. Content safety is middleware you write over `Azure.AI.ContentSafety`, attached with `agent.AsBuilder().Use(...)` or written as a `DelegatingAIAgent`. The full input-and-output screening example is on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#net-content-safety-integration), and middleware in general on the [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/).

```bash
dotnet add package Azure.AI.ContentSafety
dotnet add package Azure.AI.TextAnalytics   # for PII redaction
```

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

`Microsoft.Agents.AI.Hosting` registers agents (and workflows) as keyed `AIAgent` services. A workflow registered with `AddWorkflow(...).AddAsAIAgent()` is callable exactly like an agent, and both can then be mapped to OpenAI-compatible, A2A or AG-UI endpoints (see the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/)).

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

app.MapOpenAIResponses();   // POST /v1/responses with "agent": {"name": "researcher" | "writer" | "research-brief"}
app.Run();

public record BriefRequest(string Topic);
```

The shared Responses route selects the agent with `agent.name` in the request body (or `metadata.entity_id`); the `model` field does not select anything. Earlier versions of this page said otherwise.

### Lifetimes

- Register the `IChatClient` as a singleton; it owns the HTTP connection pool.
- Agents are safe to share as singletons (the default for `AddAIAgent`); keep per-user state in sessions.
- Use `ServiceLifetime.Scoped` on `AddAIAgent` only when the agent depends on scoped services (for example a per-request `DbContext` used by its tools).
- Build workflows per run (a factory, or `AddWorkflow`'s delegate) in services that run several at once: one `Workflow` instance can only be in one run at a time.

---

## Best Practices

### 1. Async all the way
- Use `async`/`await` throughout; never block on `.Result` or `.Wait()`.
- Flow `CancellationToken`s from the request into `RunAsync` and into workflow runs.

### 2. Dependency injection
- One `IChatClient` per model deployment, registered once.
- Resolve agents as `AIAgent` (keyed by name) so implementations can change without touching callers.
- Bind configuration with `IOptions<T>` and validate it on start-up.

### 3. Type safety
- Use records for workflow messages and structured output.
- Prefer `RunAsync<T>` over parsing JSON out of `response.Text`.

### 4. Error handling
- Catch `ClientResultException` (OpenAI SDK) or `RequestFailedException` (Azure SDK) around model calls; add retries with `Microsoft.Extensions.Http.Resilience` on the `HttpClient` ([production guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_production_guide/#resilience)).
- Workflow executors do not throw out of the run: watch for `WorkflowErrorEvent` and `ExecutorFailedEvent`.

### 5. Performance
- Stream (`RunStreamingAsync`) for user-facing latency.
- Use `AgentWorkflowBuilder.BuildConcurrent` or fan-out edges for independent work.
- Keep histories bounded with compaction or a chat reducer.

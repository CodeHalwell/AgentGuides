---
title: "Microsoft Agent Framework .NET - Comprehensive Technical Guide"
description: "Comprehensive .NET reference for Microsoft.Agents.AI — ChatClientAgent, hosting and DI, workflows, tools, structured output, MCP, memory, context providers and Azure integration, with compiled examples."
framework: microsoft-agent-framework
language: dotnet
---

# Microsoft Agent Framework .NET - Comprehensive Technical Guide

**Framework version:** `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%; first written for the October 2025 preview). Every C# block on this page was compiled against that release.
**Target platform:** .NET 8.0+ (the core packages also target .NET Standard 2.0 / .NET Framework 4.7.2)

Earlier drafts of this page used an `AgentFactory` / `AddAgentFactory` / `UseAzureOpenAI` registration model, `IAgentMemory`, `AddCosmosDBState`, `MapAgents` and a `Workflow.AddAgent` / `AddConditionalEdge` graph API. None of those exist. The page below uses only the shipped surface: agents built from an `IChatClient`, hosting through `Microsoft.Agents.AI.Hosting`, graphs through `Microsoft.Agents.AI.Workflows`.

---

## Introduction

The Microsoft Agent Framework is the open-source successor to **Semantic Kernel**'s agent features and **AutoGen**. It gives .NET developers one SDK for single agents, tool use, and multi-agent workflows.

- **From Semantic Kernel:** enterprise plumbing — dependency injection, telemetry, connectors, and the `Microsoft.Extensions.AI` abstractions (`IChatClient`, `AIFunction`) the framework is built on.
- **From AutoGen:** multi-agent orchestration — sequential, concurrent, handoff, group chat and Magentic patterns.

### Target use cases

- **Customer support automation:** classify, route and resolve issues across specialist agents.
- **Autonomous task execution:** agents that plan and carry out multi-step tasks with tools.
- **Data analysis and RAG:** agents grounded in your own documents and databases.
- **Content pipelines:** drafting, reviewing and publishing as an explicit workflow.

---

## Core Fundamentals

### Architecture

```
+------------------------------------------------------------+
| Your application: APIs, services, background workers        |
+------------------------------------------------------------+
| Hosting (Microsoft.Agents.AI.Hosting.*)                     |
|   AddAIAgent, session stores, OpenAI / A2A / AG-UI endpoints|
+------------------------------------------------------------+
| Workflows (Microsoft.Agents.AI.Workflows)                   |
|   WorkflowBuilder, executors, AgentWorkflowBuilder          |
+------------------------------------------------------------+
| Agents (Microsoft.Agents.AI)                                |
|   AIAgent, ChatClientAgent, AgentSession, context providers |
+------------------------------------------------------------+
| Microsoft.Extensions.AI                                     |
|   IChatClient, AIFunction, ChatMessage, middleware pipeline |
+------------------------------------------------------------+
| Providers: Azure OpenAI, OpenAI, Foundry, Ollama, ...       |
+------------------------------------------------------------+
```

### Installation

```bash
dotnet new console -n MyAgentApp
cd MyAgentApp

# Core agent package (stable)
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%

# OpenAI / Azure OpenAI bridge (AsAIAgent on the OpenAI SDK's ChatClient)
dotnet add package Microsoft.Agents.AI.OpenAI --version %%version=Microsoft.Agents.AI%%
dotnet add package Azure.AI.OpenAI --prerelease
dotnet add package Azure.Identity

# Optional
dotnet add package Microsoft.Agents.AI.Workflows --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.Hosting --prerelease      # DI registration, session stores
dotnet add package Microsoft.Agents.AI.Foundry --prerelease      # Microsoft Foundry agents
```

### Authentication and configuration

Keep secrets out of source code. For local development, user secrets:

```bash
dotnet user-secrets init
dotnet user-secrets set "AzureOpenAI:Endpoint" "https://your-resource.openai.azure.com"
```

```json
{
  "AzureOpenAI": {
    "DeploymentName": "gpt-4o-mini"
  }
}
```

In production, use `DefaultAzureCredential` (managed identity on App Service, Container Apps or AKS) and no API key at all. The [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#microsoft-entra-id-authentication-patterns) covers managed and workload identity.

### Dependency injection

Register the `IChatClient` once, then the agents. `builder.AddAIAgent(name, instructions)` (from `Microsoft.Agents.AI.Hosting`) builds a `ChatClientAgent` on the registered chat client and makes it resolvable as a keyed `AIAgent`.

```csharp
// Program.cs
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateApplicationBuilder(args);

var endpoint = builder.Configuration["AzureOpenAI:Endpoint"]!;
var deployment = builder.Configuration["AzureOpenAI:DeploymentName"] ?? "gpt-4o-mini";

// One chat client for the whole application.
builder.Services.AddChatClient(
    new AzureOpenAIClient(new Uri(endpoint), new DefaultAzureCredential())
        .GetChatClient(deployment)
        .AsIChatClient());

// A named agent built on that chat client.
builder.AddAIAgent("dotnet-helper", "You are a helpful AI assistant for .NET developers.");

// Your own services can take the agent as a dependency.
builder.Services.AddTransient<MySimpleAgent>();

using var host = builder.Build();

var myAgent = host.Services.GetRequiredService<MySimpleAgent>();
Console.WriteLine(await myAgent.AskAsync("Tell me a joke about .NET."));

public class MySimpleAgent([FromKeyedServices("dotnet-helper")] AIAgent agent)
{
    public async Task<string> AskAsync(string input)
    {
        AgentResponse response = await agent.RunAsync(input);
        return response.Text;
    }
}
```

Without the hosting package, register the agent yourself — `builder.Services.AddSingleton<AIAgent>(sp => new ChatClientAgent(sp.GetRequiredService<IChatClient>(), instructions: "..."))` — which is all `AddAIAgent` does for you, plus naming.

---

## Simple Agents

### `AIAgent` and `ChatClientAgent`

- **`AIAgent`** is the abstract base class every agent derives from: `RunAsync`, `RunStreamingAsync`, `CreateSessionAsync`, `SerializeSessionAsync` / `DeserializeSessionAsync`. Code that *uses* agents should depend on `AIAgent`.
- **`ChatClientAgent`** is the implementation for any `IChatClient` — Azure OpenAI, OpenAI, Ollama, Foundry models and so on. It adds instructions, tools, function calling, context providers and chat history. Other implementations wrap remote agents: `A2AAgent`, `CopilotStudioAgent`, Foundry service agents, and workflows exposed via `workflow.AsAIAgent()`.

Agents are stateless with respect to conversations. History lives in an **`AgentSession`**, which you create per conversation and pass to each run.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public class ConversationalAgent(IChatClient chatClient)
{
    private readonly AIAgent _agent =
        new ChatClientAgent(chatClient, instructions: "You are a helpful AI assistant for .NET developers.");

    public async Task StartConversationAsync()
    {
        AgentSession session = await _agent.CreateSessionAsync();

        AgentResponse first = await _agent.RunAsync("What is the latest LTS version of .NET?", session);
        Console.WriteLine($"Assistant: {first.Text}");

        // Same session: the agent sees the previous turn.
        AgentResponse second = await _agent.RunAsync("What were the headline features in that release?", session);
        Console.WriteLine($"Assistant: {second.Text}");
    }
}
```

The convenience extension `AsAIAgent(...)` exists on the OpenAI SDK's `ChatClient` and `ResponsesClient` (package `Microsoft.Agents.AI.OpenAI`) and on Foundry's `AIProjectClient`; for a general `IChatClient`, construct `ChatClientAgent` directly. (It was called `CreateAIAgent` in early previews, and `AgentThread` / `GetNewThread()` became `AgentSession` / `CreateSessionAsync()` before GA.)

### Agent lifecycle

Register the `IChatClient` (and the `AzureOpenAIClient` behind it) once as a singleton — it owns the HTTP connection pool. `ChatClientAgent` is cheap and thread-safe to share; per-conversation state is in the `AgentSession`, so a singleton agent serving many users is normal.

### Execution models

`RunAsync` takes a string, a `ChatMessage` or a list of messages, plus an optional `AgentSession` and `AgentRunOptions`, and returns an `AgentResponse` (`.Text`, `.Messages`, `.Usage`). `RunStreamingAsync` yields `AgentResponseUpdate`s as tokens arrive. `RunAsync<T>` returns structured output (below).

### Testing agents

`ChatClientAgent` sits on `IChatClient`, so a mocked chat client is the natural seam: no live model calls, no framework internals to fake.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Moq;
using Xunit;

public class AgentTests
{
    [Fact]
    public async Task Agent_returns_the_model_reply()
    {
        var chatClient = new Mock<IChatClient>();
        chatClient
            .Setup(c => c.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(new ChatMessage(ChatRole.Assistant, "This is a mock response.")));

        var agent = new ChatClientAgent(chatClient.Object, instructions: "You are helpful.");

        AgentResponse response = await agent.RunAsync("test");

        Assert.Equal("This is a mock response.", response.Text);
    }
}
```

---

## Multi-Agent Systems

Multi-agent orchestration lives in `Microsoft.Agents.AI.Workflows`.

### Orchestration patterns

| Pattern | API |
| --- | --- |
| Sequential: each agent adds to the conversation in turn | `AgentWorkflowBuilder.BuildSequential(a, b, c)` |
| Concurrent: same input to all, results aggregated | `AgentWorkflowBuilder.BuildConcurrent(agents, aggregator)` |
| Handoff: agents transfer control to each other | `AgentWorkflowBuilder.CreateHandoffBuilderWith(triage).WithHandoffs(...)` |
| Group chat: a manager picks the next speaker | `AgentWorkflowBuilder.CreateGroupChatBuilderWith(agents => new RoundRobinGroupChatManager(agents, ...))` |
| Magentic: a planner agent coordinates specialists | `AgentWorkflowBuilder.CreateMagenticBuilderWith(manager)` |
| Custom graph | `new WorkflowBuilder(start).AddEdge(...)...Build()` |

### Sequential workflow

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public class ResearchWorkflow(IChatClient chatClient)
{
    public async Task<string> RunAsync(string topic)
    {
        AIAgent researcher = new ChatClientAgent(chatClient,
            instructions: "You are a research assistant. Find information on the topic.", name: "researcher");
        AIAgent writer = new ChatClientAgent(chatClient,
            instructions: "You are a content writer. Summarise the research above in one paragraph.", name: "writer");

        Workflow workflow = AgentWorkflowBuilder.BuildSequential(researcher, writer);

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, $"Research the topic: {topic}") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent output && output.Is<List<ChatMessage>>(out var conversation))
            {
                return conversation.Last().Text;   // the writer's turn
            }
        }

        return "";
    }
}
```

Don't share one `AgentSession` across two different agents to "pass context"; a session belongs to the agent that created it. Use a workflow, or pass the first agent's output into the second agent's input.

### Shared state

Inside a workflow, executors share state through the workflow context rather than through a global object: `context.QueueStateUpdateAsync(key, value)` and `context.ReadStateAsync<T>(key)`, scoped and checkpointed with the run.

```csharp
using Microsoft.Agents.AI.Workflows;

public sealed class CountingExecutor() : Executor<string, string>("counter")
{
    public override async ValueTask<string> HandleAsync(string message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        int seen = await context.ReadStateAsync<int>("messages-seen", cancellationToken: cancellationToken);
        await context.QueueStateUpdateAsync("messages-seen", seen + 1, cancellationToken: cancellationToken);
        return $"{message} (message #{seen + 1})";
    }
}
```

### Custom graphs and conditional edges

`WorkflowBuilder` builds a directed graph of executors. Agents convert to executors implicitly, and plain functions become executors with `BindAsExecutor`. Conditional routing is an `AddEdge<T>` with a predicate over the message.

```csharp
using Microsoft.Agents.AI.Workflows;

public record Ticket(string Text, bool IsUrgent);

public static class TriageGraph
{
    public static async Task RunAsync()
    {
        var classify = ((Func<string, Ticket>)(text =>
            new Ticket(text, text.Contains("outage", StringComparison.OrdinalIgnoreCase)))).BindAsExecutor("classify");

        // An executor's return value is its output message; on an output executor it becomes a workflow output.
        var page = ((Func<Ticket, string>)(t => $"PAGED ON-CALL: {t.Text}")).BindAsExecutor("page");
        var queue = ((Func<Ticket, string>)(t => $"Queued: {t.Text}")).BindAsExecutor("queue");

        Workflow workflow = new WorkflowBuilder(classify)
            .AddEdge<Ticket>(classify, page, condition: t => t!.IsUrgent)
            .AddEdge<Ticket>(classify, queue, condition: t => !t!.IsUrgent)
            .WithOutputFrom(page, queue)
            .Build();

        await using Run run = await InProcessExecution.RunAsync(workflow, "Total outage in EU-West");
        foreach (WorkflowEvent evt in run.OutgoingEvents)
        {
            if (evt is WorkflowOutputEvent output)
            {
                Console.WriteLine(output.As<string>());   // PAGED ON-CALL: Total outage in EU-West
            }
        }
    }
}
```

The [graph workflows and declarative agents page](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) covers fan-out/fan-in, switches, checkpointing and human-in-the-loop request ports.

---

## Tools Integration

Tools are `AIFunction`s. `AIFunctionFactory.Create` (from `Microsoft.Extensions.AI`) turns a delegate or method into one, generating the JSON schema from the parameters and `[Description]` attributes. `ChatClientAgent` runs the function-calling loop for you.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public class WeatherTools
{
    [Description("Get the current weather for a specified city.")]
    public Task<string> GetCurrentWeatherAsync(
        [Description("The city name, e.g. 'Seattle, WA'")] string city)
    {
        // Call a real weather API here.
        return Task.FromResult(city.Contains("Seattle")
            ? "The weather in Seattle is rainy, 13°C."
            : $"The weather in {city} is sunny, 22°C.");
    }
}

public static class WeatherAgent
{
    public static async Task RunAsync(IChatClient chatClient)
    {
        var weather = new WeatherTools();

        AIAgent agent = new ChatClientAgent(
            chatClient,
            instructions: "You answer weather questions using your tools.",
            tools: [AIFunctionFactory.Create(weather.GetCurrentWeatherAsync)]);

        AgentResponse response = await agent.RunAsync("What's the weather like in Seattle today?");
        Console.WriteLine(response.Text);   // the agent called GetCurrentWeatherAsync first
    }
}
```

### Hosted tools

Some providers run tools server-side. `Microsoft.Extensions.AI` models these as `HostedCodeInterpreterTool`, `HostedWebSearchTool`, `HostedFileSearchTool` and `HostedMcpServerTool`; pass them in `tools:` like any other tool, on a provider that supports them (Foundry agents, the OpenAI Responses API).

### Tool approval

Wrap a function in `ApprovalRequiredAIFunction` to make the agent stop and ask before calling it. The run then returns a `ToolApprovalRequestContent` instead of executing the tool; you answer it on the next run in the same session.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class ApprovalFlow
{
    [Description("Transfer money between two accounts.")]
    public static string Transfer(string from, string to, decimal amount) => $"Moved {amount} from {from} to {to}.";

    public static async Task RunAsync(IChatClient chatClient)
    {
        AIAgent agent = new ChatClientAgent(chatClient,
            instructions: "You are a banking assistant.",
            tools: [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(Transfer))]);

        AgentSession session = await agent.CreateSessionAsync();
        AgentResponse response = await agent.RunAsync("Move 50 from ACC-1 to ACC-2", session);

        var requests = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();
        while (requests.Count > 0)
        {
            var answers = requests.Select(r =>
            {
                var call = (FunctionCallContent)r.ToolCall;
                Console.Write($"Approve {call.Name}? [y/N] ");
                bool approved = Console.ReadLine()?.Trim().ToLowerInvariant() == "y";
                return new ChatMessage(ChatRole.User, [r.CreateResponse(approved)]);
            });

            response = await agent.RunAsync(answers, session);
            requests = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();
        }

        Console.WriteLine(response.Text);
    }
}
```

### Errors in tools

If a tool throws, the function-invocation loop catches it and tells the model the call failed, but by default **without** the exception message, so internal details do not leak into the prompt. Return an explanatory string when the model should know what went wrong:

```csharp
using System.ComponentModel;

public class RobustTools
{
    [Description("Performs a division operation.")]
    public string Divide(double numerator, double denominator) =>
        denominator == 0
            ? "Error: cannot divide by zero. Ask the user for a non-zero denominator."
            : (numerator / denominator).ToString();
}
```

---

## Structured Output

`RunAsync<T>` asks the model for JSON matching `T`'s schema and deserialises it; the typed value is on `response.Result`.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public record Person(
    [property: Description("The full name of the person.")] string Name,
    [property: Description("The person's age.")] int Age,
    [property: Description("A list of the person's skills.")] List<string> Skills);

public class DataExtractionAgent(IChatClient chatClient)
{
    private readonly AIAgent _agent =
        new ChatClientAgent(chatClient, instructions: "You extract structured data from text.");

    public async Task<Person> ExtractPersonAsync(string text)
    {
        AgentResponse<Person> response = await _agent.RunAsync<Person>(text);
        return response.Result;
    }
}

// var person = await new DataExtractionAgent(chatClient)
//     .ExtractPersonAsync("John Doe is a 42-year-old expert in C# and .NET who also knows Python.");
// person.Name == "John Doe", person.Age == 42, person.Skills == ["C#", ".NET", "Python"]
```

If the model's output cannot be deserialised into `T`, `RunAsync<T>` throws (a `JsonException`); catch it and retry or fall back. Models with native structured-output support (the GPT-4o family and later) rarely fail.

---

## Model Context Protocol (MCP)

MCP support comes from the official C# SDK (`ModelContextProtocol`, `ModelContextProtocol.AspNetCore`), whose client tools are already `AIFunction`s, so they drop straight into an agent.

### Using MCP tools in an agent

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;

public static class McpToolsAgent
{
    public static async Task RunAsync(IChatClient chatClient)
    {
        await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new StdioClientTransportOptions
        {
            Name = "github",
            Command = "npx",
            Arguments = ["-y", "@modelcontextprotocol/server-github"],
        }));

        IList<McpClientTool> tools = await mcp.ListToolsAsync();

        AIAgent agent = new ChatClientAgent(
            chatClient,
            instructions: "You help with GitHub repositories.",
            tools: [.. tools.Cast<AITool>()]);

        Console.WriteLine((await agent.RunAsync("List the open issues in microsoft/agent-framework")).Text);
    }
}
```

For a remote server use `new HttpClientTransport(new HttpClientTransportOptions { Endpoint = new Uri("https://...") })`.

### Exposing tools (or an agent) as an MCP server

```csharp
// Program.cs (ASP.NET Core)
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Server;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithTools<MyCustomTools>();

var app = builder.Build();
app.MapMcp("/mcp");     // Streamable HTTP endpoint
app.Run();

[McpServerToolType]
public class MyCustomTools
{
    [McpServerTool, Description("Look up an order's status by ID.")]
    public static string GetOrderStatus(string orderId) => $"Order {orderId}: shipped";
}
```

To publish a whole agent as one MCP tool, convert it with `agent.AsAIFunction()` and register it with `McpServerTool.Create(...)`.

---

## Agentic Patterns

### Planning and execution

Give a planner agent the goal, then hand its plan to an executor agent that has the tools. Each agent keeps its own session.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class PlanAndExecute
{
    public static async Task RunAsync(IChatClient chatClient, IList<AITool> travelTools)
    {
        AIAgent planner = new ChatClientAgent(chatClient,
            instructions: "You are a planner. Break goals into short numbered steps.");
        AIAgent executor = new ChatClientAgent(chatClient,
            instructions: "You carry out plans step by step with your tools.", tools: travelTools);

        const string goal = "Plan a 3-day trip to Seattle: flights, a hotel and an itinerary.";
        AgentResponse plan = await planner.RunAsync(goal);
        Console.WriteLine($"The plan:\n{plan.Text}");

        AgentSession execution = await executor.CreateSessionAsync();
        AgentResponse result = await executor.RunAsync($"Execute this plan:\n{plan.Text}", execution);
        Console.WriteLine(result.Text);
    }
}
```

For long or open-ended plans, the Magentic orchestration (`AgentWorkflowBuilder.CreateMagenticBuilderWith(manager)`) does the planning, progress tracking and re-planning for you.

### Routing

A router classifies the request with structured output and dispatches to a specialist; see Recipe 5 in the [.NET recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-5-router-to-specialist-agents). When specialists need to pass the conversation back and forth, use the handoff builder.

### Reflection

A writer drafts, a critic reviews, the writer revises in its own session (so it remembers its draft).

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class Reflection
{
    public static async Task<string> RunAsync(IChatClient chatClient, string topic)
    {
        AIAgent writer = new ChatClientAgent(chatClient, instructions: "You are a blog post writer.");
        AIAgent critic = new ChatClientAgent(chatClient,
            instructions: "You are a critical editor. Give specific, actionable feedback.");

        AgentSession writerSession = await writer.CreateSessionAsync();

        AgentResponse draft = await writer.RunAsync($"Write a short blog post about: {topic}", writerSession);
        AgentResponse feedback = await critic.RunAsync($"Review this draft:\n\n{draft.Text}");
        AgentResponse final = await writer.RunAsync($"Revise your draft using this feedback:\n\n{feedback.Text}", writerSession);

        return final.Text;
    }
}
```

---

## Memory Systems

### Conversation history

By default a `ChatClientAgent` keeps history in the session itself (`InMemoryChatHistoryProvider`) — or, for providers with server-side conversations such as the Responses API or Foundry agents, as a conversation ID. Persist a session between requests by serialising it:

```csharp
using System.Text.Json;
using Microsoft.Agents.AI;

public static class SessionPersistence
{
    public static async Task<string> SaveAsync(AIAgent agent, AgentSession session)
    {
        JsonElement state = await agent.SerializeSessionAsync(session);
        return state.GetRawText();                     // store in Redis, SQL, Blob, ...
    }

    public static async Task<AgentSession> LoadAsync(AIAgent agent, string json)
    {
        using var doc = JsonDocument.Parse(json);
        return await agent.DeserializeSessionAsync(doc.RootElement.Clone());
    }
}
```

### Cosmos DB chat history

`Microsoft.Agents.AI.CosmosNoSql` (preview) stores each message in Cosmos DB instead of in the session blob, which suits long conversations and lets you query or expire history (`MessageTtlSeconds`).

```csharp
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class CosmosHistory
{
    public static AIAgent Create(IChatClient chatClient) =>
        new ChatClientAgent(chatClient, new ChatClientAgentOptions
        {
            Name = "support",
            ChatOptions = new() { Instructions = "You are a support assistant." },
        }.WithCosmosDBChatHistoryProviderUsingManagedIdentity(
            accountEndpoint: "https://contoso.documents.azure.com:443/",
            databaseId: "agents",
            containerId: "chat-history",
            tokenCredential: new DefaultAzureCredential()));
}
```

The same package provides `CosmosCheckpointStore` for workflow checkpoints.

### Retrieval (RAG)

Retrieval is a context provider. `TextSearchProvider` calls your search function — Azure AI Search, a vector database, SQL full-text, anything — and injects the hits before each model call (or exposes search as a tool, with `SearchTime = OnDemandFunctionCalling`). Ingestion is done with the search service's own SDK. Recipe 6 in the [.NET recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-6-retrieval-with-azure-ai-search) has a complete Azure AI Search example.

For long-term user memory across sessions, `ChatHistoryMemoryProvider` (vector search over past conversations) and `FoundryMemoryProvider` (Foundry's managed memory, in `Microsoft.Agents.AI.Foundry`) are the shipped options.

---

## Context Engineering

Everything the model sees on a turn is assembled from the agent's instructions, the session history, and the output of its **context providers**. A custom `AIContextProvider` can add instructions, messages or tools per run, and observe the result afterwards.

```csharp
using Microsoft.Agents.AI;

public sealed class UserProfileProvider(string displayName, string locale) : AIContextProvider
{
    protected override ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(new AIContext
        {
            Instructions = $"The user is {displayName}. Reply in {locale}.",
        });
}

// new ChatClientAgent(chatClient, new ChatClientAgentOptions
// {
//     ChatOptions = new() { Instructions = "You are a support assistant." },
//     AIContextProviders = [new UserProfileProvider("Ana", "pt-PT")],
// });
```

Context compaction (summarising or trimming long histories) is built in: see `Microsoft.Agents.AI.Compaction` and the `CompactionProvider`.

### Multi-tenant isolation

Sessions are the isolation boundary: never let a request choose a session that belongs to someone else. Key stored sessions by the authenticated user (and tenant), not by an ID from the request body. With `Microsoft.Agents.AI.Hosting`, a hosted agent's session store can do this for you: `builder.AddAIAgent(...).WithInMemorySessionStore(withIsolation: true)` together with `builder.Services.UseClaimsBasedAgentIsolation()` (from `Microsoft.Agents.AI.Hosting.AspNetCore`) partitions stored sessions by a claim of the signed-in user.

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddChatClient(
    new AzureOpenAIClient(new Uri(builder.Configuration["AzureOpenAI:Endpoint"]!), new DefaultAzureCredential())
        .GetChatClient("gpt-4o-mini").AsIChatClient());
builder.Services.AddHttpContextAccessor();
builder.Services.UseClaimsBasedAgentIsolation();               // isolation key = the user's "oid" claim by default
builder.AddAIAgent("assistant", "You are a helpful assistant.")
       .WithInMemorySessionStore(withIsolation: true);         // swap for a durable AgentSessionStore in production
```

---

## Copilot Studio Integration

The integration runs in both directions:

- **Calling a Copilot Studio agent from .NET:** `Microsoft.Agents.AI.CopilotStudio` (preview) provides `CopilotStudioAgent`, an `AIAgent` over the Copilot Studio client (`Microsoft.Agents.CopilotStudio.Client`). It can then take part in workflows like any other agent.
- **Calling your .NET tools from Copilot Studio:** expose them as an MCP server (above) and add it in Copilot Studio as an MCP tool, or publish an OpenAPI description and add it as a custom connector.

---

## Azure AI Integration

- **Azure OpenAI / Foundry models** — the primary model providers (`Azure.AI.OpenAI` with `AsAIAgent`, or `Microsoft.Agents.AI.Foundry`).
- **Foundry Agent Service** — service-managed agents with server-side threads and hosted tools, surfaced as `AIAgent`.
- **Azure AI Search** — retrieval through `TextSearchProvider` or `HostedFileSearchTool`.
- **Azure AI Content Safety** — call it from agent middleware; see the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#azure-ai-content-safety-integration).
- **Application Insights** — `.AsBuilder().UseOpenTelemetry(...)` plus the Azure Monitor exporter.

### Cost and governance

- Token usage is on every `AgentResponse.Usage`; export it as a metric per agent and tenant.
- Use Azure budgets, per-deployment quotas, and Azure Policy (allowed models, no key-based auth).

---

## Semantic Kernel Integration

The Agent Framework is the successor to Semantic Kernel's agent APIs, and both sit on `Microsoft.Extensions.AI`, so migration is mostly mechanical.

### Reusing Semantic Kernel plugins

A `KernelFunction` converts to an `AIFunction` with `AsAIFunction()`, so existing plugins work as agent tools without wrappers.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;

public class MySKPlugin
{
    [KernelFunction, Description("A function from an existing SK plugin.")]
    public string OldPluginFunction(string input) => $"Processed by SK plugin: {input}";
}

public static class SkInterop
{
    public static AIAgent Create(IChatClient chatClient)
    {
        KernelPlugin plugin = KernelPluginFactory.CreateFromType<MySKPlugin>();
        IList<AITool> tools = plugin.Select(f => (AITool)f.AsAIFunction()).ToList();

        return new ChatClientAgent(chatClient, instructions: "Use your tools.", tools: tools);
    }
}
```

### Migration strategy

- **Tools:** convert plugins with `AsAIFunction()` as above, then move them to plain methods with `[Description]` at your own pace.
- **Agents:** `ChatCompletionAgent` → `ChatClientAgent`; `AgentThread` → `AgentSession`; `InvokeAsync` → `RunAsync`.
- **Orchestration:** SK planners and `AgentGroupChat` → `Microsoft.Agents.AI.Workflows` (sequential, concurrent, handoff, group chat, Magentic, or a custom graph).
- **History:** move stored chat history to a `ChatHistoryProvider` (in-memory, Cosmos DB, or your own).

The [migration notice](/microsoft-agent-framework-guide/migration_notice/) has the full mapping.

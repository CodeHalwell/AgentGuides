---
title: "Microsoft Agent Framework (.NET) — Recipes"
description: "Practical, compiled C# recipes for common agent patterns with the Microsoft Agent Framework for .NET: chat, tools, workflows, routing, RAG, ASP.NET Core hosting, Azure Functions and streaming."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Recipes"
---

A collection of practical C# recipes for common agent patterns with the Microsoft Agent Framework for .NET.

**Target platform:** .NET 8.0+
**Framework version:** `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%). Every C# block on this page was compiled against that release.

Packages used across the recipes:

```bash
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.OpenAI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.Workflows --version %%version=Microsoft.Agents.AI%%
dotnet add package Azure.AI.OpenAI --prerelease
dotnet add package Azure.Identity
```

There is no `AgentFactory`, `AddAgentFactory`, `UseAzureOpenAI`, `AgentOptions` or `IAgentMemory` in the framework. An agent is built from a chat client — `AzureOpenAIClient(...).GetChatClient(deployment).AsAIAgent(...)`, or `new ChatClientAgent(IChatClient, ...)` — and you register it in DI yourself (or with `builder.AddAIAgent(...)` from the hosting package, shown in Recipe 9). The recipes share this small helper:

```csharp
// Agents.cs
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Chat;

public static class Agents
{
    private static readonly AzureOpenAIClient Client = new(
        new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!),
        new DefaultAzureCredential());

    private static string Deployment =>
        Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai-small%%";

    /// The IChatClient abstraction, for DI registration and ChatClientAgent.
    public static IChatClient ChatClient() => Client.GetChatClient(Deployment).AsIChatClient();

    public static ChatClientAgent Create(string instructions, string? name = null, IList<AITool>? tools = null) =>
        Client.GetChatClient(Deployment).AsAIAgent(instructions: instructions, name: name, tools: tools);
}
```

---

## Beginner Recipes

### Recipe 1: Simple chat agent

The "Hello, World!" of the framework: a conversational agent that keeps history in an `AgentSession`.

```csharp
// SimpleChatAgent.cs
using Microsoft.Agents.AI;

public class SimpleChatAgent
{
    private readonly AIAgent _agent =
        Agents.Create("You are a friendly AI assistant. Keep your responses concise.", name: "chat");

    public async Task RunInteractiveAsync()
    {
        Console.WriteLine("Chat agent ready. Type 'exit' to quit.");
        AgentSession session = await _agent.CreateSessionAsync();

        while (true)
        {
            Console.Write("You: ");
            var userInput = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(userInput) || userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            AgentResponse response = await _agent.RunAsync(userInput, session);
            Console.WriteLine($"Assistant: {response.Text}");
        }
    }
}

// Program.cs:  await new SimpleChatAgent().RunInteractiveAsync();
```

### Recipe 2: Agent with a single tool

Tools are `AIFunction`s created with `AIFunctionFactory.Create` (from `Microsoft.Extensions.AI`). The `[Description]` attributes become the tool and parameter descriptions the model sees.

```csharp
// AgentWithTool.cs
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class TimeTools
{
    [Description("Gets the current UTC date and time.")]
    public static string GetCurrentTime() => DateTime.UtcNow.ToString("F");
}

public class ToolAgent
{
    public AIAgent Agent { get; } = Agents.Create(
        "You have a tool to get the current time. Use it when asked about the time.",
        tools: [AIFunctionFactory.Create(TimeTools.GetCurrentTime)]);

    public async Task<string> AskAboutTimeAsync()
    {
        AgentResponse response = await Agent.RunAsync("What time is it right now?");
        return response.Text;
    }
}
```

### Recipe 3: Error handling in tools

If a tool throws, the function-invocation loop catches the exception and tells the model the call failed — but by default **without** the exception message, so internal details do not leak into the conversation. When the model should know *why*, return the explanation as the result instead of throwing.

```csharp
// ErrorHandlingTool.cs
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class MathTools
{
    [Description("Divides two numbers.")]
    public static string Divide(
        [Description("The number to divide")] double numerator,
        [Description("The number to divide by")] double denominator)
    {
        if (denominator == 0)
        {
            // A result string reaches the model verbatim; it can then ask the user for another value.
            return "Error: cannot divide by zero. Ask the user for a non-zero denominator.";
        }

        return (numerator / denominator).ToString();
    }

    public static async Task DemoAsync()
    {
        AIAgent agent = Agents.Create("You are a calculator.", tools: [AIFunctionFactory.Create(Divide)]);
        AgentResponse response = await agent.RunAsync("What is 10 divided by 0?");
        Console.WriteLine(response.Text);   // e.g. "I can't divide by zero — could you give me a different number?"
    }
}
```

If you build the pipeline yourself (`chatClient.AsBuilder().UseFunctionInvocation(configure: f => f.IncludeDetailedErrors = true)`), exception messages are passed to the model too; only do that when they contain nothing sensitive.

---

## Intermediate Recipes

### Recipe 4: Sequential multi-agent workflow

Two agents in a pipeline — a researcher, then a summariser — built with `AgentWorkflowBuilder.BuildSequential` from `Microsoft.Agents.AI.Workflows`. Each agent sees the conversation so far and adds its turn. (Do not share one `AgentSession` between two different agents; a session belongs to the agent that created it.)

```csharp
// SequentialWorkflow.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class SequentialWorkflow
{
    public static async Task<string> RunAsync(string topic)
    {
        AIAgent researcher = Agents.Create(
            "You are a world-class researcher. Find detailed information on the given topic.", name: "researcher");
        AIAgent summariser = Agents.Create(
            "You are a skilled editor. Summarise the conversation so far into one concise paragraph.", name: "summariser");

        Workflow workflow = AgentWorkflowBuilder.BuildSequential(researcher, summariser);

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, $"Research the topic: {topic}") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));   // start the agents

        string summary = "";
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent output && output.Is<List<ChatMessage>>(out var conversation))
            {
                summary = conversation.Last().Text;   // the summariser's turn
            }
        }

        return summary;
    }
}
```

### Recipe 5: Router to specialist agents

A small router agent classifies the request with structured output (`RunAsync<T>`), then the matching specialist answers. For routing where the specialists can hand the conversation back and forth, use `AgentWorkflowBuilder.CreateHandoffBuilderWith(...)` instead (see the [workflows page](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/)).

```csharp
// RouterWorkflow.cs
using System.Text.Json.Serialization;
using Microsoft.Agents.AI;

[JsonConverter(typeof(JsonStringEnumConverter<Route>))]
public enum Route { Billing, TechnicalSupport }

public record RouteDecision(Route Route, string Reason);

public class RouterWorkflow
{
    private readonly AIAgent _router = Agents.Create(
        "Classify the user's request as Billing or TechnicalSupport.", name: "router");

    private readonly Dictionary<Route, AIAgent> _specialists = new()
    {
        [Route.Billing] = Agents.Create("You are a billing support specialist.", name: "billing"),
        [Route.TechnicalSupport] = Agents.Create("You are a technical support specialist.", name: "tech"),
    };

    public async Task<string> RouteAndHandleRequestAsync(string userInput)
    {
        AgentResponse<RouteDecision> decision = await _router.RunAsync<RouteDecision>(userInput);
        AIAgent specialist = _specialists[decision.Result.Route];

        AgentResponse answer = await specialist.RunAsync(userInput);
        return answer.Text;
    }
}
```

### Recipe 6: Retrieval with Azure AI Search

The framework's retrieval hook is `TextSearchProvider`, an `AIContextProvider` that runs your search function and injects the results into the prompt before each model call. The search itself is plain `Azure.Search.Documents` (`dotnet add package Azure.Search.Documents`). Ingestion is done with the search SDK too; there is no framework-level "memory upsert".

```csharp
// KnowledgeWorker.cs
using Azure.Identity;
using Azure.Search.Documents;
using Azure.Search.Documents.Models;
using Microsoft.Agents.AI;

public record KnowledgeDocument(string Id, string Content, string Source);

public class KnowledgeWorker
{
    private readonly SearchClient _search = new(
        new Uri(Environment.GetEnvironmentVariable("AZURE_SEARCH_ENDPOINT")!),
        indexName: "knowledge",
        new DefaultAzureCredential());

    private readonly AIAgent _ragAgent;

    public KnowledgeWorker()
    {
        var retrieval = new TextSearchProvider(SearchAsync, new TextSearchProviderOptions
        {
            SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
        }, loggerFactory: null);

        _ragAgent = new ChatClientAgent(Agents.ChatClient(), new ChatClientAgentOptions
        {
            Name = "rag",
            ChatOptions = new() { Instructions = "Answer using the provided sources and cite them. Say so if they don't cover the question." },
            AIContextProviders = [retrieval],
        });
    }

    public async Task IngestKnowledgeAsync(string documentId, string content, string source)
    {
        await _search.MergeOrUploadDocumentsAsync([new KnowledgeDocument(documentId, content, source)]);
    }

    public async Task<string> AnswerQuestionAsync(string question)
    {
        AgentResponse response = await _ragAgent.RunAsync(question);
        return response.Text;
    }

    private async Task<IEnumerable<TextSearchProvider.TextSearchResult>> SearchAsync(string query, CancellationToken ct)
    {
        SearchResults<KnowledgeDocument> results =
            await _search.SearchAsync<KnowledgeDocument>(query, new SearchOptions { Size = 5 }, ct);

        var hits = new List<TextSearchProvider.TextSearchResult>();
        await foreach (SearchResult<KnowledgeDocument> hit in results.GetResultsAsync())
        {
            hits.Add(new TextSearchProvider.TextSearchResult
            {
                Text = hit.Document.Content,
                SourceName = hit.Document.Source,
            });
        }
        return hits;
    }
}
```

`ChatClientAgentOptions` is the long-form way to configure an agent: instructions go in `ChatOptions.Instructions`, context providers in `AIContextProviders`. Set `SearchTime = OnDemandFunctionCalling` to expose search as a tool the model calls only when it needs to.

---

## Advanced Recipes

### Recipe 7: RAG end to end

```csharp
// RagSystem.cs
public class RagSystem
{
    private readonly KnowledgeWorker _knowledge = new();

    public async Task RunExampleAsync()
    {
        // 1. Ingest documents into the index
        await _knowledge.IngestKnowledgeAsync("doc-001",
            "The Microsoft Agent Framework unifies Semantic Kernel and AutoGen.", "overview.md");
        await _knowledge.IngestKnowledgeAsync("doc-002",
            "The framework is available for .NET and Python.", "platforms.md");

        // 2. Ask a question that needs retrieval
        const string question = "What two frameworks were unified?";
        var answer = await _knowledge.AnswerQuestionAsync(question);

        Console.WriteLine($"Question: {question}");
        Console.WriteLine($"Answer: {answer}");   // mentions Semantic Kernel and AutoGen, citing overview.md
    }
}
```

Azure AI Search indexes documents asynchronously, so in a real system ingestion and querying are separate processes; this example runs them together for brevity.

### Recipe 8: Explicit multi-agent graph with `WorkflowBuilder`

`WorkflowBuilder` gives you the graph directly: agents (or custom executors) as nodes, edges between them, and a declared output. An `AIAgent` converts to an executor binding implicitly.

```csharp
// AdvancedWorkflow.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public class AdvancedWorkflow
{
    public async Task<string> RunPipelineAsync(string topic)
    {
        AIAgent researcher = Agents.Create("You are a researcher. Find three key points on the topic.", name: "researcher");
        AIAgent analyst = Agents.Create("You are a data analyst. Assess the key points critically.", name: "analyst");
        AIAgent writer = Agents.Create("You are a technical writer. Write a short brief from the analysis.", name: "writer");

        Workflow workflow = new WorkflowBuilder(researcher)
            .AddEdge(researcher, analyst)
            .AddEdge(analyst, writer)
            .WithOutputFrom(writer)
            .WithName("ResearchBriefWorkflow")
            .Build();

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new ChatMessage(ChatRole.User, topic));
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

        var brief = new System.Text.StringBuilder();
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case AgentResponseUpdateEvent update when update.ExecutorId.StartsWith("writer"):
                    brief.Append(update.Update.Text);
                    break;
                case WorkflowErrorEvent error:
                    throw new InvalidOperationException("Workflow failed", error.Exception);
            }
        }

        return brief.ToString();
    }
}
```

Use `AddEdge<T>(source, target, condition)` for conditional routing, `AddFanOutEdge` / `AddFanInBarrierEdge` for parallel branches, and `AddSwitch` for multi-way branching.

---

## Integration Recipes

### Recipe 9: Exposing an agent through ASP.NET Core

`Microsoft.Agents.AI.Hosting` (preview) registers agents in DI by name; `AddAIAgent` resolves the `IChatClient` you registered. You can write your own endpoint, keeping one session per conversation, or let `Microsoft.Agents.AI.Hosting.OpenAI` expose an OpenAI-compatible Responses endpoint for you.

```bash
dotnet add package Microsoft.Agents.AI.Hosting --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.OpenAI --prerelease   # optional: OpenAI-compatible endpoints
```

```csharp
// Program.cs (ASP.NET Core minimal API)
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddChatClient(Agents.ChatClient());
builder.Services.AddDistributedMemoryCache();          // swap for Redis / SQL in production
builder.AddAIAgent("assistant", "You are a helpful assistant. Keep answers short.");
builder.AddOpenAIResponses();                           // optional, for app.MapOpenAIResponses()

var app = builder.Build();

app.MapPost("/chat/{conversationId}", async (
    string conversationId,
    ChatRequest request,
    [FromKeyedServices("assistant")] AIAgent agent,
    IDistributedCache cache,
    CancellationToken ct) =>
{
    // Restore this conversation's session, or start a new one.
    string? saved = await cache.GetStringAsync($"session:{conversationId}", ct);
    AgentSession session = saved is null
        ? await agent.CreateSessionAsync(ct)
        : await agent.DeserializeSessionAsync(JsonDocument.Parse(saved).RootElement, cancellationToken: ct);

    AgentResponse response = await agent.RunAsync(request.Message, session, cancellationToken: ct);

    JsonElement state = await agent.SerializeSessionAsync(session, cancellationToken: ct);
    await cache.SetStringAsync($"session:{conversationId}", state.GetRawText(), ct);

    return Results.Ok(new { reply = response.Text });
});

// OpenAI Responses-compatible endpoint for every registered agent (POST /v1/responses).
app.MapOpenAIResponses();

app.Run();

public record ChatRequest(string Message);
```

Authenticate the endpoint before exposing it (`app.MapPost(...).RequireAuthorization()`), and derive `conversationId` from the authenticated user rather than trusting the URL alone.

### Recipe 10: Event-driven agents with Azure Functions

A queue-triggered function on the isolated worker model. The agent is created once and reused across invocations.

```csharp
// AgentTriggerFunction.cs
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;

public class AgentTriggerFunction(ILogger<AgentTriggerFunction> logger)
{
    private static readonly AdvancedWorkflow Workflow = new();

    [Function("ProcessQueueItem")]
    public async Task Run([QueueTrigger("agent-tasks", Connection = "AzureWebJobsStorage")] string topic)
    {
        logger.LogInformation("Processing topic {Topic}", topic);

        string brief = await Workflow.RunPipelineAsync(topic);

        logger.LogInformation("Workflow completed with {Length} characters", brief.Length);
        // Write the result to another queue, a blob or a database.
    }
}

// Program.cs for the Functions isolated worker:
//   var builder = FunctionsApplication.CreateBuilder(args);
//   builder.Build().Run();
```

Long-running or human-in-the-loop work fits Durable Functions better than a single queue trigger, because a durable orchestration can wait days for an approval without holding a worker.

---

## Troubleshooting Patterns

### Pattern 11: Watching tool calls with streaming

`RunStreamingAsync` yields `AgentResponseUpdate`s. Their `Contents` show the model's text as it arrives and the tool calls and results as they happen.

```csharp
// StreamingDebug.cs
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class StreamingDebug
{
    public static async Task DebugWithStreamingAsync(string userInput)
    {
        AIAgent agent = new ToolAgent().Agent;
        AgentSession session = await agent.CreateSessionAsync();

        await foreach (AgentResponseUpdate update in agent.RunStreamingAsync(userInput, session))
        {
            foreach (AIContent content in update.Contents)
            {
                switch (content)
                {
                    case FunctionCallContent call:
                        Console.ForegroundColor = ConsoleColor.Gray;
                        Console.WriteLine($"[tool call] {call.Name}({string.Join(", ", call.Arguments ?? new Dictionary<string, object?>())})");
                        Console.ResetColor();
                        break;
                    case FunctionResultContent result:
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"[tool result] {result.Result}");
                        Console.ResetColor();
                        break;
                    case TextContent text:
                        Console.Write(text.Text);
                        break;
                }
            }
        }

        Console.WriteLine();
    }
}
```

For anything beyond local debugging, wrap the agent with `.AsBuilder().UseOpenTelemetry(...)` and look at the spans instead; see the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#opentelemetry-instrumentation).

---
title: "Semantic Kernel Recipes (.NET)"
description: "Ready-to-use C# patterns for Semantic Kernel — ASP.NET Core endpoints, agents with dependency injection, orchestrations, RAG and streaming."
framework: "semantic-kernel"
language: "dotnet"
sidebar:
  label: "Recipes"
  order: 2
---

> Targets `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%% (checked %%versions-checked%%). Agents need `Microsoft.SemanticKernel.Agents.Core`; orchestrations need the preview packages `Microsoft.SemanticKernel.Agents.Orchestration` and `.Agents.Runtime.InProcess` and `<NoWarn>$(NoWarn);SKEXP0110</NoWarn>`. For new agent projects consider [Microsoft Agent Framework](/microsoft-agent-framework-guide/dotnet/), Semantic Kernel's successor.

**See also:** [language-agnostic recipes](/semantic-kernel-guide/semantic_kernel_recipes/) · [comprehensive .NET guide](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/)

---

## 1. ASP.NET Core minimal API with a transient kernel

```csharp
// Program.cs
using Azure.Identity;
using Microsoft.SemanticKernel;

var builder = WebApplication.CreateBuilder(args);

// AI services are registered once (they are thread-safe and hold HTTP clients)...
builder.Services.AddAzureOpenAIChatCompletion(
    deploymentName: builder.Configuration["AzureOpenAI:Deployment"]!,
    endpoint: builder.Configuration["AzureOpenAI:Endpoint"]!,
    credentials: new DefaultAzureCredential());

builder.Services.AddSingleton<WeatherPlugin>();   // WeatherPlugin: see the comprehensive guide, section 4.1

// ...but the Kernel is transient, so plugins and filters added per request do not leak
builder.Services.AddTransient(sp =>
{
    var kernel = new Kernel(sp);
    kernel.Plugins.AddFromObject(sp.GetRequiredService<WeatherPlugin>(), "weather");
    return kernel;
});

var app = builder.Build();

app.MapPost("/api/chat", async (ChatRequest request, Kernel kernel, ILogger<Program> logger) =>
{
    try
    {
        var answer = await kernel.InvokePromptAsync<string>(
            "Answer the following question concisely: {{$input}}",
            new() { ["input"] = request.Message });
        return Results.Ok(new { response = answer });
    }
    catch (HttpOperationException ex)
    {
        logger.LogError(ex, "Model call failed");
        return Results.Problem("The AI service is unavailable.", statusCode: 503);   // don't leak exception text
    }
});

app.Run();

public sealed record ChatRequest(string Message);
```

`HttpOperationException` is what the OpenAI and Azure OpenAI connectors throw for HTTP failures; its `StatusCode` tells you whether retrying makes sense.

---

## 2. A chat endpoint that keeps conversation state

```csharp
using System.Collections.Concurrent;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;

public sealed class ConversationService(Kernel kernel)
{
    // Demo only: use a distributed store (Redis, SQL) in production and serialise the ChatHistory
    private static readonly ConcurrentDictionary<string, ChatHistory> Histories = new();

    private readonly ChatCompletionAgent _agent = new()
    {
        Name = "Support",
        Instructions = "You are a concise support assistant.",
        Kernel = kernel,
    };

    public async Task<string> SendAsync(string sessionId, string message)
    {
        var history = Histories.GetOrAdd(sessionId, _ => new ChatHistory());
        var thread = new ChatHistoryAgentThread(history);

        string reply = "";
        await foreach (var item in _agent.InvokeAsync(message, thread))
            reply = item.Message.Content ?? "";
        return reply;
    }
}

// builder.Services.AddTransient<ConversationService>();
```

`ChatHistoryAgentThread` appends to the `ChatHistory` you give it, so the dictionary entry stays current.

---

## 3. Research-and-write pipeline with dependency injection

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Orchestration;
using Microsoft.SemanticKernel.Agents.Orchestration.Sequential;
using Microsoft.SemanticKernel.Agents.Runtime.InProcess;

public interface IAgentOrchestrator
{
    Task<string> ProcessAsync(string input, CancellationToken ct = default);
}

public sealed class ResearchWritingOrchestrator(Kernel kernel, ILogger<ResearchWritingOrchestrator> logger)
    : IAgentOrchestrator
{
    public async Task<string> ProcessAsync(string input, CancellationToken ct = default)
    {
        ChatCompletionAgent researcher = new()
        {
            Name = "Researcher",
            Description = "Gathers facts",
            Instructions = "Research the topic and list key facts with sources.",
            Kernel = kernel,
        };
        ChatCompletionAgent writer = new()
        {
            Name = "Writer",
            Description = "Writes the article",
            Instructions = "Write a 200-word article from the research notes.",
            Kernel = kernel,
        };

        SequentialOrchestration orchestration = new(researcher, writer)
        {
            ResponseCallback = msg =>
            {
                logger.LogInformation("[{Agent}] {Length} chars", msg.AuthorName, msg.Content?.Length ?? 0);
                return ValueTask.CompletedTask;
            },
        };

        InProcessRuntime runtime = new();
        await runtime.StartAsync(ct);
        try
        {
            var result = await orchestration.InvokeAsync(input, runtime, ct);
            return await result.GetValueAsync(TimeSpan.FromMinutes(3), ct);
        }
        finally
        {
            await runtime.RunUntilIdleAsync();
        }
    }
}

// builder.Services.AddScoped<IAgentOrchestrator, ResearchWritingOrchestrator>();
```

A sequential orchestration has a defined end point (the last agent's answer), so there is no need to rely on the model saying a stop word.

---

## 4. Streaming tokens over Server-Sent Events

```csharp
app.MapGet("/api/stream", async (string q, Kernel kernel, HttpResponse response, CancellationToken ct) =>
{
    response.Headers.ContentType = "text/event-stream";
    await foreach (var chunk in kernel.InvokePromptStreamingAsync<string>(q, cancellationToken: ct))
    {
        await response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(chunk)}\n\n", ct);
        await response.Body.FlushAsync(ct);
    }
});
```

---

## 5. RAG tool from a vector store

```csharp
using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Data;

public sealed class Doc
{
    [VectorStoreKey] public string Id { get; set; } = "";
    [VectorStoreData, TextSearchResultValue] public string Text { get; set; } = "";
    [VectorStoreVector(1536)] public string Embedding => Text;
}

var store = new InMemoryVectorStore(new()
{
    EmbeddingGenerator = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>(),
});
var docs = store.GetCollection<string, Doc>("docs");
await docs.EnsureCollectionExistsAsync();
await docs.UpsertAsync(new Doc { Id = "1", Text = "Refunds are processed within 5 working days." });

kernel.Plugins.Add(new VectorStoreTextSearch<Doc>(docs).CreateWithGetTextSearchResults("Docs"));
// An agent with FunctionChoiceBehavior.Auto() can now call Docs.GetTextSearchResults
```

Register the embedding generator with `builder.AddOpenAIEmbeddingGenerator("text-embedding-3-small", apiKey)` (or the Azure equivalent).

---

For more, see the [comprehensive guide](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/), the [production guide](/semantic-kernel-guide/dotnet/semantic_kernel_production_dotnet/) and the [general recipes](/semantic-kernel-guide/semantic_kernel_recipes/).

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 1.81.0 | Rewritten on current APIs (orchestrations, `CommunityToolkit.VectorData`); all blocks compile against 1.81.0. |

**[Back to .NET overview](/semantic-kernel-guide/dotnet/)**

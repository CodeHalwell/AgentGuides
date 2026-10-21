---
title: "Semantic Kernel Comprehensive Guide (.NET)"
description: "C#/.NET reference for Semantic Kernel — kernel and services, functions and plugins, function calling, agents, orchestrations, vector stores, MCP, filters, ASP.NET Core and migration to Microsoft Agent Framework."
framework: semantic-kernel
language: dotnet
---

# Semantic Kernel Comprehensive Guide (.NET)

> Verified against `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%% (checked %%versions-checked%%). Type and member names were checked against the published packages' API documentation and the Semantic Kernel source at that release. The package targets .NET 8, .NET 10 and .NET Standard 2.0.

> **Semantic Kernel and Microsoft Agent Framework.** [Microsoft Agent Framework](/microsoft-agent-framework-guide/dotnet/) (MAF, `Microsoft.Agents.AI`) is the successor to Semantic Kernel, built by the same team, and has been generally available since April 2026. Semantic Kernel 1.x remains supported — critical bug and security fixes for at least a year after MAF's GA — but most new features go to MAF. New agent projects should start on MAF; existing SK applications can migrate gradually, and every SK agent type can be wrapped as a MAF `AIAgent` ([section 14](#14-migrating-to-microsoft-agent-framework)).

---

## Table of contents

1. [Installation and packages](#1-installation-and-packages)
2. [Building a kernel](#2-building-a-kernel)
3. [AI services](#3-ai-services)
4. [Functions and plugins](#4-functions-and-plugins)
5. [Function calling (planners are gone)](#5-function-calling-planners-are-gone)
6. [Agents](#6-agents)
7. [Multi-agent orchestration](#7-multi-agent-orchestration)
8. [Vector stores and RAG](#8-vector-stores-and-rag)
9. [Model Context Protocol (MCP)](#9-model-context-protocol-mcp)
10. [A2A](#10-a2a)
11. [Structured output](#11-structured-output)
12. [Filters](#12-filters)
13. [ASP.NET Core, resilience and telemetry](#13-aspnet-core-resilience-and-telemetry)
14. [Migrating to Microsoft Agent Framework](#14-migrating-to-microsoft-agent-framework)
15. [Removed and renamed APIs](#15-removed-and-renamed-apis)

---

## 1. Installation and packages

```bash
dotnet new console -n SkApp && cd SkApp
dotnet add package Microsoft.SemanticKernel --version %%version=Microsoft.SemanticKernel%%
```

`Microsoft.SemanticKernel` bundles the core plus the OpenAI and Azure OpenAI connectors. Other packages version in lockstep with it, but not all are stable — check the suffix on NuGet before depending on one:

| Package | Status at %%version=Microsoft.SemanticKernel%% | Purpose |
|---|---|---|
| `Microsoft.SemanticKernel.Agents.Core` | stable | `ChatCompletionAgent`, `ChatHistoryAgentThread` |
| `Microsoft.SemanticKernel.Agents.Orchestration` | `-preview` | Sequential / concurrent / group chat / hand-off / Magentic orchestrations |
| `Microsoft.SemanticKernel.Agents.Runtime.InProcess` | `-preview` | Runtime the orchestrations run on |
| `Microsoft.SemanticKernel.Agents.OpenAI`, `.AzureAI`, `.A2A` | `-preview` | Assistants/Responses, Azure AI Foundry and A2A agents |
| `Microsoft.SemanticKernel.Plugins.OpenApi` | stable | Import OpenAPI documents as plugins |
| `Microsoft.SemanticKernel.Plugins.Core` | `-preview` | Time, math, text, HTTP plugins |
| `Microsoft.SemanticKernel.Connectors.Google`, `.Ollama`, ... | `-alpha` | Other model providers |
| `Microsoft.SemanticKernel.Process.*` | `-alpha` | Process framework |

```bash
# Prerelease packages need an explicit version (or --prerelease)
dotnet add package Microsoft.SemanticKernel.Agents.Core --version %%version=Microsoft.SemanticKernel%%
dotnet add package Microsoft.SemanticKernel.Agents.Orchestration --prerelease
dotnet add package Microsoft.SemanticKernel.Agents.Runtime.InProcess --prerelease
```

Vector store connectors no longer ship as `Microsoft.SemanticKernel.Connectors.*`: they were renamed to `CommunityToolkit.VectorData.*` (stable), implementing `Microsoft.Extensions.VectorData` (section 8).

**Experimental APIs** raise diagnostics such as `SKEXP0110` (agents, orchestration) and `SKEXP0001`. Opt in per project:

```xml
<PropertyGroup>
  <NoWarn>$(NoWarn);SKEXP0001;SKEXP0110</NoWarn>
</PropertyGroup>
```

**Configuration** — keep keys in user secrets or environment variables, never in `appsettings.json` committed to source control:

```bash
dotnet user-secrets init
dotnet user-secrets set "OpenAI:ApiKey" "sk-..."
dotnet user-secrets set "AzureOpenAI:Endpoint" "https://my-resource.openai.azure.com/"
```

---

## 2. Building a kernel

```csharp
using Microsoft.SemanticKernel;

var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(
    modelId: "gpt-5-mini",
    apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);
builder.Plugins.AddFromType<WeatherPlugin>("weather");

Kernel kernel = builder.Build();

var answer = await kernel.InvokePromptAsync("Give me one sentence about Semantic Kernel.");
Console.WriteLine(answer);
```

The `Kernel` holds an `IServiceProvider` (AI services, loggers, your own services), a plugin collection and filters. It is lightweight: in web apps register it as **transient** so per-request plugins and filters do not leak (section 13). Model names in this guide are examples.

---

## 3. AI services

### 3.1 OpenAI and Azure OpenAI

```csharp
// OpenAI
builder.AddOpenAIChatCompletion("gpt-5-mini", apiKey, serviceId: "openai");
builder.AddOpenAIEmbeddingGenerator("text-embedding-3-small", apiKey);

// Azure OpenAI with an API key
builder.AddAzureOpenAIChatCompletion(
    deploymentName: "gpt-5-mini",
    endpoint: "https://my-resource.openai.azure.com/",
    apiKey: azureKey,
    serviceId: "azure");

// Azure OpenAI with Entra ID (managed identity in production)
builder.AddAzureOpenAIChatCompletion(
    deploymentName: "gpt-5-mini",
    endpoint: "https://my-resource.openai.azure.com/",
    credentials: new Azure.Identity.DefaultAzureCredential());
```

`AddOpenAIEmbeddingGenerator` / `AddAzureOpenAIEmbeddingGenerator` register a `Microsoft.Extensions.AI` `IEmbeddingGenerator<string, Embedding<float>>`, the type the vector store abstractions use. (The older `ITextEmbeddingGenerationService` registrations are obsolete.)

### 3.2 `Microsoft.Extensions.AI` chat clients

Any `IChatClient` can back the kernel — `AddOpenAIChatClient` / `AddAzureOpenAIChatClient` register one directly, and an existing `IChatClient` (for example from another provider's SDK) can be added with `builder.Services.AddSingleton<IChatClient>(...)`. This is also the abstraction Microsoft Agent Framework uses, which makes it the easiest path for later migration.

### 3.3 Selecting a service

```csharp
using Microsoft.SemanticKernel.Connectors.OpenAI;

var settings = new OpenAIPromptExecutionSettings { ServiceId = "openai", MaxTokens = 400 };
var result = await kernel.InvokePromptAsync("Summarise: {{$input}}", new KernelArguments(settings) { ["input"] = text });
```

### 3.4 Using the chat service directly

```csharp
using Microsoft.SemanticKernel.ChatCompletion;

var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory("You are terse.");
history.AddUserMessage("What is RAG?");

var reply = await chat.GetChatMessageContentAsync(history, kernel: kernel);
Console.WriteLine(reply.Content);

await foreach (var chunk in chat.GetStreamingChatMessageContentsAsync(history, kernel: kernel))
    Console.Write(chunk.Content);
```

---

## 4. Functions and plugins

### 4.1 Native functions

```csharp
using System.ComponentModel;
using Microsoft.SemanticKernel;

public sealed class WeatherPlugin
{
    [KernelFunction, Description("Get the current weather for a city.")]
    public string GetWeather([Description("City name, e.g. Paris")] string city)
        => $"Sunny, 22°C in {city}";

    [KernelFunction("get_forecast"), Description("Get a multi-day forecast.")]
    public async Task<string[]> GetForecastAsync(string city, int days = 3)
    {
        await Task.Yield();
        return Enumerable.Range(1, days).Select(d => $"Day {d}: sunny").ToArray();
    }
}

kernel.ImportPluginFromType<WeatherPlugin>("weather");          // or builder.Plugins.AddFromType<T>()
var result = await kernel.InvokeAsync("weather", "GetWeather", new() { ["city"] = "Paris" });
```

Plugins created with `AddFromType<T>()` are constructed through the kernel's service provider, so constructor dependencies are injected.

### 4.2 Prompt functions

```csharp
var summarise = kernel.CreateFunctionFromPrompt(
    "Summarise the following in {{$sentences}} sentences:\n\n{{$input}}",
    functionName: "summarise");

var summary = await kernel.InvokeAsync(summarise, new() { ["input"] = longText, ["sentences"] = "2" });
```

Templates can call other functions (`{{weather.GetWeather $city}}`). Handlebars and Liquid formats are available via `Microsoft.SemanticKernel.PromptTemplates.Handlebars` / `.Liquid`; YAML prompt files via `Microsoft.SemanticKernel.Yaml` (`kernel.CreateFunctionFromPromptYaml(...)`).

### 4.3 Other plugin sources

```csharp
// Group arbitrary functions into a plugin
kernel.Plugins.AddFromFunctions("text", [summarise]);

// OpenAPI (package Microsoft.SemanticKernel.Plugins.OpenApi)
await kernel.ImportPluginFromOpenApiAsync(
    pluginName: "petstore",
    uri: new Uri("https://petstore3.swagger.io/api/v3/openapi.json"));
```

MCP servers (section 9) and vector-store searches (section 8) also become plugins.

---

## 5. Function calling (planners are gone)

`SequentialPlanner`, `StepwisePlanner`, `HandlebarsPlanner` and `FunctionCallingStepwisePlanner` are obsolete: the `Microsoft.SemanticKernel.Planners.*` packages stopped at 1.47.0-preview and are deprecated on NuGet, with a migration guide pointing to function calling. Let the model call your functions directly:

```csharp
using Microsoft.SemanticKernel.Connectors.OpenAI;

var settings = new OpenAIPromptExecutionSettings
{
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
};

var result = await kernel.InvokePromptAsync(
    "What's the weather in Paris and Rome?",
    new KernelArguments(settings));
```

| Behaviour | Meaning |
|---|---|
| `FunctionChoiceBehavior.Auto()` | Model may call any advertised function; the kernel invokes them |
| `FunctionChoiceBehavior.Auto(functions: [...])` | Advertise only these functions |
| `FunctionChoiceBehavior.Auto(autoInvoke: false)` | Return the calls to you instead of invoking them |
| `FunctionChoiceBehavior.Required(...)` | Model must call one of the functions |
| `FunctionChoiceBehavior.None()` | Advertise but do not call — useful for producing a plan for human review |

---

## 6. Agents

### 6.1 `ChatCompletionAgent`

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Connectors.OpenAI;

ChatCompletionAgent agent = new()
{
    Name = "Assistant",
    Instructions = "You are a helpful assistant. Use the weather tool for weather questions.",
    Kernel = kernel,                                  // supplies the chat service and plugins
    Arguments = new KernelArguments(new OpenAIPromptExecutionSettings
    {
        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
    }),
};

await foreach (AgentResponseItem<ChatMessageContent> item in agent.InvokeAsync("What's the weather in Paris?"))
{
    Console.WriteLine(item.Message.Content);
}
```

### 6.2 Threads

```csharp
AgentThread? thread = null;

foreach (var input in new[] { "My name is Ada.", "What is my name?" })
{
    await foreach (var item in agent.InvokeAsync(input, thread))
    {
        thread = item.Thread;                    // reuse for the next turn
        Console.WriteLine(item.Message.Content);
    }
}

if (thread is not null) await thread.DeleteAsync();
```

`ChatHistoryAgentThread` keeps history in memory; pass `new ChatHistoryAgentThread(existingHistory)` to resume a stored conversation. Service-backed agents (Azure AI, OpenAI Assistants/Responses) have their own thread types stored server-side.

### 6.3 Streaming and intermediate messages

```csharp
var options = new AgentInvokeOptions
{
    OnIntermediateMessage = msg =>
    {
        foreach (var call in msg.Items.OfType<FunctionCallContent>())
            Console.WriteLine($"  calling {call.PluginName}.{call.FunctionName}");
        return Task.CompletedTask;
    },
};

await foreach (var chunk in agent.InvokeStreamingAsync("Weather in Rome?", thread, options))
    Console.Write(chunk.Message.Content);
```

### 6.4 Other agent types

| Type | Package | Backed by |
|---|---|---|
| `ChatCompletionAgent` | `Agents.Core` | Any `IChatCompletionService` / `IChatClient` |
| `OpenAIResponseAgent`, `OpenAIAssistantAgent` | `Agents.OpenAI` (preview) | OpenAI Responses / Assistants APIs |
| `AzureAIAgent` | `Agents.AzureAI` (preview) | Azure AI Foundry Agent Service |
| `CopilotStudioAgent`, `BedrockAgent` | preview packages | Copilot Studio, Amazon Bedrock |
| `A2AAgent` | `Agents.A2A` (preview) | A remote A2A agent (section 10) |

`AgentGroupChat` with selection/termination strategies still exists but is superseded by the orchestrations below.

---

## 7. Multi-agent orchestration

All orchestrations share one pattern (preview package, `SKEXP0110`):

```csharp
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Orchestration;
using Microsoft.SemanticKernel.Agents.Orchestration.Sequential;
using Microsoft.SemanticKernel.Agents.Runtime.InProcess;

ChatCompletionAgent analyst = new() { Name = "Analyst", Description = "Extracts features",
    Instructions = "List the product's key features and audience.", Kernel = kernel };
ChatCompletionAgent writer = new() { Name = "Writer", Description = "Writes copy",
    Instructions = "Write 50 words of marketing copy from the analysis.", Kernel = kernel };

SequentialOrchestration orchestration = new(analyst, writer)
{
    ResponseCallback = response =>
    {
        Console.WriteLine($"[{response.AuthorName}] {response.Content}");
        return ValueTask.CompletedTask;
    },
};

InProcessRuntime runtime = new();
await runtime.StartAsync();

OrchestrationResult<string> result = await orchestration.InvokeAsync("An eco-friendly steel water bottle.", runtime);
string output = await result.GetValueAsync(TimeSpan.FromMinutes(2));
Console.WriteLine(output);

await runtime.RunUntilIdleAsync();
```

| Pattern | Type (namespace `...Orchestration.*`) |
|---|---|
| Sequential | `SequentialOrchestration(params Agent[])` |
| Concurrent | `ConcurrentOrchestration(params Agent[])` — result is `string[]` |
| Group chat | `GroupChatOrchestration(new RoundRobinGroupChatManager { MaximumInvocationCount = 5 }, agents)` |
| Hand-off | `HandoffOrchestration(OrchestrationHandoffs.StartWith(triage).Add(triage, billing, tech).Add(billing, triage, "Not a billing issue"), triage, billing, tech)` |
| Magentic | `MagenticOrchestration(new StandardMagenticManager(...), agents)` (package `Agents.Magentic`) |

Hand-off and group chat accept an `InteractiveCallback` for human input. `StreamingResponseCallback` streams tokens from whichever agent is active.

---

## 8. Vector stores and RAG

Vector data uses the `Microsoft.Extensions.VectorData` abstractions. Connectors are separate packages — `CommunityToolkit.VectorData.InMemory`, `.AzureAISearch`, `.Qdrant`, `.Redis`, `.SqlServer`, `.AzureCosmosDB`, `.Weaviate`, plus `Microsoft.SemanticKernel.Connectors.PgVector` and `.SqliteVec` — and are interchangeable.

```bash
dotnet add package CommunityToolkit.VectorData.InMemory
```

```csharp
using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;

public sealed class FaqEntry
{
    [VectorStoreKey] public string Id { get; set; } = "";
    [VectorStoreData(IsFullTextIndexed = true)] public string Question { get; set; } = "";
    [VectorStoreData] public string Answer { get; set; } = "";

    // A string-typed vector property is embedded automatically by the store's generator
    [VectorStoreVector(1536)] public string Embedding => $"{Question} {Answer}";
}

var generator = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
var store = new InMemoryVectorStore(new() { EmbeddingGenerator = generator });

var faq = store.GetCollection<string, FaqEntry>("faq");
await faq.EnsureCollectionExistsAsync();
await faq.UpsertAsync([
    new FaqEntry { Id = "1", Question = "How do I reset my password?", Answer = "Settings > Security." },
    new FaqEntry { Id = "2", Question = "What are your hours?", Answer = "9–5 UK time, Mon–Fri." },
]);

await foreach (var hit in faq.SearchAsync("forgot password", top: 1))
    Console.WriteLine($"{hit.Score:F3} {hit.Record.Answer}");
```

Filters are LINQ expressions: `new VectorSearchOptions<FaqEntry> { Filter = r => r.Question.Contains("password") }`.

**Search as a plugin for agents**

```csharp
using Microsoft.SemanticKernel.Data;

var textSearch = new VectorStoreTextSearch<FaqEntry>(faq);
kernel.Plugins.Add(textSearch.CreateWithGetTextSearchResults("FaqSearch"));
```

Records need a text result mapping (`[TextSearchResultValue]` on the answer property, or a custom mapper) for `CreateWithGetTextSearchResults`. The old `ISemanticTextMemory` / `IMemoryStore` API is obsolete.

---

## 9. Model Context Protocol (MCP)

Semantic Kernel consumes MCP through the official C# SDK (`ModelContextProtocol` package): list a server's tools and convert each one into a kernel function.

```bash
dotnet add package ModelContextProtocol
```

```csharp
using ModelContextProtocol.Client;

await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "GitHub",
    Command = "docker",
    Arguments = ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
}));

IList<McpClientTool> tools = await mcp.ListToolsAsync();
kernel.Plugins.AddFromFunctions("GitHub", tools.Select(t => t.AsKernelFunction()));
```

`McpClientTool` is a `Microsoft.Extensions.AI` `AIFunction`, and `AsKernelFunction()` (in `Microsoft.SemanticKernel`) adapts it. Use `HttpClientTransport` for Streamable HTTP servers.

To *expose* kernel functions as an MCP server, host them with the same SDK's server package (`ModelContextProtocol.AspNetCore`), registering each `KernelFunction` as a tool — Microsoft's samples include a ready-made adapter.

---

## 10. A2A

`Microsoft.SemanticKernel.Agents.A2A` (preview) provides `A2AAgent`, an SK `Agent` that talks to a remote agent over the Agent-to-Agent protocol, so a remote A2A agent can take part in threads and orchestrations like any local agent. Hosting SK agents *as* A2A servers is done with the A2A .NET SDK. Agent Framework has fuller first-party A2A support.

---

## 11. Structured output

```csharp
using System.Text.Json;
using Microsoft.SemanticKernel.Connectors.OpenAI;

public sealed record Recipe(string Title, string[] Ingredients, int Minutes);

var settings = new OpenAIPromptExecutionSettings { ResponseFormat = typeof(Recipe) };
var result = await kernel.InvokePromptAsync("Give me a quick pasta recipe.", new(settings));

Recipe recipe = JsonSerializer.Deserialize<Recipe>(result.ToString())!;
```

`ResponseFormat = typeof(T)` generates a strict JSON schema from the type; `"json_object"` requests unstructured JSON.

---

## 12. Filters

```csharp
public sealed class LoggingFilter(ILogger<LoggingFilter> logger) : IFunctionInvocationFilter
{
    public async Task OnFunctionInvocationAsync(FunctionInvocationContext context, Func<FunctionInvocationContext, Task> next)
    {
        logger.LogInformation("→ {Plugin}.{Function}", context.Function.PluginName, context.Function.Name);
        await next(context);
        logger.LogInformation("← {Result}", context.Result);
    }
}

public sealed class ApprovalFilter : IAutoFunctionInvocationFilter
{
    public async Task OnAutoFunctionInvocationAsync(AutoFunctionInvocationContext context, Func<AutoFunctionInvocationContext, Task> next)
    {
        if (context.Function.Name == "DeleteAccount")
        {
            context.Result = new FunctionResult(context.Function, "Blocked: requires human approval.");
            context.Terminate = true;
            return;
        }
        await next(context);
    }
}

builder.Services.AddSingleton<IFunctionInvocationFilter, LoggingFilter>();
builder.Services.AddSingleton<IAutoFunctionInvocationFilter, ApprovalFilter>();
// or: kernel.FunctionInvocationFilters.Add(...), kernel.AutoFunctionInvocationFilters.Add(...)
```

`IPromptRenderFilter` can inspect or rewrite the rendered prompt (redaction, prompt-injection checks) before it is sent.

---

## 13. ASP.NET Core, resilience and telemetry

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenAIChatCompletion("gpt-5-mini", builder.Configuration["OpenAI:ApiKey"]!);
builder.Services.AddSingleton<WeatherPlugin>();
builder.Services.AddTransient(sp =>
{
    var kernel = new Kernel(sp);
    kernel.Plugins.AddFromObject(sp.GetRequiredService<WeatherPlugin>(), "weather");
    return kernel;
});

var app = builder.Build();

app.MapPost("/ask", async (AskRequest req, Kernel kernel) =>
    (await kernel.InvokePromptAsync(req.Question)).ToString());

app.Run();

public sealed record AskRequest(string Question);
```

**Resilience.** The OpenAI SDK retries transient errors itself. For more control, add the standard handler to the `HttpClient` the connector uses (`Microsoft.Extensions.Http.Resilience`):

```csharp
builder.Services.ConfigureHttpClientDefaults(http => http.AddStandardResilienceHandler());
```

**Telemetry.** Semantic Kernel emits OpenTelemetry traces, metrics and logs under the `Microsoft.SemanticKernel*` sources:

```csharp
AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics", true);
// AppContext.SetSwitch("...EnableOTelDiagnosticsSensitive", true);  // includes prompts — dev only

builder.Services.AddOpenTelemetry()
    .WithTracing(t => t.AddSource("Microsoft.SemanticKernel*").AddOtlpExporter())
    .WithMetrics(m => m.AddMeter("Microsoft.SemanticKernel*").AddOtlpExporter());
```

Azure Monitor / Application Insights consumes the same data via `Azure.Monitor.OpenTelemetry.AspNetCore` (`builder.Services.AddOpenTelemetry().UseAzureMonitor()`).

---

## 14. Migrating to Microsoft Agent Framework

Every SK agent type has an `AsAIAgent()` extension that returns a MAF `Microsoft.Agents.AI.AIAgent`, so you can move orchestration code to MAF first and keep SK agents (and their plugins) underneath:

```csharp
using Microsoft.SemanticKernel.Agents;

Microsoft.Agents.AI.AIAgent mafAgent = agent.AsAIAgent();   // agent is an SK ChatCompletionAgent
```

| Semantic Kernel | Agent Framework (.NET) |
|---|---|
| `Kernel` + connectors | An `IChatClient`; no kernel object |
| `ChatCompletionAgent` | `chatClient.AsAIAgent(instructions: ..., tools: [...])` / `ChatClientAgent` |
| `[KernelFunction]` plugins | `AIFunctionFactory.Create(method)` tools |
| `AgentThread` | `AgentSession` |
| Filters | Middleware |
| Orchestrations, Process framework | Workflows |

See Microsoft's [migration guide](https://learn.microsoft.com/agent-framework/migration-guide/from-semantic-kernel/) and the [SK → MAF samples](https://github.com/microsoft/semantic-kernel/tree/main/dotnet/samples/AgentFrameworkMigration).

---

## 15. Removed and renamed APIs

| Old | Status | Use instead |
|---|---|---|
| `SequentialPlanner`, `StepwisePlanner`, `HandlebarsPlanner`, `FunctionCallingStepwisePlanner` | Packages deprecated (last 1.47.0-preview) | Function calling (section 5) |
| `ISemanticTextMemory`, `IMemoryStore`, `*MemoryStore` connectors | Obsolete | `Microsoft.Extensions.VectorData` collections (section 8) |
| `Microsoft.SemanticKernel.Connectors.InMemory` / `.AzureAISearch` / `.Qdrant` / ... | Deprecated (last 1.74.x-preview), renamed | `CommunityToolkit.VectorData.*` |
| `ITextEmbeddingGenerationService` | Obsolete | `IEmbeddingGenerator<string, Embedding<float>>` |
| `agent.InvokeAsync(chatHistory)` returning messages | Replaced in the agent GA | `InvokeAsync(message, thread)` returning `AgentResponseItem<ChatMessageContent>` |
| `AgentGroupChat` | Superseded | Orchestrations (section 7) |
| `Microsoft.SemanticKernel.Connectors.MCP`, `MCPClient`, `MCPServer` | Never existed | `ModelContextProtocol` SDK + `AsKernelFunction()` |
| `Microsoft.AgentFramework.UnifiedAgent`, `AgentRegistry` | Never existed | `Microsoft.Agents.AI` (see section 14) |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 1.81.0 | Rewritten. The previous edition documented planners and the `SemanticTextMemory` API as current, invented MCP and Agent Framework types (`MCPClient`, `MCPServer`, `UnifiedAgent`, `AgentRegistry`) and listed .NET 6 as supported. Now covers orchestrations, `Microsoft.Extensions.VectorData` with the `CommunityToolkit.VectorData` connectors, MCP via the official SDK, A2A agents, OpenTelemetry, and `AsAIAgent()` for migration. Version banner tracks the site's version registry. |
| April 2026 | 1.74.0 | Previous edition. |

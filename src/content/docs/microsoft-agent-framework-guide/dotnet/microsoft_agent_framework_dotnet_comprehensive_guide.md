---
title: "Microsoft Agent Framework (.NET) — Comprehensive Guide"
sidebar:
  label: "Comprehensive guide"
  order: 2
description: "Start here for Microsoft.Agents.AI in .NET: installation, configuration and DI, ChatClientAgent, structured output, agentic patterns, the harness agent and Semantic Kernel interop, with short summaries linking to the pages on providers, tools, MCP, middleware, sessions, workflows, orchestrations, hosting, observability and testing."
framework: "microsoft-agent-framework"
language: "dotnet"
---

**Framework version:** `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%; first written for the October 2025 preview). Every C# block on this page was compiled against that release.
**Target platform:** .NET 8.0+ (the core packages also target .NET Standard 2.0 / .NET Framework 4.7.2)

This page is the starting point: what the framework is, how to install and configure it, how a `ChatClientAgent` works, structured output, and the pieces that have no page of their own (agentic patterns, the harness agent, Copilot Studio, Semantic Kernel interop). Each major topic has its own page, summarised in a short section here and linked:

| Topic | Page |
| --- | --- |
| Every model provider and agent type, credentials, routing | [Providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/) |
| Function tools, the tool loop, approvals, hosted tools, shell and code execution | [Tools](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/) |
| MCP clients, hosted MCP, MCP servers | [MCP](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/) |
| Agent, function and chat-client middleware, retries and errors | [Middleware](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/) |
| Sessions, history, session stores, context providers, memory, compaction | [Sessions and memory](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/) |
| Foundry agents, agents as chat clients and tools, workflows as agents | [Advanced patterns](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_advanced/) |
| Executors, edges, request ports, checkpointing, sub-workflows, declarative workflows | [Workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) |
| Sequential, concurrent, group chat, handoff and Magentic orchestrations | [Orchestrations](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/) |
| OpenAI-compatible, A2A and AG-UI endpoints, DevUI, SSE, Foundry hosted agents, Azure Functions | [Hosting](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/) |
| OpenTelemetry traces, metrics and logs | [Observability](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/) |
| Tests without a model, evaluation | [Testing and evaluation](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/) |

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
dotnet add package Microsoft.Agents.AI.Anthropic --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.Hosting --prerelease      # DI registration, session stores
dotnet add package Microsoft.Agents.AI.Foundry --prerelease      # Microsoft Foundry agents
dotnet add package ModelContextProtocol                          # MCP servers as tools
```

The full package list, with which are stable and which are previews, is on the [providers page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#packages).

### Authentication and configuration

Keep secrets out of source code. For local development, user secrets:

```bash
dotnet user-secrets init
dotnet user-secrets set "AzureOpenAI:Endpoint" "https://your-resource.openai.azure.com"
```

```json
{
  "AzureOpenAI": {
    "DeploymentName": "%%model=openai-small%%"
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
var deployment = builder.Configuration["AzureOpenAI:DeploymentName"] ?? "%%model=openai-small%%";

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

The convenience extension `AsAIAgent(...)` is equivalent to `new ChatClientAgent(...)`. It exists on any `IChatClient` (in `Microsoft.Agents.AI`), on the OpenAI SDK's `ChatClient` and `ResponsesClient` (package `Microsoft.Agents.AI.OpenAI`), and on Foundry's `AIProjectClient`. (It was called `CreateAIAgent` in early previews, and `AgentThread` / `GetNewThread()` became `AgentSession` / `CreateSessionAsync()` before GA.)

### Agent lifecycle

Register the `IChatClient` (and the `AzureOpenAIClient` behind it) once as a singleton — it owns the HTTP connection pool. `ChatClientAgent` is cheap and thread-safe to share; per-conversation state is in the `AgentSession`, so a singleton agent serving many users is normal.

### Execution models

`RunAsync` takes a string, a `ChatMessage` or a list of messages, plus an optional `AgentSession` and `AgentRunOptions`, and returns an `AgentResponse` (`.Text`, `.Messages`, `.Usage`). `RunStreamingAsync` yields `AgentResponseUpdate`s as tokens arrive; serving them over server-sent events is under [your own streaming endpoint](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#your-own-streaming-endpoint). `RunAsync<T>` returns structured output (below).

### Testing agents

`ChatClientAgent` sits on `IChatClient`, so a fake chat client (a small class implementing `IChatClient`, or a mock) is the natural seam: no live model calls and no framework internals to fake. The [testing and evaluation page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/#a-scripted-chat-client) builds a scripted chat client and uses it to test agents, workflows and hosted endpoints.

---

## Multi-Agent Systems

Multi-agent work lives in `Microsoft.Agents.AI.Workflows`, at two levels. `AgentWorkflowBuilder` assembles the prebuilt orchestrations from a list of agents, and `WorkflowBuilder` builds any graph you like from executors and agent nodes. Both produce an ordinary `Workflow`, which you run with `InProcessExecution`, stream, checkpoint, pause for human input, or wrap as an agent with `AsAIAgent()`.

| Pattern | Entry point | Covered on |
| --- | --- | --- |
| Sequential | `AgentWorkflowBuilder.BuildSequential(...)` | [Orchestrations: sequential](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#sequential) |
| Concurrent | `AgentWorkflowBuilder.BuildConcurrent(...)` | [Orchestrations: concurrent](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#concurrent) |
| Group chat | `AgentWorkflowBuilder.CreateGroupChatBuilderWith(managerFactory)` | [Orchestrations: group chat](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#group-chat) |
| Handoff | `AgentWorkflowBuilder.CreateHandoffBuilderWith(initialAgent)` | [Orchestrations: handoff](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#handoff) |
| Magentic | `AgentWorkflowBuilder.CreateMagenticBuilderWith(managerAgent)` | [Orchestrations: Magentic](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#magentic) |
| Custom graph | `new WorkflowBuilder(start).AddEdge(...).Build()` | [Workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) |

Agent nodes in .NET wait for a `TurnToken` before calling the model, so a run you start yourself needs `run.TrySendMessageAsync(new TurnToken(emitEvents: true))`; a workflow wrapped with `AsAIAgent()` sends it for you. The [orchestrations page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#choosing-a-pattern) explains when to use each pattern, and the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) covers executors, shared state, [request ports](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#human-input-with-request-ports) and [checkpointing](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#checkpointing).

Don't share one `AgentSession` across two different agents to "pass context"; a session belongs to the agent that created it. Use a workflow, or pass the first agent's output into the second agent's input.

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

The [tools page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/) covers schemas and parameter binding, the function loop and its limits, tools that change during a run, agents as tools, and the shell and code-execution packages.

### Hosted tools

Provider-run tools (`HostedWebSearchTool`, `HostedCodeInterpreterTool`, `HostedFileSearchTool`, `HostedImageGenerationTool`, `HostedMcpServerTool`) and Foundry's own (`FoundryAITool`) are on the [tools page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#hosted-tools).

### Tool approval

Wrap a function in `ApprovalRequiredAIFunction` and the run stops with a `ToolApprovalRequestContent` instead of calling it; you answer with `request.CreateResponse(approved)` on the next run in the same session. The round trip, what the framework enforces around it, and standing approvals with `ToolApprovalAgent` are under [approvals](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#approvals).

### Errors in tools

A tool that throws does not end the run: the model is told "Error: Function failed." without the exception message, unless you enable `IncludeDetailedErrors`. See [what the model sees when a tool fails](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#what-the-model-sees-when-a-tool-fails).

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

How it works: `RunAsync<T>` is defined on `AIAgent`, so it works with any agent. It builds a JSON schema for `T` (`ChatResponseFormat.ForJsonSchema<T>()`), puts it in `AgentRunOptions.ResponseFormat`, and runs normally. Schemas that are not objects (a `List<T>`, a string, a number) are wrapped in an object, because providers require an object at the top level; `AgentResponse<T>.IsWrappedInObject` records that and `Result` unwraps it. The JSON is only parsed when you read `Result`: if the response contained no JSON it throws `InvalidOperationException`, and if the JSON does not fit `T` the `JsonException` from `System.Text.Json` propagates, so catch both and retry or fall back. `response.Text` still holds the raw JSON. Pass a `JsonSerializerOptions` (from a source-generated context for trimming and AOT) as the `serializerOptions` argument to control naming and converters.

To ask for JSON without deserialising, set `AgentRunOptions.ResponseFormat` yourself (`ChatResponseFormat.Json`, or `ChatResponseFormat.ForJsonSchema(schema)`), or set `ChatOptions.ResponseFormat` on the agent for every run. Whether a schema is enforced or only requested depends on the provider: OpenAI and Azure OpenAI enforce it, other `IChatClient`s may treat it as a hint.

---

## Model Context Protocol (MCP)

MCP support comes from the official C# SDK (`ModelContextProtocol`, `ModelContextProtocol.AspNetCore`). Its client tools are `AIFunction`s, so `await mcpClient.ListToolsAsync()` goes straight into an agent's `tools:`. The [MCP page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/) covers stdio and Streamable HTTP connections, filtering and approving MCP tools, authentication, long-running MCP tasks, hosted MCP, and exposing tools or whole agents as MCP servers.

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

For long or open-ended plans, the [Magentic orchestration](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#magentic) does the planning, progress tracking and re-planning for you.

### Routing

A router classifies the request with structured output and dispatches to a specialist; see Recipe 5 in the [.NET recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-5-router-to-specialist-agents). When specialists need to pass the conversation back and forth, use the [handoff orchestration](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#handoff).

### Reflection

A writer drafts, a critic reviews, the writer revises in its own session (so it remembers its draft). Recipe 12 in the [recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-12-writer-and-critic-with-a-round-cap) runs the same loop as a workflow with a round cap.

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

By default a `ChatClientAgent` keeps the conversation in the session's state bag (`InMemoryChatHistoryProvider`), or, for services that store it themselves (the Responses API, Foundry), as a conversation ID. Persist a session with `agent.SerializeSessionAsync(session)` and restore it with `agent.DeserializeSessionAsync(json)`. Database-backed history (`CosmosChatHistoryProvider`, `ValkeyChatHistoryProvider`, or your own `ChatHistoryProvider`), retrieval (`TextSearchProvider`), long-term memory (`ChatHistoryMemoryProvider`, `FileMemoryProvider`, `FoundryMemoryProvider`) and compaction are on the [sessions and memory page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/).

---

## Context Engineering

Everything the model sees on a turn is assembled from the agent's instructions, the session history, and the output of its **context providers**: `AIContextProvider`s that add instructions, messages or tools on each run and observe the result afterwards. Writing one, how their output is ordered and stored, and the built-in providers are under [context providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#context-providers).

### Multi-tenant isolation

Sessions are the isolation boundary: never let a request choose a session that belongs to someone else. Key stored sessions by the authenticated user (and tenant), never by an ID from the request body. With `Microsoft.Agents.AI.Hosting`, `builder.AddAIAgent(...).WithInMemorySessionStore()` (or another session store) together with `builder.Services.UseClaimsBasedAgentIsolation()` partitions stored sessions by a claim of the signed-in user, `ClaimTypes.NameIdentifier` by default. See [session stores](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#session-stores-hosting).

---

## More agent types

`ChatClientAgent` covers any `IChatClient`. `Microsoft.Agents.AI.Harness` adds an agent with planning, memory and compaction already wired in; other agent types (Anthropic, GitHub Copilot, A2A, Copilot Studio, Foundry-defined agents, your own `AIAgent` subclass) are on the [providers page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/).

### `HarnessAgent`

`Microsoft.Agents.AI.Harness` wraps a chat client in an agent that already has the harness context providers: a todo list, plan/execute modes, file memory, skills, background sub-agents, standing tool approvals, compaction and OpenTelemetry. Each one can be turned off (`DisableTodoProvider`, `DisableAgentModeProvider`, `DisableFileMemory`, `DisableAgentSkillsProvider`, `DisableCompaction`, `DisableWebSearch`, `DisableOpenTelemetry`) or configured through `HarnessAgentOptions`.

```bash
dotnet add package Microsoft.Agents.AI.Harness --version %%version=Microsoft.Agents.AI%%
```

```csharp
#pragma warning disable MAAI001   // the context-window settings are experimental
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

IChatClient chatClient = new AzureOpenAIClient(
        new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!),
        new DefaultAzureCredential())
    .GetChatClient("%%model=openai%%")
    .AsIChatClient();

HarnessAgent agent = chatClient.AsHarnessAgent(new HarnessAgentOptions
{
    Name = "research-agent",
    ChatOptions = new ChatOptions { Instructions = "Focus on primary sources." },
    MaxContextWindowTokens = 200_000,   // enables context-window compaction
    MaxOutputTokens = 32_000,
    DisableWebSearch = true,
});

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("Compare three heat-pump designs and summarise.", session);
Console.WriteLine(response.Text);
```

Setting `MaxContextWindowTokens` and `MaxOutputTokens` (both experimental, `MAAI001`, in %%version=Microsoft.Agents.AI%%) turns on compaction sized to the model; `LoopEvaluators` keeps the agent working until a condition (for example, no open todos) is met. The same providers (`TodoProvider`, `AgentModeProvider`, `AgentSkillsProvider`, `BackgroundAgentsProvider`, file memory and file access) live in `Microsoft.Agents.AI` and can be added to a plain `ChatClientAgent` individually through `ChatClientAgentOptions.AIContextProviders`.

### Anthropic and GitHub Copilot

`Microsoft.Agents.AI.Anthropic` adds `AsAIAgent` to the Anthropic SDK client (including its beta service and Claude in Foundry), and `Microsoft.Agents.AI.GitHub.Copilot` exposes the GitHub Copilot SDK as `GitHubCopilotAgent`. Both are on the providers page: [Anthropic](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#anthropic), [GitHub Copilot](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#github-copilot).

---

## Evaluation

Evaluation is in the core package. `LocalEvaluator` runs deterministic checks (`EvalChecks`, or your own through `FunctionEvaluator.Create`), `agent.EvaluateAsync(queries, evaluator)` runs the agent on each query and scores the replies, and the results have assertion helpers such as `AssertAllPassed`, so the same code serves as a unit test. Model-graded evaluators from `Microsoft.Extensions.AI.Evaluation` and Foundry's `FoundryEvals` plug into the same calls. The checks, model-graded evaluation, workflow evaluation and running the suite in CI are on the [testing and evaluation page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/#evaluating-with-localevaluator).

---

## Local testing with DevUI

`Microsoft.Agents.AI.DevUI` (preview) is a browser UI for chatting with your registered agents and workflows during development, served from your own app with `AddDevUI()` and `MapDevUI()` on top of the OpenAI-compatible endpoints. Map it only in development; set-up and options are on the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#devui).

---

## Copilot Studio Integration

The integration runs in both directions:

- **Calling a Copilot Studio agent from .NET:** `Microsoft.Agents.AI.CopilotStudio` (preview) provides `CopilotStudioAgent`, an `AIAgent` over the Copilot Studio client; see the [providers page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#copilot-studio).
- **Calling your .NET tools or agents from Copilot Studio:** expose them as an [MCP server](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/#exposing-tools-and-agents-as-mcp-servers) and add it in Copilot Studio as an MCP tool, or publish an OpenAPI description and add it as a custom connector.

---

## Azure AI Integration

- **Azure OpenAI / Foundry models** — the primary model providers: the OpenAI client against the Azure v1 endpoint, `Azure.AI.OpenAI`, or `Microsoft.Agents.AI.Foundry` ([providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#azure-openai)).
- **Foundry Agent Service** — service-managed agents with server-side conversations and hosted tools, surfaced as `FoundryAgent` ([providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#agents-defined-in-foundry), [advanced](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_advanced/)).
- **Azure AI Search** — retrieval through `TextSearchProvider`, `FoundryAITool.CreateAzureAISearchTool` or `HostedFileSearchTool`.
- **Cosmos DB and Blob Storage** — chat history (`CosmosChatHistoryProvider`) and session stores (`AzureBlobAgentSessionStore`); see [sessions and memory](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/).
- **Azure AI Content Safety** — call it from agent middleware; see the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#azure-ai-content-safety-integration).
- **Application Insights** — `.AsBuilder().UseOpenTelemetry(...)` plus the Azure Monitor exporter; see [observability](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/#azure-monitor).

### Cost and governance

- Token usage is on every `AgentResponse.Usage`; export it as a metric per agent and tenant.
- Use Azure budgets, per-deployment quotas, and Azure Policy (allowed models, no key-based auth).

---

## Semantic Kernel Integration

The Agent Framework is the successor to Semantic Kernel's agent APIs, and both sit on `Microsoft.Extensions.AI`, so migration is mostly mechanical.

### Reusing Semantic Kernel plugins

In current Semantic Kernel releases `KernelFunction` derives from `AIFunction`, so existing plugin functions are agent tools as they are. (`KernelFunction.AsAIFunction()` is obsolete as of Semantic Kernel 1.81.0.)

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
        IList<AITool> tools = plugin.Select(f => (AITool)f).ToList();   // a KernelFunction is an AIFunction

        return new ChatClientAgent(chatClient, instructions: "Use your tools.", tools: tools);
    }
}
```

### Migration strategy

- **Tools:** pass plugin functions as tools as above, then move them to plain methods with `[Description]` at your own pace.
- **Agents:** `ChatCompletionAgent` → `ChatClientAgent`; `AgentThread` → `AgentSession`; `InvokeAsync` → `RunAsync`.
- **Orchestration:** SK planners and `AgentGroupChat` → `Microsoft.Agents.AI.Workflows` (sequential, concurrent, handoff, group chat, Magentic, or a custom graph).
- **History:** move stored chat history to a `ChatHistoryProvider` (in-memory, Cosmos DB, or your own).

Microsoft's [Semantic Kernel migration guide](https://learn.microsoft.com/agent-framework/migration-guide/from-semantic-kernel/) has the full mapping.

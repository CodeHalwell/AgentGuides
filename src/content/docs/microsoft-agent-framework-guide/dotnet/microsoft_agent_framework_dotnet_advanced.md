---
title: "Microsoft Agent Framework (.NET) — Advanced Patterns"
description: "Service-managed agents in Microsoft Foundry and the classic Persistent Agents service, agent middleware with AIAgentBuilder and DelegatingAIAgent, and workflows exposed as agents."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Advanced patterns"
  order: 40
---

Checked against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% on %%versions-checked%%; every C# block compiles against that release. Deployment, resilience and security are on the [production guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_production_guide/).

This page covers:

- Agents whose definition lives in Microsoft Foundry (`Microsoft.Agents.AI.Foundry`), and the older Persistent Agents service.
- Wrapping any agent with your own behaviour (`AIAgentBuilder.Use`, `DelegatingAIAgent`).
- Turning a workflow into an `AIAgent`.

## Microsoft Foundry agents

`Microsoft.Agents.AI.Foundry` (preview) adds `AsAIAgent` overloads to `AIProjectClient` (package `Azure.AI.Projects`). There are two kinds of agent:

- **A model in your project, agent defined in code.** `AsAIAgent(model, instructions, ...)` returns a `ChatClientAgent` over the project's Responses endpoint. Instructions and tools stay in your code.
- **An agent defined in Foundry.** `AsAIAgent(agentReference)` (or an agent endpoint, record or version) returns a `FoundryAgent` that runs the agent as configured in the Foundry portal, with its hosted tools.

```bash
dotnet add package Microsoft.Agents.AI.Foundry --prerelease
dotnet add package Azure.Identity
```

```csharp
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;

var project = new AIProjectClient(
    new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!),
    new DefaultAzureCredential());

// Agent defined in code, model deployed in the project.
AIAgent writer = project.AsAIAgent(
    model: "%%model=openai-small%%",
    instructions: "You are a helpful writing assistant.",
    name: "writer");

AgentSession session = await writer.CreateSessionAsync();
AgentResponse response = await writer.RunAsync("Draft a summary of LangGraph.", session);
Console.WriteLine(response.Text);
```

Either kind is an ordinary `AIAgent`: sessions, streaming, middleware, workflows and hosting work unchanged. Foundry also offers evaluation (`FoundryEvals`, see [evaluation](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#evaluation)) and memory stores (`FoundryMemoryProvider`) through the same package. To run your own code *as* a Foundry hosted agent, see the [production guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_production_guide/#foundry-hosted-agents).

### Classic Persistent Agents

Projects that still use the Azure AI Agents service (`Azure.AI.Agents.Persistent`, server-side threads and runs) can bring those agents in with `Microsoft.Agents.AI.AzureAI.Persistent` (preview). Microsoft Learn lists Agents (classic) as deprecated, with retirement on 31 March 2027, and the classic service does not take models later than GPT-5, so use the Foundry package for new work.

```bash
dotnet add package Microsoft.Agents.AI.AzureAI.Persistent --prerelease
```

```csharp
using Azure.AI.Agents.Persistent;
using Azure.Identity;
using Microsoft.Agents.AI;

var client = new PersistentAgentsClient(
    Environment.GetEnvironmentVariable("AZURE_AI_PROJECT_ENDPOINT")!, new DefaultAzureCredential());

// Create the server-side agent and wrap it; use client.GetAIAgentAsync(agentId) for an existing one.
AIAgent agent = await client.CreateAIAgentAsync(
    model: Environment.GetEnvironmentVariable("MODEL_DEPLOYMENT_NAME")!,   // a model the classic service supports
    name: "writer",
    instructions: "You are a helpful writing assistant.");

AgentSession session = await agent.CreateSessionAsync();   // backed by a service-side thread
Console.WriteLine((await agent.RunAsync("Draft a summary of LangGraph.", session)).Text);
```

## Wrapping agents with your own behaviour

`agent.AsBuilder().Use(...)` adds agent-level middleware: a function that sees the messages, session and options of each run and decides whether and how to call the inner agent. Pass `runFunc` and `runStreamingFunc` separately, or one shared function that wraps both. This is the place for audit logging, input screening, per-tenant limits and caching.

```csharp
using System.Diagnostics;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class TimingMiddleware
{
    public static AIAgent WithTiming(this AIAgent agent) =>
        agent.AsBuilder()
            .Use(
                runFunc: async (messages, session, options, inner, ct) =>
                {
                    var stopwatch = Stopwatch.StartNew();
                    AgentResponse response = await inner.RunAsync(messages, session, options, ct);
                    Console.WriteLine($"{inner.Name}: {stopwatch.ElapsedMilliseconds} ms, {response.Usage?.TotalTokenCount} tokens");
                    return response;
                },
                runStreamingFunc: null)
            .Build();
}
```

For something you want to unit-test or register in DI, derive from `DelegatingAIAgent` instead; the [.NET feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/#a-reusable-safety-wrapper) has a Content Safety wrapper written that way. Chat-level middleware (around each model call) goes on the `IChatClient` with `ChatClientBuilder.Use(...)`, and tool-level approval with `ApprovalRequiredAIFunction` (see [tool approval](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#tool-approval)).

## Workflows as agents

`workflow.AsAIAgent(...)` turns a workflow whose input is a list of chat messages (any `AgentWorkflowBuilder` workflow qualifies) into an `AIAgent`. A multi-agent pipeline can then be called, hosted or used as a tool exactly like a single agent.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class Pipeline
{
    public static async Task<string> RunAsync(IChatClient chatClient, string topic)
    {
        AIAgent researcher = new ChatClientAgent(chatClient, instructions: "List the key facts about the topic.", name: "researcher");
        AIAgent writer = new ChatClientAgent(chatClient, instructions: "Turn the facts into one paragraph.", name: "writer");

        AIAgent pipeline = AgentWorkflowBuilder.BuildSequential(researcher, writer)
            .AsAIAgent(name: "research-pipeline");

        AgentResponse response = await pipeline.RunAsync(topic);
        return response.Text;
    }
}
```

Requests raised inside the workflow (a `RequestPort` waiting for a person, or a tool waiting for approval) come back on the agent's response as user-input requests. The [.NET feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/#human-in-the-loop) covers request ports and checkpointing; the orchestration patterns (sequential, concurrent, handoff, group chat, Magentic) are in the [comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#multi-agent-systems).

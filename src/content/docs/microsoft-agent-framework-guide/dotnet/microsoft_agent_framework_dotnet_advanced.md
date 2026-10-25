---
title: "Microsoft Agent Framework (.NET) — Advanced Patterns"
description: "Workflows exposed as agents, agents exposed as chat clients and tools, and pointers to Foundry-defined agents and agent middleware in Microsoft.Agents.AI."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Advanced patterns"
  order: 8
---

Checked against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% on %%versions-checked%%; every C# block compiles against that release. Deployment, resilience and security are on the [production guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_production_guide/).

Everything in the framework converts to and from `AIAgent`, which is what makes the pieces compose. This page collects the conversions:

| From | To | How | Covered |
| --- | --- | --- | --- |
| An `IChatClient` or provider client | `AIAgent` | `AsAIAgent(...)`, `new ChatClientAgent(...)` | [Providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/) |
| A Foundry agent definition | `AIAgent` | `project.AsAIAgent(agentVersion)` → `FoundryAgent` | [Providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#agents-defined-in-foundry) |
| An `AIAgent` | A wrapped `AIAgent` | `agent.AsBuilder().Use(...)`, `DelegatingAIAgent` | [Middleware](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#agent-run-middleware) |
| An `AIAgent` | A tool | `agent.AsAIFunction()` | [Tools](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#agents-as-tools) |
| An `AIAgent` | An MCP tool | `McpServerTool.Create(agent.AsAIFunction())` | [MCP](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/#an-agent-as-an-mcp-tool) |
| An `AIAgent` | An `IChatClient` | `agent.AsIChatClient()` (experimental) | [below](#agents-as-chat-clients) |
| A workflow | `AIAgent` | `workflow.AsAIAgent(...)` | [below](#workflows-as-agents) |

## Microsoft Foundry agents

`Microsoft.Agents.AI.Foundry` (preview) gives you two kinds of agent from an `AIProjectClient`: a `ChatClientAgent` over a model in the project, with the agent defined in your code, and a `FoundryAgent` over an agent defined and versioned in Foundry with its hosted tools. Both, the server-side conversation options, hosted-agent sessions, and the deprecated classic Persistent Agents package, are covered on the [providers page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#microsoft-foundry). Foundry's hosted tools are on the [tools page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#foundry-tools), Foundry Memory on the [sessions and memory page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#foundry-memory), and running your own code *as* a Foundry hosted agent on the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#microsoft-foundry-hosted-agents).

## Wrapping agents with your own behaviour

Agent-level middleware (`agent.AsBuilder().Use(...)` with run delegates, function-invocation callbacks, or a `DelegatingAIAgent` subclass) and chat-client middleware (`ChatClientBuilder.Use(...)`) are on the [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/), with ordering, streaming, short-circuiting, retries and error handling.

## Agents as chat clients

`agent.AsIChatClient()` (experimental `MAAI001`) presents an agent as an `IChatClient`, for libraries and frameworks that accept only chat clients. By default it accepts only a `ChatClientAgent` (or a decorator around one), because only then is the mapping from `ChatOptions` well defined. Pass a session to keep the agent's history across calls through the client; without one, each call stands alone and the caller sends the history. `allowNonChatClientAgents: true` wraps any agent (A2A, Copilot, a workflow agent), but then only `ChatOptions.ResponseFormat` reaches the agent; tools, instructions, temperature and the other options are silently ignored.

```csharp
#pragma warning disable MAAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You answer in one sentence.", name: "one-liner");

AgentSession session = await agent.CreateSessionAsync();
IChatClient asClient = agent.AsIChatClient(session);   // history kept in the session

ChatResponse first = await asClient.GetResponseAsync("What is HTTP?");
ChatResponse second = await asClient.GetResponseAsync("And HTTPS?");
Console.WriteLine(first.Text);
Console.WriteLine(second.Text);
Console.WriteLine($"conversation id reported to the caller: {second.ConversationId}");
```

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

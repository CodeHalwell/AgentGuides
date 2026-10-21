---
title: "Microsoft Agent Framework Comprehensive Guide"
description: "Language-neutral overview of the Microsoft Agent Framework — architecture, packages, authentication, simple agents, tools, execution models, testing and multi-agent systems — with verified Python and C# side by side."
framework: "microsoft-agent-framework"
---

**Current releases:** `agent-framework` %%version=agent-framework%% (Python) · `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (.NET) — the two version independently. Reviewed %%versions-checked%%: every Python example was import- and signature-checked (and the agent logic run against a stub chat client), and every C# example compiled, against those releases.
**Target audience:** developers from beginner to expert level

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Fundamentals](#core-fundamentals)
3. [Simple Agents](#simple-agents)
4. [Multi-Agent Systems](#multi-agent-systems)

This page is the language-neutral overview. Tools, structured output, MCP, memory, context engineering and the Azure/Copilot Studio integrations are covered in depth per language in the [Python guide](/microsoft-agent-framework-guide/python/) and the [.NET guide](/microsoft-agent-framework-guide/dotnet/), and across both in the [A2A protocol](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/), [graph workflows & declarative agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) and [enterprise](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/) pages.

---

## Introduction

The Microsoft Agent Framework (generally available since April 2026) is the open-source SDK that unifies **Semantic Kernel**'s agent features and **AutoGen** into one framework for building agents and multi-agent workflows in .NET and Python.

### Key objectives

- **One programming model across languages:** the same concepts — agents, sessions, tools, middleware, context providers, workflows — in Python and .NET.
- **Provider choice:** Azure OpenAI, OpenAI, Microsoft Foundry, Anthropic, Ollama and others behind one chat-client abstraction.
- **Production readiness:** OpenTelemetry instrumentation, checkpointed workflows, human-in-the-loop, and hosting integrations.
- **Open protocols:** MCP for tools, A2A for agent-to-agent calls, AG-UI for front ends.

### Target use cases

- **Conversational AI** with tools and memory
- **Autonomous task execution** with planning and approvals
- **Multi-agent systems** — sequential, concurrent, handoff, group chat, Magentic, or custom graphs
- **Enterprise integration** with business systems through tools and MCP servers
- **Knowledge assistance** grounded in organisational content (RAG)

---

## Core Fundamentals

### Architecture

The framework is layered, and each layer is replaceable:

```
┌──────────────────────────────────────────────────────────────┐
│ Workflows: graphs of executors, orchestrations, checkpoints  │
├──────────────────────────────────────────────────────────────┤
│ Agents: Agent / ChatClientAgent, sessions, middleware,       │
│         context providers (memory, RAG), tools               │
├──────────────────────────────────────────────────────────────┤
│ Chat clients: OpenAI, Azure OpenAI, Foundry, Anthropic, ...  │
│ (.NET: Microsoft.Extensions.AI IChatClient)                  │
├──────────────────────────────────────────────────────────────┤
│ Cross-cutting: OpenTelemetry, hosting (HTTP, A2A, AG-UI),    │
│                MCP, Azure Functions / durable agents         │
└──────────────────────────────────────────────────────────────┘
```

| Concept | Python | .NET |
| --- | --- | --- |
| Agent | `agent_framework.Agent` | `AIAgent` (abstract), `ChatClientAgent` |
| Model client | `OpenAIChatClient`, `FoundryChatClient`, ... | `IChatClient` (`Microsoft.Extensions.AI`) |
| Conversation state | `AgentSession` (`agent.create_session()`) | `AgentSession` (`agent.CreateSessionAsync()`) |
| Tools | `@tool` functions, MCP tools | `AIFunctionFactory.Create(...)`, MCP tools |
| Middleware | `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware` | `agent.AsBuilder().Use(...)`, `ChatClientBuilder.Use(...)` |
| Memory / RAG | `ContextProvider`, history providers | `AIContextProvider`, `ChatHistoryProvider` |
| Workflows | `WorkflowBuilder`, `SequentialBuilder`, ... | `WorkflowBuilder`, `AgentWorkflowBuilder` |
| Telemetry | `agent_framework.observability` | `.UseOpenTelemetry()` |

### Installation

**Python** (3.10+):

```bash
pip install agent-framework==%%version=agent-framework%%   # core plus the common integrations
```

`agent-framework` is a meta-package that pulls in `agent-framework-core` and the provider packages (`agent-framework-openai`, `agent-framework-foundry`, `agent-framework-orchestrations`, ...). To keep the dependency set small, install `agent-framework-core` plus only the provider packages you use.

**.NET** (8.0+):

```bash
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.OpenAI --version %%version=Microsoft.Agents.AI%%   # AsAIAgent() for the OpenAI SDK
dotnet add package Azure.AI.OpenAI --prerelease
dotnet add package Azure.Identity
dotnet add package Microsoft.Agents.AI.Workflows --version %%version=Microsoft.Agents.AI%%  # multi-agent workflows
```

### Integration with Azure AI, Copilot Studio and Semantic Kernel

- **Microsoft Foundry / Azure OpenAI** — model hosting, Foundry Agent Service (server-side agents and hosted tools), evaluation. Python: `agent_framework.foundry.FoundryChatClient`; .NET: `Microsoft.Agents.AI.Foundry` and `Azure.AI.OpenAI`.
- **Azure AI Search** — retrieval through context providers (`agent_framework.azure.AzureAISearchContextProvider`; .NET `TextSearchProvider`).
- **Copilot Studio** — call a Copilot Studio agent as a framework agent (`agent_framework.microsoft.CopilotStudioAgent`; .NET `Microsoft.Agents.AI.CopilotStudio`), or expose your tools to Copilot Studio as an MCP server.
- **Semantic Kernel** — the framework is SK's successor for agents; SK plugins convert to framework tools (`KernelFunction.AsAIFunction()` in .NET). See Microsoft's [Semantic Kernel migration guide](https://learn.microsoft.com/agent-framework/migration-guide/from-semantic-kernel/).

### Authentication

Authentication belongs to the **chat client**. Use Microsoft Entra ID credentials from `azure-identity` / `Azure.Identity`: Azure CLI sign-in for local development, managed identity in Azure.

```python
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import AzureCliCredential, ManagedIdentityCredential

# Local development: uses your `az login` session.
dev_client = FoundryChatClient(credential=AzureCliCredential())   # FOUNDRY_PROJECT_ENDPOINT, FOUNDRY_MODEL

# In Azure: the app's managed identity, against an Azure OpenAI resource.
prod_client = OpenAIChatClient(
    model="%%model=openai-small%%",                                   # the deployment name
    azure_endpoint="https://your-resource.openai.azure.com",
    credential=ManagedIdentityCredential(),
)
```

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;

// System-assigned managed identity (in Azure); use AzureCliCredential locally.
var client = new AzureOpenAIClient(
    new Uri("https://your-resource.openai.azure.com/"),
    new ManagedIdentityCredential(ManagedIdentityId.SystemAssigned));
```

The [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#microsoft-entra-id-authentication-patterns) covers workload identity, multi-tenant apps and government clouds.

### Environment and configuration

**Python** clients read their settings from environment variables (or a `.env` file via `env_file_path=`), so code stays free of endpoints:

```bash
# .env
FOUNDRY_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
FOUNDRY_MODEL=gpt-4o-mini
# or, for OpenAI / Azure OpenAI with OpenAIChatClient:
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

**.NET** uses the standard configuration system and DI:

```json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com",
    "DeploymentName": "%%model=openai-small%%"
  }
}
```

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .AddEnvironmentVariables()
    .Build();

var services = new ServiceCollection();
services.AddChatClient(
    new AzureOpenAIClient(new Uri(config["AzureOpenAI:Endpoint"]!), new DefaultAzureCredential())
        .GetChatClient(config["AzureOpenAI:DeploymentName"]!)
        .AsIChatClient());
```

### Development tools

- **DevUI** (`agent-framework-devui`, `Microsoft.Agents.AI.DevUI`) — a local web UI for chatting with agents and stepping through workflows.
- **AI Toolkit for VS Code** — model playground, tracing viewer (`configure_otel_providers(vs_code_extension_port=...)` sends traces to it) and agent scaffolding.
- **Azure Developer CLI templates** for deploying agents to Container Apps or Foundry.

---

## Simple Agents

### Creating a basic agent

**Python** — `Agent` wraps a chat client with instructions:

```python
import asyncio

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential


async def main() -> None:
    async with AzureCliCredential() as credential:
        agent = Agent(
            client=FoundryChatClient(credential=credential),
            name="python-helper",
            instructions=(
                "You are a helpful assistant specialising in Python programming. "
                "Give clear, concise explanations with code examples when appropriate."
            ),
        )
        response = await agent.run("How do I implement async/await in Python?")
        print(f"Assistant: {response.text}")


asyncio.run(main())
```

**.NET** — `AsAIAgent` on the OpenAI SDK's `ChatClient` returns a `ChatClientAgent`:

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var client = new AzureOpenAIClient(new Uri("https://your-resource.openai.azure.com/"), new DefaultAzureCredential());

AIAgent agent = client.GetChatClient("%%model=openai-small%%")
    .AsAIAgent(instructions: "You are a helpful assistant that explains concepts clearly.");

AgentResponse result = await agent.RunAsync("Explain machine learning to a beginner.");
Console.WriteLine(result.Text);
```

Both agents are stateless between calls unless you pass a **session** (below). In .NET, `AIAgent` is the abstract base type; `ChatClientAgent` is the implementation over any `IChatClient`. There is no separate "stateless" agent class.

### Agents with tools

**Python** — decorate a function with `@tool`; type hints and `Annotated` descriptions become the schema:

```python
import asyncio
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential


@tool(description="Get the current weather for a specified location")
async def get_weather(
    location: Annotated[str, "City name, e.g. 'London'"],
    unit: Annotated[str, "Temperature unit: 'celsius' or 'fahrenheit'"] = "celsius",
) -> str:
    weather_data = {"London": "15°C, cloudy", "Paris": "16°C, sunny", "Amsterdam": "14°C, rainy"}
    return f"Weather in {location}: {weather_data.get(location, 'unknown')}"


async def main() -> None:
    async with AzureCliCredential() as credential:
        agent = Agent(
            client=FoundryChatClient(credential=credential),
            tools=[get_weather],
            instructions="You are a weather assistant.",
        )
        response = await agent.run("What's the weather in London and Paris?")
        print(response.text)   # the agent called get_weather twice


asyncio.run(main())
```

**.NET** — `AIFunctionFactory.Create` turns a method into a tool:

```csharp
using System.ComponentModel;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Chat;

public static class WeatherExample
{
    [Description("Get current weather for a location")]
    public static string GetWeather(
        [Description("The city name")] string location,
        [Description("Temperature unit: celsius or fahrenheit")] string unit = "celsius")
    {
        var weatherData = new Dictionary<string, string>
        {
            ["London"] = "15°C, cloudy",
            ["Paris"] = "16°C, sunny",
            ["Amsterdam"] = "14°C, rainy",
        };
        return $"Weather in {location}: {weatherData.GetValueOrDefault(location, "unknown")}";
    }

    public static async Task RunAsync()
    {
        var client = new AzureOpenAIClient(new Uri("https://your-resource.openai.azure.com/"), new DefaultAzureCredential());

        AIAgent agent = client.GetChatClient("%%model=openai-small%%").AsAIAgent(
            instructions: "You are a weather assistant.",
            tools: [AIFunctionFactory.Create(GetWeather)]);

        AgentResponse result = await agent.RunAsync("What's the weather in London?");
        Console.WriteLine(result.Text);
    }
}
```

### Agent lifecycle

```
Create client  →  Create agent  →  Create session(s)  →  Run  →  Persist / close
(once per app)    (cheap; share)   (one per conversation)        (sessions; MCP tools)
```

- Create the **chat client** once and share it; it owns the HTTP connection pool.
- **Agents** are cheap and safe to share across requests; they hold configuration, not conversation state.
- **Sessions** hold the conversation. Create one per conversation, pass it to every run, and persist it (`session.to_dict()` / `AgentSession.from_dict(...)` in Python; `SerializeSessionAsync` / `DeserializeSessionAsync` in .NET) if the conversation spans requests.
- In Python, use `async with agent:` when the agent owns resources that need opening and closing — MCP tools and some context providers. The agent opens them on entry and closes them on exit.

```python
from agent_framework import Agent, AgentSession, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient


async def conversation() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You answer questions about Microsoft Learn content.",
        tools=[MCPStreamableHTTPTool(name="learn", url="https://learn.microsoft.com/api/mcp")],
    )
    async with agent:                                   # connects the MCP tool; closes it afterwards
        session = agent.create_session()
        await agent.run("What is Azure Container Apps?", session=session)
        follow_up = await agent.run("How does it scale?", session=session)   # remembers the topic
        print(follow_up.text)

        saved = session.to_dict()                       # store it (Redis, Cosmos DB, ...)
        restored = AgentSession.from_dict(saved)        # ...and resume later
        await agent.run("Summarise our conversation.", session=restored)
```

### Execution models

Every agent call is asynchronous. There are three shapes: one awaited response, many concurrent calls, and streaming.

**Python**

```python
import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), instructions="You tell short, clean jokes.")


async def single() -> None:
    response = await agent.run("Tell a joke about AI")
    print(response.text)


async def concurrent() -> None:
    # Independent calls run concurrently; no session is shared between them.
    results = await asyncio.gather(
        agent.run("Tell a joke about AI"),
        agent.run("Tell a joke about Python"),
        agent.run("Tell a joke about Azure"),
    )
    for result in results:
        print(result.text)


async def streaming() -> None:
    async for update in agent.run("Write a short story about AI", stream=True):
        print(update.text, end="", flush=True)
    print()
```

**.NET**

```csharp
using Microsoft.Agents.AI;

public static class ExecutionModels
{
    public static async Task RunAsync(AIAgent agent)
    {
        // One awaited response.
        AgentResponse joke = await agent.RunAsync("Tell me a joke.");
        Console.WriteLine(joke.Text);

        // Concurrent independent calls.
        string[] prompts = ["Joke about AI", "Joke about C#", "Joke about Azure"];
        AgentResponse[] results = await Task.WhenAll(prompts.Select(p => agent.RunAsync(p)));
        foreach (var r in results) Console.WriteLine(r.Text);

        // Streaming.
        await foreach (AgentResponseUpdate update in agent.RunStreamingAsync("Write a short story about AI"))
        {
            Console.Write(update.Text);
        }
        Console.WriteLine();
    }
}
```

Do not block on async calls (`.Result`, `.Wait()` in .NET; `asyncio.run` inside a running loop in Python). There is no synchronous agent API, and wrapping the async one in a blocking call costs threads and risks deadlocks. If a synchronous caller must use an agent, give it its own entry point (`asyncio.run(main())` at the top of a script, or `GetAwaiter().GetResult()` in a console `Main` that has no synchronisation context).

### Single-purpose agent design

Small agents with one job, precise instructions and only the tools they need are easier to test and cheaper to run than one agent that does everything. You don't need a class hierarchy for this; a factory function per role is enough.

```python
from agent_framework import Agent, SupportsChatGetResponse


def data_analyst(client: SupportsChatGetResponse) -> Agent:
    return Agent(
        client=client,
        name="data_analyst",
        instructions="You analyse data. Report trends, outliers and one recommendation. Be concise.",
    )


def code_reviewer(client: SupportsChatGetResponse) -> Agent:
    return Agent(
        client=client,
        name="code_reviewer",
        instructions="You review code for correctness, security and readability. List issues by severity.",
    )


# reviewer = code_reviewer(OpenAIChatClient())
# print((await reviewer.run("def hello(): print('hi')")).text)
```

### Testing agents

The seam for unit tests is the chat client: substitute one that returns canned responses, and test your instructions wiring, tools, middleware and parsing without calling a model.

**Python** — a minimal stub client (tests use `pytest` with `pytest-asyncio`):

```python
import pytest

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)


class StubChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Returns a fixed reply and records what the agent sent."""

    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply
        self.calls: list[list[Message]] = []

    async def _respond(self, messages) -> ChatResponse:
        self.calls.append(list(messages))
        return ChatResponse(messages=[Message(role="assistant", contents=[Content.from_text(self.reply)])])

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        return self._respond(messages)


@pytest.mark.asyncio
async def test_agent_returns_model_reply() -> None:
    client = StubChatClient("Hello! How can I help?")
    agent = Agent(client=client, instructions="You are helpful.")

    response = await agent.run("Hello")

    assert response.text == "Hello! How can I help?"
    assert client.calls[0][-1].text == "Hello"


@pytest.mark.asyncio
async def test_session_carries_history() -> None:
    client = StubChatClient("ok")
    agent = Agent(client=client, instructions="You are helpful.")
    session = agent.create_session()

    await agent.run("My name is Alice", session=session)
    await agent.run("What is my name?", session=session)

    sent = [m.text for m in client.calls[1]]
    assert "My name is Alice" in sent      # the second call included the first turn
```

**.NET** — mock `IChatClient` (here with Moq):

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
            .Setup(c => c.GetResponseAsync(It.IsAny<IEnumerable<ChatMessage>>(), It.IsAny<ChatOptions?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(new ChatMessage(ChatRole.Assistant, "Hello! How can I help?")));

        var agent = new ChatClientAgent(chatClient.Object, instructions: "You are helpful.");

        AgentResponse response = await agent.RunAsync("Hello");

        Assert.Equal("Hello! How can I help?", response.Text);
    }
}
```

For **integration tests** against a real model, assert on properties of the answer (it mentions a fact, it called a tool, it returned valid structured output) rather than exact text, and keep them out of the fast unit-test run. The evaluation pages ([Python](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_evaluation/)) cover scoring answers systematically.

---

## Multi-Agent Systems

### Orchestration patterns

| Pattern | Python (`agent_framework.orchestrations`) | .NET (`Microsoft.Agents.AI.Workflows`) |
| --- | --- | --- |
| Sequential | `SequentialBuilder(participants=[...])` | `AgentWorkflowBuilder.BuildSequential(...)` |
| Concurrent | `ConcurrentBuilder(participants=[...])` | `AgentWorkflowBuilder.BuildConcurrent(...)` |
| Handoff | `HandoffBuilder` | `AgentWorkflowBuilder.CreateHandoffBuilderWith(...)` |
| Group chat | `GroupChatBuilder` | `AgentWorkflowBuilder.CreateGroupChatBuilderWith(...)` |
| Magentic (planner-led) | `MagenticBuilder` | `AgentWorkflowBuilder.CreateMagenticBuilderWith(...)` |
| Custom graph | `WorkflowBuilder(start_executor=...)` | `new WorkflowBuilder(start)` |

```
            ┌──────────────┐
  input ──▶ │  researcher  │──┐
            └──────────────┘  │   sequential: each agent sees the conversation so far
            ┌──────────────┐  │
            │   analyst    │◀─┘──┐
            └──────────────┘     │
            ┌──────────────┐     │
            │    writer    │◀────┘──▶ output
            └──────────────┘
```

### A sequential pipeline

**Python**

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="Research the topic. List key facts.")
analyst = Agent(client=client, name="analyst", instructions="Analyse the research. Identify the main insights.")
writer = Agent(client=client, name="writer", instructions="Write a concise summary of the analysis.")

workflow = SequentialBuilder(participants=[researcher, analyst, writer]).build()


async def orchestrate(topic: str) -> str:
    result = await workflow.run(topic)
    final = result.get_outputs()[-1]             # an AgentResponse: the writer's turn
    return final.text
```

**.NET**

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class Pipeline
{
    public static async Task<string> OrchestrateAsync(IChatClient chatClient, string topic)
    {
        AIAgent researcher = new ChatClientAgent(chatClient, instructions: "Research the topic. List key facts.", name: "researcher");
        AIAgent analyst = new ChatClientAgent(chatClient, instructions: "Analyse the research. Identify the main insights.", name: "analyst");
        AIAgent writer = new ChatClientAgent(chatClient, instructions: "Write a concise summary of the analysis.", name: "writer");

        Workflow workflow = AgentWorkflowBuilder.BuildSequential(researcher, analyst, writer);

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, topic) });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent output && output.Is<List<ChatMessage>>(out var conversation))
            {
                return conversation.Last().Text;
            }
        }
        return "";
    }
}
```

### Communication between agents

There are three ways for agents to work together, from loosest to most structured:

1. **Pass outputs explicitly** — run agent A, put its text in agent B's prompt. Simple and transparent; fine for two steps.
2. **Agent as a tool** — give a coordinating agent other agents as tools, and let the model decide when to call them.
3. **A workflow** — an explicit graph with typed messages, checkpointing and events (above, and below).

Agent-as-tool:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", description="Researches a topic and lists key facts.",
                   instructions="Research the topic thoroughly.")
writer = Agent(client=client, name="writer", description="Writes polished prose from notes.",
               instructions="Write clearly and concisely.")

coordinator = Agent(
    client=client,
    name="coordinator",
    instructions="Produce briefs. Use the researcher for facts and the writer for the final text.",
    tools=[researcher.as_tool(), writer.as_tool()],
)

# brief = await coordinator.run("A one-paragraph brief on solid-state batteries")
```

In .NET the same is `researcher.AsAIFunction()`, passed in `tools:`.

For agents in **different processes or organisations**, use A2A: each agent publishes an agent card, and the caller wraps the remote agent as a local one (`agent_framework.a2a.A2AAgent`; .NET `A2ACardResolver.GetAIAgentAsync()`). A message queue between services is ordinary distributed-systems plumbing — the framework does not need its own.

### Shared state

Within a workflow, executors share state through the workflow context, which is checkpointed with the run — not through a global object.

```python
from agent_framework import WorkflowBuilder, WorkflowContext, executor


@executor(id="collect")
async def collect(topic: str, ctx: WorkflowContext[str]) -> None:
    ctx.set_state("topic", topic)                    # visible to later executors, saved in checkpoints
    await ctx.send_message(f"Notes on {topic}")


@executor(id="report")
async def report(notes: str, ctx: WorkflowContext[str, str]) -> None:
    topic = ctx.get_state("topic")
    await ctx.yield_output(f"Report on {topic}: {notes}")


workflow = WorkflowBuilder(start_executor=collect).add_edge(collect, report).build()

# result = await workflow.run("tidal energy"); result.get_outputs() == ["Report on tidal energy: Notes on tidal energy"]
```

Between *agents*, the equivalent is a shared `AgentSession` (Python `AgentExecutor(agent, session=shared)`) whose `state` dictionary tools and context providers can read and write; see the [Python orchestration page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/).

### Persisting conversation state

To keep history outside the process, attach a history provider. Cosmos DB is built in (`agent-framework-azure-cosmos`; .NET `Microsoft.Agents.AI.CosmosNoSql`):

```python
from agent_framework import Agent
from agent_framework.azure import CosmosHistoryProvider
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential

history = CosmosHistoryProvider(
    endpoint="https://contoso.documents.azure.com:443/",
    database_name="agents",
    container_name="chat-history",
    credential=DefaultAzureCredential(),
)

agent = Agent(client=OpenAIChatClient(), instructions="You are a support assistant.", context_providers=[history])

# Messages for session "customer-42" are loaded from and written to Cosmos DB on every run:
#   await agent.run("Where is my order?", session=agent.create_session(session_id="customer-42"))
```

Workflow checkpoints are persisted separately, with a checkpoint store (`FileCheckpointStorage`, Cosmos DB, or your own).

### Graph workflows

When the flow has branches, loops or non-agent steps, build the graph directly. Edges can carry conditions.

```python
from dataclasses import dataclass

from agent_framework import WorkflowBuilder, WorkflowContext, executor


@dataclass
class Ticket:
    text: str
    urgent: bool


@executor(id="classify")
async def classify(text: str, ctx: WorkflowContext[Ticket]) -> None:
    await ctx.send_message(Ticket(text, urgent="outage" in text.lower()))


@executor(id="page_on_call")
async def page_on_call(ticket: Ticket, ctx: WorkflowContext[str, str]) -> None:
    await ctx.yield_output(f"Paged on-call: {ticket.text}")


@executor(id="queue")
async def queue(ticket: Ticket, ctx: WorkflowContext[str, str]) -> None:
    await ctx.yield_output(f"Queued: {ticket.text}")


workflow = (
    WorkflowBuilder(start_executor=classify)
    .add_edge(classify, page_on_call, condition=lambda t: t.urgent)
    .add_edge(classify, queue, condition=lambda t: not t.urgent)
    .build()
)

# (await workflow.run("Total outage in EU-West")).get_outputs() == ["Paged on-call: Total outage in EU-West"]
```

The [graph workflows & declarative agents page](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) covers fan-out/fan-in, switch-case edges, checkpointing, human-in-the-loop and YAML-defined workflows, and the [.NET feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/#graph-based-workflows---net) the C# equivalents.

### Agent discovery

The framework has no agent registry. For agents inside one application, a dictionary of named agents (or .NET keyed DI services via `builder.AddAIAgent(name, ...)`) is all you need. Across services, A2A **agent cards** are the discovery mechanism: each agent serves a card at `/.well-known/agent-card.json` describing its skills and endpoint, and callers resolve the card to get a usable agent. Foundry Agent Service also lists the agents in a project.

### Scaling

Agents are I/O-bound — they spend their time waiting for the model — so scale with concurrency, not threads:

- In one process, run independent requests concurrently (`asyncio.gather`, `Task.WhenAll`) and bound concurrency with a semaphore sized to your model quota.
- Across processes, keep the service stateless (sessions and checkpoints in external stores) and scale out horizontally: Container Apps or AKS with KEDA, or Azure Functions with durable agents.
- Watch the model's tokens-per-minute quota; it is usually the real limit before CPU or memory.

```python
import asyncio

from agent_framework import Agent


async def run_bounded(agent: Agent, prompts: list[str], max_concurrency: int = 8) -> list[str]:
    gate = asyncio.Semaphore(max_concurrency)

    async def one(prompt: str) -> str:
        async with gate:
            return (await agent.run(prompt)).text

    return await asyncio.gather(*(one(p) for p in prompts))
```

The [production guide](/microsoft-agent-framework-guide/microsoft_agent_framework_production_guide/) goes into deployment, scaling and monitoring.

---

## Further reading

- [Python guide](/microsoft-agent-framework-guide/python/) · [.NET guide](/microsoft-agent-framework-guide/dotnet/)
- [A2A protocol](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/) · [Graph workflows & declarative agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) · [Enterprise](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/)
- [Recipes](/microsoft-agent-framework-guide/microsoft_agent_framework_recipes/) · [Production guide](/microsoft-agent-framework-guide/microsoft_agent_framework_production_guide/)

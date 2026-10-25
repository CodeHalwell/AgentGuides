---
title: "Semantic Kernel: Concepts and Cross-Language Reference"
description: "Language-neutral Semantic Kernel reference — architecture, core concepts, agents, orchestrations, vector stores, MCP, filters and the relationship to Microsoft Agent Framework, with Python and C# side by side."
framework: "semantic-kernel"
sidebar:
  label: "Concepts"
  order: 1
---

> Current releases: `semantic-kernel` %%version=semantic-kernel%% (Python) and `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%% (.NET), checked %%versions-checked%%. A Java SDK also exists but is not covered here. For complete, verified code see the [Python guide](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/) and the [.NET guide](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/).

This page explains *what* the pieces of Semantic Kernel are and how they fit together, with short Python and C# snippets so you can map one SDK onto the other. C# snippets assume `using System.ComponentModel;`, `using Microsoft.SemanticKernel;` and, for agents, `using Microsoft.SemanticKernel.Agents;`.

---

## Table of contents

1. [Where Semantic Kernel stands today](#1-where-semantic-kernel-stands-today)
2. [Architecture](#2-architecture)
3. [The kernel and AI services](#3-the-kernel-and-ai-services)
4. [Functions and plugins](#4-functions-and-plugins)
5. [Function calling](#5-function-calling)
6. [Agents and threads](#6-agents-and-threads)
7. [Orchestrations](#7-orchestrations)
8. [Vector stores](#8-vector-stores)
9. [MCP and A2A](#9-mcp-and-a2a)
10. [Filters](#10-filters)
11. [Structured output](#11-structured-output)
12. [Observability](#12-observability)
13. [Python and .NET differences](#13-python-and-net-differences)
14. [Removed concepts](#14-removed-concepts)

---

## 1. Where Semantic Kernel stands today

- **Microsoft Agent Framework (MAF) is the successor.** It is built by the Semantic Kernel and AutoGen teams, reached 1.0 (general availability) in April 2026, and is where most new features go. Microsoft's documentation calls it the direct successor to Semantic Kernel and AutoGen, created by the same teams.
- **Semantic Kernel 1.x is still supported.** Microsoft has committed to critical bug and security fixes, and to taking some existing features to GA, for as long as substantial numbers of developers use it and for at least one year after MAF's GA ([Semantic Kernel blog](https://devblogs.microsoft.com/semantic-kernel/semantic-kernel-and-microsoft-agent-framework/)).
- **Practical advice.** Keep existing SK applications on SK and keep them updated. Start new agent projects on [Microsoft Agent Framework](/microsoft-agent-framework-guide/). Migration can be gradual. In .NET, SK agents wrap as MAF agents with `agent.AsAIAgent()`. In Python, SK functions can serve as MAF tools, but not through the built-in `KernelFunction.as_agent_framework_tool()`: as of %%version=semantic-kernel%% it imports `AIFunction`, a class every released MAF version calls `FunctionTool`, and fails with "agent_framework is not installed". Use the [wrapper in the Python guide](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/#16-migrating-to-microsoft-agent-framework) instead.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Your application (API, worker, notebook)                     │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│ Agents & orchestrations                                      │
│   ChatCompletionAgent, Azure AI / OpenAI / Bedrock agents    │
│   Sequential · Concurrent · Group chat · Hand-off · Magentic │
└──────────────┬───────────────────────────────────────────────┘
               │ invoke via
┌──────────────▼───────────────────────────────────────────────┐
│ Kernel                                                       │
│   AI services ─ chat, embeddings, image, audio               │
│   Plugins     ─ native functions, prompt functions, OpenAPI, │
│                 MCP tools, vector-store search               │
│   Filters     ─ prompt render · function · auto-invocation   │
└──────────────┬──────────────────────────┬────────────────────┘
               │                          │
┌──────────────▼─────────────┐ ┌──────────▼───────────────────┐
│ Model providers            │ │ Vector stores                │
│ OpenAI, Azure OpenAI,      │ │ In-memory, Azure AI Search,  │
│ Anthropic, Google, Ollama, │ │ Qdrant, Postgres, Redis,     │
│ Bedrock, Mistral, ...      │ │ Cosmos DB, ...               │
└────────────────────────────┘ └──────────────────────────────┘
```

The kernel is deliberately thin: a container that hands the right service to each function call and runs filters around it. Everything interesting is a plugin, a service or an agent.

---

## 3. The kernel and AI services

**Python**

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="%%model=openai-small%%"))
result = await kernel.invoke_prompt("Say hello.")
```

**C#**

```csharp
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("%%model=openai-small%%", apiKey, serviceId: "chat");
Kernel kernel = builder.Build();
var result = await kernel.InvokePromptAsync("Say hello.");
```

Several services can be registered; execution settings select one by `service_id` / `ServiceId`. Model names here are examples.

---

## 4. Functions and plugins

A **kernel function** is either a *prompt function* (a template rendered and sent to a model) or a *native function* (your code). A **plugin** is a named group of functions. The model sees each function's name, description and parameter schema.

**Python**

```python
from typing import Annotated
from semantic_kernel.functions import kernel_function

class OrdersPlugin:
    @kernel_function(description="Look up an order's status.")
    def status(self, order_id: Annotated[str, "Order ID, e.g. A-1042"]) -> str:
        return "shipped"

kernel.add_plugin(OrdersPlugin(), plugin_name="orders")
summarise = kernel.add_function(plugin_name="text", function_name="summarise",
                                prompt="Summarise:\n{{$input}}")
```

**C#**

```csharp
public sealed class OrdersPlugin
{
    [KernelFunction, Description("Look up an order's status.")]
    public string Status([Description("Order ID, e.g. A-1042")] string orderId) => "shipped";
}

kernel.ImportPluginFromType<OrdersPlugin>("orders");
var summarise = kernel.CreateFunctionFromPrompt("Summarise:\n{{$input}}", functionName: "summarise");
```

Other plugin sources: OpenAPI documents, MCP servers, vector-store search functions, and prompt directories/YAML files.

**Template input is encoded.** When a prompt template inserts a variable or a function result, Semantic Kernel HTML-encodes it, so text such as `</message><message role="system">` arrives as content of the current message rather than as a new system message. Opt out per variable or per template only for values you trust: `allow_dangerously_set_content=True` (Python `InputVariable` / `PromptTemplateConfig`) or `AllowDangerouslySetContent = true` (.NET `InputVariable`, `PromptTemplateConfig` or `KernelPromptTemplateFactory`).

---

## 5. Function calling

The model chooses which functions to call; the kernel executes them and feeds results back until the model produces an answer. This replaced the planners.

| Python | C# | Meaning |
|---|---|---|
| `FunctionChoiceBehavior.Auto()` | `FunctionChoiceBehavior.Auto()` | Model may call any advertised function |
| `FunctionChoiceBehavior.Required(...)` | `FunctionChoiceBehavior.Required(...)` | Model must call a function |
| `FunctionChoiceBehavior.NoneInvoke()` | `FunctionChoiceBehavior.None()` | Advertise but do not call |
| `Auto(auto_invoke=False)` | `Auto(autoInvoke: false)` | Return calls to your code |

Agents enable automatic function calling by default (Python) or via their `Arguments` execution settings (C#).

---

## 6. Agents and threads

An **agent** packages instructions, a model and plugins behind a common interface; a **thread** holds one conversation.

**Python**

```python
from semantic_kernel.agents import ChatCompletionAgent

agent = ChatCompletionAgent(service=OpenAIChatCompletion(ai_model_id="%%model=openai-small%%"),
                            name="Support", instructions="Be concise.", plugins=[OrdersPlugin()])
response = await agent.get_response(messages="Where is order A-1042?")
follow_up = await agent.get_response(messages="And when will it arrive?", thread=response.thread)
```

**C#**

```csharp
ChatCompletionAgent agent = new() { Name = "Support", Instructions = "Be concise.", Kernel = kernel };
AgentThread? thread = null;
await foreach (var item in agent.InvokeAsync("Where is order A-1042?", thread))
{
    thread = item.Thread;
    Console.WriteLine(item.Message.Content);
}
```

Agent types beyond `ChatCompletionAgent`: OpenAI Assistants and Responses agents, Azure AI Foundry agents, Amazon Bedrock agents, Copilot Studio agents, and (in .NET) A2A agents. Each has its own thread type, often stored server-side.

**Memory providers (.NET only).** A .NET `AgentThread` holds a list of `AIContextProviders`; before each model call every provider can add instructions or functions, and it sees each new message. `Mem0Provider` (long-term memory in Mem0) and `WhiteboardProvider` (a running summary of requirements and decisions) ship in the main package; you can write your own by subclassing `AIContextProvider`. Python SK has no equivalent.

---

## 7. Orchestrations

All five orchestration patterns run on an in-process runtime and share one API: build, start the runtime, invoke with a task, await the result.

| Pattern | Use when |
|---|---|
| Sequential | A fixed pipeline: each agent transforms the previous output |
| Concurrent | Independent perspectives on the same task, gathered at the end |
| Group chat | Agents take turns under a manager that picks speakers and decides when to stop |
| Hand-off | Agents transfer control to the right specialist; can ask a human for input |
| Magentic | A manager plans, delegates and tracks progress (from the Magentic-One design) |

They are experimental in both SDKs (Python: marked experimental; .NET: preview packages, `SKEXP0110`). The older `AgentGroupChat` with selection and termination strategies remains but is superseded. Microsoft Agent Framework's graph-based **workflows** are the long-term replacement for both orchestrations and the Process framework.

---

## 8. Vector stores

A vector store collection is typed by a record class whose fields are marked as key, data or vector. Collections can embed text automatically through an embedding generator, support filtering, and can be turned into a search function for agents.

| | Python | .NET |
|---|---|---|
| Abstractions | `semantic_kernel.data.vector` | `Microsoft.Extensions.VectorData` |
| Field markers | `VectorStoreField("key" \| "data" \| "vector")` | `[VectorStoreKey]`, `[VectorStoreData]`, `[VectorStoreVector(n)]` |
| Connectors | `semantic_kernel.connectors.*` (in-memory, Azure AI Search, Qdrant, Postgres, Redis, ...) | `CommunityToolkit.VectorData.*` (`.InMemory`, `.AzureAISearch`, `.Qdrant`, `.Redis`, `.PgVector`, `.SqlServer`, `.SqliteVec`, `.AzureCosmosDB`, `.CosmosMongoDB`, ...) |
| As a tool | `collection.create_search_function(...)` | `VectorStoreTextSearch<T>` + `CreateWithGetTextSearchResults(...)` |

The Python API was redesigned in 1.34.0; the .NET connectors were renamed to `CommunityToolkit.VectorData.*` in July 2026. The older memory-store API is legacy in both.

---

## 9. MCP and A2A

- **Consuming MCP servers.** Python: `MCPStdioPlugin`, `MCPStreamableHttpPlugin` (and SSE/WebSocket variants) are plugins. .NET: use the official `ModelContextProtocol` SDK and convert tools with `tool.AsKernelFunction()`.
- **Exposing a kernel over MCP.** Python: `kernel.as_mcp_server(...)`. .NET: host kernel functions with the MCP SDK's server package.
- **A2A.** .NET has an `A2AAgent` (package `Microsoft.SemanticKernel.Agents.A2A`, still `-alpha`) that calls a remote A2A agent. Python has no built-in A2A connector. Microsoft Agent Framework has fuller A2A support.

---

## 10. Filters

Filters are middleware around three points:

| Point | Python | C# |
|---|---|---|
| Prompt rendering | `FilterTypes.PROMPT_RENDERING` | `IPromptRenderFilter` |
| Function invocation | `FilterTypes.FUNCTION_INVOCATION` | `IFunctionInvocationFilter` |
| Automatic function calling | `FilterTypes.AUTO_FUNCTION_INVOCATION` | `IAutoFunctionInvocationFilter` |

Typical uses: logging and timing, redaction, prompt-injection checks, approval gates on sensitive tools, caching, and stopping the tool loop early (`terminate` / `Terminate`).

---

## 11. Structured output

Pass a type as the response format and the connector requests a strict JSON schema:

```python
from pydantic import BaseModel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

class MyRecord(BaseModel):
    title: str
    tags: list[str]

settings = OpenAIChatPromptExecutionSettings(response_format=MyRecord)
```

```csharp
public sealed record MyRecord(string Title, string[] Tags);   // any C# type describing the JSON shape

var settings = new OpenAIPromptExecutionSettings { ResponseFormat = typeof(MyRecord) };
```

---

## 12. Observability

Both SDKs emit OpenTelemetry traces, metrics and logs. The metrics differ: both record function duration (`semantic_kernel.function.invocation.duration`), but only .NET records token usage per function (`semantic_kernel.function.invocation.token_usage.prompt` / `.completion`). In Python, token counts are on each completion's metadata (`message.metadata["usage"]`) and on GenAI spans, and you record them yourself ([example](/semantic-kernel-guide/python/semantic_kernel_production_python/#3-telemetry-and-token-usage)). GenAI semantic-convention spans for model calls are switched on with an environment variable (Python: `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true`) or an `AppContext` switch (.NET: `Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics`). The "sensitive" variants also record prompts and completions — development only.

---

## 13. Python and .NET differences

| Topic | Python | .NET |
|---|---|---|
| Install | one package plus extras (`semantic-kernel[azure]`, ...) | many NuGet packages; several still `-preview`/`-alpha` |
| Async | `asyncio`, `await` everywhere | `async`/`await`, `IAsyncEnumerable` for streams |
| DI | none built in | `IServiceCollection`; kernel built from the service provider |
| Agent call | `get_response(messages=..., thread=...)` | `InvokeAsync(message, thread)` |
| Experimental APIs | marked in docs and warnings | compiler diagnostics `SKEXP0001`, `SKEXP0110`, ... |
| Process framework | `semantic_kernel.processes` (experimental) | `Microsoft.SemanticKernel.Process.*` (alpha) |

---

## 14. Removed concepts

| Concept | Status | Replacement |
|---|---|---|
| Skills, `SKContext`, `ISKFunction` | Renamed before 1.0 | Plugins, `KernelArguments`, `KernelFunction` |
| Planners (Sequential, Stepwise, Action, Handlebars) | Removed (Python 1.33.0) / deprecated (.NET) | Function calling |
| Memory stores, `SemanticTextMemory` | Legacy | Vector store collections |
| `AgentGroupChat` | Superseded | Orchestrations |
| `kernel.add_chat_service`, `invoke_async` (Python) | Pre-1.0 | `add_service`, `invoke` |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-25 | Py 1.45.0 / .NET 1.81.0 | Removed an unverified quotation about Agent Framework. |
| 2026-10-21 | Py 1.45.0 / .NET 1.81.0 | Python Agent Framework bridge documented as broken. Token-usage metrics corrected (.NET only). Vector-store package names and A2A status corrected. Added template input encoding and .NET memory providers. |
| 2026-10-21 | Py 1.45.0 / .NET 1.81.0 | Rewritten as a concise cross-language concepts page. The previous edition mixed pre-1.0 APIs (`invoke_async`, `create_function_from_prompt`, memory stores, planners) with current ones and left sections 4–14 as empty stubs. Versions now read from the site's version registry. |
| 2026-06-04 | Py 1.43.0 / .NET 1.74.0 | Version bump. |

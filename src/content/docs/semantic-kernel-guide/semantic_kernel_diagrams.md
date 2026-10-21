---
title: "Semantic Kernel: Architecture and Pattern Diagrams"
description: "Diagrams of how Semantic Kernel works: the filter pipeline around a function call, the function-calling loop, agents and threads, the five orchestrations, vector-store RAG, MCP and A2A, hosting with a kernel per request, and the bridge to Agent Framework."
framework: "semantic-kernel"
sidebar:
  label: "Diagrams"
  order: 4
---

Diagrams of the mechanisms behind Semantic Kernel, the same in Python (`semantic-kernel` %%version=semantic-kernel%%) and .NET (`Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%%) unless a diagram says otherwise. The overall component map is in the [concepts page](/semantic-kernel-guide/semantic_kernel_comprehensive_guide/#2-architecture); code for each flow is in the [Python](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/) and [.NET](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/) guides.

---

## 1. Invoking a prompt function

What happens inside `kernel.invoke(fn, args)` / `kernel.InvokeAsync(fn, args)` for a prompt function. Filters are nested: each calls `next` to continue, so the first one registered is the outermost.

```
kernel.invoke(function, arguments)
  │
  ▼
┌─ function-invocation filters ─────────────────────────────────────────┐
│  (logging, timing, caching: set a result and skip `next`)             │
│                                                                       │
│   ┌─ prompt-render filters ─────────────────────────────────────┐     │
│   │  template engine renders the prompt                         │     │
│   │    {{$var}}          → argument, HTML-encoded               │     │
│   │    {{plugin.fn $x}}  → nested function call, result encoded │     │
│   │  filter may inspect or rewrite rendered_prompt              │     │
│   └──────────────────────────────┬──────────────────────────────┘     │
│                                  ▼                                    │
│   rendered text parsed into chat messages (<message role="...">)      │
│                                  │                                    │
│                                  ▼                                    │
│   AI service chosen by execution settings' service_id (or default)    │
│                                  │                                    │
│                                  ▼                                    │
│   model call  ──(tool calls?)──▶ function-calling loop (section 2)    │
│                                  │                                    │
│                                  ▼                                    │
│   FunctionResult (value + metadata, e.g. token usage)                 │
└───────────────────────────────────────────────────────────────────────┘
```

A native function skips the rendering and model steps: the function-invocation filters wrap your code directly. Because arguments are encoded before the rendered text is split into messages, a user cannot close a `<message>` tag and inject a system message unless the template opts out with `allow_dangerously_set_content` / `AllowDangerouslySetContent`.

---

## 2. The function-calling loop

The Sequential, Stepwise, Action and Handlebars planners were removed from Python (1.33.0) and deprecated in .NET. The model now plans implicitly through native function calling:

```
┌────────────────────────────────────────────────────────────┐
│ Prompt / agent turn                                        │
│   execution settings: FunctionChoiceBehavior.Auto()        │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │ Chat completion request │◀────────────────────┐
              │ (messages + tool schema │                     │
              │  of kernel functions)   │                     │
              └────────────┬────────────┘                     │
                           ▼                                  │
                 ┌────────────────────┐   no tool calls       │
                 │ Model response     │──────────────▶ Final answer
                 └─────────┬──────────┘                       │
                           │ tool calls                       │
                           ▼                                  │
          ┌──────────────────────────────────────┐            │
          │ Auto-function-invocation filters     │            │
          │ (approve / block / terminate)        │            │
          └─────────────────┬────────────────────┘            │
                            ▼                                 │
          ┌──────────────────────────────────────┐            │
          │ Kernel invokes functions             │            │
          │ (function-invocation filters wrap    │            │
          │  each call; results appended to      │            │
          │  chat history)                       │────────────┘
          └──────────────────────────────────────┘
```

For a plan a human must approve first, request it with `FunctionChoiceBehavior.NoneInvoke()` / `None()` or with structured output, review it, then execute.

---

## 3. Agents and threads

Where the conversation lives depends on the agent type:

```
ChatCompletionAgent                         Service-backed agents
(any chat service)                          (OpenAI Responses, Azure AI Foundry,
                                             OpenAI Assistants, Bedrock, Copilot Studio)

 your process                                your process           provider
┌──────────────────────────────┐            ┌──────────────┐       ┌──────────────────────┐
│ agent ── kernel ── plugins   │            │ agent        │──────▶│ agent definition     │
│   │                          │            │  (plugins run│       │ thread / response    │
│   ▼                          │            │  here, where │◀──────│ history              │
│ ChatHistoryAgentThread       │            │  supported)  │       │ hosted tools         │
│   (full message history)     │            │ thread = ID  │       └──────────────────────┘
└──────────────────────────────┘            └──────────────┘
 persist the history yourself               persist only the thread / response ID
```

Both kinds share one call shape: `get_response(messages=..., thread=...)` in Python, `InvokeAsync(message, thread)` in .NET, with the thread returned on each response and passed back on the next turn. In .NET a thread can also carry `AIContextProviders` (memory such as Mem0 or a whiteboard) that add instructions before every model call.

---

## 4. Orchestrations

All five patterns run on the in-process runtime and share one call shape: build, start the runtime, invoke with a task, wait for the result with a timeout.

```
orchestration.invoke(task, runtime)  /  InvokeAsync(task, runtime)
        │
        ▼
┌──────────────────────────────── InProcessRuntime ────────────────────────────────┐
│                                                                                  │
│  Sequential   Agent A ──▶ Agent B ──▶ Agent C ──▶ result (last agent's output)   │
│                                                                                  │
│  Concurrent        ┌─▶ Agent A ─┐                                                │
│               task ┼─▶ Agent B ─┼──▶ list of results                             │
│                    └─▶ Agent C ─┘                                                │
│                                                                                  │
│  Group chat   manager: select next speaker ──▶ agent replies ──▶ should stop?    │
│               (RoundRobinGroupChatManager, max_rounds / MaximumInvocationCount;  │
│                subclass the manager for custom selection or termination)         │
│                                                                                  │
│  Hand-off     Triage ──handoff──▶ Billing ──handoff──▶ Triage ...                │
│               each agent gets hand-off functions for its allowed targets;        │
│               human input via human_response_function / InteractiveCallback     │
│                                                                                  │
│  Magentic     manager plans ──▶ picks an agent ──▶ checks progress ──▶ replans   │
│               (StandardMagenticManager, after the Magentic-One design)           │
└──────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
result.get(timeout=...)  /  GetValueAsync(timeout)
```

Every pattern accepts a response callback that sees each agent message, and input/output transforms that convert between your types and chat messages. Orchestrations are experimental; their long-term replacement is Agent Framework workflows.

---

## 5. Vector stores and RAG

The old "memory store" API (`VolatileMemoryStore`, `SemanticTextMemory`, `*MemoryStore` connectors) is legacy. Current releases use typed vector store collections — `semantic_kernel.data.vector` in Python, `Microsoft.Extensions.VectorData` in .NET.

```
┌────────────────────────────────────────────────────┐
│ Record class                                       │
│   key field · data fields · vector field(s)        │
│   (VectorStoreField / [VectorStoreKey] etc.)       │
└─────────────────────┬──────────────────────────────┘
                      │ typed by
                      ▼
┌────────────────────────────────────────────────────┐
│ VectorStore ── get_collection / GetCollection ──▶  │
│ VectorStoreCollection<TKey, TRecord>               │
│   ensure_collection_exists · upsert · get · delete │
│   search(text or vector, top, filter) · hybrid     │
│   embedding generator ─▶ embeds on upsert/search   │
└─────────────────────┬──────────────────────────────┘
                      │ implemented by connectors
      ┌───────────────┼───────────────┬─────────────────┐
      ▼               ▼               ▼                 ▼
  In-memory     Azure AI Search    Qdrant / Redis   Postgres, Cosmos DB,
                                   Weaviate ...     SQL Server, Pinecone ...
                      │
                      ▼
      create_search_function / VectorStoreTextSearch
      ─▶ a kernel function agents can call (RAG as a tool)
```

An agent with the search function in its kernel decides when to retrieve; nothing is written back to the store as part of a conversation unless your own code does so.

---

## 6. MCP

```
Consuming MCP servers                              Exposing a kernel over MCP (Python)

 Python                                             kernel with plugins
   MCPStdioPlugin / MCPStreamableHttpPlugin            │ kernel.as_mcp_server(...)
     │ connect(), list tools                           ▼
     ▼                                              MCP server: one tool per function,
   KernelPlugin of tool-backed functions            optional prompts
                                                       │ stdio / Streamable HTTP
 .NET                                                  ▼
   McpClient (ModelContextProtocol SDK)             any MCP client
     │ ListToolsAsync()
     ▼
   tool.AsKernelFunction() → kernel.Plugins
```

In .NET, exposing kernel functions as an MCP server is done with the MCP SDK's server package; there is no `as_mcp_server` equivalent in Semantic Kernel itself.

---

## 7. A2A

Semantic Kernel has no A2A message bus or router. What exists:

```
.NET client side                                   Python / .NET server side

 orchestration or thread                            SK agent
   │                                                  │ wrapped by your code using the
   ▼                                                  ▼ A2A SDK (a2a-sdk / A2A .NET SDK)
 A2AAgent (Microsoft.SemanticKernel.Agents.A2A,    A2A server: agent card + JSON-RPC
           alpha)                                     ▲
   │ A2A JSON-RPC over HTTP                           │
   └──────────────────────────────────────────────────┘  or any other A2A agent
```

Python SK has no A2A connector; Microsoft's samples host SK agents with the separate `a2a-sdk` package. Agent Framework has first-party A2A support.

---

## 8. Hosting: shared services, a kernel per request

```
process start                                  each HTTP request
┌────────────────────────────────────┐         ┌──────────────────────────────────────────┐
│ chat / embedding services          │────────▶│ new Kernel                               │
│   (HTTP pool, credentials; shared) │         │   + shared services                      │
│ vector store client (shared)       │────────▶│   + plugins for this tenant / user       │
│ stateless plugin instances         │────────▶│   + filters (approval, redaction, usage) │
│ telemetry exporter                 │         │ invoke function or agent                 │
└────────────────────────────────────┘         │ thread loaded from / saved to a store    │
                                               └──────────────────────────────────────────┘
 .NET: services as singletons,                  Python: build the kernel in a FastAPI
       Kernel registered as transient                   dependency (Depends(get_kernel))
```

A kernel shared across requests would share whatever plugins and filters one request added. Runnable versions: [Python production guide](/semantic-kernel-guide/python/semantic_kernel_production_python/#2-the-app-one-service-one-kernel-per-request), [.NET recipes](/semantic-kernel-guide/dotnet/semantic_kernel_recipes_dotnet/#1-aspnet-core-minimal-api-with-a-transient-kernel).

---

## 9. Moving to Agent Framework

```
Semantic Kernel                         Microsoft Agent Framework
─────────────────                       ─────────────────────────
.NET  SK agent ── agent.AsAIAgent() ──▶ AIAgent   (works today)

Py    KernelFunction ──as_agent_framework_tool()──X  broken in 1.45.0 (imports a renamed class)
      KernelFunction ──your wrapper (FunctionTool)──▶ tool for an Agent Framework Agent

      Kernel + services                 chat client, no kernel
      plugins                           tools
      threads                           sessions
      filters                           middleware
      orchestrations / processes        workflows
```

The Python wrapper is in the [Python guide's migration section](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/#16-migrating-to-microsoft-agent-framework).

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | Py 1.45.0 / .NET 1.81.0 | Rewritten around Semantic Kernel's real mechanisms. Removed diagrams of behaviour SK does not have ("semantic" plugins, volatile memory, memory updates after every request, a structured-output validation and retry pipeline, generic master-worker and peer-to-peer patterns, a fixed ReAct loop limit). The Python and .NET diagram pages were merged in; their invented A2A message bus was replaced by the actual A2A support. |

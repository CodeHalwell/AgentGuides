---
title: "Semantic Kernel Diagrams (.NET)"
description: "Visual architecture and flow diagrams for Semantic Kernel in .NET — request flow, dependency injection lifetimes and agent orchestration."
framework: semantic-kernel
language: dotnet
---

# Semantic Kernel Diagrams (.NET)

Diagrams for `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%%. Code for each flow is in the [comprehensive .NET guide](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/).

---

## ASP.NET Core request flow

```
HTTP request
     │
     ├─→ Endpoint / controller
     │        │
     │        ├─→ Kernel (transient, resolved per request)
     │        │        │
     │        │        ├─→ IChatCompletionService / IChatClient   (singleton)
     │        │        ├─→ Plugins  ──→ your services via DI
     │        │        ├─→ Filters  (logging, approval, redaction)
     │        │        └─→ Vector store search (RAG plugin)
     │        │
     │        └─→ ILogger + OpenTelemetry spans (Microsoft.SemanticKernel*)
     │
     └─────────────────→ HTTP response (JSON or SSE stream)
```

---

## Dependency injection lifetimes

```
┌──────────────────────────────────────────────────────┐
│ Program.cs                                           │
│                                                      │
│  services.AddAzureOpenAIChatCompletion(...)          │  singleton
│  services.AddSingleton<VectorStore>(...)             │  singleton
│  services.AddSingleton<WeatherPlugin>()              │  singleton (stateless)
│  services.AddTransient(sp => new Kernel(sp) {...})   │  transient
│  services.AddScoped<IAgentOrchestrator, ...>()       │  scoped
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ Endpoint / service                                   │
│                                                      │
│  (Kernel kernel, IAgentOrchestrator orchestrator,    │
│   ILogger<T> logger) => ...                          │
└──────────────────────────────────────────────────────┘
```

Why transient: plugins and filters are added to a `Kernel` instance; a singleton kernel would share whatever one request added with every other request.

---

## Agent orchestration on the in-process runtime

```
InvokeAsync(task, runtime)
        │
        ▼
┌───────────────────────────── InProcessRuntime ─────────────────────────────┐
│                                                                             │
│  Sequential:   Agent A ──▶ Agent B ──▶ Agent C ──▶ result                   │
│                                                                             │
│  Concurrent:   ┌─▶ Agent A ─┐                                               │
│        task ───┼─▶ Agent B ─┼──▶ string[] result                            │
│                └─▶ Agent C ─┘                                               │
│                                                                             │
│  Group chat:   manager picks next speaker ──▶ agent replies ──▶ repeat      │
│                (RoundRobinGroupChatManager, MaximumInvocationCount)         │
│                                                                             │
│  Hand-off:     triage ──handoff──▶ billing ──handoff──▶ triage …            │
│                (OrchestrationHandoffs; InteractiveCallback for the user)    │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
OrchestrationResult<T>.GetValueAsync(timeout)
```

---

For more diagrams, see the [language-agnostic diagrams](/semantic-kernel-guide/semantic_kernel_diagrams/).

**[Back to .NET overview](/semantic-kernel-guide/dotnet/)**

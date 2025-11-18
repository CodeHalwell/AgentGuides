# Semantic Kernel Diagrams (.NET)

**Visual Architecture and Flow Diagrams for .NET Implementation**

Last Updated: November 2025

---

## ASP.NET Core Request Flow

```
HTTP Request
     │
     ├─→ Controller Action
     │        │
     │        ├─→ Kernel (DI injected)
     │        │        │
     │        │        ├─→ Chat Completion Service
     │        │        ├─→ Plugins
     │        │        └─→ Memory
     │        │
     │        └─→ ILogger (structured logging)
     │
     └─────────────────→ HTTP Response
```

---

## Dependency Injection Pattern

```
┌─────────────────────────────────────────────────┐
│              Program.cs / Startup.cs            │
│                                                 │
│  builder.Services.AddSingleton<Kernel>(...)     │
│  builder.Services.AddScoped<IAgentService>(...) │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│           Dependency Injection Container        │
│                                                 │
│  - Kernel (Singleton)                           │
│  - IChatCompletionService (Singleton)           │
│  - IMemoryStore (Singleton)                     │
│  - IAgentService (Scoped)                       │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────┐
│               Controller / Service               │
│                                                  │
│  public MyController(                            │
│      Kernel kernel,                              │
│      IAgentService agentService,                 │
│      ILogger<MyController> logger)               │
│  {                                               │
│      _kernel = kernel;                           │
│      _agentService = agentService;               │
│      _logger = logger;                           │
│  }                                               │
└──────────────────────────────────────────────────┘
```

---

For more diagrams, see [../semantic_kernel_diagrams.md](../semantic_kernel_diagrams.md)

**[Back to .NET README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

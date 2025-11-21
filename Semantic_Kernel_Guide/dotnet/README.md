# Semantic Kernel .NET Guide

**A Comprehensive C#/.NET-Specific Technical Reference for Microsoft Semantic Kernel**

Last Updated: November 2025
.NET Version: 6.0+
Semantic Kernel .NET: 1.67.1+

---

## Overview

This directory contains comprehensive, .NET-specific documentation for building AI agents and agentic systems with Microsoft's Semantic Kernel framework. All examples, patterns, and best practices are tailored specifically for C# and .NET developers.

### What You'll Find Here

- **Production-Ready Patterns:** Enterprise deployment, monitoring, and scaling strategies for .NET
- **Multi-Agent Systems:** Advanced orchestration using .NET async patterns
- **2025 Features:** Model Context Protocol (MCP - March 2025), Vector Store overhaul, Microsoft Agent Framework
- **Real-World Examples:** Complete, tested C# code recipes for common use cases
- **Best Practices:** Error handling, resilience (Polly), security, and performance optimization

---

## Quick Start

### Installation

```bash
dotnet new console -n SkAgents
cd SkAgents

# Core packages
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Connectors.OpenAI
dotnet add package Microsoft.SemanticKernel.Plugins.Core

# Azure integrations
dotnet add package Microsoft.SemanticKernel.Connectors.AzureOpenAI
dotnet add package Microsoft.SemanticKernel.Connectors.AzureAISearch

# Additional packages
dotnet add package Azure.Identity
dotnet add package Microsoft.Extensions.Configuration
dotnet add package Microsoft.Extensions.Logging
dotnet add package Polly
```

### Hello World Agent

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var builder = Kernel.CreateBuilder();

builder.AddOpenAIChatCompletion(
    modelId: "gpt-4",
    apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
);

var kernel = builder.Build();

// Create and invoke function
var prompt = "Explain Semantic Kernel in one sentence.";
var function = kernel.CreateFunctionFromPrompt(prompt);

var result = await kernel.InvokeAsync<string>(function);
Console.WriteLine(result);
```

---

## Documentation Structure

### Core Guides

1. **[semantic_kernel_comprehensive_dotnet.md](semantic_kernel_comprehensive_dotnet.md)**
   - Complete C#/.NET reference from basics to advanced topics
   - Agents, functions, plugins, memory, planners
   - 2025 features: MCP (March 2025), Agent Framework, Vector Store
   - Async/await patterns, dependency injection, configuration
   - Type-safe implementations with generics

2. **[semantic_kernel_production_dotnet.md](semantic_kernel_production_dotnet.md)**
   - Production deployment strategies (Docker, Kubernetes, Azure App Service)
   - Performance optimization and caching with MemoryCache
   - Monitoring with Application Insights and OpenTelemetry
   - Error handling with Polly (retries, circuit breakers)
   - Security best practices with Azure Key Vault and Managed Identity
   - Cost tracking and optimization
   - Testing strategies (xUnit, NUnit, integration tests)

3. **[semantic_kernel_recipes_dotnet.md](semantic_kernel_recipes_dotnet.md)**
   - 30+ production-ready C# code examples
   - Document processing and RAG implementations
   - Multi-agent systems and orchestration
   - Plugin development patterns
   - ASP.NET Core integration
   - Integration with existing .NET ecosystems

4. **[semantic_kernel_diagrams_dotnet.md](semantic_kernel_diagrams_dotnet.md)**
   - .NET-specific architecture diagrams
   - Request flow visualizations
   - Dependency injection patterns
   - Async execution patterns

### Index and Navigation

5. **[GUIDE_INDEX.md](GUIDE_INDEX.md)**
   - Complete topic index
   - Quick reference by use case
   - Cross-reference to related sections
   - Learning paths for different skill levels

---

## 2025 Features Highlighted

### Model Context Protocol (MCP) - March 2025

- **Host/Client Implementation:** Connect to MCP servers for tool/resource access
- **MCP Server Creation:** Expose SK functions as MCP tools
- **.NET SDK:** Full MCP support in .NET (v1.38+ - March 2025)
- See: Comprehensive Guide → Section 15-17

### Microsoft Agent Framework Integration - March 2025

- **Unified Agent SDK:** Integration with Microsoft's broader agent ecosystem
- **Cross-Framework Communication:** SK agents working with AutoGen, Python SK
- **Enterprise Features:** Enhanced governance, monitoring, and compliance
- **Primary Platform:** .NET is the primary platform for Agent Framework
- See: Comprehensive Guide → Section 19

### Vector Store Overhaul (v1.34)

- **Unified API:** Consistent interface across all vector stores
- **Improved Performance:** Optimized embedding and retrieval
- **New Connectors:** Enhanced Azure AI Search, Qdrant support
- See: Comprehensive Guide → Section 10

### Google A2A Protocol Integration

- **Agent-to-Agent Communication:** Standardized messaging protocol
- **Interoperability:** Connect SK agents with other A2A-compliant systems
- See: Comprehensive Guide → Section 18

---

## Learning Paths

### Beginner Path (1-2 weeks)
1. Start with: **Comprehensive Guide → Sections 1-3** (Fundamentals, Simple Agents)
2. Try: **Quick Start** (above) and first examples
3. Explore: **Recipes → Basic Patterns** (Q&A, summarization)
4. Practice: Build a simple chatbot with memory

### Intermediate Path (2-4 weeks)
1. Study: **Comprehensive Guide → Sections 4-8** (Plugins, Memory, Planners)
2. Build: **Recipes → Multi-Agent Examples** (orchestration patterns)
3. Learn: **ASP.NET Core Integration** (Web APIs, DI)
4. Implement: RAG system with Azure AI Search

### Advanced Path (4-8 weeks)
1. Master: **Production Guide** (deployment, monitoring, scaling)
2. Deep Dive: **2025 Features** (MCP, Agent Framework, Vector Store)
3. Architect: Enterprise-grade multi-agent system
4. Deploy: Production deployment on Azure

---

## Common Use Cases

### Intelligent Document Processing
**Example:** Upload PDFs → Extract text → Chunk → Embed → Store in Azure AI Search → Query → Generate answers

**See:**
- Recipes → Document QA Recipe
- Comprehensive Guide → Memory & Vector Stores
- Production Guide → Azure Integration

### Multi-Agent Enterprise System
**Example:** User query → Coordinator agent → Specialist agents (research, analysis, reporting) → Aggregated result

**See:**
- Comprehensive Guide → Agent Orchestration
- Recipes → Multi-Agent Patterns
- Production Guide → Scaling Strategies

### ASP.NET Core API with SK
**Example:** REST API → SK agent processes → Stream response → Return JSON

**See:**
- Recipes → ASP.NET Core Integration
- Production Guide → Web API Patterns
- Comprehensive Guide → Dependency Injection

---

## .NET-Specific Best Practices

### Dependency Injection

```csharp
// Program.cs (ASP.NET Core)
builder.Services.AddSingleton(sp =>
{
    var kernelBuilder = Kernel.CreateBuilder();
    kernelBuilder.AddAzureOpenAIChatCompletion(
        deploymentName: builder.Configuration["AzureOpenAI:Deployment"]!,
        endpoint: builder.Configuration["AzureOpenAI:Endpoint"]!,
        credential: new DefaultAzureCredential()
    );
    return kernelBuilder.Build();
});

builder.Services.AddScoped<IAgentService, AgentService>();
```

### Async/Await Patterns

```csharp
// ✅ GOOD: Proper async/await
public async Task<string> ProcessQueryAsync(string query)
{
    var result = await _kernel.InvokeAsync<string>(_function, new() { ["input"] = query });
    return result;
}

// ❌ BAD: Blocking async code
public string ProcessQuery(string query)
{
    var result = _kernel.InvokeAsync<string>(_function, new() { ["input"] = query }).Result;
    return result;  // Don't do this!
}
```

### Error Handling with Polly

```csharp
using Polly;
using Polly.Retry;

var retryPolicy = Policy
    .Handle<HttpRequestException>()
    .WaitAndRetryAsync(
        retryCount: 3,
        sleepDurationProvider: attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)),
        onRetry: (exception, timeSpan, retryCount, context) =>
        {
            _logger.LogWarning($"Retry {retryCount} after {timeSpan.TotalSeconds}s due to: {exception.Message}");
        }
    );

var result = await retryPolicy.ExecuteAsync(async () =>
    await kernel.InvokeAsync<string>(function, arguments)
);
```

### Configuration Management

```csharp
// appsettings.json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com/",
    "Deployment": "gpt-4",
    "ApiKey": ""  // Use Key Vault in production
  },
  "AzureAISearch": {
    "Endpoint": "https://your-search.search.windows.net",
    "IndexName": "documents"
  }
}

// Configuration class
public class SkConfiguration
{
    public AzureOpenAIConfig AzureOpenAI { get; set; } = new();
    public AzureAISearchConfig AzureAISearch { get; set; } = new();
}

// Usage
var config = builder.Configuration.Get<SkConfiguration>();
```

---

## Integration with Azure Services

### Supported Azure Services

| Service | Purpose | NuGet Package |
|---------|---------|---------------|
| Azure OpenAI | LLM/Chat/Embeddings | `Microsoft.SemanticKernel.Connectors.AzureOpenAI` |
| Azure AI Search | Vector Store | `Microsoft.SemanticKernel.Connectors.AzureAISearch` |
| Azure Key Vault | Secrets Management | `Azure.Security.KeyVault.Secrets` |
| Application Insights | Monitoring | `Microsoft.ApplicationInsights.AspNetCore` |
| Azure App Service | Web Hosting | - |
| Azure Functions | Serverless | `Microsoft.Azure.Functions.Worker` |

### Quick Azure OpenAI Setup

```csharp
using Azure.Identity;

var builder = Kernel.CreateBuilder();

builder.AddAzureOpenAIChatCompletion(
    deploymentName: Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT")!,
    endpoint: Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!,
    credential: new DefaultAzureCredential()  // Managed Identity
);

var kernel = builder.Build();
```

---

## Performance Optimization

### Key Strategies

1. **Async Everywhere:** Use `async/await` for all I/O operations
2. **Connection Pooling:** Use HttpClient factory pattern
3. **Caching:** Use `IMemoryCache` for function results and embeddings
4. **Batching:** Process multiple requests using `Task.WhenAll`
5. **Streaming:** Stream responses for long-running operations

**See:** Production Guide → Performance Section

---

## Testing

### Test Structure

```
tests/
├── UnitTests/
│   ├── FunctionsTests.cs
│   ├── PluginsTests.cs
│   └── AgentsTests.cs
├── IntegrationTests/
│   ├── AzureIntegrationTests.cs
│   ├── MemoryStoreTests.cs
│   └── WorkflowTests.cs
└── E2ETests/
    ├── ChatScenariosTests.cs
    └── MultiAgentTests.cs
```

### Example xUnit Test

```csharp
using Xunit;
using Moq;

public class KernelTests
{
    [Fact]
    public async Task SimpleFunction_ReturnsExpectedResult()
    {
        // Arrange
        var mockService = new Mock<IChatCompletionService>();
        mockService.Setup(s => s.GetChatMessageContentsAsync(
            It.IsAny<ChatHistory>(),
            It.IsAny<PromptExecutionSettings>(),
            It.IsAny<Kernel>(),
            It.IsAny<CancellationToken>()
        )).ReturnsAsync(new List<ChatMessageContent> {
            new(AuthorRole.Assistant, "Hello from SK")
        });

        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton(mockService.Object);
        var kernel = builder.Build();

        var function = kernel.CreateFunctionFromPrompt("Say hello");

        // Act
        var result = await kernel.InvokeAsync<string>(function);

        // Assert
        Assert.Contains("hello", result, StringComparison.OrdinalIgnoreCase);
    }
}
```

**See:** Production Guide → Testing Strategies

---

## Security Checklist

- [ ] Use Azure Key Vault for secrets (never hardcode API keys)
- [ ] Enable Managed Identity for Azure services
- [ ] Implement input validation and sanitization
- [ ] Use HTTPS for all communications
- [ ] Enable rate limiting and request throttling
- [ ] Log security events (authentication, authorization)
- [ ] Regularly update NuGet packages
- [ ] Use principle of least privilege (Azure RBAC)
- [ ] Implement content filtering and guardrails
- [ ] Audit and monitor API usage and costs

**See:** Production Guide → Security Best Practices

---

## Troubleshooting

### Common Issues

**Issue:** `NuGet package not found`
```bash
# Solution: Ensure you're using correct package names
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Connectors.OpenAI
```

**Issue:** `API key not found` or `Authentication failed`
```csharp
// Solution: Check environment variables or Key Vault
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
if (string.IsNullOrEmpty(apiKey))
{
    throw new InvalidOperationException("OPENAI_API_KEY not set");
}
```

**Issue:** `Token limit exceeded`
```csharp
// Solution: Implement chunking or use smaller models
var settings = new OpenAIPromptExecutionSettings
{
    MaxTokens = 1000
};
```

**Issue:** Async deadlocks in .NET Framework
```csharp
// Solution: Use ConfigureAwait(false) in libraries
var result = await kernel.InvokeAsync(function).ConfigureAwait(false);
```

---

## Additional Resources

### Official Documentation
- [Semantic Kernel GitHub](https://github.com/microsoft/semantic-kernel)
- [.NET Documentation](https://learn.microsoft.com/semantic-kernel/get-started/quick-start-guide?pivots=programming-language-csharp)
- [API Reference](https://learn.microsoft.com/dotnet/api/microsoft.semantickernel)

### .NET Community
- [SK .NET Samples](https://github.com/microsoft/semantic-kernel/tree/main/dotnet/samples)
- [Discussion Forum](https://github.com/microsoft/semantic-kernel/discussions)

### Related Frameworks
- [AutoGen .NET](../AutoGen_Guide/)
- [LangGraph Guide](../LangGraph_Guide/)

---

## Version Support

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| .NET | 6.0 | 8.0+ |
| Microsoft.SemanticKernel | 1.0.0 | 1.67.1+ |
| C# | 10.0 | 12.0+ |
| Azure.Identity | 1.10.0 | Latest |
| Polly | 8.0.0 | Latest |

---

## Contributing

Found an issue or want to contribute?
- Submit issues with .NET version, SK version, and minimal reproduction
- Include full error messages and stack traces
- Provide code examples demonstrating the problem

---

## Quick Navigation

**New to Semantic Kernel?** → Start with [Comprehensive Guide](semantic_kernel_comprehensive_dotnet.md)
**Building for Production?** → See [Production Guide](semantic_kernel_production_dotnet.md)
**Need Examples?** → Check [Recipes](semantic_kernel_recipes_dotnet.md)
**Complete Index?** → Browse [GUIDE_INDEX.md](GUIDE_INDEX.md)

---

**Start building intelligent .NET applications with Semantic Kernel today!**

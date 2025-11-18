# Semantic Kernel Comprehensive Guide (.NET)

**Complete C#/.NET Reference for Building AI Agents and Agentic Systems**

Last Updated: November 2025
.NET Version: 6.0+
Semantic Kernel: 1.38.0+

---

## Overview

This guide provides a complete C#/.NET reference for Semantic Kernel. For detailed conceptual explanations of features, see [Python Comprehensive Guide](../python/semantic_kernel_comprehensive_python.md) - the concepts translate directly, with this guide focusing on C#-specific implementation.

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Kernel Initialization](#2-kernel-initialization)
3. [Service Configuration](#3-service-configuration)
4. [Functions & Plugins](#4-functions--plugins)
5. [Agents & Multi-Agent Systems](#5-agents--multi-agent-systems)
6. [Memory & Vector Stores v1.34](#6-memory--vector-stores-v134)
7. [Planners](#7-planners)
8. [Model Context Protocol (MCP) - March 2025](#8-model-context-protocol-mcp---march-2025)
9. [Google A2A Protocol](#9-google-a2a-protocol)
10. [Microsoft Agent Framework - March 2025](#10-microsoft-agent-framework---march-2025)
11. [Structured Output](#11-structured-output)
12. [Error Handling & Resilience (Polly)](#12-error-handling--resilience-polly)
13. [Observability (Application Insights)](#13-observability-application-insights)
14. [ASP.NET Core Integration](#14-aspnet-core-integration)
15. [Best Practices](#15-best-practices)

---

## 1. Installation & Setup

```bash
# Create project
dotnet new console -n SkApp
cd SkApp

# Core packages
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Connectors.OpenAI
dotnet add package Microsoft.SemanticKernel.Connectors.AzureOpenAI
dotnet add package Microsoft.SemanticKernel.Connectors.AzureAISearch
dotnet add package Microsoft.SemanticKernel.Plugins.Core

# Azure & Production
dotnet add package Azure.Identity
dotnet add package Microsoft.Extensions.Configuration
dotnet add package Microsoft.Extensions.Logging
dotnet add package Polly
dotnet add package Microsoft.ApplicationInsights.AspNetCore
```

### Configuration

```json
// appsettings.json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com/",
    "Deployment": "gpt-4",
    "ApiKey": ""  // Use Key Vault in production
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.SemanticKernel": "Debug"
    }
  }
}
```

---

## 2. Kernel Initialization

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

// Basic kernel
var builder = Kernel.CreateBuilder();

builder.AddOpenAIChatCompletion(
    modelId: "gpt-4",
    apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
);

var kernel = builder.Build();
```

### Azure OpenAI with Managed Identity

```csharp
using Azure.Identity;

builder.AddAzureOpenAIChatCompletion(
    deploymentName: config["AzureOpenAI:Deployment"]!,
    endpoint: config["AzureOpenAI:Endpoint"]!,
    credential: new DefaultAzureCredential()  // Managed Identity
);
```

---

## 3. Service Configuration

### Multiple Services

```csharp
// Add multiple models
builder.Services.AddKeyedSingleton<IChatCompletionService>(
    "gpt-4",
    (sp, key) => new OpenAIChatCompletionService("gpt-4", apiKey)
);

builder.Services.AddKeyedSingleton<IChatCompletionService>(
    "gpt-3.5",
    (sp, key) => new OpenAIChatCompletionService("gpt-3.5-turbo", apiKey)
);

// Use specific service
var result = await kernel.InvokeAsync(
    function,
    new KernelArguments { ["input"] = "query" },
    serviceId: "gpt-4"
);
```

---

## 4. Functions & Plugins

### Native Functions with Attributes

```csharp
using Microsoft.SemanticKernel;
using System.ComponentModel;

public class MathPlugin
{
    [KernelFunction]
    [Description("Adds two numbers")]
    public int Add(
        [Description("First number")] int a,
        [Description("Second number")] int b)
    {
        return a + b;
    }

    [KernelFunction]
    [Description("Multiplies two numbers")]
    public int Multiply(int a, int b) => a * b;
}

// Register plugin
kernel.ImportPluginFromObject(new MathPlugin(), "Math");

// Invoke
var result = await kernel.InvokeAsync(
    "Math",
    "Add",
    new KernelArguments { ["a"] = 5, ["b"] = 3 }
);
```

### Async Plugin

```csharp
public class WebPlugin
{
    private readonly HttpClient _httpClient;

    public WebPlugin(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    [KernelFunction]
    [Description("Fetches content from a URL")]
    public async Task<string> FetchUrlAsync(
        [Description("URL to fetch")] string url)
    {
        var response = await _httpClient.GetAsync(url);
        return await response.Content.ReadAsStringAsync();
    }
}

// Register with DI
builder.Services.AddHttpClient<WebPlugin>();
kernel.ImportPluginFromObject(serviceProvider.GetRequiredService<WebPlugin>(), "Web");
```

### Semantic Functions

{% raw %}
```csharp
var summarizePrompt = """
    Summarize the following text in 3 bullet points:

    {{$input}}
    """;

var settings = new OpenAIPromptExecutionSettings
{
    MaxTokens = 500,
    Temperature = 0.7,
    TopP = 0.9
};

var function = kernel.CreateFunctionFromPrompt(
    summarizePrompt,
    functionName: "Summarize",
    executionSettings: settings
);

var result = await kernel.InvokeAsync<string>(
    function,
    new KernelArguments { ["input"] = "Long text here..." }
);
```
{% endraw %}

---

## 5. Agents & Multi-Agent Systems

### ChatCompletionAgent

```csharp
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;

var agent = new ChatCompletionAgent
{
    Name = "assistant",
    Instructions = "You are a helpful AI assistant.",
    Kernel = kernel
};

var history = new ChatHistory();
history.AddUserMessage("What is Semantic Kernel?");

await foreach (var message in agent.InvokeAsync(history))
{
    Console.WriteLine($"{message.Role}: {message.Content}");
    history.Add(message);
}
```

### AgentGroupChat (Multi-Agent)

```csharp
var researcher = new ChatCompletionAgent
{
    Name = "researcher",
    Instructions = "Research and cite sources.",
    Kernel = kernel
};

var writer = new ChatCompletionAgent
{
    Name = "writer",
    Instructions = "Write engaging content.",
    Kernel = kernel
};

var editor = new ChatCompletionAgent
{
    Name = "editor",
    Instructions = "Edit for quality and consistency.",
    Kernel = kernel
};

var groupChat = new AgentGroupChat(researcher, writer, editor);

await groupChat.AddChatMessageAsync(new ChatMessageContent(
    AuthorRole.User,
    "Create a blog post about Semantic Kernel agents with sources"
));

await foreach (var message in groupChat.InvokeAsync())
{
    Console.WriteLine($"[{message.AuthorName}]: {message.Content}");

    if (message.Content.Contains("DONE", StringComparison.OrdinalIgnoreCase))
        break;
}
```

---

## 6. Memory & Vector Stores v1.34

### Azure AI Search (v1.34)

```csharp
using Microsoft.SemanticKernel.Connectors.AzureAISearch;
using Microsoft.SemanticKernel.Memory;
using Azure.Search.Documents.Indexes;

// Create memory store (v1.34+)
var searchClient = new SearchIndexClient(
    new Uri(config["AzureAISearch:Endpoint"]!),
    new DefaultAzureCredential()
);

var memoryStore = new AzureAISearchMemoryStore(
    searchClient,
    vectorSize: 1536
);

// Embedding service
var embeddingService = new OpenAITextEmbeddingGenerationService(
    modelId: "text-embedding-ada-002",
    apiKey: apiKey
);

// Create memory
var memory = new SemanticTextMemory(memoryStore, embeddingService);

// Save
await memory.SaveInformationAsync(
    collection: "documents",
    id: "doc_1",
    text: "Semantic Kernel is a lightweight SDK...",
    description: "SK Overview",
    additionalMetadata: new Dictionary<string, object>
    {
        ["category"] = "documentation",
        ["version"] = "1.0"
    }
);

// Search with metadata filtering (NEW in v1.34)
var results = await memory.SearchAsync(
    collection: "documents",
    query: "What is Semantic Kernel?",
    limit: 5,
    minRelevanceScore: 0.75,
    filter: new { category = "documentation" }  // NEW: Metadata filtering
).ToListAsync();
```

---

## 7. Planners

### Sequential Planner

```csharp
using Microsoft.SemanticKernel.Planning;

var planner = new SequentialPlanner(kernel);

var ask = "Get the weather in Seattle and email it to alice@example.com";
var plan = await planner.CreatePlanAsync(ask);

Console.WriteLine("Plan steps:");
foreach (var step in plan.Steps)
{
    Console.WriteLine($"  - {step.Name}: {step.Description}");
}

var result = await plan.InvokeAsync(kernel);
Console.WriteLine($"Result: {result}");
```

### Stepwise Planner

```csharp
var config = new StepwisePlannerConfig
{
    MaxIterations = 10,
    MinIterationTimeMs = 1000,
    MaxTokens = 4000
};

var planner = new StepwisePlanner(kernel, config);

var result = await planner.ExecuteAsync("Research AI trends and create a summary");
Console.WriteLine($"Result: {result.FinalAnswer}");
```

---

## 8. Model Context Protocol (MCP) - March 2025

### MCP Client

```csharp
using Microsoft.SemanticKernel.Connectors.MCP;

// Connect to MCP server
var mcpClient = new MCPClient(new MCPClientConfig
{
    ServerUrl = "http://localhost:3000",
    Timeout = TimeSpan.FromSeconds(30)
});

await mcpClient.ConnectAsync();

// List tools
var tools = await mcpClient.ListToolsAsync();
foreach (var tool in tools)
{
    Console.WriteLine($"Tool: {tool.Name} - {tool.Description}");
}

// Create plugin from MCP tools
var mcpPlugin = await MCPToolPlugin.FromMcpServerAsync(
    "http://localhost:3000",
    "MCPTools"
);

kernel.ImportPluginFromObject(mcpPlugin, "MCP");

// Agent can now use MCP tools
var agent = new ChatCompletionAgent
{
    Name = "mcp_agent",
    Instructions = "You have access to MCP tools. Use them to answer questions.",
    Kernel = kernel
};
```

### MCP Server

```csharp
using Microsoft.SemanticKernel.Connectors.MCP;

public class SkMcpServer
{
    private readonly Kernel _kernel;
    private readonly MCPServer _server;

    public SkMcpServer(Kernel kernel, int port = 3000)
    {
        _kernel = kernel;
        _server = new MCPServer(new MCPServerConfig
        {
            Name = "SK MCP Server",
            Version = "1.0.0",
            Port = port
        });

        RegisterHandlers();
    }

    private void RegisterHandlers()
    {
        // Expose SK plugins as MCP tools
        _server.OnListTools(async () =>
        {
            var tools = new List<MCPTool>();

            foreach (var plugin in _kernel.Plugins)
            {
                foreach (var function in plugin)
                {
                    tools.Add(new MCPTool
                    {
                        Name = $"{plugin.Name}.{function.Name}",
                        Description = function.Description ?? "",
                        InputSchema = ConvertToJsonSchema(function)
                    });
                }
            }

            return tools;
        });

        _server.OnCallTool(async (name, arguments) =>
        {
            var parts = name.Split('.');
            var result = await _kernel.InvokeAsync(
                parts[0],
                parts[1],
                new KernelArguments(arguments)
            );

            return new { result = result.ToString() };
        });
    }

    public async Task StartAsync()
    {
        await _server.StartAsync();
        Console.WriteLine($"MCP Server running on port {_server.Config.Port}");
    }
}
```

---

## 9. Google A2A Protocol

**See:** [Python Comprehensive Guide → Section 18](../python/semantic_kernel_comprehensive_python.md#18-google-a2a-protocol-integration---2025) for detailed A2A protocol patterns.

**C# Implementation:**

```csharp
public record A2AMessage(
    string Id,
    A2AMessageType Type,
    string Sender,
    string Recipient,
    Dictionary<string, object> Payload,
    string? CorrelationId = null
);

public enum A2AMessageType
{
    Request,
    Response,
    Event,
    Error
}

public class SkA2AAgent
{
    private readonly ChatCompletionAgent _agent;
    private readonly A2AMessageBus _messageBus;
    private readonly string _agentId;

    public SkA2AAgent(
        ChatCompletionAgent agent,
        A2AMessageBus messageBus,
        string agentId)
    {
        _agent = agent;
        _messageBus = messageBus;
        _agentId = agentId;

        _messageBus.RegisterAgent(agentId, this);
    }

    public async Task HandleMessageAsync(A2AMessage message)
    {
        if (message.Type == A2AMessageType.Request)
        {
            var history = new ChatHistory();
            history.AddUserMessage(message.Payload["content"].ToString()!);

            var result = "";
            await foreach (var msg in _agent.InvokeAsync(history))
            {
                result = msg.Content;
            }

            var response = new A2AMessage(
                Guid.NewGuid().ToString(),
                A2AMessageType.Response,
                _agentId,
                message.Sender,
                new Dictionary<string, object> { ["result"] = result },
                message.Id
            );

            await _messageBus.SendAsync(response);
        }
    }
}
```

---

## 10. Microsoft Agent Framework - March 2025

**.NET is the primary platform for Microsoft Agent Framework (March 2025)**

```csharp
using Microsoft.AgentFramework;

// Create unified agent (March 2025+)
var agent = new UnifiedAgent
{
    Name = "sk_unified_agent",
    Framework = "semantic-kernel",
    Capabilities = new[]
    {
        AgentCapability.Chat,
        AgentCapability.ToolUse,
        AgentCapability.Memory
    },
    GovernancePolicy = new AgentGovernance
    {
        MaxTokensPerHour = 100000,
        AllowedTools = new[] { "approved_tools/*" },
        ContentFilterEnabled = true,
        AuditLogging = true
    }
};

// Register with enterprise registry
var registry = new AgentRegistry(
    registryEndpoint,
    new DefaultAzureCredential()
);

await registry.RegisterAsync(agent);

// Cross-framework communication
var client = new AgentClient(registryEndpoint, new DefaultAzureCredential());

var agents = await client.DiscoverAgentsAsync(new AgentQuery
{
    Capabilities = new[] { AgentCapability.Research },
    Framework = "autogen"  // Find AutoGen agents
});

// Send message to AutoGen agent from SK
var response = await client.SendMessageAsync(
    toAgent: agents[0].Id,
    message: "Research quantum computing trends",
    senderContext: new { framework = "semantic-kernel", agent = "sk_agent" }
);
```

---

## 11. Structured Output

{% raw %}
```csharp
using System.Text.Json;
using System.Text.Json.Serialization;

public record CustomerInfo(
    [property: JsonPropertyName("customer_id")] string CustomerId,
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("email")] string Email,
    [property: JsonPropertyName("phone")] string? Phone = null,
    [property: JsonPropertyName("purchase_history")] List<string>? PurchaseHistory = null,
    [property: JsonPropertyName("lifetime_value")] decimal LifetimeValue = 0
);

var prompt = """
    Extract customer information from the following text and return as JSON:

    Text: {{$input}}

    Return JSON matching this schema:
    {
      "customer_id": "string",
      "name": "string",
      "email": "string",
      "phone": "string (optional)",
      "purchase_history": ["string"],
      "lifetime_value": number
    }
    """;

var function = kernel.CreateFunctionFromPrompt(prompt);

var text = "Customer John Doe (ID: C12345, email: john@example.com) has purchased items A, B, C. LTV: $1,250.50";
var result = await kernel.InvokeAsync<string>(function, new() { ["input"] = text });

var customer = JsonSerializer.Deserialize<CustomerInfo>(result);
Console.WriteLine($"Customer: {customer.Name}, LTV: ${customer.LifetimeValue}");
```
{% endraw %}

---

## 12. Error Handling & Resilience (Polly)

```csharp
using Polly;
using Polly.CircuitBreaker;
using Polly.Retry;

public class ResilientKernel
{
    private readonly Kernel _kernel;
    private readonly AsyncRetryPolicy _retryPolicy;
    private readonly AsyncCircuitBreakerPolicy _circuitBreakerPolicy;

    public ResilientKernel(Kernel kernel)
    {
        _kernel = kernel;

        // Retry policy
        _retryPolicy = Policy
            .Handle<HttpRequestException>()
            .Or<TimeoutException>()
            .WaitAndRetryAsync(
                retryCount: 3,
                sleepDurationProvider: attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt))
            );

        // Circuit breaker
        _circuitBreakerPolicy = Policy
            .Handle<HttpRequestException>()
            .CircuitBreakerAsync(
                exceptionsAllowedBeforeBreaking: 5,
                durationOfBreak: TimeSpan.FromMinutes(1)
            );
    }

    public async Task<T> InvokeWithResilienceAsync<T>(
        KernelFunction function,
        KernelArguments arguments)
    {
        return await _circuitBreakerPolicy.ExecuteAsync(async () =>
            await _retryPolicy.ExecuteAsync(async () =>
                await _kernel.InvokeAsync<T>(function, arguments)
            )
        );
    }
}
```

---

## 13. Observability (Application Insights)

```csharp
using Microsoft.ApplicationInsights;
using Microsoft.ApplicationInsights.DataContracts;

public class MonitoredKernel
{
    private readonly Kernel _kernel;
    private readonly TelemetryClient _telemetry;

    public MonitoredKernel(Kernel kernel, TelemetryClient telemetry)
    {
        _kernel = kernel;
        _telemetry = telemetry;
    }

    public async Task<T> InvokeWithMonitoringAsync<T>(
        KernelFunction function,
        KernelArguments arguments)
    {
        using var operation = _telemetry.StartOperation<RequestTelemetry>("SK_Invoke");
        operation.Telemetry.Properties["function"] = function.Name;

        try
        {
            var result = await _kernel.InvokeAsync<T>(function, arguments);
            operation.Telemetry.Success = true;
            return result;
        }
        catch (Exception ex)
        {
            operation.Telemetry.Success = false;
            _telemetry.TrackException(ex);
            throw;
        }
    }
}
```

---

## 14. ASP.NET Core Integration

```csharp
// Program.cs
var builder = WebApplication.CreateBuilder(args);

// Add Semantic Kernel to DI
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

// Add Application Insights
builder.Services.AddApplicationInsightsTelemetry();

// Add controllers
builder.Services.AddControllers();

var app = builder.Build();
app.MapControllers();
app.Run();
```

{% raw %}
```csharp
// ChatController.cs
[ApiController]
[Route("api/[controller]")]
public class ChatController : ControllerBase
{
    private readonly Kernel _kernel;
    private readonly ILogger<ChatController> _logger;

    public ChatController(Kernel kernel, ILogger<ChatController> logger)
    {
        _kernel = kernel;
        _logger = logger;
    }

    [HttpPost]
    public async Task<IActionResult> Chat([FromBody] ChatRequest request)
    {
        try
        {
            var function = _kernel.CreateFunctionFromPrompt(
                "Answer the following: {{$input}}"
            );

            var result = await _kernel.InvokeAsync<string>(
                function,
                new KernelArguments { ["input"] = request.Message }
            );

            return Ok(new { response = result });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Chat failed");
            return StatusCode(500, new { error = ex.Message });
        }
    }
}

public record ChatRequest(string Message);
```
{% endraw %}

---

## 15. Best Practices

### Async/Await

```csharp
// ✅ GOOD
public async Task<string> ProcessAsync(string input)
{
    return await _kernel.InvokeAsync<string>(_function, new() { ["input"] = input });
}

// ❌ BAD
public string Process(string input)
{
    return _kernel.InvokeAsync<string>(_function, new() { ["input"] = input }).Result;
}
```

### Dependency Injection

```csharp
// ✅ GOOD
builder.Services.AddSingleton<Kernel>(...);
builder.Services.AddScoped<IAgentService, AgentService>();

// Use constructor injection
public class MyService
{
    private readonly Kernel _kernel;

    public MyService(Kernel kernel)
    {
        _kernel = kernel;
    }
}
```

### Configuration

```csharp
// ✅ GOOD: Use IOptions pattern
public class SkOptions
{
    public string OpenAIApiKey { get; set; } = string.Empty;
    public string AzureOpenAIEndpoint { get; set; } = string.Empty;
}

builder.Services.Configure<SkOptions>(builder.Configuration.GetSection("SemanticKernel"));

// Inject
public class MyService
{
    private readonly SkOptions _options;

    public MyService(IOptions<SkOptions> options)
    {
        _options = options.Value;
    }
}
```

---

## Conclusion

This guide covers Semantic Kernel .NET with 2025 features:

- ✅ Complete C# implementation patterns
- ✅ ASP.NET Core integration
- ✅ Polly resilience patterns
- ✅ Application Insights monitoring
- ✅ **Model Context Protocol (MCP)** - March 2025
- ✅ **Microsoft Agent Framework** - Primary platform (March 2025)
- ✅ **Vector Store v1.34**
- ✅ **Google A2A Protocol**

**Related Guides:**
- [Production Guide](semantic_kernel_production_dotnet.md)
- [Recipes](semantic_kernel_recipes_dotnet.md)
- [Python Comprehensive Guide](../python/semantic_kernel_comprehensive_python.md) - for detailed conceptual patterns

**[Back to .NET README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

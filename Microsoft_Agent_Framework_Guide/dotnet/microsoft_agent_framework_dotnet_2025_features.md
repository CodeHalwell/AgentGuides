# Microsoft Agent Framework - .NET 2025 Features Guide
## October 2025 Release - Advanced .NET-Specific Features

**Document Version:** 1.0
**Last Updated:** November 2025
**Target Audience:** .NET/C# Developers
**Preview Status:** October 2025 Public Preview
**.NET Version:** 8.0+

---

## Table of Contents

1. [Agent2Agent (A2A) Protocol - .NET](#agent2agent-a2a-protocol---net)
2. [Graph-Based Workflows - .NET](#graph-based-workflows---net)
3. [Declarative Agents with .NET](#declarative-agents-with-net)
4. [OpenTelemetry Integration - .NET](#opentelemetry-integration---net)
5. [Content Safety - .NET](#content-safety---net)
6. [Dependency Injection Patterns](#dependency-injection-patterns)

---

## Agent2Agent (A2A) Protocol - .NET

### Quick Start with A2A

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.A2A;
using Azure.Identity;
using System;
using System.Threading.Tasks;

public class A2AExample
{
    public static async Task SetupA2AAgentAsync()
    {
        // Create base agent
        var agent = new ChatAgent(
            instructions: "You are a helpful sales data agent.",
            name: "SalesAgent"
        );

        // Wrap with A2A adapter
        var a2aAdapter = new A2AProtocolAdapter(
            agent: agent,
            agentId: "agent_sales_001",
            endpoint: new Uri("https://api.contoso.com/agents/sales"),
            authentication: new A2AAuthentication
            {
                Method = A2AAuthMethod.EntraId,
                Credential = new DefaultAzureCredential()
            }
        );

        // Register message handler
        a2aAdapter.OnMessage(async (message) =>
        {
            var query = message.Payload.Content?.ToString();
            var response = await agent.RunAsync(query);

            return new A2AResponse
            {
                Type = "response",
                Status = "success",
                Content = response
            };
        });

        await a2aAdapter.RegisterAsync();
        return a2aAdapter;
    }

    public static async Task CallOpenAIAgentAsync()
    {
        // Call external OpenAI agent via A2A
        var client = new A2AClient(
            agentId: "agent_sales_001",
            authentication: new A2AAuthentication
            {
                Method = A2AAuthMethod.OAuth2,
                ClientId = "your_client_id",
                ClientSecret = "your_client_secret"
            }
        );

        var message = new A2AMessage
        {
            RecipientAgentId = "agent_openai_analytics",
            RecipientEndpoint = new Uri("https://api.partner.com/agents/analytics"),
            RecipientFramework = "openai-sdk",
            Payload = new A2APayload
            {
                Type = "request",
                Action = "query",
                Content = "Analyze Q3 sales trends"
            }
        };

        var response = await client.SendMessageAsync(message);
        Console.WriteLine($"Response: {response.Payload.Content}");
    }
}
```

### Type-Safe A2A with Records

```csharp
using System.Text.Json;

// Define strongly-typed request/response models
public record SalesQuery(
    string QueryType,
    string DateRange,
    string? Region = null,
    string Currency = "USD"
);

public record SalesResponse(
    decimal TotalSales,
    string Period,
    string Region,
    string Currency,
    Dictionary<string, decimal> Breakdown
);

public class TypeSafeA2A
{
    public static async Task<SalesResponse> TypeSafeA2ACallAsync(SalesQuery query)
    {
        var client = new A2AClient(agentId: "sales_agent");

        var message = new A2AMessage
        {
            RecipientAgentId = "analytics_agent",
            RecipientEndpoint = new Uri("https://api.partner.com/agents/analytics"),
            Payload = new A2APayload
            {
                Type = "request",
                Action = "query_sales",
                Content = JsonSerializer.Serialize(query)
            }
        };

        var response = await client.SendMessageAsync(message);

        // Deserialize with type safety
        var salesData = JsonSerializer.Deserialize<SalesResponse>(
            response.Payload.Content?.ToString()
        );

        return salesData!;
    }

    public static async Task Main()
    {
        var query = new SalesQuery(
            QueryType: "regional_sales",
            DateRange: "2025-Q3",
            Region: "EMEA",
            Currency: "EUR"
        );

        var result = await TypeSafeA2ACallAsync(query);
        Console.WriteLine($"Total Sales: {result.TotalSales} {result.Currency}");
        Console.WriteLine($"Breakdown: {string.Join(", ", result.Breakdown)}");
    }
}
```

---

## Graph-Based Workflows - .NET

### Building Complex Workflows

```csharp
using Microsoft.Agents.AI.Graphs;
using Microsoft.Agents.AI;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public class ContentPipeline
{
    public static async Task<AgentGraph> CreateContentPipelineAsync()
    {
        // Define specialized agents
        var researchAgent = new ChatAgent(
            instructions: "You research topics thoroughly.",
            name: "ResearchAgent"
        );

        var writerAgent = new ChatAgent(
            instructions: "You write engaging content.",
            name: "WriterAgent"
        );

        var editorAgent = new ChatAgent(
            instructions: "You edit and improve content quality.",
            name: "EditorAgent"
        );

        var seoAgent = new ChatAgent(
            instructions: "You optimize content for SEO.",
            name: "SEOAgent"
        );

        // Create graph
        var graph = new AgentGraph("ContentPipeline");

        // Add nodes
        graph.AddNode("research", researchAgent);
        graph.AddNode("write", writerAgent);
        graph.AddNode("edit", editorAgent);
        graph.AddNode("seo", seoAgent);

        // Linear workflow
        graph.AddEdge("research", "write");
        graph.AddEdge("write", "edit");
        graph.AddEdge("edit", "seo");

        graph.SetEntryPoint("research");

        return graph;
    }

    public static async Task RunWithStreamingAsync(string topic)
    {
        var graph = await CreateContentPipelineAsync();

        Console.WriteLine($"Creating content for: {topic}\n");

        // Stream execution events
        await foreach (var evt in graph.StreamAsync(
            initialState: new Dictionary<string, object>
            {
                ["topic"] = topic,
                ["target_length"] = 1500
            },
            streamMode: StreamMode.Updates
        ))
        {
            var nodeId = evt.Node;
            var eventType = evt.Type;

            switch (eventType)
            {
                case "node_start":
                    Console.WriteLine($"▶ Starting: {nodeId}");
                    break;
                case "node_output":
                    var output = evt.Data?["output"]?.ToString();
                    Console.WriteLine($"  Output preview: {output?[..Math.Min(100, output.Length)]}...");
                    break;
                case "node_complete":
                    Console.WriteLine($"✓ Completed: {nodeId}\n");
                    break;
            }
        }

        var finalResult = await graph.GetFinalStateAsync();
        Console.WriteLine($"Final content ready!");
    }
}
```

### Conditional Routing

```csharp
public class ConditionalWorkflow
{
    public static async Task<AgentGraph> CreateConditionalWorkflowAsync()
    {
        var classifierAgent = new ChatAgent(
            instructions: "Classify customer queries by urgency and type."
        );

        var simpleAgent = new ChatAgent(
            instructions: "Handle simple queries quickly."
        );

        var complexAgent = new ChatAgent(
            instructions: "Handle complex queries with detailed analysis."
        );

        var urgentAgent = new ChatAgent(
            instructions: "Handle urgent queries with priority."
        );

        var graph = new AgentGraph("CustomerSupport");

        graph.AddNode("classifier", classifierAgent);
        graph.AddNode("simple", simpleAgent);
        graph.AddNode("complex", complexAgent);
        graph.AddNode("urgent", urgentAgent);

        // Conditional routing function
        string RouteQuery(Dictionary<string, object> state)
        {
            var urgency = state.GetValueOrDefault("urgency", "normal")?.ToString();
            var complexity = state.GetValueOrDefault("complexity", "simple")?.ToString();

            return urgency == "high" ? "urgent"
                : complexity == "high" ? "complex"
                : "simple";
        }

        graph.AddConditionalEdge(
            source: "classifier",
            pathFunction: RouteQuery,
            pathMap: new Dictionary<string, string>
            {
                ["simple"] = "simple",
                ["complex"] = "complex",
                ["urgent"] = "urgent"
            }
        );

        return graph;
    }
}
```

### Parallel Execution

```csharp
public class ParallelWorkflow
{
    public static async Task<AgentGraph> CreateParallelAnalysisAsync()
    {
        var sentimentAgent = new ChatAgent(instructions: "Analyze sentiment");
        var keywordAgent = new ChatAgent(instructions: "Extract keywords");
        var summaryAgent = new ChatAgent(instructions: "Summarize text");
        var aggregatorAgent = new ChatAgent(instructions: "Combine results");

        var graph = new AgentGraph("ParallelAnalysis");

        graph.AddNode("sentiment", sentimentAgent);
        graph.AddNode("keywords", keywordAgent);
        graph.AddNode("summary", summaryAgent);
        graph.AddNode("aggregator", aggregatorAgent);

        // Parallel fan-out
        var parallelNodes = new[] { "sentiment", "keywords", "summary" };
        foreach (var node in parallelNodes)
        {
            graph.AddEdge("START", node);
            graph.AddEdge(node, "aggregator");
        }

        return graph;
    }

    public static async Task RunParallelAsync()
    {
        var graph = await CreateParallelAnalysisAsync();

        var result = await graph.RunAsync(new Dictionary<string, object>
        {
            ["text"] = "Sample text to analyze..."
        });

        Console.WriteLine($"Aggregated results: {result["aggregator"]}");
    }
}
```

### Checkpointing and Recovery

```csharp
using Microsoft.Agents.AI.Graphs;

public class CheckpointedWorkflow
{
    public static async Task<AgentGraph> CreateCheckpointedWorkflowAsync()
    {
        var graph = new AgentGraph("DataProcessing");

        // Configure checkpointing
        var checkpointConfig = new CheckpointConfig
        {
            Backend = "azure-storage",
            ConnectionString = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION"),
            Container = "agent-checkpoints",
            CheckpointInterval = CheckpointInterval.NodeCompletion,
            RetentionDays = 7
        };

        graph.ConfigureCheckpointing(checkpointConfig);

        // Add nodes...

        return graph;
    }

    public static async Task<object> ResumeFromCheckpointAsync(string checkpointId)
    {
        var graph = await CreateCheckpointedWorkflowAsync();

        // Load checkpoint
        var checkpoint = await graph.LoadCheckpointAsync(checkpointId);

        Console.WriteLine($"Resuming from: {checkpoint.CurrentNode}");
        Console.WriteLine($"State: {checkpoint.State}");

        // Resume execution
        var result = await graph.ResumeAsync(checkpoint);

        return result;
    }
}
```

### Human-in-the-Loop

```csharp
using Microsoft.Agents.AI.Graphs;

public class ApprovalWorkflow
{
    public static async Task<AgentGraph> CreateApprovalWorkflowAsync()
    {
        var graph = new AgentGraph("ContentApproval");

        var draftAgent = new ChatAgent(instructions: "Draft content");
        var reviewAgent = new ChatAgent(instructions: "Review content");
        var publishAgent = new ChatAgent(instructions: "Publish content");

        graph.AddNode("draft", draftAgent);
        graph.AddNode("review", reviewAgent);
        graph.AddNode("publish", publishAgent);

        // Configure HITL
        var hitlConfig = new HITLConfig
        {
            Enabled = true,
            ApprovalEndpoint = new Uri("https://approval.contoso.com/api/requests"),
            NotificationChannels = new[] { "email", "teams" },
            TimeoutSeconds = 3600
        };

        graph.AddHITLCheckpoint(
            afterNode: "review",
            config: hitlConfig,
            approvalPrompt: "Review and approve this content"
        );

        graph.AddEdge("draft", "review");
        graph.AddEdge("review", "publish");

        return graph;
    }

    public static async Task RunInteractiveApprovalAsync()
    {
        var graph = await CreateApprovalWorkflowAsync();

        await foreach (var evt in graph.StreamAsync(
            new Dictionary<string, object> { ["topic"] = "Product Launch" }
        ))
        {
            if (evt.Type == "hitl_required")
            {
                var content = evt.Content?.ToString();
                var requestId = evt.RequestId;

                Console.WriteLine("\n" + new string('=', 60));
                Console.WriteLine("APPROVAL REQUIRED");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine($"Content:\n{content}");
                Console.WriteLine(new string('=', 60));

                Console.Write("Approve? (yes/no): ");
                var decision = Console.ReadLine()?.ToLower();

                if (decision == "yes")
                {
                    await graph.ApproveHITLAsync(requestId);
                }
                else
                {
                    Console.Write("Rejection reason: ");
                    var reason = Console.ReadLine();
                    await graph.RejectHITLAsync(requestId, reason);
                }
            }
        }
    }
}
```

---

## Declarative Agents with .NET

### JSON Configuration

```json
{
  "agent": {
    "name": "CustomerSupportAgent",
    "version": "1.0.0",
    "model": {
      "provider": "azure-openai",
      "deployment": "gpt-4o-mini",
      "temperature": 0.7,
      "maxTokens": 1000
    },
    "instructions": "You are a helpful customer support agent.\nProvide clear, concise answers.",
    "tools": [
      {
        "name": "search_kb",
        "type": "function",
        "function": {
          "assembly": "Tools.KnowledgeBase",
          "type": "KnowledgeBaseTools",
          "method": "SearchKnowledgeBase"
        }
      }
    ],
    "memory": {
      "type": "persistent",
      "backend": "cosmos-db",
      "connectionString": "${COSMOS_CONNECTION_STRING}"
    },
    "authentication": {
      "method": "entra-id",
      "tenantId": "${AZURE_TENANT_ID}"
    }
  }
}
```

### Loading Configuration

```csharp
using Microsoft.Agents.AI.Config;
using System.Text.Json;
using Azure.Identity;

public class ConfigurationLoader
{
    public static async Task<ChatAgent> LoadAgentFromJsonAsync(string configPath)
    {
        // Read configuration file
        var json = await File.ReadAllTextAsync(configPath);

        // Deserialize to strongly-typed config
        var config = JsonSerializer.Deserialize<AgentConfiguration>(json);

        // Load agent from configuration
        var agent = await AgentConfiguration.LoadAgentAsync(
            config: config!,
            credential: new DefaultAzureCredential()
        );

        return agent;
    }

    public static async Task Main()
    {
        var agent = await LoadAgentFromJsonAsync("agent-config.json");

        // Agent is fully configured and ready
        var result = await agent.RunAsync("Help me with my order");
        Console.WriteLine(result);
    }
}
```

### Strongly-Typed Configuration Models

```csharp
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

public record ModelConfiguration
{
    [Required]
    [JsonPropertyName("provider")]
    public string Provider { get; init; } = "azure-openai";

    [Required]
    [JsonPropertyName("deployment")]
    public string Deployment { get; init; } = default!;

    [Range(0.0, 2.0)]
    [JsonPropertyName("temperature")]
    public double Temperature { get; init; } = 0.7;

    [Range(1, int.MaxValue)]
    [JsonPropertyName("maxTokens")]
    public int MaxTokens { get; init; } = 1000;
}

public record ToolConfiguration
{
    [Required]
    public string Name { get; init; } = default!;

    public string Type { get; init; } = "function";

    public FunctionConfig Function { get; init; } = default!;
}

public record FunctionConfig
{
    [Required]
    public string Assembly { get; init; } = default!;

    [Required]
    public string Type { get; init; } = default!;

    [Required]
    public string Method { get; init; } = default!;
}

public record AgentConfiguration
{
    [Required]
    public string Name { get; init; } = default!;

    public string Version { get; init; } = "1.0.0";

    [Required]
    public ModelConfiguration Model { get; init; } = default!;

    [Required]
    public string Instructions { get; init; } = default!;

    public List<ToolConfiguration> Tools { get; init; } = new();

    // Validation method
    public bool Validate()
    {
        return !string.IsNullOrWhiteSpace(Name)
            && !string.IsNullOrWhiteSpace(Instructions)
            && Model != null;
    }
}
```

### Configuration with Options Pattern

```csharp
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;

public class AgentOptions
{
    public const string Section = "Agent";

    public string Name { get; set; } = default!;
    public string Instructions { get; set; } = default!;
    public ModelOptions Model { get; set; } = default!;
}

public class ModelOptions
{
    public string Provider { get; set; } = "azure-openai";
    public string Deployment { get; set; } = default!;
    public double Temperature { get; set; } = 0.7;
    public int MaxTokens { get; set; } = 1000;
}

public class AgentService
{
    private readonly AgentOptions _options;
    private readonly ChatAgent _agent;

    public AgentService(IOptions<AgentOptions> options)
    {
        _options = options.Value;
        _agent = new ChatAgent(
            instructions: _options.Instructions,
            name: _options.Name
        );
    }

    public async Task<string> ExecuteAsync(string query)
    {
        var result = await _agent.RunAsync(query);
        return result;
    }
}

// appsettings.json
/*
{
  "Agent": {
    "Name": "CustomerSupportAgent",
    "Instructions": "You are a helpful support agent.",
    "Model": {
      "Provider": "azure-openai",
      "Deployment": "gpt-4o-mini",
      "Temperature": 0.7,
      "MaxTokens": 1000
    }
  }
}
*/

// Startup configuration
public class Startup
{
    public void ConfigureServices(IServiceCollection services, IConfiguration configuration)
    {
        // Bind configuration
        services.Configure<AgentOptions>(
            configuration.GetSection(AgentOptions.Section)
        );

        // Register service
        services.AddSingleton<AgentService>();
    }
}
```

---

## OpenTelemetry Integration - .NET

### Complete Observability Setup

```csharp
using OpenTelemetry;
using OpenTelemetry.Trace;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using Microsoft.Agents.AI.Telemetry;
using Azure.Monitor.OpenTelemetry.Exporter;
using System.Diagnostics;

public class TelemetrySetup
{
    public static void ConfigureOpenTelemetry()
    {
        var serviceName = "customer-support-agents";
        var serviceVersion = "1.0.0";

        // Configure Tracing
        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(ResourceBuilder.CreateDefault()
                .AddService(serviceName: serviceName, serviceVersion: serviceVersion)
                .AddAttributes(new Dictionary<string, object>
                {
                    ["deployment.environment"] = "production",
                    ["service.region"] = "eastus"
                }))
            .AddSource("Microsoft.Agents.AI.*")
            .AddHttpClientInstrumentation()
            .AddAspNetCoreInstrumentation()
            .AddOtlpExporter(options =>
            {
                options.Endpoint = new Uri("https://otel-collector.contoso.com:4317");
            })
            .AddAzureMonitorTraceExporter(options =>
            {
                options.ConnectionString = Environment.GetEnvironmentVariable(
                    "APPLICATIONINSIGHTS_CONNECTION_STRING"
                );
            })
            .Build();

        // Configure Metrics
        using var meterProvider = Sdk.CreateMeterProviderBuilder()
            .SetResourceBuilder(ResourceBuilder.CreateDefault()
                .AddService(serviceName: serviceName, serviceVersion: serviceVersion))
            .AddMeter("Microsoft.Agents.AI.*")
            .AddHttpClientInstrumentation()
            .AddAspNetCoreInstrumentation()
            .AddOtlpExporter()
            .AddAzureMonitorMetricExporter()
            .Build();

        // Enable agent telemetry
        AgentTelemetry.Configure(new AgentTelemetryOptions
        {
            TraceAllOperations = true,
            IncludeMessageContent = false, // Exclude PII
            CustomTags = new Dictionary<string, string>
            {
                ["environment"] = "production",
                ["region"] = "eastus"
            }
        });
    }

    public static async Task ExecuteWithTracingAsync()
    {
        ConfigureOpenTelemetry();

        var activitySource = new ActivitySource("MyApp.Agents");

        using var activity = activitySource.StartActivity("customer_support_workflow");
        activity?.SetTag("customer.id", "cust_12345");
        activity?.SetTag("query.type", "order_status");

        var agent = new ChatAgent(
            instructions: "You help customers with orders."
        );

        var response = await agent.RunAsync("Where is my order?");

        activity?.SetTag("response.length", response.Length);
        activity?.AddEvent(new ActivityEvent("response_generated"));

        Console.WriteLine(response);
    }
}
```

### Custom Metrics

```csharp
using System.Diagnostics.Metrics;

public class AgentMetrics
{
    private static readonly Meter Meter = new("MyApp.Agents", "1.0");

    private static readonly Counter<long> RequestsCounter = Meter.CreateCounter<long>(
        name: "agent.requests.total",
        unit: "1",
        description: "Total agent requests"
    );

    private static readonly Histogram<double> LatencyHistogram = Meter.CreateHistogram<double>(
        name: "agent.latency",
        unit: "ms",
        description: "Agent request latency"
    );

    private static readonly UpDownCounter<int> ActiveRequestsCounter = Meter.CreateUpDownCounter<int>(
        name: "agent.requests.active",
        unit: "1",
        description: "Currently active requests"
    );

    public static async Task<string> MonitoredExecutionAsync(string query)
    {
        var tags = new TagList
        {
            { "agent", "support" }
        };

        ActiveRequestsCounter.Add(1, tags);

        var stopwatch = Stopwatch.StartNew();

        try
        {
            var agent = new ChatAgent(instructions: "You help customers.");
            var response = await agent.RunAsync(query);

            // Record success
            RequestsCounter.Add(1, new TagList
            {
                { "agent", "support" },
                { "status", "success" }
            });

            stopwatch.Stop();
            LatencyHistogram.Record(stopwatch.Elapsed.TotalMilliseconds, tags);

            return response;
        }
        catch (Exception ex)
        {
            // Record failure
            RequestsCounter.Add(1, new TagList
            {
                { "agent", "support" },
                { "status", "error" },
                { "error_type", ex.GetType().Name }
            });

            throw;
        }
        finally
        {
            ActiveRequestsCounter.Add(-1, tags);
        }
    }
}
```

---

## Content Safety - .NET

### Azure AI Content Safety Integration

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Safety;
using Azure.AI.ContentSafety;
using Azure.Identity;

public class SafeAgent
{
    public static ChatAgent CreateSafeAgent()
    {
        // Configure Content Safety
        var contentSafetyConfig = new ContentSafetyConfig
        {
            Endpoint = new Uri(Environment.GetEnvironmentVariable("CONTENT_SAFETY_ENDPOINT")!),
            Credential = new DefaultAzureCredential(),
            Categories = new Dictionary<string, CategoryConfig>
            {
                ["Hate"] = new CategoryConfig
                {
                    SeverityThreshold = 2,
                    BlockOnViolation = true
                },
                ["Sexual"] = new CategoryConfig
                {
                    SeverityThreshold = 2,
                    BlockOnViolation = true
                },
                ["Violence"] = new CategoryConfig
                {
                    SeverityThreshold = 4,
                    BlockOnViolation = true
                },
                ["SelfHarm"] = new CategoryConfig
                {
                    SeverityThreshold = 2,
                    BlockOnViolation = true
                }
            },
            CheckInput = true,
            CheckOutput = true,
            ActionOnViolation = ViolationAction.BlockAndLog
        };

        var agent = new ChatAgent(
            instructions: "You are a customer support agent.",
            contentSafety: contentSafetyConfig
        );

        return agent;
    }

    public static async Task SafeExecutionAsync()
    {
        var agent = CreateSafeAgent();

        try
        {
            var result = await agent.RunAsync("User query here");
            Console.WriteLine(result);
        }
        catch (ContentSafetyViolationException ex)
        {
            Console.WriteLine($"Content violation: {ex.Category} (Severity: {ex.Severity})");
            // Handle violation
        }
    }
}
```

### Custom Safety Handler

```csharp
using Microsoft.Agents.AI.Safety;

public class CustomSafetyHandler : ISafetyViolationHandler
{
    public async Task<SafetyResponse> HandleInputViolationAsync(
        ContentSafetyViolation violation,
        CancellationToken cancellationToken = default)
    {
        // Log to security monitoring
        await LogSecurityEventAsync(new SecurityEvent
        {
            Type = "input_violation",
            Category = violation.Category,
            Severity = violation.Severity,
            UserId = violation.Context?.GetValueOrDefault("user_id")?.ToString(),
            Timestamp = DateTime.UtcNow
        });

        return new SafetyResponse
        {
            Blocked = true,
            Message = "Your message violates our content policy.",
            ReferenceId = violation.Id
        };
    }

    public async Task<SafetyResponse> HandleOutputViolationAsync(
        ContentSafetyViolation violation,
        CancellationToken cancellationToken = default)
    {
        await LogSecurityEventAsync(new SecurityEvent
        {
            Type = "output_violation",
            Category = violation.Category,
            AgentId = violation.AgentId
        });

        return new SafetyResponse
        {
            Blocked = true,
            Message = "I apologize, but I cannot provide that information.",
            Fallback = true
        };
    }

    private async Task LogSecurityEventAsync(SecurityEvent evt)
    {
        // Log to security system
        await Task.CompletedTask;
    }
}

public record SecurityEvent
{
    public string Type { get; init; } = default!;
    public string Category { get; init; } = default!;
    public int Severity { get; init; }
    public string? UserId { get; init; }
    public string? AgentId { get; init; }
    public DateTime Timestamp { get; init; }
}
```

### PII Detection

```csharp
using Microsoft.Agents.AI.Safety;

public class PIIProtection
{
    public static ChatAgent CreatePIIProtectedAgent()
    {
        var piiConfig = new PIIDetectionConfig
        {
            Enabled = true,
            DetectCategories = new[]
            {
                "Email",
                "PhoneNumber",
                "SSN",
                "CreditCard",
                "PersonName",
                "Address"
            },
            Action = PIIAction.Redact,
            RedactionPattern = "[REDACTED_{category}]"
        };

        var agent = new ChatAgent(
            instructions: "You handle customer data.",
            piiDetection: piiConfig
        );

        return agent;
    }

    public static async Task TestPIIRedactionAsync()
    {
        var agent = CreatePIIProtectedAgent();

        var response = await agent.RunAsync(
            "My email is john@example.com and SSN is 123-45-6789"
        );

        Console.WriteLine(response);
        // Output: "My email is [REDACTED_Email] and SSN is [REDACTED_SSN]"
    }
}
```

---

## Dependency Injection Patterns

### Registering Agents in DI Container

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Agents.AI;
using Azure.Identity;

public class Program
{
    public static async Task Main(string[] args)
    {
        var host = Host.CreateDefaultBuilder(args)
            .ConfigureServices((context, services) =>
            {
                // Register credential
                services.AddSingleton<DefaultAzureCredential>();

                // Register agents
                services.AddSingleton<ChatAgent>(sp =>
                {
                    return new ChatAgent(
                        instructions: "You are a helpful assistant.",
                        credential: sp.GetRequiredService<DefaultAzureCredential>()
                    );
                });

                // Register agent service
                services.AddScoped<IAgentService, AgentService>();

                // Register hosted service
                services.AddHostedService<AgentWorker>();
            })
            .Build();

        await host.RunAsync();
    }
}

public interface IAgentService
{
    Task<string> ExecuteQueryAsync(string query, CancellationToken cancellationToken = default);
}

public class AgentService : IAgentService
{
    private readonly ChatAgent _agent;
    private readonly ILogger<AgentService> _logger;

    public AgentService(ChatAgent agent, ILogger<AgentService> logger)
    {
        _agent = agent;
        _logger = logger;
    }

    public async Task<string> ExecuteQueryAsync(
        string query,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Executing query: {Query}", query);

        var result = await _agent.RunAsync(query, cancellationToken);

        _logger.LogInformation("Query completed successfully");

        return result;
    }
}

public class AgentWorker : BackgroundService
{
    private readonly IAgentService _agentService;
    private readonly ILogger<AgentWorker> _logger;

    public AgentWorker(IAgentService agentService, ILogger<AgentWorker> logger)
    {
        _agentService = agentService;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var result = await _agentService.ExecuteQueryAsync(
                    "Process next item",
                    stoppingToken
                );

                _logger.LogInformation("Result: {Result}", result);

                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in agent worker");
            }
        }
    }
}
```

---

## Best Practices

### 1. Async All the Way
- Always use `async`/`await` throughout
- Avoid `.Result` or `.Wait()` - deadlock risk
- Use `ConfigureAwait(false)` in library code
- Implement cancellation tokens

### 2. Dependency Injection
- Register agents with appropriate lifetimes
- Use `IOptions<T>` for configuration
- Implement proper disposal patterns
- Leverage DI for testability

### 3. Type Safety
- Use record types for immutable data
- Leverage nullable reference types
- Use attributes for metadata
- Implement validation

### 4. Error Handling
- Use specific exception types
- Implement proper logging
- Design for resilience
- Handle cancellation gracefully

### 5. Performance
- Use `ValueTask` where appropriate
- Implement object pooling for high-throughput
- Configure connection pooling
- Monitor and optimize allocations

---

**Last Updated:** November 2025
**Document Version:** 1.0
**Preview Status:** October 2025 Public Preview

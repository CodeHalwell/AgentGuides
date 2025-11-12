Latest: 1.1.0
# Microsoft Agent Framework .NET - Comprehensive Technical Guide

**Release:** October 2025  
**Framework Version:** 1.0+  
**Target Platform:** .NET 8.0+

---

## Introduction

Welcome to the comprehensive technical guide for the Microsoft Agent Framework for .NET. This guide is designed for .NET developers looking to build, orchestrate, and deploy sophisticated AI agents and multi-agent systems.

### Framework Overview

The Microsoft Agent Framework is an open-source SDK that unifies the capabilities of two powerful predecessors: **Semantic Kernel** and **AutoGen**. It provides a single, cohesive platform for building everything from simple, single-purpose agents to complex, distributed multi-agent workflows.

- **From Semantic Kernel:** It inherits enterprise-grade features like robust tool integration (plugins), memory management, and a rich ecosystem of connectors.
- **From AutoGen:** It adopts advanced multi-agent orchestration, conversational patterns, and group chat coordination.

The framework is designed to be modular, extensible, and production-ready, empowering developers to create scalable, real-world AI applications within the .NET ecosystem.

### Key Objectives

- **Unified Development:** Provide a single SDK for tasks that previously required both Semantic Kernel and AutoGen.
- **Enterprise-Ready:** Offer features like observability, security, and scalability out of the box.
- **Developer-Friendly:** Leverage familiar .NET patterns like dependency injection, `async/await`, and strong typing.
- **Extensible:** Allow for custom agents, tools, and memory systems to fit any use case.
- **Interoperable:** Support open standards like the Model Context Protocol (MCP) to ensure compatibility across different AI ecosystems.

### Target Use Cases

- **Customer Support Automation:** Multi-agent systems that can classify, route, and resolve customer issues.
- **Autonomous Task Execution:** Agents that can plan and execute complex tasks with multiple steps and tools.
- **Data Analysis & RAG:** Agents that can query databases, analyze data, and generate insights using Retrieval-Augmented Generation.
- **Software Development Lifecycle:** Agents that can write, test, and debug code.
- **Content Generation:** Orchestrated workflows for drafting, reviewing, and publishing content.

---

## Core Fundamentals

### Architecture Principles

The framework is built on a layered architecture that promotes separation of concerns:

```
+-----------------------------------+
|      Application Layer            |
| (Your Agents, APIs, Services)     |
+-----------------------------------+
|      Orchestration Layer          |
| (Workflows, Coordination)         |
+-----------------------------------+
|      Agent Abstraction Layer      |
| (AIAgent, ChatAgent, State)       |
+-----------------------------------+
|      Core Components Layer          |
| (Tools, Memory, LLM Providers)    |
+-----------------------------------+
|      Integration Layer            |
| (Azure, OpenAI, LlamaHub, etc.)   |
+-----------------------------------+
```

### Installation

To get started, you need the .NET 8.0 SDK and the core NuGet package.

```bash
# Create a new .NET console application
dotnet new console -n MyAgentApp
cd MyAgentApp

# Add the core Agent Framework package
dotnet add package Microsoft.Agents.AI --prerelease

# Add a specific LLM provider (e.g., Azure OpenAI)
dotnet add package Azure.AI.OpenAI
dotnet add package Azure.Identity

# For full Azure integration features
dotnet add package Microsoft.Agents.AI.Azure --prerelease
```

### Authentication and Configuration

It is a security best practice to manage secrets outside of your source code. The framework integrates seamlessly with .NET's configuration system.

**1. Using `appsettings.json` and User Secrets:**

For local development, use `secrets.json`, which is not checked into source control.

```bash
# Initialize user secrets
dotnet user-secrets init

# Set your Azure OpenAI credentials
dotnet user-secrets set "AzureOpenAI:Endpoint" "https://your-resource.openai.azure.com"
dotnet user-secrets set "AzureOpenAI:ApiKey" "your-api-key"
```

Your `appsettings.json` would reference these settings:

```json
// appsettings.json
{
  "AzureOpenAI": {
    "DeploymentName": "gpt-4o"
  }
}
```

**2. Using `DefaultAzureCredential` (Recommended for Production):**

For production environments, `DefaultAzureCredential` provides a secure, passwordless way to authenticate by using the managed identity of the host service (e.g., Azure App Service, Azure Container Apps).

### Environment Setup & Dependency Injection

The framework is designed to work with .NET's dependency injection (DI) container, making it easy to manage services and configurations.

```csharp
// Program.cs
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Agents.AI;
using Azure.AI.OpenAI;
using Azure.Identity;
using System;

var builder = Host.CreateApplicationBuilder(args);

// --- Register Configuration ---
// This automatically loads from appsettings.json, environment variables, and user secrets
var openAiEndpoint = builder.Configuration["AzureOpenAI:Endpoint"];
var openAiDeployment = builder.Configuration["AzureOpenAI:DeploymentName"];

// --- Register Services with DI ---
builder.Services.AddSingleton(sp =>
{
    // Use DefaultAzureCredential in production, or a specific credential for dev
    var credential = new DefaultAzureCredential(); 
    return new OpenAIClient(new Uri(openAiEndpoint), credential);
});

// Register the core AgentFactory
builder.Services.AddAgentFactory(b =>
{
    b.UseAzureOpenAI(openAiDeployment);
});

// Register your custom agent
builder.Services.AddTransient<MySimpleAgent>();

var host = builder.Build();

// --- Run your agent ---
var myAgent = host.Services.GetRequiredService<MySimpleAgent>();
var response = await myAgent.RunAsync("Tell me a joke about .NET.");
Console.WriteLine(response);


// Your custom agent class
public class MySimpleAgent
{
    private readonly AgentFactory _agentFactory;

    public MySimpleAgent(AgentFactory agentFactory)
    {
        _agentFactory = agentFactory;
    }

    public async Task<string> RunAsync(string input)
    {
        var agent = _agentFactory.CreateAgent<ChatAgent>();
        var thread = agent.CreateThread();
        var response = await thread.InvokeAsync(input);
        return response.GetContent<string>();
    }
}
```

This setup provides a clean and maintainable structure for building complex agentic applications.

---

## Simple Agents

This section covers the creation and management of individual agents.

### Creating Basic Agents: `AIAgent` vs. `ChatAgent`

The framework provides two primary agent types:

-   **`AIAgent`**: A stateless agent ideal for single-turn, request/response interactions. It does not maintain conversation history automatically.
-   **`ChatAgent`**: A stateful agent that automatically manages conversation history within a `Thread`. It's the most common choice for building conversational experiences.

**Example: Creating a `ChatAgent`**

```csharp
using Microsoft.Agents.AI;
using System.Threading.Tasks;

public class ConversationalAgent
{
    private readonly ChatAgent _agent;

    // AgentFactory is injected via DI
    public ConversationalAgent(AgentFactory factory)
    {
        // Define the agent's role or system prompt
        var options = new AgentOptions
        {
            Instructions = "You are a helpful AI assistant for .NET developers."
        };
        _agent = factory.CreateAgent<ChatAgent>(options);
    }

    public async Task StartConversation()
    {
        // Each conversation is a separate thread
        var thread = _agent.CreateThread();

        var response1 = await thread.InvokeAsync("What is the latest version of .NET?");
        Console.WriteLine($"Assistant: {response1.GetContent<string>()}");

        // The agent remembers the previous turn
        var response2 = await thread.InvokeAsync("What were the key features in that release?");
        Console.WriteLine($"Assistant: {response2.GetContent<string>()}");
    }
}
```

### Agent Lifecycle Management

Agents are typically registered as transient or scoped services in a DI container. The `AgentFactory` manages the underlying resources, ensuring that components like `OpenAIClient` are reused efficiently.

### Task Execution Models

Agents execute tasks asynchronously. The `InvokeAsync` method is the primary way to interact with an agent. It takes an input, processes it through the LLM (along with any history and tools), and returns the resulting `AgentResponse`.

### Testing Individual Agents

Testing is crucial. You can mock the `AgentFactory` and `ChatAgent` to test your application logic without making live LLM calls.

```csharp
// In your test project
using Moq;
using Microsoft.Agents.AI;
using System.Threading.Tasks;
using Xunit;

public class AgentTests
{
    [Fact]
    public async Task MyAgent_Should_Respond_Correctly()
    {
        // Arrange
        var mockResponse = new AgentResponse(new { Content = "This is a mock response." });
        
        var mockThread = new Mock<AgentThread>();
        mockThread.Setup(t => t.InvokeAsync(It.IsAny<object>(), default))
                  .ReturnsAsync(mockResponse);

        var mockAgent = new Mock<ChatAgent>();
        mockAgent.Setup(a => a.CreateThread()).Returns(mockThread.Object);

        var mockFactory = new Mock<AgentFactory>();
        mockFactory.Setup(f => f.CreateAgent<ChatAgent>(It.IsAny<AgentOptions>()))
                   .Returns(mockAgent.Object);

        var myConversationalAgent = new ConversationalAgent(mockFactory.Object);

        // Act
        // Since we are mocking, the actual conversation logic is bypassed
        // In a real test, you would call your method that uses the agent
        // For demonstration, we assume the test directly verifies the mock setup.
        var thread = myConversationalAgent.CreateThread(); // This would be inside your method
        var result = await thread.InvokeAsync("test");


        // Assert
        Assert.Equal("This is a mock response.", result.GetContent<string>());
        mockThread.Verify(t => t.InvokeAsync("test", default), Times.Once);
    }
}
```

---

## Multi-Agent Systems

The true power of the framework lies in orchestrating multiple agents to solve complex problems.

### Orchestration Patterns

The framework supports several patterns for agent collaboration:

-   **Sequential Workflow:** Agent A completes a task and passes its output to Agent B.
-   **Parallel Workflow:** Multiple agents work on different sub-tasks simultaneously.
-   **Group Chat / Broadcast:** A user query is sent to a group of agents, and a primary agent or a voting mechanism selects the best response.

### Communication Patterns Between Agents

Agents communicate through a shared `AgentThread`. One agent can add a message to the thread, and another agent can then be invoked with that thread's state, allowing it to see the full history.

**Example: A Simple Sequential Workflow**

```csharp
public class ResearchWorkflow
{
    private readonly ChatAgent _researcher;
    private readonly ChatAgent _writer;

    public ResearchWorkflow(AgentFactory factory)
    {
        _researcher = factory.CreateAgent<ChatAgent>(new() { 
            Instructions = "You are a research assistant. Find information on a topic." 
        });
        _writer = factory.CreateAgent<ChatAgent>(new() { 
            Instructions = "You are a content writer. Write a summary based on research." 
        });
    }

    public async Task<string> RunAsync(string topic)
    {
        var thread = _researcher.CreateThread();

        // 1. Researcher agent finds information
        var researchResult = await thread.InvokeAsync($"Research the topic: {topic}");
        Console.WriteLine($"Researcher found: {researchResult.GetContent<string>().Substring(0, 100)}...");

        // 2. Writer agent is invoked with the same thread, so it sees the research
        var finalReport = await _writer.InvokeAsync(thread, "Write a one-paragraph summary of your findings.");
        
        return finalReport.GetContent<string>();
    }
}
```

### Shared State Management

For more complex scenarios, agents might need to share more than just conversation history. A shared state object can be passed between agents. This object can be a simple dictionary or a strongly-typed C# class, managed by an orchestrator.

### Workflow Coordination with `Workflows`

For explicit control over multi-agent execution, the framework provides a `Workflow` engine.

```csharp
using Microsoft.Agents.AI.Orchestration;

public class ExplicitWorkflow
{
    private readonly AgentFactory _factory;

    public ExplicitWorkflow(AgentFactory factory)
    {
        _factory = factory;
    }

    public async Task<string> RunAsync(string topic)
    {
        // Define agents as workflow steps
        var researcher = _factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a researcher." });
        var writer = _factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a writer." });

        // Create a workflow
        var workflow = new Workflow();

        // Define the steps
        var researchStep = workflow.AddAgent(researcher, "ResearchStep");
        var writeStep = workflow.AddAgent(writer, "WriteStep");

        // Define the flow
        workflow.AddEdge(researchStep, writeStep, (thread) => 
            !string.IsNullOrEmpty(thread.GetLastMessage()?.GetContent<string>())
        );

        // Execute the workflow
        var initialMessage = new { Topic = topic };
        var result = await workflow.ExecuteAsync(initialMessage);

        return result.GetLastMessage().GetContent<string>();
    }
}
```

---

## Tools Integration

Tools (formerly "skills" or "plugins" in Semantic Kernel) are functions that agents can invoke to interact with the outside world.

### Tool Definition with `AIFunctionFactory`

The primary way to define tools is by creating plain C# methods and using `AIFunctionFactory` to generate the necessary schema for the LLM.

```csharp
using Microsoft.Agents.AI.Tools;
using System.ComponentModel;
using System.Threading.Tasks;

public class WeatherTools
{
    [Description("Get the current weather for a specified city.")]
    public async Task<string> GetCurrentWeatherAsync(
        [Description("The city name, e.g., 'Seattle, WA'")] string city)
    {
        // In a real app, call a weather API
        if (city.Contains("Seattle"))
        {
            return "The weather in Seattle is rainy, 55°F.";
        }
        return $"Weather for {city} is sunny, 72°F.";
    }
}

// --- In your agent setup ---
var weatherTools = new WeatherTools();
var tool = AIFunctionFactory.CreateTool(weatherTools);

var agentOptions = new AgentOptions
{
    Tools = { tool }
};

var agent = factory.CreateAgent<ChatAgent>(agentOptions);
var thread = agent.CreateThread();

var response = await thread.InvokeAsync("What's the weather like in Seattle today?");
// The agent will automatically call the GetCurrentWeatherAsync tool
Console.WriteLine(response.GetContent<string>());
```

### Built-in Azure Tools

The `Microsoft.Agents.AI.Azure` package provides pre-built tools for interacting with Azure services, such as Azure AI Search for RAG.

### Custom Tool Creation

You can create complex custom tools by simply defining more methods. The framework handles the schema generation and invocation logic.

### Error Handling in Tools

If a tool throws an exception, the framework catches it and passes the error message back to the agent. The agent can then reason about the error and decide whether to retry the tool with different parameters or inform the user.

```csharp
public class RobustTools
{
    [Description("Performs a division operation.")]
    public double Divide(double numerator, double denominator)
    {
        if (denominator == 0)
        {
            throw new ArgumentException("Cannot divide by zero.");
        }
        return numerator / denominator;
    }
}

// When the agent calls Divide(10, 0), the exception will be caught.
// The agent's next step will be based on the error message "Cannot divide by zero."
```

---

## Structured Output

The framework supports generating strongly-typed, structured responses from an LLM instead of just plain text. This is incredibly useful for data extraction and ensuring predictable outputs.

### Record Types for Schema Definition

Using C# `record` types is the recommended way to define the desired output schema. Records are immutable and provide a concise syntax for defining data structures.

```csharp
using System.Collections.Generic;
using System.ComponentModel;

// Define the desired output structure as a C# record
public record Person(
    [property: Description("The full name of the person.")] string Name,
    [property: Description("The person's age.")] int Age,
    [property: Description("A list of the person's skills.")] List<string> Skills
);
```

### Forcing Structured Output from an Agent

You can instruct an agent to respond with a specific structure by passing the type to the `InvokeAsync` call.

```csharp
public class DataExtractionAgent
{
    private readonly ChatAgent _agent;

    public DataExtractionAgent(AgentFactory factory)
    {
        _agent = factory.CreateAgent<ChatAgent>(new() {
            Instructions = "You are an expert at extracting structured data from text."
        });
    }

    public async Task<Person> ExtractPersonAsync(string text)
    {
        var thread = _agent.CreateThread();
        
        // The agent will format its response to match the Person record schema
        var response = await thread.InvokeAsync<Person>(text);
        
        return response.GetContent();
    }
}

// --- Usage ---
var extractor = new DataExtractionAgent(host.Services.GetRequiredService<AgentFactory>());
var text = "John Doe is a 42-year-old expert in C# and .NET who also knows Python.";
var person = await extractor.ExtractPersonAsync(text);

Console.WriteLine($"Name: {person.Name}"); // Output: John Doe
Console.WriteLine($"Age: {person.Age}");   // Output: 42
Console.WriteLine($"Skills: {string.Join(", ", person.Skills)}"); // Output: C#, .NET, Python
```

### Schema Validation and Error Handling

If the LLM fails to produce a response that matches the requested schema, the framework will throw an exception, which you can handle gracefully. This ensures that you always receive data in the expected format.

---

## Model Context Protocol (MCP)

MCP is an open standard that the Microsoft Agent Framework supports to promote interoperability between different AI agent ecosystems. It defines a standard way for agents to discover and invoke tools.

### Implementing an MCP Server

You can expose your agent's tools over an MCP endpoint, allowing other MCP-compliant agents (even those built with different frameworks) to use them.

```csharp
// In an ASP.NET Core application (Program.cs)
using Microsoft.Agents.AI.Hosting;

var builder = WebApplication.CreateBuilder(args);

// Add Agent Framework services
builder.Services.AddAgentFactory(b => b.UseAzureOpenAI(...));

// Define your tools
builder.Services.AddSingleton<MyCustomTools>();

var app = builder.Build();

// Map the MCP endpoint
app.MapAgents(options =>
{
    // Expose tools from the MyCustomTools class
    options.AddTool<MyCustomTools>();
});

app.Run();
```

Now, the tools within `MyCustomTools` are discoverable and callable via a standard RESTful API at the `/agents` endpoint.

---

## Agentic Patterns

Beyond simple request/response, the framework enables advanced agentic patterns for autonomous problem-solving.

### Planning and Reasoning

By providing a high-level goal, you can have an agent create and execute a multi-step plan.

```csharp
var plannerAgent = factory.CreateAgent<ChatAgent>(new() {
    Instructions = "You are a master planner. Break down complex goals into simple, executable steps."
});

var goal = "Plan a 3-day trip to Seattle, including finding flights, booking a hotel, and suggesting an itinerary.";
var thread = plannerAgent.CreateThread();

// The agent will first generate a plan
var planResponse = await thread.InvokeAsync(goal);
Console.WriteLine($"The Plan:\n{planResponse.GetContent<string>()}");

// You can then have another agent (or the same one) execute the plan step-by-step
var executorAgent = factory.CreateAgent<ChatAgent>(new() { Tools = { ... } });
var executionResponse = await executorAgent.InvokeAsync(thread, "Execute the first step of the plan.");
```

### Autonomous Decision-Making

In a multi-agent system, you can create a "supervisor" or "router" agent that delegates tasks to other agents based on the nature of the request.

```csharp
// (See the Multi-Agent Systems section for a code example of a router)
```

### Reflection Patterns

An agent can be prompted to review and critique its own work, leading to higher-quality outputs.

```csharp
var writerAgent = factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a blog post writer." });
var criticAgent = factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a critical editor. Review the following text and provide feedback for improvement." });

var topic = "The Future of AI in .NET";
var thread = writerAgent.CreateThread();

// 1. Draft the post
var draft = await thread.InvokeAsync($"Write a blog post about: {topic}");

// 2. Have the critic review the draft
var feedback = await criticAgent.InvokeAsync(thread, $"Please review this draft:\n\n{draft.GetContent<string>()}");

// 3. Have the writer revise the draft based on feedback
var finalVersion = await writerAgent.InvokeAsync(thread, $"Revise the draft based on this feedback:\n\n{feedback.GetContent<string>()}");

Console.WriteLine($"Final Version:\n{finalVersion.GetContent<string>()}");
```

---

## Memory Systems

Memory allows agents to persist information within and across conversations, enabling more personalized and context-aware interactions.

### Memory Backends

The framework supports various backends for storing agent memory:

-   **In-Memory:** Default, for development and testing. State is lost when the application restarts.
-   **Azure AI Search:** For scalable, production-grade vector memory.
-   **SQL Server / PostgreSQL:** For structured, relational memory storage.
-   **Azure Cosmos DB:** For globally distributed, multi-model database storage.

### Persistent State and Conversation History

When you use a persistent backend, conversation history (`AgentThread` state) is automatically saved. This means you can resume a conversation with a user even after the application has been restarted.

**Example: Configuring Cosmos DB for State Persistence**

```csharp
// Program.cs
builder.Services.AddAgentFactory(b =>
{
    b.UseAzureOpenAI(deploymentName);
    
    // Configure Cosmos DB as the state backend
    b.AddCosmosDBState(options =>
    {
        options.ConnectionString = builder.Configuration["CosmosDB:ConnectionString"];
        options.DatabaseName = "AgentState";
        options.ContainerName = "Threads";
    });
});
```

With this configuration, all agent threads will be automatically persisted to Cosmos DB.

### Vector Memory for RAG

For Retrieval-Augmented Generation (RAG), you need a vector store to hold your knowledge base.

**Example: Using Azure AI Search for RAG**

```csharp
// 1. Configure Azure AI Search as the memory backend
builder.Services.AddAgentFactory(b =>
{
    b.UseAzureOpenAI(deploymentName);
    b.AddAzureAISearchMemory(options =>
    {
        options.Endpoint = new Uri(builder.Configuration["AzureAISearch:Endpoint"]);
        options.Credential = new DefaultAzureCredential();
    });
});

// 2. In your agent, ingest data into memory
public class KnowledgeAgent
{
    private readonly IAgentMemory _memory;

    public KnowledgeAgent(IAgentMemory memory)
    {
        _memory = memory;
    }

    public async Task IngestDocumentAsync(string documentContent, string documentId)
    {
        // The framework handles chunking and embedding automatically
        await _memory.UpsertAsync(documentId, documentContent);
    }
}

// 3. The agent can now retrieve this information during conversations
var ragAgent = factory.CreateAgent<ChatAgent>(new() {
    Instructions = "Answer questions based on the knowledge provided to you."
});

// When you invoke ragAgent, it will automatically search the configured
// Azure AI Search index to augment its response.
```

---

## Context Engineering

Context engineering is the practice of carefully managing the information provided to the LLM to elicit the best possible response. This includes managing conversation history, providing relevant data for RAG, and isolating context in multi-tenant scenarios.

### Context Propagation and `AsyncLocal`

The framework uses `AsyncLocal<T>` to implicitly carry context, like the `AgentThread`, through the call stack. This is a powerful pattern that simplifies method signatures, as you don't need to pass the thread object around manually.

### Multi-Tenant Isolation

In a multi-tenant application, it is critical to ensure that one user's data and conversation history are never exposed to another. The framework's `AgentThread` model provides a natural boundary for this isolation.

**Best Practice:** Always key your threads to a unique, authenticated user ID.

```csharp
public class MultiTenantService
{
    private readonly AgentFactory _agentFactory;

    public MultiTenantService(AgentFactory agentFactory)
    {
        _agentFactory = agentFactory;
    }

    public async Task<string> ProcessUserRequestAsync(string userId, string userInput)
    {
        var agent = _agentFactory.CreateAgent<ChatAgent>();

        // Each user gets their own thread, ensuring conversation isolation.
        // The thread ID is a combination of agent ID and a unique user identifier.
        var threadId = $"{agent.Id}:{userId}"; 
        var thread = agent.GetThread(threadId);

        var response = await thread.InvokeAsync(userInput);
        return response.GetContent<string>();
    }
}
```
When using a persistent memory backend (like Cosmos DB or SQL Server), the framework automatically partitions the data based on the thread ID, providing a strong security boundary between tenants.

---

## Copilot Studio Integration

Microsoft Copilot Studio provides a low-code interface for building and managing copilots. The Agent Framework allows you to publish your .NET-built agents as skills that can be consumed directly within Copilot Studio.

### Publishing an Agent to Copilot Studio

1.  **Expose Tools via MCP:** First, expose your agent's tools via an MCP endpoint as described in the MCP section. This makes your agent's capabilities discoverable.
2.  **Register in Copilot Studio:** In the Copilot Studio portal, you can add a new "Agent Skill" and point it to your MCP endpoint's manifest URL (`/agents/manifest.json`).
3.  **Invoke from Topics:** Once registered, your agent's tools will appear as nodes that you can call from any Copilot Studio topic, allowing you to blend low-code conversation design with powerful, custom .NET backend logic.

---

## Azure AI Integration

The framework is designed for deep integration with the Azure AI ecosystem.

### Leveraging Azure AI Services

-   **Azure OpenAI:** The primary service for accessing GPT models.
-   **Azure AI Search:** The recommended backend for production-grade RAG and vector memory.
-   **Azure AI Content Safety:** Can be integrated as a middleware step to moderate inputs and outputs, ensuring responsible AI practices.
-   **Azure AI Speech:** For building voice-enabled agents.

### Cost Optimization and Governance in Azure

-   **Azure Budgets:** Set spending limits on your Azure resources.
-   **Azure Policy:** Enforce rules on your AI resources, such as restricting which regions they can be deployed in or which VM sizes can be used.
-   **Monitoring Token Usage:** Use Application Insights to log token usage for each agent interaction, allowing you to identify high-cost operations and optimize your prompts or workflows.

---

## Semantic Kernel Integration

As the successor to Semantic Kernel (SK), the Agent Framework provides a clear migration path and interoperability.

### Plugin Compatibility

The concept of "plugins" in Semantic Kernel is equivalent to "tools" in the Agent Framework. You can easily wrap existing SK plugins for use in the new framework.

**Example: Using a Semantic Kernel Plugin**

```csharp
using Microsoft.SemanticKernel;
using Microsoft.Agents.AI.Tools;
using System.ComponentModel;

// Assume you have an existing Semantic Kernel plugin
public class MySKPlugin
{
    [KernelFunction, Description("A function from an old SK plugin.")]
    public string OldPluginFunction(string input)
    {
        return $"Processed by SK plugin: {input}";
    }
}

// You can expose it as a tool in the new framework
public class SKWrapperTool
{
    private readonly MySKPlugin _plugin = new();

    [Description("Wrapper for the old SK plugin function.")]
    public string UseOldPlugin(string data)
    {
        return _plugin.OldPluginFunction(data);
    }
}

// Then, register SKWrapperTool with your agent:
// options.AddTool<SKWrapperTool>();
```

### Migration Strategies

-   **Gradual Adoption:** You don't need to rewrite your entire application at once. You can start by wrapping your existing Semantic Kernel plugins as tools and using them within a new Agent Framework orchestrator.
-   **State Management:** Migrate your conversation history and memory from SK's storage solutions to the Agent Framework's `IAgentMemory` providers to take advantage of the unified memory system.
-   **Orchestration:** Replace SK's `Planner` with the Agent Framework's more explicit multi-agent orchestration patterns or `Workflow` engine for more predictable and controllable execution.

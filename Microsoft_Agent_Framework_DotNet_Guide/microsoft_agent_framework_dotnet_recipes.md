# Microsoft Agent Framework .NET - Recipes and Code Patterns

This document provides a collection of practical, copy-paste-ready C# recipes for building common agentic patterns with the Microsoft Agent Framework for .NET.

**Target Platform:** .NET 8.0+  
**Framework Version:** 1.0+

---

## Beginner Recipes

These recipes are for developers new to the framework and cover fundamental concepts.

### Recipe 1: Simple Chat Agent (C#)

This is the "Hello, World!" of the Agent Framework—a basic conversational agent that maintains history.

```csharp
// SimpleChatAgent.cs
using Microsoft.Agents.AI;
using System;
using System.Threading.Tasks;

public class SimpleChatAgent
{
    private readonly ChatAgent _agent;

    public SimpleChatAgent(AgentFactory agentFactory)
    {
        var options = new AgentOptions
        {
            Instructions = "You are a friendly AI assistant. Keep your responses concise."
        };
        _agent = agentFactory.CreateAgent<ChatAgent>(options);
    }

    public async Task RunInteractiveAsync()
    {
        Console.WriteLine("Chat Agent Initialized. Type 'exit' to quit.");
        var thread = _agent.CreateThread();

        while (true)
        {
            Console.Write("You: ");
            var userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput) || userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            var response = await thread.InvokeAsync(userInput);
            Console.WriteLine($"Assistant: {response.GetContent<string>()}");
        }
    }
}

// --- To run this (in Program.cs) ---
// var chatAgent = host.Services.GetRequiredService<SimpleChatAgent>();
// await chatAgent.RunInteractiveAsync();
```

### Recipe 2: Agent with a Single Tool

This recipe shows how to add a simple tool to an agent, allowing it to perform an action.

```csharp
// AgentWithTool.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Tools;
using System.ComponentModel;
using System.Threading.Tasks;

// 1. Define the tool
public class TimeTool
{
    [Description("Gets the current UTC date and time.")]
    public string GetCurrentTime()
    {
        return DateTime.UtcNow.ToString("F");
    }
}

// 2. Create the agent and add the tool
public class ToolAgent
{
    private readonly ChatAgent _agent;

    public ToolAgent(AgentFactory agentFactory)
    {
        var timeTool = new TimeTool();
        var tool = AIFunctionFactory.CreateTool(timeTool);

        var options = new AgentOptions
        {
            Instructions = "You have a tool to get the current time. Use it when asked about the time.",
            Tools = { tool }
        };
        _agent = agentFactory.CreateAgent<ChatAgent>(options);
    }

    public async Task<string> AskAboutTimeAsync()
    {
        var thread = _agent.CreateThread();
        var response = await thread.InvokeAsync("What time is it right now?");
        return response.GetContent<string>();
    }
}
```

### Recipe 3: Basic Error Handling in Tools

This recipe demonstrates how to handle errors within a tool gracefully.

```csharp
// ErrorHandlingTool.cs
using System.ComponentModel;

public class MathTool
{
    [Description("Divides two numbers.")]
    public string Divide(double numerator, double denominator)
    {
        if (denominator == 0)
        {
            // The framework will pass this exception message to the LLM,
            // allowing it to understand what went wrong.
            throw new ArgumentException("Cannot divide by zero. Please ask the user for a non-zero denominator.");
        }
        return (numerator / denominator).ToString();
    }
}

// --- Agent Setup ---
// var mathTool = new MathTool();
// var agentOptions = new AgentOptions { Tools = { AIFunctionFactory.CreateTool(mathTool) } };
// var agent = factory.CreateAgent<ChatAgent>(agentOptions);
// var response = await agent.CreateThread().InvokeAsync("What is 10 divided by 0?");
// The agent will respond with something like: "I cannot divide by zero. Could you please provide a different number?"
```

---

## Intermediate Recipes

These recipes cover multi-agent systems, memory, and more complex tool interactions.

### Recipe 4: Simple Multi-Agent Workflow (Sequential)

This recipe shows how to chain two agents together: a researcher and a summarizer.

```csharp
// SequentialWorkflow.cs
public class SequentialWorkflow
{
    private readonly ChatAgent _researcher;
    private readonly ChatAgent _summarizer;

    public SequentialWorkflow(AgentFactory factory)
    {
        _researcher = factory.CreateAgent<ChatAgent>(new() {
            Instructions = "You are a world-class researcher. Find detailed information on the given topic."
        });
        _summarizer = factory.CreateAgent<ChatAgent>(new() {
            Instructions = "You are a skilled editor. Summarize the provided text into a single, concise paragraph."
        });
    }

    public async Task<string> RunAsync(string topic)
    {
        // Use a single thread to pass context between agents
        var thread = _researcher.CreateThread();

        // Step 1: Researcher gathers information
        var researchResponse = await thread.InvokeAsync($"Please research the topic: {topic}");

        // Step 2: Summarizer is invoked on the same thread. It sees the researcher's output.
        var summaryResponse = await _summarizer.InvokeAsync(thread, "Please summarize your findings.");

        return summaryResponse.GetContent<string>();
    }
}
```

### Recipe 5: Agent with Multiple Tools and a Router

This recipe demonstrates a router agent that decides which specialist agent should handle a request.

```csharp
// RouterAgent.cs
public class RouterWorkflow
{
    private readonly ChatAgent _routerAgent;
    private readonly ChatAgent _billingAgent;
    private readonly ChatAgent _techSupportAgent;

    public RouterWorkflow(AgentFactory factory)
    {
        _routerAgent = factory.CreateAgent<ChatAgent>(new() {
            Instructions = "You are a request router. Your job is to determine if a user's query is about 'Billing' or 'Technical Support'. Respond with only one of those two words."
        });
        _billingAgent = factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a billing support specialist." });
        _techSupportAgent = factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a technical support specialist." });
    }

    public async Task<string> RouteAndHandleRequestAsync(string userInput)
    {
        var routingResponse = await _routerAgent.CreateThread().InvokeAsync(userInput);
        var route = routingResponse.GetContent<string>();

        var conversationThread = _billingAgent.CreateThread(); // Thread can be from any agent
        await conversationThread.AddMessageAsync(new { Role = "user", Content = userInput });

        if (route.Contains("Billing", StringComparison.OrdinalIgnoreCase))
        {
            var response = await _billingAgent.InvokeAsync(conversationThread, "Provide a response to the user's billing question.");
            return response.GetContent<string>();
        }
        else // Default to technical support
        {
            var response = await _techSupportAgent.InvokeAsync(conversationThread, "Provide a response to the user's technical question.");
            return response.GetContent<string>();
        }
    }
}
```

### Recipe 6: Memory Integration with Azure AI Search

This recipe shows how to configure an agent to use Azure AI Search for long-term memory and RAG.

```csharp
// In Program.cs
builder.Services.AddAgentFactory(b =>
{
    b.UseAzureOpenAI(deploymentName);
    
    // Configure Azure AI Search as the memory backend
    b.AddAzureAISearchMemory(options =>
    {
        options.Endpoint = new Uri(builder.Configuration["AzureAISearch:Endpoint"]);
        options.Credential = new DefaultAzureCredential();
    });
});

// Agent that can ingest and retrieve knowledge
public class KnowledgeWorker
{
    private readonly IAgentMemory _memory;
    private readonly ChatAgent _ragAgent;

    public KnowledgeWorker(IAgentMemory memory, AgentFactory factory)
    {
        _memory = memory;
        _ragAgent = factory.CreateAgent<ChatAgent>(new() {
            Instructions = "You answer questions based on the knowledge you have stored."
        });
    }

    public async Task IngestKnowledgeAsync(string documentId, string content)
    {
        await _memory.UpsertAsync(documentId, content);
        Console.WriteLine($"Ingested document: {documentId}");
    }

    public async Task<string> AnswerQuestionAsync(string question)
    {
        var thread = _ragAgent.CreateThread();
        var response = await thread.InvokeAsync(question);
        return response.GetContent<string>();
    }
}
```

---

## Advanced Recipes

These recipes cover more complex scenarios like RAG and custom tool development.

### Recipe 7: RAG (Retrieval-Augmented Generation) Agent

This builds on the memory recipe to create a full RAG agent that can ingest and query documents.

```csharp
// RagAgent.cs - (See Recipe 6 for DI setup)

public class RagSystem
{
    private readonly KnowledgeWorker _knowledge;

    public RagSystem(KnowledgeWorker knowledge)
    {
        _knowledge = knowledge;
    }

    public async Task RunExampleAsync()
    {
        // 1. Ingest documents into memory
        await _knowledge.IngestKnowledgeAsync("doc-001", "The Microsoft Agent Framework unifies Semantic Kernel and AutoGen.");
        await _knowledge.IngestKnowledgeAsync("doc-002", "The framework is available for .NET and Python.");

        // 2. Ask a question that requires retrieval
        var answer = await _knowledge.AnswerQuestionAsync("What two frameworks were unified?");
        
        Console.WriteLine($"Question: What two frameworks were unified?");
        Console.WriteLine($"Answer: {answer}");
        // Expected output will mention Semantic Kernel and AutoGen.
    }
}
```

### Recipe 8: Complex Multi-Agent Orchestration with `Workflow`

This recipe uses the `Workflow` engine for a more robust and explicit multi-agent collaboration.

```csharp
// WorkflowOrchestration.cs
using Microsoft.Agents.AI.Orchestration;

public class AdvancedWorkflow
{
    private readonly AgentFactory _factory;

    public AdvancedWorkflow(AgentFactory factory)
    {
        _factory = factory;
    }

    public async Task<string> RunHandoffWorkflowAsync(string topic)
    {
        // 1. Define agents
        var researcher = _factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a researcher." });
        var analyst = _factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a data analyst." });
        var writer = _factory.CreateAgent<ChatAgent>(new() { Instructions = "You are a technical writer." });

        // 2. Create a workflow
        var workflow = new Workflow("ResearchPaperWorkflow");

        // 3. Add agents as steps
        var researchStep = workflow.AddAgent(researcher);
        var analysisStep = workflow.AddAgent(analyst);
        var writeStep = workflow.AddAgent(writer);

        // 4. Define the execution flow
        workflow.AddEdge(researchStep, analysisStep); // Researcher hands off to Analyst
        workflow.AddEdge(analysisStep, writeStep);   // Analyst hands off to Writer

        // 5. Execute the workflow
        var initialContext = new { Topic = topic, Instructions = "Find three key points on the topic." };
        var result = await workflow.ExecuteAsync(initialContext);

        return result.GetLastMessage().GetContent<string>();
    }
}
```

---

## Integration Recipes

These recipes show how to integrate the Agent Framework with other .NET and Azure technologies.

### Recipe 9: Exposing an Agent via ASP.NET Core API

This recipe shows how to create a web API endpoint to interact with an agent.

```csharp
// In an ASP.NET Core project (e.g., Program.cs for minimal APIs)

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddAgentFactory(b => b.UseAzureOpenAI(...));
builder.Services.AddScoped<SimpleChatAgent>(); // Register our agent

var app = builder.Build();

// Define the API endpoint
app.MapPost("/chat", async (ChatRequest request, SimpleChatAgent agent) =>
{
    var response = await agent.RespondToAsync(request.UserId, request.Message);
    return Results.Ok(new { Reply = response });
});

app.Run();

public record ChatRequest(string UserId, string Message);

// Modify the SimpleChatAgent to support specific threads
public class SimpleChatAgent 
{
    // ... constructor ...
    public async Task<string> RespondToAsync(string userId, string message)
    {
        var thread = _agent.GetThread($"user-{userId}");
        var response = await thread.InvokeAsync(message);
        return response.GetContent<string>();
    }
}
```

### Recipe 10: Event-Driven Agents with Azure Functions

This recipe demonstrates how to trigger an agent workflow from an Azure Queue Storage message.

```csharp
// AgentTriggerFunction.cs
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using System.Threading.Tasks;

public class AgentTriggerFunction
{
    private readonly AdvancedWorkflow _workflow;
    private readonly ILogger<AgentTriggerFunction> _logger;

    public AgentTriggerFunction(AdvancedWorkflow workflow, ILogger<AgentTriggerFunction> logger)
    {
        _workflow = workflow;
        _logger = logger;
    }

    [Function("ProcessQueueItem")]
    public async Task Run([QueueTrigger("agent-tasks", Connection = "AzureWebJobsStorage")] string myQueueItem)
    {
        _logger.LogInformation($"C# Queue trigger function processed: {myQueueItem}");

        // myQueueItem could be a JSON payload with the topic
        var result = await _workflow.RunHandoffWorkflowAsync(myQueueItem);

        _logger.LogInformation($"Workflow completed. Result: {result}");
        // Could write the result to another queue, a database, etc.
    }
}

// --- Startup.cs for Azure Functions ---
// public class Startup : FunctionsStartup
// {
//     public override void Configure(IFunctionsHostBuilder builder)
//     {
//         builder.Services.AddAgentFactory(...);
//         builder.Services.AddSingleton<AdvancedWorkflow>();
//     }
// }
```

---

## Troubleshooting Patterns

These patterns help with debugging and monitoring your agents.

### Pattern 11: Debugging Agent Execution with Streaming

Stream the agent's thoughts and tool calls in real-time to understand its decision-making process.

```csharp
public async Task DebugWithStreamingAsync(string userInput)
{
    var agent = new ToolAgent(factory).GetAgent(); // Assuming ToolAgent exposes its agent
    var thread = agent.CreateThread();

    await foreach (var message in thread.StreamAsync(userInput))
    {
        if (message.Role == "tool")
        {
            // This is a tool call
            var toolCall = message.GetContent<ToolCall>();
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine($"[Tool Call: {toolCall.ToolName}]");
            Console.ResetColor();
        }
        else if (message.Role == "assistant")
        {
            // This is the final response or an intermediate thought
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Assistant]: {message.GetContent<string>()}");
            Console.ResetColor();
        }
    }
}
```

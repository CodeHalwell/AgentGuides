---
title: "Microsoft Agent Framework (.NET) — Recipes"
description: "Practical, compiled C# recipes for the Microsoft Agent Framework for .NET: chat, tools, routing, RAG, a writer-critic loop with a round cap, a parallel review panel inside a pipeline, a support handoff with approval-gated refunds, bounded agent calls with timeouts and retries, map-reduce summarisation, ASP.NET Core and Azure Functions."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Recipes"
  order: 15
---

A collection of practical C# recipes for common agent patterns with the Microsoft Agent Framework for .NET.

**Target platform:** .NET 8.0+
**Framework version:** `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%). Every C# block on this page was compiled against that release, and all but the interactive chat loop, the Azure AI Search recipes and the Functions trigger were run against a local stand-in for the Azure OpenAI API (so the replies they print are placeholders, not model output).

Each recipe combines features that are explained in depth elsewhere: [workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/), [orchestrations](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/), [hosting](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/), [observability](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/) and [testing](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/).

Packages used across the recipes:

```bash
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.OpenAI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.Workflows --version %%version=Microsoft.Agents.AI%%
dotnet add package Azure.AI.OpenAI --prerelease
dotnet add package Azure.Identity
```

There is no `AgentFactory`, `AddAgentFactory`, `UseAzureOpenAI`, `AgentOptions` or `IAgentMemory` in the framework. An agent is built from a chat client — `AzureOpenAIClient(...).GetChatClient(deployment).AsAIAgent(...)`, or `new ChatClientAgent(IChatClient, ...)` — and you register it in DI yourself (or with `builder.AddAIAgent(...)` from the hosting package, shown in Recipe 9). The recipes share this small helper, which uses Entra ID unless `AZURE_OPENAI_API_KEY` is set:

```csharp
// Agents.cs
using Azure;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Chat;

public static class Agents
{
    // Entra ID (managed identity, az login) by default; an API key if AZURE_OPENAI_API_KEY is set.
    private static readonly AzureOpenAIClient Client =
        Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY") is { Length: > 0 } key
            ? new(new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!), new AzureKeyCredential(key))
            : new(new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!), new DefaultAzureCredential());

    private static string Deployment =>
        Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai-small%%";

    /// The IChatClient abstraction, for DI registration and ChatClientAgent.
    public static IChatClient ChatClient() => Client.GetChatClient(Deployment).AsIChatClient();

    public static ChatClientAgent Create(string instructions, string? name = null, IList<AITool>? tools = null) =>
        Client.GetChatClient(Deployment).AsAIAgent(instructions: instructions, name: name, tools: tools);
}
```

---

## Beginner Recipes

### Recipe 1: Simple chat agent

The "Hello, World!" of the framework: a conversational agent that keeps history in an `AgentSession`.

```csharp
// SimpleChatAgent.cs
using Microsoft.Agents.AI;

public class SimpleChatAgent
{
    private readonly AIAgent _agent =
        Agents.Create("You are a friendly AI assistant. Keep your responses concise.", name: "chat");

    public async Task RunInteractiveAsync()
    {
        Console.WriteLine("Chat agent ready. Type 'exit' to quit.");
        AgentSession session = await _agent.CreateSessionAsync();

        while (true)
        {
            Console.Write("You: ");
            var userInput = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(userInput) || userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            AgentResponse response = await _agent.RunAsync(userInput, session);
            Console.WriteLine($"Assistant: {response.Text}");
        }
    }
}

// Program.cs:  await new SimpleChatAgent().RunInteractiveAsync();
```

### Recipe 2: Agent with a single tool

Tools are `AIFunction`s created with `AIFunctionFactory.Create` (from `Microsoft.Extensions.AI`). The `[Description]` attributes become the tool and parameter descriptions the model sees.

```csharp
// AgentWithTool.cs
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class TimeTools
{
    [Description("Gets the current UTC date and time.")]
    public static string GetCurrentTime() => DateTime.UtcNow.ToString("F");
}

public class ToolAgent
{
    public AIAgent Agent { get; } = Agents.Create(
        "You have a tool to get the current time. Use it when asked about the time.",
        tools: [AIFunctionFactory.Create(TimeTools.GetCurrentTime)]);

    public async Task<string> AskAboutTimeAsync()
    {
        AgentResponse response = await Agent.RunAsync("What time is it right now?");
        return response.Text;
    }
}
```

### Recipe 3: Error handling in tools

If a tool throws, the function-invocation loop catches the exception and tells the model the call failed — but by default **without** the exception message, so internal details do not leak into the conversation. When the model should know *why*, return the explanation as the result instead of throwing.

```csharp
// ErrorHandlingTool.cs
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class MathTools
{
    [Description("Divides two numbers.")]
    public static string Divide(
        [Description("The number to divide")] double numerator,
        [Description("The number to divide by")] double denominator)
    {
        if (denominator == 0)
        {
            // A result string reaches the model verbatim; it can then ask the user for another value.
            return "Error: cannot divide by zero. Ask the user for a non-zero denominator.";
        }

        return (numerator / denominator).ToString();
    }

    public static async Task DemoAsync()
    {
        AIAgent agent = Agents.Create("You are a calculator.", tools: [AIFunctionFactory.Create(Divide)]);
        AgentResponse response = await agent.RunAsync("What is 10 divided by 0?");
        Console.WriteLine(response.Text);   // e.g. "I can't divide by zero — could you give me a different number?"
    }
}
```

If you build the pipeline yourself (`chatClient.AsBuilder().UseFunctionInvocation(configure: f => f.IncludeDetailedErrors = true)`), exception messages are passed to the model too; only do that when they contain nothing sensitive.

---

## Intermediate Recipes

### Recipe 4: Sequential multi-agent workflow

Two agents in a pipeline — a researcher, then a summariser — built with `AgentWorkflowBuilder.BuildSequential` from `Microsoft.Agents.AI.Workflows`. Each agent sees the conversation so far and adds its turn. (Do not share one `AgentSession` between two different agents; a session belongs to the agent that created it.)

```csharp
// SequentialWorkflow.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class SequentialWorkflow
{
    public static async Task<string> RunAsync(string topic)
    {
        AIAgent researcher = Agents.Create(
            "You are a world-class researcher. Find detailed information on the given topic.", name: "researcher");
        AIAgent summariser = Agents.Create(
            "You are a skilled editor. Summarise the conversation so far into one concise paragraph.", name: "summariser");

        Workflow workflow = AgentWorkflowBuilder.BuildSequential(researcher, summariser);

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, $"Research the topic: {topic}") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));   // start the agents

        string summary = "";
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent output && output.Is<List<ChatMessage>>(out var conversation))
            {
                summary = conversation.Last().Text;   // the summariser's turn
            }
        }

        return summary;
    }
}
```

### Recipe 5: Router to specialist agents

A small router agent classifies the request with structured output (`RunAsync<T>`), then the matching specialist answers. For routing where the specialists can hand the conversation back and forth, use `AgentWorkflowBuilder.CreateHandoffBuilderWith(...)` instead (see [handoff](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#handoff) and [Recipe 14](#recipe-14-support-handoff-with-approval-gated-refunds)).

```csharp
// RouterWorkflow.cs
using System.Text.Json.Serialization;
using Microsoft.Agents.AI;

[JsonConverter(typeof(JsonStringEnumConverter<Route>))]
public enum Route { Billing, TechnicalSupport }

public record RouteDecision(Route Route, string Reason);

public class RouterWorkflow
{
    private readonly AIAgent _router = Agents.Create(
        "Classify the user's request as Billing or TechnicalSupport.", name: "router");

    private readonly Dictionary<Route, AIAgent> _specialists = new()
    {
        [Route.Billing] = Agents.Create("You are a billing support specialist.", name: "billing"),
        [Route.TechnicalSupport] = Agents.Create("You are a technical support specialist.", name: "tech"),
    };

    public async Task<string> RouteAndHandleRequestAsync(string userInput)
    {
        AgentResponse<RouteDecision> decision = await _router.RunAsync<RouteDecision>(userInput);
        AIAgent specialist = _specialists[decision.Result.Route];

        AgentResponse answer = await specialist.RunAsync(userInput);
        return answer.Text;
    }
}
```

### Recipe 6: Retrieval with Azure AI Search

The framework's retrieval hook is `TextSearchProvider`, an `AIContextProvider` that runs your search function and injects the results into the prompt before each model call. The search itself is plain `Azure.Search.Documents` (`dotnet add package Azure.Search.Documents`). Ingestion is done with the search SDK too; there is no framework-level "memory upsert".

```csharp
// KnowledgeWorker.cs
using Azure.Identity;
using Azure.Search.Documents;
using Azure.Search.Documents.Models;
using Microsoft.Agents.AI;

public record KnowledgeDocument(string Id, string Content, string Source);

public class KnowledgeWorker
{
    private readonly SearchClient _search = new(
        new Uri(Environment.GetEnvironmentVariable("AZURE_SEARCH_ENDPOINT")!),
        indexName: "knowledge",
        new DefaultAzureCredential());

    private readonly AIAgent _ragAgent;

    public KnowledgeWorker()
    {
        var retrieval = new TextSearchProvider(SearchAsync, new TextSearchProviderOptions
        {
            SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
        }, loggerFactory: null);

        _ragAgent = new ChatClientAgent(Agents.ChatClient(), new ChatClientAgentOptions
        {
            Name = "rag",
            ChatOptions = new() { Instructions = "Answer using the provided sources and cite them. Say so if they don't cover the question." },
            AIContextProviders = [retrieval],
        });
    }

    public async Task IngestKnowledgeAsync(string documentId, string content, string source)
    {
        await _search.MergeOrUploadDocumentsAsync([new KnowledgeDocument(documentId, content, source)]);
    }

    public async Task<string> AnswerQuestionAsync(string question)
    {
        AgentResponse response = await _ragAgent.RunAsync(question);
        return response.Text;
    }

    private async Task<IEnumerable<TextSearchProvider.TextSearchResult>> SearchAsync(string query, CancellationToken ct)
    {
        SearchResults<KnowledgeDocument> results =
            await _search.SearchAsync<KnowledgeDocument>(query, new SearchOptions { Size = 5 }, ct);

        var hits = new List<TextSearchProvider.TextSearchResult>();
        await foreach (SearchResult<KnowledgeDocument> hit in results.GetResultsAsync())
        {
            hits.Add(new TextSearchProvider.TextSearchResult
            {
                Text = hit.Document.Content,
                SourceName = hit.Document.Source,
            });
        }
        return hits;
    }
}
```

`ChatClientAgentOptions` is the long-form way to configure an agent: instructions go in `ChatOptions.Instructions`, context providers in `AIContextProviders`. Set `SearchTime = OnDemandFunctionCalling` to expose search as a tool the model calls only when it needs to.

---

## Advanced Recipes

### Recipe 7: RAG end to end

```csharp
// RagSystem.cs
public class RagSystem
{
    private readonly KnowledgeWorker _knowledge = new();

    public async Task RunExampleAsync()
    {
        // 1. Ingest documents into the index
        await _knowledge.IngestKnowledgeAsync("doc-001",
            "The Microsoft Agent Framework unifies Semantic Kernel and AutoGen.", "overview.md");
        await _knowledge.IngestKnowledgeAsync("doc-002",
            "The framework is available for .NET and Python.", "platforms.md");

        // 2. Ask a question that needs retrieval
        const string question = "What two frameworks were unified?";
        var answer = await _knowledge.AnswerQuestionAsync(question);

        Console.WriteLine($"Question: {question}");
        Console.WriteLine($"Answer: {answer}");   // mentions Semantic Kernel and AutoGen, citing overview.md
    }
}
```

Azure AI Search indexes documents asynchronously, so in a real system ingestion and querying are separate processes; this example runs them together for brevity.

### Recipe 8: Explicit multi-agent graph with `WorkflowBuilder`

`WorkflowBuilder` gives you the graph directly: agents (or custom executors) as nodes, edges between them, and a declared output. An `AIAgent` converts to an executor binding implicitly.

```csharp
// AdvancedWorkflow.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public class AdvancedWorkflow
{
    public async Task<string> RunPipelineAsync(string topic)
    {
        AIAgent researcher = Agents.Create("You are a researcher. Find three key points on the topic.", name: "researcher");
        AIAgent analyst = Agents.Create("You are a data analyst. Assess the key points critically.", name: "analyst");
        AIAgent writer = Agents.Create("You are a technical writer. Write a short brief from the analysis.", name: "writer");

        Workflow workflow = new WorkflowBuilder(researcher)
            .AddEdge(researcher, analyst)
            .AddEdge(analyst, writer)
            .WithOutputFrom(writer)
            .WithName("ResearchBriefWorkflow")
            .Build();

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new ChatMessage(ChatRole.User, topic));
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

        var brief = new System.Text.StringBuilder();
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case AgentResponseUpdateEvent update when update.ExecutorId.StartsWith("writer"):
                    brief.Append(update.Update.Text);
                    break;
                case WorkflowErrorEvent error:
                    throw new InvalidOperationException("Workflow failed", error.Exception);
            }
        }

        return brief.ToString();
    }
}
```

Use `AddEdge<T>(source, target, condition)` for conditional routing, `AddFanOutEdge` / `AddFanInBarrierEdge` for parallel branches, and `AddSwitch` for multi-way branching; all are covered on the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#edges-and-routing). Agent nodes only start when they receive the `TurnToken`, and they ignore plain `string` input, which is why the run starts with a `ChatMessage`.

---

## Integration Recipes

### Recipe 9: Exposing an agent through ASP.NET Core

`Microsoft.Agents.AI.Hosting` (preview) registers agents in DI by name; `AddAIAgent` resolves the `IChatClient` you registered. You can write your own endpoint, keeping one session per conversation, or let `Microsoft.Agents.AI.Hosting.OpenAI` expose an OpenAI-compatible Responses endpoint for you.

```bash
dotnet add package Microsoft.Agents.AI.Hosting --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.OpenAI --prerelease   # optional: OpenAI-compatible endpoints
```

```csharp
// Program.cs (ASP.NET Core minimal API)
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddChatClient(Agents.ChatClient());
builder.Services.AddDistributedMemoryCache();          // swap for Redis / SQL in production
builder.AddAIAgent("assistant", "You are a helpful assistant. Keep answers short.");
builder.AddOpenAIResponses();                           // optional, for app.MapOpenAIResponses()

var app = builder.Build();

app.MapPost("/chat/{conversationId}", async (
    string conversationId,
    ChatRequest request,
    [FromKeyedServices("assistant")] AIAgent agent,
    IDistributedCache cache,
    CancellationToken ct) =>
{
    // Restore this conversation's session, or start a new one.
    string? saved = await cache.GetStringAsync($"session:{conversationId}", ct);
    AgentSession session = saved is null
        ? await agent.CreateSessionAsync(ct)
        : await agent.DeserializeSessionAsync(JsonDocument.Parse(saved).RootElement, cancellationToken: ct);

    AgentResponse response = await agent.RunAsync(request.Message, session, cancellationToken: ct);

    JsonElement state = await agent.SerializeSessionAsync(session, cancellationToken: ct);
    await cache.SetStringAsync($"session:{conversationId}", state.GetRawText(), ct);

    return Results.Ok(new { reply = response.Text });
});

// OpenAI Responses-compatible endpoint for every registered agent (POST /v1/responses).
app.MapOpenAIResponses();

app.Run();

public record ChatRequest(string Message);
```

Authenticate the endpoint before exposing it (`app.MapPost(...).RequireAuthorization()`), and derive `conversationId` from the authenticated user rather than trusting the URL alone. A request to the shared `/v1/responses` route names the agent with `"agent": {"name": "assistant"}`, not with `model`. `AIHostAgent` and the built-in session stores replace the hand-written cache code here; see [your own streaming endpoint](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#your-own-streaming-endpoint).

### Recipe 10: Event-driven agents with Azure Functions

A queue-triggered function on the isolated worker model. The agent is created once and reused across invocations.

```csharp
// AgentTriggerFunction.cs
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;

public class AgentTriggerFunction(ILogger<AgentTriggerFunction> logger)
{
    private static readonly AdvancedWorkflow Workflow = new();

    [Function("ProcessQueueItem")]
    public async Task Run([QueueTrigger("agent-tasks", Connection = "AzureWebJobsStorage")] string topic)
    {
        logger.LogInformation("Processing topic {Topic}", topic);

        string brief = await Workflow.RunPipelineAsync(topic);

        logger.LogInformation("Workflow completed with {Length} characters", brief.Length);
        // Write the result to another queue, a blob or a database.
    }
}

// Program.cs for the Functions isolated worker:
//   var builder = FunctionsApplication.CreateBuilder(args);
//   builder.Build().Run();
```

Long-running or human-in-the-loop work fits Durable Functions better than a single queue trigger, because a durable orchestration can wait days for an approval without holding a worker; see [durable agents and workflows](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#durable-agents-and-workflows-on-azure-functions).

---

## Workflow and Orchestration Recipes

### Recipe 12: Writer and critic with a round cap

Two agents iterate on a draft until the critic approves it. Each agent is called from its own executor, so the critic's verdict is a typed record (`RunAsync<Verdict>`) the workflow can route on, and the writer is a multi-handler executor that takes either the brief or a revision request. Workflows in .NET have no superstep limit, so the round cap in the critic is what guarantees the loop ends.

```csharp
// WriterCritic.cs
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;

[Description("The critic's verdict on a draft.")]
public sealed record Verdict(bool Approved, string Feedback);

public sealed record Draft(string Text, int Round);

// The writer handles the brief and every revision request.
public sealed class WriterExecutor(AIAgent writer) : Executor("writer")
{
    protected override ProtocolBuilder ConfigureProtocol(ProtocolBuilder protocol) =>
        protocol.ConfigureRoutes(routes => routes
            .AddHandler<string, Draft>(async (brief, context, ct) =>
                new Draft((await writer.RunAsync(brief, cancellationToken: ct)).Text, Round: 1))
            .AddHandler<(Draft Draft, Verdict Verdict), Draft>(async (review, context, ct) =>
                new Draft((await writer.RunAsync(
                    $"Revise this draft.\nFeedback: {review.Verdict.Feedback}\nDraft: {review.Draft.Text}", cancellationToken: ct)).Text,
                    review.Draft.Round + 1)));
}

[SendsMessage(typeof((Draft, Verdict)))]
[YieldsOutput(typeof(string))]
public sealed partial class CriticExecutor(AIAgent critic, int maxRounds) : Executor<Draft>("critic")
{
    public override async ValueTask HandleAsync(Draft draft, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        Verdict verdict = (await critic.RunAsync<Verdict>($"Review this draft:\n{draft.Text}", cancellationToken: cancellationToken)).Result;

        if (verdict.Approved || draft.Round >= maxRounds)   // the round cap is the loop's only guaranteed exit
        {
            await context.YieldOutputAsync($"[round {draft.Round}, {(verdict.Approved ? "approved" : "cap reached")}] {draft.Text}", cancellationToken);
        }
        else
        {
            await context.SendMessageAsync((draft, verdict), cancellationToken: cancellationToken);
        }
    }
}

public static class WriterCriticRecipe
{
    public static async Task RunAsync()
    {
        var writer = new WriterExecutor(Agents.Create("You write short, vivid product copy.", name: "writer"));
        var critic = new CriticExecutor(Agents.Create("You are a demanding copy editor. Approve only tight, concrete copy.", name: "critic"), maxRounds: 3);

        Workflow workflow = new WorkflowBuilder(writer)
            .AddEdge(writer, critic)
            .AddEdge(critic, writer)
            .WithOutputFrom(critic)
            .Build();

        await using Run run = await InProcessExecution.RunAsync(workflow, "Copy for a folding e-bike, under 40 words.");
        foreach (WorkflowEvent evt in run.OutgoingEvents)
        {
            if (evt is WorkflowOutputEvent output) Console.WriteLine(output.Data);
            if (evt is WorkflowErrorEvent error) Console.WriteLine(error.Exception);
        }
    }
}
```

Because the critic decides whether to loop, the cycle is two plain edges rather than a switch. Value tuples are fine as messages here, but not as checkpointed state (their items are fields), so use a record if you add checkpointing.

### Recipe 13: A review panel inside a pipeline

A concurrent orchestration wrapped as an agent becomes one step of a sequential one: a summariser restates the proposal, three reviewers critique it in parallel, and a chair writes the recommendation from their combined critique.

```csharp
// PanelInPipeline.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class PanelInPipelineRecipe
{
    public static async Task RunAsync()
    {
        // Process-wide, once at start-up. Without it the inner agents' streamed updates leak into the
        // outer conversation as fragments, and the aggregator's message is lost.
        Futures.EnableAgentResponseOutputTaggingAndFiltering = true;

        // Three reviewers in parallel, wrapped as one agent ...
        AIAgent panel = AgentWorkflowBuilder.BuildConcurrent(
            "review-panel",
            [
                Agents.Create("Critique the proposal's cost in one sentence.", name: "finance"),
                Agents.Create("Critique the proposal's legal risk in one sentence.", name: "legal"),
                Agents.Create("Critique the proposal's effect on customers in one sentence.", name: "customer"),
            ],
            // One message for the next step, rather than three.
            aggregator: perAgent =>
            [
                new ChatMessage(ChatRole.Assistant, string.Join("\n", perAgent.Select(m => $"{m.Last().AuthorName}: {m.Last().Text}"))) { AuthorName = "review-panel" },
            ]).AsAIAgent(name: "review-panel", includeWorkflowOutputsInResponse: true);   // the aggregated message is the reply

        // ... which is then one step of a sequential pipeline.
        Workflow pipeline = AgentWorkflowBuilder.BuildSequential(
            "proposal-review",
            Agents.Create("Restate the proposal in three bullet points.", name: "summariser"),
            panel,
            Agents.Create("Write a recommendation from the critiques: go, no-go or revise.", name: "chair"));

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            pipeline, new List<ChatMessage> { new(ChatRole.User, "Close the high-street branches and move all support online.") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: false));

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent output && output.Is(out List<ChatMessage>? conversation))
            {
                foreach (ChatMessage message in conversation.Skip(1))
                {
                    Console.WriteLine($"{message.AuthorName}: {message.Text}");
                }
            }
        }
    }
}
```

Without the `Futures` switch, the nested workflow's agents stream their updates straight into the outer conversation, which then holds dozens of fragments (one per streamed token, interleaved across the three reviewers) and never the aggregated message. With it, agent events go through the inner workflow's output filter, and `includeWorkflowOutputsInResponse: true` makes the aggregator's message the panel's reply. The switch is process-wide and becomes the default in a later major version.

### Recipe 14: Support handoff with approval-gated refunds

A triage agent hands customers to an orders specialist or a refunds specialist. The refund tool requires approval, so issuing money pauses the workflow until a person (here a rule standing in for one) answers.

```csharp
// SupportHandoff.cs
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class SupportTools
{
    [Description("Look up an order's status.")]
    public static string LookUpOrder([Description("Order number")] string orderId) => $"Order {orderId}: delivered on Monday.";

    [Description("Issue a refund. Needs a person's approval.")]
    public static string IssueRefund([Description("Order number")] string orderId, [Description("Amount in pounds")] decimal amount) =>
        $"Refunded £{amount:0.00} for order {orderId}.";
}

public static class SupportHandoffRecipe
{
    public static async Task RunAsync()
    {
        AIAgent triage = Agents.Create("Greet the customer and hand off to orders or refunds. Never answer yourself.", name: "triage");
        AIAgent orders = Agents.Create("You answer order-status questions with the LookUpOrder tool.", name: "orders",
            tools: [AIFunctionFactory.Create(SupportTools.LookUpOrder)]);
        AIAgent refunds = Agents.Create("You issue refunds with the IssueRefund tool.", name: "refunds",
            tools: [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(SupportTools.IssueRefund))]);

        Workflow workflow = AgentWorkflowBuilder.CreateHandoffBuilderWith(triage)
            .WithHandoff(triage, orders, "Order status and delivery questions")
            .WithHandoff(triage, refunds, "Refund requests")
            .WithHandoffs([orders, refunds], triage)
            .Build();

        List<ChatMessage> history = [new(ChatRole.User, "My order 1042 arrived broken. Can I have my £25 back?")];

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, history);
        await run.TrySendMessageAsync(new TurnToken(emitEvents: false));

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                // The refund tool waits for a person; here a rule stands in for the support lead.
                case RequestInfoEvent pending when pending.Request.TryGetDataAs(out ToolApprovalRequestContent? approval):
                    var call = (FunctionCallContent)approval.ToolCall;
                    bool approved = call.Arguments?.TryGetValue("amount", out object? amount) == true && Convert.ToDecimal(amount?.ToString()) <= 50m;
                    Console.WriteLine($"approval for {call.Name}: {(approved ? "granted" : "refused")}");
                    await run.SendResponseAsync(pending.Request.CreateResponse(approval.CreateResponse(approved)));
                    break;
                case WorkflowOutputEvent output when output.Is(out List<ChatMessage>? conversation):
                    Console.WriteLine($"{conversation.Last().AuthorName}: {conversation.Last().Text}");
                    break;
            }
        }
    }
}
```

In a web app, return the approval request to a support lead's queue and answer it later; with checkpointing the pending approval survives restarts, as on the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#resuming-in-another-process). Run the workflow again with the returned conversation for the customer's next message.

### Recipe 15: Bounded agent calls

A model call can hang or be throttled. A linked `CancellationTokenSource` gives each attempt a time limit while still honouring the caller's cancellation, and a small retry loop handles timeouts, 429s and 5xx responses before falling back to a fixed answer.

```csharp
// BoundedCalls.cs
using Microsoft.Agents.AI;

public static class BoundedCalls
{
    // A time limit per attempt, a few retries, and a fallback answer instead of an exception.
    public static async Task<string> AskWithLimitsAsync(AIAgent agent, string question, CancellationToken requestAborted = default)
    {
        for (int attempt = 1; attempt <= 3; attempt++)
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(requestAborted);
            timeout.CancelAfter(TimeSpan.FromSeconds(20));
            try
            {
                return (await agent.RunAsync(question, cancellationToken: timeout.Token)).Text;
            }
            catch (OperationCanceledException) when (!requestAborted.IsCancellationRequested)
            {
                Console.WriteLine($"attempt {attempt} timed out");
            }
            catch (System.ClientModel.ClientResultException ex) when (ex.Status is 429 or >= 500)
            {
                Console.WriteLine($"attempt {attempt} failed with HTTP {ex.Status}");
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), requestAborted);
            }
        }

        return "Sorry, the assistant is not available right now. Please try again shortly.";
    }

    public static async Task RunAsync() =>
        Console.WriteLine(await AskWithLimitsAsync(Agents.Create("You are a helpful assistant."), "What is a superstep?"));
}
```

For retries at the HTTP level, with jitter and a circuit breaker, use `Microsoft.Extensions.Http.Resilience` as on the [production guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_production_guide/#resilience); keep the timeout here, which also covers tool calls and streaming that an HTTP policy cannot see. Do not retry a call whose tools have side effects unless they are idempotent.

### Recipe 16: Map-reduce summarisation

A long document is split into chunks, each chunk is summarised by an agent with bounded parallelism, and a final call combines the partial summaries. Fan-out edges suit a fixed set of branches; when the number of branches depends on the input, as here, run the parallel work inside one executor.

```csharp
// MapReduceSummary.cs
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;

public static class MapReduceSummary
{
    public static Workflow Build(AIAgent summariser, int maxParallel = 4)
    {
        // Map: one agent call per chunk, a few at a time. The number of chunks is only known at run time,
        // so the parallelism lives inside one executor rather than in fan-out edges.
        ExecutorBinding map = ((Func<string, CancellationToken, ValueTask<string[]>>)(async (document, ct) =>
        {
            string[] chunks = document.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
            using var gate = new SemaphoreSlim(maxParallel);
            return await Task.WhenAll(chunks.Select(async chunk =>
            {
                await gate.WaitAsync(ct);
                try { return (await summariser.RunAsync($"Summarise in one sentence:\n{chunk}", cancellationToken: ct)).Text; }
                finally { gate.Release(); }
            }));
        })).BindAsExecutor("map");

        // Reduce: one call over the partial summaries.
        ExecutorBinding reduce = ((Func<string[], CancellationToken, ValueTask<string>>)(async (partials, ct) =>
            (await summariser.RunAsync($"Combine these into one paragraph:\n- {string.Join("\n- ", partials)}", cancellationToken: ct)).Text))
            .BindAsExecutor("reduce");

        return new WorkflowBuilder(map).AddEdge(map, reduce).WithOutputFrom(reduce).Build();
    }

    public static async Task RunAsync()
    {
        string report = string.Join("\n\n", Enumerable.Range(1, 6).Select(i => $"Section {i}: findings from region {i}."));
        Workflow workflow = Build(Agents.Create("You summarise text faithfully.", name: "summariser"));

        await using Run run = await InProcessExecution.RunAsync(workflow, report);
        Console.WriteLine(run.OutgoingEvents.OfType<WorkflowOutputEvent>().Single().Data);
    }
}
```

An array is a valid message type, so `map` returns `string[]` and `reduce` receives it. The semaphore bounds concurrent model calls; set it below your deployment's rate limit.

---

## Troubleshooting Patterns

### Pattern 11: Watching tool calls with streaming

`RunStreamingAsync` yields `AgentResponseUpdate`s. Their `Contents` show the model's text as it arrives and the tool calls and results as they happen.

```csharp
// StreamingDebug.cs
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class StreamingDebug
{
    public static async Task DebugWithStreamingAsync(string userInput)
    {
        AIAgent agent = new ToolAgent().Agent;
        AgentSession session = await agent.CreateSessionAsync();

        await foreach (AgentResponseUpdate update in agent.RunStreamingAsync(userInput, session))
        {
            foreach (AIContent content in update.Contents)
            {
                switch (content)
                {
                    case FunctionCallContent call:
                        Console.ForegroundColor = ConsoleColor.Gray;
                        Console.WriteLine($"[tool call] {call.Name}({string.Join(", ", call.Arguments ?? new Dictionary<string, object?>())})");
                        Console.ResetColor();
                        break;
                    case FunctionResultContent result:
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"[tool result] {result.Result}");
                        Console.ResetColor();
                        break;
                    case TextContent text:
                        Console.Write(text.Text);
                        break;
                }
            }
        }

        Console.WriteLine();
    }
}
```

For anything beyond local debugging, wrap the agent with `.AsBuilder().UseOpenTelemetry(...)` and look at the spans instead; see the [observability page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/#tracing-an-agent).

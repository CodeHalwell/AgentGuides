---
title: "Microsoft Agent Framework (.NET) — Hosting"
description: "Serving .NET agents and workflows: AddAIAgent and AddWorkflow registration, OpenAI Responses, Conversations and Chat Completions endpoints, A2A servers and clients, AG-UI servers and clients, DevUI, your own SSE endpoint with AIHostAgent, human-in-the-loop workflows over HTTP with HostedWorkflowState, session stores and per-user isolation, Foundry hosted agents and durable agents on Azure Functions."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Hosting"
  order: 12
---

The hosting packages turn agents and workflows registered in an ASP.NET Core app into HTTP endpoints, one package per wire protocol. They share one registration model: `builder.AddAIAgent(name, ...)` puts an `AIAgent` in DI as a keyed service under its name, and each `Map...` call exposes registered agents in a protocol. You can serve one agent in several protocols from one app, and mix framework endpoints with your own routes.

| Package | Status | Gives you |
| --- | --- | --- |
| `Microsoft.Agents.AI.Hosting` | preview | `AddAIAgent`, `AddWorkflow`, session stores, `AIHostAgent`, `HostedWorkflowState` |
| `Microsoft.Agents.AI.Hosting.AspNetCore` | preview | Claims-based session isolation |
| `Microsoft.Agents.AI.Hosting.OpenAI` | alpha | OpenAI Responses, Conversations and Chat Completions endpoints |
| `Microsoft.Agents.AI.Hosting.A2A.AspNetCore` | preview | A2A server (JSON-RPC and HTTP+JSON bindings) |
| `Microsoft.Agents.AI.A2A` | preview | A2A client: a remote A2A agent as an `AIAgent` |
| `Microsoft.Agents.AI.Hosting.AGUI.AspNetCore` | preview | AG-UI server |
| `AGUI.Client` | stable | AG-UI client (`AGUIChatClient`), from the AG-UI project |
| `Microsoft.Agents.AI.DevUI` | preview | Browser UI for trying agents and workflows in development |
| `Microsoft.Agents.AI.Hosting.AzureStorage` | preview | Azure Blob session store |
| `Microsoft.Agents.AI.Foundry.Hosting` | preview | Your agent as a Microsoft Foundry hosted agent |
| `Microsoft.Agents.AI.Hosting.AzureFunctions`, `Microsoft.Agents.AI.DurableTask` | preview | Durable agents and workflows on Azure Functions or the Durable Task Scheduler |

The C# blocks on this page compile against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% with the matching preview packages (checked %%versions-checked%%). The ASP.NET Core apps were started and called with `curl` and with the client examples, with the agents' model calls going to a local stand-in for the OpenAI API, so the replies are placeholders. The Foundry and Azure Functions blocks were compiled but not run, as they need Azure resources.

```bash
dotnet add package Microsoft.Agents.AI.Hosting --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.OpenAI --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.A2A.AspNetCore --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.AGUI.AspNetCore --prerelease
dotnet add package Microsoft.Agents.AI.DevUI --prerelease
```

The examples use the `Models` helper from the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/), which returns an `IChatClient` for `OPENAI_API_KEY` and an optional `OPENAI_BASE_URL`.

## How it differs from Python

Python serves agents through separate entry points: DevUI's `serve(...)`, the A2A and AG-UI integrations, and the Azure Functions package. In .NET everything goes through DI and ASP.NET Core routing: you register agents once and choose endpoints with `Map...` calls, and authentication, CORS, rate limiting and OpenTelemetry are the ordinary ASP.NET Core features applied to those routes. There is no `serve()` helper and no CLI; `dotnet run` starts the app.

## Registering agents and workflows

`AddAIAgent` has overloads for the common cases: name and instructions (the agent uses the `IChatClient` registered in DI), an explicit `IChatClient`, a keyed chat client (`chatClientServiceKey`), or a factory `(IServiceProvider sp, string name) => AIAgent` for anything else, including agents that are not `ChatClientAgent`s. All take a `ServiceLifetime`, `Singleton` by default. The returned `IHostedAgentBuilder` adds tools (`WithAITool`, `WithAITools`, or a tool factory resolved from DI) and a [session store](#session-stores-and-isolation).

`AddWorkflow(name, (sp, name) => Workflow)` registers a workflow, and `.AddAsAIAgent()` also registers it as an `AIAgent` under the same name, so every endpoint below serves it like any agent. `includeWorkflowOutputsInResponse: true` adds non-message workflow outputs to the response.

Use singleton agents unless an agent's tools depend on scoped services, such as a per-request `DbContext`; then register it as `Scoped`. Per-conversation state belongs in sessions, never in the agent.

## OpenAI-compatible endpoints

`Microsoft.Agents.AI.Hosting.OpenAI` (alpha) implements the OpenAI Responses, Conversations and Chat Completions APIs over your agents, so any OpenAI client library, and tools built on one, can call them.

```csharp
// Program.cs: two agents and a workflow behind OpenAI-compatible endpoints.
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

// One IChatClient for the app; AddAIAgent resolves it from DI.
builder.Services.AddChatClient(Models.ChatClient());

[Description("Look up an order's status.")]
static string GetOrderStatus([Description("The order number")] string orderId) => $"Order {orderId} shipped yesterday.";

// Agents are registered as keyed AIAgent services under their name.
builder.AddAIAgent("support", "You answer questions about orders. Be brief.")
    .WithAITool(AIFunctionFactory.Create(GetOrderStatus, name: "get_order_status"))
    // Keep conversations between requests, in this process. Isolation (the default) needs an
    // AgentIsolationKeyProvider, normally UseClaimsBasedAgentIsolation(); without auth, turn it off.
    .WithInMemorySessionStore(withIsolation: false);

IHostedAgentBuilder translator = builder.AddAIAgent("translator", "Translate the user's message into French.");

// A workflow registered with AddAsAIAgent is served exactly like an agent.
builder.AddWorkflow("support-then-translate", (sp, name) => AgentWorkflowBuilder.BuildSequential(
        name,
        sp.GetRequiredKeyedService<AIAgent>("support"),
        sp.GetRequiredKeyedService<AIAgent>("translator")))
    .AddAsAIAgent();

builder.AddOpenAIResponses();         // POST /v1/responses
builder.AddOpenAIConversations();     // /v1/conversations, for stored conversation state
builder.AddOpenAIChatCompletions();   // POST /{agent}/v1/chat/completions

var app = builder.Build();

app.MapOpenAIResponses();                  // every registered agent; the request's agent.name picks one
app.MapOpenAIConversations();
app.MapOpenAIChatCompletions(translator);  // per agent: POST /translator/v1/chat/completions

app.Run();
```

| Call | Route | Selects the agent by |
| --- | --- | --- |
| `MapOpenAIResponses()` | `POST /v1/responses`, plus `GET`, `DELETE`, `/cancel` and `/input_items` per response | `agent.name` in the body, or `metadata.entity_id` |
| `MapOpenAIResponses(agentBuilder)` / `(agent)` | `POST /{name}/v1/responses` | The route |
| `MapOpenAIConversations()` | `/v1/conversations` and its items | |
| `MapOpenAIChatCompletions(agentBuilder)` / `(agent)` | `POST /{name}/v1/chat/completions` | The route |

Each needs the matching `AddOpenAIResponses()`, `AddOpenAIConversations()` or `AddOpenAIChatCompletions()` on the builder. A request to the shared Responses route picks the agent with `"agent": {"name": "support"}` (or `"metadata": {"entity_id": "support"}`); the `model` field is informational and does **not** select an agent. A request with neither fails with `No 'agent.name' or 'metadata["entity_id"]' specified in the request`.

```bash
curl http://localhost:5784/v1/responses -H 'content-type: application/json' \
  -d '{"agent": {"name": "support"}, "input": "Where is order 77?"}'

# Continue the conversation from a previous response, or stream with "stream": true
curl http://localhost:5784/v1/responses -H 'content-type: application/json' \
  -d '{"agent": {"name": "support"}, "input": "Thanks", "previous_response_id": "resp_..."}'
```

By default the endpoints reject any request that tries to change the agent's configuration. A body with `temperature`, `instructions`, `tools` or `tool_choice` fails with `The following request setting(s) are not supported by this agent endpoint: temperature`. To honour such settings, set `OpenAIResponsesMapOptions.RunOptionsFactory` (or `OpenAIChatCompletionsMapOptions.RunOptionsFactory`) to a function that maps the `OpenAIResponseRequestInfo` onto `AgentRunOptions`. `DangerouslyAllowClientFunctionTools = true` forwards client-declared functions to the model; as the name says, the caller can then steer which tools the model picks.

From the client side, the endpoint is just another OpenAI-compatible server:

```csharp
using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

// Any OpenAI client can call the hosted agents; here the framework itself is the client.
public static class CallHostedAgent
{
    public static async Task RunAsync()
    {
        var hostedTranslator = new OpenAIClient(
                new ApiKeyCredential("unused"),   // the host does no authentication of its own: add it
                new OpenAIClientOptions { Endpoint = new Uri("http://localhost:5784/translator/v1") })
            .GetChatClient("translator")
            .AsIChatClient();

        AIAgent remote = new ChatClientAgent(hostedTranslator, name: "remote-translator");
        AgentResponse response = await remote.RunAsync("Good morning");
        Console.WriteLine(response.Text);
    }
}
```

Two behaviours to know about. The Chat Completions endpoint returns each message the agent produced as a separate choice, so for an agent with tools `choices[0]` is the agent's internal tool call and the answer comes later. A standard client reading the first choice sees a tool call it cannot run, which is why the client above calls the tool-less translator; prefer the Responses endpoint for agents with tools. And none of these endpoints authenticate: add authentication and call `.RequireAuthorization()` on the returned builders, as in the [session store example](#session-stores-and-isolation). Response and conversation IDs are not secrets.

## A2A

[Agent2Agent](https://a2a-protocol.org/) lets agents on different frameworks and hosts call each other. The server package publishes an `AIAgent` over A2A protocol version 1.0 with the JSON-RPC and HTTP+JSON bindings; the client package turns a remote A2A agent into an `AIAgent`.

```csharp
// Program.cs: publish an agent over A2A (JSON-RPC and HTTP+JSON bindings) with its agent card.
using A2A;
using A2A.AspNetCore;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.A2A;

var builder = WebApplication.CreateBuilder(args);
string publicUrl = builder.Configuration["PublicUrl"] ?? "http://localhost:5785/agents/policy";

AIAgent policyAgent = new ChatClientAgent(Models.ChatClient(),
    "You answer questions about the returns policy in two sentences.",
    name: "policy", description: "Answers questions about the returns policy.");

builder.AddA2AServer(policyAgent, options =>
{
    // ReturnMessage (the default) answers with a message; ReturnTask creates an A2A task the client can poll.
    options.AgentRunMode = AgentRunMode.ReturnMessage;
    options.ServerOptions = new A2AServerOptions { SupportsStreaming = true };
});

var app = builder.Build();

app.MapA2AJsonRpc(policyAgent, "/agents/policy");
app.MapA2AHttpJson(policyAgent, "/agents/policy");

// Clients discover the agent from its card, which you publish yourself.
app.MapWellKnownAgentCard(new AgentCard
{
    Name = "policy",
    Description = "Answers questions about the returns policy.",
    Version = "1.0.0",
    DefaultInputModes = ["text/plain"],
    DefaultOutputModes = ["text/plain"],
    Capabilities = new AgentCapabilities { Streaming = true },
    Skills = [new A2A.AgentSkill { Id = "returns", Name = "Returns policy", Description = "Explains the returns policy.", Tags = ["policy"] }],
    SupportedInterfaces =
    [
        new AgentInterface { Url = publicUrl, ProtocolBinding = ProtocolBindingNames.JsonRpc, ProtocolVersion = "1.0" },
        new AgentInterface { Url = publicUrl, ProtocolBinding = ProtocolBindingNames.HttpJson, ProtocolVersion = "1.0" },
    ],
}, "/agents/policy");

app.Run();
```

`AddA2AServer` registers the agent's A2A request handler, and `MapA2AJsonRpc` / `MapA2AHttpJson` map it at a path. They do not publish an agent card: clients discover an A2A agent from `/.well-known/agent-card.json`, so map one with `MapWellKnownAgentCard` (from the `A2A.AspNetCore` package), giving it the public URL of each binding. A server without a card works only for clients that are configured with the URL directly.

| `A2AServerRegistrationOptions` | Default | Effect |
| --- | --- | --- |
| `AgentRunMode` | `ReturnMessage` | `ReturnMessage` answers with an A2A message; `ReturnTask` creates a task the client can poll or subscribe to; `ReturnTaskWhen(func)` decides per request |
| `ServerOptions` | | The A2A SDK's `A2AServerOptions`: `SupportsStreaming`, `SupportedInputModes`, `SupportsExtendedAgentCard`, `AutoAppendHistory` |

The overloads taking an agent name or an `IHostedAgentBuilder` serve a registered agent. With a session isolation provider configured, A2A tasks are scoped to the caller as well (`IsolationKeyScopedTaskStore`).

```csharp
using A2A;
using Microsoft.Agents.AI;

public static class CallA2AAgent
{
    public static async Task RunAsync()
    {
        using var http = new HttpClient();
        // e.g. http.DefaultRequestHeaders.Authorization = new("Bearer", token);

        // Reads {baseUrl}/.well-known/agent-card.json and builds an AIAgent from the card.
        var resolver = new A2ACardResolver(new Uri("http://localhost:5785/agents/policy/"), http);
        AIAgent policy = await resolver.GetAIAgentAsync(http);
        Console.WriteLine($"{policy.Name}: {policy.Description}");

        // A session maps to an A2A context ID, so the remote side sees one conversation.
        AgentSession session = await policy.CreateSessionAsync();
        Console.WriteLine((await policy.RunAsync("Can I return an opened item?", session)).Text);

        await foreach (AgentResponseUpdate update in policy.RunStreamingAsync("And without a receipt?", session))
        {
            Console.Write(update.Text);
        }
        Console.WriteLine();
    }
}
```

`A2ACardResolver.GetAIAgentAsync` reads the card and returns an `A2AAgent`. Its sessions are `A2AAgentSession`s, which carry the A2A `ContextId`, `TaskId` and `TaskState`; `CreateSessionAsync(contextId)` resumes a known context. `card.AsAIAgent(http)` and `a2aClient.AsAIAgent()` build the same agent from a card or client you already have. Authentication is whatever the remote card declares; set it on the `HttpClient`. A remote A2A agent can sit in a workflow, be wrapped in middleware, or become a tool with `AsAIFunction()`. The cross-language [A2A page](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/) covers the protocol itself.

## AG-UI

[AG-UI](https://docs.ag-ui.com/) is the event protocol between agent back ends and front ends such as CopilotKit: a client POSTs a `RunAgentInput` (thread ID, messages, front-end tools, state) and receives server-sent events (`RUN_STARTED`, `TEXT_MESSAGE_CONTENT`, `TOOL_CALL_START`, `RUN_FINISHED` and so on).

```csharp
// Program.cs: an AG-UI endpoint for CopilotKit or any other AG-UI client.
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Hosting.AGUI.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddChatClient(Models.ChatClient());
builder.Services.AddAGUIServer();

// The AG-UI endpoint keeps the thread's session in the agent's session store. withIsolation: false
// because this sample has no authentication; see "Session stores and isolation" below.
builder.AddAIAgent("assistant", "You are a helpful assistant.").WithInMemorySessionStore(withIsolation: false);
builder.AddAIAgent("planner", "You turn requests into numbered plans.").WithInMemorySessionStore(withIsolation: false);

var app = builder.Build();

// One endpoint per agent ...
app.MapAGUIServer("assistant", "/agui/assistant");

// ... or one endpoint that picks the agent per request. Returning null answers 404.
app.MapAGUIServer("/agui/by-name/{agent}", context =>
    context.Request.RouteValues["agent"] is string name ? context.RequestServices.GetKeyedService<AIAgent>(name) : null);

app.Run();
```

`MapAGUIServer` has four overloads: a fixed `AIAgent`, a registered agent by name, an `IHostedAgentBuilder`, and a selector `Func<HttpContext, AIAgent?>` that picks the agent per request (returning `null` answers 404 without touching session state). The thread ID keys the session in the agent's session store; sessions are partitioned by the agent's `Name`, so give agents distinct, non-empty names. The name and builder overloads respect the agent's DI lifetime: a singleton agent is resolved once, scoped and transient agents once per request.

The server calls the store with session isolation on by default, so `WithInMemorySessionStore()` without an `AgentIsolationKeyProvider` fails every request with `Agent isolation key is required but was not provided`. Configure claims-based isolation, or pass `withIsolation: false` for an unauthenticated demo as above.

On the client side, the AG-UI client now comes from the AG-UI project's own `AGUI.Client` package. `Microsoft.Agents.AI.AGUI` is no longer updated (its last release is 1.13.0-preview); use `AGUI.Client` with current framework versions.

```csharp
using AGUI.Client;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public static class CallAGUIAgent
{
    public static async Task RunAsync()
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

        // AGUIChatClient (package AGUI.Client) speaks AG-UI over HTTP and SSE.
        var chatClient = new AGUIChatClient(new(http, "http://localhost:5786/agui/assistant"));
        AIAgent remote = chatClient.AsAIAgent(name: "agui-client");

        AgentSession session = await remote.CreateSessionAsync();
        await foreach (AgentResponseUpdate update in remote.RunStreamingAsync("Plan a team offsite.", session))
        {
            foreach (AIContent content in update.Contents)
            {
                if (content is TextContent text) Console.Write(text.Text);
                if (content is ErrorContent error) Console.WriteLine($"[error] {error.Message}");
            }
        }
        Console.WriteLine();
    }
}
```

Front-end tools (tools the browser runs), shared state and human-in-the-loop approval are covered by Microsoft's AG-UI samples (`dotnet/samples/02-agents/AGUI`, steps 3 to 5); the server side needs no extra code for them.

## DevUI

DevUI is a browser UI for chatting with registered agents and running registered workflows during development. It drives them through the Responses and Conversations endpoints, so register those too.

```csharp
// Program.cs: DevUI, a browser UI for trying registered agents and workflows during development.
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.DevUI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Workflows;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddChatClient(Models.ChatClient());

builder.AddAIAgent("writer", "You write short product descriptions.");
builder.AddAIAgent("reviewer", "You review product descriptions for accuracy.");
builder.AddWorkflow("write-and-review", (sp, name) => AgentWorkflowBuilder.BuildSequential(
        name, sp.GetRequiredKeyedService<AIAgent>("writer"), sp.GetRequiredKeyedService<AIAgent>("reviewer")))
    .AddAsAIAgent(includeWorkflowOutputsInResponse: true);

// DevUI talks to the agents through the Responses and Conversations endpoints.
builder.AddOpenAIResponses();
builder.AddOpenAIConversations();

if (builder.Environment.IsDevelopment())
{
    builder.AddDevUI();   // DevUIOptions: AllowRemoteAccess (false), AuthToken
}

var app = builder.Build();
app.MapOpenAIResponses();
app.MapOpenAIConversations();

if (app.Environment.IsDevelopment())
{
    app.MapDevUI();       // http://localhost:<port>/devui
}

app.Run();
```

Open `/devui`. The UI lists every registered entity from `/v1/entities`, including each agent's instructions, tools, model ID and provider URL, so never expose it outside development: map it only when `IsDevelopment()`, as above. `DevUIOptions.AllowRemoteAccess` is `false` by default, and DevUI then rejects requests that do not come from the loopback address. If you allow remote access, set `AuthToken` (or the environment variable named by `DevUIOptions.AuthTokenEnvironmentVariable`) or attach an authorisation policy through `ConfigureEndpoints`; DevUI logs a warning when remote access has neither. Python's DevUI is a separate `devui` command with directory discovery; the .NET one is middleware in your own app.

## Your own streaming endpoint

When none of the protocols fits, write the endpoint yourself. `AIHostAgent` wraps an agent with a session store and adds `GetOrCreateSessionAsync(conversationId)` and `SaveSessionAsync(conversationId, session)`, so a route can load a conversation, stream the reply as server-sent events, and save it.

```csharp
// Program.cs: your own streaming endpoint, with conversations kept in a session store.
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;

var builder = WebApplication.CreateBuilder(args);

AIAgent agent = new ChatClientAgent(Models.ChatClient(), "You are a concise assistant.", name: "assistant");

// AIHostAgent pairs an agent with a session store. Swap InMemoryAgentSessionStore for a durable
// store (Azure Blob, or your own) when you run more than one replica.
var hosted = new AIHostAgent(agent, new InMemoryAgentSessionStore());
builder.Services.AddSingleton(hosted);

var app = builder.Build();

app.MapPost("/chat/{conversationId}", async (string conversationId, ChatRequest request, AIHostAgent host, HttpContext http) =>
{
    CancellationToken ct = http.RequestAborted;   // a closed tab cancels the model call
    http.Response.Headers.ContentType = "text/event-stream";
    http.Response.Headers.CacheControl = "no-cache";

    AgentSession session = await host.GetOrCreateSessionAsync(conversationId, ct);

    await foreach (AgentResponseUpdate update in host.RunStreamingAsync(request.Message, session, cancellationToken: ct))
    {
        if (string.IsNullOrEmpty(update.Text)) continue;   // tool calls, usage and so on
        await http.Response.WriteAsync($"data: {JsonSerializer.Serialize(new { delta = update.Text })}\n\n", ct);
        await http.Response.Body.FlushAsync(ct);
    }

    await host.SaveSessionAsync(conversationId, session, ct);
    await http.Response.WriteAsync("event: done\ndata: {}\n\n", ct);
});

app.Run();

public record ChatRequest(string Message);
```

Pass `http.RequestAborted` through, so a closed browser tab cancels the model call. The second request to the same conversation ID sends the model the earlier exchange as well, because the session was saved. Derive the conversation ID from the authenticated user rather than trusting the URL, or give `AIHostAgent` an isolation-scoped store and call `BindIsolationKey(userId)`, which returns a host agent whose session keys carry that user's key. For the browser side, `EventSource` only supports GET; use `fetch` with a readable stream for POST.

## Human-in-the-loop workflows over HTTP

A workflow that waits for a person cannot hold an HTTP request open. `HostedWorkflowState` turns it into turns: `RunOrResumeAsync(sessionId, input)` starts the workflow on the first call and, on later calls, restores the session's latest checkpoint and delivers the new input (a message for the start executor, or an `ExternalResponse` for a pending request). The result has the turn's `Events` and the new `Checkpoint`.

```csharp
// Program.cs: a human-in-the-loop workflow over HTTP. Each POST is one turn; checkpoints carry the state between turns.
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Agents.AI.Workflows.Checkpointing;

var builder = WebApplication.CreateBuilder(args);

// The factory builds a fresh workflow per turn, so different sessions can run at the same time.
// With CheckpointManager.CreateJson(...) over a durable store, sessions also survive restarts.
builder.Services.AddSingleton(new HostedWorkflowState(
    ct => ValueTask.FromResult(ExpenseWorkflow.Build()),
    CheckpointManager.CreateInMemory()));

var app = builder.Build();

// Turn 1: submit a claim. The reply lists the request the workflow is waiting on.
app.MapPost("/claims/{sessionId}", async (string sessionId, ExpenseClaim claim, HostedWorkflowState state, CancellationToken ct) =>
    ExpenseWorkflow.Summarise(await state.RunOrResumeAsync(sessionId, claim, ct)));

// Turn 2: answer it. RunOrResumeAsync restores the session's latest checkpoint and delivers the response.
app.MapPost("/claims/{sessionId}/requests/{requestId}", async (string sessionId, string requestId, ApprovalDecision decision, HostedWorkflowState state, CancellationToken ct) =>
{
    var response = new ExternalResponse(ExpenseWorkflow.ApprovalPort, requestId, new PortableValue(decision));
    return ExpenseWorkflow.Summarise(await state.RunOrResumeAsync(sessionId, response, ct));
});

app.Run();

public record ExpenseClaim(string Employee, decimal Amount);
public record ApprovalRequest(string Summary);
public record ApprovalDecision(bool Approved, string? Reason = null);

public static class ExpenseWorkflow
{
    public static readonly RequestPortInfo ApprovalPort =
        new(new TypeId(typeof(ApprovalRequest)), new TypeId(typeof(ApprovalDecision)), "approval");

    public static Workflow Build()
    {
        ExecutorBinding intake = ((Func<ExpenseClaim, ApprovalRequest>)(c => new ApprovalRequest($"{c.Employee}: {c.Amount:0.00}"))).BindAsExecutor("intake");
        RequestPort approval = RequestPort.Create<ApprovalRequest, ApprovalDecision>(ApprovalPort.PortId);
        ExecutorBinding settle = ((Func<ApprovalDecision, string>)(d => d.Approved ? "paid" : $"rejected: {d.Reason}")).BindAsExecutor("settle");
        return new WorkflowBuilder(intake).AddEdge(intake, approval).AddEdge(approval, settle).WithOutputFrom(settle).Build();
    }

    public static object Summarise(HostedWorkflowRunResult result) => new
    {
        result.SessionId,
        pending = result.Events.OfType<RequestInfoEvent>()
            .Where(e => !result.Events.OfType<WorkflowOutputEvent>().Any())
            .Select(e => new { e.Request.RequestId, e.Request.Data.As<ApprovalRequest>()?.Summary }),
        outputs = result.Events.OfType<WorkflowOutputEvent>().Select(e => e.Data?.ToString()),
    };
}
```

```bash
curl -X POST localhost:5789/claims/s42 -H 'content-type: application/json' -d '{"employee":"Ana","amount":120.5}'
# {"sessionId":"s42","pending":[{"requestId":"ec0c6af8...","summary":"Ana: 120.50"}],"outputs":[]}
curl -X POST localhost:5789/claims/s42/requests/ec0c6af8... -H 'content-type: application/json' -d '{"approved":false,"reason":"no receipt"}'
# {"sessionId":"s42","pending":[],"outputs":["rejected: no receipt"]}
```

An `ExternalResponse` needs the port's `RequestPortInfo` (port ID plus request and response types), which the example builds once from the port's types. The pending request is raised again when the checkpoint is restored, so the second turn's events include it as well as the output; the example only reports requests when there is no output. Constructed with a factory, `HostedWorkflowState` builds a workflow per turn and different sessions run concurrently; turns for the same session are not serialised, so guard against double submission yourself. The `sessionId` is a partition key: authenticate the caller and check they own it before using it.

## Session stores and isolation

Every endpoint above that keeps conversations loads and saves sessions through an `AgentSessionStore` (experimental, `MAAI001`).

| Store | Package | Use |
| --- | --- | --- |
| `InMemoryAgentSessionStore` (`WithInMemorySessionStore()`) | Hosting | One process; lost on restart |
| `AzureBlobAgentSessionStore` (`WithAzureBlobSessionStore(container, options)`) | Hosting.AzureStorage | One blob per session; `CreateContainerIfNotExists`, `BlobNamePrefix` |
| `NoopAgentSessionStore` | Hosting | Never saves: every request starts fresh |
| Your own | | Derive from `AgentSessionStore` |

There is no Cosmos DB session store. To keep long histories out of the session, give the agent a `CosmosChatHistoryProvider` (package `Microsoft.Agents.AI.CosmosNoSql`): the messages then live in Cosmos DB and the session only records where. `Microsoft.Agents.AI.Foundry.Hosting` has its own `FoundryAgentSessionStore`, `InMemoryAgentSessionStore` and `FileSystemAgentSessionStore` for Foundry hosted agents.

The store contract (the three abstract methods, `AgentSessionStoreKey` and its partitions, and the `DeleteSessionAsync` change in 1.26.0) is on the [sessions and memory page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#session-stores-hosting). For hosting, the point is that the storage key must include the key's partitions as well as its `SessionId`, because the isolation key travels as a partition. A store over `IDistributedCache`, for Redis or SQL Server, looks like this:

```csharp
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.Caching.Distributed;

// A session store over IDistributedCache (Redis, SQL Server, ...), so every replica sees every conversation.
public sealed class DistributedCacheSessionStore(IDistributedCache cache, TimeSpan slidingExpiry) : AgentSessionStore
{
    private static string Key(AIAgent agent, AgentSessionStoreKey key) =>
        $"session:{agent.Name}:{string.Join(":", (key.Partitions ?? new Dictionary<string, string>()).OrderBy(p => p.Key).Select(p => $"{p.Key}={p.Value}"))}:{key.SessionId}";

    public override async ValueTask<AgentSession?> GetSessionAsync(AIAgent agent, AgentSessionStoreKey key, CancellationToken cancellationToken = default)
    {
        string? json = await cache.GetStringAsync(Key(agent, key), cancellationToken);
        return json is null ? null : await agent.DeserializeSessionAsync(JsonDocument.Parse(json).RootElement, cancellationToken: cancellationToken);
    }

    public override async ValueTask SaveSessionAsync(AIAgent agent, AgentSessionStoreKey key, AgentSession session, CancellationToken cancellationToken = default)
    {
        JsonElement state = await agent.SerializeSessionAsync(session, cancellationToken: cancellationToken);
        await cache.SetStringAsync(Key(agent, key), state.GetRawText(),
            new DistributedCacheEntryOptions { SlidingExpiration = slidingExpiry }, cancellationToken);
    }

    // Abstract since 1.26.0. Deleting a session that does not exist must not throw.
    public override async ValueTask DeleteSessionAsync(AIAgent agent, AgentSessionStoreKey key, CancellationToken cancellationToken = default) =>
        await cache.RemoveAsync(Key(agent, key), cancellationToken);
}
```

`WithInMemorySessionStore`, `WithSessionStore` and `WithAzureBlobSessionStore` wrap the store in an `IsolationKeyScopedAgentSessionStore` unless you pass `withIsolation: false`. That decorator asks the registered `AgentIsolationKeyProvider` for the caller's key and adds it to every session key, so two users who send the same conversation ID get different sessions; with no provider registered it throws. `UseClaimsBasedAgentIsolation()` (from `Microsoft.Agents.AI.Hosting.AspNetCore`, together with `AddHttpContextAccessor()`) registers a provider that reads a claim from the authenticated user, `sub` by default (`ClaimsIdentityAgentIsolationKeyProviderOptions.ClaimType`).

```csharp
// Program.cs: session stores and per-user isolation for hosted agents.
using Azure.Identity;
using Azure.Storage.Blobs;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.Caching.Distributed;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddChatClient(Models.ChatClient());

// Authenticate callers (here with JWT bearer tokens from your identity provider).
builder.Services.AddAuthentication().AddJwtBearer(options => builder.Configuration.Bind("Jwt", options));
builder.Services.AddAuthorization();
builder.Services.AddDistributedMemoryCache();   // AddStackExchangeRedisCache(...) in production

// Sessions are keyed by the authenticated caller as well as by conversation, so one user
// cannot load another's session by guessing its ID. Uses the "sub" claim by default.
builder.Services.AddHttpContextAccessor();
builder.Services.UseClaimsBasedAgentIsolation();

// Development: in memory.
builder.AddAIAgent("scratch", "You are a scratchpad assistant.").WithInMemorySessionStore();

// Azure Blob Storage, one blob per session (package Microsoft.Agents.AI.Hosting.AzureStorage).
builder.AddAIAgent("support", "You are a support assistant.")
    .WithAzureBlobSessionStore(
        new BlobContainerClient(new Uri("https://contoso.blob.core.windows.net/agent-sessions"), new DefaultAzureCredential()),
        new() { CreateContainerIfNotExists = true, BlobNamePrefix = "support/" });

// Your own store, created from DI.
builder.AddAIAgent("sales", "You are a sales assistant.")
    .WithSessionStore((sp, agentName) => new DistributedCacheSessionStore(sp.GetRequiredService<IDistributedCache>(), TimeSpan.FromDays(7)));

builder.AddOpenAIResponses();
builder.AddOpenAIConversations();

var app = builder.Build();
app.MapOpenAIResponses().RequireAuthorization();
app.MapOpenAIConversations().RequireAuthorization();
app.Run();
```

The OpenAI endpoints also keep responses and conversations in their own storage, which is scoped by the same isolation provider. Requests without a valid token get 401 from `RequireAuthorization()` before any agent code runs.

## Microsoft Foundry hosted agents

`Microsoft.Agents.AI.Foundry.Hosting` runs your agent inside Foundry's agent hosting, behind Foundry's Responses protocol, with Foundry-managed sessions, identity and tracing.

```csharp
// Program.cs: a Foundry hosted agent (package Microsoft.Agents.AI.Foundry.Hosting).
using Azure.AI.AgentServer.Core;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Foundry.Hosting;

var projectEndpoint = new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!);

AIAgent agent = new AIProjectClient(projectEndpoint, new DefaultAzureCredential())
    .AsAIAgent(
        model: Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? "%%model=openai%%",
        instructions: "You are a helpful assistant.",
        name: "my-agent");

var builder = AgentHost.CreateBuilder(args);
builder.Services.AddFoundryResponses(agent);
builder.RegisterProtocol("responses", endpoints => endpoints.MapFoundryResponses());

var app = builder.Build();
app.Run();
```

`AddFoundryResponses(agent, agentSessionStore)` takes an optional session store (`FoundryAgentSessionStore` in Foundry; `InMemoryAgentSessionStore` or `FileSystemAgentSessionStore` locally) and a `FoundryResponsesOptions` callback. `AddFoundryToolboxes(credential, "toolbox-name")` connects Foundry toolboxes as MCP tools. Deployment with the Azure Developer CLI is in [Microsoft's hosted-agent guide](https://learn.microsoft.com/agent-framework/hosting/foundry-hosted-agent); the [production guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_production_guide/#choosing-a-host) compares it with the other hosts.

## Durable agents and workflows on Azure Functions

`Microsoft.Agents.AI.Hosting.AzureFunctions` hosts agents as durable entities and workflows as durable orchestrations. State lives in Durable Functions storage, so a conversation survives restarts and scale-out, and a workflow can wait days for input without holding a worker. The packages now live in the [agent-framework-durable-extension](https://github.com/microsoft/agent-framework-durable-extension) repository and release separately from the core: the current release is 1.16.0-preview, which works with core 1.26.0.

```bash
dotnet add package Microsoft.Agents.AI.Hosting.AzureFunctions --prerelease
dotnet add package Microsoft.Azure.Functions.Worker.Extensions.Http.AspNetCore
dotnet add package Microsoft.Azure.Functions.Worker.Extensions.DurableTask
```

```csharp
// Program.cs: durable agents on Azure Functions (package Microsoft.Agents.AI.Hosting.AzureFunctions).
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AzureFunctions;
using Microsoft.Azure.Functions.Worker.Builder;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Hosting;

AIAgent agent = new AzureOpenAIClient(
        new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!),
        new DefaultAzureCredential())
    .GetChatClient(Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai-small%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You write well-structured documents on a given topic.", name: "DocumentPublisher");

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options => options.AddAIAgent(agent, timeToLive: TimeSpan.FromHours(1)))
    .Build();

app.Run();
```

Each registered agent gets an HTTP trigger; `options.AddAIAgent(agent, enableHttpTrigger: true, enableMcpToolTrigger: true)` also exposes it as an MCP tool, and `timeToLive` expires idle sessions. `AddAIAgentFactory(name, sp => ...)` builds the agent from DI. `ConfigureDurableWorkflows(o => o.AddWorkflow(workflow, enableStatusEndpoint: true))` hosts a workflow as an orchestration. Inside a function, `durableClient.AsDurableAgentProxy(context, "DocumentPublisher")` returns an `AIAgent` that runs the durable agent, and `durableClient.AsWorkflowClient(context)` starts and drives durable workflows. Without Functions, `Microsoft.Agents.AI.DurableTask` provides the same through `services.ConfigureDurableAgents(...)` and `ConfigureDurableWorkflows(...)` on the Durable Task Scheduler. Unlike in-process workflows, durable workflows enforce a superstep limit (`MaxSuperstepsExceededException`). Microsoft's [durable agents documentation](https://learn.microsoft.com/azure/durable-task/sdks/durable-agents-microsoft-agent-framework) and the extension repository's samples cover orchestration patterns, reliable streaming and long-running tools.

## Choosing an endpoint

| You need | Use |
| --- | --- |
| Any OpenAI client, including tools built on the OpenAI SDKs | Responses (agents with tools) or Chat Completions (tool-less agents) |
| Agents on other frameworks or in other organisations to call yours | A2A |
| A CopilotKit or other AG-UI front end | AG-UI |
| To try agents and workflows by hand | DevUI, in development only |
| A bespoke front end or API shape | Your own route with `AIHostAgent` |
| Human approval steps in a web API | `HostedWorkflowState` plus a durable checkpoint store |
| Managed hosting with sessions and identity handled for you | Foundry hosted agents |
| Long-running, restart-proof agents and workflows | Durable agents on Azure Functions or the Durable Task Scheduler |

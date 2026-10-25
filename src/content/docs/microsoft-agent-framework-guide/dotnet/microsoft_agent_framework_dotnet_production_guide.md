---
title: "Microsoft Agent Framework (.NET) — Production Guide"
description: "Running .NET agents and workflows in production: choosing a host, sessions and scale-out, securing the protocol endpoints, durable workflows, observability, identity and secrets, prompt-injection defence, resilience with Microsoft.Extensions.Http.Resilience, evaluation in CI, cost control and a pre-deployment checklist."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Production guide"
  order: 16
---

This page collects what is specific to running `Microsoft.Agents.AI` agents and workflows in production, and links to the pages that cover each piece in depth. Container builds, networking and CI/CD pipelines are standard ASP.NET Core practice and are not repeated; see the [Azure Container Apps documentation](https://learn.microsoft.com/azure/container-apps/) for those.

The C# blocks on this page compile against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%). The resilience settings were checked by creating and calling the client, which is when `Microsoft.Extensions.Http.Resilience` validates them.

## Choosing a host

| Host | Package | What you get | Trade-off |
| --- | --- | --- | --- |
| Your own ASP.NET Core service | `Microsoft.Agents.AI.Hosting` and the protocol packages | `AddAIAgent` registration, session stores, OpenAI, A2A and AG-UI endpoints, full control | You run, scale and secure it |
| Microsoft Foundry hosted agent | `Microsoft.Agents.AI.Foundry.Hosting` (preview) | Your agent behind Foundry's Responses protocol: managed sessions, toolboxes, identity and tracing | Foundry's protocol and deployment model |
| Durable agents on Azure Functions | `Microsoft.Agents.AI.Hosting.AzureFunctions` (preview) | Agents as durable entities, workflows as durable orchestrations; state survives restarts and scale-in | Functions programming model; separate release train (1.16.0-preview) |
| Durable Task Scheduler in any host | `Microsoft.Agents.AI.DurableTask` (preview) | The same durable agents without Functions | A Durable Task Scheduler to run |

Code for each is on the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/): [OpenAI-compatible endpoints](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#openai-compatible-endpoints), [A2A](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#a2a), [AG-UI](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#ag-ui), [Foundry hosted agents](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#microsoft-foundry-hosted-agents) and [durable agents](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#durable-agents-and-workflows-on-azure-functions).

## Sessions and scale-out

An agent is stateless and safe to share; a conversation lives in its `AgentSession`. With more than one replica, sessions must live in a store every replica can reach: `WithAzureBlobSessionStore`, or a store of your own over Redis or SQL (see [session stores and isolation](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#session-stores-and-isolation)). The in-memory store loses every conversation on restart and splits them across replicas.

Keep isolation on. The built-in stores scope every session key by the caller's isolation key, and `UseClaimsBasedAgentIsolation()` takes it from the authenticated user's `sub` claim, so one user cannot load another's conversation by guessing its ID. With isolation on and no provider registered, every request fails with `Agent isolation key is required`, which is the safe failure. Turning isolation off (`withIsolation: false`) is for unauthenticated local demos only.

A custom `AgentSessionStore` must implement `DeleteSessionAsync` from 1.26.0 on (it became abstract again), and deleting a missing session should do nothing. The contract is on the [sessions and memory page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#session-stores-hosting).

## Securing the endpoints

None of the protocol endpoints authenticate on their own. For each one you map:

1. **Authenticate and authorise.** Add your authentication scheme and call `.RequireAuthorization()` on the builder each `Map...` call returns, including `MapOpenAIConversations`. Response and conversation IDs are not secrets.
2. **Keep request settings rejected.** The OpenAI endpoints refuse `temperature`, `instructions`, `tools` and `tool_choice` from the caller by default. Map only the settings you mean to allow with `RunOptionsFactory`, and leave `DangerouslyAllowClientFunctionTools` off unless the callers are trusted: it lets them steer which tools the model picks.
3. **Publish an accurate A2A card.** Clients use its URLs and security schemes; a card that advertises an unauthenticated interface invites unauthenticated calls.
4. **Never ship DevUI.** It lists every agent's instructions, tools and model endpoint. Map it only when `IsDevelopment()` is true; by default it rejects non-loopback requests.
5. **Limit request size and rate** with ASP.NET Core's rate-limiting middleware, per user or tenant, before the agent runs. A single request can trigger many model calls through tools.
6. **Set CORS** explicitly for AG-UI front ends; do not allow any origin.

## Workflows in production

The workflow engine has [no superstep limit](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#executors-that-send-yield-and-loop), and handoff orchestrations have no limit on handoffs per turn. Give every cycle an explicit cap, set `WithMaxRounds` and `WithMaxResets` on Magentic workflows, and put a cancellation timeout on each run.

For any workflow that waits for people or runs for minutes, use a durable checkpoint store and resume by session ID, so a deploy or crash does not lose work. `HostedWorkflowState` does the bookkeeping for request-response hosts:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Agents.AI.Workflows.Checkpointing;

// Workflow checkpoints on shared storage, and a workflow agent that writes to it.
public static class DurableWorkflowSetup
{
    public static void AddDurableReviewWorkflow(this IHostApplicationBuilder builder, string checkpointDirectory)
    {
        // A mounted volume (Azure Files) here; CosmosCheckpointStore or your own JsonCheckpointStore in a database.
        var store = new FileSystemJsonCheckpointStore(new DirectoryInfo(checkpointDirectory));
        CheckpointManager checkpoints = CheckpointManager.CreateJson(store);
        builder.Services.AddSingleton(store);   // disposed with the host

        builder.Services.AddSingleton(new HostedWorkflowState(
            ct => ValueTask.FromResult(BuildReviewWorkflow()),   // a fresh instance per run
            checkpoints));
    }

    public static Workflow BuildReviewWorkflow()
    {
        ExecutorBinding draft = ((Func<string, string>)(topic => $"Draft about {topic}")).BindAsExecutor("draft");
        RequestPort approve = RequestPort.Create<string, bool>("approve");
        ExecutorBinding publish = ((Func<bool, string>)(ok => ok ? "published" : "discarded")).BindAsExecutor("publish");
        return new WorkflowBuilder(draft).AddEdge(draft, approve).AddEdge(approve, publish).WithOutputFrom(publish).Build();
    }
}
```

Build workflows through factories (or `HostedWorkflowState`'s factory constructor) in services that run several sessions at once: one `Workflow` instance can only be run by one run at a time, and executor fields carry over between runs of the same instance. Resume with a workflow of the same shape as the one that wrote the checkpoint; renaming an executor ID or changing a port's types breaks resumption of runs already in flight, so version long-running workflows deliberately. Checkpoints are never pruned automatically; delete them when a run finishes, or with a retention job.

## Observability

Wrap agents with `.AsBuilder().UseOpenTelemetry(sourceName)`, instrument the chat client in DI, call `WithOpenTelemetry()` on workflows, and subscribe to those sources and to `Microsoft.Agents.AI.Workflows` in your tracer and meter providers. Spans follow the GenAI semantic conventions (`invoke_agent`, `chat`, `execute_tool`) and carry token usage; prompt text is excluded unless you set `EnableSensitiveData`. Set-up for OTLP, the Aspire dashboard and Azure Monitor is on the [observability page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/).

## Identity and secrets

Use managed identity (`DefaultAzureCredential` locally, `ManagedIdentityCredential` in Azure) for Azure OpenAI, Foundry, Cosmos DB, Storage and Key Vault, with the least-privileged roles listed on the [Azure RBAC page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_azure_rbac/). Secrets for third-party services go in Key Vault:

```csharp
// Program.cs: secrets from Azure Key Vault (Azure.Extensions.AspNetCore.Configuration.Secrets)
using Azure.Identity;

var builder = WebApplication.CreateBuilder(args);

var keyVaultEndpoint = new Uri(Environment.GetEnvironmentVariable("KEY_VAULT_ENDPOINT")!);
builder.Configuration.AddAzureKeyVault(keyVaultEndpoint, new DefaultAzureCredential());

// Secrets are now ordinary configuration values (e.g. a third-party API key used by a tool).
// Prefer managed identity over keys for Azure OpenAI itself.
var weatherApiKey = builder.Configuration["WeatherApiKey"];

var app = builder.Build();
app.Run();
```

## Prompt-injection defence

There is no information-flow-control layer in .NET (FIDES is Python-only). Combine:

1. **Approval for consequential tools.** Wrap the function in `ApprovalRequiredAIFunction`; the run returns a `ToolApprovalRequestContent` and continues only after you answer it on the same session ([tool approval](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#tool-approval)). In workflows and orchestrations the request surfaces as a `RequestInfoEvent` ([tool approval inside orchestrations](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/#tool-approval-inside-orchestrations)).
2. **Prompt Shields.** Azure AI Content Safety's [Prompt Shields](https://learn.microsoft.com/azure/ai-services/content-safety/concepts/jailbreak-detection) API detects jailbreak and indirect-injection attempts in user input and in documents or tool results. Call it from agent middleware before the model call, in the same way as the text-moderation middleware on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#azure-ai-content-safety-integration).
3. **Purview.** `Microsoft.Agents.AI.Purview` adds `.AsBuilder().WithPurview(credential, settings)` to apply your organisation's DLP policies to prompts and responses.
4. **Least privilege for tools.** Give each tool its own scoped identity or token, and keep untrusted content out of the instructions.
5. **No client-supplied tools** on public endpoints (see [securing the endpoints](#securing-the-endpoints)).

## Resilience

`Microsoft.Extensions.Http.Resilience` (built on Polly v8) adds retries, a circuit breaker and timeouts to an `HttpClient` in one call. To apply it to model calls, give the Azure OpenAI client that `HttpClient` as its transport and turn off the SDK's own retries so the two do not compound.

```csharp
// Program.cs: resilience for the model client (Microsoft.Extensions.Http.Resilience)
using System.ClientModel.Primitives;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var builder = WebApplication.CreateBuilder(args);

// Retries with back-off (including 429s), a circuit breaker and timeouts, on a named HttpClient.
builder.Services.AddHttpClient("openai").AddStandardResilienceHandler(options =>
{
    // Model calls are slow: the defaults (10 s per attempt, 30 s in total) are too short for long answers.
    options.AttemptTimeout.Timeout = TimeSpan.FromSeconds(60);
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(3);
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(2);   // must be at least twice the attempt timeout
});

builder.Services.AddSingleton<AIAgent>(sp =>
{
    HttpClient http = sp.GetRequiredService<IHttpClientFactory>().CreateClient("openai");
    var options = new AzureOpenAIClientOptions
    {
        Transport = new HttpClientPipelineTransport(http),
        RetryPolicy = new ClientRetryPolicy(maxRetries: 0),   // let the resilience handler own retries
    };

    return new AzureOpenAIClient(new Uri(builder.Configuration["AzureOpenAI:Endpoint"]!), new DefaultAzureCredential(), options)
        .GetChatClient("%%model=openai-small%%")
        .AsAIAgent(instructions: "You are a helpful assistant.");
});

var app = builder.Build();
app.Run();
```

The standard handler's defaults (10 seconds per attempt, 30 seconds in total) are meant for ordinary APIs and cut long model answers off mid-stream; raise them as above. The options are validated when the client is first created and used, not at start-up: a total timeout shorter than the attempt timeout, or a circuit-breaker sampling duration shorter than twice the attempt timeout, throws `OptionsValidationException` on that first call. For time limits that also cover tool calls, and a fallback answer, see [bounded agent calls](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-15-bounded-agent-calls). For multi-step work that must not be lost, prefer a durable host or workflow checkpointing over retries.

## Evaluation in CI

Keep the pull-request gate to deterministic tests against a scripted chat client and `LocalEvaluator` checks (keywords, expected tool calls); `AgentEvaluationResults.AssertAllPassed()` turns them into ordinary test failures. Run model-graded `Microsoft.Extensions.AI.Evaluation` evaluators and `dotnet aieval` reports on a schedule against the production deployment. The [testing and evaluation page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_testing_and_evaluation/) has the code.

## Cost control

- Use the small tier (`%%model=openai-small%%`) for routing and classification; reserve the larger model for reasoning.
- Alert on `gen_ai.client.token.usage` per agent and tenant, or read `AgentResponse.Usage` yourself.
- Keep long sessions bounded with a compaction strategy (built in to `HarnessAgent` when you set `MaxContextWindowTokens`).
- Cap tool loops and workflow rounds, as above; a loop that never ends is also a bill that never ends.
- Set per-deployment quotas and Azure budgets as the hard ceiling.

## Before you deploy

- Exercise agents and workflows locally with [DevUI](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#devui), and check it is not mapped in production.
- Run the evaluation suite against the production model deployment.
- Check that sessions survive a restart and a second replica (shared session store or durable host), and that a second user cannot read the first user's session.
- Call every endpoint without a token and confirm a 401.
- Resume a workflow from a checkpoint written by the previous release.
- Confirm traces arrive in your backend with your source names, and that no prompt text is in them.

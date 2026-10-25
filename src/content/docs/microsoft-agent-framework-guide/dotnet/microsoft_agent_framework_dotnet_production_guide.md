---
title: "Microsoft Agent Framework (.NET) — Production Guide"
description: "Hosting .NET agents in production (Foundry hosted agents, Azure Functions durable agents, ASP.NET Core), observability, identity, prompt-injection defence, resilience and evaluation in CI."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Production guide"
  order: 85
---

This page covers what is specific to running `Microsoft.Agents.AI` agents in production. Container builds, networking and CI/CD pipelines are standard ASP.NET Core practice and are not repeated; see the [Azure Container Apps documentation](https://learn.microsoft.com/azure/container-apps/) for those.

## 1. Choosing a host

| Host | Package | What you get |
|---|---|---|
| Microsoft Foundry hosted agent | `Microsoft.Agents.AI.Foundry.Hosting` (preview) | Your agent behind Foundry's Responses protocol: managed sessions (`FoundryAgentSessionStore`), checkpoints, Foundry toolboxes, identity and tracing. |
| Azure Functions durable agent | `Microsoft.Agents.AI.Hosting.AzureFunctions` (preview) | Agents as durable entities and workflows as durable orchestrations; state survives restarts and scale-in, with generated HTTP (and optional MCP) triggers. |
| Durable Task Scheduler in any host | `Microsoft.Agents.AI.DurableTask` (preview) | The same durable agents without Functions (`services.ConfigureDurableAgents(...)`). |
| Your own ASP.NET Core service | `Microsoft.Agents.AI.Hosting` | `AddAIAgent` registration, session stores, and protocol endpoints (`MapOpenAIResponses`, `MapA2AJsonRpc`, `MapAGUIServer`). |

### Foundry hosted agents

```bash
dotnet add package Microsoft.Agents.AI.Foundry.Hosting --prerelease
```

```csharp
using Azure.AI.Projects;
using Azure.AI.AgentServer.Core;
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

`AddFoundryResponses(agent, agentSessionStore)` takes an optional session store (`FoundryAgentSessionStore` in Foundry, `InMemoryAgentSessionStore` or `FileSystemAgentSessionStore` locally) and a `FoundryResponsesOptions` callback (`ResilientBackground`, `SteerableConversations`, ...). `AddFoundryToolboxes(credential, "toolbox-name")` connects Foundry toolboxes as MCP tools. Deployment with the Azure Developer CLI is described in [Microsoft's hosted-agent guide](https://learn.microsoft.com/agent-framework/hosting/foundry-hosted-agent).

### Durable agents on Azure Functions

```bash
dotnet add package Microsoft.Agents.AI.Hosting.AzureFunctions --prerelease
```

```csharp
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
    .ConfigureDurableAgents(options => options.AddAIAgent(agent))
    .Build();

app.Run();
```

Each registered agent gets an HTTP endpoint and its conversation state is kept by Durable Functions, so a session continues across restarts and instances. `options.AddAIAgent(agent, enableHttpTrigger: true, enableMcpToolTrigger: true)` also exposes it as an MCP tool, and `ConfigureDurableWorkflows(o => o.AddWorkflow(workflow))` hosts a workflow as a durable orchestration that can wait for human input. Inside a function, `durableClient.AsDurableAgentProxy(context, "DocumentPublisher")` returns an `AIAgent` that runs the durable agent. See Microsoft's [durable agents documentation](https://learn.microsoft.com/azure/durable-task/sdks/durable-agents-microsoft-agent-framework).

### Your own ASP.NET Core service

Register one `IChatClient` and the agents as singletons, keep conversations in an `AgentSessionStore` that all replicas share, and isolate sessions per user (`UseClaimsBasedAgentIsolation`). The [.NET comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#multi-tenant-isolation) shows the registration, and [Recipe 9](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-9-exposing-an-agent-through-aspnet-core) an HTTP endpoint.

A store of your own derives from `AgentSessionStore` (experimental, `MAAI001`) and overrides `GetSessionAsync`, `SaveSessionAsync` and, since 1.26.0, `DeleteSessionAsync`, which is abstract again: a custom store written for 1.25 no longer compiles until you add it. Deleting a session that does not exist should do nothing, as the built-in stores do.

`MapAGUIServer` (in `Microsoft.Agents.AI.Hosting.AGUI.AspNetCore`) has, since 1.26.0, an overload that picks the agent per request, so one endpoint can serve several registered agents. Returning `null` answers 404 without loading any session state. Sessions are partitioned by the selected agent's `Name`, so it must be non-empty and distinct for agents that must not share sessions:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AGUI.AspNetCore;

// The {agent} route value picks a keyed AIAgent registered with builder.AddAIAgent(...).
app.MapAGUIServer("/agui/{agent}", context =>
    context.Request.RouteValues["agent"] is string name
        ? context.RequestServices.GetKeyedService<AIAgent>(name)
        : null);
```

The overloads that take a hosted agent builder or an agent name now respect the registration's DI lifetime: a singleton agent is resolved once when the endpoint is mapped, scoped and transient agents once per request.

## 2. Observability

Wrap agents with `.AsBuilder().UseOpenTelemetry(sourceName)` (and chat clients with `ChatClientBuilder.UseOpenTelemetry`), then add that source to your OpenTelemetry tracer provider with the Azure Monitor or OTLP exporter. Spans follow the GenAI semantic conventions (`invoke_agent`, `chat`, `execute_tool`) and carry token usage; prompt text is excluded unless you set `EnableSensitiveData`. Workflows have `WorkflowBuilder.WithOpenTelemetry()`. A complete set-up is in the [.NET feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/#opentelemetry-integration---net).

## 3. Identity and secrets

Use managed identity (`DefaultAzureCredential` locally, `ManagedIdentityCredential` in Azure) for Azure OpenAI, Foundry, Cosmos DB and Key Vault, with the least-privileged roles listed on the [Azure RBAC page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_azure_rbac/). Secrets for third-party services go in Key Vault:

```csharp
// Program.cs - Integrating Azure Key Vault (Azure.Extensions.AspNetCore.Configuration.Secrets)
using Azure.Identity;

var builder = WebApplication.CreateBuilder(args);

var keyVaultEndpoint = new Uri(Environment.GetEnvironmentVariable("KEY_VAULT_ENDPOINT")!);
builder.Configuration.AddAzureKeyVault(keyVaultEndpoint, new DefaultAzureCredential());

// Secrets are now ordinary configuration values (e.g. a third-party API key used by a tool).
// Prefer managed identity over keys for Azure OpenAI itself.
var weatherApiKey = builder.Configuration["WeatherApiKey"];
```


## 4. Prompt-injection defence

There is no information-flow-control layer in .NET (FIDES is Python-only). Combine:

1. **Approval for consequential tools.** Wrap the function in `ApprovalRequiredAIFunction`; the run returns a `ToolApprovalRequestContent` and continues only after you answer it on the same session ([tool approval](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#tool-approval)). In workflows, use a request port for the same purpose.
2. **Prompt Shields.** Azure AI Content Safety's [Prompt Shields](https://learn.microsoft.com/azure/ai-services/content-safety/concepts/jailbreak-detection) API detects jailbreak and indirect-injection attempts in user input and in documents or tool results. Call it from agent middleware before the model call, in the same way as the text-moderation middleware on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#azure-ai-content-safety-integration).
3. **Purview.** `Microsoft.Agents.AI.Purview` adds `.AsBuilder().WithPurview(credential, settings)` to apply your organisation's DLP policies to prompts and responses.
4. **Least privilege for tools.** Give each tool its own scoped identity or token, and keep untrusted content out of the instructions.

## 5. Resilience

`Microsoft.Extensions.Http.Resilience` (built on Polly v8) adds retries, a circuit breaker and timeouts to an `HttpClient` in one call. To apply it to model calls, give the Azure OpenAI client that `HttpClient` as its transport and turn off the SDK's own retries so the two don't compound.

```csharp
// Resilience for the model client (Microsoft.Extensions.Http.Resilience)
using System.ClientModel.Primitives;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var builder = WebApplication.CreateBuilder(args);

// Retries with back-off (including 429s), a circuit breaker and timeouts, on a named HttpClient.
builder.Services.AddHttpClient("openai").AddStandardResilienceHandler();

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
```


For multi-step work that must not be lost, prefer a durable host (section 1) or workflow checkpointing over retries.

## 6. Evaluation in CI

`LocalEvaluator` and `agent.EvaluateAsync(...)` are in the core package, and `AgentEvaluationResults.AssertAllPassed()` throws on failure, so a quality gate is an ordinary unit test. Keep the PR gate to deterministic checks (keywords, expected tool calls) and run model-graded `Microsoft.Extensions.AI.Evaluation` or Foundry evaluators on a schedule. Code is in the [.NET comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#evaluation).

## 7. Cost control

- Use the small tier (`%%model=openai-small%%`) for routing and classification; reserve the larger model for reasoning.
- Read `AgentResponse.Usage` (or the span token attributes) and alert per agent and tenant.
- Keep long sessions bounded with a compaction strategy (built in to `HarnessAgent` when you set `MaxContextWindowTokens`).
- Set per-deployment quotas and Azure budgets as the hard ceiling.

## 8. Before you deploy

- Exercise agents and workflows locally with [DevUI](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#local-testing-with-devui) (development only).
- Run the evaluation suite against the production model deployment.
- Check that sessions survive a restart and a second replica (shared session store or durable host).

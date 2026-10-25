---
title: "Microsoft Agent Framework — Enterprise Patterns"
description: "Enterprise patterns for the Microsoft Agent Framework — OpenTelemetry, Entra ID, Azure AI Content Safety, PII redaction, Purview, governance and audit logging — built from the framework's real middleware, client and observability APIs."
framework: "microsoft-agent-framework"
sidebar:
  label: "Enterprise patterns"
  order: 30
---

**Target audience:** enterprise architects, security teams, compliance officers
**Verified against:** `agent-framework` %%version=agent-framework%% (Python) and `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (.NET), checked %%versions-checked%%. First written during the October 2025 public preview.

The framework itself is deliberately small. It gives you four extension points, and every enterprise concern on this page is built from them:

| Extension point | Python | .NET |
| --- | --- | --- |
| Credentials on the chat client | `credential=` on `FoundryChatClient` / `OpenAIChatClient` | `AzureOpenAIClient(uri, TokenCredential)`, `AIProjectClient` |
| Middleware (agent, chat, function) | `AgentMiddleware`, `@agent_middleware`, `@chat_middleware`, `@function_middleware` | `agent.AsBuilder().Use(...)`, `ChatClientBuilder.Use(...)` |
| Context providers | `ContextProvider`, `SecureAgentConfig` | `AIContextProvider`, `ChatHistoryProvider` |
| Telemetry | `agent_framework.observability` | `.UseOpenTelemetry()` on the agent builder |

There is **no** compliance, governance, audit or content-safety module in the framework — no `HIPAAConfig`, `PolicyEngine`, `AuditConfig`, `ContentSafetyConfig` or `Agent(content_safety=...)`. What follows shows how to get each outcome with the APIs that do exist, plus the relevant Azure SDKs. Where the honest answer is "this is platform configuration, not agent code", the page says so.

---

## Table of Contents

1. [OpenTelemetry Instrumentation](#opentelemetry-instrumentation)
2. [Azure AI Content Safety Integration](#azure-ai-content-safety-integration)
3. [Microsoft Entra ID Authentication Patterns](#microsoft-entra-id-authentication-patterns)
4. [Regulated Industry Compliance](#regulated-industry-compliance)
5. [Enterprise Governance](#enterprise-governance)
6. [Advanced Security Features](#advanced-security-features)
7. [Audit and Compliance Logging](#audit-and-compliance-logging)

---

## OpenTelemetry Instrumentation

The framework emits OpenTelemetry spans, metrics and logs that follow the GenAI semantic conventions (`invoke_agent`, `chat`, `execute_tool`, token-usage histograms).

### Automatic instrumentation (Python)

```python
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.observability import configure_otel_providers
from azure.identity.aio import DefaultAzureCredential


def setup_telemetry() -> None:
    # Builds tracer / meter / logger providers with an OTLP exporter.
    # Point it at an OpenTelemetry Collector, which can fan out to
    # Azure Monitor, Datadog, Grafana and so on.
    configure_otel_providers(
        service_name="customer-support-agents",
        service_version="2.3.0",                      # your service's version
        resource_attributes={"deployment.environment": "production"},
        otlp_endpoint="https://otel-collector.contoso.com:4317",
        otlp_headers={"api-key": os.environ["OTEL_API_KEY"]},
        enable_sensitive_data=False,                  # keep prompts and completions out of spans
    )


async def traced_agent_execution() -> None:
    setup_telemetry()

    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(credential=credential)  # FOUNDRY_PROJECT_ENDPOINT / FOUNDRY_MODEL from env
        agent = Agent(client=client, name="support", instructions="You are helpful.")

        # Emits invoke_agent -> chat -> execute_tool spans.
        response = await agent.run("Hello!")
        print(response.text)
```

`configure_otel_providers()` also reads the standard `OTEL_EXPORTER_OTLP_*` environment variables. If your application already owns its OpenTelemetry set-up (for example via `configure_azure_monitor()` below), call `agent_framework.observability.enable_instrumentation()` instead so the framework only emits into your providers. `enable_sensitive_data` defaults to off; leave it off anywhere prompts can contain personal or regulated data. The [Python observability page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_observability/) covers the environment variables and exporters in detail.

### Custom spans and metrics

Your own spans nest naturally around the framework's spans, because both use the active OpenTelemetry context.

```python
import time

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from opentelemetry import metrics, trace

tracer = trace.get_tracer("contoso.support")
meter = metrics.get_meter("contoso.support")

agent_requests = meter.create_counter("agent.requests", unit="1", description="Agent requests")
agent_latency = meter.create_histogram("agent.latency", unit="ms", description="Agent request latency")

agent = Agent(client=OpenAIChatClient(), name="customer_support", instructions="You help customers.")


async def execute_with_custom_telemetry(customer_id: str, question: str) -> str:
    with tracer.start_as_current_span("customer_support_workflow") as span:
        span.set_attribute("workflow.name", "customer_support")
        span.set_attribute("customer.id", customer_id)   # an opaque ID, never a name or email

        start = time.perf_counter()
        response = await agent.run(question)
        elapsed_ms = (time.perf_counter() - start) * 1000

        agent_requests.add(1, {"agent": agent.name})
        agent_latency.record(elapsed_ms, {"agent": agent.name})
        span.set_attribute("response.length", len(response.text))
        return response.text
```

### .NET OpenTelemetry integration

In .NET the agent is wrapped with `UseOpenTelemetry()`; you then subscribe to the same source name in your tracer provider. There is no `Microsoft.Agents.AI.Telemetry` namespace and no `AgentTelemetry` static class.

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Azure.Monitor.OpenTelemetry.Exporter;
using Microsoft.Agents.AI;
using OpenAI.Chat;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

public static class AgentTelemetrySetup
{
    private const string SourceName = "Contoso.Agents";

    public static async Task RunAsync()
    {
        var resource = ResourceBuilder.CreateDefault()
            .AddService(serviceName: "enterprise-agents", serviceVersion: "1.0.0");

        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(resource)
            .AddSource(SourceName)
            .AddOtlpExporter(o => o.Endpoint = new Uri("https://otel-collector.contoso.com:4317"))
            .AddAzureMonitorTraceExporter(o =>
                o.ConnectionString = Environment.GetEnvironmentVariable("APPLICATIONINSIGHTS_CONNECTION_STRING"))
            .Build();

        using var meterProvider = Sdk.CreateMeterProviderBuilder()
            .SetResourceBuilder(resource)
            .AddMeter(SourceName)
            .AddOtlpExporter()
            .Build();

        AIAgent agent = new AzureOpenAIClient(
                new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!),
                new DefaultAzureCredential())
            .GetChatClient("%%model=openai-small%%")
            .AsAIAgent(instructions: "You are a helpful assistant.", name: "support")
            .AsBuilder()
            .UseOpenTelemetry(sourceName: SourceName, configure: o => o.EnableSensitiveData = false)
            .Build();

        AgentResponse result = await agent.RunAsync("Hello!");
        Console.WriteLine(result.Text);
    }
}
```

### Azure Monitor integration

`configure_azure_monitor()` (package `azure-monitor-opentelemetry`) installs the providers; `enable_instrumentation()` turns the framework's spans on inside them.

```python
import os

from agent_framework import Agent
from agent_framework.observability import enable_instrumentation
from agent_framework.openai import OpenAIChatClient
from azure.monitor.opentelemetry import configure_azure_monitor

configure_azure_monitor(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
    enable_live_metrics=True,
)
enable_instrumentation(enable_sensitive_data=False)

agent = Agent(client=OpenAIChatClient(), instructions="You are helpful.")


async def main() -> None:
    response = await agent.run("Query")
    print(response.text)

# In Application Insights: Application Map shows the model endpoint as a
# dependency, Transaction search shows invoke_agent / chat / execute_tool spans,
# and the gen_ai.* metrics carry token usage per model.
```

### Distributed tracing across agents

`agent.run()` has no `trace_context` argument and needs none: within one process, every agent call made inside an active span becomes a child of it.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from opentelemetry import trace

tracer = trace.get_tracer("contoso.research")
client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="You research.")
analyst = Agent(client=client, name="analyst", instructions="You analyse.")


async def multi_agent_traced_workflow() -> str:
    with tracer.start_as_current_span("multi_agent_workflow") as span:
        research = await researcher.run("Research AI trends")
        analysis = await analyst.run(f"Analyse: {research.text}")
        span.set_attribute("agents.count", 2)
        return analysis.text   # both invoke_agent spans share this trace
```

Across process boundaries (an agent behind an HTTP API, or an A2A call) propagation is ordinary W3C `traceparent` handling: instrument your HTTP server and client (`opentelemetry-instrumentation-fastapi`, `-httpx`, or ASP.NET Core / `HttpClient` instrumentation in .NET) and the trace continues. Workflows add their own `workflow.run` and executor spans on top.

---

## Azure AI Content Safety Integration

The framework has no built-in content-safety switch. You call [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/) yourself from agent middleware, which sees both the incoming messages and the final response. (Azure OpenAI and Foundry deployments also apply their own content filters server-side; in Python a filtered request surfaces as `agent_framework.openai.OpenAIContentFilterException`. The middleware below is for your own, stricter or auditable, thresholds.)

### Screening input and output with middleware

```bash
pip install agent-framework azure-ai-contentsafety
```

```python
import os

from agent_framework import Agent, AgentContext, AgentMiddleware, AgentResponse, Content, Message
from agent_framework.openai import OpenAIChatClient
from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.identity.aio import DefaultAzureCredential

# Severity is 0, 2, 4 or 6 with the default output type. Block at or above:
THRESHOLDS = {"Hate": 2, "Sexual": 2, "Violence": 4, "SelfHarm": 2}


class ContentSafetyMiddleware(AgentMiddleware):
    """Screens the user's input before the model sees it, and the model's output before the caller does."""

    def __init__(self, client: ContentSafetyClient, thresholds: dict[str, int] = THRESHOLDS) -> None:
        self.client = client
        self.thresholds = thresholds

    async def violations(self, text: str) -> list[tuple[str, int]]:
        if not text.strip():
            return []
        result = await self.client.analyze_text(AnalyzeTextOptions(text=text[:10_000]))
        return [
            (item.category, item.severity)
            for item in result.categories_analysis
            if item.severity is not None and item.severity >= self.thresholds.get(item.category, 7)
        ]

    @staticmethod
    def blocked(reason: str) -> AgentResponse:
        return AgentResponse(messages=[Message(role="assistant", contents=[Content.from_text(reason)])])

    async def process(self, context: AgentContext, call_next) -> None:
        user_text = "\n".join(m.text or "" for m in context.messages if m.role == "user")
        if hits := await self.violations(user_text):
            context.metadata["content_safety"] = {"stage": "input", "violations": hits}
            context.result = self.blocked("I can't help with that request.")
            return                                   # the model is never called

        await call_next()

        if isinstance(context.result, AgentResponse) and (hits := await self.violations(context.result.text)):
            context.metadata["content_safety"] = {"stage": "output", "violations": hits}
            context.result = self.blocked("The response was withheld by content policy.")


async def main() -> None:
    async with (
        DefaultAzureCredential() as credential,
        ContentSafetyClient(os.environ["CONTENT_SAFETY_ENDPOINT"], credential) as safety,
    ):
        agent = Agent(
            client=OpenAIChatClient(),
            instructions="You are a helpful assistant.",
            middleware=[ContentSafetyMiddleware(safety)],
        )
        print((await agent.run("User query")).text)
```

This version handles non-streaming runs. For `stream=True`, check `context.stream` and either buffer the output before releasing it or screen the input only; the [middleware page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/) shows the streaming hooks.

### Custom actions on a violation

Because the check is your own code, the action on a violation is too: log it, raise, route to a human, or count strikes per user. To make a block an error the caller handles, raise your own exception type — it propagates out of `agent.run()` unchanged. (Don't use `MiddlewareTermination` for this: in agent middleware it never reaches the caller, and `agent.run()` returns whatever is in `context.result`, or `None`.)

```python
import logging

from agent_framework import AgentContext

security_log = logging.getLogger("contoso.security")


class ContentPolicyViolation(Exception):
    pass


class StrictContentSafetyMiddleware(ContentSafetyMiddleware):
    async def process(self, context: AgentContext, call_next) -> None:
        user_text = "\n".join(m.text or "" for m in context.messages if m.role == "user")
        if hits := await self.violations(user_text):
            security_log.warning(
                "content_safety_violation",
                extra={"agent": context.agent.name, "violations": hits},
            )
            raise ContentPolicyViolation(f"content policy violation: {hits}")
        await call_next()


# Caller side:
#   try:
#       response = await agent.run(text)
#   except ContentPolicyViolation as exc:
#       return http_422(str(exc))
```

### .NET content safety integration

In .NET the same idea is a delegating agent built with `AsBuilder().Use(...)`. The `Azure.AI.ContentSafety` package provides the client.

```csharp
using Azure.AI.ContentSafety;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Chat;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

public static class SafeAgentConfiguration
{
    private static readonly Dictionary<TextCategory, int> Thresholds = new()
    {
        [TextCategory.Hate] = 2,
        [TextCategory.Sexual] = 2,
        [TextCategory.Violence] = 4,
        [TextCategory.SelfHarm] = 2,
    };

    public static AIAgent CreateSafeAgent()
    {
        var credential = new DefaultAzureCredential();
        var safety = new ContentSafetyClient(
            new Uri(Environment.GetEnvironmentVariable("CONTENT_SAFETY_ENDPOINT")!), credential);

        AIAgent inner = new AzureOpenAIClient(
                new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!), credential)
            .GetChatClient("%%model=openai-small%%")
            .AsAIAgent(instructions: "You are a helpful assistant.");

        return inner.AsBuilder()
            .Use(
                runFunc: async (messages, session, options, innerAgent, cancellationToken) =>
                {
                    var input = string.Join("\n", messages.Where(m => m.Role == ChatRole.User).Select(m => m.Text));
                    if (await IsHarmfulAsync(safety, input, cancellationToken))
                    {
                        return new AgentResponse(new ChatMessage(ChatRole.Assistant, "I can't help with that request."));
                    }

                    var response = await innerAgent.RunAsync(messages, session, options, cancellationToken);
                    return await IsHarmfulAsync(safety, response.Text, cancellationToken)
                        ? new AgentResponse(new ChatMessage(ChatRole.Assistant, "The response was withheld by content policy."))
                        : response;
                },
                runStreamingFunc: null)
            .Build();
    }

    private static async Task<bool> IsHarmfulAsync(ContentSafetyClient safety, string text, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        var result = await safety.AnalyzeTextAsync(new AnalyzeTextOptions(text), ct);
        return result.Value.CategoriesAnalysis.Any(c =>
            c.Severity is int severity && Thresholds.TryGetValue(c.Category, out var max) && severity >= max);
    }
}
```

Passing `runStreamingFunc: null` makes streaming calls go through the non-streaming path, so output screening still applies.

### PII detection and redaction

There is no PII detector in the framework. Use [Azure AI Language PII detection](https://learn.microsoft.com/azure/ai-services/language-service/personally-identifiable-information/overview) (`azure-ai-textanalytics`) or an open-source recogniser such as Presidio, again from middleware. Redacting the *input* is the more useful half: the model, its logs and any stored history never see the raw values.

```python
import os

from agent_framework import Agent, AgentContext, AgentMiddleware, Content, Message
from agent_framework.openai import OpenAIChatClient
from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.identity.aio import DefaultAzureCredential


class PiiRedactionMiddleware(AgentMiddleware):
    def __init__(self, client: TextAnalyticsClient, categories: list[str] | None = None) -> None:
        self.client = client
        self.categories = categories   # e.g. ["Email", "USSocialSecurityNumber", "CreditCardNumber"]

    async def process(self, context: AgentContext, call_next) -> None:
        redacted: list[Message] = []
        for message in context.messages:
            if message.role == "user" and message.text:
                [doc] = await self.client.recognize_pii_entities(
                    [message.text], categories_filter=self.categories, language="en"
                )
                if not doc.is_error and doc.entities:
                    message = Message(role="user", contents=[Content.from_text(doc.redacted_text)])
            redacted.append(message)
        context.messages[:] = redacted
        await call_next()


async def main() -> None:
    async with (
        DefaultAzureCredential() as credential,
        TextAnalyticsClient(os.environ["LANGUAGE_ENDPOINT"], credential) as language,
    ):
        agent = Agent(
            client=OpenAIChatClient(),
            instructions="You are a helpful assistant.",
            middleware=[PiiRedactionMiddleware(language)],
        )
        response = await agent.run("My email is user@example.com and my SSN is 859-98-0987")
        # The model receives: "My email is ******************** and my SSN is ***********"
        print(response.text)
```

---

## Microsoft Entra ID Authentication Patterns

Authentication lives on the **chat client**, not on `Agent`. Every Azure-backed client takes a `credential=` (an `azure.identity` credential or a bearer-token provider); in .NET you pass a `TokenCredential` to the Azure SDK client the agent is built from.

### Managed identity for production

```python
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import ManagedIdentityCredential


async def production_agent_with_managed_identity() -> str:
    # System-assigned identity; pass client_id="..." for a user-assigned one.
    async with ManagedIdentityCredential() as credential:
        client = FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["FOUNDRY_MODEL"],
            credential=credential,
        )
        agent = Agent(client=client, instructions="Production agent")
        return (await agent.run("Query")).text

# Grant the identity data-plane access, for example:
#   az role assignment create --assignee <principal-id> \
#     --role "53ca6127-db72-4b80-b1b0-d745d6d5456d" --scope <foundry-project-resource-id>   # Foundry User (formerly Azure AI User)
# or, for a plain Azure OpenAI resource, "Cognitive Services OpenAI User".
```

For an Azure OpenAI resource without a Foundry project, use `OpenAIChatClient(model="<deployment>", azure_endpoint="https://<resource>.openai.azure.com", credential=credential)`.

### .NET managed identity

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

public static class ProductionAgent
{
    public static async Task<string> ExecuteWithManagedIdentity()
    {
        var credential = new ManagedIdentityCredential(ManagedIdentityId.SystemAssigned);

        AIAgent agent = new AzureOpenAIClient(
                new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!), credential)
            .GetChatClient(Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai-small%%")
            .AsAIAgent(instructions: "Production agent");

        AgentResponse result = await agent.RunAsync("Query");
        return result.Text;
    }
}
```

### Workload identity for Kubernetes

On AKS with [workload identity](https://learn.microsoft.com/azure/aks/workload-identity-overview), `DefaultAzureCredential` (or `WorkloadIdentityCredential` if you want to be explicit) picks up the federated token that the webhook mounts into the pod. The agent code does not change.

```python
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import WorkloadIdentityCredential


async def kubernetes_agent_with_workload_identity() -> str:
    # Reads AZURE_CLIENT_ID, AZURE_TENANT_ID and AZURE_FEDERATED_TOKEN_FILE from the pod.
    async with WorkloadIdentityCredential() as credential:
        agent = Agent(client=FoundryChatClient(credential=credential), instructions="Agent in AKS")
        return (await agent.run("Query")).text
```

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-service-account
  namespace: agents
  annotations:
    azure.workload.identity/client-id: <managed-identity-client-id>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
  namespace: agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
        azure.workload.identity/use: "true"
    spec:
      serviceAccountName: agent-service-account
      containers:
        - name: agent
          image: contoso.azurecr.io/agent:latest
          env:
            - name: FOUNDRY_PROJECT_ENDPOINT
              value: https://contoso.services.ai.azure.com/api/projects/support
            - name: FOUNDRY_MODEL
              value: %%model=openai-small%%
```

### User-delegated authentication

For desktop tools or CLIs where the model call should run as the signed-in user (so Azure RBAC and audit logs reflect the person, not a service), use an interactive credential.

```python
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import DeviceCodeCredential   # sync: azure.identity.aio has no device-code credential


async def user_delegated_agent() -> str:
    # InteractiveBrowserCredential (also sync-only) works similarly on a desktop.
    credential = DeviceCodeCredential(tenant_id="<tenant-id>")
    agent = Agent(client=FoundryChatClient(credential=credential), instructions="User-scoped agent")
    return (await agent.run("Summarise my project's open issues")).text
```

For a web API acting on behalf of a signed-in user against *downstream* resources (Graph, SharePoint), use the on-behalf-of flow (`OnBehalfOfCredential`) inside the tools that call those resources; the model call itself normally stays on the service's managed identity.

### Multi-tenant authentication

There is no multi-tenant auth object in the framework. In a multi-tenant SaaS you typically keep one credential per customer tenant and build (and cache) one client per tenant:

```python
import os
from functools import cache

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import ClientSecretCredential


@cache
def client_for_tenant(tenant_id: str, endpoint: str) -> OpenAIChatClient:
    """One credential and client per customer tenant; created once, then reused."""
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=os.environ["MULTITENANT_APP_CLIENT_ID"],
        client_secret=os.environ["MULTITENANT_APP_SECRET"],   # prefer a certificate or federated credential
    )
    return OpenAIChatClient(model="%%model=openai-small%%", azure_endpoint=endpoint, credential=credential)


async def run_for_tenant(tenant_id: str, endpoint: str, question: str) -> str:
    agent = Agent(client=client_for_tenant(tenant_id, endpoint), instructions="Tenant-scoped assistant")
    return (await agent.run(question)).text
```

On ASP.NET Core, the hosting package can also isolate stored sessions per caller: `builder.Services.UseClaimsBasedAgentIsolation()` (from `Microsoft.Agents.AI.Hosting.AspNetCore`) keys session storage on a claim of the authenticated user, so one user cannot resume another user's conversation.

---

## Regulated Industry Compliance

No agent framework makes an application HIPAA-, GDPR-, SOX- or FedRAMP-compliant, and this one has no `agent_framework.compliance` module. Compliance comes from the Azure services you choose, how you configure them, and the controls in your own code. The table maps common requirements to the real knobs.

| Requirement | Where it is met | Framework knob, if any |
| --- | --- | --- |
| Data processed under a BAA / in-scope services | Azure OpenAI / Foundry, Cosmos DB, Key Vault are HIPAA-eligible under Microsoft's BAA | none — choose in-scope services |
| Data residency | Deploy models and stores in the required region (or a Data Zone deployment) | the endpoint you give the client |
| No prompts in telemetry | Telemetry configuration | `enable_sensitive_data=False` (default) / `EnableSensitiveData = false` |
| No server-side conversation storage | Use Chat Completions, or Responses with storage off | `default_options={"store": False}` |
| PII minimisation | Redact before the model sees it | agent middleware (above) |
| DLP and sensitivity labels | Microsoft Purview | `PurviewPolicyMiddleware` (Python), `.WithPurview(...)` (.NET) |
| Human approval for consequential actions | Your workflow | `@tool(approval_mode="always_require")` |
| Audit trail | Log Analytics / Sentinel / immutable storage | agent and function middleware (below) |
| Retention and erasure | The store that holds sessions | your `HistoryProvider` / session store |

### Microsoft Purview policies (DLP)

Purview integration is real and shipped (`agent-framework-purview`, re-exported from `agent_framework.microsoft`). The middleware sends prompts and responses to Purview for policy evaluation, blocks what your Purview DLP policies block, and records the interaction for audit and eDiscovery.

```python
from agent_framework import Agent
from agent_framework.microsoft import PurviewPolicyMiddleware, PurviewSettings
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential

credential = DefaultAzureCredential()

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a clinical documentation assistant.",
    middleware=[
        PurviewPolicyMiddleware(
            credential=credential,
            settings=PurviewSettings(
                app_name="Clinical Notes Assistant",
                blocked_prompt_message="This request was blocked by your organisation's data policy.",
            ),
        )
    ],
)
```

`PurviewChatPolicyMiddleware` is the chat-client-level equivalent. In .NET, call `.WithPurview(credential, new PurviewSettings("app name"))` on the agent's `AIAgentBuilder` (package `Microsoft.Agents.AI.Purview`, currently a release candidate). The app needs Microsoft Graph permissions for Purview's `processContent` API; see the [Purview developer docs](https://learn.microsoft.com/purview/developer/).

### HIPAA-style controls in code

The code-level part of a HIPAA programme is mostly the controls already shown: managed identity, PHI redaction or a BAA-covered model, no PHI in telemetry, and an audit trail. Keep the model provider's server-side storage off if you do not need it:

```python
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import ManagedIdentityCredential

phi_agent = Agent(
    client=FoundryChatClient(credential=ManagedIdentityCredential()),
    instructions="You draft discharge summaries from the notes provided.",
    default_options={"store": False},       # don't persist the conversation service-side
    middleware=[audit_middleware],          # defined in "Audit and Compliance Logging"
)
```

### GDPR

Data residency is a deployment decision (an EU region or the EU Data Zone). The right to erasure applies to wherever **you** store conversation history: if you persist sessions with a history provider (Cosmos DB, Redis, your own `HistoryProvider`), give each session a key you can look up by data subject and delete it on request. Transient, unpersisted sessions (`agent.create_session()` kept in memory) leave nothing to erase once the request ends.

### Financial services (SOX, PCI-DSS)

Two framework features matter here. Tool approval stops an agent from executing a consequential action until a human approves it, and function middleware gives you a tamper-evident record of every tool call.

```python
from typing import Annotated

from agent_framework import tool


@tool(approval_mode="always_require")
async def transfer_funds(
    from_account: Annotated[str, "Source account ID"],
    to_account: Annotated[str, "Destination account ID"],
    amount: Annotated[float, "Amount in GBP"],
) -> str:
    """Move money between accounts. Never runs without explicit human approval."""
    ...
```

The run then returns approval requests instead of executing the tool; see the [human-in-the-loop page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/) for the approve/reject round trip. For PCI-DSS, keep card data out of the conversation altogether (tokenise upstream, redact in middleware) so the agent stack stays out of the cardholder-data environment.

### Government cloud (FedRAMP)

FedRAMP High and DoD workloads run against Azure Government endpoints. The only code difference is the endpoint, the authority and the token scope, which you supply through a bearer-token provider:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureAuthorityHosts
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

credential = DefaultAzureCredential(authority=AzureAuthorityHosts.AZURE_GOVERNMENT)
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.us/.default")

gov_agent = Agent(
    client=OpenAIChatClient(
        model="%%model=openai%%",
        azure_endpoint="https://contoso-gov.openai.azure.us",
        credential=token_provider,
    ),
    instructions="Government assistant",
)
```

Check which models and features are available in Azure Government before you design around them; the list lags the commercial cloud.

---

## Enterprise Governance

There is no `agent_framework.governance` module. Governance policies are ordinary code in middleware, which is the right place for them: they run on every call, they are testable, and they cannot be bypassed by a prompt.

### Policy enforcement (tool allow-lists by role)

Function middleware runs around every tool call. This one enforces a role-to-tool allow-list for the caller of the current run.

```python
from agent_framework import FunctionInvocationContext, FunctionMiddleware

ROLE_TOOLS = {
    "analyst": {"search_documents", "summarise"},
    "manager": {"search_documents", "summarise", "approve_expense"},
}


class ToolPolicyMiddleware(FunctionMiddleware):
    def __init__(self, role: str) -> None:
        self.allowed = ROLE_TOOLS.get(role, set())

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        if context.function.name not in self.allowed:
            # The model receives this as the tool result and can explain it to the user.
            context.result = f"Policy: tool '{context.function.name}' is not permitted for this user."
            return
        await call_next()


# Per request, with the role taken from the caller's validated token:
#   await agent.run(question, middleware=[ToolPolicyMiddleware(role=claims["role"])])
```

Passing middleware to `agent.run(...)` scopes it to that one call, which is how you inject per-user policy into a shared agent instance.

### Cost management and budgets

Every chat response carries token usage. A chat middleware can meter it per tenant and refuse calls once a budget is spent. Persist the counters somewhere shared (Redis, a database) in a multi-instance deployment; the dictionary below is for illustration.

```python
from collections import defaultdict

from agent_framework import ChatContext, ChatMiddleware, ChatResponse


class BudgetExceeded(Exception):
    pass


class TokenBudgetMiddleware(ChatMiddleware):
    def __init__(self, tenant_id: str, monthly_token_budget: int, usage: dict[str, int]) -> None:
        self.tenant_id = tenant_id
        self.budget = monthly_token_budget
        self.usage = usage

    async def process(self, context: ChatContext, call_next) -> None:
        if self.usage[self.tenant_id] >= self.budget:
            raise BudgetExceeded(f"tenant {self.tenant_id} has exhausted its monthly token budget")
        await call_next()
        if isinstance(context.result, ChatResponse) and context.result.usage_details:
            self.usage[self.tenant_id] += context.result.usage_details.get("total_token_count") or 0


usage_store: dict[str, int] = defaultdict(int)
# await agent.run(question, middleware=[TokenBudgetMiddleware("tenant-42", 2_000_000, usage_store)])
```

Hard spending limits belong in Azure as well: per-deployment tokens-per-minute quotas, Azure API Management token-limit policies in front of the model, and Cost Management budgets with alerts.

### Model governance

Decide centrally which deployments may be used, and make that the only way application code gets a client:

```python
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import ManagedIdentityCredential

APPROVED_MODELS = {
    "general": "%%model=openai-small%%",
    "reasoning": "%%model=openai-small%%",
}

_credential = ManagedIdentityCredential()


def governed_client(purpose: str) -> FoundryChatClient:
    """The only sanctioned way to obtain a model client in this codebase."""
    try:
        model = APPROVED_MODELS[purpose]
    except KeyError:
        raise ValueError(f"No approved model for purpose '{purpose}'") from None
    return FoundryChatClient(model=model, credential=_credential)
```

Back it with Azure Policy on the Foundry or Azure OpenAI resource (allowed model deployments, disabling key-based auth) so the rule holds outside your code too.

---

## Advanced Security Features

### Encryption

Encryption is a platform property, not an agent setting. Traffic to Azure OpenAI, Foundry, Cosmos DB and the rest is TLS 1.2+; data at rest is encrypted by default, and you can switch the resource to customer-managed keys in Key Vault where policy requires it. The framework does not add or need an encryption layer. If you persist sessions yourself (`AgentSession.to_dict()` / `AgentSession.from_dict()` in Python, `agent.SerializeSessionAsync(...)` in .NET), store them in a service that is encrypted at rest — or encrypt the blob before you write it.

### Network security

Lock the model and data services down with private endpoints and disable public network access on them; run the agent inside the same virtual network. If outbound traffic must go through a proxy, give the client an HTTP client that uses it:

```python
import httpx
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import ManagedIdentityCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI

http_client = httpx.AsyncClient(proxy="http://egress-proxy.internal:3128", timeout=60)

client = OpenAIChatClient(
    model="%%model=openai-small%%",
    async_client=AsyncAzureOpenAI(
        azure_endpoint="https://contoso-private.openai.azure.com",
        azure_ad_token_provider=get_bearer_token_provider(
            ManagedIdentityCredential(), "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2024-10-21",
        http_client=http_client,
    ),
)
```

Tools that call the internet are the other egress path; give them their own allow-listed HTTP client.

### Zero trust and prompt-injection defence

Zero trust for agents means: no keys (managed identity everywhere), least-privilege role assignments per agent, verified caller identity on every request, and treating tool output as untrusted input. The last of these is where the framework does ship something. `agent_framework.security` contains the experimental FIDES defences: `SecureAgentConfig` is a context provider that labels tool results by integrity and confidentiality, hides untrusted content from the planning model, and blocks (or asks for approval on) tool calls that would act on it.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import SecureAgentConfig

security = SecureAgentConfig(
    allow_untrusted_tools={"fetch_web_page"},   # may still run once untrusted content is in context
    block_on_violation=True,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You research topics using the tools provided.",
    tools=[fetch_web_page, send_email],
    context_providers=[security],
)
```

Tool results are untrusted unless a tool declares otherwise (`@tool(additional_properties={"source_integrity": "trusted"})`), so here `send_email` is blocked once fetched web content is in context. It is marked experimental and may change between releases; the [Python comprehensive guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#prompt-injection-defence-fides) has the options. .NET has no equivalent; use tool approval and Prompt Shields there.

---

## Audit and Compliance Logging

There is no `agent_framework.audit` module. An audit trail is two small middlewares — one around each agent run, one around each tool call — writing structured records to a logger. Ship that logger wherever your audit records must live.

### Comprehensive audit logging

```python
import hashlib
import json
import logging
import time
import uuid

from agent_framework import (
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    FunctionInvocationContext,
    FunctionMiddleware,
)

audit_log = logging.getLogger("contoso.audit")


def fingerprint(text: str) -> str:
    """Record a hash, not the content, so the audit log holds no personal data."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AgentAuditMiddleware(AgentMiddleware):
    def __init__(self, user_id: str, tenant_id: str) -> None:
        self.user_id = user_id
        self.tenant_id = tenant_id

    async def process(self, context: AgentContext, call_next) -> None:
        record = {
            "event": "agent_run",
            "run_id": str(uuid.uuid4()),
            "agent": context.agent.name,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "input_sha": fingerprint("\n".join(m.text or "" for m in context.messages)),
        }
        context.metadata["audit_run_id"] = record["run_id"]
        start = time.perf_counter()
        try:
            await call_next()
            record["outcome"] = "success"
            if isinstance(context.result, AgentResponse):
                record["output_sha"] = fingerprint(context.result.text)
                record["usage"] = dict(context.result.usage_details or {})
        except Exception as exc:
            record["outcome"] = f"error:{type(exc).__name__}"
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - start) * 1000)
            audit_log.info(json.dumps(record))


class ToolAuditMiddleware(FunctionMiddleware):
    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        audit_log.info(json.dumps({
            "event": "tool_call",
            "tool": context.function.name,
            "arguments_sha": fingerprint(str(context.arguments)),
        }))
        await call_next()


audit_middleware = ToolAuditMiddleware()

# Agent-wide tool auditing, plus per-request run auditing with the caller's identity:
#   agent = Agent(client=..., tools=[...], middleware=[audit_middleware])
#   await agent.run(question, middleware=[AgentAuditMiddleware(user_id=claims["oid"], tenant_id=claims["tid"])])
```

Decide deliberately whether the audit log holds content or only fingerprints. Regulators often want the full exchange retained; privacy teams usually do not want it in a general-purpose log store. A common split is fingerprints in the operational log and full transcripts in a separate, access-controlled, immutable store (Azure Storage with immutability policies).

### SIEM integration (Microsoft Sentinel)

Use the logging pipeline you already have. With `configure_otel_providers()` or `configure_azure_monitor()` in place, Python `logging` records are exported as OpenTelemetry logs; in Application Insights they land in the `traces` table of the Log Analytics workspace, which Sentinel can query directly.

```python
import logging

from agent_framework.observability import configure_otel_providers

configure_otel_providers(service_name="agent-api", otlp_endpoint="https://otel-collector.contoso.com:4317")
logging.getLogger("contoso.audit").setLevel(logging.INFO)

# A Sentinel analytics rule over the exported records might look like:
#   AppTraces
#   | where Properties["logger_name"] == "contoso.audit"
#   | extend r = parse_json(Message)
#   | where r.event == "tool_call" and r.tool == "transfer_funds"
#   | summarize count() by tostring(r.tool), bin(TimeGenerated, 5m)
```

The content-safety middleware's `security_log.warning(...)` records travel the same route, so policy violations and tool use end up queryable side by side.

---

## Best Practices

### 1. Observability
- Configure OpenTelemetry at start-up and keep `enable_sensitive_data` off in production.
- Name agents (`name=`) so spans and metrics are attributable.
- Alert on token usage per tenant and on tool-call error rates.

### 2. Content safety
- Rely on the model deployment's content filter as the baseline; add Content Safety middleware where you need your own thresholds or an auditable decision.
- Redact PII on input, before it reaches the model, history or logs.

### 3. Authentication
- Managed identity (or workload identity) in production; no API keys in configuration.
- One identity per agent service, with the narrowest role that works.
- Disable key-based auth on the Azure OpenAI / Foundry resource once nothing uses it.

### 4. Compliance
- Map each requirement to a service setting or a middleware, and test the middleware.
- Use Purview where your organisation already runs DLP policies.
- Require approval for consequential tools.

### 5. Security
- Private endpoints for model and data services.
- Treat tool output as untrusted; consider `SecureAgentConfig` for agents that read external content.
- Review the audit log routinely, not only after an incident.

---

## Migration Checklist

- [ ] Configure OpenTelemetry (`configure_otel_providers` / `UseOpenTelemetry`)
- [ ] Route telemetry to Azure Monitor or your collector
- [ ] Add Content Safety middleware to user-facing agents where required
- [ ] Add PII redaction middleware
- [ ] Move model clients to managed identity
- [ ] Turn on Purview policy middleware if DLP applies
- [ ] Mark consequential tools `approval_mode="always_require"`
- [ ] Add agent and tool audit middleware; ship the audit logger to Sentinel
- [ ] Add per-tenant budget middleware and Azure-side quotas
- [ ] Centralise model client creation on approved deployments
- [ ] Private endpoints and disabled public access on dependent services
- [ ] Document the security architecture and review it

---

## Resources

### Documentation
- [Agent Framework observability](https://learn.microsoft.com/agent-framework/user-guide/observability)
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/azure/azure-monitor/app/opentelemetry-overview)
- [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/)
- [Azure AI Language PII detection](https://learn.microsoft.com/azure/ai-services/language-service/personally-identifiable-information/overview)
- [Microsoft Entra ID](https://learn.microsoft.com/entra/identity/)
- [Azure compliance offerings](https://learn.microsoft.com/azure/compliance/)

### Tools
- [Azure Policy](https://learn.microsoft.com/azure/governance/policy/)
- [Microsoft Defender for Cloud](https://learn.microsoft.com/azure/defender-for-cloud/)
- [Microsoft Sentinel](https://learn.microsoft.com/azure/sentinel/)
- [Microsoft Purview](https://learn.microsoft.com/purview/)

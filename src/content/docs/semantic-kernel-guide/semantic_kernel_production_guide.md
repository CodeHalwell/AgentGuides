---
title: "Semantic Kernel: Production Deployment and Operations Guide"
description: "Language-neutral production guidance for Semantic Kernel — deployment shape, configuration and secrets, resilience, observability, security, scaling, cost and testing."
framework: "semantic-kernel"
---

> Applies to `semantic-kernel` %%version=semantic-kernel%% (Python) and `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%% (.NET), checked %%versions-checked%%. Runnable, language-specific versions of these patterns are in the [Python production guide](/semantic-kernel-guide/python/semantic_kernel_production_python/) and the [.NET production guide](/semantic-kernel-guide/dotnet/semantic_kernel_production_dotnet/).

Semantic Kernel is in long-term support while its successor, [Microsoft Agent Framework](/microsoft-agent-framework-guide/), receives most new features. Everything below applies equally to both, which keeps a later migration cheap.

---

## Table of contents

1. [Deployment shape](#1-deployment-shape)
2. [Configuration and secrets](#2-configuration-and-secrets)
3. [Resilience](#3-resilience)
4. [Observability](#4-observability)
5. [Security](#5-security)
6. [Scaling and state](#6-scaling-and-state)
7. [Performance and caching](#7-performance-and-caching)
8. [Cost management](#8-cost-management)
9. [Testing](#9-testing)
10. [Release checklist](#10-release-checklist)

---

## 1. Deployment shape

- **Stateless service, stateful stores.** Run SK inside an ordinary web or worker process (FastAPI, ASP.NET Core, Azure Functions, a queue consumer). Keep conversation history, vector data and caches in external stores so any replica can serve any request.
- **Build the kernel per request (or per tenant).** AI service clients are expensive and thread-safe — create them once. Kernels are cheap; creating one per request stops plugins and filters added for one user leaking to another. In .NET register `Kernel` as transient; in Python build it in a dependency or factory.
- **Long-running agent work goes to a queue.** Multi-agent orchestrations can take minutes. Accept the request, enqueue it, process it in a worker and let the client poll or subscribe (SSE/WebSocket) for the result.
- **Containers.** Use slim official base images (`python:3.12-slim`, `mcr.microsoft.com/dotnet/aspnet:10.0`), run as a non-root user, and expose a health endpoint that does *not* call the model (a readiness check may verify configuration and store connectivity).

---

## 2. Configuration and secrets

| Concern | Recommendation |
|---|---|
| Azure OpenAI auth | Managed identity / Entra ID (`DefaultAzureCredential`) rather than API keys |
| Other provider keys | A secret store (Azure Key Vault, AWS Secrets Manager, Kubernetes secrets) injected as environment variables |
| Model and deployment names | Configuration, not code — they change more often than the code does |
| Prompt templates | Versioned files (YAML/`skprompt.txt`) reviewed like code |
| Per-environment differences | Settings objects (`OpenAISettings`, `AzureOpenAISettings` in Python; `IConfiguration` in .NET) |

Never log API keys, and never return raw exception text to clients — connector errors can include request details.

---

## 3. Resilience

- **Know what already retries.** The OpenAI and Azure OpenAI connectors use the official OpenAI SDKs, which retry 429 and 5xx responses with back-off. Adding a second retry layer multiplies attempts; if you add one, keep it small and limited to service errors (`ServiceResponseException` in Python, `HttpOperationException` with a 429/5xx `StatusCode` in .NET).
- **Timeouts everywhere.** Model calls, tool calls and orchestration results (`result.get(timeout=...)` / `GetValueAsync(timeout)`) all need explicit timeouts.
- **Circuit breakers** around flaky tools and secondary providers, so one failing dependency does not tie up every worker.
- **Fallback models.** Register a second chat service and fall back to it on repeated failure; record which model answered.
- **Idempotent tools.** The model may call a tool twice, and retries replay turns. Tools with side effects need idempotency keys.
- **Loop limits.** Cap automatic function-calling iterations and orchestration rounds, and use an auto-function-invocation filter to terminate runaway tool loops.

---

## 4. Observability

Semantic Kernel emits OpenTelemetry traces, metrics and logs:

- **Traces** for function invocations and, with GenAI diagnostics enabled, model calls using the GenAI semantic conventions (model, token counts, finish reason).
- **Metrics** such as `semantic_kernel.function.invocation.duration` and `semantic_kernel.function.invocation.token_usage.prompt` / `.completion`.
- **Logs** through the standard logging frameworks (`logging` / `ILogger`).

Enable GenAI diagnostics with `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true` (Python) or the `Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics` `AppContext` switch (.NET). The *sensitive* variants also capture prompts and completions; use them only in development.

Export with OTLP to any back end, or with the Azure Monitor distro to Application Insights. Alert on: error rate per function, p95 latency per model, token usage per tenant (sudden growth usually means a prompt or loop regression), and tool-call failure rate.

---

## 5. Security

- **Prompt injection.** Treat retrieved documents, tool output and user input as untrusted. Use a prompt-render filter to apply checks, keep system instructions separate from data, and never let model output choose arbitrary URLs, file paths or SQL.
- **Least-privilege tools.** Give each agent only the plugins it needs (restrict with function-choice filters), use read-only credentials for read-only tools, and gate destructive tools with an approval filter.
- **Data handling.** Redact personal data before it reaches prompts or logs (filters again), and check the data-residency settings of each model provider and vector store.
- **MCP servers** run code with your credentials — pin their versions, run local ones in containers, and review the tools they expose.
- **Dependencies.** Pin SK exactly, update deliberately, and watch both SDKs' release notes; several .NET packages are still preview or alpha.

---

## 6. Scaling and state

| State | Where to keep it |
|---|---|
| Chat history / agent threads | A database or cache keyed by session; reload into a `ChatHistory` / thread per request. Service-backed agents (Azure AI, OpenAI Responses) keep threads server-side. |
| Long-term knowledge | A vector store collection |
| Orchestration progress | Persist inputs and outputs per step if a run must survive a restart (SK orchestrations are in-process); for durable multi-step processes prefer Agent Framework workflows |
| Caches | Redis or equivalent shared cache |

Scale horizontally on CPU and request concurrency; model latency, not CPU, is usually the bottleneck, so watch queue depth and provider rate limits (tokens per minute) before adding replicas. Spread load across several deployments or regions when a single quota is the limit.

---

## 7. Performance and caching

- **Stream** responses to users; perceived latency matters more than total time.
- **Parallelise** independent calls (`asyncio.gather` / `Task.WhenAll`), and let models emit parallel tool calls.
- **Cache deterministic work** — classification, extraction, embeddings of unchanged content — keyed by a hash of prompt template, arguments and model. Exact-match caches are simple and safe; semantic caches need a similarity threshold and an eviction policy.
- **Trim context.** Use chat-history reducers (truncation or summarisation), retrieve fewer but better chunks, and keep plugin descriptions short — every advertised function costs prompt tokens on every call.
- **Use the smallest model that works** per task, routing by `service_id` / `ServiceId`.

---

## 8. Cost management

- Track token usage per function, tenant and model from SK's metrics, and convert to cost using your provider's *current* price list held in configuration (prices change too often to hard-code).
- Set budgets per tenant and refuse or degrade (smaller model, shorter answers) when exceeded.
- Watch for silent multipliers: retries, multi-agent rounds, large tool schemas, and verbose system prompts repeated on every turn.

---

## 9. Testing

| Level | Approach |
|---|---|
| Unit | Test native plugin functions as plain code. Test prompt functions and agents against a stub chat service (Python: subclass `ChatCompletionClientBase`; .NET: mock `IChatCompletionService`). |
| Filters and routing | Call filters directly with a constructed context; routers and group-chat managers are deterministic and easy to unit test. |
| Integration | Run against a real model in CI on a small, fixed set of cases; assert on structure (valid JSON, required fields, tool called) rather than exact wording. |
| Evaluation | Keep a golden set of questions with reference answers; score relevance, groundedness and safety on each prompt or model change. |
| Load | Exercise the real provider quota in a staging subscription; rate limits are the usual first failure. |

---

## 10. Release checklist

- [ ] Semantic Kernel pinned exactly; lock file committed
- [ ] Secrets in a secret store; managed identity for Azure resources
- [ ] Kernel built per request; services created once
- [ ] Timeouts on model calls, tools and orchestrations; loop limits set
- [ ] Filters for logging, redaction and approval of sensitive tools
- [ ] OpenTelemetry exporting traces and metrics; sensitive diagnostics off
- [ ] Conversation state in a shared store
- [ ] Token budgets and alerts per tenant
- [ ] Golden-set evaluation runs on prompt and model changes
- [ ] Migration path to Microsoft Agent Framework noted for new features

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | Py 1.45.0 / .NET 1.81.0 | Rewritten as language-neutral guidance. The previous edition's code used pre-1.0 Python APIs (`invoke_async`, `create_function_from_prompt`), wrong .NET signatures and hard-coded 2024 model prices; runnable code now lives in the language-specific production guides. |

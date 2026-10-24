---
title: "Microsoft Agent Framework (Python) — Production Guide"
description: "Hosting Python agents in production (Foundry hosted agents, Azure Functions durable agents, your own service), observability, prompt-injection defence, evaluation in CI and framework-specific operational patterns."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "Production guide"
  order: 85
---

This page is about the parts of running an agent in production that are specific to Agent Framework. General container, networking and CI/CD setup is not repeated here; Microsoft's [Azure Container Apps documentation](https://learn.microsoft.com/azure/container-apps/) covers it.

## 1. Choosing a host

| Host | Package | What you get |
|---|---|---|
| Microsoft Foundry hosted agent | `agent-framework-foundry-hosting` (beta) | `ResponsesHostServer` serves your agent with the OpenAI Responses contract inside Foundry's agent runtime: managed conversation history, background responses, HITL tool approval, Foundry identity and tracing. |
| Azure Functions durable agent | `agent-framework-azurefunctions` (beta) | `AgentFunctionApp` turns agents or workflows into durable entities and orchestrations: history survives restarts and scale-in, HTTP endpoints are generated, and workflows can wait days for a human. |
| Durable Task Scheduler without Functions | `agent-framework-durabletask` (beta) | The same durable agents hosted in any process, with `DurableAIAgentWorker` / `DurableAIAgentClient`. |
| Your own web service | `agent-framework-core` | Full control: FastAPI or similar on Container Apps, App Service or AKS. You own history storage, scaling and auth. |

### Foundry hosted agents

```python
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient, ResponsesHostServer
from azure.identity import DefaultAzureCredential

agent = Agent(
    client=FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["FOUNDRY_MODEL"],
        credential=DefaultAzureCredential(),
    ),
    instructions="You are a friendly assistant. Keep your answers brief.",
    default_options={"store": False},     # the host keeps the conversation, not the model service
)

if __name__ == "__main__":
    ResponsesHostServer(agent).run()      # POST /responses
```

`ResponsesHostServer` also accepts a workflow (`workflow=`), and options for where history lives (`history_source="agent_server" | "agent" | "service"`), session and checkpoint stores, and the origins allowed for OAuth consent prompts from hosted MCP tools (`allowed_oauth_consent_origins=`). Package and deploy the container with the Azure Developer CLI as described in [Microsoft's hosted-agent guide](https://learn.microsoft.com/agent-framework/hosting/foundry-hosted-agent).

### Durable agents on Azure Functions

```python
import os

from agent_framework import Agent
from agent_framework.azure import AgentFunctionApp
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

writer = Agent(
    client=FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["FOUNDRY_MODEL"],
        credential=DefaultAzureCredential(),
    ),
    name="DocumentPublisher",
    instructions="You write well-structured documents on a given topic.",
)

app = AgentFunctionApp(agents=[writer])   # the Functions entry point (function_app.py)
```

Each agent becomes a durable entity whose conversation state is stored by Durable Functions, so a session continues after a restart or on another instance. Pass `workflow=` (or `workflows=`) to host a graph workflow as a durable orchestration: agent nodes become entities, other executors become activities, and `request_info` pauses can last as long as you need. Other options: `http_auth_level=`, `enable_http_endpoints=`, `enable_health_check=` and `enable_mcp_tool_trigger=` (expose each agent as an MCP tool). See Microsoft's [durable agents documentation](https://learn.microsoft.com/azure/durable-task/sdks/durable-agents-microsoft-agent-framework).

### Your own service

Build one agent per process at start-up, keep a session per conversation in a shared history provider (Redis or Cosmos DB when you run more than one replica), and close the agent at shutdown (`await agent.close()`, or `async with agent:`), which closes the MCP connections it opened. Section 10 below has the FastAPI pattern, per-tenant scoping, timeouts, concurrency limits and health checks.

## 2. Observability

```python
from agent_framework.observability import configure_otel_providers

configure_otel_providers(service_name="support-agent")   # reads OTEL_EXPORTER_OTLP_* from the environment
```

Agent runs, model calls, tool calls and workflow supersteps then emit OpenTelemetry spans and metrics with the GenAI semantic conventions. Message contents are excluded unless you pass `enable_sensitive_data=True`. If your application already configures OpenTelemetry (for example with `azure-monitor-opentelemetry`), call `enable_instrumentation()` instead. Azure Monitor setup and the span reference are on the [observability page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_observability/).

## 3. Identity and secrets

- Use `DefaultAzureCredential` (managed identity in Azure) for Azure OpenAI and Foundry, and grant the identity the narrowest role on the resource. The roles are listed on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/).
- Keep any remaining API keys in Key Vault or your platform's secret store, load them with `load_settings` into `SecretString` fields, and unwrap them only at the SDK boundary.
- Pass per-user tokens to tools and MCP servers through `function_invocation_kwargs`, not instructions, so they never reach the model.

## 4. Prompt-injection defence

Treat tool output, retrieved documents and MCP results as untrusted. Agent Framework gives you three layers:

1. **Approval for consequential tools.** `@tool(approval_mode="always_require")` (or `approval_mode=` on MCP tools) stops the run until a person approves the call. See [Human in the loop](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/).
2. **Information-flow control (FIDES, experimental).** `SecureAgentConfig` labels tool results as trusted or untrusted, hides untrusted content behind variable references, and blocks privileged tools (or asks for approval) once untrusted data is in context:

   ```python
   from agent_framework import Agent
   from agent_framework.foundry import FoundryChatClient
   from agent_framework.security import SecureAgentConfig
   from azure.identity import DefaultAzureCredential

   agent = Agent(
       client=FoundryChatClient(credential=DefaultAzureCredential()),   # endpoint and model from FOUNDRY_* env vars
       instructions="You triage inbound email.",
       tools=[read_inbox, send_reply],
       context_providers=[
           SecureAgentConfig(allow_untrusted_tools={"read_inbox"}, approval_on_violation=True),
       ],
   )
   ```

   Constructor options and the MCP proxy are in the [comprehensive guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#prompt-injection-defence-fides).
3. **Content filtering.** Azure AI Content Safety (text moderation, and [Prompt Shields](https://learn.microsoft.com/azure/ai-services/content-safety/concepts/jailbreak-detection) for jailbreak and indirect-injection detection) and Microsoft Purview policies run as agent middleware; the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#azure-ai-content-safety-integration) shows the middleware pattern and the Purview integration.

## 5. Evaluation in CI

Run a small, deterministic evaluation on every pull request and a larger model-judged one on a schedule. `LocalEvaluator` checks are plain functions, so the PR gate costs only the agent's own calls:

```python
# tests/test_agent_quality.py
import pytest
from agent_framework import LocalEvaluator, evaluate_agent, keyword_check, tool_called_check

from app.agents import build_support_agent   # your factory


@pytest.mark.asyncio
async def test_refund_questions_use_the_refund_tool() -> None:
    agent = build_support_agent()
    results = await evaluate_agent(
        agent=agent,
        queries=["I want a refund for order 1042", "How do I return a damaged item?"],
        evaluators=LocalEvaluator(keyword_check("refund"), tool_called_check("lookup_order")),
    )
    for result in results:
        result.assert_no_failed_items()
```

Evaluation is experimental (`ExperimentalFeature.EVALS`). Model-graded evaluators, Foundry evaluators, `evaluate_workflow` and repetition for flaky prompts are on the [evaluation page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_evaluation/).

## 6. Cost control

- Use the small tier (`%%model=openai-small%%`) for routing, classification and summarisation, and override per call with `options={"model": ...}`.
- Read `response.usage_details` (or the `gen_ai.usage.*` span attributes) and alert on per-tenant spend.
- Add a [compaction](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_compaction/) strategy to long-lived sessions so prompt size stays flat.
- Cap tool loops with `FunctionInvocationConfiguration` (`max_iterations`, `max_function_calls`) and workflows with `max_iterations` / Magentic's `max_round_count`.

## 7. Resilience

Retries belong in middleware, so they apply to every model or tool call. The [middleware page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/) has retry and per-tool circuit-breaker examples, and section 10 below has a model-level circuit breaker. For multi-step work that must not be lost, prefer a durable host (section 1) or [checkpointing](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_checkpointing/) over in-process retries.

---

## 8. Local testing with DevUI

Before pushing to ACA, smoke-test the agent against a real model using `agent-framework-devui`. It's a built-in FastAPI/Uvicorn server (re-exported as `agent_framework.devui`) that exposes any agent or workflow over an OpenAI-compatible HTTP API plus a web UI.

```python
# devui_app.py
from agent_framework import Agent
from agent_framework.devui import serve
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential


def build_agent() -> Agent:
    return Agent(
        client=FoundryChatClient(credential=DefaultAzureCredential()),   # endpoint and model from FOUNDRY_* env vars
        instructions="You are an internal support agent.",
    )


if __name__ == "__main__":
    serve(
        entities=[build_agent()],
        host="127.0.0.1",
        port=8080,
        auto_open=True,
        instrumentation_enabled=True,                  # surfaces OpenTelemetry traces in the UI
    )
```

Authentication and exposure:

- Bearer-token auth on the `/v1/*` API is **on by default** (`auth_enabled=True`). Set `DEVUI_AUTH_TOKEN` (or pass `auth_token=`) to keep the same token across restarts; on a loopback host without one, the server generates a token and logs it at start-up.
- `auth_enabled=False` (CLI: `--no-auth`) is only allowed on loopback hosts. Binding to a network address such as `host="0.0.0.0"` without a token raises `ValueError` at start-up.
- Use `mode="user"` instead of the default `"developer"` before showing it to others: generic error messages and restricted APIs.
- DevUI is a development harness, not a production runtime.

For directory-style discovery (one folder per agent), use `serve(entities_dir="./agents", ...)` or run `devui ./agents --port 8080` from the CLI (`--auth-token`, `--mode user` and `--no-auth` are the related flags).

---

## 9. Choosing a chat client

Three first-party clients dominate Azure deployments:

| Client | Best for | Notes |
|---|---|---|
| `OpenAIChatClient` (Azure mode) | Standard Azure OpenAI deployments — direct endpoint + key/Entra | Smallest dependency footprint; reads `AZURE_OPENAI_*` env vars |
| `agent_framework.foundry.FoundryChatClient` | Microsoft Foundry projects — model deployment, evaluation, private networking | Reads `FOUNDRY_PROJECT_ENDPOINT` / `FOUNDRY_MODEL`; supports `AIProjectClient` reuse for OAuth helpers |
| `agent_framework.foundry.FoundryAgent` | Service-managed agents — identity, threads, tool definitions live in Foundry | Use when you want the agent to outlive your process / be visible in the Foundry portal |

`FoundryChatClient` constructor accepts either `(project_endpoint=, model=, credential=)` or a pre-built `project_client=AIProjectClient(...)`. The latter is the right choice when your service already shares an `AIProjectClient` for evaluations or memory:

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

project = AIProjectClient(
    endpoint="https://corp-foundry.services.ai.azure.com/api/projects/support",
    credential=DefaultAzureCredential(),
)

client = FoundryChatClient(project_client=project, model="%%model=openai-small%%")
agent = Agent(client=client, instructions="...")
```

All three clients implement the same `SupportsChatGetResponse` protocol, so you can swap them without touching agent / tool / middleware code.

---

## 10. Operational patterns specific to `agent_framework`

### Reuse one `Agent` per process

Build the agent (and its chat client, which holds the HTTP connection pool) once at start-up and pass it through dependency injection:

```python
# app.py
from contextlib import asynccontextmanager

from azure.identity.aio import DefaultAzureCredential
from fastapi import Depends, FastAPI
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.redis import RedisHistoryProvider

_agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    async with DefaultAzureCredential() as credential:
        _agent = Agent(
            client=FoundryChatClient(credential=credential),
            instructions="You are the customer-support agent.",
            context_providers=[RedisHistoryProvider(redis_url="redis://redis:6379", application_id="support")],
            default_options={"store": False},     # history lives in Redis, not in the Foundry service
        )
        async with _agent:                        # closes MCP connections on shutdown
            yield


app = FastAPI(lifespan=lifespan)


def get_agent() -> Agent:
    assert _agent is not None
    return _agent


@app.post("/chat")
async def chat(conversation_id: str, message: str, agent: Agent = Depends(get_agent)) -> dict:
    session = agent.create_session(session_id=conversation_id)   # authorise access to this id first
    response = await agent.run(message, session=session)
    return {"reply": response.text}
```

Per-request state (conversation history, user context) lives in the `AgentSession` and the history provider, not the agent. Return `response.text` (or your own model) rather than the `AgentResponse` object, which is not a Pydantic model.

### Per-tenant scoping

For SaaS deployments, keep one agent and scope each request instead of building per-tenant agents. Put tenant data that context providers need in `session.state`, and data tools need in `function_invocation_kwargs`:

```python
session = agent.create_session(session_id=conversation_id)
session.state["tenant_id"] = tenant_id                       # read by context providers
await agent.run(
    user_message,
    session=session,
    function_invocation_kwargs={"tenant_id": tenant_id},     # passed to tools that accept **kwargs
)
```

Neither reaches the model. One agent instance serves every tenant.

### Graceful timeouts on workflows

`Workflow.run(...)` doesn't enforce a wall-clock deadline. Wrap it in `asyncio.wait_for` so a stuck tool or runaway Magentic loop doesn't tie up your worker pool:

```python
import asyncio
import logging
from agent_framework import CheckpointStorage, Workflow

logger = logging.getLogger(__name__)


async def safe_run(
    workflow: Workflow,
    user_input: str,
    *,
    storage: CheckpointStorage,
    timeout_s: float = 60.0,
):
    try:
        return await asyncio.wait_for(workflow.run(user_input), timeout=timeout_s)
    except asyncio.TimeoutError:
        # Save the latest checkpoint so a human can resume manually.
        latest = await storage.get_latest(workflow_name=workflow.name)
        logger.warning(
            "workflow_timeout checkpoint_id=%s",
            latest.checkpoint_id if latest else None,
        )
        raise
```

Combine with `MagenticBuilder(max_round_count=20, max_stall_count=3)` so the orchestrator gives up internally before you ever hit the outer timeout.

### Concurrency limits

Front-line `OpenAIChatClient` / `FoundryChatClient` traffic is rate-limited by your model deployment. Add a semaphore at the agent boundary so a flood of requests doesn't all hit the model at once:

```python
import asyncio
from agent_framework import Agent, AgentContext, AgentMiddleware
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential


class ConcurrencyLimit(AgentMiddleware):
    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    async def process(self, context: AgentContext, call_next) -> None:
        async with self._sem:
            await call_next()


agent = Agent(
    client=FoundryChatClient(credential=DefaultAzureCredential()),
    instructions="…",
    middleware=[ConcurrencyLimit(max_concurrent=8)],
)
```

For workflow-level limits put the semaphore in your handler instead of around `agent.run` — workflow supersteps execute concurrently, so you want one semaphore per executor type.

### Health checks that don't burn tokens

A liveness probe that calls `agent.run("ping")` costs real money on every probe interval. Keep `/healthz` as a lightweight local liveness signal — it answers "is this Python process still running?" — and let the orchestrator's network probe answer "is the upstream model up?":

```python
@app.get("/healthz")
async def healthz(agent: Agent = Depends(get_agent)):
    # Liveness only — does NOT touch the chat client. The process is up if this returns.
    return {"status": "ok", "agent": agent.name, "version": agent.id}
```

If you genuinely need a readiness probe that proves the model deployment is reachable, run a separate `/readyz` that issues a token-free call (e.g. an Azure OpenAI `models` listing or an `AIProjectClient.connections.list()` call) on startup, caches the result for 60 seconds, and returns 503 until the probe succeeds. Never make the model probe synchronous on every probe — the cost adds up fast at K8s default 10-second intervals.

### Checkpoint hygiene

`FileCheckpointStorage` and Cosmos / Redis backends accumulate over time. Add a scheduled task that prunes old checkpoints per workflow (`list_checkpoint_ids(workflow_name=...)` and `delete(checkpoint_id)` are part of the `CheckpointStorage` protocol):

```python
async def prune_checkpoints(storage, *, workflow_name: str, keep: int = 50) -> int:
    ids = await storage.list_checkpoint_ids(workflow_name=workflow_name)
    deleted = 0
    for old in ids[:-keep]:
        if await storage.delete(old):
            deleted += 1
    return deleted
```

Run it nightly per workflow name. Aim for `keep` ≥ longest plausible HITL pause (don't auto-delete checkpoints with `pending_request_info_events`).

### Circuit breaker for upstream model outages

When the model API is degraded, fail fast rather than waiting for every request to time out:

```python
import asyncio
import time
from agent_framework import ChatContext, ChatMiddleware


class ModelUnavailable(Exception):
    """Raised while the circuit is open; catch it around agent.run()."""


class ModelCircuitBreaker(ChatMiddleware):
    """A simple async-safe circuit breaker.

    `ChatMiddleware.process` runs concurrently across requests, so the failure
    counter and open-until timestamp are guarded by an `asyncio.Lock`. The lock
    is only held while reading/mutating state — never around `call_next()` —
    so it does not serialise model traffic.
    """

    def __init__(self, *, fail_threshold: int = 5, recover_seconds: float = 30.0) -> None:
        self._fail_count = 0
        self._open_until: float = 0.0
        self._fail_threshold = fail_threshold
        self._recover_seconds = recover_seconds
        self._lock = asyncio.Lock()

    async def process(self, context: ChatContext, call_next) -> None:
        async with self._lock:
            if time.time() < self._open_until:
                raise ModelUnavailable("model unavailable: circuit open")

        try:
            await call_next()
        except Exception:
            async with self._lock:
                self._fail_count += 1
                if self._fail_count >= self._fail_threshold:
                    self._open_until = time.time() + self._recover_seconds
            raise
        else:
            async with self._lock:
                self._fail_count = 0
```

The exception propagates out of `agent.run()`, so the request handler can catch `ModelUnavailable` and fall back to another agent or a static answer. Do not use `MiddlewareTermination` here: raised from chat middleware without a `context.result`, it makes `agent.run()` fail with an `AttributeError` in %%version=agent-framework%%. To answer without calling the model, set `context.result` to a `ChatResponse` and return instead.

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | 1.22.0 | Fixed `FoundryChatClient()` examples that omitted the required credential, the FastAPI example (returned a non-serialisable `AgentResponse`, no session, no shutdown), DevUI auth defaults (auth is on by default; network binds without a token fail), the project-endpoint format and the circuit breaker (`MiddlewareTermination` from chat middleware breaks the run). |

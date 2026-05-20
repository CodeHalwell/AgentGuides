---
title: "Microsoft Agent Framework (Python) — 2025 / 2026 Features"
description: "Verified feature set in agent-framework-core 1.5.0 and beta sub-packages (a2a, declarative, copilotstudio, observability) as of April 2026."
framework: microsoft-agent-framework
language: python
---

# 2025 / 2026 Features — Python

> **Errata (April 2026).** An earlier draft of this page documented `from agent_framework.a2a import A2AProtocolAdapter, A2AClient, A2AMessage` and `from agent_framework.graphs import AgentGraph, GraphNode, CheckpointConfig, HITLConfig` — **none of those classes exist**. This page has been rewritten after direct introspection of the installed packages.
>
> For the real graph/workflow API and A2A integration, see the dedicated pages:
> - [Workflows & Declarative Agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/)
> - [A2A Protocol](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/)

## Package status (April 2026)

| Package | Version | Status |
|---|---|---|
| `agent-framework-core` | 1.4.0 | Stable |
| `agent-framework-foundry` | 1.4.0 | Stable |
| `agent-framework-openai` | 1.4.0 | Stable |
| `agent-framework-a2a` | 1.0.0b260514 | Beta |
| `agent-framework-declarative` | 1.0.0b260514 | Beta |
| `agent-framework-copilotstudio` | 1.0.0b260514 | Beta |
| `agent-framework-mem0` | 1.0.0b260514 | Beta |
| `agent-framework-anthropic` | 1.0.0b260514 | Beta |
| `agent-framework-bedrock` | 1.0.0b260514 | Beta |
| `agent-framework-azurefunctions` | 1.0.0b260514 | Beta |
| `agent-framework-durabletask` | 1.0.0b260514 | Beta |
| `agent-framework-orchestrations` | 1.0.0b260514 | Beta |
| `agent-framework-redis` | 1.0.0b260514 | Beta |

The `agent-framework` meta package (1.4.0) installs core + provider packages in one go. When you're ready to prune dependencies, install the sub-packages directly (e.g. `pip install --pre agent-framework-foundry agent-framework-mem0`).

## Agent2Agent (A2A) protocol

`agent-framework-a2a` exposes exactly one class — `A2AAgent` — which wraps a remote A2A-compliant agent as a local framework `Agent`. Full detail on the [A2A protocol page](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/).

```python
from agent_framework.a2a import A2AAgent

remote = A2AAgent(url="https://analyst.example.com/a2a", name="Analyst")
response = await remote.run("Summarise Q3")
```

## Workflows (graph-based orchestration)

The real workflow API is `WorkflowBuilder` + `Executor`/`FunctionExecutor` + `Edge` groups, all exported from top-level `agent_framework`. See [Workflows & Declarative Agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) for the full reference. One-glance example:

```python
from agent_framework import WorkflowBuilder, FunctionExecutor

research = FunctionExecutor(lambda q: {"data": q}, id="research")
write    = FunctionExecutor(lambda d: f"report: {d['data']}", id="write")

workflow = (
    WorkflowBuilder(start_executor=research, name="ContentPipeline")
    .add_edge(research, write)
    .build()
)
```

Workflows support **checkpointing** via `FileCheckpointStorage` / `InMemoryCheckpointStorage`, **fan-in / fan-out / switch-case** routing, and **exposing a workflow as an agent** via `WorkflowAgent`.

## Declarative agents & workflows (YAML)

`agent-framework-declarative` provides `AgentFactory` (single agent from YAML) and `WorkflowFactory` (multi-agent workflow from YAML). Actions are a Power-Fx-expression dialect — `SetVariable`, `InvokeAzureAgent`, `InvokeFunctionTool`, `If`, `Foreach`, `RepeatUntil`, `Question`, `Confirmation`, `SendActivity`, etc. Full reference: [Microsoft Learn — Declarative Workflows](https://learn.microsoft.com/agent-framework/workflows/declarative/).

### `AgentFactory` — load a single agent from YAML

The factory parses a `kind: Prompt` YAML definition, picks the chat-client class from the `model.provider` field (`AzureOpenAI`, `OpenAI`, `Foundry`, …), and returns a configured `Agent` ready for `agent.run(...)`.

```python
from agent_framework.declarative import AgentFactory


# Method 1: from a YAML file path
factory = AgentFactory()
agent = factory.create_agent_from_yaml_path("agent.yaml")
response = await agent.run("Hello!")


# Method 2: from inline YAML
yaml_content = """
kind: Prompt
name: GreetingAgent
instructions: You are a friendly assistant.
model:
  id: gpt-4o
  provider: AzureOpenAI
"""
agent = factory.create_agent_from_yaml(yaml_content)
```

Two security-relevant knobs on the factory constructor:

- `safe_mode=True` (default) blocks `=Env.*` Power-Fx lookups in the YAML so untrusted YAML can't read your environment. Flip to `False` only for YAML you fully control.
- `additional_mappings={...}` extends the built-in provider table — point a custom `Provider.ApiType` key at your own `SupportsChatGetResponse` implementation when you want to instantiate non-first-party clients from YAML.

If you've already constructed a chat client (shared connection pool, custom credentials), pass it via `client=` so the factory reuses it instead of building a fresh one from the YAML's `model:` block.

### `WorkflowFactory` — load a multi-action workflow from YAML

`WorkflowFactory` parses `kind: Workflow` YAML and returns a real `Workflow` object — every action becomes a node in the graph, so checkpointing, visualization, and pause/resume work the same as a code-built workflow.

```python
from agent_framework import FileCheckpointStorage
from agent_framework.declarative import WorkflowFactory
from agent_framework.openai import OpenAIChatClient


client = OpenAIChatClient()

# Pre-register agents that the workflow YAML references via `kind: InvokeAzureAgent`.
factory = WorkflowFactory(
    agents={
        "Writer": client.as_agent(name="Writer", instructions="Write content."),
        "Reviewer": client.as_agent(name="Reviewer", instructions="Review content."),
    },
    checkpoint_storage=FileCheckpointStorage("./checkpoints"),
    max_iterations=200,           # raise the default 100 for GotoAction loops
)

workflow = factory.create_workflow_from_yaml_path("review_pipeline.yaml")

async for event in workflow.run({"input": "draft the agent-framework launch post"}, stream=True):
    if event.type == "output":
        print(event.data)
```

Pre-registering agents (`agents={...}`) is the common pattern when the same agent definition is reused across many workflows. For one-off agents, embed the `Prompt` definition inline in the YAML and let the inner `AgentFactory` build them.

## Middleware & hooks

Middleware is a first-class concept in `agent-framework-core`. Three decorators — `@chat_middleware`, `@agent_middleware`, `@function_middleware` — and three base classes — `ChatMiddleware`, `AgentMiddleware`, `FunctionMiddleware` — let you wrap the corresponding call type:

```python
from agent_framework import Agent, chat_middleware, ChatContext
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import DefaultAzureCredential

@chat_middleware
async def logging_middleware(context: ChatContext, next):
    print(f"before: {len(context.messages)} messages")
    await next(context)
    print(f"after: {context.result}")

agent = Agent(
    client=FoundryChatClient(credential=DefaultAzureCredential(), model="gpt-4o"),
    instructions="You are a helpful assistant.",
    middleware=[logging_middleware],  # must be a list in 2026 releases
)
```

Breaking changes in the 2026 beta line (verified against release notes):

- `display_name` parameter removed from agents.
- `context_providers` (plural list) renamed to `context_provider` (singular; only one provider allowed).
- `middleware` now requires a list — no longer accepts a single instance.
- `AgentRunResponse` / `AgentRunResponseUpdate` renamed to `AgentResponse` / `AgentResponseUpdate`.
- `AggregateContextProvider` removed — build your own aggregator from examples.

## OpenTelemetry observability

`agent-framework-core` emits OpenTelemetry traces and metrics out of the box. Wire it up with the standard OTel SDK:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Framework code from here on emits agent_framework.* spans
```

For Azure Monitor, use `azure.monitor.opentelemetry.configure_azure_monitor(connection_string=...)`.

## Content safety

Content safety is **not** a framework-specific wrapper class. Call Azure AI Content Safety directly from a middleware:

```python
from agent_framework import chat_middleware, ChatContext
from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

cs_client = ContentSafetyClient("https://<region>.cognitiveservices.azure.com/",
                                AzureKeyCredential("<key>"))

@chat_middleware
async def safety_guard(context: ChatContext, next):
    last_user = next((m for m in reversed(context.messages) if m.role == "user"), None)
    if last_user:
        verdict = await cs_client.analyze_text(text=last_user.content)
        if verdict.categories_analysis[0].severity > 2:
            raise ValueError("Blocked by content safety")
    await next(context)
```

## Async patterns

The framework is async-first:

- Agents expose `await agent.run(...)` and `async for update in agent.run(..., stream=True): ...`.
- Chat clients implement the `SupportsChatGetResponse` protocol — `await client.get_response(...)`.
- Checkpoint storage (`FileCheckpointStorage`, `InMemoryCheckpointStorage`) exposes awaitable `save` / `load` / `get_latest`.
- `A2AAgent` uses `httpx.AsyncClient` under the hood; pass your own if you need custom TLS / retry / auth.

## Mem0 — semantic long-term memory

`agent-framework-mem0` ships `Mem0ContextProvider` — a `ContextProvider` that searches a Mem0-hosted memory store before each run and persists request+response messages after. Use it for cross-session recall ("the user prefers metric units"), facts you want to surface independently of the conversation history.

```python
from agent_framework import Agent, FileHistoryProvider
from agent_framework.mem0 import Mem0ContextProvider
from agent_framework.openai import OpenAIChatClient


memory = Mem0ContextProvider(
    api_key="m0_...",                    # or set MEM0_API_KEY
    user_id="alice@example.com",         # one of user_id / agent_id / application_id is required
    context_prompt="## Memories\nConsider these when answering:",
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[
        FileHistoryProvider(storage_path="./sessions"),     # short-term
        memory,                                              # long-term semantic recall
    ],
)

async with memory:                                           # opens the underlying Mem0 client
    session = agent.create_session(session_id="alice-2026")
    await agent.run("I'm planning a trip — remember I always want vegetarian options.", session=session)
    await agent.run("Suggest a weekend in Lisbon.", session=session)
    # The second run sees the vegetarian preference even if it's not in the recent chat history.
```

`Mem0ContextProvider` accepts either an `AsyncMemoryClient` (cloud Mem0 platform) or `AsyncMemory` (open-source Mem0 instance). Pass `mem0_client=` when you've already constructed one in your app — the provider will not close a client it didn't create.

Filtering is enforced at construction time: at least one of `user_id`, `agent_id`, or `application_id` must be set, otherwise `_validate_filters` raises before the first search hits Mem0. Use `application_id` for tenant scoping, `user_id` for per-end-user memory, and `agent_id` when multiple agents share the same user.

## Local interactive testing — DevUI

`agent-framework-devui` (re-exported as `agent_framework.devui`) is a built-in FastAPI/Uvicorn dev server that exposes any list of agents or workflows over an OpenAI-compatible HTTP API plus an interactive web UI. Useful for poking at an agent without writing your own front-end.

```python
from agent_framework import Agent
from agent_framework.devui import serve
from agent_framework.openai import OpenAIChatClient


writer = Agent(
    client=OpenAIChatClient(),
    name="writer",
    instructions="You write concise marketing copy.",
)


# Blocks the current process — uvicorn under the hood.
serve(
    entities=[writer, my_workflow],            # mix agents and workflows
    host="127.0.0.1",
    port=8080,
    auto_open=True,                            # pop the browser
    instrumentation_enabled=True,              # wire OpenTelemetry traces in the UI
)
```

Three knobs that matter when deploying beyond localhost:

- `mode="developer"` (default) returns full error tracebacks and exposes admin APIs. Use `mode="user"` for shareable demos — it returns generic errors and locks down the management endpoints.
- `auth_enabled=True` requires a Bearer token on every request. Without `auth_token=`, the server auto-generates one and prints it on startup; in CI / production, set `DEVUI_AUTH_TOKEN` in the environment so it's stable across restarts.
- Network-exposing the server (`host="0.0.0.0"`) without `auth_enabled=True` triggers a startup warning — DevUI is not a hardened production runtime, it's a developer harness.

For directory-based discovery (one folder per agent), point at the directory instead of passing instances:

```python
serve(entities_dir="./agents", port=8080)
```

The dev server is also runnable from the CLI: `devui ./agents --port 8080 --auth`.

## Production patterns

- **Package minimisation.** Pin only the sub-packages you need: `pip install --pre agent-framework-foundry agent-framework-redis`.
- **Session persistence.** Use `agent.create_session()` with the Redis, Cosmos, or mem0 providers depending on your backend.
- **Checkpointing long-running workflows.** Use `FileCheckpointStorage` for local, or wrap a `CheckpointStorage`-compatible backend for cloud storage.
- **Durable agents on Azure Functions.** `agent-framework-azurefunctions` + `agent-framework-durabletask` expose `DurableAIAgent`, `DurableAIAgentClient`, and `AgentFunctionApp` for long-running orchestrations.
- **Multiple LLM providers.** Swap Foundry, OpenAI, Anthropic, Bedrock, Ollama, and local Foundry via the chat-client of the same interface (`SupportsChatGetResponse`).

## Sessions and conversation history

`AgentSession` is a lightweight state container — provider instances live on the agent, the session only carries `session_id`, an optional `service_session_id` (for service-side history like OpenAI Responses), and a mutable `state: dict[str, Any]` shared across providers.

```python
from agent_framework import Agent, AgentSession, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

history = FileHistoryProvider(storage_path="./sessions", skip_excluded=True)
agent = Agent(client=OpenAIChatClient(), context_providers=[history])

session = agent.create_session(session_id="user-42")        # picks up ./sessions/user-42.jsonl
await agent.run("Continue the chat", session=session)

# Session round-trips cleanly through to_dict/from_dict — handy if you want to
# stash a snapshot in your own datastore alongside the JSONL.
snapshot = session.to_dict()
restored = AgentSession.from_dict(snapshot)
```

Built-in providers (`InMemoryHistoryProvider`, `FileHistoryProvider`) plus the Redis/Cosmos backends in the beta sub-packages all subclass `HistoryProvider`. To roll your own, override `get_messages()` and `save_messages()` — the rest of the lifecycle (`load_messages` / `store_inputs` / `store_outputs` / `store_context_messages` flags) is handled by the parent.

## MCP — three transports, one tool surface

```python
from agent_framework import Agent, MCPStreamableHTTPTool, MCPStdioTool, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient

# Streamable HTTP (most remote MCP servers)
async with MCPStreamableHTTPTool(
    name="learn",
    url="https://learn.microsoft.com/api/mcp",
    request_timeout=30,
    # Per-request auth headers from agent.run(..., function_invocation_kwargs=...)
    header_provider=lambda kwargs: {"Authorization": f"Bearer {kwargs['token']}"},
) as learn:
    agent = Agent(client=OpenAIChatClient(), tools=learn)
    await agent.run("Explain DefaultAzureCredential",
                    function_invocation_kwargs={"token": user_token})

# Stdio (local servers — filesystem, git, sqlite, etc.)
async with MCPStdioTool(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
) as fs:
    agent = Agent(client=OpenAIChatClient(), tools=fs)

# WebSocket (bidirectional realtime)
async with MCPWebsocketTool(name="realtime", url="wss://service.example.com/mcp") as rt:
    agent = Agent(client=OpenAIChatClient(), tools=rt)
```

Per-tool approval gates work uniformly:

```python
mcp = MCPStdioTool(
    name="git",
    command="uvx",
    args=["mcp-server-git"],
    approval_mode={
        "always_require_approval": ["git_push", "git_reset"],
        "never_require_approval": ["git_status", "git_diff"],
    },
)
```

For hosted MCP (the provider runs the MCP client server-side), feature-detect via `isinstance(client, SupportsMCPTool)` and call `client.get_mcp_tool(name=..., url=...)`. See the [MCP page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/) for full transport, header, and approval coverage.

## Magentic — manager + workers + replanning

`MagenticBuilder` is the most sophisticated of the orchestration patterns: a manager agent maintains a task ledger, dispatches to workers, and can replan when a stall is detected.

```python
from agent_framework_orchestrations import MagenticBuilder, StandardMagenticManager

workflow = (
    MagenticBuilder(
        participants=[researcher, analyst, writer],
        manager_agent=StandardMagenticManager(client=OpenAIChatClient(model="gpt-4o")),
        enable_plan_review=True,        # pause after the initial plan for human review
        checkpoint_storage=storage,     # resume across process restarts
    )
    .with_human_input_on_stall()        # ask a human when the manager loops
    .build()
)
```

The plan-review HITL event is durable — combined with checkpointing, a user can come back hours later in a different pod and approve the plan, and the workflow resumes from the exact superstep that paused.

## Custom workflow executors

Beyond agents, workflows accept arbitrary executors. The `@executor` decorator is the lightweight form; subclassing `Executor` gives access to per-instance state and `@response_handler` for HITL.

```python
from agent_framework import AgentExecutorResponse, WorkflowContext, executor


@executor(
    id="upper_case_executor",
    input=AgentExecutorResponse,
    output=AgentExecutorResponse,
    workflow_output=str,
)
async def upper_case(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, str],
) -> None:
    upper_text = response.agent_response.text.upper()
    # with_text preserves the prior conversation chain so downstream
    # AgentExecutors still see the full history.
    await ctx.send_message(response.with_text(upper_text))
    await ctx.yield_output(upper_text)
```

The `with_text` helper on `AgentExecutorResponse` matters when you transform agent output and want the next `AgentExecutor` to retain conversation history — sending a plain `str` instead would invoke `from_str` on the downstream executor and reset its message cache.

## Further reading

- [Microsoft Learn — Agent Framework](https://learn.microsoft.com/agent-framework/)
- [Migration from Semantic Kernel](https://learn.microsoft.com/agent-framework/migration-guide/from-semantic-kernel/)
- [Source (microsoft/agent-framework)](https://github.com/microsoft/agent-framework)

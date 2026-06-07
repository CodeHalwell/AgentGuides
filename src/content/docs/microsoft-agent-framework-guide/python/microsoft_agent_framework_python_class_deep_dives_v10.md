---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 10"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.0: A2AAgent+A2AAgentSession+A2AContinuationToken, A2AExecutor, AGUIChatClient+AGUIHttpService+AGUIRequest+AGUIChatOptions, AnthropicClient+AnthropicChatOptions+ThinkingConfig, BedrockChatClient+BedrockChatOptions+BedrockGuardrailConfig, CopilotStudioAgent+CopilotStudioSettings, GroupChatOrchestrator+GroupChatState+GroupChatSelectionFunction, AgentBasedGroupChatOrchestrator+AgentOrchestrationOutput, HandoffBuilder+HandoffConfiguration+HandoffSentEvent, ExternalInputRequest+ExternalInputResponse+MCPToolApprovalRequest."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 33
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 10

Verified against **agent-framework 1.8.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source at `/usr/local/lib/python3.11/dist-packages/`. No API name has been guessed or
inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — `AgentFileStore` hierarchy, `FileAccessProvider`, `MCPSkill` + `MCPSkillsSource`, `ToolMode`, `AgentEvalConverter` + `CheckResult` + `RubricScore`, `ChatContext`, `WorkflowAgent` + `WorkflowContext`, `TruncationStrategy`, `HistoryProvider` + `InMemoryHistoryProvider`, `DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent` + `DurableAIAgentClient` + `DurableAIAgentWorker`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool` + `HyperlightCodeActProvider`, `Mem0ContextProvider`, `RedisContextProvider` + `RedisHistoryProvider`, `StandardMagenticManager` + `MagenticContext`, `FileSkillsSource` + `FilteringSkillsSource`

This volume covers **ten class groups** drawn from four integration packages and the
orchestrations package: **A2A protocol consumer & server bridges**, **AG-UI protocol**,
**Anthropic & Bedrock chat clients**, **Microsoft Copilot Studio**, and **advanced
orchestration** patterns (custom group-chat, LLM-driven selection, explicit handoff, and
declarative HITL).

---

## Table of Contents

1. [`A2AAgent` + `A2AAgentSession` + `A2AContinuationToken`](#1-a2aagent--a2aagentsession--a2acontinuationtoken)
2. [`A2AExecutor`](#2-a2aexecutor)
3. [`AGUIChatClient` + `AGUIHttpService` + `AGUIRequest` + `AGUIChatOptions`](#3-aguichatclient--aguihttpservice--aguirequest--aguichatoptions)
4. [`AnthropicClient` + `AnthropicChatOptions` + `ThinkingConfig`](#4-anthropicclient--anthropicchatoptions--thinkingconfig)
5. [`BedrockChatClient` + `BedrockChatOptions` + `BedrockGuardrailConfig`](#5-bedrockchatclient--bedrockchatoptions--bedrockguardrailconfig)
6. [`CopilotStudioAgent` + `CopilotStudioSettings`](#6-copilotstudioagent--copilotstudiosettings)
7. [`GroupChatOrchestrator` + `GroupChatState` + `GroupChatSelectionFunction` + `TerminationCondition`](#7-groupchatorchestrator--groupchatstate--groupchatselectionfunction--terminationcondition)
8. [`AgentBasedGroupChatOrchestrator` + `AgentOrchestrationOutput`](#8-agentbasedgroupchatorchestrator--agentorchestrationoutput)
9. [`HandoffBuilder` + `HandoffConfiguration` + `HandoffSentEvent`](#9-handoffbuilder--handoffconfiguration--handoffsentevent)
10. [`ExternalInputRequest` + `ExternalInputResponse` + `MCPToolApprovalRequest`](#10-externalinputrequest--externalinputresponse--mcptoolapprovalrequest)

---

## 1. `A2AAgent` + `A2AAgentSession` + `A2AContinuationToken`

**Source:** `agent_framework_a2a._agent`
**Package:** `pip install agent-framework[a2a]`

`A2AAgent` wraps the **A2A (Agent-to-Agent) protocol** client so you can call any remote
A2A-compliant agent (OpenAI Agents SDK, Claude Agent SDK, LangGraph, Google ADK, …) using
the same `agent.run()` surface you use with local agents.

`A2AAgentSession` extends `AgentSession` with the three pieces of A2A protocol state required
for multi-turn continuations: `context_id`, `task_id`, and `task_state`.

`A2AContinuationToken` is the framework's `ContinuationToken` sub-type returned when the remote
agent enters the `input-required` state — resuming with the token lets you continue that task
rather than opening a new one.

### Constructor — `A2AAgent`

```python
class A2AAgent(AgentTelemetryLayer, BaseAgent):
    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        description: str | None = None,
        agent_card: AgentCard | None = None,        # Pre-fetched AgentCard
        url: str | None = None,                      # Remote A2A base URL
        client: Client | None = None,                # Pre-built a2a.Client
        http_client: httpx.AsyncClient | None = None,
        auth_interceptor: AuthInterceptor | None = None,
        timeout: float | httpx.Timeout | None = None, # Default: 10 connect / 60 read
        supported_protocol_bindings: list[str] | None = None,  # Default: ["JSONRPC"]
        **kwargs: Any,
    ) -> None: ...
```

`url` is the most common entry point — the agent fetches the `/.well-known/agent.json` card
automatically. Pass `agent_card` when you have already fetched it (saves an HTTP round-trip).
Pass `client` when you need full control of the underlying `a2a.Client` (e.g. custom TLS).

### Constructor — `A2AAgentSession`

```python
class A2AAgentSession(AgentSession):
    def __init__(
        self,
        *,
        context_id: str | None = None,
        task_id: str | None = None,
        task_state: TaskState | None = None,
    ) -> None: ...
```

`to_dict()` / `from_dict()` round-trips the A2A keys alongside the standard
`AgentSession` state, making sessions safe to persist in `FileHistoryProvider` or Redis.

### Key `A2AAgent` terminal task states

| Constant | Value | Meaning |
|---|---|---|
| `TASK_STATE_COMPLETED` | terminal | Task finished successfully |
| `TASK_STATE_FAILED` | terminal | Task ended with an error |
| `TASK_STATE_CANCELED` | terminal | Task was cancelled |
| `TASK_STATE_INPUT_REQUIRED` | in-progress | Remote agent paused, needs more user input |
| `TASK_STATE_AUTH_REQUIRED` | in-progress | Remote agent needs authentication |

### Example 1 — basic single-turn call

```python
import asyncio
from agent_framework_a2a import A2AAgent

async def main():
    # Point at any A2A-compliant remote agent
    agent = A2AAgent(url="http://localhost:9000")
    response = await agent.run("What is the capital of France?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — multi-turn conversation with session persistence

```python
import asyncio
import json
from agent_framework_a2a import A2AAgent, A2AAgentSession

async def main():
    agent = A2AAgent(url="http://localhost:9000", name="RemoteAssistant")

    # First turn — create a fresh session
    session = A2AAgentSession()
    response = await agent.run("Start a recipe for banana bread.", session=session)
    print(response.text)

    # Persist session state between requests
    saved = json.dumps(session.to_dict())

    # Second turn — restore session so the remote agent remembers context
    restored_session = A2AAgentSession.from_dict(json.loads(saved))
    response = await agent.run("Add chocolate chips to the recipe.", session=restored_session)
    print(response.text)

asyncio.run(main())
```

### Example 3 — streaming response + input-required continuation

```python
import asyncio
from agent_framework_a2a import A2AAgent, A2AAgentSession, A2AContinuationToken

async def main():
    agent = A2AAgent(url="http://localhost:9000")
    session = A2AAgentSession()

    # Run the agent and check if it needs more input
    response = await agent.run("Plan a 3-day itinerary for Tokyo.", session=session)
    print(response.text)

    if isinstance(response.continuation_token, A2AContinuationToken):
        # Remote agent paused and needs clarification — send follow-up on same session
        follow_up = await agent.run(
            "Focus the itinerary on food tours.",
            session=session,
        )
        print(follow_up.text)

asyncio.run(main())
```

### Example 4 — authenticated remote agent

```python
import asyncio
import httpx
from a2a.client.auth.interceptor import BearerTokenInterceptor
from agent_framework_a2a import A2AAgent

async def main():
    token = "my-bearer-token"
    agent = A2AAgent(
        url="https://secure.remote-agent.example.com",
        auth_interceptor=BearerTokenInterceptor(token),
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    )
    response = await agent.run("Summarise Q2 earnings report.")
    print(response.text)

asyncio.run(main())
```

---

## 2. `A2AExecutor`

**Source:** `agent_framework_a2a._a2a_executor`
**Package:** `pip install agent-framework[a2a]`

`A2AExecutor` is the **server-side** complement to `A2AAgent`. It implements the `a2a.server.AgentExecutor`
interface and bridges an `agent_framework` agent into an A2A JSON-RPC server so that any external
A2A client can call it.

### Constructor

```python
class A2AExecutor(AgentExecutor):
    def __init__(
        self,
        agent: SupportsAgentRun,     # Any Agent, WorkflowAgent, or SupportsAgentRun
        stream: bool = False,         # Enable streaming (TaskArtifactUpdateEvent chunks)
        run_kwargs: Mapping[str, Any] | None = None,  # Extra kwargs forwarded to agent.run()
    ) -> None: ...
```

`run_kwargs` cannot contain `"session"` or `"stream"` — both are managed internally.

### Key methods

| Method | Description |
|---|---|
| `execute(context, event_queue)` | Main dispatch — runs the agent, converts response to A2A events |
| `cancel(context, event_queue)` | Sends `TASK_STATE_CANCELED` through the event queue |
| `handle_events(item, updater, ...)` | Override to customise agent-output → A2A event mapping |

`handle_events` is the main extension point. Override it to support custom content types, add
metadata, or route events differently.

### Example 1 — minimal A2A server

```python
import asyncio
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent
from agent_framework_a2a import A2AExecutor
from starlette.applications import Starlette
import uvicorn

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="You are a helpful assistant.",
)

card = AgentCard(
    name="MyAgent",
    description="A simple agent exposed over A2A.",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(url="http://localhost:9000/", protocol_binding="JSONRPC"),
    ],
    skills=[],
)

handler = DefaultRequestHandler(
    agent_executor=A2AExecutor(agent, stream=True),
    task_store=InMemoryTaskStore(),
    agent_card=card,
)

app = Starlette(
    routes=[
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

### Example 2 — custom `handle_events` to add metadata

```python
from a2a.server.tasks import TaskUpdater
from a2a.types import Part
from agent_framework import AgentResponse, AgentResponseUpdate, Message
from agent_framework_a2a import A2AExecutor

class TaggedA2AExecutor(A2AExecutor):
    """Prepends an [AGENT] tag to every text response."""

    async def handle_events(
        self,
        item: Message | AgentResponseUpdate,
        updater: TaskUpdater,
        streamed_artifact_ids: set[str] | None = None,
        default_artifact_id: str | None = None,
    ) -> None:
        # Prefix text parts with a tag before forwarding
        for content in getattr(item, "contents", []):
            if content.type == "text" and content.text:
                content.text = f"[AGENT] {content.text}"
        await super().handle_events(item, updater, streamed_artifact_ids, default_artifact_id)
```

### Example 3 — wiring `A2AAgent` + `A2AExecutor` for cross-framework routing

```python
# agent_b.py — exposes a LangGraph agent over A2A (run separately)
# agent_a.py — uses A2AAgent to call agent_b from an agent_framework workflow

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_a2a import A2AAgent

# Agent A calls Agent B transparently through the A2A protocol
agent_b_proxy = A2AAgent(url="http://agent-b-host:9001", name="AgentB")

agent_a = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="Route specialised tasks to Agent B when appropriate.",
)

# Use agent_b_proxy exactly like a local sub-agent in a handoff or sequential workflow
```

---

## 3. `AGUIChatClient` + `AGUIHttpService` + `AGUIRequest` + `AGUIChatOptions`

**Source:** `agent_framework_ag_ui._client`, `._http_service`, `._types`
**Package:** `pip install agent-framework[ag-ui]`

[AG-UI](https://github.com/ag-ui/ag-ui-protocol) is a streaming SSE protocol for connecting
AI agents to front-end UIs. This package provides both sides of the bridge:

- **`AGUIChatClient`** — a `BaseChatClient` that speaks to a remote AG-UI server (use it anywhere you'd use `OpenAIChatClient`)
- **`AGUIHttpService`** — thin HTTP + SSE layer; useful when you need raw event access
- **`AGUIRequest`** — Pydantic request model validated on the server side
- **`AGUIChatOptions`** — `ChatOptions` extension with AG-UI-specific keys

### Constructor — `AGUIChatClient`

```python
class AGUIChatClient(
    FunctionInvocationLayer,
    ChatMiddlewareLayer,
    ChatTelemetryLayer,
    BaseChatClient,
):
    def __init__(
        self,
        endpoint: str,                                   # AG-UI server URL
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    ) -> None: ...
```

### Constructor — `AGUIHttpService`

```python
class AGUIHttpService:
    def __init__(
        self,
        endpoint: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None: ...
```

### Key `AGUIChatOptions` fields

| Key | Description |
|---|---|
| `forward_props` | Extra props forwarded verbatim to the AG-UI server |
| `context` | Shared state dict sent to the server (for generative UI) |
| `available_interrupts` | Interrupt descriptors the server may resume |
| `resume` | Resume payload for continuing a paused run |
| `metadata` | Pass `thread_id` here for multi-turn continuity |

### Example 1 — using `AGUIChatClient` as a drop-in chat client

```python
import asyncio
from agent_framework import Agent
from agent_framework_ag_ui import AGUIChatClient

async def main():
    # Any AG-UI-compliant server works as a backend
    client = AGUIChatClient("http://localhost:8888/api/agent")
    agent = Agent(
        client=client,
        instructions="Answer questions about our product catalogue.",
    )
    response = await agent.run("What colours does the Apex hoodie come in?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — multi-turn chat with `thread_id` continuity

```python
import asyncio
import uuid
from agent_framework import Agent
from agent_framework_ag_ui import AGUIChatClient

async def main():
    thread_id = str(uuid.uuid4())
    client = AGUIChatClient("http://localhost:8888/api/agent")
    agent = Agent(client=client, instructions="You are a shopping assistant.")

    for user_msg in [
        "I am looking for running shoes.",
        "Size 10, please.",
        "Do you have them in blue?",
    ]:
        response = await agent.run(
            user_msg,
            options={"metadata": {"thread_id": thread_id}},
        )
        print(f"User: {user_msg}")
        print(f"Agent: {response.text}\n")

asyncio.run(main())
```

### Example 3 — server side: FastAPI endpoint with `add_agent_framework_fastapi_endpoint`

```python
from fastapi import FastAPI
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint

app = FastAPI()

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="You are a helpful assistant for generative UI demos.",
)

# Registers GET /.well-known/agent.json + POST /api/agent SSE endpoint
add_agent_framework_fastapi_endpoint(app, agent, path="/api/agent")
```

### Example 4 — raw SSE event access with `AGUIHttpService`

```python
import asyncio
import uuid
from agent_framework_ag_ui import AGUIHttpService

async def main():
    svc = AGUIHttpService("http://localhost:8888/api/agent")
    async with svc:
        async for event in svc.post_run(
            thread_id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
            messages=[{"role": "user", "content": "Hello!"}],
            state={"user_name": "Alice"},
        ):
            event_type = event.get("type")
            if event_type == "TEXT_MESSAGE_CONTENT":
                print(event["delta"], end="", flush=True)
            elif event_type == "RUN_FINISHED":
                print("\n[done]")

asyncio.run(main())
```

---

## 4. `AnthropicClient` + `AnthropicChatOptions` + `ThinkingConfig`

**Source:** `agent_framework_anthropic._chat_client`
**Package:** `pip install agent-framework[anthropic]`

`AnthropicClient` is the full-stack Anthropic chat client with middleware, telemetry, and
function invocation. `AnthropicChatOptions` extends the standard `ChatOptions` with
Anthropic-specific parameters including `thinking` for Claude's extended reasoning.

`AnthropicFoundryClient`, `AnthropicBedrockClient`, and `AnthropicVertexClient` are
drop-in replacements for Foundry, Bedrock, and Vertex endpoints respectively — same
constructor shape, different credential wiring.

### Constructor — `AnthropicClient`

```python
class AnthropicClient(
    FunctionInvocationLayer,
    ChatMiddlewareLayer,
    ChatTelemetryLayer,
    BaseChatClient,
):
    def __init__(
        self,
        *,
        api_key: str | None = None,          # env: ANTHROPIC_API_KEY
        model: str | None = None,            # env: ANTHROPIC_CHAT_MODEL
        base_url: str | None = None,         # env: ANTHROPIC_BASE_URL
        anthropic_client: AsyncAnthropic | None = None,  # Bring-your-own client
        additional_beta_flags: list[str] | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

Default beta flags enabled automatically: `"mcp-client-2025-04-04"` and
`"code-execution-2025-08-25"`.

### Key `AnthropicChatOptions` fields

| Key | Type | Notes |
|---|---|---|
| `max_tokens` | `int` | **Required** — defaults to 1024 if omitted |
| `thinking` | `ThinkingConfig` | Enable extended reasoning |
| `top_k` | `int` | Top-K sampling |
| `service_tier` | `"auto" \| "standard_only"` | Routing preference |
| `additional_beta_flags` | `list[str]` | Extra Anthropic beta headers |
| `container` | `dict` | Container config for Skills |

`temperature`, `top_p`, `stop`, `tools`, `tool_choice`, and `response_format` are inherited
from `ChatOptions` and fully supported. `seed`, `frequency_penalty`, `presence_penalty`,
`store`, and `logit_bias` are typed `None` (Anthropic does not support them).

### `ThinkingConfig` fields

```python
class ThinkingConfig(TypedDict, total=False):
    type: Literal["enabled", "disabled"]
    budget_tokens: int   # Minimum 1024; counts against max_tokens
```

### Example 1 — basic agent with Anthropic

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework_anthropic import AnthropicClient

@tool
def word_count(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())

async def main():
    client = AnthropicClient(model="claude-sonnet-4-5-20250929")
    agent = Agent(
        client=client,
        instructions="You are a writing assistant.",
        tools=[word_count],
    )
    response = await agent.run("How many words are in 'The quick brown fox'?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — extended thinking for complex reasoning

```python
import asyncio
from agent_framework import Agent
from agent_framework_anthropic import AnthropicClient, AnthropicChatOptions

async def main():
    client = AnthropicClient(model="claude-opus-4-5")
    agent = Agent(
        client=client,
        instructions="You are a strategic advisor. Think deeply before answering.",
    )
    # Enable extended thinking — budget_tokens counts against max_tokens
    options: AnthropicChatOptions = {
        "max_tokens": 16_000,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 10_000,  # Up to 10k tokens of reasoning
        },
    }
    response = await agent.run(
        "What are the key strategic risks in entering the Japanese market for a SaaS company?",
        options=options,
    )
    print(response.text)

asyncio.run(main())
```

### Example 3 — using `AnthropicFoundryClient` for Azure AI Foundry deployments

```python
import asyncio
from agent_framework import Agent
from agent_framework_anthropic import AnthropicFoundryClient

async def main():
    # Credentials resolved from AZURE_ENDPOINT, ANTHROPIC_API_KEY env vars
    client = AnthropicFoundryClient(model="claude-sonnet-4-5-20250929")
    agent = Agent(client=client, instructions="You are a helpful Azure-hosted assistant.")
    response = await agent.run("Explain Azure Container Apps in one paragraph.")
    print(response.text)

asyncio.run(main())
```

### Example 4 — structured output with Pydantic model

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework_anthropic import AnthropicClient

class BookReview(BaseModel):
    title: str
    rating: float       # 1.0 – 5.0
    summary: str
    recommended: bool

async def main():
    client = AnthropicClient(model="claude-sonnet-4-5-20250929")
    agent = Agent(client=client, instructions="You are a book reviewer.")
    response = await agent.run(
        "Review 'The Pragmatic Programmer'.",
        options={"response_format": BookReview, "max_tokens": 2048},
    )
    review: BookReview = response.value
    print(f"Rating: {review.rating}/5 — Recommended: {review.recommended}")
    print(review.summary)

asyncio.run(main())
```

---

## 5. `BedrockChatClient` + `BedrockChatOptions` + `BedrockGuardrailConfig`

**Source:** `agent_framework_bedrock._chat_client`
**Package:** `pip install agent-framework[bedrock]`

`BedrockChatClient` wraps Amazon Bedrock's **Converse API** — a single unified surface for
Claude, Titan, Llama, Mistral, and other foundation models. `BedrockGuardrailConfig` applies
Amazon Bedrock Guardrails for content filtering, PII detection, and grounding checks.

### Constructor

```python
class BedrockChatClient(
    FunctionInvocationLayer,
    ChatMiddlewareLayer,
    ChatTelemetryLayer,
    BaseChatClient,
):
    def __init__(
        self,
        *,
        region: str | None = None,            # env: BEDROCK_REGION (default: us-east-1)
        model: str | None = None,             # env: BEDROCK_CHAT_MODEL
        access_key: str | None = None,        # AWS_ACCESS_KEY_ID
        secret_key: str | None = None,        # AWS_SECRET_ACCESS_KEY
        session_token: str | None = None,     # AWS_SESSION_TOKEN (temp credentials)
        client: BaseClient | None = None,     # Preconfigured boto3 bedrock-runtime client
        boto3_session: Boto3Session | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### `BedrockGuardrailConfig` fields

```python
class BedrockGuardrailConfig(TypedDict, total=False):
    guardrailIdentifier: str          # Guardrail ID or ARN
    guardrailVersion: str             # e.g. "1" or "DRAFT"
    trace: Literal["enabled", "disabled"]
    streamProcessingMode: Literal["sync", "async"]  # sync blocks streaming
```

### Key `BedrockChatOptions` fields not in base `ChatOptions`

| Key | Type | Description |
|---|---|---|
| `guardrailConfig` | `BedrockGuardrailConfig` | Content filtering / PII guardrail |
| `performanceConfig` | `dict` | Latency profile (`{"latency": "optimized"}`) |
| `requestMetadata` | `dict[str, str]` | Key-value metadata on the request |
| `promptVariables` | `dict[str, str]` | Variables for managed prompt templates |

`seed`, `frequency_penalty`, `presence_penalty`, `store`, `logit_bias`, `user`,
and `metadata` are not supported by Bedrock and are typed `None`.

### Example 1 — basic Bedrock agent

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework_bedrock import BedrockChatClient

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI from weight in kg and height in metres."""
    return weight_kg / (height_m ** 2)

async def main():
    # IAM credentials resolved from environment or instance profile
    client = BedrockChatClient(
        region="us-east-1",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    agent = Agent(
        client=client,
        instructions="You are a health information assistant.",
        tools=[calculate_bmi],
    )
    response = await agent.run("What is the BMI for someone 70kg and 1.75m tall?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — applying Bedrock Guardrails

```python
import asyncio
from agent_framework import Agent
from agent_framework_bedrock import BedrockChatClient, BedrockChatOptions, BedrockGuardrailConfig

async def main():
    client = BedrockChatClient(
        region="us-east-1",
        model="amazon.titan-text-express-v1",
    )
    agent = Agent(client=client, instructions="Customer support agent.")

    guardrail: BedrockGuardrailConfig = {
        "guardrailIdentifier": "my-guardrail-id",
        "guardrailVersion": "1",
        "trace": "enabled",
        "streamProcessingMode": "sync",
    }
    options: BedrockChatOptions = {"guardrailConfig": guardrail, "max_tokens": 512}

    response = await agent.run(
        "Help me return a defective product.",
        options=options,
    )
    print(response.text)

asyncio.run(main())
```

### Example 3 — structured output (Bedrock `outputConfig.textFormat`)

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework_bedrock import BedrockChatClient

class ProductSummary(BaseModel):
    name: str
    price_usd: float
    in_stock: bool
    description: str

async def main():
    client = BedrockChatClient(
        region="us-east-1",
        model="amazon.nova-pro-v1:0",
    )
    agent = Agent(client=client, instructions="Extract product information.")
    raw = "Apex Runner Pro — $129.99, currently available. Lightweight trail shoe."
    response = await agent.run(
        f"Extract structured data: {raw}",
        options={"response_format": ProductSummary, "max_tokens": 256},
    )
    product: ProductSummary = response.value
    print(f"{product.name} — ${product.price_usd} — in stock: {product.in_stock}")

asyncio.run(main())
```

### Example 4 — using a pre-configured boto3 session (cross-account role)

```python
import asyncio
import boto3
from agent_framework import Agent
from agent_framework_bedrock import BedrockChatClient

async def main():
    sts = boto3.client("sts")
    creds = sts.assume_role(
        RoleArn="arn:aws:iam::123456789012:role/BedrockAccess",
        RoleSessionName="AgentSession",
    )["Credentials"]

    session = boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name="us-east-1",
    )
    client = BedrockChatClient(
        boto3_session=session,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )
    agent = Agent(client=client, instructions="Cross-account agent.")
    response = await agent.run("Hello from a cross-account role!")
    print(response.text)

asyncio.run(main())
```

---

## 6. `CopilotStudioAgent` + `CopilotStudioSettings`

**Source:** `agent_framework_copilotstudio._agent`
**Package:** `pip install agent-framework[copilotstudio]`

`CopilotStudioAgent` wraps a published **Microsoft Copilot Studio** bot and exposes it through
the standard `BaseAgent` interface. You can use it as a participant in handoff workflows,
as a sub-agent in sequential pipelines, or anywhere else you'd use a local agent.

### Constructor

```python
class CopilotStudioAgent(BaseAgent):
    def __init__(
        self,
        client: CopilotClient | None = None,
        settings: ConnectionSettings | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: list[AgentMiddlewareTypes] | None = None,
        environment_id: str | None = None,      # env: COPILOTSTUDIOAGENT__ENVIRONMENTID
        agent_identifier: str | None = None,    # env: COPILOTSTUDIOAGENT__SCHEMANAME
        client_id: str | None = None,           # env: COPILOTSTUDIOAGENT__AGENTAPPID
        tenant_id: str | None = None,           # env: COPILOTSTUDIOAGENT__TENANTID
        token: str | None = None,               # Pre-acquired auth token
        cloud: PowerPlatformCloud | None = None,
        agent_type: AgentType | None = None,
        custom_power_platform_cloud: str | None = None,
        username: str | None = None,
        token_cache: Any | None = None,
        scopes: list[str] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### `CopilotStudioSettings` TypedDict keys

| Key | Env var | Description |
|---|---|---|
| `environmentid` | `COPILOTSTUDIOAGENT__ENVIRONMENTID` | Power Platform environment ID |
| `schemaname` | `COPILOTSTUDIOAGENT__SCHEMANAME` | Agent schema name / identifier |
| `agentappid` | `COPILOTSTUDIOAGENT__AGENTAPPID` | Azure App Registration client ID |
| `tenantid` | `COPILOTSTUDIOAGENT__TENANTID` | Azure AD tenant ID |

### Example 1 — minimal connection via environment variables

```python
# .env
# COPILOTSTUDIOAGENT__ENVIRONMENTID=abc123
# COPILOTSTUDIOAGENT__SCHEMANAME=my-copilot-bot
# COPILOTSTUDIOAGENT__AGENTAPPID=app-client-id
# COPILOTSTUDIOAGENT__TENANTID=tenant-id

import asyncio
from agent_framework_copilotstudio import CopilotStudioAgent

async def main():
    agent = CopilotStudioAgent(name="HRBot")
    response = await agent.run("How many days of annual leave do I have?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — explicit credentials, GCC cloud

```python
import asyncio
from microsoft_agents.copilotstudio.client import PowerPlatformCloud
from agent_framework_copilotstudio import CopilotStudioAgent

async def main():
    agent = CopilotStudioAgent(
        environment_id="prod-env-abc123",
        agent_identifier="ITSupportAgent",
        client_id="your-app-id",
        tenant_id="your-tenant-id",
        cloud=PowerPlatformCloud.GCC,
        name="ITSupport",
    )
    response = await agent.run("My printer is offline. What should I do?")
    print(response.text)

asyncio.run(main())
```

### Example 3 — Copilot Studio bot as a participant in a handoff workflow

`HandoffBuilder` requires `Agent` instances, so wrap the `CopilotStudioAgent` in a
`@tool` and expose it through a thin local `Agent`. The triage agent then routes
HR queries to that wrapper via the standard handoff mechanism.

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework_copilotstudio import CopilotStudioAgent
from agent_framework.orchestrations import HandoffBuilder

hr_bot = CopilotStudioAgent(
    environment_id="env-id",
    agent_identifier="HRBot",
    client_id="app-id",
    tenant_id="tenant-id",
    name="hr_bot",
    description="Handles HR queries: leave, benefits, payroll.",
)

@tool
async def call_hr_bot(query: str) -> str:
    """Call the Copilot Studio HR bot for HR queries."""
    response = await hr_bot.run(query)
    return response.text

hr_agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="hr_agent",
    instructions="Handles HR queries by calling the HR bot tool.",
    tools=[call_hr_bot],
)

triage_agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="triage",
    instructions=(
        "Classify incoming queries. Hand off to hr_agent for any HR-related "
        "topics (leave, benefits, payroll). Handle general questions yourself."
    ),
)

workflow = (
    HandoffBuilder()
    .participants([triage_agent, hr_agent])
    .add_handoff(triage_agent, hr_agent, "HR-related query")
    .start(triage_agent)
    .build()
)

async def main():
    response = await workflow.run("How do I submit a holiday request?")
    print(response.text)

asyncio.run(main())
```

---

## 7. `GroupChatOrchestrator` + `GroupChatState` + `GroupChatSelectionFunction` + `TerminationCondition`

**Source:** `agent_framework_orchestrations._group_chat`, `._base_group_chat_orchestrator`
**Package:** `pip install agent-framework[orchestrations]`

`GroupChatOrchestrator` is the **function-driven** group chat: you supply a plain Python
callable (`GroupChatSelectionFunction`) that receives the current `GroupChatState` and
returns the name of the next speaker. This gives deterministic, testable routing without
any LLM calls in the selection layer.

`TerminationCondition` is a `Callable[[list[Message]], bool | Awaitable[bool]]` that halts
the conversation when it returns `True`.

### Key types

```python
@dataclass(frozen=True)
class GroupChatState:
    current_round: int                         # 0-indexed
    participants: OrderedDict[str, str]        # name -> description
    conversation: list[Message]                # full history


GroupChatSelectionFunction = Callable[
    [GroupChatState],
    Awaitable[str] | str,
]

TerminationCondition = Callable[
    [list[Message]],
    bool | Awaitable[bool],
]
```

### `GroupChatBuilder` quick-reference

`GroupChatBuilder` is the high-level builder that wires a `GroupChatOrchestrator` (or
`AgentBasedGroupChatOrchestrator`) into a `Workflow`. It is the preferred entry point.

```python
from agent_framework.orchestrations import GroupChatBuilder
builder = (
    GroupChatBuilder()
    .participants([agent_a, agent_b, agent_c])
    .selection_function(my_selection_fn)
    .termination_condition(my_termination)
    .max_rounds(20)
    .build()
)
```

### Example 1 — round-robin selection function

```python
import asyncio
from collections import OrderedDict
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

def round_robin(state: GroupChatState) -> str:
    names = list(state.participants.keys())
    return names[state.current_round % len(names)]

writer = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="writer",
    instructions="You draft short creative paragraphs on the given topic.",
)
critic = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="critic",
    instructions="You provide concise constructive feedback on writing.",
)

workflow = (
    GroupChatBuilder()
    .participants([writer, critic])
    .selection_function(round_robin)
    .max_rounds(6)
    .build()
)

async def main():
    async for event in workflow.run_stream("Write about the future of renewable energy."):
        if hasattr(event, "text"):
            print(f"[{event.role}] {event.text}\n")

asyncio.run(main())
```

### Example 2 — async termination condition (keyword check)

```python
import asyncio
from agent_framework import Message
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

async def done_when_approved(messages: list[Message]) -> bool:
    """Stop when the last assistant message contains 'APPROVED'."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return any(
                "APPROVED" in (c.text or "") for c in (msg.contents or [])
            )
    return False

def select_reviewer_then_approver(state: GroupChatState) -> str:
    names = list(state.participants.keys())
    # Reviewer speaks first, then approver, cycling
    return names[state.current_round % len(names)]

# ... build agents and workflow as above, using done_when_approved
```

### Example 3 — observing `GroupChatRequestSentEvent` and `GroupChatResponseReceivedEvent`

```python
import asyncio
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState
from agent_framework.orchestrations import GroupChatRequestSentEvent, GroupChatResponseReceivedEvent

async def main():
    # ... build workflow ...
    async for event in workflow.run_stream("Discuss quantum computing trends."):
        if isinstance(event, GroupChatRequestSentEvent):
            print(f"→ Round {event.round_index}: asking {event.participant_name}")
        elif isinstance(event, GroupChatResponseReceivedEvent):
            print(f"← Round {event.round_index}: received from {event.participant_name}")
        elif hasattr(event, "text"):
            print(f"  {event.text}")

asyncio.run(main())
```

---

## 8. `AgentBasedGroupChatOrchestrator` + `AgentOrchestrationOutput`

**Source:** `agent_framework_orchestrations._group_chat`
**Package:** `pip install agent-framework[orchestrations]`

`AgentBasedGroupChatOrchestrator` replaces the Python selection function with a **dedicated
orchestrator agent** that reasons about conversation context to decide the next speaker and
whether to terminate. The agent must support structured output (it returns
`AgentOrchestrationOutput`).

### `AgentOrchestrationOutput` — the structured schema

```python
class AgentOrchestrationOutput(BaseModel):
    terminate: bool           # Whether to stop the group chat
    reason: str               # Explanation for the decision
    next_speaker: str | None  # Name of next participant (None if terminating)
    final_message: str | None # Optional summary if terminating
```

`json_schema_extra` forces all four fields into the JSON schema's `required` array so that
OpenAI strict mode models always populate them — set `model_config = {"extra": "forbid"}`.

### Constructor — `AgentBasedGroupChatOrchestrator`

```python
class AgentBasedGroupChatOrchestrator(BaseGroupChatOrchestrator):
    def __init__(
        self,
        agent: Agent,                               # Orchestrator agent (structured output)
        participant_registry: ParticipantRegistry,
        *,
        max_rounds: int | None = None,
        termination_condition: TerminationCondition | None = None,
        retry_attempts: int | None = None,
        session: AgentSession | None = None,        # Shared session across rounds
    ) -> None: ...
```

Use via `GroupChatBuilder.orchestrator_agent(my_agent)` to avoid constructing the orchestrator
directly.

### Example 1 — LLM-driven expert panel

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import GroupChatBuilder

orchestrator = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="moderator",
    instructions="""You moderate a panel discussion. Based on the conversation so far,
decide which expert should speak next and whether the discussion is complete.
Return a JSON object with: terminate (bool), reason (str), next_speaker (str|null),
final_message (str|null).""",
)

climate_scientist = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="climate_scientist",
    instructions="You are a climate scientist. Provide data-driven insights on climate change.",
)
economist = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="economist",
    instructions="You are an economist. Analyse climate policy from an economic perspective.",
)
policy_expert = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="policy_expert",
    instructions="You are a policy expert. Suggest practical legislation and governance approaches.",
)

workflow = (
    GroupChatBuilder()
    .participants([climate_scientist, economist, policy_expert])
    .orchestrator_agent(orchestrator)
    .max_rounds(12)
    .build()
)

async def main():
    async for event in workflow.run_stream(
        "What are the most effective policies to reduce carbon emissions by 2035?"
    ):
        if hasattr(event, "text") and event.text:
            role = getattr(event, "role", "?")
            print(f"[{role}]: {event.text}\n")

asyncio.run(main())
```

### Example 2 — retry on orchestrator failure

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import GroupChatBuilder

orchestrator = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="orchestrator",
    instructions="Select the best expert for each question and terminate when answered.",
)

workflow = (
    GroupChatBuilder()
    .participants([agent_a, agent_b])          # defined elsewhere
    .orchestrator_agent(orchestrator)
    # retry_attempts=3: if structured output fails, retry up to 3 times
    .orchestrator_retry_attempts(3)
    .max_rounds(10)
    .build()
)

async def main():
    response = await workflow.run("Explain the CAP theorem.")
    print(response.text)

asyncio.run(main())
```

---

## 9. `HandoffBuilder` + `HandoffConfiguration` + `HandoffSentEvent`

**Source:** `agent_framework_orchestrations._handoff`
**Package:** `pip install agent-framework[orchestrations]`

The **handoff pattern** is *decentralised* routing: agents themselves decide who to hand
off to next by calling a synthetic tool (e.g. `handoff_to_billing`). There is no central
orchestrator LLM — the routing is embedded in each agent's system prompt and tool calls.

`HandoffBuilder` wires participating agents, declares handoff links, and produces a
ready-to-run `Workflow`.

`HandoffConfiguration` stores a directed `(source → target)` link. `HandoffSentEvent` is
emitted as a `WorkflowEvent` payload whenever a handoff occurs so you can observe routing.

### `HandoffConfiguration`

```python
@dataclass
class HandoffConfiguration:
    target_id: str
    description: str | None

    def __init__(
        self,
        *,
        target: str | SupportsAgentRun,   # Agent instance or agent ID string
        description: str | None = None,
    ) -> None: ...
```

### `HandoffBuilder` key methods

| Method | Signature | Description |
|---|---|---|
| `.participants(agents)` | `Sequence[Agent]` | Register all agents |
| `.add_handoff(source, target, description)` | `Agent \| str, Agent \| str, str \| None` | Add a directed handoff link |
| `.start(agent)` | `Agent \| str` | Override which agent handles first user input |
| `.termination_condition(fn)` | `TerminationCondition` | Stop condition |
| `.max_turns(n)` | `int` | Hard ceiling on total agent turns |
| `.autonomous(flag)` | `bool` | If `True`, no user input between hops |
| `.checkpoint_storage(store)` | `CheckpointStorage` | Persist workflow state |
| `.build()` | `→ Workflow` | Compile to runnable workflow |

### Example 1 — customer support routing mesh

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder

def make_agent(name: str, instructions: str) -> Agent:
    return Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name=name,
        instructions=instructions,
    )

triage = make_agent("triage",
    "Classify the user's issue as billing, technical, or general. "
    "Hand off to the appropriate specialist immediately.")
billing = make_agent("billing",
    "Resolve billing and payment issues. "
    "Hand off back to triage if the issue is actually technical.")
technical = make_agent("technical",
    "Resolve technical issues. Hand off to billing if payment is involved.")
general = make_agent("general",
    "Handle general questions. Hand off to triage if specialised help is needed.")

workflow = (
    HandoffBuilder()
    .participants([triage, billing, technical, general])
    # Explicit directed links; omitting .add_handoff() defaults to full mesh
    .add_handoff(triage, billing, "Billing or payment issue")
    .add_handoff(triage, technical, "Technical or product issue")
    .add_handoff(triage, general, "General enquiry")
    .add_handoff(billing, triage, "Needs re-classification")
    .add_handoff(technical, billing, "Billing component discovered")
    .start(triage)
    .max_turns(15)
    .build()
)

async def main():
    async for event in workflow.run_stream("My invoice shows a double charge."):
        if hasattr(event, "text") and event.text:
            print(event.text)

asyncio.run(main())
```

### Example 2 — observing handoff events

```python
import asyncio
from agent_framework.orchestrations import HandoffBuilder
from agent_framework.orchestrations import HandoffSentEvent

async def main():
    async for event in workflow.run_stream("I need help with my subscription."):
        event_data = getattr(event, "data", None)
        if isinstance(event_data, HandoffSentEvent):
            print(f"[HANDOFF] {event_data.source} → {event_data.target}")
        elif hasattr(event, "text") and event.text:
            print(event.text)

asyncio.run(main())
```

### Example 3 — autonomous multi-agent pipeline (no HITL)

```python
import asyncio
from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder

researcher = make_agent("researcher",
    "Research the topic thoroughly then hand off to the writer.")
writer = make_agent("writer",
    "Write a 200-word summary based on the research. Hand off to editor when done.")
editor = make_agent("editor",
    "Polish the summary for clarity and grammar. Output 'FINAL:' followed by the text.")

def is_final(messages: list[Message]) -> bool:
    return any(
        "FINAL:" in (c.text or "")
        for m in messages if m.role == "assistant"
        for c in (m.contents or [])
    )

workflow = (
    HandoffBuilder()
    .participants([researcher, writer, editor])
    .add_handoff(researcher, writer)
    .add_handoff(writer, editor)
    .start(researcher)
    .autonomous(True)
    .termination_condition(is_final)
    .build()
)

async def main():
    response = await workflow.run("Summarise the impact of LLMs on software development.")
    print(response.text)

asyncio.run(main())
```

---

## 10. `ExternalInputRequest` + `ExternalInputResponse` + `MCPToolApprovalRequest`

**Source:** `agent_framework_declarative._workflows._executors_external_input`,
`agent_framework_declarative._workflows._executors_mcp`
**Package:** `pip install agent-framework[declarative]`

Declarative workflows (defined in YAML / JSON) support two kinds of human-in-the-loop pauses:

- **`ExternalInputRequest`** — emitted when a `Question` or `RequestExternalInput` action needs
  user text. The workflow suspends via `request_info` and waits for an `ExternalInputResponse`.
- **`MCPToolApprovalRequest`** — emitted before any MCP tool invocation when the YAML action
  sets `approvalMode: required`. The workflow pauses for a human to approve or deny the call.
  Header values are intentionally omitted (they may carry auth secrets); only header names
  are surfaced.

### `ExternalInputRequest` fields

```python
@dataclass
class ExternalInputRequest:
    request_id: str          # Unique ID matching the workflow event
    message: str             # Prompt text to show the user
    request_type: str        # "question" | "external" | any custom string
    metadata: dict[str, Any] # choices, output_property, timeout, etc.
```

### `ExternalInputResponse` fields

```python
@dataclass
class ExternalInputResponse:
    user_input: str   # User's text answer
    value: Any        # Optional typed value (bool, selected choice, etc.)
```

### `MCPToolApprovalRequest` fields

```python
@dataclass
class MCPToolApprovalRequest:
    request_id: str               # Matches the suspended workflow event ID
    tool_name: str                # Evaluated tool name
    server_url: str               # Evaluated MCP server URL
    server_label: str | None      # Human-readable label
    arguments: dict[str, Any]     # Tool arguments to be sent
    header_names: list[str]       # Auth header names only (no values)
```

### Declarative YAML — question action

```yaml
# agent_definition.yaml
name: OnboardingAgent
model:
  type: AzureOpenAI
  endpoint: "${AZURE_ENDPOINT}"
  model: gpt-4o
actions:
  - type: Question
    prompt:
      text: "What is your preferred programming language?"
    variable: preferred_language
  - type: SendActivity
    activity: "Great, I'll tailor the onboarding to ${preferred_language}."
```

### Example 1 — loading a declarative agent and handling `ExternalInputRequest`

```python
import asyncio
from agent_framework_declarative import AgentFactory
from agent_framework_declarative import ExternalInputRequest, ExternalInputResponse

async def main():
    factory = AgentFactory.from_yaml("agent_definition.yaml")
    workflow = factory.create_workflow()

    async for event in workflow.run_stream("Start onboarding."):
        event_data = getattr(event, "data", None)
        if isinstance(event_data, ExternalInputRequest):
            req: ExternalInputRequest = event_data
            print(f"[INPUT REQUIRED] {req.message}")
            # Simulate user input (in production: read from API / UI)
            user_text = "Python"
            await workflow.resume(ExternalInputResponse(user_input=user_text))
        elif hasattr(event, "text") and event.text:
            print(event.text)

asyncio.run(main())
```

### Example 2 — MCP tool approval gate in YAML

```yaml
# agent_with_mcp.yaml
name: ResearchAgent
model:
  type: AzureOpenAI
  endpoint: "${AZURE_ENDPOINT}"
  model: gpt-4o
actions:
  - type: InvokeMcpTool
    server:
      url: "https://internal-mcp.example.com"
      label: "InternalSearch"
    tool: search_documents
    approvalMode: required          # triggers MCPToolApprovalRequest
    arguments:
      query: "${user_query}"
```

### Example 3 — handling `MCPToolApprovalRequest` in code

```python
import asyncio
from agent_framework_declarative import AgentFactory
from agent_framework_declarative import MCPToolApprovalRequest
from agent_framework_declarative import ToolApprovalResponse

async def approve_or_deny(req: MCPToolApprovalRequest) -> bool:
    """Simple allowlist-based approval."""
    allowed_tools = {"search_documents", "list_files"}
    return req.tool_name in allowed_tools

async def main():
    factory = AgentFactory.from_yaml("agent_with_mcp.yaml")
    workflow = factory.create_workflow()

    async for event in workflow.run_stream(
        "Find documents about the Q3 budget review."
    ):
        event_data = getattr(event, "data", None)
        if isinstance(event_data, MCPToolApprovalRequest):
            req: MCPToolApprovalRequest = event_data
            print(f"[APPROVAL REQUEST] Tool: {req.tool_name} on {req.server_url}")
            print(f"  Arguments: {req.arguments}")
            print(f"  Header names present: {req.header_names}")

            approved = await approve_or_deny(req)
            await workflow.resume(
                ToolApprovalResponse(
                    request_id=req.request_id,
                    approved=approved,
                )
            )
        elif hasattr(event, "text") and event.text:
            print(event.text)

asyncio.run(main())
```

### Example 4 — multi-step declarative workflow with HITL question + MCP approval

```yaml
# pipeline.yaml
name: DocumentReviewPipeline
model:
  type: AzureOpenAI
  endpoint: "${AZURE_ENDPOINT}"
  model: gpt-4o
actions:
  - type: Question
    prompt:
      text: "Which document category should I search?"
    variable: category

  - type: InvokeMcpTool
    server:
      url: "https://internal-mcp.example.com"
    tool: search_documents
    approvalMode: required
    arguments:
      category: "${category}"

  - type: SendActivity
    activity: "Search complete. Found documents in category: ${category}."
```

```python
import asyncio
from agent_framework_declarative import AgentFactory
from agent_framework_declarative import (
    ExternalInputRequest,
    ExternalInputResponse,
    MCPToolApprovalRequest,
    ToolApprovalResponse,
)

async def main():
    factory = AgentFactory.from_yaml("pipeline.yaml")
    workflow = factory.create_workflow()

    async for event in workflow.run_stream("Begin document review."):
        event_data = getattr(event, "data", None)
        if isinstance(event_data, ExternalInputRequest):
            print(f"Question: {event_data.message}")
            await workflow.resume(ExternalInputResponse(user_input="quarterly-reports"))

        elif isinstance(event_data, MCPToolApprovalRequest):
            req = event_data
            print(f"Approve search_documents with args {req.arguments}? [y]")
            await workflow.resume(
                ToolApprovalResponse(request_id=req.request_id, approved=True)
            )

        elif hasattr(event, "text") and event.text:
            print(event.text)

asyncio.run(main())
```

---

*Previous: [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — integration package deep dives (Ollama, Purview, Durable, Hyperlight, Mem0, Redis, Magentic-One internals, skill discovery).*

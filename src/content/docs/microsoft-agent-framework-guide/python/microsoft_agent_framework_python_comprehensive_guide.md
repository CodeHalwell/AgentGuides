---
title: "Microsoft Agent Framework Python - Comprehensive Technical Guide"
description: "Release: GA April 3, 2026; patch 1.0.1 April 10, 2026 Framework Version: 1.0.1 Target Platform: Python 3.10+"
framework: microsoft-agent-framework
language: python
---

Latest: 1.0.1 | Updated: April 20, 2026
# Microsoft Agent Framework Python - Comprehensive Technical Guide

**Release:** GA April 3, 2026; patch 1.0.1 April 10, 2026
**Framework Version:** 1.0.1
**Target Platform:** Python 3.10+

---

> **API reference (verified against `agent-framework-core==1.1.0`).**
>
> - **Package name / import root:** `agent_framework` (underscores). Install with `pip install agent-framework`.
> - **Primary agent class:** `Agent`. Construct with `Agent(client=<ChatClient>, instructions=..., tools=[...])`.
> - **Chat clients:** `agent_framework.foundry.FoundryChatClient`, `agent_framework.openai.OpenAIChatClient`, `agent_framework.anthropic.AnthropicClient`, plus Bedrock / Ollama in the `1.0.0b` provider line.
> - **Tool decorator:** `@tool` from `agent_framework`.
> - **Multi-turn state:** `session = agent.create_session()`, then `await agent.run(prompt, session=session)`.
> - **Workflows:** `WorkflowBuilder` (with `.add_edge` / `.add_fan_in_edges` / `.add_fan_out_edges`) from `agent_framework`.
> - **Declarative YAML agents (beta):** `AgentFactory` / `WorkflowFactory` from `agent_framework.declarative`.

---

## Introduction

This guide provides a comprehensive technical overview of the Microsoft Agent Framework for Python, designed for developers building advanced AI agents and multi-agent systems.

### Framework Overview

The Microsoft Agent Framework is an open-source SDK that unifies the capabilities of **Semantic Kernel** and **AutoGen**. It offers a single, cohesive platform for Python developers to build everything from simple conversational bots to complex, orchestrated multi-agent systems.

-   **Inheritance from Semantic Kernel:** It brings enterprise-grade features, including a robust plugin/tool system, memory management, and a wide array of connectors.
-   **Inheritance from AutoGen:** It incorporates sophisticated multi-agent orchestration, group chat coordination, and flexible conversation patterns.

The framework is designed with a Python-first approach, embracing `asyncio` for scalability and integrating seamlessly with the rich Python data science and web development ecosystems.

### Key Objectives

-   **Unified SDK:** A single, Pythonic library for all agent development needs.
-   **Production-Ready:** Built-in features for observability, security, and scalable deployment.
-   **Extensibility:** A modular design that allows for custom agents, tools, and memory backends.
-   **Azure Integration:** Deep, native integration with Azure AI services while remaining platform-agnostic.

---

## Core Fundamentals

### Architecture Principles

The framework's architecture is layered to promote modularity and ease of use.

```
+-----------------------------------+
|      Application Layer            |
| (Your Agents, FastAPI/Flask APIs) |
+-----------------------------------+
|      Orchestration Layer          |
| (Workflows, GroupChatManager)     |
+-----------------------------------+
|      Agent Abstraction Layer      |
| (Agent, AgentThread, BaseAgent)   |
+-----------------------------------+
|      Core Components Layer          |
| (Tools, Memory, LLM Providers)    |
+-----------------------------------+
|      Integration Layer            |
| (Azure, OpenAI, Custom Connectors)|
+-----------------------------------+
```

### Installation

Setting up your Python environment is straightforward.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install the core package
pip install agent-framework

# 3. Install provider-specific packages (pick what you need)
pip install agent-framework-azure-ai       # Azure AI Foundry chat client
pip install agent-framework-openai         # OpenAI / Azure OpenAI chat clients
pip install azure-identity                 # DefaultAzureCredential, managed identity
```

### Authentication and Configuration

Manage credentials securely using environment variables and `azure-identity`.

**1. Environment Variables:**

Create a `.env` file in your project root.

```
# .env
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
```

**2. Loading Configuration:**

Use a library like `python-dotenv` to load these variables.

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
```

**3. Using `DefaultAzureCredential` (Recommended):**

For production, rely on managed identities and `DefaultAzureCredential`.

```python
from azure.identity.aio import DefaultAzureCredential

# This will automatically use the managed identity of the host,
# environment variables, or local Azure CLI login.
credential = DefaultAzureCredential()
```

### Environment Setup & Basic Usage

```python
# main.py
import asyncio
from azure.identity.aio import DefaultAzureCredential

from agent_framework import Agent
from agent_framework.openai import AzureOpenAIChatClient
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT

async def main():
    # Use DefaultAzureCredential for secure, passwordless auth.
    credential = DefaultAzureCredential()

    # Construct the chat client — pick any first-party provider
    # (OpenAIChatClient, AzureOpenAIChatClient, FoundryChatClient,
    # AnthropicClient, OllamaChatClient, BedrockChatClient).
    client = AzureOpenAIChatClient(
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        credential=credential,
    )

    # Create a simple chat agent.
    agent = Agent(
        client=client,
        instructions="You are a helpful AI assistant for Python developers.",
    )

    # Single-turn call.
    response = await agent.run("What are decorators in Python?")
    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Simple Agents

### The single `Agent` class

Unlike the .NET API (which distinguishes `AIAgent` as the stateless base class and `ChatClientAgent` as the concrete stateful implementation), the Python package ships a single `Agent` class in `agent_framework` that covers both scenarios. How it behaves is driven by how you invoke it:

- **Stateless / single-turn** — call `await agent.run(prompt)` without a session. Each call is independent; no conversation history persists.
- **Stateful / multi-turn** — attach a session (`session = agent.create_session()`) and pass it to each `agent.run(prompt, session=session)` call. The session's `ChatHistoryProvider` (in-memory by default) accumulates turns so follow-ups have context.

For low-level subclassing, inherit from `BaseAgent` (`from agent_framework import BaseAgent`) which provides the minimal surface without the middleware and telemetry layers that `Agent` adds on top.

### Creating an `Agent`

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def run_chat_agent() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a friendly and helpful assistant.",
    )

    # A session carries conversation history across turns.
    session = agent.create_session()

    print("Starting conversation... (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = await agent.run(user_input, session=session)
        print(f"Assistant: {response.text}")
```

### Agent Lifecycle

The chat client owns the underlying HTTP session and credentials; when the client supports it, use `async with` to close those resources deterministically. Agents themselves are cheap to construct — you typically build one per role and reuse it across requests. Sessions are per-conversation and hold `ChatHistoryProvider` state (in-memory by default); create a new session per user or request.

---

## Multi-Agent Systems

### Orchestration Patterns

-   **Sequential Workflow:** The output of one agent is passed as the input to the next.
-   **Router/Dispatcher:** A primary agent routes tasks to specialized agents based on the query.
-   **Group Chat:** Multiple agents collaborate in a shared conversation, moderated by a manager.

### Example: Router Pattern

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

class RouterWorkflow:
    def __init__(self, client: OpenAIChatClient):
        self._router = Agent(
            client=client,
            instructions="You are a router. Classify the user's query as 'Billing' or 'Technical'. Respond with only one of those words.",
        )
        self._billing_agent = Agent(
            client=client,
            instructions="You are a billing support expert.",
        )
        self._tech_agent = Agent(
            client=client,
            instructions="You are a technical support expert.",
        )

    async def handle_request(self, user_query: str) -> str:
        route_response = await self._router.run(user_query)
        route = route_response.text

        target_agent = self._billing_agent if "Billing" in route else self._tech_agent

        final_response = await target_agent.run(user_query)
        return final_response.text

# --- Usage ---
# workflow = RouterWorkflow(OpenAIChatClient())
# response = await workflow.handle_request("I have a problem with my invoice.")
# print(response)
```

---

## Tools Integration

### Defining and Using Tools

Tools are standard Python functions decorated with `@tool` from `agent_framework`.

```python
from agent_framework import tool
from typing import Annotated

@tool(description="Get the current time in a specified timezone.")
async def get_current_time(
    timezone: Annotated[str, "The IANA timezone name, e.g., 'America/New_York'."]
) -> str:
    from datetime import datetime
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        return f"The current time in {timezone} is {datetime.now(tz).strftime('%H:%M:%S')}."
    except zoneinfo.ZoneInfoNotFoundError:
        return "Unknown timezone."

# --- Attaching the tool to an agent ---
# from agent_framework import Agent
# from agent_framework.openai import OpenAIChatClient
#
# agent = Agent(
#     client=OpenAIChatClient(),
#     instructions="You can get the current time.",
#     tools=[get_current_time],
# )
# response = await agent.run("What time is it in New York?")
```

### Built-in Azure Tools

The `agent-framework-azure-ai` package provides chat clients and tool wrappers for Azure AI services. Retrieval against Azure AI Search is typically exposed as a `@tool`-decorated function that wraps the `azure-search-documents` SDK (see Recipe 6 in the [recipes page](./microsoft_agent_framework_python_recipes/)).

---

## Structured Output

Force an agent to respond in a specific JSON schema using Pydantic models.

```python
from pydantic import BaseModel, Field
from typing import List

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

class UserProfile(BaseModel):
    """A model to hold structured user information."""
    name: str = Field(description="The user's full name.")
    age: int = Field(description="The user's age.")
    interests: List[str] = Field(description="A list of the user's interests.")

async def extract_structured_data(client: OpenAIChatClient, text: str) -> UserProfile:
    agent = Agent(
        client=client,
        instructions="Extract user profile information from the text provided.",
    )

    # Pass the Pydantic model as the expected response type.
    response = await agent.run(text, response_format=UserProfile)
    return response.value

# --- Usage ---
# text_blob = "My name is Jane Doe, I'm 28, and I love hiking and programming in Python."
# profile = await extract_structured_data(OpenAIChatClient(), text_blob)
# print(profile.model_dump_json(indent=2))
```

For streaming structured output, the same `response_format=` argument works against `agent.run(..., stream=True)` — the framework buffers updates until enough JSON has arrived to validate, then emits the parsed `value` once on the final `AgentResponseUpdate`.

---

## Streaming Responses

The `Agent.run` method returns either an awaitable (`stream=False`, default) or a `ResponseStream[AgentResponseUpdate, AgentResponse]` (`stream=True`). The `ResponseStream` is async-iterable and exposes the assembled final response on `await stream.get_response()` once consumption finishes.

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )

    stream = agent.run("Explain backpressure in 3 short paragraphs.", stream=True)
    async for update in stream:
        # Each update is an AgentResponseUpdate. update.text is the
        # incremental text fragment for this chunk.
        if update.text:
            print(update.text, end="", flush=True)
    print()
    # Optional: get the final assembled AgentResponse, including aggregated tool calls.
    final = await stream.get_response()
    print(f"\n--- finish_reason={final.finish_reasons!r}")


asyncio.run(main())
```

For HITL flows that need to inject an approval response **mid-stream**, the same `ResponseStream` exposes `await stream.send_response(...)` — used for `function_approval_request` events without restarting the run.

---

## Sessions and Conversation History

A `session = agent.create_session()` plus a history provider stores the conversation across turns. By default, `Agent` auto-attaches an `InMemoryHistoryProvider` for sessions that don't have one — fine for in-process bots, but ephemeral.

For durable sessions, swap in `FileHistoryProvider` (one JSONL file per `session_id`):

```python
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

history = FileHistoryProvider(
    storage_path="./sessions",
    skip_excluded=True,        # don't reload messages compaction marked excluded
    store_inputs=True,
    store_outputs=True,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[history],
)

session = agent.create_session(session_id="user-42")        # picks up ./sessions/user-42.jsonl
await agent.run("Continue where we left off.", session=session)
```

`FileHistoryProvider` validates every resolved path against the storage root, so user-supplied `session_id`s can't escape via `../` traversal. Use Redis (`agent-framework-redis`) or Cosmos DB (`agent-framework-azure-cosmos`) providers when you need cross-process safety.

The same class behaves as a write-only audit log when configured with `load_messages=False`:

```python
audit = FileHistoryProvider(
    storage_path="./audit",
    source_id="audit",
    load_messages=False,           # purely a write destination
    store_context_messages=True,   # also capture messages from other providers
)
agent = Agent(client=client, context_providers=[primary_history, audit])
```

### Building a custom history backend

Subclass `HistoryProvider` and implement two methods. Anything that lets you persist messages keyed by `session_id` works — Postgres, S3, even a Notion table.

```python
from agent_framework import HistoryProvider, Message
from collections.abc import Sequence
from typing import Any


class PostgresHistoryProvider(HistoryProvider):
    DEFAULT_SOURCE_ID = "postgres_history"

    def __init__(self, pool, *, source_id: str | None = None, **kwargs) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID, **kwargs)
        self._pool = pool

    async def get_messages(
        self, session_id: str | None, *, state: dict[str, Any] | None = None, **_: Any
    ) -> list[Message]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT payload FROM agent_history WHERE session_id = $1 ORDER BY id",
                session_id,
            )
            return [Message.from_dict(r["payload"]) for r in rows]

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO agent_history (session_id, payload) VALUES ($1, $2)",
                [(session_id, m.to_dict()) for m in messages],
            )
```

The `load_messages`, `store_inputs`, `store_outputs`, and `store_context_messages` flags inherited from `HistoryProvider` work exactly the same as the file-backed implementation — your subclass only needs the two storage methods.

---

## Compaction in 30 lines

Long conversations exceed the model's context window. Compaction strategies decide what stays in the model's view per turn — the source history is preserved.

```python
from agent_framework import (
    Agent,
    CompactionProvider,
    InMemoryHistoryProvider,
    SlidingWindowStrategy,
    ToolResultCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient

history = InMemoryHistoryProvider()

compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
    history_source_id=history.source_id,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    context_providers=[history, compaction],
)

session = agent.create_session()
await agent.run("Run the analysis.", session=session)   # history is compacted between turns
```

Six strategies ship in the box: `TruncationStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `ToolResultCompactionStrategy`, `SummarizationStrategy` (LLM-driven), and `TokenBudgetComposedStrategy`. See the [compaction page](./microsoft_agent_framework_python_compaction/) for trade-offs.

---

## Middleware in 30 lines

Middleware wraps `agent.run(...)` (`AgentMiddleware`), each model call inside the tool loop (`ChatMiddleware`), or each tool invocation (`FunctionMiddleware`).

```python
from agent_framework import Agent, AgentMiddleware, AgentContext, MiddlewareTermination
from agent_framework.openai import OpenAIChatClient


class BudgetGuard(AgentMiddleware):
    def __init__(self, max_runs: int) -> None:
        self.remaining = max_runs

    async def process(self, context: AgentContext, call_next) -> None:
        if self.remaining <= 0:
            raise MiddlewareTermination("budget exhausted")
        self.remaining -= 1
        await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    middleware=[BudgetGuard(max_runs=20)],
)
```

Decorator forms (`@agent_middleware`, `@chat_middleware`, `@function_middleware`) tag plain async functions for the same pipeline. See the [middleware page](./microsoft_agent_framework_python_middleware/) for redaction, retries, and streaming hooks.

---

## Workflows — Graph-Based Orchestration

`WorkflowBuilder` lets you wire agents (and arbitrary executors) into a directed graph that runs in Pregel-style supersteps. Each `Workflow` exposes `.run(message)` (returns `WorkflowRunResult`) and `.run(message, stream=True)` (returns a `ResponseStream` of events).

```python
import asyncio
from agent_framework import Agent, AgentExecutor, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    researcher = Agent(client=client, name="researcher", instructions="Bullet-point findings.")
    writer = Agent(client=client, name="writer", instructions="One-paragraph summary.")

    # AgentExecutor wraps an agent so it can sit inside a workflow graph.
    research_node = AgentExecutor(researcher)
    write_node = AgentExecutor(writer)

    workflow = (
        WorkflowBuilder(start_executor=research_node, name="research-pipeline")
        .add_edge(research_node, write_node)
        .build()
    )

    result = await workflow.run("Quantum sensors in 2026")
    # result is a list[WorkflowEvent]; output events carry yielded data.
    for event in result:
        if event.type == "output":
            print(event.data)


asyncio.run(main())
```

Note that `AgentExecutor` is *only* needed when you want the agent inside a graph. If you pass an `Agent` directly to `WorkflowBuilder(start_executor=agent)`, the framework wraps it for you. Wrapping explicitly gives access to `context_mode`:

- `context_mode="full"` (default) — append the entire prior conversation when chaining.
- `context_mode="last_agent"` — pass only the most recent agent response downstream.
- `context_mode="custom"` — supply a `context_filter` callable to shape the conversation per node.

```python
research_node = AgentExecutor(researcher, context_mode="last_agent")
```

Use `context_mode="last_agent"` when the next agent doesn't need the full chain — keeps token costs predictable on long pipelines.

### Custom executors with `@handler`

Inserting deterministic logic into a workflow is just a function-style executor:

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
    # AgentExecutorResponse.with_text preserves the full conversation chain so
    # the next AgentExecutor still sees the prior history. Returning a plain str
    # via send_message would lose that context.
    await ctx.send_message(response.with_text(upper_text))
    await ctx.yield_output(upper_text)
```

`with_text(...)` matters: if your custom executor sends a plain `str` to the next `AgentExecutor`, only that string lands in the downstream agent's cache and the conversation history is lost. `AgentExecutorResponse.with_text(...)` keeps the message type, so `from_response` is invoked instead of `from_str` and history is preserved.

### Workflow checkpointing

Pass a `CheckpointStorage` to the builder and every superstep saves automatically:

```python
from agent_framework import FileCheckpointStorage, WorkflowBuilder

storage = FileCheckpointStorage("/var/lib/agents/checkpoints")
workflow = (
    WorkflowBuilder(start_executor=research_node, checkpoint_storage=storage, name="research-pipeline")
    .add_edge(research_node, write_node)
    .build()
)

# Resume the latest run after a process restart.
latest = await storage.get_latest(workflow_name="research-pipeline")
if latest:
    result = await workflow.run(checkpoint_id=latest.checkpoint_id)
```

`InMemoryCheckpointStorage`, `FileCheckpointStorage`, the Redis backend, and the Cosmos backend all share the `CheckpointStorage` protocol — six async methods (`save`, `load`, `list_checkpoints`, `delete`, `get_latest`, `list_checkpoint_ids`). Roll your own backend by implementing those six methods and pass it to the builder. See the [checkpointing page](./microsoft_agent_framework_python_checkpointing/) for an S3-backed reference implementation.

### Workflow human-in-the-loop

Inside an executor, call `ctx.request_info(payload, response_type)` to pause the workflow. A matching `@response_handler` on the same executor receives the reply when the caller resumes with `workflow.run(responses={...})`.

```python
from dataclasses import dataclass
from agent_framework import Executor, WorkflowContext, handler, response_handler


@dataclass
class Approval:
    summary: str


class ReviewExecutor(Executor):
    @handler
    async def submit(self, draft: str, ctx: WorkflowContext[str, str]) -> None:
        # Pause and wait for a human to approve the draft.
        await ctx.request_info(Approval(summary=draft[:280]), response_type=bool)

    @response_handler
    async def on_decision(
        self,
        original_request: Approval,
        approved: bool,
        ctx: WorkflowContext[str, str],
    ) -> None:
        await ctx.yield_output("approved" if approved else "rejected")
```

`response_handler` infers the request and response types from the parameter annotations. To skip introspection (when you're working with forward references or want to keep the parameters un-annotated), use the explicit-types form:

```python
@response_handler(request=Approval, response=bool, workflow_output=str)
async def on_decision(self, original_request, approved, ctx):
    await ctx.yield_output("approved" if approved else "rejected")
```

The full HITL loop on the caller side is in the [HITL page](./microsoft_agent_framework_python_hitl/).

---

## Multi-Agent Orchestration Patterns

`agent-framework-orchestrations` ships five fluent builders. Each produces a regular `Workflow`, so checkpointing, streaming, and HITL apply uniformly.

### Sequential — pipeline

```python
from agent_framework_orchestrations import SequentialBuilder

workflow = SequentialBuilder(participants=[researcher, analyst, writer]).build()
result = await workflow.run("Quantum computing in 2026")
print(result.get_outputs()[-1])
```

### Concurrent — fan-out / fan-in

```python
from agent_framework_orchestrations import ConcurrentBuilder

# Default aggregator returns list[Message] from each participant.
workflow = ConcurrentBuilder(participants=[fact_checker, sentiment, summariser]).build()


# Or supply a callback aggregator (sync or async). The return value is the workflow output.
async def stitch(results) -> str:
    return " | ".join(r.agent_response.messages[-1].text for r in results)


workflow = (
    ConcurrentBuilder(participants=[fact_checker, sentiment, summariser])
    .with_aggregator(stitch)
    .build()
)
```

### Handoff — agent-to-agent routing

Triage agent decides which specialist to delegate to. Each participant must be an `Agent` instance because handoff relies on cloning, tool injection, and middleware:

```python
from agent_framework_orchestrations import HandoffBuilder

workflow = (
    HandoffBuilder(participants=[triage, billing, refund, escalation])
    .add_handoff(triage, [billing, refund, escalation])
    .add_handoff(billing, [refund, escalation])
    .build()
)
```

If you skip `add_handoff`, every agent can hand off to every other (mesh topology). Termination is decided by either a built-in heuristic or your own `termination_condition=lambda messages: ...` callable on the builder.

### GroupChat — moderated panel

```python
from agent_framework_orchestrations import GroupChatBuilder

workflow = GroupChatBuilder(participants=[engineer, pm, security]).build()
```

### Magentic — manager + workers + replanning

```python
from agent_framework_orchestrations import MagenticBuilder

workflow = (
    MagenticBuilder(
        participants=[researcher, analyst, writer],
        manager_agent=manager_agent,
        enable_plan_review=True,        # pause for HITL after the initial plan
    )
    .with_human_input_on_stall()        # ask a human when the manager loops
    .build()
)
```

For the full set of optional knobs (intermediate outputs, request-info filters, autonomous mode for handoff, custom selection functions for group chat) see the [orchestration page](./microsoft_agent_framework_python_orchestration/).

---

## MCP Integration

Connect to Model Context Protocol servers as a tool source. Three transports cover the common cases:

```python
import asyncio
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    async with MCPStreamableHTTPTool(
        name="learn",
        url="https://learn.microsoft.com/api/mcp",
        description="Search official Microsoft Learn documentation.",
        request_timeout=30,
    ) as learn:
        agent = Agent(
            client=OpenAIChatClient(),
            instructions="Use the learn tool to answer Microsoft documentation questions.",
            tools=learn,
        )
        response = await agent.run("How does DefaultAzureCredential pick a credential?")
        print(response.text)


asyncio.run(main())
```

For local stdio servers (filesystem, git, SQLite), use `MCPStdioTool(name=..., command=..., args=[...])`. For real-time bidirectional servers, use `MCPWebsocketTool(name=..., url="wss://...")`.

### Per-request headers (multi-tenant auth)

```python
mcp = MCPStreamableHTTPTool(
    name="billing-api",
    url="https://mcp.example.com",
    header_provider=lambda kwargs: {"Authorization": f"Bearer {kwargs['token']}"},
)

await agent.run("What's my balance?", function_invocation_kwargs={"token": user_token})
```

`header_provider` reads from `function_invocation_kwargs` on the outer `agent.run(...)` call — no per-tenant `httpx.AsyncClient` needed. See the [MCP page](./microsoft_agent_framework_python_mcp/) for approval gates, custom result parsers, and the `SupportsMCPTool` protocol for hosted MCP.

---

## Custom Chat Clients

`BaseChatClient` is the abstract parent every first-party client inherits from. Implement one method (`_inner_get_response`) and the framework wraps your code with the tool loop, middleware, telemetry, and serialization:

```python
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any, ClassVar
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Message,
    ResponseStream,
)


class EchoChatClient(BaseChatClient):
    """Test double — echoes the last user message back as the assistant response."""

    OTEL_PROVIDER_NAME: ClassVar[str] = "echo"

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        text = (last_user.text if last_user else "") or "<no input>"

        if stream:

            async def _iter() -> AsyncIterable[ChatResponseUpdate]:
                for token in text.split():
                    yield ChatResponseUpdate(role="assistant", contents=[token + " "])

            return self._build_response_stream(_iter())

        async def _single() -> ChatResponse:
            return ChatResponse(messages=[Message(role="assistant", contents=[text])])

        return _single()


agent = Agent(client=EchoChatClient(), instructions="Echo only.")
response = await agent.run("Hello")
assert response.text == "Hello"
```

Wrap any real client to add caching, request coalescing, or shadow traffic — see the [Advanced Patterns page](./microsoft_agent_framework_python_advanced/#caching-wrapper) for a SHA-256-keyed cache wrapper.

---

## Capability Detection — `Supports*` Protocols

Different providers ship different hosted tools. Feature-detect at runtime via `runtime_checkable` protocols rather than `try/except` on import:

```python
from agent_framework import (
    Agent,
    SupportsCodeInterpreterTool,
    SupportsFileSearchTool,
    SupportsMCPTool,
    SupportsWebSearchTool,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework.anthropic import AnthropicClient


def build_tools(client) -> list:
    tools: list = []
    if isinstance(client, SupportsWebSearchTool):
        tools.append(client.get_web_search_tool())
    if isinstance(client, SupportsFileSearchTool):
        tools.append(client.get_file_search_tool(vector_store_ids=["vs_123"]))
    if isinstance(client, SupportsCodeInterpreterTool):
        tools.append(client.get_code_interpreter_tool())
    if isinstance(client, SupportsMCPTool):
        tools.append(client.get_mcp_tool(name="learn", url="https://learn.microsoft.com/api/mcp"))
    return tools


# OpenAI → web search + file search + code interpreter.
# Anthropic → MCP only.
for client in [OpenAIChatClient(), AnthropicClient()]:
    agent = Agent(client=client, tools=build_tools(client))
```

Same pattern works for `SupportsAgentRun`, `SupportsChatGetResponse`, and `SupportsImageGenerationTool`. See the [Advanced Patterns page](./microsoft_agent_framework_python_advanced/) for the full table.

---

## Production Deployment Cheatsheet

- **Pin sub-packages** rather than the umbrella meta-install — `pip install agent-framework-core agent-framework-openai agent-framework-orchestrations` keeps the dependency tree tight.
- **DefaultAzureCredential** in production; environment-variable fallback in dev. Construct the credential once and reuse it across chat clients.
- **One agent per role**, reused across requests. Sessions are per-conversation. Chat clients own HTTP pools — close them with `async with` at process shutdown.
- **Compaction** — pair an `InMemoryHistoryProvider` (or Redis/Cosmos for cross-process) with a `CompactionProvider` so long-lived sessions stay inside the context window.
- **Checkpointing** — `FileCheckpointStorage` for single-process services; Cosmos / Redis for multi-process workers; custom `CheckpointStorage` (S3, etc.) for cross-cloud.
- **Observability** — call `configure_otel_providers()` once at startup, or `enable_instrumentation()` if you already wire OTel yourself. See the [observability page](./microsoft_agent_framework_python_observability/) for Azure Monitor wiring.
- **HITL durability** — combine HITL request_info with checkpointing so a human can come back hours later in a different process and the workflow resumes exactly where it paused.

---

## Where to go next

| Topic | Page |
|---|---|
| Per-call middleware, retries, redaction | [Middleware](./microsoft_agent_framework_python_middleware/) |
| Six compaction strategies + custom strategies | [Compaction](./microsoft_agent_framework_python_compaction/) |
| Workflow checkpoint backends + S3 example | [Checkpointing](./microsoft_agent_framework_python_checkpointing/) |
| Sequential / Concurrent / Handoff / GroupChat / Magentic | [Orchestration](./microsoft_agent_framework_python_orchestration/) |
| `request_info` + tool approval + plan review | [HITL](./microsoft_agent_framework_python_hitl/) |
| OpenTelemetry traces / metrics / Azure Monitor | [Observability](./microsoft_agent_framework_python_observability/) |
| MCPStdio / HTTP / WebSocket transports | [MCP](./microsoft_agent_framework_python_mcp/) |
| Skills (progressive-disclosure knowledge) | [Skills](./microsoft_agent_framework_python_skills/) |
| BaseChatClient / BaseEmbeddingClient / ContextProvider extension points | [Advanced Patterns](./microsoft_agent_framework_python_advanced/) |

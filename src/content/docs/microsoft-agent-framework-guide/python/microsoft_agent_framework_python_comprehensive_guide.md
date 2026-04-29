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

#### Custom JSON encoders — encrypted history at rest

`FileHistoryProvider` accepts `dumps=` / `loads=` callables. Each one receives a dict (for `dumps`) or text/bytes (for `loads`) and must round-trip cleanly. The hook is the right place to add envelope encryption, schema redaction, or version migration:

```python
import json
import os
from cryptography.fernet import Fernet
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient


# Key management is your problem — pull from KMS, Key Vault, AWS SSM, etc.
fernet = Fernet(os.environ["AF_HISTORY_KEY"].encode())


def encrypted_dumps(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    # Fernet tokens are already URL-safe base64 — no extra encoding needed.
    return fernet.encrypt(body).decode("ascii")


def encrypted_loads(line: str | bytes) -> dict:
    token = line if isinstance(line, bytes) else line.encode("ascii")
    return json.loads(fernet.decrypt(token))


encrypted_history = FileHistoryProvider(
    storage_path="./sessions-encrypted",
    dumps=encrypted_dumps,
    loads=encrypted_loads,
    skip_excluded=True,
)
agent = Agent(client=OpenAIChatClient(), context_providers=[encrypted_history])
```

`FileHistoryProvider` writes one line per message, which is what makes the per-line encrypt/decrypt pattern safe — corruption of one line never tanks the entire session file. Two operational notes:

- **Validate the round-trip in tests.** A buggy `dumps`/`loads` pair will surface as `ValueError("History line N in '<file>' did not deserialize to a mapping.")`. The provider re-raises with the offending line number so failures pinpoint the corrupt entry.
- **Treat the provider as single-host trust boundary.** The path-traversal guards (`session_id` validated against the storage root, encoded fallback for reserved names like `CON`/`NUL` on Windows, striped per-file locks) defend against malicious session ids — but **not** against another process scribbling into the same directory. Use `agent-framework-redis` or `agent-framework-azure-cosmos` for multi-host setups.

#### Selective storage — capture only what you need

The `store_*` flags compose freely. A common pattern is a primary store plus a redacted audit copy:

```python
primary = FileHistoryProvider(
    storage_path="./sessions",
    source_id="primary",
    store_inputs=True,
    store_outputs=True,
    store_context_messages=False,   # don't bloat with retrieved snippets
)

audit = FileHistoryProvider(
    storage_path="./audit",
    source_id="audit",
    load_messages=False,             # never reload — audit is write-only
    store_inputs=True,
    store_outputs=True,
    store_context_messages=True,
    store_context_from={"doc_retriever"},  # only retain retrieval traces
    skip_excluded=False,             # capture compacted messages too — full forensic trail
)

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[doc_retriever, primary, audit],
)
```

`store_context_from` accepts a set of `source_id` strings — only context messages tagged with one of those ids are persisted. Pair with the [advanced page's `ContextProvider` example](./microsoft_agent_framework_python_advanced/#custom-context-provider--contextprovider) so each provider's `source_id` is distinct and your audit log tells you which provider produced each captured message.

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

### Serializing sessions across requests — `AgentSession.to_dict()`

`AgentSession` itself is a lightweight wrapper around a `session_id` and a mutable `state` dict. The history (messages) lives **inside** the session's `state` when you use `InMemoryHistoryProvider` — so `session.to_dict()` captures everything you need to send a session to another worker, store it in Redis between requests, or hand off across a network boundary.

```python
import json

from agent_framework import Agent, AgentSession, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[InMemoryHistoryProvider()],
)

# Turn 1 — serialize after the first turn.
session = agent.create_session(session_id="user-7")
await agent.run("Remember: my favourite colour is teal.", session=session)

snapshot = session.to_dict()
# Persist somewhere durable. The dict is JSON-safe — every value either is
# a primitive or implements SerializationProtocol (e.g. Message.to_dict()).
redis_client.set(f"session:{session.session_id}", json.dumps(snapshot))
```

A separate worker can rehydrate the session and continue:

```python
raw = redis_client.get(f"session:user-7")
restored = AgentSession.from_dict(json.loads(raw))
response = await agent.run("What's my favourite colour?", session=restored)
print(response.text)        # mentions teal — full history is restored
```

Two practical notes:

- `to_dict()` skips `service_session_id` if you didn't set one (provider-side conversation IDs from OpenAI Responses, Anthropic, etc.). When the chat client manages history server-side, persisting only `session_id` + `service_session_id` is enough — no message bodies cross the wire.
- Custom values you put into `session.state` round-trip cleanly **only** if they implement `to_dict()`/`from_dict()` (the framework's `SerializationProtocol`). Strings, ints, floats, bools, `None`, lists, and dicts are passed through unchanged.

For longer-lived agents, prefer a real `HistoryProvider` subclass (Postgres, Redis, Cosmos) over `to_dict()` round-trips — the provider handles incremental writes per turn, so you don't pay to re-serialize the whole conversation on every request.

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

For class-based executors with multiple handlers — and per-instance state that survives across invocations — subclass `Executor` directly:

```python
from agent_framework import Executor, WorkflowContext, handler


class CounterExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="counter")
        self._count = 0

    @handler
    async def tick(self, _: str, ctx: WorkflowContext[str, str]) -> None:
        self._count += 1
        await ctx.send_message(f"count={self._count}")

    @handler
    async def reset(self, _: int, ctx: WorkflowContext[str]) -> None:
        # Distinct input type → distinct handler. The framework dispatches
        # on the runtime type of the message.
        self._count = 0
        await ctx.send_message("reset")
```

The `@handler` decorator infers the input/output types from the parameter annotations. When you need forward references, union types you'd rather not import, or are building executors dynamically, use the **explicit-types** form. **All** types must come from decorator parameters — annotation-based introspection is disabled the moment any explicit param is supplied:

```python
@handler(input=str | int, output=bool, workflow_output=str)
async def handle_data(self, message, ctx):
    # No annotations on message/ctx. Types come from the decorator.
    if isinstance(message, str):
        await ctx.send_message(True)
    await ctx.yield_output(f"saw {type(message).__name__}")


# String forward references resolve against the decorated function's globals.
@handler(input="MyEvent", output="ResponseType")
async def handle_custom(self, message, ctx): ...
```

### Routing patterns — fan-out, fan-in, switch-case

Beyond linear `add_edge`, `WorkflowBuilder` exposes four routing primitives. Pick the one that matches the topology you want.

**Fan-out** — broadcast one source to many targets:

```python
workflow = (
    WorkflowBuilder(start_executor=parser)
    .add_fan_out_edges(parser, [enricher_a, enricher_b, enricher_c])
    .build()
)
```

**Fan-in** — converge many sources onto one target. The target's handler receives the **list** of upstream messages, so its input type must be `list[T]`:

```python
from typing import Never


class Aggregator(Executor):
    @handler
    async def aggregate(
        self,
        results: list[str],          # one entry per fan-in source
        ctx: WorkflowContext[Never, str],
    ) -> None:
        await ctx.yield_output(" | ".join(results))


workflow = (
    WorkflowBuilder(start_executor=parser)
    .add_fan_out_edges(parser, [worker_a, worker_b, worker_c])
    .add_fan_in_edges([worker_a, worker_b, worker_c], Aggregator())
    .build()
)
```

**Switch-case** — first-match routing on a payload predicate. Always include a `Default(...)` to catch the fall-through:

```python
from dataclasses import dataclass
from agent_framework import Case, Default, Executor, WorkflowBuilder, WorkflowContext, handler


@dataclass
class Result:
    score: int


class Evaluator(Executor):
    @handler
    async def evaluate(self, text: str, ctx: WorkflowContext[Result]) -> None:
        await ctx.send_message(Result(score=len(text)))


workflow = (
    WorkflowBuilder(start_executor=Evaluator(id="eval"))
    .add_switch_case_edge_group(
        Evaluator(id="eval"),
        [
            Case(condition=lambda r: r.score > 100, target=long_form_handler),
            Case(condition=lambda r: r.score > 10, target=mid_handler),
            Default(target=short_handler),
        ],
    )
    .build()
)
```

Conditions evaluate top-to-bottom — the first one that returns truthy wins. The `Default` branch fires only if none matched.

**Multi-selection** — like fan-out, but a `selection_func(message, target_ids)` returns the *subset* of targets that should receive each payload:

```python
def select_workers(task, available: list[str]) -> list[str]:
    return available if task.priority == "high" else [available[0]]


workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_multi_selection_edge_group(dispatcher, [worker_a, worker_b], selection_func=select_workers)
    .build()
)
```

Use `add_chain([a, b, c])` as a shortcut for `.add_edge(a, b).add_edge(b, c)` when you have a long linear pipeline.

### Filtering which executors yield outputs — `output_executors`

By default, every executor that calls `ctx.yield_output(...)` contributes to `WorkflowRunResult.get_outputs()`. In a fan-out / fan-in graph that's noisy — you typically only care about the final aggregator. Pass `output_executors=[...]` to the builder to filter:

```python
from agent_framework import WorkflowBuilder

workflow = (
    WorkflowBuilder(
        start_executor=parser,
        name="research-pipeline",
        output_executors=[final_writer],   # only this executor's yields surface
    )
    .add_fan_out_edges(parser, [worker_a, worker_b, worker_c])
    .add_fan_in_edges([worker_a, worker_b, worker_c], final_writer)
    .build()
)

result = await workflow.run("seed text")
print(result.get_outputs())                # contains only final_writer's output
```

Outputs from upstream executors still flow through the graph (they're consumed by the next handler), they just aren't surfaced through the run result. This is the cheapest way to keep `result.get_outputs()` deterministic when many nodes can yield.

### Conditional edges — gate a single connection

`add_edge(source, target, condition=...)` accepts a predicate that runs against the routed message. Useful for "route to specialist only if confidence high enough" patterns without falling back to switch-case:

```python
from agent_framework import WorkflowBuilder

def is_high_confidence(payload) -> bool:
    return getattr(payload, "confidence", 0.0) >= 0.85

workflow = (
    WorkflowBuilder(start_executor=triager)
    .add_edge(triager, fast_responder)                          # always runs
    .add_edge(triager, specialist, condition=is_high_confidence)  # only if confident
    .build()
)
```

The condition is `Callable[[Any], bool | Awaitable[bool]]` — synchronous or async, both work. Returning `False` (or a falsy value) skips the edge silently; the source executor isn't told whether the message was routed.

### Auto-wrapping — pass agents directly to the builder

Every builder method (`add_edge`, `add_fan_out_edges`, `add_fan_in_edges`, `add_switch_case_edge_group`, `add_multi_selection_edge_group`, `add_chain`, plus the `start_executor=` constructor parameter) accepts either an `Executor` or an `Agent`. Agents are auto-wrapped in an `AgentExecutor` once and reused across calls — same agent, same wrapper:

```python
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="...")
writer = Agent(client=client, name="writer", instructions="...")

# No AgentExecutor wrapping needed — the builder handles it.
workflow = (
    WorkflowBuilder(start_executor=researcher, name="research")
    .add_edge(researcher, writer)
    .build()
)
```

Reach for an explicit `AgentExecutor` only when you need a non-default `context_mode` (see above) or you want to give the wrapper a custom `id` that differs from the agent name.

### Visualizing a workflow

`WorkflowViz` ships with the framework — render any built workflow to Mermaid (no extra deps), DOT, or SVG/PNG/PDF (needs `graphviz`):

```python
from agent_framework import WorkflowViz

viz = WorkflowViz(workflow)
print(viz.to_mermaid())            # paste into Markdown
viz.save_svg("workflow.svg")       # needs `pip install graphviz>=0.20.0` + the dot binary
```

Pass `include_internal_executors=True` when you're debugging routing — the diagram then includes the framework's auto-injected glue nodes.

### Nesting a workflow inside another with `WorkflowExecutor`

A built workflow is just an `Executor` with extra type metadata — wrap it in a `WorkflowExecutor` and it becomes a single node inside a larger workflow. Useful for building reusable building blocks: a "draft → review → approve" sub-pipeline that you can drop into multiple parents.

```python
from agent_framework import (
    Agent,
    AgentExecutor,
    WorkflowBuilder,
    WorkflowExecutor,
)
from agent_framework.openai import OpenAIChatClient


client = OpenAIChatClient()

# Inner workflow: draft + critique
drafter = AgentExecutor(Agent(client=client, name="drafter"))
critic = AgentExecutor(Agent(client=client, name="critic"))
inner = (
    WorkflowBuilder(start_executor=drafter, name="draft-and-critique")
    .add_edge(drafter, critic)
    .build()
)

# Outer workflow: the inner pipeline becomes a single node, followed by a publisher.
publisher = AgentExecutor(Agent(client=client, name="publisher"))
outer = (
    WorkflowBuilder(
        start_executor=WorkflowExecutor(inner, id="draft-pipeline"),
        name="publish-pipeline",
    )
    .add_edge(WorkflowExecutor(inner, id="draft-pipeline"), publisher)
    .build()
)
```

Two flags shape how the inner workflow's outputs reach the parent:

- `allow_direct_output=False` (default) — outputs from the inner workflow are forwarded to the next executor as messages. Use this when the next executor in the parent wants to react to the sub-pipeline's result.
- `allow_direct_output=True` — outputs are yielded directly into the parent workflow's event stream. Use this when the inner workflow's output **is** the outer workflow's output and you don't have a downstream executor.

Sub-workflow request_info events propagate by default as `SubWorkflowRequestMessage` so a parent executor can intercept and respond locally; set `propagate_request=True` if you want the original `WorkflowEvent` to bubble out to the outer caller (useful when the same human handles both inner and outer HITL gates).

`WorkflowViz` walks the composition tree automatically — a multi-level nest renders as Mermaid clusters that mirror the call hierarchy.

### Workflow event types — what comes out of `workflow.run(stream=True)`

`workflow.run(message, stream=True)` yields `WorkflowEvent` objects. The `type` discriminator tells you what kind of event it is; lifecycle, executor, and orchestration events all flow through the same stream:

| `event.type` | Useful fields | Emitted by |
|---|---|---|
| `started` | — | Once per run, when the workflow begins |
| `status` | `event.state` (`STARTED`, `IN_PROGRESS`, `IDLE`, `IDLE_WITH_PENDING_REQUESTS`, `FAILED`, `CANCELLED`) | Lifecycle transitions |
| `output` | `event.executor_id`, `event.data` | Executor called `ctx.yield_output(...)` |
| `data` | `event.executor_id`, `event.data` (typed payload, e.g. `AgentResponse`) | Executor emitted typed data (e.g. an `AgentExecutor` finishing) |
| `request_info` | `event.request_id`, `event.source_executor_id`, `event.data` | Executor called `ctx.request_info(...)` — caller must reply |
| `superstep_started` / `superstep_completed` | `event.iteration` | Pregel-style superstep boundaries |
| `executor_invoked` / `executor_completed` / `executor_failed` | `event.executor_id`, `event.details` (on failure) | Per-executor lifecycle |
| `executor_bypassed` | `event.executor_id` | Replay hit a cached result |
| `warning` / `error` | `event.data` (str/Exception) | Diagnostic — non-fatal |
| `failed` | `event.details` (`WorkflowErrorDetails`) | Workflow terminated with an unrecoverable error |
| `group_chat` / `handoff_sent` / `magentic_orchestrator` | `event.data` (typed orchestrator payload) | Specific orchestration patterns |

A typical consumer pattern:

```python
async for event in workflow.run(message, stream=True):
    if event.type == "output":
        print(f"[{event.executor_id}] {event.data}")
    elif event.type == "request_info":
        # Pause for human input — see the HITL section above.
        responses[event.request_id] = await ask_human(event.data)
    elif event.type == "executor_failed":
        print(f"FAIL {event.executor_id}: {event.details.error_type}: {event.details.message}")
    elif event.type == "status" and event.state == "IDLE":
        break
```

The factory methods (`WorkflowEvent.output(...)`, `WorkflowEvent.status(...)`, etc.) are what executors and the runtime use internally — you almost never construct events yourself, but the discriminator pattern means a single `for event in result:` loop handles every signal the framework can produce.

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

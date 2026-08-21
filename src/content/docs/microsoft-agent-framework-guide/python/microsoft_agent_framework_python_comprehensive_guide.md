---
title: "Microsoft Agent Framework Python - Comprehensive Technical Guide"
description: "Comprehensive technical guide for the Microsoft Agent Framework on Python. Verified against agent-framework 1.14.0 — chat clients, tools, sessions, middleware, MCP, skills, workflows, long-term memory, evaluation, security (FIDES), file access/memory providers, workflow visualization, and observability."
framework: microsoft-agent-framework
language: python
---

Latest verified release: 1.14.0 | Python 3.10+
# Microsoft Agent Framework Python - Comprehensive Technical Guide

**Framework Version:** 1.14.0 (`agent-framework` and `agent-framework-core`)
**Target Platform:** Python 3.10+
**Quick check:** `pip index versions agent-framework`

---

> **API reference (verified against `agent-framework==1.14.0`).**
>
> - **Package name / import root:** `agent_framework` (underscores). Install with `pip install agent-framework`.
> - **Agent classes:** `Agent` (full stack with middleware + telemetry), `RawAgent` (same interface, skips the middleware/telemetry wrappers for latency-sensitive paths), `BaseAgent` (abstract base for custom subclasses).
> - **Chat clients:** `agent_framework.foundry.FoundryChatClient`, `agent_framework.openai.OpenAIChatClient`, `agent_framework.anthropic.AnthropicClient`, plus Bedrock / Ollama in the `1.0.0b` provider line.
> - **Tool decorator:** `@tool` from `agent_framework`.
> - **Multi-turn state:** `session = agent.create_session()`, then `await agent.run(prompt, session=session)`.
> - **Workflows:** `WorkflowBuilder` (with `.add_edge` / `.add_fan_in_edges` / `.add_fan_out_edges` / `.add_chain` / `.add_multi_selection_edge_group`) and the experimental `@workflow` / `@step` functional API from `agent_framework`. Visualise any workflow with `WorkflowViz`.
> - **Long-term memory (experimental):** `MemoryStore` + `MemoryContextProvider` from `agent_framework`. File-backed memory: `FileMemoryProvider` (direct import: `from agent_framework._harness._file_memory import FileMemoryProvider` — not yet re-exported from `agent_framework` in 1.14.0).
> - **File access (experimental):** `FileAccessProvider` + `AgentFileStore` — shared, persistent CRUD/grep access for agents across sessions.
> - **Agent mode (experimental):** `AgentModeProvider` — plan/execute mode switching, persisted in session state.
> - **Security (experimental — FIDES):** `agent_framework.security` — `LabelTrackingFunctionMiddleware`, `PolicyEnforcementFunctionMiddleware`, `SecureMCPToolProxy`, `SecureAgentConfig` for prompt-injection defence.
> - **Settings:** `load_settings` + `SecretString` — `TypedDict`-based settings from env vars, `.env` files, and overrides with masked secret values.
> - **Evaluation:** `LocalEvaluator`, `evaluate_agent`, `evaluate_workflow`, `AgentEvalConverter` from `agent_framework`.
> - **Declarative YAML agents (beta):** `AgentFactory` / `WorkflowFactory` from `agent_framework.declarative`.

---

## What's new in 1.14.0

| Area | What changed |
|---|---|
| **Workflow events** | `WorkflowEvent[DataT]` unified event bus with factory methods (`started`, `status`, `failed`, `executor_invoked`, `executor_bypassed`, etc.). Replaces ad-hoc callbacks. `WorkflowEvent.emit()` deprecated — use `ctx.yield_output()`. |
| **Workflow visualization** | `WorkflowViz` — export any `Workflow` to Mermaid, DOT, SVG, PNG, or PDF. No extra dependencies for DOT/Mermaid. |
| **File access provider** | `FileAccessProvider` + `AgentFileStore` hierarchy (`InMemoryAgentFileStore`, `FileSystemAgentFileStore`). 7 agent-facing tools (write, read, delete, ls, grep, replace, replace_lines). Configurable approval modes. |
| **File memory provider** | `FileMemoryProvider` — session-scoped file-based memory with descriptions index (`memories.md`), `scope` override for cross-session sharing. |
| **Agent mode** | `AgentModeProvider` — plan/execute modes (fully customisable), `get_agent_mode` / `set_agent_mode` helpers for external orchestrators. |
| **Settings** | `SecretString` replaces `pydantic.SecretStr` — masked repr, `get_secret_value()` shim. `load_settings` — TypedDict-based config from env/dotenv with `required_fields` validation. |
| **Compaction** | `ToolResultCompactionStrategy` (replace old tool-call groups with summary messages) and `TokenBudgetComposedStrategy` (pipeline multiple strategies under a token budget cap). |
| **FIDES security** | `agent_framework.security` promoted to public experimental: `LabelTrackingFunctionMiddleware` (3-tier label propagation), `PolicyEnforcementFunctionMiddleware` (block/approve on violation), `SecureMCPToolProxy` (local MCP enforcement), `SecureAgentConfig` (all-in-one context provider). |
| **Workflow evaluation** | `evaluate_workflow` — post-hoc or run+evaluate, per-agent `sub_results` breakdown. `AgentEvalConverter` — message/tool conversion to Foundry evaluator format. |

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

# 4. Optional extras (1.14.0+)
pip install agent-framework[security]      # FIDES prompt-injection defence
pip install agent-framework[evals]         # Evaluation tooling (evaluate_workflow, AgentEvalConverter)
pip install "graphviz>=0.20.0"             # WorkflowViz SVG/PNG/PDF export (also needs system graphviz)
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

**2. Loading Configuration (1.14.0+ preferred — `load_settings`):**

Use the built-in `load_settings` helper, which reads from env vars, an optional `.env` file, and explicit overrides — no extra library needed.

```python
# config.py
from typing import TypedDict
from agent_framework import load_settings, SecretString

class AzureSettings(TypedDict, total=False):
    endpoint: str | None
    api_key: str | None          # plain str in the TypedDict; wrap in SecretString after loading
    deployment_name: str | None

settings = load_settings(
    AzureSettings,
    env_prefix="AZURE_OPENAI_",         # reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, etc.
    env_file_path=".env",               # optional, but the file must exist if this path is provided
    required_fields=["endpoint", "deployment_name"],
)

AZURE_OPENAI_ENDPOINT = settings["endpoint"]
# Only wrap in SecretString when a value is actually present — avoids masking a missing key
AZURE_OPENAI_KEY = SecretString(settings["api_key"]) if settings.get("api_key") else None
AZURE_OPENAI_DEPLOYMENT = settings["deployment_name"]
```

Alternatively, use `python-dotenv` + `os.getenv` for simpler cases:

```python
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

### Agent types — `Agent`, `RawAgent`, and `BaseAgent`

The Python package ships three agent types in `agent_framework`:

| Class | When to use |
|---|---|
| `Agent` | Default — full middleware + telemetry stack. Use for all production agents. |
| `RawAgent` | Same `__init__` and `run()` signature as `Agent`, but skips the middleware and telemetry layers. For latency-sensitive inner loops, test harnesses, and scenarios where you control the full pipeline yourself. |
| `BaseAgent` | Abstract base class for custom agent subclasses. Provides the minimal interface without the built-in layers. |

Both `Agent` and `RawAgent` behave the same way based on how you invoke them:

- **Stateless / single-turn** — call `await agent.run(prompt)` without a session. Each call is independent; no conversation history persists.
- **Stateful / multi-turn** — attach a session (`session = agent.create_session()`) and pass it to each `agent.run(prompt, session=session)` call. The session's history provider (in-memory by default) accumulates turns so follow-ups have context.

```python
import asyncio
from agent_framework import Agent, RawAgent
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Standard agent — middleware and telemetry included
agent = Agent(client=client, instructions="You are a helpful assistant.")

# Raw agent — same API, thinner stack (no middleware, no OTel)
raw_agent = RawAgent(client=client, instructions="You are a low-latency classifier.")

async def main() -> None:
    response = await agent.run("Explain async/await in Python.")
    print(response.text)

    label = await raw_agent.run("Classify: 'I need a refund' → billing|tech|other")
    print(label.text)

asyncio.run(main())
```

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

#### `Agent.__init__` parameter reference

All parameters except `client` are optional keyword arguments.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client` | `SupportsChatGetResponse` | — | **Required.** Chat client (`OpenAIChatClient`, `FoundryChatClient`, etc.). |
| `instructions` | `str \| None` | `None` | System prompt prepended to every conversation. |
| `id` | `str \| None` | `None` | Stable identifier for this agent instance. Used in workflow graphs and telemetry. Auto-generated if omitted. |
| `name` | `str \| None` | `None` | Human-readable name surfaced in multi-agent event streams (`update.author_name`). |
| `description` | `str \| None` | `None` | Short description shown to an orchestrating LLM (e.g. in `HandoffBuilder`) to help it decide when to route to this agent. |
| `tools` | `Sequence[tool] \| None` | `None` | Default tools available on every run. Can be overridden per-call via `agent.run(..., tools=...)`. |
| `default_options` | `ChatOptions \| None` | `None` | Default model options (`model`, `temperature`, `max_tokens`, etc.) applied to every call. Per-call `options=` overrides these at the field level. |
| `context_providers` | `Sequence[ContextProvider] \| None` | `None` | Ordered list of `ContextProvider` instances (history, memory, skills, …). Providers run `before_run` / `after_run` hooks and can inject messages. |
| `middleware` | `Sequence[MiddlewareTypes] \| None` | `None` | Ordered middleware chain for the agent run. Only applicable to `Agent` (skipped by `RawAgent`). |
| `compaction_strategy` | `CompactionStrategy \| None` | `None` | Default history compaction strategy. Applied on every run before the LLM call; can be overridden per-call. |
| `tokenizer` | `TokenizerProtocol \| None` | `None` | Tokenizer used by compaction strategies to count tokens. Falls back to a fast estimate if `None`. |
| `require_per_service_call_history_persistence` | `bool` | `False` | When `True`, history is persisted after **every** inner LLM call (including tool-call round trips), not just at end-of-turn. Useful when a long chain of tool calls must survive a crash mid-turn. |
| `additional_properties` | `MutableMapping[str, Any] \| None` | `None` | Arbitrary key-value bag attached to the agent instance. Middleware and context providers can read these values via `agent.additional_properties`. |

**Full constructor example:**

```python
import asyncio
from agent_framework import Agent, SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient
from tiktoken import encoding_for_model

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="You are a senior code reviewer.",
    id="code-reviewer-1",
    name="CodeReviewer",
    description="Reviews code diffs and flags issues.",
    tools=[search_codebase, run_tests],
    default_options={"temperature": 0.2, "max_tokens": 2048},
    compaction_strategy=SlidingWindowStrategy(keep_last_groups=20),
    tokenizer=encoding_for_model("gpt-4o"),
    require_per_service_call_history_persistence=True,
    additional_properties={"team": "platform", "tier": "internal"},
)
```

#### `Agent.run()` parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `str \| list[Message] \| None` | `None` | Input message(s). A plain string is treated as a user message. |
| `stream` | `bool` | `False` | Return a `ResponseStream` for token-by-token output instead of an awaitable. |
| `session` | `AgentSession \| None` | `None` | Session carrying conversation history. Omit for stateless single-turn calls. |
| `tools` | `Sequence[tool] \| None` | `None` | Per-call tool list. **Replaces** (not extends) the agent's default `tools`. |
| `options` | `ChatOptions \| None` | `None` | Per-call model options (`model`, `temperature`, `max_tokens`, `response_format`, …). Merged on top of `default_options`. |
| `compaction_strategy` | `CompactionStrategy \| None` | `None` | Per-call override for the agent's default compaction strategy. |
| `tokenizer` | `TokenizerProtocol \| None` | `None` | Per-call tokenizer override used by the compaction strategy. |
| `function_invocation_kwargs` | `Mapping[str, Any] \| None` | `None` | Extra kwargs forwarded to every tool/resource/script callable that accepts `**kwargs`. Useful for injecting request-scoped data (tenant ID, auth token, trace ID) without globals. |
| `client_kwargs` | `Mapping[str, Any] \| None` | `None` | Extra kwargs forwarded directly to the underlying chat client call. Use for provider-specific params not exposed by `ChatOptions`. |

**Per-call override example:**

```python
import asyncio
from agent_framework import Agent, TruncationStrategy
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="You are a helpful assistant.",
    tools=[search_docs],
    default_options={"temperature": 0.7},
)

async def main():
    # Override model and temperature for a single classification call
    label = await agent.run(
        "Classify the sentiment: 'The product is amazing!'",
        options={"model": "gpt-4o-mini", "temperature": 0.0},
        tools=[],                          # no tools needed for classification
        compaction_strategy=TruncationStrategy(keep_last_groups=5),
        function_invocation_kwargs={"tenant_id": "acme", "user_id": "u-42"},
    )
    print(label.text)

asyncio.run(main())
```

### Agent Lifecycle

The chat client owns the underlying HTTP session and credentials; when the client supports it, use `async with` to close those resources deterministically. Agents themselves are cheap to construct — you typically build one per role and reuse it across requests. Sessions are per-conversation and hold `ChatHistoryProvider` state (in-memory by default); create a new session per user or request.

---

## Multi-Agent Systems

### Orchestration Patterns

`agent-framework-orchestrations` provides five high-level builders that cover the most common topologies. Each builder returns a standard `Workflow` object, so checkpointing, streaming, and HITL work uniformly across all patterns.

| Pattern | Builder | When to use |
|---|---|---|
| Sequential | `SequentialBuilder` | Document pipeline — research → analyse → write |
| Concurrent | `ConcurrentBuilder` | Parallel opinions aggregated once |
| Handoff | `HandoffBuilder` | Support triage routed to specialists |
| GroupChat | `GroupChatBuilder` | Moderated panel, LLM or code-driven speaker selection |
| Magentic | `MagenticBuilder` | Open-ended research with replanning |

### Example: Sequential pipeline

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import SequentialBuilder

client = OpenAIChatClient()

researcher = Agent(client=client, name="researcher",
                   instructions="Return concise bullet-point facts on the topic.")
analyst    = Agent(client=client, name="analyst",
                   instructions="Synthesise the facts into an analysis.")
writer     = Agent(client=client, name="writer",
                   instructions="Write a polished one-paragraph summary.")

workflow = SequentialBuilder(participants=[researcher, analyst, writer]).build()

async def main() -> None:
    result = await workflow.run("Advances in post-quantum cryptography")
    print(result.get_outputs()[-1].text)

asyncio.run(main())
```

### Example: Concurrent opinions

```python
from agent_framework_orchestrations import ConcurrentBuilder

# All three agents receive the same prompt and run in parallel.
# The default aggregator returns one assistant message per participant.
workflow = ConcurrentBuilder(participants=[researcher, analyst, writer]).build()

# Custom aggregator — join responses with a separator.
async def stitch(results) -> str:
    return "\n---\n".join(r.agent_response.text for r in results)

workflow_custom = (
    ConcurrentBuilder(participants=[researcher, analyst, writer])
    .with_aggregator(stitch)
    .build()
)
```

### Example: Handoff routing

```python
from agent_framework_orchestrations import HandoffBuilder

triage    = Agent(client=client, name="triage",
                  instructions="Classify the request and hand off to billing or technical.")
billing   = Agent(client=client, name="billing",
                  instructions="Resolve billing questions.",
                  description="Handles invoices, refunds, plan changes.")
technical = Agent(client=client, name="technical",
                  instructions="Resolve technical questions.",
                  description="Handles bugs, API errors, outages.")

workflow = (
    HandoffBuilder(participants=[triage, billing, technical])
    .add_handoff(triage, [billing, technical])
    .build()
)

result = await workflow.run("My card was charged twice last month.")
```

For the full set of knobs — `with_request_info`, `with_autonomous_mode`, `enable_plan_review`, custom selection functions, etc. — see the [Multi-Agent Orchestration page](./microsoft_agent_framework_python_orchestration/).

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

### Inspecting an `AgentResponseUpdate`

Each chunk is a fully-typed `AgentResponseUpdate` carrying `contents`, `role`, `author_name`, `agent_id`, `response_id`, `message_id`, `created_at`, `finish_reason`, and `continuation_token`. A few attributes are particularly useful when building richer UIs over the raw stream:

```python
async for update in agent.run("Plan tomorrow's release.", stream=True):
    # 1. Plain text fragment (None for non-text chunks like tool calls).
    if update.text:
        ui.append_text(update.text)

    # 2. In multi-agent runs, `author_name` and `agent_id` distinguish which
    #    agent emitted the chunk so you can colour-code it in the UI.
    if update.author_name:
        ui.set_speaker(update.author_name)

    # 3. HITL approvals surface as Content items inside the update — there's
    #    a property that filters them out for you.
    for request in update.user_input_requests:
        await approval_queue.put((update.response_id, request))

    # 4. The `finish_reason` lands on the **final** update of a streamed run.
    if update.finish_reason is not None:
        ui.mark_complete(update.finish_reason)
```

### Persisting and replaying updates

`AgentResponseUpdate` is a `SerializationMixin` dataclass — round-trips through `to_dict()` / `from_dict()` and `to_json()` / `from_json()`. Useful for buffering chunks to a queue, replaying them in tests, or shipping them over a websocket without the framework on the receiving end:

```python
from agent_framework import AgentResponseUpdate

# Persist each chunk as it arrives
chunks: list[str] = []
async for update in agent.run("Hello", stream=True):
    chunks.append(update.to_json())

# Later — restore the exact same updates
restored = [AgentResponseUpdate.from_json(line) for line in chunks]
```

For non-streaming consumers that received a chunked feed, rebuild a single `AgentResponse` from the buffer:

```python
from agent_framework import AgentResponse, AgentResponseUpdate

updates = [AgentResponseUpdate.from_json(line) for line in chunks]
final = AgentResponse.from_updates(updates)
print(final.text)            # joined text
print(final.user_input_requests)
```

When a Pydantic schema is configured for structured output, pass `output_format_type=` and the assembled response lazily validates `final.value`:

```python
from pydantic import BaseModel
from agent_framework import AgentResponse


class ReleasePlan(BaseModel):
    version: str
    date: str


final = AgentResponse.from_updates(updates, output_format_type=ReleasePlan)
plan: ReleasePlan = final.value      # validated on first access
```

For a streaming source, the async equivalent `AgentResponse.from_update_generator(stream)` consumes an async iterator and returns a single `AgentResponse` — handy when you want to forward a streaming provider's output to a non-streaming caller without dropping tool calls or the `finish_reason`.

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

### Routing patterns — fan-out, fan-in, switch-case, multi-selection, chain shortcut

Beyond linear `add_edge`, `WorkflowBuilder` exposes five routing primitives. Pick the one that matches the topology you want.

**Fan-out** — broadcast one source to many targets concurrently:

```python
# parser, enricher_a, enricher_b, enricher_c are Executor instances
workflow = (
    WorkflowBuilder(start_executor=parser)
    .add_fan_out_edges(parser, [enricher_a, enricher_b, enricher_c])
    .build()
)
```

**Fan-in** — converge many sources onto one target. The target's handler receives the **list** of upstream messages in one call, so its input type must be `list[T]`:

```python
from typing import NoReturn
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


class Aggregator(Executor):
    @handler
    async def aggregate(
        self,
        results: list[str],          # one entry per fan-in source
        ctx: WorkflowContext[NoReturn, str],
    ) -> None:
        combined = " | ".join(results)
        await ctx.yield_output(combined)


# parser, worker_a, worker_b, worker_c are Executor instances
workflow = (
    WorkflowBuilder(start_executor=parser)
    .add_fan_out_edges(parser, [worker_a, worker_b, worker_c])
    .add_fan_in_edges([worker_a, worker_b, worker_c], Aggregator(id="agg"))
    .build()
)
```

**Switch-case** — first-match routing on a payload predicate. Always include a `Default(...)` to catch the fall-through. Conditions are evaluated top-to-bottom; the first truthy match wins.

```python
import asyncio
from dataclasses import dataclass
from typing import NoReturn
from agent_framework import (
    Case, Default, Executor,
    WorkflowBuilder, WorkflowContext, executor, handler,
)


@dataclass
class ScoredText:
    text: str
    score: int


class Scorer(Executor):
    """Assign a word-count score to input text."""
    @handler
    async def score(self, text: str, ctx: WorkflowContext[ScoredText]) -> None:
        await ctx.send_message(ScoredText(text=text, score=len(text.split())))


@executor(id="long-handler")
async def long_handler(payload: ScoredText, ctx: WorkflowContext[NoReturn, str]) -> None:
    await ctx.yield_output(f"[LONG] {payload.text[:40]}…")


@executor(id="short-handler")
async def short_handler(payload: ScoredText, ctx: WorkflowContext[NoReturn, str]) -> None:
    await ctx.yield_output(f"[SHORT] {payload.text}")


scorer = Scorer(id="scorer")

workflow = (
    WorkflowBuilder(start_executor=scorer)
    .add_switch_case_edge_group(
        scorer,
        cases=[
            Case(condition=lambda p: p.score > 50, target=long_handler),
            Default(target=short_handler),
        ],
    )
    .build()
)


async def main() -> None:
    short_result = await workflow.run("Hello world")
    print(short_result.get_outputs()[-1])   # [SHORT] Hello world

    long_input = " ".join(["word"] * 60)
    long_result = await workflow.run(long_input)
    print(long_result.get_outputs()[-1])    # [LONG] word word word…


asyncio.run(main())
```

**Multi-selection** — like fan-out, but a `selection_func(message, target_ids) -> list[str]` chooses *which subset* of targets receives each message. Use this when routing logic depends on the message payload at runtime (e.g. high-priority tasks use all workers; low-priority tasks use only one):

```python
import asyncio
from dataclasses import dataclass
from typing import NoReturn
from agent_framework import (
    Executor, FunctionExecutor, WorkflowBuilder,
    WorkflowContext, executor, handler,
)


@dataclass
class Task:
    description: str
    priority: str   # "high" | "low"


class Dispatcher(Executor):
    @handler
    async def dispatch(self, raw: str, ctx: WorkflowContext[Task]) -> None:
        priority = "high" if "urgent" in raw.lower() else "low"
        await ctx.send_message(Task(description=raw, priority=priority))


@executor(id="specialist-a")
async def specialist_a(task: Task, ctx: WorkflowContext[NoReturn, str]) -> None:
    await ctx.yield_output(f"[A] handled: {task.description}")


@executor(id="specialist-b")
async def specialist_b(task: Task, ctx: WorkflowContext[NoReturn, str]) -> None:
    await ctx.yield_output(f"[B] handled: {task.description}")


def route_by_priority(task: Task, target_ids: list[str]) -> list[str]:
    """Send high-priority tasks to ALL workers; low-priority to the first only."""
    return target_ids if task.priority == "high" else target_ids[:1]


dispatcher = Dispatcher(id="dispatcher")

workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_multi_selection_edge_group(
        dispatcher,
        targets=[specialist_a, specialist_b],
        selection_func=route_by_priority,
    )
    .build()
)


async def main() -> None:
    # Low-priority — only specialist_a runs
    result = await workflow.run("Fix the login bug")
    print(result.get_outputs())   # ['[A] handled: Fix the login bug']

    # High-priority — both run concurrently
    result = await workflow.run("Urgent: production is down")
    print(result.get_outputs())   # ['[A] handled: ...', '[B] handled: ...']


asyncio.run(main())
```

**Chain shortcut** — `add_chain([a, b, c])` is equivalent to `.add_edge(a, b).add_edge(b, c)`. Use it for long linear pipelines:

```python
# parser, enricher, writer are Executor instances
workflow = WorkflowBuilder(start_executor=parser).add_chain([parser, enricher, writer]).build()
```

### Filtering which executors yield outputs — `output_from` / `intermediate_output_from`

By default, every executor that calls `ctx.yield_output(...)` contributes to `WorkflowRunResult.get_outputs()`. In a fan-out / fan-in graph that's noisy — you typically only care about the final aggregator. Use `output_from=` on the builder to filter:

```python
from agent_framework import WorkflowBuilder

workflow = (
    WorkflowBuilder(
        start_executor=parser,
        name="research-pipeline",
        output_from=[final_writer],          # only this executor's yields surface in get_outputs()
    )
    .add_fan_out_edges(parser, [worker_a, worker_b, worker_c])
    .add_fan_in_edges([worker_a, worker_b, worker_c], final_writer)
    .build()
)

result = await workflow.run("seed text")
print(result.get_outputs())                  # contains only final_writer's output
```

Use `intermediate_output_from=` when you want some intermediate nodes visible separately from the primary outputs — for example, to surface per-worker results alongside the aggregator's final answer:

```python
workflow = (
    WorkflowBuilder(
        start_executor=parser,
        name="research-pipeline",
        output_from=[final_writer],
        intermediate_output_from=[worker_a, worker_b, worker_c],  # also visible, labelled intermediate
    )
    .add_fan_out_edges(parser, [worker_a, worker_b, worker_c])
    .add_fan_in_edges([worker_a, worker_b, worker_c], final_writer)
    .build()
)

result = await workflow.run("seed text")
primary = result.get_outputs()              # final_writer's output only
intermediates = result.get_intermediate_outputs()  # worker_a/b/c outputs
```

Outputs from unfiltered executors still flow through the graph (consumed by the next handler) — they simply aren't surfaced via the run result. This keeps `result.get_outputs()` deterministic when many nodes can yield.

> **Deprecated:** The old `output_executors=[...]` parameter still works in 1.5.0 but is superseded by `output_from=`. Prefer `output_from=` in new code.

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

### Exposing a workflow as an agent — `Workflow.as_agent()`

Every `Workflow` has an `as_agent(name=..., description=..., context_providers=...)` method that returns a `WorkflowAgent`. The wrapper satisfies `SupportsAgentRun`, so the workflow drops into anywhere an `Agent` is expected — multi-agent orchestrations, `Agent.as_tool()` chains, FastAPI routes, etc.

```python
from agent_framework import Agent, AgentExecutor, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Inner pipeline: classify → resolve.
classifier = AgentExecutor(Agent(client=client, name="classifier", instructions="Tag the message."))
resolver = AgentExecutor(Agent(client=client, name="resolver", instructions="Answer."))

triage = (
    WorkflowBuilder(start_executor=classifier, name="support-triage")
    .add_edge(classifier, resolver)
    .build()
)

# Wrap the whole graph as an agent — same interface as a single-LLM Agent.
triage_agent = triage.as_agent(
    name="support_triage",
    description="Classifies a support ticket and produces a resolution.",
)

# Drop it into a higher-level supervisor as a tool.
supervisor = Agent(
    client=client,
    name="supervisor",
    instructions="Route messages to specialised tools.",
    tools=[triage_agent.as_tool()],
)

response = await supervisor.run("My laptop won't charge — please help.")
print(response.text)
```

A few facts that aren't obvious from the signature alone:

- The wrapper streams `WorkflowEvent` objects under the hood and surfaces them as `AgentResponseUpdate` chunks when called with `stream=True`. Pending HITL requests inside the workflow surface as `Content` items with `user_input_request` set, so the same UI code that handles per-tool approval handles workflow-level HITL too.
- `context_providers=` on `as_agent()` attaches the providers to the wrapper — they see the *outer* `Agent.run` calls, not the inner workflow's executors.
- Workflow state is preserved across `agent.run(...)` calls (the same workflow instance is reused). To get a fresh run, build a new `Workflow` and call `as_agent` again.

### Exposing an agent as an MCP server — `Agent.as_mcp_server()`

`RawAgent` (and `Agent`, which inherits it) can expose itself as an **MCP server**. Any MCP-compatible client — another agent using `MCPStreamableHTTPTool`, a third-party tool, or VS Code Copilot — can then invoke it as a tool. This is how you publish a specialist agent for use outside your Python process without building a separate REST API:

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from mcp.server.stdio import stdio_server


@tool
def search_inventory(sku: str) -> str:
    """Return real-time stock count for a SKU."""
    return f"SKU {sku}: 142 units in stock"


inventory_agent = Agent(
    client=OpenAIChatClient(),
    name="inventory-agent",
    instructions="You are an inventory assistant. Use search_inventory to answer stock questions.",
    tools=[search_inventory],
)

# as_mcp_server() returns mcp.server.lowlevel.Server — it is transport-agnostic.
# Wire it to a transport by calling server.run(read_stream, write_stream, init_options).
mcp_server = inventory_agent.as_mcp_server(
    server_name="InventoryAgent",
    version="1.0.0",
    instructions="Call this agent to query real-time inventory levels.",
)

# Option 1 — stdio transport (CLI tools, VS Code extensions, local testing)
async def run_stdio():
    init_options = mcp_server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, init_options)

asyncio.run(run_stdio())

# Option 2 — streamable HTTP transport (production)
# from mcp.server.streamable_http import StreamableHTTPServerTransport
# transport = StreamableHTTPServerTransport(mcp_session_id=None)
# init_options = mcp_server.create_initialization_options()
# async def run_http():
#     async with transport.connect() as (read_stream, write_stream):
#         await mcp_server.run(read_stream, write_stream, init_options)
# # transport.handle_request is an ASGI callable — mount it in Starlette / FastAPI:
# from starlette.applications import Starlette
# from starlette.routing import Route
# app = Starlette(routes=[Route("/mcp", transport.handle_request, methods=["GET", "POST"])])
# # uvicorn.run(app, host="0.0.0.0", port=8080)
```

Consuming the published agent from another agent in the same or a different process:

```python
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

# The inventory agent is now running at http://localhost:8080
async with MCPStreamableHTTPTool(
    name="inventory",
    url="http://localhost:8080/mcp",
    description="Remote inventory agent",
) as inventory_mcp:
    orchestrator = Agent(
        client=OpenAIChatClient(),
        instructions="You coordinate warehouse operations.",
        tools=inventory_mcp,
    )
    response = await orchestrator.run("Do we have enough SKU-9921 for the weekend sale?")
    print(response.text)
```

`as_mcp_server()` parameters:

| Parameter | Default | Effect |
|---|---|---|
| `server_name` | `"Agent"` | Prefix for the MCP tool name exposed to clients (`"<server_name>_run"`). |
| `version` | `None` (auto) | Semantic version string advertised in the MCP server handshake. |
| `instructions` | `None` | Override the server-level instructions hint (shown to MCP clients). |
| `lifespan` | `None` | `AsyncContextManager` called once when the server starts/stops — use it to connect pools, wire telemetry, or warm caches. |

The method requires `mcp` to be installed (included in the default `agent-framework` install). The returned `mcp.server.lowlevel.Server` is transport-agnostic — mount it over stdio, streamable HTTP, or WebSocket depending on how your clients connect.

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

## Long-Term Memory — `MemoryStore` and `MemoryContextProvider`

> **Experimental.** `MemoryStore` and `MemoryContextProvider` are marked `ExperimentalFeature` in 1.4.0. The API is functional but may change between minor releases.

The memory system gives agents durable, cross-session recall. It works in two phases:

1. **Extraction** — after each session, an LLM extracts "durable facts" (preferences, decisions, patterns) from the conversation transcript.
2. **Injection** — at the start of each future session, the most relevant topics are loaded into the system prompt automatically.

The agent never "remembers" by keeping messages forever; instead it builds a compact, topic-indexed knowledge base that stays small regardless of conversation volume.

### Quickstart with `MemoryFileStore`

`MemoryFileStore` requires `owner_state_key` — a string naming the key in `session.state` that holds the logical owner (typically a user ID). The store uses that value to partition memory files on disk. Set `session.state[owner_state_key]` before the first `agent.run()` call.

```python
import asyncio
from datetime import timedelta
from agent_framework import Agent, MemoryContextProvider, MemoryFileStore
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# owner_state_key tells the store which session.state key holds the user/owner ID.
# Each unique value gets its own directory under base_path.
store = MemoryFileStore(
    base_path="./memory",
    owner_state_key="user_id",   # session.state["user_id"] drives per-user partitioning
)

memory = MemoryContextProvider(
    store=store,
    source_id="memory",           # identifies this provider's data within the store
    recent_turns=2,               # inject the last N turns as additional context
    selection_limit=3,            # load at most 3 topic files per session
    max_extractions=5,            # extract at most 5 memories per session
    consolidation_interval=timedelta(hours=24),  # consolidate topics once per day
    consolidation_min_sessions=5, # don't consolidate until at least 5 sessions exist
    consolidation_client=client,  # LLM used for consolidation (defaults to same as agent)
)

agent = Agent(
    client=client,
    instructions="You are a helpful personal assistant with long-term memory.",
    context_providers=[memory],
)


async def main() -> None:
    # Session 1 — store a preference.
    # session.state["user_id"] MUST be set before run() — the store raises if it's missing.
    session1 = agent.create_session(session_id="user-42-s1")
    session1.state["user_id"] = "user-42"
    await agent.run("I prefer concise bullet-point answers over long paragraphs.", session=session1)

    # Session 2 — same user_id so the provider loads memory from the same directory.
    session2 = agent.create_session(session_id="user-42-s2")
    session2.state["user_id"] = "user-42"
    response = await agent.run("Summarise the benefits of asyncio.", session=session2)
    print(response.text)  # Likely uses bullet points — remembered from session 1


asyncio.run(main())
```

**Multi-user isolation.** Every distinct value of `session.state["user_id"]` gets its own subtree under `base_path`. Two users can share a single agent and store instance without their memories crossing:

```python
async def handle_request(user_id: str, message: str) -> str:
    session = agent.create_session()
    session.state["user_id"] = user_id   # partitions memory to ./memory/<user_id>/
    response = await agent.run(message, session=session)
    return response.text
```

### `MemoryContextProvider` constructor reference

```python
MemoryContextProvider(
    store: MemoryStore,                     # storage backend (MemoryFileStore, custom)
    *,
    source_id: str = "memory",             # partition key within the store
    recent_turns: int = 0,                 # inject last N conversation turns as context
    load_tool_turns: bool = True,          # include tool-call turns when loading recent
    context_prompt: str | None = None,     # override the default "## Memory" header
    selection_limit: int = 3,             # max topic files loaded per session
    max_extractions: int = 5,             # max memories extracted per session
    consolidation_interval: timedelta = timedelta(hours=24),
    consolidation_min_sessions: int = 5,
    extraction_prompt: str | None = None,  # override LLM extraction prompt
    consolidation_prompt: str | None = None,
    consolidation_client: SupportsChatGetResponse | None = None,
    history_message_filter: Callable | None = None,
    history_dumps: JsonDumps | None = None,
    history_loads: JsonLoads | None = None,
)
```

### How the index works

`MemoryFileStore` organises data under `base_path` as:

```
memory/
└── <owner_id>/           # derived from session_id or set explicitly
    ├── MEMORY.md         # index: one line per topic with a summary
    ├── topics/
    │   ├── communication-style.md
    │   ├── tech-preferences.md
    │   └── ...
    ├── transcripts/      # raw session transcripts for extraction
    └── state.json        # metadata (last extraction timestamp, etc.)
```

At session start the provider reads `MEMORY.md`, selects the `selection_limit` most relevant topics (currently all, with future semantic ranking), and injects them into the system prompt. The injection is cheap — only the compact index and selected topic bodies are included.

### Inspecting and managing the store

```python
import asyncio
from agent_framework import AgentSession, MemoryFileStore

store = MemoryFileStore(base_path="./memory", owner_state_key="user_id")

# session.state["user_id"] must be set so the store knows which directory to read.
session = AgentSession(session_id="user-42-s1")
session.state["user_id"] = "user-42"


async def inspect_memory() -> None:
    # List all extracted topics for this user
    topics = store.list_topics(session, source_id="memory")
    for t in topics:
        print(f"{t.name}: {t.summary}")

    # Read a specific topic
    record = store.get_topic(session, source_id="memory", topic="communication-style")
    print(record.content)

    # Delete a topic the user wants forgotten (right-to-erasure flows)
    store.delete_topic(session, source_id="memory", topic="communication-style")

    # Rebuild the MEMORY.md index after manual edits to topic files
    store.rebuild_index(session, source_id="memory", line_limit=200, line_length=150)


asyncio.run(inspect_memory())
```

Note: `MemoryFileStore` methods (`list_topics`, `get_topic`, `delete_topic`, `rebuild_index`) are synchronous — they perform filesystem I/O directly. The async wrapper lives in `MemoryContextProvider`, which calls them from async lifecycle hooks.

### Custom `MemoryStore` backend

Subclass `MemoryStore` to use any durable backend — database, blob storage, vector DB. All abstract methods are **synchronous** (no `async`); `MemoryContextProvider` calls them from thread-pool workers when needed:

```python
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_framework import AgentSession, MemoryIndexEntry, MemoryStore, MemoryTopicRecord


class MyMemoryStore(MemoryStore):
    # get_owner_id is not abstract — override it to enable per-user isolation
    def get_owner_id(self, session: AgentSession) -> str | None:
        return session.state.get("user_id")

    # --- 10 abstract methods that must be implemented ---

    def list_topics(self, session: AgentSession, *, source_id: str) -> list[MemoryTopicRecord]:
        ...

    def get_topic(self, session: AgentSession, *, source_id: str, topic: str) -> MemoryTopicRecord:
        ...

    def write_topic(self, session: AgentSession, record: MemoryTopicRecord, *, source_id: str) -> None:
        ...

    def delete_topic(self, session: AgentSession, *, source_id: str, topic: str) -> None:
        ...

    def rebuild_index(
        self, session: AgentSession, *, source_id: str, line_limit: int, line_length: int
    ) -> list[MemoryIndexEntry]:   # returns MemoryIndexEntry objects, not strings
        ...

    def get_index_text(
        self,
        session: AgentSession,
        *,
        source_id: str,
        line_limit: int,
        line_length: int,
        index_entries: Sequence[MemoryIndexEntry] | None = None,
    ) -> str:
        ...

    def read_state(self, session: AgentSession, *, source_id: str) -> dict[str, Any]:
        ...

    def write_state(self, session: AgentSession, state: Mapping[str, Any], *, source_id: str) -> None:
        ...

    def get_transcripts_directory(self, session: AgentSession, *, source_id: str) -> Path:
        ...

    def search_transcripts(
        self,
        session: AgentSession,
        *,
        source_id: str,
        query: str,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        ...
```

Wire it up exactly like `MemoryFileStore` — pass it as the `store=` argument to `MemoryContextProvider`. Override `get_owner_id` to return the owner key from session state so the provider can scope memory per user.

---

## Agent Todo List — `TodoProvider` (Experimental HARNESS)

> **Experimental.** `TodoProvider` and its backing stores are `ExperimentalFeature.HARNESS` in 1.4.0.

`TodoProvider` gives an agent a structured task list it can manage itself. The agent receives five tools — `add_todos`, `complete_todos`, `remove_todos`, `get_remaining_todos`, and `get_all_todos` — and a default system-prompt injection that tells it how to use them. The provider stores state in the session (in-memory by default) or on disk via `TodoFileStore`.

### Quickstart — in-session todos

```python
import asyncio
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a project-planning assistant.",
    context_providers=[TodoProvider()],   # in-session store by default
)

async def main() -> None:
    session = agent.create_session()

    # Turn 1 — agent adds todos as it plans
    r1 = await agent.run(
        "Plan a three-day product launch: marketing, engineering, and support tasks.",
        session=session,
    )
    print(r1.text)

    # Turn 2 — agent checks get_remaining_todos and marks items complete as it works
    r2 = await agent.run(
        "Draft the engineering checklist and mark those tasks done.",
        session=session,
    )
    print(r2.text)


asyncio.run(main())
```

The agent sees instructions like "Break complex work into trackable items… Use `add_todos`… `complete_todos`… `get_remaining_todos`…". It manages the list autonomously — no application code needed to drive it.

### Persisting todos to disk with `TodoFileStore`

For todos that should survive process restarts or span multiple sessions, swap in `TodoFileStore`. Unlike `MemoryFileStore`, the `owner_state_key` parameter is optional — when omitted, the `session_id` itself is used as the file path component:

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoFileStore, TodoProvider
from agent_framework.openai import OpenAIChatClient


# Todos written to ./todos/<session_id>/todos.json  (no owner_state_key required)
store = TodoFileStore(base_path="./todos")

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a long-running task assistant.",
    context_providers=[TodoProvider(store=store)],
)


async def main() -> None:
    # First run — agent creates todos
    session = agent.create_session(session_id="project-launch-42")
    await agent.run("Break down the launch into 10 concrete tasks.", session=session)

    # Second run (new process, same session_id) — agent picks up existing todos
    session2 = agent.create_session(session_id="project-launch-42")
    r = await agent.run("What's still left to do?", session=session2)
    print(r.text)


asyncio.run(main())
```

Pass `owner_state_key="user_id"` when multiple users share a `base_path` so their todo files are partitioned:

```python
store = TodoFileStore(base_path="./todos", owner_state_key="user_id")

session = agent.create_session()
session.state["user_id"] = "alice"   # todos written to ./todos/alice/todos.json
```

### `TodoProvider` constructor reference

```python
TodoProvider(
    source_id="todo",          # key in session.state — change if you stack multiple providers
    *,
    instructions=None,         # override the default system-prompt block (None = use built-in)
    store=None,                # TodoStore subclass; defaults to TodoSessionStore (in-memory)
)
```

**Custom instructions.** The default text explains all five tools. Override to restrict the agent or tune the tone:

```python
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient

focused_provider = TodoProvider(
    instructions=(
        "You have a task list. Use `add_todos` to create tasks when the user asks you to plan. "
        "Use `complete_todos` when a task is done. "
        "Never remove tasks unless the user explicitly says to drop them."
    ),
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a focused sprint assistant.",
    context_providers=[focused_provider],
)
```

### Inspecting todos from application code

Read the task list from outside the agent — useful for dashboards, webhooks, or status APIs:

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoFileStore, TodoProvider, TodoSessionStore
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a task assistant.",
    context_providers=[TodoProvider()],
)


async def main() -> None:
    session = agent.create_session(session_id="s1")
    await agent.run("Add tasks: write tests, review PR, deploy.", session=session)

    # Read from in-memory store — uses the same session object
    in_mem_store = TodoSessionStore()
    items, _next_id = await in_mem_store.load_state(session, source_id="todo")
    for item in items:
        status = "✓" if item.is_complete else "·"
        print(f"  {status} [{item.id}] {item.title}")

    # --- File-based: load from disk by session_id ---
    file_store = TodoFileStore(base_path="./todos")
    items2, _ = await file_store.load_state(
        AgentSession(session_id="project-launch-42"),
        source_id="todo",
    )
    remaining = [i for i in items2 if not i.is_complete]
    completed = [i for i in items2 if i.is_complete]
    print(f"{len(remaining)} pending, {len(completed)} done")


asyncio.run(main())
```

**`TodoItem` fields:** `id` (int), `title` (str), `description` (str | None), `is_complete` (bool). The agent calls `complete_todos([id, ...])` and `remove_todos([id, ...])` using the integer IDs.

---

## Agent Mode Provider — `AgentModeProvider` (Experimental HARNESS)

> **Experimental.** `AgentModeProvider`, `set_agent_mode`, and `get_agent_mode` are `ExperimentalFeature.HARNESS` in 1.4.0.

`AgentModeProvider` lets an agent switch between named operating modes at runtime. Two modes ship out of the box — **plan** (interactive, ask questions) and **execute** (autonomous, minimise interruptions). You can define any set of modes and inject custom descriptions for each.

The provider exposes `get_mode` and `set_mode` tools to the agent, and injects the current mode into the system prompt so the agent knows how to behave.

### Quickstart — plan / execute cycle

```python
import asyncio
from agent_framework import Agent, AgentModeProvider
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a task-planning and execution assistant.",
    context_providers=[AgentModeProvider()],  # default modes: "plan" and "execute"
)

async def main() -> None:
    session = agent.create_session()

    # Phase 1: planning — the agent should ask clarifying questions
    await agent.run(
        "I want to migrate our Postgres database to a new schema.",
        session=session,
    )

    # Phase 2: switch to execute mode and let the agent work autonomously
    await agent.run(
        "Looks good. Switch to execute mode and start the migration.",
        session=session,
    )

asyncio.run(main())
```

In **plan** mode the agent is encouraged to ask for clarification before acting. In **execute** mode it works autonomously and avoids unnecessary check-ins.

### Custom modes

Define your own mode names and descriptions when the defaults don't fit. Mode names come from the keys of `mode_descriptions`:

```python
from agent_framework import Agent, AgentModeProvider
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a code-review assistant.",
    context_providers=[
        AgentModeProvider(
            default_mode="review",
            mode_descriptions={
                "review":  "Read the diff and identify issues. Do not suggest fixes yet.",
                "suggest": "For each issue, propose a concrete code fix.",
                "approve": "All issues resolved. Write the approval comment and exit.",
            },
        )
    ],
)
```

### Reading and setting mode from application code

```python
from agent_framework import AgentSession, get_agent_mode, set_agent_mode

session = AgentSession(session_id="review-pr-88")

# Read the current mode (returns the default if not yet set).
# available_modes must match what the provider was configured with.
current = get_agent_mode(
    session,
    default_mode="review",
    available_modes=["review", "suggest", "approve"],
)
print(current)   # "review"

# Programmatically advance to the next stage
set_agent_mode(session, "suggest", available_modes=["review", "suggest", "approve"])
```

Use `set_agent_mode` from your application layer when an external event (e.g. a CI gate passing) should trigger a mode transition, rather than relying on the agent to call `set_mode` itself.

### `AgentModeProvider` constructor reference

```python
AgentModeProvider(
    source_id="agent_mode",  # session.state partition key
    *,
    default_mode=None,       # starting mode; defaults to first key in mode_descriptions
    mode_descriptions=None,  # Mapping[mode_name, description]; defaults to plan/execute
    instructions=None,       # override the default system-prompt block (must contain
                             # {available_modes} and {current_mode} placeholders)
)
```

---

## Prompt Injection Defense — `SecureAgentConfig` (Experimental)

> **Experimental.** `SecureAgentConfig` is `ExperimentalFeature.FIDES` in 1.4.0. The API is functional but may change between minor releases.

`SecureAgentConfig` is a `ContextProvider` that defends against prompt injection attacks using **information-flow control**. It labels every tool result as either `TRUSTED` or `UNTRUSTED`, prevents untrusted content from calling privileged tools, and optionally logs policy violations. The approach is inspired by the FIDES research on taint tracking for LLM pipelines.

- The label tracker (`LabelTrackingFunctionMiddleware`) marks results from tools that touch external or user-controlled data as `IntegrityLabel.UNTRUSTED`.
- The policy enforcer (`PolicyEnforcementFunctionMiddleware`) prevents untrusted-labelled context from invoking privileged tools.
- When a violation is detected, the agent either blocks the call (`block_on_violation=True`, default) or routes it for human approval (`approval_on_violation=True`).

Import from `agent_framework.security` — this is a different sub-module from the main `agent_framework` namespace:

```python
from agent_framework.security import SecureAgentConfig, IntegrityLabel, ConfidentialityLabel
```

### Quickstart

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import SecureAgentConfig, IntegrityLabel, ConfidentialityLabel


@tool
async def fetch_news(query: str) -> str:
    """Fetch news headlines for a query — untrusted external content."""
    return f"[external] Top story about {query}: ..."


@tool
async def summarize(text: str) -> str:
    """Summarize trusted, internal content."""
    return f"Summary: {text[:100]}"


security = SecureAgentConfig(
    auto_hide_untrusted=True,                          # hide UNTRUSTED results from the model
    default_integrity=IntegrityLabel.UNTRUSTED,        # tool calls default to untrusted
    default_confidentiality=ConfidentialityLabel.PUBLIC,
    block_on_violation=True,                           # block on policy violation (default)
    enable_audit_log=True,
    enable_policy_enforcement=True,
)

# SecureAgentConfig is a ContextProvider — pass it via context_providers=
agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    tools=[fetch_news, summarize],
    context_providers=[security],
)


async def main() -> None:
    response = await agent.run("What's the latest news about quantum computing?")
    print(response.text)


asyncio.run(main())
```

### Allowing specific tools in untrusted context

Use `allow_untrusted_tools` to whitelist tools that may run even when the call stack is tainted with untrusted data. Pair with `approval_on_violation=True` to request human approval instead of hard-blocking unknown cases:

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import SecureAgentConfig, IntegrityLabel, ConfidentialityLabel


@tool
async def fetch_news(query: str) -> str:
    """Fetch external news — produces UNTRUSTED output."""
    return f"[external] News about {query}"


@tool
async def send_email(to: str, body: str) -> str:
    """Send an email — privileged, must not be reachable from untrusted data."""
    return f"sent to {to}"


@tool
async def log_search(query: str) -> str:
    """Log the search query for auditing — allowed even in untrusted context."""
    return f"logged: {query}"


security = SecureAgentConfig(
    auto_hide_untrusted=True,
    default_integrity=IntegrityLabel.UNTRUSTED,
    default_confidentiality=ConfidentialityLabel.PUBLIC,
    allow_untrusted_tools={"log_search"},   # these tools may run in untrusted context
    block_on_violation=False,
    approval_on_violation=True,             # request human approval instead of hard block
    enable_audit_log=True,
    enable_policy_enforcement=True,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant. Do not send emails based on news content.",
    tools=[fetch_news, send_email, log_search],
    context_providers=[security],
)


async def main() -> None:
    response = await agent.run("Find news about AI and send a summary to boss@example.com.")
    print(response.text)


asyncio.run(main())
```

### `SecureAgentConfig` constructor reference

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `auto_hide_untrusted` | `bool` | `True` | Hide tool results labelled `UNTRUSTED` from the model's view |
| `default_integrity` | `IntegrityLabel` | `UNTRUSTED` | Default integrity label applied to all tool results |
| `default_confidentiality` | `ConfidentialityLabel` | `PUBLIC` | Default confidentiality label |
| `allow_untrusted_tools` | `set[str]` | `set()` | Tool names allowed to execute even when context is untrusted |
| `block_on_violation` | `bool` | `True` | Block the call on a policy violation |
| `approval_on_violation` | `bool` | `False` | Route violation to human approval instead of blocking |
| `enable_audit_log` | `bool` | `True` | Log every policy decision to the audit logger |
| `enable_policy_enforcement` | `bool` | `True` | Enable the `PolicyEnforcementFunctionMiddleware` |
| `quarantine_chat_client` | `SupportsChatGetResponse \| None` | `None` | Optional isolated LLM for the `quarantined_llm` built-in security tool |

`SecureAgentConfig` automatically injects `LabelTrackingFunctionMiddleware`, and (when `enable_policy_enforcement=True`) `PolicyEnforcementFunctionMiddleware`. It also adds two built-in security tools: `quarantined_llm` (runs a prompt through an isolated model without access to privileged tools) and `inspect_variable` (lets the agent inspect labelled variables before acting on them).

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

## Class & API Reference

> Consolidated from the full "class deep dive" volume series that used to ship as 49 separate pages (the base volume plus `_v2`–`_v44` of the main series, and `microsoft_agent_framework_python_sdk_class_deep_dives` plus `_v3`–`_v6` covering the `azure-ai-agents` add-on). Two independent verification passes checked every symbol below against an installed `agent-framework==1.14.0` venv with `inspect.signature()` / `inspect.getsource()`; the `azure-ai-agents` appendix at the end was checked only for internal consistency across its five source volumes — no fresh `pip install` of `azure-ai-agents` was available for that pass, so treat it as best-effort rather than independently re-verified.
>
> Several older per-volume pages invented methods or parameters that never existed on any released version of the package. Rather than silently drop those, the corrections are called out inline as **Correction** notes — useful if you're migrating code that was written against one of the old pages.

### Workflows & Execution

#### `WorkflowBuilder`
**Module:** `agent_framework._workflows._workflow_builder`

```python
WorkflowBuilder(
    max_iterations: int = 100, name: str | None = None, description: str | None = None, *,
    start_executor: Executor | SupportsAgentRun,
    checkpoint_storage: CheckpointStorage | None = None,
    output_from: list[Executor | SupportsAgentRun] | Literal["all"] | None = ...,
    intermediate_output_from: list[...] | Literal["all", "all_other"] | None = ...,
    output_executors: ...,  # deprecated alias for output_from
)
```

Key methods: `.add_edge(source, target, condition=None)`, `.add_chain([executors...])` (sequential shorthand), `.add_fan_out_edges(source, targets, selection_func=None)`, `.add_fan_in_edges(sources, target)`, `.add_switch_case_edge_group(source, [Case(condition, target), ..., Default(target)])`, `.add_multi_selection_edge_group(source, targets, selection_func)`, `.build()`.

**Output routing:** `output_from` picks which executors' `yield_output` calls become `type="output"` events; `intermediate_output_from` (list, `"all"`, or `"all_other"`) routes the rest to `type="intermediate"`. With both omitted, every `yield_output` is `"output"` (back-compat mode).

```python
from agent_framework import WorkflowBuilder, WorkflowContext, Executor, handler

class Normalize(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text.strip().lower())

class Count(Executor):
    @handler
    async def run(self, tokens: list[str], ctx: WorkflowContext[None, int]) -> None:
        await ctx.yield_output(len(tokens))

workflow = (
    WorkflowBuilder(start_executor=Normalize(id="norm"), output_from=[Count(id="count")])
    .add_chain([Normalize(id="norm"), Count(id="count")])
    .build()
)
result = await workflow.run("  Hello World  ")
print(result.get_outputs())
```

> **Correction — fabricated methods:** several old per-volume pages (source volumes 5, 11, 16, 19) invented `add_node`, `add_executor`, `connect`, `set_output_from`, `fan_in`, and `fan_out` methods on `WorkflowBuilder`. None of these exist on any version of the installed package — the verified method set is exactly the seven shown above (`add_edge`, `add_chain`, `add_fan_out_edges`, `add_fan_in_edges`, `add_switch_case_edge_group`, `add_multi_selection_edge_group`, `build`). Discard any example using the fabricated names.

#### `Workflow` + `WorkflowRunResult`
**Module:** `agent_framework._workflows._workflow`

`Workflow` is the immutable engine built by `.build()`. `workflow.run(message, *, stream=False, responses=None, checkpoint_id=None, checkpoint_storage=None, include_status_events=False, function_invocation_kwargs=None, client_kwargs=None)` returns a `WorkflowRunResult` (non-streaming) or a `ResponseStream[WorkflowEvent, WorkflowRunResult]` (streaming). `WorkflowRunResult` is a `list[WorkflowEvent]` subclass with helper methods: `get_outputs()`, `get_intermediate_outputs()`, `get_request_info_events()`, `get_final_state()`, `status_timeline()`.

```python
result = await workflow.run("hello")
print(result.get_outputs())
print(result.get_final_state())        # WorkflowRunState.IDLE

async for event in workflow.run("hello", stream=True):
    if event.type == "output":
        print(event.data)
```

HITL resume: `workflow.run(responses={request_id: value}, checkpoint_id=...)` — `message` and `checkpoint_id` are mutually exclusive; **there is no `workflow.respond()` method and no `workflow.run_from_checkpoint()` method** (an older per-volume page showed `await workflow.run_from_checkpoint(checkpoint_id, checkpoint_storage=...)` — that method does not exist on the installed 1.14.0 `Workflow` class; resume through `workflow.run(checkpoint_id=...)` instead, as the rest of this guide's "Workflow checkpointing" section already does).

Every `Workflow` also exposes `.as_agent(name=..., description=..., context_providers=...)` (wraps the graph as a `SupportsAgentRun`-compatible agent) and `.as_tool(name=..., description=...)` (wraps it as a callable `FunctionTool` for use inside another agent's `tools=[...]`) — see this guide's "Exposing a workflow as an agent" section above for a full example.

#### `WorkflowContext`
**Module:** `agent_framework._workflows._workflow_context`

Injected into every executor `@handler`. Provides `await ctx.send_message(payload)` (routes to downstream executors per the edge graph; emits an OTel `PRODUCER` span) and `await ctx.yield_output(payload)` (surfaces a workflow-level result). Generic over `WorkflowContext[OutT, W_OutT]`. Fan-in executors receive multiple `source_executor_ids`; call `ctx.get_source_executor_id()` only when there is exactly one source (raises `RuntimeError` otherwise).

#### `WorkflowEvent` + `WorkflowRunState` + `WorkflowEventSource` + `WorkflowErrorDetails`
**Module:** `agent_framework._workflows._events`

`WorkflowEvent[DataT]` is the single generic event type for the whole workflow event bus; prefer the factory classmethods over the constructor: `.started()`, `.status(state)`, `.failed(details)`, `.warning(message)`, `.error(exception)`, `.request_info(request_id, source_executor_id, request_data, response_type)`, `.superstep_started(n)` / `.superstep_completed(n)`, `.executor_invoked(id)` / `.executor_completed(id)` / `.executor_failed(id, details)` / `.executor_bypassed(id)` (fires on a checkpoint-replay cache hit). `.emit()` (type `"data"`) still exists on the installed version but is deprecated — use `yield_output` + `intermediate_output_from` instead.

`WorkflowRunState(str, Enum)`: `STARTED, IN_PROGRESS, IN_PROGRESS_PENDING_REQUESTS, IDLE, IDLE_WITH_PENDING_REQUESTS, FAILED, CANCELLED`. `WorkflowEventSource(str, Enum)`: `FRAMEWORK, EXECUTOR`.

`WorkflowErrorDetails` (`@dataclass`): `error_type: str, message: str, traceback: str | None, executor_id: str | None, extra: dict | None` — build with `.from_exception(exc, executor_id=..., extra=...)`.

```python
async for event in workflow.run("do work", stream=True):
    if event.type == "request_info":
        print(event.request_id, event.data, event.response_type)
    elif event.type == "failed":
        print(event.details.error_type, event.details.message)
```

#### `WorkflowViz`
**Module:** `agent_framework._workflows._viz`

```python
WorkflowViz(workflow: Workflow)
```
`.to_digraph(include_internal_executors=False)` → Graphviz DOT string. `.to_mermaid(...)` → Mermaid `flowchart TD`. `.export(format="svg"|"png"|"pdf"|"dot", filename=None, include_internal_executors=False)` renders to a file (`pip install graphviz` needed for non-`dot` formats). `.save_svg/.save_png/.save_pdf(...)` are thin wrappers around `.export()`. Conditional edges render dashed with a `conditional` label; sub-workflows hosted by a `WorkflowExecutor` render as nested `subgraph` blocks.

```python
viz = WorkflowViz(workflow)
print(viz.to_mermaid())
viz.export(format="svg", filename="pipeline.svg")
```

#### `SwitchCaseEdgeGroup` + `Case` + `Default`
**Module:** `agent_framework._workflows._edge`

Runtime API: `builder.add_switch_case_edge_group(source, [Case(condition=fn, target=executor), ..., Default(target=executor)])` — evaluates cases in order, first match wins; exactly one `Default` required. Serialization-only classes `SwitchCaseEdgeGroup(source_id, cases=[SwitchCaseEdgeGroupCase(condition, target_id), SwitchCaseEdgeGroupDefault(target_id)])` back `.to_dict()` inspection. `FanInEdgeGroup` / `FanOutEdgeGroup` are the base primitives switch/case and multi-selection build on.

```python
builder.add_switch_case_edge_group(
    classifier,
    [
        Case(condition=lambda msg: get_sentiment(msg) == "positive", target=positive_agent),
        Default(target=neutral_agent),
    ],
)
```

#### EdgeGroup / EdgeRunner hierarchy (overview)
Internal representation of every edge type the builder methods create: `SingleEdgeGroup` (plain `add_edge`), `FanOutEdgeGroup`, `FanInEdgeGroup`, `SwitchCaseEdgeGroup`, `MultiSelectionEdgeGroup`. Each has a matching `EdgeRunner` that the runner invokes at execution time. Useful when reading stack traces from workflow errors — the class name in the trace maps directly to which builder call produced that edge.

#### `Executor`, `@executor`, `@handler`
Base unit of work in a workflow graph. `@executor` decorates a plain async function into an `Executor`; `@handler` marks a method inside a class-based `Executor` subclass as the entry point for a given message type (supports overloads for multiple input types on one executor class — see "Custom executors with `@handler`" above for the full pattern, including the explicit-types decorator form).

```python
from agent_framework import Executor, handler, WorkflowContext

class SplitText(Executor):
    def __init__(self, id: str = "split"):
        super().__init__(id=id)

    @handler
    async def run(self, text: str, ctx: WorkflowContext[list[str]]) -> None:
        await ctx.send_message(text.split())
```

#### `AgentExecutor` + `AgentExecutorResponse.with_text()`
**Module:** `agent_framework._workflows._agent_executor`

`AgentExecutor` wraps a `SupportsAgentRun` as a graph node. `AgentExecutorResponse.with_text(text: str) -> AgentExecutorResponse` returns a new response with the text replaced but `full_conversation` preserved — required when a custom executor transforms an agent's text but must keep the conversation chain intact for downstream `AgentExecutor`s (returning a bare `str` breaks that chain; see "Custom executors with `@handler`" above for a worked example).

#### Runner / superstep model (overview)
Internal Pregel-style engine that drives workflow execution: each "superstep" delivers all pending messages to their target executors in parallel, collects new outgoing messages, and repeats until no more messages are pending or `max_iterations` is hit. Not directly instantiated by users — documented here because it explains observed ordering/concurrency behavior (fan-out targets run concurrently within a superstep; fan-in targets wait for all upstream sources to have contributed for the current step).

#### `InProcRunnerContext`
**Module:** `agent_framework._workflows._runner_context`

The concrete execution context behind a local `workflow.run()`. Handles inter-executor message routing (`send_message`/`drain_messages`), event streaming with **lazy `asyncio.Queue` binding** (so one context can be reused across separate `asyncio.run()` calls), HITL request/response correlation (`add_request_info_event`/`send_request_info_response`/`get_pending_request_info_events`), and per-run checkpoint-storage override (`set_runtime_checkpoint_storage` / `clear_runtime_checkpoint_storage`, the latter called automatically after each run).

#### `RequestInfoExecutor` / `request_info`
Pauses a workflow superstep to solicit external (typically human) input mid-graph, then resumes once a response is supplied — used to build human-in-the-loop approval steps inside otherwise automated graphs. See "Workflow human-in-the-loop" above for the `@response_handler` pattern used to receive the reply.

#### `FunctionalWorkflow` + `FunctionalWorkflowAgent` + `RunContext` + `StepWrapper` + `WorkflowInterrupted`
**Module:** `agent_framework._workflows._functional`

The `@workflow` / `@step` decorator API authors workflows as plain async functions — no graph wiring.

```python
FunctionalWorkflow(func: Callable[..., Awaitable[Any]], *, name=None, description=None,
                    checkpoint_storage: CheckpointStorage | None = None)
FunctionalWorkflowAgent(workflow: FunctionalWorkflow, *, name=None, description=None,
                         context_providers: Sequence[Any] | None = None)
StepWrapper(func: Callable[..., Awaitable[R]], *, name: str | None = None)   # raises TypeError if func isn't async
```

`@step` caches results by `(step_name, call_index)` for checkpoint replay; a cache hit emits `executor_bypassed` instead of re-running. `RunContext` (injected by parameter name `ctx` or type annotation) exposes `await ctx.request_info(request_data, *, response_type, request_id=None)` (raises `WorkflowInterrupted` internally to suspend), `await ctx.add_event(WorkflowEvent(...))`, `ctx.get_state(key, default=None)` / `ctx.set_state(key, value)`, `ctx.is_streaming`. `WorkflowInterrupted` subclasses `BaseException` (not `Exception`) — a caller's `except Exception:` cannot accidentally swallow it.

```python
from agent_framework import workflow, step, RunContext

@step
async def draft(requirements: str) -> str: ...

@workflow
async def approval_workflow(requirements: str, ctx: RunContext) -> str:
    text = await draft(requirements)
    approval: str = await ctx.request_info({"draft": text}, request_id="approval", response_type=str)
    return text if approval.lower().startswith("approve") else "revise"

result = await approval_workflow.run("Build a feature")
pending = result.get_request_info_events()
result2 = await approval_workflow.run(responses={"approval": "Approved!"})
```

#### `WorkflowGraphValidator` + `EdgeDuplicationError` + `GraphConnectivityError` + `TypeCompatibilityError`
**Module:** `agent_framework._workflows._validation`

Runs at `.build()` time: edge duplication, type compatibility between connected executors (a `FanIn` target's declared input type must accept a `list` of the upstream type), DFS graph connectivity from the start node, isolated-executor / self-loop / dead-end checks, and output-designation validation. Each error carries a `ValidationTypeEnum` tag and (for `TypeCompatibilityError`) `source_executor_id`/`target_executor_id`/`source_output_types`/`target_input_types`.

#### `SubWorkflowRequestMessage` + `SubWorkflowResponseMessage` + `ExecutionContext`
**Module:** `agent_framework._workflows._workflow_executor`

When a child workflow hosted inside a `WorkflowExecutor` node emits `request_info`, the parent wraps it in a `SubWorkflowRequestMessage(source_event, executor_id)`; the caller answers via `req.create_response(data)` (type-validates against `source_event.response_type`, raising `TypeError` on mismatch), producing a `SubWorkflowResponseMessage(data, source_event)`. `ExecutionContext(execution_id, collected_responses, expected_response_count, pending_requests)` tracks how many of several concurrent sub-workflow requests have been answered.

#### Durable execution — `DurableAIAgentWorker` / `DurableAIAgentClient` / `DurableAIAgent` / `AgentEntity`
**Package:** `agent-framework-durabletask` (`agent_framework_durabletask`)

Turns any agent-framework agent into a long-running Azure Durable Entity. `DurableAIAgentWorker(worker: TaskHubGrpcWorker, callback=None)` registers agents via `.add_agent(agent, callback=None)` — each becomes a durable entity named `dafx-{agent.name}`. `DurableAIAgentClient(client: TaskHubGrpcClient, max_poll_retries=..., poll_interval_seconds=...)` returns `DurableAIAgent` proxies via `.get_agent(name)`; `DurableAIAgent.run()` returns `TaskT` **synchronously** (not a coroutine) and raises `ValueError` if `stream=True` is passed — durable agents don't support streaming. `AgentEntity` is the platform-agnostic execution kernel: it tries `agent.run(stream=True)` first, falls back to non-streaming on failure, and persists conversation history via an injected `AgentEntityStateProviderMixin` (`DurableAgentState`, schema version `"1.1.0"`, JSON-serialized). Errors are flagged `is_error=True` **in-memory only** (not serialized — a reloaded entity forgets which past turns failed). Session addressing uses `AgentSessionId` (`@name@key` wire format) / `DurableAgentSession`. `DurableAIAgentOrchestrationContext(context: OrchestrationContext)` gives orchestration functions the same `get_agent()` pattern, returning yield-compatible `DurableAgentTask`s.

For standalone (non-Azure-Functions) hosting, `WorkflowOrchestrationContext` is a `@runtime_checkable` Protocol abstracting host differences; `DurableTaskWorkflowContext` implements it for the standalone Durable Task Scheduler (`supports_event_streaming=True`, no 16 KB status cap, unlike the Azure Functions adapter). `DurableWorkflowClient` (`from agent_framework.azure import DurableWorkflowClient`) drives a durable *workflow* (as opposed to a single durable agent) from an external caller: `.start_workflow(input) -> str`, `.await_workflow_output(instance_id, timeout_seconds=None)`, `.stream_workflow(instance_id) -> AsyncIterator[WorkflowEvent]`, `.get_pending_hitl_requests(instance_id)` / `.send_hitl_response(...)`. `plan_workflow_registration(workflow) -> WorkflowRegistrationPlan` classifies each `AgentExecutor` node as a durable **entity** and every other `Executor` as a durable **activity**; the orchestrator itself is always named `"workflow_orchestrator"`.

For Azure Functions specifically, `AzureFunctionsAgentExecutor(context)` extends `DurableAgentExecutor` and overrides `generate_unique_id()` to use `context.new_uuid()` (deterministic across replays); `PreCompletedTask` is an already-completed `TaskBase` used for fire-and-forget (`wait_for_response=False`) responses; `AgentTask` wraps a raw entity-call task and parses its result into a typed `AgentResponse` given a Pydantic `response_format`.

```python
worker = TaskHubGrpcWorker(host_address="localhost:4001")
agent_worker = DurableAIAgentWorker(worker)
agent_worker.add_agent(Agent(client=OpenAIChatCompletionClient(model="gpt-4o"), name="assistant"))
agent_worker.start()

grpc_client = TaskHubGrpcClient(host_address="localhost:4001")
agent = DurableAIAgentClient(grpc_client).get_agent("assistant")
response = agent.run("Summarise the latest earnings report.")   # returns synchronously
```

### Agents, Modes & Orchestration

> **Reconciled naming:** an older per-volume page described this layer's base class as `ChatAgent`. The installed 1.14.0 API — and every other section of this guide — uses `Agent` / `RawAgent`; `ChatAgent` is not the current class name. Likewise `AgentRunResponse` / `AgentRunResponseUpdate` are the pre-rename names for what 1.14.0 ships as `AgentResponse` / `AgentResponseUpdate` (used throughout this guide already).

#### `Agent` + `RawAgent`
**Module:** `agent_framework._agents`

`RawAgent` is the low-latency inner implementation (no telemetry, no middleware pipeline, no default compaction) that `Agent` wraps with the full `AgentMiddlewareLayer` + telemetry stack. Both share the same constructor shape and are `Generic[OptionsCoT]` for provider-specific option typing:

```python
Agent(client: SupportsChatGetResponse[OptionsCoT], instructions: str | None = None, *,
      id=None, name=None, description=None, tools=None, default_options: OptionsCoT | None = None,
      context_providers: Sequence[ContextProvider] | None = None,
      middleware: MiddlewareTypes | Sequence[MiddlewareTypes] | None = None,
      require_per_service_call_history_persistence: bool = False,
      compaction_strategy: CompactionStrategy | None = None, tokenizer: TokenizerProtocol | None = None,
      additional_properties: MutableMapping[str, Any] | None = None)
```

`agent.run(messages=None, *, stream=False, session=None, middleware=None, tools=None, options=None, compaction_strategy=None, tokenizer=None, function_invocation_kwargs=None, client_kwargs=None)` has three practical shapes: plain text, structured output (`options={"response_format": Model}` → read via `response.value`), and streaming (`stream=True` → `ResponseStream`; `await stream.get_final_response()` gives the accumulated `AgentResponse`). `RawAgent` accepts a `middleware=` constructor kwarg but never invokes it — use `Agent` for middleware support.

```python
agent = RawAgent(client=OpenAIChatClient(model="gpt-4o-mini"), instructions="You are a fast, terse assistant.")
response = await agent.run("What is 2 + 2?")
```

#### `AgentResponse` + `AgentResponseUpdate`
**Module:** `agent_framework._sessions` (top-level `agent_framework`)

`AgentResponse` is the result of `await agent.run(...)`: `messages`, `response_id`, `finish_reason`, `usage_details: UsageDetails | None`, `value: ResponseModelT | None` (structured output), `continuation_token`. `.text` concatenates the last assistant message's text. Class methods `AgentResponse.from_updates(updates)`, `.from_dict()`/`.from_json()`; instance `.to_dict()`/`.to_json()`. `AgentResponseUpdate` is the streaming chunk type consumed via `async for update in stream`.

#### `create_harness_agent`
**Module:** `agent_framework._harness._agent`

Batteries-included factory that assembles a fully-wired `Agent`: function invocation, per-service-call history persistence, compaction (when token budgets given), `TodoProvider`, `AgentModeProvider`, `FileMemoryProvider`, `FileAccessProvider`, `SkillsProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, optional `AgentLoopMiddleware`, and OTel tracing.

> **Correction vs. earlier docs:** the installed 1.14.0 signature has **no `disable_file_access` parameter** (present in older docs) — instead pass `file_access_store=None` to skip it. It gained `before_compaction_strategy` / `after_compaction_strategy` (replacing a single `compaction_strategy` shortcut), `file_access_disable_readonly_tool_approval` / `file_access_disable_write_tool_approval` (mirroring `FileAccessProvider`'s own new flags), and an entirely new **`shell_executor` / `shell_environment_provider_options`** pair not documented in any prior per-volume page.

```python
agent = create_harness_agent(
    OpenAIChatClient(model="gpt-4o"),
    agent_instructions="You are a thorough task executor.",
    loop_should_continue=todos_remaining(),
    loop_next_message=todos_remaining_message,
    loop_max_iterations=10,
)
```

#### `AgentModeProvider` + `get_agent_mode` / `set_agent_mode`
**Module:** `agent_framework._harness._mode`

Gives an agent named operating modes (default `plan`/`execute`) persisted in session state and injected into the system prompt; exposes `mode_set`/`mode_get` tools to the LLM.

```python
AgentModeProvider(source_id: str = "agent_mode", *, default_mode: str | None = None,
                   mode_instructions: Mapping[str, str] | None = None, instructions: str | None = None)
get_agent_mode(session: AgentSession, *, source_id="agent_mode", default_mode=None,
               available_modes=None) -> str
set_agent_mode(session: AgentSession, mode: str, *, source_id="agent_mode", available_modes=None) -> str
```

> **Correction:** the constructor keyword is `mode_instructions` (not `mode_descriptions`, as several older per-volume pages named it — and as the "Agent Mode Provider" section of this guide's own body currently still shows). Confirmed against the installed signature; see this guide's "Agent Mode Provider" section above for the older, superseded `mode_descriptions=` examples — those should be updated to `mode_instructions=` when you copy them.

#### `AgentLoopMiddleware` + `JudgeVerdict`
**Module:** `agent_framework._harness._loop`

Re-runs an agent after each response until `should_continue` returns `False`.

```python
AgentLoopMiddleware(should_continue: ShouldContinueCallable, *, max_iterations: int | None = 10,
                     next_message: NextMessageCallable | None = None,
                     record_feedback: FeedbackCallable | None = None, inject_progress: bool = True,
                     fresh_context: bool = False, return_final_only: bool = False,
                     additional_instructions: str | None = None)

AgentLoopMiddleware.with_judge(judge_client, *, criteria=None, instructions=None,
                                max_iterations: int | None = 5, next_message=None, fresh_context=False)
```

`should_continue(**kwargs)` receives `iteration`, `last_result`, `messages`, `original_messages`, `session`, `agent`, `progress`, `feedback`. `with_judge` wires a separate LLM to emit a structured `JudgeVerdict {answered: bool, reasoning: str = ""}` each iteration. `fresh_context=True` restores the session to its pre-loop snapshot between iterations (only the `record_feedback` progress log carries over). `todos_remaining(*, looping_modes=None)` / `background_tasks_running()` are ready-made `ShouldContinueCallable`s that auto-resolve `TodoProvider` / `BackgroundAgentsProvider` from `agent.context_providers`; pair with `todos_remaining_message` / `background_tasks_running_message`.

#### `BackgroundAgentsProvider` + `BackgroundTaskInfo` + `BackgroundTaskStatus`
**Module:** `agent_framework._harness._background_agents`

Lets a parent agent delegate to named sub-agents running concurrently in separate sessions (`asyncio.Task` fan-out). Six tools: `background_agents_start_task`, `_wait_for_first_completion`, `_get_task_results`, `_get_all_tasks`, `_continue_task`, `_clear_completed_task`.

```python
BackgroundAgentsProvider(agents: Sequence[SupportsAgentRun], *, source_id="background_agents",
                          instructions: str | None = None)
BackgroundTaskInfo(id: int, agent_name: str, description: str,
                   status: BackgroundTaskStatus = RUNNING, result_text=None, error_text=None)
```
`BackgroundTaskStatus`: `RUNNING, COMPLETED, FAILED, LOST` (`LOST` = the in-process `asyncio.Task` reference was lost, e.g. process restart). Each agent needs a unique, non-empty `name`.

#### Orchestration builders — `MagenticBuilder`, `GroupChatBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `HandoffBuilder`
**Package:** `agent-framework-orchestrations` (`agent_framework.orchestrations`)

> **Correction:** an older per-volume page documented `SequentialBuilder`/`ConcurrentBuilder`/`GroupChatBuilder`/`HandoffBuilder`/`MagenticBuilder` as taking a positional `agents: list[Agent]` constructor argument, with a `SequentialBuilder.with_request_info()` method and `WorkflowBuilder.add_group_chat()` / `.add_agent_based_group_chat()` helpers. None of that is real: `WorkflowBuilder` has no group-chat methods at all, and every orchestration builder's actual constructor takes agents via the **keyword-only `participants=`** argument (confirmed live against the installed 1.14.0 `agent_framework_orchestrations` package with `inspect.signature()`). Each builder is fully independent and produces its own `Workflow` via `.build()` — see the "Multi-Agent Orchestration Patterns" section above for the current, verified `participants=` usage.

- **`MagenticOrchestrator`** drives Magentic-One's two-level loop (outer replan / inner coordinate): `MagenticOrchestrator(manager: MagenticManagerBase, participant_registry, *, require_plan_signoff=False)`. `MagenticContext` (`@dataclass`) holds `task, chat_history, participant_descriptions, round_count, stall_count, reset_count` — always cloned before being passed to manager calls. `MagenticResetSignal` is a no-field sentinel used only by custom `BaseGroupChatOrchestrator` subclasses. When `require_plan_signoff=True`, the workflow suspends with a `MagenticPlanReviewRequest(plan, current_progress, is_stalled)`; answer with `.approve()` or `.revise(feedback)` → `MagenticPlanReviewResponse(review: list[Message])` (empty = approved). `StandardMagenticManager(agent, *, max_stall_count=3, max_reset_count=None, max_round_count=None, progress_ledger_retry_count=None, ...8 prompt overrides)` wraps an `Agent` for every orchestration LLM call. `MagenticProgressLedger` is a 5-field structured output — `is_request_satisfied, is_in_loop, is_progress_being_made, next_speaker, instruction_or_question`, each a `MagenticProgressLedgerItem(reason: str, answer: str | bool)` (bool for yes/no fields, str for `next_speaker`/`instruction_or_question`) — access as attributes, not by iterating. `MagenticOrchestratorEvent(event_type: MagenticOrchestratorEventType, content: Message | MagenticProgressLedger)`; `MagenticOrchestratorEventType`: `PLAN_CREATED, REPLANNED, PROGRESS_LEDGER_UPDATED`.

- **`GroupChatOrchestrator`** is the lightweight, non-LLM-manager alternative: a plain `selection_func(state: GroupChatState) -> str` (or `async`) picks the next speaker. `GroupChatBuilder(*, participants=None, participant_factories=None, orchestrator_agent=None, orchestrator=None, selection_func=None, orchestrator_name=None, termination_condition=None, max_rounds=None, checkpoint_storage=None, output_from=..., intermediate_output_from=None)` — pass either `selection_func` (deterministic routing) or `orchestrator_agent` (LLM decides). `GroupChatState` (frozen dataclass): `current_round: int, participants: OrderedDict[str, str], conversation: list[Message]`. `TerminationCondition = Callable[[list[Message]], bool | Awaitable[bool]]`. `GroupChatRequestMessage(additional_instruction=None, metadata=None)` / `GroupChatRequestSentEvent(round_index, participant_name)` / `GroupChatResponseReceivedEvent(round_index, participant_name)` are the observable per-round events.

- `SequentialBuilder`, `ConcurrentBuilder`, `HandoffBuilder` + `HandoffConfiguration` + `HandoffSentEvent` round out the orchestration-pattern builders (fan-out sequential chain, parallel fan-out/fan-in, and agent-to-agent handoff respectively) — all take `participants=` plus `checkpoint_storage=`/`output_from=`/`intermediate_output_from=`, matching the signatures shown in "Multi-Agent Orchestration Patterns" above. `AgentOrchestrationOutput`, `create_completion_message`, and `clean_conversation_for_handoff` are the shared result/utility types across these builders.

#### `AgentApprovalExecutor` + `AgentRequestInfoExecutor` + `AgentRequestInfoResponse`
**Package:** `agent-framework-orchestrations` (private module `_orchestration_request_info`; only `AgentRequestInfoResponse` is re-exported at `agent_framework.orchestrations` top level)

Implements the agent-approval-gate pattern: an agent runs, a human reviews the output, and either approves (forwarding downstream) or supplies correction messages (re-running the agent).

```python
AgentApprovalExecutor(agent: SupportsAgentRun,
                       context_mode: Literal["full", "last_agent", "custom"] | None = None, *,
                       allow_direct_output: bool = False)
```
`allow_direct_output=True` is required when the executor is the workflow's terminal node — it makes the workflow's `output` event carry an `AgentResponse` rather than an `AgentExecutorResponse`. `AgentRequestInfoResponse(messages: list[Message])` — `.approve()` (empty `messages` = approval), `.from_strings([...])`, `.from_messages([...])`.

#### Copilot Studio / Claude Code CLI / GitHub Copilot — alternate agent backends

- **`CopilotStudioAgent`** (`agent-framework-copilotstudio`, `agent_framework.microsoft`) wraps a published Microsoft Copilot Studio app as a native agent, authenticating via MSAL (silent-then-interactive) with connection params resolvable from `COPILOTSTUDIOAGENT__*` environment variables (`ENVIRONMENTID`, `SCHEMANAME`, `AGENTAPPID`, `TENANTID`).
- **`ClaudeAgent` / `RawClaudeAgent`** (`agent-framework-claude`) shell out to the **Claude Code CLI** (not the Anthropic HTTP API — that's `agent_framework.anthropic`). `AGENT_PROVIDER_NAME = "anthropic.claude"`. `ClaudeAgentOptions` (TypedDict) covers `model` (`"sonnet"|"opus"|"haiku"`), `permission_mode` (`"default"|"acceptEdits"|"plan"|"bypassPermissions"`), `allowed_tools`/`disallowed_tools`, `mcp_servers`, `max_turns`, `max_budget_usd`, `sandbox`. Always use as an async context manager.
- **`RawGitHubCopilotAgent`** (`agent-framework-github-copilot`, `agent_framework.github`) drives the GitHub Copilot CLI (`AGENT_PROVIDER_NAME = "github.copilot"`) with an `on_pre_tool_use`/`on_permission_request` hook pair and BYOK provider config, analogous in shape to the Claude Code integration.

#### Provider chat/embedding clients — Gemini, Mistral, Anthropic Vertex, Bedrock, OpenAI Chat Completions

- **`GeminiChatClient` / `RawGeminiChatClient`** (`agent-framework-gemini`, `agent_framework.gemini`) target either the Gemini Developer API or Vertex AI: `GeminiChatClient(*, api_key=None, model=None, vertexai=None, project=None, location=None, credentials=None, client=None, middleware=None, function_invocation_configuration=None)`. `GeminiChatOptions` adds `top_k`, `response_schema`, `thinking_config: ThinkingConfig {include_thoughts, thinking_budget (0=off, -1=dynamic), thinking_level}`. Env resolution: `GOOGLE_*` vars win over `GEMINI_*` (`GeminiSettings` vs `GoogleGeminiSettings`).
- **`MistralEmbeddingClient`** (`agent-framework-mistral`) — `OTEL_PROVIDER_NAME = "mistralai"`; resolves `MISTRAL_API_KEY`, `MISTRAL_EMBEDDING_MODEL`, `MISTRAL_SERVER_URL` (self-hosted override).
- **`AnthropicVertexClient` / `RawAnthropicVertexClient`** (`agent_framework.anthropic`, also importable as `agent_framework.google`) run Claude via Google Vertex AI (`OTEL_PROVIDER_NAME = "google.vertex.ai"`); resolves `CLOUD_ML_REGION` / `ANTHROPIC_VERTEX_PROJECT_ID`.
- **`BedrockGuardrailConfig`** (`agent_framework.amazon`) is a TypedDict mapping to Bedrock's Converse-API `guardrailConfig` — `guardrailIdentifier`, `guardrailVersion`, `trace`, `streamProcessingMode`; set it in `Agent(default_options={"guardrailConfig": ...})` or per-call via `agent.run(options=...)`.
- **`OpenAIChatCompletionClient`** (Chat Completions API, complementing the Responses-API `OpenAIChatClient`) adds `OpenAIChatCompletionOptions` fields `logprobs`, `top_logprobs`, `prediction: Prediction` (speculative decoding — supply expected output text to reuse cached tokens), `store`, `web_search_options`, `stream_options: StreamOptions {include_usage}`. `ReasoningOptions {effort: "none"|"low"|"medium"|"high"|"xhigh", summary}` and `OpenAIContinuationToken {response_id}` (background response polling) extend the Responses-API surface. `OpenAIContentFilterException` (Azure content-filter block) carries `content_filter_code: ContentFilterCodes` and a per-category `content_filter_result: dict[str, ContentFilterResult{filtered, severity: ContentFilterResultSeverity}]`.

### Memory, Sessions & File Stores

#### `AgentSession`
**Module:** `agent_framework._sessions`

```python
AgentSession(*, session_id: str | None = None, service_session_id: str | ServiceSessionId | None = None)
```
`session_id` auto-generates a UUID; `service_session_id` is a provider-managed identifier (e.g. an Azure AI Agents thread ID). `session.state: dict[str, Any]` is the free-form bag every `ContextProvider` reads/writes. `.to_dict()` / `AgentSession.from_dict()` round-trip for persistence.

#### `ContextProvider` (base class)
The common extension point underlying memory, todo, mode, file-access, background-agent, and skills harnesses. Subclasses override `before_run`/`after_run` hooks to inspect/mutate the outgoing request and incoming response respectively.

```python
class ContextProvider:
    async def before_run(self, context: "AgentContext") -> None: ...
    async def after_run(self, context: "AgentContext") -> None: ...
```

#### `HistoryProvider` + `InMemoryHistoryProvider` + `FileHistoryProvider`
**Module:** `agent_framework._sessions`

`HistoryProvider(ContextProvider)` ABC — subclass and implement `async get_messages(session_id, *, state=None, **kwargs) -> list[Message]` and `async save_messages(session_id, messages, *, state=None, **kwargs) -> None`. Both built-ins share the flags `load_messages`, `store_inputs`, `store_context_messages`, `store_context_from: set[str] | None`, `store_outputs`, `skip_excluded` (omit compaction-excluded messages on reload).

```python
InMemoryHistoryProvider(source_id: str | None = None, *, load_messages=True, store_inputs=True,
                         store_context_messages=False, store_context_from=None, store_outputs=True,
                         skip_excluded=False)
FileHistoryProvider(storage_path: str | Path, *, source_id="file_history", load_messages=True,
                     store_inputs=True, store_context_messages=False, store_context_from=None,
                     store_outputs=True, skip_excluded=False, serialization_format: Literal["json","msgpack"]="json",
                     dumps=None, loads=None)
```
`InMemoryHistoryProvider` stores under `session.state[provider.source_id]["messages"]` — clear history by resetting that key, not a top-level `"messages"` key. `FileHistoryProvider` writes one JSON-Lines file per session; 64-slot `threading.Lock` stripe array for concurrent writes; path-traversal-guards `session_id` and encodes Windows-reserved filenames. `serialization_format="msgpack"` is a newer addition (not present in older docs, which only showed JSON).

`PerServiceCallHistoryPersistingMiddleware(*, agent, session, providers: Sequence[HistoryProvider], service_stores_history=False)` persists history after **every individual model call** within one agent run (not just at the end) — activated by `Agent(require_per_service_call_history_persistence=True)`. When `service_stores_history=True` it becomes write-only (loading is skipped; the real `conversation_id` is preserved instead of a local sentinel).

#### `MemoryStore` + `MemoryFileStore` + `MemoryContextProvider`
**Module:** `agent_framework._harness._memory`

Durable, cross-session, per-owner topic memory (distinct from per-session `FileMemoryProvider`, below). `MemoryStore` is the abstract backing interface (`list_topics`, `get_topic`, `write_topic`, `delete_topic`, `rebuild_index`, `get_index_text`, `read_state`/`write_state`, `get_transcripts_directory`, `search_transcripts` — all synchronous, plus concrete `get_owner_id`/`export_provider_state`/`import_provider_state`). `MemoryFileStore` is the file-backed implementation:

```python
MemoryFileStore(base_path: str | Path, *, kind="memory", owner_prefix="", owner_state_key: str,
                 index_file_name="MEMORY.md", topics_directory_name="topics",
                 transcripts_directory_name="transcripts", state_file_name="state.json", dumps=None, loads=None)
```
Both `source_id` and the owner ID (from `session.state[owner_state_key]`) are URL-safe base64 encoded (no padding) into the directory path — e.g. source `"memory"` + owner `"carol"` → `bWVtb3J5/Y2Fyb2w/`. `get_owner_id` rejects path-traversal owner values. `MemoryTopicRecord(*, topic, slug=None, summary, memories: Sequence[str], updated_at, session_ids=None)` auto-slugifies and de-duplicates `memories`.

```python
MemoryContextProvider(recent_turns: int = 0, load_tool_turns: bool = True, *, store: MemoryStore,
                       source_id="memory", context_prompt=None, index_line_limit=200, index_line_length=150,
                       selection_limit=3, max_extractions=5,
                       consolidation_interval: timedelta = timedelta(days=1), consolidation_min_sessions=5,
                       extraction_prompt=..., consolidation_prompt=..., consolidation_client=None,
                       history_message_filter=None, history_dumps=None, history_loads=None)
```
A `HistoryProvider` subclass that also injects the memory index + selected topics into the system prompt and runs a background LLM extraction/consolidation pass. `consolidation_client` lets you use a cheaper model for periodic consolidation than the main agent client. This matches the constructor reference already documented in this guide's "Long-Term Memory" section above.

```python
store = MemoryFileStore(base_path="./agent_memory", owner_state_key="user_id")
memory = MemoryContextProvider(store=store, recent_turns=3, selection_limit=5)
agent = Agent(client=client, context_providers=[memory])
session = agent.create_session()
session.state["user_id"] = "alice"
await agent.run("My favourite language is Python.", session=session)
```

#### `FileMemoryProvider` + `FileAccessProvider` + `AgentFileStore` hierarchy
**Module:** `agent_framework._harness._file_memory` / `._file_access`

`FileMemoryProvider` gives an agent **session-scoped** (or `scope=`-shared) file memory through seven tools (`file_memory_write/read/delete/ls/grep/replace/replace_lines`):

```python
FileMemoryProvider(store: AgentFileStore, *, source_id="file_memory", scope: str | None = None,
                    instructions: str | None = None)
```
`scope=None` isolates per session (keyed by `session_id`); pass a user ID to share across sessions. All write/delete ops serialize through an `asyncio.Lock` so the `memories.md` index stays consistent.

`FileAccessProvider` is the analogous **shared, persistent** CRUD+grep provider (7 tools: `file_access_write/read/delete/ls/grep/replace/replace_lines`):

```python
FileAccessProvider(store: AgentFileStore, *, source_id="file_access", instructions=None,
                    disable_write_tools=False, disable_readonly_tool_approval=False,
                    disable_write_tool_approval=False)
```
Every tool defaults to `approval_mode="always_require"`; two static helpers produce standing `ToolApprovalRule`s for `ToolApprovalMiddleware`: `FileAccessProvider.read_only_tools_auto_approval_rule` and `.all_tools_auto_approval_rule`.

`AgentFileStore` (ABC): `async write(path, content, *, overwrite=True)`, `read(path) -> str | None`, `delete(path) -> bool`, `list_children(directory="") -> list[FileStoreEntry]`, `search(pattern, *, glob=None, base_dir="") -> list[FileSearchResult]`. Paths are always relative; implementations must reject `..` traversal and make `overwrite=False` an atomic exclusive create. `FileStoreEntry(name: str, type: str)` — class constants `FILE="file"` / `DIRECTORY="directory"`. `InMemoryAgentFileStore()` is dict-backed (lower-cased keys, single `asyncio.Lock`). `FileSystemAgentFileStore(root_directory: str | os.PathLike)` is disk-backed with symlink rejection (`O_NOFOLLOW` on POSIX) and lazy root creation. `FileSearchMatch(line_number, line)` / `FileSearchResult(file_name, snippet="", matching_lines=None)` are the search DTOs.

#### `TodoProvider` + `TodoItem` + `TodoSessionStore` + `TodoFileStore`
**Module:** `agent_framework._harness._todo`

Structured task planning inside a session. Five tools: `todos_add`, `todos_complete`, `todos_remove`, `todos_get_remaining`, `todos_get_all`.

```python
TodoProvider(source_id: str = "todo", *, instructions: str | None = None, store: TodoStore | None = None)
TodoItem(id: int, title: str, description: str | None = None, is_complete: bool = False)
TodoFileStore(base_path: str | Path, *, kind="todos", owner_prefix="", owner_state_key: str | None = None,
              state_filename="todos.json")
```

> **Correction vs. earlier docs:** several older per-volume pages documented `TodoProvider`'s default `source_id` as `"todos"` — the installed default is **`"todo"`** (singular), matching this guide's own "Agent Todo List" section above.

`TodoInput(title, description=None)` / `TodoCompleteInput(id, reason)` are the tool argument schemas. `TodoSessionStore()` (default backing store) keeps state in `session.state[source_id]`; `TodoFileStore` persists per-session JSON, sanitizing owner/session path segments against traversal and Windows-reserved names. A `WeakKeyDictionary[AgentSession, asyncio.Lock]` guards concurrent read-modify-write.

#### Provider-specific session/memory stores
- **`FoundrySessionStore`** (`agent_framework.foundry`) extends `FileSessionStore` with per-`user_id` subdirectory isolation (`validate_path_segment` + `.resolve().is_relative_to()` traversal guard) for Foundry-hosted deployments; `FoundrySessionStore(storage_path, *, serialisation_format: Literal["json", "msgpack"] = "json")`.
- **`CosmosHistoryProvider`** (`agent-framework-azure-cosmos`), **`AzureAISearchContextProvider`** (`agent-framework-azure-ai-search`), **`Mem0ContextProvider`** (`agent-framework-mem0`) — alternate managed-service-backed `HistoryProvider`/`ContextProvider` implementations following the same `ContextProvider` contract as the built-ins above.
- **`ConversationStore` / `InMemoryConversationStore` / `CheckpointConversationManager`** (`agent_framework_devui._conversations`) back the DevUI's OpenAI-compatible `/v1/conversations` surface; `CheckpointConversationManager` hands each conversation its own isolated `InMemoryCheckpointStorage` so `FunctionalWorkflow` HITL state survives across multiple HTTP turns in the same DevUI session.
- **`AGUIThreadSnapshot` / `AGUIThreadSnapshotStore` / `InMemoryAGUIThreadSnapshotStore`** (`agent-framework-ag-ui`, `agent_framework.ag_ui`) persist replayable AG-UI thread state: `AGUIThreadSnapshot(messages: list[dict], state: dict | None, interrupt: list[dict] | None, session_state: dict | None)` — `session_state` is **private** and must never be replayed to the client. `AGUIThreadSnapshotStore` is a `@runtime_checkable` Protocol (`save`/`get`/`delete`/`clear`, all keyed by `(scope, thread_id)`). `InMemoryAGUIThreadSnapshotStore(*, max_snapshots=100)` LRU-evicts the oldest entry when full; raises `ValueError` if `max_snapshots < 1`.

**`FileMemoryProvider`, `FileSessionStore`, `SessionStore`** postdate the source volumes that only went up to ~1.9.0 — the fully documented `FileMemoryProvider` above is the current 1.14.0 shape. `SessionStore` is the general abstract persistence interface for conversation/session state; `FileSessionStore` is its local-disk implementation.

### Tools & MCP

#### `FunctionTool` + `FunctionInvocationConfiguration` + `FunctionInvocationLayer`
**Module:** `agent_framework._tools`

```python
FunctionTool(*, name: str, description: str = "", approval_mode: ApprovalMode | None = None,
             kind: str | None = None, max_invocations: int | None = None,
             max_invocation_exceptions: int | None = None, additional_properties: dict | None = None,
             func: Callable[..., Any] | None = None, input_model: type[BaseModel] | Mapping | None = None,
             result_parser: Callable[[Any], str | list[Content]] | _SkipParsingSentinel | None = None)
```
`max_invocations` is a **lifetime** counter on the instance (accumulates across all requests — reset it manually via `tool.invocation_count = 0`). `func=None` creates a **declaration-only** tool: it appears in the schema but is never executed server-side (the call short-circuits into `response.user_input_requests`). `result_parser` (or the sentinel `SKIP_PARSING`) controls how the return value is serialized back to the model. `FunctionTool` is the object the `@tool` decorator used throughout this guide builds under the hood (an older per-volume page called this wrapper `AIFunction` / `@ai_function` — that name predates the current release; `@tool` / `FunctionTool` is what 1.14.0 actually ships).

`FunctionInvocationLayer` is the MRO mixin every concrete chat client inherits, adding the tool-execution loop:

```python
FunctionInvocationLayer(*, middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
                         function_invocation_configuration: FunctionInvocationConfiguration | None = None)
```
`FunctionInvocationConfiguration` (TypedDict): `enabled: bool`, `max_iterations: int` (LLM round-trips, not individual calls), `max_function_calls: int | None` (total individual invocations, checked *after* each parallel batch — best-effort), `max_consecutive_errors_per_request: int`, `terminate_on_unknown_calls: bool`, `additional_tools: Sequence[FunctionTool]` (hidden from the model's schema but callable), `include_detailed_errors: bool`. Set per-client: `client.function_invocation_configuration["max_iterations"] = 3`.

#### MCP tools — `MCPStdioTool` / `MCPStreamableHTTPTool` / `MCPWebsocketTool` + `MCPTaskOptions`
**Module:** `agent_framework._mcp`

All three transports share the same keyword surface (`tool_name_prefix`, `load_tools`, `parse_tool_results`, `load_prompts`, `parse_prompt_results`, `request_timeout`, `session`, `description`, `approval_mode: Literal["always_require","never_require"] | MCPSpecificApproval | None`, `allowed_tools: Collection[str] | None`, `use_progressive_disclosure`, `always_load`, `client`, `sampling_approval_callback`, `sampling_max_tokens=4096`, `sampling_max_requests=25`, `task_options: MCPTaskOptions | None`, `additional_tool_argument_names: Sequence[str] | Mapping[str, Sequence[str]] | None`), plus transport-specific positionals: `MCPStdioTool(name, command, *, args=None, env=None, ...)`, `MCPStreamableHTTPTool(name, url, *, http_client=None, header_provider=None, terminate_on_close=None, ...)`, `MCPWebsocketTool(name, url, ...)`.

- `allowed_tools` restricts which remote tools are exposed; `MCPSpecificApproval` is a per-tool-name approval-mode override dict.
- `additional_tool_argument_names` injects extra hidden arguments into every tool call, never shown in the schema sent to the model — useful for `user_id`/`tenant_id` passed via `agent.run(function_invocation_kwargs=...)`.
- `MCPTaskOptions(default_ttl: timedelta | None = None, cancel_remote_task_on_local_cancellation: bool = True, max_task_wait: timedelta | None = None)` configures SEP-2663 long-running MCP task lifecycle — this is the concrete class backing the "Long-running MCP tasks (SEP-2663)" support mentioned by older docs: a tool call can return a task handle immediately and the client polls/subscribes for completion instead of blocking the whole agent turn on a slow remote tool.

```python
fs_tool = MCPStdioTool(
    name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    allowed_tools={"read_file", "list_directory"}, tool_name_prefix="fs_", approval_mode="never_require",
)
async with fs_tool:
    agent = Agent(client=client, tools=[fs_tool])
```

> **Correction:** MCP sampling (an MCP server itself requesting an LLM completion from the connecting client — the "confused deputy" mitigation surface) is configured via `sampling_approval_callback` on the tool constructor above, not a `sampling_handler=` parameter as an older per-volume page named it.

#### MCP error taxonomy — `ToolExecutionException` / `_MCPTaskAbandoned` / `_MCPDeadlineExpired`
**Module:** `agent_framework.exceptions` / `agent_framework._mcp`

```
AgentFrameworkException
 └─ ToolException
      └─ ToolExecutionException
           └─ _MCPTaskAbandoned      # internal — catch ToolExecutionException instead
Exception
 └─ _MCPDeadlineExpired              # internal sentinel, NOT a ToolException subclass
```
`_MCPTaskAbandoned` fires when `MCPTaskOptions.max_task_wait` expires while the remote task may still be running — the leading underscore signals it's internal; catch `ToolExecutionException` in application code. `UserInputRequiredException(contents: list[Any], message="Tool requires user input to proceed.")` (subclass of `ToolException`) propagates `oauth_consent_request`/`function_approval_request` content items from a tool-wrapped sub-agent back into the parent's response rather than swallowing them as a generic error.

#### Sandboxed code execution — `MontyExecuteCodeTool` + `FileMount` + `MontyCodeActProvider`
**Package:** `agent-framework-monty` (`agent_framework.monty`)

`MontyExecuteCodeTool` is a `FunctionTool` that runs Python inside **Monty**, a lightweight sandbox interpreter (alternative to Hyperlight); registered host tools appear as typed async functions inside the sandbox, argument-checked by the `ty` type checker.

```python
MontyExecuteCodeTool(*, tools=None, approval_mode: ApprovalMode | None = None,
                      workspace_root: str | Path | None = None,
                      file_mounts: FileMountInput | Sequence[FileMountInput] | None = None,
                      resource_limits: dict[str, Any] | None = None)
FileMount(host_path: str | Path, mount_path: str, mode: MountMode = "overlay", write_bytes_limit=None)
```
`mode`: `"overlay"` (writes captured in-process, host dir untouched), `"rw"` (writes go to `host_path`), `"ro"` (write attempts raise). `MontyCodeActProvider` is the `ContextProvider` wrapper (same params, forwards `.add_tools()/.get_tools()/.remove_tool()/.clear_tools()/.add_file_mounts()` to the inner tool) — the recommended way to add code execution without managing the tool directly.

#### `FoundryToolbox`
**Package:** `agent-framework-foundry-hosting` (`agent_framework.foundry`)

`MCPStreamableHTTPTool` subclass targeting a Microsoft Foundry toolbox MCP endpoint with bearer-token auth.

```python
FoundryToolbox(credential: TokenCredential, *, url: str | None = None, name: str | None = None,
               token_scope: str = "https://ai.azure.com/.default", load_prompts: bool = False,
               load_tools: bool = True, timeout: float = 120.0)
```
> **Correction vs. earlier docs:** an older per-volume page documented `timeout: float = 30.0` — the installed default is **`120.0`**. Endpoint resolves from `TOOLBOX_ENDPOINT`, or `FOUNDRY_PROJECT_ENDPOINT` + `TOOLBOX_NAME`.

#### `MessageInjectionMiddleware` + `enqueue_messages`
**Module:** `agent_framework._middleware` (top-level `agent_framework`)

A `ChatMiddleware` letting tool code enqueue messages into the *next* model call within the same session, without breaking a running tool-call loop.

```python
MessageInjectionMiddleware()
enqueue_messages(session: AgentSession, messages: AgentRunInputs) -> None   # module-level free function
middleware.get_pending_messages(session) -> list[Message]                   # point-in-time snapshot
```
If the middleware drains new messages and the last model response had no pending tool calls, it loops internally with the model until nothing new is queued. `MiddlewareBundle` (also new relative to the older per-volume pages) groups multiple middleware instances so they can be attached to an agent as a single named unit.

#### Declarative-workflow tool executors
**Package:** `agent-framework-declarative` (see also **A2A Protocol & Declarative Graphs** below) — `FunctionTool` / `OpenApiTool` / `WebSearchTool` / `FileSearchTool` / `CodeInterpreterTool` (the YAML-driven declarative-model subclasses, distinct from the harness `FunctionTool` above) and `BaseToolExecutor` / `InvokeFunctionToolExecutor` / `ToolApprovalRequest` / `ToolApprovalResponse` / `ToolInvocationResult` (the declarative-workflow HITL/tool executor pair) are covered under **A2A Protocol & Declarative Graphs** to keep the two `ToolApprovalRule` namespaces (harness vs. declarative) from being conflated.

### Security & FIDES Content Labels

"FIDES" is the framework's information-flow-control (IFC) subsystem: every piece of content flowing through an agent run can carry integrity and confidentiality labels, and policy enforcement can block, redact, or quarantine actions based on those labels (e.g. preventing content sourced from an untrusted tool result from being used to justify a sensitive action). All symbols below are `@experimental(feature_id=ExperimentalFeature.FIDES)` in `agent_framework.security`.

#### `IntegrityLabel` + `ConfidentialityLabel` + `ContentLabel` + `LabeledMessage`

```python
class IntegrityLabel(str, Enum): TRUSTED = "trusted"; UNTRUSTED = "untrusted"
class ConfidentialityLabel(str, Enum): PUBLIC = "public"; PRIVATE = "private"; USER_IDENTITY = "user_identity"
ContentLabel(integrity: IntegrityLabel = TRUSTED, confidentiality: ConfidentialityLabel = PUBLIC,
             metadata: dict[str, Any] | None = None)
LabeledMessage(role: str, content: Any, security_label: ContentLabel | None = None, message_index=None,
                source_labels: list[ContentLabel] | None = None, metadata: dict | None = None)
```
`ContentLabel.is_trusted()` / `.is_public()`; string constructor args are coerced to the enum (`ContentLabel("untrusted", "private")` works). `combine_labels(*labels)` merges to the most restrictive result (any `UNTRUSTED` wins on integrity; `max()` over `{PUBLIC:0, PRIVATE:1, USER_IDENTITY:2}` on confidentiality; metadata merged left-to-right; zero args → fresh `ContentLabel()`). `check_confidentiality_allowed(context_label, max_allowed) -> bool` returns `True` only when it's safe to send — e.g. blocks `PRIVATE` context data reaching a `PUBLIC` destination.

#### `LabelTrackingFunctionMiddleware` + `PolicyEnforcementFunctionMiddleware`

```python
LabelTrackingFunctionMiddleware(default_integrity: IntegrityLabel = UNTRUSTED,
                                 default_confidentiality: ConfidentialityLabel = PUBLIC,
                                 auto_hide_untrusted: bool = True, hide_threshold: IntegrityLabel = UNTRUSTED)
PolicyEnforcementFunctionMiddleware(allow_untrusted_tools: set[str] | None = None,
                                     block_on_violation: bool = True, enable_audit_log: bool = True,
                                     approval_on_violation: bool = False)
```
`LabelTrackingFunctionMiddleware` labels every tool result via a strict 3-tier priority: (1) a per-item embedded `additional_properties.security_label` in the result always wins; (2) the tool's declared `@tool(additional_properties={"source_integrity": "trusted"|"untrusted"})`; (3) `combine_labels()` of the input argument labels. `PolicyEnforcementFunctionMiddleware` runs *before* execution and blocks (or, with `approval_on_violation=True`, routes to HITL approval instead of blocking) any tool call made in an untrusted context unless the tool name is in `allow_untrusted_tools`. Stack `LabelTrackingFunctionMiddleware` **before** `PolicyEnforcementFunctionMiddleware` in the `middleware=[...]` list.

#### `SecureAgentConfig`

One-liner `ContextProvider` that wires the whole stack (label tracker + optional policy enforcer + quarantine tool injection) into an agent.

```python
SecureAgentConfig(auto_hide_untrusted: bool = True, default_integrity: IntegrityLabel = UNTRUSTED,
                   default_confidentiality: ConfidentialityLabel = PUBLIC,
                   allow_untrusted_tools: set[str] | None = None, block_on_violation: bool = True,
                   approval_on_violation: bool = False, enable_audit_log: bool = True,
                   enable_policy_enforcement: bool = True, quarantine_chat_client: SupportsChatGetResponse | None = None,
                   source_id: str | None = None)
```

> **Correction — re-verified live against the installed 1.14.0 signature (`inspect.signature(SecureAgentConfig.__init__)`):** `auto_hide_untrusted` and `enable_audit_log` both default to **`True`**, not `False` as several older per-volume pages showed — the "Prompt Injection Defense" section's constructor-reference table above has been corrected to match.

After construction: `security.label_tracker: LabelTrackingFunctionMiddleware`, `security.policy_enforcer: PolicyEnforcementFunctionMiddleware | None`. The `quarantined_llm` tool it injects reads a **process-global** quarantine-client slot (set via `set_quarantine_client()`) — the most-recently-constructed `SecureAgentConfig`'s client wins process-wide if you build more than one with different `quarantine_chat_client` values.

#### `SecureMCPToolProxy` + `apply_mcp_security_labels`

Auto-labels every tool an MCP server advertises at connection time, so `LabelTrackingFunctionMiddleware` / `PolicyEnforcementFunctionMiddleware` intercept MCP tool calls exactly like local ones. **This matters because hosted MCP** (`client.get_mcp_tool()`) runs the MCP server on the provider's infrastructure, bypassing all local middleware entirely — `SecureMCPToolProxy` runs it locally instead.

```python
SecureMCPToolProxy(mcp_tool: MCPTool | None = None, *, url: str | None = None,
                    headers: dict[str, str] | None = None, name: str | None = None, description: str | None = None,
                    default_integrity: IntegrityLabel = UNTRUSTED,
                    annotation_overrides: dict[str, tuple[IntegrityLabel, ConfidentialityLabel | None]] | None = None,
                    mark_write_tools_as_sinks: bool = True)
```
Exactly one of `mcp_tool` (wrap an existing `MCPStdioTool`/etc.) or `url` (auto-creates an `MCPStreamableHTTPTool`) is required. `proxy.tools` (alias `proxy.functions`) is the labeled tool list to pass to `Agent(tools=...)`. `apply_mcp_security_labels(mcp_tool, *, default_integrity=..., mark_write_tools_as_sinks=...)` is the underlying async function, for callers managing the MCP tool lifecycle themselves.

```python
async with SecureMCPToolProxy(MCPStdioTool(name="github", command="gh-mcp", args=["stdio"])) as proxy:
    agent = Agent(client=client, tools=proxy.tools,
                   middleware=[LabelTrackingFunctionMiddleware(), PolicyEnforcementFunctionMiddleware()])
```

#### `ContentVariableStore` + `VariableReferenceContent` + `InspectVariableInput`

The variable-indirection layer that keeps untrusted content out of the LLM context window — the mechanism behind "quarantined LLM calls": a model call made with untrusted (low-integrity) input is routed through a restricted context whose own output is itself labeled low-integrity and cannot directly trigger high-privilege actions without an explicit re-validation step. This is the core mitigation for prompt-injection-via-tool-output scenarios.

```python
ContentVariableStore()                       # dict-backed; store()/retrieve()/exists()/list_variables()/clear()
VariableReferenceContent(variable_id: str, label: ContentLabel, description: str | None = None)
InspectVariableInput(variable_id: str, reason: str | None = None)   # tool-arg schema for inspect_variable
```
`store(content, label) -> str` returns a `var_{16hex}` ID; the LLM only ever sees a `VariableReferenceContent` placeholder. The `inspect_variable` tool taints the calling context to `UNTRUSTED` when used — call sparingly. `store_untrusted_content(content, label=None, description=None)` always writes to the module-level `_global_variable_store` singleton; **when `LabelTrackingFunctionMiddleware` is active, `inspect_variable`/`quarantined_llm` look up the *middleware's own* store instead** — a variable ID from `store_untrusted_content` will `KeyError` there. Use `middleware.get_variable_store().store(...)` directly when middleware is running (an older per-volume page described this indirection as free-standing `get_variable_store()`/`set_variable_store()` functions — the real accessor is scoped to the middleware instance, as shown here). `get_security_tools() -> [quarantined_llm, inspect_variable]` returns the two `FunctionTool`s for `Agent(tools=[..., *get_security_tools()])`. `quarantined_llm` sends the raw content of a variable to a separate chat client (registered via `set_quarantine_client()` / read via `get_quarantine_client()`) isolated from the main context; if none is registered it logs a warning and returns a placeholder string rather than raising.

#### Microsoft Purview compliance
**Package:** `agent-framework-purview` (`agent_framework.microsoft`)

`PurviewPolicyMiddleware(credential, settings: PurviewSettings, cache_provider: CacheProvider | None = None)` enforces Purview DLP at the agent-middleware layer: a pre-check on the prompt (`Activity.UPLOAD_TEXT` — on block, sets `context.result` to `blocked_prompt_message` and raises `MiddlewareTermination`), then the agent runs, then a post-check on the response (`Activity.GENERATE_RESPONSE`). `PurviewChatPolicyMiddleware` is the chat-middleware-layer equivalent for bare chat-client usage. `PurviewSettings` (TypedDict): `app_name, app_version, tenant_id, purview_app_location: PurviewAppLocation, graph_base_uri, blocked_prompt_message, blocked_response_message, ignore_exceptions, ignore_payment_required, cache_ttl_seconds (default 14400), max_cache_size_bytes`. `CacheProvider` is a Protocol (`get`/`set`/`remove`) for sharing the policy cache across worker instances (e.g. Redis).

### Skills

The Agent Skills spec (agentskills.io) defines a `Skill` as a bundle of instructions plus optional `SkillResource`s (reference material loaded on demand) and `SkillScript`s (executable helpers), following a progressive-disclosure pattern — the agent sees only the skill's frontmatter/summary until it actively loads a resource or runs a script. `Skill`, `SkillResource`, and `SkillScript` are the abstract base classes defining that contract.

#### `InlineSkill` + `ClassSkill` + `FileSkillsSource`
**Module:** `agent_framework._skills`

```python
InlineSkill(*, frontmatter: SkillFrontmatter, instructions: str, resources: Sequence[SkillResource] | None = None,
            scripts: Sequence[SkillScript] | None = None, argument_parser=None)
FileSkillsSource(skill_paths: str | Path | Sequence[str | Path], *, script_runner=None,
                  resource_extensions: tuple[str, ...] = (".md",".json",".yaml",".yml",".csv",".xml",".txt"),
                  script_extensions: tuple[str, ...] = (".py",), search_depth: int = 2,
                  script_filter: Callable[[str, str], bool] | None = None,
                  resource_filter: Callable[[str, str], bool] | None = None)
```
`InlineSkill` requires a `SkillFrontmatter(name, description, *, version=None)` built first — an older per-volume page constructed it directly as `InlineSkill(name=..., description=..., instructions=...)`, which is not a valid call; the frontmatter object is mandatory. `ClassSkill` lets a skill be defined as a Python class instead of a filesystem directory — `@ClassSkill.resource(name=...)` / `@ClassSkill.script(name=...)` decorate methods for auto-discovery. `search_depth` (default 2) / `script_filter` / `resource_filter` are 1.10.0-era additions controlling how deep `FileSkillsSource` scans and which files it includes.

```python
frontmatter = SkillFrontmatter(name="pdf_summarizer", description="Summarize PDF documents.")
skill = InlineSkill(frontmatter=frontmatter, instructions="Read the PDF and produce a 3-bullet summary.")
```

#### `SkillsProvider` + `SkillsSource` composition pipeline

```python
SkillsProvider(source: SkillsSource | Sequence[Skill] | Skill, *, instruction_template=None,
               disable_caching=False, cache_refresh_interval: timedelta | None = None,
               disable_load_skill_approval=False, disable_read_skill_resource_approval=False,
               disable_run_skill_script_approval=False, source_id: str | None = None)
```

> **Correction:** an older per-volume page documented a `require_script_approval` parameter and a plain `skills: list[Skill]` first argument — neither matches the installed signature. The first argument accepts a `SkillsSource`, a single `Skill`, or a sequence of them, and approval is controlled by the three separate `disable_*_approval` flags shown above (plus the newer `disable_caching`/`cache_refresh_interval`/`instruction_template`/`source_id`).

`SkillsProvider.from_paths([...], **FileSkillsSource kwargs)` is the convenience wrapper. The decorator-pattern composition classes each wrap another `SkillsSource`:

| Class | Purpose |
|---|---|
| `AggregatingSkillsSource(sources: Sequence[SkillsSource])` | fan-in multiple sources into one |
| `FilteringSkillsSource(inner_source, predicate: Callable[[Skill, SkillsSourceContext], bool])` | context-aware predicate filter |
| `DeduplicatingSkillsSource(inner_source)` | first-one-wins by case-insensitive name |
| `CachingSkillsSource(inner_source, *, cache_isolation_key_selector=None, refresh_interval: timedelta \| None = None)` | per-key `asyncio.Lock`-guarded cache; a failed fetch never poisons the cache |
| `InMemorySkillsSource(skills: Sequence[Skill])` | serves pre-built `Skill` instances, no I/O |
| `MCPSkillsSource` | discovers skills from an MCP server |

`SkillsSourceContext(agent: SupportsAgentRun, session: AgentSession | None = None)` is the frozen context every source/decorator receives on `get_skills(context)`.

```python
combined = DeduplicatingSkillsSource(
    AggregatingSkillsSource([FileSkillsSource("./skills/core"), InMemorySkillsSource([my_inline_skill])])
)
agent = Agent(client=client, context_providers=[SkillsProvider(combined)])
```

### Checkpointing & Compaction

#### `CheckpointStorage` (Protocol) + `FileCheckpointStorage` + `InMemoryCheckpointStorage` + `WorkflowCheckpoint`
**Module:** `agent_framework._workflows._checkpoint`

`CheckpointStorage` is a structural `Protocol` (six methods: `save`, `load`, `list_checkpoints`, `delete`, `get_latest`, `list_checkpoint_ids`), **not** `@runtime_checkable` — `isinstance()` checks raise `TypeError`; use duck typing or `typing.cast()`.

```python
FileCheckpointStorage(storage_path: str | Path, *, allowed_checkpoint_types: list[str] | None = None)
InMemoryCheckpointStorage()   # no args — recommended for tests
```
Each checkpoint file is JSON at the top level; complex Python objects (Pydantic models, dataclasses) are pickled, base64-encoded, and embedded as strings. `allowed_checkpoint_types` extends the deserialization allowlist (built-ins, `datetime`, `uuid`, all `agent_framework.*`, `openai.types.*`) with application types in `"module:qualname"` form — a security control against arbitrary pickle deserialization.

`WorkflowCheckpoint` (`@dataclass(slots=True)`): `workflow_name, graph_signature_hash, checkpoint_id, previous_checkpoint_id, timestamp, messages, state, pending_request_info_events, iteration_count, metadata, version`. `state` holds only **committed** values (never in-flight `State` mutations); `_executor_state` is a reserved per-executor sub-key. Pass `checkpoint_storage=` per-run to `workflow.run(...)` to override the storage the workflow was built with; resume with `checkpoint_id=<id>` (there is no separate `run_from_checkpoint()` method — see the "Workflows & Execution" `Workflow` entry above).

#### Compaction strategy library
**Module:** `agent_framework._compaction`

All strategies implement `async def __call__(self, messages: list[Message]) -> bool` (returns whether it changed anything) and share the annotation constants `GROUP_ANNOTATION_KEY`, `GROUP_ID_KEY`, `GROUP_KIND_KEY` (`"system"|"user"|"assistant_text"|"tool_call"`), `GROUP_INDEX_KEY`, `GROUP_TOKEN_COUNT_KEY`, `EXCLUDED_KEY`, `EXCLUDE_REASON_KEY`, `SUMMARY_OF_MESSAGE_IDS_KEY` / `SUMMARY_OF_GROUP_IDS_KEY` / `SUMMARIZED_BY_SUMMARY_ID_KEY`, `COMPACTION_STATE_KEY`.

```python
SlidingWindowStrategy(*, keep_last_groups: int, preserve_system: bool = True)
SelectiveToolCallCompactionStrategy(*, keep_last_tool_call_groups: int = 1)   # 0 = drop all tool-call groups
TruncationStrategy(*, max_n: int, compact_to: int, tokenizer: TokenizerProtocol | None = None,
                    preserve_system: bool = True)      # token-based when tokenizer given, else message count
ToolResultCompactionStrategy(*, keep_last_tool_call_groups: int = 1)  # collapses old tool groups into a
                                                                       # summary message instead of excluding them
SummarizationStrategy(*, client: SupportsChatGetResponse, target_count: int = 4, threshold: int | None = 2,
                       prompt: str | None = None, max_summary_input_tokens: int | None = 8000, tokenizer=None)
TokenBudgetComposedStrategy(*, token_budget: int, tokenizer: TokenizerProtocol,
                             strategies: Sequence[CompactionStrategy], early_stop: bool = True)
ContextWindowCompactionStrategy(*, max_context_window_tokens: int, max_output_tokens: int, tokenizer=None,
                                 tool_eviction_threshold: float = 0.5, truncation_threshold: float = 0.8,
                                 keep_last_tool_call_groups: int = 4)
CharacterEstimatorTokenizer()   # count_tokens = max(1, len(text) // 4); the zero-dependency default
```

> **Correction:** an older per-volume page gave `SummarizationStrategy`'s constructor as `SummarizationStrategy(summarizer_client: ChatClient, *, max_n: int, compact_to: int)` — those parameter names (`summarizer_client`, `max_n`, `compact_to`) actually belong to a different class (`TruncationStrategy`). The verified `SummarizationStrategy` signature is the one shown above (`client`, `target_count`, `threshold`, `prompt`, `max_summary_input_tokens`, `tokenizer`) — re-checked live via `inspect.signature()`. `TruncationStrategy`'s own signature (also shown above, `max_n`/`compact_to`) is correct as documented in older pages and matches live.

`ContextWindowCompactionStrategy` composes two independent budget phases sized to `max_context_window_tokens - max_output_tokens`: tool eviction fires at 50% of that budget, truncation at 80%. `SummarizationStrategy` triggers when `included_non_system_count > target_count + threshold`; its LLM output becomes a **trusted** part of history going forward — only point `client` at a summarizer you trust as much as the primary model (a compromised one is a persistent indirect-prompt-injection vector). `TokenBudgetComposedStrategy` runs its `strategies` in order until the included-token count is within budget (`early_stop=True` stops as soon as it is); if none succeed, a deterministic fallback excludes oldest groups, then system groups.

```python
CompactionProvider(*, before_strategy: CompactionStrategy | None = None,
                    after_strategy: CompactionStrategy | None = None, tokenizer=None,
                    source_id: str = "compaction", history_source_id: str = "in_memory")
```
`before_strategy` compacts context already loaded before the model call; `after_strategy` compacts the persisted history so the *next* turn starts smaller.

Pipeline helpers: `group_messages(messages, *, id_offset=0, reserved_ids=None)` computes span descriptors; `annotate_message_groups(messages, *, from_index=None, force_reannotate=False, tokenizer=None)` writes them into `additional_properties` and returns the group-ID order; `apply_compaction(messages, *, strategy, tokenizer=None)` runs the full annotate → tokenize → strategy → project pipeline; `included_messages(messages)` / `included_token_count(messages)` read back what a strategy kept; `annotate_token_counts(messages, *, tokenizer, from_index=None, force_retokenize=False)` fills in per-message token counts incrementally.

```python
strategy = ContextWindowCompactionStrategy(max_context_window_tokens=128_000, max_output_tokens=16_384)
agent = Agent(client=client, context_providers=[InMemoryHistoryProvider(), CompactionProvider(before_strategy=strategy)])
```

#### `SerializationMixin` / `DictConvertible`
The framework's two serialization mixins used throughout checkpoint and session persistence: `DictConvertible` provides `to_dict()`/`from_dict()` for plain-dict round-tripping (used by simple state objects), while `SerializationMixin` adds versioned, type-tagged serialization suitable for polymorphic checkpoint payloads (executor state unions, etc.) — `AgentResponseUpdate`, documented in this guide's "Persisting and replaying updates" section above, is a `SerializationMixin` dataclass.

### Middleware & Function Invocation

#### `AgentMiddleware` + `AgentContext` + `AgentMiddlewareLayer`
**Module:** `agent_framework._middleware`

```python
@agent_middleware
async def my_middleware(ctx: AgentContext, call_next):
    ...                 # before: inspect/mutate ctx.messages, ctx.options, ctx.session
    await call_next()   # zero-argument continuation
    ...                 # after: ctx.result holds the AgentResponse (or ResponseStream)
```
`AgentContext` fields: `agent, messages: list[Message], session, tools, options, stream, compaction_strategy, tokenizer, metadata: dict (cross-middleware side-channel), result, kwargs, client_kwargs, function_invocation_kwargs, stream_transform_hooks, stream_result_hooks, stream_cleanup_hooks`. Raise `MiddlewareTermination()` to short-circuit — **set `ctx.result` first**; the pipeline suppresses the exception and returns `ctx.result` (the exception's own `result=` kwarg, if any, is not read by the agent pipeline). `AgentMiddlewareLayer(*, middleware=None)` is the MRO mixin `Agent` inherits: it categorizes middleware into agent/chat/function buckets, merges base + per-call middleware, wraps streaming in `ResponseStream.from_awaitable()`, and bypasses the pipeline entirely (zero overhead) when there are no middlewares. `ChatMiddlewareLayer(*, middleware=None)` is the analogous mixin on `BaseChatClient` for `ChatMiddleware`/`ChatContext`. `MiddlewareType(str, Enum)`: `AGENT, FUNCTION, CHAT` — the discriminator used internally to route a given middleware into the right layer/bucket.

#### `FunctionInvocationContext`
**Module:** `agent_framework._middleware`

The per-tool-call middleware context (parallel to `AgentContext` one layer down, at the individual function-invocation level):

```python
FunctionInvocationContext(function: FunctionTool, arguments: BaseModel | Mapping[str, Any],
                           session: AgentSession | None = None, metadata: Mapping[str, Any] | None = None,
                           result: Any = None, kwargs: Mapping[str, Any] | None = None,
                           tools: list[ToolTypes] | None = None)
```
`.add_tools(...)` / `.remove_tools(...)` let function middleware add or remove tools available to *later* calls within the same run. As with `AgentContext`/`ChatContext`, the continuation (`call_next`) is passed as a separate argument to the decorated middleware callable — it is not a method on the context object itself.

#### `ChatContext` + `BaseChatClient` + `ResponseStream` + `BaseEmbeddingClient`
**Module:** `agent_framework._sessions` / `agent_framework._clients`

`ChatContext` mirrors `AgentContext` one layer down (per chat-client call rather than per agent run): same `stream_transform_hooks`/`stream_result_hooks`/`stream_cleanup_hooks` triad, plus `client`, `metadata`.

`ResponseStream` is the async-iterable wrapper for `stream=True` calls:
```python
ResponseStream(stream: AsyncIterable[UpdateT] | Awaitable[AsyncIterable[UpdateT]], *,
                finalizer: Callable[[Sequence[UpdateT]], FinalT | Awaitable[FinalT]] | None = None,
                transform_hooks=None, cleanup_hooks=None, result_hooks=None)
```
`await stream.get_final_response()` drains + runs the finalizer; `.with_result_hook(fn)` / `.with_transform_hook(fn)` / `.with_cleanup_hook(fn)` attach hooks after construction.

`BaseChatClient` is the ABC for custom LLM providers (see this guide's "Custom Chat Clients" section above for a worked example) — implement `_inner_get_response(*, messages, stream, options, **kwargs)` as a **plain `def`** returning either an `Awaitable[ChatResponse]` (nest your `async def` logic inside and return the coroutine) when `stream=False`, or `self._build_response_stream(async_gen())` when `stream=True`. `BaseEmbeddingClient(*, additional_properties=None)` is the embedding-provider ABC — implement `async def get_embeddings(values, *, options=None) -> GeneratedEmbeddings`. `GeneratedEmbeddings` extends `list[Embedding[T]]` (each `Embedding.vector: list[float]`), carrying `.usage: UsageDetails | None`. `EmbeddingGenerationOptions` (TypedDict): `model`, `dimensions`. Override `OTEL_PROVIDER_NAME` (ClassVar, default `"unknown"`) for correct telemetry attribution.

#### `ToolApprovalMiddleware` + `ToolApprovalRule` + `ToolApprovalState`
**Module:** `agent_framework._harness._tool_approval`

```python
ToolApprovalMiddleware(*, source_id: str = "tool_approval",
                        auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None)
ToolApprovalRule(tool_name: str, arguments: Mapping[str, str] | None = None, *, server_label: str | None = None)
ToolApprovalState(*, rules=None, queued_approval_requests=None, collected_approval_responses=None)
```
`arguments=None` matches any call to the tool; `{}` matches only no-argument calls; `server_label` scopes the rule to one hosted-tool server. `auto_approval_rules` are `ToolApprovalRuleCallback = Callable[[Content], bool | Awaitable[bool]]` callbacks inspecting the `function_call` content — **security note**: a callback approved for one tool auto-approves *any* local tool sharing that name, so avoid name collisions across unrelated tools registered on the same middleware. `create_always_approve_tool_response(request, *, reason=None)` and `create_always_approve_tool_with_arguments_response(request, *, reason=None)` build standing `ToolApprovalRule`s from a pending `function_approval_request` content item — feed the resulting response messages back via `agent.run([Message(role="user", contents=[...])], session=session)`. Requires `AgentSession`; raises `RuntimeError` without one.

#### `AgentFrameworkException` hierarchy
**Module:** `agent_framework.exceptions`

```
Exception
└── AgentFrameworkException(message, inner_exception=None, log_level=logging.DEBUG)
    ├── AgentException ── AgentContentFilterException, AgentInvalidAuthException,
    │                      AgentInvalidRequestException, AgentInvalidResponseException
    ├── ChatClientException ── ChatClient{ContentFilter,InvalidAuth,InvalidRequest,InvalidResponse}Exception
    ├── IntegrationException ── Integration{ContentFilter,InitializationError,InvalidAuth,
    │                            InvalidRequest,InvalidResponse}Exception
    ├── MiddlewareException
    ├── SettingNotFoundError
    ├── ToolException ── ToolExecutionException, UserInputRequiredException
    ├── WorkflowException ── WorkflowRunnerException ── WorkflowCheckpointException, WorkflowConvergenceException
    └── ContentError
```
Pass `log_level=None` to suppress the automatic debug log; `inner_exception` chains via `__cause__`. The root hierarchy shape (`AgentFrameworkException` at the root, with `AgentException`, `ChatClientException`, `IntegrationException`, `ContentError`, `ToolException`, `WorkflowException`, and `SettingNotFoundError` as the main first-level subclasses) is unchanged from older docs — the nested leaf exceptions shown above are the additional detail confirmed live against 1.14.0.

### Evaluation

#### `Evaluator` (Protocol) + `LocalEvaluator`
**Module:** `agent_framework._evaluation` (`@experimental(ExperimentalFeature.EVALS)`)

```python
class Evaluator(Protocol):
    name: str
    async def evaluate(self, items: Sequence[EvalItem], *, eval_name: str) -> EvalResults: ...

LocalEvaluator(*checks: EvalCheck)   # EvalCheck = Callable[[EvalItem], CheckResult | Awaitable[CheckResult]]
```
Any object with a `name: str` and `async evaluate(...)` satisfies `Evaluator` — no inheritance required. `LocalEvaluator` runs every check concurrently (`asyncio.gather`); an item passes only if **all** checks pass. Built-in check factories: `keyword_check(*keywords, case_sensitive=False)`, `tool_called_check(*tool_names, mode: Literal["all","any"]="all")`, plus the standalone `tool_calls_present(item)` / `tool_call_args_match(item)` (read `item.expected_tool_calls`), and the `@evaluator` decorator to wrap any plain function (parameter names dispatch on `EvalItem` fields: `query, response, expected_output, expected_tool_calls, conversation, tools, context`).

#### `EvalItem` + `ConversationSplit`

```python
EvalItem(conversation: list[Message], tools: list[FunctionTool] | None = None, context: str | None = None,
         expected_output: str | None = None, expected_tool_calls: list[ExpectedToolCall] | None = None,
         split_strategy: ConversationSplitter | None = None)
```
`.query` / `.response` are computed properties derived by splitting `conversation` (default `ConversationSplit.LAST_TURN`). `.split_messages(split=None) -> (query_msgs, response_msgs)`. `EvalItem.per_turn_items(conversation, *, tools=None, context=None)` splits a multi-turn conversation into one cumulative `EvalItem` per user turn.

> **Correction:** `ConversationSplit` (`str, Enum`) has exactly **two** members in the installed version — `LAST_TURN` and `FULL`. A `FIRST_TURN` member documented in an older per-volume page does not exist on 1.14.0. Also note that `EvalItem` takes a `conversation: list[Message]` rather than the flat `input=`/`output=` keyword shape shown in an older per-volume's usage example.

#### `EvalResults` + `EvalItemResult` + `EvalScoreResult` + `RubricScore` + `ExpectedToolCall`

```python
EvalResults(*, provider: str, eval_id="", run_id="", status="completed", result_counts=None,
            report_url=None, error=None, per_evaluator=None, items=None, sub_results=None)
```
Computed: `.passed`, `.failed`, `.total`, `.all_passed`. CI assertion helpers (all recurse into `sub_results`): `raise_for_status(msg=None)`, `assert_score_at_least(threshold, evaluator=None)`, `assert_dimension_score_at_least(dimension_id, threshold)`, `assert_no_failed_items()`. `EvalItemResult(item_id, status: Literal["pass","fail","error"], scores: list[EvalScoreResult], error_code=None, error_message=None, response_id=None, input_text=None, output_text=None, token_usage=None, metadata=None)` — booleans `.is_passed/.is_failed/.is_error`. `EvalScoreResult(name, score: float, passed=None, sample=None, dimensions: list[RubricScore] | None = None)`. `RubricScore` (frozen dataclass): `id, score: int | None, applicable: bool, weight: int, reason: str` (`score=None` when non-applicable for an item). `ExpectedToolCall(name: str, arguments: dict | None = None)`. `EvalNotPassedError(Exception)` is raised by `raise_for_status()` on any failure.

#### `evaluate_agent` + `evaluate_workflow` + `AgentEvalConverter`

```python
evaluate_agent(*, agent=None, queries=None, expected_output=None, expected_tool_calls=None, responses=None,
               evaluators, eval_name=None, context=None, conversation_split=None, num_repetitions=1) -> list[EvalResults]
evaluate_workflow(*, workflow, workflow_result=None, queries=None, expected_output=None, evaluators,
                  eval_name=None, include_overall=True, include_per_agent=True, conversation_split=None,
                  num_repetitions=1) -> list[EvalResults]
```
`evaluate_agent(responses=...)` lets you score pre-existing responses without a second model call. `evaluate_workflow` supports **post-hoc** mode (pass a prior `workflow_result`) or **run+evaluate** mode (pass `queries`, optionally `num_repetitions>1` for stability runs); `include_per_agent=True` populates `EvalResults.sub_results` with one nested `EvalResults` per agent/executor.

`AgentEvalConverter` (all `@staticmethod`) bridges internal `Message`/`Content`/`FunctionTool` types to the Foundry evaluation-service schema:

> **Correction:** the actual method set is `convert_message(message) -> list[dict]`, `convert_messages(messages) -> list[dict]`, `extract_tools(...)`, `to_eval_item(...)`. A `convert_tool` / `convert_tools` pair documented in an older per-volume page does **not** exist on the installed version — use `extract_tools` for tool-schema conversion instead.

`function_result` content items each produce a **separate** output dict (one per tool result); unparseable `function_call` arguments are sanitized to `{"_raw_arguments": "[unparseable]"}` before being sent to an external evaluator.

For teams using Azure AI Foundry, evaluation can also run against Foundry's hosted evaluator catalog instead of `LocalEvaluator`, sharing the same `EvalItem`/`EvalCheck` data shapes but executing server-side and integrating with Foundry's evaluation-run tracking UI.

#### GAIA benchmark harness
**Package:** `agent-framework-lab` (`agent_framework.lab.gaia`), `pip install "agent-framework-lab[gaia]"`

```python
Task(task_id: str, question: str, answer: str | None = None, level: int | None = None,
     file_name: str | None = None, metadata: dict | None = None)
Prediction(prediction: str, messages: list[Any] = [], metadata: dict = {})
Evaluation(is_correct: bool, score: float, details: dict = {})
TaskResult(task_id, task: Task, prediction: Prediction, evaluation: Evaluation,
           runtime_seconds: float | None = None, error: str | None = None)
class Evaluator(Protocol):  # distinct from the core evaluation Evaluator above, same shape idea
    async def __call__(self, task: Task, prediction: Prediction) -> Evaluation: ...
GAIA(evaluator: Evaluator | None = None, data_dir: str | None = None, hf_token: str | None = None,
     telemetry_config: GAIATelemetryConfig | None = None)
GAIATelemetryConfig(enable_tracing=False, otlp_endpoint=None, trace_to_file=False, file_path=None)
```
`await gaia.run(task_runner, level=1, max_n=None, parallel=1, timeout=None, out=None) -> list[TaskResult]`. `TaskRunner` is a bare Protocol (`async def __call__(task) -> Prediction`) — wrap an `Agent` in a thin adapter rather than passing it directly (`Agent` doesn't satisfy the protocol itself). Default evaluator uses the official GAIA exact-match scorer; supply a custom `evaluator=` for fuzzy/LLM-judged scoring.

### Settings & Configuration

#### `SecretString` + `load_settings`
**Module:** `agent_framework._settings` (top-level `agent_framework`)

```python
class SecretString(str):
    def __repr__(self) -> str: return "SecretString('**********')"
    def get_secret_value(self) -> str: return str(self)   # back-compat shim

load_settings(settings_type: type[SettingsT], *, env_prefix: str = "", env_file_path: str | None = None,
              env_file_encoding: str | None = None,
              required_fields: Sequence[str | tuple[str, ...]] | None = None, **overrides: Any) -> SettingsT
```
`SecretString` behaves as a normal `str` everywhere except `repr()` (masked) — safe to `logging.info("%r", key)`. `load_settings` resolution order, highest to lowest: explicit `**overrides` (ignoring `None`) → `.env` file (only if `env_file_path` given) → environment variables (`<env_prefix><FIELD_NAME>`) → TypedDict class defaults. `required_fields` entries: a bare string must resolve non-`None`; a `tuple[str, ...]` means exactly one member of the group must resolve non-`None` (mutually-exclusive constraint, e.g. `("api_key", "azure_api_key")`).

By contrast, individual provider-specific chat clients (OpenAI, Anthropic, Bedrock, Foundry, Ollama, GitHub Copilot, etc.) resolve their own credentials/config with a similar but distinct precedence: explicit constructor kwargs override process environment variables, which override values loaded from a `.env` file if `python-dotenv` support is enabled — that's the client-constructor-level resolution order, separate from the general-purpose `load_settings()` helper above.

#### Observability — `ObservabilitySettings` + telemetry layers + feature staging
**Module:** `agent_framework.observability` / `agent_framework._feature_stage`

Built-in OTel instrumentation emits spans for agent runs, chat-completion calls, tool invocations, and workflow supersteps, plus metrics (token counts, latency, tool-call counts); it's configured via standard OTel SDK setup (exporter/processor wiring) rather than framework-specific config — the framework only emits to whatever OTel pipeline is already configured in-process.

`OBSERVABILITY_SETTINGS` is a module-level `ObservabilitySettings` singleton read by every telemetry layer. `enable_instrumentation(*, enable_sensitive_data=None, force=False)` / `disable_instrumentation()` / `enable_sensitive_telemetry(force=False)`: instrumentation is on by default; `disable_instrumentation()` sets a **sticky** `_user_disabled` flag that silently no-ops later `enable_instrumentation()` calls (and direct `.enable_instrumentation = True` assignment) unless `force=True` clears it explicitly. Sensitive telemetry (message/tool-argument contents in spans) is opt-in even when instrumentation is enabled. `ChatTelemetryLayer` / `EmbeddingTelemetryLayer` are the MRO mix-ins provider clients apply for GenAI-semantic-convention spans; `OtelAttr(str, Enum)` enumerates every span-attribute key (`gen_ai.operation.name`, `gen_ai.usage.input_tokens`, `gen_ai.agent.name`, ...). `create_resource(service_name=None, service_version=None, **attributes)` builds an OTel `Resource` (env var `OTEL_RESOURCE_ATTRIBUTES` entries win over both explicit kwargs and `OTEL_SERVICE_NAME`/`OTEL_SERVICE_VERSION`). `create_metric_views()` returns 3 `View`s passing `agent_framework*`/`gen_ai*` metrics and dropping everything else — it does **not** itself configure the `TOKEN_USAGE_BUCKET_BOUNDARIES` (14 values, ×4 each) / `OPERATION_DURATION_BUCKET_BOUNDARIES` (14 values, ×2 each) constants; wire those into your own histogram view if needed. `get_tracer()`/`get_meter()` are thin pass-throughs to the OTel SDK (no `OBSERVABILITY_SETTINGS` gate themselves — `create_mcp_client_span` is what checks it and substitutes a `NoOpTracer` when disabled). `create_mcp_client_span(method, target=None, attributes=None)` / `set_mcp_span_error(span, error_type, description=None)` instrument MCP client calls per the OTel MCP semantic conventions. `EdgeGroupDeliveryStatus` enumerates why a workflow edge dispatch was delivered/dropped (`DELIVERED, DROPPED_TYPE_MISMATCH, DROPPED_TARGET_MISMATCH, DROPPED_CONDITION_FALSE, EXCEPTION, BUFFERED`).

`ExperimentalFeature(str, Enum)` lists the current experimental-tier feature IDs (`FIDES`, `HARNESS`, `EVALS`, `FUNCTIONAL_WORKFLOWS`, `DECLARATIVE_AGENTS`, `MCP_LONG_RUNNING_TASKS`, `MCP_SKILLS`, `FOUNDRY_TOOLS`, ...) — membership changes across releases, so guard with `getattr(obj, "__feature_id__", None)` rather than assuming stability. `ReleaseCandidateFeature` is the RC tier (empty as of 1.14.0 — nothing has reached RC without also graduating to stable). `FeatureStageWarning(FutureWarning)` / `ExperimentalWarning(FeatureStageWarning)` is the warning hierarchy — `warnings.filterwarnings("ignore", category=ExperimentalWarning)` silences experimental noise without also silencing unrelated `FutureWarning`s from other libraries (the reverse — filtering `FutureWarning` — does catch both).

#### `DevServer` (local debug host)
**Package:** `agent-framework-devui` (`agent_framework.devui`)

```python
DevServer(entities_dir: str | None = None, port: int = 8080, host: str = "127.0.0.1",
          cors_origins: list[str] | None = None, ui_enabled: bool = True, mode: str = "developer",
          auth_enabled: bool = True, auth_token: str | None = None)
```
Local-development-only OpenAI-compatible API server; **not for production** (use Azure Container Apps / Functions / the `ResponsesHostServer`/`InvocationsHostServer` pair from `agent-framework-foundry-hosting` instead). Auth cannot be disabled on non-loopback hosts (raises `ValueError`); CORS default is an empty allowlist (no implicit wildcard-on-localhost); Host-header enforcement guards against DNS rebinding on loopback binds. `DiscoveryResponse{entities: list[EntityInfo]}` backs `GET /discovery`; `EntityInfo` is an 18-field Pydantic model (`id, type, name, framework, tools, source, required_env_vars, deployment_supported, instructions, model, ..., workflow_dump, input_schema, start_executor_id`, the last few populated only for detailed workflow views). `AgentFrameworkRequest` follows the OpenAI Responses-API shape, using its `model` field to select which registered agent/workflow to route to. `ResponsesHostServer(agent, *, prefix="", options=None, store=None)` / `InvocationsHostServer(agent, *, openapi_spec=None)` are the production Foundry hosting servers (`agent-framework-foundry-hosting`) — `ResponsesHostServer` implements the full OpenAI Responses contract with managed checkpointing and HITL tool approval; `InvocationsHostServer` implements a simpler `{"message", "stream"}` → `{"response", "session_id"}` JSON contract with in-process session storage and no checkpointing.

### A2A Protocol & Declarative Graphs

#### `A2AAgent` + `A2AAgentSession` + `A2AContinuationToken`
**Package:** `agent-framework-a2a` (`agent_framework.a2a`)

`A2AAgent` bridges to a remote Agent-to-Agent (A2A) protocol server, letting a `ChatAgent`/`Workflow`-backed agent interoperate with agents built on other A2A-compliant frameworks over the wire, using A2A's task/message envelope rather than the framework's native in-process `Agent` protocol. Conversation continuity across HTTP round-trips uses:

```python
A2AAgentSession(*, context_id: str | None = None, task_id: str | None = None, task_state: TaskState | None = None)
```
`context_id` maps to A2A's `contextId`; stored via `service_session_id`. When `task_state == "input-required"`, the next `agent.run()` sends a **task continuation**; otherwise a new task starts. `A2AContinuationToken(ContinuationToken)` (fields `task_id, context_id`) surfaces on a long-running task's `AgentResponseUpdate.continuation_token` — resume with `agent.run(None, session=session, continuation_token=token)` in a loop until it's `None`.

> **Reconciled naming:** an older per-volume page named this integration's classes `A2AServer`/`A2AClientAgent`. Neither name exists in the installed `agent_framework.a2a` module (confirmed via `dir(agent_framework.a2a)`) — the real, importable names are `A2AAgent` / `A2AAgentSession` / `A2AExecutor` / `A2AServiceSessionId`, as shown above.

#### Declarative agents/workflows (YAML-driven)
**Package:** `agent-framework-declarative` (`agent_framework.declarative`) — a separate package that lets a workflow (or agent) graph be authored as YAML/JSON rather than Python code, with expressions evaluated via embedded PowerFx rather than arbitrary Python for safety/portability.

```python
AgentFactory(*, client=None, bindings=None, connections=None, client_kwargs=None, additional_mappings=None,
             default_provider: str = "Foundry", safe_mode: bool = True, env_file_path=None, env_file_encoding=None)
```
`.create_agent_from_yaml_path(path)` / `.create_agent_from_yaml(yaml_str) -> Agent`. Resolves `model.provider` via a built-in table (`AzureOpenAI[.Responses|.Chat]`, `OpenAI[.Responses|.Chat]`, `Foundry[.Chat]` (default), `Anthropic.Chat`) extendable with `additional_mappings: Mapping[str, ProviderTypeMapping]` (`ProviderTypeMapping` TypedDict: `package, name, model_field, endpoint_field, api_key_field`). **`Ollama` and `Bedrock` are not built into the provider table** — register them via `additional_mappings`. `safe_mode=True` (default) makes `=Env.*` PowerFx expressions in the YAML always resolve empty, so untrusted YAML can't exfiltrate environment variables — it does **not** validate file paths, so still guard `create_agent_from_yaml_path` against path traversal yourself. `ProviderLookupError` (unknown provider/apiType) and the broader `DeclarativeLoaderError` are the two failure modes to catch. `Binding(name=None, input=None)` links a `FunctionTool` argument to a callable registered in `AgentFactory(bindings={...})` by name — **`Binding.input` is not read at runtime**, despite the field existing on the model. The declarative tool models `FunctionTool` / `OpenApiTool` / `WebSearchTool` / `FileSearchTool` / `CodeInterpreterTool` are YAML-`kind`-dispatched subclasses of a common `Tool` base, distinct from the harness `FunctionTool` in the Tools & MCP section above — note that `AgentFactory` has no `OpenApiTool` branch (`kind: openapi` in an agent YAML raises `ValueError`) even though the data model exists.

```python
WorkflowFactory(*, agent_factory=None, agents=None, bindings=None, env_file=None, checkpoint_storage=None,
                 max_iterations=None, http_request_handler=None, mcp_tool_handler=None, configuration=None,
                 restrict_env_to_configuration: bool = True)
```
`.create_workflow_from_yaml_path/_yaml/_definition(...) -> Workflow`. `max_iterations` resolution: factory arg > YAML `maxTurns:` > core default (100). Raises `DeclarativeWorkflowError` at **build time** (not run time) if the YAML contains `HttpRequestAction` without `http_request_handler`, or `InvokeMcpTool` without `mcp_tool_handler`. `DeclarativeWorkflowBuilder` is the lower-level graph assembler `WorkflowFactory` delegates to, for callers with a pre-parsed `dict` instead of raw YAML text.

**Declarative action executors** (one node type per YAML action `kind`) — spans two internal modules, `agent_framework_declarative._workflows._executors_control_flow` and `._executors_basic`, plus the HTTP/MCP/tool-specific executor modules:

- `QuestionExecutor` / `RequestExternalInputExecutor` + `ExternalInputRequest{request_id, message, request_type, metadata}` / `ExternalInputResponse{user_input, value}` — HITL; resume via `workflow.run(responses={request_event.request_id: ExternalInputResponse(...)})` (there is no `workflow.respond()`).
- `HttpRequestActionExecutor` — evaluates method/url/headers/body from state (PowerFx-capable), raises `DeclarativeActionError` on non-2xx or transport failure; `asyncio.CancelledError` propagates unwrapped.
- `InvokeMcpToolActionExecutor` + `MCPToolApprovalRequest` — unlike HTTP, **non-2xx/error responses are written into `output.result` as `"Error: ..."` without raising**, matching the .NET `AssignErrorAsync` contract; check the result string in downstream conditions rather than expecting an exception.
- `BaseToolExecutor` / `InvokeFunctionToolExecutor` + `ToolApprovalRequest{request_id, function_name, arguments}` / `ToolApprovalResponse{approved, reason}` / `ToolInvocationResult{success, result, error, messages, rejected, rejection_reason}` — the declarative `InvokeFunctionTool` action's own approval contract, parallel to (but a separate namespace from) the harness `ToolApprovalRule`/`ToolApprovalState` covered under Middleware.
- **Control flow** (`_executors_control_flow`): `ConditionGroupEvaluatorExecutor` / `IfConditionEvaluatorExecutor` (conditional branching), `ForeachInitExecutor` / `ForeachNextExecutor` / `BreakLoopExecutor` / `ContinueLoopExecutor` (loop constructs implementing break/continue semantics), `JoinExecutor`, `EndWorkflowExecutor`, `EndConversationExecutor`, `CancelDialogExecutor`, `CancelAllDialogsExecutor` (graph merge/termination nodes).
- **Basic variable manipulation** (`_executors_basic`): `SetValueExecutor` / `SetVariableExecutor` / `SetMultipleVariablesExecutor` / `SetTextVariableExecutor`, `ClearAllVariablesExecutor`, `ResetVariableExecutor`, `EditTableExecutor` / `EditTableV2Executor`, `ParseValueExecutor`, `CreateConversationExecutor`, `SendActivityExecutor`.
- `ActionComplete` / `ActionTrigger` / `DeclarativeStateData` / `ConversationData` are the typed inter-executor message contracts carrying result data and the 8-namespace declarative state.

Declarative graphs can also include action nodes that call out to an HTTP endpoint or an MCP tool using the same `MCPStdioTool`/`MCPStreamableHTTPTool` wrappers documented in **Tools & MCP** under the hood — the declarative layer is a thin authoring surface over the same runtime primitives, not a parallel implementation.

```python
factory = AgentFactory(default_provider="AzureOpenAI", safe_mode=True)
agent = factory.create_agent_from_yaml_path("agents/assistant.yaml")

wf_factory = WorkflowFactory(agents={"WriterAgent": agent}, max_iterations=200)
workflow = wf_factory.create_workflow_from_yaml_path("workflow.yaml")
result = await workflow.run({"topic": "Quantum computing"})
```

---

## Appendix: azure-ai-agents Add-on (separate package)

> `azure-ai-agents` is an **optional integration add-on** for the Azure AI Agents service — a distinct package from `agent-framework`, not a replacement for it (see [the migration notice page](./microsoft_agent_framework_python_sdk_migration_notice/) for when to reach for it alongside the framework). Condensed from five source-verified per-volume pages claiming `azure-ai-agents==1.1.0`; **not** independently re-installed/re-verified in this consolidation pass (no network access was used to `pip install` it fresh) — best-effort cross-checked only for internal consistency across the five source pages, which largely repeated the same ~10 core classes with increasing depth (the later volume in particular re-covered the first volume's classes almost entirely, which is folded in here rather than duplicated).

```bash
pip install azure-ai-agents azure-identity
```

### `AgentsClient`
`AgentsClient(endpoint: str, credential: TokenCredential, **kwargs)` — sync from `azure.ai.agents`, async from `azure.ai.agents.aio`. Sub-operation namespaces: `.threads`, `.messages`, `.runs`, `.run_steps`, `.files`, `.vector_stores`, `.vector_store_files`, `.vector_store_file_batches`. Top-level convenience methods: `create_agent(*, model, name=None, description=None, instructions=None, tools=None, tool_resources=None, toolset=None, temperature=None, top_p=None, response_format=None, metadata=None) -> Agent`, `get_agent`, `list_agents`, `update_agent`, `delete_agent`, `enable_auto_function_calls(tools: Set[Callable] | FunctionTool | ToolSet, max_retry=10)` (auto-dispatches tool calls during `create_and_process`/`stream` — no manual polling loop needed), `create_thread_and_process_run(...)` (create + run + poll to completion in one call), `create_thread_and_run(...)` (same but returns immediately without polling).

### `Agent` (model) + `AgentThread`
Read-only snapshots returned by the client, not constructed directly. `Agent`: `id, object="assistant", created_at, name, description, model, instructions, tools, tool_resources, temperature, top_p, response_format, metadata` (≤16 pairs; keys ≤64 chars, values ≤512 chars — Microsoft recommends changing `temperature` OR `top_p`, not both). `AgentThread`: `id, object="thread", created_at, tool_resources (thread-level, overrides agent-level for this thread), metadata`. Deleting a thread permanently removes all its messages.

### `ThreadMessage`
`id, thread_id, role, content: list[MessageContent], status, created_at`, plus convenience properties added in `_patch.py`: `.text_messages`, `.image_contents`, `.file_citation_annotations`, `.file_path_annotations`, `.url_citation_annotations`. `MessageAttachment(*, tools: list[FileSearchToolDefinition | CodeInterpreterToolDefinition], file_id=None, data_source: VectorStoreDataSource | None = None)` attaches a file to one message (exactly one of `file_id`/`data_source` required) — ephemeral/per-message, vs. the agent-level `tool_resources` which persist across every run on the thread. Multimodal input uses `ThreadMessageOptions(role, content: str | list[MessageInputContentBlock])` with `MessageInputTextBlock(text)`, `MessageInputImageFileBlock(image_file: MessageImageFileParam{file_id, detail: ImageDetailLevel})`, `MessageInputImageUrlBlock(image_url: MessageImageUrlParam{url, detail})` — `ImageDetailLevel.LOW` is a fixed 85-token crop, `HIGH` tiles at ~170 tokens/tile (~765 tokens for a 1080×1080 image), `AUTO` (default) picks based on image size.

### `FunctionTool` + `AsyncFunctionTool` + `ToolSet` + `AsyncToolSet`
```python
FunctionTool(functions: Set[Callable])       # sync — .execute() catches exceptions, returns {"error": ...} JSON
AsyncFunctionTool(functions: Set[Callable])  # async — awaits coroutines, calls sync callables directly (auto-detected)
```
Schema is derived from type annotations + Sphinx-style `:param name: description` docstrings — both are required for a correct/complete JSON schema. `.add_functions({...})` registers more callables after construction. `ToolSet`/`AsyncToolSet` aggregate one instance of each tool type (`.add`, `.remove`, `.get_tool`, `.execute_tool_calls`, `.definitions`, `.resources`) and enforce sync/async separation — adding an `AsyncFunctionTool` to a sync `ToolSet` (or vice versa) raises `ValueError` immediately. `AsyncToolSet.execute_tool_calls` runs all of one batch's tool calls **concurrently** via `asyncio.gather` (wall time = `max(latencies)` instead of `sum(latencies)`).

### `CodeInterpreterTool` + `FileSearchTool` + `VectorStore` family
```python
CodeInterpreterTool(file_ids: list[str] | None = None)   # .add_file()/.remove_file()
FileSearchTool(vector_store_ids: list[str] | None = None)  # .add_vector_store()/.remove_vector_store()
```
`VectorStore` (read model, from `client.vector_stores.*`): `id, name, status (expired|in_progress|completed), file_counts: VectorStoreFileCount{in_progress, completed, failed, cancelled, total}, usage_bytes, expires_after, expires_at, last_active_at, metadata`. `VectorStoreExpirationPolicy(anchor="last_active_at", days: int)` — the only anchor; TTL resets on every query, service caps `days` at 365. Chunking: `VectorStoreAutoChunkingStrategyRequest()` (default 800/400) or `VectorStoreStaticChunkingStrategyRequest(static=VectorStoreStaticChunkingStrategyOptions(max_chunk_size_tokens, chunk_overlap_tokens))` — overlap must be `< max_chunk_size_tokens / 2`; set at creation time, immutable afterward. `VectorStoreFileBatch` / `VectorStoreFile` + `VectorStoreFileError{code, message}` track individual per-file embedding status/errors within a batch (a batch can reach `completed` while still having per-file failures — check `file_counts.failed`). `VectorStoreDataSource(asset_identifier, asset_type: URI_ASSET | ID_ASSET)` + `VectorStoreConfigurations(store_name, store_configuration: VectorStoreConfiguration(data_sources=[...]))` reference existing Azure Blob/ADLS-Gen2/AML assets directly without re-uploading through `client.files.upload`.

### `AzureAISearchTool` + `BingGroundingTool` + `ConnectedAgentTool` + `OpenApiTool`
```python
AzureAISearchTool(index_connection_id, index_name, query_type: AzureAISearchQueryType = SIMPLE, filter="",
                   top_k=5, index_asset_id="")
```
`AzureAISearchQueryType`: `SIMPLE, SEMANTIC, VECTOR, VECTOR_SIMPLE_HYBRID, VECTOR_SEMANTIC_HYBRID`. Both `.definitions` **and** `.resources` must be passed to `create_agent` (unlike most tools, which only need `.definitions`). Execution happens server-side — `.execute()` is a no-op.

```python
BingGroundingTool(connection_id, market="", set_lang="", count=5, freshness="")
```
`freshness`: `"Day"|"Week"|"Month"` or an ISO date range. Underlying config type: `BingGroundingSearchConfiguration{connection_id, market, set_lang, count, freshness}` wrapped in `BingGroundingSearchToolParameters{search_configurations}` (service currently caps this list at 1 entry).

```python
ConnectedAgentTool(id: str, name: str, description: str)   # id = a real sub-agent's Agent.id
```
Multi-agent delegation: the orchestrator's model reads `description` to decide when to route to the sub-agent as a black-box tool call. Delete the orchestrator agent before its sub-agents during cleanup.

```python
OpenApiTool(name, description, spec: Any, auth: OpenApiAuthDetails, default_parameters=None)
```
`auth` is one of `OpenApiAnonymousAuthDetails()` (no auth), `OpenApiManagedAuthDetails(security_scheme=OpenApiManagedSecurityScheme(audience=...))` (Azure managed identity), or `OpenApiConnectionAuthDetails(security_scheme=OpenApiConnectionSecurityScheme(connection_id=...))` (Foundry connection — third-party OAuth/API-key). `.add_definition(...)`/`.remove_definition(name)` register additional API specs on the same tool instance.

### `AzureFunctionTool`
```python
AzureFunctionTool(name, description, parameters: dict, input_queue: AzureFunctionStorageQueue,
                   output_queue: AzureFunctionStorageQueue)
AzureFunctionStorageQueue(*, storage_service_endpoint: str, queue_name: str)
```
Calls Azure Functions via Storage Queues rather than in-process code: the service writes a base64-JSON job to `input_queue`; your Function reads it, and writes the result to `output_queue`, which the service polls. `.execute()` is a no-op — entirely server-managed. If the output queue never receives a result before the run's `expires_at`, the run status becomes `"expired"`.

### `ThreadRun` + `RunStep` + usage/cost accounting
`ThreadRun`: `id, thread_id, agent_id, status: RunStatus, required_action, last_error, model, instructions, tools, usage: RunCompletionUsage | None, truncation_strategy: TruncationObject, parallel_tool_calls, temperature, top_p, max_prompt_tokens, max_completion_tokens, created_at/started_at/completed_at/failed_at/expires_at, metadata`. `RunStatus`: `QUEUED, IN_PROGRESS, REQUIRES_ACTION, CANCELLING, CANCELLED, FAILED, COMPLETED, INCOMPLETE, EXPIRED`. `RunCompletionUsage{prompt_tokens, completion_tokens, total_tokens}` is `None` until a terminal state.

`RunStep`: `id, run_id, thread_id, agent_id, type: RunStepType (MESSAGE_CREATION|TOOL_CALLS), status, step_details (discriminated union: RunStepMessageCreationDetails | RunStepToolCallDetails), last_error, usage: RunStepCompletionUsage | None`. `RunStepFileSearchToolCall.file_search.results[i]` gives `{file_id, file_name, score (0.0–1.0), content: list[FileSearchToolCallContent] | None}` — `content` is `None` unless you pass `include=[RunAdditionalFieldList.FILE_SEARCH_CONTENTS]` to `run_steps.list(...)`.

`TruncationObject(type: TruncationStrategy = "auto"|"last_messages", last_messages: int | None = None)` controls how thread history is trimmed to fit the context window per run. `AgentsNamedToolChoice(type: AgentsNamedToolChoiceType, function: FunctionName | None = None)` / `AgentsToolChoiceOptionMode.NONE|AUTO` force (or forbid) tool use on the first model turn of a run via the `tool_choice=` run parameter.

### `ToolOutput` + `SubmitToolOutputsAction` + manual dispatch
```python
ToolOutput(tool_call_id: str | None, output: str | None)   # output must be a string — json.dumps() dicts
```
When `run.status == "requires_action"`, `run.required_action` is a `SubmitToolOutputsAction{submit_tool_outputs: SubmitToolOutputsDetails{tool_calls: list[RequiredFunctionToolCall]}}`; each `RequiredFunctionToolCall{id, function: RequiredFunctionToolCallDetails{name, arguments: str}}` needs `arguments` JSON-parsed before calling your function, and its `id` echoed back verbatim as `ToolOutput.tool_call_id`. Submit **all** pending outputs in one `client.runs.submit_tool_outputs(...)` call — partial submission isn't supported. `enable_auto_function_calls()` handles this whole loop automatically; the manual path exists for custom streaming handlers or fine-grained control.

### Streaming — `AgentEventHandler` / `AsyncAgentEventHandler` + `AgentRunStream`
```python
class AgentEventHandler:  # override any subset
    def on_message_delta(self, delta: MessageDeltaChunk) -> None: ...
    def on_thread_message(self, message: ThreadMessage) -> None: ...
    def on_thread_run(self, run: ThreadRun) -> None: ...
    def on_run_step(self, step: RunStep) -> None: ...
    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None: ...
    def on_error(self, data: str) -> None: ...
    def on_done(self) -> None: ...
    def on_unhandled_event(self, event_type: str, event_data: Any) -> None: ...
    def set_max_retry(self, max_retry: int) -> None: ...   # tool-output resubmission retries, default 10
```
`AsyncAgentEventHandler` mirrors this with `async def` hooks for the async client. `MessageDeltaChunk.text` (a `_patch.py` convenience property) concatenates all text fragments in one delta chunk. `with client.runs.stream(thread_id, agent_id, event_handler=...) as handler:` — `__enter__` returns the **handler**, not the stream; iterate `for event_type, event_data, raw in handler: ...` or call `handler.until_done()`. If `on_thread_run` returns a non-empty `list[ToolOutput]`, the SDK submits it as the tool-output result for that `requires_action` pause — the hook for custom streaming tool dispatch alongside `enable_auto_function_calls`. The full streaming event taxonomy: `ThreadStreamEvent.THREAD_CREATED`; `RunStreamEvent.THREAD_RUN_{CREATED,QUEUED,IN_PROGRESS,REQUIRES_ACTION,COMPLETED,INCOMPLETE,FAILED,CANCELLING,CANCELLED,EXPIRED}`; `MessageStreamEvent.THREAD_MESSAGE_{CREATED,IN_PROGRESS,DELTA,COMPLETED,INCOMPLETE}`; `RunStepStreamEvent.THREAD_RUN_STEP_{CREATED,IN_PROGRESS,DELTA,COMPLETED,FAILED,CANCELLED,EXPIRED}`; plus `AgentStreamEvent.DONE` / `.ERROR` (master union of all the above).

### `FileInfo`
```python
FileInfo(object="file", id, bytes, filename, created_at, purpose: FilePurpose, status=None, status_details=None)
```
`FilePurpose`: `"assistants"` (code interpreter / file search input), `"assistants_output"` (code-interpreter output, read-only), `"vision"` (image input). `bytes` is the raw upload size — embedding storage size is `VectorStoreFile.usage_bytes`, tracked separately. `status`/`FileState` are Azure-OpenAI-endpoint-only fields; often `None` on Foundry endpoints.

```python
import asyncio, os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential

client = AgentsClient(endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"], credential=DefaultAzureCredential())

def get_weather(city: str) -> str:
    """Return current weather for a city.
    :param city: Name of the city.
    """
    return f"{city}: 18°C, partly cloudy"

toolset = ToolSet()
toolset.add(FunctionTool(functions={get_weather}))
client.enable_auto_function_calls(tools=toolset)

agent = client.create_agent(model="gpt-4o", name="weather-agent",
                             instructions="Use get_weather to answer weather questions.", toolset=toolset)
thread = client.threads.create()
client.messages.create(thread_id=thread.id, role=MessageRole.USER, content="Weather in Edinburgh?")
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

---

## Revision history

| Version | Date | Changes |
|---------|------|---------|
| 1.6.0 | May 22, 2026 | Core bumped 1.5.0 → 1.6.0; `Latest verified release`, `**Framework Version:**`, API reference header, and description frontmatter updated. All 23 core symbols verified against installed `agent-framework-core==1.6.0` (`.routine-envs/check-0522-ms-agent`); 242 public symbols confirmed; no deprecations detected. Note: opentelemetry `importlib.metadata` DeprecationWarning at import (Python 3.11 compat issue) is an upstream opentelemetry bug — suppressed with `warnings.filterwarnings('ignore')` for verification purposes. | Claude routine |
| 1.4.0 | May 15, 2026 | Core bumped 1.3.0 → 1.4.0. Added `SecureAgentConfig` (`ExperimentalFeature.FIDES`) section covering information-flow control, label tracking, policy enforcement, and audit logging. Added `InMemorySkillsSource`, `DelegatingSkillsSource`, `FunctionExecutor`/`@executor`, and `WorkflowViz` to the relevant reference pages. Version strings updated throughout; verified against installed `agent-framework==1.4.0`. |
| 1.3.0 | May 9, 2026 | Core bumped 1.2.2 → 1.3.0. `agent-framework-foundry` and `agent-framework-openai` promoted to stable 1.3.0. `MemoryStore` and `SkillResource` now emit `ExperimentalWarning` on import. Version strings updated throughout; `Agent` and `FoundryChatClient` verified against installed `agent-framework==1.3.0` (`.routine-envs/check-0509-py`). |
| 1.2.2 | May 2026 | Guide verified against `agent-framework-core==1.2.2`; skills, functional workflows, and `Agent.as_tool()` added. |

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
| Long-term memory (`MemoryStore`, `MemoryContextProvider`) | See "Long-Term Memory" section above |
| BaseChatClient / BaseEmbeddingClient / ContextProvider extension points | [Advanced Patterns](./microsoft_agent_framework_python_advanced/) |

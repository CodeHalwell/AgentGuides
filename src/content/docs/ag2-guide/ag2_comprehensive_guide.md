---
title: "AG2 Comprehensive Guide"
description: "AG2 1.x reference — Agent, model configs, tools, structured output, human-in-the-loop, event streams, the multi-agent Network, MCP and A2A — plus AG2 Classic and migration from 0.x."
framework: ag2
---

> **Verified against `ag2` %%version=ag2%%** (checked %%versions-checked%%). Every 1.x example on this page was executed offline against the installed package using `ag2.testing.TestConfig` in place of a real model. The Classic section was checked against the `autogen` distribution (AG2 Classic) 0.14.1.

## Overview

AG2 began in November 2024 as the community fork of AutoGen 0.2, led by AutoGen's original creators. Until mid-2026 it kept the AutoGen-style API (`ConversableAgent`, `GroupChat`, `initiate_chat`). **AG2 1.0** replaced that with a new framework, previewed earlier as `autogen.beta`:

- **Async throughout** — `await agent.ask(...)` returns an `AgentReply`.
- **Plain functions as tools** — decorate with `@tool`; dependency injection via `Context`.
- **Typed events** — every model call, tool call and human-input request is an event you can subscribe to on a `MemoryStream`.
- **Network** — a `Hub` plus typed channels (`conversation`, `consulting`, `discussion`, `workflow`) replaces `GroupChat`, swarms and nested chats.
- **Interop** — MCP servers as toolkits, agents served or consumed over A2A, AG-UI and ACP adapters.

## What changed in 1.0

| | AG2 Classic (`ag2` ≤ 0.14, `autogen`) | AG2 1.x (`ag2` ≥ 1.0) |
|---|---|---|
| Import | `import autogen` | `import ag2` |
| Core agent | `ConversableAgent`, `AssistantAgent`, `UserProxyAgent` | `Agent` |
| Model config | `LLMConfig` / `OAI_CONFIG_LIST` dicts | Provider classes: `OpenAIConfig`, `AnthropicConfig`, `GeminiConfig`, … |
| Running | `initiate_chat`, `generate_reply` (sync) | `await agent.ask(...)`, `await reply.ask(...)` |
| Tools | `register_function(f, caller=..., executor=...)` | `@tool` + `Agent(tools=[...])` |
| Multi-agent | `GroupChat` / `GroupChatManager`, swarms, nested chats | `ag2.network.Hub` + channels, `TransitionGraph` |
| Human input | `human_input_mode="ALWAYS"` | `await context.input(...)` inside a tool + `hitl_hook=` |
| Testing | — | `ag2.testing.TestConfig` scripted model |

There is no compatibility shim: `ag2>=1.0` does not ship the `autogen` package at all.

## Installation and model configuration

```bash
pip install "ag2[openai]"            # or ag2[anthropic], ag2[gemini], ag2[bedrock], ag2[ollama], ...
pip install "ag2[openai]==%%version=ag2%%"   # pinned
```

The base install is deliberately small (pydantic, httpx, anyio); each provider and integration is an extra: `openai`, `anthropic`, `gemini`, `bedrock`, `ollama`, `mistral`, `xai`, `zai`, `dashscope`, `typesafe`, plus `mcp`, `a2a`, `ag-ui`, `acp`, `redis`, `tavily`, `ddgs`, `perplexity`, `tracing`, `metrics`.

Each provider has its own config class in `ag2.config`, which reads the provider's standard environment variable when `api_key` is omitted:

| Provider | Config class | Key variable |
|---|---|---|
| OpenAI Chat Completions (and compatible endpoints via `base_url`) | `OpenAIConfig` | `OPENAI_API_KEY` |
| OpenAI Responses API | `OpenAIResponsesConfig` | `OPENAI_API_KEY` |
| Anthropic | `AnthropicConfig` | `ANTHROPIC_API_KEY` |
| Gemini API / Vertex AI | `GeminiConfig` / `VertexAIConfig` | `GEMINI_API_KEY` / ADC |
| Amazon Bedrock | `BedrockConfig` | AWS credentials |
| Ollama, Mistral, xAI, Z.AI, DashScope, TypeSafe | `OllamaConfig`, `MistralConfig`, `XAIConfig`, `ZAIConfig`, `DashScopeConfig`, `TypeSafeConfig` | provider-specific |

```python
from ag2.config import AnthropicConfig, GeminiConfig, OpenAIConfig

openai_cfg = OpenAIConfig(model="gpt-4.1-mini", temperature=0.2, streaming=True)
local_cfg = OpenAIConfig(model="llama3.1", base_url="http://localhost:11434/v1", api_key="unused")
claude_cfg = AnthropicConfig(model="claude-sonnet-4-6")
gemini_cfg = GeminiConfig(model="gemini-2.5-flash", thinking_budget=1024)
```

A config can be set on the agent and overridden per call (`agent.ask(..., config=...)`).

## Your first agent

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig

agent = Agent(
    "assistant",                       # name (positional)
    prompt="You are a helpful assistant.",
    config=OpenAIConfig(model="gpt-4.1-mini"),
)


async def main() -> None:
    reply = await agent.ask("Summarise the difference between lists and tuples.")
    print(reply.body)                  # text of the final message

    follow_up = await reply.ask("Which is hashable?")   # same conversation, history kept
    print(follow_up.body)


asyncio.run(main())
```

- `Agent.ask(...)` starts a **new** conversation; `AgentReply.ask(...)` continues it.
- `prompt=` accepts a string, a callable returning a string (sync or async), or a list of either — useful for prompts built from runtime state.
- `Agent.run(...)` is the observable counterpart of `ask`: it returns an `AgentRun` you can drive (`run.start()`), watch (`run.stream.join()`), feed (`run.enqueue(...)`) and finally `await run.result()`.
- `agent.as_tool(description=...)` turns an agent into a tool another agent can call (registered as `task_<name>`) — the simplest sub-agent pattern.

## Tools

```python
from datetime import datetime

from ag2 import Agent, tool
from ag2.config import OpenAIConfig


@tool
async def get_weekday(date_string: str) -> str:
    """Get the day of the week for a date formatted as YYYY-MM-DD."""
    return datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")


@tool(description="Add two integers and return the sum.")
def add(a: int, b: int) -> int:
    return a + b


agent = Agent(
    "date_agent",
    prompt="Answer date and arithmetic questions using your tools.",
    config=OpenAIConfig(model="gpt-4.1-mini"),
    tools=[get_weekday, add],
)
```

- The docstring (or `description=`) and type hints become the tool schema. Sync and async functions both work.
- AG2 runs the whole loop: the model requests a call, AG2 executes it and feeds the result back until the model produces a final answer.
- A parameter typed `Context` is injected rather than exposed to the model; it gives tools access to `context.input(...)` (human input), variables and dependencies.
- Built-in tools live in `ag2.tools`: web search and fetch (`WebSearchTool`, `WebFetchTool`, `TavilySearchTool`, `DuckDuckSearchTool`), code execution and shells (`CodeExecutionTool`, `SandboxCodeTool`, `ShellTool`), `FilesystemToolkit`, skills (`SkillsToolkit`) and MCP (`MCPToolkit`). Provider-hosted tools such as web search only work with providers that support them.

## Structured output

```python
from pydantic import BaseModel

from ag2 import Agent
from ag2.config import OpenAIConfig


class TicketTriage(BaseModel):
    category: str  # billing, bug, account_access
    urgency: str   # low, medium, high


triage = Agent(
    "triage",
    prompt="You triage customer support messages. Be conservative with urgency.",
    config=OpenAIConfig(model="gpt-4.1-mini"),
    response_schema=TicketTriage,
)


async def classify(text: str) -> TicketTriage:
    reply = await triage.ask(f"Classify this ticket:\n\n{text}")
    return await reply.content()   # parsed and validated TicketTriage
```

`reply.body` is always the raw text; `await reply.content()` parses it against the schema in force for that turn. `response_schema` accepts a plain type (`int`, a dataclass, a Pydantic model), a `ResponseSchema(...)` with an explicit name and description, a function decorated with `@response_schema` for custom parsing, or `PromptedSchema(...)` for models without native structured output.

## Human in the loop

```python
from ag2 import Agent, Context, tool
from ag2.config import OpenAIConfig
from ag2.events import HumanInputRequest, HumanMessage


@tool
async def publish_plan(context: Context, plan: str) -> str:
    """Publish a lesson plan once a human has approved it."""
    answer = await context.input(f"Approve this plan?\n\n{plan}")
    return "Published." if answer.strip().lower().startswith("y") else f"Rejected: {answer}"


def hitl_hook(event: HumanInputRequest) -> HumanMessage:
    # Replace with a web UI, queue or CLI prompt.
    return HumanMessage(content=input(f"{event.content}\n> "))


teacher = Agent(
    "teacher",
    prompt="Draft a short lesson plan, then call publish_plan. Never ask for approval in plain text.",
    config=OpenAIConfig(model="gpt-4.1-mini"),
    tools=[publish_plan],
    hitl_hook=hitl_hook,
)
```

The hook may be sync or async and may return a `HumanMessage` or a plain string. Tools that need approval before running can also be wrapped declaratively (see the upstream "approval required" tools docs).

## Observing a turn with MemoryStream

```python
from ag2 import MemoryStream
from ag2.events import BaseEvent, ToolCallEvent

stream = MemoryStream()


@stream.subscribe()
async def log_everything(event: BaseEvent) -> None:
    print(type(event).__name__)


@stream.where(ToolCallEvent).subscribe()
async def on_tool_call(event: ToolCallEvent) -> None:
    print("tool requested:", event.name)


# reply = await agent.ask("What day was 1995-03-25?", stream=stream)
```

Events are typed (`ModelResponse`, `ModelMessageChunk` when `streaming=True`, `ToolCallEvent`, `ToolResultEvent`, `HumanInputRequest`, compaction events, …), so persistence, UIs and audit logs subscribe to exactly what they need. Longer-lived concerns — metrics, tracing, policies — are better expressed as `observers=` and `middleware=` on the agent.

## Multi-agent: the Network

Multi-agent systems in 1.x run on `ag2.network`: a `Hub` owns the agent registry, a write-ahead log (WAL) and an audit trail; agents talk over **channels**:

| Channel type | Shape | Ends when |
|---|---|---|
| `conversation` | Free-form, two parties | You close it |
| `consulting` | One question, one reply | Automatically, after the reply |
| `discussion` | Round-robin across N agents | You close it |
| `workflow` | Turn order from a declarative `TransitionGraph` | The graph terminates or `max_turns` is reached |

A three-step pipeline (the 1.x equivalent of a sequential `GroupChat`):

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import EV_CHANNEL_CLOSED, EV_PACKET, Hub, TransitionGraph


async def main() -> None:
    config = OpenAIConfig(model="gpt-4.1-mini")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

    researcher = await hub.register(Agent("researcher", prompt="You gather facts.", config=config))
    writer = await hub.register(Agent("writer", prompt="You turn facts into a short draft.", config=config))
    editor = await hub.register(Agent("editor", prompt="You tighten drafts.", config=config))

    graph = TransitionGraph.sequence([researcher.agent_id, writer.agent_id, editor.agent_id])
    channel = await researcher.open(
        type="workflow",
        target=[writer.agent_id, editor.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    await channel.send("Topic: how does HTTPS work?")

    # Wait for the graph to finish, then replay the exchange from the WAL.
    while not any(e.event_type == EV_CHANNEL_CLOSED for e in await hub.read_wal(channel.channel_id)):
        await asyncio.sleep(0.1)
    for envelope in await hub.read_wal(channel.channel_id):
        if envelope.event_type == EV_PACKET:
            print(envelope.event_data["body"])

    await hub.close()


asyncio.run(main())
```

- `TransitionGraph.round_robin(participants=[...], max_turns=N)` and `TransitionGraph.sequence([...])` cover the common cases; custom graphs use `Transition(when=..., then=...)` with conditions such as `FromSpeaker(...)` or `ToolCalled("escalate")` and targets such as `AgentTarget(...)`, `StayTarget()`, `RevertToInitiatorTarget()` and `TerminateTarget(reason=...)`.
- Graphs are plain data (`to_dict()`), so a channel survives `Hub.hydrate()` after a restart. Use `DiskKnowledgeStore`, `SqliteKnowledgeStore` or `RedisKnowledgeStore` instead of `MemoryKnowledgeStore` for durable hubs.
- For a two-agent exchange, `type="conversation"` with `target="reviewer"` is simpler; close it yourself with `await channel.close()`.

## MCP servers

```python
from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.tools import MCPServerConfig, MCPStdioServerConfig, MCPToolkit

agent = Agent(
    "ops",
    prompt="Use the filesystem and ticketing tools to answer questions.",
    config=OpenAIConfig(model="gpt-4.1-mini"),
    tools=[
        # local server launched as a subprocess over stdio
        MCPToolkit(MCPStdioServerConfig(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])),
        # remote streamable-HTTP server
        MCPToolkit(MCPServerConfig(server_url="https://mcp.example.com/mcp", authorization_token="...")),
    ],
)
```

`MCPToolkit` connects from your process and exposes each MCP tool as an ordinary function tool, so it works with any provider; `allowed_tools=` / `blocked_tools=` / `tool_name_prefix=` on the server config narrow what the model sees. `MCPServerTool` instead forwards the server URL to providers that connect to MCP themselves (for example Anthropic). Requires `ag2[mcp]`.

## A2A protocol

Serve an agent over A2A, or call a remote A2A agent as if it were local (requires `ag2[a2a]`):

```python
import uvicorn

from ag2 import Agent, tool
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.config import AnthropicConfig


@tool(description="Add two integers and return the sum as a string.")
async def calc_add(a: int, b: int) -> str:
    return f"{a + b}"


async def serve() -> None:
    agent = Agent("calculator", config=AnthropicConfig(model="claude-sonnet-4-6"), tools=[calc_add])
    card = build_card(agent, url="http://127.0.0.1:8000", description="Adds integers.")
    app = A2AServer(agent).build_jsonrpc(url="http://127.0.0.1:8000", card=card)
    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000)).serve()


# Client side: the remote agent's card URL is the "model config".
remote = Agent("remote", config=A2AConfig(card_url="http://127.0.0.1:8000"))
# reply = await remote.ask("What is 2 + 40?")
```

The card is served at `/.well-known/agent-card.json`. `A2AServer` can also build REST and gRPC transports sharing one task store, and supports push notifications (pass `push_url_validator=` to guard against SSRF).

## Testing

`ag2.testing.TestConfig` replays scripted model outputs — strings for answers, `ToolCallEvent(...)` for tool calls — so agents can be unit-tested without network access:

```python
import pytest

from ag2 import Agent, tool
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig


@tool
def lookup(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}: shipped"


@pytest.mark.asyncio
async def test_lookup_flow() -> None:
    agent = Agent("support", tools=[lookup])
    reply = await agent.ask(
        "Where is order 42?",
        config=TestConfig(ToolCallEvent(name="lookup", arguments='{"order_id": "42"}'), "It has shipped."),
    )
    assert reply.body == "It has shipped."
```

## AG2 Classic and migrating from 0.x

**AG2 Classic** is the `autogen.*` API that `ag2` shipped up to 0.14.0. It now lives in [ag2ai/ag2-classic](https://github.com/ag2ai/ag2-classic), is published on PyPI as `autogen`, and is in **maintenance mode** (security and critical fixes only). Documentation: [classic.docs.ag2.ai](https://classic.docs.ag2.ai).

To keep Classic code running, pin it explicitly:

```bash
pip install "autogen[openai]"   # AG2 Classic distribution
# or keep the last Classic ag2 release
pip install "ag2[openai]<1.0"
```

For reference, idiomatic Classic code looks like this (verified against `autogen` 0.14.1 — note it uses `ConversableAgent` and `register_function`, not a bare `Agent`):

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig, register_function

llm_config = LLMConfig({"api_type": "openai", "model": "gpt-4.1-mini"})

coder = ConversableAgent("coder", system_message="You write Python.", llm_config=llm_config)
reviewer = ConversableAgent("reviewer", system_message="You review Python.", llm_config=llm_config)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)


def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."


register_function(get_weather, caller=coder, executor=user, description="Get the weather.")

groupchat = GroupChat(agents=[user, coder, reviewer], messages=[], max_round=6)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
# user.initiate_chat(manager, message="Write a script that prints today's weather in Paris.")
```

Porting checklist:

1. `ConversableAgent(name, system_message=..., llm_config=...)` → `Agent(name, prompt=..., config=OpenAIConfig(...))`.
2. `LLMConfig` / `OAI_CONFIG_LIST` → one provider config object per model; keys come from environment variables.
3. `register_function(...)` / `register_for_llm` → `@tool` functions passed in `tools=[...]`; executors disappear because AG2 runs tools itself.
4. `initiate_chat` / `generate_reply` → `await agent.ask(...)`; wrap entry points in `asyncio.run`.
5. `human_input_mode` → a tool that calls `context.input(...)` plus `hitl_hook=`.
6. `GroupChat`, swarms and handoffs → a `workflow` channel with a `TransitionGraph` (upstream guide: "Migrating from Group Chat" on docs.ag2.ai).
7. Code executors → `CodeExecutionTool` / `SandboxCodeTool` / `ShellTool` in `ag2.tools`.

Code written against `autogen.beta` in 0.11–0.14 ports almost mechanically: the same names now import from `ag2`.

## Resources

- [AG2 documentation](https://docs.ag2.ai) · [GitHub: ag2ai/ag2](https://github.com/ag2ai/ag2)
- [AG2 Classic documentation](https://classic.docs.ag2.ai) · [GitHub: ag2ai/ag2-classic](https://github.com/ag2ai/ag2-classic)
- [Recipes](./ag2_recipes/) · [Production guide](./ag2_production_guide/) · [Diagrams](./ag2_diagrams/)

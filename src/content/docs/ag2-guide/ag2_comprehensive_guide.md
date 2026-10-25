---
title: "AG2 Comprehensive Guide"
description: "AG2 1.x reference: Agent, model configs, tools, dependency injection, code execution, structured output, multimodal input, human-in-the-loop, events, middleware, knowledge and compaction, the multi-agent Network, MCP, A2A and testing."
framework: "ag2"
language: "python"
sidebar:
  label: "Comprehensive guide"
---

> **Verified against `ag2` %%version=ag2%%** (checked %%versions-checked%%). Every 1.x example on this page was executed offline against the installed package using `ag2.testing.TestConfig` in place of a real model. For the Classic `import autogen` API, see [AG2 Classic](/ag2-guide/classic/).

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

The base install is deliberately small (pydantic, httpx, anyio, PyYAML and FastDepends); each provider and integration is an extra: `openai`, `anthropic`, `gemini`, `bedrock`, `ollama`, `mistral`, `xai`, `zai`, `dashscope`, `typesafe`, `elevenlabs`, plus `mcp`, `mcp-ui`, `a2a`, `a2ui`, `ag-ui`, `acp`, `redis`, `tavily`, `ddgs`, `perplexity`, `tracing`, `metrics`. Importing something whose extra is missing raises an `ImportError` that names the extra to install.

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

openai_cfg = OpenAIConfig(model="%%model=openai-small%%", temperature=0.2, streaming=True)
local_cfg = OpenAIConfig(model="llama3.1", base_url="http://localhost:11434/v1", api_key="unused")
claude_cfg = AnthropicConfig(model="%%model=anthropic%%")
gemini_cfg = GeminiConfig(model="%%model=gemini%%", thinking_budget=1024)
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
    config=OpenAIConfig(model="%%model=openai-small%%"),
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
- `Agent.run(...)` is the observable counterpart of `ask`: it returns an `AgentRun`, an async context manager you drive (`run.start()` or `await run.result()`), watch (`run.stream`), and feed (`run.enqueue(...)`). See [Streaming and observing a turn](#streaming-and-observing-a-turn).
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
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[get_weekday, add],
)
```

- The docstring (or `description=`) and type hints become the tool schema. Sync and async functions both work.
- AG2 runs the whole loop: the model requests a call, AG2 executes it and feeds the result back until the model produces a final answer.
- A parameter typed `Context` is injected rather than exposed to the model; it gives tools access to `context.input(...)` (human input), variables and dependencies.
- Built-in tools live in `ag2.tools`. They include web search and fetch (`WebSearchTool`, `WebFetchTool`, `TavilySearchTool`, `DuckDuckSearchTool`), local code and shell execution (`SandboxCodeTool`, `SandboxShellTool`; see [Code execution](#code-execution)), `FilesystemToolkit`, skills (`SkillsToolkit`) and MCP (`MCPToolkit`).
- Some tools are **provider-hosted**: the model provider runs them on its own side. Examples are `WebSearchTool`, `CodeExecutionTool` (the provider's code sandbox) and `ShellTool` (OpenAI's hosted container via the Responses API; not supported on Anthropic). They only work with providers that support them.

## Dependency injection and variables

Tool parameters can be filled by AG2 instead of the model. Annotate them and they disappear from the tool schema:

- `Annotated[T, Inject()]` takes the value from `dependencies=` (on the `Agent` or per `ask`), looked up by parameter name or by `Inject("name")`. Use it for database clients, HTTP sessions and settings.
- `Annotated[T, Variable()]` takes the value from `variables=`, the per-conversation state (tools can change it through `context.variables`).
- `Annotated[T, Depends(factory)]` calls `factory()` (FastDepends-style) and passes the result.

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated

from ag2 import Agent, Depends, Inject, Variable, tool
from ag2.config import OpenAIConfig


@dataclass
class Database:
    balances: dict[str, float]


def get_currency() -> str:
    return "EUR"


@tool
def account_balance(
    account_id: str,                                  # the only parameter the model sees
    db: Annotated[Database, Inject()],
    user_id: Annotated[str, Variable()],
    currency: Annotated[str, Depends(get_currency)],
) -> str:
    """Return the balance of one of the current user's accounts."""
    return f"{user_id}/{account_id}: {db.balances[account_id]:.2f} {currency}"


agent = Agent(
    "bank",
    prompt="Answer balance questions with your tool.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[account_balance],
    dependencies={"db": Database({"current": 120.5})},
)


async def main() -> None:
    reply = await agent.ask("What is in my current account?", variables={"user_id": "u-42"})
    print(reply.body)


asyncio.run(main())
```

A `Context` parameter gives the same access programmatically (`context.dependencies`, `context.variables`).

## Code execution

Two kinds of code tool exist, and they are not interchangeable:

| Tool | Where code runs | Works with |
|---|---|---|
| `SandboxCodeTool(environment)`, `SandboxShellTool(environment)` | Client-side, in the environment you pass: `LocalEnvironment` (a subprocess on your machine, no isolation), `ag2.extensions.docker.DockerEnvironment` (needs the `docker` package and a daemon), `ag2.extensions.daytona.DaytonaEnvironment` | Every provider |
| `CodeExecutionTool()`, `ShellTool()` | Server-side, in the model provider's sandbox | Only providers with native code execution or shell tools |

```python
from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.tools import LocalEnvironment, SandboxCodeTool, SandboxShellTool

workspace = LocalEnvironment("./sandbox", timeout=30)   # trusted code only; use DockerEnvironment otherwise

analyst = Agent(
    "analyst",
    prompt="Answer by running Python. Inspect files with the shell tool.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[
        SandboxCodeTool(workspace, languages=("python",)),               # exposes run_code(code, language)
        SandboxShellTool(workspace, allowed=["ls", "cat"], readonly=True),  # exposes run_shell_command(command)
    ],
)
```

`SandboxCodeTool` has no default environment. It runs whatever the model writes, so you must choose the backend explicitly. `SandboxShellTool` takes its environment as an optional argument and also accepts `allowed=` / `blocked=` command lists and `readonly=True`. A command outside the allow-list is not run: the model receives a `Command not allowed` result instead. The tool names the model sees are `run_code(code, language)` and `run_shell_command(command)`.

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
    config=OpenAIConfig(model="%%model=openai-small%%"),
    response_schema=TicketTriage,
)


async def classify(text: str) -> TicketTriage:
    reply = await triage.ask(f"Classify this ticket:\n\n{text}")
    return await reply.content()   # parsed and validated TicketTriage
```

`reply.body` is always the raw text; `await reply.content()` parses it against the schema in force for that turn. `response_schema` accepts a plain type (`int`, a dataclass, a Pydantic model), a `ResponseSchema(...)` with an explicit name and description, a function decorated with `@response_schema` for custom parsing, or `PromptedSchema(...)` for models without native structured output.

## Multimodal input

Pass inputs alongside the text in `ask(...)`. `ImageInput`, `DocumentInput`, `AudioInput` and `VideoInput` accept a URL, `path=`, raw `data=` with a `media_type=`, or a provider `file_id=`:

```python
from ag2 import Agent, DocumentInput, ImageInput
from ag2.config import OpenAIConfig

agent = Agent("analyst", prompt="Describe what you are shown.", config=OpenAIConfig(model="%%model=openai-small%%"))


async def describe() -> None:
    reply = await agent.ask("What does this chart show?", ImageInput(path="chart.png"))
    print(reply.body)
    reply = await reply.ask("Does the report agree?", DocumentInput(url="https://example.com/q3-report.pdf"))
    print(reply.body)
```

Support depends on the provider and model.

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
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[publish_plan],
    hitl_hook=hitl_hook,
)
```

The hook may be sync or async and may return a `HumanMessage` or a plain string. Tools that need approval before running can also be wrapped declaratively with the `approval_required` tool middleware from `ag2.middleware`.

## Streaming and observing a turn

Set `streaming=True` on the provider config and the model's text arrives as `ModelMessageChunk` events while the turn runs. A `MemoryStream` you pass to `ask()` lets you subscribe to them. The final `ModelResponse` still carries the complete text, and `reply.body` is unchanged.

```python
import asyncio

from ag2 import Agent, MemoryStream
from ag2.config import OpenAIConfig
from ag2.events import ModelMessageChunk, ModelResponse

agent = Agent("writer", prompt="Write one sentence.", config=OpenAIConfig(model="%%model=openai-small%%", streaming=True))


async def main() -> None:
    # 1. Subscribe to chunks on a stream you own.
    stream = MemoryStream()

    @stream.where(ModelMessageChunk).subscribe()
    async def show(chunk: ModelMessageChunk) -> None:
        print(chunk.content, end="", flush=True)

    reply = await agent.ask("Describe tea.", stream=stream)
    print("\nfinal:", reply.body)

    # 2. Or open the turn as a scope and drive it yourself.
    async with agent.run("Describe coffee.") as run:
        seen: list[str] = []

        @run.stream.where(ModelResponse).subscribe()
        async def on_response(event: ModelResponse) -> None:
            seen.append(type(event).__name__)

        result = await run.result()     # drives the turn; run.start() would drive it in the background
    print(result.body, seen)


asyncio.run(main())
```

`AgentRun` owns the turn's task: leaving the `async with` block cancels a turn that is still running, so await `run.result()` (or `run.start()` and then the result) inside the block. Subscribe before driving, because callbacks fire inline as events are published.

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

## Middleware and observers

**Middleware** wraps the agent turn, each LLM call and each tool execution, and can change what passes through. **Observers** only watch events and raise alerts. Both are passed as lists on the `Agent` (or per `ask`), and the built-ins are in `ag2.middleware` and `ag2.observers`:

```python
import logging

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.middleware import HistoryLimiter, LoggingMiddleware, RetryMiddleware, approval_required
from ag2.observers import LoopDetector, TokenMonitor

agent = Agent(
    "support",
    prompt="You answer support questions.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    middleware=[
        RetryMiddleware(max_retries=2),            # retry failed LLM calls
        HistoryLimiter(max_events=200),            # or TokenLimiter(max_tokens=...)
        LoggingMiddleware(logging.getLogger("support")),
    ],
    observers=[
        TokenMonitor(warn_threshold=50_000, alert_threshold=100_000),
        LoopDetector(repeat_threshold=3),          # identical tool calls in a row
    ],
)
```

Other built-ins: `TelemetryMiddleware` (OpenTelemetry spans; see [production](/ag2-guide/ag2_production_guide/#observability)) and `MetricsMiddleware` (extra `metrics`). `approval_required()` is tool middleware: `@tool(middleware=[approval_required()])` asks a human (through `hitl_hook`) before the tool runs. Write your own turn-level middleware by subclassing `BaseMiddleware`, and your own observer by subclassing `BaseObserver`. `ToolMiddleware` is not a base class: it is the type of a tool-middleware callable, `async def mw(call_next, event, context)`, which you pass to `@tool(middleware=[...])` (the next section shows both). A `Plugin(tools=..., middleware=..., observers=..., prompt=...)` bundles all of these so you can attach them with `plugins=[...]`.

### Writing your own middleware and observers

`BaseMiddleware` has four hooks, each receiving `call_next` so it can run code before and after the wrapped step: `on_turn`, `on_llm_call`, `on_tool_execution` and `on_human_input`. Register it with `Middleware(Class, **options)`, which builds a fresh instance per turn. Observers publish an `ObserverAlert` on the stream, which you can subscribe to like any other event.

```python
import asyncio
import time

from ag2 import Agent, MemoryStream, tool
from ag2.config import OpenAIConfig
from ag2.events import ModelResponse, ObserverAlert
from ag2.middleware import BaseMiddleware, Middleware, approval_required
from ag2.observers import LoopDetector


class Timing(BaseMiddleware):
    """Print how long each LLM call takes."""

    def __init__(self, event, context, *, label: str) -> None:
        super().__init__(event, context)
        self.label = label

    async def on_llm_call(self, call_next, events, context) -> ModelResponse:
        started = time.perf_counter()
        response = await call_next(events, context)
        print(f"[{self.label}] LLM call took {time.perf_counter() - started:.3f}s")
        return response


@tool(middleware=[approval_required()])      # asks the human before every call
def delete_account(user_id: str) -> str:
    """Delete a user account."""
    return f"deleted {user_id}"


def human(event) -> str:                      # hitl_hook: answer "y", "n" or "always"
    return input(event.content)


agent = Agent(
    "ops",
    prompt="Operate customer accounts.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[delete_account],
    middleware=[Middleware(Timing, label="ops")],
    observers=[LoopDetector(repeat_threshold=3)],
    hitl_hook=human,
)


async def main() -> None:
    stream = MemoryStream()

    @stream.where(ObserverAlert).subscribe()
    async def on_alert(alert: ObserverAlert) -> None:
        print("alert:", alert.severity, alert.message)

    reply = await agent.ask("Delete account u1.", stream=stream)
    print(reply.body)


asyncio.run(main())
```

`approval_required()` accepts `message=`, `denied_message=`, `timeout=` and `allow_always=`. A denied call returns `denied_message` to the model as the tool result. Answering `always` skips the question for later calls of the same tool in that conversation. `LoopDetector` emits a warning alert after `repeat_threshold` consecutive identical tool calls; it does not stop the turn, so a policy that should stop it belongs in middleware or the tool.

## Knowledge, memory and compaction

`knowledge=KnowledgeConfig(...)` attaches a `KnowledgeStore` to an agent. By default the agent gets a `knowledge` tool for reading and writing the store, and each conversation's event log is persisted there. A compaction strategy keeps long histories inside the context window:

```python
from ag2 import Agent, KnowledgeConfig
from ag2.compact import CompactTrigger, SummarizeCompact, TailWindowCompact
from ag2.config import OpenAIConfig
from ag2.knowledge import SqliteKnowledgeStore

config = OpenAIConfig(model="%%model=openai-small%%")

agent = Agent(
    "assistant",
    prompt="You are a long-running assistant.",
    config=config,
    knowledge=KnowledgeConfig(
        store=SqliteKnowledgeStore("agent-memory.db"),
        compact=TailWindowCompact(target=40),               # free: keep the last 40 events
        # compact=SummarizeCompact(target=40, config=config),  # one LLM call: summarise the rest
        compact_trigger=CompactTrigger(max_events=80),       # or max_tokens=...
    ),
)
```

- Stores: `MemoryKnowledgeStore` (tests), `SqliteKnowledgeStore`, `DiskKnowledgeStore` (needs `watchdog`), `RedisKnowledgeStore` (extra `redis`).
- `expose_tool=False` hides the store from the model when only your own code should use it. `aggregate=` / `aggregate_trigger=` distil conversations into longer-term memory.
- `assembly=[...]` on the `Agent` takes `AssemblyPolicy` objects that decide what goes into each LLM request (for example working or episodic memory read from the store).

## Sub-tasks and evaluation

- `tasks=TaskConfig(...)` gives an agent `run_subtask(task)` and `run_subtasks(tasks, parallel=True)` tools for delegating work to sub-agents. The sub-agents inherit its tools, narrowed with `include_tools=` / `exclude_tools=`, and cannot delegate further. `TaskConfig(config=...)` picks a cheaper model for them, `extra_tools=` adds tools only they have, and `max_concurrency=` bounds how many run at once. `agent.task(...)` is the lifecycle handle for tracking a unit of work yourself.
- `ag2.eval` runs agents offline against datasets with scorers (`Suite`, `Scorer`), replays model output deterministically with `TestConfig` cassettes, and saves run JSON for run-against-run diffs.

```python
import asyncio

from ag2 import Agent, TaskConfig, tool
from ag2.config import OpenAIConfig


@tool
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


lead = Agent(
    "lead",
    prompt="Split big jobs into sub-tasks with run_subtasks, then combine the results.",
    config=OpenAIConfig(model="%%model=openai-flagship%%"),
    tools=[word_count],
    tasks=TaskConfig(
        config=OpenAIConfig(model="%%model=openai-small%%"),   # cheaper model for the workers
        include_tools=["word_count"],                          # workers may only count words
        max_concurrency=2,
    ),
)

print(asyncio.run(lead.ask("How many words are in each of these: 'a b c' and 'd e'?")).body)
```

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
    config = OpenAIConfig(model="%%model=openai-small%%")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

    requester = await hub.register(Agent("requester", prompt="You commission short pieces.", config=config))
    researcher = await hub.register(Agent("researcher", prompt="You gather facts.", config=config))
    writer = await hub.register(Agent("writer", prompt="You turn facts into a short draft.", config=config))
    editor = await hub.register(Agent("editor", prompt="You tighten drafts.", config=config))

    graph = TransitionGraph.sequence([requester.agent_id, researcher.agent_id, writer.agent_id, editor.agent_id])
    channel = await requester.open(
        type="workflow",
        target=[researcher.agent_id, writer.agent_id, editor.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    await channel.send("Topic: how does HTTPS work?")   # this message is the requester's turn

    # Wait for the graph to finish, then replay the exchange from the WAL.
    while not any(e.event_type == EV_CHANNEL_CLOSED for e in await hub.read_wal(channel.channel_id)):
        await asyncio.sleep(0.1)
    for envelope in await hub.read_wal(channel.channel_id):
        if envelope.event_type == EV_PACKET:
            print(envelope.event_data["body"])

    await hub.close()


asyncio.run(main())
```

- The initiator's `channel.send(...)` is the initiator's own turn. If the initiator is the first step of the graph, its model never runs, so give the first step to an agent whose job is just to hand over the brief (`requester` above), and put the agents you want to run after it.
- `TransitionGraph.round_robin(participants=[...], max_turns=N)` and `TransitionGraph.sequence([...])` cover the common cases; custom graphs use `Transition(when=..., then=...)` with conditions such as `FromSpeaker(...)` or `ToolCalled("escalate")` and targets such as `AgentTarget(...)`, `StayTarget()`, `RevertToInitiatorTarget()` and `TerminateTarget(reason=...)`.
- Graphs are plain data (`to_dict()`), so a channel survives `Hub.hydrate()` after a restart. Use `DiskKnowledgeStore`, `SqliteKnowledgeStore` or `RedisKnowledgeStore` instead of `MemoryKnowledgeStore` for durable hubs.
- For a two-agent exchange, `type="conversation"` with `target="reviewer"` is simpler; close it yourself with `await channel.close()`.

### A consulting channel: one question, one answer

`consulting` is the lightest channel. The initiator sends one message, the respondent replies once, and the hub closes the channel with the reason `consulting_complete`. Targets can be given by agent name or `agent_id`.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import EV_CHANNEL_CLOSED, EV_TEXT, Hub


async def main() -> None:
    config = OpenAIConfig(model="%%model=openai-small%%")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    asker = await hub.register(Agent("asker", prompt="You ask questions.", config=config))
    await hub.register(Agent("tax_expert", prompt="Answer tax questions in one sentence.", config=config))

    channel = await asker.open(type="consulting", target="tax_expert")
    await channel.send("What is the standard UK VAT rate?")

    while not any(e.event_type == EV_CHANNEL_CLOSED for e in await hub.read_wal(channel.channel_id)):
        await asyncio.sleep(0.05)
    answer = [e.event_data["text"] for e in await hub.read_wal(channel.channel_id) if e.event_type == EV_TEXT][-1]
    print(answer)
    await hub.close()


asyncio.run(main())
```

### Routing on a tool call

A transition can depend on what an agent did, not only on who spoke. `ToolCalled("escalate")` fires when the speaker's turn included a call to a tool named `escalate`, and the graph's `default_target` handles every other case. Below, the desk submits a ticket, triage answers it or escalates it, and a specialist only speaks when triage escalated:

```python
import asyncio

from ag2 import Agent, tool
from ag2.config import OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED, EV_PACKET, AgentTarget, FromSpeaker, Hub, TerminateTarget, ToolCalled, Transition, TransitionGraph,
)


@tool
def escalate(reason: str) -> str:
    """Hand the ticket to a human specialist."""
    return "escalated"


async def main() -> None:
    config = OpenAIConfig(model="%%model=openai-small%%")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    desk = await hub.register(Agent("desk", prompt="You submit tickets.", config=config))
    triage = await hub.register(Agent("triage", prompt="Answer simple questions. Call escalate for refunds over 500 pounds.", config=config, tools=[escalate]))
    specialist = await hub.register(Agent("specialist", prompt="You handle escalated refund requests.", config=config))

    graph = TransitionGraph(
        initial_speaker=desk.agent_id,
        transitions=[
            Transition(when=FromSpeaker(desk.agent_id), then=AgentTarget(triage.agent_id)),
            Transition(when=ToolCalled("escalate"), then=AgentTarget(specialist.agent_id)),
        ],
        default_target=TerminateTarget(reason="resolved"),   # anything else ends the channel
        max_turns=4,
    )
    channel = await desk.open(type="workflow", target=[triage.agent_id, specialist.agent_id], knobs={"graph": graph.to_dict()})
    await channel.send("Customer wants a 900 pound refund.")

    while not any(e.event_type == EV_CHANNEL_CLOSED for e in await hub.read_wal(channel.channel_id)):
        await asyncio.sleep(0.1)
    for envelope in await hub.read_wal(channel.channel_id):
        if envelope.event_type == EV_PACKET:
            print(envelope.event_data["body"])
    await hub.close()


asyncio.run(main())
```

A tool can also choose the next speaker at run time by returning `Handoff(target="specialist", reason=...)` (the target is the agent's registered name), or end the channel by returning `Finish(summary=...)`; both are exported from `ag2.network`.

## MCP servers

```python
from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.tools import MCPServerConfig, MCPStdioServerConfig, MCPToolkit

agent = Agent(
    "ops",
    prompt="Use the filesystem and ticketing tools to answer questions.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[
        # local server launched as a subprocess over stdio
        MCPToolkit(MCPStdioServerConfig(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])),
        # remote streamable-HTTP server
        MCPToolkit(MCPServerConfig(server_url="https://mcp.example.com/mcp", authorization_token="...")),
    ],
)
```

`MCPToolkit` connects from your process and exposes each MCP tool as an ordinary function tool, so it works with any provider. The stdio subprocess gets a minimal environment, so pass `env=dict(os.environ)` (or the variables it needs) to `MCPStdioServerConfig` if `npx` must reach a registry through a proxy.  `allowed_tools=` / `blocked_tools=` / `tool_name_prefix=` on the server config narrow what the model sees. `MCPServerTool` instead forwards the server URL to providers that connect to MCP themselves (for example Anthropic). Requires `ag2[mcp]`.

### Serving an agent as an MCP server

`ag2.mcp.MCPServer` does the reverse: it exposes an `Agent` to other MCP clients as a single tool (named `ask` unless you set `tool_name=`) that takes a `message`. It speaks stdio through `run_stdio()`, and the server object is also an ASGI app that serves streamable HTTP at `/mcp`. Requires `ag2[mcp]`.

```python
# tax_server.py
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.mcp import MCPServer

agent = Agent("tax_expert", prompt="Answer tax questions in one sentence.", config=OpenAIConfig(model="%%model=openai-small%%"))
server = MCPServer(agent, name="tax-expert", tool_name="ask_tax_expert")

if __name__ == "__main__":
    asyncio.run(server.run_stdio())           # or: uvicorn tax_server:server  (HTTP at /mcp)
```

Another AG2 agent can then use it with `MCPToolkit(MCPStdioServerConfig(command="python", args=["tax_server.py"]))`, or with `MCPServerConfig(server_url="http://127.0.0.1:8000/mcp")` for the HTTP form. Any other MCP client works too.

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
    agent = Agent("calculator", config=AnthropicConfig(model="%%model=anthropic%%"), tools=[calc_add])
    card = build_card(agent, url="http://127.0.0.1:8000", description="Adds integers.")
    app = A2AServer(agent).build_jsonrpc(url="http://127.0.0.1:8000", card=card)
    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000)).serve()


# Client side: the remote agent's card URL is the "model config".
remote = Agent("remote", config=A2AConfig(card_url="http://127.0.0.1:8000"))
# reply = await remote.ask("What is 2 + 40?")
```

The card is served at `/.well-known/agent-card.json`. `A2AServer` can also build REST and gRPC transports sharing one task store, and supports push notifications via `push_config_store=`; pass `push_url_validator=` (added in 1.2.0) to guard against SSRF through client-supplied webhook URLs.

## Testing

`ag2.testing.TestConfig` replays scripted model outputs (strings for answers, `ToolCallEvent(...)` for tool calls), so agents can be unit-tested without network access. The example uses `pytest-asyncio` (`pip install pytest pytest-asyncio`):

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

The Classic `import autogen` API (`ConversableAgent`, `GroupChat`), which `ag2` shipped up to 0.14.0, now lives in [ag2ai/ag2-classic](https://github.com/ag2ai/ag2-classic). It is published on PyPI as `autogen` and is in maintenance mode. It has its own section on this site: [AG2 Classic](/ag2-guide/classic/). To port Classic code to 1.x (or AutoGen 0.2 code to Classic), follow the [migration guide](/ag2-guide/classic/ag2_classic_migration_guide/).

## Resources

- [AG2 documentation](https://docs.ag2.ai) · [GitHub: ag2ai/ag2](https://github.com/ag2ai/ag2)
- [AG2 Classic documentation](https://classic.docs.ag2.ai) · [GitHub: ag2ai/ag2-classic](https://github.com/ag2ai/ag2-classic)
- [Recipes](/ag2-guide/ag2_recipes/) · [Production guide](/ag2-guide/ag2_production_guide/) · [Diagrams](/ag2-guide/ag2_diagrams/) · [AG2 Classic](/ag2-guide/classic/)

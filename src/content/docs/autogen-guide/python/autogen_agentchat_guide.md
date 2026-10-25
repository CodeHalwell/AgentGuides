---
title: "Microsoft AutoGen 0.7 (AgentChat & Core)"
description: "Reference for Microsoft AutoGen's current API — autogen-agentchat, autogen-core and autogen-ext: AssistantAgent, tools, structured output, teams, GraphFlow, state, MCP and the Core runtime. Includes the project's archived status."
framework: "autogen"
language: "python"
sidebar:
  order: 1
---

> **Verified against `autogen-agentchat` %%version=autogen-agentchat%%** with `autogen-core` and `autogen-ext` at the same version (the three packages are released in lockstep; checked %%versions-checked%%). Every example was run offline with `autogen_ext.models.replay.ReplayChatCompletionClient` or constructed against the installed packages. Examples that use a literal `gpt-5-mini` need that older ID because this release does not recognise newer ones; [Model clients](#model-clients) shows the `model_info` that lets any other model run.

> **Status: archived.** Microsoft moved AutoGen to maintenance mode when it announced [Microsoft Agent Framework](/microsoft-agent-framework-guide/) as the successor, and the repository is now archived and no longer maintained by Microsoft (the upstream README states there are no bug fixes, security fixes or support). The last release is 0.7.5 (30 September 2025). Use this page to understand and maintain existing AutoGen 0.4–0.7 code; for new work, Microsoft points to Agent Framework and its [AutoGen migration guide](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/).

This page is about **Microsoft AutoGen** (`autogen_agentchat`, `autogen_core`, `autogen_ext`) — the actor-model rewrite Microsoft shipped as 0.4 in January 2025. It is **not** AG2. The community fork's `import autogen` API (AG2 Classic) is covered in the [AG2 guide](/ag2-guide/classic/); see [the AutoGen overview](/autogen-guide/) for how the two diverged.

## Packages

| Package | Role |
|---|---|
| `autogen-core` | Event-driven actor runtime: agents, messages, topics, subscriptions, `SingleThreadedAgentRuntime`, distributed gRPC runtime. |
| `autogen-agentchat` | High-level API on top of Core: `AssistantAgent`, `UserProxyAgent`, teams, termination conditions, `Console`. |
| `autogen-ext` | Integrations: model clients (OpenAI, Azure OpenAI, Anthropic, Ollama, …), MCP, code executors, memory stores. |

```bash
pip install "autogen-agentchat==%%version=autogen-agentchat%%" "autogen-ext[openai]==%%version=autogen-ext%%"
# MCP support: the 0.7.5 MCP adapter does not import against mcp 2.x, so pin the 1.x line
pip install "autogen-ext[mcp]==%%version=autogen-ext%%" "mcp<2"
```

Python 3.10+ is required. Because the project is archived, pin all three packages to the same version and pin integration dependencies (`openai`, `mcp`, `anthropic`) that later broke compatibility.

## Your first agent

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 23 °C and sunny."


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-5-mini")  # reads OPENAI_API_KEY; 0.7.5 only knows models released before it
    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        tools=[get_weather],
        system_message="You are a helpful assistant. Use tools when useful.",
        reflect_on_tool_use=True,   # summarise the tool result in a final model call
    )
    result = await agent.run(task="What is the weather in Paris?")
    print(result.messages[-1].content)
    await model_client.close()


asyncio.run(main())
```

- Agents are async; `run()` returns a `TaskResult` with the full `messages` list and a `stop_reason`. `run_stream()` yields messages as they are produced.
- Plain Python functions become tools via their signature and docstring (wrapped as `autogen_core.tools.FunctionTool`).
- Without `reflect_on_tool_use=True`, the agent's final message is the raw tool result.
- `AssistantAgent` is stateful: call `run()` again to continue the conversation, or `await agent.on_reset(CancellationToken())` to clear it.

## Streaming to the console

```python
from autogen_agentchat.ui import Console

# await Console(agent.run_stream(task="Write a haiku about tea."))
```

Pass `model_client_stream=True` to `AssistantAgent` to stream tokens as `ModelClientStreamingChunkEvent`s.

## Structured output

```python
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent


class Verdict(BaseModel):
    approved: bool
    reason: str


def make_reviewer(model_client) -> AssistantAgent:
    return AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="Approve the change only if it is safe to merge.",
        output_content_type=Verdict,   # final message becomes a StructuredMessage[Verdict]
    )

# result = await make_reviewer(client).run(task="Review: ...")
# verdict: Verdict = result.messages[-1].content
```

The model client must support structured output (`model_info["structured_output"]`), as current OpenAI models do.

## Teams

Teams coordinate several agents and stop when a **termination condition** fires. Conditions combine with `|` (either) and `&` (both).

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console


async def write_and_review(model_client) -> None:
    writer = AssistantAgent("writer", model_client=model_client, system_message="Write and revise short copy.")
    critic = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Give concrete feedback. Reply APPROVE when the copy is ready.",
    )
    team = RoundRobinGroupChat(
        [writer, critic],
        termination_condition=TextMentionTermination("APPROVE") | MaxMessageTermination(10),
    )
    result = await Console(team.run_stream(task="Write a one-line tagline for a tea shop."))
    print(result.stop_reason)
```

| Team | How the next speaker is chosen |
|---|---|
| `RoundRobinGroupChat` | Fixed rotation. |
| `SelectorGroupChat` | A model picks the next speaker from agent `description`s (`selector_prompt=`, optional `selector_func=` / `candidate_func=` overrides). |
| `Swarm` | Agents hand off explicitly with `handoffs=[...]`; `HandoffTermination(target="user")` pauses for a human. |
| `MagenticOneGroupChat` | An orchestrator plans, tracks progress and re-plans (Magentic-One). |
| `GraphFlow` | A directed graph built with `DiGraphBuilder`: sequences, parallel fan-out, conditional edges and loops. |

```python
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow


def build_pipeline(writer, editor) -> GraphFlow:
    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(editor)
    builder.add_edge(writer, editor)          # writer → editor
    return GraphFlow(participants=builder.get_participants(), graph=builder.build())
```

Useful termination conditions in `autogen_agentchat.conditions`: `MaxMessageTermination`, `TextMentionTermination`, `TokenUsageTermination`, `TimeoutTermination`, `HandoffTermination`, `SourceMatchTermination`, `FunctionCallTermination`, `ExternalTermination` (stop from outside, e.g. a UI button).

Running a team again continues from where it stopped; `await team.reset()` starts afresh.

## Human in the loop

```python
from autogen_agentchat.agents import UserProxyAgent

user = UserProxyAgent("user", input_func=input)  # replace input with your UI's async callback
```

Put the `UserProxyAgent` in a team to collect feedback during a run — it blocks the team while waiting — or stop the team (for example with `HandoffTermination` in a `Swarm`) and resume with a new `run(task=...)` once the human replies, which suits web back ends better.

## Agents as tools

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool


def build_coordinator(model_client) -> AssistantAgent:
    translator = AssistantAgent(
        "translator",
        model_client=model_client,
        description="Translates English text into French.",
        system_message="Translate the input into French. Reply with the translation only.",
    )
    return AssistantAgent("coordinator", model_client=model_client, tools=[AgentTool(translator)])
```

`TeamTool` does the same for a whole team.

## MCP tools

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


async def files_agent(model_client) -> None:
    params = StdioServerParams(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
    async with McpWorkbench(params) as workbench:
        agent = AssistantAgent("files", model_client=model_client, workbench=workbench)
        result = await agent.run(task="List the Markdown files here.")
        print(result.messages[-1].content)
```

The stdio subprocess gets a minimal environment; pass `env=dict(os.environ)` (or just the variables it needs) to `StdioServerParams` if `npx` has to reach a registry through a proxy. `StreamableHttpServerParams` and `SseServerParams` connect to remote servers. Only connect to MCP servers you trust. Remember the `mcp<2` pin above.

## Saving and restoring state

```python
import json


async def checkpoint(team, path: str) -> None:
    with open(path, "w") as f:
        json.dump(await team.save_state(), f)


async def restore(team, path: str) -> None:
    with open(path) as f:
        await team.load_state(json.load(f))
```

`save_state()` / `load_state()` exist on agents and teams; the state is JSON-serialisable.

## Autogen Core: the actor runtime

AgentChat is built on `autogen_core`. Use Core directly when you need custom message types, pub/sub topics or a distributed runtime.

```python
import asyncio
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler


@dataclass
class Greeting:
    text: str


class Greeter(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Replies to greetings")

    @message_handler
    async def on_greeting(self, message: Greeting, ctx: MessageContext) -> Greeting:
        return Greeting(text=f"Hello back, {message.text}!")


async def main() -> None:
    runtime = SingleThreadedAgentRuntime()
    await Greeter.register(runtime, "greeter", lambda: Greeter())
    runtime.start()
    reply = await runtime.send_message(Greeting("Ada"), AgentId("greeter", "default"))
    print(reply.text)
    await runtime.stop_when_idle()


asyncio.run(main())
```

Handlers are routed by the message's type annotation. `publish_message(..., topic_id=...)` broadcasts to subscribers (`@default_subscription`, `TypeSubscription`); `autogen_ext.runtimes.grpc` provides a distributed host and worker runtime.

### Publish and subscribe

Direct `send_message` calls reach one agent. `publish_message` broadcasts to a topic, and every agent subscribed to it receives the message, with no reply expected. `@default_subscription` subscribes an agent type to the default topic, which is the simplest case. `TypeSubscription` and explicit `TopicId`s route by topic type and source instead.

```python
import asyncio
from dataclasses import dataclass

from autogen_core import (
    DefaultTopicId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, default_subscription, message_handler,
)


@dataclass
class Order:
    sku: str
    quantity: int


@default_subscription
class Warehouse(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Reserves stock")
        self.reserved: list[Order] = []

    @message_handler
    async def on_order(self, message: Order, ctx: MessageContext) -> None:
        self.reserved.append(message)
        print(f"warehouse reserved {message.quantity} x {message.sku}")


@default_subscription
class Billing(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Invoices orders")

    @message_handler
    async def on_order(self, message: Order, ctx: MessageContext) -> None:
        print(f"billing invoiced {message.quantity} x {message.sku}")


async def main() -> None:
    runtime = SingleThreadedAgentRuntime()
    await Warehouse.register(runtime, "warehouse", lambda: Warehouse())
    await Billing.register(runtime, "billing", lambda: Billing())
    runtime.start()
    await runtime.publish_message(Order("tea-100g", 3), topic_id=DefaultTopicId())   # both subscribers receive it
    await runtime.stop_when_idle()


asyncio.run(main())
```

## Memory, model context and code execution

`memory=` adds retrieved content to the model context before each call. `model_context=` controls how much history the model sees. `CodeExecutorAgent` runs the code blocks in the messages it receives:

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-5-mini")  # 0.7.5 only knows models released before it

    memory = ListMemory()
    await memory.add(MemoryContent(content="The user prefers metric units.", mime_type=MemoryMimeType.TEXT))

    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        memory=[memory],
        model_context=BufferedChatCompletionContext(buffer_size=10),   # last 10 messages only
    )
    result = await agent.run(task="How tall is Mont Blanc?")
    print(result.messages[-1].content)

    executor = CodeExecutorAgent("executor", code_executor=LocalCommandLineCodeExecutor(work_dir="coding"))
    result = await executor.run(task="```python\nprint(6 * 7)\n```")
    print(result.messages[-1].content)   # 42

    await model_client.close()


asyncio.run(main())
```

- Other contexts: `UnboundedChatCompletionContext` (default), `HeadAndTailChatCompletionContext`, `TokenLimitedChatCompletionContext`.
- Vector memories live in `autogen_ext.memory` (for example `ChromaDBVectorMemory`, extra `autogen-ext[chromadb]`).
- `LocalCommandLineCodeExecutor` runs code on the host. Use `autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` (extra `autogen-ext[docker]`) for model-written code. Passing `model_client=` to `CodeExecutorAgent` lets it write the code as well as run it.

## Magentic-One and AutoGen Studio

- **`MagenticOneGroupChat`** (`autogen_agentchat.teams`) is the orchestrator team from Microsoft's Magentic-One system. An orchestrator keeps a task ledger, picks the next agent and re-plans after `max_stalls` turns without progress: `MagenticOneGroupChat([researcher, writer], model_client=model_client, max_turns=20)`. The full Magentic-One agent set (web surfer, file surfer, coder) is `autogen_ext.teams.magentic_one.MagenticOne` (extra `autogen-ext[magentic-one]`, plus Playwright for the browser).
- **AutoGen Studio** (`pip install autogenstudio`, then `autogenstudio ui --port 8080`) is a low-code web UI for building and testing teams from component JSON. It is a prototyping tool, not a deployment target. Its last release was 0.4.2.2 (May 2025), and it is archived along with the rest of the repository.

## Testing offline with ReplayChatCompletionClient

`autogen_ext.models.replay.ReplayChatCompletionClient` returns scripted responses in order, so tests need no network and no API key. A string is a plain text answer. A `CreateResult` with `finish_reason="function_calls"` and a list of `FunctionCall`s makes the agent call a tool, and the real tool function runs. Pass `model_info` so the client reports the capabilities (here, function calling) that `AssistantAgent` checks for.

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient


async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 23 °C and sunny."


def tool_call(name: str, arguments: str) -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id="call-1", name=name, arguments=arguments)],
        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
        cached=False,
    )


async def main() -> None:
    model_client = ReplayChatCompletionClient(
        [tool_call("get_weather", '{"city": "Paris"}'), "It is 23 °C and sunny in Paris."],
        model_info={"vision": False, "function_calling": True, "json_output": False, "family": "unknown", "structured_output": False},
    )
    agent = AssistantAgent("assistant", model_client=model_client, tools=[get_weather], reflect_on_tool_use=True)
    result = await agent.run(task="What is the weather in Paris?")

    kinds = [type(m).__name__ for m in result.messages]
    assert kinds == ["TextMessage", "ToolCallRequestEvent", "ToolCallExecutionEvent", "TextMessage"], kinds
    assert result.messages[2].content[0].content == "The weather in Paris is 23 °C and sunny."   # the real tool ran
    assert result.messages[-1].content == "It is 23 °C and sunny in Paris."
    print(kinds)

asyncio.run(main())
```

The replay client raises an error once the script is exhausted, which is useful: a test that makes more model calls than you scripted fails instead of hanging. Teams share one client in this style of test, so script the responses in the order the agents (and, for `SelectorGroupChat`, the selector) will call it.

## Model clients

| Client | Import | Extra |
|---|---|---|
| OpenAI / OpenAI-compatible | `autogen_ext.models.openai.OpenAIChatCompletionClient` | `autogen-ext[openai]` |
| Azure OpenAI | `autogen_ext.models.openai.AzureOpenAIChatCompletionClient` | `autogen-ext[openai,azure]` |
| Azure AI Foundry models | `autogen_ext.models.azure.AzureAIChatCompletionClient` | `autogen-ext[azure]` |
| Anthropic | `autogen_ext.models.anthropic.AnthropicChatCompletionClient` | `autogen-ext[anthropic]` |
| Ollama | `autogen_ext.models.ollama.OllamaChatCompletionClient` | `autogen-ext[ollama]` |
| Tests | `autogen_ext.models.replay.ReplayChatCompletionClient` | none |

AutoGen 0.7.5 was the final release, so its built-in model table stops at the models available in September 2025 (the examples on this page use `gpt-5-mini` for that reason). For any model the client does not recognise, including current ones, pass `model_info` describing its capabilities, or the constructor raises `ValueError: model_info is required`:

```python
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="%%model=openai-small%%",
    model_info=ModelInfo(vision=True, function_calling=True, json_output=True, structured_output=True, family="unknown"),
)
```

Model clients are components: `client.dump_component()` and `ChatCompletionClient.load_component(config)` round-trip them through JSON, which is how AutoGen Studio stores configurations.

## Moving on: Microsoft Agent Framework

Microsoft Agent Framework (`agent-framework`, currently %%version=agent-framework%% on PyPI) is the supported successor. Rough mapping:

| AutoGen 0.7 | Agent Framework |
|---|---|
| `AssistantAgent` + model client | `agent_framework.Agent` wrapping a provider chat client |
| `FunctionTool` / plain functions | Function tools (plain functions with type hints) |
| `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`, `MagenticOneGroupChat` | Orchestration patterns (sequential, concurrent, group chat, handoff, Magentic) on Workflows |
| `GraphFlow` | `WorkflowBuilder` workflows (typed, graph-based, with checkpointing) |
| `autogen_core` runtime | Workflow executors and the hosting layer |

See the [Microsoft Agent Framework guide](/microsoft-agent-framework-guide/) and Microsoft's official migration guide for exact APIs.

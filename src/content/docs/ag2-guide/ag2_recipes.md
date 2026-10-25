---
title: "AG2 Recipes"
description: "Copy-pasteable AG2 1.x recipes — chatbot loop, tool use, structured extraction, sub-agents, a reviewed pipeline on the Network, MCP tools, human approval for risky tools and offline tests."
framework: "ag2"
language: "python"
sidebar:
  label: "Recipes"
---

> **Verified against `ag2` %%version=ag2%%** (checked %%versions-checked%%). These recipes use the 1.x API (`import ag2`). For AG2 Classic (`import autogen`) recipes, see [AG2 Classic recipes](/ag2-guide/classic/ag2_classic_recipes/).

All recipes assume `pip install "ag2[openai]"` and `OPENAI_API_KEY` in the environment. Swap `OpenAIConfig` for any other provider config.

## Recipe 1: Terminal chatbot

`AgentReply.ask` continues the same conversation, so a chat loop just keeps the latest reply.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig

assistant = Agent("assistant", prompt="You are a helpful assistant.", config=OpenAIConfig(model="%%model=openai-small%%"))


async def main() -> None:
    reply = None
    while (text := input("you> ").strip()) not in {"", "exit"}:
        reply = await (reply.ask(text) if reply else assistant.ask(text))
        print("bot>", reply.body)


asyncio.run(main())
```

## Recipe 2: Calculator tool

```python
import asyncio
import operator

from ag2 import Agent, tool
from ag2.config import OpenAIConfig

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


@tool
def calculator(a: float, b: float, op: str) -> float:
    """Apply a binary operator (+, -, *, /) to two numbers."""
    return OPS[op](a, b)


agent = Agent(
    "maths",
    prompt="Use the calculator for every arithmetic step.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[calculator],
)

print(asyncio.run(agent.ask("What is 123 * 456?")).body)
```

## Recipe 3: Structured extraction

```python
import asyncio
from dataclasses import dataclass

from ag2 import Agent
from ag2.config import OpenAIConfig


@dataclass
class Invoice:
    supplier: str
    total: float
    currency: str


extractor = Agent(
    "extractor",
    prompt="Extract invoice fields exactly as written.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    response_schema=Invoice,
)


async def main() -> None:
    reply = await extractor.ask("ACME Ltd — total due GBP 1,240.50 by 30 November.")
    invoice = await reply.content()
    print(invoice)


asyncio.run(main())
```

## Recipe 4: A specialist as a sub-agent

`Agent.as_tool()` lets a coordinator call another agent like any other tool.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig

config = OpenAIConfig(model="%%model=openai-small%%")

translator = Agent("translator", prompt="Translate the given text into French. Reply with the translation only.", config=config)

coordinator = Agent(
    "coordinator",
    prompt="Answer the user. When they ask for French, delegate to the translator tool.",
    config=config,
    tools=[translator.as_tool(description="Translate text into French.")],
)

print(asyncio.run(coordinator.ask("Say 'good morning, team' in French.")).body)
```

## Recipe 5: Write → review pipeline on the Network

The 1.x replacement for a sequential `GroupChat`: a `workflow` channel driven by a `TransitionGraph`. The client's `channel.send(...)` is the client's own turn, so the client's model does not run; the writer and reviewer are the agents that call a model.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import EV_CHANNEL_CLOSED, EV_PACKET, Hub, TransitionGraph


async def main() -> None:
    config = OpenAIConfig(model="%%model=openai-small%%")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

    client = await hub.register(Agent("client", prompt="You commission short pieces.", config=config))
    writer = await hub.register(Agent("writer", prompt="Write 100 words on the brief.", config=config))
    reviewer = await hub.register(Agent("reviewer", prompt="List concrete improvements, then a revised draft.", config=config))

    graph = TransitionGraph.sequence([client.agent_id, writer.agent_id, reviewer.agent_id])
    channel = await client.open(type="workflow", target=[writer.agent_id, reviewer.agent_id], knobs={"graph": graph.to_dict()})
    await channel.send("Brief: why teams should write ADRs.")

    while not any(e.event_type == EV_CHANNEL_CLOSED for e in await hub.read_wal(channel.channel_id)):
        await asyncio.sleep(0.1)
    for envelope in await hub.read_wal(channel.channel_id):
        if envelope.event_type == EV_PACKET:
            print(envelope.event_data["body"], "\n---")
    await hub.close()


asyncio.run(main())
```

## Recipe 6: Filesystem tools over MCP

Requires `pip install "ag2[openai,mcp]"` and Node.js for `npx`.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.tools import MCPStdioServerConfig, MCPToolkit

files = MCPToolkit(
    MCPStdioServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        allowed_tools=["list_directory", "read_text_file"],  # read-only surface
    )
)

agent = Agent("librarian", prompt="Answer questions about files in the current directory.", config=OpenAIConfig(model="%%model=openai-small%%"), tools=[files])

print(asyncio.run(agent.ask("Which Markdown files are here, and what is the first heading of each?")).body)
```

## Recipe 7: Human approval before a risky tool

`approval_required()` wraps a single tool. Before the tool runs, the agent's `hitl_hook` is asked the question; any answer other than `y`, `yes` or `1` (and `always`, unless `allow_always=False`) is treated as a refusal, and the model receives `denied_message` as the tool result.

```python
import asyncio

from ag2 import Agent, tool
from ag2.config import OpenAIConfig
from ag2.middleware import approval_required

RECORDS = {"u1": "Ada", "u2": "Grace"}


@tool(middleware=[approval_required(denied_message="The operator refused. Do not retry.", allow_always=False)])
def delete_record(record_id: str) -> str:
    """Permanently delete a record."""
    RECORDS.pop(record_id)
    return f"deleted {record_id}"


def operator(event) -> str:
    """hitl_hook: show the question and return the operator's answer."""
    return input(event.content)


agent = Agent(
    "janitor",
    prompt="Delete records when asked. Report what happened.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[delete_record],
    hitl_hook=operator,
)


async def main() -> None:
    reply = await agent.ask("Delete record u1.")
    print(reply.body, RECORDS)


asyncio.run(main())
```

In a web service, replace `input` with a function that puts the question on a queue and awaits the answer (the hook may be `async`).

## Recipe 8: Test a tool-using agent offline

`TrackingConfig` wraps any model config and records what the framework sent on each model call: the tools advertised, the prompt, and the dependencies and variables in force. Combine it with a scripted `TestConfig` to assert both the behaviour and the request. Run it with `pip install pytest pytest-asyncio` and `pytest --asyncio-mode=strict`.

```python
import pytest

from ag2 import Agent, tool
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig


@tool
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount:.2f} on {order_id}"


@pytest.mark.asyncio
async def test_agent_offers_refund_tool_and_uses_result() -> None:
    script = TestConfig(
        ToolCallEvent(name="refund", arguments='{"order_id": "A1", "amount": 9.5}'),
        "Your refund of 9.50 is on its way.",
    )
    tracked = TrackingConfig(script)           # wraps any config and records every model call
    agent = Agent("support", prompt="You handle refunds.", tools=[refund], config=tracked)

    reply = await agent.ask("Refund order A1, 9.50 please.")

    assert reply.body == "Your refund of 9.50 is on its way."
    first_call = tracked.calls[0]
    assert [t.function.name for t in first_call.tools] == ["refund"]
    assert "You handle refunds." in first_call.prompt
    assert len(tracked.calls) == 2             # one call that requested the tool, one that answered
```

---
title: "OpenAI Agents SDK: Migrating from Swarm"
description: "Moving Python code from the experimental Swarm library to the OpenAI Agents SDK: concept mapping, side-by-side examples and a checklist."
framework: "openai-agents-sdk"
language: "python"
sidebar:
  label: "Migrating from Swarm"
---

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%). The Swarm snippets are checked against the final state of the [openai/swarm](https://github.com/openai/swarm) repository, an educational project that is no longer developed. Swarm was Python-only.

Swarm showed the two ideas the Agents SDK is built on, agents with instructions and functions, and handoffs by returning another agent, in a few hundred lines. The SDK keeps those ideas and adds what Swarm left out: a typed run loop with limits, guardrails, sessions, tracing, structured output, MCP, approvals and test doubles. Most migrations are mechanical.

## Concept mapping

| Swarm | Agents SDK | Notes |
|-------|------------|-------|
| `from swarm import Swarm, Agent` | `from agents import Agent, Runner` | `pip install openai-agents` |
| `client = Swarm(); client.run(agent, messages)` | `await Runner.run(agent, input)` or `Runner.run_sync(...)` | `input` is a string or a list of input items |
| `Agent(functions=[f])` | `Agent(tools=[f])` with `@function_tool` on `f` | The schema comes from type hints and the docstring |
| A function that returns an `Agent` | `Agent(handoffs=[other])` | The SDK generates a `transfer_to_<name>` tool |
| `context_variables` dict | `context=` any object; tools take `RunContextWrapper[T]` as first parameter | The context is never sent to the model |
| `Result(value, agent, context_variables)` | Return a value from the tool; mutate `ctx.context`; use `handoffs` | |
| `instructions=lambda context_variables: ...` | `instructions=lambda ctx, agent: ...` | `ctx.context` is your object |
| `response.messages` | `result.to_input_list()`, or a `session=` | Sessions store history for you |
| `response.agent` | `result.last_agent` | |
| `response.context_variables` | your context object, after the run | |
| `max_turns=float("inf")` (default) | `max_turns=10` (default); `None` for no limit | Raises `MaxTurnsExceeded` |
| `stream=True` | `Runner.run_streamed(...)` then `stream_events()` | |
| `debug=True` | Tracing (on by default) or `enable_verbose_stdout_logging()` | |
| `model_override=` | `RunConfig(model=...)` | |
| `tool_choice`, `parallel_tool_calls` on `Agent` | `ModelSettings(tool_choice=..., parallel_tool_calls=...)` | |
| Chat Completions API | Responses API by default | `OpenAIChatCompletionsModel` keeps Chat Completions |

## Agents, tools and context

Before (Swarm):

```python
from swarm import Agent, Swarm

def get_order(order_id: str, context_variables: dict) -> str:
    """Look up an order for the current customer."""
    return f"Order {order_id} for {context_variables['customer']} has shipped."

support = Agent(
    name="Support",
    instructions=lambda context_variables: f"Help {context_variables['customer']}.",
    functions=[get_order],
)

client = Swarm()
response = client.run(
    agent=support,
    messages=[{"role": "user", "content": "Where is order 42?"}],
    context_variables={"customer": "Alice"},
)
print(response.messages[-1]["content"])
```

After (Agents SDK):

```python
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool


@dataclass
class Customer:
    name: str


@function_tool
def get_order(ctx: RunContextWrapper[Customer], order_id: str) -> str:
    """Look up an order for the current customer."""
    return f"Order {order_id} for {ctx.context.name} has shipped."


def instructions(ctx: RunContextWrapper[Customer], agent: Agent[Customer]) -> str:
    return f"Help {ctx.context.name}."


support = Agent[Customer](name="Support", instructions=instructions, tools=[get_order])

result = Runner.run_sync(support, "Where is order 42?", context=Customer(name="Alice"))
print(result.final_output)
```

## Handoffs

Before: a function returns the next agent.

```python
from swarm import Agent

refunds = Agent(name="Refunds", instructions="Handle refunds.")

def transfer_to_refunds():
    return refunds

triage = Agent(name="Triage", instructions="Route the user.", functions=[transfer_to_refunds])
```

After: list the agent in `handoffs`. `handoff_description` tells the routing agent when to use it, and `handoff()` adds an input schema, an `on_handoff` callback or a history filter when you need one.

```python
from agents import Agent, Runner

refunds = Agent(
    name="Refunds",
    handoff_description="Handles refund requests",
    instructions="Handle refunds.",
)
triage = Agent(name="Triage", instructions="Route the user.", handoffs=[refunds])

result = Runner.run_sync(triage, "I want a refund for order 42.")
print(result.last_agent.name, result.final_output)
```

If the routing agent should stay in charge and only consult the specialist, use `refunds.as_tool(tool_name=..., tool_description=...)` instead of a handoff.

## Conversation history

Swarm returned the new messages and left you to keep them. In the SDK, either pass the previous result back in, or let a session do it:

```python
import asyncio

from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant", instructions="Be brief.")


async def main() -> None:
    # Manual: carry the history yourself
    first = await Runner.run(agent, "My name is Alice.")
    second = await Runner.run(agent, first.to_input_list() + [{"role": "user", "content": "What's my name?"}])
    print(second.final_output)

    # Session: the SDK loads and saves history for this conversation id
    session = SQLiteSession("conversation-1")
    await Runner.run(agent, "My name is Alice.", session=session)
    print((await Runner.run(agent, "What's my name?", session=session)).final_output)


asyncio.run(main())
```

## Checklist

- Replace `swarm` imports and `Swarm().run(...)` calls with `agents` and `Runner`.
- Decorate every function with `@function_tool`; add type hints and a docstring, which become the schema.
- Replace `context_variables` with a dataclass or Pydantic model passed as `context=`, and add a `RunContextWrapper[...]` first parameter to tools that need it.
- Replace "return an agent" functions with `handoffs=[...]`.
- Set `max_turns` deliberately: Swarm had no limit by default, the SDK stops at 10.
- Set `model=` on each agent. Swarm defaulted to `gpt-4o`; the SDK's default is different and has changed between releases.
- Decide where history lives (a session backend, or `previous_response_id` / `conversation_id` for server-side state).
- Add what Swarm did not have where you need it: `output_type`, [guardrails](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/#guardrails), [approvals](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#human-in-the-loop-approvals), and offline tests with [`ScriptedModel`](/openai-agents-sdk-guides/openai_agents_sdk_production_guide/#testing-without-the-api).
- Pin the SDK to a minor version: it is `0.Y.Z`, and minor releases can include breaking changes ([release table](/openai-agents-sdk-guides/#whats-new)).

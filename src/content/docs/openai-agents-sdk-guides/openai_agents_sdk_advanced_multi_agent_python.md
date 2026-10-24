---
title: "OpenAI Agents SDK Advanced Multi‑Agent (Python)"
description: "Choosing between handoffs, agents as tools and code orchestration in the Python OpenAI Agents SDK."
sidebar:
  label: "Multi-agent patterns"
  order: 3
framework: "openai-agents-sdk"
language: "python"
---

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%).
> Upstream: https://github.com/openai/openai-agents-python

## Patterns

- **Handoffs** — the active agent passes the whole conversation to a specialist, which then owns the reply. Good for routing (triage → billing).
- **Agents as tools** — a manager agent calls specialists via `Agent.as_tool()` and keeps control of the final answer. Good for orchestration and synthesis.
- **Guardrails** — input guardrails run on the first agent's input, output guardrails on the final agent's output; tool guardrails wrap individual function tools.
- **Code orchestration** — chain `Runner.run` calls in Python (or `asyncio.gather` them) when the flow is deterministic.

## Handoffs vs agents as tools

```python
import asyncio
from agents import Agent, Runner

researcher = Agent(
    name="Researcher",
    handoff_description="Finds facts and sources",
    instructions="Research the question and list key facts with sources.",
)
writer = Agent(
    name="Writer",
    handoff_description="Writes polished prose",
    instructions="Write a clear, concise answer from the notes you are given.",
)

# 1. Routing: the triage agent hands the conversation over
triage = Agent(
    name="Triage",
    instructions="Hand off research questions to Researcher and writing requests to Writer.",
    handoffs=[researcher, writer],
)

# 2. Orchestration: the manager keeps control and calls specialists as tools
manager = Agent(
    name="Manager",
    instructions="Use the research tool to gather facts, then the writing tool to draft the answer.",
    tools=[
        researcher.as_tool(tool_name="research", tool_description="Research a topic"),
        writer.as_tool(tool_name="write", tool_description="Draft prose from notes"),
    ],
)

async def main():
    routed = await Runner.run(triage, "What causes the northern lights?")
    print(routed.last_agent.name, "→", routed.final_output)

    orchestrated = await Runner.run(manager, "Explain the northern lights for a ten-year-old.")
    print(orchestrated.final_output)

asyncio.run(main())
```

Since 0.7.0 handoffs no longer nest prior history by default; set `RunConfig(nest_handoff_history=True)` (or `handoff(agent, nest_handoff_history=True)`) if a specialist needs the earlier transcript packaged as a single recap message.

## Handoffs in more detail

Listing an agent in `handoffs=[...]` is the short form. `handoff()` wraps the agent when you need more control: the tool name and description the model sees, a typed payload the model fills in when it hands off, a callback that runs at the moment of transfer, and `is_enabled` to hide the handoff for some runs.

```python
import asyncio

from pydantic import BaseModel

from agents import Agent, RunContextWrapper, Runner, handoff


class Escalation(BaseModel):
    reason: str


async def log_escalation(ctx: RunContextWrapper[None], data: Escalation) -> None:
    # Runs as the handoff happens: open a ticket, page someone, warm a cache.
    print("escalating:", data.reason)


billing = Agent(
    name="Billing",
    handoff_description="Billing disputes and refunds",
    instructions="Resolve billing disputes.",
)

triage = Agent(
    name="Triage",
    instructions="Hand billing problems to the billing team, with the reason.",
    handoffs=[
        handoff(
            billing,
            tool_name_override="escalate_to_billing",
            tool_description_override="Send a billing problem to the billing team",
            input_type=Escalation,
            on_handoff=log_escalation,
        )
    ],
)


async def main() -> None:
    result = await Runner.run(triage, "I was charged twice for order 42.")
    print(result.last_agent.name, result.final_output)


asyncio.run(main())
```

`input_type` is validated before `on_handoff` is called. It only describes the handoff tool call, so the receiving agent does not get it as input; use the callback to act on it. After a handoff, `result.last_agent` is the agent that produced the answer, and a follow-up turn in the same session starts from that agent only if you pass it in yourself (`Runner.run(result.last_agent, ...)`). The runner does not remember it for you.

## Agents as tools in more detail

`as_tool()` runs the specialist as a nested run and returns its output to the manager as the tool result. Four options are worth knowing:

- `parameters=` takes a Pydantic model, so the manager passes structured input instead of a free-text string. The specialist receives it as JSON.
- `custom_output_extractor=` receives the nested `RunResult` and returns the string the manager sees. Use it to return one field of a structured output, or to trim long answers.
- `is_enabled=` is a bool or a function of `(context, agent)`. A disabled tool is not offered to the model for that run.
- `needs_approval=` pauses the run before the specialist starts, like any function tool.

```python
import asyncio

from pydantic import BaseModel

from agents import Agent, RunContextWrapper, Runner


class Brief(BaseModel):
    topic: str
    audience: str


researcher = Agent(
    name="Researcher",
    instructions="Research the topic for the stated audience. Lead with the key fact.",
)


async def first_sentence(result) -> str:
    return str(result.final_output).split(".")[0] + "."


def research_enabled(ctx: RunContextWrapper[dict], agent: Agent) -> bool:
    return bool(ctx.context.get("research_allowed"))


manager = Agent[dict](
    name="Manager",
    instructions="Use the research tool when you need facts, then answer.",
    tools=[
        researcher.as_tool(
            tool_name="research",
            tool_description="Research a topic for an audience",
            parameters=Brief,
            custom_output_extractor=first_sentence,
            is_enabled=research_enabled,
        )
    ],
)


async def main() -> None:
    result = await Runner.run(
        manager, "Explain the northern lights to a child.", context={"research_allowed": True}
    )
    print(result.final_output)


asyncio.run(main())
```

Nested runs share the parent's context object but not its session; pass `session=` to `as_tool()` if the specialist should keep its own history.

## Choosing between them

| You want | Use |
|----------|-----|
| The specialist to talk to the user from now on | Handoff |
| The manager to combine several specialists' answers | Agent as tool |
| A fixed sequence (extract, then validate, then write) | Plain Python: await one `Runner.run` after another |
| Independent work on the same input | `asyncio.gather` over `Runner.run` calls |
| A structured final answer from a team | Agents as tools, with `output_type` on the manager |

The last row matters because of how the run ends. After a handoff, the specialist's output type is the run's output type, so a manager with `output_type=Report` and `handoffs=[...]` can finish with whatever the specialist returns, typically a string. Keeping the manager in charge with `as_tool()` guarantees the final output has the manager's type.

To stop the run as soon as a tool has answered, without a second model call, set `tool_use_behavior="stop_on_first_tool"` on the agent. `tool_use_behavior={"stop_at_tool_names": ["lookup"]}` stops on the named tools only.

## Parallel specialists in code

Using the `researcher` agent above and any second agent:

```python
import asyncio

from agents import Agent, Runner

researcher = Agent(name="Researcher", instructions="List key facts.")
critic = Agent(name="Critic", instructions="List weaknesses and open questions.")


async def compare(question: str) -> tuple[str, str]:
    facts, doubts = await asyncio.gather(
        Runner.run(researcher, question),
        Runner.run(critic, question),
    )
    return facts.final_output, doubts.final_output


print(asyncio.run(compare("Is a four-day week good for productivity?")))
```

`asyncio.gather` raises the first exception and leaves the other runs going; pass `return_exceptions=True` if you want every outcome. Do not share one session between concurrent runs: their items would interleave.

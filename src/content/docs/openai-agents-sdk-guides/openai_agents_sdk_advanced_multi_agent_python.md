---
title: "OpenAI Agents SDK Advanced Multi‑Agent (Python)"
description: "Advanced multi-agent coordination, handoffs, and guardrails in Python."
framework: openai-agents-sdk
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

## Parallel specialists in code

```python
async def compare(question: str):
    a, b = await asyncio.gather(
        Runner.run(researcher, question),
        Runner.run(writer, question),
    )
    return a.final_output, b.final_output
```

---
title: "Haystack Multi-Agent Systems Guide"
description: "Building collaborative multi-agent applications with Haystack 3 — AgentTool coordinators, agents as pipeline stages, routing, parallel fan-out, refinement loops and shared state."
framework: haystack
---

# Haystack Multi-Agent Systems Guide

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). First written in 2025 for Haystack 2.x; rewritten for the 3.x `Agent` API.

---

## Table of Contents

1. [Introduction to Multi-Agent Systems](#introduction-to-multi-agent-systems)
2. [Multi-Agent Architecture Patterns](#multi-agent-architecture-patterns)
3. [Agent Specialization & Roles](#agent-specialization--roles)
4. [Agent Communication Protocols](#agent-communication-protocols)
5. [Coordination & Orchestration](#coordination--orchestration)
6. [Complete Production Multi-Agent Example](#complete-production-multi-agent-example)
7. [Summary](#summary)

For the short version see [Multi-Agent Pipeline Design Patterns](/haystack-guide/haystack_comprehensive_guide/#multi-agent-pipeline-design-patterns); for deployment, scaling and caching see the [production guide](/haystack-guide/haystack_production_guide/).

---

## Introduction to Multi-Agent Systems

A single agent with twenty tools and a long system prompt tends to pick the wrong tool, forget instructions and burn tokens re-reading schemas. Splitting the work between specialised agents keeps each prompt short and each tool set small, at the cost of more LLM calls and more moving parts. Reach for multiple agents when the sub-tasks genuinely need different instructions or tools; otherwise one agent plus a deterministic pipeline is cheaper and easier to debug.

### Building blocks

Haystack has no separate "multi-agent" module. You compose agents with the same primitives used everywhere else:

| Primitive | Role in a multi-agent system |
|-----------|------------------------------|
| `Agent` | A specialist: a chat generator, a system prompt and a few tools |
| `AgentTool` | Exposes an agent as a tool, so a coordinator agent can delegate to it |
| `Pipeline` | Fixed wiring between agents (and retrievers, validators, routers) |
| `ConditionalRouter` / `LLMMessagesRouter` | Rule-based or model-based dispatch to the right specialist |
| `State` + `state_schema` | Structured data shared between tools during one agent's run |
| `run_async` / `Pipeline.run_async` | Concurrent execution of independent agents |
| Hooks | Guard-rails and control flow around each agent's loop |

All examples below assume these imports and a small helper:

```python
from typing import Annotated

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import AgentTool, tool


def make_agent(system_prompt: str, tools=None, **kwargs) -> Agent:
    return Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5-mini"),
        system_prompt=system_prompt,
        tools=tools,
        **kwargs,
    )


@tool
def web_search(query: Annotated[str, "Search query"]) -> str:
    """Search the web for current information."""
    return f"(stub) results for {query!r}"
```

---

## Multi-Agent Architecture Patterns

### Pattern 1: Sequential Collaboration

Each agent's final message feeds the next agent's `user_prompt`. The order is fixed, so a pipeline is the natural fit.

```python
researcher = make_agent(
    "Research the topic and list the key facts with sources.",
    tools=[web_search],
    user_prompt="Topic: {{ topic }}",
)
writer = make_agent(
    "Write a clear 300-word article from the research notes.",
    user_prompt="Research notes:\n{{ research.text }}",
)

sequential = Pipeline()
sequential.add_component("researcher", researcher)
sequential.add_component("writer", writer)
sequential.connect("researcher.last_message", "writer.research")

result = sequential.run({
    "researcher": {"topic": "grid-scale batteries", "messages": []},
    "writer": {"messages": []},
})
print(result["writer"]["last_message"].text)
```

`messages` is a mandatory input of every `Agent`; pass `[]` when the `user_prompt` supplies the turn.

### Pattern 2: Parallel Processing

Independent analyses can run at the same time. Either gather `run_async` calls yourself, or put the agents on parallel branches of a pipeline and use `await pipeline.run_async(...)`.

```python
import asyncio

analysts = {
    "market": make_agent("You analyse market size and growth."),
    "competition": make_agent("You analyse competitors and their positioning."),
    "customers": make_agent("You analyse customer needs and sentiment."),
}
synthesiser = make_agent("Combine the analyses into one balanced brief.")


async def parallel_brief(question: str) -> str:
    results = await asyncio.gather(*[
        agent.run_async(messages=[ChatMessage.from_user(question)]) for agent in analysts.values()
    ])
    sections = "\n\n".join(
        f"## {name}\n{r['last_message'].text}" for name, r in zip(analysts, results)
    )
    final = await synthesiser.run_async(messages=[ChatMessage.from_user(sections)])
    return final["last_message"].text


print(asyncio.run(parallel_brief("Should we enter the UK e-bike market?")))
```

### Pattern 3: Hierarchical Delegation

A manager agent decides which specialists to call, in what order, and how often. Each specialist is wrapped in an `AgentTool`.

```python
research_agent = make_agent("You find information and cite sources.", tools=[web_search])
data_agent = make_agent("You answer questions about sales data. Be precise with numbers.")
coding_agent = make_agent("You write small, correct Python snippets.")

manager = make_agent(
    "You manage a team. Break the request into sub-tasks, delegate each to the right specialist, "
    "and assemble the final answer. Do not do specialist work yourself.",
    tools=[
        AgentTool(agent=research_agent, name="research", description="Finds information and best practice."),
        AgentTool(agent=data_agent, name="data", description="Answers questions about sales data."),
        AgentTool(agent=coding_agent, name="code", description="Writes Python code."),
    ],
    max_agent_steps=12,
)

result = manager.run(messages=[ChatMessage.from_user("Build a dashboard plan for our Q4 sales.")])
print(result["last_message"].text)
print(result["tool_call_counts"])   # how often each specialist was used
```

`AgentTool` sends the sub-agent a single user message and, by default, returns its final text to the manager. Use `outputs_to_string` to return something else, or `outputs_to_state` to capture the sub-agent's outputs in the manager's state. Set `tool_streaming_callback_passthrough=True` on the manager to stream sub-agent output too.

### Pattern 4: Iterative Refinement Loop

A creator drafts; a critic reviews; the creator revises. The simplest robust version is an explicit loop in Python with a hard cap:

```python
creator = make_agent("You write concise product descriptions.")
critic = make_agent(
    "You review product descriptions. Reply with APPROVED if it is clear, accurate and under 80 words; "
    "otherwise list specific fixes."
)


def refine(brief: str, max_rounds: int = 3) -> str:
    history = [ChatMessage.from_user(brief)]
    draft = ""
    for _ in range(max_rounds):
        draft = creator.run(messages=history)["last_message"].text
        review = critic.run(messages=[ChatMessage.from_user(draft)])["last_message"].text
        if "APPROVED" in review:
            break
        history += [ChatMessage.from_assistant(draft), ChatMessage.from_user(f"Revise: {review}")]
    return draft


print(refine("Describe a stainless-steel water bottle."))
```

The same loop can be expressed inside a pipeline with a `ConditionalRouter` and a `BranchJoiner` (see [Iterative refinement loop](/haystack-guide/haystack_comprehensive_guide/#iterative-refinement-loop)), or inside one agent with an `on_exit` hook that sets `continue_run`.

### Pattern 5: Consensus Building

Several agents with different stances answer independently; a judge reconciles them. Varying the system prompt usually matters more than varying temperature.

```python
panel = {
    "optimist": make_agent("You argue the strongest case in favour."),
    "sceptic": make_agent("You argue the strongest case against."),
    "pragmatist": make_agent("You weigh costs and practicalities."),
}
judge = make_agent("You read several opinions and write a balanced recommendation with a confidence level.")


async def consensus(question: str) -> str:
    opinions = await asyncio.gather(*[
        a.run_async(messages=[ChatMessage.from_user(question)]) for a in panel.values()
    ])
    transcript = "\n\n".join(f"{n}: {o['last_message'].text}" for n, o in zip(panel, opinions))
    verdict = await judge.run_async(messages=[ChatMessage.from_user(f"Question: {question}\n\n{transcript}")])
    return verdict["last_message"].text


print(asyncio.run(consensus("Should a 20-person company adopt a four-day week?")))
```

---

## Agent Specialization & Roles

### Defining Agent Expertise

Keep role definitions as data so they are easy to review and test:

```python
from dataclasses import dataclass, field

from haystack.tools import Tool


@dataclass
class Role:
    name: str
    description: str          # what the coordinator sees
    instructions: str         # the specialist's system prompt
    tools: list[Tool] = field(default_factory=list)
    max_steps: int = 6


ROLES = [
    Role("researcher", "Finds facts and sources.", "Research thoroughly; cite every claim.", [web_search]),
    Role("analyst", "Interprets data and identifies trends.", "Analyse the material; state assumptions."),
    Role("writer", "Produces polished prose.", "Write clearly for a non-specialist reader."),
]


def build_team(roles: list[Role]) -> list[AgentTool]:
    return [
        AgentTool(
            agent=make_agent(r.instructions, tools=r.tools or None, max_agent_steps=r.max_steps),
            name=r.name,
            description=r.description,
        )
        for r in roles
    ]


lead = make_agent(
    "Coordinate your team to answer the request. Delegate, then synthesise.",
    tools=build_team(ROLES),
    max_agent_steps=12,
)
```

Guidelines that hold up in practice:

- Give each specialist two to five tools. If a role needs more, wrap them in a `SearchableToolset`.
- Write the `AgentTool` description for the coordinator: say what the specialist is good at and what input it expects.
- Cap `max_agent_steps` per specialist; a runaway sub-agent multiplies cost.
- Use a cheaper model for narrow specialists and a stronger one for the coordinator.

---

## Agent Communication Protocols

### Message Passing Between Agents

Agents communicate through `ChatMessage`s. In a pipeline you can connect an agent's `messages` output (the full transcript) or `last_message` (just the final reply) to the next agent:

```python
reviewer = make_agent("Review the conversation so far and point out any factual errors.")

chain = Pipeline()
chain.add_component("drafter", make_agent("Answer the user's question."))
chain.add_component("reviewer", reviewer)
chain.connect("drafter.messages", "reviewer.messages")   # full transcript, including tool calls

out = chain.run({"drafter": {"messages": [ChatMessage.from_user("When was the Eiffel Tower built?")]}})
print(out["reviewer"]["last_message"].text)
```

Passing the whole transcript gives the next agent full context but costs tokens. Passing `last_message` into a `user_prompt` variable is cheaper and keeps each agent's context focused.

### Shared Context & State

Within one agent's run, tools share data through `State`. Declare keys in `state_schema`, write them with `outputs_to_state` and read them with `inputs_from_state`. Combined with `AgentTool`, this lets a coordinator collect structured results from its specialists:

```python
@tool(outputs_to_state={"facts": {"source": "facts"}})
def record_fact(fact: Annotated[str, "A verified fact with its source"]) -> dict:
    """Record a verified fact."""
    return {"facts": [fact]}


fact_finder = make_agent(
    "Find facts and record each with record_fact.",
    tools=[web_search, record_fact],
    state_schema={"facts": {"type": list[str]}},
)

coordinator = make_agent(
    "Use the fact finder, then write a summary.",
    tools=[AgentTool(
        agent=fact_finder,
        name="fact_finder",
        description="Finds and records facts on a topic.",
        outputs_to_state={"facts": {"source": "facts"}},   # copy the sub-agent's facts into our state
    )],
    state_schema={"facts": {"type": list[str]}},
)

result = coordinator.run(messages=[ChatMessage.from_user("Summarise what is known about perovskite solar cells.")])
print(result.get("facts", []))
```

List-typed state keys are merged (appended) by default; other types are replaced. Supply a `handler` in the schema entry to change that, e.g. `{"type": dict, "handler": my_merge}`.

Across separate runs or processes, persist what you need explicitly: serialise messages with `ChatMessage.to_dict()` and store state values in your database.

---

## Coordination & Orchestration

### Dynamic Agent Selection

Route each request to one specialist instead of asking a coordinator to delegate. `LLMMessagesRouter` classifies messages with a model and sends them to the matching output; unmatched messages go to `unmatched`.

```python
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import LLMMessagesRouter

router = LLMMessagesRouter(
    chat_generator=OpenAIChatGenerator(model="gpt-5-mini"),
    output_names=["billing", "technical"],
    output_patterns=["billing", "technical"],
    system_prompt="Classify the request as 'billing' or 'technical'. Reply with one word.",
)

dispatch = Pipeline()
dispatch.add_component("router", router)
dispatch.add_component("billing", make_agent("You handle billing questions."))
dispatch.add_component("technical", make_agent("You handle technical support questions."))
dispatch.add_component("general", make_agent("You handle general enquiries."))
dispatch.add_component("joiner", BranchJoiner(ChatMessage))

dispatch.connect("router.billing", "billing.messages")
dispatch.connect("router.technical", "technical.messages")
dispatch.connect("router.unmatched", "general.messages")
for name in ["billing", "technical", "general"]:
    dispatch.connect(f"{name}.last_message", "joiner")

out = dispatch.run({"router": {"messages": [ChatMessage.from_user("I was charged twice this month.")]}})
print(out["joiner"]["value"].text)
```

For deterministic rules (keywords, metadata, user tier), use `ConditionalRouter` instead; it costs nothing and is easy to test. To branch on how an agent finished, route on its `exit_reason` output.

---

## Complete Production Multi-Agent Example

A support desk: a cheap triage step, specialists with real tools, a confirmation hook on the one tool with side effects, and tracing.

```python
import os

from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    SimpleConsoleUI,
)

ORDERS = {"ORD-1": {"status": "delivered", "amount": 49.0}}


@tool
def get_order(order_id: Annotated[str, "Order ID"]) -> dict:
    """Look up an order."""
    return ORDERS.get(order_id, {"error": "not found"})


@tool
def issue_refund(order_id: Annotated[str, "Order ID"], amount: Annotated[float, "Amount in GBP"]) -> str:
    """Issue a refund. This moves money."""
    return f"Refunded £{amount:.2f} for {order_id}"


confirm_refunds = ConfirmationHook(confirmation_strategies={
    "issue_refund": BlockingConfirmationStrategy(
        confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
    )
})

billing_agent = make_agent(
    "You resolve billing issues. Check the order before refunding.",
    tools=[get_order, issue_refund],
    hooks={"before_tool": [confirm_refunds]},
    max_agent_steps=6,
)
tech_agent = make_agent("You resolve technical issues step by step.", tools=[web_search], max_agent_steps=6)

desk = make_agent(
    "You are the front desk. Hand billing issues to `billing` and technical issues to `tech`. "
    "Answer simple questions yourself.",
    tools=[
        AgentTool(agent=billing_agent, name="billing", description="Handles orders, charges and refunds."),
        AgentTool(agent=tech_agent, name="tech", description="Handles technical problems."),
    ],
    max_agent_steps=8,
)

# Tracing (pip install langfuse-haystack); instantiating the connector enables it globally
if os.getenv("LANGFUSE_SECRET_KEY"):
    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    from haystack_integrations.components.connectors.langfuse import LangfuseConnector
    LangfuseConnector("support-desk")

result = desk.run(messages=[ChatMessage.from_user("Order ORD-1 arrived broken, can I get my money back?")])
print(result["last_message"].text)
```

In a web service, replace `SimpleConsoleUI` with a custom `ConfirmationUI` that asks the user through your front end, and pass request-scoped objects with `desk.run(..., hook_context={...})`.

---

## Summary

- Start with one agent; split only when roles need different prompts or tools.
- **Coordinator + `AgentTool`** when the order of work is decided at run time.
- **Pipeline stages** when the order is fixed; wire `last_message` into the next `user_prompt`.
- **Routers** to send each request to exactly one specialist.
- **`asyncio.gather` over `run_async`** for independent, parallel work.
- **State** (`state_schema`, `outputs_to_state`, `inputs_from_state`) for structured data shared between tools and sub-agents.
- **Hooks** for guard-rails and approvals; **tracing** to see what each agent did.

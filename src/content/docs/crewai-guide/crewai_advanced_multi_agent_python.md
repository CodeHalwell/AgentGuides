---
title: "Multi-agent patterns"
description: "Delegation between agents, hierarchical managers, planning, parallel specialists, remote agents over A2A, human review, and resilience settings in CrewAI."
framework: "crewai"
language: "python"
sidebar:
  label: "Multi-agent patterns"
  order: 40
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%). The Python examples were run against that release with a stub LLM, except the A2A one, which was checked for construction only (it needs `crewai[a2a]` and a remote agent).

This page collects the ways CrewAI agents work together beyond a straight sequence of tasks. The fields themselves are documented on [Agent API](/crewai-guide/crewai_agent_api_python/) and [Task API](/crewai-guide/crewai_task_api_python/); here the focus is on when to use each.

| Pattern | Mechanism | Use when |
|---|---|---|
| Hand-offs | Sequential tasks, `context=[...]` | The steps are known in advance. |
| Delegation | `allow_delegation=True` | An agent should be able to ask a colleague mid-task. |
| Manager-led | `Process.hierarchical` + `manager_llm` / `manager_agent` | The order and assignment of work should be decided at run time. |
| Parallel specialists | `async_execution=True` + a synthesis task | Independent analyses that are merged at the end. |
| Remote agents | `a2a=A2AClientConfig(...)` | Part of the work lives in another service. |
| Human review | `Task(human_input=True)`, flow `@human_feedback` | A person must approve or correct output. |
| Explicit control | A Flow around several crews | You need branches, loops or persistence. |

## Hand-offs

```python
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="openai/%%model=openai-small%%")
researcher = Agent(role="Researcher", goal="Find and cite reliable sources", backstory="Methodical analyst.", llm=llm)
writer = Agent(role="Writer", goal="Draft concise copy from research", backstory="Plain-English editor.", llm=llm)

research = Task(
    description="Research {topic} and cite sources.",
    expected_output="Bullet list of findings with URLs.",
    agent=researcher,
)
draft = Task(
    description="Draft a concise summary from the research.",
    expected_output="Three-paragraph summary.",
    agent=writer,
    context=[research],  # explicit hand-off
)

crew = Crew(agents=[researcher, writer], tasks=[research, draft], process=Process.sequential)
result = crew.kickoff(inputs={"topic": "agent evaluation"})
```

Each task already sees every earlier task's output by default; `context=[...]` makes the dependency explicit and keeps unrelated outputs out of the prompt.

## Delegation

With `allow_delegation=True` an agent gets two extra tools, **Delegate work to coworker** and **Ask question to coworker**, whose targets are the other agents in the crew:

```python
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")

lead = Agent(
    role="Lead Analyst",
    goal="Deliver a sound investment memo on {company}",
    backstory="Portfolio manager who checks numbers with specialists before concluding.",
    allow_delegation=True,
    llm=llm,
)
accountant = Agent(
    role="Forensic Accountant",
    goal="Check financial statements for red flags",
    backstory="Former auditor.",
    llm=llm,
)
lawyer = Agent(
    role="Regulatory Counsel",
    goal="Identify regulatory exposure",
    backstory="Securities lawyer.",
    llm=llm,
)

memo = Task(
    description="Write an investment memo on {company}. Ask colleagues where you need specialist input.",
    expected_output="A one-page memo with a recommendation and the main risks.",
    agent=lead,
)

crew = Crew(agents=[lead, accountant, lawyer], tasks=[memo])
print(crew.kickoff(inputs={"company": "ACME Robotics"}).raw)
```

- Delegation is off by default. Turn it on only for the agent that coordinates; specialists that can also delegate tend to bounce work between each other.
- The coworker list is every other agent in the crew, so the `role` and `goal` strings are what the delegating agent reads when choosing whom to ask. Make them distinct.
- Each delegation is a full agent run and costs tokens; cap it with `max_iter` on the lead.

## Manager-led crews

A hierarchical crew has a manager that decides which agent handles each task, delegates, reviews results and asks for rework:

```python
from crewai import Agent, Crew, Process, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")
researcher = Agent(role="Researcher", goal="Gather facts", backstory="Analyst at a think tank.", llm=llm)
writer = Agent(role="Writer", goal="Write clear reports", backstory="Former newspaper editor.", llm=llm)

manager = Agent(
    role="Editor-in-Chief",
    goal="Ship an accurate, readable report on {topic}",
    backstory="Runs a small research desk; assigns work and checks it before it goes out.",
    allow_delegation=True,
    llm=LLM(model="openai/%%model=openai%%"),
)

crew = Crew(
    agents=[researcher, writer],        # the manager is not listed here
    tasks=[
        Task(description="Research the state of {topic}.", expected_output="Bulleted findings with sources."),
        Task(description="Write a one-page brief on {topic}.", expected_output="A one-page brief."),
    ],
    process=Process.hierarchical,
    manager_agent=manager,
)
print(crew.kickoff(inputs={"topic": "grid-scale storage"}).raw)
```

- Set exactly one of `manager_llm` (CrewAI builds a generic manager) or `manager_agent` (your own role, goal and backstory). Neither raises a `ValidationError`; listing the `manager_agent` in `agents` also does.
- Tasks may leave `agent` unset; the manager assigns them. Tasks that do name an agent can still be delegated elsewhere.
- Give the manager the strongest model in the crew: its decisions shape every other call.

## Planning

`Crew(planning=True)` runs a planning step before the tasks and adds the plan to each task's prompt; `planning_llm` picks the model. Per-agent planning (`Agent(planning=True, planning_config=PlanningConfig(...))`) makes one agent plan before acting on each of its tasks. Planning helps on long, multi-step tasks and is wasted on short ones.

## Parallel specialists

```python
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")
market = Agent(role="Market Analyst", goal="Size the market for {product}", backstory="Industry analyst.", llm=llm)
tech = Agent(role="Technology Analyst", goal="Assess the technology behind {product}", backstory="Former CTO.", llm=llm)
editor = Agent(role="Editor", goal="Merge analyses into one view", backstory="Report editor.", llm=llm)

market_view = Task(description="Market view of {product}.", expected_output="Five bullets.", agent=market, async_execution=True)
tech_view = Task(description="Technology view of {product}.", expected_output="Five bullets.", agent=tech, async_execution=True)
summary = Task(
    description="Merge both views into a single assessment.",
    expected_output="One page with a recommendation.",
    agent=editor,
    context=[market_view, tech_view],
)

crew = Crew(agents=[market, tech, editor], tasks=[market_view, tech_view, summary])
print(crew.kickoff(inputs={"product": "solid-state batteries"}).raw)
```

Consecutive async tasks run concurrently on threads; the next synchronous task waits for all of them. A crew may end with at most one async task (more is a `ValidationError`). For independent runs of a whole crew over many inputs, use `kickoff_for_each` / `akickoff_for_each` instead.

## Remote agents over A2A

An agent can delegate to an agent in another service through the Agent-to-Agent protocol (`pip install "crewai[a2a]"`):

```python
from crewai import Agent, LLM
from crewai.a2a.config import A2AClientConfig

coordinator = Agent(
    role="Support Coordinator",
    goal="Resolve customer issues, using the billing agent for anything about invoices",
    backstory="Tier-2 support lead.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    a2a=A2AClientConfig(
        endpoint="https://billing.example.com/.well-known/agent-card.json",
        timeout=60,
        max_turns=5,
    ),
)
```

- `endpoint` points at the remote agent card. `auth`, `timeout`, `max_turns`, `response_model` and `fail_fast` are the main client options.
- `a2a=` also takes a list (several remote agents) and an `A2AServerConfig`, which describes the agent card this agent advertises when it is itself exposed over A2A.

## Human review

- **Task level:** `Task(human_input=True)` pauses after the task and asks on the console for approval or corrections, which are fed back to the agent. It suits local runs and demos.
- **Flow level:** `@human_feedback` routes on the reviewer's answer and, with a custom provider, pauses the flow until someone replies from Slack, a ticket or a web form, then resumes it from persisted state. See [Flows → Human feedback](/crewai-guide/crewai_flows_guide/#human-feedback--human_feedback).
- **Tool level:** a `before_tool_call` hook can refuse a dangerous tool call unless it has been approved. See [Production → Hooks](/crewai-guide/crewai_production_guide/#hooks-block-or-rewrite-calls).

## Resilience

- Retries on transient failures: `Agent(max_retry_limit=...)`; `guardrail` + `guardrail_max_retries` on tasks re-run output that fails validation.
- Timeouts: `Agent(max_execution_time=...)` caps wall-clock seconds per task; `LLM(timeout=...)` caps each model call.
- Tool failures: `tool_failure_policy="raise"` on an agent, task or crew aborts instead of carrying on with a failed tool result.
- Side effects: keep compensating actions in your own tool code, and use checkpoints (`Crew(checkpoint=...)`) to resume rather than re-run completed tasks.

## Observability

Delegation and manager decisions show up as ordinary tool calls (`ToolUsageStartedEvent` / `ToolUsageFinishedEvent` with the delegation tool's name) and agent runs (`AgentExecutionStartedEvent`) on the event bus, so the listeners on [Production → Event listeners](/crewai-guide/crewai_production_guide/#event-listeners) capture who did what.

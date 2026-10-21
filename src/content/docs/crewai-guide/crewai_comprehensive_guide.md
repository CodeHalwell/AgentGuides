---
title: "CrewAI comprehensive guide"
description: "A single-page tour of CrewAI — install, models, agents, tasks, crews and processes, running and reading results, project layout, tools and MCP, memory, knowledge, flows and checkpoints."
framework: "crewai"
language: "python"
sidebar:
  label: "Comprehensive guide"
  order: 10
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%). The Python examples were run against that release with a stub LLM. Each section links to the reference page that covers its topic in full.

CrewAI builds multi-agent applications from four pieces:

- an **Agent** is an LLM with a role, a goal, a backstory, and optionally tools, memory and knowledge;
- a **Task** is one piece of work with a description and an expected output, assigned to an agent;
- a **Crew** runs a list of tasks with its agents, either in order (sequential) or under a manager agent (hierarchical);
- a **Flow** is an event-driven Python class that wires steps together, holds state, and can call crews or agents from any step.

Crews are the quickest way to get a team of agents working on a problem. Flows are for the control around them: branching, loops, plain code between LLM calls, persistence and human review.

## Core fundamentals

### Install

CrewAI supports Python 3.10 to 3.13.

```bash
pip install "crewai[tools]"        # crewai, the CLI and the crewai-tools package
# or, as a uv project
uv add "crewai[tools]"
```

Provider SDKs other than OpenAI are optional extras, as is LiteLLM (used for providers without a native integration):

```bash
pip install "crewai[anthropic]"    # also: google-genai, bedrock, azure-ai-inference, litellm
```

The `crewai` command comes from the `crewai-cli` distribution, which is installed with `crewai`. `crewai create crew <name>` scaffolds a project; see [Project layout](#project-layout) and the [CLI reference](/crewai-guide/crewai_cli_python/).

### Versions

This guide tracks `crewai` %%version=crewai%%. When pinning an older release, note that **1.10.0, 1.12.0 and 1.14.0 were yanked from PyPI** (1.14.0 for a broken Anthropic dependency); use 1.10.1, 1.12.1 or 1.14.1 and later instead. The [overview](/crewai-guide/#whats-new-in-the-115-line-junesoctober-2026) lists the changes in the 1.14 and 1.15 lines.

### Models

`LLM` takes a `provider/model` string and routes it to the provider's own SDK where CrewAI has a native integration:

```python
from crewai import LLM

openai_llm = LLM(model="openai/%%model=openai-small%%", temperature=0.2)
claude = LLM(model="anthropic/%%model=anthropic%%")             # needs crewai[anthropic]
gemini = LLM(model="gemini/%%model=gemini%%")                   # needs crewai[google-genai]
local = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
```

API keys come from the usual environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …). An agent without `llm=` reads the `MODEL` environment variable and otherwise uses CrewAI's built-in default (`gpt-4.1-mini` in 1.15.28), so set the model explicitly in anything you deploy. Routing rules, reasoning settings and custom providers are on [LLM providers](/crewai-guide/crewai_llm_providers_python/).

## Simple agents

### A first crew

```python
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")

analyst = Agent(
    role="Research Analyst",
    goal="Explain {topic} accurately for a general audience",
    backstory="Science journalist who checks every claim against a primary source.",
    llm=llm,
)

explainer = Task(
    description="Write a short explainer on {topic}.",
    expected_output="Three paragraphs in plain English, no jargon.",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[explainer])
result = crew.kickoff(inputs={"topic": "how tides work"})
print(result.raw)
```

- `role`, `goal` and `backstory` are required on every `Agent`; `description` and `expected_output` are required on every `Task`. Leaving one out raises a `ValidationError`.
- `{topic}` placeholders in agents and tasks are filled from `kickoff(inputs=...)`.
- The backstory shapes tone and judgement. A sentence or two that says who the agent is and what it cares about is enough; long persona essays rarely help.
- `verbose=True` on the agent or crew prints each step while you develop.

The full list of agent fields (limits, planning, guardrails, knowledge, MCP, A2A, checkpoints) is on the [Agent API](/crewai-guide/crewai_agent_api_python/) page.

### One agent without a crew

For a single question, `Agent.kickoff()` skips the crew and task:

```python
from crewai import Agent, LLM

analyst = Agent(
    role="Analyst",
    goal="Answer questions about unit economics",
    backstory="Finance analyst at a SaaS company.",
    llm=LLM(model="openai/%%model=openai-small%%"),
)
out = analyst.kickoff("What is a reasonable LTV:CAC ratio for B2B SaaS?")
print(out.raw)
```

It accepts `response_format=SomeModel` for structured output and works inside flow methods.

## Tasks

### Chaining and context

```python
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")
researcher = Agent(role="Researcher", goal="Find facts about {topic}", backstory="Careful analyst.", llm=llm)
writer = Agent(role="Writer", goal="Explain {topic} plainly", backstory="Science journalist.", llm=llm)
editor = Agent(role="Editor", goal="Tighten copy", backstory="Sub-editor.", llm=llm)

facts = Task(description="Collect five facts about {topic}.", expected_output="Five bullets with sources.", agent=researcher)
draft = Task(description="Write a short article on {topic}.", expected_output="Three paragraphs.", agent=writer)
polish = Task(
    description="Edit the article for clarity.",
    expected_output="The edited article.",
    agent=editor,
    context=[draft],            # only the draft, not the raw facts
)

print(Crew(agents=[researcher, writer, editor], tasks=[facts, draft, polish]).kickoff(inputs={"topic": "tides"}).raw)
```

When `context` is unset, a task sees the outputs of **all** earlier tasks; `context=[...]` narrows it, and `context=[]` gives the task none. Tasks with `async_execution=True` run concurrently until the next synchronous task, which waits for them.

### Structured output

```python
from pydantic import BaseModel
from crewai import Agent, Crew, Task, LLM


class Brief(BaseModel):
    headline: str
    bullets: list[str]


writer = Agent(
    role="Copywriter",
    goal="Write product briefs",
    backstory="Senior copywriter at a B2B agency.",
    llm=LLM(model="openai/%%model=openai-small%%"),
)
brief = Task(
    description="Write a brief for {product}.",
    expected_output="A headline and three bullets.",
    agent=writer,
    output_pydantic=Brief,
)

result = Crew(agents=[writer], tasks=[brief]).kickoff(inputs={"product": "a USB-C dock"})
print(result.pydantic.headline)      # final task's model
print(result.tasks_output[0].raw)    # raw text of any task
```

`output_pydantic` asks for JSON in the prompt and parses it; `response_model` uses the provider's native structured output; `output_json` keeps a dict in `json_dict`. A `CrewOutput` is not a string or a dict: read `.raw`, `.pydantic` or `.json_dict` (or `json.loads(result.raw)` when you asked for JSON). Guardrails, conditional tasks, file inputs and `human_input` are covered on the [Task API](/crewai-guide/crewai_task_api_python/) page.

## Crews and processes

`Process` has two values: `sequential` (the default) and `hierarchical`.

```python
from crewai import Agent, Crew, Process, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")
researcher = Agent(role="Researcher", goal="Gather facts", backstory="Analyst at a think tank.", llm=llm)
writer = Agent(role="Writer", goal="Write clear reports", backstory="Former newspaper editor.", llm=llm)

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        Task(description="Research the state of {topic}.", expected_output="Bulleted findings."),
        Task(description="Write a one-page brief on {topic}.", expected_output="A one-page brief."),
    ],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/%%model=openai%%"),   # or manager_agent=Agent(...)
    planning=True,                                       # optional: plan the steps first
    planning_llm=llm,
)
print(crew.kickoff(inputs={"topic": "heat pumps"}).raw)
```

- A hierarchical crew **requires** `manager_llm` or `manager_agent`; without either, `Crew(...)` raises `ValidationError: manager_llm or manager_agent is required`. A `manager_agent` must not also appear in `agents`.
- Tasks in a hierarchical crew may omit `agent`; the manager assigns and delegates them.
- `planning=True` adds a planning step before the tasks run; `planning_llm` chooses its model.
- Delegation between agents (`allow_delegation=True`), manager design and A2A are on [Multi-agent patterns](/crewai-guide/crewai_advanced_multi_agent_python/).

Useful `Crew` fields beyond these: `memory`, `knowledge_sources`, `embedder`, `max_rpm`, `cache`, `tool_failure_policy`, `checkpoint`, `stream`, `tracing`, `step_callback`, `task_callback`, `before_kickoff_callbacks` and `after_kickoff_callbacks`. There is no `share_crew_state` field, and passing an unknown keyword raises a `ValidationError`.

## Running a crew

```python
import asyncio
from crewai import Agent, Crew, Process, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")

researcher = Agent(role="Researcher", goal="Find facts about {topic}", backstory="Careful analyst.", llm=llm)
writer = Agent(role="Writer", goal="Explain {topic} plainly", backstory="Science journalist.", llm=llm)

research = Task(description="Collect five facts about {topic}.", expected_output="Five bullets.", agent=researcher)
article = Task(description="Write a short explainer on {topic}.", expected_output="Three paragraphs.", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research, article], process=Process.sequential)

# One run
result = crew.kickoff(inputs={"topic": "tides"})
print(result.raw[:80])

# The same crew over several inputs, one after another
results = crew.kickoff_for_each(inputs=[{"topic": "tides"}, {"topic": "auroras"}])
print(len(results))

# Native async (use inside FastAPI handlers, async flows, etc.)
async def main():
    return await crew.akickoff(inputs={"topic": "eclipses"})

print(asyncio.run(main()).raw[:80])
```

`kickoff_async`, `kickoff_for_each_async` and `akickoff_for_each` cover the remaining combinations. The result is a `CrewOutput`:

| Attribute | Contents |
|---|---|
| `.raw` | Final task's text output. |
| `.pydantic` / `.json_dict` | Final task's structured output, if it had one. |
| `.tasks_output` | One `TaskOutput` per task, in execution order. |
| `.token_usage` | `UsageMetrics`: prompt, completion, cached, reasoning and cache-creation tokens, request count. |

## Project layout

`crewai create crew <name>` starts an interactive wizard and writes a JSON-first project (`crew.jsonc` plus `agents/*.jsonc`). `crewai create crew <name> --classic` writes the Python/YAML layout used in most CrewAI examples: agents and tasks in `config/agents.yaml` and `config/tasks.yaml`, wired up by a `@CrewBase` class in `crew.py`. `crewai run` runs either. The layout, the YAML format and every CLI command are on the [CLI](/crewai-guide/crewai_cli_python/) page.

## Tools and MCP

Give agents tools with the `@tool` decorator, a `BaseTool` subclass, the ready-made tools in `crewai_tools`, or whole MCP servers:

```python
from crewai import Agent
from crewai.mcp import MCPServerStdio
from crewai.tools import tool


@tool("order_status")
def order_status(order_id: str) -> str:
    """Look up the shipping status of an order."""
    return f"order {order_id}: shipped"


support = Agent(
    role="Support",
    goal="Answer order questions accurately",
    backstory="Level-2 support agent.",
    tools=[order_status],
    mcps=[MCPServerStdio(command="uvx", args=["mcp-server-fetch"])],   # every tool the server lists
)
```

MCP support is built in (`crewai.mcp`): stdio, HTTP and SSE servers, tool filters, and `"server-url#tool"` references. Some `crewai_tools` tools need extra packages (`GithubSearchTool` needs `PyGithub`, for example). See [Tools](/crewai-guide/crewai_tools_python/).

## Memory and knowledge

- **Memory** is one `Memory` class with scoped views. `Crew(memory=True)` gives the crew a default store (LanceDB, OpenAI for analysis and embeddings); pass `memory=Memory(...)` to choose the storage, embedder and scope. The older short-term, long-term and entity memories were folded into it. See [Memory](/crewai-guide/crewai_memory_python/).
- **Knowledge** is a read-mostly vector store built from files or strings and attached with `knowledge_sources=[...]` on an agent or crew. See [Knowledge](/crewai-guide/crewai_knowledge_python/).

Both need an embedder; without one configured they use OpenAI and need `OPENAI_API_KEY`.

## Flows

```python
from pydantic import BaseModel
from crewai.flow import Flow, start, listen


class State(BaseModel):
    topic: str = ""
    outline: str = ""


class OutlineFlow(Flow[State]):
    @start()
    def make_outline(self):
        self.state.outline = f"1. What {self.state.topic} is\n2. Why it matters"

    @listen(make_outline)
    def finish(self):
        return self.state.outline


print(OutlineFlow().kickoff(inputs={"topic": "tides"}))
```

Routers, `and_`/`or_`, crews inside flows, `@persist`, human feedback, conversational and declarative flows are on the [Flows](/crewai-guide/crewai_flows_guide/) page.

## Checkpoints

`Crew(checkpoint=CheckpointConfig(...))` writes a snapshot when chosen events fire (by default each completed task), and `kickoff(from_checkpoint=CheckpointConfig(restore_from=...))` resumes a run. `crewai checkpoint list|info|resume|diff|prune` manages the files. See [Production → Recovery with checkpoints](/crewai-guide/crewai_production_guide/#recovery-with-checkpoints).

## CrewAI AMP

CrewAI AMP is CrewAI's hosted platform for deploying, tracing and evaluating crews and flows. You reach it through the CLI (`crewai login`, `crewai deploy ...`, `crewai traces ...`, `crewai eval`) and through `Agent(apps=[...])` for its connected app integrations; there is no `crewai.amp` Python module. See [Production → Deploying](/crewai-guide/crewai_production_guide/#deploying).

## Where to go next

| Topic | Page |
|---|---|
| Every agent field | [Agent API](/crewai-guide/crewai_agent_api_python/) |
| Tasks, structured output, guardrails | [Task API](/crewai-guide/crewai_task_api_python/) |
| Delegation, managers, A2A, human review | [Multi-agent patterns](/crewai-guide/crewai_advanced_multi_agent_python/) |
| Event-driven workflows | [Flows](/crewai-guide/crewai_flows_guide/) |
| Worked examples | [Recipes](/crewai-guide/crewai_recipes/) |
| Observability, testing, deployment | [Production](/crewai-guide/crewai_production_guide/) |
| Diagrams of the moving parts | [Diagrams](/crewai-guide/crewai_diagrams/) |

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-21 | 1.15.28 | Rewritten against 1.15.28: required `backstory`/`expected_output` everywhere, hierarchical example with a manager, `CrewOutput` attributes, `kickoff_for_each`/`akickoff`, `Agent.kickoff`, MCP via `mcps=`, yanked releases. Removed the invented "Custom Process" and `share_crew_state`, the MCP stub, the duplicated Flows chapter (now on the Flows page) and the UV and backstory padding. |
| 2026-05-28 | 1.14.6 | Patch release. |
| 2026-04-17 | 1.14.2 | Checkpoint forking; `from_checkpoint` on `kickoff`; checkpoint CLI; reasoning and cache-creation token tracking. |
| 2026-04-08 | 1.14.1 | First usable 1.14 release (1.14.0, released 2026-04-07, was yanked): checkpoint system, `CodeInterpreterTool` removed, native OpenAI-compatible providers, tool hooks. |
| 2026-04-02 | 1.13.0 | `Flow` and `LLM` became Pydantic models. |

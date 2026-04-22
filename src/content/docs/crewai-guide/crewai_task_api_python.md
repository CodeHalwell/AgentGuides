---
title: "Task API and process types"
description: "Every field on crewai.Task, ConditionalTask, and TaskOutput — structured outputs, guardrails, async, context, file inputs, and the sequential vs hierarchical process."
framework: crewai
language: python
sidebar:
  label: "Task API"
  order: 21
---

> **Verified against crewai==1.14.3a2** (source: `crewai/task.py`, `crewai/tasks/conditional_task.py`, `crewai/tasks/task_output.py`, `crewai/process.py`).

Tasks are the units of work a crew executes. Their outputs feed forward into `{variables}` available to later tasks. This page covers the real fields, the two process types, and the surprisingly subtle rules around structured outputs.

## Minimal runnable example

```python
from crewai import Agent, Crew, Task, Process, LLM
from pydantic import BaseModel

class Brief(BaseModel):
    headline: str
    bullets: list[str]

writer = Agent(
    role="Writer",
    goal="Draft a marketing brief",
    backstory="Senior copywriter.",
    llm=LLM(model="openai/gpt-4o-mini"),
)

draft = Task(
    description="Write a brief for {product}.",
    expected_output="A Brief with a punchy headline and 3 bullets.",
    agent=writer,
    output_pydantic=Brief,
)

out = Crew(agents=[writer], tasks=[draft], process=Process.sequential).kickoff(
    inputs={"product": "NoSQL analytics warehouse"}
)

print(out.tasks_output[0].pydantic.headline)
```

## Core fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `description` | `str` | required | The instruction sent to the agent. `{vars}` pulled from crew `inputs` and prior task outputs. |
| `expected_output` | `str` | required | Plain-language description of what "done" looks like. |
| `agent` | `Agent \| None` | `None` | Executor. Hierarchical crews can leave this `None` — the manager assigns. |
| `async_execution` | `bool` | `False` | Run in a thread. Several async tasks in a row fan out in parallel. |
| `context` | `list[Task] \| None` | *auto* | Explicit upstream tasks. Defaults to the prior task's output in sequential crews. |
| `tools` | `list[BaseTool] \| None` | `[]` | Restrict the tools available **for this task only**. |
| `callback` | `Callable \| None` | `None` | Called with the `TaskOutput` on completion. |
| `output_file` | `str \| None` | `None` | File path (template vars allowed) for raw output. |
| `create_directory` | `bool \| None` | `True` | Create the parent dir of `output_file` if missing. |
| `markdown` | `bool` | `False` | Add a Markdown format instruction to the prompt. |
| `human_input` | `bool` | `False` | Pause after this task and ask the user to accept/modify. |
| `guardrail` | `str \| Callable` | `None` | Single validator on `TaskOutput`. |
| `guardrails` | `list[...]` | `None` | Multiple validators; also accepts a single callable. |
| `guardrail_max_retries` | `int` | `3` | Retries across all guardrails. |
| `name` | `str \| None` | `None` | Friendly name used in logs and events. |

Full structured-output fields are covered below — they interact with each other in ways the docstrings don't spell out.

## Structured output: three fields, one answer

| Field | Effect | Output lives in |
|---|---|---|
| `output_pydantic=ModelCls` | Prompt-level: the agent is asked to emit JSON; the framework parses into `ModelCls`. | `TaskOutput.pydantic` |
| `output_json=ModelCls` | Same prompt trick but leaves the dict raw, without instantiating the model. | `TaskOutput.json_dict` |
| `response_model=ModelCls` | Uses the **LLM provider's native structured outputs** (OpenAI `response_format`, etc.) — no prompt hack. | `TaskOutput.pydantic` |

Pick **one**. Set `response_model` when the provider supports it (OpenAI, Gemini 2 flash, Anthropic tool-use mode) — the JSON is guaranteed by the provider. Fall back to `output_pydantic` otherwise.

```python
from pydantic import BaseModel, Field

class Report(BaseModel):
    score: int = Field(ge=0, le=100)
    reasons: list[str]

task = Task(
    description="Score the article on {topic}.",
    expected_output="Report with integer score and 3 reasons.",
    agent=analyst,
    response_model=Report,  # provider-enforced
)
```

Access after kickoff:

```python
out = crew.kickoff(inputs={"topic": "..."})
print(out.tasks_output[-1].pydantic.score)
```

## Guardrails

Single or multiple validators run after the task finishes; a failing guardrail feeds the reason back to the agent for another attempt up to `guardrail_max_retries`.

```python
from crewai.tasks.task_output import TaskOutput

def min_length(output: TaskOutput) -> tuple[bool, str]:
    if len(output.raw) < 200:
        return False, "Output under 200 chars — please expand."
    return True, output.raw

def no_dollars(output: TaskOutput) -> tuple[bool, str]:
    if "$" in output.raw:
        return False, "Strip currency symbols, they upset our CMS."
    return True, output.raw

task = Task(
    description="Summarise earnings.",
    expected_output="Plain-text summary, 200+ chars, no $ symbols.",
    agent=writer,
    guardrails=[min_length, no_dollars],
    guardrail_max_retries=2,
)
```

String guardrails let the LLM itself judge the output: `guardrail="Output must not mention competitors by name."`

## Context and task chaining

```python
research = Task(description="Find 5 sources on {topic}.", expected_output="Bulleted list of links.", agent=researcher)
outline  = Task(description="Outline an article from the sources.", expected_output="Section headers.", agent=editor, context=[research])
draft    = Task(description="Write the full article.", expected_output="Markdown article.", agent=writer, context=[outline])
```

- In `Process.sequential`, **context defaults to the previous task** — you only need `context=` when you want to skip or merge tasks.
- `context=[]` (empty list) explicitly isolates a task from previous outputs.

## Async execution and parallel fan-out

```python
t1 = Task(description="Research US market", ..., async_execution=True)
t2 = Task(description="Research EU market", ..., async_execution=True)
t3 = Task(description="Research APAC market", ..., async_execution=True)
summary = Task(description="Synthesise the three regional reports",
               expected_output="Exec summary",
               agent=analyst,
               context=[t1, t2, t3])
```

`t1`/`t2`/`t3` run on threads concurrently; `summary` blocks on all three. If you forget the `context=[...]`, the sequential process keeps them isolated and `summary` sees nothing.

## File inputs

```python
from crewai.tasks.task_output import TaskOutput

task = Task(
    description="Extract tables from {doc}.",
    expected_output="Markdown tables.",
    agent=analyst,
    input_files={"doc": "/path/to/filing.pdf"},
)
```

`input_files` is a dict of reference names to paths (or `File` objects). They're passed through to the LLM using whichever mechanism the provider supports (OpenAI file upload, Anthropic files API, etc.). This replaces the deprecated `Agent.multimodal=True` flag.

## Conditional tasks

```python
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

def needs_fact_check(prev: TaskOutput) -> bool:
    return "citation needed" in prev.raw.lower()

draft = Task(description="Write the article", expected_output="Article", agent=writer)

fact_check = ConditionalTask(
    condition=needs_fact_check,
    description="Fact-check the article.",
    expected_output="Citations list.",
    agent=checker,
)

crew = Crew(agents=[writer, checker], tasks=[draft, fact_check])
```

Constraints from the source:
- A `ConditionalTask` **cannot** be the first or only task — it reads the prior task's `TaskOutput`.
- When the condition returns `False`, the task yields an empty `TaskOutput` (`raw=""`).

## Process types

`crewai.process.Process` is the enum:

```python
class Process(str, Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"
```

`consensual` is commented out in source — **don't use it**; it doesn't exist.

### Sequential

```python
Crew(agents=[a, b, c], tasks=[t1, t2, t3], process=Process.sequential)
```

- Tasks run in list order.
- Output of `t_{n}` is implicit context for `t_{n+1}` (override with `context=[...]`).
- Async tasks at the same "level" run concurrently until a synchronous task is reached.

### Hierarchical — manager coordinates

```python
from crewai import Crew, Process, Agent, LLM

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[write_brief, write_article, polish],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o"),   # manager built by CrewAI
    # OR:
    # manager_agent=Agent(role="Manager", goal="...", backstory="...", allow_delegation=True),
)
```

Rules:
- You must set **exactly one** of `manager_llm` or `manager_agent`.
- Tasks in a hierarchical crew don't need an `agent=` — the manager routes them (agent-less tasks go into the pool).
- The manager can delegate to any worker, ask clarifying questions, and synthesise answers.

## CrewOutput — what `kickoff()` returns

`CrewOutput` exposes:

- `.raw` — the final task's raw string output
- `.tasks_output` — list of `TaskOutput`, in execution order
- `.token_usage` — `UsageMetrics` (total / prompt / completion / reasoning / cache-creation tokens)
- `.json_dict` / `.pydantic` — shortcut to the final task's structured output if set
- `.to_dict()` — dict of all outputs keyed by task name

```python
out = crew.kickoff(inputs={...})

for t in out.tasks_output:
    print(t.name or t.summary, "->", len(t.raw), "chars")

print("total tokens:", out.token_usage.total_tokens)
```

## Patterns

### 1. Markdown-clean final deliverable

```python
final = Task(
    description="Assemble the report.",
    expected_output="Well-formed Markdown, with a top-level H1 and 3 H2 sections.",
    agent=editor,
    markdown=True,
    output_file="reports/{topic}.md",
)
```

`{topic}` in the file path interpolates from crew inputs.

### 2. Retry-on-failure with strict JSON

```python
class Plan(BaseModel):
    steps: list[str]

Task(
    description="Plan the campaign.",
    expected_output="Plan with step list.",
    agent=planner,
    response_model=Plan,          # native structured outputs
    guardrail=lambda o: (len(o.pydantic.steps) >= 3, "need ≥3 steps"),
    guardrail_max_retries=3,
)
```

### 3. Human-in-the-loop gate

```python
Task(
    description="Produce the press-release draft.",
    expected_output="Approved draft.",
    agent=writer,
    human_input=True,    # blocks for stdin approval / edit before moving on
)
```

### 4. File-fed comparison

```python
Task(
    description="Diff the two contracts in {a} and {b}; list the risky clauses.",
    expected_output="Ranked list with clause numbers.",
    agent=legal,
    input_files={"a": "contracts/v1.pdf", "b": "contracts/v2.pdf"},
)
```

### 5. Skip a task when upstream produced nothing

```python
cond = ConditionalTask(
    condition=lambda prev: bool(prev.raw.strip()),
    description="Summarise the research.",
    expected_output="Summary.",
    agent=summariser,
)
```

## Gotchas

- **`output_pydantic` vs `response_model`** — the first is prompt-driven and can still parse-fail; the second uses provider-native structured outputs and is enforced.
- **`max_retries` is deprecated on `Task`** — use `guardrail_max_retries`. The source docstring says "will be removed in v1.0.0" (written before the 1.x line, so effectively: a future major). Treat it as already-deprecated and avoid it in new code.
- **Sequential crews don't let you wait for a specific task** — use `context=[t1, t2]` to pin ordering.
- **Hierarchical tasks with `agent=` still obey delegation** — the manager can route them elsewhere.
- **`input_files` supersedes `Agent.multimodal=True`.** That flag is deprecated and is removed in 2.0.
- **`Process.consensual` doesn't exist.** It's commented out in `process.py`.

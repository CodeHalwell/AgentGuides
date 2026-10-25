---
title: "Task API and process types"
description: "Every field on crewai.Task, ConditionalTask, and TaskOutput — structured outputs, guardrails, async, context, file inputs, and the sequential vs hierarchical process."
framework: "crewai"
language: "python"
sidebar:
  label: "Task API"
  order: 21
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/task.py`, `crewai/tasks/conditional_task.py`, `crewai/tasks/task_output.py`, `crewai/process.py`).

Tasks are the units of work a crew executes. A task's output reaches later tasks as **context** appended to their prompt; it is not substituted into `{variables}`, which are filled only from the `inputs` passed to `kickoff()`. This page covers the fields, the two process types, and the surprisingly subtle rules around structured outputs and guardrails.

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
    llm=LLM(model="openai/%%model=openai-small%%"),
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
| `agent` | `Agent \| None` | `None` | Executor in a sequential crew. Hierarchical crews can leave this `None`; there the manager runs every task and decides whom to delegate to. |
| `async_execution` | `bool` | `False` | Run in a thread. Several async tasks in a row fan out in parallel. |
| `context` | `list[Task] \| None` | *auto* | Explicit upstream tasks. When unset, the task sees the outputs of **all** earlier tasks. |
| `tools` | `list[BaseTool] \| None` | `[]` | Restrict the tools available **for this task only**. |
| `tool_failure_policy` | `"ignore" \| "warn" \| "raise" \| None` | `None` | Overrides the agent's policy for tools that run but report failure, for this task only. |
| `callback` | `Callable \| None` | `None` | Called with the `TaskOutput` on completion. |
| `output_file` | `str \| None` | `None` | File path (template vars allowed) for raw output. |
| `create_directory` | `bool \| None` | `True` | Create the parent dir of `output_file` if missing. |
| `markdown` | `bool` | `False` | Add a Markdown format instruction to the prompt. |
| `human_input` | `bool` | `False` | Pause after this task and ask the user to accept/modify. |
| `guardrail` | `str \| Callable` | `None` | Single validator on `TaskOutput`. A string is judged by the agent's LLM. |
| `guardrails` | `list[...]` | `None` | Several validators, run in order, each seeing the previous one's result. If both `guardrail` and `guardrails` are set, only `guardrails` runs. |
| `guardrail_max_retries` | `int` | `3` | Retries across all guardrails. |
| `name` | `str \| None` | `None` | Friendly name used in logs and events. |
| `input_files` | `dict[str, File]` | `{}` | Named files attached to the task. Needs `crewai[file-processing]`; see [File inputs](#file-inputs). |
| `converter_cls` | `type[Converter] \| None` | `None` | Custom converter used to parse `output_pydantic` / `output_json` results. |

Full structured-output fields are covered below — they interact with each other in ways the docstrings don't spell out.

## Structured output: three fields, one answer

| Field | What the agent is told | Where the parsed result lives |
|---|---|---|
| `output_pydantic=ModelCls` | The model's JSON schema is added to the prompt, and the model is also passed to the LLM as its `response_model` (native structured output where the provider supports it). The final text is parsed into `ModelCls`. | `TaskOutput.pydantic` (and `CrewOutput.pydantic` for the last task) |
| `output_json=ModelCls` | Same as above, but the result is kept as a dict. | `TaskOutput.json_dict` |
| `response_model=ModelCls` | The model is passed to the LLM as its `response_model` only. | Nothing is parsed. `TaskOutput.raw` holds the JSON string; `.pydantic` stays `None`. |

Use `output_pydantic` when you want a model instance back. In 1.15.28 `response_model` on its own does not populate `.pydantic`, even when the provider returns a validated model: we checked this with a stub that returned a `BaseModel` instance from `call()`, and `raw` held the JSON while `pydantic` and `json_dict` were empty. If you want provider-enforced JSON and a parsed result, set `output_pydantic`; to parse `response_model` output yourself, use `Brief.model_validate_json(result.raw)`.

```python
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, LLM


class Report(BaseModel):
    score: int = Field(ge=0, le=100)
    reasons: list[str]


analyst = Agent(
    role="Analyst",
    goal="Score articles for credibility",
    backstory="Fact-checking editor.",
    llm=LLM(model="openai/%%model=openai-small%%"),
)
task = Task(
    description="Score the article on {topic}.",
    expected_output="Report with integer score and 3 reasons.",
    agent=analyst,
    output_pydantic=Report,
)
crew = Crew(agents=[analyst], tasks=[task])
out = crew.kickoff(inputs={"topic": "remote work"})
print(out.tasks_output[-1].pydantic.score)   # a Report instance
print(out.to_dict())                          # {'score': ..., 'reasons': [...]}
```

`CrewOutput.to_dict()` returns the final task's `json_dict`, or its `pydantic` result dumped to a dict, and an empty dict when the final task had no structured output (so it is empty with `response_model` alone).

## Guardrails

Single or multiple validators run after the task finishes. A failing guardrail sends the reason back to the agent for another attempt, up to `guardrail_max_retries`.

```python
from crewai import Agent, Crew, Task, LLM
from crewai.tasks.task_output import TaskOutput


def min_length(output: TaskOutput) -> tuple[bool, str]:
    if len(output.raw) < 200:
        return False, "Output under 200 chars. Please expand."
    return True, output.raw


def no_dollars(output: TaskOutput) -> tuple[bool, str]:
    if "$" in output.raw:
        return False, "Strip currency symbols, they upset our CMS."
    return True, output.raw


writer = Agent(
    role="Writer",
    goal="Summarise earnings clearly",
    backstory="Financial copy editor.",
    llm=LLM(model="openai/%%model=openai-small%%"),
)
task = Task(
    description="Summarise earnings.",
    expected_output="Plain-text summary, 200+ chars, no $ symbols.",
    agent=writer,
    guardrails=[min_length, no_dollars],
    guardrail_max_retries=2,
)
print(Crew(agents=[writer], tasks=[task]).kickoff().raw)
```

How it behaves (each point checked by running it):

- **Attempts.** `guardrail_max_retries=2` allows three attempts in total: the first, then two retries. If the last attempt still fails, the task raises `Exception("Task failed guardrail validation after 2 retries. Last error: ...")`.
- **What the agent sees on a retry.** The task prompt gains a block reading "Previous attempt failed validation: <your reason>" followed by the previous result, so make the reason specific.
- **Chaining.** With `guardrails=[a, b]`, `b` receives the output `a` returned. What a guardrail returns on success becomes the task output, so return `output.raw` unless you mean to rewrite it. Returning `(True, "ok")` replaces the real answer with `"ok"`.
- **String guardrails.** `guardrail="Output must not mention competitors by name."` makes a separate "Guardrail Agent" call on the agent's LLM for every attempt. That is an extra model call per check; use a function when a code check will do.

## Context and task chaining

```python
research = Task(description="Find 5 sources on {topic}.", expected_output="Bulleted list of links.", agent=researcher)
outline  = Task(description="Outline an article from the sources.", expected_output="Section headers.", agent=editor, context=[research])
draft    = Task(description="Write the full article.", expected_output="Markdown article.", agent=writer, context=[outline])
```

- When `context` is unset, a task receives the outputs of **all** earlier tasks in the crew. Use `context=[...]` to narrow it to specific ones.
- `context=[]` (empty list) explicitly isolates a task from previous outputs.
- Context is appended to the prompt as "This is the context you're working with:" followed by the raw text of each output. Nothing is substituted into `{variables}`: a `{research}` placeholder in a later task's description is left as the literal text `{research}`.

## Async execution and parallel fan-out

```python
t1 = Task(description="Research the US market", expected_output="Five bullets.", agent=analyst, async_execution=True)
t2 = Task(description="Research the EU market", expected_output="Five bullets.", agent=analyst, async_execution=True)
t3 = Task(description="Research the APAC market", expected_output="Five bullets.", agent=analyst, async_execution=True)
summary = Task(description="Synthesise the three regional reports",
               expected_output="Exec summary",
               agent=analyst,
               context=[t1, t2, t3])
```

`t1`/`t2`/`t3` run on threads concurrently; `summary`, a synchronous task, waits for all three before it starts. Without `context=[...]` it would still see all three outputs (the default is every earlier task); the explicit list documents the dependency and keeps any other earlier outputs out.

## File inputs

File inputs need the optional `crewai-files` package: `pip install "crewai[file-processing]"`. Without it, `Task(input_files=...)` is accepted and silently ignored.

```python
from crewai import Agent, Crew, Task, LLM
from crewai_files import File

analyst = Agent(
    role="Analyst",
    goal="Extract data from filings",
    backstory="Financial analyst.",
    llm=LLM(model="openai/%%model=openai-small%%"),
)
task = Task(
    description="Extract the tables from the attached filing.",
    expected_output="Markdown tables.",
    agent=analyst,
    input_files={"filing": File(source="/path/to/filing.pdf")},
)
```

Wrap each path in a `File` (or `PDFFile`, `ImageFile`, `TextFile`, ...). A plain string or `Path` as the dict value fails validation once `crewai-files` is installed. The prompt gains an "Available input files" list naming each file by its dict key, and the files are attached to the message for the provider integration to format (inline or upload, depending on the provider). The key is not a template variable: writing `{filing}` in the description leaves the literal text in place, so refer to the file by name in plain words. `crew.kickoff(input_files=...)` and `Agent.kickoff(input_files=...)` take the same dict. This replaces the deprecated `Agent.multimodal=True` flag.

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

Rules, each checked by running it:
- The first task cannot be a `ConditionalTask` (`ValidationError: The first task cannot be a ConditionalTask`), and a crew of only conditional tasks is rejected.
- The condition receives the `TaskOutput` of the task immediately before it in the list, not of every earlier task.
- When the condition returns `False`, the task does not run and yields an empty `TaskOutput` (`raw=""`). It still appears in `tasks_output`.
- `CrewOutput.raw` is the last **non-empty** task output, so a skipped conditional task at the end does not blank the crew's result; `tasks_output[-1].raw` is `""`.

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
- Every earlier task's output is implicit context for the next task (narrow it with `context=[...]`).
- Async tasks at the same "level" run concurrently until a synchronous task is reached.

### Hierarchical — manager coordinates

```python
from crewai import Crew, Process, Agent, LLM

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[write_brief, write_article, polish],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/%%model=openai%%"),   # manager built by CrewAI
    # OR:
    # manager_agent=Agent(role="Manager", goal="...", backstory="...", allow_delegation=True),
)
```

Rules:
- Set one of `manager_llm` or `manager_agent`. With neither, `Crew(...)` raises a `ValidationError`. If you set both, the `manager_agent` is used.
- In a hierarchical crew the **manager runs every task**. It gets delegation tools addressed to your worker agents and decides which of them does the work. Each `TaskOutput.agent` therefore shows the manager's role (for example "Crew Manager"), not the worker's. A `Task(agent=...)` is not binding on the manager.
- Tasks don't need an `agent=`; leave it out and let the manager choose.
- Crew inputs are interpolated into worker agents and tasks, but not into a `manager_agent`'s role, goal or backstory.

## CrewOutput — what `kickoff()` returns

`CrewOutput` exposes:

- `.raw` — the raw string output of the last task that produced non-empty output
- `.tasks_output` — list of `TaskOutput`, in execution order
- `.token_usage` — `UsageMetrics` (total / prompt / completion / reasoning / cache-creation tokens)
- `.json_dict` / `.pydantic` — shortcut to the final task's structured output if set
- `.to_dict()` — the final task's `json_dict`, or its `pydantic` output dumped to a dict (empty if the final task had no structured output)

```python
out = crew.kickoff(inputs={"topic": "remote work"})

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
from pydantic import BaseModel
from crewai.tasks.task_output import TaskOutput

class Plan(BaseModel):
    steps: list[str]

def at_least_three_steps(output: TaskOutput):
    # Guardrails run before a string result is parsed, so output.pydantic can be None here.
    try:
        plan = output.pydantic or Plan.model_validate_json(output.raw)
    except ValueError:
        return False, "Return JSON that matches the Plan schema."
    if len(plan.steps) < 3:
        return False, "Need at least 3 steps."
    return True, output.raw    # a string result is parsed into output_pydantic again

Task(
    description="Plan the campaign.",
    expected_output="Plan with step list.",
    agent=planner,
    output_pydantic=Plan,
    guardrail=at_least_three_steps,
    guardrail_max_retries=3,
)
```

What the guardrail returns on success becomes the task output: a string replaces `raw` and is parsed into `output_pydantic`/`output_json`; a `TaskOutput` is used as it is. Returning a message string on success (`(True, "ok")`) would replace the real output.

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
from crewai import Task
from crewai_files import File

Task(
    description="Compare the two attached contracts (v1 and v2) and list the risky clauses.",
    expected_output="Ranked list with clause numbers.",
    agent=legal,
    input_files={"v1": File(source="contracts/v1.pdf"), "v2": File(source="contracts/v2.pdf")},
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

- **`output_pydantic` vs `response_model`** — `output_pydantic` parses the result into `.pydantic` (and can still fail to parse if the model ignores the schema). `response_model` only asks the provider for structured output and leaves the parsed value to you.
- **Guardrails see unparsed output.** When a guardrail is set, `output.pydantic` is only filled if the agent already returned a model; otherwise parse `output.raw` yourself, as in the strict-JSON pattern above.
- **`max_retries` is deprecated on `Task`** — use `guardrail_max_retries`. The source docstring says "will be removed in v1.0.0" (written before the 1.x line, so effectively: a future major). Treat it as already-deprecated and avoid it in new code.
- **Async tasks and ordering.** A synchronous task waits for all pending async tasks before it starts; `context=[t1, t2]` controls which outputs it sees, not when it runs.
- **Hierarchical tasks with `agent=` are run by the manager** — the manager can route them to any worker.
- **`input_files` supersedes `Agent.multimodal=True`.** That flag is deprecated and is removed in 2.0. `input_files` needs `crewai[file-processing]`.
- **`Process.consensual` doesn't exist.** It's commented out in `process.py`.

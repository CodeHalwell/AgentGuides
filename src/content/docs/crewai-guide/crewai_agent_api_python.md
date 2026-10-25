---
title: "Agent API reference"
description: "Every field on crewai.Agent, verified against installed source — role/goal/backstory, tools, planning_config, guardrails, knowledge, MCP servers, A2A, and checkpoint."
framework: "crewai"
language: "python"
sidebar:
  label: "Agent API"
  order: 20
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/agent/core.py`, `crewai/agents/agent_builder/base_agent.py`). Every import, field, and default below was read from the installed package.

`Agent` is a Pydantic model. Fields inherit from `BaseAgent` and `Agent` extends the set. Any unknown kwarg raises a validation error — this page lists the real fields.

## Minimal runnable example

```python
from crewai import Agent, Task, Crew, LLM

analyst = Agent(
    role="Research Analyst",
    goal="Surface the strongest 3 claims about {topic}",
    backstory="Senior analyst at an investigative newsroom.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    verbose=True,
)

task = Task(
    description="Produce a bullet list on {topic}.",
    expected_output="3 bullets, each citing a source.",
    agent=analyst,
)

result = Crew(agents=[analyst], tasks=[task]).kickoff(inputs={"topic": "RAG evaluation"})
print(result.raw)
```

`role`, `goal`, and `backstory` are the only truly required fields. Everything else has a default.

## Constructor fields

Fields defined on `BaseAgent` — the ones you'll touch most:

| Field | Type | Default | Notes |
|---|---|---|---|
| `role` | `str` | required | Role name; also used as knowledge collection name. |
| `goal` | `str` | required | Objective. `{placeholders}` are interpolated from crew inputs. |
| `backstory` | `str` | required | Persona text. `{placeholders}` also interpolated. |
| `llm` | `str \| BaseLLM \| None` | `None` | Model name (`"openai/%%model=openai%%"`) or `LLM(...)` / `BaseLLM` instance. `None` reads the `MODEL` environment variable, then falls back to CrewAI's default (`gpt-4.1-mini`). |
| `function_calling_llm` | `str \| BaseLLM \| None` | `None` | Deprecated in the 1.15 line (marked for removal). Overrides the tool-calling LLM only. |
| `tools` | `list[BaseTool] \| None` | `[]` | Tools the agent may call. See the tools page. |
| `skills` | `list[Path \| Skill \| str] \| None` | `None` | Agent Skills: directories to discover, inline `SKILL.md` text, `Skill` objects, or `"@org/name"` registry refs. |
| `tool_failure_policy` | `"ignore" \| "warn" \| "raise" \| None` | `None` | What to do when a tool runs but reports failure. `None` inherits from the crew, then `"warn"`. |
| `memory` | `bool \| Memory \| MemoryScope \| MemorySlice \| None` | `None` | Per-agent memory; falls back to the crew's memory when unset. |
| `max_iter` | `int` | `25` | Hard cap on reasoning iterations per task. |
| `max_rpm` | `int \| None` | `None` | Requests-per-minute throttle for this agent. |
| `max_tokens` | `int \| None` | `None` | Max tokens per LLM call. |
| `max_execution_time` | `int \| None` | `None` | Wall-clock seconds per task. |
| `verbose` | `bool` | `False` | Print the agent's inner monologue. |
| `allow_delegation` | `bool` | `False` | Give the agent the `delegate_work_to_coworker` and `ask_question_to_coworker` tools (see [Multi-agent patterns](/crewai-guide/crewai_advanced_multi_agent_python/#delegation)). |
| `cache` | `bool` | `True` | Whether the agent may take part in tool-result caching. The default only permits it: caching turns on when the crew sets `cache=True` or the agent passes `cache=True` explicitly. `False` keeps the agent out. See [Tools → Caching](/crewai-guide/crewai_tools_python/#caching). |
| `respect_context_window` | `bool` | `True` | Summarise messages to stay under the model's context limit. |
| `max_retry_limit` | `int` | `2` | Retries on task-level errors. |
| `use_system_prompt` | `bool \| None` | `True` | Whether to set a system message (some o1-style models want `False`). |
| `inject_date` | `bool` | `False` | Inject the current date into the agent's prompt. |
| `date_format` | `str` | `"%Y-%m-%d"` | Format when `inject_date=True`. |
| `planning` | `bool` | `False` | Enable planning (reflection) before execution. |
| `planning_config` | `PlanningConfig \| None` | `None` | Fine-tune planning (max attempts, LLM override). |
| `embedder` | `EmbedderConfig \| None` | `None` | Embedder for the agent's own knowledge collection. |
| `knowledge_sources` | `list[BaseKnowledgeSource] \| None` | `None` | Per-agent knowledge (see Knowledge page). |
| `knowledge_config` | `KnowledgeConfig \| None` | `None` | Limits/threshold for knowledge retrieval. |
| `mcps` | `list[str \| MCPServerConfig] \| None` | `None` | MCP server references — URL, bare slug, or config object. `"server#tool"` filters to one tool. |
| `apps` | `list[str] \| None` | `None` | CrewAI Platform app integrations (e.g. `"gmail"`, `"gmail/send_email"`). |
| `a2a` | `A2AConfig \| A2AClientConfig \| A2AServerConfig \| list[A2AConfig \| A2AClientConfig \| A2AServerConfig] \| None` | `None` | Agent-to-Agent config (remote agents). |
| `guardrail` | `str \| Callable \| None` | `None` | Validator for output produced by `Agent.kickoff()`. It is not applied to tasks run inside a crew; use `Task.guardrail` there. See [Guardrails](#guardrails). |
| `guardrail_max_retries` | `int` | `3` | Retries when the `Agent.kickoff()` guardrail fails. |
| `step_callback` | `Callable \| None` | `None` | Step hook. What it receives depends on how the model is called; see [Callbacks and observability](#callbacks-and-observability). |
| `system_template` / `prompt_template` / `response_template` | `str \| None` | `None` | Override the built-in prompt strings. |
| `checkpoint` | `CheckpointConfig \| bool \| None` | `None` | Opt the agent into checkpointing. |
| `executor_class` | `type[CrewAgentExecutor] \| type[AgentExecutor]` | `AgentExecutor` | The newer `AgentExecutor` (from `crewai.experimental`) is now the default; pass `CrewAgentExecutor` to get the older loop back. |
| `from_repository` | `str \| None` | `None` | Load base config from the user's CrewAI repository. |

## Deprecated fields

These were deprecated during the 1.14 line and are still marked so in the current release — **don't introduce them in new code**:

- `allow_code_execution` / `code_execution_mode` — `CodeInterpreterTool` was removed. Call out to E2B / Modal / your own sandbox.
- `multimodal` — pass files natively via `Task.input_files`. The field description says it will be removed in v2.0.
- `reasoning`, `max_reasoning_attempts` — superseded by `planning` + `planning_config`.

These fields are still accepted and are marked deprecated in the model schema. `allow_code_execution` and `code_execution_mode` no longer do anything, because the tool they enabled has been removed. `function_calling_llm` (on `Agent` and `Crew`) is also marked deprecated in 1.15.

## Tools

```python
from crewai import Agent
from crewai.tools import tool

@tool("lookup_order")
def lookup_order(order_id: str) -> str:
    """Fetch order status from the orders DB."""
    return f"order {order_id}: shipped"

support = Agent(
    role="Support",
    goal="Answer order questions accurately",
    backstory="Level-2 support with DB access.",
    tools=[lookup_order],
    allow_delegation=False,
)
```

- In a sequential crew, `allow_delegation=True` adds two more tools to this agent, one to delegate work and one to ask a question, each addressed to the other agents in the crew. Workers in a hierarchical crew do not need it: the manager gets the delegation tools.
- Tool-result caching is off unless the crew or the agent opts in, so a default agent calls its tools every time. If you do enable it, pass `cache=False` for an agent whose tool outputs change between calls (live APIs, RNG).
- Use `max_usage_count` on individual tools to cap how often the agent can call them.

## Planning — plan before acting

```python
from crewai import Agent, PlanningConfig, LLM

planner_llm = LLM(model="openai/%%model=openai%%")  # a stronger model for the plan only

researcher = Agent(
    role="Researcher",
    goal="Gather the strongest 3 sources on {topic}",
    backstory="Veteran librarian.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    planning=True,
    planning_config=PlanningConfig(max_attempts=3, llm=planner_llm),
)
```

With planning on, the agent first asks the planning model for a step-by-step plan, refines it until the model says it is ready (or `max_attempts` is reached, after which it proceeds with the current plan and logs a warning), then executes. In a run with a stub we saw the planner model take the planning calls and the agent's own model take the task call.

- `planning=True` with no `planning_config` uses the defaults below.
- `planning_config.llm` swaps a different model in for planning while the executor stays as it is. When it is `None` the agent's own LLM plans.
- `planning_enabled` is a property: `True` if either `planning` or `planning_config` is set.

`PlanningConfig` fields (all optional):

| Field | Default | Effect |
|---|---|---|
| `reasoning_effort` | `"medium"` | How much the agent re-checks its plan while executing. `"low"` skips the per-step observer call. `"medium"` observes each step with the LLM and replans on failure. `"high"` also allows early finish, refinement and replanning after every step; most adaptive, most latency. |
| `observe_steps` | `None` | Force per-step LLM observation on or off regardless of `reasoning_effort`. |
| `max_attempts` | `None` | Cap on plan-refinement rounds. `None` continues until the model reports it is ready. |
| `max_steps` | `20` | Maximum steps in the generated plan. |
| `max_replans` | `3` | Maximum full replans before finalising. |
| `max_step_iterations` | `15` | Model iterations allowed inside one plan step. |
| `step_timeout` | `None` | Seconds allowed for one step; an overrun marks the step failed. |
| `system_prompt`, `plan_prompt`, `refine_prompt` | `None` | Replace the built-in planning prompts. |
| `llm` | `None` | Planning model. |

Planning adds model calls (one or more before the task, plus observer calls per step), so it pays off on long, multi-step tasks and is wasted on short ones. `Crew(planning=True)` is the crew-level variant: one planner writes a plan per task and appends it to each task's description ([Multi-agent patterns → Planning](/crewai-guide/crewai_advanced_multi_agent_python/#planning)).

## Guardrails

`Agent.guardrail` checks the output of **`Agent.kickoff()`**: the standalone, crew-less run. It does not run for tasks that a crew executes. We ran an agent with a failing `guardrail` as a crew task: the function was never called and the crew returned the unvalidated answer. In a crew, put the check on the task (`Task.guardrail`, see [Task API → Guardrails](/crewai-guide/crewai_task_api_python/#guardrails)).

```python
from crewai import Agent, LLM
from crewai.tasks.task_output import TaskOutput


def no_ssn(output: TaskOutput) -> tuple[bool, str]:
    if "ssn" in output.raw.lower():
        return False, "The answer mentions an SSN. Rewrite it without personal identifiers."
    return True, output.raw


writer = Agent(
    role="Writer",
    goal="Draft short customer replies",
    backstory="Seasoned content editor.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    guardrail=no_ssn,
    guardrail_max_retries=2,
)

reply = writer.kickoff("Write a two-line reply about the customer's refund.")
print(reply.raw)
```

- `guardrail` takes a `(output) -> (bool, value)` callable or a string. A string is judged by the agent's own LLM.
- On failure the agent runs again with the reason added to its input, up to `guardrail_max_retries` times. When the retries are spent, `kickoff()` raises `ValueError("Agent's guardrail failed validation after N retries. Last error: ...")`.
- On success the second value replaces the output: a string replaces `.raw`, a Pydantic model replaces `.pydantic`. Return `output.raw` if you do not want to change it.

## Knowledge

```python
from crewai import Agent
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

research = Agent(
    role="Legal Research",
    goal="Summarise clauses from the supplied statutes",
    backstory="Lawyer with 15 years of contract experience.",
    knowledge_sources=[PDFKnowledgeSource(file_paths=["gdpr.pdf"])],   # read from ./knowledge/gdpr.pdf
)
```

Knowledge is separate from memory: it is a vector store you fill ahead of time and attach to the agent (or crew). Before each task the agent rewrites the task text into a search query with its own LLM, retrieves matching chunks and appends them to the prompt, so each task costs one extra model call. File paths given as strings are looked up under `./knowledge/`. See [Knowledge](/crewai-guide/crewai_knowledge_python/).

## MCP servers

```python
from crewai import Agent
from crewai.mcp import MCPServerStdio, MCPServerHTTP

agent = Agent(
    role="DevOps",
    goal="Inspect infra",
    backstory="SRE.",
    mcps=[
        MCPServerStdio(command="uvx", args=["mcp-server-fetch"]),
        MCPServerHTTP(url="https://tools.example.com/mcp", headers={"Authorization": "Bearer …"}),
        "notion",                   # bare slug → CrewAI Platform integration
        "https://tools.example.com/mcp#search",  # single-tool filter
    ],
)
```

- Stdio / HTTP / SSE are the three transports exposed in `crewai.mcp`.
- `tool_filter=` on the config object lets you allow-list or block-list tools without a `#tool` suffix.
- `cache_tools_list=True` reuses the tool listing after the first discovery.
- The model sees each MCP tool under a namespaced name (a sanitised server name plus a hash), not the server's own tool name. Filters receive the original name.

[Tools → MCP server tools](/crewai-guide/crewai_tools_python/#mcp-server-tools--agentmcps) has an end-to-end example against a local server.

## A2A — remote agents

```python
from crewai import Agent
from crewai.a2a.config import A2AClientConfig

remote = Agent(
    role="Index Lookups",
    goal="Look up internal knowledge base",
    backstory="Specialised retrieval agent.",
    a2a=A2AClientConfig(
        endpoint="https://kb.example.com/.well-known/agent-card.json",
    ),
)
```

Pass a single config, a single `A2AServerConfig`, or a list combining clients with one server to expose this agent as an A2A endpoint.

## Callbacks and observability

```python
from crewai import Agent
from crewai.agents.parser import AgentAction, AgentFinish


def on_step(step):
    if isinstance(step, AgentAction):          # a tool call and its result
        print(f"{step.tool}({step.tool_input}) -> {str(step.result)[:80]}")
    elif isinstance(step, AgentFinish):        # the final answer
        print("finished:", step.output[:80])


agent = Agent(role="Researcher", goal="Find sources", backstory="Librarian.", step_callback=on_step)
```

What `step_callback` receives depends on how the executor talks to the model:

- With a model that reports native function calling (`supports_function_calling()` returns true), the callback fires **once, with an `AgentFinish`**, when the agent produces its final answer. Tool calls do not reach it.
- With a model on the text (ReAct) path, such as a custom `BaseLLM` that does not report function calling, it fires with a `ToolResult` and an `AgentAction` after each tool call, then an `AgentFinish`.

We confirmed both by running the same agent against a function-calling stub and a text stub. To watch tool calls whatever the model, listen for `ToolUsageStartedEvent` / `ToolUsageFinishedEvent` instead; see [Production → Event listeners](/crewai-guide/crewai_production_guide/#event-listeners). `Crew(step_callback=...)` behaves the same way for every agent in the crew, and `Task(callback=...)` / `Crew(task_callback=...)` receive the `TaskOutput` when a task completes.

## Patterns

### 1. Parallel fan-out with `async_execution`

```python
analyst_a = Agent(role="Market Analyst", goal="Size the market", backstory="Industry analyst.")
analyst_b = Agent(role="Tech Analyst", goal="Assess the technology", backstory="Former CTO.")
editor = Agent(role="Editor", goal="Merge findings", backstory="Report editor.")

task_a = Task(description="Market view of {topic}.", expected_output="Five bullets.", agent=analyst_a, async_execution=True)
task_b = Task(description="Technology view of {topic}.", expected_output="Five bullets.", agent=analyst_b, async_execution=True)
summary = Task(description="Merge both views.", expected_output="One page.", agent=editor, context=[task_a, task_b])
```

Consecutive async tasks run concurrently; the next synchronous task waits for them. `context=[...]` limits the summary to those two outputs.

### 2. Cheap executor, smart planner

Keep the reasoning model cheap, use a bigger model only for the planning reflection:

```python
Agent(
    role="Doc Writer",
    goal="Write API docs",
    backstory="Technical writer.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    planning=True,
    planning_config=PlanningConfig(llm=LLM(model="openai/%%model=openai%%")),
)
```

### 3. Delegation-capable manager

```python
manager = Agent(
    role="Manager",
    goal="Break the problem down and route tasks",
    backstory="Skilled at decomposition.",
    allow_delegation=True,
    llm=LLM(model="openai/%%model=openai%%"),
)
crew = Crew(
    agents=[researcher, writer],      # workers only: listing the manager here is a ValidationError
    tasks=[brief, article],
    process=Process.hierarchical,
    manager_agent=manager,
)
```

Pick one of `manager_llm` and `manager_agent`; if both are set, the `manager_agent` is used. `manager_agent` gives you full control over the manager's prompts. Do not put `{placeholders}` in a `manager_agent`'s role, goal or backstory: crew inputs are interpolated into the worker agents and tasks only, so the manager would see the literal `{topic}`. A manager agent must not have tools; `Crew` raises `Exception: Manager agent should not have tools` at kickoff.

### 4. Per-agent rate limiting

```python
Agent(role="Web Scraper", goal="Collect pages", backstory="Crawler operator.", max_rpm=20)
```

`max_rpm` throttles this agent's model requests (the crew-level `Crew(max_rpm=...)` applies to the whole run); `max_execution_time` covers the whole task wall-clock.

### 5. Checkpoint one slow agent only

```python
from crewai import Agent, CheckpointConfig

slow = Agent(role="Heavy Research", goal="Survey the literature", backstory="Research fellow.",
             checkpoint=CheckpointConfig(on_events=["task_completed", "llm_call_completed"]))
```

Other agents in the same crew can leave `checkpoint=None`. See [Production → Recovery with checkpoints](/crewai-guide/crewai_production_guide/#recovery-with-checkpoints) for the crew-level setting.

## Gotchas

- **Field name is `planning`, not `reasoning`.** The `reasoning=...` kwarg still works but is deprecated.
- **`function_calling_llm` isn't the agent's primary LLM.** It's used only for tool-call formatting, and it is marked deprecated in 1.15; prefer a model that handles tool calls itself.
- **`allow_code_execution` is a no-op.** `CodeInterpreterTool` was removed from the public API in 1.14; use E2B or Modal.
- **`knowledge_sources` needs an embedder.** If neither the agent's `embedder` nor the crew's is set, the default `OpenAIEmbeddingFunction` tries to reach OpenAI and needs `OPENAI_API_KEY`.
- **`from_repository`** loads an agent definition from your CrewAI repository and merges the keyword arguments you pass on top of it, so explicit values override the stored ones.

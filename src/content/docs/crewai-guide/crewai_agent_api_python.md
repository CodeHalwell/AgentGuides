---
title: "Agent API reference"
description: "Every field on crewai.Agent, verified against installed source — role/goal/backstory, tools, planning_config, guardrails, knowledge, MCP servers, A2A, and checkpoint."
framework: crewai
language: python
sidebar:
  label: "Agent API"
  order: 20
---

> **Verified against crewai==1.14.3a2** (source: `crewai/agent/core.py`, `crewai/agents/agent_builder/base_agent.py`). Every import, field, and default below was read from the installed package.

`Agent` is a Pydantic model. Fields inherit from `BaseAgent` and `Agent` extends the set. Any unknown kwarg raises a validation error — this page lists the real fields.

## Minimal runnable example

```python
from crewai import Agent, Task, Crew, LLM

analyst = Agent(
    role="Research Analyst",
    goal="Surface the strongest 3 claims about {topic}",
    backstory="Senior analyst at an investigative newsroom.",
    llm=LLM(model="openai/gpt-4o-mini"),
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
| `llm` | `str \| BaseLLM \| None` | `None` | Model name (`"openai/gpt-4o"`) or `LLM(...)` / `BaseLLM` instance. `None` uses default from `crewai.llm`. |
| `function_calling_llm` | `str \| BaseLLM \| None` | `None` | Override the tool-calling LLM only (agent keeps `llm` for reasoning). |
| `tools` | `list[BaseTool] \| None` | `[]` | Tools the agent may call. See the tools page. |
| `max_iter` | `int` | `25` | Hard cap on reasoning iterations per task. |
| `max_rpm` | `int \| None` | `None` | Requests-per-minute throttle for this agent. |
| `max_tokens` | `int \| None` | `None` | Max tokens per LLM call. |
| `max_execution_time` | `int \| None` | `None` | Wall-clock seconds per task. |
| `verbose` | `bool` | `False` | Print the agent's inner monologue. |
| `allow_delegation` | `bool` | `False` | Let the agent call `Delegate Work` / `Ask Question` tools. |
| `cache` | `bool` | `True` | Cache tool-call results. |
| `respect_context_window` | `bool` | `True` | Summarise messages to stay under the model's context limit. |
| `max_retry_limit` | `int` | `2` | Retries on task-level errors. |
| `use_system_prompt` | `bool \| None` | `True` | Whether to set a system message (some o1-style models want `False`). |
| `inject_date` | `bool` | `False` | Prepend current date to the task description. |
| `date_format` | `str` | `"%Y-%m-%d"` | Format when `inject_date=True`. |
| `planning` | `bool` | `False` | Enable planning (reflection) before execution. |
| `planning_config` | `PlanningConfig \| None` | `None` | Fine-tune planning (max attempts, LLM override). |
| `embedder` | `EmbedderConfig \| None` | `None` | Embedder for the agent's own knowledge collection. |
| `knowledge_sources` | `list[BaseKnowledgeSource] \| None` | `None` | Per-agent knowledge (see Knowledge page). |
| `knowledge_config` | `KnowledgeConfig \| None` | `None` | Limits/threshold for knowledge retrieval. |
| `mcps` | `list[str \| MCPServerConfig] \| None` | `None` | MCP server references — URL, bare slug, or config object. `"server#tool"` filters to one tool. |
| `apps` | `list[PlatformAppOrAction] \| None` | `None` | CrewAI Platform app integrations (e.g. `"gmail"`, `"gmail/send_email"`). |
| `a2a` | `A2AConfig \| A2AClientConfig \| A2AServerConfig \| list[A2AConfig \| A2AClientConfig \| A2AServerConfig] \| None` | `None` | Agent-to-Agent config (remote agents). |
| `guardrail` | `str \| GuardrailType \| None` | `None` | Validator applied to agent output. |
| `guardrail_max_retries` | `int` | `3` | Retries when the guardrail fails. |
| `step_callback` | `Callable \| None` | `None` | Called after every agent step. |
| `system_template` / `prompt_template` / `response_template` | `str \| None` | `None` | Override the built-in prompt strings. |
| `checkpoint` | `CheckpointConfig \| bool \| None` | `None` | Opt the agent into checkpointing. |
| `executor_class` | `type[CrewAgentExecutor] \| type[AgentExecutor]` | `CrewAgentExecutor` | Swap in the experimental `AgentExecutor` (ReAct-style). |
| `from_repository` | `str \| None` | `None` | Load base config from the user's CrewAI repository. |

## Deprecated in 1.14.x

The installed source marks these as deprecated — **don't introduce them in new code**:

- `allow_code_execution` / `code_execution_mode` — `CodeInterpreterTool` was removed. Call out to E2B / Modal / your own sandbox.
- `multimodal` — pass files natively via `Task.input_files`.
- `reasoning`, `max_reasoning_attempts` — superseded by `planning` + `planning_config`.

Setting any of these still works but emits a `DeprecationWarning`.

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

- `tools=[]` + `allow_delegation=True` in a hierarchical crew still lets the agent call the manager's delegation tools.
- Set `cache=False` on an agent whose tool outputs change between calls (live APIs, RNG).
- Use `max_usage_count` on individual tools to cap how often the agent can call them.

## Planning — plan before acting

```python
from crewai import Agent, PlanningConfig, LLM

planner_llm = LLM(model="openai/gpt-4o")  # cheaper than the exec LLM is fine

researcher = Agent(
    role="Researcher",
    goal="Gather the strongest 3 sources on {topic}",
    backstory="Veteran librarian.",
    llm=LLM(model="openai/gpt-4o-mini"),
    planning=True,
    planning_config=PlanningConfig(max_attempts=3, llm=planner_llm),
)
```

- `planning=True` with no `planning_config` uses defaults.
- `planning_config.llm` lets you swap a smarter model in for the reflection step while the executor stays cheap.
- `planning_enabled` is a property — it's `True` if either `planning` or `planning_config` is set.

## Guardrails

```python
from crewai import Agent
from crewai.tasks.task_output import TaskOutput

def no_pii(output: TaskOutput) -> tuple[bool, str]:
    if "ssn" in output.raw.lower():
        return False, "Output contains what looks like an SSN — rewrite without it."
    return True, output.raw

writer = Agent(
    role="Writer",
    goal="Draft blog posts",
    backstory="Seasoned content editor.",
    guardrail=no_pii,
    guardrail_max_retries=2,
)
```

- `guardrail` accepts a `(TaskOutput) -> tuple[bool, Any]` callable or a string describing the check (the agent itself runs the check).
- On failure the agent re-executes up to `guardrail_max_retries` with the failure reason injected into the prompt.

## Knowledge

```python
from crewai import Agent
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

research = Agent(
    role="Legal Research",
    goal="Summarise clauses from the supplied statutes",
    backstory="Lawyer with 15 years of contract experience.",
    knowledge_sources=[PDFKnowledgeSource(file_paths=["gdpr.pdf"])],
)
```

Knowledge is separate from memory: it's an immutable vector store attached to the agent (or crew). See the Knowledge page for the full list of sources.

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
- `tool_filter=` on the config object lets you allow-list tools without a `#tool` suffix.
- `cache_tools_list=True` speeds up subsequent runs after discovery.

## A2A — remote agents

```python
from crewai import Agent
from crewai.a2a.config import A2AClientConfig

remote = Agent(
    role="Index Lookups",
    goal="Look up internal knowledge base",
    backstory="Specialised retrieval agent.",
    a2a=A2AClientConfig(
        agent_card_url="https://kb.example.com/.well-known/agent.json",
    ),
)
```

Pass a single config, a single `A2AServerConfig`, or a list combining clients with one server to expose this agent as an A2A endpoint.

## Callbacks and observability

```python
def on_step(step):
    print(f"[{step.agent.role}] {step.tool} -> {step.observation[:80]}")

agent = Agent(role="...", goal="...", backstory="...", step_callback=on_step)
```

`step_callback` fires after each reasoning step. For crew-wide callbacks, use `Crew(step_callback=...)` — the crew-level one runs for every agent.

## Patterns

### 1. Parallel fan-out with `async_execution`

```python
analyst_a = Agent(role="Market Analyst", ...)
analyst_b = Agent(role="Tech Analyst", ...)

task_a = Task(description="...", expected_output="...", agent=analyst_a, async_execution=True)
task_b = Task(description="...", expected_output="...", agent=analyst_b, async_execution=True)
summary = Task(description="Merge", expected_output="...", agent=editor, context=[task_a, task_b])
```

Async tasks at the same level run concurrently; `context=[...]` forces the summariser to wait for both.

### 2. Cheap executor, smart planner

Keep the reasoning model cheap, use a bigger model only for the planning reflection:

```python
Agent(
    role="Doc Writer",
    goal="...",
    backstory="...",
    llm=LLM(model="openai/gpt-4o-mini"),
    planning=True,
    planning_config=PlanningConfig(llm=LLM(model="openai/gpt-4o")),
)
```

### 3. Delegation-capable manager

```python
manager = Agent(
    role="Manager",
    goal="Break the problem down and route tasks",
    backstory="Skilled at decomposition.",
    allow_delegation=True,
    llm=LLM(model="openai/gpt-4o"),
)
crew = Crew(agents=[manager, researcher, writer], tasks=[...], process=Process.hierarchical, manager_agent=manager)
```

Don't set both `manager_llm` and `manager_agent` — pick one. `manager_agent` gives you full control over prompts.

### 4. Per-agent rate limiting

```python
Agent(role="Web Scraper", goal="...", backstory="...", max_rpm=20)
```

`max_rpm` is enforced before each LLM call; `max_execution_time` covers the whole task wall-clock.

### 5. Checkpoint one slow agent only

```python
from crewai import Agent, CheckpointConfig

slow = Agent(role="Heavy Research", goal="...", backstory="...",
             checkpoint=CheckpointConfig(on_events=["task_completed", "llm_call_completed"]))
```

Other agents in the same crew can leave `checkpoint=None` (the crew's setting then applies).

## Gotchas

- **Field name is `planning`, not `reasoning`.** The `reasoning=...` kwarg still works but is deprecated.
- **`function_calling_llm` isn't the agent's primary LLM.** It's used only for tool-call formatting on models that need a separate call for JSON-schema arguments.
- **`allow_code_execution` is a no-op.** `CodeInterpreterTool` was removed from the public API in 1.14; use E2B or Modal.
- **`knowledge_sources` needs an embedder.** If neither the agent's `embedder` nor the crew's is set, the default `OpenAIEmbeddingFunction` tries to reach OpenAI and needs `OPENAI_API_KEY`.
- **`from_repository`** merges settings from your CrewAI repo on top of the explicit kwargs you passed — explicit values win.

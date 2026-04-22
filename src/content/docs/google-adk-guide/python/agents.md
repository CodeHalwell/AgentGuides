---
title: "Agents (LlmAgent, Parallel, Loop, Sequential)"
description: "The agent primitives shipped by google-adk and when to pick each."
framework: google-adk
language: python
sidebar:
  order: 20
---

Verified against google-adk==2.0.0b1 (`google/adk/agents/`).

ADK exposes one LLM-backed agent (`LlmAgent`, also re-exported as `Agent`) and three *shell* agents for composition (`SequentialAgent`, `ParallelAgent`, `LoopAgent`). As of 2.x the three shell agents are `@deprecated` in favour of `Workflow` — see the [workflows page](./workflows/). They still work but new projects should compose with `Workflow`.

## Minimal example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

root = LlmAgent(
    name="tutor",
    model="gemini-2.5-flash",
    instruction="Answer concisely. If you do maths, show the steps.",
)

async def main():
    runner = InMemoryRunner(agent=root, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug("What is 15 + 27?", user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

`Agent` is a type alias for `LlmAgent` (`agents/llm_agent.py:end`). `InMemoryRunner` wires in-memory session/memory/artifact services so the example runs with no GCP setup.

## The four agents at a glance

| Class | Runs sub-agents | Terminates when | Status |
|---|---|---|---|
| `LlmAgent` | No (uses `tools=` + `sub_agents=`) | model stops emitting function calls | Stable |
| `SequentialAgent` | yes, in order | all finished | **Deprecated → `Workflow`** |
| `ParallelAgent` | yes, concurrently | all finished | **Deprecated → `Workflow`** |
| `LoopAgent` | yes, in order, repeated | `max_iterations` reached OR sub-agent sets `actions.escalate=True` | **Deprecated → `Workflow`** |

The deprecation notices are emitted via `typing_extensions.deprecated` at class level (see `sequential_agent.py:48`, `parallel_agent.py:150`, `loop_agent.py:52`).

## LlmAgent

Pydantic model. Constructor accepts every field as a kwarg.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from pydantic import BaseModel

class Reply(BaseModel):
    answer: str
    confidence: float

agent = LlmAgent(
    name="research_assistant",          # required; must be a Python identifier
    model="gemini-2.5-flash",            # str or BaseLlm; inherits from ancestors when ""
    description="Answers research questions with web search.",
    instruction="You are a research assistant. Cite the URLs you consulted.",
    tools=[google_search],
    output_schema=Reply,                 # optional; forbids tools when set
    output_key="latest_reply",           # writes final text to session.state[key]
    include_contents="default",          # or "none" to wipe history
    disallow_transfer_to_parent=False,
    disallow_transfer_to_peers=False,
)
```

**Field reference** (verified in `agents/llm_agent.py`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Identifier; used for agent-transfer routing |
| `model` | `str \| BaseLlm` | `""` | Empty inherits; built-in default is `gemini-2.5-flash` (`DEFAULT_MODEL`) |
| `instruction` | `str \| InstructionProvider` | `""` | Supports `{state_key}` placeholders resolved from session state |
| `global_instruction` | same | `""` | **Deprecated** → use `GlobalInstructionPlugin` |
| `static_instruction` | `types.ContentUnion` | `None` | For context-cache friendly prefixes |
| `tools` | `list[Callable \| BaseTool \| BaseToolset]` | `[]` | Callables are auto-wrapped as `FunctionTool` |
| `generate_content_config` | `types.GenerateContentConfig` | `None` | Temperature, safety, thinking, etc. |
| `mode` | `'chat' \| 'task' \| 'single_turn' \| None` | `None` | Root `LlmAgent` must have `mode='chat'` |
| `input_schema` / `output_schema` | Pydantic model / schema | `None` | Setting `output_schema` disables tool use |
| `output_key` | `str` | `None` | Writes final text to `session.state[key]` |
| `include_contents` | `'default' \| 'none'` | `'default'` | `'none'` → stateless single-turn |
| `planner` | `BasePlanner` | `None` | `BuiltInPlanner` forwards `thinking_config` to the model |
| `code_executor` | `BaseCodeExecutor` | `None` | See [code executors](#code-executors) |
| `disallow_transfer_to_parent` / `disallow_transfer_to_peers` | `bool` | `False` | Governs agent-transfer reachability |
| `before_model_callback` / `after_model_callback` / `on_model_error_callback` | fn or list | `None` | See [callbacks-and-plugins](./callbacks-and-plugins/) |
| `before_tool_callback` / `after_tool_callback` / `on_tool_error_callback` | fn or list | `None` | Same |
| `before_agent_callback` / `after_agent_callback` | fn or list | `None` | Inherited from `BaseAgent` |

### Model resolution

A bare string is looked up in `LLMRegistry`. The registered prefixes are `Gemini`, `Gemma`, `ApigeeLlm`, optionally `Claude`, `LiteLlm`, `Gemma3Ollama` (`models/__init__.py`). Setting `LlmAgent.set_default_model("gemini-2.5-pro")` changes the class-level default used when `model=""` and no ancestor sets it.

```python
from google.adk.models import Gemini, LiteLlm

# Gemini with explicit base URL and speech config
agent = LlmAgent(name="voice", model=Gemini(model="gemini-2.5-pro"))

# OpenAI via LiteLlm (requires `pip install google-adk[extensions]`)
agent = LlmAgent(name="gpt", model=LiteLlm(model="openai/gpt-4o"))
```

### Dynamic instructions

`instruction` can be a callable receiving a `ReadonlyContext`:

```python
async def instruction_provider(ctx):
    user = ctx.state.get("user_name", "there")
    return f"You are talking to {user}. Be friendly."

agent = LlmAgent(name="greeter", instruction=instruction_provider)
```

When you set `static_instruction`, the runtime places it as `system_instruction` (ideal for cache keys) and routes `instruction` into the user content instead (`agents/llm_agent.py:248-297`).

## Transfer and routing

An `LlmAgent` with `sub_agents=[...]` gets the `transfer_to_agent` tool injected automatically. The runner's `_find_agent_to_run` routes the next user message to the last-replying transferable agent (`runners.py:1456`).

```python
from google.adk.agents import LlmAgent

billing = LlmAgent(name="billing", description="Handles refunds, invoices.", instruction="...")
support = LlmAgent(name="support", description="Handles tech issues.", instruction="...")
root = LlmAgent(
    name="triage",
    instruction="Route the user to the right specialist.",
    sub_agents=[billing, support],
)
```

Set `disallow_transfer_to_parent=True` on a specialist to prevent it from returning control to the triage agent. Pair with `disallow_transfer_to_peers=True` to lock the conversation to the single agent (the runtime will then use `SingleFlow` instead of `AutoFlow`, disabling transfer-tool injection entirely — `llm_agent.py:788-797`).

## Code executors

Set `code_executor=` on an `LlmAgent` to let the model run code:

| Executor | Where it runs | Extra install |
|---|---|---|
| `BuiltInCodeExecutor` | Gemini-side (safe, sandboxed) | none |
| `UnsafeLocalCodeExecutor` | current Python process | none — **unsafe** |
| `VertexAiCodeExecutor` | Vertex AI extension | `google-adk[extensions]` |
| `ContainerCodeExecutor` | Local Docker container | `google-adk[extensions]` |
| `GkeCodeExecutor` | GKE pod | `google-adk[extensions]` |
| `AgentEngineSandboxCodeExecutor` | Agent Engine sandbox | `google-adk[extensions]` |

```python
from google.adk.code_executors import BuiltInCodeExecutor

agent = LlmAgent(
    name="analyst",
    model="gemini-2.5-pro",
    instruction="Use Python to compute anything numeric.",
    code_executor=BuiltInCodeExecutor(),
)
```

Note: when `RunConfig.support_cfc=True` and the agent's model is `gemini-2.*`, the runner swaps in `BuiltInCodeExecutor` automatically (`runners.py:1806-1814`).

## Deprecated shell agents (still supported)

All three accept `name` and `sub_agents` via `BaseAgent`. They emit `DeprecationWarning` on import and will be removed in a future release.

### SequentialAgent

```python
from google.adk.agents import SequentialAgent, LlmAgent

draft = LlmAgent(name="drafter", instruction="Draft an essay.", output_key="draft")
polish = LlmAgent(
    name="polisher",
    instruction="Polish the draft in state['draft']. Return only the final text.",
)

pipeline = SequentialAgent(name="writer", sub_agents=[draft, polish])
```

Sub-agents run in order. State is shared across them; use `output_key` to pass a value forward.

### ParallelAgent

```python
from google.adk.agents import ParallelAgent, LlmAgent

algo_a = LlmAgent(name="algo_a", instruction="Answer using approach A.")
algo_b = LlmAgent(name="algo_b", instruction="Answer using approach B.")
fanout = ParallelAgent(name="multi_try", sub_agents=[algo_a, algo_b])
```

Sub-agents run concurrently in isolated branches. `run_live` is **not supported** (`parallel_agent.py:219`).

### LoopAgent

```python
from google.adk.agents import LoopAgent, LlmAgent

critic = LlmAgent(
    name="critic",
    instruction=(
        "Read state['draft']. If good enough, set actions.escalate=True. "
        "Otherwise rewrite it and store it back to state['draft']."
    ),
    output_key="draft",
)

loop = LoopAgent(name="refine", sub_agents=[critic], max_iterations=5)
```

Exit conditions: `max_iterations` reached, or any event with `actions.escalate=True`. `max_iterations=None` means loop until an escalate.

## Migration to `Workflow`

The deprecated shells map to `Workflow` like this (full details in the [workflows page](./workflows/)):

```python
from google.adk.workflow import Workflow, START

# Sequential
pipeline = Workflow(name="pipeline", edges=[(START, draft, polish)])

# Parallel (fan-out)
fanout = Workflow(name="fanout", edges=[(START, (algo_a, algo_b))])

# Loop — use a router node + a routing map. See workflows page.
```

`Workflow` is a `BaseNode`, not a `BaseAgent`. `App(root_agent=workflow)` is the recommended way to wire it to a `Runner`.

## Patterns

### 1 — Specialists with a triage parent
Set `sub_agents=[a, b]` on a parent `LlmAgent` to get `transfer_to_agent` routing. Disable `disallow_transfer_to_peers` on specialists so they can bounce between one another without returning to root.

### 2 — Produce → Validate with `output_schema`
An `LlmAgent` with `output_schema=MyPydanticModel` can't use tools but emits a validated structured reply. Wire it downstream of a tool-enabled agent via `Workflow` or chain `output_key` → prompt template (`{draft}` placeholders).

### 3 — ReAct via `BuiltInPlanner`
```python
from google.adk.planners import BuiltInPlanner
from google.genai import types

agent = LlmAgent(
    name="thoughtful",
    model="gemini-2.5-pro",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True)),
    tools=[...],
)
```
`PlanReActPlanner` is also available for a textual plan-then-act flow (`planners/plan_re_act_planner.py`).

### 4 — Reflection loop
`LoopAgent` (or the `Workflow` equivalent) with a single reflective `LlmAgent` that rewrites `state['draft']` each turn and escalates when satisfied. Use `include_contents="none"` on the reflective agent to avoid feeding the full history back each iteration.

### 5 — Parallel multi-try with a judge
`ParallelAgent` (or `Workflow` fan-out) of N candidate agents, followed by a "judge" `LlmAgent` that reads `state['candidate_1..N']` and picks the best. Pair each candidate's `output_key` to a distinct state slot.

## Gotchas

- Setting `output_schema` **disables tool use** for that agent, including sub-agent transfer (`llm_agent.py:348-372`).
- `global_instruction` is deprecated at the agent level; use `GlobalInstructionPlugin` at the `App` level.
- A root `LlmAgent` must have `mode='chat'` or the runner auto-sets it; other modes are only valid inside a `Workflow`.
- `LoopAgent.run_live` is **not implemented** — `ParallelAgent.run_live` also raises `NotImplementedError`.
- When a sub-agent has no `model`, it inherits from the nearest ancestor `LlmAgent`. If the root also omits `model`, the default is resolved via `LlmAgent._default_model` (`gemini-2.5-flash`).
- Callables passed to `tools=` are wrapped as `FunctionTool(func=callable)` automatically. Pass an explicit `FunctionTool` only when you need `require_confirmation=`.

---
title: "Agents (LlmAgent, LangGraphAgent, RemoteA2aAgent, Parallel, Loop, Sequential)"
description: "All agent primitives shipped by google-adk: LlmAgent, LangGraphAgent, RemoteA2aAgent, and the deprecated shell agents."
framework: google-adk
language: python
sidebar:
  order: 20
---

Verified against google-adk==2.1.0 (`google/adk/agents/`).

ADK exposes one LLM-backed agent (`LlmAgent`, also re-exported as `Agent`), three *shell* agents for composition (`SequentialAgent`, `ParallelAgent`, `LoopAgent` — deprecated in 2.x), a LangGraph bridge (`LangGraphAgent`), and a remote-agent client (`RemoteA2aAgent`). New projects should compose with `Workflow` rather than the deprecated shell agents — see the [workflows page](./workflows/).

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

## Agent types at a glance

| Class | Module | Purpose | Status |
|---|---|---|---|
| `LlmAgent` | `google.adk.agents` | LLM-backed; uses `tools=` + `sub_agents=` for orchestration | Stable |
| `SequentialAgent` | `google.adk.agents` | Runs sub-agents in sequence | **Deprecated → `Workflow`** |
| `ParallelAgent` | `google.adk.agents` | Runs sub-agents concurrently | **Deprecated → `Workflow`** |
| `LoopAgent` | `google.adk.agents` | Runs sub-agents in a loop until `escalate` or `max_iterations` | **Deprecated → `Workflow`** |
| `LangGraphAgent` | `google.adk.agents.langgraph_agent` | Wraps a compiled LangGraph graph as a `BaseAgent` | Stable (concept) |
| `RemoteA2aAgent` | `google.adk.agents.remote_a2a_agent` | Calls a remote A2A-compatible agent over HTTP | `@a2a_experimental` |

The deprecation notices are emitted via `typing_extensions.deprecated` at class level (see `sequential_agent.py:48`, `parallel_agent.py:150`, `loop_agent.py:52`).

> **LangGraphAgent** and **RemoteA2aAgent** have source-verified deep dives in [Class deep dives — vol. 2](./google_adk_class_deep_dives_v2/).

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

### `generate_content_config` — temperature, safety, thinking

All generation parameters (temperature, top-p, max tokens, safety settings, thinking mode) live inside `types.GenerateContentConfig`. Do **not** pass them as top-level fields on `LlmAgent` — they are not accepted there.

```python
from google.adk.agents import LlmAgent
from google.genai import types

agent = LlmAgent(
    name="precise_analyst",
    model="gemini-2.5-pro",
    instruction="Analyse the data carefully.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,          # lower = more deterministic
        max_output_tokens=4096,
        top_p=0.95,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            )
        ],
    ),
)

# Gemini 2.5 thinking mode
thinking_agent = LlmAgent(
    name="thoughtful",
    model="gemini-2.5-pro",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=8192,
        )
    ),
)
```

### Model resolution

A bare string is looked up in `LLMRegistry`. The registered prefixes are `Gemini`, `Gemma`, `ApigeeLlm`, optionally `Claude`, `LiteLlm`, `Gemma3Ollama` (`models/__init__.py`). Setting `LlmAgent.set_default_model("gemini-2.5-pro")` changes the class-level default used when `model=""` and no ancestor sets it. The built-in class constants are `LlmAgent.DEFAULT_MODEL = "gemini-2.5-flash"` and `LlmAgent.DEFAULT_LIVE_MODEL = "gemini-live-2.5-flash-native-audio"`.

```python
from google.adk.models import Gemini, LiteLlm

# Gemini with explicit base URL and speech config
agent = LlmAgent(name="voice", model=Gemini(model="gemini-2.5-pro"))

# OpenAI via LiteLlm (requires `pip install google-adk[extensions]`)
agent = LlmAgent(name="gpt", model=LiteLlm(model="openai/gpt-4o"))

# Change the class-level default for all agents in the process
LlmAgent.set_default_model("gemini-2.5-pro")
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

## LangGraphAgent

`LangGraphAgent` bridges an existing LangGraph compiled graph into ADK's agent system. Install `langgraph` and `langchain-core` first (`pip install langgraph langchain-core`). Verified in `agents/langgraph_agent.py`.

```python
import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from google.adk.agents import LangGraphAgent
from google.adk.runners import InMemoryRunner

# --- Build a minimal LangGraph ------------------------------------------------
def chatbot(state: MessagesState):
    # Replace this with your actual LLM call (e.g. ChatGoogleGenerativeAI)
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Echo: {last}")]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- Wrap as an ADK agent ------------------------------------------------------
agent = LangGraphAgent(
    name="echo_bot",
    graph=graph,
    instruction="You are a helpful assistant.",   # prepended as SystemMessage if graph is empty
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug("hello", user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)   # "Echo: hello"

asyncio.run(main())
```

**How it works (from `agents/langgraph_agent.py`):**

- When `graph.checkpointer` is set the agent passes only the latest user messages — the graph owns its own memory via the checkpointer, keyed by `ctx.session.id` as the LangGraph `thread_id`.
- When there is no checkpointer, the agent passes the full conversation (user ↔ this agent only) as `HumanMessage` / `AIMessage` so the graph has context.
- The `instruction` is prepended as a `SystemMessage` only when the graph has no prior messages yet (initial turn).
- `LangGraphAgent` emits a single `Event` per invocation — it does **not** stream partial responses.

**Constructor fields:**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Agent name |
| `graph` | `CompiledGraph` | required | The compiled LangGraph graph |
| `instruction` | `str` | `""` | System instruction injected on first turn |

**Gotchas:**
- `sub_agents=` is **not supported** on `LangGraphAgent` — it does not participate in ADK's agent-transfer routing.
- If you need ADK tools inside the graph, call `FunctionTool.run_async` from your LangGraph nodes directly; `LangGraphAgent` itself holds no `tools=` list.
- Multi-turn with a checkpointer requires that the runner's session id is stable across calls (it is, by default).

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

`BuiltInPlanner` delegates planning entirely to Gemini's native thinking feature. It injects a `ThinkingConfig` into every `LlmRequest` — the model then produces `thought=True` parts internally which are stripped before the response reaches the user.

Source: `planners/built_in_planner.py`.

```python
from google.adk.planners import BuiltInPlanner
from google.adk.agents import LlmAgent
from google.genai import types

# ── Minimal — enable thinking, no token budget ────────────────────────────────
agent = LlmAgent(
    name="thoughtful",
    model="gemini-2.5-pro",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
    tools=[...],
)

# ── Cap thinking tokens (reduces latency / cost) ──────────────────────────────
agent_capped = LlmAgent(
    name="budgeted",
    model="gemini-2.5-flash",
    instruction="Research and summarise the topic.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=4096,   # max thinking tokens; 0 disables thinking
        )
    ),
    tools=[google_search],
)
```

**How `BuiltInPlanner` works (verified source `built_in_planner.py`):**

1. `apply_thinking_config(llm_request)` is called before each model call — sets `llm_request.config.thinking_config`. If the request already has a thinking config it is overwritten (a warning is logged).
2. `build_planning_instruction(...)` returns `None` — no extra system instruction is prepended.
3. `process_planning_response(...)` returns `None` — no response post-processing needed (Gemini handles it natively).

**Accessing thought parts in callbacks:**

```python
async def capture_thoughts(callback_context, llm_response):
    """Log thinking tokens for debugging."""
    for part in (llm_response.content.parts or []):
        if getattr(part, "thought", False) and part.text:
            print("[THOUGHT]", part.text[:200])
    return None   # don't modify the response

agent = LlmAgent(
    name="transparent",
    model="gemini-2.5-pro",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
    after_model_callback=capture_thoughts,
    tools=[...],
)
```

### PlanReActPlanner (no thinking-model required)

`PlanReActPlanner` works with **any** Gemini model — it does not require `thinking_config` support. Instead it injects a structured prompt that instructs the model to emit planning, reasoning, action, and final-answer blocks using XML-style tags. The planner strips all content except function calls and the final answer from what is sent back to the user.

Source: `planners/plan_re_act_planner.py`.

```python
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.tools import google_search

agent = LlmAgent(
    name="planner_agent",
    model="gemini-2.5-flash",
    instruction="Answer research questions with web search.",
    planner=PlanReActPlanner(),
    tools=[google_search],
)
```

**What the model produces internally** (never shown to the user as-is):

```
/*PLANNING*/
1. Use google_search to find the current CEO of Anthropic.
2. Return the name in the final answer.
/*REASONING*/
Executing step 1.
/*ACTION*/
<function_call>google_search(query="Anthropic CEO 2025")</function_call>
/*REASONING*/
Found: Dario Amodei is CEO.
/*FINAL_ANSWER*/
The CEO of Anthropic is Dario Amodei.
```

**Tag semantics (from `planners/plan_re_act_planner.py`):**

| Tag | Constant | Purpose | Shown to user |
|---|---|---|---|
| `/*PLANNING*/` | `PLANNING_TAG` | Initial plan in natural language | No — marked as `thought=True` |
| `/*REPLANNING*/` | `REPLANNING_TAG` | Revised plan when initial plan fails | No — marked as `thought=True` |
| `/*REASONING*/` | `REASONING_TAG` | Inline reasoning between tool calls | No — marked as `thought=True` |
| `/*ACTION*/` | `ACTION_TAG` | Function call block | The function call is executed |
| `/*FINAL_ANSWER*/` | `FINAL_ANSWER_TAG` | The response visible to the user | Yes |

**Processing pipeline (verified source):**

1. `build_planning_instruction(...)` — prepends a comprehensive NL instruction block to the system prompt covering: plan format, reasoning tags, action tags, tool-use rules, and the final-answer tag requirement.
2. After the model responds, `process_planning_response(...)` splits the response parts:
   - Text before the first `/*FINAL_ANSWER*/` tag → marked `thought=True` (never reaches the user).
   - Text after `/*FINAL_ANSWER*/` → returned as the visible reply.
   - Function calls → preserved and executed normally.

**Multi-tool research agent:**

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import InMemoryRunner
from google.adk.apps import App
from google.adk.tools import google_search
from google.genai import types

async def get_stock_price(ticker: str) -> dict:
    """Get the latest price for a stock ticker.

    Args:
      ticker: Stock symbol, e.g. 'GOOG'.
    Returns:
      A dict with price and currency.
    """
    # In production: call a real market data API
    return {"ticker": ticker, "price": 175.32, "currency": "USD"}

async def get_company_news(company: str) -> dict:
    """Fetch recent news about a company.

    Args:
      company: Full company name.
    Returns:
      A dict with headlines list.
    """
    return {"company": company, "headlines": ["Q1 earnings beat", "New product launch"]}

analyst = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a financial analyst. Use the available tools to research "
        "companies and produce a concise investment brief."
    ),
    planner=PlanReActPlanner(),
    tools=[get_stock_price, get_company_news, google_search],
)

async def main():
    app = App(name="finance", root_agent=analyst)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="finance", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Give me a brief on Alphabet Inc.")]
        ),
    ):
        if event.is_final_response() and event.content:
            print("Brief:", "".join(p.text or "" for p in event.content.parts))

asyncio.run(main())
```

**When to use `PlanReActPlanner` vs `BuiltInPlanner`:**

| Planner | Requires thinking model | Latency overhead | Reasoning visible in traces | Cost |
|---|---|---|---|---|
| `BuiltInPlanner(thinking_config=...)` | Yes (Gemini 2.5 Pro/Flash only) | Low (native) | Yes (thought parts) | Thinking tokens charged separately |
| `PlanReActPlanner()` | No (any Gemini model) | Higher (extra output tokens) | Yes (thought parts) | Normal output token rate |

Use `PlanReActPlanner` when:
- You need structured planning on a non-thinking model (e.g. `gemini-2.0-flash`).
- You want the plan/reasoning explicitly tagged and capturable via callbacks.
- You need to debug multi-step tool chains — the `/*REASONING*/` blocks reveal the model's intent at each step.

Use `BuiltInPlanner` when:
- You're using Gemini 2.5 Pro or Flash (supports `ThinkingConfig`).
- Latency matters — native thinking is faster than generating reasoning tokens.
- You want to cap reasoning cost with `thinking_budget`.

### 4 — Reflection loop
`LoopAgent` (or the `Workflow` equivalent) with a single reflective `LlmAgent` that rewrites `state['draft']` each turn and escalates when satisfied. Use `include_contents="none"` on the reflective agent to avoid feeding the full history back each iteration.

### 5 — Parallel multi-try with a judge
`ParallelAgent` (or `Workflow` fan-out) of N candidate agents, followed by a "judge" `LlmAgent` that reads `state['candidate_1..N']` and picks the best. Pair each candidate's `output_key` to a distinct state slot.

## LangGraphAgent

`LangGraphAgent` wraps a compiled LangGraph `CompiledGraph` as a `BaseAgent`. Install prerequisite: `pip install langchain-core langgraph langchain-google-genai`.

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
graph = create_react_agent(llm, tools=[])

agent = LangGraphAgent(
    name="langgraph_agent",
    description="Answers questions using a LangGraph ReAct graph.",
    graph=graph,
    instruction="You are a helpful assistant.",
)
```

**Fields:** `graph: CompiledGraph` (required), `instruction: str` (injected as `SystemMessage` on the first turn only). All `BaseAgent` fields (`name`, `description`, `mode`, callbacks) also apply.

**Memory rules:** If `graph.checkpointer` is set, ADK sends only the latest user messages and LangGraph manages history via its checkpointer. If no checkpointer, ADK sends the full conversation for that agent.

> For detailed examples — multi-turn with `MemorySaver`, as a sub-agent in a multi-agent system — see [Class deep dives — vol. 2](./google_adk_class_deep_dives_v2/#2--langgraphagent).

## RemoteA2aAgent

`RemoteA2aAgent` calls a remote A2A-compatible agent over HTTP, exposing it as a local `BaseAgent`. See also [MCP & A2A](./mcp-and-a2a/).

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents import LlmAgent

remote = RemoteA2aAgent(
    name="remote_specialist",
    agent_card="https://specialist.internal/.well-known/agent.json",
    timeout=30.0,
)

root = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="For specialist tasks, delegate to 'remote_specialist'.",
    sub_agents=[remote],
)
```

> For full constructor reference and examples (signed requests, file-based cards, interceptors) — see [Class deep dives — vol. 2](./google_adk_class_deep_dives_v2/#1--remotea2aagent).

## Gotchas

- Setting `output_schema` **disables tool use** for that agent, including sub-agent transfer (`llm_agent.py:348-372`).
- `global_instruction` is deprecated at the agent level; use `GlobalInstructionPlugin` at the `App` level.
- A root `LlmAgent` must have `mode='chat'` or the runner auto-sets it; other modes are only valid inside a `Workflow`.
- `LoopAgent.run_live` is **not implemented** — `ParallelAgent.run_live` also raises `NotImplementedError`.
- When a sub-agent has no `model`, it inherits from the nearest ancestor `LlmAgent`. If the root also omits `model`, the default is resolved via `LlmAgent._default_model` (`gemini-2.5-flash`).
- Callables passed to `tools=` are wrapped as `FunctionTool(func=callable)` automatically. Pass an explicit `FunctionTool` only when you need `require_confirmation=`.
- `LangGraphAgent` requires `langchain-core` and `langgraph` installed separately — they are not ADK dependencies.
- `RemoteA2aAgent` is `@a2a_experimental` — import paths and wire protocol may change in future minor releases.

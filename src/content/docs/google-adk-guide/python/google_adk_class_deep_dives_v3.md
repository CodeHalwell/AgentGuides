---
title: "Class deep dives — volume 3 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: AgentTool, BuiltInPlanner, PlanReActPlanner, InMemorySessionService, VertexAiRagMemoryService, UnsafeLocalCodeExecutor, BuiltInCodeExecutor, LiteLlm, AgentEvaluator, and the deprecated shell agents."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 3"
  order: 62
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and code example is taken directly from the installed package source.

| Class | Module | Status |
|---|---|---|
| `AgentTool` | `google.adk.tools.agent_tool` | Stable |
| `BuiltInPlanner` | `google.adk.planners.built_in_planner` | Stable |
| `PlanReActPlanner` | `google.adk.planners.plan_re_act_planner` | Stable |
| `InMemorySessionService` | `google.adk.sessions.in_memory_session_service` | Stable |
| `VertexAiRagMemoryService` | `google.adk.memory.vertex_ai_rag_memory_service` | Stable |
| `UnsafeLocalCodeExecutor` | `google.adk.code_executors.unsafe_local_code_executor` | Stable |
| `BuiltInCodeExecutor` | `google.adk.code_executors.built_in_code_executor` | Stable |
| `LiteLlm` | `google.adk.models.lite_llm` | Stable |
| `AgentEvaluator` | `google.adk.evaluation.agent_evaluator` | Stable |
| `LoopAgent` / `ParallelAgent` / `SequentialAgent` | `google.adk.agents` | **Deprecated → `Workflow`** |

---

## 1 · `AgentTool`

`google.adk.tools.agent_tool.AgentTool` wraps any `BaseAgent` so an `LlmAgent` can call it as a function tool. The wrapped agent spins up its own isolated `Runner` (backed by an in-memory session) for the duration of the tool call, then returns its final text response (or a validated Pydantic object when `output_schema` is set on the wrapped agent).

### Constructor (verified `agent_tool.py:116-131`)

```python
AgentTool(
    agent: BaseAgent,
    skip_summarization: bool = False,
    *,
    include_plugins: bool = True,
    propagate_grounding_metadata: bool = False,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `agent` | required | Any `BaseAgent` (or `Workflow`). `agent.name` becomes the tool name; `agent.description` becomes the tool description |
| `skip_summarization` | `False` | When `True` sets `actions.skip_summarization = True`, telling the parent `LlmAgent` not to summarise the tool result |
| `include_plugins` | `True` | Propagate the parent runner's plugins down into the wrapped agent's runner |
| `propagate_grounding_metadata` | `False` | When `True`, stores the final grounding metadata in `state['temp:_adk_grounding_metadata']` for the parent agent to inspect |

### How it works internally

When the model calls the tool the `run_async` method (`agent_tool.py:196-253`):

1. Creates a new `InMemorySessionService` + `InMemoryMemoryService` scoped to the child invocation.
2. Copies all **non-internal** state keys (those not starting with `_adk`) from the parent session into the child session — so the wrapped agent inherits the parent's state.
3. Runs the wrapped agent via its own `Runner`, collecting all events.
4. Forwards any `state_delta` events from the child back to the parent's `ToolContext`.
5. Returns the merged text of the **last content event** (or a Pydantic-validated object when `output_schema` is configured).
6. Calls `runner.close()` to clean up MCP connections and avoid `asyncio` scope warnings.

### Example 1 — basic agent-as-tool

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import InMemoryRunner

# Sub-agent that specialises in translation
translator = LlmAgent(
    name="translator",
    model="gemini-2.5-flash",
    description="Translates text into the requested language.",
    instruction=(
        "You receive a JSON request like "
        '{"text": "...", "target_language": "..."}. '
        "Return only the translated text."
    ),
    mode="single_turn",
)

# Orchestrator calls the translator as a tool
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful assistant. When asked to translate, "
        "call the `translator` tool. For everything else answer directly."
    ),
    tools=[AgentTool(agent=translator)],
)

async def main():
    runner = InMemoryRunner(agent=orchestrator, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        'Please translate "Hello, world!" into French.',
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — structured input/output schema

When the wrapped agent declares `input_schema` and `output_schema`, the `FunctionDeclaration` for the tool is built from those Pydantic models, giving the model strong type hints.

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

class SentimentRequest(BaseModel):
    text: str
    language: str = "en"

class SentimentResult(BaseModel):
    label: str       # "positive" | "neutral" | "negative"
    confidence: float

sentiment_agent = LlmAgent(
    name="sentiment_analyzer",
    model="gemini-2.5-flash",
    description="Analyzes the sentiment of a piece of text.",
    instruction=(
        "Given the input JSON, reply with JSON matching exactly: "
        '{"label": "positive"|"neutral"|"negative", "confidence": 0.0-1.0}'
    ),
    input_schema=SentimentRequest,
    output_schema=SentimentResult,
    mode="single_turn",
)

# The parent agent gets a tool whose parameter schema mirrors SentimentRequest
# and whose return value is validated as SentimentResult.
orchestrator = LlmAgent(
    name="report_generator",
    model="gemini-2.5-flash",
    instruction="Use sentiment_analyzer to score the review, then write a brief summary.",
    tools=[AgentTool(agent=sentiment_agent)],
)
```

### Example 3 — state sharing between parent and child

The child agent inherits a snapshot of the parent's non-internal state, and any `state_delta` emitted by the child propagates back to the parent `ToolContext`.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import InMemoryRunner

def get_user_tier(tool_context: ToolContext) -> str:
    """Returns the current user's subscription tier from session state."""
    return tool_context.state.get("user_tier", "free")

# The child can read state that the parent set
pricing_agent = LlmAgent(
    name="pricing_advisor",
    model="gemini-2.5-flash",
    description="Returns the correct pricing for the current user tier.",
    instruction=(
        "Call get_user_tier to find out if the user is 'free' or 'pro', "
        "then quote the matching price: free=$0/mo, pro=$29/mo."
    ),
    tools=[get_user_tier],
    mode="single_turn",
)

orchestrator = LlmAgent(
    name="sales_bot",
    model="gemini-2.5-flash",
    instruction="Answer sales questions. For pricing, delegate to pricing_advisor.",
    tools=[AgentTool(agent=pricing_agent)],
)

async def main():
    runner = InMemoryRunner(agent=orchestrator, app_name="demo")
    # Set user tier in session state before the conversation
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1",
        state={"user_tier": "pro"},
    )
    events = await runner.run_debug(
        "How much does the pro plan cost?",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 4 — skip summarization for large payloads

When the wrapped agent returns a large blob (e.g. a data extraction result), instruct the parent to forward it verbatim instead of asking the model to summarise it.

```python
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

data_extractor = LlmAgent(
    name="data_extractor",
    model="gemini-2.5-flash",
    description="Extracts structured data from documents.",
    instruction="Extract all tables from the document and return them as JSON.",
    mode="single_turn",
)

pipeline = LlmAgent(
    name="pipeline",
    model="gemini-2.5-flash",
    instruction=(
        "Use data_extractor to pull out the tables. "
        "Then analyse the extracted data and answer the user's question."
    ),
    # skip_summarization=True means the model receives the raw JSON, not an LLM summary of it
    tools=[AgentTool(agent=data_extractor, skip_summarization=True)],
)
```

### `AgentToolConfig` — YAML/JSON config support

`AgentTool` can be declared in an agent config file using `AgentToolConfig`:

```yaml
# agent_config.yaml
tools:
  - type: agent_tool
    agent:
      ref: ./translator_agent.yaml
    skip_summarization: false
    include_plugins: true
```

```python
# Programmatic equivalent
from google.adk.tools.agent_tool import AgentToolConfig, AgentTool

config = AgentToolConfig(
    agent={"ref": "./translator_agent.yaml"},
    skip_summarization=False,
    include_plugins=True,
)
```

---

## 2 · `BuiltInPlanner`

`google.adk.planners.built_in_planner.BuiltInPlanner` enables **Gemini's native thinking** capability for an `LlmAgent`. It injects a `ThinkingConfig` into every `LlmRequest` before the model is called, so the model produces explicit reasoning traces (which appear as `thought=True` parts in the response).

### Constructor (verified `built_in_planner.py:52`)

```python
BuiltInPlanner(*, thinking_config: types.ThinkingConfig)
```

`ThinkingConfig` lives in `google.genai.types`. The two main fields are:

| Field | Type | Purpose |
|---|---|---|
| `thinking_budget` | `int \| None` | Maximum tokens the model may spend thinking (`None` = dynamic) |
| `include_thoughts` | `bool` | When `True`, thought tokens are included in the response parts with `part.thought == True` |

### Attaching to an agent

Pass an instance to `LlmAgent.planner=`:

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

agent = LlmAgent(
    name="deep_thinker",
    model="gemini-2.5-flash",   # Must be Gemini 2.x+ (thinking-capable model)
    instruction="Solve the problem step by step. Show your reasoning.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            thinking_budget=8192,      # Up to 8 k tokens for reasoning
            include_thoughts=True,     # Return thought tokens in the event stream
        )
    ),
)
```

### What it does at runtime (verified `built_in_planner.py:57-80`)

`BuiltInPlanner.apply_thinking_config(llm_request)` is called by the LLM flow before sending a request. It:

1. Creates `llm_request.config` if `None`.
2. Sets `llm_request.config.thinking_config` to the value passed at construction.
3. Logs a `DEBUG` warning if the request already had a `thinking_config` (overwrite notice).

`build_planning_instruction()` and `process_planning_response()` both return `None` — the planner delegates all reasoning to the model itself; no instruction injection or response post-processing is needed.

### Example 1 — math reasoning with visible thoughts

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="math_solver",
    model="gemini-2.5-flash",
    instruction="Solve the problem. Be precise.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            thinking_budget=4096,
            include_thoughts=True,
        )
    ),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="If 3x² + 5x - 2 = 0, find x.")],
        ),
    ):
        if event.content:
            for part in event.content.parts:
                if part.thought:
                    print("[THOUGHT]", part.text[:120], "...")
                elif part.text:
                    print("[ANSWER]", part.text)

asyncio.run(main())
```

### Example 2 — minimal thinking budget (fast, cheap)

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

# Use a small budget to improve accuracy without spending too many tokens
agent = LlmAgent(
    name="quick_reasoner",
    model="gemini-2.5-flash",
    instruction="Answer concisely.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            thinking_budget=1024,    # Low budget: quick sanity-check level thinking
            include_thoughts=False,  # Suppress thought tokens in the stream
        )
    ),
)
```

### Example 3 — dynamic budget (`None`)

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

# Let Gemini decide how much to think based on query complexity
agent = LlmAgent(
    name="adaptive_thinker",
    model="gemini-2.5-flash",
    instruction="Think as hard as needed.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=None)
    ),
)
```

### Combining `BuiltInPlanner` with `generate_content_config`

If the agent already has a `thinking_config` in `generate_content_config`, `BuiltInPlanner` **overwrites it** (with a debug log). Prefer setting the thinking config exclusively through the planner to avoid confusion.

```python
# Avoid: mixing both
agent = LlmAgent(
    name="confused",
    model="gemini-2.5-flash",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=512),  # will be overwritten
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=4096),  # wins
    ),
)
```

---

## 3 · `PlanReActPlanner`

`google.adk.planners.plan_re_act_planner.PlanReActPlanner` implements **Plan-Re-Act** — a structured reasoning pattern that does *not* require a thinking-capable model. It enforces a response format using XML-like tags so that reasoning and actions are cleanly separated.

### Constructor (verified `plan_re_act_planner.py:32`)

```python
PlanReActPlanner()   # No arguments required
```

### Response format the planner enforces

The planner instructs the model to produce responses in this structure:

| Tag | Role |
|---|---|
| `/*PLANNING*/` | High-level plan before any tool calls |
| `/*REPLANNING*/` | Revised plan if the initial plan fails |
| `/*REASONING*/` | Interleaved reasoning between tool calls |
| `/*ACTION*/` | Wraps each tool call block |
| `/*FINAL_ANSWER*/` | The final answer to the user |

All content before `/*FINAL_ANSWER*/` is marked as `part.thought = True` by `process_planning_response`, making those parts invisible to the user by default.

### How it works at runtime (verified `plan_re_act_planner.py:47-100`)

1. **`build_planning_instruction()`** injects a ~500-word system instruction explaining the tag format, tool usage rules, and planning requirements into the LLM request.
2. **`process_planning_response()`** post-processes each LLM turn:
   - Splits the response at `/*FINAL_ANSWER*/` — content before is marked as a thought.
   - Extracts the first group of function calls, skipping empty-name function calls.
   - Strips trailing non-function-call parts after the first function call group (so the model cannot interleave tool calls with prose).

### Example 1 — attaching `PlanReActPlanner` to an agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import InMemoryRunner

def search_database(query: str) -> str:
    """Search the internal product database."""
    # Simulated result
    return f"Found 3 items matching '{query}': Widget A, Widget B, Widget C."

def get_item_price(item_name: str) -> float:
    """Get the current price of a product."""
    prices = {"Widget A": 9.99, "Widget B": 14.99, "Widget C": 24.99}
    return prices.get(item_name, 0.0)

agent = LlmAgent(
    name="shop_assistant",
    model="gemini-2.5-flash",  # Works with non-thinking models
    instruction="Help customers find products and their prices.",
    tools=[search_database, get_item_price],
    planner=PlanReActPlanner(),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What widgets do you have and how much do they cost?",
        user_id="u1", session_id="s1",
    )
    # Only the FINAL_ANSWER section is shown; planning/reasoning parts are hidden
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — exposing thought parts for debugging

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="debuggable_agent",
    model="gemini-2.5-flash",
    instruction="Use tools to answer questions.",
    tools=[],  # add your tools here
    planner=PlanReActPlanner(),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What is the capital of France?")]
        ),
    ):
        if event.content:
            for part in event.content.parts:
                if part.thought:
                    # Captures /*PLANNING*/, /*REASONING*/, /*ACTION*/ blocks
                    print("🧠", part.text[:200])
                elif part.text:
                    print("💬", part.text)

asyncio.run(main())
```

### `BuiltInPlanner` vs `PlanReActPlanner` — when to use which

| | `BuiltInPlanner` | `PlanReActPlanner` |
|---|---|---|
| Requires thinking model? | Yes (Gemini 2.x+) | No — works with any model |
| Reasoning style | Native model thinking | Structured tags in text |
| Thought tokens billed? | Yes | No (part of output tokens) |
| Response post-processing | None | Tag extraction + thought marking |
| Best for | Complex reasoning on Gemini | Any model, structured tool use |

---

## 4 · `InMemorySessionService`

`google.adk.sessions.in_memory_session_service.InMemorySessionService` is the reference implementation of `BaseSessionService` backed by nested Python dicts. It is **thread-safe** (uses a single copy per operation) and **not suitable for multi-process production** — but it is perfect for tests, local development, and unit-testing agent behaviour.

### Constructor (verified `in_memory_session_service.py:67`)

```python
InMemorySessionService()
```

The service maintains three internal maps:
- `self.sessions[app_name][user_id][session_id]` — `Session` objects
- `self.user_state[app_name][user_id]` — user-scoped state, keyed by app
- `self.app_state[app_name]` — app-scoped state

### State scoping (source: `in_memory_session_service.py:167-181`)

ADK has three state scopes that share a single `session.state` dict at read time:

| Scope | Key prefix | Lifetime |
|---|---|---|
| Session | *(no prefix)* | Current session only |
| User | `user:` | All sessions for this user within the app |
| App | `app:` | All sessions in the entire app |
| Temporary | `temp:` | Current invocation only; not persisted |

When you call `create_session(state={"user:name": "Alice", "theme": "dark"})`, `InMemorySessionService` extracts `user:name` → `user_state`, `theme` → session-local state. On `get_session()`, `_merge_state()` reconstructs the unified view.

### Key methods

```python
# Create
session = await svc.create_session(
    app_name="my_app",
    user_id="u123",
    state={"user:name": "Alice", "app:version": "2.0", "greeting": "Hello"},
    session_id="my-session-id",  # optional; auto-generated UUID if omitted
)

# Read (with optional slicing)
from google.adk.sessions.base_session_service import GetSessionConfig

session = await svc.get_session(
    app_name="my_app",
    user_id="u123",
    session_id="my-session-id",
    config=GetSessionConfig(
        num_recent_events=10,       # Only last 10 events
        after_timestamp=1700000000, # Events after this UNIX timestamp
    ),
)

# List all sessions for a user (events are stripped for efficiency)
response = await svc.list_sessions(app_name="my_app", user_id="u123")
for s in response.sessions:
    print(s.id, s.last_update_time)

# Delete
await svc.delete_session(app_name="my_app", user_id="u123", session_id="s1")
```

### Example 1 — full session lifecycle

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

svc = InMemorySessionService()
agent = LlmAgent(name="chat", model="gemini-2.5-flash", instruction="Be concise.")
runner = Runner(
    app_name="myapp",
    agent=agent,
    session_service=svc,
    memory_service=InMemoryMemoryService(),
)

async def main():
    # Create with pre-populated state
    s = await svc.create_session(
        app_name="myapp",
        user_id="u1",
        state={
            "user:preferred_name": "Alice",   # persists across all Alice's sessions
            "app:motd": "Welcome to v2!",     # visible to every user
            "topic": "Python",                 # session-local
        },
    )
    print("Session created:", s.id)
    print("State:", dict(s.state))
    # → {'user:preferred_name': 'Alice', 'app:motd': 'Welcome to v2!', 'topic': 'Python'}

    # Run the agent
    async for event in runner.run_async(
        user_id="u1",
        session_id=s.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Hi, what is my topic?")]
        ),
    ):
        if event.content and not event.partial:
            print("Agent:", event.content.parts[0].text)

    # Inspect the session after the turn
    s2 = await svc.get_session(app_name="myapp", user_id="u1", session_id=s.id)
    print("Events logged:", len(s2.events))

asyncio.run(main())
```

### Example 2 — cross-session user state

```python
import asyncio
from google.adk.sessions import InMemorySessionService

svc = InMemorySessionService()

async def main():
    # Session A — set a user preference
    s_a = await svc.create_session(
        app_name="app", user_id="u1",
        state={"user:language": "French"},
    )
    print("S_A language:", s_a.state["user:language"])  # French

    # Session B — same user; preference is automatically inherited
    s_b = await svc.create_session(app_name="app", user_id="u1")
    s_b_fetched = await svc.get_session(
        app_name="app", user_id="u1", session_id=s_b.id
    )
    print("S_B language:", s_b_fetched.state.get("user:language"))  # French

    # A different user does NOT inherit u1's preference
    s_c = await svc.create_session(app_name="app", user_id="u2")
    s_c_fetched = await svc.get_session(
        app_name="app", user_id="u2", session_id=s_c.id
    )
    print("S_C language:", s_c_fetched.state.get("user:language"))  # None

asyncio.run(main())
```

### Example 3 — event slicing with `GetSessionConfig`

```python
import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
import time

svc = InMemorySessionService()

async def main():
    s = await svc.create_session(app_name="app", user_id="u1")
    # (In a real scenario the runner appends events; here we omit that for brevity)

    # Only retrieve the last 5 events
    recent = await svc.get_session(
        app_name="app", user_id="u1", session_id=s.id,
        config=GetSessionConfig(num_recent_events=5),
    )
    print("Recent events:", len(recent.events))  # ≤ 5

    # Only events after a certain point in time
    cutoff = time.time() - 3600  # Last hour
    hour = await svc.get_session(
        app_name="app", user_id="u1", session_id=s.id,
        config=GetSessionConfig(after_timestamp=cutoff),
    )
    print("Hour events:", len(hour.events))

asyncio.run(main())
```

---

## 5 · `VertexAiRagMemoryService`

`google.adk.memory.vertex_ai_rag_memory_service.VertexAiRagMemoryService` stores conversation history in a **Vertex AI RAG corpus** and retrieves it with vector similarity search. Unlike `InMemoryMemoryService` (keyword-only), this service provides semantic search across sessions.

### Constructor (verified `vertex_ai_rag_memory_service.py:95-127`)

```python
VertexAiRagMemoryService(
    rag_corpus: Optional[str] = None,
    similarity_top_k: Optional[int] = None,
    vector_distance_threshold: float = 10,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `rag_corpus` | `None` | Full resource name `projects/{p}/locations/{l}/ragCorpora/{id}` or just the numeric ID |
| `similarity_top_k` | `None` | Number of chunks to retrieve (model default when `None`) |
| `vector_distance_threshold` | `10` | Only return results with vector distance below this threshold; lower = stricter |

**Prerequisites**: `pip install google-cloud-aiplatform[rag]` and a RAG corpus created in Vertex AI.

### How sessions are stored (verified `vertex_ai_rag_memory_service.py:133-178`)

`add_session_to_memory(session)` serialises each event as a JSON line:
```json
{"author": "model", "timestamp": 1748000000.0, "text": "The capital of France is Paris."}
```
and uploads the file to the RAG corpus. The `display_name` of the uploaded file encodes `app_name`, `user_id`, and `session_id` in URL-safe base64 — this is how the service filters results by user on retrieval.

### Example 1 — basic setup with RAG memory

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.memory import VertexAiRagMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from google.genai import types

# Set up Google Cloud auth
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

memory_service = VertexAiRagMemoryService(
    rag_corpus="projects/my-gcp-project/locations/us-central1/ragCorpora/123456789",
    similarity_top_k=5,
    vector_distance_threshold=0.8,  # Lower = higher similarity required
)

agent = LlmAgent(
    name="remembering_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful assistant with long-term memory. "
        "Use `load_memory` to recall relevant past conversations."
    ),
    tools=[load_memory],
)

runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=memory_service,
)

async def main():
    session = await runner.session_service.create_session(
        app_name="my_app", user_id="alice", session_id="session-001"
    )

    # First turn
    async for event in runner.run_async(
        user_id="alice", session_id="session-001",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="My favourite colour is blue.")],
        ),
    ):
        pass

    # Save this session to RAG memory for future recall
    session_obj = await runner.session_service.get_session(
        app_name="my_app", user_id="alice", session_id="session-001"
    )
    await runner.memory_service.add_session_to_memory(session_obj)

    # Later — new session, Alice asks about her favourite colour
    session2 = await runner.session_service.create_session(
        app_name="my_app", user_id="alice", session_id="session-002"
    )
    async for event in runner.run_async(
        user_id="alice", session_id="session-002",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Do you remember my favourite colour?")],
        ),
    ):
        if event.content and not event.partial:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — incremental memory updates (add latest turn only)

```python
import asyncio
from google.adk.memory import VertexAiRagMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.genai import types

memory = VertexAiRagMemoryService(
    rag_corpus="my-corpus-id",
    similarity_top_k=3,
)
svc = InMemorySessionService()

async def save_latest_turn(runner: Runner, session_id: str, user_id: str) -> None:
    """Save only the most recent user+model exchange to memory."""
    session = await svc.get_session(
        app_name="my_app", user_id=user_id, session_id=session_id
    )
    if session and len(session.events) >= 2:
        # Last two events: user message + model response
        await memory.add_events_to_memory(
            app_name="my_app",
            user_id=user_id,
            events=session.events[-2:],
            session_id=session_id,
        )
```

### Example 3 — explicit memory injection

```python
import asyncio
from google.adk.memory import VertexAiRagMemoryService
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

memory = VertexAiRagMemoryService(rag_corpus="my-corpus-id")

async def add_facts(user_id: str) -> None:
    """Seed the memory with explicit facts (not derived from a session)."""
    await memory.add_memory(
        app_name="my_app",
        user_id=user_id,
        memories=[
            MemoryEntry(
                content=types.Content(
                    parts=[types.Part(text="The user is a vegetarian.")]
                ),
                author="system",
            ),
            MemoryEntry(
                content=types.Content(
                    parts=[types.Part(text="The user's timezone is UTC+9 (Tokyo).")]
                ),
                author="system",
            ),
        ],
    )
```

---

## 6 · `UnsafeLocalCodeExecutor`

`google.adk.code_executors.unsafe_local_code_executor.UnsafeLocalCodeExecutor` runs Python code in a **spawned subprocess** on the local machine. It is the simplest executor — no cloud dependencies — but is *unsafe* in the sense that it executes untrusted code on your host. Use only in controlled, sandboxed environments.

### Constructor (verified `unsafe_local_code_executor.py:55-65`)

```python
UnsafeLocalCodeExecutor(
    *,
    timeout_seconds: float = 30.0,  # inherited from BaseCodeExecutor
    # stateful and optimize_data_file are frozen=False on the base but frozen=True here
)
```

| Parameter | Default | Notes |
|---|---|---|
| `timeout_seconds` | `30.0` | Maximum wall-clock seconds per code execution |
| `stateful` | `False` (frozen) | Cannot be set to `True`; each run gets a fresh `globals` dict |
| `optimize_data_file` | `False` (frozen) | Cannot be set to `True` |

### How it executes code (verified `unsafe_local_code_executor.py:90-125`)

1. Spawns a new **`multiprocessing.Process`** (using `spawn` context — not `fork`).
2. Redirects `stdout` to a `StringIO` buffer inside the child process.
3. Calls `exec(code, globals, globals)` where `globals` is a fresh empty dict.
4. Puts `(stdout_text, error_traceback_or_None)` into a `multiprocessing.Queue`.
5. Parent waits up to `timeout_seconds`; on timeout it terminates the process and returns an error string.

### Attaching to an agent

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor

agent = LlmAgent(
    name="code_runner",
    model="gemini-2.5-flash",
    instruction=(
        "You are a Python coding assistant. "
        "Write and run Python code to solve problems."
    ),
    code_executor=UnsafeLocalCodeExecutor(timeout_seconds=10.0),
)
```

### Example 1 — data analysis agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction=(
        "Write Python code to answer data questions. "
        "Use standard library only (no pandas/numpy — this is a minimal environment)."
    ),
    code_executor=UnsafeLocalCodeExecutor(timeout_seconds=15.0),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Calculate the mean, median, and standard deviation of [4, 8, 15, 16, 23, 42].",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — `__main__` guard support

The executor injects `__name__ = '__main__'` when the code contains an `if __name__ == '__main__':` guard, so typical Python scripts work out-of-the-box:

```python
from google.adk.code_executors.unsafe_local_code_executor import (
    _prepare_globals, UnsafeLocalCodeExecutor
)

code = '''
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("world"))
'''

# _prepare_globals injects __name__ = '__main__' so the guarded block runs
g = {}
_prepare_globals(code, g)
assert g.get('__name__') == '__main__'
```

### Example 3 — timeout handling

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor

# The agent will handle timeout errors gracefully: the executor returns
# "Code execution timed out after X seconds." as stderr; the LLM sees this
# and can retry or inform the user.
agent = LlmAgent(
    name="safe_sandbox",
    model="gemini-2.5-flash",
    instruction="Run user-provided code snippets safely.",
    code_executor=UnsafeLocalCodeExecutor(timeout_seconds=5.0),
)
```

---

## 7 · `BuiltInCodeExecutor`

`google.adk.code_executors.built_in_code_executor.BuiltInCodeExecutor` delegates Python execution to **Gemini's native code execution tool** (`types.ToolCodeExecution`). The model generates and runs code server-side inside Google's sandbox — no code ever runs on your machine.

### Constructor (verified `built_in_code_executor.py:28`)

```python
BuiltInCodeExecutor()  # No parameters
```

### Requirements

- Model must be **Gemini 2.0 or above** (enforced at `process_llm_request` time; raises `ValueError` otherwise).
- The executor injects `types.Tool(code_execution=types.ToolCodeExecution())` into `llm_request.config.tools` before each LLM call.

### Example 1 — attaching to an agent

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor

agent = LlmAgent(
    name="gemini_coder",
    model="gemini-2.5-flash",   # Must be Gemini 2.x+
    instruction=(
        "Write and run Python code using the built-in code execution tool. "
        "Show your work and explain the output."
    ),
    code_executor=BuiltInCodeExecutor(),
)
```

### Example 2 — full pipeline with output capture

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="data_scientist",
    model="gemini-2.5-flash",
    instruction=(
        "Use code execution to compute exact answers. "
        "Always verify numerical results by running code."
    ),
    code_executor=BuiltInCodeExecutor(),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=(
                "Plot the first 20 Fibonacci numbers and return the 20th value."
            ))],
        ),
    ):
        if event.content and not event.partial:
            for part in event.content.parts:
                if part.text:
                    print(part.text)
                elif part.code_execution_result:
                    print("[CODE OUTPUT]", part.code_execution_result.output)
                elif part.executable_code:
                    print("[CODE]", part.executable_code.code[:100], "...")

asyncio.run(main())
```

### `BuiltInCodeExecutor` vs `UnsafeLocalCodeExecutor` vs `VertexAiCodeExecutor`

| | `BuiltInCodeExecutor` | `UnsafeLocalCodeExecutor` | `VertexAiCodeExecutor` |
|---|---|---|---|
| Execution environment | Gemini server-side | Spawned local subprocess | Vertex AI Code Interpreter Extension |
| Requires cloud? | Yes (Gemini API) | No | Yes (Vertex AI) |
| Stateful sessions? | No | No | Yes (optional) |
| File I/O support? | Limited | No | Yes (CSV, images) |
| Safe? | Yes | No | Yes |
| Model requirement | Gemini 2.0+ | Any | Any |

---

## 8 · `LiteLlm`

`google.adk.models.lite_llm.LiteLlm` is a `BaseLlm` subclass that routes LLM calls through the **[LiteLLM](https://github.com/BerriAI/litellm)** library, giving ADK agents access to 100+ model providers (OpenAI, Anthropic, Azure, AWS Bedrock, Cohere, Mistral, Ollama, etc.) with a single interface.

**Installation**: `pip install 'google-adk[extensions]'`

### Constructor (verified `lite_llm.py:2189-2210`)

```python
LiteLlm(
    model: str,       # LiteLLM model string, e.g. "openai/gpt-4o"
    **kwargs,         # Passed to litellm.acompletion() on every call
)
```

The `model` string follows LiteLLM's `{provider}/{model-name}` convention.

| Provider | Example model string |
|---|---|
| OpenAI | `"openai/gpt-4o"`, `"openai/gpt-4o-mini"` |
| Anthropic | `"anthropic/claude-opus-4-7"` |
| Azure OpenAI | `"azure/my-deployment-name"` |
| AWS Bedrock | `"bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"` |
| Vertex AI (non-Gemini) | `"vertex_ai/claude-3-7-sonnet@20250219"` |
| Ollama (local) | `"ollama/llama3.2"` |
| Mistral | `"mistral/mistral-large-latest"` |
| Cohere | `"cohere/command-r-plus"` |

> **Note**: For Gemini models, use ADK's native `GoogleLlm` (pass a plain model string like `"gemini-2.5-flash"` to `LlmAgent`). Using `LiteLlm(model="gemini/...")` will log a deprecation warning and work, but the native path is preferred.

### Example 1 — OpenAI GPT-4o

```python
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

os.environ["OPENAI_API_KEY"] = "sk-..."

agent = LlmAgent(
    name="openai_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    instruction="You are a helpful assistant.",
    description="An agent powered by GPT-4o.",
)
```

### Example 2 — Anthropic Claude on Vertex AI

```python
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

os.environ["VERTEXAI_PROJECT"] = "my-gcp-project"
os.environ["VERTEXAI_LOCATION"] = "us-east5"

agent = LlmAgent(
    name="claude_agent",
    model=LiteLlm(model="vertex_ai/claude-opus-4-7@20250514"),
    instruction="You are an expert code reviewer.",
)
```

### Example 3 — AWS Bedrock (Claude)

```python
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

os.environ["AWS_ACCESS_KEY_ID"] = "AKIA..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_REGION_NAME"] = "us-east-1"

agent = LlmAgent(
    name="bedrock_agent",
    model=LiteLlm(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    ),
    instruction="Answer concisely.",
)
```

### Example 4 — local Ollama model

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Assumes `ollama serve` is running locally on port 11434
agent = LlmAgent(
    name="local_agent",
    model=LiteLlm(model="ollama/llama3.2"),
    instruction="You are a helpful local assistant.",
)
```

### Example 5 — Azure OpenAI with custom base URL

```python
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "https://my-resource.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-02-01"

agent = LlmAgent(
    name="azure_agent",
    model=LiteLlm(
        model="azure/my-gpt4o-deployment",
        # Extra kwargs forwarded to litellm.acompletion()
        api_base="https://my-resource.openai.azure.com/",
        api_version="2024-02-01",
    ),
    instruction="You are a corporate assistant.",
)
```

### Example 6 — multi-provider agent orchestration

Mix models from different providers in the same agent graph:

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.workflow import Workflow, node, START

# Each agent can run on a different LLM
brainstormer = LlmAgent(
    name="brainstormer",
    model=LiteLlm(model="openai/gpt-4o"),
    instruction="Generate 5 creative ideas for the given topic.",
    mode="single_turn",
)

critic = LlmAgent(
    name="critic",
    model="gemini-2.5-flash",  # Native Gemini; no LiteLlm needed
    instruction="Evaluate the 5 ideas and pick the best one with reasons.",
    mode="single_turn",
)

writer = LlmAgent(
    name="writer",
    model=LiteLlm(model="anthropic/claude-opus-4-7"),
    instruction="Write a 200-word pitch for the selected idea.",
    mode="single_turn",
)

pipeline = Workflow(
    name="idea_pipeline",
    edges=[(START, brainstormer, critic, writer)],
)
```

### `drop_params` — dropping unsupported parameters

Some providers reject parameters like `response_format` that they don't support. Pass `drop_params=True` to silently ignore unrecognised fields:

```python
agent = LlmAgent(
    name="cohere_agent",
    model=LiteLlm(model="cohere/command-r-plus", drop_params=True),
    instruction="Summarise the following text.",
)
```

---

## 9 · `AgentEvaluator`

`google.adk.evaluation.agent_evaluator.AgentEvaluator` is ADK's built-in testing harness. It runs an agent against a set of test cases (`EvalSet`) and scores the results against configurable metrics.

### Key methods (verified `agent_evaluator.py:97-280`)

| Method | Purpose |
|---|---|
| `AgentEvaluator.evaluate(agent_module, eval_dataset_file_path_or_dir, config)` | Primary entry point: load agent from module, load eval data, run, score |
| `AgentEvaluator.evaluate_eval_set(agent_module, eval_set, eval_config, ...)` | Lower-level: pass an `EvalSet` object directly |
| `AgentEvaluator.find_config_for_test_file(test_file)` | Auto-discovers `test_config.json` in the same directory as the test |

### `EvalConfig` structure

```python
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

config = EvalConfig(
    criteria={
        # Threshold (float): simple pass/fail at this score
        PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE: 0.8,
        # BaseCriterion: threshold + optional judge model options
        PrebuiltMetrics.RESPONSE_MATCH_SCORE: 0.5,
        # LLM-as-judge
        PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2: {
            "threshold": 0.7,
            "judge_model_options": {
                "judge_model": "gemini-2.5-flash",
                "num_samples": 3,
            },
        },
    },
)
```

### Available `PrebuiltMetrics`

| Metric | Measures |
|---|---|
| `TOOL_TRAJECTORY_AVG_SCORE` | Whether the agent called the expected tools in the expected order |
| `RESPONSE_EVALUATION_SCORE` | LLM-judged response quality |
| `RESPONSE_MATCH_SCORE` | Lexical similarity between actual and reference response |
| `SAFETY_V1` | Safety/harm evaluation |
| `FINAL_RESPONSE_MATCH_V2` | LLM-judged final answer quality (v2 rubric) |
| `HALLUCINATIONS_V1` | Detects factual hallucinations |
| `RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1` | Custom rubric-based final response quality |
| `RUBRIC_BASED_TOOL_USE_QUALITY_V1` | Custom rubric-based tool use quality |
| `MULTI_TURN_TASK_SUCCESS_V1` | Task completion in multi-turn conversations |
| `MULTI_TURN_TRAJECTORY_QUALITY_V1` | Multi-turn tool trajectory quality |
| `MULTI_TURN_TOOL_USE_QUALITY_V1` | Multi-turn tool call quality |

### Example 1 — pytest integration (recommended pattern)

**Directory structure:**
```
tests/
  my_agent_test.py
  eval_data/
    test_config.json
    test_cases.json
```

**`test_config.json`:**
```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.9,
    "response_match_score": 0.5
  }
}
```

**`test_cases.json`:**
```json
[
  {
    "name": "capital_city_lookup",
    "initial_session_state": {},
    "turns": [
      {
        "user_content": {"parts": [{"text": "What is the capital of Japan?"}], "role": "user"},
        "expected_tool_use": [
          {"tool_name": "search_wikipedia", "tool_input": {"query": "capital of Japan"}}
        ],
        "reference": "The capital of Japan is Tokyo."
      }
    ]
  }
]
```

**`my_agent_test.py`:**
```python
import asyncio
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

AGENT_MODULE = "my_package.agent"  # Module that defines root_agent


@pytest.mark.asyncio
async def test_capital_city_queries():
    """Verify tool usage and response quality."""
    await AgentEvaluator.evaluate(
        agent_module=AGENT_MODULE,
        eval_dataset_file_path_or_dir="tests/eval_data",
        config=AgentEvaluator.find_config_for_test_file(__file__),
    )
```

### Example 2 — programmatic eval with `EvalSet`

```python
import asyncio
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.genai import types as genai_types

# Build test cases in code
eval_cases = [
    EvalCase(
        eval_id="test_greet",
        conversation=[
            Invocation(
                user_content=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="Hello!")],
                ),
                expected_tool_use=[],
                reference="Hello! How can I help you today?",
            )
        ],
    ),
]

eval_set = EvalSet(eval_set_id="smoke_tests", eval_cases=eval_cases)

config = EvalConfig(
    criteria={PrebuiltMetrics.RESPONSE_MATCH_SCORE: 0.3}
)

async def run_eval():
    manager = InMemoryEvalSetsManager()
    await manager.create_eval_set(app_name="demo", eval_set_id="smoke_tests")
    for case in eval_cases:
        await manager.add_eval_case(app_name="demo", eval_set_id="smoke_tests", eval_case=case)

    await AgentEvaluator.evaluate_eval_set(
        agent_module="my_package.agent",
        eval_set=eval_set,
        eval_config=config,
        num_runs=1,
        print_detailed_results=True,
    )

asyncio.run(run_eval())
```

### Example 3 — rubric-based evaluation

```python
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

# Use a rubric score to measure response helpfulness on a 0–5 scale
config = EvalConfig(
    criteria={
        PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1: {
            "threshold": 3.0,   # Must score at least 3/5
            "judge_model_options": {
                "judge_model": "gemini-2.5-flash",
                "num_samples": 5,  # Average over 5 judge runs for stability
            },
        }
    }
)
```

---

## 10 · `LoopAgent`, `ParallelAgent`, `SequentialAgent` *(deprecated)*

These three **shell agents** were the primary orchestration primitives in ADK 1.x. In ADK 2.x they have been **deprecated in favour of `Workflow`** (`@deprecated` decorator on each class; verified in `loop_agent.py:52`, `parallel_agent.py:150`, `sequential_agent.py:48`).

> **Migration**: New code should use `Workflow` with `edges=`. See the [workflows page](./workflows/) for the full API.

They are documented here because a large body of existing ADK 1.x code uses them and they will continue to work until they are removed in a future major version.

### `SequentialAgent` — runs sub-agents one after another

```python
from google.adk.agents import LlmAgent, SequentialAgent  # deprecated
from google.adk.runners import InMemoryRunner
import asyncio

step1 = LlmAgent(
    name="extractor",
    model="gemini-2.5-flash",
    instruction="Extract all product names from the text. Return a JSON list.",
    mode="single_turn",
)
step2 = LlmAgent(
    name="enricher",
    model="gemini-2.5-flash",
    instruction="For each product name in the list, add a short description. Return JSON.",
    mode="single_turn",
)
step3 = LlmAgent(
    name="formatter",
    model="gemini-2.5-flash",
    instruction="Format the enriched product list as a Markdown table.",
    mode="single_turn",
)

# ⚠️ Deprecated — use Workflow instead (see migration below)
pipeline = SequentialAgent(
    name="product_pipeline",
    sub_agents=[step1, step2, step3],
)

async def main():
    runner = InMemoryRunner(agent=pipeline, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Products: Widget Pro, Gadget Plus, SuperTool 3000",
        user_id="u1", session_id="s1",
    )

asyncio.run(main())
```

**`Workflow` equivalent:**
```python
from google.adk.workflow import Workflow, START
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

pipeline = Workflow(
    name="product_pipeline",
    edges=[(START, step1, step2, step3)],
)
runner = InMemoryRunner(app=App(name="demo", root_agent=pipeline))
```

---

### `ParallelAgent` — runs sub-agents concurrently in isolated branches

Each sub-agent gets its own **branch context** (isolated `branch` ID) so their events don't interfere. On Python 3.11+, `asyncio.TaskGroup` is used; on 3.10, individual tasks with manual cancellation.

```python
from google.adk.agents import LlmAgent, ParallelAgent  # deprecated
from google.adk.runners import InMemoryRunner
import asyncio

# Three independent research agents running concurrently
tech_researcher = LlmAgent(
    name="tech_researcher",
    model="gemini-2.5-flash",
    instruction="Research recent breakthroughs in quantum computing.",
    mode="single_turn",
)
bio_researcher = LlmAgent(
    name="bio_researcher",
    model="gemini-2.5-flash",
    instruction="Research recent breakthroughs in gene therapy.",
    mode="single_turn",
)
space_researcher = LlmAgent(
    name="space_researcher",
    model="gemini-2.5-flash",
    instruction="Research recent breakthroughs in space exploration.",
    mode="single_turn",
)

# ⚠️ Deprecated — use Workflow with fan-out edges instead
team = ParallelAgent(
    name="research_team",
    sub_agents=[tech_researcher, bio_researcher, space_researcher],
)

async def main():
    runner = InMemoryRunner(agent=team, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Summarise the latest breakthroughs in your domain.",
        user_id="u1", session_id="s1",
    )

asyncio.run(main())
```

**`Workflow` equivalent (fan-out + join):**
```python
from google.adk.workflow import Workflow, JoinNode, START
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

join = JoinNode(name="join")

team = Workflow(
    name="research_team",
    edges=[
        (START, tech_researcher),
        (START, bio_researcher),
        (START, space_researcher),
        (tech_researcher, join),
        (bio_researcher, join),
        (space_researcher, join),
    ],
)
runner = InMemoryRunner(app=App(name="demo", root_agent=team))
```

---

### `LoopAgent` — runs sub-agents repeatedly until escalation or max iterations

`LoopAgent` runs its `sub_agents` list in order, then loops back to the start. It exits when:
1. Any sub-agent emits an event with `actions.escalate = True` (e.g. via the built-in `exit_loop` tool).
2. `max_iterations` is reached.

```python
from google.adk.agents import LlmAgent, LoopAgent  # deprecated
from google.adk.tools import exit_loop
from google.adk.runners import InMemoryRunner
import asyncio

generator = LlmAgent(
    name="generator",
    model="gemini-2.5-flash",
    instruction=(
        "Generate a one-sentence story idea. "
        "Keep a counter in state['iteration'] and increment it each turn."
    ),
    output_key="story_idea",
    mode="single_turn",
)

evaluator = LlmAgent(
    name="evaluator",
    model="gemini-2.5-flash",
    instruction=(
        "Read state['story_idea']. If it's interesting (score ≥ 8/10), "
        "call exit_loop to stop. Otherwise reply 'try again'."
    ),
    tools=[exit_loop],
    mode="single_turn",
)

# ⚠️ Deprecated — use Workflow with conditional routing instead
refiner = LoopAgent(
    name="story_refiner",
    sub_agents=[generator, evaluator],
    max_iterations=5,   # Safety cap
)

async def main():
    runner = InMemoryRunner(agent=refiner, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Generate an interesting sci-fi story idea.",
        user_id="u1", session_id="s1",
    )

asyncio.run(main())
```

**`Workflow` equivalent (loop until condition):**
```python
from google.adk.workflow import Workflow, node, DEFAULT_ROUTE, START
from google.adk.apps import App

@node
def check_quality(node_input: str) -> str:
    """Return 'done' if quality is good, 'retry' otherwise."""
    # In practice use an LlmAgent for this; simplified here
    return "done" if "interesting" in node_input.lower() else "retry"

loop = Workflow(
    name="story_refiner",
    edges=[
        (START, generator),
        (generator, check_quality),
        # Loop back to generator or exit
        (check_quality, {
            "done": None,    # Exit the workflow
            DEFAULT_ROUTE: generator,  # Loop
        }),
    ],
)
```

### Shell agent migration summary

| `v1.x` (deprecated) | `v2.x` replacement |
|---|---|
| `SequentialAgent(sub_agents=[a, b, c])` | `Workflow(edges=[(START, a, b, c)])` |
| `ParallelAgent(sub_agents=[a, b, c])` | `Workflow(edges=[(START, a), (START, b), (START, c), (a, join), ...])` |
| `LoopAgent(sub_agents=[a, b], max_iterations=5)` | `Workflow` with conditional back-edges and `max_iterations` in `Workflow` state or a loop exit node |

---

## Version notes

All source references verified against **google-adk==2.1.0** installed from PyPI (May 2026).

| Class | Source file | Key line refs |
|---|---|---|
| `AgentTool` | `tools/agent_tool.py` | `116-131` (constructor), `196-253` (run_async) |
| `BuiltInPlanner` | `planners/built_in_planner.py` | `52` (constructor), `57-80` (apply_thinking_config) |
| `PlanReActPlanner` | `planners/plan_re_act_planner.py` | `32` (constructor), `47-100` (process_planning_response) |
| `InMemorySessionService` | `sessions/in_memory_session_service.py` | `67` (constructor), `167-181` (_merge_state) |
| `VertexAiRagMemoryService` | `memory/vertex_ai_rag_memory_service.py` | `95-127` (constructor), `133-178` (add_session_to_memory) |
| `UnsafeLocalCodeExecutor` | `code_executors/unsafe_local_code_executor.py` | `55-65` (constructor), `90-125` (execute_code) |
| `BuiltInCodeExecutor` | `code_executors/built_in_code_executor.py` | `28` (constructor), `34-52` (process_llm_request) |
| `LiteLlm` | `models/lite_llm.py` | `2161` (class), `2189-2210` (constructor) |
| `AgentEvaluator` | `evaluation/agent_evaluator.py` | `97` (class), `108-185` (evaluate_eval_set) |
| `LoopAgent` | `agents/loop_agent.py` | `52` (`@deprecated`), `60-130` (_run_async_impl) |
| `ParallelAgent` | `agents/parallel_agent.py` | `150` (`@deprecated`), `160-230` (_run_async_impl) |
| `SequentialAgent` | `agents/sequential_agent.py` | `48` (`@deprecated`), `55-105` (_run_async_impl) |

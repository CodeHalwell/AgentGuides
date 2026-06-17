---
title: "Class deep dives — volume 20 (compaction internals, HITL utilities, workflow composition & backend detection)"
description: "Source-verified 2.2.0 deep dives: compaction pipeline internals (token-threshold + sliding-window algorithms), inject_session_state (async template substitution), run_llm_agent_as_node (single_turn/task/chat workflow modes), ToolConfig YAML DSL, RequestInput HITL model, HITL workflow utilities (request_input/auth builders), TaskResultAggregator (A2A state machine), retry internals (exponential backoff formula), GoogleLLMVariant + model-name utilities, SpannerAdminToolset (7 admin tools)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 20"
  order: 89
---

Source-verified against **google-adk==2.2.0** (installed in `/tmp/adk-env`, June 2026). Every field name, formula, and code example is drawn directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | Compaction pipeline internals | `google.adk.apps.compaction` | Stable (internal) |
| 2 | `inject_session_state` | `google.adk.utils.instructions_utils` | Stable |
| 3 | `run_llm_agent_as_node` + mode helpers | `google.adk.workflow._llm_agent_wrapper` | Stable (internal) |
| 4 | `ToolConfig` + `BaseToolConfig` + `ToolArgsConfig` | `google.adk.tools.tool_configs` | `@experimental` |
| 5 | `RequestInput` | `google.adk.events.request_input` | Stable |
| 6 | HITL workflow utilities | `google.adk.workflow.utils._workflow_hitl_utils` | Stable (internal) |
| 7 | `TaskResultAggregator` | `google.adk.a2a.executor.task_result_aggregator` | `@a2a_experimental` |
| 8 | Retry internals | `google.adk.workflow.utils._retry_utils` | Stable (internal) |
| 9 | `GoogleLLMVariant` + model-name utilities | `google.adk.utils.variant_utils` / `google.adk.utils.model_name_utils` / `google.adk.utils.output_schema_utils` | Stable (internal) |
| 10 | `SpannerAdminToolset` | `google.adk.tools.spanner.admin_toolset` | `@experimental` |

---

## 1 · Compaction pipeline internals

**Source:** `google.adk.apps.compaction`

`EventsCompactionConfig` on `App` is the public knob; the actual work happens in `google.adk.apps.compaction`, which is called by the runner after each invocation. The module implements two orthogonal compaction strategies that can run simultaneously.

### Token-threshold compaction

```
_run_compaction_for_token_threshold_config()
  ├── _latest_prompt_token_count()        # reads usage_metadata; falls back to chars÷4
  ├── _events_to_compact_for_token_threshold()
  │     ├── _latest_compaction_event()    # find most-recent non-subsumed compaction
  │     ├── candidate_events              # raw events after last compaction's end_timestamp
  │     ├── _safe_token_compaction_split_index()   # avoid orphaning retained tool responses
  │     ├── _truncate_events_before_pending_function_call()   # HITL guard 1
  │     └── _truncate_events_before_hitl_signal()             # HITL guard 2
  └── _summarize_events_with_trace()      # calls config.summarizer; wraps in OTel span
```

**`_latest_prompt_token_count`** — first tries `event.usage_metadata.prompt_token_count` (most recent event with a value); falls back to `_estimate_prompt_token_count` which calls `_contents._get_contents()` and applies `chars // 4`.

**`_safe_token_compaction_split_index`** — protects retained events from orphaned function responses. Iterates backwards collecting `function_response` IDs; if the matching `function_call` is in the compacted prefix, shifts the split earlier so call and response stay together.

**`_pending_function_call_ids`** — scans all session events, collecting all call IDs and all response IDs, then returns `call_ids - response_ids`. Events with pending calls must not be compacted (the LLM is still mid-tool-call).

**`_is_compaction_subsumed`** — if two compaction events cover identical ranges, the earlier one is subsumed by the later. Only the latest non-subsumed compaction event is considered "active."

### Sliding-window compaction

The sliding-window compaction triggers after `compaction_interval` new invocations since the last compaction, and looks back `overlap_size` invocations before the new block to create overlap.

**Worked example** from source docstring (compaction_interval=2, overlap_size=1):

| After invocation | New inv IDs | Action | Session state after |
|---|---|---|---|
| inv 2 | [1, 2] | First compaction: range [1,2] | Raw events [1,2] + CompactedEvent([1,2]) |
| inv 3 | [3] | Only 1 new — skip | — |
| inv 4 | [3, 4] | 2 new → range [2,4] (overlap from inv 2) | + CompactedEvent([2,4]) |

**Rolling-summary seed** — when token-threshold fires and a previous compaction exists, the previous compaction's `compacted_content` is injected as a seed event at the start of `events_to_compact`. This lets the new summary supersede (not duplicate) the old one.

### Example 1 — inspect what the compactor would compact (dry-run)

```python
import asyncio
from google.adk.apps.compaction import (
    _events_to_compact_for_token_threshold,
    _latest_prompt_token_count,
    _pending_function_call_ids,
)
from google.adk.sessions.session import Session

async def inspect_compaction(session: Session, event_retention_size: int = 5):
    """Prints which events the token-threshold compactor would select."""
    token_count = _latest_prompt_token_count(session.events)
    pending = _pending_function_call_ids(session.events)

    print(f"Estimated prompt tokens: {token_count}")
    print(f"Pending function calls: {pending}")

    events_to_compact = _events_to_compact_for_token_threshold(
        events=session.events,
        event_retention_size=event_retention_size,
    )
    print(f"Events selected for compaction: {len(events_to_compact)}")
    for ev in events_to_compact:
        role = ev.content.role if ev.content else "?"
        text = ""
        if ev.content and ev.content.parts:
            text = (ev.content.parts[0].text or "")[:60]
        print(f"  [{role}] {text!r}")
```

### Example 2 — wire both strategies simultaneously

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

# Token-threshold triggers if prompt > 50k tokens, retaining last 10 events.
# Sliding-window triggers every 5 new invocations with 2-invocation overlap.
# When both are active, token-threshold is checked first; if it fires,
# sliding-window is skipped for that turn.
app = App(
    root_agent=agent,
    name="dual-compact-demo",
    events_compaction_config=EventsCompactionConfig(
        token_threshold=50_000,
        event_retention_size=10,
        compaction_interval=5,
        overlap_size=2,
    ),
)

async def main():
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="dual-compact-demo", user_id="u1"
    )
    for i in range(20):
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=f"Message {i}: tell me something interesting.")]),
        ):
            pass  # events flow; compaction runs post-invocation
    print("Done — check session.events for CompactedEvent entries.")

asyncio.run(main())
```

### Example 3 — custom summarizer that avoids LLM calls

```python
from typing import Optional
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event_actions import EventCompaction
from google.adk.events.event import Event
import time

class BulletSummarizer(BaseEventsSummarizer):
    """Summarises events as a bullet list without an LLM call."""

    async def maybe_summarize_events(
        self, events: list[Event]
    ) -> Optional[Event]:
        if not events:
            return None
        lines = []
        for ev in events:
            if ev.content and ev.content.parts:
                text = "".join(p.text for p in ev.content.parts if p.text)
                if text.strip():
                    lines.append(f"• [{ev.content.role}] {text[:80].strip()}")
        if not lines:
            return None
        summary_text = "Conversation summary:\n" + "\n".join(lines)
        from google.genai import types
        start_ts = events[0].timestamp
        end_ts = events[-1].timestamp
        from google.adk.events.event import EventActions
        return Event(
            author="model",
            content=types.Content(
                role="model",
                parts=[types.Part(text=summary_text)],
            ),
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    compacted_content=types.Content(
                        role="model",
                        parts=[types.Part(text=summary_text)],
                    ),
                )
            ),
        )

from google.adk.apps._configs import EventsCompactionConfig

config = EventsCompactionConfig(
    token_threshold=20_000,
    event_retention_size=5,
    # compaction_interval and overlap_size are required integer fields in 2.2.0
    compaction_interval=10,
    overlap_size=3,
    summarizer=BulletSummarizer(),
)
```

---

## 2 · `inject_session_state` — async template substitution

**Source:** `google.adk.utils.instructions_utils`

`inject_session_state(template, readonly_context)` is the async engine behind `{var_name}` interpolation in `LlmAgent.instruction` callable strings. It supports three interpolation forms:

| Pattern | What it resolves |
|---|---|
| `{var_name}` | `session.state["var_name"]`; raises `KeyError` if missing |
| `{var_name?}` | Same but returns `""` if missing (optional) |
| `{artifact.file_name}` | Loads the named artifact and calls `str()` on it; raises `KeyError` if not found |
| `{artifact.file_name?}` | Same but returns `""` if the artifact does not exist — **requires `artifact_service` to be configured**; if `artifact_service is None` raises `ValueError` even with `?` |

Scope-prefixed keys work too: `{app:shared_var}`, `{user:prefs}`, `{temp:step_result}`.

The function uses an async regex substitution (`_async_sub`) because artifact loading is async. The regex pattern `r'{+[^{}]*}+'` matches one or more braces on each side, so double-brace escape `{{literal_brace}}` is NOT used — any `{...}` is attempted as an interpolation.

### Constructor / signature (source-verified)

```python
from google.adk.utils.instructions_utils import inject_session_state
from google.adk.agents.readonly_context import ReadonlyContext

async def inject_session_state(
    template: str,
    readonly_context: ReadonlyContext,
) -> str: ...
```

### Example 1 — state variable + optional variable injection

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.utils.instructions_utils import inject_session_state
from google.adk.agents.readonly_context import ReadonlyContext

async def build_instruction(ctx: ReadonlyContext) -> str:
    return await inject_session_state(
        "You are helping user '{user_name}'. "
        "Their account tier is '{user:tier}'. "
        "Current topic (may be empty): '{topic?}'. "  # optional state var — '' if absent
        "Focus on {mode?}.",  # second optional var — no error if 'mode' not in state
        ctx,
    )

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction=build_instruction,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo",
        user_id="u1",
        state={
            "user_name": "Alice",
            "user:tier": "premium",
            # "topic" intentionally absent — {topic?} → ""
        },
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What can you help me with?")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — inject per-language system instruction

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils.instructions_utils import inject_session_state

SYSTEM_PROMPTS = {
    "en": "You are a helpful assistant. Respond in English.",
    "fr": "Tu es un assistant utile. Réponds en français.",
    "de": "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
}

async def multilingual_instruction(ctx: ReadonlyContext) -> str:
    lang = ctx.state.get("lang", "en")
    base = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    # Still supports {user_name} from state
    return await inject_session_state(
        f"{base} You are speaking with {{user_name?}}.", ctx
    )

agent = LlmAgent(
    name="multilingual",
    model="gemini-2.5-flash",
    instruction=multilingual_instruction,
)
```

### Example 3 — validate state names (`_is_valid_state_name`)

The internal `_is_valid_state_name` guard means only valid Python identifiers (or `scope:identifier`) trigger substitution. Strings like `{2024-01-01}` or `{user input}` are returned as-is rather than raising an error.

```python
# Patterns that ARE interpolated:
# {user_name}         → valid identifier
# {user:tier}         → valid scope:identifier
# {app:feature_flag}  → valid scope:identifier
# {temp:result123}    → valid scope:identifier

# Patterns that are NOT interpolated (returned verbatim):
# {2024-01-01}        → not a valid identifier (starts with digit)
# {user input}        → not a valid identifier (contains space)
# {my-var}            → not a valid identifier (contains hyphen)

from google.adk.utils.instructions_utils import _is_valid_state_name

print(_is_valid_state_name("user_name"))     # True
print(_is_valid_state_name("user:tier"))     # True
print(_is_valid_state_name("2024-01-01"))    # False → returned verbatim
print(_is_valid_state_name("user input"))    # False → returned verbatim
```

---

## 3 · `run_llm_agent_as_node` — LlmAgent in three workflow modes

**Source:** `google.adk.workflow._llm_agent_wrapper`

When an `LlmAgent` is placed inside a `Workflow`, `NodeRunner` calls `run_llm_agent_as_node()`. The agent's `mode` field controls which dispatch path executes:

| `mode` | Behaviour | `include_contents` | Isolation |
|---|---|---|---|
| `None` → auto-set to `"single_turn"` | One LLM round; input appended as user event | `"none"` (forced) | `isolation_scope` from ctx |
| `"single_turn"` | Same as above | `"none"` | `isolation_scope` |
| `"task"` | Driven by `FinishTaskTool`; retries on validation failure | unchanged | `isolation_scope` |
| `"chat"` | Multi-round; task-delegation FC dispatch loop | unchanged | parent scope |

### `single_turn` / `task` input injection

`prepare_llm_agent_input` appends a user-role `Event` to `session.events` carrying the node's `node_input` converted via `_node_input_to_content`:

```
node_input type          → Content form
─────────────────────────────────────────
types.Content            → passed through (role set to "user")
str                      → Part(text=node_input)
BaseModel                → Part(text=model.model_dump_json())
dict / list              → Part(text=json.dumps(node_input))
anything else            → Part(text=str(node_input))
```

### `chat` mode — task-delegation dispatch loop

The chat wrapper replays unresolved task FCs from prior turns (step 1), then enters a `while True` loop (step 2): run `agent.run_async`; for each task-delegation FC emitted, call `ctx.run_node()` with `run_id=fc.id` (idempotency) and `override_isolation_scope=fc.id` (per-task scope isolation), synthesize a FR event, and re-enter `agent.run_async`. The loop exits when the LLM emits no task FC or transfers away.

### `task` mode — FinishTaskTool handshake

The task wrapper sniffs `finish_task` FCs and waits for the matching FR from `FinishTaskTool`. A validation-failure FR (containing an `error` key) is NOT terminal — the LLM sees the error and retries. Only a success FR causes `event.output = pending_fc_args` and returns.

### Example 1 — single_turn LlmAgent node with typed output

```python
import asyncio
from google.genai import types
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow
from google.adk.workflow._node import node

class SentimentResult(BaseModel):
    sentiment: str    # "positive", "negative", or "neutral"
    confidence: float
    reason: str

# mode defaults to None → "single_turn" inside the workflow
sentiment_node = LlmAgent(
    name="sentiment",
    model="gemini-2.5-flash",
    instruction=(
        "Classify the sentiment of the text you receive. "
        "Return only JSON matching the schema."
    ),
    output_schema=SentimentResult,
    output_key="sentiment_result",
)

@node
def format_output(node_input, ctx):
    result = ctx.session.state.get("sentiment_result", {})
    ctx.actions.state_delta["report"] = (
        f"Sentiment: {result.get('sentiment')} "
        f"(confidence {result.get('confidence', 0):.0%})"
    )

wf = Workflow(
    name="sentiment_pipeline",
    edges=[("START", sentiment_node, format_output)],
)

async def main():
    runner = InMemoryRunner(agent=wf, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="I absolutely love this product!")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text if event.content else "")

asyncio.run(main())
```

### Example 2 — task mode with FinishTaskTool retry

```python
import asyncio
from google.genai import types
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow
from google.adk.workflow._node import node
from google.adk.workflow._llm_agent_wrapper import run_llm_agent_as_node

class ExtractedEntities(BaseModel):
    people: list[str]
    organisations: list[str]
    locations: list[str]

# mode="task" → agent must call finish_task(entities=...) to complete.
# If the output doesn't match ExtractedEntities, FinishTaskTool returns
# an error and the LLM is asked to retry.
entity_agent = LlmAgent(
    name="entity_extractor",
    model="gemini-2.5-flash",
    mode="task",
    instruction=(
        "Extract all named entities from the text you receive. "
        "Call finish_task with the structured result."
    ),
    output_schema=ExtractedEntities,
)

# Workflow._validate_no_task_mode_graph_nodes() rejects LlmAgent(mode="task")
# placed directly as a static graph node. Wrap it in a @node function instead.
@node
async def entity_node(node_input, ctx):
    async for event in run_llm_agent_as_node(entity_agent, ctx=ctx, node_input=node_input):
        yield event

@node
def summarise(node_input, ctx):
    # node_input carries the finish_task args dict from the task-mode agent.
    # task-mode does not write output_key to session state — use node_input instead.
    entities = node_input or {}
    summary = (
        f"Found {len(entities.get('people', []))} people, "
        f"{len(entities.get('organisations', []))} orgs, "
        f"{len(entities.get('locations', []))} locations."
    )
    ctx.actions.state_delta["summary"] = summary

wf = Workflow(name="ner_pipeline", edges=[("START", entity_node, summarise)])

async def main():
    runner = InMemoryRunner(agent=wf, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=(
            "Apple Inc. was founded by Steve Jobs in Cupertino. "
            "Tim Cook now leads the company."
        ))]),
    ):
        pass
    state = (await runner.session_service.get_session(
        app_name="demo", user_id="u1", session_id=session.id
    )).state
    print(state.get("summary"))

asyncio.run(main())
```

### Example 3 — chat mode coordinator with sequential task delegation

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow

research_agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    mode="task",
    instruction="Research the topic provided to you. Return a concise summary.",
)
writer_agent = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    mode="task",
    instruction=(
        "Write a polished article based on the research provided to you. "
        "Call finish_task with the article text."
    ),
    output_key="article",
)

# The coordinator uses mode="chat" so it can delegate to sub-agents
# across multiple LLM rounds without closing the event loop.
# sub_agents with mode="task" → model_post_init() creates _TaskAgentTool for each,
# which is what _extract_task_delegation_fcs() checks for. Using AgentTool in
# tools=[] bypasses this check and task delegation never fires.
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro",
    mode="chat",
    instruction=(
        "You coordinate a research + writing pipeline. "
        "First, ask the researcher sub-agent to gather information. "
        "Then pass the result to the writer sub-agent to produce the article. "
        "Finally, summarise what was accomplished."
    ),
    sub_agents=[research_agent, writer_agent],
)

wf = Workflow(name="research_write", edges=[("START", coordinator)])

async def main():
    runner = InMemoryRunner(agent=wf, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Write a short article about the history of quantum computing.")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text[:200] if event.content else "")

asyncio.run(main())
```

---

## 4 · `ToolConfig` + `BaseToolConfig` + `ToolArgsConfig` — YAML tool DSL

**Source:** `google.adk.tools.tool_configs`

These three `@experimental(FeatureName.TOOL_CONFIG)` Pydantic models form the YAML-level tool reference system used by `AgentLoader` when loading agents from `.yaml` config files. They are not normally instantiated in Python code — they exist so that YAML can refer to tools by name with optional constructor args.

### Class signatures (source-verified)

```python
from google.adk.tools.tool_configs import (
    BaseToolConfig,   # base class; extra="forbid"
    ToolArgsConfig,   # holds free key-value pairs; extra="allow"
    ToolConfig,       # name + optional args
)

class BaseToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ToolArgsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # arbitrary kwargs

class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str      # FQN or built-in name
    args: ToolArgsConfig | None = None
```

### Five reference patterns from source docstring

```yaml
# Pattern 1 — ADK built-in tool instance
tools:
  - name: google_search

# Pattern 2 — ADK built-in tool class with args
tools:
  - name: AgentTool
    args:
      agent: ./sub_agent.yaml
      skip_summarization: true

# Pattern 3 — user-defined tool instance (FQN to an already-constructed object)
tools:
  - name: my_package.my_module.my_tool_instance

# Pattern 4 — user-defined tool class with constructor args
tools:
  - name: my_package.my_module.MyToolClass
    args:
      api_base_url: https://api.example.com
      timeout_seconds: 30

# Pattern 5 — factory function (must accept a ToolArgsConfig and return BaseTool)
tools:
  - name: my_package.my_module.build_weather_tool
    args:
      api_key: ${WEATHER_API_KEY}
```

### Example 1 — YAML agent config with ToolConfig

```yaml
# agents/research_agent.yaml
name: research_agent
model: gemini-2.5-flash
instruction: |
  You are a research assistant. Use available tools to answer questions.
tools:
  - name: google_search
  - name: my_tools.web_scraper.WebScraperTool
    args:
      max_pages: 5
      timeout: 10
```

```python
# config_agent_utils.from_config is the current loader for YAML configs.
# Note: deprecated in 2.2.0 — expected to be replaced by a stable public API.
from google.adk.agents.config_agent_utils import from_config

agent = from_config("agents/research_agent.yaml")
print(f"Loaded: {agent.name}")
```

### Example 2 — custom tool factory for ToolConfig pattern 5

```python
# my_tools/toolbox.py
from google.adk.tools.tool_configs import ToolArgsConfig
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool

def build_http_tool(args: ToolArgsConfig) -> BaseTool:
    """Factory function loadable via ToolConfig pattern 5."""
    # ToolArgsConfig uses extra="allow"; in Pydantic v2 extra fields are stored
    # in model_extra, not as regular attributes — use model_extra.get() to access them.
    extras = args.model_extra or {}
    base_url = extras.get("base_url", "https://api.example.com")
    timeout = extras.get("timeout", 10)

    async def call_api(endpoint: str, method: str = "GET") -> dict:
        """Call the configured API endpoint."""
        import httpx
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as c:
            resp = await c.request(method, endpoint)
            resp.raise_for_status()
            return resp.json()

    return FunctionTool(func=call_api)
```

```yaml
# agents/api_agent.yaml
tools:
  - name: my_tools.toolbox.build_http_tool
    args:
      base_url: https://api.myservice.com
      timeout: 30
```

### Example 3 — BaseToolConfig for custom tool extensions

```python
from google.adk.tools.tool_configs import BaseToolConfig
from pydantic import Field

class DatabaseToolConfig(BaseToolConfig):
    """Custom config for database tools — loaded via YAML discriminated union."""
    connection_string: str = Field(description="SQLAlchemy connection string")
    pool_size: int = Field(default=5, ge=1, le=20)
    echo: bool = False

    def build(self):
        from my_tools.database import DatabaseQueryTool
        return DatabaseQueryTool(
            connection_string=self.connection_string,
            pool_size=self.pool_size,
            echo=self.echo,
        )

# Usage: config = DatabaseToolConfig(
#     connection_string="postgresql://user:pass@host/db"
# )
# tool = config.build()
```

---

## 5 · `RequestInput` — structured HITL interrupt model

**Source:** `google.adk.events.request_input`

`RequestInput` is the data model for workflow HITL interrupts that request user input (as opposed to auth interrupts which use `AuthConfig`). It is **yielded** (or returned) from a `@node` function; `BaseNode.run()` intercepts the yielded value, calls `create_request_input_event()` to wrap it in an interrupt `Event`, and marks the node as WAITING. There is no `ctx.request_input()` method — the yield is the mechanism.

### Constructor (source-verified)

```python
from google.adk.events.request_input import RequestInput

RequestInput(
    interrupt_id: str = Field(default_factory=lambda: str(uuid.uuid4())),
    payload: Any | None = None,
    message: str | None = None,
    response_schema: SchemaType | None = None,  # Pydantic type / generic alias / raw dict
)
```

**camelCase aliases** — all fields have camelCase aliases via `alias_generator=alias_generators.to_camel` with `populate_by_name=True`, so `interrupt_id` == `interruptId` in JSON / event serialisation.

> **Field names in 2.2.0:** the Python fields are `interrupt_id` and `message`. Older community examples sometimes use `id` or `hint` — those are **not** valid field names or aliases in 2.2.0 and will be silently ignored (or raise a validation error if `extra="forbid"` is set). Always use `interrupt_id` / `message` with this version.

**`response_schema`** accepts:
- `type[BaseModel]` — a Pydantic model class
- Generic alias — e.g. `list[str]`, `dict[str, int]`
- Raw `dict` — JSON Schema

### Example 1 — basic user-input gate in a FunctionNode

```python
import asyncio
from google.genai import types
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow
from google.adk.workflow._node import node
from google.adk.agents.context import Context
from google.adk.events.request_input import RequestInput
from google.adk.runners import InMemoryRunner

class ApprovalResponse(BaseModel):
    approved: bool
    comment: str = ""

@node
async def approval_gate(node_input, ctx: Context):
    """Pause workflow and ask the user for approval."""
    # Yield RequestInput — BaseNode.run() intercepts the yield and converts
    # it to an interrupt Event. With rerun_on_resume=False (the default), the
    # user's response dict becomes this node's output and flows to the next
    # node as its node_input.
    yield RequestInput(
        message="Please approve or reject this action.",
        response_schema=ApprovalResponse,
        payload={"action": node_input},  # custom context for the UI
    )

@node
def process_if_approved(node_input, ctx: Context):
    # node_input is the user's response from approval_gate
    # (resolved as a dict matching ApprovalResponse when rerun_on_resume=False)
    if not isinstance(node_input, dict) or not node_input.get("approved"):
        ctx.actions.state_delta["status"] = "rejected"
        return
    ctx.actions.state_delta["status"] = "approved"
    ctx.actions.state_delta["comment"] = node_input.get("comment", "")

wf = Workflow(name="approval_wf", edges=[("START", approval_gate, process_if_approved)])

async def main():
    runner = InMemoryRunner(agent=wf, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    from google.adk.workflow.utils._workflow_hitl_utils import (
        get_request_input_interrupt_ids,
        create_request_input_response,
    )

    # First run — workflow pauses at approval_gate
    interrupted_event = None
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Process order #42")])
    ):
        # Use get_request_input_interrupt_ids to detect RequestInput interrupts:
        # it reads interrupt IDs from the adk_request_input function-call args,
        # which is more explicit than checking event.long_running_tool_ids
        # (long_running_tool_ids carries general FC IDs, not just HITL ones).
        if get_request_input_interrupt_ids(event):
            interrupted_event = event
            break

    if interrupted_event:
        interrupt_ids = get_request_input_interrupt_ids(interrupted_event)
        interrupt_id = interrupt_ids[0]
        print(f"Workflow paused. Interrupt ID: {interrupt_id}")

        # Resume with approval
        resume_part = create_request_input_response(
            interrupt_id=interrupt_id,
            response={"approved": True, "comment": "Looks good!"},
        )
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[resume_part]),
        ):
            pass

asyncio.run(main())
```

### Example 2 — typed list response schema

```python
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.workflow._node import node

@node
async def collect_tags(node_input, ctx: Context):
    """Request a list of string tags from the user."""
    # Yielding RequestInput pauses the node. With rerun_on_resume=False (default),
    # the user's list response flows to save_tags as its node_input.
    yield RequestInput(
        message="Please provide tags for this item (list of strings).",
        response_schema=list[str],
        interrupt_id="collect-tags-001",  # stable ID for idempotency
    )

@node
def save_tags(node_input, ctx: Context):
    # node_input is the user's response list from collect_tags's output
    tags = node_input if isinstance(node_input, list) else []
    ctx.actions.state_delta["tags"] = tags
    print(f"Saved tags: {tags}")
```

### Example 3 — `interrupt_id` reuse for rejection/retry cycles

Reusing the same `interrupt_id` across retries is explicitly supported by the source docstring. The framework matches FCs and FRs by count (not unique ID), so a stable ID lets you track the same logical interrupt across multiple attempts.

```python
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.workflow._node import node

REVIEW_INTERRUPT_ID = "document-review-v1"

@node
async def request_document_review(node_input, ctx: Context):
    draft = ctx.session.state.get("draft", "")
    yield RequestInput(
        interrupt_id=REVIEW_INTERRUPT_ID,
        message=f"Please review this draft: {draft[:200]}...",
        payload={"draft_length": len(draft)},
        response_schema=dict,  # free-form reviewer comments
    )
    # The reviewer's comment dict becomes this node's output and flows to
    # the next graph node as its node_input.
```

---

## 6 · HITL workflow utilities

**Source:** `google.adk.workflow.utils._workflow_hitl_utils`

This internal module provides the builder functions that `NodeRunner` and `FunctionNode` use to create and respond to HITL interrupts. You rarely need these directly, but understanding them is essential for debugging interrupt flows or building custom HITL tooling.

### Key functions

| Function | Purpose |
|---|---|
| `create_request_input_event(request_input)` | Wraps `RequestInput` into an `Event` with an `adk_request_input` FC |
| `create_request_input_response(interrupt_id, response)` | Builds the `FunctionResponse` Part for resuming |
| `has_request_input_function_call(event)` | Checks if an event has a pending `adk_request_input` FC |
| `has_auth_request_function_call(event)` | Checks for a pending `adk_request_credential` FC |
| `get_request_input_interrupt_ids(event)` | Returns all `interrupt_id`s from an event's request-input FCs |
| `create_auth_request_event(auth_config, interrupt_id)` | Builds an `adk_request_credential` FC event |
| `process_auth_resume(response_data, auth_config, state)` | Stores credentials from resume data into session state |
| `has_auth_credential(auth_config, state)` | Returns True if credentials already exist in state |

**Constant names:**
```python
REQUEST_INPUT_FUNCTION_CALL_NAME = "adk_request_input"
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = "adk_request_credential"
```

### Example 1 — scan events for pending interrupts

```python
from google.adk.workflow.utils._workflow_hitl_utils import (
    has_request_input_function_call,
    has_auth_request_function_call,
    get_request_input_interrupt_ids,
)

def summarise_pending_interrupts(events):
    """Print all pending HITL interrupts in a session's event list."""
    for i, event in enumerate(events):
        if has_request_input_function_call(event):
            ids = get_request_input_interrupt_ids(event)
            print(f"Event {i}: input interrupt(s) {ids}")
        if has_auth_request_function_call(event):
            print(f"Event {i}: auth credential interrupt")
```

### Example 2 — manually build a resume response

```python
from google.genai import types
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_response,
)

def build_resume_message(interrupt_id: str, user_response: dict) -> types.Content:
    """Build the user Content needed to resume a paused workflow."""
    part = create_request_input_response(
        interrupt_id=interrupt_id,
        response=user_response,
    )
    return types.Content(role="user", parts=[part])

# Example usage:
# user_response must be a dict (Mapping[str, Any]); wrap lists/scalars in a dict.
resume_content = build_resume_message(
    interrupt_id="collect-tags-001",
    user_response={"tags": ["python", "async", "google-adk"]},
)
```

### Example 3 — process_auth_resume for three response formats

`process_auth_resume` uses a two-stage dispatch:

1. **Stage 1** — attempt `AuthConfig.model_validate(response_data)`. If it succeeds, the
   full auth config (including the exchanged credential) is stored directly.
2. **Stage 2** (if stage 1 raises `ValidationError` / `TypeError`) — call
   `_build_credential_from_value`, which branches on `raw_auth_credential.auth_type`:
   - `API_KEY` → wrap the raw value as a plain-string api key
   - anything else → `AuthCredential.model_validate(value)` (expects an AuthCredential dict)

```python
import asyncio
# APIKey and APIKeyIn are the correct auth-scheme types from FastAPI's OpenAPI models;
# google.adk.auth.auth_schemes re-exports SecurityScheme (which includes APIKey).
from fastapi.openapi.models import APIKey, APIKeyIn
from google.adk.workflow.utils._workflow_hitl_utils import process_auth_resume
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.sessions.state import State

auth_config = AuthConfig(
    auth_scheme=APIKey(name="x-api-key", in_=APIKeyIn.header),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="",  # will be filled by resume
    ),
)

# State requires value dict and delta dict positional args.
state = State({}, {})

async def demo():
    # Format 1: full AuthConfig dict — stage 1 succeeds (sends a complete
    # AuthConfig dict including the exchanged credential; works for any auth type).
    # AuthCredentialTypes values are lowercase strings: "apiKey", "oauth2", etc.
    await process_auth_resume(
        response_data={
            "auth_scheme": {"type": "apiKey", "name": "x-api-key", "in": "header"},
            "exchanged_auth_credential": {
                "auth_type": "apiKey",   # AuthCredentialTypes.API_KEY.value
                "api_key": "sk-test-123",
            },
        },
        auth_config=auth_config,
        state=state,
    )

    # Format 2: plain string — stage 1 fails; stage 2 sees auth_type==API_KEY
    # and wraps the string directly as the api_key value
    await process_auth_resume(
        response_data="sk-test-456",
        auth_config=auth_config,
        state=state,
    )

    # Format 3: AuthCredential dict — stage 1 fails; stage 2 branches on
    # raw_auth_credential.auth_type, NOT the auth scheme.
    # auth_type=OAUTH2 causes stage 2 to call AuthCredential.model_validate(response_data).
    # (The APIKey scheme here is intentional: stage 2 only checks auth_type.)
    oauth_config = AuthConfig(
        auth_scheme=APIKey(name="x-api-key", in_=APIKeyIn.header),
        raw_auth_credential=AuthCredential(auth_type=AuthCredentialTypes.OAUTH2),
    )
    await process_auth_resume(
        response_data={"auth_type": "oauth2", "oauth2": {"access_token": "tok-abc"}},
        auth_config=oauth_config,
        state=state,
    )

asyncio.run(demo())
```

---

## 7 · `TaskResultAggregator` — A2A task state priority machine

**Source:** `google.adk.a2a.executor.task_result_aggregator`

> **Install note:** The examples in this section import from the `a2a` package (the A2A Python SDK), which is a separate PyPI package required for A2A features. Install it with `pip install a2a-sdk` alongside `google-adk`. The `a2a` types (`TaskState`, `TaskStatusUpdateEvent`, etc.) are defined there — not in `google.adk.a2a`.

`TaskResultAggregator` is used inside `A2aAgentExecutorImpl` to accumulate `TaskStatusUpdateEvent`s from the ADK runner and emit the correct final A2A task state. The challenge: ADK runners emit `working` events during execution, but intermediate state transitions (e.g. to `auth_required`) must not prematurely terminate the event stream for the A2A client. The aggregator solves this by recording the true state internally while re-writing intermediate events to `working`.

### State priority (source-verified)

```
failed > auth_required > input_required > working
```

Once a higher-priority state is reached, lower-priority transitions are ignored.

### Constructor / signature

```python
from google.adk.a2a.executor.task_result_aggregator import TaskResultAggregator

aggregator = TaskResultAggregator()
# Fields:
#   _task_state: TaskState = TaskState.working  (mutable)
#   _task_status_message: Message | None = None
```

### Example 1 — simulate the aggregator state machine

```python
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Message, TextPart
from google.adk.a2a.executor.task_result_aggregator import TaskResultAggregator

def make_status_event(state: TaskState, text: str) -> TaskStatusUpdateEvent:
    return TaskStatusUpdateEvent(
        id="task-1",
        status=TaskStatus(
            state=state,
            message=Message(parts=[TextPart(text=text)]),
        ),
        final=False,
    )

aggregator = TaskResultAggregator()

# Sequence: working → auth_required → input_required → working
events = [
    make_status_event(TaskState.working, "Starting..."),
    make_status_event(TaskState.auth_required, "Please authenticate."),
    make_status_event(TaskState.input_required, "Need more input."),   # ignored — auth > input
    make_status_event(TaskState.working, "Processing..."),
]

for ev in events:
    aggregator.process_event(ev)
    print(f"After event: aggregator.task_state={aggregator.task_state}, event.status.state={ev.status.state}")

# Output:
# working  | working      (working → working, no change)
# working  | auth_required (re-written to working; aggregator records auth_required)
# working  | input_required (re-written to working; aggregator ignores — already auth_required)
# working  | working      (re-written to working; aggregator state stays auth_required)

print(f"Final state: {aggregator.task_state}")           # auth_required
print(f"Final message: {aggregator.task_status_message}")
```

### Example 2 — use in a custom A2A executor

```python
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent
from google.adk.a2a.executor.task_result_aggregator import TaskResultAggregator

async def stream_task(runner, user_id, session_id, message):
    """Runs an ADK agent and yields A2A-safe events."""
    aggregator = TaskResultAggregator()
    all_events = []

    async for adk_event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        # Map each ADK event to a simple A2A state: completed for the final
        # response, working for everything else. This example does not map ADK
        # interrupt events (auth/input_required) — those would need separate
        # handling before calling process_event with the matching TaskState.
        state = TaskState.completed if adk_event.is_final_response() else TaskState.working
        a2a_event = TaskStatusUpdateEvent(
            id="task-1",
            status=TaskStatus(state=state),
            final=False,
        )
        aggregator.process_event(a2a_event)
        all_events.append(a2a_event)
        yield a2a_event  # stream to A2A client

    # Aggregator only records failed/auth_required/input_required; it never
    # promotes to completed. Derive the final state manually.
    final_state = (
        TaskState.completed
        if aggregator.task_state == TaskState.working
        else aggregator.task_state
    )
    final_status = TaskStatus(
        state=final_state,
        message=aggregator.task_status_message,
    )
    yield TaskStatusUpdateEvent(id="task-1", status=final_status, final=True)
```

### Example 3 — failed state is terminal

```python
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus
from google.adk.a2a.executor.task_result_aggregator import TaskResultAggregator

aggregator = TaskResultAggregator()

def status_event(state):
    return TaskStatusUpdateEvent(
        id="t",
        status=TaskStatus(state=state),
        final=False,
    )

aggregator.process_event(status_event(TaskState.failed))
aggregator.process_event(status_event(TaskState.auth_required))  # ignored

print(aggregator.task_state)  # TaskState.failed — cannot be overridden
```

---

## 8 · Retry internals — exponential backoff formula

**Source:** `google.adk.workflow.utils._retry_utils`

`_should_retry_node` and `_get_retry_delay` are the two functions `NodeRunner._attempt_retry` calls before sleeping and re-running a failed node. Understanding the exact formula is important when tuning `RetryConfig`.

### `_should_retry_node` logic (source-verified)

```python
def _should_retry_node(exception, retry_config, node_state) -> bool:
    if not retry_config:                                          return False
    if node_state.attempt_count >= retry_config.max_attempts:    return False  # max_attempts default: 5
    if retry_config.exceptions is not None:
        if type(exception).__name__ not in retry_config.exceptions:
            return False
    return True
```

- `attempt_count` starts at **1** for the original request; `>=` comparison means retry stops when `attempt_count` equals `max_attempts` (after `max_attempts - 1` retries total).
- `retry_config.exceptions` is a list of exception **type names** (strings), not types themselves.

### `_get_retry_delay` formula (source-verified)

```
delay = initial_delay × backoff_factor^(attempt_count − 1)
delay = min(delay, max_delay)
jitter_offset = random.uniform(-jitter × delay, +jitter × delay)
final_delay = max(0.0, delay + jitter_offset)
```

**Defaults:** `initial_delay=1.0`, `max_delay=60.0`, `backoff_factor=2.0`, `jitter=1.0`

| Attempt (attempt_count) | Exponent | Base delay | With default jitter range |
|---|---|---|---|
| 1st failure (attempt 1) | 0 | 1.0 s | [0.0, 2.0] s |
| 2nd failure (attempt 2) | 1 | 2.0 s | [0.0, 4.0] s |
| 3rd failure (attempt 3) | 2 | 4.0 s | [0.0, 8.0] s |
| 4th failure (attempt 4) | 3 | 8.0 s | [0.0, 16.0] s |

### Example 1 — RetryConfig with exception filter

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._node import node
from google.adk.agents.context import Context
import httpx

retry_cfg = RetryConfig(
    max_attempts=4,
    initial_delay=2.0,
    backoff_factor=3.0,
    max_delay=30.0,
    jitter=0.5,  # smaller jitter for predictable delays
    # Use bare class names: _should_retry_node compares type(exception).__name__
    exceptions=["TimeoutException", "ConnectError"],
)

@node(retry_config=retry_cfg)
async def fetch_data(node_input, ctx: Context):
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get("https://api.example.com/data")
        resp.raise_for_status()
        ctx.actions.state_delta["data"] = resp.json()
```

### Example 2 — simulate delay calculation

```python
import random
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._node_state import NodeState
from google.adk.workflow.utils._retry_utils import _get_retry_delay

config = RetryConfig(
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=60.0,
    jitter=1.0,
)

random.seed(42)  # reproducible
for attempt in range(1, 6):
    state = NodeState(attempt_count=attempt)
    delay = _get_retry_delay(config, state)
    print(f"Attempt {attempt}: delay={delay:.2f}s")

# Example output (seed=42):
# Attempt 1: delay=0.63s
# Attempt 2: delay=2.17s
# Attempt 3: delay=5.94s
# Attempt 4: delay=14.23s
# Attempt 5: delay=43.85s
```

### Example 3 — custom exception names in retry filter

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._node_state import NodeState
from google.adk.workflow.utils._retry_utils import _should_retry_node

config = RetryConfig(
    max_attempts=3,
    exceptions=["ValueError", "TimeoutError"],
)

state = NodeState(attempt_count=1)
print(_should_retry_node(ValueError("bad"), config, state))    # True
print(_should_retry_node(RuntimeError("crash"), config, state)) # False — not in list

state2 = NodeState(attempt_count=3)
print(_should_retry_node(ValueError("bad"), config, state2))   # False — attempt >= max
```

---

## 9 · `GoogleLLMVariant` + model-name utilities + `can_use_output_schema_with_tools`

**Sources:** `google.adk.utils.variant_utils`, `google.adk.utils.model_name_utils`, `google.adk.utils.output_schema_utils`

These three small but critical utility modules control how ADK selects between Vertex AI and Gemini API backends, parses model strings, and decides whether to use native output-schema-with-tools support or fall back to the `SetModelResponseTool` workaround.

### `GoogleLLMVariant` (source-verified)

```python
from google.adk.utils.variant_utils import GoogleLLMVariant, get_google_llm_variant

class GoogleLLMVariant(Enum):
    VERTEX_AI = "VERTEX_AI"    # GOOGLE_GENAI_USE_VERTEXAI=1
    GEMINI_API = "GEMINI_API"  # default (no env var)

# Runtime detection:
variant = get_google_llm_variant()  # reads GOOGLE_GENAI_USE_VERTEXAI
```

### `extract_model_name` — path-based model strings

ADK agents can receive model strings in three forms. `extract_model_name` normalises them all:

| Input form | Example | Extracted |
|---|---|---|
| Simple name | `"gemini-2.5-flash"` | `"gemini-2.5-flash"` |
| Vertex AI path | `"projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash"` | `"gemini-2.5-flash"` |
| Apigee path | `"apigee/google/v1/gemini-2.5-flash"` | `"gemini-2.5-flash"` |
| `models/` prefix | `"models/gemini-2.5-pro"` | `"gemini-2.5-pro"` |

### `is_gemini_eap_or_2_or_above` — EAP + semver check

Matches EAP models (`gemini-flash-early-exp`, `gemini-flash-early-exp3`) first using a regex; then for standard names, parses the version component with `packaging.version.Version` and checks `major >= 2`.

### `can_use_output_schema_with_tools` — output schema gating

```python
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools

# Returns True when:
# 1. Model is a LiteLlm instance (LiteLLM handles tools+response_format per-provider)
# 2. GOOGLE_GENAI_USE_VERTEXAI=1 AND model is Gemini EAP or 2.x+
# Otherwise returns False → SetModelResponseTool workaround is activated
```

This function is the single gating point for whether `SetModelResponseTool` is injected. If it returns `False`, `LlmAgent` adds `SetModelResponseTool` to the tool list and removes `response_mime_type`/`response_schema` from the generation config.

### Example 1 — detect backend and model version

```python
import os
from google.adk.utils.variant_utils import get_google_llm_variant, GoogleLLMVariant
from google.adk.utils.model_name_utils import (
    extract_model_name,
    is_gemini_model,
    is_gemini_1_model,
    is_gemini_eap_or_2_or_above,
)

# Simulate Vertex AI mode
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

model_strings = [
    "gemini-2.5-flash",
    "gemini-1.5-pro",
    "gemini-flash-early-exp3",
    "projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-pro",
    "gpt-4o",
]

for ms in model_strings:
    name = extract_model_name(ms)
    print(
        f"{ms[:55]:55s} → name={name:25s} "
        f"gemini={is_gemini_model(ms)!s:5} "
        f"v1={is_gemini_1_model(ms)!s:5} "
        f"2+={is_gemini_eap_or_2_or_above(ms)!s}"
    )
```

### Example 2 — check output schema + tools support at runtime

```python
import os
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools

# Gemini API (no Vertex AI env) — native support NOT available for Gemini 2.x
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
print(can_use_output_schema_with_tools("gemini-2.5-flash"))  # False

# Vertex AI mode — native support available for Gemini 2.x
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
print(can_use_output_schema_with_tools("gemini-2.5-flash"))  # True
print(can_use_output_schema_with_tools("gemini-1.5-pro"))    # False (v1.x)

# LiteLlm always True regardless of variant/model
from google.adk.models.lite_llm import LiteLlm
llm = LiteLlm(model="openai/gpt-4o")
print(can_use_output_schema_with_tools(llm))                 # True
```

### Example 3 — choose model + backend at agent construction time

```python
import os
from google.adk.agents import LlmAgent
from google.adk.utils.variant_utils import get_google_llm_variant, GoogleLLMVariant
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    sections: list[str]
    word_count: int

def build_report_agent(model: str = "gemini-2.5-flash") -> LlmAgent:
    variant = get_google_llm_variant()
    native = can_use_output_schema_with_tools(model)
    print(
        f"Backend: {variant.value}, "
        f"native output_schema+tools: {native}, "
        f"SetModelResponseTool: {'yes' if not native else 'no'}"
    )
    return LlmAgent(
        name="report_agent",
        model=model,
        instruction="Generate a structured report from the provided text.",
        output_schema=Report,
        # ADK automatically adds SetModelResponseTool when native=False
    )

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
agent = build_report_agent("gemini-2.5-flash")  # native=True
```

---

## 10 · `SpannerAdminToolset` — 7 Spanner admin tools

**Source:** `google.adk.tools.spanner.admin_toolset`

`SpannerAdminToolset` is a `@experimental(FeatureName.SPANNER_ADMIN_TOOLSET)` `BaseToolset` that wraps seven Cloud Spanner admin operations as `GoogleTool` instances, giving an LLM the ability to manage Spanner infrastructure (instances, databases, configs) directly.

### Constructor (source-verified)

```python
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig

SpannerAdminToolset(
    tool_filter=None,              # list[str] | ToolPredicate | None
    credentials_config=None,       # SpannerCredentialsConfig | None
    spanner_tool_settings=None,    # SpannerToolSettings | None → defaults
)
```

### 7 built-in tools

| Tool name | Function | Description |
|---|---|---|
| `spanner_list_instances` | `admin_tool.list_instances` | List all instances in a project |
| `spanner_get_instance` | `admin_tool.get_instance` | Get details of a named instance |
| `spanner_create_instance` | `admin_tool.create_instance` | Create a new instance |
| `spanner_list_databases` | `admin_tool.list_databases` | List databases on an instance |
| `spanner_create_database` | `admin_tool.create_database` | Create a new database |
| `spanner_list_instance_configs` | `admin_tool.list_instance_configs` | List available regional configs |
| `spanner_get_instance_config` | `admin_tool.get_instance_config` | Get details of a named config |

All tools are `async`, accept `credentials: Credentials` (injected by `GoogleTool`), and return `{"status": "SUCCESS", "results": ...}` or `{"status": "ERROR", "error_details": ...}`.

Note: `SpannerAdminToolset` provides **management plane** operations. For **data plane** operations (SQL queries, DML, vector search) use `SpannerToolset` (covered in vol. 2/16).

### Example 1 — basic admin agent

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset
from google.adk.runners import InMemoryRunner

# credentials_config=None (default) uses Application Default Credentials.
# To pass explicit credentials: credentials_config=SpannerCredentialsConfig(
#     credentials=google.auth.default(scopes=[...])[0]
# )
admin_toolset = SpannerAdminToolset()

admin_agent = LlmAgent(
    name="spanner_admin",
    model="gemini-2.5-flash",
    instruction=(
        "You are a Cloud Spanner administrator. "
        "Use the available tools to manage Spanner instances and databases. "
        "Always confirm destructive operations with the user before executing."
    ),
    tools=[admin_toolset],
)

async def main():
    runner = InMemoryRunner(agent=admin_agent, app_name="spanner-mgmt")
    session = await runner.session_service.create_session(
        app_name="spanner-mgmt", user_id="admin"
    )
    async for event in runner.run_async(
        user_id="admin",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="List all Spanner instances in project my-gcp-project.")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — filter to read-only tools

```python
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset

# Only allow listing operations — no create/mutate.
# tool_filter is checked against the pre-prefix tool name (the raw function
# name); get_tools_with_prefix() adds "spanner_" AFTER filtering.
read_only_toolset = SpannerAdminToolset(
    tool_filter=[
        "list_instances",
        "get_instance",
        "list_databases",
        "list_instance_configs",
        "get_instance_config",
    ],
)

from google.adk.agents import LlmAgent

read_only_agent = LlmAgent(
    name="spanner_reader",
    model="gemini-2.5-flash",
    instruction="You can only read Spanner configuration — you cannot create or modify anything.",
    tools=[read_only_toolset],
)
```

### Example 3 — combined admin + data agent

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.runners import InMemoryRunner

# Admin toolset for management plane
admin_tools = SpannerAdminToolset()

# Data toolset for query plane (requires instance + database config)
data_tools = SpannerToolset(
    project_id="my-gcp-project",
    instance_id="my-instance",
    database_id="my-database",
    spanner_tool_settings=SpannerToolSettings(
        max_executed_query_result_rows=100,
    ),
)

combined_agent = LlmAgent(
    name="spanner_full_agent",
    model="gemini-2.5-pro",
    instruction=(
        "You are a full-stack Spanner assistant. "
        "You can manage infrastructure (instances, databases) AND run queries. "
        "Use admin tools for management and data tools for queries."
    ),
    tools=[admin_tools, data_tools],
)

async def main():
    runner = InMemoryRunner(agent=combined_agent, app_name="spanner-full")
    session = await runner.session_service.create_session(
        app_name="spanner-full", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=(
            "Check if there's an instance called 'prod-instance'. "
            "If so, list its databases. If not, let me know."
        ))]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## Summary table

| # | Class / symbol | Module | Key insight |
|---|---|---|---|
| 1 | Compaction pipeline | `apps.compaction` | Two strategies run in priority order; HITL guards prevent compacting pending FC/auth events; rolling-summary seed avoids duplicate summaries |
| 2 | `inject_session_state` | `utils.instructions_utils` | `{var?}` optional form; `{artifact.file_name}` async load (requires `artifact_service`; `?` skips missing artifact but not missing service); scope prefixes work; non-identifier patterns returned verbatim |
| 3 | `run_llm_agent_as_node` | `workflow._llm_agent_wrapper` | `single_turn` forces `include_contents=none`; `task` waits for FinishTaskTool success FR; `chat` dispatch loop replays unresolved task FCs on resume |
| 4 | `ToolConfig` / YAML DSL | `tools.tool_configs` | 5 reference patterns; `ToolArgsConfig(extra="allow")` for free kwargs; `BaseToolConfig(extra="forbid")` for custom configs |
| 5 | `RequestInput` | `events.request_input` | camelCase aliases; `response_schema` accepts Pydantic type/generic alias/dict; stable `interrupt_id` for retry cycles |
| 6 | HITL utilities | `workflow.utils._workflow_hitl_utils` | `adk_request_input` FC name constant; `process_auth_resume` stage 1: full AuthConfig dict; stage 2: API_KEY type → plain string, others → AuthCredential dict |
| 7 | `TaskResultAggregator` | `a2a.executor.task_result_aggregator` | Priority: failed > auth_required > input_required > working; intermediate events re-written to `working` to prevent premature A2A stream termination |
| 8 | Retry internals | `workflow.utils._retry_utils` | `attempt_count` is 1-based; delay formula: `initial × backoff^(attempt-1)`; exception filter uses type **name** strings |
| 9 | `GoogleLLMVariant` + model utils | `utils.variant_utils` / `model_name_utils` / `output_schema_utils` | `GOOGLE_GENAI_USE_VERTEXAI` env switch; `extract_model_name` handles Vertex/Apigee paths; `can_use_output_schema_with_tools` gates `SetModelResponseTool` injection |
| 10 | `SpannerAdminToolset` | `tools.spanner.admin_toolset` | 7 management-plane tools; `tool_filter` list or predicate; `GoogleTool` wrapping injects credentials automatically; complements `SpannerToolset` for data plane |

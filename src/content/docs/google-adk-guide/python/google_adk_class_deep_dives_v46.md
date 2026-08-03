---
title: "Class deep dives — volume 46 (App, EventsCompactionConfig, ResumabilityConfig, LlmEventSummarizer, BaseEventsSummarizer, ManagedAgent, LangGraphAgent, ToolConfirmation, McpInstructionProvider, SkillToolset)"
description: "10 source-verified deep dives for google-adk 2.6.1: App (top-level agentic application container; root_agent + plugins + compaction + resumability; strict name validation), EventsCompactionConfig (token-threshold pair + sliding-window pair; mutual-exclusion validator; custom summarizer slot), ResumabilityConfig (at-least-once pause/resume on long-running function calls; per-app opt-in), LlmEventSummarizer (LLM-based sliding-window compactor; _DEFAULT_PROMPT_TEMPLATE with language-identification directive; tool-call truncation at 2 000 chars; EventCompaction timestamp range), BaseEventsSummarizer (abstract compaction interface; @experimental; maybe_summarize_events contract), ManagedAgent (Managed Agents API via interactions.create; background=True streaming only; server-side tools + RemoteMcpServer; single_turn composition mode; lazy enterprise-or-developer Client), LangGraphAgent (CompiledStateGraph adapter; checkpointer-aware multi-turn; SystemMessage injection guard; _get_messages bifurcation), ToolConfirmation (@experimental TOOL_CONFIRMATION; camelCase aliases; from_response_dict wraps ADK client JSON-string wrapper), McpInstructionProvider (InstructionProvider from MCP Prompt; argument injection from session state; MCPSessionManager reuse), SkillToolset (five built-in skill tools; registry + local skills; adk_additional_tools metadata; invocation-scoped retry guard; script execution via code_executor or subprocess)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 46"
  order: 115
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.6.1**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `App` — the top-level application container

**Source:** `google/adk/apps/app.py`

### Why it matters

`App` replaces the ad-hoc pattern of wiring a `Runner` directly to a `BaseAgent`.
It bundles the root entry point, application-wide plugins, context-cache policy,
event-compaction strategy, and resumability config into one validated object that
the `Runner` and `AgentEngine` can introspect.

### Class signature

```python
from google.adk.apps import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents import LlmAgent

app = App(
    name="my_app",                     # must match ^[a-zA-Z][a-zA-Z0-9_-]*$
    root_agent=...,                    # BaseAgent OR BaseNode — required
    plugins=[],                        # list[BasePlugin], app-wide
    events_compaction_config=None,     # Optional[EventsCompactionConfig]
    context_cache_config=None,         # Optional[ContextCacheConfig]
    resumability_config=None,          # Optional[ResumabilityConfig]
)
```

Name rules are enforced by `validate_app_name`:

```python
import re
_VALID_APP_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

def validate_app_name(name: str) -> None:
    if not _VALID_APP_NAME_RE.match(name):
        raise ValueError(...)
    if name == "user":
        raise ValueError("App name cannot be 'user'; reserved for end-user input.")
```

### Example 1 — minimal App with an LlmAgent

```python
from google.adk.apps import App
from google.adk.agents import LlmAgent

root = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

app = App(name="demo", root_agent=root)
```

### Example 2 — App with compaction and resumability

```python
from google.adk.apps import App, EventsCompactionConfig, ResumabilityConfig

app = App(
    name="long_session_app",
    root_agent=root,
    events_compaction_config=EventsCompactionConfig(
        token_threshold=30_000,
        event_retention_size=10,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

### Example 3 — App with a Workflow root node

`App.root_agent` accepts a `BaseNode` (a `Workflow` instance) in addition to
any `BaseAgent`, so you can use the same container for graph-based orchestration.

```python
from google.adk.apps import App
from google.adk.workflow import Workflow

workflow = Workflow(
    name="pipeline",
    # ... nodes and edges defined elsewhere ...
)

app = App(name="workflow_app", root_agent=workflow)
# model_validator accepts BaseNode; "root_agent" is the field name regardless.
```

---

## 2 · `EventsCompactionConfig` — token-threshold and sliding-window compaction

**Source:** `google/adk/apps/_configs.py`

### Why it matters

As conversations grow, the prompt injected into each LLM call expands with the
full event history, eventually hitting model context limits.
`EventsCompactionConfig` configures two complementary strategies:

| Strategy | When it fires | Key params |
|---|---|---|
| Token-threshold | After any invocation, when the estimated prompt token count ≥ `token_threshold` | `token_threshold`, `event_retention_size` |
| Sliding-window | After every `compaction_interval` new user invocations | `compaction_interval`, `overlap_size` |

### Class signature

```python
@experimental
class EventsCompactionConfig(BaseModel):
    summarizer: Optional[BaseEventsSummarizer] = None
    compaction_interval: Optional[int] = Field(default=None, gt=0)
    overlap_size: Optional[int] = Field(default=None, ge=0)
    token_threshold: Optional[int] = Field(default=None, gt=0)
    event_retention_size: Optional[int] = Field(default=None, ge=0)
```

**Validation rules** (enforced by `@model_validator`):
- `token_threshold` and `event_retention_size` must be set **together**.
- `compaction_interval` and `overlap_size` must be set **together**.
- At least **one** trigger pair must be present.

### Example 1 — token-threshold only

```python
from google.adk.apps import EventsCompactionConfig

cfg = EventsCompactionConfig(
    token_threshold=20_000,   # compact when prompt ≥ 20k tokens
    event_retention_size=5,   # keep the 5 most recent raw events un-compacted
)
# ADK auto-creates an LlmEventSummarizer from the root LlmAgent's model.
```

### Example 2 — sliding-window only

```python
cfg = EventsCompactionConfig(
    compaction_interval=4,   # compact every 4 new user turns
    overlap_size=1,          # include 1 previous invocation for continuity
)
```

### Example 3 — both triggers active (token-threshold wins when both fire)

```python
cfg = EventsCompactionConfig(
    token_threshold=25_000,
    event_retention_size=8,
    compaction_interval=6,
    overlap_size=2,
)
# ADK checks token-threshold first; sliding-window fires only when the
# token check does not trigger.
```

---

## 3 · `ResumabilityConfig` — pause and resume long-running invocations

**Source:** `google/adk/apps/_configs.py`

### Why it matters

When an agent triggers a long-running function call (e.g. a multi-minute Cloud
Run job), ADK can *pause* the invocation — serialising its in-flight state —
and *resume* it later from the last recorded event without re-running earlier
steps. `ResumabilityConfig` is the opt-in switch.

### Class signature

```python
@experimental
class ResumabilityConfig(BaseModel):
    is_resumable: bool = False
```

**Guarantees:**
- Resume is **at-least-once**: idempotency of resumed tool calls is the
  caller's responsibility.
- Any ephemeral / in-memory state is lost on pause; only session-persisted
  events survive.

### Example 1 — enable globally for an app

```python
from google.adk.apps import App, ResumabilityConfig

app = App(
    name="durable_app",
    root_agent=root,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

### Example 2 — combine with token-threshold compaction

Long durable sessions benefit from compaction so the resumed prompt fits in
context even after many paused turns.

```python
from google.adk.apps import App, EventsCompactionConfig, ResumabilityConfig

app = App(
    name="durable_compact",
    root_agent=root,
    events_compaction_config=EventsCompactionConfig(
        token_threshold=30_000,
        event_retention_size=10,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

### Example 3 — guard: tool calls must be idempotent

```python
import asyncio

async def send_report(report_id: str) -> str:
    """Send a report — safe to call more than once for the same report_id."""
    # Idempotency key: the backend deduplicates on report_id.
    response = await _backend.send(report_id, dedupe_key=report_id)
    return response.status
```

---

## 4 · `LlmEventSummarizer` — LLM-based compaction summarizer

**Source:** `google/adk/apps/llm_event_summarizer.py`

### Why it matters

`LlmEventSummarizer` is the built-in `BaseEventsSummarizer` implementation.
When `EventsCompactionConfig.summarizer` is `None` and the root agent is an
`LlmAgent`, ADK auto-creates one of these from `agent.canonical_model`.
You can also instantiate it directly with a custom model and prompt template.

### Class signature

```python
class LlmEventSummarizer(BaseEventsSummarizer):
    _DEFAULT_PROMPT_TEMPLATE: str  # contains {conversation_history}
    _MAX_TOOL_CONTENT_CHARS: int = 2000

    def __init__(
        self,
        llm: BaseLlm,
        prompt_template: Optional[str] = None,
    ): ...

    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Optional[Event]: ...
```

### Default prompt template

The built-in template instructs the LLM to:
1. Explicitly state the **primary language** used by the user.
2. List **exact tool names** that were called (prevents tool-grounding drift).
3. Produce a **concise summary** of decisions, information, and open tasks.

Tool call args and responses are **truncated to 2 000 characters** so compaction
does not inflate the context it is meant to shrink.

### Example 1 — auto-creation via `EventsCompactionConfig` (no direct init)

```python
from google.adk.apps import App, EventsCompactionConfig

app = App(
    name="auto_summarizer",
    root_agent=LlmAgent(name="agent", model="gemini-2.5-pro", instruction="..."),
    events_compaction_config=EventsCompactionConfig(
        token_threshold=20_000,
        event_retention_size=5,
    ),
    # LlmEventSummarizer is created from root_agent.canonical_model automatically.
)
```

### Example 2 — explicit init with a fast model

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

summarizer = LlmEventSummarizer(llm=Gemini("gemini-2.5-flash"))

from google.adk.apps import EventsCompactionConfig
cfg = EventsCompactionConfig(
    summarizer=summarizer,
    compaction_interval=5,
    overlap_size=1,
)
```

### Example 3 — custom prompt template

```python
CUSTOM_TEMPLATE = (
    "You are summarizing a support conversation.\n"
    "Conversation:\n{conversation_history}\n\n"
    "Produce a bullet-point summary including: user problem, steps taken, "
    "current state, and any open action items."
)

summarizer = LlmEventSummarizer(
    llm=Gemini("gemini-2.5-flash"),
    prompt_template=CUSTOM_TEMPLATE,
)
```

---

## 5 · `BaseEventsSummarizer` — custom compaction interface

**Source:** `google/adk/apps/base_events_summarizer.py`

### Why it matters

Implement this ABC to plug any summarization backend — a fine-tuned model, a
RAG-based extractor, or a rule-based condenser — into ADK's compaction pipeline
without modifying the runner.

### Class signature

```python
@experimental
class BaseEventsSummarizer(abc.ABC):
    @abc.abstractmethod
    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Optional[Event]:
        """Return a compaction Event, or None to skip compaction this round."""
        ...
```

**Contract:**
- Return `None` if no compaction should happen (e.g. too few events).
- Return an `Event` whose `actions.compaction` field is populated with an
  `EventCompaction(start_timestamp, end_timestamp, compacted_content)`.
- The returned event is **appended** to the session by the runner; do not
  persist it yourself.

### Example 1 — rule-based summarizer (no LLM required)

```python
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.genai.types import Content, Part

class BulletSummarizer(BaseEventsSummarizer):
    async def maybe_summarize_events(self, *, events):
        if len(events) < 4:
            return None
        lines = []
        for e in events:
            if e.content and e.content.parts:
                for p in e.content.parts:
                    if p.text:
                        lines.append(f"- [{e.author}] {p.text[:120]}")
        summary_text = "\n".join(lines)
        compaction = EventCompaction(
            start_timestamp=events[0].timestamp,
            end_timestamp=events[-1].timestamp,
            compacted_content=Content(
                role="model",
                parts=[Part(text=summary_text)],
            ),
        )
        return Event(
            author="user",
            actions=EventActions(compaction=compaction),
            invocation_id=Event.new_id(),
        )
```

### Example 2 — wire into EventsCompactionConfig

```python
from google.adk.apps import App, EventsCompactionConfig

app = App(
    name="rule_based_compaction",
    root_agent=root,
    events_compaction_config=EventsCompactionConfig(
        summarizer=BulletSummarizer(),
        compaction_interval=3,
        overlap_size=0,
    ),
)
```

### Example 3 — async RAG-based summarizer

```python
class RagSummarizer(BaseEventsSummarizer):
    def __init__(self, retriever):
        self._retriever = retriever

    async def maybe_summarize_events(self, *, events):
        if not events:
            return None
        raw = "\n".join(
            f"{e.author}: {p.text}"
            for e in events
            if e.content
            for p in e.content.parts
            if p.text
        )
        # Retrieve relevant context then summarize.
        context = await self._retriever.fetch(raw)
        summary = await _llm_summarize(raw, context)
        compaction = EventCompaction(
            start_timestamp=events[0].timestamp,
            end_timestamp=events[-1].timestamp,
            compacted_content=Content(role="model", parts=[Part(text=summary)]),
        )
        return Event(
            author="user",
            actions=EventActions(compaction=compaction),
            invocation_id=Event.new_id(),
        )
```

---

## 6 · `ManagedAgent` — Managed Agents API agent

**Source:** `google/adk/agents/_managed_agent.py`

### Why it matters

`ManagedAgent` wraps Google's **Managed Agents API** (`interactions.create`).
Instead of running inference locally, it delegates to a server-hosted agent
identified by `agent_id`. This enables code-execution sandboxes, stateful remote
environments, and server-side tools (Google Search, Code Execution, URL Context,
Computer Use) without client-side infrastructure.

### Class signature

```python
class ManagedAgent(BaseAgent):
    agent_id: str
    environment: Optional[CreateAgentInteractionEnvironmentParam] = None
    agent_config: Optional[CreateAgentInteractionAgentConfigParam] = None
    instruction: Union[str, InstructionProvider] = ''
    tools: list[Union[types.Tool, BaseTool, RemoteMcpServer]] = []
    mode: Literal['single_turn'] | None = None
```

**Key constraints:**
- Supports **server-side tools only**: `types.Tool` with `google_search`,
  `code_execution`, `url_context`, or `computer_use`; `RemoteMcpServer` specs.
  Client-executed `FunctionTool` / callables are rejected at runtime.
- Always streams (`background=True`) via SSE; non-streaming polling is not yet
  supported.
- The Managed Agents API is only served from the `global` location; enterprise
  clients targeting any other location are rejected at init.

### Example 1 — basic remote agent with Google Search

```python
from google.adk.agents import ManagedAgent
from google.genai import types

agent = ManagedAgent(
    name="search_agent",
    description="Answers questions using Google Search.",
    agent_id="antigravity-preview-05-2026",
    tools=[types.Tool(google_search=types.GoogleSearch())],
)
```

### Example 2 — single-turn composition inside an LlmAgent

`mode='single_turn'` makes `ManagedAgent` behave like an inline tool of its
parent `LlmAgent`, preserving internal events in the shared session.

```python
from google.adk.agents import LlmAgent, ManagedAgent
from google.genai import types

coder = ManagedAgent(
    name="code_executor",
    description="Execute Python code and return results.",
    agent_id="code-execution-agent-id",
    mode="single_turn",
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-pro",
    instruction="Use the code_executor to run computations.",
    tools=[coder],          # ManagedAgent is valid in LlmAgent.tools
)
```

### Example 3 — persistent remote environment with a stateful sandbox

```python
agent = ManagedAgent(
    name="env_agent",
    agent_id="remote-sandbox-agent-id",
    environment={"type": "remote"},     # request a fresh sandbox
    instruction="You have a persistent Python environment. Use it.",
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)
# On the second turn, ADK reuses environment_id from the first interaction's
# session events so the sandbox state is preserved across turns.
```

---

## 7 · `LangGraphAgent` — LangGraph state graph adapter

**Source:** `google/adk/agents/langgraph_agent.py`

### Why it matters

`LangGraphAgent` wraps a **compiled LangGraph `StateGraph`** as a first-class
ADK `BaseAgent`. This lets you use LangGraph's graph DSL, persistence
checkpointers, and custom reducers while still running inside the ADK runner,
session service, and evaluation framework.

### Class signature

```python
class LangGraphAgent(BaseAgent):
    graph: CompiledStateGraph
    instruction: str = ''
```

### Memory / multi-turn behaviour

| `graph.checkpointer` | Message strategy |
|---|---|
| `None` (no persistence) | `_get_conversation_with_agent`: all user ↔ agent messages across the full session |
| Set (e.g. `MemorySaver`) | `_get_last_human_messages`: only the latest block of user messages (LangGraph owns memory) |

When a checkpointer is configured, `thread_id` is set to `ctx.session.id` so
LangGraph and ADK share the same conversation identity.

### Example 1 — stateless graph (no checkpointer)

```python
from langgraph.graph import StateGraph, MessagesState, END
from google.adk.agents.langgraph_agent import LangGraphAgent

def my_node(state: MessagesState):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", my_node)
builder.set_entry_point("assistant")
builder.add_edge("assistant", END)
graph = builder.compile()

agent = LangGraphAgent(
    name="lg_agent",
    description="A LangGraph-powered assistant.",
    graph=graph,
    instruction="You are a helpful assistant.",
)
```

### Example 2 — multi-turn graph with MemorySaver

```python
from langgraph.checkpoint.memory import MemorySaver
import os

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"  # required before compile

graph = builder.compile(checkpointer=MemorySaver())

agent = LangGraphAgent(
    name="lg_memory_agent",
    graph=graph,
    instruction="Remember the user's preferences across turns.",
)
# ADK passes thread_id=session.id; LangGraph replays its own memory.
# Only the latest user messages are extracted from ADK events each turn.
```

### Example 3 — embedding LangGraphAgent in a multi-agent system

```python
from google.adk.agents import LlmAgent, SequentialAgent

lg_agent = LangGraphAgent(name="lg_worker", graph=graph)

pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[lg_agent, LlmAgent(name="reviewer", model="gemini-2.5-flash",
                                   instruction="Review the previous output.")],
)
```

---

## 8 · `ToolConfirmation` — human-in-the-loop tool gate

**Source:** `google/adk/tools/tool_confirmation.py`

### Why it matters

`ToolConfirmation` is the data model that flows between the agent runtime and
a human (or automated policy) when a tool requires explicit approval before
execution. A tool calls `tool_context.request_confirmation(hint="...")` to
**pause** the turn; the runner surfaces the hint to the frontend; the frontend
sends back a response that is read from `tool_context.tool_confirmation` on
re-entry — if `tool_context.tool_confirmation.confirmed` is `True`, the tool
proceeds; otherwise it cancels.

### Class signature

```python
import json
from typing import Any, Optional

from pydantic import alias_generators, BaseModel, ConfigDict

from google.adk.features import experimental, FeatureName

@experimental(FeatureName.TOOL_CONFIRMATION)
class ToolConfirmation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=alias_generators.to_camel,   # hint → hint, confirmed → confirmed, payload → payload
        populate_by_name=True,
    )
    hint: str = ""          # shown to the user explaining why approval is needed
    confirmed: bool = False # True = approved, False = pending/rejected
    payload: Optional[Any] = None  # any JSON-serialisable data from the user

    @classmethod
    def from_response_dict(cls, response: dict[str, Any]) -> ToolConfirmation:
        """Handles the ADK client's {'response': json_string} wrapper."""
        if response and len(response) == 1 and "response" in response:
            return cls.model_validate(json.loads(response["response"]))
        return cls.model_validate(response)
```

### Example 1 — tool that requires confirmation before deletion

The correct pattern is to call `tool_context.request_confirmation(hint=...)` on the
first invocation, which pauses the turn. The runner re-invokes the tool once the
frontend responds; the reply is available via `tool_context.tool_confirmation`.

```python
from google.adk.tools.tool_context import ToolContext

async def delete_record(record_id: str, tool_context: ToolContext) -> dict:
    if not tool_context.tool_confirmation:
        # First call: request approval and pause.
        tool_context.request_confirmation(
            hint=f"Are you sure you want to delete record '{record_id}'?",
        )
        return {"pending": "Awaiting confirmation before deleting record."}

    if not tool_context.tool_confirmation.confirmed:
        return {"status": "cancelled"}

    # Confirmed — proceed with deletion.
    _db.delete(record_id)
    return {"status": "deleted", "id": record_id}
```

### Example 2 — reading a confirmation response from a function response dict

```python
# Simulated response from an ADK web client (JSON-wrapped).
raw = {"response": '{"hint": "", "confirmed": true, "payload": {"reason": "approved by admin"}}'}
conf = ToolConfirmation.from_response_dict(raw)
assert conf.confirmed is True
assert conf.payload["reason"] == "approved by admin"
```

### Example 3 — camelCase alias compatibility

The camelCase alias generator means `ToolConfirmation` can be serialised and
deserialised from both Python-native and JSON-API forms:

```python
tc = ToolConfirmation(hint="Approve?", confirmed=False)
# JSON form (camelCase keys) — matches what the frontend sends.
assert tc.model_dump(by_alias=True) == {"hint": "Approve?", "confirmed": False, "payload": None}
# Python form (snake_case keys) — works too because populate_by_name=True.
tc2 = ToolConfirmation.model_validate({"hint": "Approve?", "confirmed": True})
assert tc2.confirmed is True
```

---

## 9 · `McpInstructionProvider` — fetch agent instructions from an MCP server

**Source:** `google/adk/agents/mcp_instruction_provider.py`

### Why it matters

`McpInstructionProvider` implements the `InstructionProvider` protocol
(`Callable[[ReadonlyContext], str | Awaitable[str]]`) by fetching a **named MCP
Prompt** at request time. This lets you centralise agent instructions in an MCP
server — updating them without redeploying the agent — and inject session state
values into the prompt's arguments automatically.

### Class signature

```python
class McpInstructionProvider(InstructionProvider):
    def __init__(
        self,
        connection_params: Any,         # e.g. StdioServerParameters or SseConnectionParams
        prompt_name: str,               # MCP Prompt name
        errlog: TextIO = sys.stderr,
    ): ...

    async def __call__(self, context: ReadonlyContext) -> str: ...
```

**Runtime behaviour:**
1. Creates (or reuses) an `MCPSessionManager` session.
2. Calls `session.list_prompts()` to discover which arguments the named prompt
   requires.
3. Extracts matching argument values from `context.state` by key name.
4. Calls `session.get_prompt(prompt_name, arguments=prompt_args)`.
5. Concatenates all `text`-type message contents and returns the result.

### Example 1 — stdio MCP server (local binary)

```python
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider

instruction_provider = McpInstructionProvider(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"],
    ),
    prompt_name="agent_system_prompt",
)

agent = LlmAgent(
    name="dynamic_agent",
    model="gemini-2.5-pro",
    instruction=instruction_provider,   # called at each invocation
)
```

### Example 2 — SSE MCP server with state-injected arguments

If the MCP Prompt declares an argument named `user_role`, ADK automatically
looks up `context.state["user_role"]` and passes it to the server.

```python
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

provider = McpInstructionProvider(
    connection_params=SseConnectionParams(url="https://prompts.internal/mcp"),
    prompt_name="role_aware_prompt",
)

# In session state before the invocation:
# session.state["user_role"] = "admin"
# → the MCP server receives arguments={"user_role": "admin"}
```

### Example 3 — error handling and fallback instruction

`McpInstructionProvider.__call__` raises `ValueError` if the prompt is empty or
if the server returns no messages. Wrap it in a provider that falls back to a
static string.

```python
class SafeInstructionProvider:
    def __init__(self, mcp_provider, fallback):
        self._mcp = mcp_provider
        self._fallback = fallback

    async def __call__(self, context):
        try:
            return await self._mcp(context)
        except Exception:
            return self._fallback

agent = LlmAgent(
    name="resilient_agent",
    model="gemini-2.5-flash",
    instruction=SafeInstructionProvider(
        mcp_provider=provider,
        fallback="You are a helpful assistant.",
    ),
)
```

---

## 10 · `SkillToolset` — discover, load, and execute agent skills

**Source:** `google/adk/tools/skill_toolset.py`

### Why it matters

`SkillToolset` is a `BaseToolset` that turns a collection of **skills** —
structured folders of markdown instructions plus optional references, assets,
and scripts — into first-class LLM tools. The agent discovers skills via
`list_skills`, loads their instructions via `load_skill`, browses resources via
`load_skill_resource`, runs shell scripts via `run_skill_script`, and (when a
`SkillRegistry` is provided) searches a remote registry via `search_skills`.

### Built-in tools exposed

| Tool | Description |
|---|---|
| `list_skills` | List all locally registered skills |
| `load_skill` | Load `SKILL.md` instructions for a skill by name |
| `load_skill_resource` | Browse `references/`, `assets/`, `scripts/` files |
| `run_skill_script` | Execute a script from `scripts/` via code executor or subprocess |
| `search_skills` | Search the remote `SkillRegistry` (only when `registry` is provided) |

### Class signature

```python
class SkillToolset(BaseToolset):
    def __init__(
        self,
        skills: list[Skill] | None = None,
        *,
        registry: SkillRegistry | None = None,
        code_executor: BaseCodeExecutor | None = None,
        script_timeout: int = 300,
        additional_tools: list[ToolUnion] | None = None,
        tool_name_prefix: str | None = None,
        tool_filter: ToolPredicate | list[str] | None = None,
    ): ...
```

### Skill directory layout

```
skills/
  my_skill/
    SKILL.md          # required: frontmatter + markdown instructions
    references/
      api_docs.md
    assets/
      email_template.txt
    scripts/
      setup.sh
```

`SKILL.md` frontmatter may include `adk_additional_tools: [tool_name]` to
unlock extra tools from `additional_tools` when the skill is activated.

### Example 1 — local skills only

```python
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.agents import LlmAgent

# load_skill_from_dir is synchronous — no await needed.
email_skill = load_skill_from_dir("skills/email_drafter")

toolset = SkillToolset(skills=[email_skill])

agent = LlmAgent(
    name="skill_agent",
    model="gemini-2.5-pro",
    instruction=(
        "You have access to skills. Use list_skills to discover them, "
        "then load_skill before acting."
    ),
    tools=[toolset],
)
```

### Example 2 — remote SkillRegistry with search

`SkillRegistry` is an abstract base class. To connect to a remote catalogue,
implement `get_skill` and `search_skills` against your own backend.

```python
from typing import Optional
from google.adk.skills import Skill, SkillRegistry
from google.adk.skills.models import Frontmatter
from google.adk.tools.skill_toolset import SkillToolset
import httpx

class HttpSkillRegistry(SkillRegistry):
    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    async def get_skill(self, *, name: str) -> Optional[Skill]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self._endpoint}/skills/{name}")
        return Skill.model_validate(resp.json()) if resp.is_success else None

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self._endpoint}/skills", params={"q": query})
        return [Frontmatter.model_validate(s) for s in resp.json()]

registry = HttpSkillRegistry(endpoint="https://skills.internal/api")
toolset = SkillToolset(registry=registry)

# Agent can now call search_skills(query="send email") to find remote skills
# on demand, without loading them all upfront.
```

### Example 3 — tool_name_prefix for multi-toolset agents

When an agent uses more than one `SkillToolset`, prefix tool names to avoid
collisions.

```python
from google.adk.tools.skill_toolset import SkillToolset

marketing_toolset = SkillToolset(
    skills=[marketing_skill],
    tool_name_prefix="marketing",
    # Exposed as: marketing_list_skills, marketing_load_skill, …
)

legal_toolset = SkillToolset(
    skills=[legal_skill],
    tool_name_prefix="legal",
    # Exposed as: legal_list_skills, legal_load_skill, …
)

agent = LlmAgent(
    name="multi_skill_agent",
    model="gemini-2.5-pro",
    instruction="Use marketing or legal skills as appropriate.",
    tools=[marketing_toolset, legal_toolset],
)
```

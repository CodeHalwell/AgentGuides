---
title: "Class deep dives — volume 34 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: App (all config knobs: plugins/events_compaction_config/resumability_config/context_cache_config), FunctionTool (Pydantic arg coercion; require_confirmation callable; async callables; mandatory args guard), Context/ToolContext unified write API (state prefixes; artifacts; memory; route; output; run_node; interrupt), InMemoryArtifactService (versioning CRUD; user namespace; list_versions/list_artifact_versions/get_artifact_version), Workflow.state_schema (Pydantic-validated shared state; max_concurrency throttle), OpenAPIToolset (preserve_property_names; httpx_client_factory; ssl_verify with custom CA), McpToolset (header_provider dynamic auth; use_mcp_resources; progress_callback factory), LongRunningFunctionTool (asyncio.create_task background; companion poll; state tracking), LoopAgent→Workflow migration (source-verified _get_start_state internals; max_iterations→routing replacement), ParallelAgent→Workflow fan-out migration (branch isolation internals; JoinNode replacement)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 34"
  order: 103
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.3.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `App` — the application container

**Source:** `google/adk/apps/app.py`

`App` is a Pydantic `BaseModel` (`extra="forbid"`) that acts as the
top-level container wiring together a root agent/node with runner-wide
configuration. It is the recommended way to hand a root node to `Runner`
because it carries four optional config blocks that the runner's
`Runner.__init__` reads directly (`runners.py:_resolve_app`).

### Constructor signature (verified `app.py:55-113`)

```python
class App(BaseModel):
    name: str
    root_agent: Union[BaseAgent, BaseNode, None] = None   # required at runtime
    plugins: list[BasePlugin] = []
    events_compaction_config: Optional[EventsCompactionConfig] = None
    context_cache_config: Optional[ContextCacheConfig] = None
    resumability_config: Optional[ResumabilityConfig] = None
```

`validate_app_name(name)` enforces `^[a-zA-Z][a-zA-Z0-9_-]*$` and bans
the name `"user"` (reserved for the user-message namespace in session
state). `root_agent` must be a `BaseAgent` or `BaseNode` instance; a bare
`None` raises `ValueError` at validation time.

### Example 1 — minimal App with a plugin

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import LoggingPlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="helper",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

app = App(
    name="demo_app",
    root_agent=agent,
    plugins=[LoggingPlugin()],  # runner-wide; applies to every agent turn
)

async def main():
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="demo_app", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Hello")]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 2 — App with `EventsCompactionConfig`

`EventsCompactionConfig` (imported from `google.adk.apps.app`) controls how
the runner summarises the conversation history when it grows long.
`compaction_interval` is the number of **new user-initiated invocations** that
must accumulate before the sliding-window compaction fires — not a token count.
`overlap_size` is the number of preceding invocations retained for context
continuity. To trigger token-based compaction instead, also set
`token_threshold` + `event_retention_size`.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

# Compact after every 5 invocations; keep the previous 2 for context overlap.
compaction_cfg = EventsCompactionConfig(
    compaction_interval=5,
    overlap_size=2,
)

agent = LlmAgent(
    name="long_chat",
    model="gemini-2.5-flash",
    instruction="You are a thorough assistant capable of multi-turn reasoning.",
)

app = App(
    name="long_chat_app",
    root_agent=agent,
    events_compaction_config=compaction_cfg,
)

async def main():
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="long_chat_app", user_id="u1"
    )
    for turn_text in ["Tell me about LLMs.", "Now explain transformers.", "Summarise both."]:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=turn_text)]),
        ):
            if event.is_final_response() and event.content:
                print(f"→ {event.content.parts[0].text[:80]}...")
    await runner.close()

asyncio.run(main())
```

### Example 3 — App with `ResumabilityConfig` for HITL workflows

`ResumabilityConfig(is_resumable=True)` enables pause-and-resume: when a
node yields `RequestInput` the runner emits an interrupt event; calling
`Runner.run_async` again with a matching `FunctionResponse` part resumes
the workflow from that point.

Key correctness rules:
- The first node receives the user's initial message as **`node_input`**,
  not a custom-named parameter (custom params are resolved from state).
- With `rerun_on_resume=True` the node restarts from scratch on resume.
  The turn-2 message contains only a `FunctionResponse` part (no text),
  so `node_input` is empty on the second run. **Always save the original
  input to `ctx.state` before yielding** and restore it from state on
  resume. Read the user's answer from `ctx.resume_inputs[interrupt_id]`;
  the `yield` expression always evaluates to `None`.
- Turn 2 must send only a `FunctionResponse` part — mixing text and
  function-response parts in the same message raises `ValueError`.

```python
import asyncio
from google.adk.apps import App
from google.adk.apps._configs import ResumabilityConfig
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_response,
    get_request_input_interrupt_ids,
)
from google.genai import types

@node(rerun_on_resume=True)
async def approval_gate(node_input: str, ctx):
    # On resume ctx.resume_inputs is populated — check before yielding.
    if decision := ctx.resume_inputs.get("review"):
        # node_input is empty on resume (turn-2 message has no text), so
        # restore the original draft that was saved to state on the first run.
        draft = ctx.state.get("hitl_draft", "")
        approved = str(decision.get("value", "no")).lower() == "yes"
        ctx.output = draft
        ctx.route = "publish" if approved else "revise"
        return
    # First run: save draft to state so it survives the resume rerun.
    ctx.state["hitl_draft"] = node_input
    yield RequestInput(
        interrupt_id="review",
        message=f"Approve this draft?\n\n{node_input}\n\nReply yes/no.",
    )

@node
def publish(node_input: str) -> str:
    return f"[PUBLISHED] {node_input}"

@node
def revise(node_input: str) -> str:
    return f"[NEEDS REVISION] {node_input}"

pipeline = Workflow(
    name="review_pipeline",
    edges=[
        (START, approval_gate, {"publish": publish, "revise": revise}),
    ],
)

app = App(
    name="hitl_app",
    root_agent=pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

async def main():
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="hitl_app", user_id="u1"
    )
    # Turn 1: start the workflow; capture the interrupt id from the event.
    interrupt_id = None
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Draft: the sky is blue.")]),
    ):
        ids = get_request_input_interrupt_ids(event)
        if ids:
            interrupt_id = ids[0]
        print(f"[turn1] {event}")

    # Turn 2: resume with a FunctionResponse — do NOT mix with text parts.
    resume_part = create_request_input_response(
        interrupt_id or "review", {"value": "yes"}
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[resume_part]),
    ):
        if event.is_final_response() and event.content:
            print(f"[turn2] {event.content.parts[0].text}")
    await runner.close()

asyncio.run(main())
```

---

## 2 · `FunctionTool` — deep dive into arg handling

**Source:** `google/adk/tools/function_tool.py`

`FunctionTool` wraps any callable (sync or async, regular function or
callable object) into an ADK tool. Key internals not obvious from the
public API:

- **`_preprocess_args`** — converts raw `dict` values from the LLM into
  Pydantic model instances when the parameter's type annotation is a
  `BaseModel` subclass (or `Optional[BaseModel]`, `list[BaseModel]`).
  Conversion happens **before** the function is called.
- **`_get_mandatory_args`** — inspects `inspect.signature` to find all
  params without defaults (excluding VAR_POSITIONAL/VAR_KEYWORD). Missing
  mandatory args short-circuit to `{"error": "...missing parameters..."}`.
- **`require_confirmation`** — a `bool` **or** a callable invoked with the
  preprocessed tool arguments as **keyword arguments** (same keys passed
  to the tool function itself) and returning `bool`. When truthy:
  1. Calls `tool_context.request_confirmation(hint=...)`.
  2. Sets `tool_context.actions.skip_summarization = True`.
  3. Returns `{"error": "This tool call requires confirmation..."}`.
  The next turn must carry a `FunctionResponse` with `ToolConfirmation`.
- **Context injection** — a param named `tool_context` **or** typed as
  `ToolContext`/`Context` is injected with the current `ToolContext`; it
  is stripped from the model's view of the tool declaration.

### Example 1 — Pydantic model auto-coercion

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

class Address(BaseModel):
    street: str
    city: str
    postcode: str

class Order(BaseModel):
    item_id: str
    quantity: int
    shipping_address: Address

# The LLM sends {"order": {"item_id": "SKU-1", "quantity": 2, "shipping_address": {"street": "..."}}}
# The outer key must match the parameter name ("order"). _preprocess_args then
# converts the inner dict → Order (and shipping_address dict → Address) automatically.
def place_order(order: Order) -> dict:
    """Place a product order.

    Args:
      order: The order details including item, quantity, and shipping address.
    Returns:
      A dict with order_id and status.
    """
    return {
        "order_id": f"ORD-{order.item_id}-{order.quantity}",
        "status": "confirmed",
        "ship_to": f"{order.shipping_address.street}, {order.shipping_address.city}",
    }

order_tool = FunctionTool(func=place_order)

agent = LlmAgent(
    name="shop",
    model="gemini-2.5-flash",
    instruction="Help the user place orders. Use the place_order tool.",
    tools=[order_tool],
)

async def main():
    app = App(name="shop_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="shop_app", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="Order 2 units of SKU-42, ship to 123 Main St, Springfield, 62701"
        )]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 2 — `require_confirmation` as a callable predicate

```python
from google.adk.tools import FunctionTool

_ALLOWED_TABLES = {"orders", "sessions", "cache"}

def delete_records(table: str, where_clause: str) -> dict:
    """Delete rows from a database table.

    Args:
      table: The table name (must be on the allow-list).
      where_clause: A structured filter like 'user_id=42' or '1=1'.
    Returns:
      A dict with rows_deleted count.

    NOTE: `table` is validated against an allow-list — it cannot be
    parameterised in SQL. `where_clause` is a free-form string and
    CANNOT be substituted with a single SQL placeholder; use a
    structured filter type (e.g. {'column': ..., 'value': ...}) in
    production to allow safe parameterisation of individual values.
    """
    if table not in _ALLOWED_TABLES:
        return {"error": f"Table '{table}' is not in the allowed list."}
    # Stub — real code would parse where_clause into column/value pairs
    # and use parameterised queries for the values.
    return {"rows_deleted": 0, "table": table}

# Only require confirmation when the where_clause is the dangerous "1=1"
def needs_confirm(table: str, where_clause: str) -> bool:
    return where_clause.strip() in ("1=1", "true", "TRUE")

delete_tool = FunctionTool(
    func=delete_records,
    require_confirmation=needs_confirm,
)

# The predicate receives the preprocessed args as keyword arguments.
# A targeted delete ("user_id = 5") proceeds directly; a full-table
# wipe ("1=1") triggers the confirmation round-trip.
```

### Example 3 — async callable + mandatory-args guard

```python
import asyncio
import httpx
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

async def fetch_weather(city: str, units: str = "metric", tool_context: ToolContext | None = None) -> dict:
    """Fetch current weather for a city.

    Args:
      city: The city name.
      units: Unit system — 'metric' or 'imperial'.
    Returns:
      A dict with temperature, description, and humidity.
    """
    # `city` is mandatory (no default); `units` and `tool_context` are optional.
    # FunctionTool._get_mandatory_args returns ["city"].
    # If the LLM forgets to pass city, the tool returns:
    #   {"error": "Invoking `fetch_weather()` failed as the following mandatory
    #              input parameters are not present:\ncity\n..."}
    if tool_context:
        tool_context.state["last_weather_city"] = city
    async with httpx.AsyncClient() as _client:
        # Simulated response — replace with a real weather API call
        return {"city": city, "temperature": 22, "units": units, "description": "sunny"}

weather_tool = FunctionTool(func=fetch_weather)

agent = LlmAgent(
    name="weather_bot",
    model="gemini-2.5-flash",
    instruction="Provide weather information. Always use the fetch_weather tool.",
    tools=[weather_tool],
)

async def main():
    app = App(name="weather_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="weather_app", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="What's the weather in London?"
        )]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

---

## 3 · `Context` unified write API (= `ToolContext` = `CallbackContext`)

**Source:** `google/adk/agents/context.py`

In 2.x, `ToolContext` and `CallbackContext` are both aliases for `Context`
(`tools/tool_context.py`: `ToolContext = Context`;
`agents/callback_context.py`: `CallbackContext = Context`). `Context`
extends `ReadonlyContext` with every write capability available inside
tools and callbacks.

Key write methods (all verified against `context.py`):

| Property / method | Available in | Purpose |
|---|---|---|
| `ctx.state[key] = val` | tools + callbacks | Mutate session state (app:/user:/temp: prefixes apply) |
| `ctx.route = "key"` | `@node` functions | Set conditional edge for the current workflow step |
| `ctx.output = val` | `@node` functions | Set the node's explicit output value |
| `await ctx.save_artifact(filename, artifact)` | tools + callbacks | Save a versioned binary/text artifact |
| `await ctx.load_artifact(filename, version=None)` | tools + callbacks | Load an artifact |
| `await ctx.list_artifacts()` | tools + callbacks | List artifact filenames in scope |
| `await ctx.get_artifact_version(filename, version=None)` | tools + callbacks | Metadata only |
| `await ctx.search_memory(query)` | tools | Search long-term memory |
| `await ctx.add_events_to_memory(events=events)` | callbacks (after_agent) | Ingest a list of `Event` objects into long-term memory |
| `await ctx.add_session_to_memory()` | callbacks (after_agent) | Ingest the entire current session into long-term memory |
| `await ctx.add_memory(memories=entries)` | callbacks (after_agent) | Inject explicit `MemoryEntry` objects directly (keyword-only) |
| `ctx.request_confirmation(hint=None)` | tool functions | Trigger HITL confirmation |
| `await ctx.run_node(node, input)` | `@node` functions | Dynamically invoke another node |
| `yield RequestInput(interrupt_id=..., message=...)` | `@node` functions | Pause workflow for user input |

### Example 1 — state management with scope prefixes

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def update_preferences(preferred_language: str, theme: str, tool_context: ToolContext) -> dict:
    """Update the user's display preferences.

    Args:
      preferred_language: The user's preferred language code (e.g. 'en', 'fr').
      theme: UI theme — 'light' or 'dark'.
    Returns:
      Confirmation dict.
    """
    # session-scoped: lost when session ends
    tool_context.state["session_theme"] = theme

    # user-scoped: persists across all sessions for this user
    tool_context.state["user:preferred_language"] = preferred_language

    # app-scoped: visible to all users and sessions of this app
    tool_context.state["app:last_pref_update"] = "2026-07-02T00:00:00Z"

    # temp-scoped: only for this invocation; stripped before state is committed
    tool_context.state["temp:validation_done"] = True

    return {"updated": True, "language": preferred_language, "theme": theme}

prefs_tool = FunctionTool(func=update_preferences)
```

### Example 2 — artifact save/load inside tools

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

async def generate_report(title: str, tool_context: ToolContext) -> dict:
    """Generate a text report and save it as an artifact.

    Args:
      title: The report title.
    Returns:
      A dict with the artifact filename and version.
    """
    report_text = f"# {title}\n\nThis is a generated report."
    artifact = types.Part(text=report_text)

    filename = f"{title.replace(' ', '_').lower()}.txt"
    version = await tool_context.save_artifact(filename=filename, artifact=artifact)
    return {"filename": filename, "version": version, "saved": True}

async def read_report(filename: str, tool_context: ToolContext) -> dict:
    """Read a previously saved report.

    Args:
      filename: The artifact filename.
    Returns:
      A dict with the report content.
    """
    part = await tool_context.load_artifact(filename=filename)
    if part is None:
        return {"error": f"No artifact found: {filename}"}
    return {"content": part.text, "filename": filename}

agent = LlmAgent(
    name="report_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Use generate_report to create reports and read_report to retrieve them. "
        "Always generate before reading."
    ),
    tools=[FunctionTool(generate_report), FunctionTool(read_report)],
)

async def main():
    artifact_service = InMemoryArtifactService()
    session_service = InMemorySessionService()
    app = App(name="report_app", root_agent=agent)
    runner = Runner(
        app=app,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    session = await session_service.create_session(app_name="report_app", user_id="u1")
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="Generate a report called Q1 Summary, then read it back."
        )]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 3 — `ctx.route` and `ctx.output` in a workflow node

```python
from google.adk.workflow import Workflow, node, START, DEFAULT_ROUTE

@node
async def classify_intent(node_input: str, ctx) -> str:
    # ctx.route controls which edge the workflow follows after this node.
    # ctx.output is the value passed to the downstream node.
    # By default ctx.output = the function's return value (if it returns one).
    # Use node_input (not a custom name) so FunctionNode passes the edge
    # payload here rather than looking it up in ctx.state.
    if "refund" in node_input.lower():
        ctx.route = "billing"
    elif "broken" in node_input.lower() or "error" in node_input.lower():
        ctx.route = "support"
    else:
        ctx.route = DEFAULT_ROUTE
    # Return the input as the node output; do NOT also set ctx.output —
    # the output setter raises ValueError('Output already set...') if both
    # the return value and ctx.output are provided.
    return node_input

@node
async def billing(node_input: str, ctx) -> str:
    return f"[BILLING] {node_input}"

@node
async def support(node_input: str, ctx) -> str:
    return f"[SUPPORT] {node_input}"

@node
async def general(node_input: str, ctx) -> str:
    return f"[GENERAL] {node_input}"

triage = Workflow(
    name="triage",
    edges=[
        (START, classify_intent, {
            "billing": billing,
            "support": support,
            DEFAULT_ROUTE: general,
        }),
    ],
)
```

---

## 4 · `InMemoryArtifactService` — versioning CRUD

**Source:** `google/adk/artifacts/in_memory_artifact_service.py`

`InMemoryArtifactService` stores artifacts in a Python dict keyed by a
path string built in `_artifact_path`. Two scoping rules (verified in
`_file_has_user_namespace` and `_artifact_path`):

- **Session-scoped** — `session_id` is required; path =
  `{app}/{user}/{session}/{filename}`.
- **User-scoped** — `filename` starts with `"user:"`; `session_id` is
  ignored; path = `{app}/{user}/user/{filename}`.

Version numbers are 0-based integers. Each call to `save_artifact` appends
a new `_ArtifactEntry`; the version number equals the current list length
before appending. `load_artifact(version=None)` returns the **last** entry
(`versions[-1]`); `load_artifact(version=0)` returns the first.

### Example 1 — full CRUD + versioning

```python
import asyncio
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

async def main():
    svc = InMemoryArtifactService()
    kwargs = dict(app_name="app", user_id="u1", session_id="s1")

    # Save three versions of the same file
    v0 = await svc.save_artifact(
        **kwargs, filename="notes.txt",
        artifact=types.Part(text="Draft 1"),
    )
    v1 = await svc.save_artifact(
        **kwargs, filename="notes.txt",
        artifact=types.Part(text="Draft 2 — with corrections"),
    )
    v2 = await svc.save_artifact(
        **kwargs, filename="notes.txt",
        artifact=types.Part(text="Final version"),
        custom_metadata={"approved_by": "alice"},
    )
    print(f"Versions saved: {v0}, {v1}, {v2}")  # 0, 1, 2

    # Load latest
    latest = await svc.load_artifact(**kwargs, filename="notes.txt")
    print(f"Latest: {latest.text}")  # Final version

    # Load a specific version
    first = await svc.load_artifact(**kwargs, filename="notes.txt", version=0)
    print(f"Version 0: {first.text}")  # Draft 1

    # List version numbers only
    versions = await svc.list_versions(**kwargs, filename="notes.txt")
    print(f"Versions: {versions}")  # [0, 1, 2]

    # List full ArtifactVersion metadata (uri, mime_type, custom_metadata)
    metas = await svc.list_artifact_versions(**kwargs, filename="notes.txt")
    for m in metas:
        print(f"  v{m.version}: {m.canonical_uri}  mime={m.mime_type}")

    # Get metadata for a single version
    meta_v2 = await svc.get_artifact_version(**kwargs, filename="notes.txt", version=2)
    print(f"v2 metadata: {meta_v2.custom_metadata}")  # {"approved_by": "alice"}

    # Delete
    await svc.delete_artifact(**kwargs, filename="notes.txt")
    keys = await svc.list_artifact_keys(**kwargs)
    print(f"After delete: {keys}")  # []

asyncio.run(main())
```

### Example 2 — user-namespace artifacts (cross-session)

```python
import asyncio
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

async def main():
    svc = InMemoryArtifactService()

    # Save a user-scoped artifact — survives session deletion
    # The "user:" prefix is detected by _file_has_user_namespace()
    version = await svc.save_artifact(
        app_name="crm",
        user_id="alice",
        session_id="s1",                    # session_id is ignored for user: files
        filename="user:profile.json",
        artifact=types.Part(text='{"name": "Alice", "tier": "gold"}'),
    )
    print(f"Saved at version: {version}")   # 0

    # Load from a completely different session — still works
    part = await svc.load_artifact(
        app_name="crm",
        user_id="alice",
        session_id="s999",                  # different session; doesn't matter
        filename="user:profile.json",
    )
    print(f"Profile: {part.text}")          # {"name": "Alice", "tier": "gold"}

    # List keys — user: files appear in both session and user-scoped listings
    all_keys = await svc.list_artifact_keys(
        app_name="crm", user_id="alice", session_id=None
    )
    print(f"User-scoped keys: {all_keys}")  # ["user:profile.json"]

asyncio.run(main())
```

### Example 3 — binary artifact (inline_data)

```python
import asyncio
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

async def main():
    svc = InMemoryArtifactService()
    kwargs = dict(app_name="app", user_id="u1", session_id="s1")

    # Save a PNG image as inline_data
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100   # fake PNG bytes
    version = await svc.save_artifact(
        **kwargs,
        filename="chart.png",
        artifact=types.Part(
            inline_data=types.Blob(mime_type="image/png", data=fake_png)
        ),
    )

    # Retrieve and check mime type via metadata
    meta = await svc.get_artifact_version(**kwargs, filename="chart.png", version=version)
    print(f"MIME: {meta.mime_type}")   # image/png
    print(f"URI: {meta.canonical_uri}")

    # Load the actual bytes back
    part = await svc.load_artifact(**kwargs, filename="chart.png")
    print(f"Bytes recovered: {len(part.inline_data.data)}")

asyncio.run(main())
```

---

## 5 · `Workflow.state_schema` — validated shared state

**Source:** `google/adk/workflow/_workflow.py`, `google/adk/workflow/_base_node.py`

`state_schema` is a Pydantic `BaseModel` class (not an instance) that
ADK uses to validate `ctx.state` mutations inside the workflow. When set:

1. `ctx.state` changes must match the schema fields (enforced by
   `StateSchemaError` in `sessions/state.py`).
2. Schema fields declared in `state_schema` are available as **injected
   parameters** in `@node` functions — you can declare them by name in
   the function signature and ADK injects the current value.
3. `max_concurrency` caps simultaneously running *graph-scheduled* nodes.
   Dynamic nodes (`ctx.run_node(...)`) are explicitly excluded from the
   semaphore to avoid deadlocks.

### Example 1 — schema-validated pipeline state

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START
from google.genai import types

class PipelineState(BaseModel):
    raw_text: str = ""
    word_count: int = 0
    summary: str = ""

@node
def preprocess(node_input: str, ctx, raw_text: str = "") -> str:
    # `raw_text` is injected from state because it matches a PipelineState field
    ctx.state["raw_text"] = node_input
    ctx.state["word_count"] = len(node_input.split())
    return node_input

summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the input in one sentence.",
    mode="single_turn",
    output_key="summary",   # writes to state["summary"]
)

pipeline = Workflow(
    name="summarise_pipeline",
    state_schema=PipelineState,
    edges=[(START, preprocess, summariser)],
)

async def main():
    app = App(name="demo", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="demo", user_id="u1")
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="Artificial intelligence is transforming many industries."
        )]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 2 — `max_concurrency` to throttle a fan-out

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, JoinNode, node, START
from google.genai import types

# Three independent analyst agents, but only 2 should run concurrently
analyst_a = LlmAgent(name="a", model="gemini-2.5-flash", instruction="Analyse from the financial angle.", mode="single_turn")
analyst_b = LlmAgent(name="b", model="gemini-2.5-flash", instruction="Analyse from the technical angle.", mode="single_turn")
analyst_c = LlmAgent(name="c", model="gemini-2.5-flash", instruction="Analyse from the market angle.", mode="single_turn")

join = JoinNode(name="merge")

@node
def synthesise(node_input: dict) -> str:
    parts = [f"[{k.upper()}]: {v}" for k, v in node_input.items()]
    return "\n".join(parts)

# max_concurrency=2 means at most 2 of a/b/c run simultaneously
# The third starts only when one of the first two finishes
analysis_workflow = Workflow(
    name="multi_analyst",
    edges=[(START, (analyst_a, analyst_b, analyst_c), join, synthesise)],
    max_concurrency=2,
)

async def main():
    app = App(name="analyst_app", root_agent=analysis_workflow)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="analyst_app", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="Analyse the prospects of electric vehicles in 2026."
        )]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 3 — state injection from `state_schema` into node params

```python
import asyncio
from pydantic import BaseModel
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START
from google.genai import types

class SearchState(BaseModel):
    query: str = ""
    max_results: int = 5
    results: list[str] = []

@node
def set_params(node_input: str, ctx) -> str:
    ctx.state["query"] = node_input
    ctx.state["max_results"] = 3
    return node_input

@node
async def search(node_input: str, ctx, query: str = "", max_results: int = 5) -> list[str]:
    # `query` and `max_results` are injected by the framework from SearchState
    # because their names match schema fields
    results = [f"Result {i+1} for '{query}'" for i in range(max_results)]
    ctx.state["results"] = results
    return results

@node
def format_output(node_input: list, max_results: int = 5) -> str:
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(node_input[:max_results]))

pipeline = Workflow(
    name="search_pipeline",
    state_schema=SearchState,
    edges=[(START, set_params, search, format_output)],
)

async def main():
    app = App(name="search_app", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="search_app", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="quantum computing")]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

---

## 6 · `OpenAPIToolset` — advanced usage

**Source:** `google/adk/tools/openapi_tool/openapi_spec_parser/openapi_toolset.py`

Three constructor parameters added or clarified in 2.3.0:

- **`preserve_property_names`** — default `False`. When `True`, property
  names in request bodies are sent as-is instead of being snake_case-ified.
  Use when the API expects `camelCase` or `PascalCase` field names.
- **`httpx_client_factory`** — zero-argument callable returning a fresh
  `httpx.AsyncClient`. Overrides per-tool default HTTP client construction.
  Enables proxies, request signing, HTTP/2, and custom transports. The
  factory is called for every request (not cached).
- **`ssl_verify`** — `True` (default system CA), `False` (disable; insecure),
  `str` (path to CA bundle or directory), or `ssl.SSLContext` (full control).
  Enterprise environments with TLS-intercepting proxies can pass a custom CA
  bundle path.

### Example 1 — `preserve_property_names` for camelCase APIs

```python
import json
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents import LlmAgent

# An API that expects camelCase in request bodies, e.g. {"firstName": "Alice"}
camel_spec = {
    "openapi": "3.0.0",
    "info": {"title": "User API", "version": "1.0"},
    # servers must be at the spec root — OpenApiSpecParser._collect_operations
    # reads openapi_spec["servers"][0] before iterating paths and never reads
    # operation-level servers.
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/users": {
            "post": {
                "operationId": "createUser",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "firstName": {"type": "string"},
                                    "lastName": {"type": "string"},
                                    "emailAddress": {"type": "string"},
                                },
                                "required": ["firstName", "lastName", "emailAddress"],
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "Created"}},
            }
        }
    },
}

# Without preserve_property_names=True, "firstName" → "first_name" in the request.
# With preserve_property_names=True, the LLM sends "firstName" unchanged.
toolset = OpenAPIToolset(
    spec_dict=camel_spec,
    preserve_property_names=True,
)

agent = LlmAgent(
    name="user_creator",
    model="gemini-2.5-flash",
    instruction="Create user accounts when asked. Use the createUser tool.",
    tools=[toolset],
)
```

### Example 2 — `ssl_verify` with a custom enterprise CA bundle

```python
import ssl
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents import LlmAgent

# Option A: path to a CA bundle file (PEM format)
toolset_ca_path = OpenAPIToolset(
    spec_dict={
        "openapi": "3.0.0",
        "info": {"title": "Internal API", "version": "1"},
        "servers": [{"url": "https://internal.corp.example.com"}],
        "paths": {
            "/data": {
                "get": {
                    "operationId": "getData",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    },
    ssl_verify="/etc/ssl/certs/corporate-ca-bundle.pem",
)

# Option B: custom ssl.SSLContext for full control (e.g. mutual TLS)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.load_verify_locations("/etc/ssl/certs/corporate-ca-bundle.pem")
ctx.load_cert_chain(certfile="/etc/ssl/client.crt", keyfile="/etc/ssl/client.key")

toolset_mtls = OpenAPIToolset(
    spec_dict={
        "openapi": "3.0.0",
        "info": {"title": "Secure API", "version": "1"},
        "servers": [{"url": "https://secure.corp.example.com"}],
        "paths": {
            "/secure": {
                "get": {
                    "operationId": "secureEndpoint",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    },
    ssl_verify=ctx,
)
```

### Example 3 — `httpx_client_factory` with request signing

```python
import asyncio
import httpx
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

# A factory that produces a client pre-configured with a custom auth header.
# The factory is called fresh on every HTTP request, so this is safe even
# when the token rotates mid-session.
def signed_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"X-Custom-Auth": "Bearer my-signed-token"},
        timeout=30.0,
        http2=True,   # Enable HTTP/2 for multiplexed connections
    )

toolset = OpenAPIToolset(
    spec_dict={
        "openapi": "3.0.0",
        "info": {"title": "Signed API", "version": "1"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/items": {
                "get": {
                    "operationId": "listItems",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    },
    httpx_client_factory=signed_client_factory,
)

agent = LlmAgent(
    name="item_browser",
    model="gemini-2.5-flash",
    instruction="List available items using the API.",
    tools=[toolset],
)
```

---

## 7 · `McpToolset` — `header_provider`, `use_mcp_resources`, `progress_callback` factory

**Source:** `google/adk/tools/mcp_tool/mcp_toolset.py`

Three features worth calling out explicitly:

- **`header_provider`** — a `(ReadonlyContext) -> dict[str, str]` callable.
  Called on every MCP request. **Header precedence differs by code path:**
  - **Session creation** (`mcp_toolset.py` → `get_tools` / `get_resources`):
    provider headers are merged first, auth headers second — **auth headers
    win** on duplicate keys.
  - **Individual tool calls** (`mcp_tool.py` → `_run_async_impl`): auth
    headers are merged first, provider headers second — **provider headers
    win** on duplicate keys.
  Use `header_provider` for per-turn tenant routing or correlation IDs. On
  the tool-call path a provider-supplied `Authorization` will overwrite the
  auth-scheme-derived one, so avoid setting that key from both sources.
- **`use_mcp_resources`** — when `True`, ADK appends a `LoadMcpResourceTool`
  to the toolset's tool list. The model can call it to fetch named MCP
  resources (not tools) from the server.
- **`progress_callback`** — accepts a `ProgressFnT(progress, total, message)`
  or a **factory** `(tool_name, callback_context, **kwargs) -> ProgressFnT`.
  The factory form gives per-tool callbacks that can read/write session state
  via the `callback_context`.

### Example 1 — `header_provider` for multi-tenant routing

> **Transport note:** `header_provider` only works with HTTP-based transports
> (`SseConnectionParams` / `StreamableHTTPConnectionParams`). For stdio
> connections `MCPSessionManager._merge_headers` returns `None` and the
> provider's headers are discarded. Use an HTTP MCP server for any scenario
> that requires per-request headers.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import SseConnectionParams

def tenant_headers(ctx) -> dict[str, str]:
    """Inject tenant ID and correlation ID from session state on every call."""
    tenant_id = ctx.state.get("user:tenant_id", "default")
    correlation_id = ctx.state.get("temp:correlation_id", "unknown")
    return {
        "X-Tenant-ID": tenant_id,
        "X-Correlation-ID": correlation_id,
    }

# header_provider requires an HTTP transport — SSE or StreamableHTTP.
# The provider is called on every tool invocation; returned headers are
# merged into the request (provider headers win on the tool-call path).
toolset = McpToolset(
    connection_params=SseConnectionParams(
        url="https://mcp.example.com/sse",
        timeout=10.0,
    ),
    header_provider=tenant_headers,
)

agent = LlmAgent(
    name="tenant_agent",
    model="gemini-2.5-flash",
    instruction="Help the user work with their files. Tenant context is in your headers.",
    tools=[toolset],
)
```

### Example 2 — `use_mcp_resources` to expose MCP resources

```python
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

# The filesystem MCP server exposes directories as resources.
# With use_mcp_resources=True, ADK adds a `load_mcp_resource` tool
# so the agent can fetch those resources directly.
toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/data"],
        ),
        timeout=10.0,
    ),
    use_mcp_resources=True,
    tool_filter=["read_file"],  # only expose the read_file tool; resource tool is always added
)

agent = LlmAgent(
    name="resource_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Use load_mcp_resource to fetch available data resources. "
        "Use read_file to read specific files."
    ),
    tools=[toolset],
)
```

### Example 3 — `progress_callback` factory for per-tool progress logging

```python
import sys
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

def make_progress_callback(tool_name: str, callback_context, **kwargs):
    """Factory: returns a per-tool progress callback that logs to session state."""
    async def on_progress(progress: float, total: float | None, message: str | None):
        pct = int(progress / total * 100) if total else "?"
        print(f"[{tool_name}] {pct}% — {message}", file=sys.stderr)
        # Also persist progress in session state for downstream tools to read
        if callback_context:
            callback_context.state[f"temp:progress_{tool_name}"] = pct
    return on_progress

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/large-data"],
        ),
        timeout=60.0,
    ),
    progress_callback=make_progress_callback,  # factory form
)

agent = LlmAgent(
    name="progress_agent",
    model="gemini-2.5-flash",
    instruction="Process large files and report progress.",
    tools=[toolset],
)
```

---

## 8 · `LongRunningFunctionTool` — fire-and-forget + companion poll

**Source:** `google/adk/tools/long_running_tool.py`

`LongRunningFunctionTool` is a thin subclass of `FunctionTool` that adds
two things (verified `long_running_tool.py`):

1. `is_long_running = True` — signals ADK that this tool returns before the
   work is done.
2. Appends to `_get_declaration()`: `"\n\nNOTE: This is a long-running
   operation. Do not call this tool again if it has already returned some
   intermediate or pending status."` — instructs the model not to re-invoke.

The function itself **must return immediately** with a `{"status":
"pending", ...}` dict. Real work is offloaded via `asyncio.create_task` or
a background queue. A second, regular `FunctionTool` (the *companion poll
tool*) checks progress by reading state the background task writes.

### Example 1 — background task with `asyncio.create_task`

```python
import asyncio
import time
from google.adk.tools import LongRunningFunctionTool, FunctionTool
from google.adk.tools.tool_context import ToolContext

# Shared in-memory store for background job results (use Redis/DB in production)
_jobs: dict[str, dict] = {}

async def _run_export_background(job_id: str, rows: int) -> None:
    """Background task — simulates a slow export."""
    await asyncio.sleep(5)   # simulate 5 s of work
    _jobs[job_id] = {"status": "done", "rows": rows, "file": f"/tmp/{job_id}.csv"}

async def start_export(dataset: str, row_limit: int = 1000, tool_context: ToolContext | None = None) -> dict:
    """Start an async data export job.

    Args:
      dataset: Name of the dataset to export.
      row_limit: Maximum number of rows to export.
    Returns:
      A dict with status='pending' and job_id.
    """
    job_id = f"exp-{dataset}-{int(time.time())}"
    _jobs[job_id] = {"status": "pending", "pct": 0}
    if tool_context:
        tool_context.state[f"export_job:{dataset}"] = job_id
    # Fire and forget — do NOT await
    asyncio.create_task(_run_export_background(job_id, row_limit))
    return {"status": "pending", "job_id": job_id, "eta_seconds": 5}

export_tool = LongRunningFunctionTool(func=start_export)

async def check_export(dataset: str, tool_context: ToolContext | None = None) -> dict:
    """Check the status of a dataset export job.

    Args:
      dataset: The dataset name originally passed to start_export.
    Returns:
      A dict with status ('pending' or 'done') and optionally file and rows.
    """
    if tool_context is None:
        return {"error": "no context"}
    job_id = tool_context.state.get(f"export_job:{dataset}")
    if not job_id:
        return {"error": f"No export job found for dataset '{dataset}'"}
    return _jobs.get(job_id, {"error": "Job not found"})

poll_tool = FunctionTool(func=check_export)
```

### Example 2 — multi-phase job with state progress updates

```python
import asyncio
import uuid
from google.adk.tools import LongRunningFunctionTool, FunctionTool
from google.adk.tools.tool_context import ToolContext

_job_store: dict[str, dict] = {}

async def _process_video(job_id: str, video_url: str) -> None:
    """Simulate multi-phase video processing."""
    for phase, pct in [("downloading", 25), ("transcoding", 60), ("indexing", 90), ("done", 100)]:
        await asyncio.sleep(2)
        _job_store[job_id] = {"status": "pending" if pct < 100 else "done", "phase": phase, "pct": pct}
        if phase == "done":
            _job_store[job_id]["url"] = f"https://cdn.example.com/{job_id}.mp4"

async def transcode_video(video_url: str, quality: str = "720p", tool_context: ToolContext | None = None) -> dict:
    """Start a video transcoding job.

    Args:
      video_url: URL of the source video to transcode.
      quality: Output quality — '480p', '720p', or '1080p'.
    Returns:
      A dict with status='pending' and job_id.
    """
    job_id = f"vid-{uuid.uuid4().hex[:8]}"
    _job_store[job_id] = {"status": "pending", "phase": "queued", "pct": 0}
    if tool_context:
        tool_context.state["video_job_id"] = job_id
    asyncio.create_task(_process_video(job_id, video_url))
    return {"status": "pending", "job_id": job_id, "quality": quality, "eta_seconds": 8}

async def get_video_status(tool_context: ToolContext | None = None) -> dict:
    """Check the status of the current video transcoding job.

    Returns:
      A dict with status, phase, pct, and optionally url.
    """
    if tool_context is None:
        return {"error": "no context"}
    job_id = tool_context.state.get("video_job_id")
    if not job_id:
        return {"error": "No active video job"}
    return _job_store.get(job_id, {"error": "Job not found"})

video_tool = LongRunningFunctionTool(func=transcode_video)
status_tool = FunctionTool(func=get_video_status)
```

### Example 3 — using `is_long_running` flag in a custom runner check

```python
from google.adk.tools import LongRunningFunctionTool, FunctionTool

async def submit_batch_job(config_json: str, tool_context=None) -> dict:
    """Submit a batch processing job to the cluster.

    Args:
      config_json: JSON string with job configuration.
    Returns:
      A dict with status='pending' and job_id.
    """
    return {"status": "pending", "job_id": "batch-001", "queue_position": 3}

batch_tool = LongRunningFunctionTool(func=submit_batch_job)

# Inspect the flag programmatically (e.g. in a custom runner or test)
print(batch_tool.is_long_running)   # True

# Compare with a regular FunctionTool
def regular_func() -> str:
    """A quick tool."""
    return "done"

regular = FunctionTool(func=regular_func)
print(regular.is_long_running)      # False (FunctionTool.is_long_running defaults to False)
```

---

## 9 · `LoopAgent` → `Workflow` migration

**Source:** `google/adk/agents/loop_agent.py`

`LoopAgent` is **deprecated** in 2.x and will be removed. Key internals
that inform the migration:

- The loop terminates when **any event** has `event.actions.escalate =
  True` (checked per-event inside the inner `async for`).
- `max_iterations` counts full passes through *all* sub-agents. After each
  complete pass, `ctx.reset_sub_agent_states(self.name)` clears sub-agent
  state so they restart fresh on the next iteration.
- `_get_start_state` (source-verified `loop_agent.py:155-179`) resumes a
  paused loop at the correct sub-agent by matching on
  `LoopAgentState.current_sub_agent`; a missing sub-agent name falls back
  to index 0 (start of the loop).

**Migration rule:** Replace the loop with a `Workflow` routing map where
the *critic* node routes either back to itself (continue) or to a terminal
node (done). Replace `actions.escalate=True` with `ctx.route = "done"`.
Replace `max_iterations` with a counter in `ctx.state`.

### Example 1 — `LoopAgent` original pattern

```python
# DEPRECATED — for migration reference only
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from google.adk.agents import LoopAgent, LlmAgent

critic = LlmAgent(
    name="critic",
    model="gemini-2.5-flash",
    instruction=(
        "Read state['draft']. If it is longer than 100 words, "
        "shorten it to 100 words and store back to state['draft']. "
        "If it is already under 100 words, set actions.escalate = True."
    ),
    output_key="draft",
)

loop = LoopAgent(name="refine", sub_agents=[critic], max_iterations=5)
```

### Example 2 — equivalent `Workflow` replacement

```python
from google.adk.workflow import Workflow, node, START, DEFAULT_ROUTE

@node(rerun_on_resume=True)
async def shorten_draft(node_input: str, ctx) -> str:
    """Shorten the draft iteratively until it is under 100 words."""
    # Track iteration count in state for max_iterations equivalent
    count = ctx.state.get("shorten_iterations", 0) + 1
    ctx.state["shorten_iterations"] = count

    words = node_input.split()
    if len(words) <= 100 or count >= 5:
        ctx.route = "done"
        return " ".join(words[:100])
    else:
        ctx.route = "continue"
        return " ".join(words[:100])  # trim and loop

@node
def publish(node_input: str) -> str:
    return f"[PUBLISHED] {node_input}"

refine_workflow = Workflow(
    name="refine",
    edges=[
        (START, shorten_draft, {
            "continue": shorten_draft,   # loop back
            "done": publish,             # terminal
        }),
    ],
)
```

### Example 3 — multi-agent loop with escalation

```python
# BEFORE (deprecated):
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from google.adk.agents import LoopAgent, LlmAgent

drafter = LlmAgent(name="drafter", model="gemini-2.5-flash",
                   instruction="Write a paragraph on the topic in state['topic'].", output_key="draft")
reviewer = LlmAgent(name="reviewer", model="gemini-2.5-flash",
                    instruction="Read state['draft']. If it is good enough, escalate. Otherwise improve it.", output_key="draft")

loop = LoopAgent(name="draft_loop", sub_agents=[drafter, reviewer], max_iterations=3)

# AFTER (Workflow replacement):
from google.adk.workflow import Workflow, node, START, DEFAULT_ROUTE

draft_agent = LlmAgent(name="drafter2", model="gemini-2.5-flash",
                       instruction="Write a paragraph on the topic given.", mode="single_turn", output_key="draft")

@node(rerun_on_resume=True)
async def review(node_input: str, ctx) -> str:
    count = ctx.state.get("review_count", 0) + 1
    ctx.state["review_count"] = count
    draft = ctx.state.get("draft", node_input)
    # Simple quality heuristic: good if > 50 words or 3 iterations done
    if len(draft.split()) > 50 or count >= 3:
        ctx.route = "done"
    else:
        ctx.route = "revise"
    return draft

@node
def final_output(node_input: str) -> str:
    return node_input

draft_workflow = Workflow(
    name="draft_loop2",
    edges=[
        (START, draft_agent, review, {
            "revise": draft_agent,   # loop back to drafter
            "done": final_output,
        }),
    ],
)
```

---

## 10 · `ParallelAgent` → `Workflow` fan-out migration

**Source:** `google/adk/agents/parallel_agent.py`

`ParallelAgent` is **deprecated** in 2.x. Key source-verified internals
that affect the migration:

- `_create_branch_ctx_for_sub_agent` gives each sub-agent an isolated
  `invocation_context` by appending `"{parent}.{sub}"` to the `branch`
  field. This isolates **event history** (each branch sees its own
  conversation events) but does **not** namespace `state_delta` — state
  writes from parallel branches still merge into the same session state, so
  branches that write the same state key will overwrite each other. The
  `Workflow` fan-out exposes the same behaviour via `Trigger.use_sub_branch`;
  use unique output keys per branch or collect results in a `JoinNode`.
- `_merge_agent_run` uses `asyncio.TaskGroup` (Python 3.11+) or manual
  task management (3.10) with a queue + resume signal to interleave events
  from concurrent sub-agents without blocking any of them.
- On resume (`is_resumable`), sub-agents that already have
  `ctx.end_of_agents[sub_agent.name]` set are skipped.
- `run_live` is **not** implemented (`NotImplementedError` raised).

**Migration rule:** Replace with a `Workflow` fan-out using a nested tuple
in `edges` and a `JoinNode`. Sub-agent isolation is handled automatically.
Results arrive in the `JoinNode` as a `dict[str, output]`.

### Example 1 — `ParallelAgent` original pattern

```python
# DEPRECATED — for migration reference only
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from google.adk.agents import ParallelAgent, LlmAgent

approach_a = LlmAgent(name="approach_a", model="gemini-2.5-flash", instruction="Solve using approach A.")
approach_b = LlmAgent(name="approach_b", model="gemini-2.5-flash", instruction="Solve using approach B.")
approach_c = LlmAgent(name="approach_c", model="gemini-2.5-flash", instruction="Solve using approach C.")

fan_out = ParallelAgent(name="multi_solve", sub_agents=[approach_a, approach_b, approach_c])
```

### Example 2 — equivalent `Workflow` fan-out + `JoinNode`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, JoinNode, node, START
from google.genai import types

a = LlmAgent(name="approach_a", model="gemini-2.5-flash",
             instruction="Solve the problem using method A. Be concise.", mode="single_turn")
b = LlmAgent(name="approach_b", model="gemini-2.5-flash",
             instruction="Solve the problem using method B. Be concise.", mode="single_turn")
c = LlmAgent(name="approach_c", model="gemini-2.5-flash",
             instruction="Solve the problem using method C. Be concise.", mode="single_turn")

join = JoinNode(name="collect_results")

@node
def pick_best(node_input: dict) -> str:
    # node_input is {"approach_a": "...", "approach_b": "...", "approach_c": "..."}
    responses = "\n\n".join(
        f"=== {k.upper()} ===\n{v}" for k, v in node_input.items()
    )
    return f"Three approaches considered:\n\n{responses}"

fan_out_workflow = Workflow(
    name="multi_solve",
    edges=[(START, (a, b, c), join, pick_best)],
)

async def main():
    app = App(name="solver_app", root_agent=fan_out_workflow)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="solver_app", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="What is the best data structure for a priority queue?"
        )]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 3 — fan-out with parallel branches (researcher + critic)

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, JoinNode, node, START

# Each branch gets its own event history (use_sub_branch=True is implicit for
# nested-tuple fan-out), but state writes are NOT isolated — use unique
# output_key values per branch to avoid collisions.

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic and write a paragraph.",
    mode="single_turn",
    output_key="research_result",  # writes to shared session state (not isolated)
)
critic = LlmAgent(
    name="critic",
    model="gemini-2.5-flash",
    instruction="Critique the research paragraph for accuracy.",
    mode="single_turn",
    output_key="critic_result",
)

join = JoinNode(name="combine")

@node
def merge_outputs(node_input: dict) -> str:
    research = node_input.get("researcher", "")
    critique = node_input.get("critic", "")
    return f"Research:\n{research}\n\nCritique:\n{critique}"

review_workflow = Workflow(
    name="parallel_review",
    edges=[(START, (researcher, critic), join, merge_outputs)],
    # No max_concurrency needed here — 2 branches is fine to run simultaneously
)
```

---

## Summary table

| Class | Module | Key new patterns |
|---|---|---|
| `App` | `google.adk.apps` | `EventsCompactionConfig`, `ResumabilityConfig`, `plugins=` |
| `FunctionTool` | `google.adk.tools` | Pydantic coercion, `require_confirmation` callable, mandatory-args guard |
| `Context` | `google.adk.agents.context` | Unified write API: state prefixes, artifacts, route, run_node |
| `InMemoryArtifactService` | `google.adk.artifacts` | Versioning, `list_versions`, `get_artifact_version`, user namespace |
| `Workflow` | `google.adk.workflow` | `state_schema`, `max_concurrency`, state injection |
| `OpenAPIToolset` | `google.adk.tools.openapi_tool` | `preserve_property_names`, `httpx_client_factory`, `ssl_verify` |
| `McpToolset` | `google.adk.tools.mcp_tool` | `header_provider`, `use_mcp_resources`, `progress_callback` factory |
| `LongRunningFunctionTool` | `google.adk.tools` | `asyncio.create_task` background, companion poll, `is_long_running` flag |
| `LoopAgent` | `google.adk.agents` | **Deprecated** — migrate to `Workflow` routing map |
| `ParallelAgent` | `google.adk.agents` | **Deprecated** — migrate to `Workflow` fan-out + `JoinNode` |

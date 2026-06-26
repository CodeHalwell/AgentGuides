---
title: "Class deep dives — volume 28 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: Runner (run_async/run_live/state_delta/auto_create_session; LlmAgent chat-mode guard; _find_agent_to_run resume logic), SlackRunner (Slack bolt integration; app_mention+DM/thread handling; channel-thread_ts session ID; Thinking... placeholder; AsyncSocketModeHandler), E2BEnvironment (remote E2B sandbox; @experimental; TTL keepalive; auto-reconnect on expiry; PurePosixPath traversal guard), InMemoryMemoryService (keyword matching NOT semantic; threading.Lock; _extract_words_lower regex; dedup by event.id; _user_key convention; prototyping only), _OutputSchemaRequestProcessor+helpers (output_schema+tools incompatibility bridge; can_use_output_schema_with_tools guard; mode='task' bypass; SetModelResponseTool injection; FC response extraction), _NlPlanningRequestProcessor+_NlPlanningResponse (NL planning flow; BuiltInPlanner.apply_thinking_config vs PlanReActPlanner.build_planning_instruction; thought-part scrubbing; module-level singleton), get_fast_api_app (production FastAPI server factory; service URI resolution; A2A; watchdog reload; CORS; cloud OTEL; trigger endpoints; express_mode), MockModel+AgentTestRunner (pytest-based deterministic testing; MockModel extends BaseLlm; EXCLUDED_EVENT_FIELDS; get_test_files directory scan), trace_call_llm+trace_tool_call (OTel span attribute writing; GCP_MCP_SERVER_DESTINATION_ID AppHub mapping; ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS toggle; gen_ai.* semantic conventions; error_type precedence), SerializedBaseModel (camelCase JSON bridge; by_alias=True model_dump_json default; populate_by_name=True; use_attribute_docstrings; underpins Event/Session web-server serialisation)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 28"
  order: 97
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source at
`/usr/local/lib/python3.11/dist-packages/google/adk/` on
**google-adk == 2.3.0**. No documentation or blog posts were used as primary
sources.
</Aside>

## 1 · `Runner` — the top-level execution harness

**Module:** `google.adk.runners`

`Runner` is the single public entry-point for all agent execution: turn-by-turn chat (`run_async`), live bidirectional audio/video (`run_live`), and the synchronous wrapper (`run`). It wires together session, artifact, memory, credential services, the plugin manager, and the root agent.

### Key constructor facts

```python
Runner(
    *,
    app: Optional[App] = None,          # preferred: all config from App
    app_name: Optional[str] = None,
    agent: Optional[BaseAgent] = None,  # wraps agent → App internally
    node: Any = None,                   # wraps node → App internally
    plugins: Optional[List[BasePlugin]] = None,  # deprecated; use App
    artifact_service: Optional[BaseArtifactService] = None,
    session_service: BaseSessionService,          # required
    memory_service: Optional[BaseMemoryService] = None,
    credential_service: Optional[BaseCredentialService] = None,
    plugin_close_timeout: float = 5.0,
    auto_create_session: bool = False,  # True → create if missing
)
```

`run_async` signature:
```python
async def run_async(
    self, *, user_id, session_id,
    invocation_id=None,    # resume interrupted invocation
    new_message=None,      # types.Content
    state_delta=None,      # dict[str, Any]
    run_config=None,       # RunConfig
    yield_user_message=False,
) -> AsyncGenerator[Event, None]
```

**Root-agent mode guard** — when `agent.mode is None`, `run_async` sets it to `"chat"` (LlmAgent as a root agent must be in chat mode). A mode of `"task"` on the root raises `ValueError`.

**`_find_agent_to_run`** — walks the session events backwards to locate the last incomplete sub-agent invocation for resume; bypassed when a task sub-agent exists (the coordinator always receives the message instead).

**`run_live`** distinguishes events that are *yielded but not saved* (inline audio blobs) from events that are *saved and yielded* (file-data references, usage metadata, transcription, function calls). Inline audio is transient so you can stream bytes directly without polluting the session.

### Example 1 — minimal turn-by-turn runner

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(name="assistant", model="gemini-2.5-flash",
                 instruction="You are a helpful assistant.")
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="demo", session_service=session_service)

async def chat(session_id: str, text: str) -> str:
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    response_parts = []
    async for event in runner.run_async(
        user_id="u1", session_id=session_id, new_message=msg
    ):
        if event.is_final_response() and event.content:
            response_parts.extend(
                p.text for p in event.content.parts if p.text
            )
    return "".join(response_parts)

async def main():
    session = await session_service.create_session(
        app_name="demo", user_id="u1"
    )
    print(await chat(session.id, "Hello!"))

asyncio.run(main())
```

### Example 2 — auto_create_session and state_delta

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(
    name="greeter", model="gemini-2.5-flash",
    instruction="Greet the user. Their name is in state['user_name'].",
)
session_service = InMemorySessionService()

# auto_create_session=True avoids a separate create_session() call
runner = Runner(
    agent=agent,
    app_name="greet_app",
    session_service=session_service,
    auto_create_session=True,
)

async def main():
    msg = types.Content(role="user", parts=[types.Part(text="Hi there")])
    async for event in runner.run_async(
        user_id="u1",
        session_id="session-42",
        new_message=msg,
        state_delta={"user_name": "Alice"},   # pre-seeds session state
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 3 — live bidirectional audio skeleton

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(name="voice", model="gemini-2.5-flash-live-preview",
                 instruction="You are a live voice assistant.")
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="voice_app",
                session_service=session_service)

async def main():
    session = await session_service.create_session(
        app_name="voice_app", user_id="u1"
    )
    live_queue = LiveRequestQueue()

    async def producer():
        # Send a text turn to the live session
        live_queue.send_content(
            types.Content(role="user",
                          parts=[types.Part(text="What is the capital of France?")])
        )
        await asyncio.sleep(3)
        live_queue.close()

    async def consumer():
        cfg = RunConfig(streaming_mode=StreamingMode.BIDI)
        async for event in runner.run_live(
            user_id="u1", session_id=session.id,
            live_request_queue=live_queue, run_config=cfg,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print("Model:", part.text)

    await asyncio.gather(producer(), consumer())

asyncio.run(main())
```

---

## 2 · `SlackRunner` — deploy any ADK agent to Slack

**Module:** `google.adk.integrations.slack.slack_runner`  
**Install:** `pip install "google-adk[slack]"`

`SlackRunner` wraps any `Runner` with a [Slack Bolt](https://slack.dev/bolt-python/) `AsyncApp` to handle `app_mention` and direct-message events. It uses **Socket Mode** for outbound connections — no public webhook URL required.

### Signature

```python
class SlackRunner:
    def __init__(self, runner: Runner, slack_app: AsyncApp): ...
    async def start(self, app_token: str): ...
    # internal helpers set up by _setup_handlers():
    #   handle_app_mentions  → listens to @mention events
    #   handle_message_events → listens to message events (IM or threaded)
```

### Session ID convention

```
thread_ts = event.get("thread_ts") or event.get("ts")
session_id = f"{channel_id}-{thread_ts}"
```

Every Slack message event carries a `ts` field, so `thread_ts` is always set and the session ID is always `f"{channel_id}-{thread_ts}"`. Threaded replies use the parent message's `thread_ts`; top-level messages use their own `ts`. This gives each thread (or unthreaded message) its own isolated ADK session.

### "Thinking..." placeholder pattern

The runner posts a `_Thinking..._` message immediately, then updates it in-place with the first text part the agent returns — giving users instant feedback without sending a blank message.

### Example 1 — basic setup

```python
import asyncio
import os
from slack_bolt.app.async_app import AsyncApp
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.integrations.slack import SlackRunner

agent = LlmAgent(
    name="slack_bot",
    model="gemini-2.5-flash",
    instruction="You are a helpful Slack assistant.",
)
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="slack_demo",
                session_service=session_service, auto_create_session=True)

slack_app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
slack_runner = SlackRunner(runner=runner, slack_app=slack_app)

async def main():
    await slack_runner.start(app_token=os.environ["SLACK_APP_TOKEN"])

asyncio.run(main())
```

### Example 2 — inspecting the session ID convention

```python
# SlackRunner maps Slack's channel+thread_ts into a deterministic session ID.
# You can replicate the logic to look up sessions directly:
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

channel_id = "C12345678"
thread_ts = "1718000000.001234"

session_id = f"{channel_id}-{thread_ts}"   # threaded conversation
# For a DM with no thread: session_id = channel_id

session = await session_service.get_session(
    app_name="slack_demo", user_id="U999", session_id=session_id
)
print(session)  # None if not yet created
```

### Example 3 — adding a before_message hook via subclass

```python
import logging
from slack_bolt.app.async_app import AsyncApp
from google.adk.integrations.slack import SlackRunner
from google.adk.runners import Runner

logger = logging.getLogger(__name__)

class AuditingSlackRunner(SlackRunner):
    async def _handle_message(self, event, say):
        user_id = event.get("user")
        text = event.get("text", "")
        logger.info("Slack user %s sent: %s", user_id, text)
        # Delegate to parent implementation
        await super()._handle_message(event, say)

# Usage:
# runner = Runner(agent=my_agent, app_name="app", session_service=svc, auto_create_session=True)
# slack_app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
# auditing_runner = AuditingSlackRunner(runner=runner, slack_app=slack_app)
# await auditing_runner.start(app_token=os.environ["SLACK_APP_TOKEN"])
```

---

## 3 · `E2BEnvironment` — remote E2B cloud sandbox

**Module:** `google.adk.integrations.e2b._e2b_environment`  
**Install:** `pip install "google-adk[e2b]"`  
**Status:** `@experimental`

`E2BEnvironment` implements `BaseEnvironment` using [E2B](https://e2b.dev/) cloud sandboxes. It provides file CRUD and shell execution inside an isolated remote workspace, with automatic TTL keepalive and transparent reconnection after expiry.

### Signature

```python
@experimental
class E2BEnvironment(BaseEnvironment):
    def __init__(
        self, *,
        image: str = "base",      # E2B template name or ID
        timeout: int = 300,        # sandbox TTL in seconds; reset on every op
        api_key: Optional[str] = None,  # falls back to E2B_API_KEY env var
        env_vars: Optional[dict[str, str]] = None,
    ): ...
```

### Key behaviours verified from source

- **TTL keepalive**: every `execute()`, `read_file()`, `write_file()` call that finds a still-running sandbox invokes `sandbox.set_timeout(self._timeout)` to extend its life.
- **Auto-reconnect**: if `is_running()` returns `False`, a fresh sandbox is created transparently — workspace state (installed packages, files) is lost.
- **Path resolution**: `_resolve_path()` uses `PurePosixPath(_SANDBOX_HOME) / path` — relative paths are anchored under `/home/user`, but absolute paths pass through unchanged (e.g. `"/etc/passwd"` resolves to `/etc/passwd`). Do not rely on this as a traversal guard for user-supplied absolute paths.
- **Working directory**: always `Path("/home/user")`.

### Example 1 — basic file write and execute

```python
import asyncio
import os
from google.adk.integrations.e2b import E2BEnvironment

async def main():
    env = E2BEnvironment(api_key=os.environ["E2B_API_KEY"])
    await env.initialize()
    try:
        await env.write_file("hello.py", b"print('Hello from E2B!')\n")
        result = await env.execute("python3 hello.py")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("exit_code:", result.exit_code)
    finally:
        await env.close()

asyncio.run(main())
```

### Example 2 — install packages and run data analysis

```python
import asyncio
import os
from google.adk.integrations.e2b import E2BEnvironment

SCRIPT = """
import json, statistics
data = [1, 4, 9, 16, 25]
print(json.dumps({"mean": statistics.mean(data), "stdev": statistics.stdev(data)}))
"""

async def main():
    env = E2BEnvironment(
        api_key=os.environ["E2B_API_KEY"],
        timeout=600,  # 10-minute TTL
        env_vars={"PYTHONUNBUFFERED": "1"},
    )
    await env.initialize()
    try:
        # Install nothing — stdlib only in this example
        await env.write_file("analyze.py", SCRIPT.encode())
        result = await env.execute("python3 analyze.py")
        import json; print(json.loads(result.stdout))
    finally:
        await env.close()

asyncio.run(main())
```

### Example 3 — using E2BEnvironment with EnvironmentToolset

```python
import asyncio, os
from google.adk.integrations.e2b import E2BEnvironment
from google.adk.tools.environment import EnvironmentToolset
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def main():
    env = E2BEnvironment(api_key=os.environ["E2B_API_KEY"])
    await env.initialize()

    toolset = EnvironmentToolset(environment=env)
    agent = LlmAgent(
        name="coder",
        model="gemini-2.5-flash",
        instruction="You are a coding assistant. Use execute to run Python scripts.",
        tools=[toolset],
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="coder_app",
                    session_service=session_service, auto_create_session=True)

    session = await session_service.create_session(app_name="coder_app", user_id="u1")
    msg = types.Content(role="user",
                        parts=[types.Part(text="Write and run a Python script that prints the first 10 Fibonacci numbers.")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

    await env.close()

asyncio.run(main())
```

---

## 4 · `InMemoryMemoryService` — keyword matching memory for prototyping

**Module:** `google.adk.memory.in_memory_memory_service`

`InMemoryMemoryService` implements `BaseMemoryService` using **keyword matching** (not semantic vector search). It is thread-safe via `threading.Lock` and suitable for development/testing only.

### Key design facts

- **Storage**: `dict[str, dict[str, list[Event]]]` — `"{app_name}/{user_id}"` → `session_id` → event list
- **Search**: `_extract_words_lower()` splits text on `\w+`, lower-cases, and intersects against query words. *Any* query word matching any event word is a hit (OR logic).
- **Dedup**: `add_events_to_memory()` tracks `event.id` in a `set` to avoid adding the same event twice.
- **Unknown session**: events added without a `session_id` are stored under the sentinel key `"__unknown_session_id__"`.

### Example 1 — basic add and search

```python
import asyncio
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.events.event import Event

async def main():
    memory = InMemoryMemoryService()
    session_service = InMemorySessionService()

    session = await session_service.create_session(
        app_name="memo_app", user_id="u1"
    )

    # Populate session with a couple of events
    event = Event(
        author="model",
        invocation_id="inv-1",
        content=types.Content(
            role="model",
            parts=[types.Part(text="The Eiffel Tower is located in Paris, France.")]
        )
    )
    session.events.append(event)

    # Index the whole session
    await memory.add_session_to_memory(session)

    # Keyword search — OR logic: any query word match fires
    result = await memory.search_memory(
        app_name="memo_app", user_id="u1", query="Paris tower"
    )
    for entry in result.memories:
        print(entry.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — add_events_to_memory with dedup

```python
import asyncio
from google.adk.memory import InMemoryMemoryService
from google.adk.events.event import Event
from google.genai import types

async def main():
    memory = InMemoryMemoryService()

    def make_event(eid: str, text: str) -> Event:
        e = Event(
            author="model",
            invocation_id="inv-1",
            content=types.Content(role="model", parts=[types.Part(text=text)])
        )
        e.id = eid  # force a stable ID for dedup testing
        return e

    e1 = make_event("e1", "Python was created by Guido van Rossum.")
    e2 = make_event("e2", "Python 3.11 added exception notes.")

    await memory.add_events_to_memory(
        app_name="app", user_id="u1", events=[e1, e2], session_id="s1"
    )
    # Adding the same events again is a no-op thanks to ID dedup
    await memory.add_events_to_memory(
        app_name="app", user_id="u1", events=[e1, e2], session_id="s1"
    )

    result = await memory.search_memory(app_name="app", user_id="u1", query="Python")
    print(f"Found {len(result.memories)} unique memories (expected 2)")

asyncio.run(main())
```

### Example 3 — multi-user isolation

```python
import asyncio
from google.adk.memory import InMemoryMemoryService
from google.adk.events.event import Event
from google.genai import types

async def main():
    memory = InMemoryMemoryService()

    def add(user: str, text: str):
        e = Event(
            author="model", invocation_id="inv",
            content=types.Content(role="model", parts=[types.Part(text=text)])
        )
        return memory.add_events_to_memory(
            app_name="shared_app", user_id=user, events=[e]
        )

    await add("alice", "Alice's secret project code: ALPHA-9.")
    await add("bob", "Bob's data: sales figures for Q4.")

    alice_result = await memory.search_memory(
        app_name="shared_app", user_id="alice", query="secret project"
    )
    bob_result = await memory.search_memory(
        app_name="shared_app", user_id="bob", query="secret project"
    )

    print("Alice sees:", len(alice_result.memories))  # 1
    print("Bob sees:", len(bob_result.memories))       # 0 — isolation

asyncio.run(main())
```

---

## 5 · `_OutputSchemaRequestProcessor` — output_schema + tools bridge

**Module:** `google.adk.flows.llm_flows._output_schema_processor`

When an `LlmAgent` has both `output_schema` *and* `tools`, and the model cannot natively use both simultaneously (`can_use_output_schema_with_tools()` returns `False`), this processor bridges the gap by injecting a `SetModelResponseTool` into the request and adding an instruction that forces the model to call it for its final answer.

### Activation guard

```python
if (
    not agent.output_schema
    or not agent.tools
    or can_use_output_schema_with_tools(agent.canonical_model)
    or getattr(agent, 'mode', None) == 'task'
):
    return   # no-op
```

`mode='task'` bypasses the processor because task agents use `FinishTaskTool` for typed output instead.

### Two extraction helpers

- **`create_final_model_response_event`** — builds a synthetic model-content `Event` from the JSON the `set_model_response` tool returned, making downstream handlers see it as a normal text response.
- **`get_structured_model_response`** — scans `function_response_event.get_function_responses()` for `name == "set_model_response"`, unwraps `{"result": ...}` if present, and JSON-serialises the payload.

### Example 1 — observe the injected instruction in the request

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.flows.llm_flows._output_schema_processor import _OutputSchemaRequestProcessor

class Summary(BaseModel):
    title: str
    word_count: int

def count_words(text: str) -> int:
    return len(text.split())

agent = LlmAgent(
    name="summarizer",
    model="gemini-2.0-flash",   # model that cannot use output_schema with tools
    output_schema=Summary,
    tools=[FunctionTool(func=count_words)],
)

# _OutputSchemaRequestProcessor is a module-level singleton
proc = _OutputSchemaRequestProcessor()
print(type(proc))  # <class '_OutputSchemaRequestProcessor'>
# In real usage the runner calls proc.run_async(invocation_context, llm_request)
# which appends SetModelResponseTool to llm_request.tools and an IMPORTANT: instruction.
```

### Example 2 — end-to-end with output_schema + tools (no manual setup needed)

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

def fetch_movie_rating(movie_name: str) -> float:
    """Returns a mock rating for a movie."""
    ratings = {"Inception": 8.8, "Interstellar": 8.6}
    return ratings.get(movie_name, 7.5)

agent = LlmAgent(
    name="reviewer",
    model="gemini-2.0-flash",
    instruction="Review movies using the fetch_movie_rating tool.",
    output_schema=MovieReview,
    tools=[fetch_movie_rating],
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="review_app",
                session_service=session_service)

async def main():
    session = await session_service.create_session(app_name="review_app", user_id="u1")
    msg = types.Content(role="user", parts=[types.Part(text="Review Inception for me.")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)  # JSON conforming to MovieReview

asyncio.run(main())
```

### Example 3 — extracting the structured response from a function response event

```python
import json
from google.adk.flows.llm_flows._output_schema_processor import get_structured_model_response
from google.adk.events.event import Event
from google.genai import types

# Simulate a function response event that set_model_response produced
fr_event = Event(
    author="tool",
    invocation_id="inv-1",
    content=types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id="fc-1",
                    name="set_model_response",
                    response={"result": {"title": "Inception", "rating": 8.8, "summary": "A mind-bending heist."}},
                )
            )
        ],
    ),
)

json_str = get_structured_model_response(fr_event)
if json_str:
    data = json.loads(json_str)
    print(data["title"])    # Inception
    print(data["rating"])   # 8.8
```

---

## 6 · `_NlPlanningRequestProcessor` + `_NlPlanningResponse` — NL planning flow processors

**Module:** `google.adk.flows.llm_flows._nl_planning`

These two module-level `BaseLlmRequestProcessor` / `BaseLlmResponseProcessor` singletons are inserted into the LLM processing pipeline whenever an `LlmAgent` has a `planner` configured. They dispatch to **either** a `BuiltInPlanner` or a `PlanReActPlanner` depending on the planner type.

### Request processor behaviour

| Planner type | Action |
|---|---|
| `BuiltInPlanner` | Calls `planner.apply_thinking_config(llm_request)` to set up extended thinking |
| `PlanReActPlanner` | Calls `planner.build_planning_instruction(context, llm_request)` and appends it; then strips thought parts from the history |

**Thought stripping** (`_remove_thought_from_request`) — sets `part.thought = None` on every part in `llm_request.contents`. This prevents prior reasoning steps from accumulating in the context and confusing the model.

### Response processor behaviour

- For `BuiltInPlanner`: extracts thought/plan parts from the response and stores them as a planning artefact on `InvocationContext`.
- For `PlanReActPlanner`: validates that the model followed the plan format (plan prefix required).

### Example 1 — agent with BuiltInPlanner (thinking-based planning)

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

agent = LlmAgent(
    name="planner_agent",
    model="gemini-2.5-pro",   # thinking-capable model
    instruction="Plan and execute multi-step research tasks.",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
        thinking_budget=8192
    )),
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="plan_app",
                session_service=session_service)

async def main():
    session = await session_service.create_session(app_name="plan_app", user_id="u1")
    msg = types.Content(role="user",
                        parts=[types.Part(text="Compare the GDP of France and Germany.")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — PlanReActPlanner with step-by-step reasoning

```python
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

def web_search(query: str) -> str:
    """Simulated web search."""
    return f"Results for '{query}': [mock result 1, mock result 2]"

agent = LlmAgent(
    name="react_agent",
    model="gemini-2.5-flash",
    instruction="You are a research assistant that plans before acting.",
    planner=PlanReActPlanner(),
    tools=[web_search],
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="react_app",
                session_service=session_service)

async def main():
    session = await session_service.create_session(app_name="react_app", user_id="u1")
    msg = types.Content(role="user",
                        parts=[types.Part(text="What are the main causes of inflation?")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 3 — verifying thought-stripping with a custom planner

```python
from google.adk.models.llm_request import LlmRequest
from google.genai import types

# _remove_thought_from_request sets part.thought = None on all contents.
# You can replicate this outside the framework if needed:
def strip_thoughts(llm_request: LlmRequest) -> None:
    if not llm_request.contents:
        return
    for content in llm_request.contents:
        if content.parts:
            for part in content.parts:
                part.thought = None

# Demonstrate:
req = LlmRequest(model="gemini-2.5-flash")
req.contents = [
    types.Content(role="model", parts=[
        types.Part(text="I will search for this.", thought=True),
        types.Part(text="Here is my answer."),
    ])
]
strip_thoughts(req)
for part in req.contents[0].parts:
    print(f"text={part.text!r} thought={part.thought}")
# text='I will search for this.' thought=None
# text='Here is my answer.' thought=None
```

---

## 7 · `get_fast_api_app` — the production FastAPI server factory

**Module:** `google.adk.cli.fast_api`

`get_fast_api_app` constructs and returns a fully-configured `FastAPI` application for serving ADK agents over HTTP/SSE. It is the function behind `adk api_server` and `adk web`.

### Abridged signature

```python
def get_fast_api_app(
    *,
    agents_dir: str,
    agent_loader: Optional[BaseAgentLoader] = None,
    session_service_uri: Optional[str] = None,   # 'memory://' | 'sqlite://' | 'postgresql://' | 'agentengine://'
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    eval_storage_uri: Optional[str] = None,
    allow_origins: Optional[list[str]] = None,   # CORS
    web: bool,                                    # serve browser UI
    a2a: bool = False,                            # A2A protocol endpoints
    task_store_uri: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    url_prefix: Optional[str] = None,
    trace_to_cloud: bool = False,
    otel_to_cloud: bool = False,
    reload_agents: bool = False,                  # watchdog hot-reload
    lifespan: Optional[Lifespan[FastAPI]] = None,
    extra_plugins: Optional[list[str]] = None,
    auto_create_session: bool = False,
    trigger_sources: Optional[list[Literal["pubsub", "eventarc"]]] = None,
    default_llm_model: Optional[str] = None,
    express_mode: bool = False,
) -> FastAPI: ...
```

### Service URI resolution

| URI scheme | Backend |
|---|---|
| `memory://` | In-memory (explicitly ephemeral) |
| `None` (default) | Local SQLite under `.adk/` when disk is writable; falls back to in-memory on Cloud Run / read-only FS |
| `sqlite:///path.db` | SQLite at an explicit path via `aiosqlite` |
| `postgresql://...` | PostgreSQL via SQLAlchemy async |
| `agentengine://resource-id` | Vertex AI Agent Engine |
| `gs://bucket/prefix` | GCS artifact service |

> **Note:** Passing `session_service_uri=None` does **not** guarantee an in-memory backend — on a local dev machine with a writable working directory, ADK will persist sessions to `.adk/`. Use `memory://` explicitly if you need an ephemeral dev backend.

### Example 1 — minimal server (for tests or local dev)

```python
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

app = get_fast_api_app(
    agents_dir="./agents",
    session_service_uri="memory://",
    web=False,         # API-only, no browser UI
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### Example 2 — production server with SQLite persistence and CORS

```python
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

app = get_fast_api_app(
    agents_dir="./agents",
    session_service_uri="sqlite:///data/sessions.db",
    artifact_service_uri="file://./artifacts",
    allow_origins=["https://myapp.example.com"],
    web=True,
    trace_to_cloud=True,    # send OTel traces to Google Cloud Trace
    reload_agents=True,     # watchdog: auto-reload on file changes (dev)
    auto_create_session=True,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Example 3 — A2A protocol + trigger sources enabled

```python
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

# A2A enables /a2a/** endpoints; trigger_sources adds /trigger/* endpoints
# for Pub/Sub and Eventarc batch invocations.
app = get_fast_api_app(
    agents_dir="./agents",
    session_service_uri="postgresql+asyncpg://user:pw@db:5432/adk",
    web=True,
    a2a=True,                               # Agent-to-Agent protocol
    trigger_sources=["pubsub", "eventarc"], # batch trigger endpoints
    express_mode=True,                       # Vertex AI Express Mode auth
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## 8 · `MockModel` + `AgentTestRunner` — deterministic pytest testing

**Module:** `google.adk.cli.agent_test_runner`

`agent_test_runner` provides pytest infrastructure for offline, deterministic ADK agent testing. The centrepiece is `MockModel`, a `BaseLlm` subclass that returns pre-scripted `LlmResponse` objects so tests never hit a real LLM.

### Key constants

```python
EXCLUDED_EVENT_FIELDS = {
    "id", "timestamp", "invocation_id", "model_version",
    "finish_reason", "usage_metadata", "avg_logprobs",
    "cache_metadata", "logprobs_result", "citation_metadata",
}
```

These non-deterministic fields are excluded when asserting event equality, making snapshot tests stable across runs.

### `get_test_files`

```python
def get_test_files(target_folder: str | None = None) -> list[pytest.ParameterSet]:
```

Walks `target_folder` (or `$ADK_TEST_FOLDER`) recursively for `tests/*.json` files. It includes them as pytest parameters only if the parent directory looks like an agent directory (`agent.py`, `__init__.py`, or `root_agent.yaml` present). Files ending in `_xfail.json` are marked `xfail` automatically.

### Example 1 — MockModel returning a scripted response

```python
import asyncio
from google.adk.cli.agent_test_runner import MockModel
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

async def main():
    scripted = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="The capital of France is Paris.")]
        )
    )

    model = MockModel(responses=[scripted])
    request = LlmRequest(model="mock", contents=[
        types.Content(role="user", parts=[types.Part(text="Capital of France?")])
    ])

    # generate_content_async yields the scripted response
    responses = []
    async for resp in model.generate_content_async(request):
        responses.append(resp)

    print(responses[0].content.parts[0].text)  # The capital of France is Paris.

asyncio.run(main())
```

### Example 2 — full offline agent test with MockModel

```python
import asyncio
import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.cli.agent_test_runner import MockModel
from google.adk.models.llm_response import LlmResponse
from google.genai import types

def make_text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )

@pytest.mark.asyncio
async def test_greeting_agent():
    scripted = make_text_response("Hello, Alice! How can I help you?")
    model = MockModel(responses=[scripted])

    # Pass the MockModel instance directly so the scripted responses are used
    agent = LlmAgent(name="greeter", model=model,
                     instruction="Greet the user by name.")

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="test_app",
                    session_service=session_service)
    session = await session_service.create_session(app_name="test_app", user_id="alice")

    msg = types.Content(role="user", parts=[types.Part(text="Hi, I'm Alice")])
    events = []
    async for event in runner.run_async(user_id="alice", session_id=session.id, new_message=msg):
        events.append(event)

    final = [e for e in events if e.is_final_response()]
    assert final, "Expected at least one final response"
```

### Example 3 — using EXCLUDED_EVENT_FIELDS for stable snapshot assertions

```python
from google.adk.cli.agent_test_runner import EXCLUDED_EVENT_FIELDS
from google.adk.events.event import Event

def event_to_comparable_dict(event: Event) -> dict:
    """Strip non-deterministic fields for stable test assertions."""
    d = event.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in d.items() if k not in EXCLUDED_EVENT_FIELDS}

# Usage in tests:
# actual_dict = event_to_comparable_dict(events[0])
# expected = {"author": "model", "content": {"role": "model", "parts": [{"text": "Hello"}]}}
# assert actual_dict == expected
```

---

## 9 · `trace_call_llm` + `trace_tool_call` — OTel span attribute writing

**Module:** `google.adk.telemetry.tracing`

These two functions write OpenTelemetry span attributes for every LLM and tool invocation. They are called by the ADK runtime — but understanding their attribute contract lets you write custom OTel exporters and dashboards.

### `trace_call_llm` key attributes

| Attribute | Value |
|---|---|
| `gen_ai.system` | `"gcp.vertex.agent"` |
| `gen_ai.request.model` | model name from `llm_request.model` |
| `gcp.vertex.agent.invocation_id` | invocation ID |
| `gcp.vertex.agent.session_id` | session ID |
| `gcp.vertex.agent.llm_request` | JSON of request (or `"{}"` when content capture disabled) |
| `gcp.vertex.agent.llm_response` | JSON of response (or `"{}"`) |

Content capture is controlled by the `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` env var (default: `true`). Set it to `"false"` to suppress PII from spans.

### `trace_tool_call` key attributes

| Attribute | Value |
|---|---|
| `gen_ai.operation.name` | `"execute_tool"` |
| `gen_ai.tool.name` | `tool.name` |
| `gen_ai.tool.type` | `tool.__class__.__name__` |
| `gen_ai.tool.description` | `tool.description` |
| `error.type` | exception class name or `error_type` string |
| `gcp.mcp.server.destination.id` | from `tool.custom_metadata` (AppHub mapping) |

**`GCP_MCP_SERVER_DESTINATION_ID` pattern**: when a `BaseTool` has `custom_metadata["gcp.mcp.server.destination.id"]`, `trace_tool_call` reads it and writes it as a span attribute. [AppHub](https://cloud.google.com/app-hub/docs/) uses this attribute to associate tool calls with their destination GCP resources.

**Error-type precedence**: if both an `error: Exception` and an `error_type: str` are passed, the exception wins (`error.type = type(error).__name__`).

### Example 1 — read live spans with an in-memory exporter

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Wire up an in-memory exporter before any ADK code runs
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# --- run your agent here ---
# from google.adk.runners import Runner; ...

# After the run, inspect spans
for span in exporter.get_finished_spans():
    attrs = dict(span.attributes or {})
    print(span.name, attrs.get("gen_ai.system"), attrs.get("gen_ai.tool.name"))
```

### Example 2 — suppress content capture for PII-sensitive deployments

```python
import os
os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "false"

# Now import and run ADK — llm_request / llm_response attributes will be "{}"
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(name="pii_safe", model="gemini-2.5-flash",
                 instruction="Answer questions.")
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="pii_app",
                session_service=session_service)

async def main():
    session = await session_service.create_session(app_name="pii_app", user_id="u1")
    msg = types.Content(role="user", parts=[types.Part(text="What is 2+2?")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 3 — add GCP_MCP_SERVER_DESTINATION_ID to a custom tool

```python
from google.adk.tools import FunctionTool
from google.adk.telemetry.tracing import GCP_MCP_SERVER_DESTINATION_ID

def query_database(sql: str) -> str:
    """Query the production database."""
    return f"[mock result for: {sql}]"

# Attach an AppHub destination ID so Cloud Trace links this tool
# to the backing Cloud SQL resource
db_tool = FunctionTool(func=query_database)
db_tool.custom_metadata = {
    GCP_MCP_SERVER_DESTINATION_ID: "projects/my-proj/locations/us-central1/cloudsqlInstances/prod-db"
}

# When trace_tool_call runs, it reads this key and sets it as a span attribute.
# AppHub dashboards then show a dependency edge to the Cloud SQL instance.
print(db_tool.custom_metadata[GCP_MCP_SERVER_DESTINATION_ID])
```

---

## 10 · `SerializedBaseModel` — the camelCase JSON bridge

**Module:** `google.adk.utils._serialized_base_model`

`SerializedBaseModel` is a thin `pydantic.BaseModel` subclass that enforces camelCase JSON serialisation across all ADK types exposed to the web server and storage layer. It is the base class for `Event`, `Session`, and many API response models.

### Class definition

```python
class SerializedBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=alias_generators.to_camel,  # snake_case → camelCase in JSON
        populate_by_name=True,                      # accept both foo_bar and fooBar on input
        use_attribute_docstrings=True,              # docstrings become field descriptions
    )

    def model_dump_json(self, **kwargs) -> str:
        kwargs.setdefault('by_alias', True)         # always camelCase output
        return super().model_dump_json(**kwargs)
```

### Key behaviours

- **`by_alias=True` default** — calling `model.model_dump_json()` always produces camelCase without any caller boilerplate. Callers that need snake_case must explicitly pass `by_alias=False`.
- **`populate_by_name=True`** — Python code can set fields using snake_case names (`event.invocation_id = ...`) while the HTTP wire format uses camelCase (`{"invocationId": ...}`).
- **`use_attribute_docstrings=True`** — field docstrings (not just `Field(description=...)`) appear in JSON Schema output, powering OpenAPI docs automatically.

### Example 1 — defining a web-API response model

```python
from google.adk.utils._serialized_base_model import SerializedBaseModel

class AgentRunResult(SerializedBaseModel):
    session_id: str
    """The session ID for this run."""
    invocation_id: str
    """The unique invocation identifier."""
    final_response: str
    """The agent's final text response."""

result = AgentRunResult(
    session_id="sess-1",
    invocation_id="inv-abc",
    final_response="The answer is 42.",
)

# JSON output is camelCase by default
print(result.model_dump_json())
# {"sessionId":"sess-1","invocationId":"inv-abc","finalResponse":"The answer is 42."}

# Python attribute access uses snake_case
print(result.session_id)  # sess-1
```

### Example 2 — accepting both camelCase and snake_case on input

```python
from google.adk.utils._serialized_base_model import SerializedBaseModel

class RunRequest(SerializedBaseModel):
    user_id: str
    session_id: str
    new_message: str

# Both forms work thanks to populate_by_name=True + alias_generator
from_api = RunRequest.model_validate(
    {"userId": "u1", "sessionId": "s1", "newMessage": "Hello"}
)
from_python = RunRequest.model_validate(
    {"user_id": "u1", "session_id": "s1", "new_message": "Hello"}
)
assert from_api == from_python  # identical
print(from_api.user_id)         # u1
print(from_api.model_dump_json())  # {"userId":"u1","sessionId":"s1","newMessage":"Hello"}
```

### Example 3 — disabling camelCase for internal snake_case output

```python
from google.adk.utils._serialized_base_model import SerializedBaseModel

class InternalData(SerializedBaseModel):
    event_type: str
    source_agent: str

data = InternalData(event_type="model_response", source_agent="summarizer")

# Default: camelCase for the wire
print(data.model_dump_json())           # {"eventType":"model_response","sourceAgent":"summarizer"}

# Override for internal serialization needing snake_case
print(data.model_dump_json(by_alias=False))  # {"event_type":"model_response","source_agent":"summarizer"}

# model_dump() (dict) also respects by_alias
import json
print(json.dumps(data.model_dump(by_alias=True)))   # camelCase dict
print(json.dumps(data.model_dump(by_alias=False)))  # snake_case dict
```

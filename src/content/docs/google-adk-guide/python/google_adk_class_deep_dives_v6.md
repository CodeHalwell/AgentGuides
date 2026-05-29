---
title: "Class deep dives — volume 6 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: ComputerUseTool/BaseComputer, OpenAPIToolset/RestApiTool, LlmEventSummarizer, Session/State, Event/EventActions, ExampleTool, GoogleSearchTool/UrlContextTool, LlmBackedUserSimulator/UserPersona, GEPARootAgentPromptOptimizer, EnvironmentSimulationPlugin."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 6"
  order: 65
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `ComputerUseTool` + `ComputerUseToolset` + `BaseComputer` | `google.adk.tools.computer_use` | ⚠️ Experimental |
| 2 | `OpenAPIToolset` + `RestApiTool` | `google.adk.tools.openapi_tool` | Stable |
| 3 | `LlmEventSummarizer` + `BaseEventsSummarizer` | `google.adk.apps` | ⚠️ Experimental |
| 4 | `Session` + `State` | `google.adk.sessions` | Stable |
| 5 | `Event` + `EventActions` + `EventCompaction` | `google.adk.events` | Stable |
| 6 | `ExampleTool` + `Example` + `BaseExampleProvider` + `VertexAiExampleStore` | `google.adk.tools` / `google.adk.examples` | Stable |
| 7 | `GoogleSearchTool` + `UrlContextTool` + `GoogleSearchAgentTool` | `google.adk.tools` | Stable |
| 8 | `LlmBackedUserSimulator` + `LlmBackedUserSimulatorConfig` + `UserPersona` + `UserBehavior` | `google.adk.evaluation.simulation` | ⚠️ Experimental |
| 9 | `GEPARootAgentPromptOptimizer` + `GEPARootAgentPromptOptimizerConfig` | `google.adk.optimization` | ⚠️ Experimental |
| 10 | `EnvironmentSimulationPlugin` + `EnvironmentSimulationConfig` + `ToolSimulationConfig` | `google.adk.tools.environment_simulation` | ⚠️ Experimental |

---

## 1 · `ComputerUseTool` + `ComputerUseToolset` + `BaseComputer`

> ⚠️ **Experimental** — decorated with `@experimental(FeatureName.COMPUTER_USE)`. API may change without notice.

These three classes give agents the ability to control a browser or GUI environment. `BaseComputer` is the abstract interface you implement for your target environment; `ComputerUseTool` wraps a single computer action function and handles coordinate normalisation; `ComputerUseToolset` wires the full `BaseComputer` into an agent.

### How coordinate normalisation works (from source)

The LLM thinks in a **virtual coordinate space** (default 1000×1000). `ComputerUseTool` automatically remaps those coordinates to the actual screen pixels before calling the underlying function:

```python
def _normalize_x(self, x: int) -> int:
    normalized = int(x / self._coordinate_space[0] * self._screen_size[0])
    return max(0, min(normalized, self._screen_size[0] - 1))
```

Drag-and-drop coordinates (`destination_x`, `destination_y`) are normalised by the same logic.

### `BaseComputer` — abstract methods you must implement

| Method | Signature | Notes |
|--------|-----------|-------|
| `screen_size()` | `async → tuple[int, int]` | Returns `(width, height)` in pixels |
| `open_web_browser()` | `async → ComputerState` | Opens browser; returns screenshot + URL |
| `click_at(x, y)` | `async → ComputerState` | Click at virtual coords |
| `hover_at(x, y)` | `async → ComputerState` | Hover — useful for sub-menus |
| `type_text_at(x, y, text, press_enter, clear_before_typing)` | `async → ComputerState` | `press_enter=True`, `clear_before_typing=True` by default |
| `scroll_document(direction)` | `async → ComputerState` | `"up" \| "down" \| "left" \| "right"` |
| `scroll_at(x, y, direction, magnitude)` | `async → ComputerState` | Scroll a specific element |
| `wait(seconds)` | `async → ComputerState` | Wait for async page processes |
| `go_back()` / `go_forward()` | `async → ComputerState` | Browser history |
| `search()` | `async → ComputerState` | Jump to search engine home page |
| `navigate(url)` | `async → ComputerState` | Navigate to a URL directly |
| `key_combination(keys)` | `async → ComputerState` | e.g. `["control", "c"]` |
| `drag_and_drop(x, y, destination_x, destination_y)` | `async → ComputerState` | Drag-and-drop |
| `current_state()` | `async → ComputerState` | Current screenshot + URL |
| `environment()` | `async → ComputerEnvironment` | `ENVIRONMENT_BROWSER` or `ENVIRONMENT_DESKTOP` |

Override `prepare(tool_context)` (not abstract) to bind session state before each tool call — used by sandbox computers to share tokens across calls.

### `ComputerUseToolset` constructor

```python
ComputerUseToolset(
    *,
    computer: BaseComputer,
    excluded_predefined_functions: Optional[list[str]] = None,
)
```

`get_tools()` calls `computer.screen_size()` to get actual dimensions, then wraps every non-private, non-excluded method of `BaseComputer` in a `ComputerUseTool`. It also injects `tool_context: ToolContext` into each wrapper so the computer's `prepare()` is called before each action.

### Example 1 — minimal in-process browser computer

```python
import asyncio
import io
from typing import Literal
from google.adk.tools.computer_use.base_computer import (
    BaseComputer, ComputerState, ComputerEnvironment,
)
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


class HeadlessBrowser(BaseComputer):
    """Minimal stub — replace with playwright/selenium integration."""

    def __init__(self):
        self._url = "about:blank"

    async def screen_size(self) -> tuple[int, int]:
        return (1280, 800)

    async def environment(self) -> ComputerEnvironment:
        return ComputerEnvironment.ENVIRONMENT_BROWSER

    async def open_web_browser(self) -> ComputerState:
        self._url = "https://www.google.com"
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def navigate(self, url: str) -> ComputerState:
        self._url = url
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def click_at(self, x: int, y: int) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def hover_at(self, x: int, y: int) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def type_text_at(
        self, x: int, y: int, text: str,
        press_enter: bool = True, clear_before_typing: bool = True,
    ) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def scroll_document(
        self, direction: Literal["up", "down", "left", "right"]
    ) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def scroll_at(
        self, x: int, y: int,
        direction: Literal["up", "down", "left", "right"], magnitude: int,
    ) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def wait(self, seconds: int) -> ComputerState:
        await asyncio.sleep(seconds)
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def go_back(self) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def go_forward(self) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def search(self) -> ComputerState:
        return await self.navigate("https://www.google.com")

    async def key_combination(self, keys: list[str]) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int,
    ) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)

    async def current_state(self) -> ComputerState:
        return ComputerState(screenshot=b"PNG_BYTES_HERE", url=self._url)


async def main():
    toolset = ComputerUseToolset(computer=HeadlessBrowser())

    agent = LlmAgent(
        name="browser_agent",
        model="gemini-2.5-flash",
        instruction="You control a web browser. Navigate and interact with pages as instructed.",
        tools=[toolset],
    )

    runner = InMemoryRunner(agent=agent, app_name="browser_demo")
    await runner.session_service.create_session(
        app_name="browser_demo", user_id="u1", session_id="s1"
    )

    events = await runner.run_debug(
        "Navigate to https://example.com and tell me the page title",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

    await toolset.close()

asyncio.run(main())
```

### Example 2 — adapting a tool (replacing `wait` with a no-op)

`ComputerUseToolset.adapt_computer_use_tool` lets you replace a built-in action at the LLM-request level:

```python
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset

async def noop_wait_adapter(original_func):
    """Replace wait with an instant no-op during tests."""
    async def instant_wait(seconds: int) -> dict:
        return {"status": "wait_skipped", "seconds": seconds}
    return instant_wait

# Call inside process_llm_request or a plugin hook:
await ComputerUseToolset.adapt_computer_use_tool(
    method_name="wait",
    adapter_func=noop_wait_adapter,
    llm_request=llm_request,
)
```

### Safety confirmation pattern

If your computer action function returns a `safety_decision` argument set to `{"decision": "require_confirmation"}`, `ComputerUseTool` automatically calls `tool_context.request_confirmation()` and blocks execution until the user approves via the HITL flow.

---

## 2 · `OpenAPIToolset` + `RestApiTool`

`OpenAPIToolset` parses an **OpenAPI 3.x spec** (JSON or YAML) and returns one `RestApiTool` per operation. Each tool handles parameter marshalling, authentication, SSL, and request execution automatically.

### `OpenAPIToolset` constructor

```python
OpenAPIToolset(
    *,
    spec_dict: Optional[dict] = None,
    spec_str: Optional[str] = None,
    spec_str_type: Literal["json", "yaml"] = "json",
    auth_scheme: Optional[AuthScheme] = None,
    auth_credential: Optional[AuthCredential] = None,
    credential_key: Optional[str] = None,
    tool_filter: Optional[Union[ToolPredicate, list[str]]] = None,
    tool_name_prefix: Optional[str] = None,
    ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None,
    header_provider: Optional[Callable[[ReadonlyContext], dict[str, str]]] = None,
    preserve_property_names: bool = False,
)
```

| Parameter | Notes |
|-----------|-------|
| `spec_dict` / `spec_str` | Provide one. `spec_str` is parsed with `json.loads` or `yaml.safe_load`. |
| `auth_scheme` + `auth_credential` | Applied to **all** operations. Use `auth_helpers` from `google.adk.tools.openapi_tool.auth` to build these. |
| `credential_key` | Stable key for interactive auth and cross-invocation credential caching. |
| `tool_filter` | `list[str]` of operation IDs to expose, or a `ToolPredicate` callable. |
| `tool_name_prefix` | Prepend to all tool names — avoids collisions when loading multiple specs. |
| `ssl_verify` | `False` to skip TLS verification in dev, or a path to a custom CA bundle. |
| `header_provider` | Callable receiving `ReadonlyContext` → `dict[str, str]`. Adds dynamic headers (correlation IDs, per-request tokens) to every call. |
| `preserve_property_names` | `True` keeps camelCase/original names instead of converting to snake_case. |

> Tool names are derived from `operationId` converted to snake_case, truncated to 60 characters (Gemini limit).

### `RestApiTool` key methods

| Method | Notes |
|--------|-------|
| `from_parsed_operation(parsed, ssl_verify, header_provider)` | Class method — preferred way to construct from parsed spec. |
| `configure_auth_scheme(auth_scheme)` | Set/replace auth scheme after construction. |
| `configure_auth_credential(auth_credential)` | Set/replace credentials after construction. |
| `configure_ssl_verify(ssl_verify)` | Set SSL policy post-construction. |
| `set_default_headers(headers)` | Merge extra headers into every request (merged before `header_provider`). |
| `call(args, tool_context)` | Execute the REST call — handles auth, path/query/body params, error parsing. |

### Example 1 — load a public API spec from a JSON string

```python
import asyncio
import json
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# Minimal OpenAPI 3.0 spec for a weather service
WEATHER_SPEC = json.dumps({
    "openapi": "3.0.0",
    "info": {"title": "Weather API", "version": "1.0"},
    "servers": [{"url": "https://api.open-meteo.com"}],
    "paths": {
        "/v1/forecast": {
            "get": {
                "operationId": "get_weather_forecast",
                "summary": "Get weather forecast for a location",
                "parameters": [
                    {"name": "latitude", "in": "query", "required": True,
                     "schema": {"type": "number"}},
                    {"name": "longitude", "in": "query", "required": True,
                     "schema": {"type": "number"}},
                    {"name": "current", "in": "query", "required": False,
                     "schema": {"type": "string",
                                "default": "temperature_2m,wind_speed_10m"}},
                ],
                "responses": {"200": {"description": "Forecast data"}},
            }
        }
    },
})

toolset = OpenAPIToolset(spec_str=WEATHER_SPEC, spec_str_type="json")

agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction="Answer weather questions using the weather API.",
    tools=toolset,
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="weather")
    await runner.session_service.create_session(
        app_name="weather", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the current temperature in London? (lat=51.5, lon=-0.12)",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)
    await toolset.close()

asyncio.run(main())
```

### Example 2 — OAuth2 bearer auth + per-request correlation ID header

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# HTTP Bearer auth credential
auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="bearer",
        credentials=HttpCredentials(token="my-api-token"),
    ),
)

def add_correlation_id(readonly_context) -> dict:
    """Inject per-request correlation ID from session state."""
    session_id = readonly_context.session.id
    return {"X-Correlation-ID": session_id}

toolset = OpenAPIToolset(
    spec_str=MY_API_SPEC,
    spec_str_type="yaml",
    auth_credential=auth_credential,
    header_provider=add_correlation_id,
    tool_filter=["create_order", "get_order", "list_orders"],  # expose subset
    tool_name_prefix="orders_",
)

agent = LlmAgent(
    name="order_agent",
    model="gemini-2.5-flash",
    instruction="Manage customer orders using the API.",
    tools=toolset,
)
```

### Example 3 — multiple specs without name collisions

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

crm_toolset = OpenAPIToolset(
    spec_dict=crm_spec,
    tool_name_prefix="crm_",          # → crm_get_contact, crm_create_deal
)
billing_toolset = OpenAPIToolset(
    spec_dict=billing_spec,
    tool_name_prefix="billing_",      # → billing_get_invoice, billing_pay
)

agent = LlmAgent(
    name="ops_agent",
    model="gemini-2.5-flash",
    tools=[crm_toolset, billing_toolset],
)
```

### SSL and enterprise proxy configuration

```python
# Custom CA bundle for TLS-intercepting corporate proxy
toolset = OpenAPIToolset(
    spec_dict=spec,
    ssl_verify="/etc/ssl/certs/corporate-ca.pem",  # path to CA bundle
)

# Or disable entirely for internal non-TLS environments (dev only)
toolset = OpenAPIToolset(spec_dict=spec, ssl_verify=False)
```

---

## 3 · `LlmEventSummarizer` + `BaseEventsSummarizer`

> ⚠️ **Experimental** — `BaseEventsSummarizer` is decorated with `@experimental`.

When you configure `EventsCompactionConfig` on an `App`, the runner periodically compacts old events into a summary to prevent context-window overflow. `LlmEventSummarizer` is the built-in implementation that uses an LLM to generate a narrative summary.

### `BaseEventsSummarizer` interface

```python
class BaseEventsSummarizer(abc.ABC):
    @abc.abstractmethod
    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Optional[Event]: ...
```

Return a new `Event` (with `actions.compaction` set) or `None` if no compaction happened. The runner replaces the summarised events with this single compaction event.

### `LlmEventSummarizer` constructor

```python
LlmEventSummarizer(
    llm: BaseLlm,
    prompt_template: Optional[str] = None,
)
```

| Parameter | Notes |
|-----------|-------|
| `llm` | Any `BaseLlm` instance (e.g. `GeminiLlm`, `LiteLlm`). Used for summarisation only — independent from the agent's model. |
| `prompt_template` | Jinja-style template with `{conversation_history}` placeholder. Defaults to a built-in narrative summary prompt. |

**Default prompt template (from source):**
```
The following is a conversation history between a user and an AI agent. Please
summarize the conversation, focusing on key information and decisions made, as
well as any unresolved questions or tasks. The summary should be concise and
capture the essence of the interaction.

{conversation_history}
```

The summariser formats events by extracting `event.content.parts[0].text` per author and joining them as `"author: text\n"` lines.

The returned `Event` contains:
- `actions.compaction.start_timestamp` — timestamp of the first summarised event
- `actions.compaction.end_timestamp` — timestamp of the last summarised event
- `actions.compaction.compacted_content` — the LLM-generated summary `Content`

### Example 1 — attach compaction to an App

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps.app import App
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import GenerateContentConfig
from google.adk.models.lite_llm import LiteLlm  # or GeminiLlm

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

summarizer = LlmEventSummarizer(
    llm=LiteLlm(model="gemini/gemini-2.5-flash"),
)

compaction_config = EventsCompactionConfig(
    compaction_invocation_threshold=20,  # compact after 20 invocations
    overlap_size=5,                      # keep last 5 events after compaction
    events_summarizer=summarizer,
)

app = App(
    name="long_running_app",
    root_agent=agent,
    events_compaction_config=compaction_config,
)

async def main():
    session_svc = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="long_running_app",
        session_service=session_svc,
    )
    await session_svc.create_session(
        app_name="long_running_app", user_id="u1", session_id="s1"
    )

    for i in range(25):
        await runner.run_debug(
            f"Tell me fact number {i+1} about space.",
            user_id="u1", session_id="s1",
        )

asyncio.run(main())
```

### Example 2 — custom summarisation prompt

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer

TECH_SUMMARY_PROMPT = """\
You are a technical note-taker. Summarise this agent conversation, preserving:
- All function calls and their outcomes
- Any error states or retry attempts
- Final decisions and outputs

Conversation:
{conversation_history}
"""

summarizer = LlmEventSummarizer(
    llm=my_llm,
    prompt_template=TECH_SUMMARY_PROMPT,
)
```

### Example 3 — custom summariser that keeps only tool calls

```python
from typing import Optional
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.genai import types


class ToolCallOnlySummarizer(BaseEventsSummarizer):
    """Drops all text turns; keeps only function call / response pairs."""

    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Optional[Event]:
        lines = []
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        lines.append(
                            f"TOOL_CALL {part.function_call.name}("
                            f"{part.function_call.args})"
                        )
                    elif part.function_response:
                        lines.append(
                            f"TOOL_RESP {part.function_response.name} → "
                            f"{part.function_response.response}"
                        )

        if not lines:
            return None

        summary = "\n".join(lines)
        summary_content = types.Content(
            role="model",
            parts=[types.Part(text=f"Tool call history:\n{summary}")],
        )

        compaction = EventCompaction(
            start_timestamp=events[0].timestamp,
            end_timestamp=events[-1].timestamp,
            compacted_content=summary_content,
        )
        return Event(
            author="user",
            actions=EventActions(compaction=compaction),
            invocation_id=Event.new_id(),
        )
```

---

## 4 · `Session` + `State`

`Session` is the persistent record of a conversation; `State` is the live, write-tracked dict that merges committed session state with in-flight deltas.

### `Session` fields (verified source)

```python
class Session(BaseModel):
    id: str                     # Unique session identifier
    app_name: str               # Name of the app
    user_id: str                # User who owns the session
    state: dict[str, Any]       # Key-value state bag (committed values)
    events: list[Event]         # Ordered history of all events
    last_update_time: float     # Unix timestamp of last update
```

`Session` uses `alias_generator=to_camel` so it serialises to/from camelCase JSON automatically (useful when exchanging with Vertex AI APIs).

### `State` — scope prefixes and delta tracking

`State` wraps the raw `dict[str, Any]` from `Session.state` and adds:
- **Uncommitted delta tracking** — writes are recorded in `_delta` and merged on storage commit
- **Scope prefixes** — by convention, keys with certain prefixes have different persistence lifetimes

| Prefix | Constant | Scope |
|--------|----------|-------|
| `"app:"` | `State.APP_PREFIX` | Shared across all users and sessions of the app |
| `"user:"` | `State.USER_PREFIX` | Shared across all sessions for a given user |
| `"temp:"` | `State.TEMP_PREFIX` | Current invocation only — not persisted |
| _(no prefix)_ | — | Session-scoped (default) |

`State` supports `__getitem__`, `__setitem__`, `__contains__`, `get`, `setdefault`, `update`, `has_delta()`, and `to_dict()`.

### `State` schema validation

Pass a Pydantic model as `schema` to `State.__init__` to validate every mutation:

```python
from pydantic import BaseModel

class AppState(BaseModel):
    cart_items: list[str] = []
    order_total: float = 0.0
    user_name: str = ""
```

When any agent writes an invalid key/value pair (type mismatch or unknown key), `StateSchemaError` (a `TypeError` subclass) is raised immediately.

### Example 1 — reading and writing state in a tool

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.context import ToolContext
from google.adk.runners import InMemoryRunner


def track_topic(topic: str, tool_context: ToolContext) -> dict:
    """Record conversation topics in session state."""
    topics = tool_context.state.get("topics", [])
    if topic not in topics:
        topics = topics + [topic]
        tool_context.state["topics"] = topics

    # Write to user-scoped state (persists across sessions for this user)
    all_topics = tool_context.state.get("user:all_topics", [])
    if topic not in all_topics:
        tool_context.state["user:all_topics"] = all_topics + [topic]

    return {"recorded_topic": topic, "session_topics": topics}


agent = LlmAgent(
    name="tracker",
    model="gemini-2.5-flash",
    instruction="Record topics the user mentions using track_topic.",
    tools=[track_topic],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="alice", session_id="s1"
    )
    await runner.run_debug(
        "I'm interested in machine learning and Python.",
        user_id="alice", session_id="s1",
    )

    session = await runner.session_service.get_session(
        app_name="demo", user_id="alice", session_id="s1"
    )
    print("Session topics:", session.state.get("topics"))
    print("User topics:", session.state.get("user:all_topics"))

asyncio.run(main())
```

### Example 2 — session state schema enforcement

```python
import asyncio
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.agents.context import ToolContext
from google.adk.runners import InMemoryRunner
from google.adk.sessions.state import State, StateSchemaError


class OrderState(BaseModel):
    item_count: int = 0
    total_usd: float = 0.0
    status: str = "pending"


def add_item(price: float, tool_context: ToolContext) -> dict:
    tool_context.state["item_count"] = tool_context.state.get("item_count", 0) + 1
    tool_context.state["total_usd"] = (
        tool_context.state.get("total_usd", 0.0) + price
    )
    try:
        # This would raise StateSchemaError — "unknown_key" not in OrderState
        tool_context.state["unknown_key"] = "bad"
    except StateSchemaError as e:
        print(f"Schema violation blocked: {e}")
    return {"item_count": tool_context.state["item_count"]}
```

### Example 3 — inspecting session events

```python
async def inspect_session(runner, app_name, user_id, session_id):
    session = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    print(f"Session {session.id}: {len(session.events)} events")
    for event in session.events:
        fc = event.get_function_calls()
        fr = event.get_function_responses()
        text = ""
        if event.content and event.content.parts:
            text = (event.content.parts[0].text or "")[:80]
        print(f"  [{event.author}] text='{text}' calls={len(fc)} responses={len(fr)}")
```

---

## 5 · `Event` + `EventActions` + `EventCompaction`

`Event` is the fundamental record of everything that happens in a conversation — user messages, model responses, tool calls, tool results, state changes, agent transfers, and compaction markers.

### `Event` field reference (source-verified)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `id` | `str` | auto-generated | UUID, assigned on first creation |
| `invocation_id` | `str` | `""` | Must be non-empty before appending to a session |
| `author` | `str` | `""` | `"user"` or the agent's `name` |
| `content` | `types.Content \| None` | `None` | Inherited from `LlmResponse` — text, tool calls, images |
| `actions` | `EventActions` | `EventActions()` | State deltas, transfers, auth requests, etc. |
| `output` | `Any \| None` | `None` | Generic data output from a Workflow `@node` |
| `node_info` | `NodeInfo` | `NodeInfo()` | Workflow node path and run ID |
| `long_running_tool_ids` | `set[str] \| None` | `None` | IDs of in-flight long-running tool calls |
| `branch` | `str \| None` | `None` | Agent branch path, e.g. `"root.sub_a.leaf"` |
| `timestamp` | `float` | `platform_time.get_time()` | Unix timestamp |

**Convenience constructor kwargs:**

```python
Event(
    message="Hello",          # alias for content (text shorthand)
    state={"key": "val"},     # alias for actions.state_delta
    route="next_node",        # alias for actions.route
    node_path="my_node",      # alias for node_info.path
)
# NOTE: 'message' and 'content' are mutually exclusive
```

### `Event` helper methods

| Method | Returns | Notes |
|--------|---------|-------|
| `is_final_response()` | `bool` | True when event has no pending calls and `skip_summarization` is not set |
| `get_function_calls()` | `list[FunctionCall]` | Extracts all `part.function_call` items from content |
| `get_function_responses()` | `list[FunctionResponse]` | Extracts all `part.function_response` items |
| `has_trailing_code_execution_result()` | `bool` | True when last part is a code execution result |
| `Event.new_id()` | `str` | Static — generates a new UUID |
| `event.message` | `Content \| None` | Property alias for `event.content` |

### `EventActions` field reference

| Field | Type | Notes |
|-------|------|-------|
| `skip_summarization` | `bool \| None` | Skip model summarisation for this function-response event |
| `state_delta` | `dict[str, Any]` | State mutations to commit with this event |
| `artifact_delta` | `dict[str, int]` | Artifact name → version number |
| `transfer_to_agent` | `str \| None` | Name of agent to transfer control to |
| `escalate` | `bool \| None` | Signal to return control to parent agent |
| `requested_auth_configs` | `dict[str, AuthConfig]` | Auth configs keyed by function-call ID |
| `requested_tool_confirmations` | `dict[str, ToolConfirmation]` | HITL confirmations keyed by function-call ID |
| `compaction` | `EventCompaction \| None` | Present when this event replaces a range of events |
| `end_of_agent` | `bool \| None` | Set by workflow when an agent finishes its run |
| `route` | `bool \| int \| str \| list \| None` | Workflow routing value for edge matching |
| `render_ui_widgets` | `list[UiWidget] \| None` | UI widgets to render alongside this event |
| `set_model_response` | `Any \| None` | Structured output override |

### Example 1 — reading events from a session

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


async def print_events(runner, user_id, session_id):
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    for event in session.events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    print(f"  → {event.author}: {part.text}")
        fc = event.get_function_calls()
        for call in fc:
            print(f"  CALL {call.name}({call.args})")
        fr = event.get_function_responses()
        for resp in fr:
            print(f"  RESP {resp.name} → {resp.response}")


agent = LlmAgent(
    name="agent",
    model="gemini-2.5-flash",
    instruction="Answer questions concisely.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="events_demo")
    await runner.session_service.create_session(
        app_name="events_demo", user_id="u1", session_id="s1"
    )
    await runner.run_debug("What is 2 + 2?", user_id="u1", session_id="s1")
    await print_events(runner, "u1", "s1")

asyncio.run(main())
```

### Example 2 — writing state delta via `EventActions` in a Workflow node

```python
from google.adk.agents.context import ToolContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

def emit_state_event(new_status: str) -> Event:
    """Create a synthetic event that commits a state update without LLM output."""
    return Event(
        author="orchestrator",
        invocation_id=Event.new_id(),
        actions=EventActions(
            state_delta={"order_status": new_status},
            skip_summarization=True,
        ),
    )
```

### Example 3 — streaming events from `run_async`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


agent = LlmAgent(
    name="streamer",
    model="gemini-2.5-flash",
    instruction="Write a short story in 3 paragraphs.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="stream_demo")
    await runner.session_service.create_session(
        app_name="stream_demo", user_id="u1", session_id="s1"
    )

    full_text = ""
    async for event in runner.run_async(
        new_message="Tell me a short story",
        user_id="u1",
        session_id="s1",
    ):
        if event.partial and event.content and event.content.parts:
            chunk = event.content.parts[0].text or ""
            print(chunk, end="", flush=True)
            full_text += chunk
        elif event.is_final_response():
            print()  # newline at end

asyncio.run(main())
```

---

## 6 · `ExampleTool` + `Example` + `BaseExampleProvider` + `VertexAiExampleStore`

`ExampleTool` implements **few-shot prompting** by injecting input/output examples into the system instruction before the LLM call. This is a powerful technique for steering model behaviour without fine-tuning.

### Data model

```python
class Example(BaseModel):
    input: types.Content     # Example user message
    output: list[types.Content]  # Expected model response (can be multi-turn)
```

### `ExampleTool` constructor

```python
ExampleTool(
    examples: Union[list[Example], BaseExampleProvider]
)
```

- Pass a static `list[Example]` for fixed few-shot examples.
- Pass a `BaseExampleProvider` for **dynamic, query-dependent** example retrieval (e.g. from a vector store).

`ExampleTool` does not appear in the agent's tool list — it operates purely via `process_llm_request`, injecting examples as a system instruction addendum before the model call.

### `BaseExampleProvider` interface

```python
class BaseExampleProvider(abc.ABC):
    @abc.abstractmethod
    def get_examples(self, query: str) -> list[Example]: ...
```

Implement this to fetch relevant examples from your store (database, vector index, etc.).

### `VertexAiExampleStore` — managed example retrieval

`VertexAiExampleStore` queries a **Vertex AI Example Store** resource using semantic similarity (top-10, `similarity_score >= 0.5` filter):

```python
VertexAiExampleStore(
    examples_store_name="projects/{project}/locations/{location}/exampleStores/{id}"
)
```

Requires `google-cloud-aiplatform`: `pip install google-adk[gcp]`.

### Example 1 — static few-shot examples for entity extraction

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.example_tool import ExampleTool
from google.adk.examples.example import Example

examples = [
    Example(
        input=types.Content(
            role="user",
            parts=[types.Part(text="Apple released the iPhone 16 in September 2024.")],
        ),
        output=[
            types.Content(
                role="model",
                parts=[types.Part(text='{"entities": [{"name": "Apple", "type": "ORG"}, {"name": "iPhone 16", "type": "PRODUCT"}, {"name": "September 2024", "type": "DATE"}]}')],
            )
        ],
    ),
    Example(
        input=types.Content(
            role="user",
            parts=[types.Part(text="Elon Musk founded SpaceX in 2002.")],
        ),
        output=[
            types.Content(
                role="model",
                parts=[types.Part(text='{"entities": [{"name": "Elon Musk", "type": "PERSON"}, {"name": "SpaceX", "type": "ORG"}, {"name": "2002", "type": "DATE"}]}')],
            )
        ],
    ),
]

agent = LlmAgent(
    name="extractor",
    model="gemini-2.5-flash",
    instruction="Extract named entities from text. Return JSON.",
    tools=[ExampleTool(examples)],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="ner")
    await runner.session_service.create_session(
        app_name="ner", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Google was founded by Larry Page and Sergey Brin in 1998.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — dynamic examples with a custom provider

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.example_tool import ExampleTool
from google.adk.examples.base_example_provider import BaseExampleProvider
from google.adk.examples.example import Example


class KeywordExampleProvider(BaseExampleProvider):
    """Retrieve examples based on keyword matching."""

    def __init__(self, example_bank: list[tuple[str, Example]]):
        self._bank = example_bank  # [(keyword, Example), ...]

    def get_examples(self, query: str) -> list[Example]:
        query_lower = query.lower()
        return [
            example
            for keyword, example in self._bank
            if keyword in query_lower
        ][:3]  # return up to 3 matching examples


provider = KeywordExampleProvider([
    ("sentiment", Example(
        input=types.Content(role="user", parts=[types.Part(text="I love this product!")]),
        output=[types.Content(role="model", parts=[types.Part(text="POSITIVE")])],
    )),
    ("refund", Example(
        input=types.Content(role="user", parts=[types.Part(text="I want a refund for my broken item.")]),
        output=[types.Content(role="model", parts=[types.Part(text="NEGATIVE")])],
    )),
])

agent = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify customer feedback sentiment: POSITIVE, NEUTRAL, or NEGATIVE.",
    tools=[ExampleTool(provider)],
)
```

### Example 3 — Vertex AI Example Store for semantic retrieval

```python
from google.adk.tools.example_tool import ExampleTool
from google.adk.examples.vertex_ai_example_store import VertexAiExampleStore
from google.adk.agents import LlmAgent

store = VertexAiExampleStore(
    examples_store_name=(
        "projects/my-project/locations/us-central1"
        "/exampleStores/my-store-id"
    )
)

agent = LlmAgent(
    name="smart_classifier",
    model="gemini-2.5-flash",
    instruction="Classify support tickets using the examples provided.",
    tools=[ExampleTool(store)],
)
```

The store fetches top-10 similar examples and filters to those with `similarity_score >= 0.5`. Supports text, function calls, and function responses in the `output` field.

---

## 7 · `GoogleSearchTool` + `UrlContextTool` + `GoogleSearchAgentTool`

These three classes are **model-native built-in tools** — they don't execute local Python functions. Instead they inject configuration into the LLM request so the **Gemini model itself** performs the search or URL fetch internally.

### `GoogleSearchTool`

Activates Google Search grounding so Gemini's response is backed by live search results.

```python
GoogleSearchTool(
    *,
    bypass_multi_tools_limit: bool = False,
    model: Optional[str] = None,
)
```

| Parameter | Notes |
|-----------|-------|
| `bypass_multi_tools_limit` | Gemini 2.x normally restricts `google_search` from mixing with other tools. Set `True` to remove this guard (use with `GoogleSearchAgentTool` pattern instead — see below). |
| `model` | Override the model for this specific tool invocation. Rarely needed. |

**Version behaviour (from source):**
- **Gemini 1.x** — injects `types.GoogleSearchRetrieval()`. Cannot be combined with other tools.
- **Gemini 2.x** — injects `types.GoogleSearch()`. Can be combined only with a few other model-native tools.

### `UrlContextTool`

Activates URL context grounding — Gemini fetches and reads the content of URLs mentioned in the prompt.

```python
UrlContextTool()
# No parameters — works only with Gemini 2.x and above
```

### `GoogleSearchAgentTool` — mixing search with function tools

Since `google_search` cannot normally coexist with custom function tools in the same agent, `GoogleSearchAgentTool` is the official workaround: it wraps a dedicated search sub-agent and exposes it as an `AgentTool`.

```python
GoogleSearchAgentTool(agent: LlmAgent)
# Internally: super().__init__(agent=agent, propagate_grounding_metadata=True)
```

### Example 1 — basic search grounding

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_search_tool import GoogleSearchTool

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Answer questions using Google Search for up-to-date information.",
    tools=[GoogleSearchTool()],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="search_demo")
    await runner.session_service.create_session(
        app_name="search_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What are the latest AI announcements from Google this week?",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — URL context for reading a specific page

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.url_context_tool import UrlContextTool

agent = LlmAgent(
    name="web_reader",
    model="gemini-2.5-flash",
    instruction="Read and summarise web pages when given URLs.",
    tools=[UrlContextTool()],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="url_demo")
    await runner.session_service.create_session(
        app_name="url_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Please summarise the content at https://ai.google.dev/adk/docs",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 3 — search + custom function tools via `GoogleSearchAgentTool`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.google_search_agent_tool import GoogleSearchAgentTool
from google.adk.agents.context import ToolContext


def save_to_database(key: str, value: str, tool_context: ToolContext) -> dict:
    tool_context.state[f"saved_{key}"] = value
    return {"saved": True, "key": key}


# Dedicated search sub-agent (only has GoogleSearchTool)
search_agent = LlmAgent(
    name="search_helper",
    model="gemini-2.5-flash",
    instruction="Search Google and return relevant information.",
    tools=[GoogleSearchTool()],
)

# Main agent: combines web search with custom function tools
main_agent = LlmAgent(
    name="research_and_save_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Research topics using the search helper, then save results to the database. "
        "Use search_helper to find information, then save_to_database to persist it."
    ),
    tools=[
        GoogleSearchAgentTool(agent=search_agent),  # search wrapped as AgentTool
        save_to_database,                            # custom function tool
    ],
)


async def main():
    runner = InMemoryRunner(agent=main_agent, app_name="research")
    await runner.session_service.create_session(
        app_name="research", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Find the current CEO of Anthropic and save it to the database with key 'anthropic_ceo'.",
        user_id="u1", session_id="s1",
    )
    session = await runner.session_service.get_session(
        app_name="research", user_id="u1", session_id="s1"
    )
    print(session.state.get("saved_anthropic_ceo"))

asyncio.run(main())
```

### Example 4 — combining `GoogleSearchTool` and `UrlContextTool`

```python
from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.url_context_tool import UrlContextTool

agent = LlmAgent(
    name="deep_researcher",
    model="gemini-2.5-flash",
    instruction=(
        "First use Google Search to find relevant pages, "
        "then read the most relevant URL using URL context."
    ),
    tools=[GoogleSearchTool(), UrlContextTool()],
)
```

---

## 8 · `LlmBackedUserSimulator` + `LlmBackedUserSimulatorConfig` + `UserPersona` + `UserBehavior`

> ⚠️ **Experimental** — `LlmBackedUserSimulator` is decorated with `@experimental`.

`LlmBackedUserSimulator` drives automated, multi-turn evaluation conversations using an LLM to generate realistic user messages. It pairs with `ConversationScenario` (covered in Vol. 5) to replace human testers in evaluation pipelines.

### `LlmBackedUserSimulatorConfig` fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | `str` | `"gemini-2.5-flash"` | LLM used for user message generation |
| `model_configuration` | `GenerateContentConfig` | Thinking budget 10240 | Controls reasoning depth of simulated user |
| `max_allowed_invocations` | `int` | `20` | Hard stop to prevent runaway loops. Set to `-1` to disable. |
| `custom_instructions` | `str \| None` | `None` | Jinja template with `{{ stop_signal }}`, `{{ conversation_plan }}`, `{{ conversation_history }}` placeholders |
| `include_function_calls` | `bool` | `False` | Whether to include tool calls/responses in the conversation history shown to the simulator |

### `UserPersona` + `UserBehavior` — conditioning simulated user style

A `UserPersona` collects multiple `UserBehavior` instances to define how the simulated user behaves. These are injected into the simulator's system prompt.

```python
class UserBehavior(BaseModel):
    name: str                          # Behaviour identifier
    description: str                   # Prose description (shown to simulator and evaluator)
    behavior_instructions: list[str]   # What the simulated user SHOULD do
    violation_rubrics: list[str]       # What the evaluator checks for violations
```

```python
class UserPersona(BaseModel):
    id: str                            # Human-readable ID used by registries
    description: str                   # Overall persona description
    behaviors: Sequence[UserBehavior]  # List of behaviors to combine
```

### Example 1 — basic evaluation run with LlmBackedUserSimulator

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulator,
    LlmBackedUserSimulatorConfig,
)
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_metrics import PrebuiltMetrics


async def main():
    agent = LlmAgent(
        name="support_agent",
        model="gemini-2.5-flash",
        instruction=(
            "You are a customer support agent. Help users with their issues "
            "and escalate when needed."
        ),
    )

    scenario = ConversationScenario(
        starting_prompt="Hi, my order hasn't arrived yet and it's been 2 weeks.",
        conversation_plan=(
            "The user is frustrated about a delayed order. "
            "They want a refund or replacement. "
            "They will provide order number ORD-12345 if asked."
        ),
    )

    simulator_config = LlmBackedUserSimulatorConfig(
        model="gemini-2.5-flash",
        max_allowed_invocations=10,
        include_function_calls=True,
    )

    simulator = LlmBackedUserSimulator(
        config=simulator_config,
        conversation_scenario=scenario,
    )

    # Run the simulated conversation (5 turns)
    eval_case = EvalCase(
        eval_id="support_test_001",
        conversation=[
            Invocation(user_content=None)  # simulator drives the conversation
        ],
    )

    # Use AgentEvaluator to run and score
    results = await AgentEvaluator.evaluate(
        agent_module="my_module",
        eval_dataset=[eval_case],
        metrics=[PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2],
    )
    print(results)

asyncio.run(main())
```

### Example 2 — custom UserPersona for a demanding user

```python
from google.adk.evaluation.simulation.user_simulator_personas import (
    UserPersona, UserBehavior,
)

demanding_behavior = UserBehavior(
    name="demanding_user",
    description="A user who is impatient and expects immediate resolution.",
    behavior_instructions=[
        "Always mention urgency in your messages.",
        "Push back if the agent asks for more information rather than acting.",
        "Repeat your main complaint if the agent doesn't acknowledge it.",
    ],
    violation_rubrics=[
        "User accepts a delay without pushing back.",
        "User doesn't reiterate urgency when the agent stalls.",
        "User is excessively polite or patient.",
    ],
)

tech_savvy_behavior = UserBehavior(
    name="tech_savvy",
    description="A user who understands technical details and asks specific questions.",
    behavior_instructions=[
        "Ask about error codes and logs.",
        "Suggest specific technical solutions if you know them.",
        "Use technical terminology correctly.",
    ],
    violation_rubrics=[
        "User uses vague language when describing technical issues.",
        "User doesn't ask follow-up questions about technical details.",
    ],
)

engineer_persona = UserPersona(
    id="senior_engineer",
    description="A senior software engineer frustrated by infrastructure issues.",
    behaviors=[demanding_behavior, tech_savvy_behavior],
)
```

### Example 3 — custom simulator prompt template

```python
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulator,
    LlmBackedUserSimulatorConfig,
)

CUSTOM_TEMPLATE = """\
You are simulating a user in a customer support conversation.

Persona: {{ persona }}

Your goal: {{ conversation_plan }}

Signal to end conversation: type exactly '{{ stop_signal }}'

Conversation so far:
{{ conversation_history }}

Generate your next message as the user. Be realistic and in-character.
"""

config = LlmBackedUserSimulatorConfig(
    model="gemini-2.5-flash",
    max_allowed_invocations=15,
    custom_instructions=CUSTOM_TEMPLATE,
)

scenario = ConversationScenario(
    starting_prompt="Hello, I need help with my subscription.",
    conversation_plan="User wants to upgrade to the Pro plan but is confused about pricing.",
)

simulator = LlmBackedUserSimulator(config=config, conversation_scenario=scenario)
```

---

## 9 · `GEPARootAgentPromptOptimizer` + `GEPARootAgentPromptOptimizerConfig`

> ⚠️ **Experimental** — decorated with `@experimental`. Requires the separate `gepa` package: `pip install gepa`.

`GEPARootAgentPromptOptimizer` implements the **GEPA (Generalized Evolutionary Prompt Algorithm)** framework to automatically improve the root agent's instruction by iterating over a training set and using reflective LLM calls. It only optimises the root agent's `instruction` field — sub-agent prompts are unchanged (with a warning).

### `GEPARootAgentPromptOptimizerConfig` fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `optimizer_model` | `str` | `"gemini-2.5-flash"` | LLM used for reflection and prompt mutation |
| `model_configuration` | `GenerateContentConfig` | Thinking budget 10240, `include_thoughts=True` | Enables extended reasoning for better prompt evolution |
| `max_metric_calls` | `int` | `100` | Total evaluations (train + validation). Lower = faster but less thorough. |
| `reflection_minibatch_size` | `int` | `3` | Number of examples per reflection step |
| `run_dir` | `str \| None` | `None` | Directory to save intermediate results. Useful for resuming long runs. |

### How GEPA works (from source walkthrough)

1. The optimizer creates a GEPA `adapter` that bridges between the ADK `Sampler` interface and GEPA's internal evaluation loop.
2. It constructs a `reflection_lm` function — a sync wrapper around the async LLM call — so GEPA can call it from its own thread executor.
3. `gepa.optimize()` runs in a thread pool (`run_in_executor`) to avoid blocking the asyncio event loop while GEPA executes its sync optimization loop.
4. The final result contains multiple candidate agents with their validation scores — you pick the best one.

### Example 1 — optimize a classification agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.gepa_root_agent_prompt_optimizer import (
    GEPARootAgentPromptOptimizer,
    GEPARootAgentPromptOptimizerConfig,
)
# Note: Sampler is the ADK bridge between your dataset and GEPA
# In practice, use AgentEvaluator or a custom Sampler implementation

async def main():
    initial_agent = LlmAgent(
        name="classifier",
        model="gemini-2.5-flash",
        instruction="Classify the following text.",  # starting prompt to optimize
    )

    config = GEPARootAgentPromptOptimizerConfig(
        optimizer_model="gemini-2.5-flash",
        max_metric_calls=50,        # budget for this optimization run
        reflection_minibatch_size=3,
        run_dir="/tmp/gepa_run",    # save progress here
    )

    optimizer = GEPARootAgentPromptOptimizer(config=config)

    # sampler bridges your eval dataset to GEPA
    # (see AgentOptimizer docs for Sampler construction)
    result = await optimizer.optimize(
        initial_agent=initial_agent,
        sampler=my_sampler,
    )

    # result.optimized_agents is sorted by overall_score (highest last)
    best = max(result.optimized_agents, key=lambda a: a.overall_score)
    print(f"Best instruction: {best.optimized_agent.instruction}")
    print(f"Validation score: {best.overall_score:.3f}")

    # Deploy the best agent
    from google.adk.runners import InMemoryRunner
    runner = InMemoryRunner(agent=best.optimized_agent, app_name="optimized")

asyncio.run(main())
```

### Example 2 — resuming a long optimization run

```python
from google.adk.optimization.gepa_root_agent_prompt_optimizer import (
    GEPARootAgentPromptOptimizer,
    GEPARootAgentPromptOptimizerConfig,
)

config = GEPARootAgentPromptOptimizerConfig(
    optimizer_model="gemini-2.5-pro",
    max_metric_calls=200,           # longer budget for production
    reflection_minibatch_size=5,
    run_dir="/checkpoints/gepa_v2", # GEPA will resume from here if it finds saved state
)
optimizer = GEPARootAgentPromptOptimizer(config=config)
```

### Comparison: `SimplePromptOptimizer` vs `GEPARootAgentPromptOptimizer`

| Aspect | `SimplePromptOptimizer` | `GEPARootAgentPromptOptimizer` |
|--------|------------------------|-------------------------------|
| Algorithm | Reflection + rewrite | GEPA evolutionary algorithm |
| Exploration | Single candidate per step | Multiple candidates per generation |
| Resumability | Not built-in | Via `run_dir` |
| Extra dependency | None | `pip install gepa` |
| Thinking budget | Optional | Enabled by default (10240 tokens) |
| Best for | Quick single-shot improvement | Systematic multi-iteration optimization |

---

## 10 · `EnvironmentSimulationPlugin` + `EnvironmentSimulationConfig` + `ToolSimulationConfig`

> ⚠️ **Experimental** — decorated with `@experimental(FeatureName.ENVIRONMENT_SIMULATION)`.

`EnvironmentSimulationPlugin` intercepts tool calls during agent runs and replaces them with **mock responses** — either LLM-generated based on a spec, from historical traces, or with injected errors. This enables deterministic, offline testing of agents without calling real external services.

### `EnvironmentSimulationConfig` fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `tool_simulation_configs` | `list[ToolSimulationConfig]` | required | One config per tool to mock. No duplicates allowed. |
| `simulation_model` | `str` | `"gemini-2.5-flash"` | LLM used internally to generate mock responses |
| `simulation_model_configuration` | `GenerateContentConfig` | Thinking budget 10240 | Controls internal LLM quality |
| `tracing` | `str \| None` | `None` | Prior agent run trace in JSON — provides historical context for mock generation |
| `environment_data` | `str \| None` | `None` | Domain data (e.g. minimal DB dump) for contextual mock responses |

### `ToolSimulationConfig` fields

| Field | Type | Notes |
|-------|------|-------|
| `tool_name` | `str` | Exact name of the tool to intercept |
| `mock_strategy_type` | `MockStrategy` | `TOOL_SPEC_MOCK` (LLM-generated from spec) or others |
| `injection_configs` | `list[InjectionConfig]` | Optional: inject specific errors or values before falling back to `mock_strategy_type` |

If `injection_configs` is empty, `mock_strategy_type` must be set (not `MOCK_STRATEGY_UNSPECIFIED`).

### Example 1 — mock two external tools for offline testing

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    ToolSimulationConfig,
    MockStrategy,
)
from google.adk.tools.environment_simulation.environment_simulation_plugin import (
    EnvironmentSimulationPlugin,
)


def get_weather(city: str) -> dict:
    """Get current weather for a city (calls real API in production)."""
    raise RuntimeError("Should not be called in test — mock should intercept")


def get_stock_price(ticker: str) -> dict:
    """Get current stock price (calls real API in production)."""
    raise RuntimeError("Should not be called in test")


sim_config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="get_weather",
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK,
        ),
        ToolSimulationConfig(
            tool_name="get_stock_price",
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK,
        ),
    ],
    simulation_model="gemini-2.5-flash",
    environment_data='{"cities": ["London", "Tokyo"], "tickers": ["AAPL", "GOOG"]}',
)

sim_plugin = EnvironmentSimulationPlugin(config=sim_config)

agent = LlmAgent(
    name="financial_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions about weather and stocks.",
    tools=[get_weather, get_stock_price],
)

app_with_mock = LlmAgent(
    name="financial_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions about weather and stocks.",
    tools=[get_weather, get_stock_price],
)


async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="financial_test",
        plugins=[sim_plugin],  # plugin intercepts tool calls
    )
    await runner.session_service.create_session(
        app_name="financial_test", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the weather in London and the price of AAPL?",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — inject errors at a specified probability

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    ToolSimulationConfig,
    InjectionConfig,
    InjectedError,
    MockStrategy,
)

# 30% of the time, inject a timeout error for the payment tool
payment_sim = ToolSimulationConfig(
    tool_name="process_payment",
    mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK,
    injection_configs=[
        InjectionConfig(
            injection_probability=0.3,
            injected_error=InjectedError(
                error_message="Payment gateway timeout",
                error_type="TIMEOUT",
            ),
        ),
    ],
)

config = EnvironmentSimulationConfig(
    tool_simulation_configs=[payment_sim],
    simulation_model="gemini-2.5-flash",
)
```

### Example 3 — seeding with a historical trace for realistic mocks

```python
import json
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, MockStrategy,
)

# Load a previous production trace for realistic mock data
with open("traces/production_run_2026_05_28.json") as f:
    historical_trace = f.read()

config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="search_inventory",
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK,
        ),
        ToolSimulationConfig(
            tool_name="create_order",
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK,
        ),
    ],
    simulation_model="gemini-2.5-flash",
    tracing=historical_trace,   # GEPA reads this to generate realistic responses
    environment_data='{"warehouse": "EU-WEST-1", "stock_level": "normal"}',
)
```

---

## Summary table

| # | Class | Key use case | Experimental? |
|---|-------|-------------|---------------|
| 1 | `ComputerUseTool` / `ComputerUseToolset` / `BaseComputer` | GUI/browser automation with coordinate normalisation | ⚠️ Yes |
| 2 | `OpenAPIToolset` / `RestApiTool` | Wrap any OpenAPI 3.x REST API as agent tools | No |
| 3 | `LlmEventSummarizer` / `BaseEventsSummarizer` | LLM-driven sliding-window event compaction | ⚠️ Yes |
| 4 | `Session` / `State` | Session and state primitives with scope prefixes and schema validation | No |
| 5 | `Event` / `EventActions` / `EventCompaction` | Event system — state deltas, transfers, auth requests, compaction markers | No |
| 6 | `ExampleTool` / `Example` / `BaseExampleProvider` / `VertexAiExampleStore` | Few-shot example injection — static list or dynamic retrieval | No |
| 7 | `GoogleSearchTool` / `UrlContextTool` / `GoogleSearchAgentTool` | Gemini-native search grounding and URL context fetching | No |
| 8 | `LlmBackedUserSimulator` / `UserPersona` / `UserBehavior` | LLM-powered user simulation for automated evaluation | ⚠️ Yes |
| 9 | `GEPARootAgentPromptOptimizer` | GEPA-based evolutionary prompt optimisation | ⚠️ Yes |
| 10 | `EnvironmentSimulationPlugin` / `EnvironmentSimulationConfig` | Mock external tools for deterministic offline testing | ⚠️ Yes |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-29 | google-adk 2.1.0 | Initial publication. All 10 class groups source-verified against installed `google-adk==2.1.0`. |

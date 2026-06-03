---
title: "Class deep dives — volume 11 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: GoogleApiToolset/GmailToolset/CalendarToolset/SheetsToolset (Google API Discovery integration), load_web_page (SSRF-protected web browsing), UiWidget (MCP iframe rendering metadata), _ToolNode (tool-as-workflow-node), SqliteSpanExporter (local OTEL tracing), RougeEvaluator/FinalResponseMatchV2 (response quality metrics), HallucinationsV1Evaluator (two-stage hallucination detection), function calling pipeline internals (handle_function_calls_async; parallel execution; client IDs), _ContentLlmRequestProcessor/_InstructionsLlmRequestProcessor (conversation history assembly)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 11"
  order: 70
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, June 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `GoogleApiToolset` + `GmailToolset` / `CalendarToolset` / `SheetsToolset` / `DocsToolset` / `SlidesToolset` / `YoutubeToolset` | `google.adk.tools.google_api_tool` | Stable |
| 2 | `load_web_page` | `google.adk.tools.load_web_page` | Stable |
| 3 | `UiWidget` | `google.adk.events.ui_widget` | Stable |
| 4 | `_ToolNode` | `google.adk.workflow._tool_node` | Stable |
| 5 | `SqliteSpanExporter` | `google.adk.telemetry.sqlite_span_exporter` | Stable |
| 6 | `RougeEvaluator` (FinalResponseMatchV1) | `google.adk.evaluation.final_response_match_v1` | Stable |
| 7 | `FinalResponseMatchV2Evaluator` | `google.adk.evaluation.final_response_match_v2` | Experimental |
| 8 | `HallucinationsV1Evaluator` | `google.adk.evaluation.hallucinations_v1` | Experimental |
| 9 | Function calling pipeline — `handle_function_calls_async` + client ID helpers | `google.adk.flows.llm_flows.functions` | Stable |
| 10 | `_ContentLlmRequestProcessor` + `_InstructionsLlmRequestProcessor` | `google.adk.flows.llm_flows.contents`, `.instructions` | Stable |

---

## 1 · `GoogleApiToolset` + pre-built Google API toolsets

**Source:** `google.adk.tools.google_api_tool.google_api_toolset`, `.google_api_toolsets`

`GoogleApiToolset` lets you surface any Google API described by the [Google API Discovery Service](https://developers.google.com/discovery) as a set of ADK tools. It fetches the OpenAPI spec for the requested API/version at runtime, converts it with `GoogleApiToOpenApiConverter`, wraps it in an `OpenAPIToolset`, and configures OIDC auth against Google's OAuth2 endpoints automatically.

Seven ready-made subclasses cover the most common Workspace and data APIs:

| Class | API name | Version |
|---|---|---|
| `GmailToolset` | `gmail` | v1 |
| `CalendarToolset` | `calendar` | v3 |
| `SheetsToolset` | `sheets` | v4 |
| `DocsToolset` | `docs` | v1 |
| `SlidesToolset` | `slides` | v1 |
| `YoutubeToolset` | `youtube` | v3 |
| `BigQueryToolset` (GA API wrapper) | `bigquery` | v2 |

### `GoogleApiToolset` constructor (source-verified)

```python
class GoogleApiToolset(BaseToolset):
    def __init__(
        self,
        api_name: str,
        api_version: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
        service_account: Optional[ServiceAccount] = None,
        tool_name_prefix: Optional[str] = None,
        *,
        additional_headers: Optional[Dict[str, str]] = None,
    ): ...
```

| Parameter | Purpose |
|---|---|
| `api_name` | Discovery API name (e.g. `"gmail"`, `"calendar"`) |
| `api_version` | API version string (e.g. `"v1"`, `"v3"`) |
| `client_id` / `client_secret` | OAuth2 credentials for the Google API |
| `tool_filter` | List of tool names to include, or a predicate `Callable[[str], bool]` |
| `service_account` | `ServiceAccount` for server-to-server auth (no user consent needed) |
| `tool_name_prefix` | Prefix prepended to every tool name to avoid clashes when multiple toolsets are used |
| `additional_headers` | Extra HTTP headers injected on every API call |

### Authentication patterns

**OAuth2 (user-delegated):** Requires a GCP OAuth2 client ID with the relevant API scopes configured.

```python
from google.adk.tools.google_api_tool.google_api_toolsets import GmailToolset

gmail = GmailToolset(
    client_id="123456-xxx.apps.googleusercontent.com",
    client_secret="GOCSPX-...",
    tool_filter=["gmail_users_messages_list", "gmail_users_messages_get"],
)
```

**Service account (server-to-server):** Best for Cloud Run / Agent Engine deployments where there is no interactive user.

```python
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.tools.google_api_tool.google_api_toolsets import SheetsToolset

sa = ServiceAccount(
    service_account_credential={
        "type": "service_account",
        "project_id": "my-project",
        "private_key_id": "...",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
        "client_email": "svc@my-project.iam.gserviceaccount.com",
    }
)
sheets = SheetsToolset(service_account=sa)
```

### Tool filtering

Without a filter `get_tools()` returns every operation in the API — often hundreds. Always filter in production.

```python
# List form — only these two operations are exposed
calendar = CalendarToolset(
    client_id="...",
    client_secret="...",
    tool_filter=["calendar_events_list", "calendar_events_insert"],
)

# Predicate form — expose only read operations
from google.adk.tools.base_toolset import ToolPredicate

read_only: ToolPredicate = lambda name: name.endswith("_list") or name.endswith("_get")

calendar = CalendarToolset(
    client_id="...",
    client_secret="...",
    tool_filter=read_only,
)
```

You can also update the filter after construction:

```python
calendar.set_tool_filter(["calendar_events_list"])
```

### Key methods (source-verified)

```python
async def get_tools(
    readonly_context: Optional[ReadonlyContext] = None
) -> list[GoogleApiTool]: ...

def configure_auth(client_id: str, client_secret: str) -> None: ...
def configure_sa_auth(service_account: ServiceAccount) -> None: ...
async def close() -> None: ...
```

### Full example — Gmail + Calendar assistant

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_api_tool.google_api_toolsets import (
    GmailToolset,
    CalendarToolset,
)

OAUTH_CLIENT_ID     = "123456-xxx.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "GOCSPX-..."

gmail_tools    = GmailToolset(
    client_id=OAUTH_CLIENT_ID,
    client_secret=OAUTH_CLIENT_SECRET,
    tool_filter=["gmail_users_messages_list", "gmail_users_messages_get",
                 "gmail_users_messages_send"],
    tool_name_prefix="gmail_",
)
calendar_tools = CalendarToolset(
    client_id=OAUTH_CLIENT_ID,
    client_secret=OAUTH_CLIENT_SECRET,
    tool_filter=["calendar_events_list", "calendar_events_insert",
                 "calendar_calendars_get"],
    tool_name_prefix="cal_",
)

agent = LlmAgent(
    name="workspace_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant with access to Gmail and Google Calendar. "
        "Help the user manage email and schedule meetings."
    ),
    tools=[gmail_tools, calendar_tools],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="workspace")
    await runner.session_service.create_session(
        app_name="workspace", user_id="alice", session_id="s1"
    )
    events = await runner.run_debug(
        "List my 5 most recent emails and any calendar events today.",
        user_id="alice",
        session_id="s1",
    )
    for e in events:
        if e.content and e.content.parts:
            print(e.content.parts[0].text)

asyncio.run(main())
```

### `tool_name_prefix` — avoiding name collisions

When two toolsets expose operations with the same short name, use a prefix:

```python
docs   = DocsToolset(..., tool_name_prefix="docs_")
slides = SlidesToolset(..., tool_name_prefix="slides_")
# Resulting tool names: "docs_documents_get", "slides_presentations_get", …
```

### Arbitrary Google API via `GoogleApiToolset` directly

Any API in the [Discovery catalogue](https://www.googleapis.com/discovery/v1/apis) works:

```python
from google.adk.tools.google_api_tool.google_api_toolset import GoogleApiToolset

# Google Drive API v3
drive = GoogleApiToolset(
    api_name="drive",
    api_version="v3",
    client_id="...",
    client_secret="...",
    tool_filter=["drive_files_list", "drive_files_get"],
)
```

---

## 2 · `load_web_page`

**Source:** `google.adk.tools.load_web_page`

`load_web_page(url: str) -> str` is a standalone tool function that fetches a URL and returns the visible text content. The implementation contains a layered SSRF (Server-Side Request Forgery) defence that blocks requests to private/internal network addresses before they are sent.

### Function signature

```python
def load_web_page(url: str) -> str:
    """Fetches the content in the url and returns the text in it."""
```

Returns the page text (BeautifulSoup `get_text`), filtered to lines with more than 3 words. Returns `"Failed to fetch url: <url>"` on any error.

### SSRF protection layers (source-verified)

The function applies four distinct checks before any network connection is made:

| Layer | What it blocks |
|---|---|
| Scheme check | Only `http` and `https` are allowed; `file://`, `ftp://`, etc. are rejected |
| Hostname check (`_is_blocked_hostname`) | `localhost` and any `*.localhost` subdomain |
| Literal IP check (`_is_blocked_address`) | Any IP that is not globally routable (`not address.is_global`) — covers 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8, link-local, etc. |
| DNS resolution check (`_resolve_direct_addresses`) | DNS names that resolve to non-global IPs |

Redirects are disabled (`allow_redirects=False`) to prevent SSRF via HTTP 301/302 chains.

### IP-pinning adapter (`_PinnedAddressAdapter`)

To prevent TOCTOU races between DNS resolution and the connection (a classic SSRF bypass), the implementation uses `_PinnedAddressAdapter`. After resolving and validating DNS, the adapter:

1. Rewrites the request URL to use the validated IP address directly
2. Preserves the original `Host` header (required for TLS SNI + virtual hosting)
3. Sets `assert_hostname` / `server_hostname` on the TLS pool to the original hostname, so TLS certificate validation still works against the domain name

```python
adapter = _PinnedAddressAdapter(
    rewritten_url="https://203.0.113.1/path",   # validated IP
    host_header="example.com",                   # original Host header
    hostname="example.com",                      # for TLS verification
)
```

### Proxy-aware behaviour

When an HTTP proxy is configured via environment variables (`HTTP_PROXY`, `HTTPS_PROXY`), the function delegates to the proxy instead of doing its own DNS resolution. It still blocks:
- Literal IP URLs that point to non-global addresses
- `localhost` / `*.localhost` hostnames

This avoids breaking proxy setups while retaining basic protections against the most obvious attacks.

### Using `load_web_page` as a tool

```python
from google.adk.agents import LlmAgent
from google.adk.tools.load_web_page import load_web_page

researcher = LlmAgent(
    name="web_researcher",
    model="gemini-2.5-flash",
    instruction=(
        "You are a research assistant. When asked about a topic, "
        "fetch the relevant web page and summarise the content."
    ),
    tools=[load_web_page],  # plain function — ADK wraps it as FunctionTool
)
```

### Wrapping into a named `FunctionTool`

If you want to control the tool description visible to the model:

```python
from google.adk.tools import FunctionTool
from google.adk.tools.load_web_page import load_web_page

web_tool = FunctionTool(
    func=load_web_page,
    # override the docstring the model sees
)
```

### Output post-processing

After fetching, BeautifulSoup converts the HTML to plain text and the result is filtered:

```python
return '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)
```

Lines with 3 words or fewer (navigation labels, one-word headings, etc.) are stripped. This meaningfully reduces token count for typical web pages.

---

## 3 · `UiWidget`

**Source:** `google.adk.events.ui_widget`

`UiWidget` is a Pydantic model that attaches rendering metadata to an event's `actions`. When the ADK web UI (or any compatible frontend) encounters a `ui_widget` on an event, it hands off rendering to the widget's declared `provider`.

### Class definition (source-verified)

```python
class UiWidget(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
    )

    id: str
    """Unique identifier of this UI widget."""

    provider: str
    """Renderer to use. Currently: 'mcp' for MCP App iframe (rendered with MCP AppBridge)."""

    payload: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific rendering data."""
```

All fields use camelCase aliases for JSON serialisation (`id`, `provider`, `payload` stay as-is since they are already camelCase-compatible).

### MCP iframe provider

The `'mcp'` provider renders an MCP App inside an iframe using the MCP AppBridge. The payload shape is:

```python
{
    "resource_uri": "ui://my-mcp-server/widget/dashboard",
    "tool": {
        "name": "render_dashboard",
        "description": "...",
        "inputSchema": { ... }
    },
    "tool_args": {
        "user_id": "alice",
        "date_range": "last_7_days"
    }
}
```

### Attaching a `UiWidget` to an event

`UiWidget` objects live on `EventActions.ui_widgets` (a `list[UiWidget]`). Attach them from inside a tool via `ToolContext.actions`:

```python
from google.adk.events.event_actions import EventActions
from google.adk.events.ui_widget import UiWidget
from google.adk.tools.tool_context import ToolContext


async def render_chart(chart_type: str, tool_context: ToolContext) -> dict:
    widget = UiWidget(
        id=f"chart-{chart_type}",
        provider="mcp",
        payload={
            "resource_uri": f"ui://charts-server/render/{chart_type}",
            "tool": {"name": "render_chart", "inputSchema": {}},
            "tool_args": {"chart_type": chart_type},
        },
    )
    # Append to the existing list (may already have widgets from parallel calls)
    tool_context.actions.ui_widgets = (
        tool_context.actions.ui_widgets or []
    ) + [widget]
    return {"status": "widget attached"}
```

### Reading `UiWidget` objects from events

When consuming the event stream programmatically:

```python
async for event in runner.run_async(new_message=msg, user_id=uid, session_id=sid):
    if event.actions and event.actions.ui_widgets:
        for widget in event.actions.ui_widgets:
            print(f"Render {widget.provider} widget id={widget.id}")
            print(f"  payload: {widget.payload}")
```

### JSON round-trip (camelCase aliases)

```python
import json
from google.adk.events.ui_widget import UiWidget

w = UiWidget(id="w1", provider="mcp", payload={"resource_uri": "ui://x"})

# Serialize with alias (camelCase) — this is what gets sent over the wire
data = json.loads(w.model_dump_json(by_alias=True))
# → {"id": "w1", "provider": "mcp", "payload": {"resource_uri": "ui://x"}}

# Deserialize
w2 = UiWidget.model_validate(data)
assert w2.id == "w1"
```

---

## 4 · `_ToolNode`

**Source:** `google.adk.workflow._tool_node`

`_ToolNode` is a `BaseNode` subclass that wraps any `BaseTool` as a first-class workflow node. This lets you compose tool calls alongside `LlmAgent` nodes and function nodes in a `Workflow` graph — with full retry, timeout, and HITL support.

### Constructor (source-verified)

```python
class _ToolNode(BaseNode):
    def __init__(
        self,
        *,
        tool: BaseTool,
        name: str | None = None,
        retry_config: RetryConfig | None = None,
        timeout: float | None = None,
    ): ...
```

| Parameter | Default | Purpose |
|---|---|---|
| `tool` | required | Any `BaseTool` instance (FunctionTool, MCP tool, etc.) |
| `name` | `tool.name` | Node name in the graph; defaults to the tool's own name |
| `retry_config` | `None` | Per-node retry policy (`RetryConfig`) |
| `timeout` | `None` | Per-node wall-clock timeout in seconds |

`rerun_on_resume` is always `False` — a `_ToolNode` does not re-execute after a HITL interrupt resumes.

### `_run_impl` — execution (source-verified)

```python
async def _run_impl(self, *, ctx: Context, node_input: Any) -> AsyncGenerator[Any, None]:
    tool_context = ToolContext(
        invocation_context=ctx.get_invocation_context(),
        function_call_id=str(uuid.uuid4()),
    )

    args = node_input
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        raise TypeError(
            "The input to ToolNode must be a dictionary of tool arguments or"
            f" None, but got {type(args)}."
        )

    response = await self.tool.run_async(args=args, tool_context=tool_context)
    state_delta = (
        dict(tool_context.actions.state_delta)
        if tool_context.actions.state_delta
        else None
    )
    if response is not None:
        yield Event(output=response, state=state_delta)
    elif state_delta:
        yield Event(state=state_delta)
```

Key observations from the source:
- **Input must be `dict | None`**: the node expects its `node_input` to be a dictionary of keyword arguments matching the tool's parameter schema, or `None` (treated as `{}`).
- **`state_delta` propagation**: any state mutations made by the tool via `tool_context.actions.state_delta` are carried on the yielded `Event.state`, merging into the workflow's session state.
- **No output → no event**: if the tool returns `None` and makes no state mutations, no event is emitted.

### Using `_ToolNode` in a `Workflow`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.workflow import Workflow, node, START
from google.adk.workflow._tool_node import _ToolNode
from google.adk.workflow._retry_config import RetryConfig


def fetch_weather(city: str) -> dict:
    """Returns mock weather data for a city."""
    return {"city": city, "temp_c": 22, "condition": "sunny"}

weather_tool = FunctionTool(func=fetch_weather)
weather_node = _ToolNode(
    tool=weather_tool,
    name="fetch_weather",
    retry_config=RetryConfig(max_retries=2, initial_delay=1.0),
    timeout=10.0,
)

summarizer = LlmAgent(
    name="summarizer",
    model="gemini-2.5-flash",
    instruction="Summarise the weather data from node_input in one sentence.",
    mode="single_turn",
)

pipeline = Workflow(
    name="weather_pipeline",
    edges=[(START, weather_node, summarizer)],
)

async def main():
    app = App(name="weather_app", agent=pipeline)
    runner = InMemoryRunner(agent=pipeline, app_name="weather_app")
    await runner.session_service.create_session(
        app_name="weather_app", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        {"city": "London"},  # dict passed as node_input to the first node
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Composing multiple `_ToolNode` instances

```python
from google.adk.tools.load_web_page import load_web_page
from google.adk.tools import FunctionTool
from google.adk.workflow._tool_node import _ToolNode

fetch_node  = _ToolNode(tool=FunctionTool(func=load_web_page), name="fetch_page")
parse_agent = LlmAgent(name="parser", model="gemini-2.5-flash",
                        instruction="Extract the main article headline from the page text.")

pipeline = Workflow(
    name="news_pipeline",
    edges=[(START, fetch_node, parse_agent)],
)
```

---

## 5 · `SqliteSpanExporter`

**Source:** `google.adk.telemetry.sqlite_span_exporter`

`SqliteSpanExporter` is an OpenTelemetry `SpanExporter` that persists spans to a local SQLite database. It is the backend used by `adk web` to make trace data available after process restart. For production, use Google Cloud Trace; use `SqliteSpanExporter` for local development, unit tests, and CI.

### Constructor

```python
class SqliteSpanExporter(SpanExporter):
    def __init__(self, *, db_path: str): ...
```

`db_path` is the path to the SQLite file. It is created automatically on first use. Schema creation is idempotent (`CREATE TABLE IF NOT EXISTS`).

### Database schema (source-verified)

```sql
CREATE TABLE IF NOT EXISTS spans (
  span_id              TEXT PRIMARY KEY,
  trace_id             TEXT NOT NULL,
  parent_span_id       TEXT,
  name                 TEXT NOT NULL,
  start_time_unix_nano INTEGER,
  end_time_unix_nano   INTEGER,
  session_id           TEXT,   -- from gcp.vertex.agent.session_id or gen_ai.conversation.id
  invocation_id        TEXT,   -- from gcp.vertex.agent.invocation_id
  attributes_json      TEXT
);

CREATE INDEX IF NOT EXISTS spans_session_id_idx ON spans(session_id);
CREATE INDEX IF NOT EXISTS spans_trace_id_idx   ON spans(trace_id);
```

### Wiring into the ADK telemetry pipeline

The exporter plugs into the standard OTEL `TracerProvider`:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter

exporter = SqliteSpanExporter(db_path="/tmp/adk_traces.db")
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Now run your agent — spans are persisted to /tmp/adk_traces.db
```

### Querying spans for a session

```python
exporter = SqliteSpanExporter(db_path="/tmp/adk_traces.db")
spans = exporter.get_all_spans_for_session("my-session-id")

for span in spans:
    print(f"{span.name:40s}  {span.start_time} → {span.end_time}")
    for k, v in (span.attributes or {}).items():
        print(f"  {k} = {v}")
```

`get_all_spans_for_session` first finds all `trace_id`s associated with the session, then returns every span in those traces — including parent spans that may lack the `session_id` attribute. This preserves the full call tree.

### Key methods

| Method | Purpose |
|---|---|
| `export(spans)` | Batch-inserts spans; uses `INSERT OR REPLACE` so re-exports are idempotent |
| `get_all_spans_for_session(session_id)` | Returns all `ReadableSpan` objects for a session (full trace trees) |
| `shutdown()` | Closes the SQLite connection |
| `force_flush(timeout_millis=30000)` | No-op — always returns `True` (SQLite writes are synchronous) |

### Thread safety

All database operations are serialised with a `threading.Lock()`. The connection is opened lazily and reused; it is opened with `check_same_thread=False` since the lock provides safety across threads.

### Integration test pattern

```python
import asyncio
import os
import tempfile
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


async def test_trace_capture():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    exporter = SqliteSpanExporter(db_path=db_path)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    agent  = LlmAgent(name="test", model="gemini-2.5-flash",
                       instruction="Answer in one word.")
    runner = InMemoryRunner(agent=agent, app_name="test_app")
    await runner.session_service.create_session(
        app_name="test_app", user_id="u1", session_id="s1"
    )
    await runner.run_debug("Say hello", user_id="u1", session_id="s1")

    spans = exporter.get_all_spans_for_session("s1")
    assert any(s.name.startswith("invoke_agent") for s in spans)
    os.unlink(db_path)
```

---

## 6 · `RougeEvaluator` — `FinalResponseMatchV1`

**Source:** `google.adk.evaluation.final_response_match_v1`

`RougeEvaluator` is the v1 text-similarity metric for evaluating final agent responses. It computes the **ROUGE-1 F-measure** (unigram overlap) between the agent's actual final response and the golden expected response.

### Class definition (source-verified)

```python
class RougeEvaluator(Evaluator):
    def __init__(self, eval_metric: EvalMetric): ...

    def evaluate_invocations(
        self,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]] = None,
        conversation_scenario: Optional[ConversationScenario] = None,
    ) -> EvaluationResult: ...
```

`expected_invocations` is **required** and must be the same length as `actual_invocations`. Each pair is scored independently, then averaged.

### Metric: ROUGE-1 F-measure

ROUGE-1 counts unigram (single-word) overlaps:

| Score component | Formula |
|---|---|
| Precision | (shared words) / (words in candidate) |
| Recall | (shared words) / (words in reference) |
| **F-measure (used)** | 2 × P × R / (P + R) |

Stemming is enabled (`use_stemmer=True`), so "running" and "run" count as the same token.

Score range: **[0.0, 1.0]**. The `EvalMetric.threshold` field sets the pass/fail boundary.

### Registering `RougeEvaluator` in an eval config

```python
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator

metric = EvalMetric(
    metric_name="response_match_v1",
    threshold=0.7,   # score ≥ 0.7 → PASSED
)
evaluator = RougeEvaluator(eval_metric=metric)
```

### End-to-end evaluation example

```python
import asyncio
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator
from google.genai import types


def make_content(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


actual = [
    Invocation(
        user_content=types.Content(parts=[types.Part(text="What is the capital of France?")]),
        final_response=make_content("The capital of France is Paris."),
    ),
    Invocation(
        user_content=types.Content(parts=[types.Part(text="Name the largest planet.")]),
        final_response=make_content("Jupiter is the largest planet in the solar system."),
    ),
]
expected = [
    Invocation(
        user_content=types.Content(parts=[types.Part(text="What is the capital of France?")]),
        final_response=make_content("Paris is the capital of France."),
    ),
    Invocation(
        user_content=types.Content(parts=[types.Part(text="Name the largest planet.")]),
        final_response=make_content("Jupiter is the largest planet."),
    ),
]

metric = EvalMetric(metric_name="response_match_v1", threshold=0.6)
result = RougeEvaluator(metric).evaluate_invocations(actual, expected)

print(f"Overall: {result.overall_score:.3f} → {result.overall_eval_status}")
for r in result.per_invocation_results:
    print(f"  turn: {r.score:.3f} → {r.eval_status}")
```

### When to use ROUGE-1 vs LLM-as-judge

| Criterion | ROUGE-1 (v1) | LLM-as-judge (v2) |
|---|---|---|
| Speed | Fast (no LLM call) | Slower (one LLM call per turn) |
| Cost | Free | Pay per token |
| Sensitivity | Lexical overlap only | Semantic equivalence |
| Best for | Short, factual answers; regression checks | Open-ended or paraphrased responses |

---

## 7 · `FinalResponseMatchV2Evaluator` `@experimental`

**Source:** `google.adk.evaluation.final_response_match_v2`

`FinalResponseMatchV2Evaluator` replaces the ROUGE-1 lexical comparison with an **LLM-as-judge** approach. A judge model reads the user prompt, the agent's response, and the golden response, then labels the agent response as `valid` or `invalid`. With multiple samples per invocation the evaluator uses **majority voting** to decide the final label.

### Class hierarchy

```
Evaluator
  └── LlmAsJudge
        └── FinalResponseMatchV2Evaluator
```

### Prompt structure (source-verified)

The judge is given:

```
User prompt:       <user's message>
Agent response:    <what the agent said>
Golden response:   <expected answer>

Is the agent response valid given the user prompt and golden response?
Reply with: {"is_the_agent_response_valid": "valid"} or {"is_the_agent_response_valid": "invalid"}
```

### Majority voting aggregation (source-verified)

```python
def aggregate_per_invocation_samples(
    self,
    per_invocation_samples: list[PerInvocationResult],
) -> PerInvocationResult:
    valid_count   = sum(1 for r in per_invocation_samples if r.score == 1.0)
    invalid_count = len(per_invocation_samples) - valid_count
    # Ties (equal counts) → invalid
    if valid_count > invalid_count:
        return next(r for r in per_invocation_samples if r.score == 1.0)
    return next(r for r in per_invocation_samples if r.score == 0.0)
```

Ties break toward `invalid` — a conservative choice.

### Usage

```python
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.final_response_match_v2 import FinalResponseMatchV2Evaluator

metric = EvalMetric(
    metric_name="response_match_v2",
    threshold=0.8,   # fraction of invocations that must be "valid"
    judge_model="gemini-2.5-flash",
    num_samples=3,   # samples per invocation for majority vote
)
evaluator = FinalResponseMatchV2Evaluator(eval_metric=metric)

result = await evaluator.evaluate_invocations(actual, expected)
print(f"Pass rate: {result.overall_score:.2%}")
```

> **`@experimental`**: The class is decorated with `@experimental`. Its interface may change between minor versions.

---

## 8 · `HallucinationsV1Evaluator` `@experimental`

**Source:** `google.adk.evaluation.hallucinations_v1`

`HallucinationsV1Evaluator` detects factual hallucinations in agent responses using a **two-stage LLM pipeline**: first the response is *segmented* into individual sentences, then each sentence is *classified* as supported, contradicted, or not applicable given the agent's conversation context (tool outputs, instructions, history).

### Two-stage pipeline

```
agent response text
        │
        ▼
  [Stage 1: Segmentation]
  LLM splits text into <sentence>…</sentence> tags
        │
        ▼
  [Stage 2: Validation]
  LLM classifies each sentence:
    - "supported"       → score contribution +1
    - "not_applicable"  → score contribution +1
    - "not_supported"   → score contribution  0
        │
        ▼
  accuracy = supported_or_na / total_sentences
```

### Class definition (source-verified)

```python
class HallucinationsV1Evaluator(Evaluator):
    def __init__(self, eval_metric: EvalMetric) -> None: ...

    async def evaluate_invocations(
        self,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]] = None,
        conversation_scenario: Optional[ConversationScenario] = None,
    ) -> EvaluationResult: ...
```

`expected_invocations` is **not required** — the evaluator compares the agent's response against its own conversation context, not a golden answer.

### Context construction

For each agent turn, the evaluator builds a `context` string that includes:

1. Developer (system) instructions
2. The user's message for that turn
3. Tool definitions (names + descriptions)
4. For every step in the turn: tool call arguments **and** the tool response

The judge model then checks whether each sentence in the NL response is grounded in this context.

### Usage

```python
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.hallucinations_v1 import HallucinationsV1Evaluator

metric = EvalMetric(
    metric_name="hallucinations_v1",
    threshold=0.9,               # 90 % of sentences must be grounded
    judge_model="gemini-2.5-flash",
    include_intermediate_responses=True,  # also check mid-turn NL responses
)
evaluator = HallucinationsV1Evaluator(eval_metric=metric)

result = await evaluator.evaluate_invocations(actual_invocations)
for r in result.per_invocation_results:
    print(f"Turn grounding: {r.score:.2%} → {r.eval_status}")
```

### Comparison: hallucination vs response-match evaluators

| Evaluator | What it measures | Needs golden? |
|---|---|---|
| `RougeEvaluator` (v1) | Lexical overlap with golden answer | Yes |
| `FinalResponseMatchV2Evaluator` | Semantic correctness vs golden | Yes |
| `HallucinationsV1Evaluator` | Factual grounding in conversation context | No |

> **`@experimental`**: Subject to interface changes between minor versions.

---

## 9 · Function calling pipeline — `handle_function_calls_async` + helpers

**Source:** `google.adk.flows.llm_flows.functions`

The `functions` module is the core of ADK's tool execution engine. Understanding it lets you reason about parallel tool calls, client-generated IDs, auth requests, and confirmation flows.

### Key constants (source-verified)

```python
AF_FUNCTION_CALL_ID_PREFIX          = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME      = 'adk_request_credential'
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = 'adk_request_confirmation'
REQUEST_INPUT_FUNCTION_CALL_NAME    = 'adk_request_input'
```

`adk_request_credential`, `adk_request_confirmation`, and `adk_request_input` are synthetic function names that ADK injects into the event stream when a tool needs user auth, confirmation, or input. They never correspond to real tools.

### Client-generated function call IDs

When the LLM returns function calls that lack IDs (some models or older APIs), ADK assigns them:

```python
def generate_client_function_call_id() -> str:
    return f'{AF_FUNCTION_CALL_ID_PREFIX}{platform_uuid.new_uuid()}'
    # → e.g. "adk-550e8400-e29b-41d4-a716-446655440000"

def populate_client_function_call_id(model_response_event: Event) -> None:
    for function_call in model_response_event.get_function_calls():
        if not function_call.id:
            function_call.id = generate_client_function_call_id()
```

Before sending conversation history back to the LLM, ADK strips the `adk-` prefix IDs so the model never sees internal tracking IDs:

```python
def remove_client_function_call_id(content: Optional[types.Content]) -> None:
    # Removes IDs starting with 'adk-' from function_call and function_response parts
    ...
```

### Parallel execution — `handle_function_calls_async` (source-verified)

When the LLM emits multiple function calls in a single response (parallel tool use), ADK executes them concurrently:

```python
async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
    function_calls = function_call_event.get_function_calls()
    # ...filters applied...
    tasks = [
        asyncio.create_task(
            _execute_single_function_call_async(
                invocation_context, function_call, tools_dict, agent,
                tool_confirmation_dict[function_call.id] if tool_confirmation_dict else None,
            )
        )
        for function_call in filtered_calls
    ]
    function_response_events = await asyncio.gather(*tasks)
    # On exception: cancel all remaining tasks before re-raising
    ...
    return merge_parallel_function_response_events(function_response_events)
```

Key behaviours:
- **All calls run concurrently** via `asyncio.gather`
- **Cancellation on failure**: if any task raises, all other tasks are cancelled before the exception propagates
- **Merged into a single event**: `merge_parallel_function_response_events` deep-merges the `EventActions` dicts (state deltas, auth requests, etc.) from all response events

### Sync vs async tool execution

ADK detects whether a tool's `run` method is synchronous and routes accordingly:

```python
def _is_sync_tool(tool: BaseTool) -> bool:
    # Checks if run_async is the default Base impl — if so, tool has a sync run()
    ...

async def _call_tool_in_thread_pool(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    max_workers: int = 4,
) -> Any:
    if _is_sync_tool(tool):
        # Run sync tool in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return await loop.run_in_executor(pool, tool.run, args, tool_context)
    else:
        # Already async — call directly
        return await tool.run_async(args=args, tool_context=tool_context)
```

In Live API mode the thread pool is taken from `RunConfig.tool_thread_pool_config` to avoid creating a new pool per call.

### Auth request events

When a tool calls `tool_context.request_credential(auth_config)`, the runner intercepts the response and creates a synthetic `adk_request_credential` event that pauses execution until the user provides credentials:

```python
def build_auth_request_event(
    invocation_context: InvocationContext,
    auth_requests: Dict[str, AuthConfig],
    *,
    author: Optional[str] = None,
    role: Optional[str] = None,
) -> Event: ...
```

The event contains a function call named `adk_request_credential` whose `args` encode the `AuthConfig` for each tool that needs credentials. The frontend renders a login prompt; on completion the user sends back a `adk_request_credential` function response and execution resumes.

### Merging parallel responses

```python
def merge_parallel_function_response_events(
    events: list[Event],
) -> Event:
    # Merges EventActions fields via deep_merge_dicts
    # Aggregates ui_widgets lists
    # Returns a single Event with all function_response parts combined
    ...
```

`deep_merge_dicts` recursively merges nested dicts (e.g. two `state_delta` dicts from concurrent tool calls):

```python
def deep_merge_dicts(d1: dict, d2: dict) -> dict:
    merged = dict(d1)
    for k, v in d2.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged
```

---

## 10 · `_ContentLlmRequestProcessor` + `_InstructionsLlmRequestProcessor`

**Sources:** `google.adk.flows.llm_flows.contents`, `.instructions`

These two processors are the heart of how ADK assembles the `LlmRequest.contents` list that gets sent to the model on every turn. Understanding them explains why certain messages appear or disappear from the LLM's context window.

### `_InstructionsLlmRequestProcessor`

Handles the `system_instruction` and dynamic instruction injection.

```python
class _InstructionsLlmRequestProcessor(BaseLlmRequestProcessor):
    async def run_async(
        self,
        invocation_context: InvocationContext,
        llm_request: LlmRequest,
    ) -> AsyncGenerator[Event, None]: ...
```

**What it does:**

1. Calls `agent.canonical_instruction(readonly_context)` which handles:
   - Static `instruction` strings
   - Callable instructions `instruction=lambda ctx: f"Today is {date.today()}"`
   - State variable injection: `"{user_name}"` → replaced with `ctx.state["user_name"]`

2. Appends to `llm_request.config.system_instruction` (the static part always becomes system instruction)

3. Dynamic instructions (callables) are appended as a user-role `Content` just before the last user message

**State variable injection:**

```python
# In the agent's instruction string, {var_name} is replaced from session state
agent = LlmAgent(
    name="greeter",
    model="gemini-2.5-flash",
    instruction="You are helping {user_name}. Today is {today}.",
)
# Before each LLM call, {user_name} and {today} are resolved from session state
# runner.run_debug sets state["user_name"] = "Alice" before running
```

**Dynamic instruction callable:**

```python
from datetime import date
from google.adk.agents.readonly_context import ReadonlyContext

def build_instruction(ctx: ReadonlyContext) -> str:
    name = ctx.state.get("user_name", "there")
    return f"You are helping {name}. Today is {date.today().isoformat()}."

agent = LlmAgent(
    name="greeter",
    model="gemini-2.5-flash",
    instruction=build_instruction,  # called fresh every turn
)
```

### `_ContentLlmRequestProcessor`

Builds the full conversation history from the session event log.

```python
class _ContentLlmRequestProcessor(BaseLlmRequestProcessor):
    async def run_async(
        self,
        invocation_context: InvocationContext,
        llm_request: LlmRequest,
    ) -> AsyncGenerator[Event, None]: ...
```

**Content selection modes** (controlled by `LlmAgent.include_contents`):

| `include_contents` | What gets included |
|---|---|
| `"default"` (default) | Full session history since start, respecting branch + isolation scope |
| `"none"` | Only the current turn's user message (stateless) |

**Event filtering pipeline (source-verified):**

The processor applies these filters in order to each event in the session history:

1. **Isolation scope filter** (`_should_include_event_in_context`): only events within the current `isolation_scope` are visible; sub-agent calls with their own scope are hidden
2. **Branch filter**: only events on the current graph branch are included
3. **Empty content filter** (`_contains_empty_content`): events with no text, inline data, file data, function calls, or function responses are dropped
4. **Compaction filter** (`_process_compaction_events`): events within a compaction range are replaced by the compaction summary event
5. **Invisible part filter** (`_is_part_invisible`): thought/reasoning parts are stripped before sending to the model

**Other-agent message presentation:**

When an agent in a `Workflow` receives output from a sibling agent, the content is reformatted as a user message with attribution:

```python
# Original: model role, author="weather_agent", text="It is 22°C in London."
# Presented as: user role, text='[weather_agent] said: "It is 22°C in London."'
```

This prevents cross-agent output from being confused with the model's own previous responses.

**Async function response reordering:**

In long-running tool scenarios (HITL), function responses may arrive many turns after the original function call. The processor re-splices them adjacent to their matching call before presenting the history to the model:

```python
# Raw history:
# turn 3: function_call(get_approval, id=adk-123)
# turn 4: user: "other message"
# turn 5: function_response(get_approval, id=adk-123)

# Presented to LLM as:
# turn 3: function_call(get_approval, id=adk-123)
# turn 3b: function_response(get_approval, id=adk-123)   ← moved here
# turn 4: user: "other message"
```

### Putting it together — what the LLM sees per turn

```
system_instruction:
  [static instruction string from agent.instruction]

contents:
  [session history, filtered + compacted + re-ordered]
  ...
  [dynamic instruction injected as user Content (if callable instruction)]
  [current user message]
```

The processors run in a defined order inside `AutoFlow.run_async`:

```
_InstructionsLlmRequestProcessor  →  adds system_instruction + dynamic instruction
_ContentLlmRequestProcessor        →  adds filtered conversation history
```

### Bypassing state injection

If you don't want ADK to replace `{var}` placeholders in your instructions (e.g. because your instruction contains literal curly braces for JSON examples), set:

```python
agent = LlmAgent(
    ...
    bypass_state_injection=True,  # {var} placeholders are left as-is
)
```

This flag is checked in `_InstructionsLlmRequestProcessor` before calling `instructions_utils.inject_session_state()`.

---

## Summary table

| # | Class | Module | What it enables |
|---|---|---|---|
| 1 | `GoogleApiToolset` + pre-built toolsets | `tools.google_api_tool` | Any Google API (Gmail, Calendar, Sheets, …) as ADK tools via Discovery API |
| 2 | `load_web_page` | `tools.load_web_page` | SSRF-protected web page fetching; IP pinning; no-redirect |
| 3 | `UiWidget` | `events.ui_widget` | Attach MCP iframe / custom UI widgets to any event |
| 4 | `_ToolNode` | `workflow._tool_node` | Use any `BaseTool` as a retryable, timeout-gated workflow node |
| 5 | `SqliteSpanExporter` | `telemetry.sqlite_span_exporter` | Local OpenTelemetry trace persistence for dev + testing |
| 6 | `RougeEvaluator` | `evaluation.final_response_match_v1` | Fast lexical response-match scoring (ROUGE-1 F-measure) |
| 7 | `FinalResponseMatchV2Evaluator` | `evaluation.final_response_match_v2` | LLM-as-judge response match with majority voting |
| 8 | `HallucinationsV1Evaluator` | `evaluation.hallucinations_v1` | Two-stage segmentation + grounding check; no golden needed |
| 9 | Function calling pipeline | `flows.llm_flows.functions` | Parallel tool calls; client IDs; auth/confirmation events |
| 10 | `_ContentLlmRequestProcessor` + `_InstructionsLlmRequestProcessor` | `flows.llm_flows.contents`, `.instructions` | How conversation history + instructions reach the LLM |

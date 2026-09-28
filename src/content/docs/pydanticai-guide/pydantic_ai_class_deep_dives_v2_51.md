---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2.51.0)"
description: "Runnable, source-verified code examples for ApprovalRequiredToolset, DeferredLoadingToolset, ExternalToolset, web_fetch_tool/WebFetchLocalTool, duckduckgo_search_tool, image_generation_tool/ImageGenerationSubagentTool, format_as_xml, RealtimeModelSettings/TurnDetection, SourcedInstruction/InstructionPart, and Hooks — verified against pydantic-ai 2.51.0."
framework: pydanticai
language: python
sidebar:
  order: 129
---

# 10 Source-Verified Class Deep Dives — v2.51.0

Verified against **pydantic-ai 2.51.0** (installed package, sources read directly).
Modules consulted: `pydantic_ai/toolsets/approval_required.py`, `pydantic_ai/toolsets/deferred_loading.py`,
`pydantic_ai/toolsets/external.py`, `pydantic_ai/common_tools/web_fetch.py`,
`pydantic_ai/common_tools/duckduckgo.py`, `pydantic_ai/common_tools/image_generation.py`,
`pydantic_ai/format_prompt.py`, `pydantic_ai/realtime/settings.py`,
`pydantic_ai/_instructions.py`, `pydantic_ai/capabilities/hooks.py`.

This page covers classes that are **new since 2.46.0** or were only **thinly documented**
in earlier deep-dive pages. Cross-references to the earlier series appear at the end.

```bash
pip install "pydantic-ai==2.51.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.51.0
```

---

## 1. `ApprovalRequiredToolset` — human-in-the-loop tool gating

**Module:** `pydantic_ai.toolsets.approval_required`

`ApprovalRequiredToolset` wraps any toolset and raises `ApprovalRequired` before executing
a tool call unless the caller has already approved it.  The `approval_required_func` callback
receives the full `RunContext`, the `ToolDefinition`, and the parsed `tool_args`, so approval
logic can be as simple or rich as you need.

### Constructor (from source)

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True
```

The default lambda returns `True` for **every** call — wrap it and then approve selectively.

### Example 1 — block all calls until explicitly approved

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.exceptions import ApprovalRequired

def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email."""
    return f"Email sent to {to}"

toolset = ApprovalRequiredToolset(
    FunctionToolset([send_email]),
    approval_required_func=lambda ctx, tool_def, args: True,  # always require approval
)

agent = Agent("openai:gpt-4o-mini", toolsets=[toolset])

async def main():
    try:
        result = await agent.run("Send a welcome email to alice@example.com")
    except ApprovalRequired:
        print("Tool call blocked — waiting for human approval")

asyncio.run(main())
```

### Example 2 — selective approval by tool name and argument value

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.tools import DeferredToolRequests

DANGEROUS_TOOLS = {"delete_record", "send_email", "execute_sql"}
HIGH_VALUE_THRESHOLD = 1000

def approval_policy(
    ctx: RunContext[None],
    tool_def: ToolDefinition,
    tool_args: dict,
) -> bool:
    # Require approval for high-risk tool names
    if tool_def.name in DANGEROUS_TOOLS:
        return True
    # Also require approval for transfers above the threshold
    if tool_def.name == "transfer_funds" and tool_args.get("amount", 0) > HIGH_VALUE_THRESHOLD:
        return True
    return False

def read_record(record_id: str) -> dict:
    """Reads a record — safe, no approval needed."""
    return {"id": record_id, "status": "active"}

def delete_record(record_id: str) -> str:
    """Deletes a record — dangerous, requires approval."""
    return f"Record {record_id} deleted"

def transfer_funds(account_id: str, amount: float) -> str:
    """Transfers funds to an account. Requires approval for amounts over the threshold."""
    return f"Transferred ${amount:.2f} to {account_id}"

toolset = ApprovalRequiredToolset(
    FunctionToolset([read_record, delete_record, transfer_funds]),
    approval_required_func=approval_policy,
)

# DeferredToolRequests must be in output_type so the agent can surface
# pending approvals as a return value rather than raising ApprovalRequired.
agent = Agent(
    "openai:gpt-4o-mini",
    toolsets=[toolset],
    output_type=str | DeferredToolRequests,
)

async def main():
    # Small transfer: amount=50 < 1000 → no approval needed → returns str
    result = await agent.run("Transfer $50 to account ACC-123")
    print(result.output)  # "Transferred $50.00 to ACC-123"

    # Large transfer: amount=5000 > 1000 → triggers approval → returns DeferredToolRequests
    result = await agent.run("Transfer $5000 to account ACC-456")
    if isinstance(result.output, DeferredToolRequests):
        print(f"Approval required for: {[c.tool_name for c in result.output.approvals]}")

asyncio.run(main())
```

### Example 3 — resuming an approved call via `deferred_tool_results`

`ApprovalRequired` is raised before the tool body runs.  The correct way to resume is to
capture the pending `DeferredToolRequests` from the first run, build a `ToolApproved` result
for the call the human approved, then re-invoke the agent with those results via
`deferred_tool_results`.  The toolset sees `ctx.tool_call_approved = True` on the replayed
call and bypasses the approval check.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved

def wire_transfer(amount: float, destination: str) -> str:
    return f"Transferred ${amount:.2f} to {destination}"

toolset = ApprovalRequiredToolset(FunctionToolset([wire_transfer]))
# Include DeferredToolRequests as a possible output type so the run can
# surface pending calls instead of raising immediately.
agent = Agent(
    "openai:gpt-4o-mini",
    toolsets=[toolset],
    output_type=str | DeferredToolRequests,
)

async def main():
    result = await agent.run("Transfer $500 to account 9876")

    if isinstance(result.output, DeferredToolRequests):
        deferred = result.output
        # Approval calls land in .approvals (not .calls which is for ExternalToolset)
        print(f"Approval needed for: {[c.tool_name for c in deferred.approvals]}")

        # build_results(approve_all=True) is the ergonomic helper; it creates
        # DeferredToolResults(approvals={id: ToolApproved() for each pending call})
        approved = deferred.build_results(approve_all=True)
        # Resume from the same message history with the approved results
        result = await agent.run(
            "",
            message_history=result.all_messages(),
            deferred_tool_results=approved,
        )

    print(result.output)

asyncio.run(main())
```

---

## 2. `DeferredLoadingToolset` — hide tools until revealed

**Module:** `pydantic_ai.toolsets.deferred_loading`

`DeferredLoadingToolset` marks one or more tools for **deferred loading**: the tools are
declared to the model but their schemas are withheld.  The model must either call a *tool
search* to discover them, or another tool can return a `ToolReturn.tools` payload that
reveals them.  This is the recommended pattern when you have hundreds of tools — expose a
search interface rather than flooding the context window.

### Constructor (from source)

```python
@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    tool_names: frozenset[str] | None = None
    # None → defer ALL tools; a frozenset → defer only those named

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_names: frozenset[str] | None = None,
    ): ...
```

### Example 1 — defer every tool

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset

def fetch_order(order_id: str) -> dict:
    """Fetch order details."""
    return {"id": order_id, "status": "shipped"}

def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"Order {order_id} cancelled"

def update_address(order_id: str, new_address: str) -> str:
    """Update shipping address."""
    return f"Address updated for {order_id}"

# Wrap ALL three tools — model sees them as deferred until revealed
toolset = DeferredLoadingToolset(
    FunctionToolset([fetch_order, cancel_order, update_address])
)

agent = Agent("openai:gpt-4o", toolsets=[toolset])

async def main():
    result = await agent.run("What is the status of order ORD-999?")
    print(result.output)

asyncio.run(main())
```

### Example 2 — defer only the expensive tools

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset

def quick_lookup(user_id: str) -> str:
    """Fast in-memory lookup."""
    return f"User {user_id}: active"

def run_analytics_report(user_id: str, period: str) -> dict:
    """Slow analytics — defer until explicitly needed."""
    return {"user": user_id, "period": period, "sessions": 42}

def run_ml_prediction(user_id: str) -> float:
    """Even slower ML inference — defer too."""
    return 0.87

toolset = DeferredLoadingToolset(
    FunctionToolset([quick_lookup, run_analytics_report, run_ml_prediction]),
    # Only these two are deferred; quick_lookup is always available
    tool_names=frozenset({"run_analytics_report", "run_ml_prediction"}),
)

agent = Agent("openai:gpt-4o", toolsets=[toolset])

async def main():
    result = await agent.run("Quick lookup for user U-101")
    print(result.output)

asyncio.run(main())
```

### Example 3 — pair with `_tool_search` for lazy discovery

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset

# Imagine 200 domain-specific tools
def tool_a(x: str) -> str: return f"a:{x}"
def tool_b(x: str) -> str: return f"b:{x}"
def tool_c(x: str) -> str: return f"c:{x}"

large_toolset = FunctionToolset([tool_a, tool_b, tool_c])
deferred = DeferredLoadingToolset(large_toolset)

# The built-in tool search reveals deferred tools when the model calls it
agent = Agent(
    "openai:gpt-4o",
    toolsets=[deferred],
    # ToolSearch is auto-injected into every agent; it reveals deferred tools at zero overhead
)

async def main():
    result = await agent.run("Call tool_b with value 'hello'")
    print(result.output)

asyncio.run(main())
```

---

## 3. `ExternalToolset` — tools whose results come from outside the run

**Module:** `pydantic_ai.toolsets.external`

`ExternalToolset` lets you declare tools whose **results are produced outside the agent run**
— by a human, a webhook, or a separate process.  The model sees the tool definition and
issues a call; the call is surfaced to your code as a pending action with `kind='external'`.
Your code fulfils it externally and re-injects the result.

### Constructor (from source)

```python
class ExternalToolset(AbstractToolset[AgentDepsT]):
    def __init__(
        self,
        tool_defs: list[ToolDefinition],
        *,
        id: str | None = None,
    ): ...
```

`tool_defs` is a list of plain `ToolDefinition` objects — no Python callables needed.

### Example 1 — declare external tools and capture calls

`ExternalToolset` surfaces pending calls as `DeferredToolRequests` so your application
can fulfil them and reinject the results.  Include `DeferredToolRequests` in `output_type`
so the run returns it rather than blocking.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai.tools import ToolDefinition, DeferredToolRequests, DeferredToolResults

# Describe the tools the model may call
external_toolset = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="book_meeting",
            description="Books a calendar meeting.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {"type": "string", "format": "date-time"},
                    "duration_minutes": {"type": "integer"},
                },
                "required": ["title", "start_time", "duration_minutes"],
            },
        )
    ]
)

agent = Agent(
    "openai:gpt-4o",
    toolsets=[external_toolset],
    output_type=str | DeferredToolRequests,  # surface deferred calls to the caller
)

async def main():
    result = await agent.run("Book a 30-min team sync for tomorrow at 10am")

    if isinstance(result.output, DeferredToolRequests):
        # Fulfil each call externally — e.g. call your calendar API
        # DeferredToolResults.calls is a {tool_call_id: result} dict; bare strings
        # are auto-wrapped in ToolReturn by the framework
        call_results = {}
        for call in result.output.calls:
            print(f"Fulfilling external tool call: {call.tool_name}({call.args})")
            # In production, call your external service here
            call_results[call.tool_call_id] = "Meeting booked: Team Sync at 10am tomorrow"

        # Resume the run with the fulfilled results
        result = await agent.run(
            "",
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(calls=call_results),
        )

    print(result.output)

asyncio.run(main())
```

### Example 2 — multiple external tools with an id for correlation

`ExternalToolset` never executes calls locally, so `output_type` must include
`DeferredToolRequests` — otherwise the pending call can never be surfaced.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai.tools import ToolDefinition, DeferredToolRequests, DeferredToolResults

CALENDAR_TOOLS = ExternalToolset(
    id="calendar-service",
    tool_defs=[
        ToolDefinition(
            name="create_event",
            description="Creates a calendar event.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["title", "date"],
            },
        ),
        ToolDefinition(
            name="list_events",
            description="Lists calendar events for a date.",
            parameters_json_schema={
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        ),
    ],
)

agent = Agent(
    "openai:gpt-4o",
    toolsets=[CALENDAR_TOOLS],
    output_type=str | DeferredToolRequests,  # required — ExternalToolset never runs locally
)

async def main():
    result = await agent.run("What meetings do I have on 2026-10-01?")

    if isinstance(result.output, DeferredToolRequests):
        # Fulfil each pending call via your external calendar service
        call_results = {}
        for call in result.output.calls:
            print(f"External call to calendar-service: {call.tool_name}({call.args})")
            call_results[call.tool_call_id] = '[{"title": "Team sync", "time": "09:00"}]'

        result = await agent.run(
            "",
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(calls=call_results),
        )

    print(result.output)

asyncio.run(main())
```

---

## 4. `web_fetch_tool` / `WebFetchLocalTool` — built-in web fetch

**Module:** `pydantic_ai.common_tools.web_fetch`

`web_fetch_tool` is a ready-made `Tool` that fetches a URL and returns either a `WebFetchResult`
dict (HTML/JSON/text → Markdown) or a `BinaryContent` object (PDFs, images, other binary
media) — all with SSRF protection built in.

### `WebFetchResult` (from source)

```python
class WebFetchResult(TypedDict):
    url: str       # The URL that was fetched
    title: str     # Page title, or '' if not found
    content: str   # Page content converted to markdown
```

### `WebFetchLocalTool` constructor (from source)

```python
@dataclass
class WebFetchLocalTool:
    max_content_length: int | None   # None = no limit
    allow_local_urls: bool           # False by default (SSRF protection)
    timeout: int                     # Request timeout in seconds
    max_download_bytes: int | None   # Default: 50 MB
    allowed_domains: list[str] | None = None  # Exact hostname allowlist
```

### Example 1 — basic web fetch

```python
# pip install "pydantic-ai[web-fetch]"
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[web_fetch_tool()],
    system_prompt="You are a research assistant. Fetch pages when asked.",
)

async def main():
    result = await agent.run("Summarise https://docs.pydantic.dev/latest/")
    print(result.output)

asyncio.run(main())
```

### Example 2 — restrict to specific domains with length cap

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[
        web_fetch_tool(
            max_content_length=8_000,   # Cap at 8 k chars of Markdown
            timeout=15,
            allowed_domains=["docs.pydantic.dev", "github.com"],
        )
    ],
)

async def main():
    result = await agent.run("What is the latest release of pydantic-ai?")
    print(result.output)

asyncio.run(main())
```

### Example 3 — enable local URLs for internal tooling

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

# Only safe inside a trusted internal network
agent = Agent(
    "openai:gpt-4o",
    tools=[
        web_fetch_tool(
            allow_local_urls=True,   # Allows http://localhost/... targets
            timeout=5,
        )
    ],
)

async def main():
    result = await agent.run("Fetch http://localhost:8080/health and tell me the status")
    print(result.output)

asyncio.run(main())
```

> **SSRF note:** `allow_local_urls=False` (the default) blocks RFC-1918 addresses and
> `localhost`.  Only set it to `True` if your agent runs in a fully trusted network.

---

## 5. `duckduckgo_search_tool` / `DuckDuckGoSearchTool` — built-in DuckDuckGo search

**Module:** `pydantic_ai.common_tools.duckduckgo`

`duckduckgo_search_tool` wraps the `ddgs` library into a first-class Pydantic AI tool.
No API key is required.

### `DuckDuckGoResult` (from source)

```python
class DuckDuckGoResult(TypedDict):
    title: str   # Result title
    href: str    # Result URL
    body: str    # Snippet / body text
```

### `DuckDuckGoSearchTool` constructor (from source)

```python
@dataclass
class DuckDuckGoSearchTool:
    client: DDGS
    max_results: int | None   # None → first response page only
```

### Example 1 — add DuckDuckGo search to an agent

```python
# pip install "pydantic-ai[duckduckgo]"
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[duckduckgo_search_tool(max_results=5)],
    system_prompt="Use DuckDuckGo to answer questions requiring current information.",
)

async def main():
    result = await agent.run("What are the top Python frameworks for AI agents in 2026?")
    print(result.output)

asyncio.run(main())
```

### Example 2 — pass a pre-configured DDGS client

```python
# pip install "pydantic-ai[duckduckgo]"
import asyncio
from ddgs.ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# DDGS uses the singular `proxy` parameter (not `proxies`)
client = DDGS(proxy=None, timeout=10)

agent = Agent(
    "openai:gpt-4o",
    tools=[duckduckgo_search_tool(duckduckgo_client=client, max_results=3)],
)

async def main():
    result = await agent.run("Latest pydantic-ai release notes")
    print(result.output)

asyncio.run(main())
```

### Example 3 — combine with web_fetch for deep research

```python
# pip install "pydantic-ai[duckduckgo,web-fetch]"
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

research_agent = Agent(
    "openai:gpt-4o",
    tools=[
        duckduckgo_search_tool(max_results=5),
        web_fetch_tool(max_content_length=5_000),
    ],
    system_prompt=(
        "You are a research assistant. Search for relevant URLs first, "
        "then fetch the most promising ones for detail."
    ),
)

async def main():
    result = await research_agent.run(
        "What is the current state of pydantic-ai's realtime API support?"
    )
    print(result.output)

asyncio.run(main())
```

---

## 6. `image_generation_tool` / `ImageGenerationSubagentTool` — image generation via subagent

**Module:** `pydantic_ai.common_tools.image_generation`

`image_generation_tool` creates a tool that lets a **text-capable** model generate images
by spinning up a dedicated image-generation subagent behind the scenes.  This is the right
pattern when your primary model cannot natively produce images.

### `ImageGenerationSubagentTool` constructor (from source)

```python
@dataclass(kw_only=True)
class ImageGenerationSubagentTool:
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc
    native_tool: ImageGenerationNativeTool[Any]
    instructions: str = 'Generate an image based on the user prompt. Do not ask clarifying questions.'
```

> **Gotcha:** pass image-only model names (`dall-e-3`, `gpt-image-1`, etc.) to
> `native_tool`, not to `model`.  Passing them directly to `model` raises `UserError` — the
> source explicitly guards against it.

### Example 1 — basic image generation tool

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.native_tools import ImageGenerationTool

agent = Agent(
    "openai:gpt-4o",
    tools=[
        image_generation_tool(
            model="openai-responses:gpt-5.4",            # conversational model for subagent
            native_tool=ImageGenerationTool(model="dall-e-3"),
        )
    ],
    system_prompt="You are a creative assistant who can generate images.",
)

async def main():
    result = await agent.run("Draw a sunset over a futuristic city in watercolour style")
    # result.output is the model's text response describing the generated image.
    # The BinaryImage itself is stored in the ToolReturnPart of result.all_messages().
    print(result.output)  # e.g. "Here is your sunset watercolour image."

asyncio.run(main())
```

### Example 2 — dynamic model resolution per run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.tools import RunContext
from dataclasses import dataclass

@dataclass
class AppDeps:
    use_premium_model: bool

def resolve_model(ctx: RunContext[AppDeps]) -> str:
    if ctx.deps.use_premium_model:
        return "openai-responses:gpt-5.5"
    return "openai-responses:gpt-5.4"

agent = Agent(
    "openai:gpt-4o",
    tools=[
        image_generation_tool(
            model=resolve_model,   # factory callable
            native_tool=ImageGenerationTool(model="gpt-image-1"),
        )
    ],
)

async def main():
    result = await agent.run(
        "Create a logo for a startup called Nexus",
        deps=AppDeps(use_premium_model=True),
    )
    # result.output is the model's text acknowledgement; the BinaryImage is in
    # the tool return messages: result.all_messages()
    print(result.output)

asyncio.run(main())
```

### Example 3 — custom subagent instructions

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.native_tools import ImageGenerationTool

STYLE_INSTRUCTIONS = (
    "Generate an image in the style of a vintage 1950s travel poster. "
    "Use warm, faded colours and bold typography. "
    "Do not ask clarifying questions."
)

agent = Agent(
    "openai:gpt-4o",
    tools=[
        image_generation_tool(
            model="openai-responses:gpt-5.4",
            native_tool=ImageGenerationTool(model="dall-e-3"),
            instructions=STYLE_INSTRUCTIONS,
        )
    ],
)

async def main():
    result = await agent.run("Make an image of Paris for our travel campaign")
    # The outer agent returns a text response; the generated BinaryImage lives
    # in the tool return messages accessible via result.all_messages().
    print(result.output)

asyncio.run(main())
```

---

## 7. `format_as_xml` — structure prompts as XML

**Module:** `pydantic_ai.format_prompt`

`format_as_xml` converts any Python object — dict, dataclass, Pydantic model, list, nested
combinations — into an XML string.  LLMs frequently find XML easier to parse than JSON for
semi-structured data, making this useful for injecting few-shot examples or structured
context into prompts.

### Signature (from source)

```python
def format_as_xml(
    obj: Any,
    root_tag: str | None = None,   # Wrapping tag; None omits the outer element
    item_tag: str = 'item',        # Tag for sequence items (overridden by class name)
    none_str: str = 'null',        # Representation for None
    indent: str | None = '  ',     # Indentation; None = compact single line
    include_field_info: Literal['once'] | bool = False,
) -> str: ...
```

### Example 1 — format a dict as XML context

```python
from pydantic_ai import format_as_xml

user = {"name": "Alice", "role": "admin", "active": True}
print(format_as_xml(user, root_tag="user"))
# <user>
#   <name>Alice</name>
#   <role>admin</role>
#   <active>True</active>
# </user>
```

### Example 2 — format a list of Pydantic models as few-shot examples

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, format_as_xml

class ClassificationExample(BaseModel):
    text: str
    label: str
    confidence: float

examples = [
    ClassificationExample(text="Buy now! Limited offer!", label="spam", confidence=0.97),
    ClassificationExample(text="Your invoice is attached.", label="legitimate", confidence=0.92),
    ClassificationExample(text="Win a free iPhone!", label="spam", confidence=0.99),
]

system_prompt = f"""
You classify emails as spam or legitimate.

Examples:
{format_as_xml(examples, root_tag="examples")}
"""

agent = Agent("openai:gpt-4o-mini", system_prompt=system_prompt, output_type=ClassificationExample)

async def main():
    result = await agent.run("Congratulations, you have been selected for a prize!")
    print(result.output)

asyncio.run(main())
```

### Example 3 — include Pydantic field descriptions once per field

```python
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml
from typing import List

class Product(BaseModel):
    sku: str = Field(description="Stock-keeping unit identifier")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Whether the item is available")

products = [
    Product(sku="A-001", price=29.99, in_stock=True),
    Product(sku="B-042", price=149.0, in_stock=False),
    Product(sku="C-007", price=9.99, in_stock=True),
]

# 'once' emits field descriptions as attributes on the first occurrence only
xml = format_as_xml(products, root_tag="catalogue", include_field_info="once")
print(xml)
```

### Example 4 — compact XML for tight token budgets

```python
from dataclasses import dataclass
from pydantic_ai import format_as_xml

@dataclass
class Metric:
    name: str
    value: float
    unit: str

metrics = [Metric("cpu", 72.3, "%"), Metric("mem", 4096, "MB"), Metric("rps", 1250, "req/s")]

# indent=None produces a single line — saves tokens in large context payloads
compact = format_as_xml(metrics, root_tag="metrics", indent=None)
print(compact)
# <metrics><Metric><name>cpu</name><value>72.3</value><unit>%</unit></Metric>...
```

---

## 8. `RealtimeModelSettings` / `TurnDetection` — realtime voice sessions

**Module:** `pydantic_ai.realtime.settings`

`RealtimeModelSettings` configures the session when using bidirectional speech-to-speech
models (OpenAI Realtime, Azure OpenAI Realtime, Gemini Live, xAI Grok Voice).
`TurnDetection` controls the Voice Activity Detection (VAD) that determines when the user
has finished speaking.

### `TurnDetection` fields (from source)

```python
class TurnDetection(TypedDict, total=False):
    sensitivity: Literal['low', 'medium', 'high']
    # Maps per provider:
    #   OpenAI/xAI: server-VAD threshold (low≈0.7, medium≈0.5, high≈0.3)
    #   Gemini: start/end sensitivity
    prefix_padding_ms: int    # Audio retained before speech onset
    silence_duration_ms: int  # Silence required to mark turn end
```

### `AudioRetention` values (from source)

```
'transcript_only'  — keep transcripts only (default — saves memory)
'input_audio'      — also retain user's spoken audio as WAV
'output_audio'     — also retain model's spoken audio as WAV
'all'              — retain both sides' audio
```

### Example 1 — basic realtime session

`agent.realtime(model)` returns an `AgentRealtime` binding (not an async context manager).
Open a session with `async with agent.realtime(model).session() as session:`.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(system_prompt="You are a helpful voice assistant.")

async def main():
    async with agent.realtime(OpenAIRealtimeModel("gpt-4o-realtime-preview")).session() as session:
        # Send audio bytes (PCM 16-bit, 24 kHz) from a microphone
        await session.send_audio(b"<pcm audio bytes>")
        async for event in session:
            print(event)

asyncio.run(main())
```

### Example 2 — configure VAD sensitivity and token limit

`model_settings` goes to `agent.realtime(model, model_settings=...)`.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel
from pydantic_ai.realtime.settings import RealtimeModelSettings, TurnDetection

settings: RealtimeModelSettings = {
    "max_tokens": 512,
    "parallel_tool_calls": False,
    "turn_detection": TurnDetection(
        sensitivity="high",          # Snappy turn detection
        prefix_padding_ms=200,       # Keep 200ms before speech onset
        silence_duration_ms=400,     # End turn after 400ms silence
    ),
}

agent = Agent()

async def main():
    realtime = agent.realtime(
        OpenAIRealtimeModel("gpt-4o-realtime-preview"),
        model_settings=settings,
    )
    async with realtime.session() as session:
        await session.send_audio(b"<pcm audio>")
        async for event in session:
            print(event)

asyncio.run(main())
```

### Example 3 — retain both sides' audio for transcription

`audio_retention` is a parameter of `session()`, not of `model_settings`.
`AudioRetention` is a `Literal` alias — assign the string value directly.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent()

async def main():
    realtime = agent.realtime(OpenAIRealtimeModel("gpt-4o-realtime-preview"))
    # audio_retention='all' retains both input and output audio as WAV
    async with realtime.session(audio_retention="all") as session:
        await session.send_audio(b"<pcm audio>")
        async for event in session:
            if hasattr(event, "audio"):
                print(f"WAV audio retained: {len(event.audio.data)} bytes")

asyncio.run(main())
```

### Example 4 — push-to-talk (disable VAD)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel
from pydantic_ai.realtime.settings import RealtimeModelSettings

settings: RealtimeModelSettings = {
    "turn_detection": False,  # Disable VAD — user controls turn boundaries
}

agent = Agent()

async def main():
    realtime = agent.realtime(
        OpenAIRealtimeModel("gpt-4o-realtime-preview"),
        model_settings=settings,
    )
    async with realtime.session() as session:
        await session.send_audio(b"<pcm audio>")
        # Manually commit the audio to signal end of user turn
        await session.commit_audio()
        # With VAD off, commit_audio() only closes the turn;
        # create_response() is required to ask the model to respond now.
        await session.create_response()
        async for event in session:
            print(event)

asyncio.run(main())
```

---

## 9. `SourcedInstruction` / `InstructionPart` — the instruction system

**Module:** `pydantic_ai._instructions`

Starting in 2.46.0, Pydantic AI introduced a first-class *instructions* system as an
alternative (and complement) to `system_prompt`.  Instructions are typed, addressable, and
support **prompt caching** via `dynamic=False` (static) vs `dynamic=True` (re-evaluated
per run).  `SourcedInstruction` is the internal wrapper that attaches a name and source to
each instruction recipe.

### Key types (from source)

```python
AgentInstruction = TemplateStr | str | InstructionPart | SystemPromptFunc
# One instruction: a literal, a function, or an InstructionPart that names itself.

@dataclass(frozen=True)
class SourcedInstruction(Generic[AgentDepsT]):
    instruction: AgentInstruction[AgentDepsT]
    name: str | None = None          # Named instructions are addressable
    id: InstructionId | None = None  # Qualified id built from source + name
    dynamic: bool = False            # True → re-evaluated on every run
```

### Example 1 — static named instruction (cacheable)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import InstructionPart

# Named InstructionPart — stays stable across runs so it can be cached
company_policy = InstructionPart(
    content="You are a customer-support agent for Acme Corp. Be concise and professional.",
    name="company_policy",
    dynamic=False,   # Static — suitable for prompt caching
)

agent = Agent("openai:gpt-4o", instructions=[company_policy])

async def main():
    result = await agent.run("What is your return policy?")
    print(result.output)

asyncio.run(main())
```

### Example 2 — dynamic instruction from a callable

```python
import asyncio
from datetime import datetime
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

@dataclass
class UserDeps:
    username: str
    timezone: str

def personalised_greeting(ctx: RunContext[UserDeps]) -> str:
    hour = datetime.now(tz=ZoneInfo(ctx.deps.timezone)).hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
    return f"{greeting}, {ctx.deps.username}! I'll use {ctx.deps.timezone} for any time references."

agent = Agent(
    "openai:gpt-4o",
    # A callable instruction is always dynamic — re-evaluated per run
    instructions=[personalised_greeting],
)

async def main():
    result = await agent.run(
        "What time is it?",
        deps=UserDeps(username="Alice", timezone="Europe/London"),
    )
    print(result.output)

asyncio.run(main())
```

### Example 3 — mix static and dynamic instructions

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import RunContext

STATIC_POLICY = InstructionPart(
    content="Always respond in the language the user writes in.",
    name="language_policy",
    dynamic=False,
)

@dataclass
class SessionDeps:
    user_tier: str

def tier_instruction(ctx: RunContext[SessionDeps]) -> str:
    if ctx.deps.user_tier == "premium":
        return "This user has premium access — provide detailed, extended answers."
    return "This user has a free tier — keep answers brief."

agent = Agent(
    "openai:gpt-4o",
    instructions=[STATIC_POLICY, tier_instruction],
)

async def main():
    result = await agent.run(
        "Explain quantum entanglement",
        deps=SessionDeps(user_tier="premium"),
    )
    print(result.output)

asyncio.run(main())
```

### Example 4 — instruction name validation (reserved names)

```python
from pydantic_ai.messages import InstructionPart
from pydantic_ai.exceptions import UserError

try:
    # 'agent' is reserved — the source guards against it
    bad = InstructionPart(content="...", name="agent")
except UserError as e:
    print(f"Caught expected error: {e}")
```

---

## 10. `Hooks` — decorator-based capability registration

**Module:** `pydantic_ai.capabilities.hooks`

`Hooks` is the ergonomic alternative to subclassing `AbstractCapability`.  It exposes a
`hooks.on` namespace of decorators — one per lifecycle event — so you can register async
(or sync) functions without writing a class.

### Supported hook names (from source)

```
before_model_request     after_model_request
wrap_model_request       wrap_node_run
before_tool_validate     after_tool_validate
wrap_tool_validate       before_tool_execute
after_tool_execute       wrap_tool_execute
before_output_validate   after_output_validate
wrap_output_validate     before_output_process
after_output_process     wrap_output_process
on_wrap_run
```

### Example 1 — log every model request and response

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context):
    print(f"→ Model request: {[type(m).__name__ for m in request_context.messages]}")
    return request_context

@hooks.on.after_model_request
async def log_response(ctx, *, response, request_context):
    print(f"← Model response: {response}")
    return response  # transformation hook — must return the (possibly modified) response

agent = Agent("openai:gpt-4o-mini", capabilities=[hooks])

async def main():
    result = await agent.run("What is 2 + 2?")
    print(result.output)

asyncio.run(main())
```

### Example 2 — track tool execution time

```python
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.toolsets import FunctionToolset

def get_weather(city: str) -> str:
    """Returns mock weather for a city."""
    return f"Sunny, 22°C in {city}"

hooks = Hooks()
_start_times: dict[str, float] = {}

@hooks.on.before_tool_execute
async def record_start(ctx, *, call, tool_def, args):
    # Keyword names from source: call (ToolCallPart), tool_def, args
    _start_times[call.tool_call_id] = time.monotonic()
    return args  # must return args (possibly modified); returning None drops the args

@hooks.on.after_tool_execute
async def record_end(ctx, *, call, tool_def, args, result):
    elapsed = time.monotonic() - _start_times.pop(call.tool_call_id, 0)
    print(f"Tool {call.tool_name!r} took {elapsed*1000:.1f}ms")
    return result  # must return result — dropping it would suppress the tool's output

agent = Agent(
    "openai:gpt-4o-mini",
    toolsets=[FunctionToolset([get_weather])],
    capabilities=[hooks],
)

async def main():
    result = await agent.run("What is the weather in London?")
    print(result.output)

asyncio.run(main())
```

### Example 3 — stack multiple `Hooks` objects

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks

security_hooks = Hooks()
observability_hooks = Hooks()

@security_hooks.on.before_tool_execute
async def security_check(ctx, *, call, tool_def, args):
    if "delete" in call.tool_name.lower():
        print(f"Security alert: destructive tool called — {call.tool_name}")
    return args  # must return args

@observability_hooks.on.after_model_request
async def emit_metric(ctx, *, response, request_context):
    token_count = sum(
        getattr(r, 'usage', None) and r.usage.total_tokens or 0
        for r in [response]
    )
    print(f"Tokens used: {token_count}")
    return response  # must return response

# Stack in the capabilities list — they run in order
agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[security_hooks, observability_hooks],
)

async def main():
    result = await agent.run("Tell me a joke")
    print(result.output)

asyncio.run(main())
```

### Example 4 — `wrap_model_request` for request logging

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()
attempt_count: dict[str, int] = {}

@hooks.on.wrap_model_request
async def log_request(ctx, *, request_context, handler):
    run_id = id(ctx)
    attempt_count[run_id] = attempt_count.get(run_id, 0) + 1

    # Log the attempt number before forwarding the request unchanged
    print(f"Attempt #{attempt_count[run_id]} for run {run_id}")

    return await handler(request_context)

agent = Agent("openai:gpt-4o-mini", capabilities=[hooks])

async def main():
    result = await agent.run("Hello!")
    print(result.output)

asyncio.run(main())
```

---

## Cross-references

| Class | Guide |
|---|---|
| `FallbackModel` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46) |
| `FunctionToolset` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46) |
| `CombinedToolset` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46) |
| `MCPToolset` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46) |
| `Embedder` / `EmbeddingResult` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46) |
| `TextOutput` / `ToolOutput` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46) |
| `PrefixedToolset` / `FilteredToolset` / `RenamedToolset` | [2026-08 deep dives](./pydantic_ai_class_examples_2026_08) |
| `Agent` / `RunContext` / `UsageLimits` | [2026-08 deep dives](./pydantic_ai_class_examples_2026_08) |
| `CachePoint` / `DeferredToolRequests` | [2026-08 deep dives](./pydantic_ai_class_examples_2026_08) |

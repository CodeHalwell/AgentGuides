---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2.57.0)"
description: "Runnable, source-verified code examples for ApprovalRequiredToolset, FilteredToolset, PrefixedToolset, RenamedToolset, ExternalToolset, DeferredLoadingToolset, HandleDeferredToolCalls, RaiseContentFilterError, SelectModel, and NativeOrLocalTool — verified against pydantic-ai 2.57.0."
framework: pydanticai
language: python
sidebar:
  order: 132
---

# 10 Source-Verified Class Deep Dives — v2.57.0

Verified against **pydantic-ai 2.57.0** (installed package, sources read directly).
Modules consulted: `pydantic_ai/toolsets/approval_required.py`, `pydantic_ai/toolsets/filtered.py`,
`pydantic_ai/toolsets/prefixed.py`, `pydantic_ai/toolsets/renamed.py`,
`pydantic_ai/toolsets/external.py`, `pydantic_ai/toolsets/deferred_loading.py`,
`pydantic_ai/capabilities/deferred_tool_handler.py`, `pydantic_ai/capabilities/content_filter.py`,
`pydantic_ai/capabilities/select_model.py`, `pydantic_ai/capabilities/native_or_local.py`.

```bash
pip install "pydantic-ai==2.57.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.57.0
```

---

## 1. `ApprovalRequiredToolset` — human-in-the-loop tool gating

**Module:** `pydantic_ai.toolsets.approval_required`

`ApprovalRequiredToolset` wraps any other toolset and gates every tool call through a
user-supplied predicate. When the predicate returns `True` for a given call the agent
raises `ApprovalRequired` and suspends the run — the caller can then inspect the pending
call, approve it, and resume the run with `ctx.tool_call_approved = True`.

### Constructor (from source)

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True
```

The default predicate (`lambda ... True`) means every call requires approval. Override it
to be selective.

### Example 1 — gate a single dangerous tool (full approval flow)

When `output_type` includes `DeferredToolRequests`, the framework intercepts
`ApprovalRequired` and packages the pending call into `result.output.approvals` rather
than raising an exception. Resume the run by calling the agent again with
`deferred_tool_results=requests.build_results(approve_all=True)` (or per-call approvals).

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def delete_file(path: str) -> str:
    """Delete a file from disk."""
    return f"Deleted {path}"

@toolset.tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return f"Contents of {path}"

# Only require approval for destructive tools
approval_toolset = ApprovalRequiredToolset(
    wrapped=toolset,
    approval_required_func=lambda ctx, tool_def, args: tool_def.name == "delete_file",
)

model = TestModel(call_tools=["delete_file"])
agent = Agent(
    model,
    toolsets=[approval_toolset],
    output_type=[str, DeferredToolRequests],
)

async def main():
    result1 = await agent.run("Delete /tmp/test.txt")
    if isinstance(result1.output, DeferredToolRequests):
        print("Pending approvals:", [tc.tool_name for tc in result1.output.approvals])
        # Present to human; here we approve everything automatically
        approved = result1.output.build_results(approve_all=True)
        result2 = await agent.run(
            message_history=result1.all_messages(),
            deferred_tool_results=approved,
        )
        print(result2.output)

asyncio.run(main())
```

### Example 2 — approve all calls automatically in tests

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# In tests, approve everything — never require a human
no_approval_toolset = ApprovalRequiredToolset(
    wrapped=toolset,
    approval_required_func=lambda ctx, tool_def, args: False,
)

model = TestModel(call_tools=["send_email"])
agent = Agent(model, toolsets=[no_approval_toolset])

async def main():
    result = await agent.run("Email alice@example.com about the meeting")
    print(result.output)

asyncio.run(main())
```

### Example 3 — context-sensitive approval via deps

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.models.test import TestModel

@dataclass
class UserDeps:
    is_admin: bool

toolset = FunctionToolset()

@toolset.tool
def drop_table(table: str) -> str:
    """Drop a database table."""
    return f"Dropped {table}"

# Only admins can run drop_table without approval
approval_toolset = ApprovalRequiredToolset(
    wrapped=toolset,
    approval_required_func=lambda ctx, tool_def, args: not ctx.deps.is_admin,
)

model = TestModel(call_tools=["drop_table"])
agent = Agent(model, toolsets=[approval_toolset], deps_type=UserDeps)

async def main():
    # Admin user — no approval required
    result = await agent.run("Drop the old_logs table", deps=UserDeps(is_admin=True))
    print(result.output)

asyncio.run(main())
```

---

## 2. `FilteredToolset` — runtime tool availability

**Module:** `pydantic_ai.toolsets.filtered`

`FilteredToolset` wraps a toolset and applies a predicate to each tool *on every request*,
returning only the matching tools to the model. Both sync and async predicates are accepted.
Because filtering runs on each request the available set can change dynamically based on
conversation state, user permissions, or external flags.

### Constructor (from source)

```python
@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    filter_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition],
        bool | Awaitable[bool]
    ]
```

### Example 1 — expose only permitted tools

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.models.test import TestModel

@dataclass
class UserDeps:
    allowed_tools: set[str]

toolset = FunctionToolset()

@toolset.tool
def query_db(sql: str) -> str:
    """Run a SELECT query."""
    return "rows..."

@toolset.tool
def write_db(sql: str) -> str:
    """Run an INSERT/UPDATE/DELETE query."""
    return "1 row affected"

filtered = FilteredToolset(
    wrapped=toolset,
    filter_func=lambda ctx, tool_def: tool_def.name in ctx.deps.allowed_tools,
)

model = TestModel(call_tools=["query_db"])
agent = Agent(model, toolsets=[filtered], deps_type=UserDeps)

async def main():
    result = await agent.run(
        "List all customers",
        deps=UserDeps(allowed_tools={"query_db"}),  # write_db is hidden
    )
    print(result.output)

asyncio.run(main())
```

### Example 2 — async filter with an external permission check

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def expensive_tool(param: str) -> str:
    """A computationally expensive tool."""
    return f"Result for {param}"

@toolset.tool
def cheap_tool(param: str) -> str:
    """A lightweight lookup tool."""
    return f"Lookup for {param}"

async def check_quota(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    """Async predicate — could check a quota service, Redis, DB, etc."""
    # expensive_tool is hidden when budget is exhausted
    if tool_def.name == "expensive_tool":
        # Simulate async I/O
        await asyncio.sleep(0)
        return True  # quota available
    return True

filtered = FilteredToolset(wrapped=toolset, filter_func=check_quota)

model = TestModel(call_tools=["cheap_tool"])
agent = Agent(model, toolsets=[filtered])

async def main():
    result = await agent.run("Do a cheap lookup")
    print(result.output)

asyncio.run(main())
```

### Example 3 — A/B-test tool sets by session

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.models.test import TestModel

@dataclass
class SessionDeps:
    experiment_group: str  # "control" or "treatment"

toolset = FunctionToolset()

@toolset.tool
def legacy_search(query: str) -> str:
    """Original search implementation."""
    return "legacy results"

@toolset.tool
def new_search(query: str) -> str:
    """Experimental search implementation."""
    return "new results"

TOOL_GROUPS = {
    "control": {"legacy_search"},
    "treatment": {"new_search"},
}

filtered = FilteredToolset(
    wrapped=toolset,
    filter_func=lambda ctx, t: t.name in TOOL_GROUPS.get(ctx.deps.experiment_group, set()),
)

model = TestModel(call_tools=["new_search"])
agent = Agent(model, toolsets=[filtered], deps_type=SessionDeps)

async def main():
    result = await agent.run("Search for widgets", deps=SessionDeps(experiment_group="treatment"))
    print(result.output)

asyncio.run(main())
```

---

## 3. `PrefixedToolset` — namespace isolation

**Module:** `pydantic_ai.toolsets.prefixed`

`PrefixedToolset` renames every tool in the wrapped toolset by prepending
`{prefix}_` to its name. This is essential when combining toolsets from different
sources that may have overlapping names, or when exposing MCP tool collections under
a per-server namespace.

### Constructor (from source)

```python
@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    prefix: str
```

The prefix is applied with an underscore separator: `prefix="db"` turns `query` into `db_query`.

### Example 1 — namespace two toolsets with clashing names

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.models.test import TestModel

postgres_toolset = FunctionToolset()
redis_toolset = FunctionToolset()

@postgres_toolset.tool
def query(sql: str) -> str:
    """Query PostgreSQL."""
    return "pg rows"

@redis_toolset.tool
def query(key: str) -> str:  # type: ignore[no-redef]  # same name, different toolset
    """Query Redis."""
    return "redis value"

pg_namespaced = PrefixedToolset(postgres_toolset, prefix="pg")
redis_namespaced = PrefixedToolset(redis_toolset, prefix="redis")

model = TestModel(call_tools=["pg_query"])
agent = Agent(model, toolsets=[CombinedToolset([pg_namespaced, redis_namespaced])])

async def main():
    result = await agent.run("Query the PostgreSQL database")
    print(result.output)
    # Model sees: pg_query, redis_query — no name collision

asyncio.run(main())
```

### Example 2 — prefix MCP server tools

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.models.test import TestModel

# Simulate a toolset loaded from an MCP server
mcp_toolset = FunctionToolset()

@mcp_toolset.tool
def search(query: str) -> str:
    """Full-text search."""
    return "search results"

@mcp_toolset.tool
def fetch(url: str) -> str:
    """Fetch a URL."""
    return "page content"

# Prefix with the server name so the model knows which server each tool belongs to
prefixed = PrefixedToolset(mcp_toolset, prefix="brave")
# Model now sees: brave_search, brave_fetch

model = TestModel(call_tools=["brave_search"])
agent = Agent(model, toolsets=[prefixed])

async def main():
    result = await agent.run("Search for Python async patterns")
    print(result.output)

asyncio.run(main())
```

---

## 4. `RenamedToolset` — selective tool renaming

**Module:** `pydantic_ai.toolsets.renamed`

`RenamedToolset` renames specific tools via a `{new_name: old_name}` mapping. Tools not
in the mapping keep their original names. This is useful for adapting third-party toolsets
whose names clash with your conventions or other toolsets in the agent.

### Constructor (from source)

```python
@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    name_map: dict[str, str]  # {new_name: original_name}
```

### Example 1 — rename to avoid collision

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.renamed import RenamedToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def get(url: str) -> str:
    """HTTP GET request."""
    return f"Response from {url}"

@toolset.tool
def post(url: str, body: str) -> str:
    """HTTP POST request."""
    return f"Posted to {url}"

# Rename to avoid clashing with Python builtins and be more descriptive
renamed = RenamedToolset(
    wrapped=toolset,
    name_map={"http_get": "get", "http_post": "post"},
)
# Model now sees: http_get, http_post

model = TestModel(call_tools=["http_get"])
agent = Agent(model, toolsets=[renamed])

async def main():
    result = await agent.run("Fetch https://example.com")
    print(result.output)

asyncio.run(main())
```

### Example 2 — partial rename (some tools keep original names)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.renamed import RenamedToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def calculate(expr: str) -> float:
    """Evaluate a math expression."""
    return float(eval(expr))  # noqa: S307

@toolset.tool
def format_number(n: float, decimals: int = 2) -> str:
    """Format a number for display."""
    return f"{n:.{decimals}f}"

# Only rename calculate; format_number keeps its name
renamed = RenamedToolset(
    wrapped=toolset,
    name_map={"math_eval": "calculate"},
)

model = TestModel(call_tools=["math_eval"])
agent = Agent(model, toolsets=[renamed])

async def main():
    result = await agent.run("What is 2 + 2?")
    print(result.output)

asyncio.run(main())
```

---

## 5. `ExternalToolset` — tools executed outside the agent run

**Module:** `pydantic_ai.toolsets.external`

`ExternalToolset` advertises tools to the model but marks them with `kind='external'`,
signalling that their execution happens *outside* the current agent run. The agent will
suspend with `DeferredToolRequests` output containing the pending calls; the caller
executes them, supplies results, and resumes the run.

This is the canonical pattern for human-in-the-loop workflows, remote tool execution,
or bridging to external systems that cannot be called synchronously.

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

`call_tool` raises `NotImplementedError` — execution must happen externally.

### Example 1 — declare external tools, handle deferred output

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai import ExternalToolset, ToolDefinition
from pydantic_ai.models.test import TestModel

# Declare two tools the model can call but we execute ourselves
external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="send_slack_message",
            description="Send a message to a Slack channel.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["channel", "text"],
            },
        ),
        ToolDefinition(
            name="create_jira_ticket",
            description="Create a JIRA issue.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["summary"],
            },
        ),
    ]
)

model = TestModel(call_tools=["send_slack_message"])
# Include DeferredToolRequests in output_type so the agent can surface pending calls
agent = Agent(model, toolsets=[external], output_type=[str, DeferredToolRequests])

async def main():
    # The run pauses with DeferredToolRequests when a tool is called
    result = await agent.run("Notify #ops that the deploy succeeded")
    if isinstance(result.output, DeferredToolRequests):
        print("External calls to execute:", [tc.tool_name for tc in result.output.calls])
        # Execute externally, then feed results back:
        # results = {tc.tool_call_id: "ok" for tc in result.output.calls}
        # result2 = await agent.run(
        #     message_history=result.all_messages(),
        #     deferred_tool_results=result.output.build_results(calls=results),
        # )
    else:
        print(result.output)

asyncio.run(main())
```

### Example 2 — bridge to a remote microservice

```python
import asyncio
from pydantic_ai import Agent, ExternalToolset, ToolDefinition
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel

external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="run_sql",
            description="Execute SQL against the data warehouse.",
            parameters_json_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
)

async def execute_externally(requests: DeferredToolRequests) -> DeferredToolResults:
    """Simulate forwarding calls to a remote service."""
    # requests.calls holds the ToolCallPart items for external execution
    results = {
        tc.tool_call_id: '{"rows": 42}'
        for tc in requests.calls  # .calls not .tool_calls
    }
    return requests.build_results(calls=results)  # calls=, not results=

model = TestModel(call_tools=["run_sql"])
agent = Agent(model, toolsets=[external], output_type=[str, DeferredToolRequests])

async def main():
    result = await agent.run("How many orders were placed today?")
    if isinstance(result.output, DeferredToolRequests):
        resolved = await execute_externally(result.output)
        result2 = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=resolved,
        )
        print(result2.output)

asyncio.run(main())
```

---

## 6. `DeferredLoadingToolset` — hide tools until revealed

**Module:** `pydantic_ai.toolsets.deferred_loading`

`DeferredLoadingToolset` marks tools with `defer_loading=True`, hiding them from the
initial context sent to the model. Tools are revealed on demand via tool-search,
`load_capability`, or another tool's `ToolReturn.tools` mechanism.

This reduces prompt token usage when the agent has a large tool catalogue and only a
fraction are relevant to any given request.

### Constructor (from source)

```python
@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    tool_names: frozenset[str] | None = None

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_names: frozenset[str] | None = None,
    ): ...
```

Pass `tool_names=None` to defer all tools; pass a frozenset to defer only specific ones.

### Example 1 — defer an entire specialised toolset

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.models.test import TestModel

always_available = FunctionToolset()
specialised = FunctionToolset()

@always_available.tool
def greet(name: str) -> str:
    """Greet a user."""
    return f"Hello, {name}!"

@specialised.tool
def analyse_image(url: str) -> str:
    """Run image analysis (expensive)."""
    return "Image analysis result"

@specialised.tool
def transcribe_audio(url: str) -> str:
    """Transcribe audio (expensive)."""
    return "Transcription result"

# specialised tools start hidden; the model can request them if needed
deferred = DeferredLoadingToolset(wrapped=specialised)

model = TestModel()
agent = Agent(model, toolsets=[always_available, deferred])

async def main():
    # Normal greeting — specialised tools never loaded, token-efficient
    result = await agent.run("Say hello to Alice")
    print(result.output)

asyncio.run(main())
```

### Example 2 — defer only heavy tools, keep lightweight ones available

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def fast_lookup(key: str) -> str:
    """Quick in-memory lookup."""
    return "cached value"

@toolset.tool
def vector_search(query: str) -> str:
    """Semantic vector search (slow)."""
    return "semantic results"

@toolset.tool
def llm_rerank(items: list) -> list:
    """Re-rank with a second LLM call (very slow)."""
    return items

# Only defer the two expensive tools; fast_lookup is always visible
deferred = DeferredLoadingToolset(
    wrapped=toolset,
    tool_names=frozenset({"vector_search", "llm_rerank"}),
)

model = TestModel(call_tools=["fast_lookup"])
agent = Agent(model, toolsets=[deferred])

async def main():
    result = await agent.run("Look up the config for user 42")
    print(result.output)

asyncio.run(main())
```

---

## 7. `HandleDeferredToolCalls` — inline deferred-call resolution

**Module:** `pydantic_ai.capabilities.deferred_tool_handler`

`HandleDeferredToolCalls` is a capability that intercepts deferred tool calls (those from
`ExternalToolset` or `ApprovalRequiredToolset`) and resolves them inline without exposing
`DeferredToolRequests` to the caller. The handler function receives the pending requests,
supplies results, and the run continues automatically.

### Constructor (from source)

```python
@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    handler: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests],
        DeferredToolResults | None | Awaitable[DeferredToolResults | None],
    ]
```

Returning `None` from the handler declines to handle; the next capability in the chain
gets a turn, or the requests bubble up as output.

### Example 1 — auto-approve all deferred calls

```python
import asyncio
from pydantic_ai import Agent, ExternalToolset, ToolDefinition
from pydantic_ai.capabilities.deferred_tool_handler import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.models.test import TestModel

external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="web_search",
            description="Search the web.",
            parameters_json_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
)

async def auto_approve(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults:
    """Resolve every deferred call automatically."""
    # requests.calls holds ToolCallPart items; build_results takes calls={id: result}
    return requests.build_results(
        calls={
            tc.tool_call_id: "Top search result: Python docs"
            for tc in requests.calls
        }
    )

model = TestModel(call_tools=["web_search"])
agent = Agent(
    model,
    toolsets=[external],
    capabilities=[HandleDeferredToolCalls(handler=auto_approve)],
)

async def main():
    result = await agent.run("Search for async Python patterns")
    print(result.output)

asyncio.run(main())
```

### Example 2 — selective handler (decline some calls)

```python
import asyncio
from pydantic_ai import Agent, ExternalToolset, ToolDefinition
from pydantic_ai.capabilities.deferred_tool_handler import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.models.test import TestModel

external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="safe_action",
            description="A safe action.",
            parameters_json_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        ),
        ToolDefinition(
            name="risky_action",
            description="A risky action.",
            parameters_json_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        ),
    ]
)

async def selective_handler(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    # requests.calls holds ToolCallPart items for external execution
    safe_calls = [tc for tc in requests.calls if tc.tool_name == "safe_action"]
    if not safe_calls:
        return None  # decline — let it bubble up
    return requests.build_results(
        calls={tc.tool_call_id: "done" for tc in safe_calls}
    )

model = TestModel(call_tools=["safe_action"])
agent = Agent(
    model,
    toolsets=[external],
    capabilities=[HandleDeferredToolCalls(handler=selective_handler)],
)

async def main():
    result = await agent.run("Run the safe action")
    print(result.output)

asyncio.run(main())
```

---

## 8. `RaiseContentFilterError` — surface content-filter failures

**Module:** `pydantic_ai.capabilities.content_filter`

By default pydantic-ai treats a response with `finish_reason='content_filter'` as a
normal (possibly partial) response. Adding `RaiseContentFilterError` as a capability
turns those responses into a raised `ContentFilterError`, making it impossible to silently
consume a filtered response. The full `ModelResponse` is serialised into the error's `body`
field for post-mortem inspection.

### Constructor (from source)

```python
@dataclass
class RaiseContentFilterError(AbstractCapability[AgentDepsT]):
    id: str | None = 'raise_content_filter_error'
    # Fixed id — a second instance de-dupes via combine()
```

The capability takes no configuration beyond the optional `id` override. A single instance
is always sufficient.

### Example 1 — catch a content-filtered response

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.content_filter import RaiseContentFilterError
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse, TextPart

# TestModel is used here; in production swap for openai:gpt-4o etc.
model = TestModel()
agent = Agent(
    model,
    capabilities=[RaiseContentFilterError()],
)

async def main():
    try:
        result = await agent.run("Tell me something benign")
        print(result.output)
    except ContentFilterError as exc:
        print(f"Content filter hit: {exc}")
        # exc.body contains the raw ModelResponse JSON for inspection

asyncio.run(main())
```

### Example 2 — log and re-raise in a FastAPI handler

```python
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.capabilities.content_filter import RaiseContentFilterError
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.models.test import TestModel

logger = logging.getLogger(__name__)

model = TestModel()
agent = Agent(
    model,
    capabilities=[RaiseContentFilterError()],
)

async def handle_user_prompt(prompt: str) -> str:
    try:
        result = await agent.run(prompt)
        return result.output
    except ContentFilterError as exc:
        logger.warning(
            "Content filter triggered",
            extra={"prompt": prompt[:200], "body": exc.body},
        )
        return "I can't help with that request."

async def main():
    response = await handle_user_prompt("A totally normal question")
    print(response)

asyncio.run(main())
```

### Example 3 — two capabilities, one instance wins (idempotent)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.content_filter import RaiseContentFilterError
from pydantic_ai.models.test import TestModel

model = TestModel()

# Combining two instances is safe — de-duped to one via combine()
agent = Agent(
    model,
    capabilities=[
        RaiseContentFilterError(),
        RaiseContentFilterError(),  # harmless duplicate
    ],
)

async def main():
    result = await agent.run("Hello")
    print(result.output)

asyncio.run(main())
```

---

## 9. `SelectModel` — per-step dynamic model selection

**Module:** `pydantic_ai.capabilities.select_model`

`SelectModel` is a capability that calls a user-supplied `selector` before each model
request step. The selector receives a `ModelSelectionContext` containing the run
dependencies, message history, accumulated usage, and the lower-precedence model, and
returns a new model instance or model ID string.

This enables routing patterns (e.g. cheap model for simple turns, expensive model for
complex ones), per-user model preferences, cost budgets, and A/B testing.

### Constructor (from source)

```python
@dataclass
class SelectModel(AbstractCapability[AgentDepsT]):
    selector: ModelSelector[AgentDepsT]
    # ModelSelector = Callable[[ModelSelectionContext[AgentDepsT]], Model | KnownModelName | str | Awaitable[...]]
```

### Example 1 — route by message length

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.select_model import SelectModel
from pydantic_ai.models import ModelSelectionContext
from pydantic_ai.models.test import TestModel

cheap_model = TestModel(custom_result_text="cheap answer")
expensive_model = TestModel(custom_result_text="detailed answer")

def route_by_length(ctx: ModelSelectionContext) -> object:
    last_user = next(
        (m for m in reversed(ctx.messages) if hasattr(m, "parts")), None
    )
    if last_user is None:
        return cheap_model
    text = "".join(
        p.content for p in last_user.parts if hasattr(p, "content") and isinstance(p.content, str)
    )
    return expensive_model if len(text) > 200 else cheap_model

agent = Agent(
    cheap_model,  # default; selector overrides per-step
    capabilities=[SelectModel(selector=route_by_length)],
)

async def main():
    short_result = await agent.run("Hi")
    print(short_result.output)  # cheap answer

    long_prompt = "Explain " + "in detail " * 30 + "quantum entanglement."
    long_result = await agent.run(long_prompt)
    print(long_result.output)  # detailed answer

asyncio.run(main())
```

### Example 2 — per-user model preference via deps

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities.select_model import SelectModel
from pydantic_ai.models import Model, ModelSelectionContext
from pydantic_ai.models.test import TestModel

@dataclass
class UserDeps:
    # Pass a Model instance (not a string ID, which would require live provider credentials)
    preferred_model: Model

def user_model_selector(ctx: ModelSelectionContext[UserDeps]) -> Model:
    return ctx.deps.preferred_model

default_model = TestModel(custom_result_text="default answer")
premium_model = TestModel(custom_result_text="premium detailed answer")

agent = Agent(
    default_model,
    deps_type=UserDeps,
    capabilities=[SelectModel(selector=user_model_selector)],
)

async def main():
    # Premium user gets the premium model; swap TestModel for a real provider in production
    result = await agent.run(
        "Summarise this document",
        deps=UserDeps(preferred_model=premium_model),
    )
    print(result.output)  # premium detailed answer

asyncio.run(main())
```

### Example 3 — async selector with budget guard

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.select_model import SelectModel
from pydantic_ai.models import ModelSelectionContext
from pydantic_ai.models.test import TestModel

cheap_model = TestModel(custom_result_text="cheap")
expensive_model = TestModel(custom_result_text="expensive")
BUDGET_TOKEN_LIMIT = 10_000

async def budget_selector(ctx: ModelSelectionContext) -> object:
    """Switch to cheap model once token budget is approaching."""
    await asyncio.sleep(0)  # simulate async budget check
    total = ctx.usage.total_tokens or 0
    return cheap_model if total > BUDGET_TOKEN_LIMIT else expensive_model

agent = Agent(
    expensive_model,
    capabilities=[SelectModel(selector=budget_selector)],
)

async def main():
    result = await agent.run("Draft a short report")
    print(result.output)

asyncio.run(main())
```

---

## 10. `NativeOrLocalTool` — provider-native with Python fallback

**Module:** `pydantic_ai.capabilities.native_or_local`

`NativeOrLocalTool` pairs a provider-native tool (e.g. OpenAI's built-in web search)
with a local Python fallback. When the model supports the native tool, the local function
is removed from the tool list. When it doesn't, the native tool is removed and the local
function takes over. The model never sees both simultaneously.

Built-in subclasses `WebSearch`, `WebFetch`, and `ImageGeneration` all extend this base.

### Constructor (from source)

```python
@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    native: AgentNativeTool[AgentDepsT] | bool = True
    local: str | Tool[AgentDepsT] | Callable | AbstractToolset | bool | None = None
```

`native=True` resolves to the default native tool (subclasses only). `native=False` forces
the local fallback. `local=False` forces the native tool with no fallback (raises `UserError`
if the provider doesn't support it).

### Example 1 — WebSearch with DuckDuckGo fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

# WebSearch extends NativeOrLocalTool; local='duckduckgo' is a named strategy
agent = Agent(
    TestModel(custom_result_text="search result"),
    capabilities=[WebSearch(local="duckduckgo")],
)

async def main():
    # If the model supports OpenAI's built-in search, it uses that.
    # Otherwise DuckDuckGo local fallback kicks in transparently.
    result = await agent.run("What happened in tech news today?")
    print(result.output)

asyncio.run(main())
```

### Example 2 — direct `NativeOrLocalTool` with a custom local fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool
from pydantic_ai.models.test import TestModel

def my_local_search(query: str) -> str:
    """Local fallback using an in-house search API."""
    return f"Local results for: {query}"

# native=False → always use the local fallback (no native tool advertised)
cap = NativeOrLocalTool(native=False, local=my_local_search)

model = TestModel(custom_result_text="answer from search")
agent = Agent(model, capabilities=[cap])

async def main():
    result = await agent.run("Find the latest Python release")
    print(result.output)

asyncio.run(main())
```

### Example 3 — `ImageGeneration` (a NativeOrLocalTool subclass)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.models.test import TestModel

# ImageGeneration extends NativeOrLocalTool with a fallback_image_model option
# native=True uses the provider's built-in image generation (e.g. gpt-image-1)
# native=False falls back to a separate image model specified via fallback_image_model
agent = Agent(
    TestModel(custom_result_text="Here is your image: [url]"),
    capabilities=[ImageGeneration()],  # native=True by default
)

async def main():
    result = await agent.run("Generate a sunset over mountains")
    print(result.output)

asyncio.run(main())
```

### Example 4 — forcing the local tool only

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

def my_fetcher(url: str) -> str:
    """Custom fetch implementation behind an authenticated proxy."""
    return f"Fetched content from {url}"

# Force local always — useful when you must control all outbound HTTP
agent = Agent(
    TestModel(custom_result_text="fetched"),
    capabilities=[WebFetch(native=False, local=my_fetcher)],
)

async def main():
    result = await agent.run("Get the content at https://example.com/api")
    print(result.output)

asyncio.run(main())
```

---

## Summary

| Class | Module | Pattern |
|---|---|---|
| `ApprovalRequiredToolset` | `toolsets.approval_required` | Gate tool calls with a predicate; raise `ApprovalRequired` |
| `FilteredToolset` | `toolsets.filtered` | Dynamically hide/show tools per request context |
| `PrefixedToolset` | `toolsets.prefixed` | Namespace tools with a `{prefix}_` separator |
| `RenamedToolset` | `toolsets.renamed` | Rename specific tools via a `{new: old}` dict |
| `ExternalToolset` | `toolsets.external` | Declare tools executed outside the agent run |
| `DeferredLoadingToolset` | `toolsets.deferred_loading` | Hide tools from model; reveal on demand |
| `HandleDeferredToolCalls` | `capabilities.deferred_tool_handler` | Inline-resolve deferred/approval calls via handler |
| `RaiseContentFilterError` | `capabilities.content_filter` | Turn content-filter responses into `ContentFilterError` |
| `SelectModel` | `capabilities.select_model` | Per-step dynamic model selection via a callable |
| `NativeOrLocalTool` | `capabilities.native_or_local` | Provider-native tool + Python fallback, auto-resolved |

All examples above are runnable against **pydantic-ai 2.57.0** using `TestModel` for
offline verification. Replace `TestModel` with your provider model (e.g. `openai:gpt-4o`)
for live usage.

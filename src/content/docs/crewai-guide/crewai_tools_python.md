---
title: "Tools, @tool decorator, and MCP"
description: "Build tools with BaseTool, the @tool decorator, or bridge MCP servers — plus caching, usage caps, result_as_answer, async variants, and from_langchain."
framework: crewai
language: python
sidebar:
  label: "Tools"
  order: 24
---

> **Verified against crewai==1.14.3a2** (source: `crewai/tools/base_tool.py`, `crewai/tools/__init__.py`, `crewai/mcp/config.py`, `crewai/tools/agent_tools/`).

There are three ways to give an agent a tool:

1. **`@tool` decorator** — fastest for one-liners.
2. **`BaseTool` subclass** — for stateful tools, DI, async.
3. **MCP server reference** — surface a whole server's tools via `Agent.mcps=[...]`.

## Minimal runnable example

```python
from crewai import Agent, Crew, Task, LLM
from crewai.tools import tool

@tool
def wiki_summary(topic: str) -> str:
    """One-paragraph summary of a Wikipedia topic."""
    import urllib.request, json
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    return data.get("extract", "not found")

researcher = Agent(
    role="Research Assistant",
    goal="Answer factual questions with citations",
    backstory="Librarian with encyclopaedic instincts.",
    llm=LLM(model="openai/gpt-4o-mini"),
    tools=[wiki_summary],
)

t = Task(
    description="Summarise the topic '{topic}'.",
    expected_output="Two-sentence summary.",
    agent=researcher,
)

print(Crew(agents=[researcher], tasks=[t]).kickoff(inputs={"topic": "Ada_Lovelace"}).raw)
```

## Three ways to build a tool

| Approach | When to use | Docs import |
|---|---|---|
| `@tool` (no args) | Quick wrapper over a function. | `from crewai.tools import tool` |
| `@tool("name", result_as_answer=True)` | Same, with overrides. | same |
| `class MyTool(BaseTool)` | Stateful, async, custom schema. | `from crewai.tools import BaseTool` |

`@tool` returns a `Tool` (subclass of `BaseTool`) — it's interchangeable with hand-rolled subclasses at runtime.

## `@tool` decorator

```python
from crewai.tools import tool

@tool
def convert_currency(amount: float, src: str, dst: str) -> float:
    """Convert amount from src currency to dst (uses the ECB rate table)."""
    ...

@tool("timezone_convert")
def tz(dt_iso: str, target: str) -> str:
    """Convert an ISO-8601 datetime to the target timezone."""
    ...

@tool("lookup_order", result_as_answer=True, max_usage_count=5)
def lookup_order(order_id: str) -> dict:
    """Fetch an order's current status and tracking URL."""
    ...
```

Rules from the source:
- The function **must** have a docstring — it becomes the LLM-facing description.
- The function **must** have typed parameters — the schema is derived from annotations.
- Three overload forms: `@tool`, `@tool("name")`, `@tool("name", ...)` / `@tool(result_as_answer=True)`.
- Async functions work; `asyncio.run` is invoked transparently if the agent runs synchronously.

## `BaseTool` subclass

```python
from typing import ClassVar
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query.")
    limit: int = Field(5, ge=1, le=50)

class SemanticSearch(BaseTool):
    name: str = "semantic_search"
    description: str = "Semantic search over the engineering KB."
    args_schema: type[BaseModel] = SearchArgs

    env_vars: ClassVar[list[EnvVar]] = [
        EnvVar(name="KB_API_KEY", description="Required", required=True),
    ]

    def _run(self, query: str, limit: int = 5) -> list[dict]:
        return search_api(query, limit=limit)

    async def _arun(self, query: str, limit: int = 5) -> list[dict]:
        return await search_api_async(query, limit=limit)
```

- `_run` is required; `_arun` is optional — override for true async.
- `args_schema` is auto-generated from `_run`'s signature when you don't set one explicitly.
- `env_vars` is a hint for CrewAI Platform; it doesn't enforce anything at runtime.

## Behaviour fields (all tool types)

| Field | Default | Effect |
|---|---|---|
| `name` | required | Unique label the LLM addresses the tool by. |
| `description` | required | Sent to the LLM, with a rendered JSON schema of args prepended automatically. |
| `args_schema` | derived | Pydantic model of arguments; validated before `_run`. |
| `cache_function` | always-cache | Callable `(args, result) -> bool` deciding whether to cache this call. |
| `result_as_answer` | `False` | When `True`, the tool's output becomes the agent's final answer for that task. |
| `max_usage_count` | `None` | Hard cap on how many times the agent can call this tool per task. Over-limit behaviour depends on the call path — see [Usage caps](#usage-caps). |

```python
def no_cache(_args, _result): return False

def only_cache_stable(args, result):
    return args.get("volatile") is not True

class PriceCheck(BaseTool):
    name: str = "price_check"
    description: str = "Look up current price."
    cache_function = staticmethod(no_cache)       # live data, never cache
```

## Caching

When `Agent.cache=True` (default), CrewAI remembers `(tool, args) -> result` across the whole crew run. Override with:

- Tool-level `cache_function` — inspect args/result per call.
- Agent-level `cache=False` — disables caching for all this agent's tools.
- Crew-level `cache=False` — global off-switch.

## `result_as_answer` — skip final LLM synthesis

```python
@tool("get_final_report", result_as_answer=True)
def get_final_report(report_id: str) -> str:
    """Return the stored final report."""
    return load_report(report_id)
```

When the agent calls `get_final_report`, its output is returned verbatim — no extra LLM round to wrap the result. Useful for deterministic pipelines.

## Usage caps

```python
@tool("external_api", max_usage_count=3)
def external_api(q: str) -> str:
    ...
```

Two call paths, two behaviours (both are in the installed source):

- **Direct call** — `BaseTool.run()` / `Tool.run()`: `_claim_usage()` returns the sentinel string `"Tool '...' has reached its usage limit of N times..."` instead of executing. The caller sees a normal return value.
- **Agent executor** — the executor wraps the tool as a `CrewStructuredTool` (internal API), whose `_run` / `_arun` **raises `ToolUsageLimitExceededError`** when over-limit. The executor catches it and feeds the message back to the LLM, so the agent sees the limit message as a tool observation and usually adapts.

Catch the exception if you're driving the structured tool yourself; if you're calling `tool.run(...)` directly, check for the sentinel string in the return value.

Reset per-tool counters between runs:

```python
for t in my_tools:
    t.reset_usage_count()
```

## MCP server tools — `Agent.mcps`

Instead of wrapping every MCP tool manually, register the server:

```python
from crewai import Agent
from crewai.mcp import MCPServerStdio, MCPServerHTTP, MCPServerSSE
from crewai.mcp.filters import create_static_tool_filter

agent = Agent(
    role="Devops",
    goal="Inspect infra",
    backstory="SRE with 10 years in on-call.",
    mcps=[
        MCPServerStdio(
            command="uvx",
            args=["mcp-server-fetch"],
            cache_tools_list=True,
        ),
        MCPServerHTTP(
            url="https://tools.example.com/mcp",
            headers={"Authorization": "Bearer ..."},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=["search", "fetch"],
            ),
        ),
        MCPServerSSE(url="https://events.example.com/mcp/sse"),
        "notion",                             # CrewAI Platform connected app
        "https://tools.example.com/mcp#search",  # single-tool filter
    ],
)
```

| Transport | Class | Notes |
|---|---|---|
| Stdio | `MCPServerStdio` | Local subprocess (`command`, `args`, `env`). |
| HTTP (streamable) | `MCPServerHTTP` | Remote server. `streamable=True` by default. |
| SSE | `MCPServerSSE` | Server-Sent-Events transport. |
| Platform shortcut | `"notion"`, `"gmail"` | Bare slugs resolve to CrewAI Platform integrations. |
| Single-tool filter | `"url#tool_name"` | Only that tool is surfaced. |

Per-server options:
- `tool_filter: ToolFilter` — allow/deny lists.
- `cache_tools_list: bool` — memoise the tool listing for faster subsequent runs.

## Bridging from LangChain tools

```python
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

lc_tool = DuckDuckGoSearchRun()
crew_tool = BaseTool.from_langchain(lc_tool)

agent = Agent(role="...", goal="...", backstory="...", tools=[crew_tool])
```

The return trip exists too — `to_langchain([crew_tool])` converts to `CrewStructuredTool` instances compatible with LangChain-style orchestrators.

## Built-in helper tools

`crewai.tools.agent_tools` exposes the tools the framework itself installs when you set `Agent.allow_delegation=True`:

- `DelegateWorkTool` — hand off a subtask to a coworker.
- `AskQuestionTool` — ask a coworker a targeted question.
- `AddImageTool` / `ReadFileTool` — file/image passing in multi-agent crews.

You rarely import these directly — they're attached automatically.

## Patterns

### 1. Human-approval gated tool

```python
@tool("push_to_prod")
def push_to_prod(artifact: str) -> str:
    """Release the artifact to production (asks for confirmation)."""
    if input(f"Deploy {artifact}? [y/N] ").strip().lower() != "y":
        return "cancelled by operator"
    return deploy(artifact)
```

Combine with `Task(human_input=True)` for a second checkpoint.

### 2. Rate-limited external API

```python
import time
from crewai.tools import BaseTool

class SlowAPI(BaseTool):
    name: str = "slow_api"
    description: str = "Call the rate-limited partner API."
    _last: float = 0.0

    def _run(self, payload: str) -> str:
        delta = time.monotonic() - self._last
        if delta < 1.0:
            time.sleep(1.0 - delta)
        self._last = time.monotonic()
        return partner_api_call(payload)
```

### 3. Per-task tool restrictions

```python
analyst = Agent(role="Analyst", ..., tools=[wiki_summary, db_lookup, web_scrape])

t = Task(
    description="Only use the DB, no web.",
    expected_output="Analysis.",
    agent=analyst,
    tools=[db_lookup],        # overrides the agent's full tool list for this task
)
```

### 4. MCP with tool filter

```python
from crewai.mcp.filters import create_dynamic_tool_filter

def allow_read_only(tool_name: str, metadata: dict) -> bool:
    return metadata.get("readOnly", False)

agent = Agent(
    role="Browser",
    goal="...",
    backstory="...",
    mcps=[
        MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/sandbox"],
            tool_filter=create_dynamic_tool_filter(allow_read_only),
        ),
    ],
)
```

### 5. Tool that returns the answer verbatim

```python
@tool("lookup_answer", result_as_answer=True)
def lookup_answer(q: str) -> str:
    """Return the canned answer from the FAQ DB."""
    return faq_db[q]
```

When the agent calls this, the task's `TaskOutput.raw` is exactly the return value — no extra LLM cost.

## Gotchas

- **Docstring + annotations are mandatory** with `@tool`. The decorator raises `ValueError` at import time if either is missing.
- **`description` is auto-extended.** `BaseTool._generate_description` prepends name/args JSON to your description; what the LLM sees is longer than what you wrote.
- **`async_execution` on the task is separate from `_arun`.** Tasks run on threads when `async_execution=True`; tools still execute on that thread.
- **MCP stdio servers spawn a subprocess.** Long-lived crews may leak processes if you don't run them under a supervisor — set `cache_tools_list=True` to avoid re-listing.
- **`CrewStructuredTool` moved to internal API.** Use `BaseTool` everywhere external. `to_structured_tool()` exists only for the executor's internal use.
- **`CodeInterpreterTool` was removed.** Use E2B, Modal, or a custom sandbox in a `BaseTool` subclass.
- **LangChain interop uses `CrewStructuredTool`.** `from_langchain()` and `to_langchain()` round-trip through it even though the class is considered internal.

---
title: "Tools, @tool decorator, and MCP"
description: "Build tools with BaseTool, the @tool decorator, or bridge MCP servers — plus caching, usage caps, result_as_answer, async variants, and from_langchain."
framework: "crewai"
language: "python"
sidebar:
  label: "Tools"
  order: 24
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/tools/base_tool.py`, `crewai/tools/__init__.py`, `crewai/mcp/config.py`, `crewai/tools/agent_tools/`).

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
    llm=LLM(model="openai/%%model=openai-small%%"),
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
- Type annotations are strongly recommended: the argument schema is derived from them. Unannotated parameters are accepted, but the LLM then sees no type for them.
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
| `cache_function` | cache every call | Callable `(args, result) -> bool` deciding whether to cache this call, once caching is on (see [Caching](#caching)). |
| `result_as_answer` | `False` | When `True`, the tool's output becomes the agent's final answer for that task. |
| `max_usage_count` | `None` | Hard cap on how many times the agent can call this tool. Over the cap the agent is told so instead of the tool running; see [Usage caps](#usage-caps). |
| `tool_failure_policy` | `None` | Per-tool override of what happens when the tool reports failure; see [Reporting tool failures](#reporting-tool-failures). |

## Caching

Tool-result caching is **opt-in** in %%version=crewai%%. When it is on, a repeated call with the same tool and the same arguments returns the first result without running the tool again. We ran the same two-call script against every combination:

| `Agent(cache=...)` | `Crew(cache=...)` | Tool executions for two identical calls |
|---|---|---|
| not passed (field default `True`) | not passed / `False` | 2 (no caching) |
| not passed | `True` | 1 |
| `True` passed explicitly | any | 1 |
| `False` | any | 2 |

So: pass `Crew(cache=True)` to cache for every agent that has not opted out, pass `Agent(cache=True)` to cache for that agent only, and pass `Agent(cache=False)` to keep an agent out of a caching crew. The cache is an in-memory `CacheHandler`, so it does not persist between processes, and a declared `ToolFailure` is never cached.

Do not turn it on for tools that read live data or change state. A tool-level `cache_function(args, result) -> bool` decides, per call, whether the result may be cached:

```python
def only_cache_stable(args, result):
    return args.get("volatile") is not True


class PriceCheck(BaseTool):
    name: str = "price_check"
    description: str = "Look up the price of a SKU."
    cache_function = staticmethod(only_cache_stable)

    def _run(self, sku: str, volatile: bool = False) -> str:
        return f"price of {sku}"
```

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
    """Query the partner API (at most three calls per task)."""
    ...
```

When the agent asks for the tool after the cap is spent, the tool does not run. The agent receives `Tool 'external_api' has reached its usage limit of 3 times and cannot be used anymore.` as the tool result and normally finishes without it. We ran a four-call script against `max_usage_count=3` on the native function-calling path: three calls executed and the fourth got that message. The count lives on the tool object (`current_usage_count`), so it accumulates across tasks and runs for as long as the object does. Reset it between runs:

```python
for t in my_tools:
    t.reset_usage_count()
```

Calling `tool.run(...)` yourself returns the same message as an ordinary string when the cap is hit.

One quirk in %%version=crewai%%: on the text (ReAct) path, used by models that do not report native function calling, each call incremented the count twice, so `max_usage_count=4` allowed only two executions. The native path counted correctly. If you cap a tool and run a custom `BaseLLM`, test the cap.

## Reporting tool failures

An exception inside a tool and an error message returned as a string are different things to CrewAI. A tool that wants the framework to know a call failed, without raising, returns a `ToolFailure`:

```python
from crewai import Agent, Crew, Task, LLM
from crewai.tools import tool, ToolFailure


@tool("send_slack")
def send_slack(channel: str, text: str) -> str | ToolFailure:
    """Post a message to a Slack channel."""
    if channel != "#general":
        return ToolFailure(
            message=f"channel {channel} not found",
            code="channel_not_found",
            retryable=False,
        )
    return "sent"


notifier = Agent(
    role="Notifier",
    goal="Tell the team about releases",
    backstory="Release coordinator.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    tools=[send_slack],
    tool_failure_policy="warn",       # the default; "ignore" and "raise" are the others
)
task = Task(description="Announce release 2.4 to the team.", expected_output="Confirmation.",
            agent=notifier, name="announce")
result = Crew(agents=[notifier], tasks=[task]).kickoff()

for failure in result.tasks_output[0].tool_failures:
    print(failure.tool_name, failure.failure.code, failure.failure.message)
```

What each policy does when a tool call fails (we ran all three against a scripted model):

| Policy | Agent sees | Recorded | Run |
|---|---|---|---|
| `"warn"` (default) | `channel #nope not found (code: channel_not_found)` | In `TaskOutput.tool_failures` and as a `ToolFailureDetectedEvent` | Continues; the agent can retry |
| `"ignore"` | The same text | Not recorded | Continues |
| `"raise"` | n/a | Event emitted | Aborts with `ToolExecutionFailedError` |

The policy is resolved most specific first: the tool's own `tool_failure_policy`, then the task's, the agent's and the crew's, then `"warn"`.

What counts as a failure: a returned `ToolFailure`, a tool that raises (the exception text is fed back to the agent and recorded with the exception class as `code`), an MCP server answering `isError`, a call over `max_usage_count`, a call to a tool that does not exist, and arguments that do not validate. A tool that merely returns a string such as `"Error: not found"` is **not** a failure: detection is declarative and CrewAI does not guess from text.

Since 1.15.30 there is one more stop that no policy overrides: when an agent makes the same call (same tool, same arguments, same result) five times in a run, CrewAI raises `ToolExecutionFailedError` with reason `repeated_call` (`ToolFailureReason.REPEATED_CALL`). We ran a scripted model that repeated one call six times: the run stopped after the fifth execution under the default policy and under `"ignore"`, and four repeats completed normally. With caching on, the repeats are served from the cache but still count, so the run stops on the fifth.

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
- `tool_filter: ToolFilter` — allow/deny lists (`create_static_tool_filter(allowed_tool_names=..., blocked_tool_names=...)`) or a function (`create_dynamic_tool_filter`).
- `cache_tools_list: bool` — memoise the tool listing for faster subsequent runs.

### An MCP server you can run locally

The `mcp` package (a CrewAI dependency) includes `FastMCP`, which is enough to try the integration without any external service. Save this as `notes_server.py`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes")


@mcp.tool()
def read_note(name: str) -> str:
    """Read a note by name."""
    return f"note:{name}"


@mcp.tool()
def delete_note(name: str) -> str:
    """Delete a note."""
    return "deleted"


@mcp.tool()
def search_notes(q: str) -> str:
    """Search notes."""
    return f"found {q}"


if __name__ == "__main__":
    mcp.run()          # stdio transport
```

Then give an agent the server, with a filter that hides the destructive tool:

```python
import sys

from crewai import Agent, Crew, Task, LLM
from crewai.mcp import MCPServerStdio
from crewai.mcp.filters import create_static_tool_filter

notes = MCPServerStdio(
    command=sys.executable,
    args=["notes_server.py"],
    tool_filter=create_static_tool_filter(blocked_tool_names=["delete_note"]),
)

librarian = Agent(
    role="Librarian",
    goal="Answer questions from the notes",
    backstory="Keeps the team's notes tidy.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    mcps=[notes],
)
task = Task(description="Find what the notes say about {topic}.", expected_output="A short answer.",
            agent=librarian)
print(Crew(agents=[librarian], tasks=[task]).kickoff(inputs={"topic": "onboarding"}).raw)
```

We ran this against a scripted model: the agent was offered two tools (`read_note` and `search_notes`), called all three tools successfully when no filter was set, and the filter removed `delete_note` when it was. Two things to know:

- The model sees each MCP tool under a **namespaced name** of the form `<sanitised server command and arguments>_<8 hex characters>`, cut to a fixed length, rather than `read_note`. Filters receive the server's original tool name, so write filters against `read_note`, not the namespaced name.
- Each run starts the stdio server as a subprocess and lists its tools. `cache_tools_list=True` skips re-listing.

## Bridging from LangChain tools

```python
from crewai import Agent
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

lc_tool = DuckDuckGoSearchRun()
crew_tool = BaseTool.from_langchain(lc_tool)

agent = Agent(role="Researcher", goal="Search the web", backstory="Analyst.", tools=[crew_tool])
```

`BaseTool.from_langchain` is a class method, so the LangChain package must be installed separately; `langchain-community` is not a CrewAI dependency. `crewai.tools.base_tool.to_langchain([crew_tool])` converts the other way and returns `CrewStructuredTool` instances.

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
analyst = Agent(
    role="Analyst",
    goal="Answer data questions",
    backstory="BI analyst.",
    tools=[wiki_summary, db_lookup, web_scrape],
)

t = Task(
    description="Only use the DB, no web.",
    expected_output="Analysis.",
    agent=analyst,
    tools=[db_lookup],        # overrides the agent's full tool list for this task
)
```

### 4. MCP with tool filter

```python
from typing import Any

from crewai import Agent
from crewai.mcp import MCPServerStdio
from crewai.mcp.filters import ToolFilterContext, create_dynamic_tool_filter

def read_only_for_reviewers(ctx: ToolFilterContext, tool: dict[str, Any]) -> bool:
    # tool has "name", "original_name", "description" and "inputSchema"
    if ctx.agent.role == "Reviewer":
        return tool["name"].startswith(("read_", "list_", "search_"))
    return True

agent = Agent(
    role="Reviewer",
    goal="Review the files in /sandbox",
    backstory="Careful code reviewer.",
    mcps=[
        MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/sandbox"],
            tool_filter=create_dynamic_tool_filter(read_only_for_reviewers),
        ),
    ],
)
```

A dynamic filter is called once per listed tool with `(ToolFilterContext, tool_dict)`. The context carries `agent`, `server_name` and `run_context`. The tool dict does not include MCP tool annotations such as `readOnlyHint`, so filter on names or descriptions.

### 5. Tool that returns the answer verbatim

```python
@tool("lookup_answer", result_as_answer=True)
def lookup_answer(q: str) -> str:
    """Return the canned answer from the FAQ DB."""
    return faq_db[q]
```

When the agent calls this, the task's `TaskOutput.raw` is exactly the return value — no extra LLM cost.

## Gotchas

- **A docstring is mandatory** with `@tool`: the decorator raises `ValueError: Function must have a docstring` at import time without one. Annotations are optional but give the LLM a typed schema.
- **`description` is auto-extended.** `BaseTool._generate_description` prepends name/args JSON to your description; what the LLM sees is longer than what you wrote.
- **`async_execution` on the task is separate from `_arun`.** Tasks run on threads when `async_execution=True`; tools still execute on that thread.
- **MCP stdio servers spawn a subprocess.** `cache_tools_list=True` avoids re-listing tools on later runs; keep long-lived processes under a supervisor.
- **`CrewStructuredTool` moved to internal API.** Use `BaseTool` everywhere external. `to_structured_tool()` exists only for the executor's internal use.
- **`CodeInterpreterTool` was removed.** Use E2B, Modal, or a custom sandbox in a `BaseTool` subclass.
- **LangChain interop uses `CrewStructuredTool`.** `from_langchain()` and `to_langchain()` round-trip through it even though the class is considered internal.

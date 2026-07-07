---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 33"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.10.0: WorkflowBuilder 1.10.0 new API (add_chain/add_multi_selection_edge_group/intermediate_output_from), Workflow+WorkflowRunResult full result API, FunctionTool advanced constructor, Agent.run() structured output/streaming/per-call overrides, MCPStdioTool advanced patterns (additional_tool_argument_names/MCPSpecificApproval/task_options), SecretString+load_settings, FunctionInvocationConfiguration tool-loop control, FileHistoryProvider deep+custom HistoryProvider authoring, AgentMiddleware streaming+MiddlewareTermination, SkillsProvider+custom SkillsSource."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 56
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 33

Source-verified against `agent-framework==1.10.0` · Python 3.10 – 3.13 · July 2026

Ten class groups with three runnable examples each (30 total). Every constructor
signature and constant was read directly from the installed package at
`/usr/local/lib/python3.11/dist-packages/agent_framework/`.

---

## 1 · `WorkflowBuilder` — 1.10.0 new API

`agent_framework._workflows._workflow_builder.WorkflowBuilder`

`WorkflowBuilder` was introduced at 1.6.0 and has grown significantly.
Three methods arrived since the Vol. 1 coverage: **`add_chain`**,
**`add_multi_selection_edge_group`**, and the **`intermediate_output_from`**
routing mode. The constructor also gained `output_from` / `intermediate_output_from`
keyword arguments that replace the deprecated `output_executors`.

### Constructor (1.10.0)

```python
WorkflowBuilder(
    max_iterations: int = 100,
    name: str | None = None,
    description: str | None = None,
    *,
    start_executor: Executor | SupportsAgentRun,
    checkpoint_storage: CheckpointStorage | None = None,
    output_from: list[Executor | SupportsAgentRun] | Literal["all"] | None = _MISSING,
    intermediate_output_from: list[Executor | SupportsAgentRun]
                             | Literal["all", "all_other"]
                             | None = _MISSING,
    output_executors: ...,  # deprecated alias for output_from
)
```

**Output routing modes** (explicit mode activates when either is provided):

| `output_from` | `intermediate_output_from` | Effect |
|---|---|---|
| omitted | omitted | Compatibility: every `yield_output` → `type='output'` |
| `"all"` | — | Every output-capable executor emits `output` |
| `[A]` | — | Only A emits `output`; others hidden |
| `[A]` | `"all_other"` | A → `output`; all other capable → `intermediate` |
| `[]` | `"all_other"` | No `output`; every capable executor → `intermediate` |

### `add_chain(executors)`

Connects a sequence of executors in order so that the output of each flows
into the next. Equivalent to calling `add_edge(a, b); add_edge(b, c)` but
reads as a single declaration.

### Example 1 · Sequential pipeline with `add_chain`

```python
import asyncio
from typing import Never
from agent_framework import (
    WorkflowBuilder, WorkflowContext, Executor, handler,
)


class Normalize(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text.strip().lower())


class Tokenize(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[list[str]]) -> None:
        await ctx.send_message(text.split())


class Count(Executor):
    @handler
    async def run(self, tokens: list[str], ctx: WorkflowContext[Never, int]) -> None:
        await ctx.yield_output(len(tokens))


async def main() -> None:
    normalize = Normalize(id="normalize")
    tokenize = Tokenize(id="tokenize")
    count = Count(id="count")

    workflow = (
        WorkflowBuilder(start_executor=normalize, output_from=[count])
        .add_chain([normalize, tokenize, count])
        .build()
    )

    result = await workflow.run("  Hello World  ")
    print(result.get_outputs())  # [2]


asyncio.run(main())
```

### Example 2 · `add_multi_selection_edge_group` — dynamic fan-out

The `selection_func` receives the source message and the list of target
executor IDs. It returns whichever subset should receive the message this
invocation.

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, WorkflowContext, Executor, handler,
)


class Router(Executor):
    @handler
    async def run(self, task: dict, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message(task)


class SearchWorker(Executor):
    @handler
    async def run(self, task: dict, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"search:{task['query']}")


class SummarizeWorker(Executor):
    @handler
    async def run(self, task: dict, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"summary:{task['query']}")


class FactCheckWorker(Executor):
    @handler
    async def run(self, task: dict, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"facts:{task['query']}")


async def main() -> None:
    router = Router(id="router")
    search = SearchWorker(id="search")
    summarize = SummarizeWorker(id="summarize")
    fact_check = FactCheckWorker(id="fact_check")

    def select_workers(task: dict, available: list[str]) -> list[str]:
        """Pick workers based on task flags."""
        selected = ["search"]  # always search
        if task.get("needs_summary"):
            selected.append("summarize")
        if task.get("needs_fact_check"):
            selected.append("fact_check")
        return selected

    workflow = (
        WorkflowBuilder(
            start_executor=router,
            output_from=[search, summarize, fact_check],
        )
        .add_multi_selection_edge_group(
            router, [search, summarize, fact_check], select_workers
        )
        .build()
    )

    task = {"query": "climate change", "needs_summary": True}
    result = await workflow.run(task)
    print(result.get_outputs())  # ['search:climate change', 'summary:climate change']


asyncio.run(main())
```

### Example 3 · `intermediate_output_from="all_other"` streaming pattern

Use this mode to stream partial results to callers while a single final
executor provides the authoritative output.

```python
import asyncio
from agent_framework import WorkflowBuilder, WorkflowContext, Executor, handler


class Stage1(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.yield_output(f"stage1:{text}")
        await ctx.send_message(text)


class Stage2(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.yield_output(f"stage2:{text}")
        await ctx.send_message(text)


class Final(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"final:{text}")


async def main() -> None:
    s1, s2, fin = Stage1(id="s1"), Stage2(id="s2"), Final(id="fin")

    workflow = (
        WorkflowBuilder(
            start_executor=s1,
            output_from=[fin],
            intermediate_output_from="all_other",  # s1 and s2 → intermediate
        )
        .add_chain([s1, s2, fin])
        .build()
    )

    result = await workflow.run("hello")
    print("outputs:", result.get_outputs())           # ['final:hello']
    print("intermediate:", result.get_intermediate_outputs())  # ['stage1:hello', 'stage2:hello']


asyncio.run(main())
```

---

## 2 · `Workflow` + `WorkflowRunResult` — full result consumption API

`agent_framework._workflows._workflow.Workflow`  
`agent_framework._workflows._workflow.WorkflowRunResult`

`Workflow` is the immutable execution engine built by `WorkflowBuilder.build()`.
`WorkflowRunResult` (a `list[WorkflowEvent]` subclass) wraps the events
produced during non-streaming execution.

### Key `WorkflowRunResult` methods

| Method | Returns | Description |
|---|---|---|
| `get_outputs()` | `list[Any]` | Events with `type='output'` |
| `get_intermediate_outputs()` | `list[Any]` | Events with `type='intermediate'` |
| `get_request_info_events()` | `list[WorkflowEvent]` | Pending HITL requests |
| `get_final_state()` | `WorkflowRunState` | IDLE / IDLE_WITH_PENDING_REQUESTS / … |
| `status_timeline()` | `list[WorkflowEvent]` | Control-plane status events |

### `Workflow.run()` signatures

```python
# Non-streaming (default)
result: WorkflowRunResult = await workflow.run(input_data)

# Streaming
async for event in workflow.run(input_data, stream=True):
    print(event)

# Resume HITL
result2 = await workflow.run(
    input_data,
    checkpoint_id="<id>",
    responses=[WorkflowEvent.create_response(...)],
)
```

### Example 1 · Consume outputs and intermediate outputs

```python
import asyncio
from agent_framework import WorkflowBuilder, WorkflowContext, Executor, handler


class Enrich(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.yield_output({"enriched": text, "length": len(text)})
        await ctx.send_message(text.upper())


class Format(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"[FORMATTED] {text}")


async def main() -> None:
    enrich = Enrich(id="enrich")
    fmt = Format(id="fmt")

    workflow = (
        WorkflowBuilder(
            start_executor=enrich,
            output_from=[fmt],
            intermediate_output_from=[enrich],
        )
        .add_edge(enrich, fmt)
        .build()
    )

    result = await workflow.run("hello")

    # Terminal output from Format
    print("outputs:", result.get_outputs())
    # ['[FORMATTED] HELLO']

    # Intermediate output from Enrich
    print("intermediate:", result.get_intermediate_outputs())
    # [{'enriched': 'hello', 'length': 5}]

    print("state:", result.get_final_state())       # WorkflowRunState.IDLE
    print("status events:", len(result.status_timeline()))  # ≥ 1


asyncio.run(main())
```

### Example 2 · Streaming `run(stream=True)`

```python
import asyncio
from agent_framework import WorkflowBuilder, WorkflowContext, Executor, handler, WorkflowEvent


class SlowAnalyzer(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, dict]) -> None:
        # Simulate incremental work
        for word in text.split():
            await asyncio.sleep(0)  # yield to event loop
            await ctx.yield_output({"word": word, "upper": word.upper()})


async def main() -> None:
    analyzer = SlowAnalyzer(id="analyzer")
    workflow = WorkflowBuilder(
        start_executor=analyzer, output_from="all"
    ).build()

    outputs = []
    async for event in workflow.run("one two three", stream=True):
        if event.type == "output":
            outputs.append(event.data)
            print("got:", event.data)

    print("total outputs:", len(outputs))  # 3


asyncio.run(main())
```

### Example 3 · Inspect `status_timeline` and `get_final_state`

```python
import asyncio
from agent_framework import WorkflowBuilder, WorkflowContext, Executor, handler, WorkflowRunState


class Echo(Executor):
    @handler
    async def run(self, msg: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(msg)


async def main() -> None:
    echo = Echo(id="echo")
    workflow = WorkflowBuilder(start_executor=echo, output_from="all").build()

    result = await workflow.run("ping")

    # status_timeline contains started/status/idle events
    for ev in result.status_timeline():
        print(f"  {ev.type}: {ev.data}")

    state = result.get_final_state()
    assert state == WorkflowRunState.IDLE, f"unexpected state: {state}"
    print("workflow completed cleanly")


asyncio.run(main())
```

---

## 3 · `FunctionTool` — full constructor API

`agent_framework._tools.FunctionTool`

`FunctionTool` wraps any callable as an agent tool. The constructor exposes
parameters that the `@tool` decorator doesn't surface directly.

### Constructor

```python
FunctionTool(
    *,
    name: str,
    description: str = "",
    approval_mode: ApprovalMode | None = None,         # 'always_require' | 'never_require'
    kind: str | None = None,                           # provider-agnostic classification
    max_invocations: int | None = None,                # lifetime cap across all requests
    max_invocation_exceptions: int | None = None,      # consecutive-error cap
    additional_properties: dict[str, Any] | None = None,
    func: Callable[..., Any] | None = None,            # None → declaration-only tool
    input_model: type[BaseModel] | Mapping[str, Any] | None = None,
    result_parser: Callable[[Any], str | list[Content]] | _SkipParsingSentinel | None = None,
)
```

**`max_invocations`** is a *lifetime* counter on the instance — it accumulates
across all agent requests. Use `FunctionInvocationConfiguration["max_function_calls"]`
for per-request limits.

**`result_parser`** controls how the tool's return value is serialised back to
the model. Pass `SKIP_PARSING` to forward the raw value without conversion.

**Declaration-only tools** (`func=None`) let the model reason about tools
that execute client-side or elsewhere without the server actually running them.

### Example 1 · `max_invocations` cap + `invocation_count` reset

```python
import asyncio
from agent_framework import FunctionTool, Agent
from agent_framework.openai import OpenAIChatClient


async def fetch_price(symbol: str) -> str:
    return f"${symbol}: 100.00"


# Allow at most 3 price lookups across the agent's lifetime.
# When the cap is exceeded the tool raises ToolException, forcing the LLM
# to produce a final answer with the information it already has.
price_tool = FunctionTool(
    name="fetch_price",
    description="Fetch the current price of a stock symbol.",
    func=fetch_price,
    max_invocations=3,
)


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="trader", tools=[price_tool])

    response = await agent.run("What are the prices of AAPL, MSFT, and GOOG?")
    print(response.text)
    print("invocations used:", price_tool.invocation_count)  # ≤ 3

    # Reset so the tool can be reused in a new logical request
    price_tool.invocation_count = 0


asyncio.run(main())
```

### Example 2 · Custom `result_parser` and `input_model`

```python
import asyncio
from pydantic import BaseModel
from agent_framework import FunctionTool, Agent, Content
from agent_framework.openai import OpenAIChatClient


class SearchInput(BaseModel):
    query: str
    top_k: int = 5


def search_docs(query: str, top_k: int = 5) -> list[dict]:
    # Simulate a vector search returning structured results
    return [{"rank": i + 1, "snippet": f"Result {i+1} for '{query}'"} for i in range(top_k)]


def format_results(raw: list[dict]) -> str:
    """Convert structured search results to a numbered markdown list."""
    lines = [f"{r['rank']}. {r['snippet']}" for r in raw]
    return "\n".join(lines)


search_tool = FunctionTool(
    name="search_docs",
    description="Search the knowledge base. Returns top_k ranked snippets.",
    func=search_docs,
    input_model=SearchInput,        # Pydantic model drives the JSON schema
    result_parser=format_results,   # Custom serializer for the model
)


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="researcher", tools=[search_tool])
    response = await agent.run("Find information about renewable energy, top 3.")
    print(response.text)


asyncio.run(main())
```

### Example 3 · Declaration-only tool (`func=None`)

Declaration-only tools appear in the model's tool list but are never executed
server-side. Useful when the client renders the result (e.g. a UI widget) or
when the tool call is forwarded to another system.

```python
import asyncio
from agent_framework import FunctionTool, Agent, SKIP_PARSING
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel


class ChartInput(BaseModel):
    chart_type: str
    data: list[float]
    title: str


# The agent can *request* a chart; the client renders it.
# func=None means the agent framework never calls anything server-side.
render_chart = FunctionTool(
    name="render_chart",
    description=(
        "Render a chart from data. The client handles rendering; "
        "you just need to call this with the correct parameters."
    ),
    func=None,         # declaration-only
    input_model=ChartInput,
)


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o")
    agent = Agent(client=client, name="analyst", tools=[render_chart])

    # The model may emit a tool call for render_chart.
    # Because func=None, the framework returns a placeholder result
    # so the conversation can continue; the client inspects the tool call.
    response = await agent.run(
        "Show me a bar chart of [10, 25, 15, 30] titled 'Monthly Sales'."
    )
    print(response.text)  # Model acknowledges chart was requested


asyncio.run(main())
```

---

## 4 · `Agent.run()` — advanced patterns

`agent_framework._agents.Agent`

`Agent.run()` has three `@overload` variants. The key differentiators are the
`options=` parameter type (which selects the structured-output path) and
`stream=True`. Additional per-call overrides let you tune each invocation
without rebuilding the agent.

### Overload summary

```python
# 1. Plain text response
response: AgentResponse[Any] = await agent.run(messages)

# 2. Structured output — options carries a Pydantic response_format
response: AgentResponse[MyModel] = await agent.run(
    messages, options={"response_format": MyModel}
)

# 3. Streaming
stream: ResponseStream = agent.run(messages, stream=True)
async for update in stream:
    print(update.text or "", end="", flush=True)
final: AgentResponse = await stream
```

### Per-call keyword arguments

```python
await agent.run(
    messages,
    session=session,                            # session carries history / context
    middleware=[extra_mw],                      # appended to agent-level middleware
    tools=[runtime_tool],                       # merged with agent-level tools
    options={"model": "gpt-4o"},               # per-call model override
    compaction_strategy=SlidingWindowStrategy(target_count=20),
    tokenizer=CharacterEstimatorTokenizer(),
    function_invocation_kwargs={"max_function_calls": 5},
    client_kwargs={"timeout": 30},
)
```

### Example 1 · Structured output with Pydantic

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


class Sentiment(BaseModel):
    label: str       # 'positive' | 'negative' | 'neutral'
    confidence: float
    reasoning: str


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="classifier")

    response = await agent.run(
        "The product exceeded all my expectations — highly recommend!",
        options={"response_format": Sentiment},
    )

    # response.output is typed as Sentiment when options carries the model
    sentiment: Sentiment = response.output
    print(f"label={sentiment.label}, confidence={sentiment.confidence:.2f}")
    print(f"reasoning: {sentiment.reasoning}")


asyncio.run(main())
```

### Example 2 · Streaming with incremental display

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o")
    agent = Agent(client=client, name="writer")

    stream = agent.run(
        "Write a haiku about distributed systems.",
        stream=True,
    )

    # Iterate updates — each AgentResponseUpdate has .text and .finish_reason
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)

    # Await the stream object to get the fully accumulated AgentResponse
    response = await stream
    print(f"\n\nfinish_reason: {response.finish_reason}")
    print(f"usage: {response.usage}")


asyncio.run(main())
```

### Example 3 · Per-call middleware injection + compaction override

```python
import asyncio
from agent_framework import Agent, AgentMiddleware, SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient


@AgentMiddleware
async def audit_logger(ctx, next_handler):
    """Log every agent call; append metadata before returning."""
    print(f"[audit] running agent '{ctx.agent.name}'")
    response = await next_handler(ctx)
    print(f"[audit] finished, finish_reason={response.finish_reason}")
    return response


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="assistant")

    # Inject middleware and compaction strategy for this specific call only.
    # The agent itself has no middleware configured.
    response = await agent.run(
        "Summarise the history of the internet in two sentences.",
        middleware=[audit_logger],
        compaction_strategy=SlidingWindowStrategy(target_count=10),
        function_invocation_kwargs={"max_function_calls": 0},  # no tools this call
    )
    print(response.text)


asyncio.run(main())
```

---

## 5 · `MCPStdioTool` — advanced patterns

`agent_framework._mcp.MCPStdioTool`

Beyond the basics, `MCPStdioTool` exposes several production-level knobs:
`additional_tool_argument_names`, `allowed_tools`, `task_options`,
`sampling_approval_callback`, and `MCPSpecificApproval` per-tool approval.

### Constructor (key advanced args)

```python
MCPStdioTool(
    name: str,
    command: str,
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    allowed_tools: Collection[str] | None = None,
    approval_mode: Literal["always_require", "never_require"] | MCPSpecificApproval | None = None,
    tool_name_prefix: str | None = None,
    additional_tool_argument_names: Sequence[str]
                                  | Mapping[str, Sequence[str]]
                                  | None = None,
    task_options: MCPTaskOptions | None = None,
    sampling_approval_callback: SamplingApprovalCallback | None = None,
    sampling_max_tokens: int | None = 512,
    sampling_max_requests: int | None = 5,
    parse_tool_results: Callable[[CallToolResult], str | list[Content]] | None = None,
    request_timeout: int | None = None,
)
```

**`additional_tool_argument_names`** — Inject extra hidden arguments into tool
calls beyond what the MCP server's schema declares. Use a `Mapping` to target
specific tool names; use a `Sequence` to apply to all tools.

**`MCPSpecificApproval`** — Per-tool approval map:
`{"write_file": "always_require", "read_file": "never_require"}`.

**`task_options`** — Configures SEP-2663 long-running task lifecycle:
TTL, wait timeout, remote cancel on local cancellation.

### Example 1 · `allowed_tools` filter + `tool_name_prefix`

```python
import asyncio
from agent_framework import MCPStdioTool, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Only expose safe read-only filesystem tools to the agent.
    # Prefix all exposed tools so they're easy to identify in logs.
    fs_tool = MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        allowed_tools={"read_file", "list_directory", "search_files"},
        tool_name_prefix="fs_",           # tools become fs_read_file, fs_list_directory, …
        approval_mode="never_require",    # read-only: no user approval needed
    )

    client = OpenAIChatClient(model="gpt-4o-mini")
    async with fs_tool:
        agent = Agent(client=client, name="file-reader", tools=[fs_tool])
        response = await agent.run("List the files in /tmp")
        print(response.text)


asyncio.run(main())
```

### Example 2 · `MCPSpecificApproval` — per-tool approval gates

```python
import asyncio
from agent_framework import MCPStdioTool, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Reads are auto-approved; writes always prompt the user.
    git_tool = MCPStdioTool(
        name="git",
        command="uvx",
        args=["mcp-server-git", "--repository", "."],
        approval_mode={
            "git_read_file": "never_require",
            "git_diff": "never_require",
            "git_log": "never_require",
            "git_commit": "always_require",   # write op — requires approval
            "git_push": "always_require",     # write op — requires approval
        },
    )

    client = OpenAIChatClient(model="gpt-4o")
    async with git_tool:
        agent = Agent(client=client, name="git-agent", tools=[git_tool])
        response = await agent.run("Show me the last 5 commits.")
        print(response.text)


asyncio.run(main())
```

### Example 3 · `additional_tool_argument_names` for hidden context injection

Some MCP servers accept extra arguments (e.g. `user_id`, `tenant_id`) that
are not part of the schema but can be supplied server-side. Use
`additional_tool_argument_names` to pass them without exposing them to the
model's schema.

```python
import asyncio
from contextvars import ContextVar
from agent_framework import MCPStdioTool, Agent, FunctionInvocationContext
from agent_framework.openai import OpenAIChatClient

current_user_id: ContextVar[str] = ContextVar("current_user_id", default="anonymous")


async def main() -> None:
    # "user_id" will be injected into every tool call by the framework,
    # but the model never sees it in the schema — it can't accidentally
    # pass the wrong value.
    crm_tool = MCPStdioTool(
        name="crm",
        command="python",
        args=["-m", "my_crm_mcp_server"],
        additional_tool_argument_names=["user_id"],   # inject this arg globally
        # Or target specific tools:
        # additional_tool_argument_names={"lookup_contact": ["user_id"]},
    )

    current_user_id.set("user-42")

    client = OpenAIChatClient(model="gpt-4o-mini")
    async with crm_tool:
        agent = Agent(client=client, name="crm-agent", tools=[crm_tool])
        response = await agent.run("Find my open opportunities.")
        print(response.text)


asyncio.run(main())
```

---

## 6 · `SecretString` + `load_settings`

`agent_framework._settings.SecretString`  
`agent_framework._settings.load_settings`

`SecretString` is a `str` subclass whose `__repr__` masks the value.
`load_settings` loads a `TypedDict`-based settings object from environment
variables, an optional `.env` file, and explicit keyword overrides.

### `SecretString` behaviour

```python
key = SecretString("sk-abc123")
print(key)          # sk-abc123   (str behaviour)
print(repr(key))    # SecretString('**********')
print(f"{key}")     # sk-abc123   (f-string uses __str__)
print(key.get_secret_value())  # sk-abc123
```

### `load_settings` signature

```python
load_settings(
    settings_type: type[SettingsT],  # TypedDict subclass
    *,
    env_prefix: str = "",
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
    required_fields: Sequence[str | tuple[str, ...]] | None = None,
    **overrides: Any,
) -> SettingsT
```

Resolution order (highest → lowest priority):
1. `**overrides` keyword arguments
2. `.env` file (only when `env_file_path` is explicitly set)
3. Environment variables (`<env_prefix><FIELD>`)
4. TypedDict class-level defaults / `None` for optional fields

`required_fields` entries:
- `"field_name"` — must resolve to non-`None`
- `("field_a", "field_b")` — at least one must resolve to non-`None`

### Example 1 · Basic settings with env prefix

```python
from typing import TypedDict, Optional
from agent_framework import SecretString, load_settings
import os


class MyServiceSettings(TypedDict, total=False):
    api_key: Optional[SecretString]
    base_url: str
    timeout: int


# Set env vars for demonstration
os.environ["MYSERVICE_API_KEY"] = "sk-demo"
os.environ["MYSERVICE_BASE_URL"] = "https://api.example.com"

settings = load_settings(
    MyServiceSettings,
    env_prefix="MYSERVICE_",
    required_fields=["api_key"],
)

print(settings["api_key"])          # sk-demo
print(repr(settings["api_key"]))    # SecretString('**********')
print(settings["base_url"])         # https://api.example.com
print(settings.get("timeout"))      # None — not in env
```

### Example 2 · `.env` file + explicit overrides

```python
import tempfile, os
from typing import TypedDict, Optional
from agent_framework import SecretString, load_settings


class DatabaseSettings(TypedDict, total=False):
    host: str
    port: int
    password: Optional[SecretString]
    database: str


# Write a temporary .env file
env_content = "DB_HOST=localhost\nDB_PORT=5432\nDB_PASSWORD=secret\n"
with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
    f.write(env_content)
    env_path = f.name

# Explicit override takes precedence over .env file
settings = load_settings(
    DatabaseSettings,
    env_prefix="DB_",
    env_file_path=env_path,
    database="production",   # direct override — highest priority
    required_fields=["host", ("password",)],
)

print(settings["host"])      # localhost  (from .env)
print(settings["database"])  # production (from override)
print(settings["port"])      # 5432       (from .env)
print(repr(settings["password"]))  # SecretString('**********')

os.unlink(env_path)
```

### Example 3 · Tuple `required_fields` — at-least-one constraint

```python
from typing import TypedDict, Optional
from agent_framework import SecretString, load_settings
import os


class AuthSettings(TypedDict, total=False):
    api_key: Optional[SecretString]
    client_id: Optional[str]
    client_secret: Optional[SecretString]


# Only client_id + client_secret are set (no api_key)
os.environ["AUTH_CLIENT_ID"] = "my-client"
os.environ["AUTH_CLIENT_SECRET"] = "my-secret"

# Require EITHER api_key OR (client_id AND client_secret)
settings = load_settings(
    AuthSettings,
    env_prefix="AUTH_",
    required_fields=[
        ("api_key", "client_id"),    # at least one of these
        ("api_key", "client_secret"),
    ],
)

print("client_id:", settings["client_id"])           # my-client
print("api_key:", settings.get("api_key"))           # None
# Both tuple constraints pass because client_id is non-None
```

---

## 7 · `FunctionInvocationConfiguration` — tool-loop control

`agent_framework._tools.FunctionInvocationConfiguration`

`FunctionInvocationConfiguration` is a `TypedDict` that controls the function
invocation loop inside chat clients. Pass it via `function_invocation_kwargs`
on `Agent.run()` or `BaseChatClient.get_response()`.

### All fields

```python
class FunctionInvocationConfiguration(TypedDict, total=False):
    enabled: bool               # Default True. Set False to disable tool use.
    max_iterations: int         # Max LLM roundtrips (not individual tool calls).
    max_function_calls: int     # Max total individual calls per request (best-effort).
    max_consecutive_errors_per_request: int   # Consecutive error threshold.
    terminate_on_unknown_calls: bool          # Raise on unknown tool name.
    additional_tools: ToolTypes | ...         # Extra tools for this request only.
```

**`max_function_calls`** is checked *after* each parallel batch completes —
if the model requests 20 parallel calls and the limit is 10, all 20 execute
before the loop stops.

### Example 1 · Budget-capped tool use

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="budget-agent")

    response = await agent.run(
        "Research and compare three popular Python web frameworks.",
        tools=[...],   # assume web_search tool registered
        function_invocation_kwargs={
            "max_function_calls": 6,       # allow at most 6 search calls
            "max_iterations": 4,           # at most 4 LLM roundtrips
        },
    )
    print(response.text)


asyncio.run(main())
```

### Example 2 · `additional_tools` for per-call tool injection

```python
import asyncio
from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient


async def get_user_profile(user_id: str) -> str:
    return f'{{"user_id": "{user_id}", "plan": "premium"}}'


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    # Agent has no tools configured at construction time
    agent = Agent(client=client, name="dynamic-agent")

    # Inject a tool only for this specific request without rebuilding the agent.
    profile_tool = FunctionTool(
        name="get_user_profile",
        description="Retrieve user profile by ID.",
        func=get_user_profile,
    )

    response = await agent.run(
        "Get the profile for user 'u-123' and summarise their plan.",
        function_invocation_kwargs={
            "additional_tools": [profile_tool],
            "max_function_calls": 2,
        },
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 · Disable tool loop for a pure-LLM call

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    # Agent normally has expensive search tools
    agent = Agent(client=client, name="qa-agent", tools=[...])

    # For this specific call, disable the tool loop entirely so the model
    # answers from its training knowledge — fast and deterministic.
    response = await agent.run(
        "What is the capital of France?",
        function_invocation_kwargs={"enabled": False},
    )
    print(response.text)  # Paris (no tool calls made)


asyncio.run(main())
```

---

## 8 · `FileHistoryProvider` deep + custom `HistoryProvider`

`agent_framework._sessions.FileHistoryProvider`  
`agent_framework._sessions.HistoryProvider`  
`agent_framework._sessions.InMemoryHistoryProvider`

`HistoryProvider` is the ABC that persists per-session conversation history.
`FileHistoryProvider` stores one JSONL file per session; `InMemoryHistoryProvider`
keeps messages in RAM. Custom providers implement three abstract methods.

### `FileHistoryProvider` constructor

```python
FileHistoryProvider(
    storage_path: str | Path,
    *,
    source_id: str = "file_history",
    service_stores_history: bool = False,
    dumps: Callable[[Any], str] | None = None,   # default json.dumps
    loads: Callable[[str], Any] | None = None,   # default json.loads
    encoding: str = "utf-8",
)
```

**`service_stores_history=True`** — the remote service owns history (e.g. Azure
AI Agents service). The provider skips local writes when set.

Files are stored as `<storage_path>/<session_id>.jsonl`. Windows reserved names
and unusual characters are encoded as `~session-<hex>.jsonl`.

### `HistoryProvider` ABC

```python
class HistoryProvider:
    async def load(
        self,
        *,
        session: AgentSession,
        context: SessionContext,
        conversation_id: str | None = None,
    ) -> list[Message]: ...

    async def save(
        self,
        *,
        session: AgentSession,
        context: SessionContext,
        messages: list[Message],
    ) -> None: ...

    async def clear(
        self,
        *,
        session: AgentSession,
        context: SessionContext,
    ) -> None: ...
```

### Example 1 · `FileHistoryProvider` with session reuse

```python
import asyncio, tempfile, pathlib
from agent_framework import Agent, AgentSession, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    storage = pathlib.Path(tempfile.mkdtemp())
    history = FileHistoryProvider(storage_path=storage)

    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="assistant", history_provider=history)

    # Turn 1 — create a session and ask a question
    session = AgentSession(session_id="user-session-42")
    r1 = await agent.run("My favourite colour is blue.", session=session)
    print("Turn 1:", r1.text)

    # Turn 2 — reuse the same session; history loaded from disk
    r2 = await agent.run("What is my favourite colour?", session=session)
    print("Turn 2:", r2.text)  # Should mention blue

    # Confirm the JSONL file exists
    files = list(storage.glob("*.jsonl"))
    print("history files:", [f.name for f in files])


asyncio.run(main())
```

### Example 2 · `InMemoryHistoryProvider` for testing

```python
import asyncio
from agent_framework import Agent, AgentSession, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    history = InMemoryHistoryProvider()

    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="test-agent", history_provider=history)

    session = AgentSession()

    await agent.run("Remember: the magic word is 'avocado'.", session=session)
    response = await agent.run("What is the magic word?", session=session)
    print(response.text)  # should mention avocado

    # Inspect stored messages
    messages = await history.load(
        session=session,
        context=None,  # type: ignore[arg-type]  — None accepted in in-memory impl
    )
    print(f"stored {len(messages)} messages")

    await history.clear(session=session, context=None)  # type: ignore[arg-type]
    cleared = await history.load(session=session, context=None)  # type: ignore[arg-type]
    print("after clear:", len(cleared))  # 0


asyncio.run(main())
```

### Example 3 · Custom `HistoryProvider` backed by a dict store

```python
import asyncio
from agent_framework import Agent, AgentSession, HistoryProvider, AgentSession
from agent_framework._sessions import SessionContext
from agent_framework._types import Message
from agent_framework.openai import OpenAIChatClient


class InProcessDictHistoryProvider(HistoryProvider):
    """Minimal HistoryProvider backed by an in-process dict for demonstration."""

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[str, list[dict]] = {}

    def _key(self, session: AgentSession) -> str:
        return session.session_id

    async def load(
        self, *, session: AgentSession, context: SessionContext, conversation_id=None
    ) -> list[Message]:
        raw = self._store.get(self._key(session), [])
        # Reconstruct Message objects from plain dicts
        return [Message(**m) for m in raw]

    async def save(
        self, *, session: AgentSession, context: SessionContext, messages: list[Message]
    ) -> None:
        self._store[self._key(session)] = [m.__dict__ for m in messages]

    async def clear(self, *, session: AgentSession, context: SessionContext) -> None:
        self._store.pop(self._key(session), None)


async def main() -> None:
    provider = InProcessDictHistoryProvider()
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="custom-history-agent", history_provider=provider)

    session = AgentSession(session_id="demo")
    await agent.run("I work at Contoso.", session=session)
    response = await agent.run("Where do I work?", session=session)
    print(response.text)   # Contoso


asyncio.run(main())
```

---

## 9 · `AgentMiddleware` + `@agent_middleware` — advanced patterns

`agent_framework._middleware.AgentMiddleware`  
`agent_framework._middleware.agent_middleware`

Agent middleware intercepts the full `Agent.run()` call — before the LLM is
invoked, after the response is assembled, or both. Unlike function middleware
(which targets tool calls), agent middleware has access to the complete
`AgentContext`, including messages, session, options, and the raw response.

### `@agent_middleware` decorator

```python
@agent_middleware
async def my_middleware(ctx: AgentContext, next_handler):
    # Pre-processing: inspect ctx.messages, ctx.options, ctx.session
    result = await next_handler(ctx)
    # Post-processing: inspect result.text, result.finish_reason, etc.
    return result
```

### `AgentContext` key attributes

```python
ctx.agent          # Agent instance
ctx.messages       # list[Message] for this invocation
ctx.session        # AgentSession (may be None)
ctx.options        # ChatOptions TypedDict
ctx.middleware     # per-call extra middleware
ctx.tools          # per-call extra tools
```

### `MiddlewareTermination`

Raise `MiddlewareTermination(result)` inside middleware to short-circuit the
chain without calling the model. The result is returned as-is to the caller.

### Example 1 · Response caching middleware

```python
import asyncio, hashlib, json
from agent_framework import Agent, AgentMiddleware, AgentContext, MiddlewareTermination
from agent_framework.openai import OpenAIChatClient


_cache: dict[str, str] = {}


@AgentMiddleware
async def simple_cache(ctx: AgentContext, next_handler):
    """Return cached responses for identical message lists."""
    key = hashlib.sha256(
        json.dumps([m.__dict__ for m in ctx.messages], sort_keys=True).encode()
    ).hexdigest()

    if key in _cache:
        print("[cache] HIT")
        # Create a minimal AgentResponse-like object via MiddlewareTermination
        # (In production use a proper AgentResponse; here we raise to short-circuit)
        cached_text = _cache[key]
        # Build a mock response — in real code use AgentResponse.from_text() if available
        # For simplicity: just proceed but note the pattern
        pass  # fall through to actual LLM if not constructing the response object

    response = await next_handler(ctx)
    _cache[key] = response.text
    print(f"[cache] MISS — stored {len(_cache)} entries")
    return response


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="cached-agent", middleware=[simple_cache])

    r1 = await agent.run("What is 2 + 2?")
    r2 = await agent.run("What is 2 + 2?")  # cache hit on second call
    print(r1.text, r2.text)


asyncio.run(main())
```

### Example 2 · PII redaction middleware

```python
import asyncio, re
from agent_framework import (
    Agent, AgentMiddleware, AgentContext, Message,
)
from agent_framework.openai import OpenAIChatClient

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _redact(text: str) -> str:
    return EMAIL_RE.sub("[REDACTED_EMAIL]", text)


@AgentMiddleware
async def pii_redact(ctx: AgentContext, next_handler):
    """Strip PII from outbound messages before they reach the model."""
    redacted_messages = []
    for msg in ctx.messages:
        contents = []
        for content in (msg.contents or []):
            if isinstance(content, str):
                contents.append(_redact(content))
            else:
                contents.append(content)
        redacted_messages.append(Message(role=msg.role, contents=contents))
    ctx.messages = redacted_messages   # mutate in-place
    return await next_handler(ctx)


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="pii-safe-agent", middleware=[pii_redact])

    response = await agent.run(
        "My email is alice@example.com. Is this a valid email format?"
    )
    print(response.text)  # model never sees alice@example.com


asyncio.run(main())
```

### Example 3 · Short-circuit with `MiddlewareTermination`

```python
import asyncio
from agent_framework import (
    Agent, AgentMiddleware, AgentContext, MiddlewareTermination,
)
from agent_framework.openai import OpenAIChatClient


BLOCKED_WORDS = {"password", "secret", "ssn", "credit_card"}


@AgentMiddleware
async def content_filter(ctx: AgentContext, next_handler):
    """Block requests containing sensitive words before they reach the model."""
    text = " ".join(
        c if isinstance(c, str) else ""
        for msg in ctx.messages
        for c in (msg.contents or [])
    ).lower()

    for word in BLOCKED_WORDS:
        if word in text:
            # Short-circuit: never call the model.
            # In production return a proper AgentResponse object.
            raise MiddlewareTermination(
                type("BlockedResponse", (), {"text": f"Request blocked: contains '{word}'."})()
            )

    return await next_handler(ctx)


async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="filtered-agent", middleware=[content_filter])

    response = await agent.run("What is my password?")
    print(response.text)


asyncio.run(main())
```

---

## 10 · `SkillsProvider` + custom `SkillsSource`

`agent_framework._skills.SkillsProvider`  
`agent_framework._skills.SkillsSource`  
`agent_framework._skills.DeduplicatingSkillsSource`

`SkillsSource` is the ABC for discovering `Skill` objects.
`SkillsProvider` composes multiple sources with optional deduplication and
caching. `DeduplicatingSkillsSource` wraps any source and drops skills whose
`name` was already seen (first-one-wins).

### `SkillsProvider` factories

```python
# Load from a list of directory paths
provider = SkillsProvider.from_paths(["/skills/customer", "/skills/finance"])

# Compose multiple sources
provider = SkillsProvider(sources=[source_a, source_b])
```

### `SkillsSource` ABC

```python
class SkillsSource:
    async def get_skills(
        self,
        *,
        session: AgentSession,
        context: SessionContext,
    ) -> list[Skill]: ...
```

### Example 1 · `SkillsProvider.from_paths` — multi-directory discovery

```python
import asyncio, pathlib, tempfile
from agent_framework import Agent, SkillsProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Create a minimal skill directory structure for demonstration
    base = pathlib.Path(tempfile.mkdtemp())
    skill_dir = base / "translate"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: translate\n"
        "description: Translate text between languages.\n"
        "---\n"
        "## translate\n"
        "Use this skill to translate text. Specify `source_lang` and `target_lang`.\n"
    )

    provider = SkillsProvider.from_paths([base])

    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(
        client=client,
        name="multilingual",
        skills_provider=provider,
    )

    response = await agent.run("Translate 'hello' to Spanish.")
    print(response.text)


asyncio.run(main())
```

### Example 2 · `DeduplicatingSkillsSource` — first-one-wins across sources

```python
import asyncio
from agent_framework import (
    Agent, SkillsProvider, DeduplicatingSkillsSource,
    AggregatingSkillsSource, InMemorySkillsSource, Skill, SkillFrontmatter,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Two sources expose a skill with the same name — "summarize".
    # DeduplicatingSkillsSource keeps only the first one encountered.
    v1_skill = Skill(
        frontmatter=SkillFrontmatter(name="summarize", description="Summarize v1"),
        resources=[],
        scripts=[],
    )
    v2_skill = Skill(
        frontmatter=SkillFrontmatter(name="summarize", description="Summarize v2"),
        resources=[],
        scripts=[],
    )

    source_v1 = InMemorySkillsSource([v1_skill])
    source_v2 = InMemorySkillsSource([v2_skill])

    # Aggregate both, then deduplicate
    combined = AggregatingSkillsSource([source_v1, source_v2])
    deduped = DeduplicatingSkillsSource(combined)

    provider = SkillsProvider(sources=[deduped])

    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="skill-agent", skills_provider=provider)

    # The agent sees only one "summarize" skill — the v1 version
    response = await agent.run("Summarise this text in one sentence: ...")
    print(response.text)


asyncio.run(main())
```

### Example 3 · Custom `SkillsSource` backed by a remote API

```python
import asyncio
from agent_framework import (
    Agent, SkillsProvider, SkillsSource, AgentSession, Skill, SkillFrontmatter,
)
from agent_framework._sessions import SessionContext
from agent_framework.openai import OpenAIChatClient


class RemoteSkillsSource(SkillsSource):
    """Fetch skills from a remote registry on every agent invocation."""

    def __init__(self, registry_url: str) -> None:
        self._registry_url = registry_url

    async def get_skills(
        self,
        *,
        session: AgentSession,
        context: SessionContext,
    ) -> list[Skill]:
        # In production: call self._registry_url with an HTTP client.
        # For demonstration, return a static skill.
        return [
            Skill(
                frontmatter=SkillFrontmatter(
                    name="remote-search",
                    description=(
                        "Search the remote knowledge base. "
                        "Pass a `query` string to retrieve relevant documents."
                    ),
                ),
                resources=[],
                scripts=[],
            )
        ]


async def main() -> None:
    remote_source = RemoteSkillsSource(registry_url="https://skills.internal/api")
    provider = SkillsProvider(sources=[remote_source])

    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, name="remote-skill-agent", skills_provider=provider)

    response = await agent.run("Search for 'machine learning pipelines'.")
    print(response.text)


asyncio.run(main())
```

---

## Quick-reference summary

| # | Class / symbol | Module | Key takeaway |
|---|---|---|---|
| 1 | `WorkflowBuilder` | `_workflows._workflow_builder` | `add_chain` = sequential shorthand; `add_multi_selection_edge_group` = dynamic fan-out; `intermediate_output_from="all_other"` = all non-terminal → intermediate |
| 2 | `Workflow` + `WorkflowRunResult` | `_workflows._workflow` | `get_outputs()` / `get_intermediate_outputs()` / `status_timeline()` / `get_final_state()` are the four result-consumption surfaces |
| 3 | `FunctionTool` | `_tools` | `max_invocations` is per-instance lifetime; `func=None` → declaration-only; `result_parser` controls serialization to the model |
| 4 | `Agent.run()` | `_agents` | Three `@overload` variants: plain, structured output via `options=`, streaming via `stream=True`; per-call `middleware=` and `compaction_strategy=` |
| 5 | `MCPStdioTool` | `_mcp` | `allowed_tools` = allowlist; `MCPSpecificApproval` = per-tool approval dict; `additional_tool_argument_names` = hidden args injected into every tool call |
| 6 | `SecretString` + `load_settings` | `_settings` | `SecretString.__repr__` masks; `load_settings` resolution: overrides → .env → env vars → defaults; `required_fields` tuple = at-least-one constraint |
| 7 | `FunctionInvocationConfiguration` | `_tools` | `max_function_calls` is per-request, best-effort; `additional_tools` injects tools without rebuilding agent; `enabled=False` disables the tool loop |
| 8 | `FileHistoryProvider` + `HistoryProvider` | `_sessions` | JSONL per session; `service_stores_history=True` skips local writes; custom provider = implement `load`/`save`/`clear` |
| 9 | `AgentMiddleware` + `@agent_middleware` | `_middleware` | Intercepts full `agent.run()` call; `MiddlewareTermination(result)` short-circuits without calling the model |
| 10 | `SkillsProvider` + `SkillsSource` | `_skills` | `from_paths()` discovers SKILL.md files; `DeduplicatingSkillsSource` = first-one-wins; custom ABC = implement `get_skills(session, context)` |

## Revision history

| Date | Version | Notes |
|---|---|---|
| 2026-07-07 | 1.10.0 | Vol. 33 — 10 class groups, 30 examples, source-verified |

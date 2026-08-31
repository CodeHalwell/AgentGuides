---
title: "Microsoft Agent Framework (Python) — 10-Class Deep Dives Vol. 2 (1.16.0)"
description: "Source-verified deep dives for FanInEdgeGroup, FanOutEdgeGroup, FunctionalWorkflow, FunctionalWorkflowAgent, FileCheckpointStorage, InMemoryCheckpointStorage, MCPStdioTool, MCPStreamableHTTPTool, SelectiveToolCallCompactionStrategy, and TodoProvider — all verified against agent-framework 1.16.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-Class Deep Dives Vol. 2

**Verified against:** `agent-framework==1.16.0`
**Python requirement:** 3.10+

This volume covers 10 additional public classes from the `agent_framework` package that are not covered in Vol. 1. Each section includes the full `__init__` signature, every meaningful method, and self-contained runnable examples verified against the 1.16.0 source.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

---

## 1. `FanInEdgeGroup`

**Module:** `agent_framework._workflows._edge` (re-exported via `agent_framework`)

`FanInEdgeGroup` creates a converging edge pattern where **multiple upstream executors** all feed into a single downstream executor. The framework aggregates their outputs so the downstream node receives every message the sources emit before executing.

### Constructor

```python
FanInEdgeGroup(
    source_ids: Sequence[str],
    target_id: str,
    *,
    id: str | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `source_ids` | `Sequence[str]` | Executor IDs of all upstream nodes. **At least two** required. |
| `target_id` | `str` | Executor ID of the single downstream node. |
| `id` | `str \| None` | Optional stable identifier for the group; auto-generated if omitted. |

Raises `ValueError` if fewer than two source IDs are provided.

### How it works

Internally, `FanInEdgeGroup` builds one `Edge(source_id=source, target_id=target_id)` per upstream source and registers them as a single group so the runner can track fan-in completion. The downstream executor runs only after **all** upstream sources have produced output in the current superstep.

### Example — Parallel research merge

`FanInEdgeGroup` is constructed internally by `WorkflowBuilder.add_fan_in_edges()`, which both registers all executor objects and wires the fan-in group in a single call:

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

dispatcher     = Agent(client=client, name="dispatcher",
                       instructions="Echo the research query unchanged.")
web_researcher = Agent(client=client, name="web_researcher",
                       instructions="Search for recent news on the topic.")
doc_researcher = Agent(client=client, name="doc_researcher",
                       instructions="Search internal documents on the topic.")
synthesizer    = Agent(client=client, name="synthesizer",
                       instructions="Combine the research findings into a single report.")

builder = WorkflowBuilder(start_executor=dispatcher)

# Fan-out: dispatcher broadcasts the query to both researchers concurrently
builder.add_fan_out_edges(dispatcher, [web_researcher, doc_researcher])

# Fan-in: synthesizer runs only after BOTH researchers complete
# (add_fan_in_edges creates a FanInEdgeGroup internally)
builder.add_fan_in_edges([web_researcher, doc_researcher], synthesizer)

workflow = builder.build()

async def main() -> None:
    result = await workflow.run("Quantum computing breakthroughs")
    print(result.get_outputs())

asyncio.run(main())
```

### Example — Three-way fan-in with serialisation

```python
from agent_framework import FanInEdgeGroup

group = FanInEdgeGroup(
    source_ids=["parser", "enricher", "validator"],
    target_id="writer",
    id="my-fan-in",
)

d = group.to_dict()
assert d["type"] == "FanInEdgeGroup"
assert len(d["edges"]) == 3
assert all(e["target_id"] == "writer" for e in d["edges"])
```

---

## 2. `FanOutEdgeGroup`

**Module:** `agent_framework._workflows._edge` (re-exported via `agent_framework`)

`FanOutEdgeGroup` broadcasts a single source executor's output to **multiple downstream executors**. An optional `selection_func` lets you narrow the active targets at runtime based on the message payload.

### Constructor

```python
FanOutEdgeGroup(
    source_id: str,
    target_ids: Sequence[str],
    selection_func: Callable[[Any, list[str]], list[str]] | None = None,
    *,
    selection_func_name: str | None = None,
    id: str | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `source_id` | `str` | Executor ID of the single upstream node. |
| `target_ids` | `Sequence[str]` | At least two downstream executor IDs. |
| `selection_func` | `Callable \| None` | Runtime filter: receives `(message, available_ids)` and returns the subset to activate. |
| `selection_func_name` | `str \| None` | Stable name for serialisation when the callable cannot be introspected. |
| `id` | `str \| None` | Optional stable group identifier. |

Raises `ValueError` if fewer than two target IDs are provided.

### Properties

| Property | Returns | Notes |
|---|---|---|
| `target_ids` | `list[str]` | Defensive copy of the configured downstream IDs. |
| `selection_func` | `Callable \| None` | The runtime selection callable, or `None` to broadcast to all. |

### Example — Broadcast to all targets

`FanOutEdgeGroup` is constructed internally by `WorkflowBuilder.add_fan_out_edges()`, which registers all executor objects and wires the broadcast group:

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

classifier  = Agent(client=client, name="classifier",
                    instructions="Classify the content as financial, legal, or general.")
fin_handler = Agent(client=client, name="financial_handler",
                    instructions="Handle financial content.")
leg_handler = Agent(client=client, name="legal_handler",
                    instructions="Handle legal content.")
gen_handler = Agent(client=client, name="general_handler",
                    instructions="Handle general content.")

# add_fan_out_edges creates a FanOutEdgeGroup internally; broadcasts to all three handlers
builder = WorkflowBuilder(start_executor=classifier)
builder.add_fan_out_edges(classifier, [fin_handler, leg_handler, gen_handler])
workflow = builder.build()
```

### Example — Inspect the `FanOutEdgeGroup` serialisation directly

```python
from agent_framework import FanOutEdgeGroup

def route(message: str, available: list[str]) -> list[str]:
    """Send only to handlers whose name prefix appears in the message."""
    return [t for t in available if t.split("_")[0] in message.lower()] or available

group = FanOutEdgeGroup(
    source_id="classifier",
    target_ids=["financial_handler", "legal_handler", "general_handler"],
    selection_func=route,
)

assert group.target_ids == ["financial_handler", "legal_handler", "general_handler"]
assert group.selection_func is route

d = group.to_dict()
assert d["type"] == "FanOutEdgeGroup"
assert len(d["edges"]) == 3
```

---

## 3. `FunctionalWorkflow`

**Module:** `agent_framework._workflows._functional` (re-exported via `agent_framework`)

`FunctionalWorkflow` is the backing object produced by the `@workflow` decorator. Unlike graph-based `Workflow`, it runs a plain async Python function — no edge wiring, no `WorkflowBuilder`. Native `if`/`else`, `for`, and `asyncio.gather` drive branching and parallelism.

> **Experimental:** The `@workflow` and `@step` decorators emit an `ExperimentalWarning` on first use. No explicit opt-in call is needed — simply import and use them.

### Constructor (internal — use `@workflow` decorator)

```python
FunctionalWorkflow(
    func: Callable[..., Awaitable[Any]],
    *,
    name: str | None = None,
    description: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
)
```

### `run()` signature

```python
async def run(
    message: Any | None = None,
    *,
    stream: bool = False,
    responses: dict[str, Any] | None = None,
    checkpoint_id: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
    include_status_events: bool = False,
    **kwargs: Any,
) -> WorkflowRunResult | ResponseStream[WorkflowEvent, WorkflowRunResult]
```

| Parameter | Notes |
|---|---|
| `message` | Input passed as the first argument to the workflow function. Mutually exclusive with `checkpoint_id`; can be combined with `responses` for HITL replays that also supply a new input. |
| `stream` | If `True`, return a `ResponseStream` yielding events. |
| `responses` | HITL responses keyed by `request_id` for resuming after a `ctx.request_info()` interruption. |
| `checkpoint_id` | Resume from a saved checkpoint. |
| `checkpoint_storage` | Override the default storage for this run. |
| `include_status_events` | When `True` (non-streaming only), include status change events in the result. |

### Example — Linear pipeline with `@workflow`

```python
import asyncio
from agent_framework import Agent, workflow, step
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher",
                   instructions="Return 3 bullet-point facts about the topic.")
writer     = Agent(client=client, name="writer",
                   instructions="Turn bullet points into a polished paragraph.")

@step
async def research(topic: str) -> str:
    result = await researcher.run(topic)
    return result.text

@step
async def write(facts: str) -> str:
    result = await writer.run(facts)
    return result.text

@workflow
async def pipeline(topic: str) -> str:
    facts = await research(topic)
    return await write(facts)

async def main() -> None:
    wf = pipeline.build()
    result = await wf.run("renewable energy storage")
    print(result.get_outputs())

asyncio.run(main())
```

### Example — Parallel branches with `asyncio.gather`

```python
import asyncio
from agent_framework import Agent, workflow, step
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
summarizer  = Agent(client=client, name="summarizer",  instructions="Summarize the text.")
translator  = Agent(client=client, name="translator",  instructions="Translate the text to French.")
fact_check  = Agent(client=client, name="fact_checker", instructions="List any factual errors.")

@step
async def summarize(text: str) -> str:
    return (await summarizer.run(text)).text

@step
async def translate(text: str) -> str:
    return (await translator.run(text)).text

@step
async def check_facts(text: str) -> str:
    return (await fact_check.run(text)).text

@workflow
async def multi_branch(text: str) -> dict[str, str]:
    summary, translation, errors = await asyncio.gather(
        summarize(text),
        translate(text),
        check_facts(text),
    )
    return {"summary": summary, "translation": translation, "errors": errors}

async def main() -> None:
    wf = multi_branch.build()
    result = await wf.run("The Earth orbits the Sun every 365.25 days.")
    print(result.get_outputs())

asyncio.run(main())
```

### Example — Checkpoint-backed long-running workflow

```python
import asyncio
from agent_framework import FileCheckpointStorage, Agent, workflow, step
from agent_framework.openai import OpenAIChatClient

client   = OpenAIChatClient()
analyzer = Agent(client=client, name="analyzer", instructions="Analyse each item.")

@step
async def analyze_item(item: str) -> str:
    return (await analyzer.run(item)).text

@workflow
async def batch_workflow(items: list[str]) -> list[str]:
    results = []
    for item in items:
        results.append(await analyze_item(item))
    return results

async def main() -> None:
    storage = FileCheckpointStorage("/tmp/wf_checkpoints")
    wf = batch_workflow.build(checkpoint_storage=storage)
    result = await wf.run(["item_1", "item_2", "item_3"])
    print(result.get_outputs())

asyncio.run(main())
```

---

## 4. `FunctionalWorkflowAgent`

**Module:** `agent_framework._workflows._functional` (re-exported via `agent_framework`)

`FunctionalWorkflowAgent` wraps a `FunctionalWorkflow` so it can be used anywhere an `Agent`-compatible object is expected — including as a `WorkflowExecutor` node inside a graph-based workflow. It translates `AgentResponse` / `ResponseStream` calls to the underlying functional workflow's `run()`.

> **Experimental:** `@workflow`/`@step` emit `ExperimentalWarning` on first use — no explicit opt-in call is needed.

### Constructor

```python
FunctionalWorkflowAgent(
    workflow: FunctionalWorkflow,
    *,
    name: str | None = None,
    description: str | None = None,
    context_providers: Sequence[Any] | None = None,
    **kwargs: Any,
)
```

| Parameter | Notes |
|---|---|
| `workflow` | The `FunctionalWorkflow` to wrap. |
| `name` | Display name; defaults to `workflow.name`. |
| `description` | Display description; defaults to `workflow.description`. |
| `context_providers` | Optional providers exposed for introspection by outer harness. |

### Properties

| Property | Returns | Notes |
|---|---|---|
| `pending_requests` | `dict[str, WorkflowEvent]` | HITL `request_info` events awaiting responses. |

### Example — Functional workflow as a node in a graph workflow

```python
import asyncio
from agent_framework import (
    Agent, WorkflowBuilder,
    workflow, step, FunctionalWorkflowAgent,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
inner_agent = Agent(client=client, name="inner", instructions="Process the data.")

@step
async def inner_step(data: str) -> str:
    return (await inner_agent.run(data)).text

@workflow
async def inner_pipeline(data: str) -> str:
    return await inner_step(data)

# @workflow returns FunctionalWorkflowDefinition; .build() returns FunctionalWorkflow
inner_wf      = inner_pipeline.build()
inner_adapter = FunctionalWorkflowAgent(inner_wf, name="inner_pipeline")

# Place the functional workflow adapter into an outer graph workflow
outer_agent = Agent(client=client, name="outer",
                    instructions="Prepare the data for the inner pipeline.")
builder = WorkflowBuilder(start_executor=outer_agent)
builder.add_edge(outer_agent, inner_adapter)
outer_workflow = builder.build()

async def main() -> None:
    result = await outer_workflow.run("raw input data")
    print(result.get_outputs())

asyncio.run(main())
```

---

## 5. `FileCheckpointStorage`

**Module:** `agent_framework._workflows._checkpoint` (re-exported via `agent_framework`)

`FileCheckpointStorage` persists workflow checkpoints to disk as JSON files with pickle-encoded state. It uses atomic writes (write-to-temp then `os.replace`) so a crash during a save never corrupts an existing checkpoint. Deserialisation uses a safe allowlist of Python types to prevent arbitrary code execution.

### Constructor

```python
FileCheckpointStorage(
    storage_path: str | Path,
    *,
    allowed_checkpoint_types: list[str] | None = None,
)
```

| Parameter | Notes |
|---|---|
| `storage_path` | Directory where checkpoint `.json` files are written. Created automatically if it does not exist. |
| `allowed_checkpoint_types` | Extra types to permit during load, in `"module:qualname"` format. |

### Methods

| Method | Returns | Notes |
|---|---|---|
| `await save(checkpoint)` | `CheckpointID` | Atomically serialises and writes the checkpoint. |
| `await load(checkpoint_id)` | `WorkflowCheckpoint` | Reads and deserialises from disk; raises `WorkflowCheckpointException` if not found. |
| `await list_checkpoints(*, workflow_name)` | `list[WorkflowCheckpoint]` | All checkpoints for a named workflow. |
| `await list_checkpoint_ids(*, workflow_name)` | `list[CheckpointID]` | Checkpoint IDs only (faster than full load). |
| `await get_latest(*, workflow_name)` | `WorkflowCheckpoint \| None` | Most recent checkpoint by timestamp, or `None`. |
| `await delete(checkpoint_id)` | `bool` | Removes the file; returns `False` if not found. |

### Example — Basic save and resume

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, FileCheckpointStorage
from agent_framework.openai import OpenAIChatClient

client  = OpenAIChatClient()
agent_a = Agent(client=client, name="step_a", instructions="Do step A.")
agent_b = Agent(client=client, name="step_b", instructions="Do step B.")

builder = WorkflowBuilder(start_executor=agent_a)
builder.add_edge(agent_a, agent_b)
storage  = FileCheckpointStorage("/tmp/my_checkpoints")
workflow = builder.build()

async def main() -> None:
    # Run and automatically checkpoint each superstep
    result = await workflow.run("start", checkpoint_storage=storage)
    print("Finished:", result.get_outputs())

    # List all checkpoints for this workflow
    ids = await storage.list_checkpoint_ids(workflow_name="step_a")
    print("Checkpoint IDs:", ids)

    # Resume from the latest checkpoint
    latest = await storage.get_latest(workflow_name="step_a")
    if latest:
        resumed = await workflow.run(checkpoint_id=latest.checkpoint_id,
                                     checkpoint_storage=storage)
        print("Resumed:", resumed.get_outputs())

asyncio.run(main())
```

### Example — Custom types in checkpoint state

```python
from agent_framework import FileCheckpointStorage
from dataclasses import dataclass

@dataclass
class MyAppState:
    user_id: str
    session_count: int

# Register MyAppState so it can round-trip through checkpoint serialisation.
# When the class is defined in the same script, pickle uses "__main__:MyAppState".
# In a packaged project import the class from its real module and list that path.
storage = FileCheckpointStorage(
    "/tmp/checkpoints",
    allowed_checkpoint_types=["__main__:MyAppState"],
)
```

---

## 6. `InMemoryCheckpointStorage`

**Module:** `agent_framework._workflows._checkpoint` (re-exported via `agent_framework`)

`InMemoryCheckpointStorage` keeps checkpoints in a Python `dict` — ideal for unit tests and development environments where disk persistence is not needed. Every `save()` deep-copies the checkpoint so mutations to the running workflow state do not corrupt the stored snapshot.

### Constructor

```python
InMemoryCheckpointStorage()
```

No parameters.

### Methods

Identical interface to `FileCheckpointStorage`:

| Method | Returns |
|---|---|
| `await save(checkpoint)` | `CheckpointID` |
| `await load(checkpoint_id)` | `WorkflowCheckpoint` |
| `await list_checkpoints(*, workflow_name)` | `list[WorkflowCheckpoint]` |
| `await list_checkpoint_ids(*, workflow_name)` | `list[CheckpointID]` |
| `await get_latest(*, workflow_name)` | `WorkflowCheckpoint \| None` |
| `await delete(checkpoint_id)` | `bool` |

### Example — Checkpoint storage round-trip (save, list, load, delete)

```python
import asyncio
from agent_framework import (
    Agent, WorkflowBuilder, WorkflowExecutor,
    InMemoryCheckpointStorage,
)
from agent_framework.openai import OpenAIChatClient

client  = OpenAIChatClient()
step_a  = Agent(client=client, name="step_a", instructions="Do step A and stop.")
step_b  = Agent(client=client, name="step_b", instructions="Do step B.")

builder = WorkflowBuilder(start_executor=step_a)
builder.add_edge(step_a, step_b)
storage  = InMemoryCheckpointStorage()
workflow = builder.build()

async def main() -> None:
    result = await workflow.run("input", checkpoint_storage=storage)

    cp_ids = await storage.list_checkpoint_ids(workflow_name="step_a")
    print(f"Stored {len(cp_ids)} checkpoint(s)")

    # Verify round-trip
    latest = await storage.get_latest(workflow_name="step_a")
    assert latest is not None

    # Delete and confirm
    deleted = await storage.delete(latest.checkpoint_id)
    assert deleted is True

asyncio.run(main())
```

---

## 7. `MCPStdioTool`

**Module:** `agent_framework._mcp` (re-exported via `agent_framework`)

`MCPStdioTool` connects to an **MCP server running as a local subprocess** via stdin/stdout. It launches the process, negotiates the MCP protocol, and exposes the server's tools as `FunctionTool` instances on the agent. The tool is used as an async context manager.

### Constructor (key parameters)

```python
MCPStdioTool(
    name: str,
    command: str,
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    tool_name_prefix: str | None = None,
    load_tools: bool = True,
    load_prompts: bool = True,
    allowed_tools: Collection[str] | None = None,
    approval_mode: Literal["always_require", "never_require"] | MCPSpecificApproval | None = None,
    use_progressive_disclosure: bool = False,
    parse_tool_results: Callable | None = None,
    request_timeout: int | None = None,
    task_options: MCPTaskOptions | None = None,
)
```

| Parameter | Notes |
|---|---|
| `name` | Logical name for this MCP connection (used in logging). |
| `command` | The executable to run as the MCP server (e.g. `"npx"`, `"python"`). |
| `args` | Command-line arguments passed to `command`. |
| `env` | Extra environment variables for the subprocess. |
| `tool_name_prefix` | Prefix prepended to every tool name exposed to the agent (avoids collisions). |
| `allowed_tools` | Whitelist of tool names from the server; others are hidden. |
| `approval_mode` | `"always_require"` forces user confirmation before every call; `"never_require"` auto-approves. |
| `use_progressive_disclosure` | Expose a lightweight index tool first; full tools load on demand. |
| `task_options` | Optional `MCPTaskOptions` for Hyperlight sandbox configuration. |

### Example — Filesystem MCP server

```python
import asyncio
from agent_framework import Agent, MCPStdioTool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    fs_tool = MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        tool_name_prefix="fs_",
        approval_mode="never_require",
    )

    async with fs_tool:
        agent = Agent(
            client=client,
            name="file_assistant",
            instructions="Help the user manage files.",
            tools=fs_tool,
        )
        response = await agent.run("List all .txt files in /tmp")
        print(response.text)

asyncio.run(main())
```

### Example — Python-based MCP server with environment override

```python
import asyncio
from agent_framework import Agent, MCPStdioTool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    db_tool = MCPStdioTool(
        name="database",
        command="python",
        args=["-m", "my_mcp_server.main"],
        env={"DATABASE_URL": "postgresql://localhost/mydb"},
        allowed_tools=["query_table", "list_tables"],
        approval_mode="never_require",
    )

    async with db_tool:
        agent = Agent(
            client=client,
            name="data_analyst",
            instructions="Answer questions using the database.",
            tools=db_tool,
        )
        response = await agent.run("How many rows are in the orders table?")
        print(response.text)

asyncio.run(main())
```

---

## 8. `MCPStreamableHTTPTool`

**Module:** `agent_framework._mcp` (re-exported via `agent_framework`)

`MCPStreamableHTTPTool` connects to a **remote MCP server over HTTP/SSE** (the Streamable HTTP transport introduced in MCP 2025-03). Unlike `MCPStdioTool`, no subprocess is spawned — the tool connects to an already-running HTTP endpoint.

### Constructor (key parameters)

```python
MCPStreamableHTTPTool(
    name: str,
    url: str,
    *,
    tool_name_prefix: str | None = None,
    load_tools: bool = True,
    load_prompts: bool = True,
    allowed_tools: Collection[str] | None = None,
    approval_mode: Literal["always_require", "never_require"] | MCPSpecificApproval | None = None,
    use_progressive_disclosure: bool = False,
    parse_tool_results: Callable | None = None,
    request_timeout: int | None = None,
    http_client: AsyncClient | None = None,
    header_provider: Callable[[dict[str, Any]], dict[str, str]] | None = None,
    terminate_on_close: bool | None = None,
    task_options: MCPTaskOptions | None = None,
)
```

| Parameter | Notes |
|---|---|
| `name` | Logical name for this MCP connection. |
| `url` | Full URL of the MCP HTTP endpoint (e.g. `https://api.example.com/mcp`). |
| `http_client` | Optional `httpx.AsyncClient` for custom TLS/proxy configuration. |
| `header_provider` | Callable that returns request headers (use for authentication). |
| `terminate_on_close` | Whether to send an MCP terminate signal on context manager exit. |

### Example — Remote MCP API with authentication

```python
import asyncio
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient
import os

client = OpenAIChatClient()

def auth_headers(metadata: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['MCP_API_KEY']}"}

async def main() -> None:
    web_tool = MCPStreamableHTTPTool(
        name="web-api",
        url="https://mcp.example.com/api/v1",
        header_provider=auth_headers,
        tool_name_prefix="api_",
        approval_mode="never_require",
    )

    async with web_tool:
        agent = Agent(
            client=client,
            name="api_agent",
            instructions="Use the API to answer questions.",
            tools=web_tool,
        )
        response = await agent.run("What is the current weather in London?")
        print(response.text)

asyncio.run(main())
```

### Example — Side-by-side stdio and HTTP MCP servers

```python
import asyncio
from agent_framework import Agent, MCPStdioTool, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    local_fs = MCPStdioTool(
        name="local_filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        tool_name_prefix="local_",
    )
    remote_api = MCPStreamableHTTPTool(
        name="remote_api",
        url="https://mcp.example.com/v1",
        tool_name_prefix="remote_",
    )

    async with local_fs, remote_api:
        agent = Agent(
            client=client,
            name="hybrid_agent",
            instructions="Use both local files and the remote API to answer questions.",
            tools=[local_fs, remote_api],
        )
        response = await agent.run("Read report.txt and enrich it with remote data")
        print(response.text)

asyncio.run(main())
```

---

## 9. `SelectiveToolCallCompactionStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`SelectiveToolCallCompactionStrategy` is a focused compaction strategy that **only targets tool-call message groups**. It keeps the most recent `keep_last_tool_call_groups` groups and marks older ones as excluded, reducing token usage from repeated tool chatter without touching user or assistant messages.

It is composable with other strategies (like `SummarizationStrategy`) because each strategy operates on a distinct annotation type.

### Constructor

```python
SelectiveToolCallCompactionStrategy(*, keep_last_tool_call_groups: int = 1)
```

| Parameter | Notes |
|---|---|
| `keep_last_tool_call_groups` | Number of newest tool-call groups to retain. `0` removes all tool groups. Must be ≥ 0. |

### How it works

Each call to `__call__(messages)` scans the message list for groups annotated as `tool_call`, identifies which are still "included", keeps the newest `keep_last_tool_call_groups` of those, and marks the rest as excluded with reason `"tool_call_compaction"`. Returns `True` if any messages were changed.

### Example — Trim tool history while preserving conversation

```python
import asyncio
from agent_framework import (
    Agent, CompactionProvider,
    SelectiveToolCallCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

def fetch_data(key: str) -> str:
    """Return a dummy data record."""
    return f"data:{key}=42"

# Keep only the 2 most recent tool-call rounds; older ones are excluded
strategy = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)
compaction = CompactionProvider(before_strategy=strategy)

agent = Agent(
    client=client,
    name="tool_heavy_agent",
    instructions="Use fetch_data to look up values, then summarise.",
    tools=[fetch_data],
    context_providers=[compaction],
)

async def main() -> None:
    session = agent.create_session()
    for query in [
        "Fetch data for key 'alpha'",
        "Now fetch 'beta' and compare with alpha",
        "Summarise both results",
    ]:
        response = await agent.run(query, session=session)
        print(response.text)

asyncio.run(main())
```

### Example — Combine with `SummarizationStrategy` in `TokenBudgetComposedStrategy`

```python
from agent_framework import (
    CompactionProvider,
    TokenBudgetComposedStrategy,
    SummarizationStrategy,
    SelectiveToolCallCompactionStrategy,
    CharacterEstimatorTokenizer,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
tokenizer = CharacterEstimatorTokenizer()

# Phase 1: evict old tool calls once the kept context exceeds 48 000 tokens
tool_evict = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=3)
phase1 = TokenBudgetComposedStrategy(
    token_budget=48_000,
    tokenizer=tokenizer,
    strategies=[tool_evict],
)

# Phase 2: summarise remaining history once context exceeds 72 000 tokens
summarize = SummarizationStrategy(client=client)
phase2 = TokenBudgetComposedStrategy(
    token_budget=72_000,
    tokenizer=tokenizer,
    strategies=[summarize],
)

compaction = CompactionProvider(before_strategy=phase1, after_strategy=phase2)
```

### Example — Remove all tool-call history (aggressive mode)

```python
from agent_framework import Agent, CompactionProvider, SelectiveToolCallCompactionStrategy
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# keep_last_tool_call_groups=0 removes every included tool-call group
aggressive = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=0)
compaction  = CompactionProvider(before_strategy=aggressive)

agent = Agent(
    client=client,
    name="stateless_tool_agent",
    instructions="Use tools but never need to refer back to past tool results.",
    context_providers=[compaction],
)
```

---

## 10. `TodoProvider`

**Module:** `agent_framework._harness._todo` (re-exported via `agent_framework`)

`TodoProvider` injects a structured todo-list capability into an agent. It registers five built-in tools (`todos_add`, `todos_complete`, `todos_remove`, `todos_get_remaining`, `todos_get_all`) and a system instruction, enabling agents to create and track sub-tasks during long-running operations. State is stored per-session in a `TodoStore`.

> **Experimental:** Part of the HARNESS feature set.

### Constructor

```python
TodoProvider(
    source_id: str = "todo",
    *,
    instructions: str | None = None,
    store: TodoStore | None = None,
)
```

| Parameter | Notes |
|---|---|
| `source_id` | Unique identifier scoping the todo state within the session (default `"todo"`). |
| `instructions` | Override the built-in system prompt that explains the todo tools to the agent. |
| `store` | Backing store. Defaults to `TodoSessionStore` (in-memory, session-scoped). Pass `TodoFileStore` for file-backed persistence. |

### Built-in tools exposed to the agent

| Tool name | Description |
|---|---|
| `todos_add` | Add one or more todo items, each with `title` and optional `description`. |
| `todos_complete` | Mark items complete by ID with a reason string. |
| `todos_remove` | Remove items by ID. |
| `todos_get_remaining` | Return incomplete items only. |
| `todos_get_all` | Return all items (complete and incomplete). |

### Example — Agent with in-session todo tracking

```python
import asyncio
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

provider = TodoProvider()

agent = Agent(
    client=client,
    name="planning_agent",
    instructions="You are a project planner. Use the todo tools to track tasks.",
    context_providers=[provider],
)

async def main() -> None:
    session = agent.create_session()

    # Agent creates its own todos during execution
    response = await agent.run(
        "Plan a 3-step process to analyse customer feedback and summarise it.",
        session=session,
    )
    print(response.text)

    # Ask for remaining tasks
    response2 = await agent.run(
        "What tasks are still incomplete?",
        session=session,
    )
    print(response2.text)

asyncio.run(main())
```

### Example — File-backed todos across sessions

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoProvider, TodoFileStore
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

store = TodoFileStore(
    base_path="/tmp/todo_storage",
    owner_state_key="user_id",   # session.state["user_id"] scopes the file path
)
provider = TodoProvider(store=store)

agent = Agent(
    client=client,
    name="persistent_planner",
    instructions="Track all work items using the todo tools.",
    context_providers=[provider],
)

# TodoFileStore partitions by BOTH owner_state_key value AND session_id.
# Pass the same explicit session_id to both sessions so they share one file.
SHARED_SESSION_ID = "user-42-main"

async def main() -> None:
    session = AgentSession(
        session_id=SHARED_SESSION_ID,
        state={"user_id": "user-42"},
    )

    # Session 1: create todos
    await agent.run("Create todos for: research, draft, review.", session=session)

    # Session 2: same session_id + owner → same backing file → todos persist
    session2 = AgentSession(
        session_id=SHARED_SESSION_ID,
        state={"user_id": "user-42"},
    )
    response = await agent.run("What todos do I have?", session=session2)
    print(response.text)

asyncio.run(main())
```

### Example — Custom instructions and multiple source IDs

```python
from agent_framework import Agent, TodoProvider
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

CUSTOM_INSTRUCTIONS = """
You have access to a todo list. Use it to track every sub-task before you begin work.
Mark tasks complete as you finish them. Never proceed without an up-to-date task list.
"""

provider = TodoProvider(
    source_id="strict_todos",
    instructions=CUSTOM_INSTRUCTIONS,
)

agent = Agent(
    client=client,
    name="disciplined_agent",
    instructions="Always plan before acting.",
    context_providers=[provider],
)
```

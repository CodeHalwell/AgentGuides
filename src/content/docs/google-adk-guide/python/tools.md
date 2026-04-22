---
title: "Tools (functions, agents, built-ins)"
description: "FunctionTool, AgentTool, Google built-ins, long-running tools, and confirmation flows."
framework: google-adk
language: python
sidebar:
  order: 30
---

Verified against google-adk==2.0.0b1 (`google/adk/tools/__init__.py`, `google/adk/tools/function_tool.py`).

Tools are the mechanism by which an `LlmAgent` calls code. Three flavours: **plain callable** (auto-wrapped into `FunctionTool`), **`BaseTool` subclass** (the built-ins + your own), and **`BaseToolset`** (dynamic tool lists — MCP, OpenAPI, custom).

## Minimal example

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, google_search

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = LlmAgent(
    name="math_and_search",
    model="gemini-2.5-flash",
    instruction="Use `add` for arithmetic. Use `google_search` for facts.",
    tools=[
        add,                       # callable → wrapped as FunctionTool
        google_search,             # built-in singleton
        FunctionTool(func=add, require_confirmation=True),  # explicit wrap
    ],
)
```

`LlmAgent` wraps bare callables with `FunctionTool(func=...)` at registration time (`llm_agent.py:178-182`). Wrap manually only when you need `require_confirmation=`.

## Public surface

Everything in `google.adk.tools` is lazy-loaded (`tools/__init__.py`):

| Name | Kind | Import note |
|---|---|---|
| `BaseTool`, `BaseToolset` | Abstract | Subclass for custom tools |
| `FunctionTool` | Class | Wraps a callable |
| `LongRunningFunctionTool` | Class | Wraps an async long-running callable |
| `AgentTool` | Class | Wraps a `BaseAgent` as a tool |
| `ExampleTool` | Class | Few-shot example injector |
| `AuthToolArguments` | Class | Auth-required tool arguments |
| `TransferToAgentTool`, `transfer_to_agent` | Class + singleton | Injected automatically when `sub_agents` is set |
| `MCPToolset` / `McpToolset` | Class | Connects to an MCP server |
| `APIHubToolset` | Class | Wraps APIs registered in Google API Hub |
| `ApiRegistry` | Class | Builds tools from OpenAPI specs |
| `ToolContext` | Class | Passed to every tool via `tool_context=` |
| `google_search` | Singleton | Built-in Google Search (Gemini-side) |
| `url_context` | Singleton | Built-in URL context (Gemini-side) |
| `google_maps_grounding` | Singleton | Built-in Maps grounding |
| `enterprise_web_search` | Singleton | Enterprise web search |
| `VertexAiSearchTool` | Class | Vertex AI Search data store |
| `DiscoveryEngineSearchTool` | Class | Discovery Engine search |
| `SearchResultMode` | Enum | For `DiscoveryEngineSearchTool` |
| `load_memory`, `preload_memory` | Singletons | Long-term memory access |
| `load_artifacts` | Singleton | Reads artifacts into the prompt |
| `exit_loop` | Singleton | Sets `actions.escalate=True` from inside `LoopAgent`/`Workflow` |
| `get_user_choice` | `LongRunningFunctionTool` | HITL multi-choice prompt |

## FunctionTool

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def list_files(folder: str, tool_context: ToolContext) -> dict:
    """List files in a given folder.

    Args:
      folder: The folder path.
    Returns:
      A dict with keys `files` and `count`.
    """
    tool_context.state["last_listed"] = folder
    return {"files": ["a.txt", "b.txt"], "count": 2}

tool = FunctionTool(func=list_files, require_confirmation=False)
```

Signature rules (`function_tool.py`):

- The tool **name** is `func.__name__` (or `func.__class__.__name__` for callable objects).
- The tool **description** is the docstring — one sentence + Google-style `Args`/`Returns`. It's passed to the model verbatim, so keep it tight.
- Parameters are introspected with `inspect.signature` + `get_type_hints`. Pydantic model params are auto-converted (`_preprocess_args`, `function_tool.py:106`).
- A parameter named `tool_context` (or typed as `ToolContext`) gets the `ToolContext` injected — it is **not** exposed to the model.
- Sync and async callables both work.

**Missing mandatory args** short-circuit to an `{"error": ...}` response without calling the function, so the LLM can retry (`function_tool.py:219-224`).

### `require_confirmation`

```python
def wipe_all(scope: str) -> dict:
    "Irreversibly wipes data."
    return {"wiped": True}

tool = FunctionTool(
    func=wipe_all,
    require_confirmation=lambda scope: scope != "dry-run",
)
```

Bool or predicate. When the callable returns truthy, the tool returns `{"error": "This tool call requires confirmation..."}` and sets `tool_context.actions.skip_summarization = True`. The user then sends back a `FunctionResponse` carrying a `ToolConfirmation` payload on the next turn.

## LongRunningFunctionTool

```python
from google.adk.tools import LongRunningFunctionTool

async def kick_off_build(project: str) -> dict:
    job_id = await build_service.start(project)
    return {"status": "pending", "job_id": job_id}

tool = LongRunningFunctionTool(func=kick_off_build)
```

The model is instructed not to call the tool again while its response is still pending — the framework surfaces intermediate status via `tool_context.request_confirmation` or an explicit status poll.

## AgentTool

Wrap a whole agent as a callable tool. The agent's `input_schema` becomes the tool's parameter schema; its reply becomes the tool's return value.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from pydantic import BaseModel

class ResearchIn(BaseModel):
    topic: str

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic and return a citation-rich paragraph.",
    input_schema=ResearchIn,
    tools=[google_search],
)

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Use the `researcher` tool, then write a crisp 150-word brief.",
    tools=[AgentTool(agent=researcher, skip_summarization=False)],
)
```

Constructor args (`agent_tool.py:111-122`):

| Arg | Default | Purpose |
|---|---|---|
| `agent` | required | Any `BaseAgent` |
| `skip_summarization` | `False` | If `True`, the caller's model sees the raw agent output rather than summarising it |
| `include_plugins` | `True` | Inherits parent runner's plugins |
| `propagate_grounding_metadata` | `False` | Forwards grounding citations up |

## Built-in Gemini tools

These run **server-side inside Gemini** and cannot be combined freely. When mixed with custom tools, ADK wraps them automatically to stay within Gemini's single-built-in constraint (see `llm_agent.py:149-176`):

| Tool | What it does | Multi-tool-safe |
|---|---|---|
| `google_search` | Gemini's built-in Google Search grounding | Auto-wrapped as `GoogleSearchAgentTool` if needed |
| `url_context` | Gemini's built-in URL-fetch grounding | Single-use |
| `google_maps_grounding` | Gemini's Maps grounding | Single-use |
| `enterprise_web_search` | Enterprise web search grounding | Single-use |
| `VertexAiSearchTool(data_store_id=..., ...)` | Vertex AI Search data store | Auto-substituted for `DiscoveryEngineSearchTool` when mixed |
| `DiscoveryEngineSearchTool(...)` | Discovery Engine (client-side) | Fine with other tools |

```python
from google.adk.tools import VertexAiSearchTool

tool = VertexAiSearchTool(
    data_store_id="projects/my-project/locations/global/collections/default_collection/dataStores/my-store",
    bypass_multi_tools_limit=True,   # auto-substitute with DiscoveryEngine if needed
)
```

## Memory and artifact tools

```python
from google.adk.tools import load_memory, preload_memory, load_artifacts

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-pro",
    instruction="Use `load_memory` to recall past facts.",
    tools=[load_memory, preload_memory, load_artifacts],
)
```

- `load_memory` — the model calls it explicitly with a query; returns memory entries.
- `preload_memory` — **no model-visible tool call**; automatically front-loads the top-k memories into the prompt before each turn.
- `load_artifacts` — lets the model fetch a saved artifact (file) by name; requires an artifact service to be configured on the runner.

## MCP toolset

```python
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

fs_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/work"],
        ),
        timeout=5.0,
    ),
    tool_filter=["read_file", "list_directory"],
)

agent = LlmAgent(name="fs_agent", tools=[fs_tools])
```

Connection params:

| Class | For | Import |
|---|---|---|
| `StdioConnectionParams(server_params, timeout)` | Local stdio MCP server (`npx`, `python3 -m ...`) | `google.adk.tools.mcp_tool` |
| `SseConnectionParams(url, headers, timeout, sse_read_timeout, httpx_client_factory)` | Remote SSE | same |
| `StreamableHTTPConnectionParams(url, headers, timeout, sse_read_timeout, terminate_on_close, ...)` | Streamable HTTP | same |

`tool_filter` accepts a list of tool names or a `ToolPredicate` callable. `McpToolset` also supports `auth_scheme` / `auth_credential` for OAuth-gated servers, `require_confirmation=` (bool or predicate), `progress_callback=`, and `use_mcp_resources=True` to expose MCP resources via a `load_mcp_resource` tool.

## OpenAPI tools

`APIHubToolset` and `ApiRegistry` generate tools from OpenAPI specs:

```python
from google.adk.tools import ApiRegistry

registry = ApiRegistry()
registry.register_openapi_spec(spec_path="./petstore.yaml", base_url="https://petstore.example")
tools = registry.get_tools()
```

Each operation becomes a `BaseTool` whose parameters are the path/query/body fields of the operation.

## Agent transfer

`transfer_to_agent` and `TransferToAgentTool` are injected automatically by ADK when the LLM agent has `sub_agents`. You rarely construct them yourself, but you can inspect them for logging.

## HITL tools

- `get_user_choice` — a `LongRunningFunctionTool` that prompts the user with a list; the LLM picks from the returned choice.
- `request_input` via `ToolContext.request_confirmation()` — any tool can pause and solicit input.

## Patterns

### 1 — Typed function tools
Annotate parameters with Pydantic models. `FunctionTool` converts `dict` → model via `model_validate`. The model sees the JSON schema; your function receives a validated Pydantic instance.

### 2 — Tool chains via `AgentTool`
Wrap a specialist agent as a tool for a generalist. Set `skip_summarization=True` when the specialist's output is already polished.

### 3 — Guardrail with `require_confirmation`
For destructive ops, pass a predicate that returns `True` only for risky inputs (e.g. `scope != "dry-run"`).

### 4 — Gemini-side search + local DB
Put `google_search` first and a `FunctionTool` wrapping your DB helper second. ADK auto-wraps `google_search` so the two coexist.

### 5 — Dynamic MCP toolset
Spin up `McpToolset` at runtime (e.g. per-tenant filesystem); pass `tool_name_prefix=` to avoid collisions with other toolsets. The `Runner` auto-closes toolsets on `runner.close()`.

## Gotchas

- Don't set `output_schema=` on an `LlmAgent` that also has `tools=` — setting `output_schema` disables tool use entirely.
- `tool_context` is injected by parameter name (`tool_context`) **or** type (`ToolContext`). Any other parameter of type `ToolContext` would also be treated as the context slot.
- `FunctionTool` treats the first sentence of the docstring as the tool description. Keep it focused — the model obeys it.
- Built-in Gemini tools (`google_search`, `url_context`, `google_maps_grounding`) cannot coexist freely. ADK tries to wrap them, but if you hit `400 INVALID_ARGUMENT` try `bypass_multi_tools_limit=True` where available.
- `LongRunningFunctionTool` is just a `FunctionTool` with `is_long_running=True`. The model is separately instructed not to re-call it while pending.
- Mutating `tool_context.state` with a reserved prefix (`app:`, `user:`, `temp:`) changes scope — see [runner-and-sessions](./runner-and-sessions/).

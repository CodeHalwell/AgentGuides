---
title: "Class deep dives — volume 18 (2.2.0 new classes)"
description: "Source-verified 2.2.0 deep dives: Context (unified ToolContext/CallbackContext), AgentCardBuilder (A2A card generation), to_a2a() (one-call Starlette A2A server), ToolConfirmation (HITL confirmation model), SessionContext (MCP background-task session), McpInstructionProvider (MCP Prompt → agent instruction), UiWidget (tool event widget rendering), EnvironmentToolset (file/shell @experimental toolset), LoadMcpResourceTool (lazy MCP resource injection), App deep-dive (all four config knobs: compaction, resumability, context cache, plugins)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 18"
  order: 87
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `Context` (unified `CallbackContext` / `ToolContext`) | `google.adk.agents.context` | Stable |
| 2 | `AgentCardBuilder` | `google.adk.a2a.utils.agent_card_builder` | `@a2a_experimental` |
| 3 | `to_a2a()` | `google.adk.a2a.utils.agent_to_a2a` | `@a2a_experimental` |
| 4 | `ToolConfirmation` | `google.adk.tools.tool_confirmation` | `@experimental(TOOL_CONFIRMATION)` |
| 5 | `SessionContext` | `google.adk.tools.mcp_tool.session_context` | Stable |
| 6 | `McpInstructionProvider` | `google.adk.agents.mcp_instruction_provider` | Stable |
| 7 | `UiWidget` | `google.adk.events.ui_widget` | Stable |
| 8 | `EnvironmentToolset` | `google.adk.tools.environment._environment_toolset` | `@experimental` |
| 9 | `LoadMcpResourceTool` | `google.adk.tools.load_mcp_resource_tool` | Stable |
| 10 | `App` (all config knobs) | `google.adk.apps.app` | Stable |

---

## 1 · `Context` — unified callback and tool context

**Source:** `google.adk.agents.context`

`Context` is the single mutable context object passed to **every callback and tool** in google-adk 2.2.0. It replaces the old `CallbackContext` / `ToolContext` split — both aliases now resolve to `Context`. It extends `ReadonlyContext` (which provides read-only properties like `state`, `session`, `user_id`, `run_config`) and adds full write access to state, artifacts, memory, auth credentials, workflow routing, and HITL.

### Constructor (source-verified)

```python
Context(
    invocation_context: InvocationContext,
    *,
    event_actions: EventActions | None = None,
    function_call_id: str | None = None,
    tool_confirmation: ToolConfirmation | None = None,
    parent_ctx: Context | None = None,
    node: BaseNode | None = None,
    node_path: str | None = None,
    run_id: str = '',
    resume_inputs: dict[str, Any] | None = None,
    attempt_count: int = 1,
    use_as_output: bool = False,
)
```

You never construct `Context` yourself — the framework builds it and passes it in. The important fields are:

| Property | Type | Description |
|---|---|---|
| `state` | `State` (delta-aware) | Read/write session state |
| `session` | `Session` | Full session object |
| `actions` | `EventActions` | Low-level event actions delta |
| `function_call_id` | `str \| None` | Tool call ID (set in tool context only) |
| `tool_confirmation` | `ToolConfirmation \| None` | HITL confirmation payload |
| `parent_ctx` | `Context \| None` | Parent node context (workflow) |
| `node_path` | `str` | Dot-separated node path in the workflow |
| `run_id` | `str` | Execution ID of the current node |
| `attempt_count` | `int` | 1-based retry counter |
| `resume_inputs` | `dict[str, Any]` | Inputs from HITL resume |
| `output` | `Any` | Node output (set once; raises if set twice) |
| `route` | `RouteValue \| list` | Workflow routing value |

### Example 1 — reading and mutating state in a tool

```python
from google.adk.agents.context import Context
from google.adk.tools.tool_context import ToolContext  # alias for Context


async def increment_counter(tool_context: ToolContext) -> dict:
    """Increment a persistent counter stored in session state."""
    current = tool_context.state.get("counter", 0)
    tool_context.state["counter"] = current + 1
    return {"counter": tool_context.state["counter"]}
```

### Example 2 — saving and loading artifacts in a callback

```python
from google.adk.agents.context import Context
from google.genai import types


async def after_model_callback(ctx: Context, response) -> None:
    """Persist the raw model response as a versioned artifact."""
    text = ""
    if response.content and response.content.parts:
        text = "".join(p.text or "" for p in response.content.parts if p.text)

    if text:
        artifact = types.Part.from_text(text=text)
        version = await ctx.save_artifact("raw_response.txt", artifact)
        ctx.state["last_response_version"] = version


async def load_previous_response(tool_context: ToolContext) -> dict:
    """Return the last saved raw response."""
    part = await tool_context.load_artifact("raw_response.txt")
    if part is None:
        return {"error": "No saved response found"}
    return {"content": part.text}
```

### Example 3 — searching memory and adding to memory

```python
from google.adk.agents.context import Context


async def knowledge_recall_tool(query: str, tool_context: ToolContext) -> dict:
    """Search long-term memory for relevant facts."""
    response = await tool_context.search_memory(query)
    memories = []
    for entry in (response.memories or []):
        memories.append({
            "content": entry.content,
            "score": entry.score,
        })
    return {"memories": memories, "count": len(memories)}


async def after_agent_callback(ctx: Context) -> None:
    """Persist the current session into long-term memory after each turn."""
    await ctx.add_session_to_memory()
```

### Example 4 — requesting auth credentials from a tool

```python
from google.adk.agents.context import Context
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth


GOOGLE_OAUTH_CONFIG = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
        ),
    ),
)


async def list_calendar_events(tool_context: ToolContext) -> dict:
    """List Google Calendar events — requests OAuth2 if not already resolved."""
    cred = tool_context.get_credential("google_oauth")
    if cred is None:
        # Suspend tool, ask the user to complete OAuth2
        tool_context.request_credential(GOOGLE_OAUTH_CONFIG)
        return {"pending": "Please complete Google sign-in"}

    # Use cred.oauth2.access_token to call the Calendar API
    return {"status": "fetched", "token_prefix": cred.oauth2.access_token[:8]}
```

### Example 5 — workflow routing and `ctx.run_node()`

```python
from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._workflow import Workflow


research_agent = LlmAgent(name="researcher", model="gemini-2.0-flash",
                          instruction="Research the topic and return a summary.")
write_agent = LlmAgent(name="writer", model="gemini-2.0-flash",
                       instruction="Write a blog post based on the summary.")


async def orchestrate(ctx: Context) -> None:
    """Dynamically chain research → write inside a workflow node."""
    summary = await ctx.run_node(research_agent, node_input=ctx.state.get("topic"))
    ctx.state["summary"] = summary
    final = await ctx.run_node(write_agent, node_input=summary, use_as_output=True)
    ctx.output = final
```

---

## 2 · `AgentCardBuilder` — A2A agent card from any ADK agent

**Source:** `google.adk.a2a.utils.agent_card_builder`

`AgentCardBuilder` converts a `BaseAgent` (or `Workflow`) into an A2A `AgentCard`. It introspects the agent tree to extract skills, planners, code executors, and sub-agents. The class is decorated with `@a2a_experimental`.

### Constructor (source-verified)

```python
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder

AgentCardBuilder(
    *,
    agent: BaseAgent | Workflow,
    rpc_url: str | None = None,          # default "http://localhost:80/a2a"
    capabilities: AgentCapabilities | None = None,
    doc_url: str | None = None,
    provider: AgentProvider | None = None,
    agent_version: str | None = None,    # default "0.0.1"
    security_schemes: dict[str, SecurityScheme] | None = None,
)
```

Call `.build()` (async) to get the `AgentCard`.

### Example 1 — minimal card for an LlmAgent

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder


agent = LlmAgent(
    name="summarizer",
    model="gemini-2.0-flash",
    description="Summarise documents in multiple languages.",
    instruction="You are a summariser. Return concise bullet-point summaries.",
)


async def main():
    builder = AgentCardBuilder(
        agent=agent,
        rpc_url="https://my-service.run.app/a2a",
        agent_version="1.0.0",
    )
    card = await builder.build()
    print(card.name)           # "summarizer"
    print(card.version)        # "1.0.0"
    print(len(card.skills))    # at least 1 (the "model" skill)
    for skill in card.skills:
        print(f"  skill: {skill.id}  tags: {skill.tags}")


asyncio.run(main())
```

### Example 2 — card with tool skills and custom capabilities

```python
import asyncio
from a2a.types import AgentCapabilities
from google.adk.agents.llm_agent import LlmAgent
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.tools.google_search_tool import GoogleSearchTool


search_agent = LlmAgent(
    name="search_agent",
    model="gemini-2.0-flash",
    description="Searches the web and returns structured answers.",
    instruction="Use Google Search to answer questions.",
    tools=[GoogleSearchTool()],
)


async def main():
    builder = AgentCardBuilder(
        agent=search_agent,
        rpc_url="https://search-agent.example.com/a2a",
        capabilities=AgentCapabilities(streaming=True),
        agent_version="2.1.0",
    )
    card = await builder.build()
    # skills: [model, google_search_retrieval (tool skill)]
    for skill in card.skills:
        print(f"{skill.id}: {skill.description[:60]}")


asyncio.run(main())
```

### Example 3 — card from a SequentialAgent (sub-agent skill extraction)

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder


triage = LlmAgent(name="triage", model="gemini-2.0-flash",
                  description="Classify incoming support ticket.")
resolver = LlmAgent(name="resolver", model="gemini-2.0-flash",
                    description="Draft resolution for classified ticket.")

pipeline = SequentialAgent(
    name="support_pipeline",
    description="End-to-end support ticket processing pipeline.",
    sub_agents=[triage, resolver],
)


async def main():
    builder = AgentCardBuilder(
        agent=pipeline,
        rpc_url="https://support.example.com/a2a",
    )
    card = await builder.build()
    print(card.name)            # "support_pipeline"
    print(len(card.skills))     # primary + sub-agent skills
    for skill in card.skills:
        print(f"  {skill.id}: tags={skill.tags}")
    # sub-agent skills are prefixed: "triage_triage", "resolver_resolver"


asyncio.run(main())
```

---

## 3 · `to_a2a()` — one-call Starlette A2A server

**Source:** `google.adk.a2a.utils.agent_to_a2a`

`to_a2a()` wraps an ADK agent or workflow in a fully configured `A2AStarletteApplication` (from the `a2a` SDK) in a single function call. Internally it wires up an `A2aAgentExecutor`, a `Runner`, and in-memory task/push-notification stores. The function is `@a2a_experimental`.

### Signature (source-verified)

```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a

to_a2a(
    agent: BaseAgent | Workflow,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: AgentCard | str | None = None,   # AgentCard object or path to JSON
    push_config_store: PushNotificationConfigStore | None = None,
    task_store: TaskStore | None = None,
    runner: Runner | None = None,
    lifespan: Callable[[Starlette], AsyncIterator[None]] | None = None,
    agent_executor_factory: Callable[[Runner], A2aAgentExecutor] | None = None,
) -> Starlette
```

Returns a `Starlette` ASGI application you can serve with `uvicorn`.

### Example 1 — minimal A2A server in 5 lines

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn

agent = LlmAgent(
    name="hello_agent",
    model="gemini-2.0-flash",
    description="A friendly greeting agent.",
    instruction="Greet the user warmly and ask how you can help.",
)

app = to_a2a(agent, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Example 2 — providing a pre-built AgentCard

```python
import asyncio
import uvicorn
from a2a.types import AgentCapabilities
from google.adk.agents.llm_agent import LlmAgent
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.a2a.utils.agent_to_a2a import to_a2a


agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-pro",
    description="Analyses data sets and produces insights.",
    instruction="You analyse CSV or JSON data and explain findings clearly.",
)


async def build_and_serve():
    # Build card explicitly so we control version and capabilities
    card = await AgentCardBuilder(
        agent=agent,
        rpc_url="http://0.0.0.0:9000/a2a",
        capabilities=AgentCapabilities(streaming=True),
        agent_version="3.0.0",
    ).build()

    starlette_app = to_a2a(agent, host="0.0.0.0", port=9000, agent_card=card)
    config = uvicorn.Config(starlette_app, host="0.0.0.0", port=9000)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(build_and_serve())
```

### Example 3 — A2A server from a Workflow with custom lifespan

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
import uvicorn
from starlette.applications import Starlette

from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.a2a.utils.agent_to_a2a import to_a2a

extract = LlmAgent(name="extract", model="gemini-2.0-flash",
                   instruction="Extract key entities from the text.")
classify = LlmAgent(name="classify", model="gemini-2.0-flash",
                    instruction="Classify the extracted entities.")

wf = Workflow(name="nlp_pipeline", description="Extract then classify entities.")
wf.graph.add_edge(START, extract)
wf.graph.add_edge(extract, classify)


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    print("NLP pipeline starting up…")
    yield
    print("NLP pipeline shutting down…")


starlette_app = to_a2a(
    wf,
    host="0.0.0.0",
    port=8765,
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8765)
```

### Example 4 — loading an AgentCard from a JSON file

```python
# agent_card.json lives at ./cards/my_card.json
import uvicorn
from google.adk.agents.llm_agent import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

agent = LlmAgent(name="card_agent", model="gemini-2.0-flash",
                 instruction="Answer questions helpfully.")

# Pass path string — to_a2a loads it via pathlib.Path.open()
app = to_a2a(agent, host="localhost", port=8000,
             agent_card="./cards/my_card.json")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
```

---

## 4 · `ToolConfirmation` — HITL tool confirmation model

**Source:** `google.adk.tools.tool_confirmation`

`ToolConfirmation` is the data model for human-in-the-loop (HITL) tool confirmation. It is `@experimental(FeatureName.TOOL_CONFIRMATION)`. When a tool calls `tool_context.request_confirmation(...)`, the framework suspends the tool, sends a `ToolConfirmation` to the caller, and resumes only after the caller responds with `confirmed=True` (and an optional `payload`).

### Model (source-verified)

```python
from google.adk.tools.tool_confirmation import ToolConfirmation

class ToolConfirmation(BaseModel):
    hint: str = ""           # displayed to the human
    confirmed: bool = False  # set to True by the human to approve
    payload: Any = None      # arbitrary JSON-serialisable data from the human
```

`model_config` uses camelCase alias generation (`alias_generators.to_camel`), so the wire format is `{ "hint": "...", "confirmed": true, "payload": {...} }`.

### Example 1 — basic tool that requests confirmation before executing

```python
from google.adk.features import FeatureName, override_feature_enabled
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.llm_agent import LlmAgent


async def delete_user_account(
    user_id: str,
    tool_context: ToolContext,
) -> dict:
    """Permanently delete a user account.

    Args:
        user_id: The ID of the user account to delete.
    """
    tc = tool_context.tool_confirmation
    if tc is None or not tc.confirmed:
        # First call — suspend and ask the human
        tool_context.request_confirmation(
            ToolConfirmation(
                hint=f"Permanently delete account '{user_id}'? This cannot be undone.",
            )
        )
        return {"status": "awaiting_confirmation"}

    # Second call — human approved (tc.confirmed is True)
    # Perform the actual deletion here
    return {"status": "deleted", "user_id": user_id}


with override_feature_enabled(FeatureName.TOOL_CONFIRMATION, True):
    agent = LlmAgent(
        name="admin_agent",
        model="gemini-2.0-flash",
        instruction="You manage user accounts. Always confirm destructive actions.",
        tools=[delete_user_account],
    )
```

### Example 2 — using `payload` to collect data from the human

```python
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext


async def transfer_funds(
    amount: float,
    destination_account: str,
    tool_context: ToolContext,
) -> dict:
    """Transfer funds after confirming with a 2FA code.

    Args:
        amount: The amount to transfer in USD.
        destination_account: The destination bank account number.
    """
    tc = tool_context.tool_confirmation
    if tc is None or not tc.confirmed:
        tool_context.request_confirmation(
            ToolConfirmation(
                hint=(
                    f"Transfer ${amount:.2f} to {destination_account}. "
                    "Please enter your 2FA code in the 'payload' field."
                ),
            )
        )
        return {"status": "awaiting_2fa"}

    two_fa_code = (tc.payload or {}).get("code")
    if two_fa_code != "123456":   # validate the 2FA code
        return {"status": "invalid_2fa"}

    return {"status": "transferred", "amount": amount, "to": destination_account}
```

### Example 3 — resuming from `tool_context.resume_inputs`

```python
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext


async def sensitive_query(
    sql: str,
    tool_context: ToolContext,
) -> dict:
    """Run a SQL query after DBA approval.

    Args:
        sql: The SQL query to execute.
    """
    confirmation = tool_context.tool_confirmation

    if confirmation is None:
        tool_context.request_confirmation(
            ToolConfirmation(
                hint=f"DBA approval required for: {sql[:120]}",
            )
        )
        return {"status": "pending_approval"}

    if not confirmation.confirmed:
        return {"status": "rejected", "reason": "DBA denied the request"}

    # Execute the approved query
    return {"status": "executed", "rows_affected": 42}
```

---

## 5 · `SessionContext` — MCP session lifecycle in a background task

**Source:** `google.adk.tools.mcp_tool.session_context`

`SessionContext` manages the full lifecycle of a single MCP `ClientSession` inside a **dedicated background asyncio task**. This design exists because AnyIO's `CancelScope` (used by MCP clients internally) requires that scope entry and exit happen in the same task. Wrapping the session in its own task satisfies this constraint.

### Constructor (source-verified)

```python
from google.adk.tools.mcp_tool.session_context import SessionContext

SessionContext(
    client: AsyncContextManager,          # e.g. streamablehttp_client(...)
    timeout: float | None,               # connection + init timeout (seconds)
    sse_read_timeout: float | None,      # per-message read timeout (SSE)
    is_stdio: bool = False,
    *,
    sampling_callback: SamplingFnT | None = None,
    sampling_capabilities: SamplingCapability | None = None,
)
```

### Key methods

| Method | Description |
|---|---|
| `await start()` | Start background task, wait for `ClientSession` to be ready |
| `await close()` | Signal background task to shut down, await cleanup |
| `async with lifecycle as session:` | Context manager that calls start/close |
| `session` property | The live `ClientSession`, or `None` before start |
| `_run_guarded(coro)` | Race a coroutine against the background task (transport crash detection) |

### Example 1 — using as an async context manager

```python
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from google.adk.tools.mcp_tool.session_context import SessionContext


async def call_mcp_tool(server_url: str, tool_name: str, args: dict):
    """Connect to an MCP server, call a tool, then disconnect cleanly."""
    client = streamablehttp_client(server_url)
    ctx = SessionContext(client=client, timeout=10.0, sse_read_timeout=30.0)

    async with ctx as session:
        result = await session.call_tool(tool_name, args)
        return result.content


async def main():
    result = await call_mcp_tool(
        "http://localhost:8080/mcp",
        "get_weather",
        {"city": "London"},
    )
    print(result)


asyncio.run(main())
```

### Example 2 — explicit start/close with error handling

```python
import asyncio
from mcp.client.sse import sse_client
from google.adk.tools.mcp_tool.session_context import SessionContext


async def robust_mcp_call(server_url: str, tool: str, args: dict):
    client = sse_client(server_url)
    ctx = SessionContext(client=client, timeout=15.0, sse_read_timeout=60.0)
    session = None
    try:
        session = await ctx.start()
        tools = await session.list_tools()
        print(f"Available tools: {[t.name for t in tools.tools]}")
        result = await session.call_tool(tool, args)
        return result.content
    except ConnectionError as e:
        print(f"Transport failure: {e}")
        raise
    finally:
        await ctx.close()


asyncio.run(robust_mcp_call("http://localhost:8888/sse", "summarise", {"text": "hello"}))
```

### Example 3 — sampling callback for LLM-backed MCP servers

```python
import asyncio
from mcp import types as mcp_types
from mcp.client.streamable_http import streamablehttp_client
from google.adk.tools.mcp_tool.session_context import SessionContext


async def my_sampling_callback(
    context: list[mcp_types.SamplingMessage],
    params: mcp_types.CreateMessageRequestParams,
) -> mcp_types.CreateMessageResult:
    """Handle sampling requests from the MCP server using a local model."""
    # In a real app, call your LLM here
    return mcp_types.CreateMessageResult(
        role="assistant",
        content=mcp_types.TextContent(type="text", text="Sampled response"),
        model="gemini-2.0-flash",
        stopReason="endTurn",
    )


async def main():
    client = streamablehttp_client("http://localhost:8080/mcp")
    ctx = SessionContext(
        client=client,
        timeout=10.0,
        sse_read_timeout=30.0,
        sampling_callback=my_sampling_callback,
    )
    async with ctx as session:
        result = await session.call_tool("smart_tool", {})
        print(result.content)


asyncio.run(main())
```

---

## 6 · `McpInstructionProvider` — dynamic agent instructions from an MCP Prompt

**Source:** `google.adk.agents.mcp_instruction_provider`

`McpInstructionProvider` implements `InstructionProvider` (the callable protocol accepted by `LlmAgent.instruction`) by fetching a **named MCP Prompt** from an MCP server at runtime. Arguments for the prompt are automatically pulled from the session state — any state key that matches a prompt argument name is forwarded.

### Constructor (source-verified)

```python
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider

McpInstructionProvider(
    connection_params: Any,   # StdioConnectionParams, SseConnectionParams, etc.
    prompt_name: str,         # name of the MCP Prompt to fetch
    errlog: TextIO = sys.stderr,
)
```

It implements `async def __call__(self, context: ReadonlyContext) -> str`.

### Example 1 — using a Stdio MCP server as the instruction source

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams

connection = StdioConnectionParams(
    command="python",
    args=["-m", "my_mcp_server"],  # local MCP server process
)

instruction_provider = McpInstructionProvider(
    connection_params=connection,
    prompt_name="agent_system_prompt",  # must exist on the server
)

agent = LlmAgent(
    name="dynamic_agent",
    model="gemini-2.0-flash",
    instruction=instruction_provider,   # fetched fresh on each invocation
)
```

### Example 2 — injecting state into prompt arguments

```python
# The MCP server exposes a prompt named "role_prompt" with argument "role".
# We set ctx.state["role"] = "finance_analyst" before invoking the agent,
# and McpInstructionProvider automatically forwards it.

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

connection = StdioConnectionParams(command="python", args=["-m", "prompts_server"])

agent = LlmAgent(
    name="role_agent",
    model="gemini-2.0-flash",
    instruction=McpInstructionProvider(connection, "role_prompt"),
)

async def run_as_role(role: str, user_message: str):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="demo", user_id="u1",
        state={"role": role},  # forwarded to prompt argument "role"
    )
    runner = Runner(agent=agent, app_name="demo", session_service=session_service)
    result = ""
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=user_message)]),
    ):
        if event.is_final_response() and event.content:
            result = "".join(p.text or "" for p in event.content.parts)
    return result
```

### Example 3 — SSE-based MCP server for instructions

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool.mcp_toolset import SseConnectionParams

sse_params = SseConnectionParams(url="http://localhost:9090/sse")

agent = LlmAgent(
    name="sse_instruction_agent",
    model="gemini-2.0-flash",
    instruction=McpInstructionProvider(
        connection_params=sse_params,
        prompt_name="custom_system_prompt",
    ),
    description="Agent with server-side-managed instructions.",
)
```

---

## 7 · `UiWidget` — tool event widget rendering metadata

**Source:** `google.adk.events.ui_widget`

`UiWidget` carries rendering metadata for a UI widget attached to a tool event. The UI reads `provider` to select a renderer, then passes `payload` to it. Today, the only built-in provider is `"mcp"` — which renders an MCP App iframe via the MCP Apps `AppBridge`. Attach a `UiWidget` to an event via `ctx.render_ui_widget(widget)`.

### Model (source-verified)

```python
from google.adk.events.ui_widget import UiWidget

class UiWidget(BaseModel):
    id: str                    # unique widget identifier
    provider: str              # "mcp" or custom provider key
    payload: dict[str, Any]    # provider-specific rendering data
```

For the `"mcp"` provider the payload is:
```python
{
    "resource_uri": "ui://...",    # MCP resource URI
    "tool": {...},                  # tool descriptor
    "tool_args": {...},             # arguments used to invoke the tool
}
```

### Example 1 — attaching a widget to a tool event via `ctx.render_ui_widget()`

```python
from google.adk.agents.context import Context
from google.adk.events.ui_widget import UiWidget


async def render_chart(
    dataset: str,
    chart_type: str,
    tool_context: Context,
) -> dict:
    """Render a chart using an MCP App iframe widget.

    Args:
        dataset: Name of the dataset to visualise.
        chart_type: One of 'bar', 'line', 'pie'.
    """
    widget = UiWidget(
        id=f"chart-{dataset}",
        provider="mcp",
        payload={
            "resource_uri": f"ui://charts/{dataset}/{chart_type}",
            "tool": {"name": "render_chart"},
            "tool_args": {"dataset": dataset, "chart_type": chart_type},
        },
    )
    tool_context.render_ui_widget(widget)
    return {"status": "widget_rendered", "widget_id": widget.id}
```

### Example 2 — custom provider widget

```python
from google.adk.agents.context import Context
from google.adk.events.ui_widget import UiWidget


async def show_map(
    latitude: float,
    longitude: float,
    zoom: int,
    tool_context: Context,
) -> dict:
    """Render an embedded map widget.

    Args:
        latitude: Map centre latitude.
        longitude: Map centre longitude.
        zoom: Zoom level (1-20).
    """
    widget = UiWidget(
        id=f"map-{latitude:.4f}-{longitude:.4f}",
        provider="google_maps",
        payload={
            "lat": latitude,
            "lng": longitude,
            "zoom": zoom,
            "map_type": "roadmap",
        },
    )
    tool_context.render_ui_widget(widget)
    return {"status": "map_rendered", "center": [latitude, longitude]}
```

### Example 3 — reading widget metadata from an event

```python
from google.adk.agents.callback_context import CallbackContext


async def after_tool_callback(ctx: CallbackContext, tool_response, event):
    """Log any UI widget metadata attached to this tool event."""
    if event and event.actions and event.actions.ui_widgets:
        for widget in event.actions.ui_widgets:
            print(f"Widget {widget.id} ({widget.provider}): {widget.payload}")
```

---

## 8 · `EnvironmentToolset` — file and shell tools for coding agents

**Source:** `google.adk.tools.environment._environment_toolset`

`EnvironmentToolset` is an `@experimental` toolset that gives an LlmAgent four environment interaction tools: **Execute** (run shell commands), **ReadFile**, **EditFile** (surgical replacement), and **WriteFile**. A `BaseEnvironment` implementation (e.g. `LocalEnvironment` or a Docker-based one) provides the actual execution backend. The toolset injects an environment-level system instruction on every LLM call.

### Constructor (source-verified)

```python
from google.adk.tools.environment._environment_toolset import EnvironmentToolset

EnvironmentToolset(
    *,
    environment: BaseEnvironment,
    max_output_chars: int | None = None,  # truncates stdout/stderr/file reads
)
```

The toolset exposes 4 tools:
| Tool class | What it does |
|---|---|
| `ExecuteTool` | Runs a shell command; returns stdout, stderr, exit_code |
| `ReadFileTool` | Reads file content (truncated to `max_output_chars`) |
| `EditFileTool` | Surgical old→new text replacement; fails if old_str absent |
| `WriteFileTool` | Creates or overwrites a file |

### Example 1 — coding agent with a local environment

```python
from google.adk.tools.environment._environment_toolset import EnvironmentToolset
from google.adk.agents.llm_agent import LlmAgent

# LocalEnvironment executes commands in the current process working directory.
# Install: pip install google-adk[environment]  (may need extra deps)
try:
    from google.adk.environment.local_environment import LocalEnvironment
    env = LocalEnvironment(work_dir="/tmp/agent_workspace")
except ImportError:
    env = None   # environment extra not installed

if env:
    env_toolset = EnvironmentToolset(environment=env, max_output_chars=8192)

    coding_agent = LlmAgent(
        name="coder",
        model="gemini-2.5-pro",
        instruction=(
            "You are a coding assistant. Use the provided tools to read files, "
            "execute code, make targeted edits, and write new files. "
            "Always read a file before editing it."
        ),
        tools=[env_toolset],
    )
```

### Example 2 — `EnvironmentSimulationConfig` for testing without side effects

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    InjectionConfig,
    InjectedError,
    MockStrategy,
    ToolSimulationConfig,
)
from google.adk.tools.environment_simulation.environment_simulation_plugin import (
    EnvironmentSimulationPlugin,
)
from google.adk.agents.llm_agent import LlmAgent

# Simulate Execute tool: 80% chance of returning a mock response,
# 20% chance of injecting a latency of 2 seconds
sim_config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="execute",
            mock_strategy=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            injection_configs=[
                InjectionConfig(
                    injection_probability=0.8,
                    injected_response={"stdout": "mock output", "exit_code": 0},
                ),
                InjectionConfig(
                    injection_probability=0.2,
                    injected_latency_seconds=2.0,
                    injected_response={"stdout": "slow mock", "exit_code": 0},
                ),
            ],
        ),
    ]
)

sim_plugin = EnvironmentSimulationPlugin(config=sim_config)

agent = LlmAgent(
    name="test_agent",
    model="gemini-2.0-flash",
    instruction="Execute shell commands as requested.",
)
```

### Example 3 — injecting HTTP errors into a simulated tool

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    InjectionConfig,
    InjectedError,
    MockStrategy,
    ToolSimulationConfig,
)

# Test your agent's error-handling by injecting a 500 error 30% of the time
error_sim = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="execute",
            mock_strategy=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            injection_configs=[
                InjectionConfig(
                    injection_probability=0.3,
                    injected_error=InjectedError(
                        injected_http_error_code=500,
                        error_message="Internal Server Error (simulated)",
                    ),
                    random_seed=42,
                ),
                InjectionConfig(
                    injection_probability=0.7,
                    injected_response={"stdout": "OK", "exit_code": 0},
                ),
            ],
        )
    ]
)
```

---

## 9 · `LoadMcpResourceTool` — lazy MCP resource injection

**Source:** `google.adk.tools.load_mcp_resource_tool`

`LoadMcpResourceTool` lets an agent load **MCP resources** on demand without pre-loading them into the context window. On each LLM call it injects a system instruction listing available resource names; when the model calls `load_mcp_resource`, it reads the resources and appends them to the LLM request as `user` content. The tool is designed to keep the context window lean until resources are actually needed.

### Constructor (source-verified)

```python
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool

LoadMcpResourceTool(mcp_toolset: McpToolset)
```

The tool name is always `"load_mcp_resource"`. The `mcp_toolset` must expose `list_resources()` and `read_resource(name)` methods.

### Example 1 — attaching `LoadMcpResourceTool` to an agent

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool

connection = StdioConnectionParams(
    command="python",
    args=["-m", "my_resource_server"],  # MCP server that exposes resources
)
mcp_toolset = MCPToolset(connection_params=connection)
resource_tool = LoadMcpResourceTool(mcp_toolset=mcp_toolset)

agent = LlmAgent(
    name="resource_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You have access to MCP resources. When the user asks about a document "
        "or dataset, call load_mcp_resource to fetch it before answering."
    ),
    tools=[mcp_toolset, resource_tool],
)
```

### Example 2 — resource injection flow (how it works)

```python
# LoadMcpResourceTool.process_llm_request() runs on EVERY LLM call and:
# 1. Calls mcp_toolset.list_resources() → gets resource names
# 2. Appends a system instruction: "You have these resources: [name1, name2, ...]"
# 3. If the last LLM content is a function_response for "load_mcp_resource":
#    - Reads each requested resource via mcp_toolset.read_resource(name)
#    - Appends the content as user Content blocks (text or binary Part)
#
# This means the model can call the tool multiple times to load different resources.

# To inspect what would be injected (pseudocode):
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool

async def demo_resource_listing(mcp_toolset):
    tool = LoadMcpResourceTool(mcp_toolset=mcp_toolset)
    resource_names = await mcp_toolset.list_resources()
    print(f"Available resources: {resource_names}")
    # The model receives: "You have a list of MCP resources: ["doc1", "doc2", ...]"
```

### Example 3 — combining with an SSE-based MCP toolset

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool

sse_toolset = MCPToolset(
    connection_params=SseConnectionParams(url="http://localhost:9090/sse")
)

agent = LlmAgent(
    name="wiki_agent",
    model="gemini-2.0-flash",
    instruction="Use load_mcp_resource to fetch wiki pages before summarising them.",
    tools=[
        sse_toolset,
        LoadMcpResourceTool(mcp_toolset=sse_toolset),
    ],
)
```

---

## 10 · `App` — complete application configuration deep-dive

**Source:** `google.adk.apps.app`

`App` is the top-level container for an ADK agentic system. It wires together the `root_agent` (or `root_node`), application-wide `plugins`, and four optional configuration knobs that the `Runner` reads at runtime.

### Model (source-verified)

```python
from google.adk.apps.app import App

class App(BaseModel):
    name: str                                           # must match [a-zA-Z][a-zA-Z0-9_-]*
    root_agent: BaseAgent | BaseNode                    # exactly one required
    plugins: list[BasePlugin] = []
    events_compaction_config: EventsCompactionConfig | None = None
    context_cache_config: ContextCacheConfig | None = None
    resumability_config: ResumabilityConfig | None = None
```

`validate_app_name()` enforces: starts with a letter, only letters/digits/underscores/hyphens, not `"user"`.

### `EventsCompactionConfig` (source: `google.adk.apps._configs`)

Controls automatic context compaction. Two modes, combinable:

| Field | Type | Description |
|---|---|---|
| `token_threshold` | `int \| None` | Compact when prompt token count exceeds this |
| `event_retention_size` | `int \| None` | Keep this many raw events after compaction |
| `compaction_interval` | `int \| None` | Compact every N new invocations (sliding window) |
| `overlap_size` | `int \| None` | Include N previous invocations in each summary |
| `summarizer` | `BaseEventsSummarizer \| None` | Custom summariser (auto-created from `LlmEventSummarizer` if `None`) |

### `ResumabilityConfig` (source: `google.adk.apps._configs`)

Enables pause/resume for long-running tools:

| Field | Description |
|---|---|
| `handle` | Unique identifier for this resumability checkpoint |
| `rerun_on_resume` | Whether the node is re-run from scratch on resume |

### `ContextCacheConfig` (source: `google.adk.agents.context_cache_config`)

Enables Gemini context caching to reduce costs on repeated large contexts.

### Example 1 — minimal App with root agent

```python
from google.adk.apps.app import App
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

app = App(name="my_assistant", root_agent=agent)
```

### Example 2 — token-threshold compaction

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(name="long_chat", model="gemini-2.0-flash",
                 instruction="You are a helpful assistant for long conversations.")

# Compact when prompt exceeds 30 000 tokens; retain the last 5 raw events
app = App(
    name="long_chat_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        token_threshold=30_000,
        event_retention_size=5,
    ),
)
```

### Example 3 — sliding-window compaction combined with token threshold

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(name="session_agent", model="gemini-2.0-flash",
                 instruction="Assist the user across a long multi-turn session.")

# Primary: compact on token threshold; fallback: compact every 10 invocations
# with a 2-invocation overlap between summaries
app = App(
    name="session_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        # Token-threshold mode (tried first)
        token_threshold=50_000,
        event_retention_size=10,
        # Sliding-window fallback
        compaction_interval=10,
        overlap_size=2,
    ),
)
```

### Example 4 — App with plugins

```python
from google.adk.apps.app import App
from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService

agent = LlmAgent(
    name="traced_agent",
    model="gemini-2.0-flash",
    instruction="You are an agent with full observability.",
)

app = App(
    name="traced_app",
    root_agent=agent,
    plugins=[
        AutoTracingPlugin(),
        SaveFilesAsArtifactsPlugin(),
    ],
)
```

### Example 5 — App with Gemini context cache

```python
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.llm_agent import LlmAgent

LARGE_SYSTEM_DOC = "..." * 10_000   # large reference document

agent = LlmAgent(
    name="cached_agent",
    model="gemini-1.5-pro",
    instruction=LARGE_SYSTEM_DOC,
)

# Cache the system instruction with a 1-hour TTL;
# only activate caching when the cached content is >= 4096 tokens
app = App(
    name="cached_app",
    root_agent=agent,
    context_cache_config=ContextCacheConfig(
        ttl_seconds=3600,
        min_token_count=4096,
    ),
)
```

### Example 6 — validate_app_name usage

```python
from google.adk.apps.app import validate_app_name

# Valid names
validate_app_name("my_agent")       # OK
validate_app_name("Agent-v2")       # OK
validate_app_name("AgentV1")        # OK

# Invalid names — raise ValueError
try:
    validate_app_name("1agent")     # starts with digit → error
except ValueError as e:
    print(e)

try:
    validate_app_name("user")       # reserved → error
except ValueError as e:
    print(e)

try:
    validate_app_name("my agent")   # space → error
except ValueError as e:
    print(e)
```

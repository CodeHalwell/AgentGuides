---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 15"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: AG-UI client layer (AGUIChatClient, AGUIEventConverter, AGUIHttpService), AG-UI protocol wrappers (AgentFrameworkAgent, AgentFrameworkWorkflow, state_update, add_agent_framework_fastapi_endpoint), ChatKit integration (ThreadItemConverter, stream_agent_response), DevServer (DevServer, serve, register_cleanup), GAIA benchmark (GAIA, GAIATelemetryConfig, TaskRunner), CopilotStudioAgent + CopilotStudioSettings, AzureAISearchContextProvider + AzureAISearchSettings, CosmosHistoryProvider, DurableAI external layer (DurableAIAgentClient, DurableAIAgentWorker, DurableAIAgentOrchestrationContext), AgentFunctionApp."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 38
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 15

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework.ag_ui`, `agent_framework.chatkit`,
`agent_framework.devui`, `agent_framework.lab.gaia`, `agent_framework.microsoft`,
`agent_framework.azure`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, compaction, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — harness providers, compaction strategies, `WorkflowViz`, MCP transports
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — message/chat types, `ResponseStream`, `AgentContext`, functional workflows, `SkillsSource`, eval model, tokenizer, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exceptions
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — feature staging, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, embedding clients, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, orchestration builders, `AgentFactory`, `SecureAgentConfig`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — file store hierarchy, `FileAccessProvider`, `MCPSkill`, `ToolMode`, eval helpers, `ChatContext`, `WorkflowAgent`, compaction, history providers, skills composition
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `Mem0ContextProvider`, Redis providers, Magentic internals, `FileSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow`, `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, memory harness, todo harness, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `BaseAgent`
- [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) — telemetry layers, `Edge`+`EdgeGroup` primitives, `Case`+`Default`, `EdgeRunner` hierarchy, `ExecutionContext`, `WorkflowGraphValidator`, `MCPTool`, serialization mixin, `Evaluator`, `PerServiceCallHistoryPersistingMiddleware`
- [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/) — Skills ABCs, `FileSkill`, `InlineSkillResource`+`InlineSkillScript`, `FileSkillScript`+`SkillScriptRunner`, `SupportsAgentRun`, `RunnerContext`, edge-routing descriptors, `WorkflowValidationError` hierarchy, `A2AAgent`+`A2AExecutor`, exception leaf classes
- [Vol. 13](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v13/) — OpenAI Responses/Completions/Embedding clients, Anthropic + Claude agent clients, multi-cloud Claude variants, group-chat + handoff + Magentic orchestration internals, declarative HTTP/MCP/approval handlers
- [Vol. 14](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v14/) — `State` (superstep cache), `OutputDesignation`, `MessageType`+`WorkflowMessage` internals, `DictConvertible` mixin, `MiddlewareWrapper`+`BaseMiddlewarePipeline`, middleware pipeline hierarchy, `MiddlewareDict`+`categorize_middleware`, `FunctionRequestResult` TypedDict, `OtelAttr`+`MessageListTimestampFilter`, security policy classes

This volume covers **ten new class groups** focused on integration interfaces that
have never been documented in any previous volume: the AG-UI protocol layer, ChatKit
UI integration, the developer server, the GAIA benchmark harness, Copilot Studio
bridging, Azure-specific storage/search providers, and the Durable Task hosting surface.

---

## Table of contents

1. [AG-UI client layer — `AGUIChatClient`, `AGUIEventConverter`, `AGUIHttpService`](#1-ag-ui-client-layer)
2. [AG-UI protocol wrappers — `AgentFrameworkAgent`, `AgentFrameworkWorkflow`, `state_update`, `add_agent_framework_fastapi_endpoint`](#2-ag-ui-protocol-wrappers)
3. [ChatKit integration — `ThreadItemConverter`, `simple_to_agent_input`, `stream_agent_response`](#3-chatkit-integration)
4. [Developer server — `DevServer`, `serve`, `register_cleanup`](#4-developer-server)
5. [GAIA benchmark harness — `GAIA`, `GAIATelemetryConfig`, `Task`, `TaskRunner`, `TaskResult`, `Prediction`, `Evaluation`](#5-gaia-benchmark-harness)
6. [Copilot Studio bridge — `CopilotStudioAgent`, `CopilotStudioSettings`](#6-copilot-studio-bridge)
7. [Azure AI Search provider — `AzureAISearchContextProvider`, `AzureAISearchSettings`](#7-azure-ai-search-context-provider)
8. [Cosmos DB history — `CosmosHistoryProvider`](#8-cosmos-db-history-provider)
9. [Durable external layer — `DurableAIAgentClient`, `DurableAIAgentWorker`, `DurableAIAgentOrchestrationContext`, `AgentCallbackContext`, `AgentResponseCallbackProtocol`](#9-durable-ai-external-layer)
10. [Azure Functions app — `AgentFunctionApp`](#10-azure-functions-app)

---

## 1. AG-UI client layer

**Module:** `agent_framework.ag_ui`  
**Install:** `pip install agent-framework[ag-ui]`

AG-UI is Microsoft's streaming protocol for stateful AI frontends. The client layer
provides three classes that work together: `AGUIHttpService` handles raw HTTP/SSE
transport, `AGUIEventConverter` normalises the AG-UI event stream into Agent Framework
`ChatResponseUpdate` objects, and `AGUIChatClient` composes both into a drop-in
`BaseChatClient` that any `Agent` can use to call a remote AG-UI server as if it were
a local model.

### `AGUIHttpService`

```
AGUIHttpService(
    endpoint: str,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 60.0,
)
```

Manages the HTTP POST + SSE streaming lifecycle. `post_run()` is an async generator
that yields raw AG-UI event dicts parsed from the server-sent-events stream.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `endpoint` | — | Base URL of the AG-UI server, e.g. `"http://localhost:8888/"` |
| `http_client` | `None` | Bring-your-own `httpx.AsyncClient`; owned and closed if not provided |
| `timeout` | `60.0` | Per-request timeout in seconds |

Key behaviour: the class is also an async context manager — `async with AGUIHttpService(...) as svc` closes the internal client on exit.

**Example 1 — stream raw events from a remote AG-UI server:**

```python
import asyncio
from agent_framework.ag_ui import AGUIHttpService

async def main():
    async with AGUIHttpService("http://localhost:8888/") as svc:
        async for event in svc.post_run(
            thread_id="t-001",
            run_id="r-001",
            messages=[{"role": "user", "content": "Summarise today's news"}],
        ):
            print(event["type"], event.get("delta", ""))

asyncio.run(main())
```

**Example 2 — pass shared state and resume a paused run:**

```python
from agent_framework.ag_ui import AGUIHttpService

async def resume_with_state():
    async with AGUIHttpService("http://localhost:8888/", timeout=120.0) as svc:
        async for event in svc.post_run(
            thread_id="t-001",
            run_id="r-002",
            messages=[],
            state={"user_preference": "brief"},
            resume={"interrupt_id": "confirm-action", "decision": "approved"},
        ):
            if event["type"] == "TEXT_MESSAGE_CONTENT":
                print(event["delta"], end="", flush=True)
```

**Example 3 — pass tools so the remote server can invoke them:**

```python
from agent_framework.ag_ui import AGUIHttpService

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get the current price of a ticker",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        },
    }
]

async def with_tools():
    async with AGUIHttpService("https://agents.example.com/api/") as svc:
        async for event in svc.post_run(
            thread_id="t-finance",
            run_id="r-001",
            messages=[{"role": "user", "content": "What is MSFT trading at?"}],
            tools=tools,
        ):
            print(event)
```

---

### `AGUIEventConverter`

```
AGUIEventConverter()
```

Stateful converter that maps AG-UI event dicts to `ChatResponseUpdate` objects.
Instantiate one converter per streaming session — it holds accumulated tool-call
arguments and the current message ID across calls.

| Attribute | Type | Notes |
|-----------|------|-------|
| `current_message_id` | `str \| None` | ID of the message being streamed |
| `current_tool_call_id` | `str \| None` | ID of the active tool call |
| `current_tool_name` | `str \| None` | Name of the active tool |
| `accumulated_tool_args` | `str` | JSON string accumulated across `TOOL_CALL_ARGS_DELTA` events |
| `thread_id` / `run_id` | `str \| None` | Populated on `RUN_STARTED` |

**Example 1 — manually drive the converter event by event:**

```python
from agent_framework.ag_ui import AGUIEventConverter

converter = AGUIEventConverter()

events = [
    {"type": "RUN_STARTED", "threadId": "t-1", "runId": "r-1"},
    {"type": "TEXT_MESSAGE_START", "messageId": "m-1"},
    {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "Hello, "},
    {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "world!"},
    {"type": "TEXT_MESSAGE_END", "messageId": "m-1"},
    {"type": "RUN_FINISHED", "threadId": "t-1", "runId": "r-1"},
]

for ev in events:
    update = converter.convert_event(ev)
    if update and update.contents:
        print(update.contents[0].text, end="")
# Output: Hello, world!
```

**Example 2 — handle tool call assembly:**

```python
from agent_framework.ag_ui import AGUIEventConverter

converter = AGUIEventConverter()

for ev in [
    {"type": "TOOL_CALL_START", "toolCallId": "tc-1", "toolCallName": "get_price"},
    {"type": "TOOL_CALL_ARGS_DELTA", "toolCallId": "tc-1", "delta": '{"tick'},
    {"type": "TOOL_CALL_ARGS_DELTA", "toolCallId": "tc-1", "delta": 'er": "MSFT"}'},
    {"type": "TOOL_CALL_END", "toolCallId": "tc-1"},
]:
    update = converter.convert_event(ev)

# After TOOL_CALL_END the converter resets accumulated_tool_args to ""
print(converter.accumulated_tool_args)  # ""
```

**Example 3 — wire the converter into a raw SSE loop:**

```python
from agent_framework.ag_ui import AGUIHttpService, AGUIEventConverter

async def convert_stream(endpoint: str, thread_id: str, messages: list):
    converter = AGUIEventConverter()
    async with AGUIHttpService(endpoint) as svc:
        async for ev in svc.post_run(thread_id=thread_id, run_id="r-auto", messages=messages):
            update = converter.convert_event(ev)
            if update:
                yield update
```

---

### `AGUIChatClient`

```
AGUIChatClient(
    *,
    endpoint: str,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 60.0,
    additional_properties: dict[str, Any] | None = None,
    middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    function_invocation_configuration: FunctionInvocationConfiguration | None = None,
)
```

A `BaseChatClient` implementation — pass it to `Agent` just like `OpenAIChatClient`.
Internally creates an `AGUIHttpService` and `AGUIEventConverter`. Manages thread IDs
per session automatically.

**Example 1 — agent that calls a remote AG-UI server:**

```python
from agent_framework import Agent
from agent_framework.ag_ui import AGUIChatClient

client = AGUIChatClient(endpoint="http://ag-ui-server.internal:8888/")
agent = Agent(
    client=client,
    name="Relay",
    instructions="You relay requests to the remote agent and return the result.",
)

async def main():
    response = await agent.run("What is the capital of France?")
    print(response.text)
```

**Example 2 — override timeout for long-running analysis requests:**

```python
import httpx
from agent_framework import Agent
from agent_framework.ag_ui import AGUIChatClient

shared_http = httpx.AsyncClient(timeout=300.0)
client = AGUIChatClient(
    endpoint="http://analysis-agent.example.com/",
    http_client=shared_http,
    timeout=300.0,
)
agent = Agent(client=client, name="Analyst")
```

**Example 3 — attach retrying middleware to the AG-UI client:**

```python
from agent_framework import Agent
from agent_framework.ag_ui import AGUIChatClient
from agent_framework._middleware import RetryMiddleware  # if available in your install

client = AGUIChatClient(
    endpoint="http://flaky-agent.example.com/",
    middleware=[RetryMiddleware(max_retries=3)],
)
agent = Agent(client=client, name="Resilient")
```

---

## 2. AG-UI protocol wrappers

**Module:** `agent_framework.ag_ui`

These classes are the *server-side* counterparts: they expose Agent Framework agents
and workflows as AG-UI compliant endpoints that browser frontends can connect to.

### `AgentFrameworkAgent`

```
AgentFrameworkAgent(
    agent: SupportsAgentRun,
    name: str | None = None,
    description: str | None = None,
    state_schema: Any | None = None,
    predict_state_config: dict[str, dict[str, str]] | None = None,
    require_confirmation: bool = True,
    use_service_session: bool = False,
)
```

Wraps any `SupportsAgentRun` for AG-UI protocol compatibility. Translates
`AgentResponseUpdate` streams to the AG-UI event sequence:
`RunStarted → content events → RunFinished`. Manages a bounded registry of pending
tool approval requests (max 10,000 entries) to prevent replay attacks and function
name spoofing.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `state_schema` | `None` | Pydantic model or plain dict class; drives frontend state type hints |
| `predict_state_config` | `None` | Keys: tool names; values: `{"state_key": "...", "tool": "..."}` |
| `require_confirmation` | `True` | Predictive state updates wait for user confirmation before applying |
| `use_service_session` | `False` | Hand session management to the AG-UI service layer |

**Example 1 — minimal FastAPI endpoint:**

```python
from fastapi import FastAPI
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.ag_ui import AgentFrameworkAgent, add_agent_framework_fastapi_endpoint

agent = Agent(client=OpenAIChatClient(), name="Assistant")
wrapped = AgentFrameworkAgent(agent, name="Assistant", description="General-purpose AI assistant")

app = FastAPI()
add_agent_framework_fastapi_endpoint(app, wrapped, path="/")
```

**Example 2 — typed state schema with predictive updates:**

```python
from pydantic import BaseModel
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.ag_ui import AgentFrameworkAgent, state_update

class TodoState(BaseModel):
    items: list[str] = []
    count: int = 0

@tool
async def add_todo(task: str):
    """Add a task to the todo list."""
    return state_update(
        text=f"Added task: {task}",
        state={"items": ["__append__", task], "count": "__increment__"},
    )

agent = Agent(client=OpenAIChatClient(), name="TodoBot", tools=[add_todo])
wrapped = AgentFrameworkAgent(
    agent,
    state_schema=TodoState,
    predict_state_config={"add_todo": {"state_key": "items", "tool": "add_todo"}},
    require_confirmation=False,
)
```

**Example 3 — disable confirmation for fast-path tools:**

```python
from agent_framework.ag_ui import AgentFrameworkAgent

# Disable approval gate — predictive state applies immediately
wrapped = AgentFrameworkAgent(
    agent,
    require_confirmation=False,
    use_service_session=True,  # session managed by AG-UI service
)
```

---

### `AgentFrameworkWorkflow`

```
AgentFrameworkWorkflow(
    workflow: Workflow | None = None,
    *,
    workflow_factory: WorkflowFactory | None = None,
    name: str | None = None,
    description: str | None = None,
)
```

Server-side wrapper for `Workflow` objects. Pass either `workflow` (a pre-built
instance shared across all sessions) or `workflow_factory` (called per thread to
create isolated workflow instances). Passing both raises `ValueError`.

**Example 1 — wrap a pre-built workflow:**

```python
from agent_framework import WorkflowBuilder
from agent_framework.ag_ui import AgentFrameworkWorkflow, add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

wf = WorkflowBuilder().add_agent(...).build()
wrapped_wf = AgentFrameworkWorkflow(wf, name="Pipeline", description="Multi-step pipeline")

app = FastAPI()
add_agent_framework_fastapi_endpoint(app, wrapped_wf, path="/pipeline")
```

**Example 2 — factory pattern for isolated per-session workflows:**

```python
from agent_framework.declarative import WorkflowFactory
from agent_framework.ag_ui import AgentFrameworkWorkflow

def build_workflow():
    from agent_framework import WorkflowBuilder
    return WorkflowBuilder().add_agent(...).build()

wrapped_wf = AgentFrameworkWorkflow(
    workflow_factory=build_workflow,  # called fresh per thread_id
    name="IsolatedPipeline",
)
```

**Example 3 — custom run behaviour via subclassing:**

```python
from agent_framework.ag_ui import AgentFrameworkWorkflow
from agent_framework import WorkflowBuilder

class AuditedWorkflow(AgentFrameworkWorkflow):
    async def run(self, messages, *, thread_id, run_id, state=None, **kwargs):
        print(f"[audit] thread={thread_id} run={run_id}")
        async for event in super().run(messages, thread_id=thread_id, run_id=run_id, state=state, **kwargs):
            yield event

wf = WorkflowBuilder().add_agent(...).build()
audited = AuditedWorkflow(wf)
```

---

### `state_update()`

```
state_update(
    text: str = "",
    *,
    state: Mapping[str, Any] | None = None,
    tool_result: Any = <UNSET>,
) -> Content
```

Returns a `Content` object from inside an `@tool` function that simultaneously:
1. Sends `text` to the LLM as the `function_result` content.
2. Sends `tool_result` (or `text` as fallback) to the AG-UI frontend as the displayed tool output.
3. Merges `state` into `FlowState.current_state` and emits a deterministic `StateSnapshotEvent`.

**Example 1 — update a counter in shared state:**

```python
from agent_framework import tool
from agent_framework.ag_ui import state_update

@tool
async def increment_counter(amount: int = 1):
    """Increment the shared counter."""
    return state_update(
        text=f"Counter incremented by {amount}",
        state={"counter": amount},  # merged into FlowState.current_state
    )
```

**Example 2 — separate LLM text from UI display payload:**

```python
from agent_framework import tool
from agent_framework.ag_ui import state_update

@tool
async def fetch_chart_data(symbol: str):
    """Fetch chart data for a symbol."""
    data = {"open": 100, "high": 105, "low": 98, "close": 102}  # from real API
    return state_update(
        text=f"{symbol} OHLC: open={data['open']} close={data['close']}",  # for LLM
        tool_result={"type": "chart", "symbol": symbol, "data": data},       # for UI
        state={"last_symbol": symbol},
    )
```

**Example 3 — state-only update with empty text:**

```python
from agent_framework import tool
from agent_framework.ag_ui import state_update

@tool
async def set_user_preference(theme: str, language: str):
    """Save user preferences without modifying the conversation."""
    return state_update(
        text="Preferences saved.",
        state={"preferences": {"theme": theme, "language": language}},
    )
```

---

### `add_agent_framework_fastapi_endpoint()`

```
add_agent_framework_fastapi_endpoint(
    app: FastAPI,
    agent: SupportsAgentRun | AgentFrameworkAgent | Workflow | AgentFrameworkWorkflow,
    path: str = "/",
    state_schema: Any | None = None,
    predict_state_config: dict[str, dict[str, str]] | None = None,
    allow_origins: list[str] | None = None,
    default_state: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    dependencies: Sequence[Depends] | None = None,
) -> None
```

Convenience function that registers an AG-UI POST endpoint on a FastAPI app. Accepts
a raw `Agent` / `Workflow` and auto-wraps it, or accepts a pre-wrapped
`AgentFrameworkAgent` / `AgentFrameworkWorkflow`. Adds CORS middleware when
`allow_origins` is provided.

**Example 1 — register multiple agents on different paths:**

```python
from fastapi import FastAPI
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint

app = FastAPI()
chat = Agent(client=OpenAIChatClient(model="gpt-4o"), name="Chat")
coder = Agent(client=OpenAIChatClient(model="gpt-4o"), name="Coder",
              instructions="You are an expert programmer.")

add_agent_framework_fastapi_endpoint(app, chat, path="/chat")
add_agent_framework_fastapi_endpoint(app, coder, path="/code")
```

**Example 2 — CORS + default state for browser clients:**

```python
from fastapi import FastAPI
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint

app = FastAPI()
add_agent_framework_fastapi_endpoint(
    app,
    agent,
    path="/",
    allow_origins=["http://localhost:3000", "https://app.example.com"],
    default_state={"mode": "standard", "history_limit": 20},
)
```

**Example 3 — dependency injection for auth:**

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer

bearer = HTTPBearer()

def verify_token(token=Security(bearer)):
    if token.credentials != "secret-key":
        raise HTTPException(status_code=401, detail="Invalid token")

app = FastAPI()
add_agent_framework_fastapi_endpoint(
    app,
    agent,
    path="/secure",
    dependencies=[Depends(verify_token)],
)
```

---

## 3. ChatKit integration

**Module:** `agent_framework.chatkit`  
**Install:** `pip install agent-framework[chatkit]`

ChatKit is Microsoft's typed thread-item protocol for Teams / Copilot-style UIs.
This integration converts ChatKit `ThreadItem` objects into Agent Framework `Message`
objects, and converts `AgentResponseUpdate` streams back into ChatKit `ThreadStreamEvent`s.

### `ThreadItemConverter`

```
ThreadItemConverter(
    attachment_data_fetcher: Callable[[str], Awaitable[bytes]] | None = None,
)
```

Base converter class. Override individual `*_to_input` methods to customise how
specific thread item types map to messages. The `to_agent_input()` method is the
main entry point — it iterates a sequence of `ThreadItem` objects and calls the
appropriate handler for each.

| Method | Input type | Notes |
|--------|-----------|-------|
| `user_message_to_input` | `UserMessageItem` | Last message flag enables role inference |
| `assistant_message_to_input` | `AssistantMessageItem` | Returns `Message \| list[Message] \| None` |
| `hidden_context_to_input` | `HiddenContextItem` | System context injected outside conversation |
| `attachment_to_message_content` | `Attachment` | Uses `attachment_data_fetcher` to resolve bytes |
| `tag_to_message_content` | `UserMessageTagContent` | @-mentions converted to special content |
| `client_tool_call_to_input` | `ClientToolCallItem` | Client-side tool invocations |

**Example 1 — basic conversion using the default converter:**

```python
from agent_framework.chatkit import simple_to_agent_input

# simple_to_agent_input uses a default ThreadItemConverter instance
messages = simple_to_agent_input(thread_items)
response = await agent.run(messages)
```

**Example 2 — custom converter that resolves SharePoint attachment URLs:**

```python
import aiohttp
from agent_framework.chatkit import ThreadItemConverter

async def fetch_sharepoint_bytes(attachment_id: str) -> bytes:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://graph.microsoft.com/v1.0/me/drive/items/{attachment_id}/content") as r:
            return await r.read()

converter = ThreadItemConverter(attachment_data_fetcher=fetch_sharepoint_bytes)
messages = converter.to_agent_input(thread_items)
response = await agent.run(messages)
```

**Example 3 — override how @-mention tags are converted:**

```python
from agent_framework.chatkit import ThreadItemConverter
from agent_framework import Content

class MentionAwareConverter(ThreadItemConverter):
    def tag_to_message_content(self, tag) -> Content:
        # Convert @mention to a plain-text hint instead of default behaviour
        return Content(text=f"[mentioned: @{tag.display_name}]")

converter = MentionAwareConverter()
messages = converter.to_agent_input(thread_items)
```

---

### `stream_agent_response()`

```
stream_agent_response(
    response_stream: AsyncIterable[AgentResponseUpdate],
    thread_id: str,
    generate_id: Callable[[str], str] | None = None,
) -> AsyncIterator[ThreadStreamEvent]
```

Converts a streaming `AgentResponseUpdate` sequence from any Agent Framework agent
into the ChatKit `ThreadStreamEvent` union that a Teams / ChatKit frontend understands.
Handles text delta accumulation, tool calls, and structured outputs.

**Example 1 — stream an agent response back to a ChatKit frontend:**

```python
from agent_framework.chatkit import stream_agent_response, simple_to_agent_input
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/chat")
async def chat(body: dict):
    thread_id = body["thread_id"]
    thread_items = body["thread_items"]

    messages = simple_to_agent_input(thread_items)
    response_stream = await agent.run(messages, stream=True)

    async def generate():
        async for event in stream_agent_response(response_stream, thread_id=thread_id):
            yield f"data: {json.dumps(event.model_dump())}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Example 2 — collect all events into a list for batch responses:**

```python
from agent_framework.chatkit import stream_agent_response

events = []
response_stream = await agent.run("Summarise this", stream=True)
async for event in stream_agent_response(
    response_stream,
    thread_id="t-001",
):
    events.append(event)

# events contains ThreadItemAddedEvent, ThreadItemUpdatedEvent, ThreadItemDoneEvent, etc.
print(events[-1].type)  # "threadItem.done"
```

**Example 3 — custom ID generation for deterministic event IDs:**

```python
import hashlib
from agent_framework.chatkit import stream_agent_response

def deterministic_id(prefix: str) -> str:
    return f"{prefix}-{hashlib.md5(prefix.encode()).hexdigest()[:8]}"

response_stream = await agent.run("Hello", stream=True)
async for event in stream_agent_response(
    response_stream,
    thread_id="t-fixed",
    generate_id=deterministic_id,
):
    print(event.type, getattr(event, "id", ""))
```

---

## 4. Developer server

**Module:** `agent_framework.devui`  
**Install:** `pip install agent-framework[devui]`

The DevUI provides an OpenAI-compatible HTTP server for local debugging of agents.
Two entry points: `serve()` (one-liner) and `DevServer` (programmatic control).

### `DevServer`

```
DevServer(
    entities_dir: str | None = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    cors_origins: list[str] | None = None,
    ui_enabled: bool = True,
    mode: str = "developer",
    auth_enabled: bool = True,
    auth_token: str | None = None,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `entities_dir` | `None` | Directory to scan for agent/workflow Python files |
| `mode` | `"developer"` | `"developer"`: full access + verbose errors; `"user"`: restricted APIs + generic errors |
| `auth_enabled` | `True` | Requires `Authorization: Bearer <token>` on `/v1/*` endpoints |
| `auth_token` | `None` | Falls back to `DEVUI_AUTH_TOKEN` env var; auto-generated on loopback if still `None` |
| `cors_origins` | `None` | Explicit CORS allowlist; previous wildcard-on-localhost default was removed as a security fix |

**Example 1 — programmatic server with in-memory entities:**

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import DevServer

agent = Agent(client=OpenAIChatClient(), name="debug-agent")
server = DevServer(port=8080, auth_enabled=False)
server.register_entities([agent])
app = server.get_app()

import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8080)
```

**Example 2 — directory-based entity scanning:**

```python
from agent_framework.devui import DevServer

server = DevServer(
    entities_dir="./agents",      # scans *.py files for Agent/Workflow instances
    port=9090,
    mode="user",                  # hide verbose errors from end users
    auth_enabled=True,
    auth_token="my-dev-token",
)
uvicorn.run(server.get_app(), host="0.0.0.0", port=9090)
```

**Example 3 — add agents dynamically after construction:**

```python
from agent_framework.devui import DevServer

server = DevServer(auth_enabled=False)
app = server.create_app()  # build the FastAPI app

# Add agents later (e.g. after loading config)
server.set_pending_entities([agent_a, agent_b])
uvicorn.run(app, port=8080)
```

---

### `serve()` and `register_cleanup()`

```
serve(
    entities: list[Any] | None = None,
    entities_dir: str | None = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    auto_open: bool = False,
    cors_origins: list[str] | None = None,
    ui_enabled: bool = True,
    instrumentation_enabled: bool = False,
    mode: str = "developer",
    auth_enabled: bool = True,
    auth_token: str | None = None,
) -> None
```

One-liner entry point. Validates `host` (regex gate prevents injection) and `port`
(must be 1–65535 integer). When `instrumentation_enabled=True`, calls
`enable_instrumentation(enable_sensitive_data=True)` before starting.

```
register_cleanup(entity: Any, *hooks: Callable[[], Any]) -> None
```

Register one or more cleanup callables (sync or async) that run during DevUI server
shutdown, before entity clients are closed.

**Example 1 — simplest possible local dev server:**

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import serve

agent = Agent(client=OpenAIChatClient(), name="MyAgent")
serve(entities=[agent], port=8080, auth_enabled=False)
```

**Example 2 — serve with OpenTelemetry enabled:**

```python
from agent_framework.devui import serve

serve(
    entities=[agent],
    instrumentation_enabled=True,  # calls enable_instrumentation(enable_sensitive_data=True)
    port=8080,
    auto_open=True,
)
```

**Example 3 — register a cleanup hook to flush a database connection:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import serve, register_cleanup

pool = None  # your DB connection pool

async def close_pool():
    if pool:
        await pool.close()

agent = Agent(client=OpenAIChatClient(), name="DBAgent")
register_cleanup(agent, close_pool)  # runs on server shutdown
serve(entities=[agent], port=8080)
```

---

## 5. GAIA benchmark harness

**Module:** `agent_framework.lab.gaia`  
**Install:** `pip install agent-framework[lab]`

GAIA (General AI Assistants benchmark) is a public benchmark for evaluating
general-purpose AI assistants. The Agent Framework lab module provides a typed
harness for running GAIA tasks with custom agents, including tracing integration.

### Core types

```python
@dataclass
class Task:
    task_id: str
    question: str
    answer: str | None = None
    level: int | None = None
    file_name: str | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class Prediction:
    prediction: str
    messages: list[Any] | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class Evaluation:
    is_correct: bool
    score: float
    details: dict[str, Any] | None = None

@dataclass
class TaskResult:
    task_id: str
    task: Task
    prediction: Prediction
    evaluation: Evaluation
    runtime_seconds: float | None = None
    error: str | None = None
```

### `GAIA`

```
GAIA(
    evaluator: Evaluator | None = None,
    data_dir: str | None = None,
    hf_token: str | None = None,
    telemetry_config: GAIATelemetryConfig | None = None,
)
```

Loads GAIA tasks (from HuggingFace Hub or a local directory) and runs them through
a custom `TaskRunner`. The `run()` method returns `list[TaskResult]`.

```
GAIA.run(
    task_runner: TaskRunner,
    level: int | list[int] = 1,
    max_n: int | None = None,
    parallel: int = 1,
    timeout: int | None = None,
    out: str | None = None,
) -> list[TaskResult]
```

| Parameter | Notes |
|-----------|-------|
| `level` | GAIA difficulty level — 1 (easiest) to 3 (hardest), or list of levels |
| `max_n` | Cap on tasks to run (useful for quick smoke tests) |
| `parallel` | Concurrent task runners |
| `out` | JSON output file path for results |

### `GAIATelemetryConfig`

```
GAIATelemetryConfig(
    enable_tracing: bool = False,
    otlp_endpoint: str | None = None,
    applicationinsights_connection_string: str | None = None,
    trace_to_file: bool = False,
    file_path: str | None = None,
)
```

Provides OpenTelemetry configuration for the benchmark run. Call `.setup_observability()`
before `GAIA.run()` to activate tracing.

### `TaskRunner` protocol

```python
class TaskRunner(Protocol):
    async def run(self, task: Task) -> Prediction:
        ...
```

Implement this protocol to make any agent a GAIA task runner.

**Example 1 — run GAIA level-1 tasks with a simple agent:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.lab.gaia import GAIA, Task, Prediction

class MyTaskRunner:
    def __init__(self, agent: Agent):
        self.agent = agent

    async def run(self, task: Task) -> Prediction:
        response = await self.agent.run(task.question)
        return Prediction(prediction=response.text, messages=response.messages)

async def main():
    agent = Agent(client=OpenAIChatClient(model="gpt-4o"), name="GAIA-Agent")
    gaia = GAIA(hf_token="hf_xxxxx")
    results = await gaia.run(MyTaskRunner(agent), level=1, max_n=10)
    correct = sum(1 for r in results if r.evaluation.is_correct)
    print(f"Score: {correct}/{len(results)}")

asyncio.run(main())
```

**Example 2 — enable tracing with Azure Monitor:**

```python
from agent_framework.lab.gaia import GAIA, GAIATelemetryConfig

config = GAIATelemetryConfig(
    enable_tracing=True,
    applicationinsights_connection_string="InstrumentationKey=...",
)
config.setup_observability()

gaia = GAIA(hf_token="hf_xxxxx", telemetry_config=config)
results = await gaia.run(runner, level=[1, 2], parallel=4)
```

**Example 3 — write results to JSON and evaluate accuracy per level:**

```python
from agent_framework.lab.gaia import GAIA
import json

gaia = GAIA(data_dir="./gaia_data")
results = await gaia.run(runner, level=[1, 2, 3], out="results.json")

for level in [1, 2, 3]:
    lvl_results = [r for r in results if r.task.level == level]
    if lvl_results:
        score = sum(r.evaluation.score for r in lvl_results) / len(lvl_results)
        print(f"Level {level}: avg score = {score:.3f}")

# results.json contains serialised TaskResult list
with open("results.json") as f:
    saved = json.load(f)
print(f"Saved {len(saved)} results")
```

---

## 6. Copilot Studio bridge

**Module:** `agent_framework.microsoft`  
**Install:** `pip install agent-framework[copilotstudio]`

`CopilotStudioAgent` connects to an existing Microsoft Copilot Studio bot and
exposes it as an Agent Framework `Agent`-compatible participant — enabling it to
join orchestrations, receive HITL pauses, and be used as a tool.

### `CopilotStudioAgent`

```
CopilotStudioAgent(
    client: CopilotClient | None = None,
    settings: ConnectionSettings | None = None,
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: list[AgentMiddlewareTypes] | None = None,
    environment_id: str | None = None,
    agent_identifier: str | None = None,
    client_id: str | None = None,
    tenant_id: str | None = None,
    token: str | None = None,
    cloud: PowerPlatformCloud | None = None,
    agent_type: AgentType | None = None,
    custom_power_platform_cloud: str | None = None,
    username: str | None = None,
    token_cache: Any | None = None,
    scopes: list[str] | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> None
```

### `CopilotStudioSettings` TypedDict

Environment variables are loaded with prefix `COPILOTSTUDIOAGENT__`:

| Key | Env var | Notes |
|-----|---------|-------|
| `environmentid` | `COPILOTSTUDIOAGENT__ENVIRONMENTID` | Power Platform environment GUID |
| `schemaname` | `COPILOTSTUDIOAGENT__SCHEMANAME` | Agent schema name (identifier) in Copilot Studio |
| `agentappid` | `COPILOTSTUDIOAGENT__AGENTAPPID` | App Registration client ID |
| `tenantid` | `COPILOTSTUDIOAGENT__TENANTID` | AAD tenant ID |

**Example 1 — minimal env-file based connection:**

```python
# .env file:
# COPILOTSTUDIOAGENT__ENVIRONMENTID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# COPILOTSTUDIOAGENT__SCHEMANAME=my_hr_bot
# COPILOTSTUDIOAGENT__AGENTAPPID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# COPILOTSTUDIOAGENT__TENANTID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

from agent_framework.microsoft import CopilotStudioAgent

cs_agent = CopilotStudioAgent(env_file_path=".env")
response = await cs_agent.run("What is the company leave policy?")
print(response.text)
```

**Example 2 — use a token string instead of app registration:**

```python
from agent_framework.microsoft import CopilotStudioAgent

token = "eyJ..."  # acquired externally via MSAL or Azure SDK

cs_agent = CopilotStudioAgent(
    environment_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    agent_identifier="my_support_bot",
    token=token,
)
```

**Example 3 — use `CopilotStudioAgent` as a tool inside a larger orchestration:**

```python
from agent_framework import Agent, WorkflowBuilder
from agent_framework.microsoft import CopilotStudioAgent
from agent_framework.openai import OpenAIChatClient

cs_agent = CopilotStudioAgent(env_file_path=".env")
cs_tool = cs_agent.as_tool(
    name="hr_bot",
    description="Answer HR policy questions using Copilot Studio.",
    propagate_session=True,  # share session state with the bot
)

orchestrator = Agent(
    client=OpenAIChatClient(),
    name="Router",
    tools=[cs_tool],
    instructions="Route HR questions to hr_bot and handle other requests yourself.",
)
response = await orchestrator.run("How many vacation days do I get?")
print(response.text)
```

---

## 7. Azure AI Search context provider

**Module:** `agent_framework.azure`  
**Install:** `pip install agent-framework[azure-ai-search]`

### `AzureAISearchContextProvider`

```
AzureAISearchContextProvider(
    source_id: str = "azure_ai_search",
    endpoint: str | None = None,
    index_name: str | None = None,
    api_key: str | AzureKeyCredential | None = None,
    credential: AzureCredentialTypes | None = None,
    *,
    mode: Literal["semantic", "agentic"] = "semantic",
    top_k: int = 5,
    semantic_configuration_name: str | None = None,
    vector_field_name: str | None = None,
    embedding_function: EmbeddingFunction | None = None,
    context_prompt: str | None = None,
    azure_openai_resource_url: str | None = None,
    model: str | None = None,
    knowledge_base_name: str | None = None,
    retrieval_instructions: str | None = None,
    azure_openai_api_key: str | None = None,
    knowledge_base_output_mode: KnowledgeBaseOutputModeLiteral = "extractive_data",
    retrieval_reasoning_effort: RetrievalReasoningEffortLiteral = "minimal",
    agentic_message_history_count: int = 10,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> None
```

Implements the `ContextProvider` hooks pattern (`before_run` / `after_run`).
Supports two modes:

| `mode` | How it retrieves | Requires |
|--------|-----------------|---------|
| `"semantic"` | Keyword + semantic ranking (`top_k` hits) | `index_name`, optional `semantic_configuration_name` |
| `"agentic"` | AI-driven multi-turn retrieval via Knowledge Base | `knowledge_base_name`, `azure_openai_resource_url`, `model` |

`AzureAISearchSettings` env-var prefix: `AZURE_SEARCH_`

| Env var | Key |
|---------|-----|
| `AZURE_SEARCH_ENDPOINT` | `endpoint` |
| `AZURE_SEARCH_INDEX_NAME` | `index_name` |
| `AZURE_SEARCH_KNOWLEDGE_BASE_NAME` | `knowledge_base_name` |
| `AZURE_SEARCH_API_KEY` | `api_key` |

**Example 1 — semantic search with managed identity:**

```python
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.azure import AzureAISearchContextProvider
from agent_framework.openai import OpenAIChatClient

search_provider = AzureAISearchContextProvider(
    endpoint="https://my-search.search.windows.net",
    index_name="products",
    credential=DefaultAzureCredential(),
    mode="semantic",
    top_k=8,
    semantic_configuration_name="semantic-config-1",
)

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[search_provider],
    name="ProductSearch",
)
response = await agent.run("What products support LDAP authentication?")
print(response.text)
```

**Example 2 — agentic mode with Knowledge Base:**

```python
from agent_framework.azure import AzureAISearchContextProvider

agentic_provider = AzureAISearchContextProvider(
    endpoint="https://my-search.search.windows.net",
    knowledge_base_name="my-kb",
    azure_openai_resource_url="https://my-openai.openai.azure.com",
    model="gpt-4o",
    mode="agentic",
    retrieval_reasoning_effort="standard",  # "minimal" | "standard" | "high"
    agentic_message_history_count=5,        # how many prior messages to include
)
```

**Example 3 — load settings from environment with `.env` file:**

```python
# .env:
# AZURE_SEARCH_ENDPOINT=https://my-search.search.windows.net
# AZURE_SEARCH_INDEX_NAME=docs
# AZURE_SEARCH_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from agent_framework.azure import AzureAISearchContextProvider

provider = AzureAISearchContextProvider(env_file_path=".env")
agent = Agent(client=..., context_providers=[provider])
```

---

## 8. Cosmos DB history provider

**Module:** `agent_framework.azure`  
**Install:** `pip install agent-framework[azure-cosmos]`

### `CosmosHistoryProvider`

```
CosmosHistoryProvider(
    source_id: str = "azure_cosmos_history",
    *,
    load_messages: bool = True,
    store_outputs: bool = True,
    store_inputs: bool = True,
    store_context_messages: bool = False,
    store_context_from: set[str] | None = None,
    endpoint: str | None = None,
    database_name: str | None = None,
    container_name: str | None = None,
    credential: str | AzureCredentialTypes | None = None,
    cosmos_client: CosmosClient | None = None,
    container_client: ContainerProxy | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> None
```

Azure Cosmos DB-backed `HistoryProvider`. Persists conversation messages so agents
can resume across process restarts. Implements `before_run` (loads prior history) and
`after_run` (stores new messages).

| Env var | Field |
|---------|-------|
| `AZURE_COSMOS_ENDPOINT` | `endpoint` |
| `AZURE_COSMOS_DATABASE_NAME` | `database_name` |
| `AZURE_COSMOS_CONTAINER_NAME` | `container_name` |
| `AZURE_COSMOS_KEY` | `credential` (when a key string is needed) |

| Parameter | Notes |
|-----------|-------|
| `store_context_messages` | If `True`, also persist context injected by `ContextProvider`s |
| `store_context_from` | Restrict storage to specific `source_id`s; only active when `store_context_messages=True` |
| `cosmos_client` | Bring-your-own `CosmosClient` (avoids recreating connections) |
| `container_client` | Fixed-container usage; bypasses database/container resolution |

**Example 1 — persist agent history to Cosmos DB with managed identity:**

```python
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.azure import CosmosHistoryProvider
from agent_framework.openai import OpenAIChatClient

history = CosmosHistoryProvider(
    endpoint="https://my-cosmos.documents.azure.com:443/",
    database_name="agents-db",
    container_name="conversation-history",
    credential=DefaultAzureCredential(),
)

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[history],
    name="PersistentAgent",
)

session = agent.create_session(session_id="user-42")
response = await agent.run("Remind me what we discussed last time.", session=session)
print(response.text)
```

**Example 2 — share a pre-created CosmosClient across providers:**

```python
from azure.cosmos.aio import CosmosClient
from azure.identity import DefaultAzureCredential
from agent_framework.azure import CosmosHistoryProvider

cosmos = CosmosClient(
    url="https://my-cosmos.documents.azure.com:443/",
    credential=DefaultAzureCredential(),
)

history = CosmosHistoryProvider(
    database_name="agents-db",
    container_name="history",
    cosmos_client=cosmos,  # shared — not owned by the provider
)
```

**Example 3 — selectively store context from specific providers:**

```python
from agent_framework.azure import CosmosHistoryProvider, AzureAISearchContextProvider

search = AzureAISearchContextProvider(source_id="my_search", ...)

history = CosmosHistoryProvider(
    endpoint="...",
    database_name="agents-db",
    container_name="history",
    store_context_messages=True,
    store_context_from={"my_search"},  # only persist search context, not other providers
)

agent = Agent(client=..., context_providers=[search, history])
```

---

## 9. Durable AI external layer

**Module:** `agent_framework.azure`  
**Install:** `pip install agent-framework[durabletask]`

Three classes bridge Agent Framework agents with the Durable Task scheduling runtime,
enabling long-running, checkpointed agent conversations that survive restarts.

### `DurableAIAgentWorker`

```
DurableAIAgentWorker(
    worker: TaskHubGrpcWorker,
    callback: AgentResponseCallbackProtocol | None = None,
)
```

Wraps a `TaskHubGrpcWorker` and allows registering `Agent` instances as durable
entities.

```
DurableAIAgentWorker.add_agent(
    agent: SupportsAgentRun,
    callback: AgentResponseCallbackProtocol | None = None,
) -> None
```

**Example 1 — register agents and start the worker:**

```python
from durabletask.worker import TaskHubGrpcWorker
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import DurableAIAgentWorker

agent = Agent(client=OpenAIChatClient(), name="durable-assistant")

worker = TaskHubGrpcWorker(host_address="localhost:4001")
durable_worker = DurableAIAgentWorker(worker)
durable_worker.add_agent(agent)
durable_worker.start()
```

**Example 2 — callback to stream updates to Azure Service Bus:**

```python
from agent_framework.azure import DurableAIAgentWorker, AgentCallbackContext, AgentResponseCallbackProtocol
from agent_framework._types import AgentResponse, AgentResponseUpdate

class ServiceBusCallback(AgentResponseCallbackProtocol):
    async def on_agent_response(self, response: AgentResponse, ctx: AgentCallbackContext):
        await bus_client.send_message(
            topic="agent-responses",
            body={"agent": ctx.agent_name, "text": response.text, "thread_id": ctx.thread_id},
        )

    async def on_streaming_response_update(self, update: AgentResponseUpdate, ctx: AgentCallbackContext):
        pass  # ignore streaming updates for final-only pattern

durable_worker = DurableAIAgentWorker(worker, callback=ServiceBusCallback())
durable_worker.add_agent(agent)
durable_worker.start()
```

**Example 3 — graceful shutdown:**

```python
import signal

durable_worker = DurableAIAgentWorker(worker)
durable_worker.add_agent(agent)
durable_worker.start()

def shutdown(sig, frame):
    durable_worker.stop()

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
```

---

### `DurableAIAgentClient`

```
DurableAIAgentClient(
    client: TaskHubGrpcClient,
    max_poll_retries: int = 30,
    poll_interval_seconds: float = 1.0,
)
```

Used in **external contexts** (e.g. HTTP handler, CLI) to get a proxy to a durable
agent and invoke it synchronously (relative to the durable runtime).

```
DurableAIAgentClient.get_agent(agent_name: str) -> DurableAIAgent[AgentResponse]
```

**Example 1 — invoke a durable agent from an HTTP endpoint:**

```python
from durabletask import TaskHubGrpcClient
from agent_framework.azure import DurableAIAgentClient
from fastapi import FastAPI

client = DurableAIAgentClient(
    TaskHubGrpcClient(host_address="localhost:4001"),
    max_poll_retries=60,
    poll_interval_seconds=0.5,
)

app = FastAPI()

@app.post("/ask")
def ask(body: dict):
    agent = client.get_agent("durable-assistant")
    response = agent.run(body["message"])  # synchronous — FastAPI runs this in a thread pool
    return {"text": response.text}
```

**Example 2 — poll until response with explicit retry config:**

```python
from durabletask import TaskHubGrpcClient
from agent_framework.azure import DurableAIAgentClient

# Increase retries for slow agents (max_poll_retries * poll_interval_seconds = max wait)
client = DurableAIAgentClient(
    TaskHubGrpcClient(host_address="scheduler.internal:4001"),
    max_poll_retries=120,      # up to 120 polls
    poll_interval_seconds=2.0, # 2 seconds apart = max 4 minutes
)
proxy = client.get_agent("long-running-analyst")
result = proxy.run("Analyse the Q2 financial data and produce a 5-page summary.")
print(result.text)
```

**Example 3 — reuse a single client across multiple requests:**

```python
from durabletask import TaskHubGrpcClient
from agent_framework.azure import DurableAIAgentClient
from fastapi import FastAPI

# Module-level singleton
_client = DurableAIAgentClient(TaskHubGrpcClient(host_address="localhost:4001"))

app = FastAPI()

@app.post("/chat/{agent_name}")
def chat(agent_name: str, body: dict):
    agent = _client.get_agent(agent_name)
    response = agent.run(body["message"])  # synchronous — FastAPI runs this in a thread pool
    return {"text": response.text}
```

---

### `DurableAIAgentOrchestrationContext`

```
DurableAIAgentOrchestrationContext(context: OrchestrationContext)
```

Used **inside Durable orchestration functions** (not from external clients) to
dispatch work to durable agents as activities.

```
DurableAIAgentOrchestrationContext.get_agent(agent_name: str) -> DurableAIAgent[DurableAgentTask]
```

The returned `DurableAIAgent.run()` here returns a `Task` (not `AgentResponse`) —
`yield` it or `await` it per the Durable Task API.

**Example 1 — orchestration that chains two agents:**

```python
from durabletask import Orchestration
from agent_framework.azure import DurableAIAgentOrchestrationContext

def my_pipeline(ctx):
    af_ctx = DurableAIAgentOrchestrationContext(ctx)
    researcher = af_ctx.get_agent("researcher")
    writer = af_ctx.get_agent("writer")

    research = yield researcher.run("Research renewable energy in 2026")
    article = yield writer.run(f"Write an article based on: {research.text}")
    return article.text
```

**Example 2 — fan-out pattern (parallel agents):**

```python
from durabletask import Orchestration
from agent_framework.azure import DurableAIAgentOrchestrationContext

def parallel_analysis(ctx):
    af_ctx = DurableAIAgentOrchestrationContext(ctx)
    analyst = af_ctx.get_agent("analyst")

    tasks = [analyst.run(f"Analyse region {r}") for r in ["NA", "EU", "APAC"]]
    results = yield ctx.task_all(tasks)
    return [r.text for r in results]
```

**Example 3 — error handling inside orchestration:**

```python
from durabletask import Orchestration, TaskFailedError
from agent_framework.azure import DurableAIAgentOrchestrationContext

def safe_pipeline(ctx):
    af_ctx = DurableAIAgentOrchestrationContext(ctx)
    agent = af_ctx.get_agent("analyst")
    try:
        result = yield agent.run("Complex task")
        return result.text
    except TaskFailedError as e:
        return f"Agent failed: {e}"
```

---

### `AgentCallbackContext` and `AgentResponseCallbackProtocol`

```python
@dataclass(frozen=True)
class AgentCallbackContext:
    agent_name: str
    correlation_id: str
    thread_id: str | None = None
    request_message: str | None = None

class AgentResponseCallbackProtocol(Protocol):
    async def on_agent_response(
        self, response: AgentResponse, context: AgentCallbackContext
    ) -> None: ...

    async def on_streaming_response_update(
        self, update: AgentResponseUpdate, context: AgentCallbackContext
    ) -> None: ...
```

`AgentCallbackContext` is frozen — all fields are set at callback dispatch time and
cannot be mutated. `correlation_id` is unique per invocation.

---

## 10. Azure Functions app

**Module:** `agent_framework.azure`  
**Install:** `pip install agent-framework[azure-functions]`

### `AgentFunctionApp`

```
AgentFunctionApp(
    agents: list[SupportsAgentRun] | None = None,
    workflow: Workflow | None = None,
    http_auth_level: func.AuthLevel = func.AuthLevel.FUNCTION,
    enable_health_check: bool = True,
    enable_http_endpoints: bool = True,
    max_poll_retries: int = 30,
    poll_interval_seconds: float = 1.0,
    enable_mcp_tool_trigger: bool = False,
    default_callback: AgentResponseCallbackProtocol | None = None,
)
```

Extends the Azure Functions `DFApp` with durable-entity-based agent hosting.
Agents registered here are exposed as:
- HTTP endpoints (when `enable_http_endpoints=True`)
- MCP tool triggers (when `enable_mcp_tool_trigger=True`)
- A health check endpoint at `GET /api/health` (when `enable_health_check=True`)

Polling defaults: `30 retries × 1.0 s = 30 s` maximum wait per request.

When `workflow` is passed, agents are extracted from the workflow's executors and
registered automatically — no need to list them in `agents`.

**Example 1 — single agent function app:**

```python
import azure.functions as func
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AgentFunctionApp

agent = Agent(
    client=OpenAIChatClient(),
    name="SupportAgent",
    instructions="You are a friendly customer support representative.",
)

app = AgentFunctionApp(
    agents=[agent],
    http_auth_level=func.AuthLevel.FUNCTION,
)
```

**Example 2 — MCP tool triggers + custom callback:**

```python
import azure.functions as func
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AgentFunctionApp, AgentResponseCallbackProtocol, AgentCallbackContext
from agent_framework._types import AgentResponse, AgentResponseUpdate
import logging

class LoggingCallback(AgentResponseCallbackProtocol):
    async def on_agent_response(self, response: AgentResponse, ctx: AgentCallbackContext):
        logging.info(f"[{ctx.agent_name}] completed thread={ctx.thread_id} tokens={len(response.text)//4}")

    async def on_streaming_response_update(self, update: AgentResponseUpdate, ctx: AgentCallbackContext):
        pass

agent = Agent(client=OpenAIChatClient(), name="CodeHelper")
app = AgentFunctionApp(
    agents=[agent],
    enable_mcp_tool_trigger=True,   # expose as MCP tool
    enable_http_endpoints=True,
    max_poll_retries=60,
    poll_interval_seconds=2.0,
    default_callback=LoggingCallback(),
)
```

**Example 3 — workflow-based app with per-agent callbacks:**

```python
import azure.functions as func
from agent_framework import WorkflowBuilder
from agent_framework.azure import AgentFunctionApp, AgentResponseCallbackProtocol

class AuditCallback(AgentResponseCallbackProtocol):
    async def on_agent_response(self, response, ctx):
        await audit_log.write(ctx.correlation_id, ctx.agent_name, response.text)

    async def on_streaming_response_update(self, update, ctx):
        pass

wf = WorkflowBuilder().add_agent(researcher).add_agent(writer).build()

app = AgentFunctionApp(
    workflow=wf,  # agents extracted automatically from workflow executors
    http_auth_level=func.AuthLevel.ANONYMOUS,
    enable_health_check=True,
)

# Add a per-agent callback after construction
app.add_agent(extra_agent, callback=AuditCallback())
```

---

## Key behaviours summary

| Class | Module | First coverage | Why it matters |
|-------|--------|---------------|----------------|
| `AGUIHttpService` | `ag_ui` | Vol. 15 | Low-level SSE transport; use directly when you need raw event access |
| `AGUIEventConverter` | `ag_ui` | Vol. 15 | Stateful normaliser; one instance per streaming session |
| `AGUIChatClient` | `ag_ui` | Vol. 15 | Drop-in `BaseChatClient` for calling remote AG-UI servers |
| `AgentFrameworkAgent` | `ag_ui` | Vol. 15 | Exposes agents as AG-UI servers with bounded approval registry |
| `AgentFrameworkWorkflow` | `ag_ui` | Vol. 15 | Exposes workflows as AG-UI servers; `workflow_factory` for thread isolation |
| `state_update()` | `ag_ui` | Vol. 15 | Decouples LLM content from UI display payload + state merge |
| `ThreadItemConverter` | `chatkit` | Vol. 15 | Extensible ChatKit→Framework message bridge |
| `stream_agent_response()` | `chatkit` | Vol. 15 | Framework→ChatKit streaming adapter |
| `DevServer` | `devui` | Vol. 15 | OpenAI-compatible local debug server; two modes, CORS-safe |
| `serve()` | `devui` | Vol. 15 | One-liner launch with host/port validation |
| `register_cleanup()` | `devui` | Vol. 15 | Shutdown hooks for resource cleanup |
| `GAIA` | `lab.gaia` | Vol. 15 | Full GAIA benchmark runner with optional OTel tracing |
| `GAIATelemetryConfig` | `lab.gaia` | Vol. 15 | Configures OTLP / Azure Monitor / file tracing for benchmarks |
| `CopilotStudioAgent` | `microsoft` | Vol. 15 | Bridge to Copilot Studio bots; exposes as `AgentSession` + tool |
| `AzureAISearchContextProvider` | `azure` | Vol. 15 | Semantic + agentic retrieval; `before_run` / `after_run` hooks |
| `CosmosHistoryProvider` | `azure` | Vol. 15 | Persistent conversation history in Cosmos DB |
| `DurableAIAgentWorker` | `azure` | Vol. 15 | Registers agents with durable-task scheduler |
| `DurableAIAgentClient` | `azure` | Vol. 15 | External (HTTP-layer) proxy to durable agents |
| `DurableAIAgentOrchestrationContext` | `azure` | Vol. 15 | Orchestration-internal proxy; returns `Task`, not `AgentResponse` |
| `AgentFunctionApp` | `azure` | Vol. 15 | Azure Functions host with HTTP + MCP triggers + health check |

---

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Sharing one `AGUIEventConverter` across multiple threads | Garbled tool args accumulation | Create one `AGUIEventConverter()` per streaming session |
| Passing both `workflow=` and `workflow_factory=` to `AgentFrameworkWorkflow` | `ValueError` at init | Pass only one |
| Calling `DurableAIAgentOrchestrationContext.get_agent().run()` without `yield` inside an orchestration | Task never executes | Always `yield` the `Task` returned by `.run()` in orchestration functions |
| Omitting `COPILOTSTUDIOAGENT__` prefix in env vars | `CopilotStudioAgent` falls back to empty settings | Prefix all four env vars with `COPILOTSTUDIOAGENT__` |
| Setting `cors_origins=None` on `DevServer` and expecting cross-origin requests to work | 403 from CORS policy | Pass explicit `cors_origins=["http://localhost:3000"]` |
| Using `serve()` with `host` containing shell metacharacters | `ValueError` from regex gate | Use only `localhost`, `127.0.0.1`, `0.0.0.0`, or a valid hostname |
| Passing `cosmos_client=` AND `container_client=` separately when a `container_client` is used | Both accepted but `container_client` takes priority | Use `container_client` only when you want a fixed container — it bypasses all resolution |

---

## Revision history

| Date | Version | Notes |
|------|---------|-------|
| June 2026 | 1.8.1 | Vol. 15 — first coverage of AG-UI, ChatKit, DevUI, GAIA lab, CopilotStudioAgent, AzureAISearchContextProvider, CosmosHistoryProvider, Durable external layer, AgentFunctionApp |

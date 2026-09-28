---
title: "Microsoft Agent Framework (Python) — 10-API Deep Dives Vol. 6 (1.19.0)"
description: "Source-verified deep dives for Agent, RawAgent, AgentSession, AgentResponse, FunctionTool, WorkflowAgent, WorkflowContext, CompactionProvider, VectorCollectionContextProvider, and TodoItem/TodoStore/TodoFileStore — all verified against agent-framework 1.19.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-API Deep Dives Vol. 6

**Verified against:** `agent-framework==1.19.0`
**Python requirement:** 3.10+

This volume covers the core building blocks every agent application depends on: the two concrete agent classes, the multi-turn session container, the response object, low-level tool construction, embedding a workflow as an agent, the executor communication context, conversation-history lifecycle management, a vector-collection context provider for RAG, and the session-scoped task-tracking trio. Each section includes the verified constructor signature, every meaningful method or property, and self-contained runnable examples.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

See [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) for `WorkflowBuilder`, `SlidingWindowStrategy`, `TruncationStrategy`, `ContextWindowCompactionStrategy`, `LocalEvaluator`, `InlineSkill`, `FileAccessProvider`, `MemoryContextProvider`, `FileHistoryProvider`, and `MCPWebsocketTool`.

See [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) for `VectorStoreField`, `VectorStoreCollectionDefinition`, `InMemoryCollection`, `InMemoryStore`, `Filter`, `FilterGroup`, `SecretString`, `load_settings`, `create_agent_hooks_middleware`, and `GroupChatBuilder`.

See [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) for `WorkflowEvent`, `AgentContext`, `MiddlewareBundle`, `ConversationSplit`/`ConversationSplitter`, `VectorStoreHistoryProvider`, `MemoryStore`/`MemoryFileStore`, `MemoryTopicRecord`, `WorkflowRunResult`, `FunctionInvocationContext`, and `ChatOptions`.

---

## 1. `Agent`

**Module:** `agent_framework._agents` (re-exported via `agent_framework`)

`Agent` is the primary, production-recommended agent class. It layers OpenTelemetry-based telemetry and the full middleware pipeline on top of `RawAgent`. Choose `Agent` for new applications; reach for `RawAgent` only when you have measured latency overhead you need to shed (see [§2](#2-rawagent)).

### Constructor

```python
Agent(
    client: SupportsChatGetResponse[OptionsCoT],
    instructions: str | None = None,
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    tools: ToolTypes | Callable[..., Any]
          | Sequence[ToolTypes | Callable[..., Any]] | None = None,
    default_options: OptionsCoT | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: Sequence[MiddlewareTypes] | None = None,
    require_per_service_call_history_persistence: bool = False,
    compaction_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    additional_properties: MutableMapping[str, Any] | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `client` | `SupportsChatGetResponse` | Required. Any first-party chat client (`OpenAIChatClient`, `FoundryChatClient`, `AnthropicClient`, etc.) or custom implementation. |
| `instructions` | `str \| None` | Injected as the system message on every run. |
| `id` | `str \| None` | Auto-generated UUID if omitted. |
| `tools` | single, callable, or list | `@tool`-decorated functions, `FunctionTool` instances, or plain callables. |
| `default_options` | provider-specific TypedDict | Merged with per-call `options=`; per-call wins on conflicts. |
| `context_providers` | `list[ContextProvider]` | History, compaction, memory, skills, vector stores, etc. Run in declaration order. |
| `middleware` | `list[MiddlewareTypes]` | Agent / chat / function middleware applied on every run. |
| `require_per_service_call_history_persistence` | `bool` | When `True` and a `HistoryProvider` is present, forces per-service-call history loading/saving. |
| `compaction_strategy` | `CompactionStrategy \| None` | Shorthand for attaching a `CompactionProvider(before_strategy=...)`. |

### Key methods

```python
# Multi-turn: create a session first
session: AgentSession = agent.create_session(session_id="optional-id")

# Single-turn (fire and forget)
response: AgentResponse = await agent.run("Summarise this document.")

# Multi-turn
response = await agent.run("Continue from before.", session=session)

# Streaming
stream = agent.run("Explain step by step.", stream=True, session=session)
async for update in stream:
    print(update.text, end="", flush=True)
final: AgentResponse = await stream.get_final_response()

# Per-call tool override
response = await agent.run(
    "Check the weather.",
    tools=[get_weather],          # merged with agent-level tools for this call (additive)
    options={"temperature": 0.0}, # merged with default_options
)
```

### Typed options for IDE autocomplete

```python
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

client = OpenAIChatClient(model="gpt-4o")

# Generic parameter unlocks provider-specific keys in IDE
agent: Agent[OpenAIChatOptions] = Agent(
    client=client,
    name="reasoner",
    instructions="You are a precise reasoning assistant.",
    default_options={"temperature": 0.0, "reasoning_effort": "high"},
)

response = await agent.run("What is 17 × 23?")
print(response.text)
```

### Full example: tools + session + streaming

```python
import asyncio
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

@tool
def celsius_to_fahrenheit(
    celsius: Annotated[float, "Temperature in Celsius"],
) -> str:
    """Convert Celsius to Fahrenheit."""
    f = celsius * 9 / 5 + 32
    return f"{celsius}°C = {f:.1f}°F"

async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        name="unit-converter",
        instructions="You help convert between units. Always show your working.",
        tools=[celsius_to_fahrenheit],
    )

    session = agent.create_session()

    # Turn 1
    r1 = await agent.run("What is 100°C in Fahrenheit?", session=session)
    print(r1.text)

    # Turn 2 — model remembers the previous exchange
    r2 = await agent.run("And what about -40°C?", session=session)
    print(r2.text)

asyncio.run(main())
```

---

## 2. `RawAgent`

**Module:** `agent_framework._agents` (re-exported via `agent_framework`)

`RawAgent` is the chat-client agent without the telemetry and middleware wrapper layers. Use it when:

- You need the absolute minimum overhead in a latency-critical hot path.
- You are implementing a custom agent subclass that provides its own telemetry.
- You are writing integration tests against the raw chat loop.

For all other cases prefer `Agent`, which inherits from `RawAgent` and adds full observability and middleware support.

### Constructor

Identical signature to `Agent` — `RawAgent` defines the constructor that `Agent` inherits:

```python
RawAgent(
    client: SupportsChatGetResponse[OptionsCoT],
    instructions: str | None = None,
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    tools: ToolTypes | Callable[..., Any]
          | Sequence[ToolTypes | Callable[..., Any]] | None = None,
    default_options: OptionsCoT | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: Sequence[MiddlewareTypes] | None = None,
    require_per_service_call_history_persistence: bool = False,
    compaction_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    additional_properties: MutableMapping[str, Any] | None = None,
)
```

### Differences from `Agent`

| Feature | `Agent` | `RawAgent` |
|---|---|---|
| OpenTelemetry spans | ✓ automatic | ✗ |
| Agent-level middleware pipeline (`AgentMiddlewareLayer`) | ✓ | ✗ |
| Function / tool middleware (`FunctionInvocationLayer`) | ✓ | ✗ |
| API surface | full (inherits `RawAgent`) | same |
| `create_session()` | ✓ | ✓ |
| Streaming | ✓ | ✓ |

### Extending `RawAgent` for a custom subclass

```python
import asyncio
from agent_framework import RawAgent, AgentSession, AgentResponse
from agent_framework.openai import OpenAIChatClient

class LoggingAgent(RawAgent):
    """Minimal agent that logs every call — no telemetry wrapper overhead."""

    def run(self, messages=None, *, stream=False, session=None, **kwargs):
        # RawAgent.run is a plain def (not async) — override must match.
        print(f"[{self.name}] run called — session={getattr(session, 'session_id', None)}")
        inner = super().run(messages, stream=stream, session=session, **kwargs)
        if stream:
            return inner  # ResponseStream — return directly, no await
        # Wrap the Awaitable to add post-run logging without changing the return type.
        async def _log_result():
            result = await inner
            print(f"[{self.name}] finished — text length={len(result.text)}")
            return result
        return _log_result()

async def main():
    agent = LoggingAgent(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        name="logger",
        instructions="You are a succinct assistant.",
    )
    response = await agent.run("What is 2 + 2?")
    print(response.text)

asyncio.run(main())
```

---

## 3. `AgentSession`

**Module:** `agent_framework._sessions` (re-exported via `agent_framework`)

`AgentSession` is a lightweight state container for one conversation. It holds a `session_id`, an optional service-issued `service_session_id`, and a mutable `state` dictionary. Context providers (history, compaction, memory) read and write `state` to persist their data across turns.

> **Session ownership**: provider instances are owned by the agent, not the session. The session only holds identifiers and the shared state dict. Multiple concurrent sessions with the same agent are safe.

### Constructor

```python
AgentSession(
    *,
    session_id: str | None = None,           # auto-generated UUID if omitted
    service_session_id: str | ServiceSessionId | None = None,
)
```

### Properties and methods

| Name | Type | Notes |
|---|---|---|
| `session_id` | `str` (property, read-only) | Unique identifier. |
| `service_session_id` | `str \| ServiceSessionId \| None` | Provider-issued session token (e.g. OpenAI response ID). Trusted app state, not an auth boundary. |
| `state` | `dict[str, Any]` | Mutable; shared with all providers. Keys are namespaced by each provider. |
| `to_dict()` | `dict` | Serialise to a plain dict — provider-registered codecs encode `state` values. |
| `from_dict(data)` | `AgentSession` (classmethod) | Restore from a previous `to_dict()`. Registered types are reconstructed. |

### Preferred creation path

Prefer `agent.create_session()` over constructing `AgentSession` directly — the agent method is aware of provider initialisation:

```python
session = agent.create_session()               # auto-generates session_id
session = agent.create_session(session_id="user-42-thread-7")
```

### Serialising sessions across processes

```python
import json, asyncio
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient

async def main():
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        name="stateful",
        instructions="You are a stateful assistant.",
    )

    # Turn 1 (process A)
    session = agent.create_session(session_id="demo-session")
    await agent.run("My favourite colour is blue.", session=session)
    
    # Serialise and hand off (e.g. persist to Redis)
    blob = json.dumps(session.to_dict())

    # Turn 2 (process B — load the session back)
    session2 = AgentSession.from_dict(json.loads(blob))
    response = await agent.run("What is my favourite colour?", session=session2)
    print(response.text)   # "Your favourite colour is blue."

asyncio.run(main())
```

### Writing to `session.state` from a context provider

```python
from agent_framework import Agent, ContextProvider, AgentSession
from agent_framework._types import Message

class UserPrefsProvider(ContextProvider):
    def __init__(self, prefs: dict):
        super().__init__("user_prefs")
        self._prefs = prefs

    async def before_run(self, *, agent, session: AgentSession, context, state: dict) -> None:
        # `state` is scoped to this provider; use session.state for cross-provider sharing
        session.state["user_prefs"] = self._prefs
        # Inject a system message into the live context via extend_messages
        context.extend_messages(self.source_id, [Message("system", [f"User preferences: {self._prefs}"])])
```

---

## 4. `AgentResponse[T]`

**Module:** `agent_framework._types` (re-exported via `agent_framework`)

`AgentResponse[T]` is the return type of `await agent.run(...)`. The generic parameter `T` is the structured-output type when `options={"response_format": MyModel}` is used.

### Constructor

You rarely construct `AgentResponse` directly — it is produced by `agent.run()`. The constructor exists for testing:

```python
AgentResponse(
    *,
    messages: Message | Sequence[Message] | None = None,
    response_id: str | None = None,
    agent_id: str | None = None,
    created_at: datetime | None = None,
    finish_reason: str | FinishReason | None = None,
    usage_details: UsageDetails | None = None,
    value: T | None = None,
    response_format: StructuredResponseFormat = None,
    continuation_token: ContinuationToken | None = None,
    additional_properties: dict[str, Any] | None = None,
)
```

### Properties

| Name | Returns | Notes |
|---|---|---|
| `text` | `str` | Text of the last assistant message. Empty string when no messages. |
| `value` | `T \| None` | Lazily parses `response_format` on first access. Raises `ValidationError` on schema mismatch. |
| `user_input_requests` | `list[Content]` | All `BaseUserInputRequest` content items — non-empty when an agent issued a `request_info` (HITL) event or a tool-approval request. |
| `messages` | `list[Message]` | All messages (including intermediate tool-call messages). |
| `response_id` | `str \| None` | Provider-issued response identifier. |
| `finish_reason` | `str \| FinishReason \| None` | `"stop"`, `"length"`, `"tool_calls"`, etc. |
| `usage_details` | `UsageDetails \| None` | Token counts for billing and budget tracking. |
| `continuation_token` | `ContinuationToken \| None` | Present when a background task has not yet completed. |

### Class methods

```python
# Assemble a response from a completed streaming session
response = AgentResponse.from_updates(updates)

# With a structured output type
response = AgentResponse.from_updates(updates, output_format_type=MyModel)
```

### Example: structured output with `value`

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent, AgentResponse
from agent_framework.openai import OpenAIChatClient

class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    conditions: str

async def main():
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        instructions="Return structured weather data only.",
    )

    response: AgentResponse[WeatherReport] = await agent.run(
        "Weather in Tokyo right now?",
        options={"response_format": WeatherReport},
    )

    # response.text is the raw JSON string
    # response.value is the parsed WeatherReport instance
    report: WeatherReport = response.value
    print(f"{report.city}: {report.temperature_celsius}°C, {report.conditions}")

asyncio.run(main())
```

### Example: inspecting token usage

`UsageDetails` is a `TypedDict` — access its values via dict keys, not attribute access.

```python
response = await agent.run("Explain quantum entanglement in one sentence.")
if response.usage_details:
    print(f"Input tokens:  {response.usage_details.get('input_token_count')}")
    print(f"Output tokens: {response.usage_details.get('output_token_count')}")
    print(f"Total tokens:  {response.usage_details.get('total_token_count')}")
```

---

## 5. `FunctionTool`

**Module:** `agent_framework._tools` (re-exported via `agent_framework`)

`FunctionTool` wraps a Python callable and exposes it as a tool the LLM can invoke. It provides automatic JSON-schema generation (via Pydantic or type-annotation introspection), approval mode gating, per-instance invocation limits, and a pluggable result parser.

The `@tool` decorator is the ergonomic shorthand — it creates a `FunctionTool` from the decorated function. Use `FunctionTool` directly when you need runtime construction, dynamic names, or custom input schemas.

### Constructor

```python
FunctionTool(
    *,
    name: str,
    description: str = "",
    approval_mode: Literal["always_require", "never_require"] | None = None,
    kind: str | None = None,
    max_invocations: int | None = None,
    max_invocation_exceptions: int | None = None,
    additional_properties: dict[str, Any] | None = None,
    func: Callable[..., Any] | None = None,
    input_model: type[BaseModel] | Mapping[str, Any] | None = None,
    result_parser: Callable[[Any], str | list[Content]]
                  | _SkipParsingSentinel | None = None,
)
```

| Parameter | Notes |
|---|---|
| `name` | Shown to the model — keep it short and descriptive. |
| `description` | Shown to the model — describe *what* the tool does and *when* to use it. |
| `approval_mode` | `"always_require"` pauses execution for human approval; `"never_require"` (default) runs automatically. |
| `max_invocations` | Hard cap on total lifetime calls on this instance. The counter is never auto-reset — use `tool.invocation_count = 0` to reset manually. |
| `max_invocation_exceptions` | Hard cap on cumulative exceptions. |
| `func` | The callable to invoke. `None` creates a declaration-only tool (schema exposed but no implementation). |
| `input_model` | Pydantic `BaseModel` subclass for schema + runtime validation, or a raw JSON-schema `Mapping` passed through to the provider. |
| `result_parser` | Transform the return value before sending to the model. Pass the top-level `SKIP_PARSING` sentinel (imported from `agent_framework`) to forward the raw value. |

### Key attributes and methods

| Name | Type | Notes |
|---|---|---|
| `name` | `str` | Read/write — the tool name sent to the LLM. |
| `description` | `str` | Read/write — the tool description sent to the LLM. |
| `invocation_count` | `int` | Total successful calls. Writable for manual reset. |
| `invoke(arguments)` | `Awaitable[str \| list[Content]]` | Call the wrapped function with validated arguments. |
| `get_json_schema()` | `dict` | The JSON schema the framework sends to the model. |

### `@tool` shorthand vs direct construction

```python
from typing import Annotated
from agent_framework import tool, FunctionTool

# Shorthand — preferred for most tools
@tool(approval_mode="never_require", max_invocations=10)
def add(
    a: Annotated[int, "First operand"],
    b: Annotated[int, "Second operand"],
) -> str:
    """Add two integers."""
    return str(a + b)

# Direct construction — useful when name/description are dynamic
def _multiply(a: int, b: int) -> str:
    return str(a * b)

multiply = FunctionTool(
    name="multiply",
    description="Multiply two integers.",
    func=_multiply,
    input_model=None,          # introspect from type hints
    approval_mode="never_require",
)
```

### Custom Pydantic input model

```python
from pydantic import BaseModel, Field
from agent_framework import FunctionTool

class BookSearchArgs(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    genre: str | None = Field(default=None, description="Optional genre filter")

def search_books(query: str, max_results: int = 5, genre: str | None = None) -> str:
    # ... actual implementation
    return f"Found {max_results} books matching '{query}'"

book_search = FunctionTool(
    name="search_books",
    description="Search the library catalogue for books.",
    func=search_books,
    input_model=BookSearchArgs,
)
```

### Declaration-only tool (schema without implementation)

```python
# Useful when the implementation is on the model side (e.g. built-in tools)
shell_tool = FunctionTool(
    name="shell",
    description="Execute a shell command.",
    kind="shell",          # provider-agnostic classification
    func=None,             # no local implementation
    input_model={"type": "object", "properties": {"command": {"type": "string"}}},
)
```

### Invocation limit pattern

```python
import asyncio
from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

call_count = 0

def expensive_api(query: str) -> str:
    global call_count
    call_count += 1
    return f"Result for '{query}' (call #{call_count})"

# Cap at 3 calls per agent instance lifetime
api_tool = FunctionTool(
    name="expensive_api",
    description="Call the expensive external API.",
    func=expensive_api,
    max_invocations=3,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o-mini"),
    instructions="Use the expensive_api tool to answer questions.",
    tools=[api_tool],
)
```

---

## 6. `WorkflowAgent`

**Module:** `agent_framework._workflows._agent` (re-exported via `agent_framework`)

`WorkflowAgent` wraps a `Workflow` and exposes it through the standard `BaseAgent` interface. This lets you drop a multi-executor workflow into any context that accepts an agent — an orchestration builder, a `GroupChatBuilder`, an A2A server, or just a direct `await` call.

> **Event filtering**: only `type='output'` events and `type='request_info'` events from the workflow are surfaced as agent responses. Intermediate events are not exposed. Use `output_from=` in `WorkflowBuilder` to control which executors contribute output events.

### Constructor

```python
WorkflowAgent(
    workflow: Workflow,
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    **kwargs: Any,
)
```

### Key methods (inherited from `BaseAgent`)

| Method | Notes |
|---|---|
| `run(messages, *, stream=False, session=None, ...)` | Executes the wrapped workflow. |
| `create_session()` | Returns an `AgentSession` compatible with the workflow's checkpointing (if configured). |

### Example: wrapping a research workflow as an agent

```python
import asyncio
from agent_framework import Agent, WorkflowAgent, tool, executor
from agent_framework._workflows import WorkflowBuilder
from agent_framework._workflows._workflow_context import WorkflowContext
from agent_framework._types import Message
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(model="gpt-4o-mini")

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"[stub] Top results for '{query}': ..."

researcher = Agent(
    client=client,
    name="researcher",
    instructions="You are a research specialist. Use web_search to find facts.",
    tools=[web_search],
)

writer = Agent(
    client=client,
    name="writer",
    instructions="You are a technical writer. Summarise the research you receive.",
)

@executor
async def research_executor(messages: list[Message], ctx: WorkflowContext[str]) -> None:
    # WorkflowAgent validates that the start executor accepts list[Message]
    response = await researcher.run(messages)
    await ctx.send_message(response.text)

@executor
async def write_executor(message: str, ctx: WorkflowContext[None, str]) -> None:
    response = await writer.run(f"Summarise this research: {message}")
    await ctx.yield_output(response.text)

workflow = (
    WorkflowBuilder(
        name="research-pipeline",
        start_executor=research_executor,  # required kwarg; marks the entry point
        output_from=[write_executor],      # list of executors whose output becomes workflow output
    )
    .add_edge(research_executor, write_executor)  # connect executors by object reference
    .build()
)

# Wrap the workflow as an agent
research_agent = WorkflowAgent(
    workflow,
    name="research-agent",
    description="A two-stage research-and-summarise pipeline exposed as a single agent.",
)

async def main():
    response = await research_agent.run("What is the current state of quantum computing?")
    print(response.text)

asyncio.run(main())
```

### Embedding `WorkflowAgent` in a `GroupChatBuilder`

`GroupChatBuilder` takes all participants in its constructor — there is no `.add_agent()` fluent method. Import from `agent_framework.orchestrations` (not the private `_orchestration` module).

```python
from agent_framework.orchestrations import GroupChatBuilder

specialist = WorkflowAgent(
    deep_analysis_workflow,
    name="specialist",
    description="Runs a deep multi-step analysis workflow.",
)

group_chat = GroupChatBuilder(
    participants=[coordinator, specialist],  # workflow drops in as a peer
    orchestrator_agent=coordinator,          # decides who speaks next
).build()

result = await group_chat.run("Analyse the quarterly earnings data.")
```

---

## 7. `WorkflowContext[OutT, W_OutT]`

**Module:** `agent_framework._workflows._workflow_context` (re-exported via `agent_framework`)

`WorkflowContext` is the single interface an executor function receives to interact with the workflow runtime. It controls message routing, workflow output, shared state, and human-in-the-loop pausing. Executors declare the types they produce in the generic parameters; the framework validates compatibility at build time.

### Generic parameters

| Signature | Meaning |
|---|---|
| `WorkflowContext` | Executor only produces side effects — no `send_message` or `yield_output`. |
| `WorkflowContext[OutT]` | Executor sends `OutT` messages to downstream executors. |
| `WorkflowContext[OutT, W_OutT]` | Executor sends `OutT` messages **and** yields `W_OutT` as workflow output. |
| `WorkflowContext[int \| str, bool]` | Union types for multi-type routing. |

### Key methods

```python
# Route a message to downstream executors (or to a specific target)
await ctx.send_message(message: OutT, target_id: str | None = None) -> None

# Emit a workflow-level output event (type='output' or 'intermediate')
await ctx.yield_output(output: W_OutT) -> None

# Add a raw WorkflowEvent to the event stream
await ctx.add_event(event: WorkflowEvent[Any]) -> None

# Pause the workflow and wait for an external response (human-in-the-loop)
await ctx.request_info(
    request_data: object,
    response_type: type,
    *,
    request_id: str | None = None,
) -> None

# Shared state accessors
ctx.get_state(key: str, default: Any = None) -> Any
ctx.set_state(key: str, value: Any) -> None

# Introspect message sources (useful in fan-in executors)
ctx.get_source_executor_id() -> str          # raises RuntimeError if multiple sources
ctx.source_executor_ids -> list[str]         # always safe

# Inspect what was sent/yielded in this invocation
ctx.get_sent_messages() -> list[Any]
```

### `request_id` (HITL response context)

```python
ctx.request_id -> str | None   # non-None only inside a @response_handler
```

### Example: fan-in aggregator with state

```python
import asyncio
from agent_framework import executor
from agent_framework._workflows import WorkflowBuilder
from agent_framework._workflows._workflow_context import WorkflowContext

@executor
async def fetch_prices(symbol: str, ctx: WorkflowContext[dict]) -> None:
    # Simulate fetching price data
    await ctx.send_message({"symbol": symbol, "price": 42.0})

@executor
async def fetch_news(symbol: str, ctx: WorkflowContext[dict]) -> None:
    await ctx.send_message({"symbol": symbol, "headline": "Earnings beat expectations"})

@executor
async def analyse(data: list[dict], ctx: WorkflowContext[None, str]) -> None:
    # Fan-in receives all upstream messages as a list
    prices = [d for d in data if "price" in d]
    news = [d for d in data if "headline" in d]

    # Write to shared workflow state
    ctx.set_state("analysis_done", True)

    summary = f"Price: ${prices[0]['price']} | News: {news[0]['headline']}"
    await ctx.yield_output(summary)

# WorkflowBuilder requires a single start_executor.
# Use a dispatcher to fan out to both fetchers, then fan-in to analyse.
@executor
async def dispatch(symbol: str, ctx: WorkflowContext[str]) -> None:
    await ctx.send_message(symbol, target_id=fetch_prices.id)
    await ctx.send_message(symbol, target_id=fetch_news.id)

workflow = (
    WorkflowBuilder(
        name="market-analysis",
        start_executor=dispatch,
        output_from=[analyse],
    )
    .add_fan_out_edges(dispatch, [fetch_prices, fetch_news])
    .add_fan_in_edges([fetch_prices, fetch_news], analyse)
    .build()
)

async def main():
    # stream=True returns ResponseStream directly (not an awaitable)
    stream = workflow.run("AAPL", stream=True)
    async for event in stream:
        if event.type == "output":
            print(event.data)   # "Price: $42.0 | News: Earnings beat expectations"

asyncio.run(main())
```

### Example: human-in-the-loop pause with `request_info`

```python
from dataclasses import dataclass
from agent_framework import Executor, handler, response_handler
from agent_framework._workflows._workflow_context import WorkflowContext

@dataclass
class ApprovalRequest:
    action: str
    details: str

@dataclass
class ApprovalResponse:
    approved: bool
    reason: str

class EmailApprovalExecutor(Executor):
    """Executor subclass that uses @handler + @response_handler for HITL approval."""

    @handler
    async def run(
        self, message: str, ctx: WorkflowContext[None, str]
    ) -> None:
        await ctx.request_info(
            ApprovalRequest(action="send_email", details=message),
            response_type=ApprovalResponse,
        )

    @response_handler(request=ApprovalRequest, response=ApprovalResponse, workflow_output=str)
    async def handle_approval(
        self,
        original_request: ApprovalRequest,
        response: ApprovalResponse,
        context: WorkflowContext[None, str],
    ) -> None:
        if response.approved:
            await context.yield_output(f"Email sent: {original_request.details}")
        else:
            await context.yield_output(f"Rejected: {response.reason}")

approval_executor = EmailApprovalExecutor()
```

---

## 8. `CompactionProvider`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`CompactionProvider` is a `ContextProvider` that attaches one or two compaction strategies to an agent's context pipeline. The `before_strategy` runs before each model call (trims loaded history before it reaches the model); the `after_strategy` runs once per turn after the model responds (compacts persisted history for the next turn). Either may be `None`.

> `after_run_once_per_turn = True` is set on the class — the `after_strategy` is only ever run once per user turn, never mid-agentic-loop.

### Constructor

```python
CompactionProvider(
    *,
    before_strategy: CompactionStrategy | None = None,
    after_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    source_id: str = "compaction",
    history_source_id: str = "in_memory",
)
```

| Parameter | Notes |
|---|---|
| `before_strategy` | Applied to context messages loaded by a history provider before the model runs. |
| `after_strategy` | Applied to stored messages after the model runs. Requires `history_source_id` to locate the target. |
| `tokenizer` | Override for token-aware strategies (e.g. `TruncationStrategy`). Falls back to `CharacterEstimatorTokenizer`. |
| `history_source_id` | Must match the `source_id` of the `HistoryProvider` whose messages `after_strategy` compacts. Default `"in_memory"`. |

### Available strategies (import from `agent_framework`)

| Class | Key params | What it does |
|---|---|---|
| `SlidingWindowStrategy` | `keep_last_groups: int` | Keeps the last N tool-call groups. |
| `TruncationStrategy` | `max_n: int, compact_to: int` | Drops oldest messages until `compact_to` messages remain, once `max_n` is exceeded. |
| `SummarizationStrategy` | `client: SupportsChatGetResponse`, `target_count: int = 4`, `threshold: int = 2` | Summarises old messages with a chat client once history exceeds `threshold` groups; retains `target_count` groups. |
| `ToolResultCompactionStrategy` | `keep_last_tool_call_groups: int` | Replaces old tool-call/result pairs with a brief summary message. |
| `SelectiveToolCallCompactionStrategy` | `keep_last_tool_call_groups: int = 1` | Evicts old tool-call/result pairs, keeping only the last N groups. |
| `ContextWindowCompactionStrategy` | `max_context_window_tokens: int`, `max_output_tokens: int` | Evicts old tool results and truncates history to fit within the token budget; does not use an LLM. |
| `TokenBudgetComposedStrategy` | `token_budget: int, tokenizer: TokenizerProtocol, strategies: list[...]` | Runs strategies in sequence, stopping once under the budget. `tokenizer` is required. |

### Example: sliding window before + tool-result compaction after

```python
import asyncio
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import SlidingWindowStrategy, ToolResultCompactionStrategy
from agent_framework.openai import OpenAIChatClient

async def main():
    # skip_excluded=True is required when using after_strategy: CompactionProvider marks
    # compacted messages as excluded, and without this flag they reload on the next turn.
    history = InMemoryHistoryProvider(skip_excluded=True)

    compaction = CompactionProvider(
        before_strategy=SlidingWindowStrategy(keep_last_groups=15),
        after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
        history_source_id=history.source_id,  # matches InMemoryHistoryProvider default
    )

    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="long-running-assistant",
        instructions="You help with extended research tasks.",
        context_providers=[history, compaction],
    )

    session = agent.create_session()
    for prompt in ["Step 1: ...", "Step 2: ...", "Step 3: ..."]:
        response = await agent.run(prompt, session=session)
        print(response.text[:100])

asyncio.run(main())
```

### Example: token-budget composed strategy

```python
from agent_framework import CompactionProvider
from agent_framework._compaction import (
    SlidingWindowStrategy,
    TruncationStrategy,
    TokenBudgetComposedStrategy,
)

from agent_framework._compaction import CharacterEstimatorTokenizer

tokenizer = CharacterEstimatorTokenizer()

compaction = CompactionProvider(
    before_strategy=TokenBudgetComposedStrategy(
        strategies=[
            SlidingWindowStrategy(keep_last_groups=30),          # first pass: drop old groups
            TruncationStrategy(max_n=40, compact_to=20),          # second pass: hard cap (message counts, not tokens)
        ],
        token_budget=4000,
        tokenizer=tokenizer,  # required by TokenBudgetComposedStrategy
    ),
)
```

### Shorthand: `compaction_strategy=` on `Agent`

For simple cases you can bypass `CompactionProvider` entirely:

```python
from agent_framework import Agent
from agent_framework._compaction import SlidingWindowStrategy

agent = Agent(
    client=client,
    compaction_strategy=SlidingWindowStrategy(keep_last_groups=20),
)
```

This is equivalent to `CompactionProvider(before_strategy=SlidingWindowStrategy(keep_last_groups=20))` with default parameters.

---

## 9. `VectorCollectionContextProvider`

**Module:** `agent_framework._vectors` (re-exported via `agent_framework`)

> **Experimental** — import from `agent_framework`; emits `ExperimentalWarning`.

`VectorCollectionContextProvider` exposes a caller-owned vector collection as a set of CRUD and semantic-search tools for the agent. It generates up to four tools (`upsert`, `get`, `delete`, `search`) from a single collection, with an optional approval gate on destructive operations.

**Use this when** your application owns the collection and wants the agent to interact with it. **Use `VectorStoreHistoryProvider`** (covered in [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/)) to let the agent persist its own conversation history in a vector store.

### Constructor

```python
VectorCollectionContextProvider(
    collection: BaseVectorCollection[KeyT, ModelT],
    source_id: str = "vector_collection",
    *,
    scope_filter: FilterExpression | None,
    instructions: str | Sequence[str] | None = None,
    include_upsert_tool: bool = True,
    include_get_tool: bool = True,
    include_delete_tool: bool = True,
    include_search_tool: bool = True,
    approval_mode: (
        Literal["always_require", "never_require"]
        | Mapping[
            Literal["get", "delete", "upsert", "search"],
            Literal["always_require", "never_require"],
        ]
    ) = ...,                # defaults: get/search = "never_require", delete/upsert = "always_require"
    additional_search_tools: Sequence[FunctionTool] | None = None,
    max_tool_batch_size: int = ...,  # default batch size for upsert
)
```

| Parameter | Notes |
|---|---|
| `collection` | Any `BaseVectorCollection` subclass (e.g. `InMemoryCollection`, Azure AI Search). |
| `scope_filter` | Logical isolation filter applied to all generated tools. Not an auth boundary. Pass `None` when no scoping is needed. |
| `instructions` | Custom tool-usage instructions injected before each run. Defaults to auto-generated text. |
| `include_*_tool` | Toggle individual generated tools off when the agent should not have that capability. |
| `approval_mode` | Either a global string or a per-operation dict. Destructive operations (`upsert`, `delete`) default to `"always_require"`. |
| `additional_search_tools` | Extra `FunctionTool` instances added alongside the auto-generated search tool. |

### Example: product catalogue RAG

```python
import asyncio
from typing import Annotated, Optional
from pydantic import BaseModel
from agent_framework import Agent, VectorCollectionContextProvider, vectorstoremodel, VectorStoreField, InMemoryStore
from agent_framework.openai import OpenAIChatClient

# @vectorstoremodel requires Annotated field metadata to describe the vector schema.
# A separate *_vec field holds the pre-computed float embeddings.
@vectorstoremodel(collection_name="products")
class Product(BaseModel):
    id: Annotated[str, VectorStoreField("key")]
    name: Annotated[str, VectorStoreField("data")]
    description: Annotated[str, VectorStoreField("data")]
    description_vec: Annotated[Optional[list[float]], VectorStoreField("vector", dimensions=1536)] = None
    price: Annotated[float, VectorStoreField("data")]

async def main():
    # An embedding generator is required so VectorCollectionContextProvider can
    # embed natural-language queries before comparing them with stored vectors.
    from agent_framework.openai import OpenAIEmbeddingClient
    embedding_client = OpenAIEmbeddingClient(model="text-embedding-3-small")
    store = InMemoryStore(embedding_generator=embedding_client)
    collection = store.get_collection(Product)
    await collection.ensure_collection_exists()

    # Seed with pre-computed vectors (generate_vectors=False skips re-embedding on upsert)
    await collection.upsert([
        Product(id="p1", name="Widget A", description="A sturdy blue widget",
                description_vec=[0.1] * 1536, price=9.99),
        Product(id="p2", name="Gadget B", description="A portable red gadget",
                description_vec=[0.2] * 1536, price=24.99),
    ], generate_vectors=False)

    vector_provider = VectorCollectionContextProvider(
        collection=collection,
        scope_filter=None,
        include_upsert_tool=False,   # read-only for the agent
        include_delete_tool=False,
        approval_mode="never_require",
    )

    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="product-assistant",
        instructions="You help customers find products. Use the search tool to answer queries.",
        context_providers=[vector_provider],
    )

    response = await agent.run("Do you have any portable items under $30?")
    print(response.text)

asyncio.run(main())
```

---

## 10. `TodoItem` · `TodoStore` · `TodoFileStore`

**Module:** `agent_framework._harness._todo` (re-exported via `agent_framework`)

> **Experimental** — all three classes emit `ExperimentalWarning` on import.

The todo trio provides session-scoped task tracking for agents. `TodoItem` is the immutable data record. `TodoStore` is the abstract persistence backend. `TodoFileStore` is the file-backed implementation (for use with `TodoProvider`, covered in [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/)).

### `TodoItem`

```python
TodoItem(
    id: int,
    title: str,
    description: str | None = None,
    is_complete: bool = False,
)
```

| Attribute | Type | Notes |
|---|---|---|
| `id` | `int` | Monotonically increasing integer within a session. |
| `title` | `str` | Short non-empty description. Required. |
| `description` | `str \| None` | Optional longer explanation. |
| `is_complete` | `bool` | `True` once the item is done. |
| `to_dict(exclude_none=True)` | `dict` | Serialise for persistence. |
| `from_dict(raw)` | `TodoItem` (classmethod) | Restore; validates types and raises `ValueError` on bad input. |

### `TodoStore` (abstract)

```python
class TodoStore(ABC):
    async def load_state(self, session, *, source_id) -> tuple[list[TodoItem], int]:
        """Return (items, next_id)."""

    async def save_state(self, session, items, *, next_id, source_id) -> None:
        """Persist items and the next-ID counter."""

    async def load_items(self, session, *, source_id) -> list[TodoItem]:
        """Convenience: load items only (next_id discarded)."""
```

Two built-in implementations:

| Class | Persistence | Notes |
|---|---|---|
| `TodoSessionStore` | `AgentSession.state` in-process | Default backend used by `TodoProvider`. Lost when the process ends. |
| `TodoFileStore` | Per-session JSON file | Survives restarts. Requires `base_path`. |

### `TodoFileStore`

```python
TodoFileStore(
    base_path: str | Path,
    *,
    kind: str = "todos",
    owner_prefix: str = "",
    owner_state_key: str | None = None,
    state_filename: str = "todos.json",
)
```

| Parameter | Notes |
|---|---|
| `base_path` | Root directory for all session todo files. |
| `kind` | Subdirectory bucket name (default `"todos"`). |
| `owner_state_key` | If set, reads `session.state[owner_state_key]` as the logical owner partition (e.g. user ID for multi-tenant stores). |
| `state_filename` | The JSON file name within each session directory. |

### Example: `TodoProvider` with `TodoFileStore` for durable tasks

```python
import asyncio
from pathlib import Path
from agent_framework import Agent, TodoProvider
from agent_framework._harness._todo import TodoFileStore
from agent_framework.openai import OpenAIChatClient

async def main():
    todo_store = TodoFileStore(base_path=Path("/tmp/agent_todos"))

    todo_provider = TodoProvider(store=todo_store)

    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        name="planner",
        instructions=(
            "You are a task planner. Use todo tools to plan and track work. "
            "Always start by creating a todo list before doing any task."
        ),
        context_providers=[todo_provider],
    )

    session = agent.create_session(session_id="project-42")

    r1 = await agent.run(
        "Plan a 3-step process to onboard a new developer.",
        session=session,
    )
    print(r1.text)

    # Resume in a new process (same session_id, same file path)
    session2 = agent.create_session(session_id="project-42")
    r2 = await agent.run("Mark step 1 as complete.", session=session2)
    print(r2.text)

asyncio.run(main())
```

### Direct `TodoStore` access for inspection or migration

```python
import asyncio
from pathlib import Path
from agent_framework import AgentSession
from agent_framework._harness._todo import TodoFileStore, TodoItem

async def list_todos(session_id: str) -> None:
    store = TodoFileStore(base_path=Path("/tmp/agent_todos"))
    session = AgentSession(session_id=session_id)
    items = await store.load_items(session, source_id="todo")  # matches TodoProvider default
    for item in items:
        status = "✓" if item.is_complete else "○"
        print(f"  {status} [{item.id}] {item.title}")
        if item.description:
            print(f"        {item.description}")

asyncio.run(list_todos("project-42"))
```

---

## Summary table

| Class | Module | Stable? | When to use |
|---|---|---|---|
| `Agent` | `agent_framework._agents` | ✓ stable | Primary agent class for all new applications. |
| `RawAgent` | `agent_framework._agents` | ✓ stable | Custom subclasses; latency-critical paths; testing. |
| `AgentSession` | `agent_framework._sessions` | ✓ stable | Multi-turn state; serialise across processes. |
| `AgentResponse[T]` | `agent_framework._types` | ✓ stable | Response from `agent.run()`; structured output via `.value`. |
| `FunctionTool` | `agent_framework._tools` | ✓ stable | Dynamic tool construction, approval gates, invocation limits. |
| `WorkflowAgent` | `agent_framework._workflows._agent` | ✓ stable | Expose a workflow as an `Agent` peer. |
| `WorkflowContext[OutT, W_OutT]` | `agent_framework._workflows._workflow_context` | ✓ stable | Route messages, yield outputs, state, HITL inside executors. |
| `CompactionProvider` | `agent_framework._compaction` | ✓ stable | Attach before/after compaction strategies to an agent. |
| `VectorCollectionContextProvider` | `agent_framework._vectors` | ⚗ experimental | Expose a caller-owned collection as agent CRUD/search tools. |
| `TodoItem` / `TodoStore` / `TodoFileStore` | `agent_framework._harness._todo` | ⚗ experimental | Session-scoped task tracking with durable file persistence. |

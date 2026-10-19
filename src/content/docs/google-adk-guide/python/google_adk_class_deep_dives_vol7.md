---
title: "Class Deep Dives Vol. 7 — v2.12.0"
description: "Source-verified deep dives for 10 entries in google-adk 2.12.0: LlmRequest, LlmResponse, BaseLlm + LLMRegistry, InvocationContext, State, AgentTool, TransferToAgentTool, BaseCodeExecutor + UnsafeLocalCodeExecutor, PluginManager, and CascadeLive."
framework: google-adk
language: python
sidebar:
  order: 137
---

All examples and field tables on this page are source-verified against **google-adk==2.12.0**. The ten entries cover the **model request/response pipeline** (`LlmRequest`, `LlmResponse`), **custom model backends** (`BaseLlm`, `LLMRegistry`), the **invocation lifecycle** (`InvocationContext`, `State`), **agent-as-tool patterns** (`AgentTool`, `TransferToAgentTool`), **code execution** (`BaseCodeExecutor`, `UnsafeLocalCodeExecutor`), **plugin lifecycle management** (`PluginManager`), and the **cascade live** speech-to-speech layer (`CascadeLive`).

| # | Class / Symbol | Module | Subject |
|---|---|---|---|
| 1 | `LlmRequest` | `google.adk.models.llm_request` | Request envelope sent to every model backend |
| 2 | `LlmResponse` | `google.adk.models.llm_response` | Response envelope from every model backend |
| 3 | `BaseLlm` + `LLMRegistry` | `google.adk.models.base_llm` / `.registry` | Custom model backends and the model registry |
| 4 | `InvocationContext` | `google.adk.agents.invocation_context` | Full per-invocation lifecycle context |
| 5 | `State` | `google.adk.sessions.state` | Session state dict with scopes and schema validation |
| 6 | `AgentTool` | `google.adk.tools.agent_tool` | Wrapping an agent as a callable tool |
| 7 | `TransferToAgentTool` | `google.adk.tools.transfer_to_agent_tool` | Type-safe agent handoffs with enum constraints |
| 8 | `BaseCodeExecutor` + `UnsafeLocalCodeExecutor` | `google.adk.code_executors` | Code execution base and local subprocess executor |
| 9 | `PluginManager` | `google.adk.plugins.plugin_manager` | Plugin registration and early-exit callback dispatch |
| 10 | `CascadeLive` | `google.adk.live._cascade_live` | STT → LLM → TTS cascade for speech-to-speech agents |

---

## 1 — `LlmRequest`

**Module:** `google.adk.models.llm_request`

`LlmRequest` is the envelope that every part of the ADK assembles before calling a model backend. It carries the conversation history, the active tool declarations, configuration, context-cache metadata, and structured output schema. Framework code builds it; you read and mutate it inside `before_model_callback`.

### Field reference

Source-verified from `google/adk/models/llm_request.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | `str \| None` | `None` | Override the agent's default model for this single call |
| `contents` | `list[types.Content]` | `[]` | Conversation history sent to the model |
| `config` | `types.GenerateContentConfig` | `GenerateContentConfig()` | Temperature, top-p, tools, response schema, etc. |
| `live_connect_config` | `types.LiveConnectConfig` | `LiveConnectConfig()` | Live-session specific config (voice, realtime audio) |
| `tools_dict` | `dict[str, BaseTool]` | `{}` | Name-keyed lookup of all active tools (excluded from serialisation) |
| `cache_config` | `ContextCacheConfig \| None` | `None` | Context cache settings for this request |
| `cache_metadata` | `CacheMetadata \| None` | `None` | Cache reuse data from previous request |
| `cacheable_contents_token_count` | `int \| None` | `None` | Token count for cache size validation |
| `previous_interaction_id` | `str \| None` | `None` | Chains stateful interactions via the Interactions API |

### Key methods

| Method | What it does |
|---|---|
| `append_instructions(instructions)` | Appends strings to the system instruction |
| `append_tools(tools, *, wrap_native_tools)` | Injects `BaseTool` instances into `tools_dict` and `config.tools` |
| `set_output_schema(schema)` | Sets `config.response_schema` and adjusts `response_mime_type` |

### Inspecting the request in a callback

```python
from google.adk.agents import LlmAgent
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents.callback_context import CallbackContext

def log_request(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    print(f"Model: {llm_request.model}")
    print(f"History turns: {len(llm_request.contents)}")
    print(f"Active tools: {list(llm_request.tools_dict.keys())}")
    # Return None to let the real model call proceed
    return None

agent = LlmAgent(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction="You are a helpful assistant.",
    before_model_callback=log_request,
)
```

### Injecting a system instruction per request

```python
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents.callback_context import CallbackContext

def inject_date(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    from datetime import date
    llm_request.append_instructions(
        [f"Today's date is {date.today().isoformat()}."]
    )
    return None  # proceed with the modified request
```

### Short-circuiting the model call

Return a `LlmResponse` from `before_model_callback` to skip the actual model call entirely — useful for caching, cost control, or testing:

```python
from google.genai import types
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents.callback_context import CallbackContext

_CACHE: dict[str, str] = {}

def cached_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    # Build a simple cache key from the last user turn
    last = llm_request.contents[-1] if llm_request.contents else None
    key = last.parts[0].text if last and last.parts else ""
    if key in _CACHE:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=_CACHE[key])],
            )
        )
    return None  # cache miss — let the real call proceed
```

---

## 2 — `LlmResponse`

**Module:** `google.adk.models.llm_response`

`LlmResponse` is the typed envelope returned by every `BaseLlm.generate_content_async` call. Streaming mode yields multiple fragments (with `partial=True`) followed by a final complete response.

### Field reference

Source-verified from `google/adk/models/llm_response.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model_version` | `str \| None` | `None` | Model version string returned by the backend |
| `content` | `types.Content \| None` | `None` | The generated content (role + parts) |
| `grounding_metadata` | `types.GroundingMetadata \| None` | `None` | Search grounding provenance |
| `partial` | `bool \| None` | `None` | `True` on streaming fragments; `None`/`False` on the complete response |
| `turn_complete` | `bool \| None` | `None` | `True` when the model signals end-of-turn (streaming only) |
| `turn_complete_reason` | `types.TurnCompleteReason \| None` | `None` | Why the turn ended (e.g. `STOP`, `MAX_TOKENS`) |
| `interaction_status` | `types.InteractionStatus \| None` | `None` | `IN_PROGRESS` / `IDLE` for multi-turn live sessions |
| `error_code` | `int \| None` | `None` | Backend error code |
| `error_message` | `str \| None` | `None` | Human-readable error description |
| `interrupted` | `bool \| None` | `None` | `True` when generation was interrupted (e.g. barge-in) |
| `input_transcription` | `str \| None` | `None` | STT transcript of user audio input |
| `output_transcription` | `str \| None` | `None` | TTS transcript of model audio output |
| `avg_logprobs` | `float \| None` | `None` | Average log-probability of generated tokens |

### Reading a response in `after_model_callback`

```python
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents.callback_context import CallbackContext

def audit_response(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    if llm_response.error_message:
        print(f"Model error: {llm_response.error_message}")
        return None

    if llm_response.content:
        for part in llm_response.content.parts:
            if part.text:
                print(f"Model said: {part.text[:80]}...")
            elif part.function_call:
                print(f"Tool call: {part.function_call.name}({part.function_call.args})")

    return None  # return the response unmodified
```

### Rewriting a response in `after_model_callback`

```python
from google.genai import types
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents.callback_context import CallbackContext

def add_disclaimer(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    if not llm_response.content:
        return None
    for part in llm_response.content.parts:
        if part.text:
            part.text += "\n\n*This is AI-generated content.*"
    return llm_response  # return modified response
```

### Streaming: detecting fragments vs. complete responses

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

async def stream_example():
    runner = InMemoryRunner(
        agent=LlmAgent(
            model="gemini-2.0-flash",
            name="streamer",
            instruction="Explain async generators in Python.",
        ),
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message={"role": "user", "parts": [{"text": "Go"}]},
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text, end="", flush=True)
    print()

asyncio.run(stream_example())
```

---

## 3 — `BaseLlm` + `LLMRegistry`

**Modules:** `google.adk.models.base_llm`, `google.adk.models.registry`

`BaseLlm` is the abstract base every model backend implements. `LLMRegistry` maps model-name patterns to their `BaseLlm` subclasses and resolves `LlmAgent(model="...")` at runtime.

### `BaseLlm` field and method reference

Source-verified from `google/adk/models/base_llm.py`:

| Symbol | Kind | Purpose |
|---|---|---|
| `model` | field `str` | The model name string passed at construction |
| `capabilities` | property → `LlmCapabilities` | Declares what the model supports (override in subclasses) |
| `generate_content_async(llm_request, stream)` | abstract async generator | Yield `LlmResponse` fragments (or one complete response) |
| `connect(llm_request)` | async context manager | Open a live/bidirectional connection; raises `NotImplementedError` if unsupported |
| `supported_models()` | classmethod → `list[str]` | Regex patterns matched against model names in the registry |

### `LlmCapabilities` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `output_schema_and_tools` | `bool` | name-based | Model can use a structured `output_schema` alongside tool declarations |

### Writing a custom `BaseLlm`

```python
from __future__ import annotations
import asyncio
from typing import AsyncGenerator
from google.genai import types
from google.adk.models import BaseLlm, LlmRequest, LlmResponse, LlmCapabilities

class EchoLlm(BaseLlm):
    """Toy model that echoes the last user message — useful for tests."""

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"^echo$"]  # matches only the name "echo"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        last_user = next(
            (c for c in reversed(llm_request.contents) if c.role == "user"),
            None,
        )
        text = last_user.parts[0].text if last_user else "(no input)"
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Echo: {text}")],
            )
        )
```

### Registering and using a custom model

```python
from google.adk.models import LLMRegistry
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
import asyncio

# Register the custom backend
LLMRegistry.register(EchoLlm)

async def run_echo():
    runner = InMemoryRunner(
        agent=LlmAgent(
            model="echo",  # matched by EchoLlm.supported_models()
            name="echo_agent",
            instruction="You are an echo.",
        ),
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    from google.genai import types as gt
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=gt.Content(
            role="user", parts=[gt.Part(text="hello world")]
        ),
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)

asyncio.run(run_echo())
# Output: Echo: hello world
```

### Using the provider-prefix syntax

`LLMRegistry` supports a `provider:model` shorthand. The prefix is matched case-insensitively against the registered class name (minus any trailing `Llm`):

```python
# "litellm:gpt-4o" resolves to LiteLlm(model="gpt-4o")
# "openai:gpt-4o"  also resolves to LiteLlm (if litellm is installed and knows "openai")
agent = LlmAgent(model="litellm:claude-3-5-haiku-20241022", ...)
```

---

## 4 — `InvocationContext`

**Module:** `google.adk.agents.invocation_context`

`InvocationContext` is created once per `runner.run_async()` call and holds everything the framework needs for that invocation: the active session, services, the current agent, abort signals, and per-agent state. Plugins and callbacks receive it as a read-mostly context object.

```
┌─────────────────────── invocation ──────────────────────────┐
┌──────────── llm_agent_call_1 ────────────┐ ┌─ agent_call_2 ─┐
┌──── step_1 ────────┐ ┌───── step_2 ──────┐
[call_llm] [call_tool] [call_llm] [transfer]
```

### Key field reference

Source-verified from `google/adk/agents/invocation_context.py`:

| Field | Type | Purpose |
|---|---|---|
| `invocation_id` | `str` | Unique ID for this invocation (set by the runner) |
| `session` | `Session` | The current session, including state and event history |
| `agent` | `BaseAgent \| BaseNode \| None` | The agent currently executing |
| `user_content` | `types.Content \| None` | The user message that started this invocation |
| `branch` | `str \| None` | Dot-separated path for branching sub-agents (e.g. `"parent.child"`) |
| `end_invocation` | `bool` | Set to `True` to stop the invocation after the current step |
| `artifact_service` | `BaseArtifactService \| None` | For reading/writing artifacts |
| `memory_service` | `BaseMemoryService \| None` | For long-term memory lookups |
| `credential_service` | `BaseCredentialService \| None` | For OAuth token management |
| `run_config` | `RunConfig \| None` | `max_llm_calls`, `streaming_mode`, etc. |

### Reading context in a `before_agent_callback`

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

def log_invocation(callback_context: CallbackContext) -> types.Content | None:
    ctx = callback_context._invocation_context
    print(f"Invocation ID : {ctx.invocation_id}")
    print(f"Session ID    : {ctx.session.id}")
    print(f"Agent         : {ctx.agent.name if ctx.agent else 'none'}")
    print(f"User content  : {ctx.user_content}")
    return None  # continue normally

agent = LlmAgent(
    model="gemini-2.0-flash",
    name="debug_agent",
    instruction="Answer the user.",
    before_agent_callback=log_invocation,
)
```

### Stopping an invocation early

```python
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

def safety_gate(callback_context: CallbackContext) -> types.Content | None:
    ctx = callback_context._invocation_context
    user_text = ""
    if ctx.user_content and ctx.user_content.parts:
        user_text = ctx.user_content.parts[0].text or ""
    if "forbidden" in user_text.lower():
        ctx.end_invocation = True  # halt the entire invocation
        return types.Content(
            role="model",
            parts=[types.Part(text="I cannot help with that request.")],
        )
    return None
```

### `LlmCallsLimitExceededError`

`InvocationContext` tracks every call to the model via `_InvocationCostManager`. When `RunConfig.max_llm_calls > 0` and the limit is exceeded, it raises `LlmCallsLimitExceededError`:

```python
from google.adk.agents import LlmAgent, RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
import asyncio

async def run_with_limit():
    runner = InMemoryRunner(
        agent=LlmAgent(
            model="gemini-2.0-flash",
            name="limited_agent",
            instruction="Loop and call tools often.",
        ),
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    from google.genai import types as gt
    try:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=gt.Content(role="user", parts=[gt.Part(text="Go")]),
            run_config=RunConfig(max_llm_calls=3),
        ):
            pass
    except Exception as e:
        print(type(e).__name__, e)
```

---

## 5 — `State`

**Module:** `google.adk.sessions.state`

`State` is the dict-like object that every agent callback and tool sees as `tool_context.state` or `callback_context.state`. It tracks a pending-commit *delta* alongside the committed *value*, so mutations are atomic: they become visible to the current turn immediately but are persisted only when the event is committed.

### Scope prefixes

| Prefix | Scope | Lifetime |
|---|---|---|
| `app:` | Application-wide | Persists across all users and sessions |
| `user:` | Per-user | Persists across all sessions for the same user |
| `temp:` | Current turn only | Discarded after the turn; not stored |
| *(none)* | Per-session | Default scope; committed with the session |

### Key operations

Source-verified from `google/adk/sessions/state.py`:

| Operation | Behaviour |
|---|---|
| `state["key"] = value` | Write to delta (and current value); validated against schema if set |
| `value = state["key"]` | Delta takes priority over committed value |
| `"key" in state` | Checks both delta and committed value |
| `state.setdefault("key", default)` | Set only when key is absent |
| `state.to_dict()` | Merge of committed value and delta (delta wins) |

### Basic read/write in a tool

```python
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

def increment_counter(tool_context: ToolContext) -> dict:
    """Increment a per-session counter and return its new value."""
    count = tool_context.state.get("counter", 0)
    tool_context.state["counter"] = count + 1
    return {"counter": tool_context.state["counter"]}

agent = LlmAgent(
    model="gemini-2.0-flash",
    name="counter_agent",
    instruction="Call increment_counter on every turn.",
    tools=[increment_counter],
)
```

### Using all three scopes

```python
from google.adk.tools import ToolContext

def multi_scope_demo(tool_context: ToolContext) -> dict:
    """Write to all three permanent scopes."""
    # Per-session (no prefix)
    tool_context.state["session_visits"] = (
        tool_context.state.get("session_visits", 0) + 1
    )
    # Per-user (persists across sessions for this user)
    tool_context.state["user:total_visits"] = (
        tool_context.state.get("user:total_visits", 0) + 1
    )
    # Application-wide (shared by all users)
    tool_context.state["app:global_count"] = (
        tool_context.state.get("app:global_count", 0) + 1
    )
    # Temp — discarded after this turn
    tool_context.state["temp:scratch"] = "will not be saved"
    return {
        "session": tool_context.state["session_visits"],
        "user_total": tool_context.state["user:total_visits"],
        "global": tool_context.state["app:global_count"],
    }
```

### Schema-validated state

Pair a `state_schema` Pydantic model with a `Workflow` or `App` to catch type errors at write time:

```python
from pydantic import BaseModel
from google.adk.workflow import Workflow, node, START
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

class PipelineState(BaseModel):
    items_processed: int = 0
    last_error: str = ""

@node
def process(node_input: str) -> str:
    return node_input.upper()

wf = Workflow(
    name="pipeline",
    edges=[(START, process)],
    state_schema=PipelineState,
)
# Writing an undeclared key will raise StateSchemaError:
# tool_context.state["typo_key"] = 1  → StateSchemaError
```

---

## 6 — `AgentTool`

**Module:** `google.adk.tools.agent_tool`

`AgentTool` wraps a `BaseAgent` as a `BaseTool`, so a parent `LlmAgent` can call it as a function. The sub-agent runs in its own isolated runner with a fresh `InMemorySessionService`.

> **Prefer `sub_agents` with `mode='single_turn'`.** The framework exposes single-turn sub-agents as tools automatically, without `AgentTool`. Use `AgentTool` only when you need fine-grained control over summarization or plugin propagation.

### Constructor parameters

Source-verified from `google/adk/tools/agent_tool.py`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `agent` | `BaseAgent` | required | The agent to wrap |
| `skip_summarization` | `bool` | `False` | Skip the post-run summarisation step |
| `include_plugins` | `bool` | `True` | Propagate the parent runner's plugins to the sub-agent runner |
| `propagate_grounding_metadata` | `bool` | `False` | Forward grounding metadata from the sub-agent's response |

### Minimal usage

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

# A specialised sub-agent
math_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="math_agent",
    description="Solves maths problems and returns numerical answers.",
    instruction="You are a maths expert. Return only numbers.",
    output_key="math_result",
)

# Wrap it as a tool
math_tool = AgentTool(agent=math_agent)

# Parent agent uses it via tool-call
coordinator = LlmAgent(
    model="gemini-2.0-flash",
    name="coordinator",
    instruction="Use the math_agent tool for all calculations.",
    tools=[math_tool],
)

async def run():
    runner = InMemoryRunner(
        agent=coordinator,
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    from google.genai import types as gt
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=gt.Content(
            role="user", parts=[gt.Part(text="What is 17 * 23?")]
        ),
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)

asyncio.run(run())
```

### Structured input/output with `input_schema`

When the wrapped `LlmAgent` has an `input_schema`, `AgentTool` generates a typed `FunctionDeclaration` from it so the LLM fills each field separately:

```python
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

class TranslationRequest(BaseModel):
    text: str = Field(description="Text to translate")
    target_language: str = Field(description="ISO 639-1 language code")

translator = LlmAgent(
    model="gemini-2.0-flash",
    name="translator",
    description="Translates text into a target language.",
    instruction="Translate the provided text into the target language. "
                "Return only the translation.",
    input_schema=TranslationRequest,
    output_key="translation",
)

translation_tool = AgentTool(agent=translator, skip_summarization=True)
# The parent LLM sees a tool with parameters {text, target_language}
```

### Isolating plugins

By default the parent's plugins are propagated to the sub-agent. Set `include_plugins=False` when the sub-agent should run in a clean environment:

```python
isolated_tool = AgentTool(
    agent=math_agent,
    include_plugins=False,  # sub-agent ignores parent's plugins
)
```

---

## 7 — `TransferToAgentTool`

**Module:** `google.adk.tools.transfer_to_agent_tool`

`TransferToAgentTool` is a `FunctionTool` subclass that gives the LLM an enum-constrained `agent_name` parameter, preventing hallucinated agent names. At runtime, it sets `tool_context.actions.transfer_to_agent` to redirect control.

### Constructor

Source-verified from `google/adk/tools/transfer_to_agent_tool.py`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `agent_names` | `list[str]` | required | The only agent names the LLM may select |
| `include_transfer_reason` | `bool` | `False` | Add a `transfer_reason` parameter to the tool signature |

### Basic multi-agent handoff

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools import TransferToAgentTool
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

billing_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="billing_agent",
    description="Handles billing and invoice queries.",
    instruction="You specialise in billing. Answer only billing questions.",
)

tech_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="tech_agent",
    description="Handles technical support queries.",
    instruction="You specialise in technical support.",
)

router = LlmAgent(
    model="gemini-2.0-flash",
    name="router",
    instruction=(
        "Route the user to the correct specialist. "
        "Billing questions → billing_agent. "
        "Technical questions → tech_agent."
    ),
    sub_agents=[billing_agent, tech_agent],
    tools=[
        TransferToAgentTool(
            agent_names=["billing_agent", "tech_agent"],
            include_transfer_reason=True,
        )
    ],
)
```

### With transfer reason for observability

```python
from google.adk.tools import TransferToAgentTool

# The LLM must provide a reason when transferring
handoff_tool = TransferToAgentTool(
    agent_names=["billing_agent", "tech_agent", "hr_agent"],
    include_transfer_reason=True,
)
# The reason is stored in tool_context.actions.transfer_reason
# and visible in telemetry spans.
```

### Why use `TransferToAgentTool` instead of the raw function

The raw `transfer_to_agent` function accepts any string as `agent_name`. `TransferToAgentTool` adds a JSON Schema `enum` to the declaration:

```json
{
  "name": "transfer_to_agent",
  "parameters": {
    "properties": {
      "agent_name": {
        "type": "string",
        "enum": ["billing_agent", "tech_agent"]
      }
    }
  }
}
```

This makes the LLM select from a closed list rather than fabricating names, which reduces routing errors in production systems.

---

## 8 — `BaseCodeExecutor` + `UnsafeLocalCodeExecutor`

**Modules:** `google.adk.code_executors.base_code_executor`, `.unsafe_local_code_executor`

Code executors allow `LlmAgent` to run Python code blocks from model responses and feed the output back as tool results. `BaseCodeExecutor` defines the interface; `UnsafeLocalCodeExecutor` runs code in the current Python interpreter via a subprocess.

### `BaseCodeExecutor` field reference

Source-verified from `google/adk/code_executors/base_code_executor.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `optimize_data_file` | `bool` | `False` | Extract CSV files from the request and pass them to the executor |
| `stateful` | `bool` | `False` | Maintain interpreter state across code blocks in one invocation |
| `error_retry_attempts` | `int` | `2` | How many times to retry on consecutive errors before giving up |
| `code_block_delimiters` | `list[tuple[str, str]]` | `[('```tool_code\n', '\n```'), ('```python\n', '\n```')]` | Patterns that identify code blocks in model output |
| `execution_result_delimiters` | `tuple[str, str]` | `('```tool_output\n', '\n```')` | How to format results injected back to the model |
| `timeout_seconds` | `int \| None` | `None` | Execution wall-clock timeout |

### `UnsafeLocalCodeExecutor`

`UnsafeLocalCodeExecutor` spawns a subprocess running the *current* Python interpreter and pipes the code via stdin. It handles:
- `__name__ == "__main__"` detection (sets the run name so guarded blocks execute)
- `SIGTERM` → `SIGKILL` escalation after `_TERMINATE_GRACE_SECONDS = 5` s
- Traceback stripping (removes the internal runner frame from error output)

> **Security warning:** `UnsafeLocalCodeExecutor` runs arbitrary model-generated code with full OS access. Use only in trusted, sandboxed environments. For production use Vertex AI Code Interpreter or a custom sandboxed executor.

### Attaching a code executor to an agent

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor

data_analyst = LlmAgent(
    model="gemini-2.0-flash",
    name="data_analyst",
    instruction=(
        "You are a data analyst. Write Python code to answer the user's question "
        "and run it to produce a numeric answer."
    ),
    code_executor=UnsafeLocalCodeExecutor(),
)
```

### With timeout and retry configuration

```python
from google.adk.code_executors import UnsafeLocalCodeExecutor

executor = UnsafeLocalCodeExecutor(
    timeout_seconds=10,        # kill after 10 s
    error_retry_attempts=1,    # retry once before giving up
    stateful=True,             # share interpreter state across blocks
)
```

### Writing a custom `BaseCodeExecutor`

```python
import subprocess
from google.adk.code_executors import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput,
    CodeExecutionResult,
)
from google.adk.agents.invocation_context import InvocationContext

class DockerCodeExecutor(BaseCodeExecutor):
    """Runs code inside a Docker container for isolation."""

    image: str = "python:3.13-slim"

    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult:
        result = subprocess.run(
            ["docker", "run", "--rm", "-i", self.image, "python", "-"],
            input=code_execution_input.code.encode(),
            capture_output=True,
            timeout=self.timeout_seconds or 30,
        )
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()
        return CodeExecutionResult(
            output=stdout or stderr,
            stderr=stderr,
        )
```

---

## 9 — `PluginManager`

**Module:** `google.adk.plugins.plugin_manager`

`PluginManager` is the internal orchestrator that invokes `BasePlugin` callbacks at each lifecycle hook. The `Runner` holds one `PluginManager`; when sub-agents run (e.g. inside `AgentTool`) they share the parent's plugin list unless `include_plugins=False`.

### Key lifecycle callbacks

Source-verified from `google/adk/plugins/plugin_manager.py`:

| Callback | Fired when | Can short-circuit? |
|---|---|---|
| `on_user_message_callback` | A new user message arrives | Yes (return a `Content`) |
| `before_run_callback` | Before `runner.run_async` starts | Yes (return a `Content`) |
| `after_run_callback` | After `runner.run_async` finishes | No |
| `on_event_callback` | Every event emitted | No |
| `before_agent_callback` | Before each agent call | Yes (return a `Content`) |
| `after_agent_callback` | After each agent call | No |
| `before_tool_callback` | Before each tool call | Yes (return a result dict) |
| `after_tool_callback` | After each tool call | Yes (return a replacement result) |
| `before_model_callback` | Before each LLM call | Yes (return an `LlmResponse`) |
| `after_model_callback` | After each LLM call | Yes (return a modified `LlmResponse`) |
| `on_tool_error_callback` | On tool error | Yes (return a recovery result) |
| `on_model_error_callback` | On model error | Yes (return a recovery response) |
| `on_agent_error_callback` | On agent error | No |
| `on_run_error_callback` | On run-level error | No |

The **early-exit** rule: if any plugin returns a non-`None` value, later plugins for that callback are skipped and the value propagates to the framework.

### Construction

```python
from google.adk.plugins import PluginManager, BasePlugin

manager = PluginManager(
    plugins=[my_plugin_a, my_plugin_b],
    close_timeout=5.0,  # seconds to wait for each plugin.close()
)
```

### Dynamic plugin registration

```python
manager.register_plugin(my_plugin_c)
# ValueError if a plugin with the same name is already registered
```

### Writing a plugin that intercepts all tool calls

```python
from typing import Any, Optional
from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool, ToolContext

class AuditPlugin(BasePlugin):
    """Logs every tool call and its result."""

    name = "audit_plugin"

    async def before_tool_callback(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        print(f"[AUDIT] Calling {tool.name}({args})")
        return None  # let the real tool run

    async def after_tool_callback(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict,
    ) -> Optional[dict]:
        print(f"[AUDIT] {tool.name} returned {tool_response}")
        return None  # return unmodified response
```

### Attaching plugins to a Runner

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

runner = Runner(
    agent=my_agent,
    session_service=InMemorySessionService(),
    plugins=[AuditPlugin()],
)
```

### Sharing plugins via `set_skip_closing_plugins`

When `AgentTool` propagates the parent runner's plugins (`include_plugins=True`, the default), the sub-agent's `PluginManager` is given `set_skip_closing_plugins(True)` so it does not tear down shared plugins when the sub-run finishes:

```python
# Internal ADK behaviour — shown for understanding, not for direct use
sub_manager = PluginManager(plugins=parent_plugins)
sub_manager.set_skip_closing_plugins(True)
# sub_manager.close() is now a no-op
```

---

## 10 — `CascadeLive`

**Module:** `google.adk.live._cascade_live`

`CascadeLive` is an **experimental** `BaseLlm` subclass that chains a speech-to-text (`LiveIngress`) transform, a standard text LLM, and a text-to-speech (`LiveEgress`) transform to create a speech-to-speech agent without requiring a native live-audio model.

```
wire audio → LiveIngress (STT) → text LLM → LiveEgress (TTS) → wire audio
```

Because the underlying model is a normal text model, all ADK features work unchanged: tools, instructions, multi-agent transfers, and workflows.

### Field reference

Source-verified from `google/adk/live/_cascade_live.py`:

| Field | Type | Purpose |
|---|---|---|
| `model` | `str` | The underlying text model name (e.g. `"gemini-2.0-flash"`) |
| `stt` | `LiveIngress` | Speech-to-text transform; consumes `AsyncIterator[types.Blob]` audio, yields `IngressEvent` |
| `tts` | `LiveEgress` | Text-to-speech transform; consumes streamed text deltas, yields `EgressEvent` (audio chunks) |

### `LiveIngress` and `LiveEgress` protocols

Both are abstract classes defined in `google.adk.live`:

```python
# LiveIngress protocol
class LiveIngress(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, audio: AsyncIterator[types.Blob]
    ) -> AsyncGenerator[IngressEvent, None]:
        ...

# LiveEgress protocol
class LiveEgress(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, text: AsyncIterator[str], *, cancel: asyncio.Event
    ) -> AsyncGenerator[EgressEvent, None]:
        ...
```

### Wiring a `CascadeLive` agent

```python
from google.adk.agents import LlmAgent
from google.adk.live import CascadeLive

# Assuming ElevenLabsSTT and ElevenLabsTTS are third-party LiveIngress/LiveEgress
# implementations (not part of ADK core — illustrative only)
from my_stt_package import ElevenLabsSTT
from my_tts_package import ElevenLabsTTS

voice_agent = LlmAgent(
    model=CascadeLive(
        model="gemini-2.0-flash",
        stt=ElevenLabsSTT(language="en"),
        tts=ElevenLabsTTS(voice_id="Rachel"),
    ),
    name="voice_assistant",
    instruction="You are a helpful voice assistant. Keep answers brief.",
    tools=[my_search_tool, my_calendar_tool],
)
```

### Non-live requests fall through

In non-live (text) requests, `CascadeLive.generate_content_async` delegates directly to the inner model with no STT/TTS overhead:

```python
# Non-live path — STT and TTS are bypassed
async for response in cascade_live.generate_content_async(llm_request, stream=False):
    ...
```

### Implementing a custom `LiveIngress` (STT)

```python
import asyncio
from typing import AsyncGenerator, AsyncIterator
from google.genai import types
from google.adk.live import LiveIngress
from google.adk.live._cascade_live_events import IngressEvent, UserTurnFinished

class WhisperSTT(LiveIngress):
    """Toy implementation using OpenAI Whisper (illustrative)."""

    def __call__(
        self, audio: AsyncIterator[types.Blob]
    ) -> AsyncGenerator[IngressEvent, None]:
        return self._transcribe(audio)

    async def _transcribe(
        self, audio: AsyncIterator[types.Blob]
    ) -> AsyncGenerator[IngressEvent, None]:
        import io
        chunks: list[bytes] = []
        async for blob in audio:
            chunks.append(blob.data)

        # In a real implementation call the Whisper API here
        transcript = "Hello, what is the weather today?"

        from google.adk.live._cascade_live_events import PartialTranscript
        yield PartialTranscript(text=transcript)
        yield UserTurnFinished(transcript=transcript)
```

### Implementing a custom `LiveEgress` (TTS)

```python
import asyncio
from typing import AsyncGenerator, AsyncIterator
from google.adk.live import LiveEgress
from google.adk.live._cascade_live_events import EgressEvent, AudioChunk

class PollyTTS(LiveEgress):
    """Toy TTS using Amazon Polly (illustrative)."""

    def __call__(
        self,
        text: AsyncIterator[str],
        *,
        cancel: asyncio.Event,
    ) -> AsyncGenerator[EgressEvent, None]:
        return self._synthesise(text, cancel)

    async def _synthesise(
        self,
        text: AsyncIterator[str],
        cancel: asyncio.Event,
    ) -> AsyncGenerator[EgressEvent, None]:
        buffer = []
        async for chunk in text:
            if cancel.is_set():
                return  # barge-in: stop synthesis
            buffer.append(chunk)

        full_text = "".join(buffer)
        # In a real implementation call the Polly API here
        audio_bytes = b"\x00" * 1024  # placeholder PCM
        yield AudioChunk(data=audio_bytes, sample_rate=24000)
```

> **Experimental flag:** `CascadeLive` requires `FeatureName.CASCADE_LIVE`. Enable it with the `GOOGLE_ADK_FEATURES=cascade_live` environment variable or via `google.adk.features.enable_feature(FeatureName.CASCADE_LIVE)`.

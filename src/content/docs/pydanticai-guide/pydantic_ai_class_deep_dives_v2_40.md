---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2.40.0)"
description: "Runnable, source-verified code examples for RealtimeSession, RealtimeModelSettings/TurnDetection, AgentRealtime, Capability, Hooks, TemplateStr, ExternalToolset, RetryConfig/HTTPX2TenacityTransport, DuckDuckGoSearchTool, and ImageGenerationSubagentTool — verified against pydantic-ai 2.40.0."
framework: pydanticai
language: python
sidebar:
  order: 126
---

# 10 Source-Verified Class Deep Dives — v2.40.0

Verified against **pydantic-ai 2.40.0** (installed package, sources read via `inspect.getsource`).
Modules consulted: `pydantic_ai/realtime/_session.py`, `pydantic_ai/realtime/settings.py`,
`pydantic_ai/agent/abstract.py`, `pydantic_ai/capabilities/capability.py`,
`pydantic_ai/capabilities/hooks.py`, `pydantic_ai/template.py`,
`pydantic_ai/toolsets/external.py`, `pydantic_ai/retries.py`,
`pydantic_ai/common_tools/duckduckgo.py`, `pydantic_ai/common_tools/image_generation.py`.

This page covers classes that were **absent or thin** in the two earlier deep-dive pages
(v2.33.0 and v2.36.0). All 10 items here are distinct from those sets.

```bash
pip install "pydantic-ai==2.40.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.40.0
```

---

## 1. `RealtimeSession` — voice / audio AI sessions

**Module:** `pydantic_ai.realtime`  
**Source:** `pydantic_ai/realtime/_session.py`

`RealtimeSession` wraps a low-level `RealtimeConnection` and builds a history-aware, tool-executing
voice session. It translates codec events (speech start/end, tool calls) into the same shared
`ModelMessage` vocabulary used by regular agent runs, so history from a voice session can be handed
off to `Agent.run` seamlessly.

### Constructor (key parameters)

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `connection` | `RealtimeConnection` | required | Low-level codec connection |
| `model` | `RealtimeModel \| None` | `None` | Provider-specific realtime model |
| `tool_manager` | `ToolManager[Any]` | required | Handles tool execution |
| `audio_retention` | `AudioRetention` | `'transcript_only'` | Which audio to keep in history |
| `handle_barge_in` | `bool` | `False` | Cancel response when user starts speaking |
| `retain_images_every_n` | `int` | `1` | Keep one image per N for high-rate streams |
| `retain_images_max` | `int \| None` | `100` | Cap on retained images (oldest evicted first) |
| `message_history` | `Sequence[ModelMessage] \| None` | `None` | Seed history from a prior text run |
| `instructions` | `str \| None` | `None` | System prompt for the session |
| `usage_limits` | `UsageLimits \| None` | `None` | Token / request budget |

### Public method reference

| Method | Signature | Purpose |
|---|---|---|
| `send` | `(content, *, respond?)` | Feed text, image, or audio into the session |
| `send_audio` | `(audio, *, sample_rate, channels?)` | Stream raw PCM16 audio chunks |
| `stream_audio` | `() → AsyncIterator[bytes]` | Receive model's audio output |
| `stream_transcripts` | `(*, user?, assistant?)` | Stream live speech-to-text updates |
| `commit_audio` | `()` | End user's push-to-talk turn |
| `create_response` | `()` | Ask model to respond (manual turn-taking) |
| `interrupt` | `()` | Barge-in: cancel model's in-progress response |
| `clear_audio` | `()` | Discard uncommitted input audio |
| `all_messages` | `()` | Full conversation history (seeded + new) |
| `new_messages` | `()` | Only messages created in this session |
| `close` | `()` | Shut down session and underlying connection |

### Example 1 — OpenAI voice assistant with a weather tool

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.realtime import TurnDetection


async def get_weather(city: str) -> str:
    """Return mock weather for demo purposes."""
    return f"Sunny, 22°C in {city}"


agent = Agent(
    "openai:gpt-4o",
    tools=[get_weather],
    instructions="You are a helpful voice assistant.",
)


async def main() -> None:
    # agent.realtime() returns an AgentRealtime; .session() opens a RealtimeSession
    async with agent.realtime(
        "openai:gpt-4o-realtime-preview",
        model_settings={
            "turn_detection": TurnDetection(
                sensitivity="medium",
                silence_duration_ms=600,
            ),
        },
    ).session() as session:
        # Feed a text prompt — VAD is on, so the model replies automatically
        await session.send("What's the weather like in Tokyo?")

        # Stream transcripts to see what the model says while it speaks
        async for update in session.stream_transcripts(assistant=True):
            print(update.content, end="", flush=True)
        print()

        # Hand history off to a standard agent run for structured follow-up
        text_result = await agent.run(
            "Summarise that exchange in one sentence.",
            message_history=session.all_messages(),
        )
        print(text_result.output)


asyncio.run(main())
```

### Example 2 — Manual push-to-talk with raw audio

```python {test="skip"}
import asyncio
import wave
import io
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o")


async def push_to_talk_demo(wav_bytes: bytes) -> None:
    """Feed a WAV file as a push-to-talk turn; stream audio back."""
    async with agent.realtime(
        "openai:gpt-4o-realtime-preview",
        model_settings={"turn_detection": False},  # push-to-talk: disable VAD
    ).session() as session:
        # Decode WAV — must be mono PCM16 for the realtime API
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            assert wf.getnchannels() == 1, "Expected mono audio"
            assert wf.getsampwidth() == 2, "Expected 16-bit (PCM16) audio"
            sample_rate = wf.getframerate()
            while chunk := wf.readframes(4096):
                await session.send_audio(chunk, sample_rate=sample_rate)

        # Signal end of user turn and request a response
        await session.commit_audio()
        await session.create_response()

        # Receive model's spoken reply as raw PCM chunks
        audio_chunks: list[bytes] = []
        async for chunk in session.stream_audio():
            audio_chunks.append(chunk)

        print(f"Received {sum(len(c) for c in audio_chunks)} bytes of audio")
        print(f"History length: {len(session.new_messages())} messages")
```

---

## 2. `RealtimeModelSettings` + `TurnDetection` — realtime session configuration

**Module:** `pydantic_ai.realtime`  
**Source:** `pydantic_ai/realtime/settings.py`

`RealtimeModelSettings` is a `TypedDict` that controls realtime session behaviour. `TurnDetection`
is a nested `TypedDict` for voice-activity detection (VAD) settings. Both are cross-provider:
unsupported fields are silently ignored per provider.

### `RealtimeModelSettings` fields

| Field | Type | Providers | Purpose |
|---|---|---|---|
| `max_tokens` | `int` | OpenAI, Azure, Gemini, xAI | Max tokens per model response |
| `parallel_tool_calls` | `bool` | OpenAI, Azure, xAI | Allow parallel function calls |
| `tool_choice` | `ToolChoice` | All (with caveats) | Control which tools the model can call |
| `input_transcription_model` | `str \| None` | OpenAI, Gemini | Transcription model for user audio |
| `output_audio_format` | `str` | OpenAI, Azure | Audio encoding (e.g. `'pcm16'`) |
| `turn_detection` | `TurnDetection \| bool \| None` | All | VAD config; `True` = provider defaults; `False` = push-to-talk |
| `instructions` | `str` | OpenAI | System prompt (overrides agent instructions) |
| `voice` | `str` | OpenAI, Azure, Gemini, xAI | TTS voice name |
| `temperature` | `float` | OpenAI, Azure, xAI | Output randomness |
| `thinking_budget_tokens` | `int` | Gemini | Extended thinking token budget |

### `TurnDetection` fields

| Field | Type | Default | Providers | Purpose |
|---|---|---|---|---|
| `sensitivity` | `'low' \| 'medium' \| 'high'` | provider default | OpenAI, Azure, xAI, Gemini | Speech detection sensitivity |
| `prefix_padding_ms` | `int` | provider default | OpenAI, Azure, xAI, Gemini | Audio buffered before detected speech |
| `silence_duration_ms` | `int` | provider default | OpenAI, Azure, xAI, Gemini | Silence required to end a turn |

### `AudioRetention` values

| Value | Stored in history |
|---|---|
| `'transcript_only'` (default) | Transcripts only; no audio bytes |
| `'input_audio'` | Transcript + user's audio |
| `'output_audio'` | Transcript + model's audio |
| `'all'` | Transcript + both sides' audio |

### Example — low-latency voice configuration

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime import TurnDetection, RealtimeModelSettings

# Tight VAD: respond quickly, use a fast voice
settings: RealtimeModelSettings = {
    "voice": "alloy",
    "turn_detection": TurnDetection(
        sensitivity="high",       # snap to speech quickly
        silence_duration_ms=300,  # 300 ms silence ends the turn
        prefix_padding_ms=100,    # keep 100 ms before onset
    ),
    "max_tokens": 512,
    "parallel_tool_calls": True,
}

agent = Agent("openai:gpt-4o")
realtime = agent.realtime("openai:gpt-4o-realtime-preview", model_settings=settings)
```

### Example — Gemini with extended thinking

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime import RealtimeModelSettings

settings: RealtimeModelSettings = {
    "thinking_budget_tokens": 2048,
    "turn_detection": True,   # use Gemini's default VAD
    "voice": "Puck",
}

agent = Agent("google:gemini-2.0-flash-exp")
realtime = agent.realtime("google:gemini-2.0-flash-live-001", model_settings=settings)
```

---

## 3. `AgentRealtime` — realtime agent binding

**Module:** `pydantic_ai.agent`  
**Source:** `pydantic_ai/agent/abstract.py`

`AgentRealtime` is the object returned by `agent.realtime(model, ...)`. It carries the agent's
realtime configuration (instructions, toolsets, capabilities, deps) so that opening multiple sessions
reuses the same setup without re-passing everything. It also exposes the WebRTC SDP offer/answer
flow for browser-to-server voice applications.

### Obtaining an `AgentRealtime`

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.realtime import TurnDetection

agent = Agent(
    "openai:gpt-4o",
    instructions="You are a helpful assistant.",
)

# All parameters are optional after the model
realtime = agent.realtime(
    "openai:gpt-4o-realtime-preview",
    deps=None,                          # agent dependency (same type as Agent.deps_type)
    model_settings={"voice": "echo"},
    instructions="Override instructions for realtime only.",
    usage_limits=None,
)
```

### Example 1 — Multiple sessions sharing the same configuration

```python {test="skip"}
import asyncio
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o")
realtime = agent.realtime("openai:gpt-4o-realtime-preview")

async def handle_caller(caller_id: str, first_message: str) -> list:
    """Open a session per caller, all sharing the same realtime binding."""
    async with realtime.session() as session:
        await session.send(first_message)
        async for update in session.stream_transcripts(assistant=True):
            print(f"[{caller_id}] {update.content}", end="", flush=True)
        print()
        return session.new_messages()


async def main() -> None:
    results = await asyncio.gather(
        handle_caller("caller-1", "What is the capital of France?"),
        handle_caller("caller-2", "Tell me a short joke."),
    )
    print(f"Total messages: {sum(len(r) for r in results)}")

asyncio.run(main())
```

### Example 2 — WebRTC offer/answer for browser voice chat

```python {test="skip"}
from fastapi import FastAPI, Request
from pydantic_ai import Agent

app = FastAPI()
agent = Agent("openai:gpt-4o", instructions="Voice assistant.")
realtime = agent.realtime("openai:gpt-4o-realtime-preview", model_settings={"voice": "alloy"})


@app.post("/rtc-connect")
async def rtc_connect(request: Request) -> dict:
    """Browser sends SDP offer; we respond with the SDP answer."""
    body = await request.json()
    answer = await realtime.answer_webrtc_offer(body["sdp"])
    return {"type": "answer", "sdp": answer.sdp}
```

---

## 4. `Capability` — bundle instructions, tools, and toolsets

**Module:** `pydantic_ai.capabilities`  
**Source:** `pydantic_ai/capabilities/capability.py`

`Capability` is the high-level, no-subclassing way to group related instructions, function tools,
and toolsets into a single reusable unit. It wraps a `FunctionToolset` internally and registers
everything with the agent at construction time.

### Constructor

```python
Capability(
    *,
    instructions: AgentInstructions | None = None,   # static str, callable, or list of either
    toolsets: Sequence[AgentToolset] | None = None,
    tools: Sequence[Tool | Callable] = (),
    id: str | None = None,         # required when defer_loading=True
    description: str | Callable | None = None,
    defer_loading: bool = False,   # hide until model calls load_capability
)
```

### Example 1 — Audit trail capability

```python {test="skip"}
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import RunContext


def get_current_time(ctx: RunContext[None]) -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def log_action(ctx: RunContext[None], action: str) -> str:
    """Record an action to the audit log and return a confirmation."""
    print(f"AUDIT: {action}")
    return f"Logged: {action}"


audit_capability = Capability(
    instructions="Always log your actions using the log_action tool before returning results.",
    tools=[get_current_time, log_action],
    id="audit",
)

agent = Agent("openai:gpt-4o", capabilities=[audit_capability])
result = agent.run_sync("What time is it? Log that you checked.")
print(result.output)
```

### Example 2 — Deferred capability (loaded on demand)

When `defer_loading=True`, the model sees only the capability's `description` and a
`load_capability` tool. The actual tools and instructions stay hidden until the model explicitly
loads the capability.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability


def search_database(query: str) -> list[str]:
    """Search the internal database."""
    return [f"Result for '{query}': item-1", "item-2", "item-3"]


def run_analysis(data: list[str]) -> str:
    """Run statistical analysis on data."""
    return f"Analysis of {len(data)} items: mean length {sum(len(d) for d in data) / len(data):.1f} chars"


heavy_capability = Capability(
    instructions="You have access to a database and analysis tools.",
    tools=[search_database, run_analysis],
    id="db-analysis",
    description="Database search and statistical analysis tools. Load when the user asks to search or analyse data.",
    defer_loading=True,
)

agent = Agent(
    "openai:gpt-4o",
    capabilities=[heavy_capability],
    instructions="Help with data queries. Load specialised tools when needed.",
)
result = agent.run_sync("Can you search for 'revenue' and analyse the results?")
print(result.output)
```

### Example 3 — Capability with an instruction function

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import RunContext
from dataclasses import dataclass


@dataclass
class UserDeps:
    username: str
    role: str


def get_user_context(ctx: RunContext[UserDeps]) -> str:
    """Dynamic instruction based on the current user."""
    return (
        f"You are helping {ctx.deps.username} who has role '{ctx.deps.role}'. "
        f"{'You may discuss admin topics.' if ctx.deps.role == 'admin' else 'Restrict to user-level topics.'}"
    )


user_capability: Capability[UserDeps] = Capability(
    instructions=get_user_context,
    id="user-context",
)

agent = Agent("openai:gpt-4o", deps_type=UserDeps, capabilities=[user_capability])
result = agent.run_sync(
    "What can you help me with?",
    deps=UserDeps(username="Alice", role="admin"),
)
print(result.output)
```

---

## 5. `Hooks` — decorator-based lifecycle callbacks

**Module:** `pydantic_ai.capabilities`  
**Source:** `pydantic_ai/capabilities/hooks.py`

`Hooks` is an `AbstractCapability` subclass that lets you register lifecycle callbacks without
subclassing. It exposes 30+ hook points covering the full run, node, model-request, tool, and
output lifecycle. Both sync and async callbacks are accepted.

### Available hook points (via `hooks.on.<hook>` decorator or constructor kwarg)

| Hook | When it fires |
|---|---|
| `before_run` / `after_run` | Before/after the entire agent run |
| `run` | Wrap the entire run (generator) |
| `run_error` | Unhandled exception during the run |
| `run_event_stream` | Wrap the event-stream iterator for the run |
| `before_node_run` / `after_node_run` | Before/after each graph node |
| `node_run` | Wrap a node execution |
| `node_run_error` | Unhandled exception from a node |
| `before_model_request` / `after_model_request` | Before/after each LLM call |
| `model_request` | Wrap the model call |
| `model_request_error` | Exception from the model |
| `prepare_tools` / `prepare_output_tools` | Modify the tool list before each request |
| `before_tool_validate` / `after_tool_validate` | Around tool argument validation |
| `tool_validate` | Wrap tool argument validation |
| `tool_validate_error` | Exception during tool argument validation |
| `before_tool_execute` / `after_tool_execute` | Around tool execution |
| `tool_execute` | Wrap a tool call |
| `tool_execute_error` | Exception during tool execution |
| `before_output_validate` / `after_output_validate` | Around output validation |
| `output_validate` | Wrap output validation |
| `output_validate_error` | Exception during output validation |
| `before_output_process` / `after_output_process` | Around output post-processing |
| `output_process` | Wrap output post-processing |
| `output_process_error` | Exception during output post-processing |
| `event` | Every `AgentStreamEvent` in the run's event stream |
| `deferred_tool_calls` | Handle deferred (HITL / external) tool calls |

### Example 1 — Observability hooks (decorator style)

```python {test="skip"}
import time
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import RunContext

hooks = Hooks()


@hooks.on.before_run
async def start_timer(ctx: RunContext[None]) -> None:
    if ctx.metadata is None:
        ctx.metadata = {}
    ctx.metadata["_start"] = time.monotonic()
    ctx.metadata["_req"] = 0
    agent_name = ctx.agent.name if ctx.agent else "unknown"
    print(f"Run started for agent: {agent_name}")


@hooks.on.after_run
async def log_duration(ctx: RunContext[None], *, result):
    elapsed = time.monotonic() - ctx.metadata.get("_start", time.monotonic())
    print(f"Run finished in {elapsed:.2f}s")
    return result


@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context):
    ctx.metadata["_req"] = ctx.metadata.get("_req", 0) + 1
    print(f"  → Model request #{ctx.metadata['_req']}")
    return request_context


@hooks.on.after_model_request
async def log_response(ctx: RunContext[None], *, response, request_context):
    print(f"  ← Model responded")
    return response


@hooks.on.before_tool_execute
async def log_tool(ctx: RunContext[None], *, call, tool_def, args):
    print(f"  🔧 Calling tool '{tool_def.name}' with {args}")
    return args


agent = Agent("openai:gpt-4o", capabilities=[hooks])
result = agent.run_sync("What is 2 + 2? Use a tool to calculate.")
print(result.output)
```

### Example 2 — Constructor kwargs style (for one-liners)

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import RunContext


async def count_tokens(ctx: RunContext[None], *, response, request_context):
    """Accumulate usage across requests."""
    usage = ctx.usage
    print(f"Usage so far: {usage.total_tokens} tokens")
    return response


hooks = Hooks(after_model_request=count_tokens)
agent = Agent("openai:gpt-4o", capabilities=[hooks])
result = agent.run_sync("Explain quantum entanglement in two sentences.")
print(result.output)
```

### Example 3 — Tool filtering via `prepare_tools`

```python {test="skip"}
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import RunContext, ToolDefinition


@dataclass
class AppDeps:
    user_role: str


hooks: Hooks[AppDeps] = Hooks()


@hooks.on.prepare_tools
async def filter_admin_tools(
    ctx: RunContext[AppDeps],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Remove admin-only tools for non-admin users."""
    if ctx.deps.user_role != "admin":
        tool_defs = [t for t in tool_defs if not t.name.startswith("admin_")]
    return tool_defs


def admin_delete_record(record_id: str) -> str:
    """Admin-only: delete a record."""
    return f"Deleted record {record_id}"


def get_record(record_id: str) -> str:
    """Public: retrieve a record."""
    return f"Record {record_id}: {{data: 'example'}}"


agent = Agent(
    "openai:gpt-4o",
    deps_type=AppDeps,
    tools=[admin_delete_record, get_record],
    capabilities=[hooks],
)

# Non-admin cannot use admin_delete_record
result = agent.run_sync(
    "Try to delete record 42.",
    deps=AppDeps(user_role="user"),
)
print(result.output)
```

### `HookTimeoutError`

Each hook registration accepts an optional `timeout` parameter. If a hook function exceeds it,
`pydantic_ai.capabilities.hooks.HookTimeoutError` (subclass of `AgentRunError` and `TimeoutError`)
is raised with `.hook_name`, `.func_name`, and `.timeout` attributes.

```python {test="skip"}
from pydantic_ai.capabilities import Hooks
from pydantic_ai.capabilities.hooks import HookTimeoutError
from pydantic_ai.tools import RunContext
import asyncio

hooks = Hooks()


@hooks.on.before_model_request(timeout=2.0)   # fail if hook takes > 2 s
async def slow_hook(ctx: RunContext[None], request_context):
    await asyncio.sleep(10)  # would cause HookTimeoutError before this returns
    return request_context   # never reached; shown here for correct hook contract
```

---

## 6. `TemplateStr` — Handlebars templates for dynamic instructions

**Module:** `pydantic_ai.template`  
**Source:** `pydantic_ai/template.py`

`TemplateStr` turns a [pydantic-handlebars](https://github.com/pydantic/pydantic-handlebars) template
string into a callable that renders against `RunContext.deps`. Use it as a drop-in replacement for
a static instruction string wherever `pydantic_ai` accepts instructions.

Strings that contain `{{` are automatically compiled during Pydantic validation, so you can pass
a plain `str` to any `TemplateStr`-typed parameter and get compilation for free.

### Constructor

```python
TemplateStr(
    source: str,               # Handlebars template, e.g. "Hello {{name}}"
    *,
    deps_type: type | None = None,   # for standalone use outside an agent
    deps_schema: dict | None = None, # JSON schema for template variable validation
)
```

### Key method

| Method | Signature | Purpose |
|---|---|---|
| `render` | `(deps=None) → str` | Render template against deps object |
| `__call__` | `(ctx: RunContext) → str` | Makes `TemplateStr` usable as an instruction callable |

### Example 1 — Dataclass deps

```python {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.template import TemplateStr


@dataclass
class UserProfile:
    username: str
    preferred_language: str
    timezone: str


agent = Agent(
    "openai:gpt-4o",
    deps_type=UserProfile,
    instructions=TemplateStr(
        "You are a helpful assistant for {{username}}. "
        "Respond in {{preferred_language}}. "
        "All times should be in {{timezone}} timezone."
    ),
)


async def main() -> None:
    result = await agent.run(
        "What time is noon UTC in my timezone?",
        deps=UserProfile(
            username="Alice",
            preferred_language="French",
            timezone="Europe/Paris",
        ),
    )
    print(result.output)


asyncio.run(main())
```

### Example 2 — Standalone rendering (outside an agent)

```python {test="skip"}
from dataclasses import dataclass
from pydantic_ai.template import TemplateStr


@dataclass
class Context:
    product: str
    audience: str
    tone: str


template = TemplateStr(
    "Write a {{tone}} product description for {{product}} targeting {{audience}}.",
    deps_type=Context,
)

rendered = template.render(
    Context(product="noise-cancelling headphones", audience="remote workers", tone="professional")
)
print(rendered)
# Write a professional product description for noise-cancelling headphones targeting remote workers.
```

### Example 3 — Template in a `Capability`

```python {test="skip"}
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.template import TemplateStr


@dataclass
class TenantDeps:
    tenant_name: str
    allowed_topics: list[str]


topics_capability: Capability[TenantDeps] = Capability(
    instructions=TemplateStr(
        "You are an assistant for {{tenant_name}}. "
        "Only discuss: {{#each allowed_topics}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}."
    ),
    id="tenant-scope",
)

agent = Agent("openai:gpt-4o", deps_type=TenantDeps, capabilities=[topics_capability])
```

---

## 7. `ExternalToolset` — tools executed outside the agent run

**Module:** `pydantic_ai.toolsets`  
**Source:** `pydantic_ai/toolsets/external.py`

`ExternalToolset` declares tools whose results are produced **outside** the agent run — for example,
by a human approver, a remote worker, or a browser client. The agent sees the tool definitions and
produces `ToolCallPart` items, but never executes them. The caller collects the calls via
`DeferredToolRequests` and submits results back in the next `Agent.run` call.

### Constructor

```python
ExternalToolset(
    tool_defs: list[ToolDefinition],
    *,
    id: str | None = None,
)
```

### Example 1 — Human-in-the-loop approval gate

```python {test="skip"}
import asyncio
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition


# Declare the tool schema without any execution logic
external_toolset = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="approve_payment",
            description="Request approval to process a payment.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "amount_usd": {"type": "number", "description": "Amount in USD"},
                    "recipient": {"type": "string", "description": "Recipient name or account"},
                    "reason": {"type": "string", "description": "Business justification"},
                },
                "required": ["amount_usd", "recipient", "reason"],
            },
        )
    ],
    id="payments",
)

# output_type includes str so the resumed run can return a normal text confirmation
agent = Agent(
    "openai:gpt-4o",
    output_type=[str, DeferredToolRequests],
    toolsets=[external_toolset],
    instructions="You can approve payments. Always justify each payment.",
)


async def main() -> None:
    # First run: agent produces tool calls but cannot execute them
    result = await agent.run("Please approve a $500 payment to Acme Corp for server hosting.")

    deferred = result.output  # str or DeferredToolRequests
    if isinstance(deferred, DeferredToolRequests) and deferred.calls:
        for call in deferred.calls:
            print(f"Pending external call: {call.tool_name}({call.args_as_dict()})")

        # Simulate the external system completing the call and returning a result
        results = deferred.build_results(
            calls={call.tool_call_id: "Approved by finance team." for call in deferred.calls},
        )

        # Second run: pass the external results back so the agent can continue
        # (external_toolset is already registered on the agent; don't pass it again)
        final = await agent.run(
            "",
            message_history=result.all_messages(),
            deferred_tool_results=results,
        )
        print(final.output)
    elif isinstance(deferred, DeferredToolRequests):
        # DeferredToolRequests with no calls — agent decided no external tool was needed
        print("No external calls were requested.")
    else:
        # Agent responded with a plain text message
        print(deferred)


asyncio.run(main())
```

### Example 2 — Remote worker pattern

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition

# Define a tool that runs in a separate worker process
external_toolset = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name="run_ml_inference",
            description="Run a machine learning inference job on the GPU cluster.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "input_data": {"type": "string"},
                    "batch_size": {"type": "integer", "default": 32},
                },
                "required": ["model_name", "input_data"],
            },
        )
    ]
)

agent = Agent("openai:gpt-4o", toolsets=[external_toolset])
```

---

## 8. `RetryConfig` + `HTTPX2TenacityTransport` — HTTP retry with tenacity

**Module:** `pydantic_ai.retries`  
**Source:** `pydantic_ai/retries.py`  
**Extra:** `pip install "pydantic-ai-slim[retries]"` (or `pip install httpx2 tenacity`)

The `retries` module integrates [tenacity](https://tenacity.readthedocs.io/) with `httpx2` HTTP
transports so you can wrap any provider's HTTP client with automatic retry logic, including
`Retry-After` header awareness.

### `RetryConfig` fields (all optional `TypedDict`)

| Field | Type | Purpose |
|---|---|---|
| `retry` | `RetryBaseT` | Which exceptions to retry (e.g. `retry_if_exception_type`) |
| `wait` | `WaitBaseT` | Wait strategy between retries |
| `stop` | `StopBaseT` | When to stop (e.g. `stop_after_attempt`) |
| `before` | `Callable` | Called before each attempt |
| `after` | `Callable` | Called after each attempt |
| `before_sleep` | `Callable` | Called before sleeping between retries |
| `reraise` | `bool` | Re-raise the last exception if all retries exhausted |
| `sleep` | `Callable` | Custom sleep function |

### `wait_retry_after` — Retry-After header-aware wait

`wait_retry_after(fallback_strategy=None, max_wait=300)` returns a tenacity wait function that:
1. Reads the `Retry-After` header from `httpx2.HTTPStatusError`
2. Handles both integer seconds and RFC-9110 HTTP date formats
3. Falls back to `fallback_strategy` (default: `wait_exponential(max=60)`) when no header is present
4. Caps at `max_wait` seconds regardless

### Example 1 — Retry 429s with Retry-After awareness

```python {test="skip"}
import httpx2
from tenacity import retry_if_exception, stop_after_attempt

from pydantic_ai.retries import (
    HTTPX2TenacityTransport,
    RetryConfig,
    wait_retry_after,
)

def _is_429(exc: BaseException) -> bool:
    return isinstance(exc, httpx2.HTTPStatusError) and exc.response.status_code == 429


transport = HTTPX2TenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception(_is_429),        # only retry 429 Too Many Requests
        wait=wait_retry_after(max_wait=120),      # respects Retry-After header; caps at 2 min
        stop=stop_after_attempt(5),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),  # 4xx/5xx → exception
)

client = httpx2.Client(transport=transport)
response = client.get("https://api.example.com/data")
print(response.json())
```

### Example 2 — Async transport with custom retry predicate

```python {test="skip"}
import asyncio
import httpx2
from tenacity import retry_if_exception, stop_after_attempt, wait_exponential

from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig


def is_server_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx2.HTTPStatusError)
        and exc.response.status_code >= 500
    )


async def fetch_with_retry(url: str) -> dict:
    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception(is_server_error),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(4),
            reraise=True,
            before_sleep=lambda state: print(
                f"  Retry {state.attempt_number} after {state.outcome.exception()}"
            ),
        ),
        validate_response=lambda r: r.raise_for_status(),
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        response = await client.get(url)
        return response.json()


# asyncio.run(fetch_with_retry("https://api.example.com/endpoint"))
```

### Example 3 — Using `RetryConfig` with an OpenAI provider

```python {test="skip"}
import httpx2
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai import Agent
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig, wait_retry_after

# Build a resilient async HTTP client
retry_transport = AsyncHTTPX2TenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(httpx2.HTTPStatusError),
        wait=wait_retry_after(max_wait=60),
        stop=stop_after_attempt(3),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),
)

# Inject into the OpenAI provider's HTTP client
from pydantic_ai.models.openai import OpenAIModel
import openai

openai_client = openai.AsyncOpenAI(
    http_client=httpx2.AsyncClient(transport=retry_transport)
)
model = OpenAIModel("gpt-4o", openai_client=openai_client)
agent = Agent(model)
```

---

## 9. `DuckDuckGoSearchTool` + `duckduckgo_search_tool` — DuckDuckGo search

**Module:** `pydantic_ai.common_tools.duckduckgo`  
**Source:** `pydantic_ai/common_tools/duckduckgo.py`  
**Extra:** `pip install "pydantic-ai-slim[duckduckgo]"` (installs `ddgs`)

`duckduckgo_search_tool()` creates a ready-to-use `Tool` backed by `DuckDuckGoSearchTool`, which
calls DuckDuckGo's text search API asynchronously via `anyio.to_thread.run_sync` and returns a
typed list of `DuckDuckGoResult` TypedDicts.

### `DuckDuckGoResult` TypedDict fields

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Page title |
| `href` | `str` | Page URL |
| `body` | `str` | Page snippet / body |

### `duckduckgo_search_tool` factory

```python
duckduckgo_search_tool(
    duckduckgo_client: DDGS | None = None,  # defaults to a new DDGS()
    max_results: int | None = None,         # None = first-response results only
) -> Tool
```

### Example 1 — Minimal setup

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[duckduckgo_search_tool(max_results=5)],
    instructions="Search the web to answer questions. Cite your sources.",
)

result = agent.run_sync("What are the latest developments in quantum computing?")
print(result.output)
```

### Example 2 — Custom DDGS client with proxy

```python {test="skip"}
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Configure DDGS with proxy or custom headers
ddgs_client = DDGS(proxy="socks5://localhost:9050", timeout=20)

agent = Agent(
    "openai:gpt-4o",
    tools=[duckduckgo_search_tool(duckduckgo_client=ddgs_client, max_results=10)],
    instructions=(
        "You are a research assistant. Search DuckDuckGo for information, "
        "synthesise the results, and always include a list of sources."
    ),
)

result = agent.run_sync("Find recent papers on LLM alignment techniques.")
print(result.output)
```

### Example 3 — Combining search with structured output

```python {test="skip"}
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool


class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str]
    sources: list[str]
    confidence: float  # 0–1


agent = Agent(
    "openai:gpt-4o",
    output_type=ResearchSummary,
    tools=[duckduckgo_search_tool(max_results=8)],
    instructions=(
        "Search the web for information, then produce a structured research summary. "
        "Extract 3–5 key findings and list the URLs you used."
    ),
)


async def main() -> None:
    result = await agent.run("Research the current state of fusion energy.")
    summary = result.output
    print(f"Topic: {summary.topic}")
    print(f"Confidence: {summary.confidence:.0%}")
    print("Findings:")
    for finding in summary.key_findings:
        print(f"  • {finding}")
    print("Sources:")
    for source in summary.sources:
        print(f"  - {source}")


asyncio.run(main())
```

---

## 10. `ImageGenerationSubagentTool` + `image_generation_tool` — image generation

**Module:** `pydantic_ai.common_tools.image_generation`  
**Source:** `pydantic_ai/common_tools/image_generation.py`

`image_generation_tool()` creates a `Tool` that generates images via a **subagent** with a
specified image-generation model. This is the fallback path for agents whose primary model doesn't
support native image generation: the tool spins up a temporary subagent, calls the provider's image
model, and returns a `BinaryImage` that the outer agent can embed in its response.

### `ImageGenerationSubagentTool` dataclass fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | `Model \| str \| Callable` | required | Model for image generation (or factory) |
| `native_tool` | `ImageGenerationTool \| Callable` | required | Provider's native image tool config |
| `instructions` | `str` | `'Generate an image...'` | Instructions for the subagent |

### `image_generation_tool` factory

```python
image_generation_tool(
    model: Model | str | Callable,         # e.g. 'openai-responses:gpt-5.4'
    native_tool: ImageGenerationTool | Callable,
    *,
    instructions: str = "Generate an image based on the user prompt. Do not ask clarifying questions.",
) -> Tool
```

> **Note:** Do **not** pass a dedicated image-only model (e.g. `'dall-e-3'`, `'imagen-3.0-generate-002'`)
> as the `model` argument — those models cannot run the subagent loop. Use a conversational model
> that supports image generation (e.g. `'openai-responses:gpt-5.4'`) instead.

### Example 1 — Image generation tool with OpenAI

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.common_tools.image_generation import image_generation_tool

# The outer agent is a text model; image generation is handled by the subagent
agent = Agent(
    "openai:gpt-4o",
    tools=[
        image_generation_tool(
            model="openai-responses:gpt-5.4",          # conversational model for the subagent
            native_tool=ImageGenerationTool(
                model="gpt-image-2",             # actual image generation model
                size="1024x1024",
            ),
        )
    ],
    instructions=(
        "You can generate images. When asked, call the generate_image tool "
        "with a detailed prompt and return the image to the user."
    ),
)


async def main() -> None:
    result = await agent.run(
        "Create an image of a futuristic city at sunset with flying cars."
    )
    print(result.output)  # str — the outer agent's text response describing the image

    # The generated BinaryImage is stored in the tool-return part of the message history
    from pydantic_ai.messages import ToolReturnPart, BinaryImage
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolReturnPart) and isinstance(part.content, BinaryImage):
                img = part.content
                with open("generated_city.png", "wb") as f:
                    f.write(img.data)
                print(f"Saved image: {len(img.data)} bytes ({img.media_type})")


asyncio.run(main())
```

### Example 2 — Dynamic model selection per request

`model` can be a callable `(RunContext) → Model | str` for per-run model choice:

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.tools import RunContext
from dataclasses import dataclass


@dataclass
class AppDeps:
    tier: str  # 'free' or 'premium'


def choose_model(ctx: RunContext[AppDeps]) -> str:
    """Select image model based on user tier."""
    return (
        "openai-responses:gpt-5.5"     # premium: highest quality
        if ctx.deps.tier == "premium"
        else "openai-responses:gpt-5.4"  # free: standard quality
    )


agent = Agent(
    "openai:gpt-4o",
    deps_type=AppDeps,
    tools=[
        image_generation_tool(
            model=choose_model,
            native_tool=ImageGenerationTool(model="gpt-image-2"),
        )
    ],
)


async def main() -> None:
    result = await agent.run(
        "Generate a logo for a coffee shop called 'Morning Brew'.",
        deps=AppDeps(tier="premium"),
    )
    print(result.output)


asyncio.run(main())
```

### Example 3 — Google Imagen via subagent

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.common_tools.image_generation import image_generation_tool

agent = Agent(
    "google:gemini-2.0-flash",   # text model for the outer agent
    tools=[
        image_generation_tool(
            model="google:gemini-3-pro-image",          # conversational model for subagent
            native_tool=ImageGenerationTool(
                model="imagen-3.0-generate-002",   # Imagen 3
            ),
            instructions=(
                "Generate a high-quality image. Include artistic style details in the prompt."
            ),
        )
    ],
)
result = agent.run_sync("Create a watercolour painting of the Eiffel Tower at dusk.")
print(result.output)
```

---

## Quick-reference: what's in each deep-dive page

| Page | Classes | Version |
|---|---|---|
| `pydantic_ai_class_examples_2026_08` | `Agent`, `RunContext`, `UsageLimits`, `ToolReturn`, `DeferredToolRequests`/`Results`, `CachePoint`, `PrefixedToolset`/`FilteredToolset`/`RenamedToolset`, `WebSearchTool`, error taxonomy | 2.33.0 |
| `pydantic_ai_class_deep_dives_v2_36` | `AgentRun`, `AgentRunResult`, `StreamedRunResult`, `ModelSettings`, `Tool`, `ToolDefinition`, `RunUsage`/`RequestUsage`, `ConcurrencyLimiter`/`ConcurrencyLimit`, `MCPToolset`, `ApprovalRequiredToolset`/`DynamicToolset` | 2.36.0 |
| **This page** | `RealtimeSession`, `RealtimeModelSettings`/`TurnDetection`, `AgentRealtime`, `Capability`, `Hooks`, `TemplateStr`, `ExternalToolset`, `RetryConfig`/`HTTPX2TenacityTransport`, `DuckDuckGoSearchTool`, `ImageGenerationSubagentTool` | **2.40.0** |

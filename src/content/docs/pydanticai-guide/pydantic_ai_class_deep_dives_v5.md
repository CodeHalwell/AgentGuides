---
title: "PydanticAI — Class Deep Dives Vol. 5"
description: "Source-verified deep dives into 10 PydanticAI classes: PendingMessage/RunContext.enqueue, AgentWorker/agent_to_a2a, WrapperAgent, safe_download/SSRF protection, DBOSAgent, StreamedResponseSync, ModelResponsePartsManager, CombinedCapability, FastMCPToolset (migration), SetToolMetadata."
sidebar:
  label: "Class deep dives (Vol. 5)"
  order: 25
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.104.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.104.0`.
</Aside>

Ten class groups from the `pydantic_ai` 1.104.0 source covering the message-enqueueing API for
mid-run injection, the Agent-to-Agent (A2A) protocol bridge, the agent-wrapping base class,
the SSRF-safe HTTP downloader, durable DBOS workflows, synchronous streaming, the internal
streaming event infrastructure, the capability composition engine, the deprecated FastMCP toolset
and its migration path, and the `SetToolMetadata` capability.

---

## 1. `PendingMessage` + `RunContext.enqueue` / `AgentRun.enqueue`

**Module:** `pydantic_ai._enqueue` (types) · `pydantic_ai.run` (enqueue method)  
**Import:** Used via `ctx.enqueue(...)` inside a tool or `agent_run.enqueue(...)` from outside

The enqueueing API lets you inject messages into a running agent conversation from anywhere:
inside a tool function, from a background task, or from an external event handler. Injected
messages are buffered as `PendingMessage` objects and delivered at the earliest safe point by
the auto-injected `PendingMessageDrainCapability`.

### Key types

```python
PendingMessagePriority: TypeAlias = Literal['asap', 'when_idle']
EnqueueContent: TypeAlias = 'UserContent | ModelRequestPart | ModelMessage'
```

- **`'asap'`** (default) — delivered into the next `ModelRequest` as it is built, or, if the
  agent would otherwise terminate, used to redirect the run for one more turn.
- **`'when_idle'`** — delivered only when the agent would otherwise end, after all `'asap'`
  messages. Useful for follow-up instructions that should not interrupt in-flight work.

`EnqueueContent` accepts:
- A plain `str` or multimodal content (`ImageUrl`, `BinaryContent`, etc.) → coalesced into a
  `UserPromptPart`.
- A `ModelRequestPart` (`SystemPromptPart`, `ToolReturnPart`, etc.) → included verbatim.
- A complete `ModelRequest` or `ModelResponse` → kept as its own wire message.

The assembled sequence must **end with a `ModelRequest`** so the agent has something to respond to.

### `PendingMessage` dataclass

```python
@dataclass
class PendingMessage:
    messages: list[ModelMessage]   # assembled at enqueue time; always ends in ModelRequest
    priority: PendingMessagePriority = 'asap'

    @classmethod
    def from_content(cls, *content: EnqueueContent, priority: ...) -> PendingMessage | None:
        ...
```

`from_content` returns `None` for an empty call (enqueueing nothing is a no-op). It raises
`UserError` if the assembled messages do not end in a `ModelRequest`.

### `RunContext.enqueue` — inject from inside a tool

```python
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

@agent.tool
async def long_running_task(ctx: RunContext[None], task_name: str) -> str:
    """Start a task and send an async update mid-run."""
    # Inject a progress message that will be delivered ASAP
    ctx.enqueue(f"Task '{task_name}' started — will update when complete.")
    # ... do async work ...
    return f"Task '{task_name}' finished."
```

### `AgentRun.enqueue` — inject from outside

`AgentRun.enqueue` has the same signature as `RunContext.enqueue` and is useful for pushing
messages from a background coroutine or event handler while the run is in progress.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import SystemPromptPart

agent = Agent('openai:gpt-4o')

async def push_update(agent_run, text: str) -> None:
    """Push an urgent system-prompt update into the next model request."""
    agent_run.enqueue(SystemPromptPart(content=text), priority='asap')

async def main():
    async with agent.iter('Monitor prices.') as agent_run:
        # Spawn a background task that will inject a message while the run is live
        asyncio.create_task(push_update(agent_run, 'ALERT: Bitcoin just hit $200k'))
        async for node in agent_run:
            pass  # drive normally; use agent_run.next(node) in production
    print(agent_run.result.output)
```

<Aside type="caution">
`AgentRun.enqueue` is not thread-safe. If calling from a different thread (e.g. a webhook
handler on its own event loop), marshal back: `loop.call_soon_threadsafe(agent_run.enqueue, msg)`.
</Aside>

### `'when_idle'` priority — follow-up turns without interruption

```python
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

agent = Agent('openai:gpt-4o')

@agent.tool
async def analyse_data(ctx: RunContext[None], dataset: str) -> dict:
    """Analyse a dataset and queue a follow-up question for when the agent is idle."""
    result = {'rows': 1000, 'nulls': 5}
    # This follow-up is only delivered once the agent would naturally finish
    ctx.enqueue(
        'Now summarise the findings in a tweet-length sentence.',
        priority='when_idle',
    )
    return result
```

### How the drain works

The `PendingMessageDrainCapability` is auto-injected (no user action required) and placed at
the `'outermost'` position in the capability ordering:

1. **`before_model_request`** — drains all `'asap'` messages into the upcoming `ModelRequest`,
   appending them to both `request_context.messages` and `ctx.messages`.
2. **`after_node_run`** — if the run is about to end (`End` result), drains remaining `'asap'`
   messages first, then `'when_idle'` messages. The last drained `ModelRequest` becomes a new
   `ModelRequestNode`, redirecting the run for one more turn instead of terminating.

---

## 2. `AgentWorker` + `agent_to_a2a`

**Module:** `pydantic_ai._a2a`  
**Import:** `from pydantic_ai._a2a import AgentWorker, agent_to_a2a`  
**Extra required:** `pip install "pydantic-ai-slim[a2a]"` (pulls `fasta2a`)

<Aside type="caution">
`agent_to_a2a()` is **deprecated** as of 1.104.0 and will be removed in v2.0. The `fasta2a`
package is now maintained independently. Migrate to
`pip install "fasta2a[pydantic-ai]>=0.6.1"` and use
`from fasta2a.pydantic_ai import agent_to_a2a`.
</Aside>

`AgentWorker` bridges a pydantic-ai `Agent` to the Agent-to-Agent (A2A) protocol, converting
A2A task messages into pydantic-ai `ModelMessage` history and packaging the agent's output as
A2A `Artifact` objects.

### `AgentWorker` class

```python
@dataclass
class AgentWorker(Worker[list[ModelMessage]], Generic[WorkerOutputT, AgentDepsT]):
    agent: AbstractAgent[AgentDepsT, WorkerOutputT]

    async def run_task(self, params: TaskSendParams) -> None: ...
    async def cancel_task(self, params: TaskIdParams) -> None: ...
    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]: ...
    def build_message_history(self, history: list[Message]) -> list[ModelMessage]: ...
```

`run_task` lifecycle:
1. Load the task from storage; reject if not in `'submitted'` state.
2. Load prior A2A context (pydantic-ai message history) from `storage.load_context`.
3. Convert incoming A2A `Message` objects to `ModelRequest`/`ModelResponse` via
   `build_message_history`.
4. Call `agent.run(message_history=...)`.
5. Persist the updated `all_messages()` back to the context store.
6. Convert new `ModelResponse` parts to A2A `Part` objects (text and thinking are exposed;
   tool calls are hidden as internal).
7. Package `result.output` as an `Artifact` (string → `TextPart`, structured → `DataPart`
   with JSON schema in `metadata`).

### Content type mappings

| pydantic-ai | A2A protocol |
|---|---|
| `TextPart` | `TextPart(kind='text', text=...)` |
| `ThinkingPart` | `TextPart(kind='text', metadata={'type': 'thinking', ...})` |
| `ToolCallPart` | skipped (internal) |
| `UserPromptPart` (str) | `TextPart(kind='text', text=...)` |
| `BinaryContent` (bytes) | `FilePart(kind='file', bytes=base64, mime_type=...)` |
| `ImageUrl`/`AudioUrl`/… | `FilePart(kind='file', uri=url)` — matched by MIME type |
| `str` output | `TextPart` artifact |
| Pydantic model output | `DataPart(data={'result': ...}, metadata={'json_schema': ...})` |

### Example — wrap an agent as an A2A server (new `fasta2a` API)

```python
# Recommended post-deprecation approach
# pip install "fasta2a[pydantic-ai]>=0.6.1"

from pydantic_ai import Agent
from fasta2a.pydantic_ai import agent_to_a2a  # new home

agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

app = agent_to_a2a(
    agent,
    name='my-agent',
    url='http://localhost:8000',
    version='1.0.0',
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### Example — typed structured output via A2A

```python
from pydantic import BaseModel
from pydantic_ai import Agent
# After migration:
from fasta2a.pydantic_ai import agent_to_a2a

class ReportSummary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

agent = Agent('openai:gpt-4o', output_type=ReportSummary)

# Structured output is packaged as DataPart with full JSON schema in metadata
app = agent_to_a2a(agent, name='report-summariser', url='http://localhost:8001')
```

---

## 3. `WrapperAgent`

**Module:** `pydantic_ai.agent.wrapper`  
**Import:** `from pydantic_ai.agent.wrapper import WrapperAgent`

`WrapperAgent` is a transparent delegation base for building agent middleware. It stores a
`wrapped: AbstractAgent` and forwards every property and method call to it. Subclassing
`WrapperAgent` lets you intercept any aspect of agent execution while keeping the rest intact.

### Class interface

```python
class WrapperAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT]):
        self.wrapped = wrapped

    # Forwarded properties (all read from self.wrapped):
    # model, name, description, deps_type, output_type,
    # event_stream_handler, root_capability, toolsets

    # Forwarded methods:
    # iter(), override(), system_prompt_parts(), output_json_schema(),
    # __aenter__(), __aexit__()
```

The `name` and `description` properties have setters that write through to `self.wrapped`.

### Example 1 — logging wrapper that records every run

```python
import time
from pydantic_ai import Agent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.output import OutputDataT
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

class TimingAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """Records wall-clock time for every agent run."""

    @asynccontextmanager
    async def iter(self, user_prompt=None, **kwargs) -> AsyncIterator:
        start = time.perf_counter()
        try:
            async with self.wrapped.iter(user_prompt, **kwargs) as run:
                yield run
        finally:
            elapsed = time.perf_counter() - start
            print(f'[TimingAgent] run completed in {elapsed:.3f}s')

base_agent = Agent('openai:gpt-4o')
timed = TimingAgent(base_agent)
result = timed.run_sync('What is 2 + 2?')
```

### Example 2 — dependency-injecting wrapper

```python
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.agent.wrapper import WrapperAgent
from contextlib import asynccontextmanager

@dataclass
class AppDeps:
    user_id: str
    tenant: str

class TenantAgent(WrapperAgent):
    """Auto-injects a tenant ID into every run."""

    def __init__(self, wrapped, tenant: str):
        super().__init__(wrapped)
        self._tenant = tenant

    @asynccontextmanager
    async def iter(self, user_prompt=None, *, deps=None, **kwargs):
        # Override the deps with tenant info before forwarding
        if deps is None:
            deps = AppDeps(user_id='anon', tenant=self._tenant)
        else:
            deps = AppDeps(user_id=deps.user_id, tenant=self._tenant)
        async with self.wrapped.iter(user_prompt, deps=deps, **kwargs) as run:
            yield run

base = Agent('openai:gpt-4o')
acme_agent = TenantAgent(base, tenant='acme-corp')
```

### `WrapperAgent` in the DBOS durable execution stack

`DBOSAgent` (see §5) extends `WrapperAgent` and replaces `model`, `toolsets`, and the
`iter()` context-manager with DBOS-step-backed equivalents, while all other properties
continue to forward to the inner `AbstractAgent`. This is the intended use pattern:
override only what needs durability, inherit the rest.

---

## 4. `safe_download` + SSRF protection

**Module:** `pydantic_ai._ssrf`  
**Import:** `from pydantic_ai._ssrf import safe_download, is_private_ip, is_cloud_metadata_ip`

`safe_download` is the internal function used by `WebFetch`, `WebSearch`, and any other
built-in tool that fetches URLs. It provides production-grade SSRF (Server-Side Request
Forgery) protection through a multi-layered defence strategy.

### `ResolvedUrl` — the validated URL descriptor

```python
@dataclass
class ResolvedUrl:
    resolved_ip: str   # IP address to actually connect to
    hostname: str      # original hostname (used for Host header and TLS SNI)
    port: int
    is_https: bool
    path: str          # includes query string and fragment
```

### Defence layers (in order)

1. **Protocol allowlist** — only `http://` and `https://`. Any other scheme (`file://`,
   `ftp://`, `data:`) raises immediately.
2. **DNS resolution** — hostname resolved with `socket.getaddrinfo` (AF_UNSPEC so both
   IPv4 and IPv6 are returned). Runs in a thread pool to avoid blocking the event loop.
3. **Cloud metadata blocklist** — always blocked, even with `allow_local=True`:

   | IP | Service |
   |---|---|
   | `169.254.169.254` | AWS IMDS / GCP / Azure / OCI / DigitalOcean / Hetzner |
   | `169.254.170.2` | AWS ECS task IAM role credentials |
   | `169.254.170.23` | AWS EKS Pod Identity Agent |
   | `168.63.129.16` | Azure WireServer (public IP — extra guard needed) |
   | `100.100.100.200` | Alibaba Cloud |
   | `192.0.0.192` | Oracle Cloud (Classic) |
   | `169.254.42.42` | Scaleway |
   | `fd00:ec2::254` | AWS IMDS IPv6 |
   | `fd00:ec2::23` | AWS EKS Pod Identity Agent IPv6 |

4. **Private range blocklist** — 20+ RFC-defined private, loopback, link-local, and
   reserved ranges (unless `allow_local=True`). IPv6 transition forms are decoded:
   IPv4-mapped `::ffff:a.b.c.d`, 6to4 `2002::/16`, NAT64 `64:ff9b::/96`, ISATAP, and Teredo.
5. **Domain allow/block lists** — optional exact-match per-hop domain filtering.
6. **Request via resolved IP** — the actual HTTP request is made to the IP address, with
   the `Host` header set to the original hostname. For HTTPS, `sni_hostname` is set so TLS
   uses the correct certificate.
7. **Manual redirect following** — up to 10 hops (configurable). Sensitive headers
   (`Authorization`, `Cookie`, `Proxy-Authorization`) are stripped on cross-origin redirects
   (RFC 7235). Each hop is independently validated.

### Usage example — safe URL download with domain restrictions

```python
import asyncio
from pydantic_ai._ssrf import safe_download

async def fetch_doc(url: str) -> str:
    response = await safe_download(
        url,
        allow_local=False,           # block private IPs (cloud metadata always blocked)
        max_redirects=5,
        timeout=10,
        headers={'Accept': 'text/html'},
        allowed_domains=['docs.example.com', 'api.example.com'],
    )
    return response.text

# Calling with a private IP raises ValueError:
# await safe_download('http://169.254.169.254/metadata')
# → ValueError: Access to cloud metadata service (169.254.169.254) is blocked.
```

### IP validation helpers

```python
from pydantic_ai._ssrf import is_private_ip, is_cloud_metadata_ip

is_private_ip('10.0.0.1')          # True
is_private_ip('8.8.8.8')           # False
is_private_ip('::ffff:192.168.1.1') # True (IPv4-mapped IPv6 decoded)

is_cloud_metadata_ip('169.254.169.254')  # True
is_cloud_metadata_ip('168.63.129.16')    # True (Azure — public IP, still blocked)
```

### IPv6 transition form decoding

The cloud-metadata guard uses `_embedded_ipv4s(ip, exhaustive=True)` to decode every
standardised IPv4-in-IPv6 encoding. For instance, the Teredo-encoded AWS metadata endpoint
`2001::XXXX` (low-32 bits XOR `0xFFFFFFFF`) maps to `169.254.169.254` and is blocked.
Private-range checking uses `exhaustive=False` (only well-recognised forms) to avoid
mis-classifying legitimate public IPv6 addresses whose bytes coincide with a private range.

---

## 5. `DBOSAgent`

**Module:** `pydantic_ai.durable_exec.dbos._agent`  
**Import:** `from pydantic_ai.durable_exec.dbos import DBOSAgent`  
**Extra required:** `pip install "pydantic-ai[dbos]"` (pulls `dbos`)

`DBOSAgent` wraps any `AbstractAgent` to run model requests, tool calls, and MCP server
interactions as **DBOS durable workflow steps** — automatically checkpointed and resumable
after crashes or restarts.

### Constructor

```python
@DBOS.dbos_class()
class DBOSAgent(WrapperAgent[AgentDepsT, OutputDataT], DBOSConfiguredInstance):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,             # required (DBOS workflow ID prefix)
        event_stream_handler: ... = None,
        mcp_step_config: StepConfig | None = None,
        model_step_config: StepConfig | None = None,
        parallel_execution_mode: DBOSParallelExecutionMode = 'parallel_ordered_events',
    ): ...
```

**`name`** is required — it becomes the DBOS configured instance name and prefixes all
workflow/step names. The inner `wrapped.model` must be set at construction time (not at
run time), because `DBOSAgent` replaces it with a `DBOSModel` wrapper.

### `DBOSParallelExecutionMode`

```python
DBOSParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
```

| Mode | Behaviour |
|---|---|
| `'parallel_ordered_events'` (default) | Tool calls run in parallel; events are emitted in order after all calls complete. Deterministic replay. |
| `'sequential'` | Tool calls run one at a time in order. |

Note: `'parallel'` (fully parallel events) is excluded from DBOS because it cannot guarantee
deterministic event ordering across workflow replays.

### Automatic toolset wrapping

When you pass toolsets to the inner agent, `DBOSAgent` automatically replaces:
- `MCPToolset` → `DBOSMCPToolset` (wraps each MCP server call in a DBOS step)
- `MCPServer` (legacy) → `DBOSMCPServer`
- `FastMCPToolset` → `DBOSFastMCPToolset`

All other toolsets are passed through unchanged.

### `StepConfig` — tune DBOS step behaviour

```python
from pydantic_ai.durable_exec.dbos._utils import StepConfig

step_config: StepConfig = {
    'retries': 3,
    'timeout_seconds': 30,
    'backoff_seconds': 2.0,
}
```

Pass separate configs for model steps and MCP steps:

```python
agent = DBOSAgent(
    inner_agent,
    name='research-agent',
    model_step_config={'retries': 5, 'timeout_seconds': 60},
    mcp_step_config={'retries': 3, 'timeout_seconds': 30},
)
```

### Example — full durable agent setup

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent
from dbos import DBOS

# Must be initialised before creating DBOSAgent
dbos = DBOS(config={'database_url': 'postgresql://user:pass@localhost/dbos'})

inner = Agent('openai:gpt-4o', name='researcher')

durable_agent = DBOSAgent(
    inner,
    name='researcher',         # required; unique per DBOS app
    parallel_execution_mode='parallel_ordered_events',
    model_step_config={'retries': 3},
)

# Use exactly like a normal agent
result = durable_agent.run_sync('Summarise the latest AI research.')
print(result.output)
```

---

## 6. `StreamedResponseSync`

**Module:** `pydantic_ai.direct`  
**Import:** `from pydantic_ai.direct import StreamedResponseSync, model_request_stream_sync`

`StreamedResponseSync` is a synchronous context manager wrapping an async streaming response.
It runs the async producer in a background `threading.Thread` and bridges events to the calling
thread via a `queue.Queue`. This is the streaming counterpart to `model_request_sync` and is
designed for CLI tools, Jupyter notebooks, and any synchronous context that cannot use `await`.

### Internal architecture

```
calling thread          background thread
─────────────           ──────────────────────────────────
with sync_stream:  ──►  _start_producer() → threading.Thread(_async_producer)
                        _async_producer() runs event loop:
                            async with async_cm as stream:
                                self._stream_response = stream
                                self._stream_ready.set()  ◄── signals ready
                                async for event in stream:
                                    queue.put(event)
                        queue.put(None)  # sentinel

for event in sync_stream:
    queue.get()    ◄── blocks until event or sentinel
```

### Constructor (used internally by `model_request_stream_sync`)

```python
@dataclass
class StreamedResponseSync:
    _async_stream_cm: AbstractAsyncContextManager[StreamedResponse]
    # internal fields created by __post_init__:
    # _queue, _thread, _stream_response, _stream_ready
```

### Properties

| Property | Type | Description |
|---|---|---|
| `response` | `ModelResponse` | Current state of the assembled response |
| `model_name` | `str` | Model name from the stream |
| `timestamp` | `datetime` | Timestamp of the response |

### Example 1 — streaming from CLI code

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync

messages = [ModelRequest.user_text_prompt('Explain quantum entanglement simply.')]

with model_request_stream_sync('anthropic:claude-haiku-4-5', messages) as stream:
    for event in stream:
        from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end='', flush=True)
    print()  # newline at end
    print(f'Model: {stream.model_name}')
```

### Example 2 — progressive Jupyter output

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, PartEndEvent
from IPython.display import display, Markdown
import ipywidgets as widgets

output = widgets.Output()
display(output)

messages = [ModelRequest.user_text_prompt('Write a haiku about pydantic.')]
with model_request_stream_sync('openai:gpt-4o-mini', messages) as stream:
    text = ''
    for event in stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            text += event.delta.content_delta
            with output:
                output.clear_output(wait=True)
                display(Markdown(text))
```

### Error propagation

Exceptions raised inside the async producer are put onto the queue as `Exception` instances
and re-raised from the `for event in stream` iteration:

```python
try:
    with model_request_stream_sync('openai:gpt-4o', messages) as stream:
        for event in stream:
            ...
except Exception as e:
    print(f'Stream failed: {e}')
```

The `_stream_ready` `threading.Event` has a 30-second timeout (`STREAM_INITIALIZATION_TIMEOUT`).
If the async stream fails to initialise within that window, `RuntimeError` is raised.

---

## 7. `ModelResponsePartsManager`

**Module:** `pydantic_ai._parts_manager`  
**Import:** `from pydantic_ai._parts_manager import ModelResponsePartsManager`

`ModelResponsePartsManager` is the internal engine behind every streamed `ModelResponse`. Each
`StreamedResponse` subclass (OpenAI, Anthropic, Gemini, …) calls methods on a manager instance
to accumulate vendor-specific delta chunks into structured `ModelResponsePart` events. You
rarely use this class directly, but understanding it explains *why* streaming events are shaped
the way they are.

### Fields

```python
@dataclass
class ModelResponsePartsManager:
    model_request_parameters: ModelRequestParameters
    # private:
    _parts: list[ManagedPart]                     # TextPart | ThinkingPart | ToolCallPart | ToolCallPartDelta
    _vendor_id_to_part_index: dict[VendorId, int] # maps provider part ID → index in _parts
    _tool_kind_by_name: dict[str, ToolPartKind]   # {tool_name: kind} from function_tools defs
```

`VendorId = Hashable` — can be an `int` (OpenAI streaming index), a `str` (Anthropic content
block ID), or any other hashable the model uses to track part identity across delta chunks.

### Core methods

| Method | Purpose |
|---|---|
| `handle_text_delta(...)` | Append or extend a `TextPart`; emit `PartStartEvent` or `PartDeltaEvent` |
| `handle_thinking_delta(...)` | Same for `ThinkingPart`; handles embedded `<think>` tags |
| `handle_tool_call_delta(...)` | Create or extend a `ToolCallPart`/`ToolCallPartDelta` |
| `handle_tool_call_part(...)` | Directly set (overwrite) a fully-formed `ToolCallPart` |
| `handle_part(...)` | Create or overwrite any `ModelResponsePart` |
| `get_parts()` | Return only fully-formed `ModelResponsePart` objects (no deltas) |

### Tool kind narrowing

The manager promotes `ToolCallPart` to a typed subclass at the first event, not after the
stream completes. The `_tool_kind_by_name` cache is built from `model_request_parameters.function_tools`
at construction time:

```python
self._tool_kind_by_name = {
    td.name: td.tool_kind
    for td in self.model_request_parameters.function_tools
    if td.tool_kind is not None
}
```

This means `isinstance(part, ToolSearchCallPart)` is true from the `PartStartEvent` itself,
letting streaming consumers branch immediately without waiting for `PartEndEvent`.

### Vendor ID semantics

- **`vendor_part_id=None`** and a `tool_name` → always create a new `ToolCallPart` (Anthropic uses
  content block IDs, not positional indices, so `None` means "no existing block to update").
- **`vendor_part_id=None`** and no `tool_name` → update the *latest* part of matching type
  (OpenAI-style positional streaming where the active part is implicit).
- **`vendor_part_id=<value>`** → look up the existing part by ID, update if found; create if not.

### Embedded thinking (`<think>` tags)

Some models stream thinking content inside `<think>...</think>` HTML tags within a `TextPart`
delta. The manager detects the open/close tags and transparently splits the stream:
1. On `<think>` — stop tracking the current `TextPart` vendor ID; create a new `ThinkingPart`.
2. On content inside `<think>` — append `ThinkingPartDelta` events.
3. On `</think>` — stop tracking the `ThinkingPart` vendor ID so the next text delta creates
   a fresh `TextPart`.

### Example — custom streaming model that uses `ModelResponsePartsManager`

```python
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.messages import ModelResponseStreamEvent

class MyStreamedResponse(StreamedResponse):
    """Skeleton showing how a model adapter uses the parts manager."""

    def __init__(self, request_params: ModelRequestParameters, vendor_stream):
        super().__init__(
            _stream=self._process(vendor_stream),
            _model_name='my-model',
        )
        self._manager = ModelResponsePartsManager(request_params)

    async def _process(self, vendor_stream) -> None:
        async for chunk in vendor_stream:
            if chunk.type == 'text_delta':
                event = self._manager.handle_text_delta(
                    vendor_part_id=chunk.index,
                    content=chunk.text,
                )
            elif chunk.type == 'tool_call':
                event = self._manager.handle_tool_call_delta(
                    vendor_part_id=chunk.index,
                    tool_name=chunk.name,
                    args=chunk.args_delta,
                    tool_call_id=chunk.id,
                )
            else:
                continue
            if event is not None:
                yield event
```

---

## 8. `CombinedCapability`

**Module:** `pydantic_ai.capabilities.combined`  
**Import:** `from pydantic_ai.capabilities import CombinedCapability`

`CombinedCapability` is the composition engine that combines multiple `AbstractCapability`
objects into a single capability object. It is constructed automatically whenever you pass a
list to the `capabilities=` parameter of `Agent` or `Agent.iter()`. Understanding it explains
why hook ordering works the way it does.

### Construction and flattening

```python
@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    capabilities: Sequence[AbstractCapability[AgentDepsT]]
```

`__post_init__` flattens nested `CombinedCapability` instances so their leaves become siblings
in the outer ordering pass. Without this, a nested combination whose leaves span both
`'outermost'` and `'innermost'` tiers would conflict.

After flattening, if any leaf declares a `CapabilityOrdering`, the list is sorted via
`sort_capabilities`. `PendingMessageDrainCapability` (always auto-injected) is placed
`'outermost'`; custom capabilities without an ordering go in the middle.

### Hook direction — forward vs reverse

The direction varies by hook phase to implement a correct middleware stack:

| Hook phase | Direction | Reason |
|---|---|---|
| `before_*` / `prepare_*` | **forward** (capabilities[0] first) | Each layer pre-processes before the next |
| `after_*` / `on_*_error` | **reverse** (capabilities[-1] first) | Each layer post-processes in reverse order |
| `wrap_*` | Closure chain built in **reverse**; innermost capability's `wrap_*` calls the actual handler | Standard middleware onion pattern |

### `get_model_settings` merging

```python
def get_model_settings(self) -> ...:
    # Collect each capability's settings in order
    # If all static → merge eagerly with merge_model_settings
    # If any dynamic → build a resolver closure that:
    #   1. For each entry, updates ctx.model_settings with accumulated merged settings
    #   2. Calls dynamic entries with the updated ctx
    #   3. Merges the result into the running total
```

This means dynamic `ModelSettings` callables see the accumulated settings from all prior
capabilities in `ctx.model_settings` when they are invoked.

### `handle_deferred_tool_calls` — first-come-first-served

```python
async def handle_deferred_tool_calls(self, ctx, *, requests):
    accumulated = DeferredToolResults()
    remaining = requests
    for capability in self.capabilities:
        result = await capability.handle_deferred_tool_calls(ctx, requests=remaining)
        if result:
            accumulated.update(result)
            remaining = remaining.remaining(result) or break
    return accumulated if any_handled else None
```

Capabilities process deferred tool calls in order. Each capability only sees the requests
not already handled by a prior capability. The loop short-circuits when all requests are
resolved.

### Example — building a capability stack explicitly

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.capabilities.thinking import Thinking
from pydantic_ai.capabilities.prefix_tools import PrefixTools

hooks = Hooks()

@hooks.before_run
async def log_start(ctx):
    print(f'Run starting: {ctx.run_id}')

@hooks.after_run
async def log_end(ctx, result):
    print(f'Run finished: {result.output!r}')
    return result

combo = CombinedCapability([
    hooks,
    Thinking(effort='low'),
    PrefixTools('v1_'),
])

agent = Agent('openai:gpt-4o', capabilities=[combo])
# Equivalent to:
agent2 = Agent('openai:gpt-4o', capabilities=[hooks, Thinking(effort='low'), PrefixTools('v1_')])
```

### Example — dynamic `ModelSettings` via `CombinedCapability`

```python
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

class TenantModelSettings(AbstractCapability):
    """Adjust temperature and max_tokens based on tenant tier."""

    def get_model_settings(self):
        def resolve(ctx: RunContext) -> ModelSettings:
            tier = (ctx.deps or {}).get('tier', 'free')
            if tier == 'enterprise':
                return ModelSettings(temperature=0.7, max_tokens=4096)
            return ModelSettings(temperature=0.3, max_tokens=512)
        return resolve

from pydantic_ai.capabilities.thinking import Thinking

# CombinedCapability merges settings from TenantModelSettings
# then applies Thinking's own settings on top
agent = Agent('openai:gpt-4o', capabilities=[TenantModelSettings(), Thinking(effort='medium')])
```

---

## 9. `FastMCPToolset` (deprecated — migration guide)

**Module:** `pydantic_ai.toolsets.fastmcp`  
**Import:** `from pydantic_ai.toolsets.fastmcp import FastMCPToolset`

<Aside type="caution">
`FastMCPToolset` is **deprecated** as of 1.104.0 and will be removed in v2.0. Use
`pydantic_ai.mcp.MCPToolset` instead — it accepts the same input shapes (including a
pre-built `fastmcp.Client`), adds caching, OAuth auth, resource methods, sampling shortcuts,
and `process_tool_call` parity.
</Aside>

`FastMCPToolset` is a `AbstractToolset` backed by a FastMCP `Client`. It is still functional
in 1.104.0 and is documented here for teams migrating existing code.

### Constructor

```python
@deprecated(...)
@dataclass(init=False)
class FastMCPToolset(AbstractToolset[AgentDepsT]):
    def __init__(
        self,
        client: Client | ClientTransport | FastMCP | FastMCP1Server
               | AnyUrl | Path | MCPConfig | dict | str,
        *,
        max_retries: int | None = None,
        tool_error_behavior: Literal['model_retry', 'error'] = 'model_retry',
        include_instructions: bool = False,
        include_return_schema: bool | None = None,
        id: str | None = None,
        process_tool_call: ProcessToolCallback | None = None,
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `client` | — | FastMCP `Client`, `ClientTransport`, `FastMCP` server, URL, path, or config |
| `max_retries` | `None` (uses agent default) | Max retries per tool call |
| `tool_error_behavior` | `'model_retry'` | On `ToolError`: retry via `ModelRetry` or re-raise |
| `include_instructions` | `False` | Expose server instructions as `InstructionPart` |
| `include_return_schema` | `None` | Include tool output schema in tool definitions |
| `id` | `None` | Unique toolset ID (for deferred-tool identification) |
| `process_tool_call` | `None` | Hook: `async (ctx, direct_call, name, args) -> Any` |

### Lifecycle

`FastMCPToolset` uses a reference-counted `AsyncExitStack` and an `anyio.Lock` (created lazily
as a `cached_property` to bind to the correct event loop):

```
__aenter__:
    acquire _enter_lock
    if _running_count == 0:
        enter async context on self.client
        read self._instructions from initialize_result
    _running_count += 1

__aexit__:
    acquire _enter_lock
    _running_count -= 1
    if _running_count == 0: close exit_stack
```

The same toolset instance can be shared across multiple concurrent `Agent.iter()` calls —
the lock ensures the client is only entered once.

### Structured content handling

When the MCP server returns structured JSON content alongside text, `FastMCPToolset` prefers
the structured form:

```python
# MCP SDK wraps primitives in {'result': value} for backward compat
# FastMCPToolset unwraps the 'result' key for single-key dicts:
if isinstance(structured, dict) and len(structured) == 1 and 'result' in structured:
    return structured['result']
return structured
```

### Migration: `FastMCPToolset` → `MCPToolset`

```python
# BEFORE (deprecated)
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

toolset = FastMCPToolset(
    'http://localhost:8000/sse',
    max_retries=3,
    include_instructions=True,
    process_tool_call=my_callback,
)

# AFTER (recommended)
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(
    'http://localhost:8000/sse',
    max_retries=3,
    include_instructions=True,
    process_tool_call=my_callback,
    # Additional MCPToolset-only features:
    # cache_tools_list=True,       # cache the tool list between requests
    # auth=OAuthParams(...),       # OAuth 2.0 support
)
```

Both classes accept the same first positional argument forms (`Client`, URL string, path, etc.).
`MCPToolset` also includes:
- **Tool list caching** (`cache_tools_list=True`) — reuses `list_tools()` between requests.
- **OAuth 2.0** — via the `auth=` parameter.
- **Resource access** — `read_resource()` and `list_resources()` methods.
- **Sampling shortcuts** — `create_message()` support.

---

## 10. `SetToolMetadata`

**Module:** `pydantic_ai.capabilities.set_tool_metadata`  
**Import:** `from pydantic_ai.capabilities import SetToolMetadata`

`SetToolMetadata` is a capability that merges arbitrary key-value metadata onto selected
tools' `ToolDefinition.metadata` dicts before each model request. Its most common use is
enabling **Code Mode** (`code_mode=True`) for all or specific tools.

### Constructor

```python
@dataclass
class SetToolMetadata(AbstractCapability[AgentDepsT]):
    tools: ToolSelector[AgentDepsT] = 'all'
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *, tools: ToolSelector[AgentDepsT] = 'all', **metadata: Any):
        self.tools = tools
        self.metadata = metadata
```

`**metadata` kwargs become the metadata dict. This lets you write:

```python
SetToolMetadata(code_mode=True)
# equivalent to: SetToolMetadata(metadata={'code_mode': True})
```

### `ToolSelector` — which tools to target

`ToolSelector[AgentDepsT]` can be:
- `'all'` — apply to every tool (default)
- `str` — exact tool name match
- `list[str]` — whitelist of tool names
- `async callable(ctx, tool_def) -> bool` — dynamic per-tool filter

### How it works

`SetToolMetadata.get_wrapper_toolset` wraps the incoming toolset in a `PreparedToolset` with a
`_set_metadata` async prepare function:

```python
async def _set_metadata(ctx, tool_defs):
    for td in tool_defs:
        if await matches_tool_selector(selector, ctx, td):
            td = replace(td, metadata={**(td.metadata or {}), **metadata})
        resolved.append(td)
    return resolved
```

Existing metadata keys are preserved; only the specified keys are set (shallow merge).

### Example 1 — Code Mode for all tools

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    'openai:gpt-4o',
    capabilities=[SetToolMetadata(code_mode=True)],
)
```

All tools exposed to this agent will have `metadata['code_mode'] = True`, signalling that
arguments should be passed as code rather than JSON.

### Example 2 — Code Mode for specific tools only

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        SetToolMetadata(tools=['run_sql', 'execute_python'], code_mode=True),
    ],
)
```

### Example 3 — Multiple metadata fields with dynamic targeting

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.tools import RunContext, ToolDefinition

async def is_expensive_tool(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    """Only mark tools that are tagged as expensive."""
    return (tool_def.metadata or {}).get('cost', 'low') == 'high'

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        SetToolMetadata(
            tools=is_expensive_tool,
            rate_limited=True,
            priority='low',
        ),
    ],
)
```

### Example 4 — Per-run metadata injection

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent('openai:gpt-4o')

async def run_with_tracing(user_prompt: str, trace_id: str):
    """Attach a trace_id to all tool calls for this run."""
    result = await agent.run(
        user_prompt,
        capabilities=[SetToolMetadata(trace_id=trace_id, env='production')],
    )
    return result
```

---

## Quick reference

| Class | Module | Key use case |
|---|---|---|
| `PendingMessage` | `pydantic_ai._enqueue` | Message injection payload; `from_content()` builder |
| `RunContext.enqueue` | `pydantic_ai.tools` | Inject messages from inside a tool |
| `AgentRun.enqueue` | `pydantic_ai.run` | Inject messages from outside during `agent.iter()` |
| `AgentWorker` | `pydantic_ai._a2a` | Bridge agent → A2A task worker |
| `agent_to_a2a` | `pydantic_ai._a2a` | Deprecated wrapper; migrate to `fasta2a.pydantic_ai` |
| `WrapperAgent` | `pydantic_ai.agent.wrapper` | Base class for agent middleware |
| `safe_download` | `pydantic_ai._ssrf` | SSRF-safe HTTP fetch with cloud metadata blocking |
| `ResolvedUrl` | `pydantic_ai._ssrf` | Validated URL descriptor after DNS resolution |
| `DBOSAgent` | `pydantic_ai.durable_exec.dbos` | Durable agent with DBOS workflow steps |
| `StreamedResponseSync` | `pydantic_ai.direct` | Synchronous streaming for CLI / notebooks |
| `ModelResponsePartsManager` | `pydantic_ai._parts_manager` | Streaming event accumulator (internal) |
| `CombinedCapability` | `pydantic_ai.capabilities.combined` | Multi-capability combinator |
| `FastMCPToolset` | `pydantic_ai.toolsets.fastmcp` | Deprecated; migrate to `MCPToolset` |
| `SetToolMetadata` | `pydantic_ai.capabilities.set_tool_metadata` | Inject metadata onto tool definitions |

---

*Verified against pydantic-ai 1.104.0 installed from PyPI. Source files inspected:
`_enqueue.py`, `capabilities/_pending_messages.py`, `run.py`, `_a2a.py`, `agent/wrapper.py`,
`_ssrf.py`, `durable_exec/dbos/_agent.py`, `direct.py`, `_parts_manager.py`,
`capabilities/combined.py`, `toolsets/fastmcp.py`, `capabilities/set_tool_metadata.py`.*

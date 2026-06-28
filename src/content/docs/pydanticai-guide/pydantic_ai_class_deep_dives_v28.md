---
title: "PydanticAI Class Deep Dives Vol. 28"
description: "Source-verified deep dives into 10 pydantic-ai 2.0.0 class groups: ToolSearchArgs + ToolSearchMatch + ToolSearchReturnContent (typed shapes for the tool-search wire format shared by native and local paths), NativeToolSearchCallPart + NativeToolSearchReturnPart (server-side typed message parts — typed_args property, queries accessor, _TYPED_PART_TAGS discriminator wiring), ToolSearchCallPart + ToolSearchReturnPart + synthesize_local_tool_search_messages (local-fallback typed parts and cross-provider history translation — flush-boundary splitting, identity-level metadata preservation), ResolvedUrl + safe_download (SSRF-protected URL download — _CLOUD_METADATA_IPV4 blocklist, IPv6 transition decoding, per-hop sensitive-header stripping, _MAX_REDIRECTS=10 guard), PendingMessage + PendingMessagePriority + _build_enqueue_messages (runtime pending message queue — asap vs when_idle priorities, UserContent coalescing, must-end-in-ModelRequest invariant), CombinedCapability (capability tree flattener — __post_init__ splat prevents conflicting positions from nested CombinedCapability, collect_leaves + sort_capabilities ordering), ProcessEventStream (dual-mode event stream capability — observer vs processor probe, anyio memory object stream tee, back-pressure semantics, durable-execution caveats), HandleDeferredToolCalls (inline deferred-tool resolver — approve_all pattern, sync/async handler, None-to-decline protocol), Thinking (unified multi-provider thinking control — ThinkingLevel, ModelSettings(thinking=...) integration, always-on model guard), ThreadExecutor + ReinjectSystemPrompt (bounded thread pool executor for sustained-load servers, system prompt reinjection with replace_existing for untrusted history). All verified against pydantic-ai 2.0.0 source."
sidebar:
  label: "Class deep dives (Vol. 28)"
  order: 54
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.0.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.x API.
</Aside>

Ten class groups covering pydantic-ai 2.0.0's tool-search message-part hierarchy, SSRF-protected URL downloads, the pending message queue, the capabilities subsystem internals, and cross-cutting utilities for thread-pool management and system-prompt reinjection.

---

## 1. `ToolSearchArgs` + `ToolSearchMatch` + `ToolSearchReturnContent` — Typed Shapes for the Tool-Search Wire Format

**Source**: `pydantic_ai/_tool_search.py`

These three `TypedDict`s form the canonical cross-provider data contract for every tool-search exchange. They are carried on both the native server-side parts (Anthropic BM25/regex, OpenAI Responses) and the local-fallback parts, so downstream code that inspects `args` or `content` can work uniformly regardless of which provider ran the search.

```python
# Key shapes verified from source:

class ToolSearchMatch(TypedDict):
    name: str          # Tool name as the model calls it
    description: str | None

class ToolSearchArgs(TypedDict):
    queries: list[str]
    # Anthropic BM25/regex: single-item list with the query string
    # OpenAI server-executed tool_search: list of tool paths the model picked
    # OpenAI client-execution / local fallback: single-item list with keywords

class ToolSearchReturnContent(TypedDict):
    discovered_tools: list[ToolSearchMatch]   # ordered by relevance; empty = "ran, nothing matched"
    message: NotRequired[str]                 # shown to model when no matches; omitted on non-empty
```

```python
# Example 1 — introspecting a tool-search return regardless of provider path
from pydantic_ai.messages import NativeToolSearchReturnPart, ToolSearchReturnPart

def log_search_results(part: NativeToolSearchReturnPart | ToolSearchReturnPart) -> None:
    """Works for both server-side (Anthropic/OpenAI) and local-fallback results."""
    content = part.content          # always ToolSearchReturnContent
    tools = content['discovered_tools']
    if not tools:
        msg = content.get('message', 'No matches')
        print(f'Search returned 0 results: {msg}')
    else:
        for match in tools:
            print(f'  {match["name"]}: {match["description"]}')
```

```python
# Example 2 — manually building a synthetic ToolSearchReturnContent for testing
from pydantic_ai._tool_search import ToolSearchReturnContent, ToolSearchMatch

synthetic: ToolSearchReturnContent = {
    'discovered_tools': [
        {'name': 'get_weather', 'description': 'Returns current weather for a city.'},
        {'name': 'search_web',  'description': 'Searches the web for current events.'},
    ]
}
# No 'message' key — NotRequired means its absence signals a non-empty match
assert 'message' not in synthetic
```

```python
# Example 3 — pattern-matching on queries field to detect multi-query searches
from pydantic_ai._tool_search import ToolSearchArgs

def is_multi_query(args: ToolSearchArgs) -> bool:
    """OpenAI tool_search can produce multiple queries; BM25/local always produce one."""
    return len(args['queries']) > 1

# Upstream code that builds ToolSearchArgs for a local search_tools call:
local_args: ToolSearchArgs = {'queries': ['weather forecast']}
native_args: ToolSearchArgs = {'queries': ['get_weather', 'search_web']}  # OpenAI tool paths
print(is_multi_query(local_args))   # False
print(is_multi_query(native_args))  # True
```

---

## 2. `NativeToolSearchCallPart` + `NativeToolSearchReturnPart` — Server-Side Typed Message Parts

**Source**: `pydantic_ai/_tool_search.py`

These dataclasses extend `NativeToolCallPart` / `NativeToolReturnPart` with narrower typed fields. The key discriminator is `tool_kind = 'tool-search'` (not `tool_name`), which lets user-defined tools accidentally named `tool_search` pass through as base parts. Both are registered in `_TYPED_PART_TAGS` so the message discriminator can route serialised dicts to the right subclass.

```python
# Key signatures verified from source:

@dataclass(repr=False)
class NativeToolSearchCallPart(NativeToolCallPart):
    tool_name: Literal['tool_search'] = 'tool_search'
    args: str | ToolSearchArgs | None = None  # str = streaming partial
    tool_kind: Literal['tool-search'] = 'tool-search'

    @property
    def typed_args(self) -> ToolSearchArgs | None:
        """Parses str args via pydantic_core.from_json. Returns None during streaming partial."""
        ...

    @property
    def queries(self) -> list[str]:
        """Convenience accessor — empty list when typed_args is None."""
        ...

@dataclass(repr=False)
class NativeToolSearchReturnPart(NativeToolReturnPart):
    content: ToolSearchReturnContent = field(kw_only=True)  # kw_only avoids default-field ordering
    tool_name: Literal['tool_search'] = 'tool_search'
    tool_kind: Literal['tool-search'] = 'tool-search'

    @property
    def discovered_tools(self) -> list[ToolSearchMatch]: ...
    @property
    def message(self) -> str | None: ...
```

```python
# Example 1 — detecting a tool-search part regardless of native vs local path
from pydantic_ai.messages import ModelResponse

def has_tool_search(response: ModelResponse) -> bool:
    return any(
        getattr(part, 'tool_kind', None) == 'tool-search'
        for part in response.parts
    )
```

```python
# Example 2 — safely reading queries from a streaming-partial call part
from pydantic_ai._tool_search import NativeToolSearchCallPart

def handle_streaming_call(part: NativeToolSearchCallPart) -> None:
    # typed_args returns None while the JSON string is still partial
    ta = part.typed_args
    if ta is None:
        print('Still streaming...')
    else:
        print(f'Searching for: {ta["queries"]}')

    # .queries short-circuits to [] when still partial
    print(f'Queries (safe): {part.queries}')
```

```python
# Example 3 — inspecting discovered tools from a native return part
from pydantic_ai._tool_search import NativeToolSearchReturnPart

def summarise_native_result(part: NativeToolSearchReturnPart) -> str:
    tools = part.discovered_tools   # property → part.content['discovered_tools']
    if not tools:
        return part.message or 'No matches'
    names = ', '.join(t['name'] for t in tools)
    return f'Found {len(tools)} tool(s): {names}'
```

---

## 3. `ToolSearchCallPart` + `ToolSearchReturnPart` + `synthesize_local_tool_search_messages` — Local-Fallback Typed Parts and Cross-Provider History Translation

**Source**: `pydantic_ai/_tool_search.py`

`ToolSearchCallPart` and `ToolSearchReturnPart` are the local-fallback equivalents (the model calls a regular `search_tools` function tool; the framework emits the typed return). `synthesize_local_tool_search_messages` converts a history containing `NativeToolSearch*Part`s — produced by Anthropic/OpenAI — into local-shape messages so the next turn on a non-native provider can still see discovered-tool state.

The split logic uses `NativeToolSearchReturnPart` as a flush boundary: parts before the return form a `ModelResponse`; the return becomes a `ModelRequest`; parts after restart a fresh `ModelResponse`. Identity-level metadata (`provider_response_id`, `usage`) is kept only on the first split response to prevent double-counting.

```python
# Key signature from source:
@dataclass(repr=False)
class ToolSearchCallPart(ToolCallPart):
    tool_name: Literal['search_tools'] = 'search_tools'
    args: str | ToolSearchArgs | None = None
    tool_kind: Literal['tool-search'] = 'tool-search'

    @property
    def typed_args(self) -> ToolSearchArgs | None: ...
    @property
    def queries(self) -> list[str]: ...

@dataclass(repr=False)
class ToolSearchReturnPart(ToolReturnPart):
    content: ToolSearchReturnContent = field(kw_only=True)
    tool_name: Literal['search_tools'] = 'search_tools'
    tool_kind: Literal['tool-search'] = 'tool-search'

def synthesize_local_tool_search_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Translate NativeToolSearch*Parts to local-shape for non-native providers."""
    ...
```

```python
# Example 1 — translate Anthropic history so OpenAI run can see discovered tools
from pydantic_ai._tool_search import synthesize_local_tool_search_messages
from pydantic_ai import Agent

async def cross_provider_handoff(
    anthropic_agent: Agent,
    openai_agent: Agent,
    prompt: str,
) -> str:
    # Run first agent (Anthropic — uses native server-side tool search)
    result = await anthropic_agent.run(prompt)
    history = result.all_messages()

    # Translate native parts so the OpenAI agent can consume the history
    translated = synthesize_local_tool_search_messages(history)

    # Continue with OpenAI agent (local fallback path)
    result2 = await openai_agent.run(
        'Continue with the tools you discovered.',
        message_history=translated,
    )
    return result2.output
```

```python
# Example 2 — understanding the flush-boundary split
# A native ModelResponse like:
#   [TextPart("Let me search"), NativeToolSearchCallPart(q="weather"),
#    NativeToolSearchReturnPart(tools=[...]), ToolCallPart(name="get_weather")]
#
# Becomes four messages after synthesis:
#   ModelResponse([TextPart, ToolSearchCallPart])    ← first split, keeps provider metadata
#   ModelRequest([ToolSearchReturnPart])             ← the search return, framework-emitted
#   ModelResponse([ToolCallPart])                    ← remaining parts, zeroed usage
from pydantic_ai.messages import ModelRequest, ModelResponse

def count_message_types(messages):
    requests = sum(1 for m in messages if isinstance(m, ModelRequest))
    responses = sum(1 for m in messages if isinstance(m, ModelResponse))
    return requests, responses
```

```python
# Example 3 — detecting a local tool-search return on a ModelRequest
from pydantic_ai._tool_search import ToolSearchReturnPart

def extract_discovered_tools(messages):
    """Collect all tools discovered via local fallback in this history."""
    found = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolSearchReturnPart):
                    found.extend(part.discovered_tools)
    return found
```

---

## 4. `ResolvedUrl` + `safe_download` — SSRF-Protected URL Download

**Source**: `pydantic_ai/_ssrf.py`

`safe_download` is the framework's hardened HTTP client used wherever pydantic-ai fetches external URLs (e.g. `WebFetch` capability, `common_tools.web_fetch`). It resolves hostnames to IP addresses before connecting, blocks all private/internal ranges and cloud metadata endpoints — even with `allow_local=True` — and strips sensitive headers (`Authorization`, `Cookie`, `Proxy-Authorization`) on cross-origin redirects.

Key design facts:
- `_CLOUD_METADATA_IPV4` hardcodes seven cloud metadata IPs (AWS, GCP, Azure, Alibaba, Oracle, Scaleway) — always blocked.
- IPv6 transition forms (6to4, NAT64, ISATAP, IPv4-mapped) are decoded by `_embedded_ipv4s()` to catch metadata IPs smuggled in IPv6 clothing. `exhaustive=True` is used for the cloud-metadata guard specifically.
- `_MAX_REDIRECTS = 10`; domain restrictions are re-checked on every hop so a redirect cannot bypass an `allowed_domains` list.
- A trailing dot on the hostname is stripped before validation (FQDN root label: `169.254.169.254.` ≡ `169.254.169.254`).

```python
# Key signatures verified from source:
@dataclass
class ResolvedUrl:
    resolved_ip: str      # first DNS result (used as connect target)
    hostname: str         # original hostname (Host header + SNI)
    port: int
    is_https: bool
    path: str             # includes query string and fragment

async def safe_download(
    url: str,
    allow_local: bool = False,         # True → skip private-IP check; metadata always blocked
    max_redirects: int = 10,
    timeout: int = 30,                 # seconds
    headers: dict[str, str] | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> httpx.Response: ...
```

```python
# Example 1 — basic safe download with domain allowlist
from pydantic_ai._ssrf import safe_download

async def fetch_trusted_source(url: str) -> str:
    response = await safe_download(
        url,
        allowed_domains=['api.example.com', 'data.example.com'],
        timeout=15,
    )
    return response.text

# This raises ValueError: domain not in allowed list
# await safe_download('https://evil.example.org/data', allowed_domains=['api.example.com'])
```

```python
# Example 2 — checking if an IP would be blocked
from pydantic_ai._ssrf import is_cloud_metadata_ip, is_private_ip

# Cloud metadata IPs are always blocked (even with allow_local=True)
print(is_cloud_metadata_ip('169.254.169.254'))   # True  — AWS/GCP/Azure IMDS
print(is_cloud_metadata_ip('168.63.129.16'))     # True  — Azure WireServer (public IP!)
print(is_cloud_metadata_ip('100.100.100.200'))   # True  — Alibaba Cloud

# Private IPs are blocked by default, allowed when allow_local=True
print(is_private_ip('10.0.1.5'))                 # True
print(is_private_ip('192.168.0.1'))              # True
print(is_private_ip('8.8.8.8'))                  # False
```

```python
# Example 3 — safe download with custom headers (sensitive headers are auto-stripped on redirect)
from pydantic_ai._ssrf import safe_download

async def authenticated_fetch(url: str, token: str) -> bytes:
    response = await safe_download(
        url,
        headers={
            'Authorization': f'Bearer {token}',
            'X-Custom-Header': 'my-value',
        },
        max_redirects=5,
    )
    # If the server redirects to a different host, Authorization is stripped.
    # X-Custom-Header is kept (not in _SENSITIVE_HEADERS).
    return response.content
```

---

## 5. `PendingMessage` + `PendingMessagePriority` + `_build_enqueue_messages` — Runtime Pending Message Queue

**Source**: `pydantic_ai/_enqueue.py`

`PendingMessage` is the internal runtime dataclass that backs `RunContext.enqueue` and `AgentRun.enqueue`. It is not part of the wire-serialisable message history — it only exists during an active run. `_build_enqueue_messages` coalesces variadic `EnqueueContent` items into a well-formed `list[ModelMessage]`, enforcing the invariant that the sequence must end in a `ModelRequest` (so the agent has something to respond to).

`PendingMessagePriority`:
- `'asap'`: prepended to the next `ModelRequest`, or used to redirect the run if it would otherwise terminate.
- `'when_idle'`: delivered only when the agent would otherwise terminate, after all `'asap'` messages.

```python
# Key signatures from source:
PendingMessagePriority: TypeAlias = Literal['asap', 'when_idle']

EnqueueContent: TypeAlias = 'UserContent | ModelRequestPart | ModelMessage'

@dataclass
class PendingMessage:
    messages: list[ModelMessage]     # always ends in ModelRequest
    priority: PendingMessagePriority = 'asap'

    @classmethod
    def from_content(
        cls,
        *content: EnqueueContent,
        priority: PendingMessagePriority = 'asap',
    ) -> PendingMessage | None:
        """Returns None for empty call. Raises UserError if assembled messages don't end in ModelRequest."""
        ...
```

```python
# Example 1 — enqueueing a follow-up question from inside a tool
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

agent = Agent('openai:gpt-5')

@agent.tool
async def fetch_data(ctx: RunContext[None], resource_id: str) -> str:
    data = f'Resource {resource_id}: [data here]'
    # Inject an asap follow-up so the model analyses the fetched data immediately
    ctx.enqueue(
        f'Here is the fetched data for {resource_id}:\n{data}\nPlease summarise it.',
        priority='asap',
    )
    return 'Data fetched and queued for analysis.'
```

```python
# Example 2 — when_idle notification to append context after the run naturally completes
from pydantic_ai import Agent
from pydantic_ai.messages import SystemPromptPart

agent = Agent('openai:gpt-5', system_prompt='You are a helpful assistant.')

async def run_with_postamble(user_prompt: str) -> str:
    # enqueue() is on AgentRun (agent.iter()), not on StreamedRunResult (agent.run_stream())
    async with agent.iter(user_prompt) as agent_run:
        # Append a reminder only after the agent is idle (won't interrupt tool calls)
        agent_run.enqueue(
            SystemPromptPart(content='Remember: always cite your sources.'),
            priority='when_idle',
        )
        async for _node in agent_run:
            pass
    return agent_run.result.output
```

```python
# Example 3 — _build_enqueue_messages coalescing behaviour
from pydantic_ai._enqueue import _build_enqueue_messages
from pydantic_ai.messages import ImageUrl, ModelRequest, UserPromptPart

# Adjacent str/UserContent items are gathered into a single UserPromptPart
messages = _build_enqueue_messages(['Hello', ImageUrl(url='https://example.com/img.png'), 'Describe this.'])
assert len(messages) == 1
assert isinstance(messages[0], ModelRequest)
# UserPromptPart content is a list because there are multiple / non-str items
part = messages[0].parts[0]
assert isinstance(part, UserPromptPart)
assert isinstance(part.content, list)
```

---

## 6. `CombinedCapability` — Capability Tree Flattener

**Source**: `pydantic_ai/capabilities/combined.py`

`CombinedCapability` wraps multiple `AbstractCapability` instances and presents them as one. Its `__post_init__` splats any nested `CombinedCapability` so that leaves participate as siblings in the `sort_capabilities` ordering pass. Without this, a nested `CombinedCapability` whose leaves span `outermost` and `innermost` tiers would cause `_effective_ordering` to merge them into one position and raise `ConflictingPositions`.

Most users interact with this indirectly: passing `capabilities=[a, b, c]` to `Agent()` creates a `CombinedCapability` under the hood.

```python
# Key signature from source:
@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    def __post_init__(self) -> None:
        # Splat nested CombinedCapabilities so all leaves are siblings
        flat: list[AbstractCapability[AgentDepsT]] = []
        for cap in self.capabilities:
            if isinstance(cap, CombinedCapability):
                flat.extend(cap.capabilities)
            else:
                flat.append(cap)
        self.capabilities = flat
```

```python
# Example 1 — combining capabilities and inspecting the flat structure
from pydantic_ai.capabilities import Thinking, ThreadExecutor, HandleDeferredToolCalls
from pydantic_ai.capabilities.combined import CombinedCapability
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=8)
group_a = CombinedCapability(capabilities=[Thinking(effort='high'), ThreadExecutor(executor)])
group_b = CombinedCapability(capabilities=[HandleDeferredToolCalls(handler=lambda ctx, req: None)])

# Nesting is flattened: group_a + group_b → 3 sibling leaves
combined = CombinedCapability(capabilities=[group_a, group_b])
print(len(combined.capabilities))  # 3 — not 2 (splat happened in __post_init__)
```

```python
# Example 2 — passing multiple capabilities to Agent (creates CombinedCapability internally)
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, ReinjectSystemPrompt

agent = Agent(
    'anthropic:claude-opus-4-8',
    capabilities=[
        Thinking(effort='high'),
        ReinjectSystemPrompt(replace_existing=True),
    ],
)
# Equivalent to:
# agent = Agent(..., capabilities=[CombinedCapability([Thinking(...), ReinjectSystemPrompt(...)])])
```

```python
# Example 3 — runtime capability injection on a single run
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('anthropic:claude-opus-4-8')

async def solve_hard_problem(prompt: str) -> str:
    # Capabilities can also be passed per-run, merged with agent-level ones
    result = await agent.run(
        prompt,
        capabilities=[Thinking(effort='xhigh')],
    )
    return result.output
```

---

## 7. `ProcessEventStream` — Dual-Mode Event Stream Capability

**Source**: `pydantic_ai/capabilities/process_event_stream.py`

`ProcessEventStream` registers a capability that fires `wrap_run_event_stream` for every `ModelRequestNode` and `CallToolsNode`. Two handler forms are supported:

- **Observer** (`async def` returning `None`): the framework tees the event stream via an `anyio` memory object stream so the handler sees all events while they also flow unchanged to the next consumer. A handler that returns early stops receiving events but does not break the main stream. Slow handlers back-pressure the stream.
- **Processor** (async generator yielding `AgentStreamEvent`s): the events it yields replace the inner stream for downstream wrappers — it can modify, drop, or inject events.

The capability probes which form is in use by calling `handler(ctx, stream)` and checking `isinstance(probe, AsyncIterator)`.

Under durable-execution runtimes (Temporal, DBOS, Prefect), model-response events are consumed inside an activity boundary, so only tool-call events and the final post-streaming batch reach the capability hook.

```python
# Key signature from source:
@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    handler: EventStreamHandler[AgentDepsT] | EventStreamProcessor[AgentDepsT]

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        # Probe: AsyncIterator → processor path, Coroutine → observer path
        ...
```

```python
# Example 1 — observer: log every event without modifying the stream
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent
from collections.abc import AsyncIterable

async def log_events(ctx, stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        print(f'[{type(event).__name__}]', event)
        # return early at any point — main stream continues unaffected

agent = Agent('openai:gpt-5', capabilities=[ProcessEventStream(log_events)])
```

```python
# Example 2 — processor: strip ThinkingPart events from the downstream view
# Track part indexes that started as ThinkingPart to also drop their delta/end events.
from pydantic_ai.messages import (
    AgentStreamEvent, PartStartEvent, PartDeltaEvent, PartEndEvent,
    ThinkingPart, ThinkingPartDelta,
)
from collections.abc import AsyncIterable, AsyncIterator

async def strip_thinking(ctx, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    skipped: set[int] = set()
    async for event in stream:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            skipped.add(event.index)
            continue
        if isinstance(event, (PartDeltaEvent, PartEndEvent)) and event.index in skipped:
            if isinstance(event, PartEndEvent):
                skipped.discard(event.index)
            continue
        yield event

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream

agent = Agent('anthropic:claude-opus-4-8', capabilities=[ProcessEventStream(strip_thinking)])
```

```python
# Example 3 — injecting a sentinel event after every function-tool call
# FunctionToolCallEvent fires when a function tool is about to be invoked.
# Inject a PartStartEvent(TextPart) afterwards for UI annotation.
from pydantic_ai.messages import AgentStreamEvent, FunctionToolCallEvent, PartStartEvent, TextPart
from collections.abc import AsyncIterable, AsyncIterator

async def annotate_tool_calls(ctx, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    async for event in stream:
        yield event
        if isinstance(event, FunctionToolCallEvent):
            # Inject a synthetic text-part event after each tool call for UI annotation
            yield PartStartEvent(index=999, part=TextPart(content='[tool call detected]'))

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream

agent = Agent('openai:gpt-5', capabilities=[ProcessEventStream(annotate_tool_calls)])
```

---

## 8. `HandleDeferredToolCalls` — Inline Deferred-Tool Resolver

**Source**: `pydantic_ai/capabilities/deferred_tool_handler.py`

When a tool is decorated with `requires_approval=True` or `deferred=True`, the agent normally pauses and surfaces a `DeferredToolRequests` output. `HandleDeferredToolCalls` intercepts these requests inline, calls the provided handler, and continues the run automatically. The handler may return `None` to decline — the next capability in chain gets a chance, and ultimately the requests bubble up as output if all decline.

```python
# Key signature from source:
@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    handler: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests],
        DeferredToolResults | None | Awaitable[DeferredToolResults | None],
    ]

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        result = self.handler(ctx, requests)
        if inspect.isawaitable(result):
            return await result
        return result
```

```python
# Example 1 — auto-approve all deferred tools (dev/testing pattern)
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

async def approve_all(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    return requests.build_results(approve_all=True)

agent = Agent(
    'openai:gpt-5',
    capabilities=[HandleDeferredToolCalls(handler=approve_all)],
)
```

```python
# Example 2 — selective approval based on tool name
# DeferredToolRequests is not iterable; combine .approvals and .calls explicitly.
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

APPROVED_TOOLS = {'search_web', 'get_weather'}

async def selective_approve(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    all_pending = [*requests.approvals, *requests.calls]
    # Decline if any pending tool is not in the allowed set
    if not all_pending or not all(req.tool_name in APPROVED_TOOLS for req in all_pending):
        return None          # Bubble up to next handler or as output
    return requests.build_results(approve_all=True)
```

```python
# Example 3 — async handler with external approval service
# DeferredToolRequests.approvals is a list[ToolCallPart] — iterate it directly.
# build_results() takes approvals={tool_call_id: True/False}, not results=.
import httpx
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

async def remote_approval(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    pending = requests.approvals  # list[ToolCallPart] requiring human approval
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'https://internal.example.com/approve',
            json=[{'tool': r.tool_name, 'args': r.args} for r in pending],
            timeout=10,
        )
        decisions = resp.json()  # [{approved: true/false}, ...]
    approvals = {
        req.tool_call_id: decision['approved']
        for req, decision in zip(pending, decisions)
    }
    return requests.build_results(approvals=approvals)
```

---

## 9. `Thinking` — Unified Multi-Provider Thinking Control

**Source**: `pydantic_ai/capabilities/thinking.py`

`Thinking` is the simplest substantive capability: it merges `ModelSettings(thinking=self.effort)` into every run it participates in, enabling model reasoning in a portable, provider-agnostic way. Provider-specific settings (`anthropic_thinking`, `openai_reasoning_effort`) take precedence when both are set. On always-on models, `effort=False` is silently ignored.

`ThinkingLevel = Literal[True, False, 'minimal', 'low', 'medium', 'high', 'xhigh']`

```python
# Key signature from source:
@dataclass
class Thinking(AbstractCapability[Any]):
    effort: ThinkingLevel = True
    # True  → provider's default effort
    # False → disable thinking (silently ignored on always-on models)
    # 'minimal'/'low'/'medium'/'high'/'xhigh' → explicit effort level

    def get_model_settings(self) -> ModelSettings | None:
        return ModelSettings(thinking=self.effort)
```

```python
# Example 1 — enable thinking at a specific effort level for complex reasoning
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent(
    'anthropic:claude-opus-4-8',
    capabilities=[Thinking(effort='high')],
    system_prompt='You are a rigorous mathematical proof assistant.',
)

async def solve_proof(problem: str) -> str:
    result = await agent.run(problem)
    return result.output
```

```python
# Example 2 — per-run thinking override via run-level capabilities
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Agent-level: medium thinking by default
agent = Agent('anthropic:claude-opus-4-8', capabilities=[Thinking(effort='medium')])

async def intensive_run(prompt: str) -> str:
    # Override to xhigh for this particular run only
    result = await agent.run(prompt, capabilities=[Thinking(effort='xhigh')])
    return result.output

async def cheap_run(prompt: str) -> str:
    result = await agent.run(prompt, capabilities=[Thinking(effort='low')])
    return result.output
```

```python
# Example 3 — mixing Thinking with provider-specific settings (provider wins)
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'openai:o3',
    capabilities=[Thinking(effort='medium')],  # sets thinking='medium'
    model_settings=ModelSettings(
        openai_reasoning_effort='high',  # provider-specific → takes precedence
    ),
)
# Effective: openai_reasoning_effort='high' (overrides capability-level 'medium')
```

---

## 10. `ThreadExecutor` + `ReinjectSystemPrompt` — Bounded Thread Pool and System-Prompt Reinjection

**Source**: `pydantic_ai/capabilities/thread_executor.py` / `pydantic_ai/capabilities/reinject_system_prompt.py`

**`ThreadExecutor`** addresses thread accumulation in long-running servers: by default, sync tool callbacks use `anyio.to_thread.run_sync` with ephemeral threads. Under sustained FastAPI load this causes unbounded thread growth. `ThreadExecutor` wraps each run with `_utils.using_thread_executor(self.executor)`, scoping a bounded `ThreadPoolExecutor` to that run's lifetime.

**`ReinjectSystemPrompt`** handles the case where `message_history` comes from a source that doesn't round-trip system prompts (UI frontends, DB persistence, compaction pipelines). It prepends the agent's configured system prompt to the first `ModelRequest` if none is present. `replace_existing=True` additionally strips any pre-existing `SystemPromptPart`s first — useful when the history is untrusted. The UI adapters use this automatically in `manage_system_prompt='server'` mode.

```python
# Key signatures from source:
@dataclass
class ThreadExecutor(AbstractCapability[Any]):
    executor: Executor   # e.g. ThreadPoolExecutor(max_workers=16)

    async def wrap_run(self, ctx, *, handler) -> AgentRunResult[Any]:
        with _utils.using_thread_executor(self.executor):
            return await handler()

@dataclass
class ReinjectSystemPrompt(AbstractCapability[AgentDepsT]):
    replace_existing: bool = False
    # False → no-op if any SystemPromptPart already present
    # True  → strip existing SystemPromptParts first, then prepend agent's prompt

    async def before_model_request(self, ctx, request_context) -> ModelRequestContext: ...
```

```python
# Example 1 — ThreadExecutor: bounded pool for a FastAPI service
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor
import contextlib

# Module-level pool — shared across all requests
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix='agent-worker')

agent = Agent(
    'openai:gpt-5',
    capabilities=[ThreadExecutor(_pool)],
)

@contextlib.asynccontextmanager
async def lifespan(app):
    yield
    _pool.shutdown(wait=True)

# With this setup, all sync tools in all agent runs use the bounded pool,
# preventing thread accumulation under load.
```

```python
# Example 2 — ThreadExecutor: per-request pool for strict isolation
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

agent = Agent('openai:gpt-5')

async def handle_request(user_prompt: str) -> str:
    import asyncio
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        result = await agent.run(
            user_prompt,
            capabilities=[ThreadExecutor(pool)],
        )
        return result.output
    finally:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pool.shutdown, True)
```

```python
# Example 3 — ReinjectSystemPrompt: server-authoritative system prompt
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    'openai:gpt-5',
    system_prompt='You are a helpful, harmless assistant.',
    capabilities=[
        ReinjectSystemPrompt(replace_existing=True),
        # replace_existing=True: strip any system prompts in the history the frontend sent
        # then prepend the server's authoritative prompt
    ],
)

async def handle_chat(history_from_frontend: list, user_message: str) -> str:
    # The frontend's history may contain a tampered or missing system prompt.
    # ReinjectSystemPrompt strips it and prepends our server-side prompt.
    result = await agent.run(user_message, message_history=history_from_frontend)
    return result.output
```

---

## Summary Table

| # | Class(es) | Module | Key insight |
|---|-----------|--------|-------------|
| 1 | `ToolSearchArgs`, `ToolSearchMatch`, `ToolSearchReturnContent` | `_tool_search` | Shared `TypedDict` shapes for both native and local tool-search paths |
| 2 | `NativeToolSearchCallPart`, `NativeToolSearchReturnPart` | `_tool_search` | Server-side typed parts; `tool_kind='tool-search'` discriminates, not `tool_name` |
| 3 | `ToolSearchCallPart`, `ToolSearchReturnPart`, `synthesize_local_tool_search_messages` | `_tool_search` | Local-fallback parts; flush-boundary splits preserve identity metadata on first response only |
| 4 | `ResolvedUrl`, `safe_download` | `_ssrf` | SSRF protection; cloud metadata always blocked; sensitive headers stripped on cross-origin redirect |
| 5 | `PendingMessage`, `PendingMessagePriority`, `_build_enqueue_messages` | `_enqueue` | Runtime message queue; `asap` vs `when_idle`; must end in `ModelRequest` |
| 6 | `CombinedCapability` | `capabilities/combined` | `__post_init__` splats nested `CombinedCapability` to prevent ordering conflicts |
| 7 | `ProcessEventStream` | `capabilities/process_event_stream` | Observer vs processor probe; anyio tee; durable-exec only sees tool events |
| 8 | `HandleDeferredToolCalls` | `capabilities/deferred_tool_handler` | Inline deferred-tool resolver; `None` to decline and pass to next handler |
| 9 | `Thinking` | `capabilities/thinking` | `ThinkingLevel` unified across providers; provider-specific settings override |
| 10 | `ThreadExecutor`, `ReinjectSystemPrompt` | `capabilities/thread_executor`, `capabilities/reinject_system_prompt` | Bounded thread pool for servers; server-authoritative system prompt injection |

## Cross-References

- **Vol. 27**: `TemporalModel`, `DBOSAgent`, `PrefectModel`, `LogfirePlugin`, OTel baggage constants
- **Vol. 26**: `AgentRun`, `AgentRunResult`, streaming-result patterns
- **Vol. 25**: `AbstractCapability`, `AbstractToolset`, capability ordering internals
- **Vol. 24**: `WrapperAgent`, `WrapperModel`, `WrapperToolset` delegation patterns
- **Vol. 9**: Core `RunContext`, `Tool`, `ToolDefinition` — foundation for `HandleDeferredToolCalls`

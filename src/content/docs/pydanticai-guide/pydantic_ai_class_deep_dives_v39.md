---
title: "PydanticAI Class Deep Dives Vol. 39"
description: "Source-verified deep dives into 10 pydantic-ai 2.18.0 class groups: RaiseContentFilterError (opt-in content-filter error — finish_reason detection, provider_details inspection, body serialisation), ResolveModelId (custom model ID resolver — sync/async ModelIdResolver, ModelResolutionContext, None passthrough), SelectModel (per-step model selection — ModelSelector callable, ModelSelectionContext fields, dynamic cost routing), HandleDeferredToolCalls (inline deferred resolution — sync/async handler, None-decline chaining, build_results(approve_all=True)), PendingMessageDrainCapability (auto-injected drain — 'asap' via before_model_request, 'when_idle' via after_node_run redirect, EnqueuedMessagesEvent), XSearch capability (xAI native + non-xAI fallback_model — allowed/excluded_x_handles max-20, from_date/to_date, enable_image/video_understanding, include_output), ImageGeneration capability (12-field config matrix — action/background/input_fidelity/moderation/image_model/output_compression/output_format/quality/size/aspect_ratio, fallback_model subagent, _image_gen_kwargs bridge), TavilySearchTool + tavily_search_tool (_UNSET sentinel pattern — partial.__signature__ freeze to hide developer-controlled params from LLM, search_depth/topic/time_range/include_domains/exclude_domains), MCP capability (url/native/local triple — auto-wrap non-URL fastmcp inputs, authorization_token/headers merge, allowed_tools filter, from_spec() for YAML/JSON), PrefixTools + ThreadExecutor (namespace prefix via PrefixedToolset + DynamicToolset delegation; bounded Executor with using_thread_executor() class-level and per-run scoping)."
sidebar:
  label: "Class deep dives (Vol. 39)"
  order: 65
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.18.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.18.x API. Three examples per class group; all code blocks pass `ast.parse()` syntax validation. Live API calls are commented out — uncomment to run.
</Aside>

Ten class groups covering opt-in content-filter error handling (`RaiseContentFilterError`), custom model ID resolution (`ResolveModelId`), per-step model switching (`SelectModel`), inline deferred tool resolution (`HandleDeferredToolCalls`), the auto-injected pending-message drain (`PendingMessageDrainCapability`), X/Twitter search with multimedia understanding (`XSearch`), the full 12-field image generation capability (`ImageGeneration`), the Tavily search factory with its signature-freeze trick (`TavilySearchTool`), the primary MCP capability (`MCP`), and tool namespace prefixing plus bounded thread execution (`PrefixTools` + `ThreadExecutor`).

---

## 1. `RaiseContentFilterError`

**Source:** `pydantic_ai/capabilities/content_filter.py`

`RaiseContentFilterError` is an `after_model_request` capability that turns a `finish_reason='content_filter'` model response into a `ContentFilterError` exception. By default pydantic-ai passes content-filtered responses through so the caller can inspect partial text — this capability opts into strict error-on-filter behaviour. The full `ModelResponse` is serialised into `ContentFilterError.body` using `ModelMessagesTypeAdapter` so callers can examine any partial content. The error message is populated from provider-specific details: `finish_reason`, `block_reason`, or `refusal` from `response.provider_details`.

```python
# Example 1 — Opt in globally: all content-filtered responses become errors
import asyncio
from unittest.mock import AsyncMock

from pydantic_ai import Agent
from pydantic_ai.capabilities import RaiseContentFilterError

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[RaiseContentFilterError()],
)

# From this point any run whose model returns finish_reason='content_filter'
# will raise ContentFilterError instead of returning partial text.
# Catch it to inspect the body:
# try:
#     result = await agent.run("tell me something controversial")
# except ContentFilterError as exc:
#     print("Filtered! Raw body:", exc.body[:200])
```

```python
# Example 2 — Per-run injection: only specific runs use strict filter handling
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import RaiseContentFilterError
from pydantic_ai.exceptions import ContentFilterError

agent = Agent('anthropic:claude-sonnet-5-20251101')

async def safe_run(prompt: str) -> str | None:
    try:
        result = await agent.run(
            prompt,
            capabilities=[RaiseContentFilterError()],
        )
        return result.output
    except ContentFilterError as exc:
        # exc.body is a JSON-serialised list[ModelResponse]
        print(f"Content filter: {exc}")
        return None

# asyncio.run(safe_run("What is quantum computing?"))
```

```python
# Example 3 — Inspect provider_details from the error body
import json
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import ModelMessagesTypeAdapter

def inspect_filter_error(exc: ContentFilterError) -> dict:
    """Decode the full ModelResponse preserved in ContentFilterError.body."""
    messages = ModelMessagesTypeAdapter.validate_json(exc.body)
    response = messages[0]  # The ModelResponse that triggered the filter
    return {
        "finish_reason": response.finish_reason,
        "provider_details": response.provider_details,
        "parts_count": len(response.parts),
    }

# Simulated usage — in practice exc comes from a failed agent run:
# try:
#     await agent.run("...", capabilities=[RaiseContentFilterError()])
# except ContentFilterError as exc:
#     info = inspect_filter_error(exc)
#     print(info)
```

---

## 2. `ResolveModelId`

**Source:** `pydantic_ai/capabilities/resolve_model_id.py`

`ResolveModelId` wraps a user-supplied callable (`ModelIdResolver`) that receives a `ModelResolutionContext` and the string model ID selected for the current step. Return a `Model` instance to override resolution, or `None` to fall through to the next capability or the default `infer_model` logic. The resolver can be synchronous or asynchronous — pydantic-ai detects this via `is_async_callable` and dispatches accordingly. This is the extension point for custom model registries, A/B test routing, and per-environment model overrides.

```python
# Example 1 — Route "gpt-5-fast" to a real model without registering a provider
from pydantic_ai import Agent
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models import ModelResolutionContext
from pydantic_ai.models.openai import OpenAIModel

def my_resolver(ctx: ModelResolutionContext, model_id: str):
    # Intercept a custom alias and return a concrete Model instance.
    if model_id == 'gpt-5-fast':
        return OpenAIModel('gpt-5.2', http_client=ctx.http_client)
    return None  # Let the default infer_model handle everything else

agent = Agent(
    'gpt-5-fast',  # Custom alias — resolved by our capability
    capabilities=[ResolveModelId(resolver=my_resolver)],
)
# await agent.run("hello")
```

```python
# Example 2 — Async resolver that loads model config from a remote registry
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models import ModelResolutionContext

async def registry_resolver(ctx: ModelResolutionContext, model_id: str):
    """Look up a model ID in a hypothetical remote registry."""
    if not model_id.startswith('registry:'):
        return None
    # In production, fetch the real model config from a registry service:
    # config = await registry_client.get(model_id[len('registry:'):])
    # return infer_model(config['resolved_id'])
    return None  # Passthrough for this example

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[ResolveModelId(resolver=registry_resolver)],
)
```

```python
# Example 3 — Chaining two resolvers: first wins, second is a fallback
from pydantic_ai import Agent
from pydantic_ai.capabilities import ResolveModelId, CombinedCapability
from pydantic_ai.models import ModelResolutionContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

def env_resolver(ctx: ModelResolutionContext, model_id: str):
    """Override models based on an environment variable."""
    import os
    override = os.getenv(f'MODEL_OVERRIDE_{model_id.upper().replace(":", "_")}')
    if override:
        from pydantic_ai.models import infer_model
        return infer_model(override)
    return None

def cost_resolver(ctx: ModelResolutionContext, model_id: str):
    """Downgrade expensive models in non-prod environments."""
    import os
    if os.getenv('ENV') != 'prod' and 'opus' in model_id:
        return AnthropicModel('claude-haiku-4-5-20251001')
    return None

agent = Agent(
    'anthropic:claude-opus-5',
    capabilities=[
        ResolveModelId(resolver=env_resolver),    # checked first
        ResolveModelId(resolver=cost_resolver),   # fallback
    ],
)
```

---

## 3. `SelectModel`

**Source:** `pydantic_ai/capabilities/select_model.py`

`SelectModel` wraps a `ModelSelector` callable that is called before each logical model-request step. The selector receives a `ModelSelectionContext` containing the run dependencies, message history, accumulated usage, and the lower-precedence model, and returns a concrete `Model` instance or model ID string. Both sync and async selectors are supported. This is the primary mechanism for dynamic per-step model routing — e.g. switching from a cheap model for tool calls to a powerful model for final synthesis, or implementing a cost-aware model ladder.

```python
# Example 1 — Switch from cheap to powerful model after N tool calls
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models import ModelSelectionContext

def model_ladder(ctx: ModelSelectionContext) -> str:
    """Use a fast model first; upgrade to a capable model once tools have run."""
    tool_calls = sum(
        1
        for msg in ctx.message_history
        if hasattr(msg, 'parts')
        for part in msg.parts
        if hasattr(part, 'tool_name')
    )
    if tool_calls >= 2:
        return 'anthropic:claude-sonnet-5-20251101'
    return 'anthropic:claude-haiku-4-5-20251001'

agent = Agent(
    'anthropic:claude-haiku-4-5-20251001',  # default
    capabilities=[SelectModel(selector=model_ladder)],
)
```

```python
# Example 2 — Async selector: cost-aware routing using token budget
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models import ModelSelectionContext

async def budget_selector(ctx: ModelSelectionContext) -> str:
    total_tokens = ctx.usage.total_tokens or 0
    # Hypothetical: if we've spent more than 10k tokens, switch to cheaper model
    if total_tokens > 10_000:
        return 'openai:gpt-4.1-mini'
    return 'openai:gpt-5.2'

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[SelectModel(selector=budget_selector)],
)
# result = await agent.run("Write a 2000-word essay on...")
```

```python
# Example 3 — Inject SelectModel per-run for different user tiers
from pydantic_ai import Agent
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models import ModelSelectionContext

agent = Agent('openai:gpt-4.1-mini')  # default for free tier

def premium_selector(ctx: ModelSelectionContext) -> str:
    return 'openai:gpt-5.2'

def standard_selector(ctx: ModelSelectionContext) -> str:
    return 'openai:gpt-4.1'

async def run_for_user(prompt: str, is_premium: bool) -> str:
    selector = premium_selector if is_premium else standard_selector
    result = await agent.run(
        prompt,
        capabilities=[SelectModel(selector=selector)],
    )
    return result.output

# asyncio.run(run_for_user("Summarise this document...", is_premium=True))
```

---

## 4. `HandleDeferredToolCalls`

**Source:** `pydantic_ai/capabilities/deferred_tool_handler.py`

`HandleDeferredToolCalls` intercepts tool calls that require approval or external execution. Normally, when an `ApprovalRequiredToolset` or `ExternalToolset` is present, the agent pauses and returns `DeferredToolRequests` as output so the caller can inspect and approve them. With this capability, you provide a `handler` function that resolves requests inline, keeping the agent running. The handler receives `(RunContext, DeferredToolRequests)` and may return `DeferredToolResults` (with results for all pending calls) or `None` to decline (passing control to the next handler in the chain or bubbling up to the caller).

```python
# Example 1 — Auto-approve all deferred tool calls (testing / automation)
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets import ApprovalRequiredToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

async def approve_all(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    return requests.build_results(approve_all=True)

def search_web(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(
    'openai:gpt-5.2',
    toolsets=[ApprovalRequiredToolset([search_web])],
    capabilities=[HandleDeferredToolCalls(handler=approve_all)],
)
# result = await agent.run("Search for pydantic-ai documentation")
# All tool calls are approved automatically — the run completes without pause.
```

```python
# Example 2 — Selective approval: approve low-risk tools, deny high-risk
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets import ApprovalRequiredToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

SAFE_TOOLS = {'search_web', 'get_weather', 'read_file'}

async def selective_handler(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    results = requests.build_results()
    for call in requests.calls:
        if call.tool_name in SAFE_TOOLS:
            results.approve(call.tool_call_id)
        else:
            results.deny(call.tool_call_id, "Tool requires manual review")
    return results

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[HandleDeferredToolCalls(handler=selective_handler)],
)
```

```python
# Example 3 — Chaining handlers: first declines, second handles
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

async def audit_only_handler(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    """Log the request but let the next handler decide."""
    print(f"Deferred tools requested: {[c.tool_name for c in requests.calls]}")
    return None  # Decline: pass to next HandleDeferredToolCalls capability

async def default_approve(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults:
    return requests.build_results(approve_all=True)

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        HandleDeferredToolCalls(handler=audit_only_handler),   # logs first
        HandleDeferredToolCalls(handler=default_approve),      # then approves
    ],
)
```

---

## 5. `PendingMessageDrainCapability`

**Source:** `pydantic_ai/capabilities/deferred_tool_handler.py` → `pydantic_ai/capabilities/_pending_messages.py`

`PendingMessageDrainCapability` is an **auto-injected internal capability** (not added by users) that manages the pending message queue populated by `RunContext.enqueue()` / `AgentRun.enqueue()`. It is always placed at `CapabilityOrdering(position='outermost')` so it wraps all user capabilities.

**Two drain moments:**
- `before_model_request`: drains `'asap'` messages into the upcoming `ModelRequest`, emitting one `EnqueuedMessagesEvent` per original `enqueue()` call.
- `after_node_run`: if the agent is about to end (`End` result), drains leftover `'asap'` and then `'when_idle'` messages, redirecting into a new `ModelRequestNode` so the agent gets another turn.

Understanding this mechanic is essential for building reactive agents that receive external events mid-run.

```python
# Example 1 — Enqueue a message from a tool: 'asap' priority
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

agent = Agent('openai:gpt-5.2')

@agent.tool
async def check_server_status(ctx: RunContext[None]) -> str:
    # Enqueue an urgent system message to be delivered before the next model call.
    await ctx.enqueue("SYSTEM: Database connection pool exhausted", priority='asap')
    return "Server status checked"

# When the agent calls check_server_status, PendingMessageDrainCapability
# drains the 'asap' message into the next ModelRequest automatically.
# result = await agent.run("Check server health and report findings")
```

```python
# Example 2 — 'when_idle' messages delivered only at run termination
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

agent = Agent('openai:gpt-5.2')

@agent.tool
async def schedule_followup(ctx: RunContext[None], message: str) -> str:
    # Enqueue a follow-up only if the agent otherwise terminates.
    await ctx.enqueue(message, priority='when_idle')
    return "Follow-up scheduled"

# The 'when_idle' message causes a new ModelRequestNode at end-of-run,
# giving the agent a chance to process it before the session truly ends.
# result = await agent.run("Analyse this data and schedule a follow-up")
```

```python
# Example 3 — Observe EnqueuedMessagesEvent in the event stream
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import EnqueuedMessagesEvent
from pydantic_ai.tools import RunContext

agent = Agent('openai:gpt-5.2')

@agent.tool
async def inject_alert(ctx: RunContext[None]) -> str:
    await ctx.enqueue("ALERT: Rate limit approaching 90%", priority='asap')
    return "Alert queued"

async def run_and_watch():
    async with agent.iter("Monitor the system") as agent_run:
        async for event in agent_run:
            if isinstance(event, EnqueuedMessagesEvent):
                print(f"Enqueued id={event.enqueue_id}, msgs={len(event.messages)}")

# asyncio.run(run_and_watch())
```

---

## 6. `XSearch` Capability

**Source:** `pydantic_ai/capabilities/x_search.py`

`XSearch` provides X (Twitter) search via the xAI native tool (on xAI models) or a subagent-based fallback (on any model, when `fallback_model` is set to an xAI model). Key constraint: there is **no default fallback** — non-xAI models must explicitly set `fallback_model`. The capability exposes fine-grained control over which X handles to include/exclude (max 20 each), date range filtering, and multimedia understanding (images and video in X posts).

**New in 2.18.0:** `include_output` exposes raw X search results as `NativeToolReturnPart` in the response.

```python
# Example 1 — xAI model using native X search with date range + image understanding
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from datetime import datetime, timezone

agent = Agent(
    'xai:grok-4.3',
    capabilities=[
        XSearch(
            from_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            to_date=datetime(2026, 7, 27, tzinfo=timezone.utc),
            enable_image_understanding=True,   # analyse images in posts
            enable_video_understanding=False,
            include_output=True,               # expose raw results in response
        )
    ],
)
# result = await agent.run("What is the latest news about pydantic-ai?")
```

```python
# Example 2 — Non-xAI model with fallback_model subagent
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'openai:gpt-5.2',  # Main model (doesn't support native X search)
    capabilities=[
        XSearch(
            fallback_model='xai:grok-4.3',  # Required for non-xAI models
            enable_image_understanding=True,
            enable_video_understanding=True,
        )
    ],
)
# Under the hood, XSearch spawns a grok-4.3 subagent that runs the native XSearchTool.
# result = await agent.run("Summarise the trending AI news from X this week")
```

```python
# Example 3 — Filter by specific X handles (max 20 per list)
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'xai:grok-4.3',
    capabilities=[
        XSearch(
            allowed_x_handles=['pydantic', 'anthropic', 'openai'],  # Only these handles
            excluded_x_handles=['bot_account', 'spam_account'],
        )
    ],
)
# The constraint fields (allowed/excluded_x_handles) require native support.
# On a non-xAI model with fallback_model set, the subagent enforces them too.
# result = await agent.run("What are pydantic and anthropic saying about agents?")
```

---

## 7. `ImageGeneration` Capability

**Source:** `pydantic_ai/capabilities/image_generation.py`

`ImageGeneration` is a `NativeOrLocalTool` subclass that exposes 12 config fields mirroring `ImageGenerationTool`, plus a `fallback_model` that triggers a subagent-based fallback for models without native image generation. The `_image_gen_kwargs()` helper collects non-`None` fields into a dict that is forwarded to both the native tool and the fallback subagent. A `fallback_model` and an explicit `local=` cannot be set simultaneously — one or the other.

```python
# Example 1 — OpenAI Responses with high-quality PNG generation
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai-responses:gpt-5.4',
    capabilities=[
        ImageGeneration(
            quality='high',
            output_format='png',
            size='1024x1024',
            action='generate',
        )
    ],
)
# result = await agent.run("Draw a futuristic city skyline at sunset")
# The model calls the native ImageGenerationTool with these settings.
```

```python
# Example 2 — fallback_model for non-image models (OpenAI chat → GPT image model)
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai:gpt-5.2',  # Chat model — no native image generation
    capabilities=[
        ImageGeneration(
            fallback_model='openai-responses:gpt-5.4',  # Image-capable model
            quality='medium',
            output_format='webp',
            output_compression=85,
        )
    ],
)
# When gpt-5.2 needs to generate an image, a gpt-5.4 subagent runs instead.
# result = await agent.run("Create a product mockup for a coffee brand")
```

```python
# Example 3 — Google image generation with aspect ratio + edit action
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'google:gemini-3-pro-image-preview',
    capabilities=[
        ImageGeneration(
            action='edit',              # Edit an existing image
            aspect_ratio='16:9',        # Supported by Google models
            size='2K',                  # Google size: '512'/'1K'/'2K'/'4K'
            output_format='jpeg',
            output_compression=90,
            background='opaque',
        )
    ],
)
# result = await agent.run("Add a sunrise to this landscape photo", ...)
```

---

## 8. `TavilySearchTool` + `tavily_search_tool`

**Source:** `pydantic_ai/common_tools/tavily.py`

The Tavily integration is a masterclass in hiding developer-controlled parameters from the LLM while keeping them available. The factory `tavily_search_tool()` accepts `search_depth`, `topic`, `time_range`, `include_domains`, and `exclude_domains` as keyword arguments. When you provide a value, it is frozen using `functools.partial` **and** the parameter is removed from `__signature__` — so the LLM never sees it in the tool schema and cannot override it. Parameters you leave unset remain visible to the LLM and can be set per-call. `max_results` is always developer-controlled and always hidden.

```python
# Example 1 — All parameters exposed to the LLM (minimal factory call)
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

# Only max_results is hidden; the LLM can set depth/topic/time_range/domains
tool = tavily_search_tool(api_key='tvly-...')

agent = Agent('openai:gpt-5.2', tools=[tool])
# The LLM schema includes: query, search_depth, topic, time_range,
# include_domains, exclude_domains — full flexibility.
# result = await agent.run("Find recent news about AI agents")
```

```python
# Example 2 — Freeze search_depth and topic; keep time_range open for LLM
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool
import inspect

tool = tavily_search_tool(
    api_key='tvly-...',
    max_results=5,
    search_depth='advanced',   # Always 'advanced' — hidden from LLM
    topic='news',              # Always 'news' — hidden from LLM
    # time_range not set → LLM can still choose 'day'/'week'/'month'/'year'
)

# Verify the frozen params are absent from the LLM-visible schema:
sig = inspect.signature(tool.function)
visible_params = list(sig.parameters.keys())
# ['query', 'time_range', 'include_domains', 'exclude_domains']
# 'search_depth' and 'topic' are gone — LLM cannot override them.
print("LLM-visible params:", visible_params)
```

```python
# Example 3 — Domain-restricted news search for a specific publication
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

# Fix to specific domains: LLM cannot search outside these
tool = tavily_search_tool(
    api_key='tvly-...',
    max_results=10,
    search_depth='basic',
    topic='news',
    include_domains=['techcrunch.com', 'theverge.com', 'wired.com'],
    # LLM still controls: query (and time_range, exclude_domains are also visible)
)

agent = Agent('openai:gpt-5.2', tools=[tool])
# result = await agent.run("What's the latest AI product launch covered by tech media?")
```

---

## 9. `MCP` Capability

**Source:** `pydantic_ai/capabilities/mcp.py`

`MCP` is the primary entry point for using MCP servers with pydantic-ai agents. It extends `NativeOrLocalTool` and accepts `url`, `native`, `local`, `authorization_token`, `headers`, and `allowed_tools`. The key design is the `local=` parameter: it accepts a URL string, `fastmcp.Client`, transport, in-process `FastMCP` server, script path, or pre-built `MCPToolset` — any non-URL, non-bool, non-string, non-`AbstractToolset`, non-callable input is auto-wrapped into an `MCPToolset`. Native MCP (`native=True`) requires a `url=`. The `from_spec()` classmethod restricts `local=` to JSON/YAML-serialisable types for `AgentSpec` compatibility.

```python
# Example 1 — HTTP MCP server with native + local dual mode
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        MCP(
            url='https://mcp.example.com/v1',
            native=False,       # Don't use provider-native MCP (not all providers support it)
            local=True,         # Use a local MCPToolset client to the same URL
            authorization_token='Bearer sk-...',
            allowed_tools=['search', 'lookup', 'fetch'],  # Filter to specific tools
        )
    ],
)
# The MCPToolset connects to the URL, fetches the tool schema, and applies
# the allowed_tools filter. Only 'search', 'lookup', 'fetch' are exposed.
# result = await agent.run("Search for recent news about agents")
```

```python
# Example 2 — In-process FastMCP server (local-only, no URL needed)
import asyncio
# from fastmcp import FastMCP  # requires pip install fastmcp
# from pydantic_ai import Agent
# from pydantic_ai.capabilities import MCP

# Hypothetical in-process FastMCP server:
# mcp_app = FastMCP("tools")

# @mcp_app.tool()
# def calculator(expression: str) -> float:
#     return eval(expression)  # simplified for example

# agent = Agent(
#     'openai:gpt-5.2',
#     capabilities=[
#         MCP(local=mcp_app)   # Non-URL input auto-wrapped into MCPToolset
#     ],
# )
# result = await agent.run("What is 2 + 2?")

# Without fastmcp installed, use MCPToolset directly:
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

# native=False (default) + local=True requires url= for URL-based derivation
# For non-URL clients, pass the toolset/client directly as local=
print("MCP capability supports in-process FastMCP servers via local=<server_instance>")
```

```python
# Example 3 — YAML/JSON spec via from_spec() (AgentSpec compatible)
from pydantic_ai.capabilities.mcp import MCP

# from_spec() restricts local= to JSON/YAML-serialisable types (str | bool | None)
# so AgentSpec round-trips work correctly.
cap = MCP.from_spec(
    url='https://tools.internal/mcp',
    native=False,
    local=True,                           # str | bool | None ✓
    authorization_token='sk-internal',
    headers={'X-Team': 'platform'},
    allowed_tools=['code_search', 'doc_lookup'],
    defer_loading=True,
)

# Equivalent YAML representation for AgentSpec:
# capabilities:
#   - type: MCP
#     url: https://tools.internal/mcp
#     local: true
#     authorization_token: sk-internal
#     headers:
#       X-Team: platform
#     allowed_tools: [code_search, doc_lookup]
#     defer_loading: true
print(f"MCP capability from spec: url={cap.url}, local={cap.local}")
```

---

## 10. `PrefixTools` + `ThreadExecutor` Capabilities

**Sources:** `pydantic_ai/capabilities/prefix_tools.py`, `pydantic_ai/capabilities/thread_executor.py`

### `PrefixTools`

`PrefixTools` wraps another capability and prefixes all its tool names with `{prefix}_`. It delegates to `PrefixedToolset` for `AbstractToolset` outputs and wraps callable toolset functions in a `DynamicToolset` first so `PrefixedToolset` can operate on it uniformly. Use it to namespace tools from different MCP servers or toolsets when their tool names collide. The `from_spec()` classmethod accepts a `capability` argument in any CapabilitySpec short-form for YAML/JSON composition.

### `ThreadExecutor`

`ThreadExecutor` replaces anyio's ephemeral per-call threads with a bounded `concurrent.futures.Executor` for all sync tool functions, output validators, and other sync callbacks within a run. This prevents thread accumulation in long-running FastAPI servers. Use `Agent.using_thread_executor()` to set it globally for all agents, or inject it per-run via `capabilities=`.

```python
# Example 1 — PrefixTools: namespace two MCP server tool sets
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, MCP

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        PrefixTools(
            wrapped=MCP(url='https://search.api/mcp', local=True),
            prefix='search',     # search_web, search_code, search_docs
        ),
        PrefixTools(
            wrapped=MCP(url='https://storage.api/mcp', local=True),
            prefix='store',      # store_read, store_write, store_delete
        ),
    ],
)
# No tool name collisions even if both MCP servers export 'read' and 'write'.
# result = await agent.run("Search for examples and store the results")
```

```python
# Example 2 — ThreadExecutor: bounded thread pool for production FastAPI
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

# Shared executor across all requests (created once at startup)
executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix='agent-worker')

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[ThreadExecutor(executor=executor)],
)

# Or set globally for ALL agents using the class-level context manager:
# with Agent.using_thread_executor(executor):
#     # All agents in this block use the shared executor
#     result = await agent.run("...")

# result = await agent.run("Process this data synchronously")
```

```python
# Example 3 — Combine PrefixTools and ThreadExecutor in one agent
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, ThreadExecutor
from pydantic_ai.toolsets import FunctionToolset

executor = ThreadPoolExecutor(max_workers=16)

toolset_a = FunctionToolset()
toolset_b = FunctionToolset()

@toolset_a.tool
def read_file(path: str) -> str:
    """Read a local file (sync I/O)."""
    # In production, this runs in the bounded ThreadPoolExecutor
    return f"Contents of {path}"

@toolset_b.tool
def write_file(path: str, content: str) -> bool:
    """Write a file (sync I/O)."""
    return True

from pydantic_ai.capabilities import Toolset

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        PrefixTools(wrapped=Toolset(toolset_a), prefix='disk'),  # disk_read_file
        PrefixTools(wrapped=Toolset(toolset_b), prefix='out'),   # out_write_file
        ThreadExecutor(executor=executor),                        # bounded threads
    ],
)
# result = await agent.run("Read config.json and write the transformed version to output.json")
```

---

## Summary

| Class / API | Source | Key role |
|---|---|---|
| `RaiseContentFilterError` | `capabilities/content_filter.py` | Strict opt-in: `finish_reason='content_filter'` → `ContentFilterError` |
| `ResolveModelId` | `capabilities/resolve_model_id.py` | Sync/async model ID override; `None` = passthrough |
| `SelectModel` | `capabilities/select_model.py` | Per-step `ModelSelector` callable; cost/usage-aware routing |
| `HandleDeferredToolCalls` | `capabilities/deferred_tool_handler.py` | Inline deferred approval; `None` → next handler |
| `PendingMessageDrainCapability` | `capabilities/_pending_messages.py` | Auto-injected `'asap'`/`'when_idle'` drain + `EnqueuedMessagesEvent` |
| `XSearch` | `capabilities/x_search.py` | xAI native + `fallback_model`; handle/date/multimedia filters |
| `ImageGeneration` | `capabilities/image_generation.py` | 12-field config + `fallback_model` subagent; `_image_gen_kwargs()` bridge |
| `TavilySearchTool` | `common_tools/tavily.py` | `_UNSET` sentinel + `partial.__signature__` freeze pattern |
| `MCP` | `capabilities/mcp.py` | Primary MCP entry: url/native/local; auth/headers; `from_spec()` |
| `PrefixTools` + `ThreadExecutor` | `capabilities/prefix_tools.py`, `capabilities/thread_executor.py` | Tool namespace isolation; bounded sync-function thread pool |

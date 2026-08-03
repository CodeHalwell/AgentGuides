---
title: "PydanticAI Class Deep Dives Vol. 40"
description: "Source-verified deep dives into 10 pydantic-ai 2.22.0 class groups: MCPToolset (prefer_tasks + direct_call_tool + use_task — durable background MCP calls, skip-optional semantics, full constructor reference), UsageLimits (per_request_input_tokens_limit + count_tokens_before_request — per-call context cap independent of cumulative limits), ModelHTTPError (headers + retry_after enrichment — lowercased headers dict, RFC-7231 Retry-After parsing, rate-limit guard pattern), OpenAIResponsesModelSettings (openai_reasoning_context + openai_reasoning_mode — all_turns/current_turn/auto context threading, standard vs pro reasoning, send_reasoning_ids history-safe replay), Gemini VALIDATED tool mode (strict=True/False/None per-tool and globally — AUTO fallback, GoogleModelProfile, InlineDefsJsonSchemaTransformer), Anthropic mid-conversation SystemPromptPart (native system-message injection via ctx.enqueue — provider-aware placement, fallback tagged-user rendering, cache-prefix preservation), AnthropicCompaction (server-side context compaction capability — token_threshold 50k–150k, custom instructions, pause_after_compaction, CompactionPart replay), MCPToolset sampling (sampling_model/sampling_handler/set_mcp_sampling_model — MCP server-driven LLM calls, SamplingHandler Protocol, client-pays-for-server-AI pattern), RunContext.is_tool_available (2.22.0 — name-form vs definition-form lookup, model-request hook timing, capability-reveal state awareness), MCPToolset.process_tool_call + CallToolFunc (intercept and mutate every MCP tool call — metadata injection, ctx-aware retry logic, audit logging, full signature reference)."
sidebar:
  label: "Class deep dives (Vol. 40)"
  order: 66
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.22.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.22.x API. Three examples per class group; all code blocks pass `ast.parse()` syntax validation. Statements that make live provider API calls are commented out or placed inside a `main()` — uncomment or call `asyncio.run(main())` to run them.
</Aside>

Ten class groups spanning the 2.19.0–2.22.0 release window: durable background MCP task execution (`MCPToolset.prefer_tasks` + `direct_call_tool`), per-request context-size budgets (`UsageLimits.per_request_input_tokens_limit`), enriched HTTP error surfaces (`ModelHTTPError.headers` + `retry_after`), OpenAI Responses API reasoning configuration (`OpenAIResponsesModelSettings`), Gemini schema-validated function calling (strict `VALIDATED` mode), Anthropic mid-conversation system messages (`SystemPromptPart` enqueue), server-side context compaction (`AnthropicCompaction`), MCP sampling (server-driven LLM calls via the client), the new `RunContext.is_tool_available` guard, and the `MCPToolset.process_tool_call` intercept hook.

---

## 1. `MCPToolset` — `prefer_tasks`, `direct_call_tool`, `use_task`

**Source:** `pydantic_ai/mcp.py`

`MCPToolset` connects a pydantic-ai agent to an MCP server. In **2.22.0** it gained `prefer_tasks` (default `True`): when the server marks a tool with `taskSupport='optional'`, the client sends the call wrapped in a durable MCP task (SEP-1686), making it cancellable and pollable. Set `prefer_tasks=False` to skip task-wrapped execution for tools whose task support is declared as `optional`; tools with `taskSupport='required'` still run as durable tasks regardless of this setting. The companion method `direct_call_tool(name, args, *, metadata, use_task)` lets you invoke a server tool outside of any agent run — useful for health checks, pre-flight validation, and batch jobs.

```python
# Example 1 — prefer_tasks=False: always use standard execution (no background tasks)
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# By default prefer_tasks=True sends optional-task tools as durable tasks.
# prefer_tasks=False skips task execution only for tools with optional task support;
# tools declared as taskSupport='required' still run as durable tasks.
toolset = MCPToolset(
    'http://localhost:8000/mcp',
    prefer_tasks=False,
)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

# async def main():
#     result = await agent.run('Summarise the sales report.')
#     print(result.output)
```

```python
# Example 2 — prefer_tasks=True (default) with a long-running optional-task tool
from fastmcp import FastMCP
from fastmcp.server.tasks import TaskConfig
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# --- Server side (separate process) ---
mcp = FastMCP('research_server')

@mcp.tool(task=TaskConfig(mode='optional'))
async def deep_research(topic: str) -> str:
    """Long-running research task — server wraps it in a durable MCP task."""
    import asyncio
    await asyncio.sleep(0)          # simulate real work
    return f'Research complete for: {topic}'

# --- Client side ---
toolset = MCPToolset('http://localhost:8000/mcp')   # prefer_tasks=True by default
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

# With prefer_tasks=True the agent sends the call with task=True per MCP SEP-1686.
# The server wraps execution in a cancellable, pollable task; client awaits completion.
# async def main():
#     result = await agent.run('Research quantum computing trends.')
#     print(result.output)
```

```python
# Example 3 — direct_call_tool: call an MCP tool without an agent run
import asyncio
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset('http://localhost:8000/mcp')

async def health_check() -> bool:
    """Call the server's echo tool directly to verify connectivity."""
    async with toolset:                    # opens the MCP connection
        result = await toolset.direct_call_tool(
            name='echo',
            args={'message': 'ping'},
            metadata={'source': 'health-check'},
            use_task=False,                # override prefer_tasks for this call
        )
    return result is not None

# asyncio.run(health_check())
```

---

## 2. `UsageLimits` — `per_request_input_tokens_limit` + `count_tokens_before_request`

**Source:** `pydantic_ai/usage.py`

`UsageLimits` controls how many tokens and requests an agent run may consume. **2.21.0** added `per_request_input_tokens_limit`: unlike `input_tokens_limit` (cumulative across all requests in a run), this field caps each individual request's context size independently. It is useful when prompt caching is active — a large cached prefix still counts toward the per-request limit, preventing runaway context growth even when cache hits are cheap. The companion flag `count_tokens_before_request` (default `False`) makes pydantic-ai call the model's token-counting API before dispatching, enforcing both `input_tokens_limit` and `per_request_input_tokens_limit` ahead of time at the cost of an extra API round-trip.

```python
# Example 1 — per_request_input_tokens_limit raises UsageLimitExceeded on oversized context
import asyncio
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-6')

async def main():
    try:
        result = await agent.run(
            'What is the capital of Italy? Answer with just the city.',
            usage_limits=UsageLimits(per_request_input_tokens_limit=10),
        )
        print(result.output)
    except UsageLimitExceeded as exc:
        print(exc)
        # Exceeded the per_request_input_tokens_limit of 10 (request_input_tokens=62).

# asyncio.run(main())
```

```python
# Example 2 — count_tokens_before_request prevents the request from ever being sent
import asyncio
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('openai:gpt-5.2')

async def main():
    # count_tokens_before_request=True calls the token-counting API first.
    # The request is rejected before being dispatched — no billing for the blocked call.
    try:
        result = await agent.run(
            'Explain the history of the Roman Empire in detail.',
            usage_limits=UsageLimits(
                per_request_input_tokens_limit=20,
                count_tokens_before_request=True,
            ),
        )
    except UsageLimitExceeded as exc:
        print(f'Blocked before dispatch: {exc}')

# asyncio.run(main())
```

```python
# Example 3 — combining per-request and cumulative limits for multi-turn budget control
import asyncio
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.usage import RunUsage
from pydantic_ai.messages import ModelMessage

agent = Agent('anthropic:claude-opus-5-20260729')

async def multi_turn_session() -> None:
    usage = RunUsage()
    history: list[ModelMessage] = []
    questions = [
        'What is the boiling point of water?',
        'And in Fahrenheit?',
        'How is Celsius converted to Fahrenheit?',
    ]
    for q in questions:
        result = await agent.run(
            q,
            message_history=history,
            usage=usage,                              # accumulates across turns
            usage_limits=UsageLimits(
                per_request_input_tokens_limit=8_000,  # each request ≤ 8k input tokens
                input_tokens_limit=20_000,             # entire session ≤ 20k cumulative
                request_limit=10,
            ),
        )
        history = result.all_messages()
        print(f'Q: {q}')
        print(f'A: {result.output}')
        print(f'Cumulative input tokens: {usage.input_tokens}')

# asyncio.run(multi_turn_session())
```

---

## 3. `ModelHTTPError` — `headers` + `retry_after`

**Source:** `pydantic_ai/exceptions.py`

`ModelHTTPError` is raised when a provider returns a 4xx or 5xx response. **2.19.0** added two new fields: `headers` (`dict[str, str] | None`) — the full response headers with keys lowercased — and the derived property `retry_after` (`float | None`) which parses the RFC-7231 `Retry-After` header (both delta-seconds and HTTP-date formats) from `headers`. Both fields are propagated by every built-in provider (Anthropic, OpenAI, Google, Bedrock, etc.); `headers` may be `None` for gRPC-based providers such as the xAI gRPC path.

```python
# Example 1 — read retry_after from a 429 rate-limit error
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

agent = Agent('openai:gpt-5.2')

async def run_with_rate_limit_handling(prompt: str) -> str:
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            result = await agent.run(prompt)
            return result.output
        except ModelHTTPError as exc:
            # Only sleep when a subsequent attempt will actually be made.
            if exc.status_code == 429 and attempt + 1 < max_attempts:
                wait = exc.retry_after if exc.retry_after is not None else 5.0
                print(f'Rate limited. Waiting {wait}s (attempt {attempt + 1}/{max_attempts})')
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError('Max retry attempts exceeded')

# asyncio.run(run_with_rate_limit_handling('Hello'))
```

```python
# Example 2 — inspect all headers for debugging / provider-specific fields
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

agent = Agent('anthropic:claude-sonnet-4-6')

def diagnose_http_error(exc: ModelHTTPError) -> dict:
    """Extract diagnostic information from a failed provider call."""
    info: dict = {
        'status_code': exc.status_code,
        'model': exc.model_name,
        'retry_after': exc.retry_after,
    }
    if exc.headers:
        # Anthropic exposes request-id in 'x-request-id'; keys are lowercased
        info['request_id'] = exc.headers.get('x-request-id')
        info['cf_ray'] = exc.headers.get('cf-ray')         # Cloudflare trace ID
        info['content_type'] = exc.headers.get('content-type')
    return info

# try:
#     agent.run_sync('...')
# except ModelHTTPError as exc:
#     print(diagnose_http_error(exc))
```

```python
# Example 3 — use AsyncTenacityTransport for smart Retry-After-aware retries
import httpx
from pydantic_ai import Agent
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import stop_after_attempt, wait_exponential

# RetryConfig is a TypedDict — pass it directly to AsyncTenacityTransport.
# wait_retry_after honours the Retry-After header when present,
# falling back to exponential back-off when it is absent (keyword: fallback_strategy).
retry_config: RetryConfig = {
    'stop': stop_after_attempt(5),
    'wait': wait_retry_after(fallback_strategy=wait_exponential(multiplier=1, min=2, max=60)),
    'reraise': True,
}

def _raise_if_transient(r: httpx.Response) -> None:
    """Raise only for retriable responses (429, 5xx); let permanent 4xx pass through."""
    if r.status_code == 429 or r.status_code >= 500:
        r.raise_for_status()

# validate_response makes Tenacity observe transient failures.
# Narrowing to 429 + 5xx avoids retrying permanent errors like 400 or 401.
transport = AsyncTenacityTransport(
    config=retry_config,
    validate_response=_raise_if_transient,
)
http_client = httpx.AsyncClient(transport=transport)

provider = OpenAIProvider(http_client=http_client)
model = OpenAIModel('gpt-5.2', provider=provider)
agent = Agent(model)

# The agent now automatically waits the server-specified Retry-After duration on 429s.
# result = agent.run_sync('Hello')
```

---

## 4. `OpenAIResponsesModelSettings` — reasoning context, mode, and replay

**Source:** `pydantic_ai/models/openai.py`

`OpenAIResponsesModelSettings` controls the OpenAI Responses API (`/v1/responses`) for reasoning models such as `gpt-5.6-sol`. Key fields added or documented in **2.20.0**: `openai_reasoning_context` (`'auto' | 'current_turn' | 'all_turns'`) — which prior-turn reasoning items the model sees. pydantic-ai defaults to `'all_turns'` on supported models to maintain continuity, though this raises input-token counts. `openai_reasoning_mode` (`'standard' | 'pro'`) — `'pro'` increases reliability for complex tasks at the cost of higher latency and token use. `openai_send_reasoning_ids` (`bool`) — when `False`, reasoning part IDs are stripped from the message history sent to the model, preventing history-mismatch errors when a custom `ProcessHistory` removes thinking parts.

```python
# Example 1 — all_turns reasoning context for full cross-turn continuity
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='high',
    openai_reasoning_context='all_turns',   # model sees reasoning from all prior turns
    openai_reasoning_summary='detailed',    # optional: human-readable reasoning summary
)
agent = Agent(model, model_settings=settings)

# async def main():
#     result = await agent.run('Plan a week-long itinerary for Japan.')
#     print(result.output)
```

```python
# Example 2 — pro reasoning mode for high-stakes tasks
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')

# 'pro' mode is independent of reasoning_effort — it activates an enhanced
# reasoning backend. Use for tasks where correctness matters more than speed.
settings = OpenAIResponsesModelSettings(
    openai_reasoning_mode='pro',
    openai_reasoning_effort='high',
)
agent = Agent(model, model_settings=settings)

# async def main():
#     result = await agent.run('Verify this proof of the Pythagorean theorem.')
#     print(result.output)
```

```python
# Example 3 — send_reasoning_ids=False for safe custom history processing
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')

def strip_thinking_parts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove ThinkingPart objects from history before each request."""
    from pydantic_ai.messages import ModelResponse, ThinkingPart
    result = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            cleaned_parts = [p for p in msg.parts if not isinstance(p, ThinkingPart)]
            result.append(ModelResponse(parts=cleaned_parts, model_name=msg.model_name))
        else:
            result.append(msg)
    return result

agent = Agent(
    model,
    capabilities=[ProcessHistory(strip_thinking_parts)],
    model_settings=OpenAIResponsesModelSettings(
        # Strip reasoning IDs so the model doesn't expect matching thinking parts
        openai_send_reasoning_ids=False,
        openai_reasoning_effort='medium',
        openai_reasoning_context='current_turn',
    ),
)

# async def main():
#     result = await agent.run('Break down this maths problem step by step.')
#     print(result.output)
```

---

## 5. Gemini `VALIDATED` tool mode — `strict` flag + `GoogleModelProfile`

**Source:** `pydantic_ai/models/google.py`, `pydantic_ai/profiles/google.py`

Google Gemini 2.5+ supports a `VALIDATED` function-calling mode which enforces schema adherence server-side, equivalent to OpenAI's strict tool definitions. pydantic-ai **2.22.0** activates this by default on supported Gemini models (`google_model_profile` sets `supports_strict_tool_definition=True`). The `strict` parameter on `Tool`, `@agent.tool`, and `@agent.tool_plain` controls this per-tool: `True` forces `VALIDATED`, `False` forces `AUTO` mode (permissive, for schemas that can't be made strict), and `None` (default) defers to the model profile. The schema transformer (`InlineDefsJsonSchemaTransformer`) rewrites `$ref` / `$defs` patterns into inline equivalents before sending to Gemini.

```python
# Example 1 — VALIDATED mode enabled by default on Gemini 2.5+
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

class FlightSearch(BaseModel):
    origin: str
    destination: str
    date: str
    cabin_class: str = 'economy'

agent = Agent(GoogleModel('gemini-2.5-flash'), output_type=FlightSearch)

# gemini-2.5-flash uses VALIDATED function-calling mode by default via GoogleModelProfile.
# This guarantees the response schema matches FlightSearch — no silent field omissions.
# result = agent.run_sync('Find a flight from London to Tokyo on 2026-12-01.')
# print(result.output)  # FlightSearch(origin='London', destination='Tokyo', ...)
```

```python
# Example 2 — strict=False forces AUTO mode for a schema VALIDATED can't handle
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.google import GoogleModel

def dynamic_query(query: str, filters: dict) -> str:
    """Run a dynamic search query with arbitrary filter keys.

    Args:
        query: The search query string.
        filters: Arbitrary key-value filter pairs (open schema).
    """
    return f'Results for {query} with {filters}'

# `dict` with arbitrary keys can't be represented in VALIDATED strict mode.
# strict=False on any tool causes the *entire request* to fall back to AUTO —
# all tools share one Gemini function-calling mode per request.
tool = Tool(dynamic_query, strict=False)

agent = Agent(
    GoogleModel('gemini-2.5-pro'),
    tools=[tool],
)
# result = agent.run_sync('Search for Python books published after 2024.')
# print(result.output)
```

```python
# Example 3 — request-level mode: all-strict keeps VALIDATED; any strict=False forces AUTO
#
# Gemini's function_calling_config mode is a single request-level setting, not
# per-tool.  pydantic-ai selects VALIDATED when every tool in the request uses
# strict=True (or strict=None with a compatible schema) and falls back to AUTO
# the moment any tool uses strict=False.  You cannot mix modes in one agent run.
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

# Agent A: all tools strict=True → Gemini uses VALIDATED mode for the whole request.
agent_validated = Agent(GoogleModel('gemini-2.5-flash'))

@agent_validated.tool_plain(strict=True)
def get_weather_data(city: str) -> dict:
    """Fetch live weather data for a city.

    Args:
        city: The city name to look up.
    """
    return {'city': city, 'temp_c': 22.5, 'cond': 'sunny', 'humidity': 58}

@agent_validated.tool_plain(strict=True)
def get_forecast(city: str, days: int) -> list:
    """Return a multi-day weather forecast.

    Args:
        city: The city name to look up.
        days: Number of forecast days (1-7).
    """
    return [{'day': i + 1, 'temp_c': 20.0 + i, 'cond': 'sunny'} for i in range(days)]

# Agent B: one tool uses strict=False → entire request falls back to AUTO mode.
agent_auto = Agent(GoogleModel('gemini-2.5-flash'))

@agent_auto.tool_plain(strict=False)    # open schema forces the request into AUTO
def log_metadata(metadata: dict) -> str:
    """Log arbitrary metadata — open-ended schema incompatible with VALIDATED mode.

    Args:
        metadata: Arbitrary key-value metadata to record.
    """
    print(f'Logged: {metadata}')
    return 'logged'

# result = agent_validated.run_sync('Weather in Paris for 3 days?')
# result = agent_auto.run_sync('Log some diagnostics.')
```

---

## 6. Anthropic mid-conversation `SystemPromptPart` via `ctx.enqueue`

**Source:** `pydantic_ai/messages.py`, `pydantic_ai/models/anthropic.py`

Anthropic models support native mid-conversation system messages — instructions injected into the conversation history rather than at the initial system prompt position. This preserves cached prefixes (the initial system prompt stays unchanged) while dynamically adjusting the model's behaviour. In **2.22.0** pydantic-ai routes `SystemPromptPart` objects enqueued with `ctx.enqueue()` through Anthropic's native system-in-conversation format when supported; it falls back to a tagged user-channel message (e.g., `<system>…</system>`) on older models or non-Anthropic providers. Placement is automatically adjusted to satisfy Anthropic's requirement that system messages sit between a user turn and the model's reply.

```python
# Example 1 — inject a system instruction mid-conversation from a tool
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart

agent = Agent(
    'anthropic:claude-opus-4-8',
    system_prompt='You are a senior code reviewer. Be thorough but constructive.',
)

@agent.tool
def require_type_annotations(ctx: RunContext[None]) -> str:
    """Signal that all suggestions must include type annotations."""
    # Enqueue a system message — Anthropic places it natively in the conversation,
    # preserving the initial system prompt's cache prefix.
    ctx.enqueue(SystemPromptPart(
        content='Every code suggestion MUST include explicit Python type annotations.'
    ))
    return 'Rule added: type annotations required.'

# async def main():
#     result = await agent.run(
#         'Review this function: def add(a, b): return a + b'
#     )
#     print(result.output)
```

```python
# Example 2 — incident mode: switch agent persona mid-conversation
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    system_prompt='You are an operations assistant. Monitor systems and guide the team.',
)

@agent.tool
def declare_incident(ctx: RunContext[None], severity: str) -> str:
    """Declare a production incident and switch the agent to incident mode.

    Args:
        severity: Incident severity level (P1, P2, P3).
    """
    ctx.enqueue(SystemPromptPart(
        content=f'INCIDENT DECLARED (severity={severity}). '
                'Switch to incident mode: be terse, action-oriented, and prioritise triage. '
                'Do not offer background explanations unless asked.'
    ))
    return f'Incident mode activated at severity {severity}.'

@agent.tool
def resolve_incident(ctx: RunContext[None]) -> str:
    """Mark the incident resolved and restore normal operating mode."""
    ctx.enqueue(SystemPromptPart(
        content='Incident resolved. Return to normal operations mode: '
                'be thorough and educational in your responses.'
    ))
    return 'Incident resolved. Normal mode restored.'

# async def main():
#     result = await agent.run('Production database is down!')
#     print(result.output)
```

```python
# Example 3 — cross-provider fallback: SystemPromptPart works on all providers
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart

# On Anthropic: native mid-conversation system message (cache-prefix-safe).
# On OpenAI / Google / others: rendered as a tagged user-channel message.
# The application code is identical either way.
def make_agent(model_name: str) -> Agent:
    agent: Agent = Agent(
        model_name,
        system_prompt='You are a helpful assistant.',
    )

    @agent.tool
    def escalate_to_formal(ctx: RunContext[None]) -> str:
        """Switch the response style to formal/professional.

        No arguments required.
        """
        ctx.enqueue(SystemPromptPart(
            content='The user has requested a formal response. '
                    'Use professional language, avoid contractions, '
                    'and structure your reply with clear headings.'
        ))
        return 'Formal mode enabled.'

    return agent

# anthropic_agent = make_agent('anthropic:claude-sonnet-4-6')
# openai_agent   = make_agent('openai:gpt-5.2')
# Both agents enqueue SystemPromptPart identically; rendering differs per provider.
```

---

## 7. `AnthropicCompaction` — server-side context compaction

**Source:** `pydantic_ai/models/anthropic.py`

`AnthropicCompaction` is an Anthropic-native capability that automatically summarises older conversation history once the input token count exceeds `token_threshold`. The summary is inserted as a `CompactionPart` in the message history, which Anthropic's API then replays on subsequent requests. Optional `instructions` steer the summary generator (e.g., "always preserve all tool call results"). Setting `pause_after_compaction=True` stops the agent run after a compaction block is produced, letting you inspect or save the compacted history before continuing. The threshold must be between 50,000 and 150,000 tokens.

```python
# Example 1 — basic compaction with a token threshold
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[AnthropicCompaction(token_threshold=100_000)],
)

# Once the conversation context exceeds 100k input tokens, Anthropic automatically
# compacts older messages into a summary. The summary is stored as a CompactionPart
# and replayed on the next request.
# async def main():
#     result = await agent.run('Summarise everything we have discussed so far.')
#     print(result.output)
```

```python
# Example 2 — custom summary instructions to preserve key data
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction

agent = Agent(
    'anthropic:claude-opus-5-20260729',
    capabilities=[AnthropicCompaction(
        token_threshold=80_000,
        instructions=(
            'When compacting, always preserve: '
            '1) All tool call results verbatim. '
            '2) Any user-provided data or figures. '
            '3) The most recent 5 assistant turns in full. '
            'Summarise earlier assistant reasoning concisely.'
        ),
    )],
)

# The custom instructions guide Anthropic's compaction model to keep
# the most important parts of the conversation intact.
# async def main():
#     result = await agent.run('What numbers did we calculate earlier?')
#     print(result.output)
```

```python
# Example 3 — pause_after_compaction to snapshot compacted history
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction
from pydantic_ai.messages import ModelMessage, CompactionPart

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[AnthropicCompaction(
        token_threshold=60_000,
        pause_after_compaction=True,   # stop after compaction so we can inspect it
    )],
)

async def run_with_compaction_snapshot(prompt: str) -> list[ModelMessage]:
    """Run the agent; if compaction fires, save a snapshot and resume."""
    result = await agent.run(prompt)
    messages = result.all_messages()

    # Check if any CompactionPart was produced in the history
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:                          # type: ignore[union-attr]
                if isinstance(part, CompactionPart):
                    print(f'Compaction triggered! Summary length: {len(part.content)} chars')
                    # Save the compacted history for durable storage
                    # with open('compacted_history.json', 'w') as f:
                    #     import json; json.dump([m.model_dump() for m in messages], f)
    return messages

# asyncio.run(run_with_compaction_snapshot('Continue our analysis...'))
```

---

## 8. `MCPToolset` sampling — `sampling_model`, `sampling_handler`, `set_mcp_sampling_model`

**Source:** `pydantic_ai/mcp.py`

MCP sampling lets an MCP server request LLM completions from the connected client — the server drives AI calls without needing its own API key. The client pays for and controls all LLM usage. In pydantic-ai, sampling is configured on `MCPToolset` via `sampling_model` (a `Model` instance the client will use to fulfil sampling requests) or `sampling_handler` (a `SamplingHandler` callable for full custom control). The convenience method `agent.set_mcp_sampling_model()` wires the agent's own model into every attached `MCPToolset` — the server gets the same LLM the agent is using.

```python
# Example 1 — agent.set_mcp_sampling_model(): server uses the agent's own model
import asyncio
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(StdioTransport(command='python', args=['svg_generator.py']))
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    # Wire the agent's model as the sampling model for all attached toolsets.
    agent.set_mcp_sampling_model()
    # The MCP server can now call ctx.session.create_message(...)
    # and pydantic-ai will fulfil it using gpt-5.2.
    result = await agent.run('Generate an SVG illustration of a mountain sunset.')
    print(result.output)

# asyncio.run(main())
```

```python
# Example 2 — sampling_model: use a different model for server-side LLM calls
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.anthropic import AnthropicModel

# The agent uses GPT-5, but sampling requests from the MCP server are
# fulfilled by claude-haiku-4-5 (cheaper, faster for simple sub-tasks).
toolset = MCPToolset(
    StdioTransport(command='python', args=['data_pipeline.py']),
    sampling_model=AnthropicModel('claude-haiku-4-5-20251001'),
)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

# async def main():
#     result = await agent.run('Process the Q3 sales data.')
#     print(result.output)
```

```python
# Example 3 — sampling_handler: full custom control over sampling requests
import asyncio
from typing import Any
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

async def audit_sampling_handler(
    messages: Any,
    model_preferences: Any,
    system_prompt: str | None,
    include_context: str,
    temperature: float | None,
    max_tokens: int,
    stop_sequences: list[str] | None,
    metadata: dict | None,
    model_hint: str | None,
) -> Any:
    """Log every sampling request and fulfil it with a custom model selection."""
    print(f'MCP sampling request: max_tokens={max_tokens}, hint={model_hint}')
    # In a real handler, call your preferred model here and return an
    # mcp.types.CreateMessageResult object.
    raise NotImplementedError('Replace with real model call')

toolset = MCPToolset(
    StdioTransport(command='python', args=['creative_server.py']),
    sampling_handler=audit_sampling_handler,   # takes full control of sampling
)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

---

## 9. `RunContext.is_tool_available` — runtime tool availability check

**Source:** `pydantic_ai/_run_context.py`

`RunContext.is_tool_available(tool: str | ToolDefinition) -> bool` was added in **2.22.0**. It answers "is this function tool currently visible to the model?" accounting for `FilteredToolset`, `PrepareTools`, `DeferredLoadingToolset`, and other capability-level mutations. The name-form (`str`) looks up the tool in the run's internal tool registry — most reliable when called from a model-request hook or during another tool's execution. The definition-form (`ToolDefinition`) checks against the fully-resolved list after all capability mutations, and remains accurate even when a wrapping toolset has removed a definition.

```python
# Example 1 — guard a tool that depends on another tool being available
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5.2')

@agent.tool
def analyse_data(ctx: RunContext[None]) -> str:
    """Analyse uploaded dataset and produce summary statistics."""
    return 'count=1000, mean=42.5, std=7.2'

@agent.tool
def export_results(ctx: RunContext[None], format: str) -> str:
    """Export results to a file. Requires analyse_data to be available.

    Args:
        format: Output format, e.g. 'csv' or 'json'.
    """
    if not ctx.is_tool_available('analyse_data'):
        return 'Error: export_results requires analyse_data to be available in this run.'
    return f'Results exported as {format}.'

# async def main():
#     result = await agent.run('Analyse the dataset and export as CSV.')
#     print(result.output)
```

```python
# Example 2 — is_tool_available inside a before_model_request hook
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelRequest

hooks = Hooks()

@hooks.before_model_request
async def log_dangerous_tools(ctx, request: ModelRequest) -> None:
    """Warn if high-privilege tools are exposed to the model."""
    if ctx.is_tool_available('send_email'):
        print('WARNING: send_email is available to the model in this request.')

# Register the hook on the same agent that owns the tool so
# ctx.is_tool_available('send_email') reflects the actual tool list.
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[hooks],
)

@agent.tool_plain
def send_email(recipient: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        recipient: Email address.
        body: Email body text.
    """
    return f'Email sent to {recipient}.'

# async def main():
#     result = await agent.run('Draft and send a status report.')
#     print(result.output)
```

```python
# Example 3 — is_tool_available with a FilteredToolset for role-based access
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset()

@toolset.tool
def read_records(ctx: RunContext[dict]) -> list[dict]:
    """Fetch all database records (read-only)."""
    return [{'id': 1, 'name': 'Alice'}]

@toolset.tool
def delete_record(ctx: RunContext[dict], record_id: int) -> str:
    """Permanently delete a database record.

    Args:
        record_id: The ID of the record to delete.
    """
    # The FilteredToolset prevents non-admins from seeing this tool at all.
    # If the tool is executing, ctx.deps should already be admin — but we
    # double-check deps directly here rather than is_tool_available(), because
    # is_tool_available('delete_record') is always True inside a running tool call.
    if ctx.deps.get('role') != 'admin':
        return 'Permission denied: admin role required to delete records.'
    return f'Record {record_id} deleted.'

# Restrict delete_record to admin users via a filtered toolset
admin_toolset = toolset.filtered(
    lambda ctx, tool_def: (
        tool_def.name != 'delete_record' or ctx.deps.get('role') == 'admin'
    )
)

agent = Agent('openai:gpt-5.2', toolsets=[admin_toolset], deps_type=dict)

# async def main():
#     # Admin run — delete_record is visible
#     r1 = await agent.run('Delete record 42', deps={'role': 'admin'})
#     print(r1.output)
#     # Non-admin run — delete_record is filtered out
#     r2 = await agent.run('Delete record 42', deps={'role': 'viewer'})
#     print(r2.output)
```

---

## 10. `MCPToolset.process_tool_call` + `CallToolFunc`

**Source:** `pydantic_ai/mcp.py`

`process_tool_call` is an optional callback on `MCPToolset` that intercepts every tool call before it reaches the MCP server. Its signature is:

```python
async def process_tool_call(
    ctx: RunContext[DepsT],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult: ...
```

`call_tool` is the default executor — wrapping it lets you inject metadata, add retry logic, enforce access policies, or log audit trails. `ToolResult` is the raw server response; the callback can return a modified result or raise `ModelRetry` to signal failure.

```python
# Example 1 — inject RunContext deps into every tool call as metadata
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult
from pydantic_ai.models.test import TestModel

async def inject_user_metadata(
    ctx: RunContext[dict],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """Pass the current user's identity to every MCP tool call via _meta."""
    user_meta = {
        'user_id': ctx.deps.get('user_id'),
        'session_id': ctx.deps.get('session_id'),
        'run_id': str(ctx.run_id),
    }
    return await call_tool(name, tool_args, user_meta)

toolset = MCPToolset(
    'http://localhost:8000/mcp',
    process_tool_call=inject_user_metadata,
)
agent = Agent(
    model=TestModel(),
    deps_type=dict,
    toolsets=[toolset],
)

# async def main():
#     result = await agent.run(
#         'Fetch the report.',
#         deps={'user_id': 'u-123', 'session_id': 'sess-abc'},
#     )
#     print(result.output)
```

```python
# Example 2 — audit logging with timing for every MCP tool call
import asyncio
import time
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult

audit_log: list[dict] = []

async def timed_audit_processor(
    ctx: RunContext[None],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """Record timing and outcome for every MCP tool invocation."""
    start = time.monotonic()
    try:
        result = await call_tool(name, tool_args)
        elapsed = time.monotonic() - start
        audit_log.append({'tool': name, 'args': tool_args, 'elapsed_s': elapsed, 'ok': True})
        return result
    except Exception as exc:
        elapsed = time.monotonic() - start
        audit_log.append({'tool': name, 'args': tool_args, 'elapsed_s': elapsed, 'ok': False, 'error': str(exc)})
        raise

toolset = MCPToolset('http://localhost:8000/mcp', process_tool_call=timed_audit_processor)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

# async def main():
#     result = await agent.run('Run the data pipeline.')
#     print(result.output)
#     print('Audit log:', audit_log)
```

```python
# Example 3 — selective retry: re-attempt a specific tool on empty results
import json
from typing import Any
from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult

MAX_EMPTY_RETRIES = 2

def _payload_has_items(decoded: object) -> bool:
    """Unwrap common server response shapes and test for non-empty results.

    Handles bare lists ([...]) and wrapped dicts ({"results": [...]} etc.).
    Adapt to your server's actual schema in production.
    """
    if isinstance(decoded, list):
        return bool(decoded)
    if isinstance(decoded, dict):
        # Unwrap {"results": [...]} / {"items": [...]} / {"data": [...]} etc.
        for value in decoded.values():
            if isinstance(value, list):
                return bool(value)
        return False  # dict with no list values → treat as empty
    return bool(decoded)

def _has_results(result: ToolResult) -> bool:
    """Return True when any content block in the result contains search items.

    Inspects every block and unwraps common JSON wrapper shapes so that
    payloads like {"results": []} are correctly treated as empty even though
    the outer dict is truthy.  Binary blocks (ImageContent, BlobResourceContents)
    are treated as non-empty on sight.
    """
    for block in result.content:
        if isinstance(block, TextContent):
            try:
                if _payload_has_items(json.loads(block.text)):
                    return True
            except (json.JSONDecodeError, ValueError):
                if block.text.strip():
                    return True
        elif isinstance(block, EmbeddedResource) and isinstance(block.resource, TextResourceContents):
            try:
                if _payload_has_items(json.loads(block.resource.text)):
                    return True
            except (json.JSONDecodeError, ValueError):
                if block.resource.text.strip():
                    return True
        else:
            return True  # ImageContent / BlobResourceContents
    return False

async def retry_empty_results(
    ctx: RunContext[None],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """Retry search tools that return empty results up to MAX_EMPTY_RETRIES times."""
    result = await call_tool(name, tool_args)

    if name == 'search' and not _has_results(result):
        for attempt in range(1, MAX_EMPTY_RETRIES + 1):
            print(f'Empty results from {name}, retry {attempt}/{MAX_EMPTY_RETRIES}')
            result = await call_tool(name, tool_args)
            if _has_results(result):
                return result
        # Tell the model to try a different query
        raise ModelRetry(f'Tool "{name}" returned no results after {MAX_EMPTY_RETRIES} retries. '
                         'Try a different search query.')
    return result

toolset = MCPToolset('http://localhost:8000/mcp', process_tool_call=retry_empty_results)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

# async def main():
#     result = await agent.run('Search for papers on quantum error correction.')
#     print(result.output)
```

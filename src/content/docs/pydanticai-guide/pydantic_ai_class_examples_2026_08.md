---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2026-08)"
description: "Runnable, source-verified code examples for Agent, RunContext, UsageLimits, ToolReturn, DeferredToolRequests/Results, CachePoint, PrefixedToolset/FilteredToolset/RenamedToolset, WebSearchTool, FunctionToolset, ModelRetry, UnexpectedModelBehavior and ModelHTTPError."
framework: pydanticai
language: python
---

# 10 Source-Verified Class Deep Dives

Verified against **pydantic-ai==2.33.0** (installed `pydantic_ai/__init__.py` — `Agent`, `RunContext`, `UsageLimits`, `ToolReturn`, `CachePoint`, `DeferredToolRequests`, `DeferredToolResults`, `PrefixedToolset`, `FilteredToolset`, `RenamedToolset`, `WebSearchTool`, `FunctionToolset`, `ModelRetry`, `UnexpectedModelBehavior`, `ModelHTTPError`).

This page focuses on **runnable code**. Every field and method mentioned below is taken directly from the installed package. Each section shows the real dataclass fields, then a runnable example that exercises them.

Install and confirm the version first:

```bash
pip install "pydantic-ai==2.33.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.33.0
```

---

## 1. `Agent` — the full constructor surface

`pydantic_ai.agent.Agent` is the entrypoint. The **complete** constructor signature at 2.33.0 is:

```python
Agent(
    model=None,
    *,
    output_type=str,
    instructions=None,
    system_prompt=(),
    deps_type=object,
    name=None,
    description=None,
    model_settings=None,
    retries=None,
    validation_context=None,
    tools=(),
    toolsets=None,
    defer_model_check=False,
    end_strategy='graceful',
    metadata=None,
    tool_timeout=None,
    max_concurrency=None,
    capabilities=None,
)
```

`Agent.run`, `Agent.run_sync` and `Agent.run_stream` add a further set of per-call overrides. The ones you reach for most often:

| Argument              | Purpose                                                                      |
| --------------------- | ---------------------------------------------------------------------------- |
| `user_prompt`         | Positional prompt (or `Sequence[UserContent]` for multimodal).               |
| `output_type`         | Overrides the agent's static `output_type` for this call.                    |
| `message_history`     | Prior `ModelMessage`s to prepend to the run.                                 |
| `deferred_tool_results` | Complete a paused run whose model had emitted `DeferredToolRequests`.       |
| `conversation_id`     | Groups runs into a conversation (auto-derived from history when omitted).    |
| `deps`                | Instance of `deps_type`, injected into `RunContext.deps`.                    |
| `model_settings`      | Per-call `ModelSettings` override.                                           |
| `usage_limits`        | See `UsageLimits` below.                                                     |
| `cancellation_token`  | Cooperative cancellation — pair with `agent.run(...)` in a task.             |
| `event_stream_handler`| Callback for streamed events without opening a `run_stream` context.         |

### Runnable example — everything wired at once

```python
from decimal import Decimal
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits, ModelSettings


class Ticket(BaseModel):
    id: int
    subject: str
    priority: str


agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Ticket,
    deps_type=dict,
    instructions='Convert the user request into a support ticket.',
    model_settings=ModelSettings(temperature=0.0, max_tokens=400),
    retries=2,
    tool_timeout=15.0,
    metadata={'service': 'support-desk'},
)


@agent.tool
def user_profile(ctx: RunContext[dict], key: str) -> str:
    """Return one field from the caller's profile dict."""
    return str(ctx.deps.get(key, 'unknown'))


result = agent.run_sync(
    'Password reset link expired, please help.',
    deps={'name': 'Alice', 'plan': 'gold'},
    usage_limits=UsageLimits(
        request_limit=6,
        output_tokens_limit=500,
        cost_limit=Decimal('0.02'),
    ),
)

print(result.output.priority)
print(result.usage())
```

Two things to notice:

1. `retries=2` at construction sets the **default** cap that tools and output validators inherit. Individual `@agent.tool(retries=...)` calls override it per tool.
2. `usage_limits=...` on `run_sync` (rather than on the `Agent`) means the same agent can enforce different budgets for internal vs. external callers.

---

## 2. `RunContext` — everything a tool can see

`pydantic_ai.RunContext` is the object PydanticAI passes to any tool, output validator, or capability hook whose signature declares it. The following fields and helpers are source-verified against 2.33.0 (`pydantic_ai/_run_context.py`):

| Field                     | Type / default                          | What it holds                                                       |
| ------------------------- | --------------------------------------- | ------------------------------------------------------------------- |
| `deps`                    | `AgentDepsT`                            | The value you passed as `deps=` to the run.                         |
| `model`                   | `AbstractModel`                         | The active model (may be a `RealtimeModel`).                        |
| `usage`                   | `RunUsage`                              | Live counters — requests, tokens, cost.                             |
| `usage_limits`            | `UsageLimits \| None`                   | The limits the run is enforcing. During a live run it is **always** a real `UsageLimits` (the default `UsageLimits()` if you passed nothing); only `None` on synthetic contexts not backed by a run (e.g. `Agent.system_prompt_parts`). |
| `agent`                   | `Agent \| None`                         | Back-reference to the agent (only set inside a run).                |
| `prompt`                  | `str \| Sequence[UserContent] \| None`  | The original user prompt.                                           |
| `messages`                | `list[ModelMessage]`                    | Full running message history.                                       |
| `validation_context`      | `Any`                                   | Passed through to Pydantic validators.                              |
| `tracer` / `instrumentation_version` / `trace_include_content` | OpenTelemetry plumbing | Wired by `logfire.instrument_pydantic_ai()`; `trace_include_content=True` embeds message bodies in spans. |
| `retries`                 | `dict[str, int]`                        | Per-tool retry counts.                                              |
| `tool_call_id`, `tool_name` | `str \| None`                         | Populated **only** while inside a specific tool's execution.        |
| `retry`, `max_retries`    | `int`                                   | Current / configured retry count for **this** invocation.           |
| `run_step`                | `int`                                   | 0-based iteration index within the run.                             |
| `tool_call_approved`      | `bool`                                  | `True` when re-entering after `ToolApproved`.                       |
| `tool_call_metadata`      | `Any`                                   | Metadata from `DeferredToolResults.metadata[tool_call_id]`.         |
| `partial_output`          | `bool`                                  | `True` if the validator is running mid-stream.                      |
| `run_id`, `conversation_id` | `str \| None`                         | Correlation ids (auto-generated as UUID7 when not passed).          |
| `metadata`                | `dict[str, Any] \| None`                | Agent-level metadata plus overrides.                                |
| `model_settings`          | `ModelSettings \| None`                 | The resolved per-run model settings.                                |
| `pending_messages`        | `list[PendingMessage] \| None`          | The internal drain queue; use `ctx.enqueue(...)` rather than mutating. |
| `tool_manager`            | `ToolManager \| None`                   | Programmatic tool dispatch (useful for sandboxes). Absent under Temporal. |
| `realtime_session`        | `RealtimeSession \| None`               | The live realtime session, once connected.                          |
| `root_capability`         | `AbstractCapability \| None`            | The effective merged capability chain for this run.                 |
| `capabilities`            | `dict[str, AbstractCapability]`         | All capabilities registered for the run.                            |
| `loaded_capability_ids`   | `set[str]`                              | Which deferred capabilities the model has explicitly loaded.        |
| `discovered_tool_names`   | `set[str]`                              | Names of deferred function tools known from history.                |
| `capability_loaded`       | `bool \| None`                          | Whether the currently-dispatching capability is loaded.             |

Helper properties / methods:

| Member                    | Kind                                          | Purpose                                                       |
| ------------------------- | --------------------------------------------- | ------------------------------------------------------------- |
| `last_attempt`            | property → `bool`                             | `True` when `retry == max_retries` — the final try before failure. |
| `enqueue(...)`            | method → `str \| None`                        | Append content to `pending_messages` from a tool or hook.     |
| `is_tool_available(name)` | method → `bool`                               | Whether a tool name is currently callable this step.          |
| `realtime`                | property → `bool`                             | Whether this is a realtime run (works even before the session connects). |

Private, framework-only attributes (`_cancellation`, `_event_stream_buffer`, `_mcp_tool_defs_cache`, `_anchored_evidence`) exist on the dataclass — they're implementation detail and should not be read from tool code.

### Runnable example — using every commonly-needed field

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, ModelRetry


@dataclass
class UserSession:
    user_id: int
    tier: str  # 'free' | 'pro'


agent = Agent('openai:gpt-4o-mini', deps_type=UserSession, retries=3)


@agent.tool
async def rate_limited_lookup(ctx: RunContext[UserSession], query: str) -> str:
    # 1) Guard on retries so we do not loop forever.
    if ctx.retry >= ctx.max_retries:
        return f'giving up after {ctx.retry} retries'

    # 2) Use deps as if they were a FastAPI dependency.
    if ctx.deps.tier == 'free' and len(query) > 40:
        raise ModelRetry(
            'Free tier: query must be <=40 chars. Shorten it and try again.'
        )

    # 3) Access the live budget the run is enforcing.
    remaining = None
    if ctx.usage_limits and ctx.usage_limits.output_tokens_limit is not None:
        remaining = (
            ctx.usage_limits.output_tokens_limit - ctx.usage.output_tokens
        )

    return (
        f'user={ctx.deps.user_id} step={ctx.run_step} '
        f'tool_call_id={ctx.tool_call_id} remaining_output_tokens={remaining}'
    )


result = agent.run_sync(
    'Look up the current status',
    deps=UserSession(user_id=42, tier='free'),
)
print(result.output)
```

The important pattern: `RunContext` is **immutable** in spirit — treat every field as read-only. Mutating `ctx.usage_limits` inside a tool actually changes what the run enforces on the next request.

---

## 3. `UsageLimits` — enforce cost and token budgets

`pydantic_ai.UsageLimits` is a dataclass (`pydantic_ai/usage.py`) with these fields:

| Field                              | Default | Description                                                    |
| ---------------------------------- | ------- | -------------------------------------------------------------- |
| `cost_limit`                       | `None`  | Max USD spend (`Decimal`).                                     |
| `request_limit`                    | `50`    | Max total requests to the model.                               |
| `tool_calls_limit`                 | `None`  | Max successful tool calls.                                     |
| `input_tokens_limit`               | `None`  | Cumulative prompt tokens across the whole run.                 |
| `output_tokens_limit`              | `None`  | Cumulative completion tokens.                                  |
| `total_tokens_limit`               | `None`  | Combined prompt+completion budget.                             |
| `per_request_input_tokens_limit`   | `None`  | Cap **each** request's context size independently.             |
| `count_tokens_before_request`      | `False` | Do a pre-flight `count_tokens` call to enforce the caps early. |

### Runnable example — three budgeting styles

```python
from decimal import Decimal
from pydantic_ai import Agent, UsageLimits, UsageLimitExceeded

agent = Agent('openai:gpt-4o-mini')

# Style 1: hard cost ceiling in USD.
cheap = UsageLimits(cost_limit=Decimal('0.001'))

# Style 2: run-wide token ceiling — good for a bounded task.
per_run = UsageLimits(
    request_limit=8,
    total_tokens_limit=4_000,
    tool_calls_limit=5,
)

# Style 3: per-request cap enforced ahead of the request itself,
# useful when a caller might pass a giant context.
preflight = UsageLimits(
    per_request_input_tokens_limit=6_000,
    count_tokens_before_request=True,
)


def run_with(limits: UsageLimits) -> str:
    try:
        result = agent.run_sync(
            'Explain OAuth 2 device flow in one paragraph.',
            usage_limits=limits,
        )
        return result.output[:80]
    except UsageLimitExceeded as exc:
        return f'BUDGET STOPPED RUN: {exc}'


for label, limits in [
    ('cheap', cheap),
    ('per_run', per_run),
    ('preflight', preflight),
]:
    print(label, run_with(limits))
```

`UsageLimits().has_token_limits()` is worth knowing when you build streaming code: if it returns `False`, PydanticAI skips per-chunk budget checks entirely, which matters on hot paths.

---

## 4. `ToolReturn` — send extra content back with the tool result

`pydantic_ai.ToolReturn` (`pydantic_ai/messages.py`) is a `dataclass(repr=False)` with these fields:

| Field          | Default | Purpose                                                              |
| -------------- | ------- | -------------------------------------------------------------------- |
| `return_value` | (required) | The value the model sees as the tool's result.                    |
| `content`      | `None`  | Extra `str | Sequence[UserContent]` sent as a fresh `UserPromptPart`. |
| `metadata`     | `None`  | Application-side data (never sent to the model).                     |
| `tools`        | `None`  | Names of deferred tools to enable in the next turn.                  |

### Runnable example — attaching an image and hidden metadata

```python
from pathlib import Path
from pydantic_ai import Agent, BinaryContent, ToolReturn

agent = Agent('openai:gpt-4o-mini')

CHART_PATH = Path('/tmp/chart.png')


@agent.tool_plain
def render_chart(spec: str) -> ToolReturn:
    """Pretend to render a chart. Return the result value plus the PNG."""
    # Only send the image when we actually have valid bytes on disk —
    # a stub payload like b'\x89PNG' is not a real PNG and downstream
    # providers reject it during content validation.
    content: list = ['Rendered chart follows:']
    if CHART_PATH.exists():
        content.append(
            BinaryContent(data=CHART_PATH.read_bytes(), media_type='image/png')
        )
    else:
        content.append('(no chart image attached — file was not generated)')

    return ToolReturn(
        return_value={'ok': True, 'spec': spec, 'points': 42},
        content=content,
        metadata={
            'internal_trace_id': 'chart-8f7a',
            'render_ms': 87,
        },
    )


result = agent.run_sync('Draw a bar chart of Q3 sales.')
print(result.output)
```

Use `metadata` for anything your application needs — evals, telemetry, PII redaction hints — that must **not** reach the model. `content=` on the other hand is sent to the LLM as a follow-up user message, which is the trick you use to feed the model back a screenshot or a document produced by the tool.

---

## 5 & 6. `DeferredToolRequests` / `DeferredToolResults` — human-in-the-loop and remote tools

`DeferredToolRequests` (`pydantic_ai/_deferred.py`) is what you set as `output_type` when the model **might** need to bail out to an external system (or a human). It carries three fields:

| Field       | Type                              | Purpose                                              |
| ----------- | --------------------------------- | ---------------------------------------------------- |
| `calls`     | `list[ToolCallPart]`              | Tool calls that need external execution.             |
| `approvals` | `list[ToolCallPart]`              | Tool calls that need human approval.                 |
| `metadata`  | `dict[str, dict[str, Any]]`       | Per-call metadata keyed by `tool_call_id`.           |

It also exposes two helper methods:

* `requests.build_results(...)` — construct the `DeferredToolResults` for the next `agent.run_sync(deferred_tool_results=...)` call. Supports `approve_all=True`.
* `requests.remaining(results)` — after applying some results, what's still pending. Returns `None` when everything is resolved.

`DeferredToolResults` mirrors it with `calls` / `approvals` / `metadata`, plus an `update()` method for merging partial resolutions.

### Runnable example — approve one, reject one, wait for a human on a third

```python
from pydantic_ai import (
    Agent, RunContext,
    DeferredToolRequests, DeferredToolResults,
    ToolApproved, ToolDenied,
)


agent = Agent(
    'openai:gpt-4o-mini',
    output_type=[str, DeferredToolRequests],
)


@agent.tool(requires_approval=True)
def transfer_funds(ctx: RunContext[None], target: str, amount: float) -> str:
    return f'transferred ${amount:.2f} to {target}'


result = agent.run_sync('Please transfer $500 to alice@example.com and $50 to bob@example.com.')

if isinstance(result.output, DeferredToolRequests):
    reqs = result.output
    print(f'{len(reqs.approvals)} tool call(s) need approval')

    # Approve one, reject another, decide the rest interactively.
    manual = {}
    for call in reqs.approvals:
        args = call.args_as_dict()
        if args.get('amount', 0) < 100:
            manual[call.tool_call_id] = ToolApproved()
        else:
            manual[call.tool_call_id] = ToolDenied(
                message='Transfers over $100 require a supervisor.',
            )

    followup = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=reqs.build_results(approvals=manual),
    )
    print(followup.output)
else:
    print(result.output)
```

Two production tips:

* If you want a "yes to everything" flow (batch backfill, CLI approval prompt), `reqs.build_results(approve_all=True)` short-circuits the loop.
* Between runs, `reqs.remaining(partial_results)` lets you re-render only the un-answered calls in your UI, without dropping the metadata attached to each.

---

## 7. `CachePoint` — provider-native prompt caching

`pydantic_ai.CachePoint` (`pydantic_ai/messages.py`) is a marker part you drop into a `UserPromptPart.content` list. Models that don't support caching simply filter it out.

Fields:

| Field  | Type                    | Default | Notes                                                                   |
| ------ | ----------------------- | ------- | ----------------------------------------------------------------------- |
| `kind` | `Literal['cache-point']`| `'cache-point'` | Discriminator (all message parts have one).                     |
| `ttl`  | `Literal['5m', '1h']`   | `'5m'`  | Anthropic / Bedrock read this per-marker; OpenAI uses request-wide TTL. |

### Runnable example — cache a long system-of-record document

```python
from pathlib import Path
from pydantic_ai import Agent, CachePoint
from pydantic_ai.messages import ModelRequest, UserPromptPart

try:
    LONG_HANDBOOK = Path('/tmp/handbook.txt').read_text()
except FileNotFoundError:
    LONG_HANDBOOK = 'X' * 20_000  # stand-in when no real handbook is on disk

agent = Agent('anthropic:claude-3-5-sonnet-latest')

# First call — cache is filled. Later calls with the same prefix hit it.
first = agent.run_sync(
    message_history=[
        ModelRequest(parts=[
            UserPromptPart(content=[
                'Consult the handbook when answering:',
                LONG_HANDBOOK,
                CachePoint(ttl='1h'),
                'How many days of PTO does a full-timer get in year one?',
            ])
        ])
    ]
)
print(first.output)

second = agent.run_sync(
    message_history=[
        ModelRequest(parts=[
            UserPromptPart(content=[
                'Consult the handbook when answering:',
                LONG_HANDBOOK,
                CachePoint(ttl='1h'),
                'And how many days for a part-timer?',
            ])
        ])
    ]
)
print(second.output)
```

If the provider does not support caching, `CachePoint` is silently dropped — the code above is safe to keep across providers.

---

## 8. Toolset composition — `PrefixedToolset`, `FilteredToolset`, `RenamedToolset`

All three live under `pydantic_ai.toolsets`. Their signatures are trivially small, which is the point — they compose. Each has a factory method on `FunctionToolset` so you rarely instantiate them directly.

| Wrapper            | Fluent factory                | Effect                                                          |
| ------------------ | ----------------------------- | --------------------------------------------------------------- |
| `PrefixedToolset`  | `toolset.prefixed(prefix)`    | Every tool renamed to `f'{prefix}_{name}'`.                     |
| `FilteredToolset`  | `toolset.filtered(fn)`        | `fn(ctx, tool_def) -> bool` decides visibility per step.        |
| `RenamedToolset`   | `toolset.renamed({new: old})` | Per-tool rename; raises `UserError` on collisions.              |

### Runnable example — one core toolset used two ways

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

# Core tools.
core = FunctionToolset()


@core.tool_plain
def get_user(user_id: int) -> dict:
    return {'id': user_id, 'name': f'user-{user_id}'}


@core.tool_plain
def delete_user(user_id: int) -> str:
    return f'deleted {user_id}'


@core.tool_plain
def create_user(name: str) -> dict:
    return {'id': 999, 'name': name}


# Read-only view for a customer-facing agent.
public = (
    core
    .filtered(lambda ctx, td: td.name in {'get_user'})
    .prefixed('crm')
    .renamed({'crm_lookup': 'crm_get_user'})
)

reader = Agent('openai:gpt-4o-mini', toolsets=[public])
print([t for t in reader.run_sync('list tools').all_messages()])
# Only `crm_lookup` is visible.

# Admin view for staff — every tool available with a role-tagged prefix.
admin = core.prefixed('admin')
writer = Agent('openai:gpt-4o-mini', toolsets=[admin])
```

The key point missed by many first-time users: **the same underlying `FunctionToolset` can be wrapped multiple ways**, so you don't have to re-declare the tool bodies to expose different subsets to different agents.

---

## 9. `WebSearchTool` — hosted web search across five providers

`pydantic_ai.WebSearchTool` (`pydantic_ai/native_tools/__init__.py`) is a native-tool _spec_ — a plain dataclass. To register it on an agent you wrap it in the [`NativeTool`][pydantic_ai.capabilities.NativeTool] capability (or the higher-level [`WebSearch`][pydantic_ai.capabilities.WebSearch] capability, which also falls back to a local implementation on providers that lack native web search). Passing a bare `WebSearchTool` into `capabilities=[...]` fails typing at construction time in 2.33.

Fields (all keyword-only):

| Field                  | Type / default                            | Providers that read it                    |
| ---------------------- | ----------------------------------------- | ----------------------------------------- |
| `search_context_size`  | `'low' | 'medium' | 'high'` (`'medium'`)  | OpenAI Responses, OpenRouter              |
| `user_location`        | `WebSearchUserLocation | None`            | Anthropic, OpenAI Responses, xAI, OpenRouter |
| `blocked_domains`      | `list[str] | None`                        | Anthropic, Groq, xAI, OpenRouter          |
| `allowed_domains`      | `list[str] | None`                        | Anthropic, Groq, OpenAI Responses, xAI, OpenRouter |
| `max_uses`             | `int | None`                              | Anthropic, OpenRouter                     |
| `external_web_access`  | `bool | None`                             | OpenAI Responses `web_search` tool        |

Note the two mutually exclusive fields: on Anthropic, `blocked_domains` and `allowed_domains` cannot be used together.

### Runnable example — provider-portable defaults with per-provider overrides

```python
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation
from pydantic_ai.capabilities import NativeTool

portable = WebSearchTool(
    search_context_size='high',
    max_uses=3,
    user_location=WebSearchUserLocation(
        city='London',
        region='England',
        country='GB',
        timezone='Europe/London',
    ),
)

news_agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    capabilities=[
        # Anthropic honours user_location + max_uses + allowed_domains.
        # The spec must be wrapped in `NativeTool(...)` to become a capability.
        NativeTool(
            WebSearchTool(
                search_context_size='high',
                max_uses=3,
                allowed_domains=['bbc.co.uk', 'reuters.com'],
                user_location=portable.user_location,
            ),
        ),
    ],
    instructions='Use web search whenever you cite a fact.',
)

print(news_agent.run_sync('What did the Bank of England do this week?').output)
```

`max_uses` gives the provider a hard budget for how many searches this run may fire; combine it with `UsageLimits(cost_limit=...)` for defence in depth.

For provider-adaptive registration, swap `NativeTool(WebSearchTool(...))` for `WebSearch(...)` from `pydantic_ai.capabilities` — it uses the provider's native tool where available and falls back to a local implementation elsewhere.

---

## 10. Error handling — `ModelRetry`, `UnexpectedModelBehavior`, `ModelHTTPError`

Three exceptions cover most production failure paths. Their public surface (`pydantic_ai/exceptions.py`) is small enough to memorise.

**`ModelRetry(message: str)`** — not really an error. Raise it from tools, output validators, or capability hooks to prompt the model to try again with your feedback.

**`UnexpectedModelBehavior(message, body=None)`** — raised **by** PydanticAI when the model produced structurally invalid output after all retries were exhausted. Has `.message` and a pretty-printed `.body`.

**`ModelHTTPError(status_code, model_name, body, *, headers=None, suggested_model_id=None)`** — a 4xx/5xx from the provider. Exposes `.status_code`, `.body`, `.headers` (lowercased), and — new in 2.x — `retry_after` which parses the `Retry-After` header (integer seconds *or* RFC-9110 date) and returns a `float | None`.

### Runnable example — layered error handling

```python
import time
from pydantic import BaseModel
from pydantic_ai import (
    Agent, RunContext, ModelRetry,
    UnexpectedModelBehavior, ModelHTTPError,
    capture_run_messages,
)


class Answer(BaseModel):
    number: int


agent = Agent('openai:gpt-4o-mini', output_type=Answer, retries=3)


@agent.output_validator
def only_positive(ctx: RunContext[None], out: Answer) -> Answer:
    if out.number <= 0:
        raise ModelRetry(
            f'{out.number} is not positive. Return a strictly positive integer.'
        )
    return out


def robust_run(prompt: str, attempts: int = 4) -> str:
    for attempt in range(attempts):
        with capture_run_messages() as msgs:
            try:
                return str(agent.run_sync(prompt).output)
            except ModelHTTPError as http:
                # Providers frequently ask us to back off. Respect it if we can.
                wait = http.retry_after or 2 ** attempt
                if http.status_code in {429, 503} and attempt < attempts - 1:
                    time.sleep(min(wait, 30))
                    continue
                return f'permanent HTTP {http.status_code}: {http.body!r}'
            except UnexpectedModelBehavior as bad:
                # Model kept returning bad shapes after all retries.
                print('captured messages before failure:', len(msgs))
                return f'model gave up: {bad.message}'
    return 'exhausted attempts'


print(robust_run('Give me the answer to life as a positive integer.'))
```

Two things this shows that the smaller examples elsewhere skip:

1. `ModelHTTPError.retry_after` interprets the header for you — you don't have to parse it yourself.
2. `capture_run_messages()` combined with a `try / except` is the idiomatic way to keep the wire log for a failed run without leaking it on the happy path.

---

## Reference

* Package version: `pydantic-ai==2.33.0` (Aug 2026).
* Source read for this page:
  * `pydantic_ai/agent/__init__.py`
  * `pydantic_ai/_run_context.py`
  * `pydantic_ai/usage.py`
  * `pydantic_ai/messages.py`
  * `pydantic_ai/_deferred.py`
  * `pydantic_ai/toolsets/{function,filtered,renamed,prefixed}.py`
  * `pydantic_ai/native_tools/__init__.py`
  * `pydantic_ai/exceptions.py`
* Selection method: ten classes chosen for coverage of the most commonly under-documented surfaces (deps injection, budgets, HITL, cache markers, toolset composition, native search, error taxonomy).

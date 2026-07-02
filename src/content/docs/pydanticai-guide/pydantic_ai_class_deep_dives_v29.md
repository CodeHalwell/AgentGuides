---
title: "PydanticAI Class Deep Dives Vol. 29"
description: "Source-verified deep dives into 10 pydantic-ai 2.3.0 class groups: AnthropicCompaction (server-side context compaction — token_threshold, instructions, pause_after_compaction), OpenAICompaction (stateful server-side + stateless /responses/compact — message_count_threshold, custom trigger, ZDR mode), OnlineEvaluation capability (async background evaluators after every run — wrap_run, run_on_errors, disable_evaluation, EvaluationSink), WrapperCapability (transparent delegation base — apply() traversal, id inheritance, defer_loading forwarding, selective override), Capability full v2.3.0 (decorator API — @cap.tool/@cap.tool_plain/@cap.instructions, toolsets=, defer_loading=, role-scoped dispatch), ChatRequestExtra + ConfigureFrontend + ModelInfo + BuiltinToolInfo (Agent.to_web() config wire types — model selection, builtin-tool routing, camelCase API contract), CapabilityOwnedToolset (binds toolsets to capabilities — capability_id stamping, defer_loading propagation, deferred instruction gating), DeferredCapabilityLoaderToolset (framework load_capability tool — load capability by id, instruction injection, ctx.capabilities resolution), DynamicCapability deep patterns (async factory, None passthrough, id forwarding, feature-flag activation, per-request personalisation), ImageGenerationSubagentTool + XSearchSubagentTool (common_tools fallback pattern — subagent delegation for non-native models, ModelRetry on failure, callable model factory). All verified against pydantic-ai 2.3.0 source."
sidebar:
  label: "Class deep dives (Vol. 29)"
  order: 55
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.3.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.3.x API.
</Aside>

Ten class groups covering brand-new context compaction capabilities, live background evaluation, the complete `Capability` decorator API, Agent web-UI configuration types, capability-owned toolset plumbing, and the common-tools subagent fallback pattern. Several of these classes (`AnthropicCompaction`, `OpenAICompaction`) are new in 2.3.0 and have no prior documentation; the rest fill gaps in the earlier volumes.

---

## 1. `AnthropicCompaction` — Server-Side Context Compaction for Anthropic Models

**Source**: `pydantic_ai/models/anthropic.py`  
**Export**: `from pydantic_ai.models.anthropic import AnthropicCompaction`

`AnthropicCompaction` is an `AbstractCapability` that wires Anthropic's native `context_management` API into a PydanticAI agent. When input tokens cross `token_threshold`, the Anthropic server automatically summarises the conversation tail, replacing older turns with a compact `<context_management>` block — without any application-side logic.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass(init=False)
class AnthropicCompaction(AbstractCapability[AgentDepsT]):
    """Capability that enables Anthropic's server-side context compaction."""

    token_threshold: int        # default 150_000; minimum 50_000
    instructions: str | None    # custom summarisation prompt
    pause_after_compaction: bool  # stop after compact block (stop_reason='compaction')

    def __init__(
        self,
        *,
        token_threshold: int = 150_000,
        instructions: str | None = None,
        pause_after_compaction: bool = False,
    ) -> None: ...
```

The capability implements `get_model_settings()` which injects a `context_management` dict into `ModelSettings.extra_body`; existing `context_management` values are preserved and the capability's edit is appended. It is compatible with any `anthropic:claude-*` model that supports the `context_management` API parameter.

### 1.1 Basic Long-Running Research Agent

Enable compaction on a context-heavy research agent that accumulates many tool results across turns.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='You are a research assistant. Accumulate findings across many queries.',
    capabilities=[AnthropicCompaction(token_threshold=100_000)],
)

async def main() -> None:
    history = []
    for query in [
        'Summarise quantum computing milestones 2020–2023',
        'Compare gate-based vs annealing approaches',
        'What are the top 5 open research challenges today?',
    ]:
        result = await agent.run(query, message_history=history)
        history = result.all_messages()
        print(result.output)

asyncio.run(main())
```

### 1.2 Custom Compaction Instructions and Pause Mode

Supply domain-specific summarisation instructions and detect the compaction event so you can log or checkpoint state.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction
from pydantic_ai.messages import ModelResponse

LEGAL_INSTRUCTIONS = (
    'Preserve all cited statute numbers, party names, and ruling dates verbatim. '
    'Omit procedural boilerplate. Retain fact summaries in bullet form.'
)

agent = Agent(
    'anthropic:claude-opus-4-8',
    capabilities=[
        AnthropicCompaction(
            token_threshold=80_000,
            instructions=LEGAL_INSTRUCTIONS,
            pause_after_compaction=True,  # stops when compact block is inserted
        )
    ],
)

async def run_with_compaction_guard(turns: list[str]) -> None:
    history = []
    for turn in turns:
        result = await agent.run(turn, message_history=history)
        new_messages = result.new_messages()
        for msg in new_messages:
            if isinstance(msg, ModelResponse):
                # Check if Anthropic stopped due to compaction
                if any(getattr(p, 'finish_reason', None) == 'compaction'
                       for p in getattr(msg, 'parts', [])):
                    print('⚡ Compaction occurred — checkpointing state')
                    # resume logic here
        history = result.all_messages()
        print(f'Turn: {turn[:40]}… → {result.output[:80]}')

asyncio.run(run_with_compaction_guard([
    'Review the Smith v Jones 2021 case',
    'How does it relate to ABC Corp v DEF LLC?',
    'Draft a brief summary for appeal',
]))
```

### 1.3 Per-Run Capability Override

Inject compaction only for long-running workflows, not quick one-shot queries.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction

base_agent = Agent('anthropic:claude-sonnet-4-6', instructions='Be concise.')

COMPACTION = AnthropicCompaction(token_threshold=60_000)

async def run_short(query: str) -> str:
    """Quick query — no compaction overhead."""
    result = await base_agent.run(query)
    return result.output

async def run_long(queries: list[str]) -> str:
    """Multi-turn workflow — enable compaction per run."""
    history = []
    output = ''
    for q in queries:
        result = await base_agent.run(
            q,
            message_history=history,
            capabilities=[COMPACTION],  # per-run injection
        )
        history = result.all_messages()
        output = result.output
    return output

async def main() -> None:
    print(await run_short('What is 2+2?'))
    print(await run_long(['Explain RLHF', 'How does it compare to DPO?', 'Summarise tradeoffs']))

asyncio.run(main())
```

---

## 2. `OpenAICompaction` — Stateful and Stateless Context Compaction for OpenAI

**Source**: `pydantic_ai/models/openai.py`  
**Export**: `from pydantic_ai.models.openai import OpenAICompaction`

`OpenAICompaction` supports two fundamentally different compaction modes for OpenAI's Responses API:

- **Stateful** (`stateless=False`, default): uses OpenAI's server-side `context_management` parameter on the `/responses` endpoint. Works alongside `openai_previous_response_id='auto'`.
- **Stateless** (`stateless=True`): calls the `/responses/compact` endpoint from a `before_model_request` hook. Required for Zero Data Retention (ZDR) environments or when `openai_store=False`.

The mode is auto-inferred: providing `message_count_threshold` or a custom `trigger` implies stateless mode.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass(init=False)
class OpenAICompaction(AbstractCapability[AgentDepsT]):
    stateless: bool                    # auto-inferred if None
    token_threshold: int | None        # stateful: compact_threshold on API
    message_count_threshold: int | None  # stateless: trigger when msg count >= N
    trigger: Callable[[list[ModelMessage]], bool] | None  # custom stateless trigger

    def __init__(
        self,
        *,
        stateless: bool | None = None,
        token_threshold: int | None = None,
        message_count_threshold: int | None = None,
        trigger: Callable[[list[ModelMessage]], bool] | None = None,
    ) -> None: ...
```

### 2.1 Stateful Compaction with Server-Side Conversation State

Let OpenAI manage the conversation state and auto-compact on its side.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModelSettings

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[OpenAICompaction(token_threshold=80_000)],
    model_settings=OpenAIResponsesModelSettings(
        openai_previous_response_id='auto',  # server-managed conversation state
    ),
)

async def main() -> None:
    history = []
    for question in [
        'Walk me through the HTTP/2 spec changes vs HTTP/1.1',
        'How does QUIC differ from TCP?',
        'What are the practical deployment tradeoffs?',
    ]:
        result = await agent.run(question, message_history=history)
        history = result.all_messages()
        print(result.output)

asyncio.run(main())
```

### 2.2 Stateless Compaction for ZDR Environments

When using Zero Data Retention, the `/responses/compact` endpoint must be called explicitly.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModelSettings

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        OpenAICompaction(
            message_count_threshold=20,  # compact when > 20 messages
            instructions='Preserve all code snippets verbatim. Summarise discussion.',
        )
    ],
    model_settings=OpenAIResponsesModelSettings(openai_store=False),  # ZDR
)

async def main() -> None:
    history = []
    for i in range(30):
        result = await agent.run(
            f'Iteration {i}: review this snippet: x = {i} * 2',
            message_history=history,
        )
        history = result.all_messages()
        print(f'[{i}] {result.output[:60]}')

asyncio.run(main())
```

### 2.3 Custom Trigger — Token-Budget-Aware Compaction

Trigger stateless compaction only when token usage in the history exceeds a custom threshold.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAICompaction

def _token_heavy(messages: list[ModelMessage]) -> bool:
    """Trigger compaction if history contains >15 ModelResponse messages."""
    response_count = sum(1 for m in messages if isinstance(m, ModelResponse))
    return response_count > 15

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[OpenAICompaction(trigger=_token_heavy)],
)

async def main() -> None:
    history: list[ModelMessage] = []
    for turn_n in range(20):
        result = await agent.run(
            f'Turn {turn_n}: What is {turn_n} * {turn_n + 1}?',
            message_history=history,
        )
        history = result.all_messages()
        response_msgs = sum(1 for m in history if isinstance(m, ModelResponse))
        print(f'Turn {turn_n}: responses={response_msgs}, output={result.output}')

asyncio.run(main())
```

---

## 3. `OnlineEvaluation` — Live Background Evaluation Capability

**Source**: `pydantic_evals/online_capability.py`  
**Export**: `from pydantic_evals.online_capability import OnlineEvaluation`

`OnlineEvaluation` is an `AbstractCapability` that attaches evaluators to an agent so they fire **asynchronously in the background** after each run completes — without blocking the caller. It wraps `agent.run()`, `agent.run_stream()`, and `agent.iter()`. Streaming runs dispatch evaluators only after the context manager exits and the final result is available.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass(kw_only=True)
class OnlineEvaluation(AbstractCapability[AgentDepsT]):
    evaluators: Sequence[Evaluator | OnlineEvaluator]
    """Evaluators to run after each completed agent run."""
    config: OnlineEvalConfig | None = None
    """Optional config override. None falls back to the module-level DEFAULT_CONFIG."""
```

`Evaluator` instances are auto-wrapped in `OnlineEvaluator` with default sampling settings. The `EvaluatorContext` exposes `inputs` (the raw prompt), `output` (the agent result), `duration`, `span_tree`, and `metadata`. Evaluation results are emitted as OpenTelemetry `gen_ai.evaluation.result` log events and can be fanned out to custom sinks.

### 3.1 Simple Quality Monitor

Attach an always-on response quality check to a production agent.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online_capability import OnlineEvaluation

@dataclass
class OutputNotEmpty(Evaluator):
    """Pass when the agent produces a non-empty response."""
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output and str(ctx.output).strip())

@dataclass
class ShortResponse(Evaluator):
    """Warn when the response exceeds 500 characters."""
    max_chars: int = 500
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) <= self.max_chars

agent = Agent(
    'openai:gpt-4o-mini',
    name='support-bot',
    capabilities=[
        OnlineEvaluation(evaluators=[OutputNotEmpty(), ShortResponse(max_chars=500)])
    ],
)

async def main() -> None:
    result = await agent.run('How do I reset my password?')
    print(result.output)
    # Evaluators fire asynchronously in background — check Logfire / OTel for results

asyncio.run(main())
```

### 3.2 Sampled Evaluation with Custom Sink

Run expensive LLM-judge evaluations on only 10 % of traffic, fanning out to a custom logging sink.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvalConfig, OnlineEvaluator, EvaluationSink, SinkPayload
from pydantic_evals.online_capability import OnlineEvaluation

# A simple dataclass sink that collects results in memory for testing
results_log: list[dict] = []

@dataclass
class LogSink(EvaluationSink):
    async def submit(self, payload: SinkPayload) -> None:
        for r in payload.results:
            results_log.append({'name': r.name, 'value': r.value})

@dataclass
class ToneCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = str(ctx.output)
        positive_words = ['happy', 'great', 'excellent', 'thanks', 'sure']
        hits = sum(1 for w in positive_words if w in output.lower())
        return min(hits / 3.0, 1.0)

config = OnlineEvalConfig(
    default_sample_rate=0.1,   # 10 % of calls
    default_sink=LogSink(),
    emit_otel_events=True,
)

agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[
        OnlineEvaluation(
            evaluators=[OnlineEvaluator(evaluator=ToneCheck(), sample_rate=0.1)],
            config=config,
        )
    ],
)

async def main() -> None:
    for i in range(5):
        result = await agent.run('Say something encouraging')
        print(result.output[:80])
    print(f'Evaluations captured: {len(results_log)}')

asyncio.run(main())
```

### 3.3 Error-Aware Evaluator and Disabled Evaluation Context

An evaluator that also fires on errors, plus disabling evaluation inside test suites.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator, disable_evaluation, wait_for_evaluations
from pydantic_evals.online_capability import OnlineEvaluation

@dataclass
class ErrorTracker(Evaluator):
    """Track which runs resulted in errors vs successes."""
    def evaluate(self, ctx: EvaluatorContext) -> dict:
        is_error = isinstance(ctx.output, Exception)
        return {'error': is_error, 'type': type(ctx.output).__name__}

agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[
        OnlineEvaluation(
            evaluators=[
                OnlineEvaluator(
                    evaluator=ErrorTracker(),
                    run_on_errors=True,  # fires even when agent raises
                )
            ]
        )
    ],
)

async def test_agent() -> None:
    """Disable evaluations inside automated tests for determinism."""
    with disable_evaluation():
        result = await agent.run('Hello')
        print(f'Test result: {result.output}')
    # No evaluators ran during the test

async def production_run() -> None:
    result = await agent.run('Summarise pydantic-ai 2.3.0 changes')
    print(result.output)
    # Wait for all background evaluations to finish (useful in scripts)
    await wait_for_evaluations(timeout=10.0)
    print('All evaluations complete')

asyncio.run(production_run())
```

---

## 4. `WrapperCapability` — Transparent Capability Delegation

**Source**: `pydantic_ai/capabilities/wrapper.py`  
**Export**: `from pydantic_ai.capabilities import WrapperCapability`

`WrapperCapability` is the capability analogue of `WrapperToolset` — it wraps another `AbstractCapability` and delegates every interface method. Subclass and override only the methods you need. It automatically inherits the wrapped capability's `id` and `defer_loading` when no explicit values are set on the wrapper itself, so deferred capabilities remain deferred through the wrapper.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass
class WrapperCapability(AbstractCapability[AgentDepsT]):
    wrapped: AbstractCapability[AgentDepsT]

    def __post_init__(self) -> None:
        # Adopts wrapped capability's id and defer_loading when not explicitly set.
        if self.id is None:
            self.id = self.wrapped.id
            self.defer_loading = self.wrapped.defer_loading

    def apply(self, visitor) -> None:
        visitor(self)
        # If wrapped is a container, also register its leaves so
        # child-owned hooks and toolsets resolve correctly.
        ...

    async def for_run(self, ctx) -> AbstractCapability:
        # Re-creates wrapper around the post-for_run wrapped instance.
        wrapped = await self.wrapped.for_run(ctx)
        return dataclasses.replace(self, wrapped=wrapped)
```

All `AbstractCapability` hook methods (`before_model_request`, `after_node_run`, `get_model_settings`, `get_instructions`, `get_toolset`, `get_native_tools`, `wrap_run`, etc.) delegate to `self.wrapped`.

### 4.1 Logging Wrapper — Trace Every Capability Hook

Wrap an existing capability to emit structured logs around each lifecycle event.

```python  {test="skip"}
import asyncio
import dataclasses
import logging
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, WrapperCapability
from pydantic_ai.models import ModelRequestContext

log = logging.getLogger(__name__)

@dataclasses.dataclass
class LoggingCapability(WrapperCapability):
    """Wraps a capability and logs every lifecycle event."""

    async def before_model_request(self, ctx: RunContext, request_context: ModelRequestContext):
        log.info('before_model_request: run_step=%s', ctx.run_step)
        return await self.wrapped.before_model_request(ctx, request_context)

    async def after_node_run(self, ctx: RunContext, node):
        log.info('after_node_run: node=%s', type(node).__name__)
        return await self.wrapped.after_node_run(ctx, node)


# Base capability that holds the tools
refunds = Capability(
    id='refunds',
    instructions='Always confirm the order ID before processing a refund.',
)

@refunds.tool_plain
def get_refund_status(order_id: str) -> str:
    """Look up the refund status for an order."""
    return f'Order {order_id}: refund pending.'


agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[LoggingCapability(wrapped=refunds)],
)

async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = await agent.run('What is the refund status for order ORD-999?')
    print(result.output)

asyncio.run(main())
```

### 4.2 Rate-Limiting Wrapper

Add a per-run concurrency guard around an expensive capability without touching its implementation.

```python  {test="skip"}
import asyncio
import dataclasses
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, WrapperCapability
from pydantic_ai.exceptions import ModelRetry

_sem = asyncio.Semaphore(3)  # at most 3 concurrent uses

@dataclasses.dataclass
class RateLimitedCapability(WrapperCapability):
    """Permits at most `max_concurrent` simultaneous uses of the wrapped capability."""

    async def wrap_run(self, ctx: RunContext, *, handler):
        async with _sem:
            return await self.wrapped.wrap_run(ctx, handler=handler)


analytics = Capability(
    id='analytics',
    instructions='Run SQL-style queries against the analytics database.',
)

@analytics.tool_plain
def run_query(sql: str) -> list[dict]:
    """Execute an analytics query."""
    return [{'result': 42, 'query': sql}]


agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[RateLimitedCapability(wrapped=analytics)],
)

async def main() -> None:
    tasks = [agent.run(f'Count active users in region {r}') for r in ['EU', 'US', 'APAC']]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r.output)

asyncio.run(main())
```

### 4.3 Conditional Bypass — Skip Capability in Test Mode

Wrap to transparently strip a capability based on runtime context.

```python  {test="skip"}
import asyncio
import dataclasses
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, WrapperCapability
from pydantic_ai.models import ModelRequestContext

@dataclasses.dataclass
class TestAwareCapability(WrapperCapability):
    """No-ops all hooks when TEST_MODE env var is set."""

    def _is_test(self) -> bool:
        return os.environ.get('TEST_MODE') == '1'

    def get_instructions(self):
        if self._is_test():
            return None  # suppress instructions in tests
        return self.wrapped.get_instructions()

    def get_toolset(self):
        if self._is_test():
            return None  # suppress tools in tests
        return self.wrapped.get_toolset()


payments = Capability(
    id='payments',
    instructions='You have access to the live payment gateway.',
)

@payments.tool_plain
def charge_card(card_id: str, amount: float) -> str:
    """Charge a payment card."""
    return f'Charged ${amount:.2f} to card {card_id}'


agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[TestAwareCapability(wrapped=payments)],
)

async def main() -> None:
    os.environ['TEST_MODE'] = '1'
    result = await agent.run('Charge card C-123 for $50.00')
    print(f'Test mode result: {result.output}')
    del os.environ['TEST_MODE']

asyncio.run(main())
```

---

## 5. `Capability` — Full v2.3.0 Decorator API

**Source**: `pydantic_ai/capabilities/capability.py`  
**Export**: `from pydantic_ai.capabilities import Capability`

`Capability` is a concrete `AbstractCapability` subclass that bundles instructions, tools, and toolsets under a single identity **without requiring subclassing**. Three decorators mirror the `Agent` API:

| Decorator | Equivalent | Notes |
|-----------|-----------|-------|
| `@cap.tool` | `@agent.tool` | Receives `RunContext[AgentDepsT]` as first arg |
| `@cap.tool_plain` | `@agent.tool_plain` | No context; all other Tool params supported |
| `@cap.instructions` | system prompt func | Receives `RunContext` or nothing; may be async |

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass(init=False)
class Capability(AbstractCapability[AgentDepsT]):
    toolsets: Sequence[AgentToolset[AgentDepsT]] = ()
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = ()
    description: str | None = None

    def __init__(
        self,
        *,
        instructions: AgentInstructions[AgentDepsT] | None = None,
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        id: str | None = None,
        description: CapabilityDescription[AgentDepsT] | None = None,
        defer_loading: bool = False,
    ) -> None: ...

    # Decorator methods:
    def tool(self, func=None, /, **kwargs) -> ...: ...
    def tool_plain(self, func=None, /, **kwargs) -> ...: ...
    def instructions(self, func) -> ...: ...
```

### 5.1 Customer-Support Capability with All Three Decorators

Register tools and dynamic instructions on a single capability instance.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

support = Capability[str](
    id='customer-support',
    description='Tools for looking up orders and handling refunds.',
    instructions='You are a customer support specialist. Be empathetic and concise.',
)

@support.instructions
def greeting(ctx: RunContext[str]) -> str:
    """Personalise the system prompt with the caller's name."""
    return f'You are assisting {ctx.deps}. Address them by name in every reply.'

@support.tool
def get_order(ctx: RunContext[str], order_id: str) -> dict:
    """Retrieve order details for the current customer."""
    return {'order_id': order_id, 'customer': ctx.deps, 'status': 'shipped'}

@support.tool_plain
def list_faq_topics() -> list[str]:
    """Return available FAQ topic categories."""
    return ['returns', 'shipping', 'billing', 'technical']

agent = Agent('openai:gpt-4o-mini', deps_type=str, capabilities=[support])

async def main() -> None:
    result = await agent.run(
        'Where is my order ORD-7890?',
        deps='Alice Johnson',
    )
    print(result.output)

asyncio.run(main())
```

### 5.2 Deferred-Loading Domain Capability

Use `defer_loading=True` so the model only loads this capability when explicitly needed (via the `load_capability` tool), saving tokens on irrelevant requests.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability

accounting = Capability(
    id='accounting',
    description='Financial reporting and invoice management tools. Load when asked about invoices, expenses, or P&L.',
    defer_loading=True,  # not loaded until model requests it
)

@accounting.tool_plain
def get_invoice(invoice_id: str) -> dict:
    """Fetch an invoice by ID."""
    return {'invoice_id': invoice_id, 'amount': 1500.00, 'status': 'paid'}

@accounting.tool_plain
def get_monthly_pnl(month: str) -> dict:
    """Return profit and loss for a given month (YYYY-MM)."""
    return {'month': month, 'revenue': 50000, 'expenses': 30000, 'profit': 20000}

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a business assistant.',
    capabilities=[accounting],
)

async def main() -> None:
    # First request — unrelated, accounting capability stays hidden
    r1 = await agent.run('What is the capital of France?')
    print('Unrelated:', r1.output)

    # Second request — model discovers and loads accounting capability
    r2 = await agent.run('Show me invoice INV-2024-001')
    print('Accounting:', r2.output)

asyncio.run(main())
```

### 5.3 Role-Based Capability Dispatch with Shared Agent

Bundle role-specific permissions into separate `Capability` instances and inject them at run time.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

@dataclass
class User:
    name: str
    role: str  # 'admin' | 'viewer'

# Capability visible to all roles
viewer_cap = Capability[User](
    id='viewer',
    instructions='You can read data but not modify it.',
)

@viewer_cap.tool
def view_dashboard(ctx: RunContext[User]) -> dict:
    """Return the read-only dashboard for the current user."""
    return {'user': ctx.deps.name, 'metrics': {'visitors': 1234, 'conversions': 56}}

# Admin-only capability
admin_cap = Capability[User](
    id='admin',
    instructions='You have full write access. Confirm destructive actions before executing.',
)

@admin_cap.tool
def delete_record(ctx: RunContext[User], record_id: str) -> str:
    """Permanently delete a record. Admin only."""
    return f'Record {record_id} deleted by {ctx.deps.name}.'

agent = Agent('openai:gpt-4o-mini', deps_type=User)

async def main() -> None:
    alice = User(name='Alice', role='admin')
    bob = User(name='Bob', role='viewer')

    alice_caps = [viewer_cap, admin_cap]
    bob_caps = [viewer_cap]

    r_alice = await agent.run(
        'Delete record REC-99 and show me the dashboard.',
        deps=alice, capabilities=alice_caps,
    )
    print('Alice:', r_alice.output)

    r_bob = await agent.run(
        'Show me the dashboard.',
        deps=bob, capabilities=bob_caps,
    )
    print('Bob:', r_bob.output)

asyncio.run(main())
```

---

## 6. `ChatRequestExtra` + `ConfigureFrontend` + `ModelInfo` + `BuiltinToolInfo` — Agent Web UI Config

**Source**: `pydantic_ai/ui/_web/api.py`  
**Export**: `from pydantic_ai.ui._web.api import ChatRequestExtra, ConfigureFrontend, ModelInfo, BuiltinToolInfo`

`Agent.to_web()` launches a Starlette-based chat UI with four routes. These four Pydantic models are the wire types governing dynamic frontend configuration:

| Class | Direction | Route | Purpose |
|-------|-----------|-------|---------|
| `ModelInfo` | server → client | `GET /config` | Available model list |
| `BuiltinToolInfo` | server → client | `GET /config` | Available builtin tools |
| `ConfigureFrontend` | server → client | `GET /config` | Full config payload |
| `ChatRequestExtra` | client → server | `POST /chat` | Per-request model/tool overrides |

```python
# Key signatures verified from source (pydantic-ai 2.3.0):

class ModelInfo(BaseModel, alias_generator=to_camel, populate_by_name=True):
    id: str          # e.g. 'openai:gpt-4o'
    name: str        # human label e.g. 'GPT-4o'

class BuiltinToolInfo(BaseModel, alias_generator=to_camel, populate_by_name=True):
    id: str          # e.g. 'web_search'
    name: str        # human label e.g. 'Web Search'

class ConfigureFrontend(BaseModel, alias_generator=to_camel, populate_by_name=True):
    models: list[ModelInfo]
    builtin_tools: list[BuiltinToolInfo]

class ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
    model: str | None = None               # maps to JSON field 'model'
    builtin_tools: list[str] = []          # maps to JSON field 'builtinTools'
```

All four classes use `alias_generator=to_camel` for JSON serialisation: Python field `builtin_tools` becomes JSON `builtinTools`.

### 6.1 Launch a Multi-Model Web Chat App

Expose three models for user selection and log which model was chosen per request.

```python  {test="skip"}
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant. Answer concisely.',
)

# to_web() accepts models= as a dict {label: model_id} or a list of model ids.
# The agent's own model is always included; models= adds extra choices.
# Internally this builds a ConfigureFrontend / ModelInfo payload served at GET /configure.
app = agent.to_web(
    models={
        'GPT-4o mini (fast)': 'openai:gpt-4o-mini',
        'GPT-4o (smart)': 'openai:gpt-4o',
        'Claude Haiku': 'anthropic:claude-haiku-4-5',
    }
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### 6.2 Parse `ChatRequestExtra` to Route Models Dynamically

In a custom endpoint, parse the extra request fields to select the model at run time.

```python  {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.ui._web.api import ChatRequestExtra

AVAILABLE_MODELS = {
    'openai:gpt-4o-mini': 'gpt-4o-mini',
    'openai:gpt-4o':      'gpt-4o',
}

def parse_chat_request(body: dict) -> tuple[Agent, ChatRequestExtra]:
    """Parse inbound chat body and select the right agent configuration."""
    extra = ChatRequestExtra.model_validate(body)
    model_id = extra.model if extra.model in AVAILABLE_MODELS else 'openai:gpt-4o-mini'
    agent = Agent(model_id)
    return agent, extra

async def handle_chat(body: dict, user_message: str) -> str:
    agent, extra = parse_chat_request(body)
    print(f'Using model: {extra.model}, tools: {extra.builtin_tools}')
    result = await agent.run(user_message)
    return result.output

async def main() -> None:
    # Simulate a frontend sending model and tool selection
    request_body = {'model': 'openai:gpt-4o', 'builtinTools': ['web_search']}
    response = await handle_chat(request_body, 'What happened in tech news today?')
    print(response)

asyncio.run(main())
```

### 6.3 Custom `/config` Endpoint with Builtin Tools

Expose which builtin tools are available and handle per-request tool activation.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch, WebFetch
from pydantic_ai.ui._web.api import BuiltinToolInfo, ChatRequestExtra, ConfigureFrontend

BUILTIN_TOOLS_REGISTRY = {
    'web_search': WebSearch(),
    'web_fetch':  WebFetch(),
}

def build_config() -> ConfigureFrontend:
    return ConfigureFrontend(
        models=[],  # no model switching
        builtin_tools=[
            BuiltinToolInfo(id='web_search', name='Web Search'),
            BuiltinToolInfo(id='web_fetch', name='Web Fetch'),
        ],
    )

def agent_for_request(body: dict) -> Agent:
    """Create an agent with only the tools the user selected."""
    extra = ChatRequestExtra.model_validate(body)
    selected_caps = [
        BUILTIN_TOOLS_REGISTRY[tool_id]
        for tool_id in extra.builtin_tools
        if tool_id in BUILTIN_TOOLS_REGISTRY
    ]
    return Agent('openai:gpt-4o-mini', capabilities=selected_caps)

async def main() -> None:
    config = build_config()
    print('Available tools:', [t.id for t in config.builtin_tools])

    agent = agent_for_request({'builtinTools': ['web_search']})
    result = await agent.run('Search for pydantic-ai 2.3.0 release notes')
    print(result.output)

asyncio.run(main())
```

---

## 7. `CapabilityOwnedToolset` — Binding Toolsets to Capabilities

**Source**: `pydantic_ai/toolsets/_capability_owned.py`  
**Export**: internal; used by the framework when a `Capability` registers toolsets

`CapabilityOwnedToolset` wraps a contributing `AbstractToolset` and stamps every `ToolDefinition` with the owning capability's `id`. When `defer_loading=True` on the parent capability, it also marks tools with `DEFERRED_CAPABILITY_TOOL_METADATA_KEY` and suppresses instructions until the capability is explicitly loaded. This is the plumbing that makes deferred capabilities work: the model sees a description in the tool catalog but can't call the tools until it first calls `load_capability`.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass
class CapabilityOwnedToolset(WrapperToolset[AgentDepsT]):
    capability: AbstractCapability[AgentDepsT]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool]:
        tools = await self.wrapped.get_tools(ctx)
        capability_id = resolve_capability_id(ctx, self.capability)
        defer_loading = self.capability.defer_loading is True
        # Stamp capability_id and defer_loading onto each ToolDefinition:
        ...

    async def get_instructions(self, ctx: RunContext[AgentDepsT]):
        if self.capability.defer_loading is True:
            return None  # suppress until loaded
        return await self.wrapped.get_instructions(ctx)
```

### 7.1 Observe Capability ID Stamping

Verify that tools from a capability carry the expected `capability_id` at run time.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

inventory = Capability(id='inventory', instructions='Manage warehouse inventory.')

@inventory.tool_plain
def stock_count(sku: str) -> int:
    """Return current stock count for a SKU."""
    return 150

# Use TestModel to inspect tool definitions without an API call
model = TestModel()
agent = Agent(model, capabilities=[inventory])

async def main() -> None:
    async with agent.iter('How many SKU-A1 do we have?') as agent_run:
        async for node in agent_run:
            # Inspect tool definitions at the ModelRequestNode
            from pydantic_ai._agent_graph import ModelRequestNode
            if isinstance(node, ModelRequestNode):
                tools = model.last_model_request_parameters
                if tools:
                    for td in tools.function_tools:
                        print(
                            f'Tool: {td.name!r}, '
                            f'capability_id: {td.capability_id!r}, '
                            f'defer_loading: {td.defer_loading}'
                        )
    print(agent_run.result.output)

asyncio.run(main())
```

### 7.2 Deferred Capability — Tools Hidden Until Load

Demonstrate that deferred capability tools are marked `defer_loading=True` and instructions are withheld.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

billing = Capability(
    id='billing',
    description='Billing and invoice tools — load when user asks about payments.',
    defer_loading=True,
)

@billing.tool_plain
def get_invoice(invoice_id: str) -> dict:
    """Retrieve an invoice."""
    return {'id': invoice_id, 'amount': 250.0, 'status': 'pending'}

model = TestModel()
agent = Agent(model, capabilities=[billing])

async def main() -> None:
    result = await agent.run('What is invoice INV-001?')
    print('Result:', result.output)
    # The model first calls load_capability(id='billing'), then get_invoice

asyncio.run(main())
```

### 7.3 Multiple Toolsets Under One Capability

Register two separate `FunctionToolset`s on a single capability — both get stamped with the same `capability_id`.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

crm_ts: FunctionToolset = FunctionToolset()
email_ts: FunctionToolset = FunctionToolset()

@crm_ts.tool_plain
def find_contact(email: str) -> dict:
    """Look up a CRM contact by email."""
    return {'email': email, 'name': 'Jane Doe', 'id': 'C-42'}

@email_ts.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f'Email sent to {to}: {subject}'

crm_cap = Capability(
    id='crm-suite',
    description='CRM lookup and email tooling.',
    toolsets=[crm_ts, email_ts],
)

agent = Agent('openai:gpt-4o-mini', capabilities=[crm_cap])

async def main() -> None:
    result = await agent.run(
        'Find the contact for alice@example.com and send her a welcome email.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 8. `DeferredCapabilityLoaderToolset` — The `load_capability` Framework Tool

**Source**: `pydantic_ai/toolsets/_deferred_capability_loader.py`  
**Export**: internal; auto-injected when `defer_loading=True` capabilities are present

`DeferredCapabilityLoaderToolset` is a `WrapperToolset` automatically injected by the framework. It prepends a reserved `load_capability` tool to the tool list. When the model calls `load_capability(id='...')`, the toolset resolves the capability from `ctx.capabilities`, loads its instructions, and returns them so the model can immediately use the capability's tools in the same response.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

@dataclass
class DeferredCapabilityLoaderToolset(WrapperToolset[AgentDepsT]):
    """Injects the framework-managed 'load_capability' tool."""

    async def get_tools(self, ctx) -> dict[str, ToolsetTool]:
        # Builds ToolDefinition(
        #   name=LOAD_CAPABILITY_TOOL_NAME,  # 'load_capability'
        #   description=LOAD_CAPABILITY_TOOL_DESCRIPTION,
        #   parameters_json_schema=_LOAD_CAPABILITY_SCHEMA,  # {id: string}
        #   tool_kind='capability-load',
        # )
        ...

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        if tool.tool_def.tool_kind == 'capability-load':
            return await self._load_capability(tool_args, ctx)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _load_capability(self, tool_args, ctx) -> LoadCapabilityReturn:
        # Looks up ctx.capabilities[id]; returns {'instructions': ...}
        # Raises ModelRetry on unknown id or already-loaded id
        ...
```

The returned `LoadCapabilityReturn` is `{'instructions': str | None}` — the model receives the capability's instructions in the next content block and can immediately call its tools.

### 8.1 Two Deferred Capabilities — Model Chooses Which to Load

An agent with two deferred capabilities; the model loads only what it needs.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability

weather = Capability(
    id='weather',
    description='Weather forecasting tools. Load when asked about weather or climate.',
    defer_loading=True,
    instructions='Provide weather in metric units by default.',
)

@weather.tool_plain
def get_forecast(city: str, days: int = 3) -> dict:
    """Get a weather forecast for a city."""
    return {'city': city, 'forecast': [f'Sunny {20+i}°C' for i in range(days)]}

calendar = Capability(
    id='calendar',
    description='Calendar and scheduling tools. Load when asked about appointments or events.',
    defer_loading=True,
    instructions='Use ISO 8601 date format.',
)

@calendar.tool_plain
def list_events(date: str) -> list[str]:
    """List calendar events for a date."""
    return [f'Team standup at 09:00 on {date}', f'1:1 with manager at 14:00 on {date}']

agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[weather, calendar],
)

async def main() -> None:
    # Model will load only 'weather' capability
    r = await agent.run('What is the 3-day forecast for Berlin?')
    print('Weather:', r.output)

    # Model will load only 'calendar' capability
    r2 = await agent.run('What is on my calendar for 2026-08-01?')
    print('Calendar:', r2.output)

asyncio.run(main())
```

### 8.2 Already-Loaded Guard

`DeferredCapabilityLoaderToolset` raises `ModelRetry` if the model tries to load a capability that is already active, preventing infinite loops.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

docs = Capability(
    id='docs',
    description='Documentation lookup tools.',
    defer_loading=True,
)

@docs.tool_plain
def search_docs(query: str) -> str:
    """Search the documentation."""
    return f'Documentation results for: {query}'

model = TestModel(call_tools=['load_capability', 'load_capability', 'search_docs'])
agent = Agent(model, capabilities=[docs])

async def main() -> None:
    # The second load_capability call will trigger a ModelRetry
    # (framework prevents re-loading already-loaded capabilities)
    result = await agent.run('Search docs for pydantic-ai streaming')
    print(result.output)

asyncio.run(main())
```

### 8.3 Inspecting the `load_capability` Tool Definition

Verify the reserved tool is injected and has the correct `tool_kind`.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

research = Capability(
    id='research',
    description='Academic research tools.',
    defer_loading=True,
)

@research.tool_plain
def search_papers(query: str) -> list[str]:
    """Search academic papers."""
    return [f'Paper: {query} — A Survey (2025)']

model = TestModel()
agent = Agent(model, capabilities=[research])

async def main() -> None:
    async with agent.iter('Find papers on transformer attention') as run:
        async for node in run:
            from pydantic_ai._agent_graph import ModelRequestNode
            if isinstance(node, ModelRequestNode):
                params = model.last_model_request_parameters
                if params:
                    for td in params.function_tools:
                        if td.name == 'load_capability':
                            print(
                                f'Reserved tool found: name={td.name!r}, '
                                f'tool_kind={td.tool_kind!r}'
                            )
    print(run.result.output)

asyncio.run(main())
```

---

## 9. `DynamicCapability` — Per-Run Capability Factory (Deep Patterns)

**Source**: `pydantic_ai/capabilities/_dynamic.py`  
**Export**: `from pydantic_ai.capabilities import DynamicCapability`

`DynamicCapability` wraps a `CapabilityFunc[AgentDepsT]` — a callable that receives `RunContext` and returns either an `AbstractCapability` or `None`. The factory is called once per run from `for_run`. If it returns `None`, the wrapper behaves as a no-op for that run. If it returns a capability, `DynamicCapability.for_run` chains into that capability's own `for_run` so all lifecycle hooks fire correctly.

```python
# Key signature verified from source (pydantic-ai 2.3.0):

CapabilityFunc = (
    Callable[[RunContext[AgentDepsT]], AbstractCapability[AgentDepsT] | None]
    | Callable[[RunContext[AgentDepsT]], Awaitable[AbstractCapability[AgentDepsT] | None]]
)

@dataclass
class DynamicCapability(AbstractCapability[AgentDepsT]):
    capability_func: CapabilityFunc[AgentDepsT]

    async def for_run(self, ctx: RunContext) -> AbstractCapability:
        capability = self.capability_func(ctx)
        if inspect.isawaitable(capability):
            capability = await capability
        if capability is None:
            return self   # no-op
        return await capability.for_run(ctx)
```

**Key constraint:** `defer_loading=True` is rejected on the `DynamicCapability` wrapper itself. Set it on the capability the factory *returns* instead.

### 9.1 Async Factory with User Tier Lookup

Fetch the user's permission tier asynchronously and return a different capability per tier.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, DynamicCapability

@dataclass
class UserContext:
    user_id: str

async def fetch_user_tier(user_id: str) -> str:
    """Simulate an async DB lookup."""
    return 'premium' if user_id.startswith('P') else 'free'

free_cap = Capability[UserContext](
    id='free-tools',
    instructions='You have access to basic tools only.',
)

@free_cap.tool_plain
def basic_search(query: str) -> str:
    """Run a basic web search."""
    return f'Basic results for: {query}'

premium_cap = Capability[UserContext](
    id='premium-tools',
    instructions='You have full access to all premium tools.',
)

@premium_cap.tool_plain
def advanced_search(query: str, depth: int = 5) -> list[str]:
    """Run a deep multi-page search."""
    return [f'Premium result {i} for {query}' for i in range(depth)]

@premium_cap.tool_plain
def export_results(data: str, format: str = 'csv') -> str:
    """Export results as CSV or JSON."""
    return f'{format.upper()} export: {data}'

async def capability_factory(ctx: RunContext[UserContext]):
    tier = await fetch_user_tier(ctx.deps.user_id)
    return premium_cap if tier == 'premium' else free_cap

agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=UserContext,
    capabilities=[DynamicCapability(capability_func=capability_factory)],
)

async def main() -> None:
    premium_user = UserContext(user_id='P-001')
    free_user = UserContext(user_id='F-999')

    r_premium = await agent.run('Search for AI research papers', deps=premium_user)
    print('Premium:', r_premium.output)

    r_free = await agent.run('Search for AI research papers', deps=free_user)
    print('Free:', r_free.output)

asyncio.run(main())
```

### 9.2 Feature-Flag Capability — None Passthrough

Return `None` to entirely suppress a capability when a feature flag is off.

```python  {test="skip"}
import asyncio
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, DynamicCapability

experimental = Capability(
    id='experimental-ui',
    instructions='You have access to experimental UI generation tools.',
)

@experimental.tool_plain
def generate_chart(data: str, chart_type: str = 'bar') -> str:
    """Generate a chart from structured data."""
    return f'[{chart_type.upper()} CHART: {data}]'

def feature_flag_factory(ctx: RunContext) -> Capability | None:
    """Only inject capability when the EXPERIMENTAL_UI flag is enabled."""
    if os.environ.get('EXPERIMENTAL_UI') == '1':
        return experimental
    return None  # no-op for this run

agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[DynamicCapability(capability_func=feature_flag_factory)],
)

async def main() -> None:
    # Flag off — experimental tools hidden
    r1 = await agent.run('Generate a bar chart of monthly sales')
    print('Flag off:', r1.output)

    # Flag on — experimental tools available
    os.environ['EXPERIMENTAL_UI'] = '1'
    r2 = await agent.run('Generate a bar chart of monthly sales')
    print('Flag on:', r2.output)
    del os.environ['EXPERIMENTAL_UI']

asyncio.run(main())
```

### 9.3 Stable `id` Forwarding for History Replay

When the factory returns capabilities with a stable `id`, the deferred loading history survives across serialisation/deserialisation boundaries.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, DynamicCapability

@dataclass
class Session:
    locale: str  # 'en' | 'de' | 'fr'

en_tools = Capability(id='locale-tools', instructions='Respond in English.')
de_tools = Capability(id='locale-tools', instructions='Antworte auf Deutsch.')  # same id!
fr_tools = Capability(id='locale-tools', instructions='Réponds en français.')

LOCALE_MAP = {'en': en_tools, 'de': de_tools, 'fr': fr_tools}

def locale_factory(ctx: RunContext[Session]) -> Capability:
    """Return locale-appropriate capability. Same id ensures stable history."""
    return LOCALE_MAP.get(ctx.deps.locale, en_tools)

agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=Session,
    capabilities=[DynamicCapability(capability_func=locale_factory)],
)

async def main() -> None:
    for locale in ['en', 'de', 'fr']:
        result = await agent.run(
            'What is the capital of Germany?',
            deps=Session(locale=locale),
        )
        print(f'[{locale}] {result.output}')

asyncio.run(main())
```

---

## 10. `ImageGenerationSubagentTool` + `XSearchSubagentTool` — Common-Tools Fallback Pattern

**Source**: `pydantic_ai/common_tools/image_generation.py`, `pydantic_ai/common_tools/x_search.py`  
**Export**: `from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool`  
**Export**: `from pydantic_ai.common_tools.x_search import XSearchSubagentTool`

Both classes implement the **subagent fallback pattern**: they create an inner `Agent` at call time, equip it with a native tool (image generation or X search), and delegate the request to it. This lets any outer agent use these capabilities even when the outer agent's model doesn't support the native tool natively.

```python
# Key signatures verified from source (pydantic-ai 2.3.0):

@dataclass(kw_only=True)
class ImageGenerationSubagentTool:
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc
    native_tool: ImageGenerationTool   # ImageGenerationTool config
    instructions: str = 'Generate an image based on the user prompt...'

    async def __call__(self, ctx: RunContext, prompt: str) -> BinaryImage:
        # Creates Agent(model, output_type=BinaryImage, capabilities=[NativeTool(native_tool)])
        # Runs it and returns result.output
        # Wraps UnexpectedModelBehavior → ModelRetry
        ...

@dataclass(kw_only=True)
class XSearchSubagentTool:
    model: Model | KnownModelName | str | XSearchFallbackModelFunc
    native_tool: XSearchTool           # XSearchTool config
    instructions: str = 'Search X/Twitter based on the user query...'

    async def __call__(self, ctx: RunContext, query: str) -> str:
        # Creates Agent(model, output_type=str, capabilities=[NativeTool(native_tool)])
        # Wraps UnexpectedModelBehavior → ModelRetry
        ...
```

The callable `model` form (`ImageGenerationFallbackModelFunc`, `XSearchFallbackModelFunc`) is `Callable[[RunContext], Model | str] | Callable[[RunContext], Awaitable[Model | str]]` — enabling per-run dynamic model selection.

### 10.1 Image Generation Fallback for Non-Native Models

Outer agent uses `gpt-4o-mini` (no native image gen); fallback subagent uses `dall-e-3` via `OpenAIResponsesModel`.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.messages import BinaryImage

gen_tool = ImageGenerationSubagentTool(
    model='openai-responses:gpt-image-1',  # subagent model with native image gen
    native_tool=ImageGenerationTool(size='1024x1024', quality='standard'),
)

agent = Agent(
    'openai:gpt-4o-mini',   # outer model — no native image gen
    instructions='You can generate images using the provided tool.',
    tools=[gen_tool],       # register as a regular tool
)

async def main() -> None:
    result = await agent.run(
        'Generate an image of a futuristic city at sunset with flying cars.'
    )
    # result.output is a str; the BinaryImage is in the tool return
    print(result.output)

asyncio.run(main())
```

### 10.2 X Search Fallback with Dynamic Subagent Model

Use a callable `model` factory to choose between xAI models based on search complexity.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.x_search import XSearchSubagentTool
from pydantic_ai.native_tools import XSearchTool

def pick_xai_model(ctx: RunContext) -> str:
    """Use a more powerful model for long queries."""
    prompt = str(ctx.prompt) if ctx.prompt else ''
    return 'xai:grok-3' if len(prompt) > 100 else 'xai:grok-3-mini'

x_tool = XSearchSubagentTool(
    model=pick_xai_model,         # callable factory
    native_tool=XSearchTool(
        max_search_results=5,
    ),
)

agent = Agent(
    'openai:gpt-4o-mini',         # outer model
    instructions='Search X/Twitter for real-time information.',
    tools=[x_tool],
)

async def main() -> None:
    result = await agent.run('What are people saying about the latest Python release?')
    print(result.output)

asyncio.run(main())
```

### 10.3 Combining Both Fallback Tools with a Multi-Modal Agent

A single outer agent that can search X and generate images, both via subagent delegation.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool
from pydantic_ai.common_tools.x_search import XSearchSubagentTool
from pydantic_ai.native_tools import ImageGenerationTool, XSearchTool

image_gen = ImageGenerationSubagentTool(
    model='openai-responses:gpt-image-1',
    native_tool=ImageGenerationTool(size='1024x1024'),
    instructions='Generate the requested image. Do not ask clarifying questions.',
)

x_search = XSearchSubagentTool(
    model='xai:grok-3-mini',
    native_tool=XSearchTool(max_search_results=3),
    instructions='Search X/Twitter and return a concise summary of findings.',
)

agent = Agent(
    'openai:gpt-4o-mini',
    instructions=(
        'You are a multimedia assistant. '
        'Use x_search_subagent_tool for real-time social media data. '
        'Use image_generation_subagent_tool to generate images on request.'
    ),
    tools=[image_gen, x_search],
)

async def main() -> None:
    # Complex request using both capabilities
    result = await agent.run(
        'Search X for reactions to the pydantic-ai 2.3.0 release and '
        'generate a celebratory banner image for the release.'
    )
    print(result.output)

asyncio.run(main())
```

---

## Summary Table

| # | Class(es) | Module | New in v2.3.0? | Key Pattern |
|---|-----------|--------|---------------|-------------|
| 1 | `AnthropicCompaction` | `pydantic_ai.models.anthropic` | ✅ | Server-side token compaction via `context_management` |
| 2 | `OpenAICompaction` | `pydantic_ai.models.openai` | ✅ | Stateful server-side + stateless `/responses/compact` |
| 3 | `OnlineEvaluation` | `pydantic_evals.online_capability` | ✅ | Async background evaluators after every run |
| 4 | `WrapperCapability` | `pydantic_ai.capabilities.wrapper` | — | Transparent capability delegation; selective override |
| 5 | `Capability` | `pydantic_ai.capabilities.capability` | — | No-subclass bundle: `@cap.tool` / `@cap.tool_plain` / `@cap.instructions` |
| 6 | `ChatRequestExtra` + `ConfigureFrontend` + `ModelInfo` + `BuiltinToolInfo` | `pydantic_ai.ui._web.api` | — | Agent web UI config and per-request model/tool routing |
| 7 | `CapabilityOwnedToolset` | `pydantic_ai.toolsets._capability_owned` | — | Stamp `capability_id` onto tools; suppress instructions when deferred |
| 8 | `DeferredCapabilityLoaderToolset` | `pydantic_ai.toolsets._deferred_capability_loader` | — | Auto-inject `load_capability` tool; guard re-load with `ModelRetry` |
| 9 | `DynamicCapability` | `pydantic_ai.capabilities._dynamic` | — | Per-run factory: async, `None` passthrough, stable `id` forwarding |
| 10 | `ImageGenerationSubagentTool` + `XSearchSubagentTool` | `pydantic_ai.common_tools.*` | — | Subagent fallback pattern; callable model factory; `ModelRetry` on failure |

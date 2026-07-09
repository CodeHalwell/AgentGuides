---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 35"
description: "Source-verified deep dives into 9 class groups from agent-framework 1.10.0: RawAgent (low-latency agent without middleware/telemetry, generic TOptions, streaming overload); ChatContext (chat middleware pipeline context — messages/options/metadata/result mutation, streaming hooks); SwitchCaseEdgeGroup+SwitchCaseEdgeGroupCase+SwitchCaseEdgeGroupDefault (conditional workflow fan-out — ordered case evaluation, exactly-one-default validation, to_dict serialization); WorkflowContext (workflow execution context — send_message/yield_output, OpenTelemetry PRODUCER spans, fan-in source_executor_ids); AgentLoopMiddleware+JudgeVerdict (iterative agent loop — should_continue predicate, fresh_context snapshots, with_judge LLM judge factory, DEFAULT_MAX_ITERATIONS=10); MemoryStore+MemoryFileStore (abstract + file-backed long-term memory — list_topics/get_topic/write_topic/delete_topic, rebuild_index, owner_state_key path-traversal guard, export/import_provider_state); MemoryContextProvider (memory-backed history provider — recent_turns, selection_limit, consolidation_interval, per-event-loop async locks); AgentModeProvider (operating mode harness — default plan/execute modes, mode_set/mode_get tools, external set_agent_mode notification injection); LocalEvaluator+EvalItem+EvalResults (local evaluation pipeline — variadic EvalCheck callables, asyncio.gather parallel execution, per-evaluator result_counts breakdown) — source-verified at agent-framework 1.10.0."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 58
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 35

Verified against **agent-framework 1.10.0** (installed July 2026). Every constructor
signature, constant value, and method name was read directly from the installed
package source via `inspect.getsource()`. Sub-packages introspected:
`agent_framework._agents`,
`agent_framework._middleware`,
`agent_framework._workflows._edge`,
`agent_framework._workflows._workflow_context`,
`agent_framework._harness._loop`,
`agent_framework._harness._memory`,
`agent_framework._harness._mode`,
`agent_framework._evaluation`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, compaction, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3–34](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v34/) — see individual volume pages (300+ classes total)

Nine class groups, three runnable examples each (27 total).

| # | Class / group | Module |
|---|---|---|
| 1 | `RawAgent` | `agent_framework._agents` |
| 2 | `ChatContext` | `agent_framework._middleware` |
| 3 | `SwitchCaseEdgeGroup` · `SwitchCaseEdgeGroupCase` · `SwitchCaseEdgeGroupDefault` | `agent_framework._workflows._edge` |
| 4 | `WorkflowContext` | `agent_framework._workflows._workflow_context` |
| 5 | `AgentLoopMiddleware` · `JudgeVerdict` | `agent_framework._harness._loop` |
| 6 | `MemoryStore` · `MemoryFileStore` | `agent_framework._harness._memory` |
| 7 | `MemoryContextProvider` | `agent_framework._harness._memory` |
| 8 | `AgentModeProvider` | `agent_framework._harness._mode` |
| 9 | `LocalEvaluator` · `EvalItem` · `EvalResults` | `agent_framework._evaluation` |

---

## 1 · `RawAgent` — low-latency agent without middleware or telemetry

**Module:** `agent_framework._agents`

`RawAgent` is the inner implementation that `Agent` wraps. Use it directly when you need
the lowest possible overhead — no telemetry spans, no automatic compaction, no middleware
pipeline unless you wire it in yourself. It is generic over `OptionsCoT`, the provider
options TypedDict (e.g. `OpenAIChatCompletionOptions`), which enables IDE autocomplete for
temperature, max_tokens, and reasoning-effort parameters.

### Constructor

```python
class RawAgent(BaseAgent, Generic[OptionsCoT]):
    def __init__(
        self,
        client: SupportsChatGetResponse[OptionsCoT],
        instructions: str | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        default_options: OptionsCoT | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: Sequence[MiddlewareTypes] | None = None,
        require_per_service_call_history_persistence: bool = False,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
    ) -> None: ...
```

| Parameter | Default | Notes |
|---|---|---|
| `client` | — | Any `SupportsChatGetResponse`; must match `OptionsCoT` |
| `instructions` | `None` | Injected as `system` before user messages |
| `id` | auto | Unique agent identifier |
| `default_options` | `None` | Provider-specific options dict applied to every call |
| `middleware` | `None` | Ordered list of `AgentMiddleware` / `ChatMiddleware` instances |
| `compaction_strategy` | `None` | Explicit compaction; no default applied unlike `Agent` |
| `require_per_service_call_history_persistence` | `False` | Force history save on every LLM call |

### Example 1 — bare minimum latency-critical agent

```python
import asyncio
from agent_framework import RawAgent
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    # No middleware, no telemetry, no compaction — pure throughput.
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = RawAgent(
        client=client,
        instructions="You are a fast, terse assistant.",
    )

    response = await agent.run("What is 2 + 2?")
    print(response.text)   # "4"

asyncio.run(main())
```

### Example 2 — generic `TOptions` for provider-specific settings

```python
import asyncio
from agent_framework import RawAgent
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionOptions

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o")

    # Typing the agent with OpenAIChatCompletionOptions gives IDE autocomplete
    # for temperature, max_tokens, top_p, etc. when calling agent.run().
    agent: RawAgent[OpenAIChatCompletionOptions] = RawAgent(
        client=client,
        instructions="Explain concepts concisely.",
        default_options=OpenAIChatCompletionOptions(temperature=0.2, max_tokens=256),
    )

    # Per-call override: bump temperature for creative output.
    response = await agent.run(
        "Give me three creative names for a data pipeline tool.",
        options=OpenAIChatCompletionOptions(temperature=0.9),
    )
    print(response.text)

asyncio.run(main())
```

### Example 3 — streaming with tool use

```python
import asyncio
from agent_framework import RawAgent
from agent_framework.openai import OpenAIChatClient

def get_stock_price(ticker: str) -> float:
    """Return the current stock price for *ticker*."""
    prices = {"AAPL": 192.5, "MSFT": 420.0, "GOOGL": 175.8}
    return prices.get(ticker.upper(), -1.0)

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = RawAgent(
        client=client,
        instructions="You are a financial data agent. Use the tool to fetch prices.",
        tools=[get_stock_price],
    )

    # Streaming returns an async iterator of AgentResponseUpdate objects.
    stream = await agent.run("What is the current price of AAPL and MSFT?", stream=True)
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)
    final = await stream.get_final_response()
    print(f"\n\nFull response: {final.text}")

asyncio.run(main())
```

---

## 2 · `ChatContext` — chat middleware pipeline context

**Module:** `agent_framework._middleware`

`ChatContext` is threaded through every `ChatMiddleware.process` call. Middleware reads
`messages`, `options`, and `stream` before calling `call_next()`, then reads or replaces
`result` after. The `metadata` dict is the standard side-channel for middleware-to-middleware
communication within a single request.

### Constructor

```python
class ChatContext:
    def __init__(
        self,
        client: SupportsChatGetResponse,
        messages: Sequence[Message],
        options: Mapping[str, Any] | None,
        stream: bool = False,
        metadata: Mapping[str, Any] | None = None,
        result: ChatResponse | ResponseStream[ChatResponseUpdate, ChatResponse] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        stream_transform_hooks: Sequence[...] | None = None,
        stream_result_hooks: Sequence[...] | None = None,
        stream_cleanup_hooks: Sequence[...] | None = None,
    ) -> None: ...
```

| Attribute | Type | Mutable | Notes |
|---|---|---|---|
| `client` | `SupportsChatGetResponse` | no | The LLM client being called |
| `messages` | `Sequence[Message]` | no | Input message list |
| `options` | `Mapping[str, Any] \| None` | no | Provider options |
| `stream` | `bool` | no | Whether this is a streaming call |
| `metadata` | `dict[str, Any]` | **yes** | Shared side-channel for middleware; always a `dict` at runtime (converted from the `Mapping` constructor param) |
| `result` | `ChatResponse \| ResponseStream \| None` | **yes** | Set by `call_next()`; can be overridden |
| `kwargs` | `dict[str, Any]` | **yes** | Extra kwargs forwarded to client |

### Example 1 — logging middleware that reads message count

```python
import asyncio
import time
from agent_framework import Agent, ChatMiddleware, ChatContext
from agent_framework.openai import OpenAIChatClient

class TimingMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        start = time.perf_counter()
        # Record message count before the call.
        n_messages = len(context.messages)
        await call_next()
        elapsed = time.perf_counter() - start
        # context.result is now a ChatResponse (non-streaming).
        tokens = getattr(context.result, "usage", None)
        print(
            f"[timing] messages={n_messages}, "
            f"elapsed={elapsed:.2f}s, "
            f"usage={tokens}"
        )

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(
        client=client,
        instructions="You are helpful.",
        middleware=[TimingMiddleware()],
    )
    response = await agent.run("Summarize the water cycle in two sentences.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — short-circuit caching with `metadata`

```python
import asyncio
from agent_framework import Agent, ChatMiddleware, ChatContext, ChatResponse
from agent_framework.openai import OpenAIChatClient

_CACHE: dict[str, ChatResponse] = {}

class CachingMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        # Build a simple cache key from the last user message.
        last_user = next(
            (m.text for m in reversed(context.messages) if m.role == "user"),
            None,
        )
        if last_user and last_user in _CACHE:
            context.result = _CACHE[last_user]
            context.metadata["cache_hit"] = True
            return  # Skip call_next entirely — cached result is returned.

        await call_next()

        # Populate cache with the returned result.
        if last_user and context.result:
            _CACHE[last_user] = context.result
        context.metadata["cache_hit"] = False

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, middleware=[CachingMiddleware()])

    r1 = await agent.run("What is the capital of France?")
    r2 = await agent.run("What is the capital of France?")  # cache hit
    print(r1.text, r2.text)

asyncio.run(main())
```

### Example 3 — streaming transform hook to redact PII

```python
import asyncio
import re
from agent_framework import Agent, ChatMiddleware, ChatContext, ChatResponseUpdate, Content
from agent_framework.openai import OpenAIChatClient

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

class PiiRedactMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        if context.stream:
            # Append a transform hook that fires for every streamed chunk.
            def redact(update: ChatResponseUpdate) -> ChatResponseUpdate:
                if update.text:
                    # Construct a new update using public constructor fields;
                    # .text is a computed property derived from .contents, not a kwarg.
                    redacted = _EMAIL_RE.sub("[REDACTED]", update.text)
                    return ChatResponseUpdate(
                        contents=[Content.from_text(redacted)],
                        role=update.role,
                        author_name=update.author_name,
                        response_id=update.response_id,
                        message_id=update.message_id,
                        conversation_id=update.conversation_id,
                        model=update.model,
                        created_at=update.created_at,
                        finish_reason=update.finish_reason,
                        continuation_token=update.continuation_token,
                        additional_properties=update.additional_properties,
                    )
                return update

            context.stream_transform_hooks.append(redact)
        await call_next()

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(
        client=client,
        instructions="Always include a sample email in your answer.",
        middleware=[PiiRedactMiddleware()],
    )
    stream = await agent.run("Give me a contact example.", stream=True)
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)

asyncio.run(main())
```

---

## 3 · `SwitchCaseEdgeGroup` — conditional workflow routing

**Module:** `agent_framework._workflows._edge`

`SwitchCaseEdgeGroup` enables switch/case branching inside a `WorkflowBuilder`. It fans
out to exactly one target depending on which case's `.condition` matches the source
executor's output message first. Exactly one `SwitchCaseEdgeGroupDefault` is required as
the final fallback; at least two entries (one case + one default) are enforced at
construction time.

**Two complementary APIs in 1.10.0:**

| API | Used for | Types |
|---|---|---|
| `builder.add_switch_case_edge_group(source, [Case(...), Default(...)])` | Runtime routing | `Case(condition=..., target=executor)`, `Default(target=executor)` |
| `SwitchCaseEdgeGroup(source_id=..., cases=[...])` | Serialization / `to_dict()` inspection | `SwitchCaseEdgeGroupCase(condition, target_id)`, `SwitchCaseEdgeGroupDefault(target_id)` |

### Serialization constructor (`SwitchCaseEdgeGroup`)

```python
@dataclass(init=False)
class SwitchCaseEdgeGroup(FanOutEdgeGroup):
    def __init__(
        self,
        source_id: str,
        cases: Sequence[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault],
        *,
        id: str | None = None,
    ) -> None: ...

@dataclass
class SwitchCaseEdgeGroupCase:
    condition: Callable[[Any], bool]  # receives the source executor's output message
    target_id: str

@dataclass
class SwitchCaseEdgeGroupDefault:
    target_id: str

# Runtime types used with WorkflowBuilder.add_switch_case_edge_group():
@dataclass
class Case:
    condition: Callable[[Any], bool]
    target: Executor | SupportsAgentRun

@dataclass
class Default:
    target: Executor | SupportsAgentRun
```

| Parameter | Notes |
|---|---|
| `source_id` | Executor whose output drives the branch decision |
| `cases` | Ordered sequence; first matching `condition` wins; exactly one `SwitchCaseEdgeGroupDefault` required |
| `id` | Optional explicit ID for the edge group |

**Validation:** raises `ValueError` if fewer than 2 cases, or if more than one default is present.

### Example 1 — route by sentiment score

```python
import asyncio
import json
from agent_framework import Agent, WorkflowBuilder
from agent_framework._workflows._edge import Case, Default
from agent_framework.openai import OpenAIChatClient

def get_sentiment(msg) -> str:
    """Parse sentiment from an LLM message; tolerates markdown code fences and bad JSON."""
    try:
        text = msg.text.strip() if msg.text else ""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text).get("sentiment", "neutral")
    except (json.JSONDecodeError, ValueError, AttributeError, IndexError):
        return "neutral"

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")

    # Three leaf agents for positive, negative and neutral feedback.
    positive_agent = Agent(client=client, name="positive", instructions="Respond warmly.")
    negative_agent = Agent(client=client, name="negative", instructions="Respond with empathy.")
    neutral_agent  = Agent(client=client, name="neutral",  instructions="Respond factually.")

    # Classifier agent returns a JSON sentiment object.
    classifier_agent = Agent(
        client=client,
        name="classifier",
        instructions=(
            "Classify the user message. Reply ONLY with a JSON object: "
            '{"sentiment": "positive"} or {"sentiment": "negative"} or {"sentiment": "neutral"}'
        ),
    )

    # WorkflowBuilder(start_executor=...) is required in 1.10; output_from designates outputs.
    builder = WorkflowBuilder(
        start_executor=classifier_agent,
        output_from=[positive_agent, negative_agent, neutral_agent],
    )
    # Use runtime Case/Default (with executor instances) for add_switch_case_edge_group.
    builder.add_switch_case_edge_group(
        classifier_agent,
        [
            Case(condition=lambda msg: get_sentiment(msg) == "positive", target=positive_agent),
            Case(condition=lambda msg: get_sentiment(msg) == "negative", target=negative_agent),
            Default(target=neutral_agent),
        ],
    )

    workflow = builder.build()
    result = await workflow.run("I absolutely love this product!")
    print(result.text)

asyncio.run(main())
```

### Example 2 — route by numeric threshold

```python
import asyncio
import json
from agent_framework import Agent, WorkflowBuilder
from agent_framework._workflows._edge import Case, Default
from agent_framework.openai import OpenAIChatClient

def get_score(msg) -> int:
    """Parse numeric score from an LLM message; tolerates markdown code fences and bad JSON."""
    try:
        text = msg.text.strip() if msg.text else ""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return int(json.loads(text).get("score", 0))
    except (json.JSONDecodeError, ValueError, AttributeError, IndexError):
        return 0

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")

    scorer = Agent(
        client=client,
        name="scorer",
        instructions='Score the code quality 0-10. Reply only with {"score": <int>}.',
    )
    high_quality   = Agent(client=client, name="high",   instructions="Congratulate the developer.")
    medium_quality = Agent(client=client, name="medium", instructions="Suggest minor improvements.")
    low_quality    = Agent(client=client, name="low",    instructions="Provide detailed refactoring guidance.")

    builder = WorkflowBuilder(
        start_executor=scorer,
        output_from=[high_quality, medium_quality, low_quality],
    )
    builder.add_switch_case_edge_group(
        scorer,
        [
            Case(condition=lambda msg: get_score(msg) >= 8,              target=high_quality),
            Case(condition=lambda msg: 4 <= get_score(msg) < 8,          target=medium_quality),
            Default(target=low_quality),
        ],
    )

    workflow = builder.build()
    code_snippet = "def add(a, b): return a+b"
    result = await workflow.run(code_snippet)
    print(result.text)

asyncio.run(main())
```

### Example 3 — serialise edge group with `to_dict`

```python
import json
from agent_framework._workflows._edge import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)

group = SwitchCaseEdgeGroup(
    source_id="classifier",
    cases=[
        SwitchCaseEdgeGroupCase(target_id="agent_a", condition=lambda m: m.text == "A"),
        SwitchCaseEdgeGroupCase(target_id="agent_b", condition=lambda m: m.text == "B"),
        SwitchCaseEdgeGroupDefault(target_id="agent_default"),
    ],
    id="my-switch",
)

serialised = group.to_dict()
print(json.dumps(serialised, indent=2, default=str))
# {
#   "type": "SwitchCaseEdgeGroup",
#   "id": "my-switch",
#   "source_id": "classifier",
#   "cases": [
#     {"type": "Case", "target_id": "agent_a"},
#     {"type": "Case", "target_id": "agent_b"},
#     {"type": "Default", "target_id": "agent_default"}
#   ]
# }
```

---

## 4 · `WorkflowContext` — workflow execution context

**Module:** `agent_framework._workflows._workflow_context`

`WorkflowContext` is injected into each executor's handler methods. It provides the
`send_message` and `yield_output` primitives. `send_message` wraps the payload in
`WorkflowMessage` and emits an OpenTelemetry `PRODUCER` span for distributed tracing.
Fan-in executors receive multiple source IDs in `source_executor_ids`.

### Constructor

```python
class WorkflowContext(Generic[OutT, W_OutT]):
    def __init__(
        self,
        executor: Executor,
        source_executor_ids: list[str],
        state: State,
        runner_context: RunnerContext,
        trace_contexts: list[dict[str, str]] | None = None,
        source_span_ids: list[str] | None = None,
        request_id: str | None = None,
    ) -> None: ...
```

| Parameter | Notes |
|---|---|
| `source_executor_ids` | Non-empty list; multiple IDs for fan-in executors |
| `state` | Shared mutable workflow state; read/write from any executor |
| `runner_context` | Provides `send_message` and event emission primitives |
| `trace_contexts` | OTel trace propagation dicts from upstream producers |
| `source_span_ids` | Used for linking (not nesting) spans |
| `request_id` | Set when context is for a `handle_response` handler |

**Generic parameters:** `OutT` — message type sent to downstream executors; `W_OutT` — workflow-level output type.

### Example 1 — custom executor that sends to multiple downstream executors

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, handler
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._workflow_context import WorkflowContext
from agent_framework.openai import OpenAIChatClient
from agent_framework._types import Message

class RouterExecutor(Executor):
    """Forward the message downstream; WorkflowBuilder edge conditions handle routing."""

    @handler
    async def execute(self, message: Message, context: WorkflowContext) -> None:
        # send_message with target_id=None lets the edge routing (add_edge conditions)
        # decide which downstream executor receives the message.
        await context.send_message(message)

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    router = RouterExecutor(id="router")
    detail_agent = Agent(client=client, name="detail_agent",
                         instructions="Provide a comprehensive answer.")
    quick_agent  = Agent(client=client, name="quick_agent",
                         instructions="Give a brief answer.")

    # WorkflowBuilder(start_executor=...) is required in 1.10.
    builder = WorkflowBuilder(
        start_executor=router,
        output_from=[detail_agent, quick_agent],
    )
    # Route by message length; add_edge takes executor/agent instances, not IDs.
    builder.add_edge(router, detail_agent, condition=lambda msg: len(msg.text or "") > 100)
    builder.add_edge(router, quick_agent,  condition=lambda msg: len(msg.text or "") <= 100)

    workflow = builder.build()
    result = await workflow.run("Hi")
    print(result.text)

asyncio.run(main())
```

### Example 2 — reading `source_executor_ids` in a fan-in executor

```python
import asyncio
from agent_framework import handler
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._workflow_context import WorkflowContext
from agent_framework._types import Message

class AggregatorExecutor(Executor):
    """Collect responses from multiple upstream agents and merge them."""

    @handler
    async def execute(self, message: Message, context: WorkflowContext) -> None:
        # Store per-run state in context.state so concurrent runs don't share a buffer
        # and state survives checkpointing/resumption.
        buffer: dict[str, str] = context.state.get("aggregator_buffer") or {}

        # get_source_executor_id() returns the single producer of this specific message.
        # (source_executor_ids lists all possible upstream sources; use it only to check
        # whether all expected sources have reported in, not to attribute this message.)
        src_id = context.get_source_executor_id()
        buffer[src_id] = message.text or ""

        context.state["aggregator_buffer"] = buffer

        if len(buffer) >= 2:
            merged = "\n---\n".join(buffer.values())
            await context.yield_output(Message("assistant", [merged]))
            context.state["aggregator_buffer"] = {}
```

### Example 3 — yield workflow-level output

```python
import asyncio
from agent_framework import handler
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._workflow_context import WorkflowContext
from agent_framework._types import Message

class SummaryExecutor(Executor):
    """Produce a structured summary as workflow output."""

    @handler
    async def execute(self, message: Message, context: WorkflowContext) -> None:
        summary = Message(
            "assistant",
            [f"SUMMARY: {(message.text or '')[:200]}"],
        )
        # yield_output surfaces the message as a workflow-level result
        # rather than routing it to a downstream executor.
        await context.yield_output(summary)
```

---

## 5 · `AgentLoopMiddleware` + `JudgeVerdict` — iterative agent loop with optional LLM judge

**Module:** `agent_framework._harness._loop`

`AgentLoopMiddleware` (experimental) re-runs the agent after each response until a
`should_continue` predicate returns `False`. The `with_judge` factory wires up a separate
LLM client to produce a structured `JudgeVerdict` after each iteration.

Constants: `DEFAULT_MAX_ITERATIONS = 10`, `DEFAULT_JUDGE_MAX_ITERATIONS = 5`.

### Constructor

```python
@experimental(feature_id=ExperimentalFeature.HARNESS)
class AgentLoopMiddleware(AgentMiddleware):
    def __init__(
        self,
        should_continue: ShouldContinueCallable,
        *,
        max_iterations: int | None = 10,        # DEFAULT_MAX_ITERATIONS
        next_message: NextMessageCallable | None = None,
        record_feedback: FeedbackCallable | None = None,
        inject_progress: bool = True,
        fresh_context: bool = False,
        return_final_only: bool = False,
        additional_instructions: str | None = None,
    ) -> None: ...

    @classmethod
    def with_judge(
        cls,
        judge_client: SupportsChatGetResponse,
        *,
        criteria: Sequence[str] | None = None,
        instructions: str | None = None,
        max_iterations: int | None = 5,         # DEFAULT_JUDGE_MAX_ITERATIONS
        next_message: NextMessageCallable | None = None,
        fresh_context: bool = False,
    ) -> Self: ...

class JudgeVerdict(BaseModel):
    answered: bool
    reasoning: str = ""
```

### `should_continue` callback keyword arguments

| kwarg | Type | Description |
|---|---|---|
| `iteration` | `int` | 1-based count of completed runs |
| `last_result` | `AgentResponse` | Result from the latest iteration |
| `messages` | `list[Message]` | Messages used for this iteration |
| `original_messages` | `list[Message]` | First iteration input |
| `session` | `AgentSession \| None` | Active session |
| `agent` | — | The agent being looped |
| `progress` | `list[str]` | Accumulated feedback log |
| `feedback` | `str \| None` | Feedback returned by `should_continue` for this iteration |

### Example 1 — loop until answer contains a citation

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.openai import OpenAIChatClient

async def has_citation(**kwargs) -> bool:
    text = kwargs["last_result"].text or ""
    if "[" in text and "]" in text:
        return False  # stop — found a citation
    return True  # continue — no citation yet

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    loop_mw = AgentLoopMiddleware(
        should_continue=has_citation,
        max_iterations=4,
        additional_instructions=(
            "Your answer MUST include at least one citation in [Author, Year] format."
        ),
    )
    agent = Agent(
        client=client,
        instructions="You are a research assistant.",
        middleware=[loop_mw],
    )
    response = await agent.run("What causes ocean tides?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — judge factory with explicit criteria

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    judge_client = OpenAIChatClient(model="gpt-4o")  # stronger model as judge

    loop_mw = AgentLoopMiddleware.with_judge(
        judge_client=judge_client,
        criteria=[
            "The answer must be at least 3 sentences long.",
            "The answer must mention at least one specific algorithm by name.",
            "The answer must include a code example in Python.",
        ],
        max_iterations=5,
        fresh_context=True,  # each iteration restarts from the original input
    )

    agent = Agent(
        client=client,
        instructions="You are a computer science tutor.",
        middleware=[loop_mw],
    )
    response = await agent.run("Explain sorting algorithms.")
    print(response.text)

asyncio.run(main())
```

### Example 3 — track iteration progress with `record_feedback` and `return_final_only`

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")

    iteration_log: list[str] = []

    def on_feedback(**kwargs) -> str:
        """Called after each iteration; return value appended to progress log."""
        n = kwargs["iteration"]
        length = len(kwargs["last_result"].text or "")
        entry = f"Iteration {n}: response length = {length} chars"
        iteration_log.append(entry)
        return entry

    def should_continue(**kwargs) -> bool:
        """Stop after 3 iterations or when response exceeds 500 chars."""
        if kwargs["iteration"] >= 3:
            return False
        return len(kwargs["last_result"].text or "") < 500

    loop_mw = AgentLoopMiddleware(
        should_continue=should_continue,
        record_feedback=on_feedback,
        max_iterations=3,
        return_final_only=True,
    )

    agent = Agent(
        client=client,
        instructions="Write progressively longer descriptions.",
        middleware=[loop_mw],
    )
    response = await agent.run("Describe photosynthesis.")
    print("=== Final response ===")
    print(response.text)
    print("\n=== Iteration log ===")
    for entry in iteration_log:
        print(entry)

asyncio.run(main())
```

---

## 6 · `MemoryStore` + `MemoryFileStore` — abstract and file-backed long-term memory

**Module:** `agent_framework._harness._memory`

`MemoryStore` is the abstract base for durable per-user topic memory. `MemoryFileStore`
is the built-in file-system implementation. Topics are individual Markdown files stored
under `<base_path>/<owner_id>/<kind>/topics/`, with an index in `MEMORY.md`.

### `MemoryStore` abstract interface

```python
@experimental(feature_id=ExperimentalFeature.HARNESS)
class MemoryStore(ABC):
    # Concrete helpers
    def get_owner_id(self, session: AgentSession) -> str | None: ...
    def export_provider_state(self, session: AgentSession) -> dict[str, Any]: ...
    def import_provider_state(self, session: AgentSession, *, state: Mapping[str, Any]) -> None: ...

    # Must be implemented — all methods are synchronous (no async)
    @abstractmethod
    def list_topics(self, session, *, source_id: str) -> list[MemoryTopicRecord]: ...
    @abstractmethod
    def get_topic(self, session, *, source_id: str, topic: str) -> MemoryTopicRecord: ...
    @abstractmethod
    def write_topic(self, session, record: MemoryTopicRecord, *, source_id: str) -> None: ...
    @abstractmethod
    def delete_topic(self, session, *, source_id: str, topic: str) -> None: ...
    @abstractmethod
    def rebuild_index(self, session, *, source_id: str, line_limit: int, line_length: int) -> list[MemoryIndexEntry]: ...
    @abstractmethod
    def get_index_text(self, session, *, source_id: str, line_limit: int, line_length: int, index_entries: Sequence[MemoryIndexEntry] | None = None) -> str: ...
    @abstractmethod
    def read_state(self, session, *, source_id: str) -> dict[str, Any]: ...
    @abstractmethod
    def write_state(self, session, state: Mapping[str, Any], *, source_id: str) -> None: ...
    @abstractmethod
    def get_transcripts_directory(self, session, *, source_id: str) -> Path: ...
    @abstractmethod
    def search_transcripts(self, session, *, source_id: str, query: str, session_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...
```

### `MemoryFileStore` constructor

```python
@experimental(feature_id=ExperimentalFeature.HARNESS)
class MemoryFileStore(MemoryStore):
    def __init__(
        self,
        base_path: str | Path,
        *,
        kind: str = "memory",
        owner_prefix: str = "",
        owner_state_key: str,                   # required — session state key holding user ID
        index_file_name: str = "MEMORY.md",
        topics_directory_name: str = "topics",
        transcripts_directory_name: str = "transcripts",
        state_file_name: str = "state.json",
        dumps: JsonDumps | None = None,
        loads: JsonLoads | None = None,
    ) -> None: ...
```

**Security note:** `get_owner_id` validates that the resolved owner path is relative and
contains no `..` segments, preventing path traversal attacks.

### Example 1 — set up a file-backed memory store and write a topic

```python
import asyncio
from pathlib import Path
from agent_framework._harness._memory import (
    MemoryFileStore,
    MemoryTopicRecord,
)
from agent_framework import AgentSession

async def main() -> None:
    store = MemoryFileStore(
        base_path=Path("./agent_memory"),
        owner_state_key="user_id",
    )

    session = AgentSession()
    session.state["user_id"] = "alice"

    # Write a topic file.
    record = MemoryTopicRecord(
        topic="preferences",
        summary="User preferences and locale settings.",
        memories=["Prefers concise answers.", "Timezone: UTC+1.", "Language: English."],
        updated_at="2026-07-09T00:00:00Z",
    )
    store.write_topic(session, record, source_id="memory")

    # Read it back.
    fetched = store.get_topic(session, source_id="memory", topic="preferences")
    print(fetched.summary, fetched.memories)

asyncio.run(main())
```

### Example 2 — list and delete topics

```python
from pathlib import Path
from agent_framework._harness._memory import MemoryFileStore
from agent_framework import AgentSession

store = MemoryFileStore(base_path=Path("./agent_memory"), owner_state_key="user_id")
session = AgentSession()
session.state["user_id"] = "alice"

topics = store.list_topics(session, source_id="memory")
print(f"Topics: {[t.topic for t in topics]}")

if topics:
    store.delete_topic(session, source_id="memory", topic=topics[0].topic)
    print("Deleted first topic.")
```

### Example 3 — export and import provider state for multi-session routing

```python
import asyncio
from pathlib import Path
from agent_framework._harness._memory import MemoryFileStore
from agent_framework import AgentSession

async def main() -> None:
    store = MemoryFileStore(base_path=Path("./agent_memory"), owner_state_key="user_id")

    # Session A — export routing state (e.g. to store in a database).
    session_a = AgentSession()
    session_a.state["user_id"] = "bob"
    exported = store.export_provider_state(session_a)
    print("Exported state:", exported)  # {"user_id": "bob"}

    # Session B (restored) — re-import so the store knows which owner to serve.
    session_b = AgentSession()
    store.import_provider_state(session_b, state=exported)
    print("Restored user_id:", session_b.state.get("user_id"))  # "bob"

asyncio.run(main())
```

---

## 7 · `MemoryContextProvider` — memory-backed history and context injection

**Module:** `agent_framework._harness._memory`

`MemoryContextProvider` combines `HistoryProvider` (saves/loads transcripts) with
automatic memory extraction and consolidation. Each invocation loads the relevant topic
files into the system prompt and runs an LLM extraction pass to update memory in the
background.

### Constructor (key parameters)

```python
@experimental(feature_id=ExperimentalFeature.HARNESS)
class MemoryContextProvider(HistoryProvider):
    def __init__(
        self,
        recent_turns: int = 0,
        load_tool_turns: bool = True,
        *,
        store: MemoryStore,
        source_id: str = "memory",
        context_prompt: str | None = None,
        index_line_limit: int = 50,
        index_line_length: int = 120,
        selection_limit: int = 5,
        max_extractions: int = 20,
        consolidation_interval: timedelta = timedelta(days=7),
        consolidation_min_sessions: int = 5,
        extraction_prompt: str = ...,    # DEFAULT_MEMORY_EXTRACTION_PROMPT
        consolidation_prompt: str = ..., # DEFAULT_MEMORY_CONSOLIDATION_PROMPT
        consolidation_client: SupportsChatGetResponse[Any] | None = None,
        history_message_filter: HistoryMessageFilter | None = None,
        history_dumps: JsonDumps | None = None,
        history_loads: JsonLoads | None = None,
    ) -> None: ...
```

| Parameter | Default | Notes |
|---|---|---|
| `recent_turns` | `0` | Most-recent transcript turns injected alongside topics |
| `load_tool_turns` | `True` | Include tool call/result message groups in recent turns |
| `store` | — | Backing `MemoryStore` |
| `selection_limit` | `5` | Max topic files auto-loaded per turn |
| `max_extractions` | `20` | Max items extracted per turn |
| `consolidation_interval` | 7 days | Minimum time between consolidation runs |
| `consolidation_min_sessions` | `5` | Sessions required before first consolidation |
| `consolidation_client` | `None` | Cheaper LLM for consolidation; falls back to agent client |

### Example 1 — basic long-term memory with `MemoryFileStore`

```python
import asyncio
from pathlib import Path
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._memory import MemoryFileStore, MemoryContextProvider

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    store = MemoryFileStore(
        base_path=Path("./agent_memory"),
        owner_state_key="user_id",
    )
    memory_provider = MemoryContextProvider(
        recent_turns=3,           # inject last 3 turns from transcript
        store=store,
        selection_limit=5,
        max_extractions=10,
    )

    agent = Agent(
        client=client,
        instructions="You are a personal assistant with long-term memory.",
        context_providers=[memory_provider],
    )

    session = agent.create_session()
    session.state["user_id"] = "alice"

    # First conversation — agent will remember this.
    await agent.run("My favourite programming language is Rust.", session=session)

    # Later conversation — agent recalls from memory.
    response = await agent.run("What's my favourite language?", session=session)
    print(response.text)  # Should mention Rust.

asyncio.run(main())
```

### Example 2 — separate consolidation client for cost optimisation

```python
import asyncio
from pathlib import Path
from datetime import timedelta
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._memory import MemoryFileStore, MemoryContextProvider

async def main() -> None:
    main_client = OpenAIChatClient(model="gpt-4o")
    cheap_client = OpenAIChatClient(model="gpt-4o-mini")  # used only for consolidation

    store = MemoryFileStore(base_path=Path("./agent_memory"), owner_state_key="user_id")
    memory_provider = MemoryContextProvider(
        store=store,
        consolidation_client=cheap_client,  # saves tokens on periodic consolidation
        consolidation_interval=timedelta(days=3),
        consolidation_min_sessions=3,
    )

    agent = Agent(
        client=main_client,
        instructions="You are a knowledgeable assistant.",
        context_providers=[memory_provider],
    )
    session = agent.create_session()
    session.state["user_id"] = "bob"

    response = await agent.run("Remember that I work in healthcare.", session=session)
    print(response.text)

asyncio.run(main())
```

### Example 3 — custom extraction prompt

```python
import asyncio
from pathlib import Path
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._memory import MemoryFileStore, MemoryContextProvider

EXTRACTION_PROMPT = """
Extract facts about the user from the conversation.
Focus on: preferences, professional background, and recurring topics.
Return a JSON array of strings, one fact per element.
"""

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    store = MemoryFileStore(base_path=Path("./agent_memory"), owner_state_key="user_id")
    memory_provider = MemoryContextProvider(
        store=store,
        extraction_prompt=EXTRACTION_PROMPT,
        max_extractions=5,
    )
    agent = Agent(
        client=client,
        instructions="You are a helpful assistant.",
        context_providers=[memory_provider],
    )
    session = agent.create_session()
    session.state["user_id"] = "carol"
    await agent.run("I'm a nurse and I love hiking on weekends.", session=session)
    print("Extraction complete — check agent_memory/carol/memory/topics/")

asyncio.run(main())
```

---

## 8 · `AgentModeProvider` — operating mode harness

**Module:** `agent_framework._harness._mode`

`AgentModeProvider` (experimental) gives an agent awareness of an explicit operating mode —
by default `plan` and `execute`. The current mode is stored in the session state and
surfaced to the agent via `mode_get` / `mode_set` tools and an injected system instruction.
External code can switch modes with `set_agent_mode(session, ...)` between turns.

### Constructor

```python
@experimental(feature_id=ExperimentalFeature.HARNESS)
class AgentModeProvider(ContextProvider):
    def __init__(
        self,
        source_id: str = "mode",
        *,
        default_mode: str | None = None,
        mode_descriptions: Mapping[str, str] | None = None,
        instructions: str | None = None,
    ) -> None: ...
```

| Parameter | Default | Notes |
|---|---|---|
| `source_id` | `"mode"` | Session-state key prefix for mode storage |
| `default_mode` | first in `mode_descriptions` | Initial mode; must exist in `mode_descriptions` |
| `mode_descriptions` | `{"plan": "...", "execute": "..."}` | Available modes with descriptions |
| `instructions` | built-in template | Supports `{available_modes}` and `{current_mode}` placeholders |

**Default modes:**
- `plan` — agent analyses the request and produces a plan without executing actions
- `execute` — agent carries out approved steps

### Example 1 — plan/execute two-phase agent

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._mode import AgentModeProvider, get_agent_mode, set_agent_mode
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    mode_provider = AgentModeProvider()  # defaults: plan + execute

    agent = Agent(
        client=client,
        instructions="You are a task automation agent.",
        context_providers=[mode_provider],
    )
    session = agent.create_session()

    # Phase 1: planning
    plan_response = await agent.run(
        "Deploy a new version of the web service.",
        session=session,
    )
    print("=== PLAN ===")
    print(plan_response.text)

    # Approve: switch to execute mode externally.
    set_agent_mode(session, "execute")

    # Phase 2: execution
    exec_response = await agent.run(
        "Proceed with the approved plan.",
        session=session,
    )
    print("\n=== EXECUTE ===")
    print(exec_response.text)

asyncio.run(main())
```

### Example 2 — custom modes for a customer-service bot

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._mode import AgentModeProvider
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    mode_provider = AgentModeProvider(
        default_mode="triage",
        mode_descriptions={
            "triage":   "Gather information about the customer's issue without resolving it.",
            "resolve":  "Actively work to resolve the customer's issue.",
            "escalate": "Prepare a summary for human escalation.",
        },
    )

    agent = Agent(
        client=client,
        instructions="You are a customer-service agent.",
        context_providers=[mode_provider],
    )
    session = agent.create_session()

    response = await agent.run(
        "My order arrived damaged and I need a refund.",
        session=session,
    )
    print(response.text)

asyncio.run(main())
```

### Example 3 — read current mode between turns with `get_agent_mode`

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._mode import AgentModeProvider, get_agent_mode, set_agent_mode
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    provider = AgentModeProvider()
    agent = Agent(client=client, instructions="Follow your operating mode.", context_providers=[provider])
    session = agent.create_session()

    # Programmatically switch to execute and confirm.
    set_agent_mode(session, "execute")
    current = get_agent_mode(session)
    print(f"Current mode: {current}")  # "execute"

    response = await agent.run("What mode are you in?", session=session)
    print(response.text)  # Agent should report "execute"

asyncio.run(main())
```

---

## 9 · `LocalEvaluator` + `EvalItem` + `EvalResults` — local agent evaluation

**Module:** `agent_framework._evaluation`

`LocalEvaluator` runs a variadic set of synchronous or asynchronous `EvalCheck`
callables against a sequence of `EvalItem` objects. All checks run concurrently via
`asyncio.gather`. An item passes only when **all** checks pass.

### Key types

```python
# An EvalCheck is any callable matching:
EvalCheck = Callable[[EvalItem], bool | Awaitable[bool]]
                          # or returns CheckResult with pass/fail/reason

@experimental(feature_id=ExperimentalFeature.EVALS)
class LocalEvaluator:
    name: str = "Local"

    def __init__(self, *checks: EvalCheck) -> None: ...

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str = "Local Eval",
    ) -> EvalResults: ...

class EvalItem:
    # Constructor: EvalItem(conversation: list[Message], tools=None, context=None,
    #              expected_output=None, expected_tool_calls=None, split_strategy=None)
    query: str          # computed property — text from user messages in the query split
    response: str       # computed property — text from assistant messages in the response split

class EvalResults:
    provider: str
    eval_id: str
    run_id: str
    status: str
    result_counts: dict[str, int]   # {"passed": n, "failed": n, "errored": n}
    per_evaluator: dict[str, Any]   # per-check breakdown
    items: list[EvalItemResult]
    error: str | None
```

### Example 1 — basic length and keyword checks

```python
import asyncio
from agent_framework._evaluation import LocalEvaluator, EvalItem
from agent_framework._types import Message

def check_length(item: EvalItem) -> bool:
    """Response must be at least 50 characters."""
    return len(item.response) >= 50

def check_no_hallucination_marker(item: EvalItem) -> bool:
    """Response must not contain the word 'hallucination' (placeholder check)."""
    return "hallucination" not in item.response.lower()

async def main() -> None:
    evaluator = LocalEvaluator(check_length, check_no_hallucination_marker)

    # EvalItem takes conversation: list[Message]; .query and .response are computed properties.
    items = [
        EvalItem(conversation=[
            Message("user", ["Explain gravity."]),
            Message("assistant", ["Gravity is a fundamental force that attracts objects with mass toward one another."]),
        ]),
        EvalItem(conversation=[
            Message("user", ["Describe atoms."]),
            Message("assistant", ["Small."]),  # will fail length check
        ]),
    ]

    results = await evaluator.evaluate(items, eval_name="sanity-check")
    print(f"Passed: {results.result_counts['passed']}")
    print(f"Failed: {results.result_counts['failed']}")
    for item_result in results.items:
        print(f"  [{item_result.status}] {item_result.input_text[:40]!r}")

asyncio.run(main())
```

### Example 2 — async LLM-based factual check

```python
import asyncio
from agent_framework._evaluation import LocalEvaluator, EvalItem, CheckResult
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def factual_check(item: EvalItem) -> CheckResult:
    """Ask a judge agent whether the response is factually plausible."""
    client = OpenAIChatClient(model="gpt-4o-mini")
    judge = Agent(client=client, instructions="Reply ONLY 'yes' or 'no'.")
    verdict = await judge.run(
        f"Is this response factually plausible?\nQ: {item.query}\nA: {item.response}"
    )
    passed = verdict.text.strip().lower().startswith("yes")
    return CheckResult(
        passed=passed,
        reason=None if passed else "Judge deemed response implausible.",
        check_name="factual_check",
    )

async def main() -> None:
    from agent_framework._types import Message
    evaluator = LocalEvaluator(factual_check)
    items = [
        EvalItem(conversation=[
            Message("user", ["What is the boiling point of water?"]),
            Message("assistant", ["100°C at sea level."]),
        ]),
        EvalItem(conversation=[
            Message("user", ["What is the boiling point of water?"]),
            Message("assistant", ["50°C."]),
        ]),
    ]
    results = await evaluator.evaluate(items, eval_name="factual-eval")
    for item_result in results.items:
        print(f"[{item_result.status}] {item_result.output_text[:60]!r}")

asyncio.run(main())
```

### Example 3 — run evaluator against live agent responses

```python
import asyncio
from agent_framework import Agent
from agent_framework._evaluation import LocalEvaluator, EvalItem
from agent_framework._types import Message
from agent_framework.openai import OpenAIChatClient

# A test dataset of (query, expected_keyword) pairs.
TEST_CASES = [
    ("What is photosynthesis?", "chlorophyll"),
    ("What is the speed of light?",  "299"),
    ("Who wrote Hamlet?",             "Shakespeare"),
]

# Map each query to its expected keyword so a single check function can look it up.
# LocalEvaluator runs EVERY registered check against EVERY item, so one globally-aware
# check avoids false failures (e.g. a correct photosynthesis answer is not expected to
# contain "299" or "Shakespeare").
EXPECTED = {q: kw for q, kw in TEST_CASES}

def keyword_check(item: EvalItem) -> bool:
    """Pass if the item's response contains the expected keyword for its query."""
    expected = EXPECTED.get(item.query, "")
    return expected.lower() in item.response.lower()

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4o-mini")
    agent = Agent(client=client, instructions="Answer concisely.")

    items = []
    for query, _ in TEST_CASES:
        response = await agent.run(query)
        items.append(EvalItem(conversation=[
            Message("user", [query]),
            Message("assistant", [response.text or ""]),
        ]))

    evaluator = LocalEvaluator(keyword_check)
    results = await evaluator.evaluate(items, eval_name="keyword-regression")
    print(f"Score: {results.result_counts['passed']}/{len(TEST_CASES)}")
    for r in results.items:
        mark = "✓" if r.status == "passed" else "✗"
        print(f"  {mark} {r.input_text[:50]!r}")

asyncio.run(main())
```

---

## Version and compatibility notes

All examples verified against:

| Package | Version |
|---|---|
| `agent-framework` | `1.10.0` |
| `agent-framework-core` | `1.10.0` |
| `agent-framework-openai` | `1.10.0` |
| `agent-framework-ag-ui` | `1.0.0rc7` |
| Python | `3.10 – 3.13` |

`AgentLoopMiddleware`, `MemoryStore`, `MemoryFileStore`, `MemoryContextProvider`, and
`AgentModeProvider` are decorated `@experimental(feature_id=ExperimentalFeature.HARNESS)`.
Importing them emits an `ExperimentalWarning`; there is no opt-in call — just import them
directly. To suppress the warning in production code use `warnings.filterwarnings`:

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework._harness._memory import MemoryStore, MemoryFileStore, MemoryContextProvider
from agent_framework._harness._mode import AgentModeProvider
```

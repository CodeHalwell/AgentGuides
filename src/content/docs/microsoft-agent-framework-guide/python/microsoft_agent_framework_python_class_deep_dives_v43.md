---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 43"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.13.0: Evaluator+LocalEvaluator (evaluation protocol — check-function dispatch, async/sync check unification, per-evaluator breakdown); EvalItem+ConversationSplit (eval data model — per_turn_items splitter, query/response derived properties, split_messages override); EvalResults+EvalItemResult+EvalScoreResult+RubricScore (rich result model — raise_for_status, assert_score_at_least, assert_dimension_score_at_least, sub_results workflow eval breakdown); ContextWindowCompactionStrategy+CharacterEstimatorTokenizer (budget-driven two-phase compaction — tool-eviction→truncation pipeline, 4-char/token heuristic, threshold validation); SummarizationStrategy (LLM-based history reduction — target_count+threshold trigger, prompt override, max_summary_input_tokens budget, indirect-prompt-injection risk); SlidingWindowStrategy+SelectiveToolCallCompactionStrategy+TruncationStrategy (windowing + tool-call pruning + oldest-first truncation — keep_last_groups, keep_last_tool_call_groups=0 clear-all, tokenizer-vs-count dual mode); TodoProvider+TodoItem+TodoInput+TodoCompleteInput+TodoSessionStore+TodoFileStore (agent task planning — per-session asyncio.Lock guard, WeakKeyDictionary eviction, path-traversal guard, Windows-reserved-name list); BackgroundAgentsProvider+BackgroundTaskInfo+BackgroundTaskStatus (parallel sub-agent delegation — background_agents_start_task/wait_for_first_completion/get_task_results/continue_task/clear_completed_task tools, RUNNING/COMPLETED/FAILED/LOST lifecycle); ToolApprovalMiddleware+ToolApprovalRule+ToolApprovalState (standing approval rules — argument-aware matching, server_label boundary, auto_approval_rules security warning, queued_approval_requests/collected_approval_responses HITL protocol); FunctionalWorkflow+FunctionalWorkflowAgent+RunContext+StepWrapper+WorkflowInterrupted (decorator-based workflow authoring — @workflow+@step, per-step checkpoint caching, RunContext.request_info HITL, WorkflowInterrupted BaseException signal, FunctionalWorkflowAgent agent adapter) — source-verified at agent-framework 1.13.0."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 66
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 43

Verified against **agent-framework 1.13.0** (installed August 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`.

Sub-packages introspected:
`agent_framework._evaluation`,
`agent_framework._compaction`,
`agent_framework._harness._todo`,
`agent_framework._harness._background_agents`,
`agent_framework._harness._tool_approval`,
`agent_framework._workflows._functional`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) through [Vol. 42](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v42/) — 420+ classes covered.

This volume covers **ten class groups**: the evaluation system end-to-end (Evaluator → EvalItem → EvalResults with rubric scoring), three complementary compaction strategies plus the context-window orchestration layer, the agent task-planning harness (TodoProvider), parallel sub-agent delegation (BackgroundAgentsProvider), standing tool-approval rules (ToolApprovalMiddleware), and the functional (decorator-based) workflow authoring API.

| # | Class / group | Package |
|---|---|---|
| 1 | `Evaluator` · `LocalEvaluator` | `agent_framework._evaluation` |
| 2 | `EvalItem` · `ConversationSplit` | `agent_framework._evaluation` |
| 3 | `EvalResults` · `EvalItemResult` · `EvalScoreResult` · `RubricScore` | `agent_framework._evaluation` |
| 4 | `ContextWindowCompactionStrategy` · `CharacterEstimatorTokenizer` | `agent_framework._compaction` |
| 5 | `SummarizationStrategy` | `agent_framework._compaction` |
| 6 | `SlidingWindowStrategy` · `SelectiveToolCallCompactionStrategy` · `TruncationStrategy` | `agent_framework._compaction` |
| 7 | `TodoProvider` · `TodoItem` · `TodoInput` · `TodoCompleteInput` · `TodoSessionStore` · `TodoFileStore` | `agent_framework._harness._todo` |
| 8 | `BackgroundAgentsProvider` · `BackgroundTaskInfo` · `BackgroundTaskStatus` | `agent_framework._harness._background_agents` |
| 9 | `ToolApprovalMiddleware` · `ToolApprovalRule` · `ToolApprovalState` | `agent_framework._harness._tool_approval` |
| 10 | `FunctionalWorkflow` · `FunctionalWorkflowAgent` · `RunContext` · `StepWrapper` · `WorkflowInterrupted` | `agent_framework._workflows._functional` |

---

## 1 · `Evaluator` · `LocalEvaluator`

**Package:** `agent_framework._evaluation` (import via `from agent_framework import Evaluator, LocalEvaluator`)

`Evaluator` is a `@runtime_checkable Protocol` that any evaluation backend must satisfy. `LocalEvaluator` is the built-in implementation that runs zero-latency check functions entirely in-process — no API calls.

Both are marked `@experimental`.

### Protocol — `Evaluator`

```python
class Evaluator(Protocol):
    name: str

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str,
    ) -> EvalResults: ...
```

Any class that exposes a `name: str` attribute and an `async evaluate(…)` method satisfies the protocol at runtime — no explicit inheritance needed.

### Constructor — `LocalEvaluator`

```python
LocalEvaluator(*checks: EvalCheck)
```

`EvalCheck` is a sync or async callable `(EvalItem) → CheckResult`. Built-in factory helpers include `keyword_check(word)`, `tool_called_check(tool_name)`, and the `@evaluator` decorator for custom check functions.

Each `EvalItem` is scored against **all** checks. An item passes only when every check passes. Per-check breakdown is available in `EvalResults.per_evaluator`.

### Example 1 — Local checks in CI

```python
import asyncio
from agent_framework import Agent, LocalEvaluator, evaluate_agent
from agent_framework._evaluation import keyword_check, tool_called_check

agent = Agent(client=client, instructions="You are a weather assistant.")

local = LocalEvaluator(
    keyword_check("temperature"),      # response must mention temperature
    tool_called_check("get_weather"),  # agent must have called the tool
)

async def main():
    results = await evaluate_agent(
        agent=agent,
        queries=["What's the weather in London?"],
        evaluators=local,
    )
    results[0].raise_for_status()  # raises EvalNotPassedError if any item failed
    print(f"Passed {results[0].passed}/{results[0].total}")

asyncio.run(main())
```

### Example 2 — Custom async check

```python
from agent_framework._evaluation import evaluator, CheckResult, EvalItem

@evaluator
async def no_hallucination_check(item: EvalItem) -> CheckResult:
    # A trivial length-based proxy — replace with a real LLM judge
    is_ok = len(item.response) < 2000
    return CheckResult(
        passed=is_ok,
        reason="Response suspiciously long" if not is_ok else "OK",
        check_name="no_hallucination",
    )

local = LocalEvaluator(no_hallucination_check)
```

### Example 3 — Mixing local + cloud evaluators

```python
import asyncio
from agent_framework.foundry import FoundryEvals
from agent_framework import LocalEvaluator, evaluate_agent

local = LocalEvaluator(keyword_check("weather"))
cloud = FoundryEvals(project_client=foundry_client, model="gpt-4o")

async def main():
    results = await evaluate_agent(
        agent=agent,
        queries=queries,
        evaluators=[local, cloud],   # list → both run, results returned per-provider
    )
    for r in results:
        print(f"{r.provider}: {r.passed}/{r.total}")

asyncio.run(main())
```

---

## 2 · `EvalItem` · `ConversationSplit`

**Package:** `agent_framework._evaluation`

`EvalItem` wraps a full `list[Message]` conversation into an evaluation unit. Two derived properties — `query` and `response` — extract the relevant text using a configurable `ConversationSplitter`.

### Constructor

```python
EvalItem(
    conversation: list[Message],
    tools: list[FunctionTool] | None = None,
    context: str | None = None,
    expected_output: str | None = None,
    expected_tool_calls: list[ExpectedToolCall] | None = None,
    split_strategy: ConversationSplitter | None = None,
)
```

| Parameter | Description |
|---|---|
| `conversation` | Full conversation as `Message` objects — the single source of truth. |
| `tools` | Typed tool objects for evaluator logic (e.g. tool-correctness evaluation). |
| `context` | Optional grounding document for retrieval/RAG quality evaluation. |
| `expected_output` | Ground-truth text for comparison-based evaluators. |
| `expected_tool_calls` | Expected tool names/args for tool-call correctness evaluation. |
| `split_strategy` | Controls how `query` / `response` are derived. Defaults to `ConversationSplit.LAST_TURN`. |

### Derived properties

| Property | Behaviour |
|---|---|
| `query` | Concatenation of `role == "user"` text from the query side of the split. |
| `response` | Concatenation of `role == "assistant"` text from the response side of the split. |

### `ConversationSplit` enum

```python
ConversationSplit.LAST_TURN    # split at the last user message (default)
ConversationSplit.FULL         # query = all messages; response = empty
ConversationSplit.FIRST_TURN   # split at the first user message
```

### `split_messages(split=None)` method

Returns `(query_messages, response_messages)` — resolution order: explicit `split` arg → `self.split_strategy` → `LAST_TURN`.

### `EvalItem.per_turn_items(conversation, *, tools, context)` — static factory

Splits a multi-turn conversation into one `EvalItem` per user turn. Each item has **cumulative** context: `query_messages` contains everything up to and including that user message; `response_messages` contains the agent's reply before the next user turn. Useful for evaluating every agent response in a dialogue independently.

### Example 1 — Single query/response pair

```python
from agent_framework._types import Message
from agent_framework._evaluation import EvalItem

messages = [
    Message(role="user",      content="What is 2+2?"),
    Message(role="assistant", content="The answer is 4."),
]
item = EvalItem(conversation=messages)
print(item.query)    # "What is 2+2?"
print(item.response) # "The answer is 4."
```

### Example 2 — Multi-turn conversation split per turn

```python
conversation = [
    Message(role="user",      content="Hello"),
    Message(role="assistant", content="Hi there!"),
    Message(role="user",      content="What's the weather?"),
    Message(role="assistant", content="It's sunny, 22°C."),
]
items = EvalItem.per_turn_items(conversation)
# Returns 2 EvalItems: one for "Hello", one for "What's the weather?"
for item in items:
    print(f"Q: {item.query!r}  A: {item.response!r}")
```

### Example 3 — Custom split strategy

```python
from agent_framework._evaluation import ConversationSplit

# Override per-item to use full conversation as query
item = EvalItem(conversation=messages, split_strategy=ConversationSplit.FULL)
query_msgs, response_msgs = item.split_messages()
```

---

## 3 · `EvalResults` · `EvalItemResult` · `EvalScoreResult` · `RubricScore`

**Package:** `agent_framework._evaluation`

The result model is a four-level hierarchy: `EvalResults` → `EvalItemResult` → `EvalScoreResult` → `RubricScore`.

### `EvalResults`

Returned by every `Evaluator.evaluate()` call.

```python
EvalResults(
    *,
    provider: str,
    eval_id: str = "",
    run_id: str = "",
    status: str = "completed",           # "completed" | "failed" | "canceled" | "timeout"
    result_counts: dict[str, int] | None = None,
    report_url: str | None = None,
    error: str | None = None,
    per_evaluator: dict[str, dict[str, int]] | None = None,
    items: list[EvalItemResult] | None = None,
    sub_results: dict[str, EvalResults] | None = None,
)
```

Key computed properties:

| Property | Returns |
|---|---|
| `passed` | `result_counts["passed"]` |
| `failed` | `result_counts["failed"]` |
| `total` | `passed + failed` |
| `all_passed` | `True` only when status is `"completed"`, `failed == 0`, `errored == 0`, and all `sub_results` also passed |

#### CI assertion helpers

```python
# Raises EvalNotPassedError on any failure or error
results.raise_for_status(msg="Custom message")

# Raises when any item's score is below the threshold
results.assert_score_at_least(0.80, evaluator="relevance")

# Raises when a rubric dimension score is below the threshold
results.assert_dimension_score_at_least("coherence", 0.75)

# Raises when any item has status "fail" or "error"
results.assert_no_failed_items()
```

`sub_results` holds per-agent breakdowns for workflow evaluations where multiple agents are evaluated separately. All assertion helpers recurse into `sub_results` automatically.

### `EvalItemResult`

Per-item result — one per `EvalItem` in the batch.

```python
@dataclass
class EvalItemResult:
    item_id: str
    status: str                           # "pass" | "fail" | "error"
    scores: list[EvalScoreResult]
    error_code: str | None = None
    error_message: str | None = None
    response_id: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
```

Boolean shorthands: `is_passed`, `is_failed`, `is_error`.

### `EvalScoreResult`

Per-evaluator score on one item.

```python
@dataclass
class EvalScoreResult:
    name: str                             # evaluator name, e.g. "relevance"
    score: float
    passed: bool | None = None
    sample: dict[str, Any] | None = None  # raw evaluator output / rationale
    dimensions: list[RubricScore] | None = None
```

`dimensions` is `None` for non-rubric evaluators (e.g. `LocalEvaluator`). For cloud rubric evaluators it contains one `RubricScore` per rubric dimension.

### `RubricScore`

```python
@dataclass(frozen=True)
class RubricScore:
    id: str         # dimension id (matches rubric definition)
    score: int | None
    applicable: bool
    weight: int
    reason: str
```

`score` is `None` when the dimension was marked non-applicable for a given item.

### Example — Full result traversal

```python
import asyncio

async def main():
    results = await evaluate_agent(agent=agent, queries=queries, evaluators=[local, cloud])

    for r in results:
        print(f"\n=== {r.provider}: {r.passed}/{r.total} passed ===")
        for item in r.items:
            print(f"  [{item.status}] {item.item_id}")
            if item.is_error:
                print(f"    ERROR: {item.error_code} — {item.error_message}")
            for score in item.scores:
                print(f"    {score.name}: {score.score:.2f}")
                if score.dimensions:
                    for dim in score.dimensions:
                        flag = "✓" if dim.applicable and dim.score and dim.score >= 3 else "✗"
                        print(f"      {flag} {dim.id}: {dim.score} (w={dim.weight}) — {dim.reason}")

    # Hard CI gate
    for r in results:
        r.assert_score_at_least(0.70)

asyncio.run(main())
```

---

## 4 · `ContextWindowCompactionStrategy` · `CharacterEstimatorTokenizer`

**Package:** `agent_framework._compaction` (import via `from agent_framework import ContextWindowCompactionStrategy, CompactionProvider`)

`ContextWindowCompactionStrategy` orchestrates a **two-phase compaction pipeline** sized to a specific model's context window. It composes two independent `TokenBudgetComposedStrategy` instances so each phase fires independently when its own threshold is exceeded.

### Constructor

```python
ContextWindowCompactionStrategy(
    *,
    max_context_window_tokens: int,
    max_output_tokens: int,
    tokenizer: TokenizerProtocol | None = None,
    tool_eviction_threshold: float = 0.5,
    truncation_threshold: float = 0.8,
    keep_last_tool_call_groups: int = 4,
)
```

| Parameter | Description |
|---|---|
| `max_context_window_tokens` | Model's maximum context window (e.g. `128_000`). |
| `max_output_tokens` | Model's maximum output tokens per response (e.g. `16_384`). |
| `tokenizer` | Token counter. Defaults to `CharacterEstimatorTokenizer` (4 chars/token). |
| `tool_eviction_threshold` | Fraction of input budget at which Phase 1 (tool eviction) triggers. Default `0.5`. |
| `truncation_threshold` | Fraction of input budget at which Phase 2 (truncation) triggers. Must be ≥ `tool_eviction_threshold`. Default `0.8`. |
| `keep_last_tool_call_groups` | Number of most-recent tool-call groups kept verbatim during Phase 1. Older groups are collapsed into summaries. Default `4`. |

The **input budget** is `max_context_window_tokens - max_output_tokens`. Phase 1 fires at 50% of input budget; Phase 2 fires at 80%.

`CharacterEstimatorTokenizer` is the zero-dependency fallback: `count_tokens(text) = max(1, len(text) // 4)`. Supply a real tokenizer (e.g. `tiktoken`) for accurate counts.

### Example 1 — Attach to an agent via `CompactionProvider`

```python
from agent_framework import Agent, CompactionProvider, ContextWindowCompactionStrategy

strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=16_384,
)
provider = CompactionProvider(before_strategy=strategy)

agent = Agent(
    client=client,
    instructions="You are a helpful assistant.",
    context_providers=[provider],
)
```

### Example 2 — Custom thresholds for a tool-heavy agent

```python
strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=200_000,
    max_output_tokens=8_192,
    tool_eviction_threshold=0.4,   # evict tool results earlier
    truncation_threshold=0.7,
    keep_last_tool_call_groups=2,  # keep only the 2 most recent tool groups verbatim
)
```

### Example 3 — Bring your own tokenizer

```python
import tiktoken
from agent_framework._compaction import TokenizerProtocol

class TiktokenWrapper:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=16_384,
    tokenizer=TiktokenWrapper(),
)
```

---

## 5 · `SummarizationStrategy`

**Package:** `agent_framework._compaction`

`SummarizationStrategy` monitors included non-system message count and, when that count exceeds `target_count + threshold`, calls out to an LLM to produce a summary that permanently replaces the oldest conversation groups.

> **Security note** (from source): This strategy calls an external LLM whose output becomes a trusted part of chat history. A compromised or malicious summarization service could inject adversarial instructions that survive indefinitely — a persistent indirect-prompt-injection vector. Only supply a `client` you trust as much as the primary model.

### Constructor

```python
SummarizationStrategy(
    *,
    client: SupportsChatGetResponse,
    target_count: int = 4,
    threshold: int | None = 2,
    prompt: str | None = None,
    max_summary_input_tokens: int | None = DEFAULT_SUMMARY_INPUT_TOKEN_BUDGET,
    tokenizer: TokenizerProtocol | None = None,
)
```

| Parameter | Description |
|---|---|
| `client` | Chat client used for summarization. Must implement `SupportsChatGetResponse`. |
| `target_count` | Target number of included non-system messages to retain after summarization. |
| `threshold` | Extra messages allowed above `target_count` before triggering. `None` = trigger every time. |
| `prompt` | Custom summarization instruction. Defaults to a built-in prompt that preserves goals, decisions, and unresolved items. |
| `max_summary_input_tokens` | Maximum estimated token count for the summarizer's input. Whole groups are added until the next group would exceed this budget. `None` = no limit. |
| `tokenizer` | Token counter. Defaults to `CharacterEstimatorTokenizer`. |

Trigger condition: `included_non_system_count > target_count + threshold`.

Trace metadata is written in both directions: summary → original message/group IDs; original → summary ID — enabling full provenance tracking.

### Example 1 — Basic summarization

```python
from agent_framework import CompactionProvider
from agent_framework._compaction import SummarizationStrategy

strategy = SummarizationStrategy(
    client=client,           # same client as the agent, or a cheaper model
    target_count=6,
    threshold=3,             # triggers when > 9 non-system messages are included
)
provider = CompactionProvider(before_strategy=strategy)
agent = Agent(client=client, context_providers=[provider])
```

### Example 2 — Cheap summarizer + custom prompt

```python
cheap_client = AzureAIChatClient(model="gpt-4o-mini")

strategy = SummarizationStrategy(
    client=cheap_client,
    target_count=8,
    threshold=4,
    prompt=(
        "Summarize the conversation below into concise bullet points. "
        "Preserve all decisions, tool results, and open questions."
    ),
    max_summary_input_tokens=4_000,
)
```

### Example 3 — Combine with truncation as a fallback

```python
from agent_framework._compaction import TokenBudgetComposedStrategy, TruncationStrategy, CharacterEstimatorTokenizer

tokenizer = CharacterEstimatorTokenizer()
composed = TokenBudgetComposedStrategy(
    token_budget=80_000,
    tokenizer=tokenizer,
    strategies=[
        SummarizationStrategy(client=cheap_client, target_count=8),
        TruncationStrategy(max_n=100_000, compact_to=60_000, tokenizer=tokenizer),
    ],
    early_stop=True,  # stop as soon as budget is satisfied
)
```

---

## 6 · `SlidingWindowStrategy` · `SelectiveToolCallCompactionStrategy` · `TruncationStrategy`

**Package:** `agent_framework._compaction`

Three lightweight compaction strategies that operate purely on message annotations — no LLM calls required.

### `SlidingWindowStrategy`

Keeps the **most recent** `keep_last_groups` included non-system groups and excludes everything older.

```python
SlidingWindowStrategy(
    *,
    keep_last_groups: int,
    preserve_system: bool = True,
)
```

- Raises `ValueError` if `keep_last_groups <= 0`.
- `preserve_system=True`: system groups are always retained, only non-system groups are eligible for exclusion.

```python
from agent_framework._compaction import SlidingWindowStrategy

strategy = SlidingWindowStrategy(keep_last_groups=10)
# Keeps the 10 most recent conversation groups; system prompts are always retained.
```

### `SelectiveToolCallCompactionStrategy`

Targets **only** groups annotated as `tool_call`. Keeps the last `keep_last_tool_call_groups` tool-call groups; older ones are excluded. Non-tool-call groups are never touched.

```python
SelectiveToolCallCompactionStrategy(
    *,
    keep_last_tool_call_groups: int = 1,
)
```

- `keep_last_tool_call_groups=0`: removes **all** tool-call groups.
- Raises `ValueError` if `keep_last_tool_call_groups < 0`.
- Ideal for tool-heavy agents where tool chatter dominates token usage but conversation history must be preserved.

```python
from agent_framework._compaction import SelectiveToolCallCompactionStrategy

strategy = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=3)
# Retains only the 3 most recent tool-call groups; everything older is excluded.
```

### `TruncationStrategy`

Oldest-first exclusion that fires when a metric threshold is exceeded. The metric is **token count** when a `tokenizer` is provided, otherwise **included message count**.

```python
TruncationStrategy(
    *,
    max_n: int,
    compact_to: int,
    tokenizer: TokenizerProtocol | None = None,
    preserve_system: bool = True,
)
```

| Parameter | Description |
|---|---|
| `max_n` | Trigger threshold (tokens or messages). |
| `compact_to` | Target value after compaction. Must be `<= max_n`. |
| `tokenizer` | When provided, switches to token-based truncation. |
| `preserve_system` | System groups are never excluded when `True`. |

```python
from agent_framework._compaction import TruncationStrategy

# Message-count based (no tokenizer)
strategy = TruncationStrategy(max_n=50, compact_to=30)

# Token-based
from agent_framework._compaction import CharacterEstimatorTokenizer
strategy = TruncationStrategy(
    max_n=100_000,
    compact_to=60_000,
    tokenizer=CharacterEstimatorTokenizer(),
)
```

### Composing all three with `TokenBudgetComposedStrategy`

```python
from agent_framework._compaction import (
    TokenBudgetComposedStrategy, CharacterEstimatorTokenizer,
    SelectiveToolCallCompactionStrategy, SlidingWindowStrategy, TruncationStrategy,
)

tokenizer = CharacterEstimatorTokenizer()
strategy = TokenBudgetComposedStrategy(
    token_budget=60_000,
    tokenizer=tokenizer,
    strategies=[
        SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2),
        SlidingWindowStrategy(keep_last_groups=15),
        TruncationStrategy(max_n=70_000, compact_to=55_000, tokenizer=tokenizer),
    ],
    early_stop=True,
)
```

---

## 7 · `TodoProvider` · `TodoItem` · `TodoInput` · `TodoCompleteInput` · `TodoSessionStore` · `TodoFileStore`

**Package:** `agent_framework._harness._todo` (import via `from agent_framework import TodoProvider`)

The todo system gives agents a persistent task list within a session. `TodoProvider` is a `ContextProvider` that injects five planning tools and manages the backing `TodoStore`.

### `TodoItem`

```python
TodoItem(id: int, title: str, description: str | None = None, is_complete: bool = False)
```

Fields: `id` (auto-assigned sequential int), `title`, `description`, `is_complete`. Implements `SerializationMixin` with `to_dict()` / `from_dict()`.

### `TodoInput` / `TodoCompleteInput`

Tool argument schemas:

```python
TodoInput(title: str, description: str | None = None)
# title is strip()-normalized; empty string raises ValueError.

TodoCompleteInput(id: int, reason: str)
# reason must be non-empty.
```

### `TodoProvider`

```python
TodoProvider(
    source_id: str = "todos",
    *,
    instructions: str | None = None,
    store: TodoStore | None = None,
)
```

Tools exposed to the agent:

| Tool | Action |
|---|---|
| `todos_add` | Add one or more `TodoInput` items |
| `todos_complete` | Mark items complete by ID + reason |
| `todos_remove` | Remove items by ID |
| `todos_get_remaining` | Retrieve incomplete items only |
| `todos_get_all` | Retrieve all items |

**Concurrency:** A `WeakKeyDictionary[AgentSession, asyncio.Lock]` guards all read-modify-write operations per session. `WeakKeyDictionary` ensures session-scoped locks are evicted automatically when the session is garbage-collected — safe for long-running services with many sessions.

### `TodoSessionStore`

Stores todo state inside `AgentSession.state` under the `source_id` key. Zero dependencies — no filesystem required. The default when `store` is omitted.

```python
TodoSessionStore()
# State layout: session.state[source_id] = {"items": [...], "next_id": N}
```

### `TodoFileStore`

Persists todo state as a JSON file per session.

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

When `owner_state_key` is set, the store reads `session.state[owner_state_key]` as the logical owner ID (e.g. `"user_id"`) and places each session's file under a per-owner subdirectory. Path segments are sanitized against path-traversal (rejects `..`) and Windows reserved file names (`CON`, `PRN`, `AUX`, `NUL`, `COMx`, `LPTx`).

`TodoFileStore` is marked `@experimental`.

### Example 1 — Session-backed todo (default)

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework import TodoProvider

provider = TodoProvider()
agent = Agent(client=client, instructions="Plan tasks before acting.", context_providers=[provider])

async def main():
    session = AgentSession()
    result = await agent.run("Research and write a report on solar energy.", session=session)
    print(result.text)

asyncio.run(main())
```

### Example 2 — File-backed todo with per-user isolation

```python
from agent_framework._harness._todo import TodoFileStore, TodoProvider
import pathlib

store = TodoFileStore(
    base_path=pathlib.Path("/data/todos"),
    owner_state_key="user_id",  # session.state["user_id"] determines the folder
)
provider = TodoProvider(store=store)
agent = Agent(client=client, context_providers=[provider])

async def handle_request(user_id: str, message: str):
    session = AgentSession()
    session.state["user_id"] = user_id
    return await agent.run(message, session=session)
```

### Example 3 — Custom instructions

```python
provider = TodoProvider(
    instructions=(
        "Before starting any task, create a todo list. "
        "Mark each item complete as you finish it. "
        "Do not proceed without a plan."
    )
)
```

---

## 8 · `BackgroundAgentsProvider` · `BackgroundTaskInfo` · `BackgroundTaskStatus`

**Package:** `agent_framework._harness._background_agents` (import via `from agent_framework import BackgroundAgentsProvider`)

`BackgroundAgentsProvider` enables a parent agent to delegate work to named sub-agents that run **concurrently in separate sessions**. Marked `@experimental`.

### `BackgroundTaskStatus`

```python
class BackgroundTaskStatus(str, Enum):
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    LOST      = "lost"    # task was started but the session was lost (e.g. process restart)
```

### `BackgroundTaskInfo`

```python
BackgroundTaskInfo(
    id: int,
    agent_name: str,
    description: str,
    status: BackgroundTaskStatus = BackgroundTaskStatus.RUNNING,
    result_text: str | None = None,
    error_text: str | None = None,
)
```

Implements `SerializationMixin`. Stored in `AgentSession.state` so task state survives across agent invocations within the same session.

### `BackgroundAgentsProvider`

```python
BackgroundAgentsProvider(
    agents: Sequence[SupportsAgentRun],
    *,
    source_id: str = "background_agents",
    instructions: str | None = None,
)
```

Exposed tools:

| Tool | Description |
|---|---|
| `background_agents_start_task` | Start a background task on a named agent with text input. Returns task ID. |
| `background_agents_wait_for_first_completion` | Block until the first of the specified task IDs completes. |
| `background_agents_get_task_results` | Retrieve the text output of a completed background task. |
| `background_agents_get_all_tasks` | List all tasks with their IDs, statuses, and descriptions. |
| `background_agents_continue_task` | Send follow-up input to a completed task's session to resume work. |
| `background_agents_clear_completed_task` | Remove a completed task and release its session. |

> **Security note** (from source): Supplied agents receive arbitrary text from the parent — which may include untrusted content from the parent's own context. A compromised agent could exfiltrate that input or return adversarial content designed to influence the parent via indirect prompt injection. Only supply background agents you have vetted and trust with the data the parent may pass to them.

### Example 1 — Parallel research tasks

```python
import asyncio
from agent_framework import Agent, AgentSession, BackgroundAgentsProvider

research_agent = Agent(client=client, name="researcher", instructions="Research the given topic thoroughly.")
writer_agent   = Agent(client=client, name="writer",     instructions="Write a clear summary of the given notes.")

provider = BackgroundAgentsProvider(agents=[research_agent, writer_agent])
orchestrator = Agent(
    client=client,
    instructions=(
        "Use background agents to research topics in parallel, "
        "then wait for results and delegate writing."
    ),
    context_providers=[provider],
)

async def main():
    session = AgentSession()
    result = await orchestrator.run(
        "Research solar energy and wind energy in parallel, then write a comparison.",
        session=session,
    )
    print(result.text)

asyncio.run(main())
```

### Example 2 — Fan-out + wait for first

```python
# The orchestrator agent calls these tools automatically.
# Illustrative tool-call sequence the model might use:

# 1. background_agents_start_task(agent_name="researcher", description="Solar energy", input="Research solar energy")
#    → {"task_id": 1}
# 2. background_agents_start_task(agent_name="researcher", description="Wind energy", input="Research wind energy")
#    → {"task_id": 2}
# 3. background_agents_wait_for_first_completion(task_ids=[1, 2])
#    → {"completed_task_id": 1, "status": "completed"}
# 4. background_agents_get_task_results(task_id=1)
#    → {"result": "Solar energy facts..."}
# 5. background_agents_clear_completed_task(task_id=1)
```

---

## 9 · `ToolApprovalMiddleware` · `ToolApprovalRule` · `ToolApprovalState`

**Package:** `agent_framework._harness._tool_approval`

`ToolApprovalMiddleware` is an `AgentMiddleware` that intercepts tool invocations and routes them through a human-in-the-loop approval flow — or auto-approves them when a matching `ToolApprovalRule` exists.

### `ToolApprovalRule`

```python
ToolApprovalRule(
    tool_name: str,
    arguments: Mapping[str, str] | None = None,
    *,
    server_label: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `tool_name` | The function tool name this rule covers. `strip()`-normalized; empty raises `ValueError`. |
| `arguments` | Optional argument-value map. `None` = match any arguments. `{}` = match only no-argument calls. |
| `server_label` | Optional hosted-tool server boundary (for MCP tools). Rules only match calls from the same server. |

### `ToolApprovalState`

Session-backed state managed by the middleware.

```python
ToolApprovalState(
    *,
    rules: Sequence[ToolApprovalRule | Mapping] | None = None,
    queued_approval_requests: Sequence[Content | Mapping] | None = None,
    collected_approval_responses: Sequence[Content | Mapping] | None = None,
)
```

| Attribute | Description |
|---|---|
| `rules` | Standing approval rules applied to future matching tool calls. |
| `queued_approval_requests` | Pending `function_approval_request` content items awaiting human response. |
| `collected_approval_responses` | `function_approval_response` items collected from the host. |

### `ToolApprovalMiddleware`

```python
ToolApprovalMiddleware(
    *,
    source_id: str = "tool_approval",
    auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None,
)
```

`auto_approval_rules` are callbacks `(Content) → bool` that can auto-approve a `function_call` without a human prompt. Each callback receives the full `function_call` content including arguments.

> **Security warning** (from source): An auto-approval callback approved for one feature may auto-approve **any** local tool with a matching name — not just the tool the rule was designed for. Ensure no unrelated tools collide with names auto-approved by any rule in this list.

Requires `AgentSession` — raises `RuntimeError` if the session is absent.

### Example 1 — Basic HITL approval

```python
from agent_framework import Agent, AgentSession
from agent_framework._harness._tool_approval import ToolApprovalMiddleware

middleware = ToolApprovalMiddleware()
agent = Agent(
    client=client,
    tools=[send_email_tool, read_file_tool],
    middleware=[middleware],
)

async def main():
    session = AgentSession()
    # First invocation — tool calls are queued as approval requests
    result = await agent.run("Send an email to alice@example.com", session=session)
    # result contains function_approval_request content items
    # Host approves / denies, then calls agent.run again with responses
    print(result.text)

asyncio.run(main())
```

### Example 2 — Standing approval rule (argument-aware)

```python
from agent_framework._harness._tool_approval import ToolApprovalRule, ToolApprovalState

# Pre-approve "read_file" calls only when path == "/safe/path"
rule = ToolApprovalRule(
    tool_name="read_file",
    arguments={"path": "/safe/path"},
)

state = ToolApprovalState(rules=[rule])
# Store in session state before the run:
session = AgentSession()
session.state["tool_approval"] = state.to_dict()
```

### Example 3 — Auto-approval callback

```python
from agent_framework._harness._tool_approval import ToolApprovalMiddleware

def always_approve_read_tools(call_content) -> bool:
    """Auto-approve any function_call whose name starts with 'read_'."""
    name = call_content.function_call.name
    return name.startswith("read_")

middleware = ToolApprovalMiddleware(auto_approval_rules=[always_approve_read_tools])
agent = Agent(client=client, tools=tools, middleware=[middleware])
```

---

## 10 · `FunctionalWorkflow` · `FunctionalWorkflowAgent` · `RunContext` · `StepWrapper` · `WorkflowInterrupted`

**Package:** `agent_framework._workflows._functional` (import via `from agent_framework import workflow, step`)

The functional workflow API lets you author orchestration logic using **plain Python async functions** decorated with `@workflow` and `@step` — no graph wiring or edge definitions required. All classes are marked `@experimental`.

### `@workflow` → `FunctionalWorkflow`

```python
FunctionalWorkflow(
    func: Callable[..., Awaitable[Any]],
    *,
    name: str | None = None,
    description: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
)
```

The `@workflow` decorator wraps an `async def` function into a `FunctionalWorkflow` that exposes a `run()` interface compatible with graph-based `Workflow` objects.

```python
import asyncio
from agent_framework import workflow, step

@step
async def to_upper(text: str) -> str:
    return text.upper()

@workflow
async def my_pipeline(data: str) -> str:
    return await to_upper(data)

async def main():
    result = await my_pipeline.run("hello")
    print(result.get_outputs())  # ["HELLO"]

asyncio.run(main())
```

### `@step` → `StepWrapper`

```python
StepWrapper(func: Callable[..., Awaitable[R]], *, name: str | None = None)
```

Raises `TypeError` if `func` is not an async function.

When called **inside a running `@workflow` function**, `StepWrapper`:

1. **Caches by `(step_name, call_index)`** — HITL replay and checkpoint restore skip already-completed work. Cache hits emit a single `executor_bypassed` event.
2. **Emits observability events** — `executor_invoked` / `executor_completed` / `executor_failed`.
3. **Injects `RunContext` automatically** — if the step function declares a parameter annotated `: RunContext` or named `ctx`.
4. **Saves a checkpoint after each live execution** — when `checkpoint_storage` is configured.

**Outside a workflow**, `StepWrapper` is transparent: it delegates directly to the original function, making `@step` functions fully testable in isolation.

### `RunContext`

Injected automatically when a `@workflow` or `@step` function declares `ctx: RunContext` or a parameter named `ctx`.

```python
RunContext(
    workflow_name: str,
    streaming: bool,
    run_kwargs: dict,
)
```

Key methods:

| Method | Description |
|---|---|
| `await ctx.request_info(request_data, *, response_type)` | Pause the workflow and request a human response. Raises `WorkflowInterrupted`. |
| `await ctx.add_event(WorkflowEvent(type=..., data=...))` | Emit a custom event into the run stream. |
| `ctx.get_state(key, default=None)` | Read workflow-scoped key/value state (survives checkpoints). |
| `ctx.set_state(key, value)` | Write workflow-scoped state. |

### `WorkflowInterrupted`

```python
class WorkflowInterrupted(BaseException):
    def __init__(self, request_id: str, request_data: Any, response_type: type) -> None: ...
```

Inherits from `BaseException` (not `Exception`) so `except Exception:` blocks in user code **cannot accidentally catch it**. Raised internally by `RunContext.request_info()` during initial execution to signal the HITL pause point.

### `FunctionalWorkflowAgent`

Adapts a `FunctionalWorkflow` to the same `run()` interface as `BaseAgent`, so functional workflows slot in anywhere an agent-compatible object is expected.

```python
FunctionalWorkflowAgent(
    workflow: FunctionalWorkflow,
    *,
    name: str | None = None,
    description: str | None = None,
    context_providers: Sequence[Any] | None = None,
)
```

`request_info` events from the workflow are surfaced as `FunctionApprovalRequestContent` items — the same format as graph `WorkflowAgent` HITL interrupts. Callers resume via `responses=` / `checkpoint_id=`.

### Example 1 — Simple pipeline

```python
import asyncio
from agent_framework import workflow, step, Agent

client = AzureAIChatClient(model="gpt-4o")
agent = Agent(client=client, instructions="You summarize text.")

@step
async def fetch_data(url: str) -> str:
    import httpx
    async with httpx.AsyncClient() as http:
        resp = await http.get(url)
        return resp.text

@step
async def summarize(text: str) -> str:
    result = await agent.run(f"Summarize this:\n\n{text}")
    return result.text

@workflow
async def research_pipeline(url: str) -> str:
    raw = await fetch_data(url)
    return await summarize(raw)

async def main():
    result = await research_pipeline.run("https://example.com/article")
    outputs = result.get_outputs()
    print(outputs[0])

asyncio.run(main())
```

### Example 2 — HITL with `RunContext.request_info`

```python
import asyncio
from agent_framework import workflow, step, RunContext
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage

storage = InMemoryCheckpointStorage()

@step
async def draft_proposal(requirements: str) -> str:
    result = await agent.run(f"Draft a proposal for: {requirements}")
    return result.text

@workflow(checkpoint_storage=storage)
async def approval_workflow(requirements: str, ctx: RunContext) -> str:
    draft = await draft_proposal(requirements)
    # Pauses here — raises WorkflowInterrupted internally
    approval = await ctx.request_info(
        {"draft": draft, "instructions": "Approve or revise."},
        request_id="approval",   # explicit ID so the resume key matches
        response_type=str,
    )
    if approval.lower().startswith("approve"):
        return draft
    return await draft_proposal(f"{requirements}\n\nFeedback: {approval}")

async def main():
    # First run — workflow pauses at request_info and auto-saves a checkpoint
    result = await approval_workflow.run("Build a new feature")

    # Retrieve the checkpoint saved by the interrupted run
    cp = await storage.get_latest(workflow_name="approval_workflow")

    # Resume: message is mutually exclusive with checkpoint_id, so omit it
    result = await approval_workflow.run(
        checkpoint_id=cp.checkpoint_id,
        responses={"approval": "Approved!"},  # keyed by the request_id set above
    )
    print(result.get_outputs())

asyncio.run(main())
```

### Example 3 — Parallel steps with `asyncio.gather`

```python
@workflow
async def parallel_research(topics: list[str]) -> list[str]:
    results = await asyncio.gather(*[research_topic(t) for t in topics])
    return list(results)

@step
async def research_topic(topic: str) -> str:
    result = await agent.run(f"Research: {topic}")
    return result.text
```

### Example 4 — Expose as an agent with `FunctionalWorkflowAgent`

```python
from agent_framework._workflows._functional import FunctionalWorkflowAgent

workflow_agent = FunctionalWorkflowAgent(
    workflow=research_pipeline,
    name="ResearchAgent",
    description="Fetches and summarizes a URL.",
)

# Now usable with BackgroundAgentsProvider or orchestrators
provider = BackgroundAgentsProvider(agents=[workflow_agent])
```

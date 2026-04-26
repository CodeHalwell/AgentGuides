---
title: "Microsoft Agent Framework (Python) — Compaction"
description: "Keep long agent conversations inside the context window: TruncationStrategy, SlidingWindowStrategy, SummarizationStrategy, ToolResultCompactionStrategy, TokenBudgetComposedStrategy, and CompactionProvider."
framework: microsoft-agent-framework
language: python
---

# Compaction — Python

Long-running agents blow through the context window. Compaction strategies decide which messages to keep, which to drop, and which to replace with shorter summaries — **per turn**, before the messages reach the model.

Six first-class strategies ship in `agent_framework`, plus a `CompactionProvider` that plugs any strategy into the session pipeline. Verified against `agent-framework-core==1.1.0` (`agent_framework._compaction`).

## Mental model

Compaction sees your conversation as an ordered list of **message groups** (user turn, assistant turn, tool-call group, system prompt). Each group is atomic — a tool call and its result stay together. A strategy marks groups `excluded=True` (or replaces them with a summary); the framework projects the included groups and ships them to the model. Your source history is preserved — only the model's view shrinks.

## Strategies at a glance

| Strategy | What it does | Use when |
|---|---|---|
| `TruncationStrategy` | Keep the first N and last M messages; exclude the middle | Simple FIFO that preserves system + recent |
| `SlidingWindowStrategy` | Keep the last `keep_last_groups` non-system groups | Predictable recency window |
| `SelectiveToolCallCompactionStrategy` | Drop older tool-call groups only | Tool chatter dominates the history |
| `ToolResultCompactionStrategy` | Collapse older tool calls into one-line summaries | Keep a readable trace but cut tokens |
| `SummarizationStrategy` | LLM-summarise older groups into a single message | Long chats where earlier context still matters |
| `TokenBudgetComposedStrategy` | Run other strategies until a token budget is met | Hard limit per turn; compose lighter strategies first |

All implement `CompactionStrategy`:

```python
class CompactionStrategy(Protocol):
    async def __call__(self, messages: list[Message]) -> bool: ...
```

Returns `True` if the message list was modified.

## Truncation — single threshold, hard cap

`TruncationStrategy` is the workhorse strategy when you want one knob ("never exceed N") with deterministic behaviour. It runs after groups are annotated, then drops whole non-system groups oldest-first until the metric is back under `compact_to`.

```python
from agent_framework import TruncationStrategy

# Message-count mode (default — no tokenizer)
msg_strategy = TruncationStrategy(
    max_n=40,         # trigger when included messages > 40
    compact_to=24,    # trim down to 24
    preserve_system=True,
)

# Token-count mode — pass a tokenizer and the same numbers become token thresholds
from agent_framework import CharacterEstimatorTokenizer

token_strategy = TruncationStrategy(
    max_n=8_000,
    compact_to=5_000,
    tokenizer=CharacterEstimatorTokenizer(),
)
```

A few things from the source worth knowing:

- The metric is **either tokens or messages**, never both — when `tokenizer=` is set, both `max_n` and `compact_to` are token counts.
- `compact_to` is **required** and must be `> 0` and `<= max_n`. The constructor raises `ValueError` otherwise.
- Tool-call groups stay atomic: the strategy never excludes a function-call message without its function-result message, even if that means crossing `compact_to` slightly.
- `preserve_system=True` (the default) excludes only non-system groups, so a multi-paragraph system prompt is never truncated.

If you need to truncate while keeping the *most-recent* tool-call results readable but dropping their full payloads, layer `ToolResultCompactionStrategy` *before* truncation in a `TokenBudgetComposedStrategy`.

## Sliding window — simplest

```python
from agent_framework import SlidingWindowStrategy, apply_compaction

strategy = SlidingWindowStrategy(
    keep_last_groups=20,    # keep the last 20 non-system groups
    preserve_system=True,   # never drop system messages
)

projected = await apply_compaction(messages, strategy=strategy)
```

Good default. Deterministic, no extra LLM calls, preserves system anchors.

## Selective tool-call compaction

```python
from agent_framework import SelectiveToolCallCompactionStrategy

strategy = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=3)
```

Only touches tool-call groups. Use when an agent polls APIs dozens of times — user/assistant turns keep full fidelity; older tool chatter is dropped.

## Tool-result compaction — readable summaries

```python
from agent_framework import ToolResultCompactionStrategy

strategy = ToolResultCompactionStrategy(keep_last_tool_call_groups=1)
```

Replaces older tool-call groups with a short summary like `[Tool results: get_weather: sunny, 18°C; get_forecast: rain Tue]` instead of excluding them. You keep the provenance at a fraction of the tokens.

## LLM summarisation

Uses a chat client to summarise older history into a single assistant message:

```python
from agent_framework import SummarizationStrategy
from agent_framework.openai import OpenAIChatClient

summariser = SummarizationStrategy(
    client=OpenAIChatClient(model="gpt-4o-mini"),
    target_count=8,     # leave 8 non-system messages untouched
    threshold=4,        # trigger when included > target + threshold
)
```

Triggers only when `included_non_system_count > target_count + threshold` — idle conversations don't pay for summaries they don't need. Once triggered, the strategy walks groups oldest-first, keeps the newest ones up to `target_count`, and replaces the rest with one summary message that links back to the original group IDs via annotations.

### Domain-specific summarisation prompt

The default prompt preserves goals, decisions, and unresolved items. Override with `prompt=` when you want a different shape:

```python
support_summariser = SummarizationStrategy(
    client=OpenAIChatClient(model="gpt-4o-mini"),
    target_count=6,
    threshold=3,
    prompt=(
        "You are summarising a customer support conversation. "
        "Produce at most 4 bullet points covering:\n"
        "- The customer's problem and any account identifiers mentioned.\n"
        "- Diagnostic steps already attempted.\n"
        "- Agreements or promises made by the support agent.\n"
        "- Any open questions or pending escalations.\n"
        "Do NOT restate pleasantries or repeat exact quotes."
    ),
)
```

The summary message keeps `group_annotation.summary_of_group_ids` and `summary_of_message_ids` metadata so you can later walk back to the original turns — useful for audit trails or "expand this summary" UI affordances.

### Graceful fallback

If the summariser client fails (timeout, rate limit, parse error), `SummarizationStrategy` logs a warning and returns `False` *without* mutating the message list. Composed strategies (`TokenBudgetComposedStrategy`) then fall through to the next strategy. Pair with a cheap sliding-window strategy so you degrade predictably when summarisation is unavailable.

## Token-budget composed strategy

Compose cheaper strategies first; fall back to hard exclusion to meet a strict cap:

```python
from agent_framework import (
    CharacterEstimatorTokenizer,
    SelectiveToolCallCompactionStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
)
from agent_framework.openai import OpenAIChatClient

summariser_client = OpenAIChatClient(model="gpt-4o-mini")

strategy = TokenBudgetComposedStrategy(
    token_budget=8_000,
    tokenizer=CharacterEstimatorTokenizer(),   # swap for a real tokenizer in prod
    strategies=[
        SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2),
        SlidingWindowStrategy(keep_last_groups=30),
        SummarizationStrategy(client=summariser_client, target_count=10),
    ],
    early_stop=True,   # stop as soon as budget is satisfied
)
```

Order matters — put cheap, deterministic strategies first; summarisation last because it spends tokens. If the composed strategies still exceed budget, the built-in fallback excludes oldest non-system groups (and finally system anchors) to enforce the cap.

### Plugging in a real tokenizer

```python
import tiktoken
from agent_framework import TokenizerProtocol


class TiktokenTokenizer:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._enc = tiktoken.encoding_for_model(model)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


strategy = TokenBudgetComposedStrategy(
    token_budget=8_000,
    tokenizer=TiktokenTokenizer(),
    strategies=[...],
)
```

`CharacterEstimatorTokenizer` ships with the framework for when you just need a rough character→token heuristic (~4 chars/token). Use the real tokenizer in production.

## Wiring compaction into an agent

Two options: **per-agent** (via the chat client) or **per-session** (via `CompactionProvider`). Per-session is more common — it integrates with history providers and persists the compacted state.

### Per-session via `CompactionProvider`

```python
from agent_framework import (
    Agent,
    CompactionProvider,
    InMemoryHistoryProvider,
    SlidingWindowStrategy,
    ToolResultCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient


history = InMemoryHistoryProvider()

compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
    history_source_id=history.source_id,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    context_providers=[history, compaction],
)

session = agent.create_session()
await agent.run("Kick off research on X.", session=session)
await agent.run("Now write the summary.", session=session)   # session history compacted between turns
```

`before_strategy` runs when messages are loaded into the run; `after_strategy` compacts what's persisted back into session state so the *next* turn starts smaller. Either can be `None` to skip that phase.

### Swapping in `FileHistoryProvider`

For durable sessions across process restarts, replace `InMemoryHistoryProvider` with `FileHistoryProvider` (experimental — `agent_framework._sessions`). It writes one JSON-Lines file per `session_id` under a root directory:

```python
from agent_framework import Agent, CompactionProvider, FileHistoryProvider, SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient


history = FileHistoryProvider(
    storage_path="./sessions",
    skip_excluded=True,             # don't reload compacted-out messages on the next turn
)

compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    history_source_id=history.source_id,
)

agent = Agent(client=OpenAIChatClient(), context_providers=[history, compaction])

# A distinct session_id picks up the corresponding file on the next call.
session = agent.create_session(session_id="user-42")
await agent.run("Continue where we left off.", session=session)
```

`FileHistoryProvider` resolves `session_id` against `storage_path` and **rejects any id that would escape the storage directory** — `../` traversal is blocked and resolved paths are validated against the storage root. Treat the storage directory as trusted application state, not a secret store; the JSONL contents are plaintext. For multi-process deployments, use the Redis or Cosmos history providers instead so concurrent writers don't race on the same file.

The `store_inputs`, `store_outputs`, `store_context_messages`, and `load_messages` flags let the same class act as an audit log (`load_messages=False`), a write-only evaluation trace, or a full primary store:

```python
audit_log = FileHistoryProvider(
    storage_path="./audit",
    source_id="audit",
    load_messages=False,            # never reload — purely a write destination
    store_inputs=True,
    store_outputs=True,
    store_context_messages=True,    # also capture messages injected by other providers
)
```

### Per-agent via chat client

Any chat client accepts `compaction_strategy=`. This applies to every call made through that client, regardless of session:

```python
client = OpenAIChatClient(
    compaction_strategy=SlidingWindowStrategy(keep_last_groups=30),
    tokenizer=TiktokenTokenizer(),
)
agent = Agent(client=client, instructions="…")
```

Use this for stateless agents or when you want a client-wide safety net independent of session compaction.

### Per-run override

Pass `compaction_strategy=` and `tokenizer=` to `agent.run(...)` for one-off overrides — handy in tests.

```python
await agent.run("…", compaction_strategy=SlidingWindowStrategy(keep_last_groups=5))
```

## Custom strategies

Anything callable that matches the `CompactionStrategy` protocol works — `__call__(messages) -> bool`. Two flavours: a plain async function for stateless logic, or a class when you need configuration.

### Stateless — drop tool errors from old turns

```python
from agent_framework import Message
from agent_framework._compaction import set_excluded


async def drop_old_errors(messages: list[Message]) -> bool:
    changed = False
    for m in messages:
        if m.role == "tool" and "error" in (m.text or "").lower():
            changed = set_excluded(m, excluded=True, reason="old_error") or changed
    return changed
```

`set_excluded(message, excluded=True, reason=...)` is the canonical way to mark a message excluded — it returns `True` when the inclusion state actually changed, which is the bool your strategy needs to return.

### Class form — keep at most N user turns

A real-world strategy usually needs config. The class form gives you a constructor:

```python
from agent_framework import Message
from agent_framework._compaction import (
    _ordered_group_ids_from_annotations,
    _group_messages_by_id,
    _group_kind_map,
    set_excluded,
)


class MaxUserTurnsStrategy:
    """Keep only the most recent N user-message groups (their assistant replies ride along)."""

    def __init__(self, *, max_user_turns: int) -> None:
        if max_user_turns <= 0:
            raise ValueError("max_user_turns must be > 0")
        self.max_user_turns = max_user_turns

    async def __call__(self, messages: list[Message]) -> bool:
        ordered_ids = _ordered_group_ids_from_annotations(messages)
        kinds = _group_kind_map(messages)
        grouped = _group_messages_by_id(messages)

        user_group_ids = [gid for gid in ordered_ids if kinds.get(gid) == "user"]
        if len(user_group_ids) <= self.max_user_turns:
            return False

        keep = set(user_group_ids[-self.max_user_turns:])
        changed = False
        for gid in user_group_ids:
            if gid in keep:
                continue
            for m in grouped.get(gid, []):
                changed = set_excluded(m, excluded=True, reason="max_user_turns") or changed
        return changed
```

The internals (`_ordered_group_ids_from_annotations`, `_group_messages_by_id`, `_group_kind_map`) are private but deliberately stable — every built-in strategy uses them. Importing them from `agent_framework._compaction` is the supported way to write strategies that respect the framework's atomic group boundaries.

### Manually walking the annotations

If you want to inspect what's been annotated (e.g. in a unit test), `annotate_message_groups` is the public entry point:

```python
from agent_framework import Message
from agent_framework._compaction import annotate_message_groups, group_messages

messages = [
    Message(role="system", contents=["You are helpful."]),
    Message(role="user", contents=["Where is Paris?"]),
    Message(role="assistant", contents=["France."]),
]

annotate_message_groups(messages)
for m in messages:
    print(m.role, m.additional_properties["_group"])
# system    {'id': 'group_msg_0', 'kind': 'system', 'index': 0, ...}
# user      {'id': 'group_msg_1', 'kind': 'user', 'index': 1, ...}
# assistant {'id': 'group_msg_2', 'kind': 'assistant_text', 'index': 2, ...}

# Just the spans, without mutating messages:
spans = group_messages(messages)
print(spans[0])  # {'group_id': 'group_msg_0', 'kind': 'system', 'start_index': 0, 'end_index': 0, 'has_reasoning': False}
```

Calling `annotate_message_groups` repeatedly is safe — it re-annotates only the un-annotated tail by default. Pass `force_reannotate=True` to re-do the whole list (e.g. after a structural change).

Compose any custom strategy with the built-ins via `TokenBudgetComposedStrategy` or call inline via `apply_compaction(messages, strategy=mine)`.

## Inspecting what compaction did

Framework helpers let you see the before/after split:

```python
from agent_framework import included_messages, included_token_count

print(len(included_messages(messages)))  # how many survived
print(included_token_count(messages))    # estimated tokens kept
```

Excluded messages stay in the list tagged with `EXCLUDED_KEY=True` and `EXCLUDE_REASON_KEY` — useful for debugging and UI display ("32 messages compacted").

## Patterns

**Default for chat UIs.** `SlidingWindowStrategy(keep_last_groups=20)` as `before_strategy`, `ToolResultCompactionStrategy(keep_last_tool_call_groups=1)` as `after_strategy`. Predictable, no judge tokens.

**Research agents with long plans.** Use `SummarizationStrategy` with a cheap model — Phi-3 / Haiku / gpt-4o-mini. The summary retains goals and open questions so the agent stays on-track past 50 turns.

**Strict SLA on context size.** `TokenBudgetComposedStrategy(token_budget=32_000, ...)` guarantees you never send >32k tokens, with deterministic fallback even if summarisation fails.

**Multi-tool pipelines.** Tool results often dwarf user/assistant turns. `SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)` preserves the important reasoning while dropping the polling noise.

**A/B test strategies.** Run `evaluate_agent` (see [Evaluation](./microsoft_agent_framework_python_evaluation/)) twice with different `compaction_strategy=` overrides and compare pass rates.

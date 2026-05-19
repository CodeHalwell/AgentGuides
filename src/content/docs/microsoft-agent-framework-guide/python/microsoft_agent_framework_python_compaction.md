---
title: "Microsoft Agent Framework (Python) — Compaction"
description: "Keep long agent conversations inside the context window: TruncationStrategy, SlidingWindowStrategy, SummarizationStrategy, ToolResultCompactionStrategy, TokenBudgetComposedStrategy, and CompactionProvider."
framework: microsoft-agent-framework
language: python
---

# Compaction — Python

Long-running agents blow through the context window. Compaction strategies decide which messages to keep, which to drop, and which to replace with shorter summaries — **per turn**, before the messages reach the model.

Six first-class strategies ship in `agent_framework`, plus a `CompactionProvider` that plugs any strategy into the session pipeline. Verified against `agent-framework-core==1.4.0` (`agent_framework._compaction`).

## Mental model

Compaction sees your conversation as an ordered list of **message groups** (user turn, assistant turn, tool-call group, system prompt). Each group is atomic — a tool call and its result stay together. A strategy marks groups `excluded=True` (or replaces them with a summary); the framework projects the included groups and ships them to the model. Your source history is preserved — only the model's view shrinks.

## Strategies at a glance

| Strategy | What it does | Use when |
|---|---|---|
| `TruncationStrategy` | Oldest-first exclusion — drop oldest groups until count (or tokens) falls to `compact_to`; triggers when `max_n` is exceeded | Simple FIFO drop; no extra LLM calls |
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

### What "groups" means in practice

A *group* is whatever the framework's group annotator considered a single logical unit — typically a user turn, an assistant turn, or a tool-call group (a function call **plus** its tool-result message stay together). Counting groups instead of messages is what lets the strategy keep tool calls atomic: it never strands a `function_call` without its `function_result`.

```python
import asyncio
from agent_framework import (
    Content,
    EXCLUDED_KEY,
    EXCLUDE_REASON_KEY,
    Message,
    SlidingWindowStrategy,
    apply_compaction,
)

# A short conversation with system anchor + 4 user/assistant pairs + a tool-call group.
messages = [
    Message(role="system", contents=[Content.from_text("Be concise.")]),
    *[
        Message(role=role, contents=[Content.from_text(f"{role} {i}")])
        for i in range(4)
        for role in ("user", "assistant")
    ],
    Message(role="assistant", contents=[Content.from_function_call(
        call_id="c1", name="lookup", arguments={"id": 42},
    )]),
    Message(role="tool", contents=[Content.from_function_result(
        call_id="c1", result={"status": "ok"},
    )]),
]

strategy = SlidingWindowStrategy(keep_last_groups=2, preserve_system=True)
asyncio.run(apply_compaction(messages, strategy=strategy))

excluded = [m for m in messages if m.additional_properties.get(EXCLUDED_KEY)]
print(len(excluded), "messages excluded with reason",
      excluded[0].additional_properties.get(EXCLUDE_REASON_KEY))
# 6 messages excluded with reason sliding_window
```

Two things this snippet illustrates:

- **System messages survive `preserve_system=True`.** Even with `keep_last_groups=2`, the system anchor is untouched and isn't counted toward the kept-groups quota.
- **The tool-call group counts as one group.** Both the `function_call` and the `function_result` survive together — `keep_last_groups=2` keeps the latest user/assistant turn and the tool group, not 2 messages out of 3.

### Edge cases worth knowing

- `keep_last_groups <= 0` raises `ValueError` at construction — you cannot configure a window of size zero.
- `preserve_system=False` lets the strategy drop the system anchor too. Use this when system instructions are duplicated by an `instructions=` parameter on the agent (the framework injects them again on each call) and you don't want to pay for both copies.
- The strategy mutates messages in place via `additional_properties[EXCLUDED_KEY] = True`. Call `included_messages(messages)` to get the projected view, or filter on `EXCLUDED_KEY` directly when you want to display "32 messages collapsed" in your UI.

## Truncation — oldest-first drop

`TruncationStrategy` measures the conversation by **included message count** (default) or **token count** (when a `tokenizer` is supplied). Once the metric exceeds `max_n`, it walks groups oldest-first and marks them excluded until the count falls to `compact_to`. System messages are skipped when `preserve_system=True` (the default).

```python
from agent_framework import TruncationStrategy

# Count-based: trigger when >40 included messages; compact down to 30.
strategy = TruncationStrategy(max_n=40, compact_to=30)
```

A `ValueError` is raised at construction if `compact_to > max_n` or either value is `<= 0`.

### Token-based truncation

Pass any `TokenizerProtocol` to switch the metric from message count to token count:

```python
from agent_framework import CharacterEstimatorTokenizer, TruncationStrategy

# Token-based: trigger when >8000 tokens; compact down to 6000.
strategy = TruncationStrategy(
    max_n=8_000,
    compact_to=6_000,
    tokenizer=CharacterEstimatorTokenizer(),  # swap for tiktoken in production
    preserve_system=True,
)
```

### Observing the effect

```python
import asyncio
from agent_framework import (
    Content,
    Message,
    TruncationStrategy,
    apply_compaction,
    included_messages,
)

messages = [
    Message(role="system", contents=[Content.from_text("You are helpful.")]),
    *[
        Message(role=role, contents=[Content.from_text(f"{role} turn {i}")])
        for i in range(8)
        for role in ("user", "assistant")
    ],
]
# 17 messages total (1 system + 16 user/assistant).

strategy = TruncationStrategy(max_n=10, compact_to=6, preserve_system=True)
asyncio.run(apply_compaction(messages, strategy=strategy))

kept = included_messages(messages)
print(len(kept))          # 6 — 1 system + 5 most-recent user/assistant
print(kept[0].role)       # system  (preserved despite oldest-first pass)
print(kept[-1].role)      # assistant
```

The metric (`max_n` and `compact_to`) counts **all** included messages — system messages included. With `compact_to=6` and one protected system message, the strategy excludes oldest non-system groups until total included count ≤ 6, leaving 1 system + 5 non-system messages. System messages cannot be excluded when `preserve_system=True`, but they still count toward the target.

### When to prefer `TruncationStrategy` over `SlidingWindowStrategy`

Both perform oldest-first exclusion.  The difference is how they trigger:

- **`SlidingWindowStrategy`** always enforces a fixed window size, every call.
- **`TruncationStrategy`** is lazy — it only fires when a threshold (`max_n`) is breached, and it stops trimming once `compact_to` is reached. This produces a *sawtooth* pattern: conversation grows from `compact_to` to `max_n` before being compacted, meaning most turns are cheap no-ops.

Use `TruncationStrategy` when you want compaction to feel like "only run when we're actually getting full", and `SlidingWindowStrategy` when you want strict, predictable window sizes on every turn.

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

`keep_last_tool_call_groups=0` collapses **every** included tool-call group; the most recent ones are summarised too. `keep_last_tool_call_groups=N` (the default is `1`) leaves the newest `N` groups untouched.

### What the summary actually looks like

Given a transcript with three tool-call groups for `get_weather` / `get_forecast`, the strategy mutates the message list **in place** and inserts a synthesised summary message at the start of each compacted group while marking the original messages excluded:

```python
import asyncio
from agent_framework import (
    Content,
    Message,
    ToolResultCompactionStrategy,
    apply_compaction,
)

messages = [
    Message(role="user", contents=[Content.from_text("Weather in Seattle?")]),
    Message(role="assistant", contents=[
        Content.from_function_call("c1", "get_weather", arguments={"city": "Seattle"}),
    ]),
    Message(role="tool", contents=[
        Content.from_function_result("c1", result="sunny, 18°C"),
    ]),
    Message(role="user", contents=[Content.from_text("And Friday?")]),
    Message(role="assistant", contents=[
        Content.from_function_call("c2", "get_forecast", arguments={"city": "Seattle"}),
    ]),
    Message(role="tool", contents=[
        Content.from_function_result("c2", result="clear, 22°C"),
    ]),
]

projected = asyncio.run(apply_compaction(
    messages,
    strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
))

for m in projected:
    text = m.text or ", ".join(c.type for c in m.contents)
    print(f"[{m.role:9}] {text}")
```

prints:

```text
[user     ] Weather in Seattle?
[assistant] [Tool results: get_weather: sunny, 18°C]
[user     ] And Friday?
[assistant] function_call
[tool     ] function_result
```

`apply_compaction` is the public helper that annotates groups, runs the strategy in place, and returns the projected list a model would actually receive. The first `(call_1, function_result)` pair is gone from the projection — replaced by the synthetic summary. The most recent group (`call_2`) survives untouched because `keep_last_tool_call_groups=1`. The raw `messages` list still contains the originals (with exclusion annotations) so you can render them in a debug UI or inspect annotations later.

The summary message carries `summary_of_message_ids` and `summary_of_group_ids` annotations linking back to the originals, so an audit-trail UI can offer a "show original tool calls" affordance without any extra bookkeeping.

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

### Tuning `target_count` and `threshold`

`SummarizationStrategy` has two knobs that govern *when* it runs and *how aggressively* it shrinks. Both default to small values that are reasonable for chat — but understanding them is the difference between a strategy that fires every turn and one that fires once an hour:

| Parameter | Default | What it controls |
|---|---|---|
| `target_count` | `4` | After summarisation, how many included non-system messages to retain. The strategy keeps groups from the *end* of history backward until at least `target_count` messages remain. |
| `threshold` | `2` | Hysteresis. The strategy only triggers when the current included non-system message count exceeds `target_count + threshold`. Set higher to trigger less often, lower to trigger sooner. |

In practice:

```python
# Aggressive — collapses anything past 6 messages, fires every turn beyond 8.
SummarizationStrategy(client=summariser_client, target_count=4, threshold=2)

# Lazy — only fires once history grows past 60, then collapses to ~40.
SummarizationStrategy(client=summariser_client, target_count=40, threshold=20)

# No hysteresis — re-summarises every turn that exceeds target_count.
SummarizationStrategy(client=summariser_client, target_count=10, threshold=0)
```

A high `threshold` is what keeps you from re-running the summariser on every single turn — the LLM call isn't free, and stable summaries across multiple turns are usually preferred. The strategy preserves atomic group boundaries, so the actual retained count may exceed `target_count` slightly (e.g. a tool-call group of 3 messages won't be split).

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
    """Implements the `TokenizerProtocol` — a single `count_tokens(text)` method."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._enc = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


strategy = TokenBudgetComposedStrategy(
    token_budget=8_000,
    tokenizer=TiktokenTokenizer(),
    strategies=[...],
)
```

`CharacterEstimatorTokenizer` ships with the framework for when you just need a rough character→token heuristic. The implementation is intentionally trivial — `max(1, len(text) // 4)` — so it never returns 0 for non-empty text and avoids dragging in a real tokenizer dependency:

```python
from agent_framework import CharacterEstimatorTokenizer

est = CharacterEstimatorTokenizer()
est.count_tokens("hello")           # 1   (5 // 4 → 1)
est.count_tokens("")                # 1   (clamped — never 0)
est.count_tokens("a" * 4_000)       # 1000
```

It's good enough for budget-aware fallbacks during dev/test. For production cost accounting use the real tokenizer that matches your model. Any object with a `count_tokens(text: str) -> int` method satisfies `TokenizerProtocol` — the protocol is structurally typed (`runtime_checkable`), so no inheritance is required.

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

#### The two-phase lifecycle in detail

Knowing exactly *when* each strategy fires lets you reason about token cost and history retention separately:

```text
agent.run(...)
   │
   ├─ HistoryProvider.load_session(...) loads stored messages into context
   ├─ CompactionProvider.before_run(...)   ← before_strategy mutates loaded context
   ├─ ──── model call(s) + tool loop ────  ← uses the (now-compacted) context
   ├─ HistoryProvider.persist(...) writes new messages into session.state
   ├─ CompactionProvider.after_run(...)    ← after_strategy mutates stored history
   │       (looked up by `history_source_id` in session.state)
   └─ return AgentResponse
```

Two practical consequences:

- **Same strategy, different goals.** A common pattern is "summarise once it gets big" — a `SummarizationStrategy` as `after_strategy` keeps storage compact, while a cheap `SlidingWindowStrategy` as `before_strategy` is a safety net in case the next turn still loads more than expected.
- **`history_source_id` must match the history provider.** The `after_strategy` looks up history under `session.state[history_source_id]`. If you change the history provider's `source_id` (say, you mount two providers as `"long_term"` and `"short_term"`) you must thread the matching id into `CompactionProvider(history_source_id=...)` or compaction silently no-ops on persist.

#### Inspecting compaction in tests

`apply_compaction` is the entry point you call directly — it's an async coroutine, so wrap it in `asyncio.run` (or `await` it from an async test). Handy in unit tests so you can assert on inclusion without spinning up an agent:

```python
import asyncio
from agent_framework import (
    Message,
    SlidingWindowStrategy,
    apply_compaction,
    included_messages,
)

messages = [
    Message(role="system", contents=["Be concise."]),
    *(Message(role="user", contents=[f"q{i}"]) for i in range(10)),
]

asyncio.run(apply_compaction(messages, strategy=SlidingWindowStrategy(keep_last_groups=3)))

assert [m.role for m in included_messages(messages)] == ["system", "user", "user", "user"]
```

Excluded messages remain in the original list with `EXCLUDED_KEY=True` annotations — `apply_compaction` mutates the list in place rather than returning a copy.

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

Anything callable that matches the protocol works:

```python
async def drop_old_errors(messages: list[Message]) -> bool:
    changed = False
    for m in messages:
        if m.role == "tool" and "error" in (m.text or "").lower():
            from agent_framework._compaction import set_excluded
            changed = set_excluded(m, excluded=True, reason="old_error") or changed
    return changed
```

Compose with the built-ins via `TokenBudgetComposedStrategy` or call inline.

## Inspecting what compaction did

Three low-level helpers give full visibility into the group structure and what survived compaction.

### `included_messages` and `included_token_count`

```python
from agent_framework import included_messages, included_token_count

kept = included_messages(messages)
print(len(kept))                    # how many messages survived
print(included_token_count(messages))   # estimated token count of kept messages
```

`included_messages(messages)` returns only the messages where `EXCLUDED_KEY` is not set — effectively the list the model will receive. `included_token_count(messages)` sums the `token_count` annotations already attached to those messages (set by the group annotator); the value is zero for messages that were never annotated.

Both helpers operate on the in-place-mutated list — call them after `apply_compaction()` or after the framework's own compaction pass completes.

### `annotate_message_groups` — low-level group inspection

`annotate_message_groups` is what the built-in strategies call internally before deciding what to exclude. It annotates each message with a `group_id`, `group_index`, and `group_type` in `additional_properties`. You can call it directly to understand the group structure without running any strategy:

```python
from agent_framework import (
    Content,
    Message,
    annotate_message_groups,
)

messages = [
    Message(role="system", contents=[Content.from_text("Be concise.")]),
    Message(role="user",   contents=[Content.from_text("What is RAG?")]),
    Message(role="assistant", contents=[Content.from_text("RAG stands for Retrieval-Augmented Generation.")]),
    Message(role="assistant", contents=[
        Content.from_function_call("c1", "search_docs", arguments={"query": "RAG"}),
    ]),
    Message(role="tool", contents=[
        Content.from_function_result("c1", result="RAG: retrieve then generate."),
    ]),
    Message(role="user",   contents=[Content.from_text("Give an example.")]),
    Message(role="assistant", contents=[Content.from_text("Sure! ...")]),
]

# Annotate without running any strategy — writes group metadata in place but does NOT mark exclusions
annotate_message_groups(messages)

for m in messages:
    props = m.additional_properties
    print(
        f"role={m.role:9}  "
        f"group={props.get('group_id', '?'):4}  "
        f"type={props.get('group_type', '?')}"
    )
```

This prints something like:

```text
role=system    group=0     type=system
role=user      group=1     type=user
role=assistant group=2     type=assistant
role=assistant group=3     type=tool_call
role=tool      group=3     type=tool_call
role=user      group=4     type=user
role=assistant group=5     type=assistant
```

Key observations:
- The `function_call` and `function_result` share **the same `group_id`** (`3`). This is how strategies keep tool calls atomic — they never exclude one without the other.
- `group_type` values are `"system"`, `"user"`, `"assistant"`, and `"tool_call"`. Strategies use this to selectively target only tool-call groups (`SelectiveToolCallCompactionStrategy`) or to skip system anchors (`preserve_system=True`).
- `annotate_message_groups` is **idempotent** — calling it twice on the same list is safe, and existing annotations are overwritten cleanly.

### Building a debug summary

Combine all three helpers to produce a human-readable before/after summary for logs or debug UIs:

```python
import asyncio
from agent_framework import (
    Content,
    EXCLUDED_KEY,
    EXCLUDE_REASON_KEY,
    Message,
    SlidingWindowStrategy,
    apply_compaction,
    included_messages,
    included_token_count,
)

messages = [
    Message(role="system", contents=[Content.from_text("Be concise.")]),
    *[
        Message(role=role, contents=[Content.from_text(f"{role} turn {i}")])
        for i in range(6)
        for role in ("user", "assistant")
    ],
]

total_before = len(messages)
asyncio.run(apply_compaction(messages, strategy=SlidingWindowStrategy(keep_last_groups=4)))

kept = included_messages(messages)
excluded = [m for m in messages if m.additional_properties.get(EXCLUDED_KEY)]

print(f"Total messages : {total_before}")
print(f"Kept           : {len(kept)}")
print(f"Excluded       : {len(excluded)}")
print(f"Est. tokens    : {included_token_count(messages)}")

# Show the reason for the first exclusion
if excluded:
    reason = excluded[0].additional_properties.get(EXCLUDE_REASON_KEY, "unknown")
    print(f"Exclusion reason: {reason}")
# → Exclusion reason: sliding_window
```

Excluded messages stay in the original list — `apply_compaction` mutates in place and never discards messages. This lets you render "32 messages compacted" affordances in a UI or reconstruct the full history for export.

## Patterns

**Default for chat UIs.** `SlidingWindowStrategy(keep_last_groups=20)` as `before_strategy`, `ToolResultCompactionStrategy(keep_last_tool_call_groups=1)` as `after_strategy`. Predictable, no judge tokens.

**Research agents with long plans.** Use `SummarizationStrategy` with a cheap model — Phi-3 / Haiku / gpt-4o-mini. The summary retains goals and open questions so the agent stays on-track past 50 turns.

**Strict SLA on context size.** `TokenBudgetComposedStrategy(token_budget=32_000, ...)` guarantees you never send >32k tokens, with deterministic fallback even if summarisation fails.

**Multi-tool pipelines.** Tool results often dwarf user/assistant turns. `SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)` preserves the important reasoning while dropping the polling noise.

**A/B test strategies.** Run `evaluate_agent` (see [Evaluation](./microsoft_agent_framework_python_evaluation/)) twice with different `compaction_strategy=` overrides and compare pass rates.

## End-to-end recipes for each strategy

The full constructor signatures are short. Each recipe builds on the same shared scaffolding — instantiate it once and reuse across the rest of this section:

```python
import tiktoken
from agent_framework import (
    InMemoryHistoryProvider,
    TokenizerProtocol,
)
from agent_framework.openai import OpenAIChatClient


class TiktokenTokenizer:
    """Implements the `TokenizerProtocol` — a single `count_tokens(text)` method."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._enc = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


# Cheap model used for summarisation only — separate from the agent's main model.
summary_client = OpenAIChatClient(model="gpt-4o-mini")
tokenizer: TokenizerProtocol = TiktokenTokenizer()
history = InMemoryHistoryProvider()
```

### `SummarizationStrategy` — long research conversations

When earlier context still matters but you don't want to ship every turn, summarise the older portion with a cheap model and keep the newest turns intact:

```python
from agent_framework import (
    Agent,
    CompactionProvider,
    SummarizationStrategy,
)

summariser = SummarizationStrategy(
    client=summary_client,           # defined in the shared scaffolding above
    target_count=6,        # keep ~6 most-recent non-system messages verbatim
    threshold=4,           # trigger once we exceed target_count + threshold = 10
    prompt=(
        "Summarise the conversation so far. Preserve: stated goals, open questions, "
        "decisions already made, and unresolved blockers. Drop chit-chat and pleasantries. "
        "Keep the summary under 200 words."
    ),
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="You are a research assistant for a multi-week project.",
    context_providers=[
        history,
        CompactionProvider(after_strategy=summariser, history_source_id=history.source_id),
    ],
)
```

`SummarizationStrategy` writes back-pointers (`SUMMARY_OF_MESSAGE_IDS_KEY`, `SUMMARY_OF_GROUP_IDS_KEY`) on the new summary message and `SUMMARIZED_BY_SUMMARY_ID_KEY` on each replaced message — so debugging tools can show "this summary covered turns 2–14".

If the summariser call fails, the strategy logs a warning and returns `False` — your conversation continues uncompacted rather than dying. Pair with a `SlidingWindowStrategy` as a safety net for the next turn.

### `TokenBudgetComposedStrategy` — guarantee a context ceiling

Run cheaper strategies first; fall through to summarisation only when needed; have a deterministic fallback that excludes oldest groups when even the LLM summariser couldn't squeeze under budget:

```python
from agent_framework import (
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    ToolResultCompactionStrategy,
)

# `summary_client` and `tokenizer` come from the shared scaffolding above.
budget_strategy = TokenBudgetComposedStrategy(
    token_budget=24_000,                 # never ship more than 24k tokens
    tokenizer=tokenizer,
    early_stop=True,                     # stop as soon as we're under budget
    strategies=[
        # 1. Cheapest first — drop tool-call chatter.
        ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
        # 2. Then trim middle of the conversation.
        SlidingWindowStrategy(keep_last_groups=30),
        # 3. Last resort — summarise older content with a cheap model.
        SummarizationStrategy(client=summary_client, target_count=12, threshold=4),
    ],
)
```

The composed strategy applies each step in order, refreshes token counts, and stops when `early_stop=True` and the budget is met. If after running every step the count is still over, it deterministically excludes the oldest non-system groups one at a time — no run ever sends more than `token_budget`. System anchors are dropped only as a last resort.

### `SelectiveToolCallCompactionStrategy` — tool-heavy agents

Polling-style tools (status checks, retries) bloat history fast. Keep just the most recent tool-call group:

```python
from agent_framework import SelectiveToolCallCompactionStrategy

trim_tools = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=1)
```

`keep_last_tool_call_groups=0` drops every tool-call group — useful when the assistant has summarised the result into prose and the raw call/response is no longer load-bearing.

### Mixing strategies — `before_strategy` ≠ `after_strategy`

The two phases run on different message lists with different goals. A common production layout:

```python
provider = CompactionProvider(
    # Read-time safety net — keeps the live request affordable even if persisted history grew.
    before_strategy=SlidingWindowStrategy(keep_last_groups=40),
    # Persist-time pruning — what gets stored for the next turn.
    after_strategy=TokenBudgetComposedStrategy(
        token_budget=8_000,
        tokenizer=tokenizer,
        strategies=[
            ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
            SummarizationStrategy(client=summary_client, target_count=8),
        ],
    ),
    history_source_id=history.source_id,
)
```

Read-side window is generous; persisted-side budget is tight. The next turn loads at most 8k tokens, then `before_strategy` prevents accidentally re-loading more if a different provider also wrote into context.

## Inspecting state with the framework constants

The annotation keys are exported as constants — use them when you need to inspect compaction decisions in middleware or debug tooling without re-typing the underlying string:

```python
from agent_framework import (
    EXCLUDED_KEY,
    EXCLUDE_REASON_KEY,
    GROUP_ANNOTATION_KEY,
    GROUP_KIND_KEY,
    SUMMARIZED_BY_SUMMARY_ID_KEY,
)


def explain_compaction(messages):
    for m in messages:
        excluded = m.additional_properties.get(EXCLUDED_KEY, False)
        reason = m.additional_properties.get(EXCLUDE_REASON_KEY, "")
        group = m.additional_properties.get(GROUP_ANNOTATION_KEY) or {}
        kind = group.get(GROUP_KIND_KEY, "?")
        replaced_by = m.additional_properties.get(SUMMARIZED_BY_SUMMARY_ID_KEY, "")
        flag = "✗" if excluded else "✓"
        suffix = f" → {reason}" if reason else ""
        if replaced_by:
            suffix += f" (summarised as {replaced_by})"
        print(f"{flag} [{kind:14}] {(m.text or '')[:80]}{suffix}")
```

Wire this into a debug `AgentMiddleware` that runs after compaction to see exactly what the model received vs. what was kept on disk.

## Low-level compaction utilities

`agent_framework` exports three utility functions that the strategies use internally. You can call them directly when building custom strategies, debug tooling, or middleware that inspects conversation state.

### `annotate_message_groups` — assign group IDs and kinds

Group annotation is the prerequisite for all compaction strategies. Each message gets stamped with a `_group` dict that carries `id`, `kind`, `index`, `has_reasoning`, and (optionally) `token_count`. The strategies then read those annotations instead of re-parsing the conversation on every call.

```python
import asyncio
from agent_framework import (
    Agent,
    Content,
    Message,
    annotate_message_groups,
    GROUP_ANNOTATION_KEY,
    GROUP_ID_KEY,
    GROUP_KIND_KEY,
    GROUP_INDEX_KEY,
)
from agent_framework.openai import OpenAIChatClient


def show_groups(messages: list[Message]) -> None:
    """Print the group annotation of each message."""
    for msg in messages:
        group = msg.additional_properties.get(GROUP_ANNOTATION_KEY) or {}
        kind = group.get(GROUP_KIND_KEY, "?")
        index = group.get(GROUP_INDEX_KEY, "?")
        snippet = (msg.text or "")[:60]
        print(f"  [{index:>3}] {kind:<20} {snippet!r}")


# Build a small synthetic conversation
conversation = [
    Message(role="system",    contents=[Content.from_text("You are an assistant.")]),
    Message(role="user",      contents=[Content.from_text("What's 2+2?")]),
    Message(role="assistant", contents=[Content.from_text("4")]),
    Message(role="user",      contents=[Content.from_text("And 3+3?")]),
    Message(role="assistant", contents=[Content.from_text("6")]),
]

group_ids = annotate_message_groups(conversation)
print(f"Group IDs: {group_ids}")    # ['g0', 'g1', 'g2', 'g3', 'g4'] or similar
show_groups(conversation)
```

`annotate_message_groups` is **incremental** — it reuses existing annotations and only re-annotates the suffix that contains new messages. When you append a new turn, call it with `from_index=len(existing)` to avoid redundant work:

```python
new_msg = Message(role="user", contents=[Content.from_text("What about 4+4?")])
conversation.append(new_msg)

# Only annotate the last message — skip the already-annotated prefix
annotate_message_groups(conversation, from_index=len(conversation) - 1)
```

Pass `force_reannotate=True` to re-stamp the whole list (e.g. after splicing in a summary message that shifts group indices):

```python
annotate_message_groups(conversation, force_reannotate=True)
```

#### With a tokenizer for token-count annotations

Pair with `CharacterEstimatorTokenizer` to also stamp token counts per message — enables `TokenBudgetComposedStrategy` and `included_token_count`:

```python
from agent_framework import annotate_message_groups, CharacterEstimatorTokenizer, Message, Content

tokenizer = CharacterEstimatorTokenizer()   # 4 chars ≈ 1 token (fast heuristic)

messages = [
    Message(role="user",      contents=[Content.from_text("Explain quantum entanglement.")]),
    Message(role="assistant", contents=[Content.from_text("Quantum entanglement is a phenomenon...")]),
]

group_ids = annotate_message_groups(messages, tokenizer=tokenizer)
# Each message now has a token_count annotation that TokenBudgetComposedStrategy reads
```

### `included_messages` — filter to non-excluded messages

After a strategy runs and marks groups as excluded, `included_messages` returns only the messages the model will actually see. Use it to verify what a strategy did without sending anything to the API:

```python
import asyncio
from agent_framework import (
    Content,
    Message,
    SlidingWindowStrategy,
    annotate_message_groups,
    apply_compaction,
    included_messages,
)


async def main() -> None:
    messages = [
        Message(role="user",      contents=[Content.from_text(f"Question {i}")]),
        Message(role="assistant", contents=[Content.from_text(f"Answer {i}")])
        for i in range(10)
    ]
    # Flatten the list comprehension (above produces nested lists)
    flat: list[Message] = []
    for i in range(10):
        flat.append(Message(role="user",      contents=[Content.from_text(f"Q{i}")]))
        flat.append(Message(role="assistant", contents=[Content.from_text(f"A{i}")]))

    annotate_message_groups(flat)
    strategy = SlidingWindowStrategy(keep_last_groups=4)
    await apply_compaction(flat, strategy=strategy)

    visible = included_messages(flat)
    print(f"Total: {len(flat)}  Visible to model: {len(visible)}")
    for m in visible:
        print(f"  {m.role}: {m.text}")


asyncio.run(main())
```

### `included_token_count` — measure context size after compaction

Combine with `CharacterEstimatorTokenizer` to compute the approximate token spend the model will see after compaction:

```python
import asyncio
from agent_framework import (
    CharacterEstimatorTokenizer,
    Content,
    Message,
    SlidingWindowStrategy,
    annotate_message_groups,
    apply_compaction,
    included_token_count,
)


async def main() -> None:
    tokenizer = CharacterEstimatorTokenizer()

    messages: list[Message] = []
    for i in range(20):
        messages.append(Message(role="user",      contents=[Content.from_text(f"Tell me about topic {i}.")]))
        messages.append(Message(role="assistant", contents=[Content.from_text(f"Topic {i} is about ...")]))

    # Annotate with token counts before compaction
    annotate_message_groups(messages, tokenizer=tokenizer)
    before = included_token_count(messages)

    # Compact to the last 6 groups
    await apply_compaction(messages, strategy=SlidingWindowStrategy(keep_last_groups=6))

    after = included_token_count(messages)
    print(f"Before compaction: ~{before} tokens")
    print(f"After compaction:  ~{after} tokens")
    print(f"Saved:             ~{before - after} tokens ({100 * (before - after) / before:.0f}%)")


asyncio.run(main())
```

`included_token_count` only sums messages that are **not** excluded. If a message lacks a token-count annotation (because `annotate_message_groups` was called without a tokenizer), it contributes zero — so always pass a tokenizer when you care about accurate counts.

### Custom strategy using the low-level utilities

The utilities compose cleanly when you need behaviour the built-in strategies don't offer. Here's a strategy that keeps the system prompt plus the `N` most recent groups that together stay under a token budget:

```python
import asyncio
from agent_framework import (
    EXCLUDED_KEY,
    GROUP_ANNOTATION_KEY,
    GROUP_INDEX_KEY,
    CharacterEstimatorTokenizer,
    CompactionStrategy,
    Content,
    Message,
    annotate_message_groups,
    apply_compaction,
    included_token_count,
)


class BudgetWindowStrategy:
    """Keep the system prompt and the most-recent groups within a token budget."""

    def __init__(self, token_budget: int) -> None:
        self._budget = token_budget
        self._tokenizer = CharacterEstimatorTokenizer()

    async def __call__(self, messages: list[Message]) -> bool:
        # Ensure all messages are annotated with token counts
        annotate_message_groups(messages, tokenizer=self._tokenizer)

        # Collect non-system groups in reverse order (newest first)
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        # Estimate system token spend
        system_tokens = sum(
            self._tokenizer.count_tokens((m.text or "")) for m in system_msgs
        )
        remaining_budget = self._budget - system_tokens

        # Walk non-system messages newest-to-oldest; include until budget runs out
        keep_indices: set[int] = set()
        running_total = 0
        for msg in reversed(non_system):
            group = msg.additional_properties.get(GROUP_ANNOTATION_KEY) or {}
            idx = group.get(GROUP_INDEX_KEY)
            tok = self._tokenizer.count_tokens(msg.text or "")
            if running_total + tok <= remaining_budget:
                running_total += tok
                if idx is not None:
                    keep_indices.add(idx)
            else:
                break

        # Exclude non-system messages whose group index is not in keep_indices
        changed = False
        for msg in non_system:
            group = msg.additional_properties.get(GROUP_ANNOTATION_KEY) or {}
            idx = group.get(GROUP_INDEX_KEY)
            should_exclude = idx is not None and idx not in keep_indices
            current = msg.additional_properties.get(EXCLUDED_KEY, False)
            if should_exclude != current:
                msg.additional_properties[EXCLUDED_KEY] = should_exclude
                changed = True

        return changed


async def main() -> None:
    messages: list[Message] = []
    messages.append(Message(role="system", contents=[Content.from_text("You are an assistant.")]))
    for i in range(15):
        messages.append(Message(role="user",      contents=[Content.from_text(f"Message {i}: " + "word " * 20)]))
        messages.append(Message(role="assistant", contents=[Content.from_text(f"Response {i}: " + "word " * 30)]))

    strategy = BudgetWindowStrategy(token_budget=200)
    await apply_compaction(messages, strategy=strategy)
    print(f"Visible: {included_token_count(messages)} tokens in {len([m for m in messages if not m.additional_properties.get(EXCLUDED_KEY)])} messages")


asyncio.run(main())
```

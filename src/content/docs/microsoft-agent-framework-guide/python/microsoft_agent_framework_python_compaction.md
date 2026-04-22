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

Triggers only when needed; preserves goals, decisions, and unresolved items via the default prompt. Override with `prompt=` for domain-specific summaries.

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

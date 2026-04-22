---
title: "PydanticAI: Message History & Multi-Turn"
description: "Model messages, message_history, new_messages vs all_messages, ModelMessagesTypeAdapter (de)serialization, and HistoryProcessor."
framework: pydanticai
language: python
---

# Message History & Multi-Turn Conversations

Verified against **pydantic-ai==1.85.1** — source modules: `pydantic_ai.messages`, `pydantic_ai.run`, `pydantic_ai._history_processor`.

Every run of a PydanticAI agent produces a list of `ModelMessage`s. Pass that list (or the serialised form) back into the next run to keep a multi-turn conversation. Message history is what makes chat applications possible on top of PydanticAI's otherwise stateless `Agent`.

## Minimal runnable example

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')

result1 = agent.run_sync('My name is Alice.')
print(result1.output)
#> Hello Alice, nice to meet you.

# Feed prior messages into the next run
result2 = agent.run_sync(
    'What is my name?',
    message_history=result1.all_messages(),
)
print(result2.output)
#> Your name is Alice.
```

`result.all_messages()` is the input to the _next_ run's `message_history`. There is no global state — you own the list.

## The message types

`ModelMessage` is a discriminated union of `ModelRequest` and `ModelResponse` (`messages.py:2030`):

| Type            | Direction      | Notable parts inside                                      |
| --------------- | -------------- | --------------------------------------------------------- |
| `ModelRequest`  | client → model | `SystemPromptPart`, `UserPromptPart`, `ToolReturnPart`, `RetryPromptPart`, `InstructionPart` |
| `ModelResponse` | model → client | `TextPart`, `ThinkingPart`, `ToolCallPart`, `BuiltinToolCallPart`, `BuiltinToolReturnPart`, `FilePart` |

Every part has a `part_kind` discriminator; delta events (`TextPartDelta`, `ThinkingPartDelta`, `ToolCallPartDelta`) exist for streaming and carry `part_delta_kind`.

## `all_messages()` vs `new_messages()`

Both return `list[ModelMessage]`. The difference is what counts as "this run":

| Method              | Includes `message_history` input | Includes this run's new messages |
| ------------------- | -------------------------------- | -------------------------------- |
| `all_messages()`    | yes                              | yes                              |
| `new_messages()`    | no                               | yes                              |

Use `all_messages()` as the next run's `message_history`; use `new_messages()` when you only want to persist what _this_ turn produced.

There are JSON variants too (`run.py:151`): `all_messages_json()` and `new_messages_json()` return `bytes` you can store in a DB column.

```python
from pydantic_ai.messages import ModelMessagesTypeAdapter

# Persist just the delta produced by this turn (don't byte-concat JSON arrays — load, extend, re-dump)
existing = ModelMessagesTypeAdapter.validate_json(row.messages_json) if row.messages_json else []
existing.extend(result.new_messages())
row.messages_json = ModelMessagesTypeAdapter.dump_json(existing)
```

## (De)serialisation with `ModelMessagesTypeAdapter`

`messages.py:2034` exports a Pydantic `TypeAdapter[list[ModelMessage]]`. This is the supported way to round-trip history through JSON / DB / queue:

```python
from pydantic_ai.messages import ModelMessagesTypeAdapter

blob: bytes = ModelMessagesTypeAdapter.dump_json(result.all_messages())

# later, on a different process / worker
history = ModelMessagesTypeAdapter.validate_json(blob)
next_result = agent.run_sync('Keep going.', message_history=history)
```

Binary content (images, files) is encoded as base64 thanks to `ser_json_bytes='base64'` / `val_json_bytes='base64'` on the adapter config.

## Passing history back in

Any of `run`, `run_sync`, `run_stream`, or `iter` accept `message_history`:

```python
async with agent.run_stream(
    'Continue the conversation.',
    message_history=prior_history,
) as stream:
    async for chunk in stream.stream_text(delta=True):
        print(chunk, end='')
```

When `message_history` is provided, the agent's system prompt is **not** re-sent — the list is assumed to already contain it. If you want to layer a new system-level instruction on top of an existing history, use `agent.override(instructions=...)` or a run-level `instructions=` argument.

## `HistoryProcessor` — trim, summarise, or redact in flight

`HistoryProcessor` is a callable that runs _before_ each model request (`_history_processor.py`). Signatures (all accepted):

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

def last_n(msgs: list[ModelMessage]) -> list[ModelMessage]:
    return msgs[-10:]

async def async_ctx(
    ctx: RunContext[None], msgs: list[ModelMessage]
) -> list[ModelMessage]:
    return msgs[-ctx.deps.window:] if ctx.deps else msgs

agent = Agent(
    'openai:gpt-5.2',
    history_processors=[last_n],
)
```

Processors run in the order given; each one receives the output of the previous. They can:

- Drop messages (context window management).
- Collapse long tool outputs into summaries.
- Redact PII before it hits the provider.
- Rewrite system messages on a per-deps basis (async + ctx variant).

Processors are **not** applied to the stored `all_messages()`; they only affect what goes over the wire to the model.

## `capture_run_messages` — inspect mid-failure

If a run raises (validation error, model error, tool retry exhaustion), you still need to see the messages up to the failure. `capture_run_messages` (`_agent_graph.py:1791`) exposes them:

```python
from pydantic_ai import Agent, capture_run_messages

agent = Agent('openai:gpt-5.2')

with capture_run_messages() as messages:
    try:
        agent.run_sync('foobar')
    except Exception:
        print(messages)  # full list, including the failed response
        raise
```

Caveat: if you call `run*` more than once inside the `with`, only the first call's messages are captured.

## Inspecting tool-call traffic

Each `ModelResponse.parts` can contain `ToolCallPart`s; each follow-up `ModelRequest.parts` contains the matching `ToolReturnPart` keyed by `tool_call_id`. When walking history:

```python
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

for msg in result.all_messages():
    if isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                print(f'{part.tool_name}({part.args_as_json_str()})')
```

## Patterns

### 1. DB-backed conversation store

```python
def load(session_id: str) -> list[ModelMessage]:
    row = db.sessions.get(session_id)
    return ModelMessagesTypeAdapter.validate_json(row.history) if row else []

def save(session_id: str, msgs: list[ModelMessage]) -> None:
    db.sessions.upsert(
        session_id,
        history=ModelMessagesTypeAdapter.dump_json(msgs),
    )

history = load(sid)
result = agent.run_sync(user_prompt, message_history=history)
save(sid, result.all_messages())
```

### 2. Sliding-window processor

```python
def window(n: int):
    def _proc(msgs: list[ModelMessage]) -> list[ModelMessage]:
        # Keep the first (system prompt + first user turn) plus the last n
        return msgs[:2] + msgs[-n:] if len(msgs) > n + 2 else msgs
    return _proc

agent = Agent('openai:gpt-5.2', history_processors=[window(20)])
```

### 3. Summarise-on-overflow

```python
async def summarise_if_long(msgs: list[ModelMessage]) -> list[ModelMessage]:
    if len(msgs) < 40:
        return msgs
    head, tail = msgs[:2], msgs[-10:]
    summary = await summariser.run(str(msgs[2:-10]))
    return head + [ModelRequest([UserPromptPart(f'Prior context: {summary.output}')])] + tail
```

### 4. Redacting PII before the provider sees it

```python
import re
PHONE = re.compile(r'\+?\d[\d\-\s]{7,}\d')

def redact(msgs: list[ModelMessage]) -> list[ModelMessage]:
    for m in msgs:
        for p in m.parts:
            if hasattr(p, 'content') and isinstance(p.content, str):
                p.content = PHONE.sub('[REDACTED]', p.content)
    return msgs
```

### 5. Branching conversations

Because `message_history` is just a list you pass in, forking a conversation is a list slice:

```python
branch_a = result.all_messages()
branch_b = result.all_messages()
a = agent.run_sync('Continue politely.', message_history=branch_a)
b = agent.run_sync('Continue sarcastically.', message_history=branch_b)
```

## Gotchas

- **Don't mutate `result.all_messages()` in place if you plan to re-use it.** The list is owned by the run context; copy via `list(...)` before modifying.
- **System prompt is in the list.** When concatenating histories across agents, drop the leading `SystemPromptPart`s from the imported history and let the receiving agent add its own.
- **`HistoryProcessor` runs on every step** of a multi-step (tool-calling) run, not just once per user turn. Keep them cheap.
- **Images / files in history**: binaries are base64-encoded in JSON. Large attachments bloat the stored blob — consider storing a reference (`ImageUrl`) instead of `BinaryImage` if you re-send across turns.

## Reference

- `ModelMessage`, `ModelRequest`, `ModelResponse` — `messages.py`
- `ModelMessagesTypeAdapter` — `messages.py:2034`
- `AgentRunResult.all_messages()` / `.new_messages()` / `.all_messages_json()` — `run.py`
- `capture_run_messages()` — `_agent_graph.py:1791`
- `HistoryProcessor` — `_history_processor.py:17`
- `Agent(history_processors=...)` — `agent/__init__.py:220`

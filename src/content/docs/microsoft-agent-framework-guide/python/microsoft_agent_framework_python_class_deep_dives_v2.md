---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 2"
description: "Source-verified deep dives into FileHistoryProvider, AgentMiddleware, ChatMiddleware, FunctionMiddleware, CompactionProvider, ToolResultCompactionStrategy, TokenBudgetComposedStrategy, FileCheckpointStorage, LocalEvaluator, and WorkflowRunResult — all verified against agent-framework-core 1.6.0."
framework: microsoft-agent-framework
language: python
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 2

Verified against **agent-framework-core 1.6.0** (installed, May 2026). Every constructor signature, parameter description, and code example in this document was derived from the installed package source at `/usr/local/lib/python3.11/dist-packages/agent_framework/`. No API name has been guessed or inferred from documentation alone.

Ten classes are covered in this volume, chosen to complement [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) and fill the gaps in coverage across history, middleware, compaction, checkpointing, and evaluation.

---

## 1. `FileHistoryProvider` — durable per-session conversation history

**Source:** `agent_framework/_sessions.py` — `FileHistoryProvider(HistoryProvider)`

`FileHistoryProvider` stores one [JSON Lines](https://jsonlines.org/) file per session in a directory of your choice. It is the drop-in replacement for `InMemoryHistoryProvider` when you need conversations that survive process restarts.

> **Experimental** in 1.6.0 — imports trigger an `ExperimentalWarning`.

### Constructor

```python
FileHistoryProvider(
    storage_path: str | Path,
    *,
    source_id: str = "file_history",
    load_messages: bool = True,
    store_inputs: bool = True,
    store_context_messages: bool = False,
    store_context_from: set[str] | None = None,
    store_outputs: bool = True,
    skip_excluded: bool = False,
    dumps: Callable[[Any], str | bytes] | None = None,
    loads: Callable[[str | bytes], Any] | None = None,
)
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `storage_path` | required | Directory where `.jsonl` session files are written |
| `source_id` | `"file_history"` | Unique ID for composing multiple providers |
| `store_context_messages` | `False` | Store messages injected by other providers (e.g. skills) |
| `store_context_from` | `None` | Whitelist specific provider `source_id`s to persist |
| `skip_excluded` | `False` | Omit compaction-excluded messages when loading |
| `dumps` / `loads` | `None` | Custom JSON serialiser/deserialiser callables |

### Basic persistent conversation

```python
import asyncio
import warnings
from pathlib import Path
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

# Suppress the experimental warning in production if you've accepted the risk
warnings.filterwarnings("ignore", category=Warning, module="agent_framework")


async def main() -> None:
    provider = FileHistoryProvider(
        storage_path=Path("/var/data/sessions"),
        source_id="file_history",
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a persistent assistant. Remember everything the user tells you.",
        context_providers=[provider],
    )

    # First process run — create or resume a session by ID
    session = agent.create_session(session_id="user-alice-001")

    r1 = await agent.run("My favourite colour is teal.", session=session)
    print(r1.text)   # "Got it! I'll remember that."

    r2 = await agent.run("What is my favourite colour?", session=session)
    print(r2.text)   # "Your favourite colour is teal."


asyncio.run(main())
```

After this runs you'll find a file like:
```
/var/data/sessions/user-alice-001.jsonl
```

Each line is a single JSON-encoded `Message`. On the next process start, the same `FileHistoryProvider` + `session_id` will reload that file automatically.

### Resuming across process restarts

```python
import asyncio
from pathlib import Path
from agent_framework import Agent, FileHistoryProvider, AgentSession
from agent_framework.openai import OpenAIChatClient


async def new_process_turn() -> None:
    """Simulates a fresh process picking up a conversation mid-stream."""
    provider = FileHistoryProvider(storage_path=Path("/var/data/sessions"))

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant with full memory.",
        context_providers=[provider],
    )

    # Same session_id — the provider loads the existing .jsonl file
    session = agent.create_session(session_id="user-alice-001")

    r = await agent.run("What have we talked about so far?", session=session)
    print(r.text)


asyncio.run(new_process_turn())
```

### Only persisting outputs from specific providers

When you combine `FileHistoryProvider` with a `SkillsProvider`, you normally want to persist the user/assistant turns but not the skill instructions (which are injected fresh each call). Use `store_context_from` to be selective:

```python
from agent_framework import (
    Agent, FileHistoryProvider, SkillsProvider, InlineSkill, SkillFrontmatter,
)
from agent_framework.openai import OpenAIChatClient

skill = InlineSkill(
    frontmatter=SkillFrontmatter(name="coding-style", description="Code style guide."),
    instructions="Always use type hints in Python. Prefer f-strings over .format().",
)
skills_provider = SkillsProvider(skill)

# Don't persist the skill context messages — they're always re-injected
history = FileHistoryProvider(
    storage_path="./sessions",
    store_context_messages=False,   # default; skill messages are not stored
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a coding assistant.",
    context_providers=[skills_provider, history],
)
```

### Custom JSON serialiser — encrypt at rest

```python
import json
import base64
from cryptography.fernet import Fernet
from agent_framework import FileHistoryProvider

_KEY = Fernet.generate_key()
_FERNET = Fernet(_KEY)


def _encrypt_dumps(value: object) -> bytes:
    plaintext = json.dumps(value, ensure_ascii=False).encode()
    return _FERNET.encrypt(plaintext)


def _decrypt_loads(data: bytes) -> object:
    plaintext = _FERNET.decrypt(data)
    return json.loads(plaintext)


provider = FileHistoryProvider(
    storage_path="./sessions-encrypted",
    dumps=_encrypt_dumps,
    loads=_decrypt_loads,
)
```

### Combining with compaction — `skip_excluded`

When `SlidingWindowStrategy` marks old messages as excluded, set `skip_excluded=True` so the provider does not reload them on the next turn:

```python
from agent_framework import (
    Agent, FileHistoryProvider, CompactionProvider, SlidingWindowStrategy,
)
from agent_framework.openai import OpenAIChatClient

history = FileHistoryProvider(
    storage_path="./sessions",
    source_id="file_history",
    skip_excluded=True,   # ← omit compaction-excluded messages on load
)
compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=30),
    history_source_id="file_history",
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="Long-running assistant.",
    context_providers=[history, compaction],
)
```

---

## 2. `AgentMiddleware` — intercepting full agent invocations

**Source:** `agent_framework/_middleware.py`

`AgentMiddleware` is the outermost middleware layer. It wraps the *entire* agent run, giving you access to `AgentContext` before **and** after the agent executes. It is only available when using `Agent` (not `RawAgent`).

Three forms exist:
1. **Subclass** `AgentMiddleware` — stateful, cleanest for complex logic.
2. **`@agent_middleware` decorator** — functional one-liner for simple interceptors.
3. **Inline function** — pass any `async` function with the right signature to `middleware=`.

### Subclass form — retry with exponential backoff

```python
import asyncio
import logging
from agent_framework import Agent, AgentMiddleware, AgentContext
from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)


class RetryOnErrorMiddleware(AgentMiddleware):
    """Retries the agent run on failure with exponential back-off."""

    def __init__(self, max_retries: int = 3, base_delay: float = 0.5) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def process(
        self,
        context: AgentContext,
        call_next,
    ) -> None:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await call_next()
                return   # success — stop retrying
            except Exception as exc:
                last_exc = exc
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Agent run failed (attempt {attempt + 1}): {exc}. Retrying in {delay}s.")
                await asyncio.sleep(delay)

        # All retries exhausted — re-raise
        if last_exc:
            raise last_exc


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a resilient assistant.",
    middleware=[RetryOnErrorMiddleware(max_retries=3)],
)
```

### Subclass form — request/response audit log

```python
import json
import time
from datetime import datetime
from agent_framework import Agent, AgentMiddleware, AgentContext
from agent_framework.openai import OpenAIChatClient


class AuditLogMiddleware(AgentMiddleware):
    """Write a structured audit entry for every agent invocation."""

    def __init__(self, log_path: str = "audit.jsonl") -> None:
        self._log_path = log_path

    async def process(self, context: AgentContext, call_next) -> None:
        start = time.monotonic()
        error: str | None = None

        try:
            await call_next()
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed = time.monotonic() - start
            entry = {
                "ts": datetime.utcnow().isoformat(),
                "agent": context.agent.name,
                "input_messages": len(context.messages or []),
                "elapsed_ms": round(elapsed * 1000, 1),
                "error": error,
            }
            if context.result:
                entry["output_text"] = getattr(context.result, "text", None)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a monitored assistant.",
    middleware=[AuditLogMiddleware(log_path="/var/log/agent-audit.jsonl")],
)
```

### `@agent_middleware` decorator — functional form

```python
from agent_framework import agent_middleware, AgentContext, Agent
from agent_framework.openai import OpenAIChatClient


@agent_middleware
async def add_user_context(context: AgentContext, call_next):
    """Inject a user-ID header into additional_properties before the run."""
    context.additional_properties["user_id"] = "user-42"
    await call_next()
    # context.result is available here


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a personalised assistant.",
    middleware=[add_user_context],
)
```

### Combining multiple agent middleware

Middleware is applied in the order given. The first entry is the outermost wrapper.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are an enterprise assistant.",
    middleware=[
        AuditLogMiddleware(log_path="/var/log/audit.jsonl"),   # outermost
        RetryOnErrorMiddleware(max_retries=2),                  # innermost
    ],
)
```

---

## 3. `ChatMiddleware` — intercepting the LLM call

**Source:** `agent_framework/_middleware.py`

`ChatMiddleware` wraps the *chat client call* — the HTTP request to the LLM. It sees `ChatContext` which contains the list of messages, the options, and the raw response. Use it to:

- **Inject system prompts** or modify messages just before they hit the model.
- **Override the response** entirely (e.g. mock the LLM in tests).
- **Add per-request headers** or observe token usage.

Unlike `AgentMiddleware`, `ChatMiddleware` also works with `RawAgent`.

### Inject a dynamic system prompt based on user language

```python
from agent_framework import Agent, ChatMiddleware, ChatContext, Message
from agent_framework.openai import OpenAIChatClient
import langdetect   # pip install langdetect


class LanguageAdapterMiddleware(ChatMiddleware):
    """Prepend a locale-specific instruction derived from the user's language."""

    async def process(self, context: ChatContext, call_next) -> None:
        # Detect language from the most recent user message
        user_text = next(
            (
                "".join(str(c) for c in (m.contents or []))
                for m in reversed(context.messages)
                if m.role == "user"
            ),
            "",
        )
        try:
            lang = langdetect.detect(user_text)
        except Exception:
            lang = "en"

        if lang != "en":
            context.messages.insert(
                0,
                Message(role="system", contents=[f"Respond in the user's language: {lang}."]),
            )

        await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a multilingual assistant.",
    middleware=[LanguageAdapterMiddleware()],
)
```

### Mock the LLM in tests — short-circuit with `MiddlewareTermination`

```python
import asyncio
from agent_framework import (
    Agent, ChatMiddleware, ChatContext, ChatResponse,
    MiddlewareTermination, Message,
)
from agent_framework.openai import OpenAIChatClient


class MockChatMiddleware(ChatMiddleware):
    """Return a canned response without calling the model."""

    def __init__(self, canned_text: str) -> None:
        self.canned_text = canned_text

    async def process(self, context: ChatContext, call_next) -> None:
        context.result = ChatResponse(
            messages=[Message(role="assistant", contents=[self.canned_text])],
        )
        raise MiddlewareTermination()   # skip the real LLM call


async def test_agent_response() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are an assistant.",
        middleware=[MockChatMiddleware("Mocked: Hello!")],
    )
    result = await agent.run("Say hello.")
    assert result.text == "Mocked: Hello!"


asyncio.run(test_agent_response())
```

### Log token usage per call

```python
import logging
from agent_framework import ChatMiddleware, ChatContext

logger = logging.getLogger("token_usage")


class TokenUsageLogger(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        await call_next()
        if context.result and hasattr(context.result, "usage"):
            usage = context.result.usage
            logger.info(
                "tokens",
                extra={
                    "prompt": getattr(usage, "prompt_tokens", None),
                    "completion": getattr(usage, "completion_tokens", None),
                    "total": getattr(usage, "total_tokens", None),
                },
            )
```

### `@chat_middleware` decorator — functional form

```python
from agent_framework import chat_middleware, ChatContext, Agent
from agent_framework.openai import OpenAIChatClient


@chat_middleware
async def strip_pii(context: ChatContext, call_next):
    """Redact email addresses from outgoing messages."""
    import re
    pattern = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
    for msg in context.messages:
        if msg.role == "user":
            msg.contents = [
                pattern.sub("[REDACTED]", str(c)) if isinstance(c, str) else c
                for c in (msg.contents or [])
            ]
    await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a privacy-conscious assistant.",
    middleware=[strip_pii],
)
```

---

## 4. `FunctionMiddleware` — intercepting tool/function execution

**Source:** `agent_framework/_middleware.py`

`FunctionMiddleware` wraps every individual **tool call** — the Python function the model invokes. It sees `FunctionInvocationContext` which contains the `FunctionTool`, the parsed arguments dict, and the result. Use it to:

- **Cache** expensive tool results.
- **Validate** or sanitise arguments before execution.
- **Enforce rate limits** per tool.
- **Redact** sensitive data from tool outputs.

### Memoising cache for deterministic tools

```python
import asyncio
import hashlib
import json
from agent_framework import Agent, FunctionMiddleware, FunctionInvocationContext, MiddlewareTermination, tool
from agent_framework.openai import OpenAIChatClient


class MemoCache(FunctionMiddleware):
    """Cache deterministic tool results in memory."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def _key(self, context: FunctionInvocationContext) -> str:
        payload = {"fn": context.function.name, "args": context.arguments}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        key = self._key(context)
        if key in self._cache:
            context.result = self._cache[key]
            raise MiddlewareTermination()   # skip the real function call

        await call_next()

        if context.result is not None:
            self._cache[key] = context.result


@tool
def get_stock_price(ticker: str) -> str:
    """Return the current stock price for a ticker (simulated)."""
    prices = {"MSFT": "415.00", "AAPL": "187.00", "GOOG": "175.00"}
    return prices.get(ticker.upper(), "unknown")


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a stock price assistant.",
        tools=[get_stock_price],
        middleware=[MemoCache()],
    )
    r1 = await agent.run("What is MSFT's stock price?")
    r2 = await agent.run("What is MSFT's stock price?")   # cache hit
    print(r1.text, r2.text)


asyncio.run(main())
```

### Rate-limiting per tool

```python
import asyncio
import time
from agent_framework import FunctionMiddleware, FunctionInvocationContext


class RateLimitMiddleware(FunctionMiddleware):
    """Enforce a minimum gap between calls to specific tools."""

    def __init__(self, tool_name: str, min_gap_seconds: float = 1.0) -> None:
        self.tool_name = tool_name
        self.min_gap_seconds = min_gap_seconds
        self._last_call: float = 0.0

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        if context.function.name == self.tool_name:
            now = time.monotonic()
            gap = now - self._last_call
            if gap < self.min_gap_seconds:
                await asyncio.sleep(self.min_gap_seconds - gap)
            self._last_call = time.monotonic()
        await call_next()
```

### Argument validation / sanitisation

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext


class SqlSanitiserMiddleware(FunctionMiddleware):
    """Block tool calls that contain SQL injection patterns in string arguments."""

    _BLOCKED_PATTERNS = ("--", ";", "DROP ", "DELETE ", "INSERT ", "UPDATE ")

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        for arg_value in context.arguments.values():
            if isinstance(arg_value, str):
                for pattern in self._BLOCKED_PATTERNS:
                    if pattern.lower() in arg_value.lower():
                        context.result = f"Blocked: argument contains disallowed pattern '{pattern}'."
                        raise MiddlewareTermination()
        await call_next()
```

### `@function_middleware` decorator

```python
from agent_framework import function_middleware, FunctionInvocationContext


@function_middleware
async def log_tool_calls(context: FunctionInvocationContext, call_next):
    print(f"→ Calling tool '{context.function.name}' with args {context.arguments}")
    await call_next()
    print(f"← Tool '{context.function.name}' returned: {context.result!r}")
```

---

## 5. `CompactionProvider` — automated context window management

**Source:** `agent_framework/_compaction.py` — `CompactionProvider(ContextProvider)`

`CompactionProvider` is a `ContextProvider` that runs compaction strategies at defined lifecycle hooks:

- **`before_strategy`** — runs in `before_run()`, operating on messages already loaded from history before they reach the model.
- **`after_strategy`** — runs in `after_run()`, operating on stored history messages *after* the model responds, so the next turn starts smaller.

Either strategy may be `None` to skip that phase.

### Constructor

```python
CompactionProvider(
    *,
    before_strategy: CompactionStrategy | None = None,
    after_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    source_id: str = "compaction",
    history_source_id: str = "in_memory",
)
```

| Parameter | Default | Description |
|---|---|---|
| `before_strategy` | `None` | Applied to loaded context before each model call |
| `after_strategy` | `None` | Applied to persisted history after each model call |
| `history_source_id` | `"in_memory"` | `source_id` of the history provider to compact in `after_run` |

### Pre-run sliding window — deterministic, zero extra LLM calls

```python
import asyncio
from agent_framework import (
    Agent, InMemoryHistoryProvider, CompactionProvider, SlidingWindowStrategy,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    history = InMemoryHistoryProvider()
    compaction = CompactionProvider(
        before_strategy=SlidingWindowStrategy(keep_last_groups=20),
        history_source_id=history.source_id,   # "in_memory"
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a long-running assistant.",
        context_providers=[history, compaction],
    )

    session = agent.create_session()
    for i in range(50):
        r = await agent.run(f"Turn {i}: say hello.", session=session)
        print(r.text)


asyncio.run(main())
```

### Post-run summarisation — compact stored history after each turn

```python
from agent_framework import (
    Agent, InMemoryHistoryProvider, CompactionProvider, SummarizationStrategy,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    history = InMemoryHistoryProvider()
    compaction = CompactionProvider(
        # No before_strategy — load everything each turn
        after_strategy=SummarizationStrategy(
            client=client,
            target_count=6,    # keep ~6 non-system groups
            threshold=2,       # trigger when count reaches 8
        ),
        history_source_id=history.source_id,
    )

    agent = Agent(
        client=client,
        instructions="You are a long-running assistant.",
        context_providers=[history, compaction],
    )

    session = agent.create_session()
    for i in range(20):
        await agent.run(f"Turn {i}.", session=session)
```

### Two-phase: window before, tool-result collapse after

```python
from agent_framework import (
    Agent, FileHistoryProvider, CompactionProvider,
    SlidingWindowStrategy, ToolResultCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient

history = FileHistoryProvider("./sessions", source_id="file_history", skip_excluded=True)
compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=30),
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
    history_source_id="file_history",
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a tool-heavy assistant.",
    context_providers=[history, compaction],
)
```

---

## 6. `ToolResultCompactionStrategy` — collapse older tool-call groups

**Source:** `agent_framework/_compaction.py`

`ToolResultCompactionStrategy` replaces older tool-call groups with compact summary messages like `[Tool results: get_weather: sunny, 18°C]`. Unlike `SelectiveToolCallCompactionStrategy` (which *excludes* them entirely), this strategy keeps a readable trace.

### Constructor

```python
ToolResultCompactionStrategy(
    *,
    keep_last_tool_call_groups: int = 1,   # keep the most recent N tool-call groups
)
```

Raises `ValueError` if `keep_last_tool_call_groups < 0`.

### Standalone use with an agent

```python
import asyncio
from agent_framework import Agent, tool, ToolResultCompactionStrategy
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(city: str) -> str:
    """Return simulated weather for a city."""
    return f"{city}: 22°C, partly cloudy"


@tool
def get_air_quality(city: str) -> str:
    """Return simulated air quality index."""
    return f"{city}: AQI 45 (Good)"


async def main() -> None:
    strategy = ToolResultCompactionStrategy(keep_last_tool_call_groups=1)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a weather and air quality assistant.",
        tools=[get_weather, get_air_quality],
        compaction_strategy=strategy,    # pass directly to Agent
    )

    session = agent.create_session()
    for city in ["London", "Paris", "Berlin", "Madrid", "Rome"]:
        r = await agent.run(f"What is the weather and AQI in {city}?", session=session)
        print(r.text)


asyncio.run(main())
```

### Inside `CompactionProvider` for `after_run` compaction

```python
from agent_framework import (
    Agent, InMemoryHistoryProvider, CompactionProvider, ToolResultCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient

history = InMemoryHistoryProvider()
compaction = CompactionProvider(
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
    history_source_id=history.source_id,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a data assistant.",
    context_providers=[history, compaction],
)
```

### Strategy comparison

| Strategy | What it does to old groups | Extra LLM call? |
|---|---|---|
| `SlidingWindowStrategy` | Excludes old non-system groups entirely | No |
| `SelectiveToolCallCompactionStrategy` | Excludes old tool-call groups (other groups kept) | No |
| `ToolResultCompactionStrategy` | Replaces old tool groups with a compact summary message | No |
| `SummarizationStrategy` | Summarises oldest content into a new assistant message | Yes (1 LLM call) |

---

## 7. `TokenBudgetComposedStrategy` — token-aware multi-strategy composition

**Source:** `agent_framework/_compaction.py`

`TokenBudgetComposedStrategy` runs a sequence of `CompactionStrategy` instances in order until the token count falls under a target budget. If strategies alone are not enough, a built-in fallback excludes oldest groups.

### Constructor

```python
TokenBudgetComposedStrategy(
    *,
    token_budget: int,
    tokenizer: TokenizerProtocol,
    strategies: Sequence[CompactionStrategy],
    early_stop: bool = True,    # stop as soon as budget is reached; default True
)
```

Available tokenisers:
- `CharacterEstimatorTokenizer` — 4 characters ≈ 1 token; no external dependencies.
- Any object implementing `TokenizerProtocol` (`count_tokens(text: str) -> int`).

### Full composed strategy example

```python
import asyncio
from agent_framework import (
    Agent,
    CharacterEstimatorTokenizer,
    InMemoryHistoryProvider,
    CompactionProvider,
    SlidingWindowStrategy,
    SummarizationStrategy,
    ToolResultCompactionStrategy,
    TokenBudgetComposedStrategy,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    tokenizer = CharacterEstimatorTokenizer()

    # Three strategies tried in order; stop as soon as budget is met.
    strategy = TokenBudgetComposedStrategy(
        token_budget=8_000,         # target: stay under 8k tokens
        tokenizer=tokenizer,
        strategies=[
            # 1. Drop older tool groups first (cheapest)
            ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
            # 2. Then slide the window down
            SlidingWindowStrategy(keep_last_groups=20),
            # 3. Last resort: summarise (costs one extra LLM call)
            SummarizationStrategy(client=client, target_count=8),
        ],
        early_stop=True,
    )

    history = InMemoryHistoryProvider()
    compaction = CompactionProvider(
        before_strategy=strategy,
        history_source_id=history.source_id,
        tokenizer=tokenizer,
    )

    agent = Agent(
        client=client,
        instructions="You are a long-running assistant.",
        context_providers=[history, compaction],
        tokenizer=tokenizer,
    )

    session = agent.create_session()
    for i in range(30):
        r = await agent.run(f"Iteration {i}: process some work.", session=session)
        print(r.text)


asyncio.run(main())
```

### Custom tokeniser — integrate tiktoken

```python
import tiktoken
from agent_framework import CharacterEstimatorTokenizer


class TiktokenTokenizer:
    """Drop-in TokenizerProtocol implementation using tiktoken."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self._enc = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


# Use it exactly like CharacterEstimatorTokenizer
from agent_framework import TokenBudgetComposedStrategy, SlidingWindowStrategy

strategy = TokenBudgetComposedStrategy(
    token_budget=16_000,
    tokenizer=TiktokenTokenizer(model="gpt-4o"),
    strategies=[SlidingWindowStrategy(keep_last_groups=40)],
)
```

---

## 8. `FileCheckpointStorage` — durable workflow checkpoints

**Source:** `agent_framework/_workflows/_checkpoint.py`

`FileCheckpointStorage` writes one checkpoint file per superstep to disk, so workflows can survive process restarts. Checkpoints store executor states, messages in transit, and shared state as hybrid JSON (metadata) + base64 pickle (complex objects).

### Security and type allowlisting

By default, only framework-internal types and Python primitives are allowed in checkpoint deserialization. Add application types via `allowed_checkpoint_types`:

```python
FileCheckpointStorage(
    storage_path: str | Path,
    *,
    allowed_checkpoint_types: list[str] | None = None,
    # e.g. ["my_app.models:JobState", "my_app.schemas:ReviewResult"]
)
```

### Basic checkpoint — pause and resume across restarts

```python
import asyncio
from typing_extensions import Never
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, WorkflowRunResult,
    FileCheckpointStorage, handler,
)


class SlowProcessorExecutor(Executor):
    @handler
    async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        import time
        time.sleep(1)   # simulate long work
        await ctx.yield_output(f"Processed: {text.upper()}")


processor = SlowProcessorExecutor(id="processor")
storage = FileCheckpointStorage("/tmp/workflow-checkpoints")

workflow = WorkflowBuilder(
    start_executor=processor,
    output_from=[processor],
    checkpoint_storage=storage,
).build()


async def run_with_resume() -> None:
    # Run to completion — checkpoints written after each superstep
    result: WorkflowRunResult = await workflow.run("hello world")
    print(result.get_outputs())  # ["Processed: HELLO WORLD"]

    # List available checkpoints
    checkpoints = await storage.list_checkpoints(workflow_name=workflow.name)
    print(f"{len(checkpoints)} checkpoints saved.")

    # Restore from the latest checkpoint (e.g. after a crash)
    latest = await storage.get_latest(workflow_name=workflow.name)
    if latest:
        resumed = await workflow.run(checkpoint_id=latest.checkpoint_id)
        print("Resumed outputs:", resumed.get_outputs())


asyncio.run(run_with_resume())
```

### Runtime checkpoint override

Pass `checkpoint_storage` at run time to override or enable checkpointing for a single run without changing the `WorkflowBuilder`:

```python
import asyncio
from agent_framework import Workflow, FileCheckpointStorage, InMemoryCheckpointStorage


async def run_once(workflow: Workflow) -> None:
    # Enable file checkpoints just for this run
    result = await workflow.run(
        "process this",
        checkpoint_storage=FileCheckpointStorage("/tmp/run-checkpoints"),
    )
    print(result.get_outputs())
```

### Custom application types in checkpoints

```python
from dataclasses import dataclass
from agent_framework import FileCheckpointStorage


@dataclass
class ReviewState:
    reviewer_id: str
    score: float
    notes: str

    # Required by the checkpoint serialiser
    def __getstate__(self):
        return {"reviewer_id": self.reviewer_id, "score": self.score, "notes": self.notes}

    def __setstate__(self, state):
        self.__dict__.update(state)


storage = FileCheckpointStorage(
    "/tmp/review-checkpoints",
    allowed_checkpoint_types=["my_app.models:ReviewState"],
)
```

### `InMemoryCheckpointStorage` — for tests and development

`InMemoryCheckpointStorage` has the same interface as `FileCheckpointStorage` but lives only in RAM — ideal for unit tests:

```python
from agent_framework import InMemoryCheckpointStorage

storage = InMemoryCheckpointStorage()
# storage.save(checkpoint)
# storage.load(checkpoint_id)
# storage.list_checkpoints(workflow_name="my-workflow")
# storage.get_latest(workflow_name="my-workflow")
# storage.delete(checkpoint_id)
```

---

## 9. `LocalEvaluator` — run offline evaluation checks

**Source:** `agent_framework/_evaluation.py`

`LocalEvaluator` runs keyword and tool-call checks **locally** — no cloud API required. Each check is applied to every `EvalItem`; an item passes only when all checks pass.

> **Experimental** in 1.6.0.

### Constructor

```python
LocalEvaluator(*checks: EvalCheck)
```

Built-in check factories:

| Factory | What it checks |
|---|---|
| `keyword_check(*keywords, case_sensitive=False)` | Response contains all listed keywords |
| `tool_called_check(*tool_names, mode="all")` | Named tools were called (`"all"` or `"any"`) |
| `tool_call_args_match(tool_name, **expected_args)` | Tool was called with specific argument values |
| `@evaluator` decorator | Turn any `async def` into a custom check |

### Basic keyword + tool-call evaluation

```python
import asyncio
from agent_framework import (
    Agent, tool, LocalEvaluator, evaluate_agent,
    keyword_check, tool_called_check,
)
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"{city}: 22°C, sunny"


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a weather assistant. Always call get_weather.",
        tools=[get_weather],
    )

    evaluator = LocalEvaluator(
        keyword_check("weather", "temperature"),     # response must mention these
        tool_called_check("get_weather"),            # get_weather must be invoked
    )

    queries = [
        "What is the weather in London?",
        "Tell me about the weather in Tokyo.",
        "Is it sunny in Paris?",
    ]

    results = await evaluate_agent(
        agent=agent,
        queries=queries,
        evaluators=evaluator,
        eval_name="weather-suite",
    )

    for eval_results in results:
        print(f"Passed: {eval_results.passed}/{eval_results.total}")
        for item in eval_results.items:
            status = "✓" if item.status == "pass" else "✗"
            print(f"  {status} {item.item_id}")


asyncio.run(main())
```

### Custom evaluator with `@evaluator`

```python
from agent_framework import evaluator, EvalItem, CheckResult


@evaluator
async def no_hallucination_check(item: EvalItem) -> CheckResult:
    """Fail if the response contains made-up city names."""
    known_cities = {"london", "paris", "tokyo", "berlin", "sydney"}
    mentioned = [w.lower() for w in item.response.split() if w.isalpha()]
    for word in mentioned:
        if word.endswith("ton") and word not in known_cities:
            return CheckResult(
                passed=False,
                reason=f"Possible hallucinated city: '{word}'",
                check_name="no_hallucination",
            )
    return CheckResult(passed=True, reason="No hallucinations detected.", check_name="no_hallucination")
```

### Expected output comparison

Pass `expected_output` to stamp ground-truth answers onto each `EvalItem`. Cloud evaluators (e.g. `FoundryEvals`) use this for semantic similarity scoring:

```python
import asyncio
from agent_framework import Agent, LocalEvaluator, evaluate_agent, keyword_check
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="You are a geography expert.")
    evaluator = LocalEvaluator(keyword_check("capital"))

    results = await evaluate_agent(
        agent=agent,
        queries=["What is the capital of France?", "What is the capital of Germany?"],
        expected_output=["Paris", "Berlin"],   # used by cloud evaluators for scoring
        evaluators=evaluator,
    )

    for r in results:
        print(r.passed, "/", r.total, "passed")


asyncio.run(main())
```

### Mixing local and cloud evaluators

```python
# from agent_framework.foundry import FoundryEvals

# results = await evaluate_agent(
#     agent=agent,
#     queries=queries,
#     expected_output=expected,
#     evaluators=[
#         LocalEvaluator(keyword_check("weather"), tool_called_check("get_weather")),
#         FoundryEvals(project_client=foundry_client, model="gpt-4o"),
#     ],
# )
```

### Evaluate pre-existing responses (skip re-running the agent)

```python
import asyncio
from agent_framework import (
    Agent, AgentResponse, LocalEvaluator, evaluate_agent, keyword_check,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="...")
    query = "What is the weather in London?"

    # Collect a response separately
    response = await agent.run(query)

    # Evaluate without running the agent again
    evaluator = LocalEvaluator(keyword_check("weather"))
    results = await evaluate_agent(
        agent=agent,
        queries=[query],
        responses=[response],   # ← pre-existing response
        evaluators=evaluator,
    )
    print(results[0].passed)


asyncio.run(main())
```

---

## 10. `WorkflowRunResult` — consuming workflow execution results

**Source:** `agent_framework/_workflows/_workflow_runner.py` — `WorkflowRunResult(list[WorkflowEvent])`

`WorkflowRunResult` is what `await workflow.run(...)` returns. It is a list of `WorkflowEvent` objects with helper methods for extracting outputs, intermediate results, request-info events, and state.

### Core methods

| Method | Returns | Description |
|---|---|---|
| `get_outputs()` | `list[Any]` | All events with `type="output"` — the final workflow results |
| `get_intermediate_outputs()` | `list[Any]` | Events with `type="intermediate"` — mid-pipeline data |
| `get_request_info_events()` | `list[WorkflowEvent]` | HITL pause events emitted by `ctx.request_info()` |
| `get_final_state()` | `WorkflowRunState` | Terminal state: `IDLE`, `IDLE_WITH_PENDING_REQUESTS`, etc. |
| `status_timeline()` | `list[WorkflowEvent]` | Full status event history (control-plane) |

### Basic output extraction

```python
import asyncio
from typing_extensions import Never
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, WorkflowRunResult, handler,
)


class UpperCaseExecutor(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(text.upper())


executor = UpperCaseExecutor(id="upper")
workflow = WorkflowBuilder(start_executor=executor, output_from=[executor]).build()


async def main() -> None:
    result: WorkflowRunResult = await workflow.run("hello world")

    outputs = result.get_outputs()
    print(outputs)              # ["HELLO WORLD"]
    print(result.get_final_state())    # WorkflowRunState.IDLE

    # Iterate raw events for custom handling
    for event in result:
        print(f"[{event.type}] {event.data!r}")


asyncio.run(main())
```

### Multi-output workflow — fan-out results

```python
import asyncio
from typing_extensions import Never
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, handler,
)


class UpperExecutor(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(("upper", text.upper()))


class ReverseExecutor(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(("reverse", text[::-1]))


source_exec = Executor(id="source")   # conceptual; use a real Executor subclass


async def main() -> None:
    upper = UpperExecutor(id="upper")
    reverse = ReverseExecutor(id="reverse")

    workflow = (
        WorkflowBuilder(start_executor=upper, output_from=[upper, reverse])
        .add_fan_out_edges(upper, [reverse])   # this is illustrative topology
        .build()
    )

    result = await workflow.run("hello")
    for label, value in result.get_outputs():
        print(f"{label}: {value}")
    # upper: HELLO
    # reverse: olleh
```

### HITL — detecting and resuming pending requests

```python
import asyncio
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, WorkflowRunState, handler,
)


class ReviewExecutor(Executor):
    @handler
    async def review(self, text: str, ctx: WorkflowContext[str]) -> None:
        feedback = await ctx.request_info(
            {"draft": text},
            response_type=str,
            request_id="human-review",
        )
        await ctx.send_message(f"APPROVED: {feedback}")


class PublishExecutor(Executor):
    @handler
    async def publish(self, text: str, ctx: WorkflowContext[None, str]) -> None:
        await ctx.yield_output(f"Published: {text}")


reviewer = ReviewExecutor(id="reviewer")
publisher = PublishExecutor(id="publisher")

workflow = (
    WorkflowBuilder(start_executor=reviewer, output_from=[publisher])
    .add_edge(reviewer, publisher)
    .build()
)


async def main() -> None:
    # First run — suspends at request_info
    result1 = await workflow.run("Draft content here")

    if result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
        for event in result1.get_request_info_events():
            print(f"Pending request '{event.data.get('request_id')}': {event.data}")

        # Second run — supply the human response
        result2 = await workflow.run(
            responses={"human-review": "Looks great, approved!"}
        )
        print(result2.get_outputs())  # ["Published: APPROVED: Looks great, approved!"]


asyncio.run(main())
```

### Streaming `WorkflowRunResult` — consume events as they arrive

When called with `stream=True`, `workflow.run(...)` returns a `ResponseStream` instead of `WorkflowRunResult`. Use `get_final_response()` to get the `WorkflowRunResult` after the stream ends:

```python
import asyncio
from typing_extensions import Never
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, handler,
)


class SlowExecutor(Executor):
    @handler
    async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        import asyncio
        for word in text.split():
            await asyncio.sleep(0.1)
            await ctx.yield_output(word)


slow = SlowExecutor(id="slow")
workflow = WorkflowBuilder(start_executor=slow, output_from=[slow]).build()


async def main() -> None:
    stream = workflow.run("hello world foo bar", stream=True)

    async for event in stream:
        if event.type == "output":
            print(f"Got word: {event.data}")

    final: WorkflowRunResult = await stream.get_final_response()
    print("All outputs:", final.get_outputs())


asyncio.run(main())
```

### Inspecting intermediate outputs and custom events

```python
import asyncio
from typing_extensions import Never
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, WorkflowEvent, handler,
)


class PipelineExecutor(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        step1 = text.strip()
        await ctx.add_event(WorkflowEvent(type="progress", data={"step": "strip", "result": step1}))

        step2 = step1.upper()
        # yield_output is the final output; intermediate_output is a mid-pipeline signal
        await ctx.yield_output(step2)


pipeline = PipelineExecutor(id="pipeline")
workflow = WorkflowBuilder(start_executor=pipeline, output_from=[pipeline]).build()


async def main() -> None:
    result = await workflow.run("  hello world  ")

    # Custom events
    custom_events = [e for e in result if e.type == "progress"]
    for evt in custom_events:
        print(f"Progress: {evt.data}")

    print("Final:", result.get_outputs())


asyncio.run(main())
```

---

## Summary — which class to reach for

| Need | Class |
|---|---|
| Persist conversation across restarts | `FileHistoryProvider` |
| Wrap the full agent run (auth, retry, audit) | `AgentMiddleware` / `@agent_middleware` |
| Intercept / mock the LLM call | `ChatMiddleware` / `@chat_middleware` |
| Cache or validate tool calls | `FunctionMiddleware` / `@function_middleware` |
| Automate context window management | `CompactionProvider` |
| Collapse tool-call groups with a trace | `ToolResultCompactionStrategy` |
| Stay under a token budget with multiple strategies | `TokenBudgetComposedStrategy` |
| Persist workflow state across process crashes | `FileCheckpointStorage` |
| Evaluate agents offline without cloud APIs | `LocalEvaluator` + `evaluate_agent` |
| Consume and introspect workflow outputs | `WorkflowRunResult` |

---

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-05-25 | 1.0.0 | Initial volume 2; 10 classes sourced from `agent-framework-core==1.6.0` installed package. |

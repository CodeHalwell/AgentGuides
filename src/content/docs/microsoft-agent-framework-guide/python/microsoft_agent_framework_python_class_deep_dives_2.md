---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 2"
description: "Source-verified deep dives into AgentMiddleware, SummarizationStrategy, TokenBudgetComposedStrategy, CompactionProvider, LocalEvaluator, FileHistoryProvider, SkillsProvider, TodoProvider, AgentModeProvider, and FileCheckpointStorage — all verified against agent-framework-core==1.6.0."
framework: microsoft-agent-framework
language: python
---

# Class Deep Dives Vol. 2 — Python

Ten classes examined directly from the installed source (`agent-framework-core==1.6.0`), with verified constructor signatures, key docstring extracts, and runnable code examples.

Classes covered: `AgentMiddleware`, `SummarizationStrategy`, `TokenBudgetComposedStrategy`, `CompactionProvider`, `LocalEvaluator`, `FileHistoryProvider`, `SkillsProvider`, `TodoProvider`, `AgentModeProvider`, `FileCheckpointStorage`.

---

## 1. `AgentMiddleware`

**Source:** `agent_framework/_middleware.py`

`AgentMiddleware` is the abstract base class for intercepting **every agent invocation**. Subclass it, implement `process()`, and attach instances to `Agent(middleware=[...])`. All three middleware levels share the same `call_next` coroutine pattern, but `AgentMiddleware` gives you the widest view: raw messages, session, tool list, streaming flag, and the final result.

### Constructor

```python
# No constructor — it's an abstract base class. Subclass and implement process().
class AgentMiddleware(ABC):
    @abstractmethod
    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None: ...
```

### `AgentContext` fields

| Field | Type | Notes |
|---|---|---|
| `agent` | `SupportsAgentRun` | The agent being invoked |
| `messages` | `list[Message]` | Input messages — mutable |
| `session` | `AgentSession \| None` | Session for this run |
| `tools` | `ToolTypes \| ...` | Run-level tool overrides |
| `options` | `Mapping[str, Any] \| None` | Model/client kwargs dict |
| `stream` | `bool` | Whether streaming is requested |
| `compaction_strategy` | `CompactionStrategy \| None` | Per-run compaction override |
| `tokenizer` | `TokenizerProtocol \| None` | Per-run tokenizer override |
| `metadata` | `dict[str, Any]` | Shared dict for passing data between middlewares |
| `result` | `AgentResponse \| ResponseStream \| None` | Set this to short-circuit; read after `call_next()` |
| `kwargs` | `dict[str, Any]` | Legacy runtime keyword arguments |
| `client_kwargs` | `dict[str, Any]` | Forwarded to the chat client |
| `function_invocation_kwargs` | `dict[str, Any]` | Forwarded to tool invocations |
| `stream_transform_hooks` | `list[Callable]` | Per-update transform hooks |
| `stream_result_hooks` | `list[Callable]` | Applied to final streaming result |
| `stream_cleanup_hooks` | `list[Callable]` | Run after stream cleanup |

### Example 1 — Latency logger

```python
import time
from agent_framework import AgentMiddleware, AgentContext, Agent
from agent_framework.openai import OpenAIChatClient


class LatencyMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next) -> None:
        t0 = time.perf_counter()
        await call_next()
        elapsed = time.perf_counter() - t0
        print(f"[{context.agent.name}] {elapsed * 1000:.1f} ms | stream={context.stream}")


agent = Agent(
    client=OpenAIChatClient(),
    name="assistant",
    middleware=[LatencyMiddleware()],
)
```

### Example 2 — Short-circuit cache

Reply instantly from an in-memory cache before any LLM call is made:

```python
from agent_framework import AgentMiddleware, AgentContext


class CacheMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    async def process(self, context: AgentContext, call_next) -> None:
        # Build a simple cache key from the last user message.
        last_user = next(
            (m for m in reversed(context.messages) if m.role == "user"), None
        )
        if last_user is None:
            await call_next()
            return

        key = str(last_user.contents)
        if key in self._cache:
            from agent_framework import AgentResponse, Message
            # Construct a synthetic result and skip call_next entirely.
            context.result = AgentResponse(
                messages=[Message(role="assistant", contents=[self._cache[key]])]
            )
            return

        await call_next()

        if context.result:
            context.metadata["cache_key"] = key
            self._cache[key] = context.result.text or ""
```

### Example 3 — Retry on transient errors

```python
import asyncio
from agent_framework import AgentMiddleware, AgentContext


class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.5) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def process(self, context: AgentContext, call_next) -> None:
        for attempt in range(1, self.max_attempts + 1):
            try:
                await call_next()
                return
            except Exception as exc:
                if attempt == self.max_attempts:
                    raise
                wait = self.base_delay * (2 ** (attempt - 1))
                print(f"Attempt {attempt} failed ({exc}). Retrying in {wait:.1f}s…")
                await asyncio.sleep(wait)
```

### Example 4 — Metadata bus between middleware layers

Multiple middleware layers can communicate through `context.metadata`:

```python
from agent_framework import AgentMiddleware, AgentContext, Agent


class TaggerMiddleware(AgentMiddleware):
    """Tag the invocation with request metadata."""
    async def process(self, context: AgentContext, call_next) -> None:
        context.metadata["user_id"] = "u-42"
        context.metadata["request_id"] = "req-abc123"
        await call_next()


class AuditMiddleware(AgentMiddleware):
    """Read tags written by an earlier middleware."""
    async def process(self, context: AgentContext, call_next) -> None:
        await call_next()
        user_id = context.metadata.get("user_id", "unknown")
        req_id = context.metadata.get("request_id", "unknown")
        print(f"Audit: user={user_id} req={req_id} result_len={len(context.result.text or '')}")


agent = Agent(
    client=...,
    middleware=[TaggerMiddleware(), AuditMiddleware()],
)
```

---

## 2. `SummarizationStrategy`

**Source:** `agent_framework/_compaction.py`

`SummarizationStrategy` is an LLM-based compaction strategy that **summarizes older conversation groups and injects the summary in-place**, annotating both directions: summary → original message IDs, and original → summary ID. It fires when the included non-system message count exceeds `target_count + threshold`.

### Constructor

```python
SummarizationStrategy(
    *,
    client: SupportsChatGetResponse[Any],
    target_count: int = 4,
    threshold: int | None = 2,
    prompt: str | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `client` | *(required)* | Chat client used to generate the summary |
| `target_count` | `4` | Target number of included non-system messages to retain after summarization. Must be > 0 |
| `threshold` | `2` | Extra messages allowed before triggering. Set to `None` for threshold of 0 |
| `prompt` | `None` | Summarization instruction; defaults to a built-in prompt that preserves goals, decisions, and unresolved items |

**Failure handling:** If the LLM call raises or returns empty text, the strategy logs a warning and returns `False` without mutating messages — the conversation continues untouched.

### Example 1 — Basic setup with CompactionProvider

```python
import asyncio
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import SummarizationStrategy
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    summarizer = SummarizationStrategy(
        client=client,
        target_count=6,   # keep ~6 recent messages
        threshold=3,      # trigger when count > 9
    )

    history = InMemoryHistoryProvider()
    compaction = CompactionProvider(
        # Compact stored history after each turn so the NEXT turn starts smaller.
        after_strategy=summarizer,
        history_source_id=history.source_id,
    )

    agent = Agent(
        client=client,
        name="researcher",
        context_providers=[history, compaction],
    )
    session = agent.create_session()

    # Run a long conversation — summarization fires automatically.
    for _ in range(20):
        response = await agent.run("Tell me more about quantum computing.", session=session)
        print(response.text[:80])
```

### Example 2 — Custom summarization prompt for domain-specific summaries

```python
LEGAL_SUMMARY_PROMPT = """
You are a legal case summarizer. Summarize the conversation below, retaining:
- All cited statutes and case precedents
- Every agreed-upon factual point
- Open disputed points and which party holds which position
- Any deadlines or procedural dates mentioned

Be concise but complete. Do not interpret — only document.
"""

from agent_framework._compaction import SummarizationStrategy
from agent_framework.openai import OpenAIChatClient

legal_summarizer = SummarizationStrategy(
    client=OpenAIChatClient(),
    target_count=8,
    prompt=LEGAL_SUMMARY_PROMPT,
)
```

### Example 3 — Inspecting summary metadata

The strategy stores `_summary_of_message_ids`, `_summary_of_group_ids`, and `_summarized_by_summary_id` in `Message.additional_properties`. Use these to trace which original messages a summary replaced:

```python
from agent_framework._compaction import (
    SUMMARY_OF_MESSAGE_IDS_KEY,
    SUMMARY_OF_GROUP_IDS_KEY,
    SUMMARIZED_BY_SUMMARY_ID_KEY,
)

# After running agent with SummarizationStrategy...
for msg in session.state.get("in_memory", {}).get("messages", []):
    props = msg.additional_properties or {}
    if SUMMARY_OF_MESSAGE_IDS_KEY in props:
        print(f"Summary message {msg.message_id} covers {props[SUMMARY_OF_MESSAGE_IDS_KEY]}")
    if SUMMARIZED_BY_SUMMARY_ID_KEY in props:
        print(f"Message {msg.message_id} summarized by {props[SUMMARIZED_BY_SUMMARY_ID_KEY]}")
```

---

## 3. `TokenBudgetComposedStrategy`

**Source:** `agent_framework/_compaction.py`

`TokenBudgetComposedStrategy` wraps a sequence of other strategies and runs them in order, re-checking the token budget after each one. If none of them reach the budget, a **deterministic fallback** excludes oldest non-system groups one at a time until the budget is met. If even system messages push you over budget, it excludes those too (strict fallback).

### Constructor

```python
TokenBudgetComposedStrategy(
    *,
    token_budget: int,
    tokenizer: TokenizerProtocol,
    strategies: Sequence[CompactionStrategy],
    early_stop: bool = True,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `token_budget` | *(required)* | Maximum included token count after compaction |
| `tokenizer` | *(required)* | Implements `count_tokens(text: str) -> int` |
| `strategies` | *(required)* | Ordered strategy list; runs before the deterministic fallback |
| `early_stop` | `True` | Stop after the first strategy that brings tokens under budget |

**Built-in tokenizer:** `CharacterEstimatorTokenizer` (4 chars ≈ 1 token) is available in `agent_framework._compaction` as a fast heuristic.

### Example 1 — Summarize first, then window, with a 16 k token hard limit

```python
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import (
    CharacterEstimatorTokenizer,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

strategy = TokenBudgetComposedStrategy(
    token_budget=16_000,
    tokenizer=CharacterEstimatorTokenizer(),
    strategies=[
        SummarizationStrategy(client=client, target_count=10, threshold=4),
        SlidingWindowStrategy(keep_last_groups=15),
    ],
)

history = InMemoryHistoryProvider()
compaction = CompactionProvider(
    before_strategy=strategy,
    history_source_id=history.source_id,
)
agent = Agent(client=client, context_providers=[history, compaction])
```

### Example 2 — Custom tokenizer backed by tiktoken

```python
from agent_framework._compaction import TokenizerProtocol

try:
    import tiktoken

    class TiktokenTokenizer:
        """Accurate tiktoken-based token counter for GPT models."""

        def __init__(self, model: str = "gpt-4o") -> None:
            self._enc = tiktoken.encoding_for_model(model)

        def count_tokens(self, text: str) -> int:
            return len(self._enc.encode(text))

except ImportError:
    from agent_framework._compaction import CharacterEstimatorTokenizer as TiktokenTokenizer  # type: ignore[misc]


from agent_framework._compaction import SlidingWindowStrategy, TokenBudgetComposedStrategy

precise_strategy = TokenBudgetComposedStrategy(
    token_budget=32_000,
    tokenizer=TiktokenTokenizer(),
    strategies=[SlidingWindowStrategy(keep_last_groups=30)],
    early_stop=True,
)
```

### Example 3 — `early_stop=False` to run all strategies regardless

```python
from agent_framework._compaction import (
    CharacterEstimatorTokenizer,
    SelectiveToolCallCompactionStrategy,
    ToolResultCompactionStrategy,
    TokenBudgetComposedStrategy,
)

# Run ALL strategies unconditionally for maximum compaction,
# then enforce the hard token limit with the deterministic fallback.
aggressive = TokenBudgetComposedStrategy(
    token_budget=8_000,
    tokenizer=CharacterEstimatorTokenizer(),
    strategies=[
        ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
        SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2),
    ],
    early_stop=False,   # run both even if the first one is enough
)
```

---

## 4. `CompactionProvider`

**Source:** `agent_framework/_compaction.py`

`CompactionProvider` is a `ContextProvider` that runs compaction **at two points**: `before_run` (on messages already loaded into context, before the model sees them) and `after_run` (on stored history messages, before the next turn loads them). Either phase can be `None` to skip.

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

| Parameter | Default | Notes |
|---|---|---|
| `before_strategy` | `None` | Applied to loaded context messages before the model runs |
| `after_strategy` | `None` | Applied to stored history messages after the model runs — **must match** the history provider's `source_id` |
| `tokenizer` | `None` | Token-aware strategies require this |
| `source_id` | `"compaction"` | Provider identity for session state |
| `history_source_id` | `"in_memory"` | Must match the `source_id` of your `HistoryProvider` |

**Key constraint:** If your history provider has a custom `source_id`, you must pass it here. Mismatches mean `after_strategy` silently finds no messages to compact.

### Example 1 — Sliding window before each turn

Trim context to the last 20 groups before sending to the model. Does not touch stored history:

```python
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import SlidingWindowStrategy

history = InMemoryHistoryProvider()
compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    history_source_id=history.source_id,  # "in_memory" by default
)
agent = Agent(client=..., context_providers=[history, compaction])
```

### Example 2 — Custom history `source_id` must match

```python
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import SlidingWindowStrategy, ToolResultCompactionStrategy

# Use a non-default source_id on the history provider.
history = InMemoryHistoryProvider(source_id="chat_history")

compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
    history_source_id="chat_history",   # ← must match history.source_id
)

agent = Agent(client=..., context_providers=[history, compaction])
```

### Example 3 — Two-provider pipeline: trim before, summarize after

```python
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import (
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    CharacterEstimatorTokenizer,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
history = InMemoryHistoryProvider()

compaction = CompactionProvider(
    # Before the model: enforce a hard token cap using a composed strategy.
    before_strategy=TokenBudgetComposedStrategy(
        token_budget=24_000,
        tokenizer=CharacterEstimatorTokenizer(),
        strategies=[SlidingWindowStrategy(keep_last_groups=25)],
    ),
    # After the model: summarize to prevent stored history from bloating over time.
    after_strategy=SummarizationStrategy(client=client, target_count=8, threshold=4),
    history_source_id=history.source_id,
)

agent = Agent(
    client=client,
    name="long-session-assistant",
    context_providers=[history, compaction],
)
```

---

## 5. `LocalEvaluator`

**Source:** `agent_framework/_evaluation.py`

`LocalEvaluator` runs structured evaluation checks against a batch of `EvalItem` instances. Pass checks as **positional variadic arguments** (not a list). Each check is an `EvalCheck` — a callable that receives an `EvalItem` and returns a `CheckResult`. Built-in checks cover exact match, contains, regex, and LLM-graded rubrics.

### Constructor

```python
LocalEvaluator(*checks: EvalCheck)
```

```python
# Core methods
LocalEvaluator.evaluate(
    self,
    items: Sequence[EvalItem],
    *,
    eval_name: str = "Local Eval",
) -> EvalResults
```

### Example 1 — Simple exact-match evaluation

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.evaluation import (
    LocalEvaluator,
    EvalItem,
    evaluate_agent,
    exact_match,
    contains_text,
)

client = OpenAIChatClient()
agent = Agent(client=client, name="qa-agent")

evaluator = LocalEvaluator(
    exact_match(field="output"),
    contains_text(field="output", text="Paris"),
)

results = await evaluate_agent(
    agent=agent,
    queries=["What is the capital of France?"],
    expected_output=["Paris"],
    evaluators=[evaluator],
    eval_name="capital-cities",
)

print(results[0].summary())
```

### Example 2 — Custom `EvalCheck` from scratch

```python
from agent_framework.evaluation import EvalItem, CheckResult, LocalEvaluator


def word_count_check(min_words: int, max_words: int):
    """Verify the agent's output falls within an expected word-count range."""
    def check(item: EvalItem) -> CheckResult:
        output = item.output or ""
        count = len(output.split())
        passed = min_words <= count <= max_words
        return CheckResult(
            name=f"word_count({min_words}–{max_words})",
            passed=passed,
            message=f"Word count: {count}",
        )
    return check


evaluator = LocalEvaluator(
    word_count_check(min_words=50, max_words=300),
)
```

### Example 3 — Running `evaluate_agent` across multiple test cases

```python
from agent_framework.evaluation import LocalEvaluator, evaluate_agent, contains_text
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), name="summarizer")

evaluator = LocalEvaluator(
    contains_text(field="output", text="summary"),
)

eval_results_list = await evaluate_agent(
    agent=agent,
    queries=[
        "Summarise the French Revolution in one sentence.",
        "Summarise the causes of World War I in one sentence.",
    ],
    expected_output=[
        "A summary of the French Revolution",
        "A summary of WWI causes",
    ],
    evaluators=[evaluator],
    eval_name="summarizer-eval",
    num_repetitions=3,          # run each query 3 times, average pass rates
)

for results in eval_results_list:
    print(results.eval_name, results.pass_rate)
```

### Example 4 — Evaluating a workflow

Use `evaluate_workflow` when your agent is packaged as a `Workflow` rather than a bare `Agent`:

```python
from agent_framework.evaluation import LocalEvaluator, evaluate_workflow, exact_match
from agent_framework import WorkflowBuilder

workflow = WorkflowBuilder(start_executor=my_executor).build()

evaluator = LocalEvaluator(exact_match(field="output"))

results = await evaluate_workflow(
    workflow=workflow,
    queries=["Compute 2 + 2"],
    expected_output=["4"],
    evaluators=[evaluator],
    eval_name="arithmetic-check",
)
```

---

## 6. `FileHistoryProvider`

**Source:** `agent_framework/_sessions.py`

`FileHistoryProvider` persists conversation history to a JSON file on disk. Each session gets its own file keyed by session ID. Use `dumps`/`loads` hooks to encrypt or transform the JSON payload on write/read.

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
    dumps: JsonDumps | None = None,
    loads: JsonLoads | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `storage_path` | *(required)* | Directory where session `.json` files are written |
| `source_id` | `"file_history"` | Provider identity; used as the key in `session.state` |
| `load_messages` | `True` | Load stored messages at the start of each turn |
| `store_inputs` | `True` | Persist user-input messages |
| `store_context_messages` | `False` | Also persist system/context injections |
| `store_context_from` | `None` | Limit `store_context_messages` to specific provider source IDs |
| `store_outputs` | `True` | Persist assistant output messages |
| `skip_excluded` | `False` | Skip compaction-excluded messages when storing |
| `dumps` | `None` | Custom JSON serializer — receive `dict`, return `str` |
| `loads` | `None` | Custom JSON deserializer — receive `str`, return `dict` |

### Example 1 — Basic file-backed session

```python
import asyncio
from agent_framework import Agent, AgentSession, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    history = FileHistoryProvider(storage_path="./sessions")

    agent = Agent(
        client=OpenAIChatClient(),
        name="assistant",
        context_providers=[history],
    )

    # Resume by loading the same session ID on subsequent runs.
    session_id = "user-42-session"
    session = AgentSession(session_id=session_id)

    response = await agent.run("Hello, remember me?", session=session)
    print(response.text)

asyncio.run(main())
```

### Example 2 — Encrypting history with custom `dumps`/`loads`

```python
import json
import base64
from cryptography.fernet import Fernet
from agent_framework import FileHistoryProvider

KEY = Fernet.generate_key()
fernet = Fernet(KEY)


def encrypted_dumps(data: dict) -> str:
    plaintext = json.dumps(data).encode()
    return base64.b64encode(fernet.encrypt(plaintext)).decode()


def encrypted_loads(raw: str) -> dict:
    ciphertext = base64.b64decode(raw.encode())
    return json.loads(fernet.decrypt(ciphertext).decode())


history = FileHistoryProvider(
    storage_path="./encrypted_sessions",
    dumps=encrypted_dumps,
    loads=encrypted_loads,
)
```

### Example 3 — Storing context provider messages for audit trails

```python
from agent_framework import FileHistoryProvider

# Store everything — including system messages injected by context providers.
history = FileHistoryProvider(
    storage_path="./audit_sessions",
    store_context_messages=True,
    # Only capture messages from the skills and todo providers, not all providers.
    store_context_from={"agent_skills", "todo"},
)
```

### Example 4 — Skipping compaction-excluded messages

When you pair `FileHistoryProvider` with a `CompactionProvider`, old summarized messages remain in the stored list but are flagged as excluded. Set `skip_excluded=True` to write only the messages the model actually sees:

```python
from agent_framework import Agent, CompactionProvider, FileHistoryProvider
from agent_framework._compaction import SlidingWindowStrategy

history = FileHistoryProvider(
    storage_path="./compact_sessions",
    skip_excluded=True,   # only store non-excluded messages
)
compaction = CompactionProvider(
    after_strategy=SlidingWindowStrategy(keep_last_groups=20),
    history_source_id=history.source_id,  # "file_history"
)

agent = Agent(client=..., context_providers=[history, compaction])
```

---

## 7. `SkillsProvider`

**Source:** `agent_framework/_skills.py`

`SkillsProvider` follows the [agentskills.io](https://agentskills.io) **progressive-disclosure** pattern: advertise → load → read resources. The agent first sees skill names and short descriptions (~100 tokens each), then requests the full body only when needed, and can access supplementary resources file-by-file. This keeps every-turn system prompt cost low while giving the agent access to rich, structured instructions.

### Constructor

```python
SkillsProvider(
    source: SkillsSource | Sequence[Skill] | Skill,
    *,
    instruction_template: str | None = None,
    require_script_approval: bool = False,
    disable_caching: bool = False,
    source_id: str | None = None,
)
```

### `from_paths` class method

```python
SkillsProvider.from_paths(
    skill_paths: str | Path | Sequence[str | Path],
    *,
    script_runner: SkillScriptRunner | None = None,
    resource_extensions: tuple[str, ...] | None = None,    # default: .md .json .yaml .yml .csv .xml .txt
    script_extensions: tuple[str, ...] | None = None,      # default: .py
    resource_directories: Sequence[str] | None = None,     # default: ("references", "assets")
    script_directories: Sequence[str] | None = None,       # default: ("scripts",)
    instruction_template: str | None = None,
    require_script_approval: bool = False,
    disable_caching: bool = False,
    source_id: str | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `source` | *(required)* | `SkillsSource`, a single `Skill`, or a `Sequence[Skill]`. Strings raise `TypeError` — use `from_paths` instead |
| `instruction_template` | `None` | Custom prompt template; must contain `{skills}` placeholder |
| `require_script_approval` | `False` | If `True`, script execution triggers a `function_approval_request` before running |
| `disable_caching` | `False` | Re-read source on every invocation — pick up file changes in dev loops |

### Example 1 — Code-defined inline skills

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework import InlineSkill, SkillsProvider


async def main() -> None:
    sql_skill = InlineSkill(
        name="sql-query-builder",
        description="Helps write efficient SQL queries for our Postgres schema.",
        instructions="""
# SQL Query Builder

Always:
- Use parameterised queries ($1, $2, …)
- Qualify table names with the schema: `public.`
- Avoid SELECT *; list columns explicitly

When asked for a query, return it inside a fenced ```sql block.
""",
    )

    provider = SkillsProvider(sql_skill)

    agent = Agent(
        client=OpenAIChatClient(),
        name="db-assistant",
        context_providers=[provider],
    )

    response = await agent.run("Write a query to find the top 5 customers by revenue.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — File-based skills with script approval

```python
from agent_framework import Agent, SkillsProvider
from agent_framework.openai import OpenAIChatClient

provider = SkillsProvider.from_paths(
    "./skills",                         # directory containing SKILL.md files
    require_script_approval=True,       # user must approve before scripts run
)

agent = Agent(
    client=OpenAIChatClient(),
    name="ops-agent",
    context_providers=[provider],
)

# Approval loop: surface function_approval_request to the user.
session = agent.create_session()
response = await agent.run("Run the deployment script for staging.", session=session)

for req in response.user_input_requests:
    if req.type == "function_approval_request":
        print(f"Approve script: {req.data.name}? (y/n) ", end="")
        answer = input().strip().lower()
        approval_response = req.data.to_function_approval_response(approved=(answer == "y"))
        response = await agent.run(approval_response, session=session)
```

### Example 3 — Hot-reload during development

```python
from agent_framework import SkillsProvider

# Disable caching so the agent picks up every edit to SKILL.md immediately.
provider = SkillsProvider.from_paths("./skills", disable_caching=True)
```

### Example 4 — Composing multiple skill sources

```python
from agent_framework import InlineSkill, SkillsProvider
from agent_framework._skills import (
    AggregatingSkillsSource,
    FileSkillsSource,
    DeduplicatingSkillsSource,
    InMemorySkillsSource,
)

# Code-defined skills: always present.
core_skills = InMemorySkillsSource([
    InlineSkill(name="date-formatter", description="Format dates per locale.", instructions="…"),
])

# File-based skills: loaded from disk, refreshed on each agent run.
file_skills = FileSkillsSource("./domain_skills")

provider = SkillsProvider(
    DeduplicatingSkillsSource(
        AggregatingSkillsSource([core_skills, file_skills])
    ),
    disable_caching=True,
)
```

---

## 8. `TodoProvider`

**Source:** `agent_framework/_harness/_todo.py`

`TodoProvider` gives an agent five tools for autonomous task management: `add_todos`, `complete_todos`, `remove_todos`, `get_remaining_todos`, `get_all_todos`. By default, state lives in `AgentSession.state` under `source_id`. Pass a `TodoFileStore` for cross-process persistence.

### Constructor

```python
TodoProvider(
    source_id: str = "todo",
    *,
    instructions: str | None = None,
    store: TodoStore | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `source_id` | `"todo"` | Key in `session.state` for todo list storage |
| `instructions` | `None` | Custom instructions injected into the system prompt |
| `store` | `None` | Defaults to `TodoSessionStore` (in-memory per session). Use `TodoFileStore` for disk persistence |

**Exposed tools:**

| Tool | Description |
|---|---|
| `add_todos` | Add one or more todos with a title and optional description |
| `complete_todos` | Mark todos as done by ID |
| `remove_todos` | Delete todos by ID |
| `get_remaining_todos` | List only incomplete todos |
| `get_all_todos` | List all todos (complete and incomplete) |

### Example 1 — Basic autonomous task tracking

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._todo import TodoProvider


async def main() -> None:
    provider = TodoProvider()

    agent = Agent(
        client=OpenAIChatClient(),
        name="planner",
        context_providers=[provider],
        instructions="You are a planning assistant. Track your progress using the todo tools.",
    )

    session = agent.create_session()
    response = await agent.run(
        "Plan and execute a research report on renewable energy. "
        "Add todos for each step and complete them as you go.",
        session=session,
    )
    print(response.text)

asyncio.run(main())
```

### Example 2 — Inspecting the todo list after a run

```python
import json

# After agent.run(...)
todo_state = session.state.get("todo", {})
items = todo_state.get("items", [])
print(json.dumps(items, indent=2))
# [{"id": 1, "title": "Research energy sources", "is_complete": true}, ...]
```

### Example 3 — Persistent todos with `TodoFileStore`

```python
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._todo import TodoProvider, TodoFileStore

# Todos survive process restarts.
store = TodoFileStore(storage_path="./todos")
provider = TodoProvider(store=store)

agent = Agent(client=OpenAIChatClient(), context_providers=[provider])
session = AgentSession(session_id="project-alpha")

response = await agent.run("Continue from where we left off.", session=session)
```

### Example 4 — Custom instructions

```python
from agent_framework._harness._todo import TodoProvider

CUSTOM_INSTRUCTIONS = """
You have access to a todo list. Use it like a Kanban board:
- Add todos BEFORE starting any task
- Complete todos IMMEDIATELY after finishing each step
- If a task is blocked, add it as a new todo with a "BLOCKED:" prefix
"""

provider = TodoProvider(instructions=CUSTOM_INSTRUCTIONS)
```

---

## 9. `AgentModeProvider`

**Source:** `agent_framework/_harness/_mode.py`

`AgentModeProvider` tracks the agent's **operating mode** in `AgentSession.state` and provides two tools (`set_mode`, `get_mode`) the agent can call to switch modes. The default modes are `"plan"` (interactive planning) and `"execute"` (autonomous execution). Mode is case-normalized and the first configured mode is used as the default.

### Constructor

```python
AgentModeProvider(
    source_id: str = "agent_mode",
    *,
    default_mode: str | None = None,
    mode_descriptions: Mapping[str, str] | None = None,
    instructions: str | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `source_id` | `"agent_mode"` | Key in `session.state` |
| `default_mode` | `None` | Initial mode. Defaults to the first key in `mode_descriptions` |
| `mode_descriptions` | `None` | `{"mode_name": "description", ...}`. Defaults to `{"plan": "…", "execute": "…"}` |
| `instructions` | `None` | Custom template; supports `{available_modes}` and `{current_mode}` placeholders |

**External mode switching:** Use the `set_agent_mode(session, mode, source_id=...)` and `get_agent_mode(session, source_id=...)` helpers to read/write mode from application code. When you change mode externally, the provider injects a user-role notification message on the next turn so the agent is aware of the change.

**Persistence note:** Mode is stored in `AgentSession.state` — it persists within a session but does **not** survive process restarts unless you pair the session with a `FileHistoryProvider` that captures state.

### Example 1 — Default plan/execute mode with session persistence

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._mode import AgentModeProvider


async def main() -> None:
    mode_provider = AgentModeProvider()

    agent = Agent(
        client=OpenAIChatClient(),
        name="planning-agent",
        context_providers=[mode_provider],
        instructions="You are an autonomous agent. Start in plan mode, then switch to execute.",
    )

    session = agent.create_session()
    response = await agent.run(
        "Write and execute a plan to analyse last quarter's sales data.",
        session=session,
    )
    print(response.text)

asyncio.run(main())
```

### Example 2 — Custom modes

```python
from agent_framework._harness._mode import AgentModeProvider

provider = AgentModeProvider(
    mode_descriptions={
        "research": "Gather information, ask clarifying questions, do not take actions.",
        "draft": "Write a first-pass document based on gathered information.",
        "review": "Check the draft for accuracy, completeness, and tone.",
        "finalise": "Apply review feedback and produce the final output.",
    },
    default_mode="research",
)
```

### Example 3 — Reading and setting mode from application code

```python
from agent_framework import AgentSession
from agent_framework._harness._mode import (
    AgentModeProvider,
    get_agent_mode,
    set_agent_mode,
)

provider = AgentModeProvider()
session = AgentSession(session_id="my-session")

# Read the current mode.
current = get_agent_mode(session, source_id=provider.source_id)
print(f"Current mode: {current}")   # "plan"

# Force the agent into execute mode before the next turn.
# The provider will inject a notification message on the next run.
set_agent_mode(session, "execute", source_id=provider.source_id)
```

### Example 4 — Mode gate in middleware

Combine `AgentModeProvider` with `AgentMiddleware` to block certain tool calls when the agent is in plan mode:

```python
from agent_framework import AgentMiddleware, AgentContext, MiddlewareTermination
from agent_framework._harness._mode import get_agent_mode


class ExecuteOnlyMiddleware(AgentMiddleware):
    """Block agent runs when mode is 'plan' and a dangerous tool is in the tool list."""

    DANGEROUS_TOOLS = {"delete_file", "send_email", "deploy_service"}

    async def process(self, context: AgentContext, call_next) -> None:
        mode = get_agent_mode(context.session, source_id="agent_mode") if context.session else "plan"
        if mode == "plan" and context.tools:
            dangerous = {t.name for t in context.tools if hasattr(t, "name")} & self.DANGEROUS_TOOLS
            if dangerous:
                raise MiddlewareTermination(
                    f"Tools {dangerous} are not allowed in plan mode.",
                    result=None,
                )
        await call_next()
```

---

## 10. `FileCheckpointStorage`

**Source:** `agent_framework/_workflows/_checkpoint.py`

`FileCheckpointStorage` saves workflow checkpoints as **JSON files with base64-encoded pickle blobs** for complex Python objects. Deserialization is restricted by default to a built-in safe set (Python primitives, datetime, uuid), all `agent_framework` types, and `openai.types`. Add application-specific types via `allowed_checkpoint_types`.

### Constructor

```python
FileCheckpointStorage(
    storage_path: str | Path,
    *,
    allowed_checkpoint_types: list[str] | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `storage_path` | *(required)* | Directory for checkpoint `.json` files; created if missing |
| `allowed_checkpoint_types` | `None` | Additional types allowed during deserialization, as `"module:qualname"` strings |

**Security:** The storage validates checkpoint IDs against path traversal. Pickle deserialization uses a restricted unpickler — do not load checkpoints from untrusted sources even with the default set.

### Key methods

```python
storage.save(checkpoint: WorkflowCheckpoint) -> CheckpointID
storage.load(checkpoint_id: CheckpointID) -> WorkflowCheckpoint
storage.list_checkpoints(*, workflow_name: str) -> list[WorkflowCheckpoint]
storage.delete(checkpoint_id: CheckpointID) -> bool
storage.get_latest(*, workflow_name: str) -> WorkflowCheckpoint | None
```

### Example 1 — Checkpoint a workflow and resume after a HITL pause

```python
import asyncio
from agent_framework import WorkflowBuilder, FileCheckpointStorage
from my_app.executors import ResearchExecutor


async def main() -> None:
    storage = FileCheckpointStorage(storage_path="./checkpoints")

    workflow = WorkflowBuilder(
        start_executor=ResearchExecutor(),
        checkpoint_storage=storage,
    ).build()

    # First run — human walks away mid-flow.
    stream = workflow.run("quantum sensors", stream=True)
    pending: dict[str, str] = {}

    async for event in stream:
        if event.type == "request_info":
            print(f"Human input needed: {event.data}")
            pending[event.request_id] = "technical"   # simulated answer
        elif event.type == "output":
            print("Output:", event.data)

    # Resume later by loading the latest checkpoint and supplying answers.
    latest = await storage.get_latest(workflow_name=workflow.name)
    if latest and pending:
        resumed_stream = workflow.run(
            checkpoint_id=latest.checkpoint_id,
            responses=pending,
            stream=True,
        )
        async for event in resumed_stream:
            if event.type == "output":
                print("Resumed output:", event.data)

asyncio.run(main())
```

### Example 2 — Allowing custom application types

If your executor state contains custom dataclasses, register them:

```python
from agent_framework import FileCheckpointStorage

storage = FileCheckpointStorage(
    storage_path="./checkpoints",
    allowed_checkpoint_types=[
        "my_app.models:ResearchState",
        "my_app.models:AnalysisResult",
    ],
)
```

### Example 3 — Checkpoint enumeration and cleanup

```python
from agent_framework import FileCheckpointStorage

storage = FileCheckpointStorage("./checkpoints")

# List all checkpoints for a workflow.
checkpoints = await storage.list_checkpoints(workflow_name="my-research-workflow")
print(f"Found {len(checkpoints)} checkpoints")

# Delete old checkpoints, keeping only the latest 5.
for checkpoint in checkpoints[:-5]:
    deleted = await storage.delete(checkpoint.checkpoint_id)
    if deleted:
        print(f"Deleted {checkpoint.checkpoint_id}")
```

### Example 4 — CI/CD resume pattern

Persist a HITL workflow, commit the checkpoint directory, and resume in a later CI job after human review:

```python
# job1.py — initial run, saves checkpoint when human input is needed
import asyncio
from agent_framework import FileCheckpointStorage, WorkflowBuilder
from my_app.executors import ReviewExecutor

storage = FileCheckpointStorage("./checkpoints")
workflow = WorkflowBuilder(start_executor=ReviewExecutor(), checkpoint_storage=storage).build()

async def job1() -> None:
    stream = workflow.run("Draft compliance report", stream=True)
    async for event in stream:
        if event.type == "request_info":
            print(f"Paused — waiting for review. Request ID: {event.request_id}")
            # Save the request_id to a file for job2 to pick up.
            with open("pending_request.txt", "w") as f:
                f.write(event.request_id)
            break

asyncio.run(job1())
```

```python
# job2.py — resume after human merges an approval PR
import asyncio
from agent_framework import FileCheckpointStorage, WorkflowBuilder
from my_app.executors import ReviewExecutor

storage = FileCheckpointStorage("./checkpoints")
workflow = WorkflowBuilder(start_executor=ReviewExecutor(), checkpoint_storage=storage).build()

async def job2() -> None:
    with open("pending_request.txt") as f:
        request_id = f.read().strip()

    latest = await storage.get_latest(workflow_name=workflow.name)
    if not latest:
        raise RuntimeError("No checkpoint found — did job1 run?")

    stream = workflow.run(
        checkpoint_id=latest.checkpoint_id,
        responses={request_id: "approved"},
        stream=True,
    )
    async for event in stream:
        if event.type == "output":
            print("Final report:", event.data)

asyncio.run(job2())
```

---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 37"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.11.0: SummarizationStrategy+TokenBudgetComposedStrategy+ToolResultCompactionStrategy+TruncationStrategy (four advanced compaction strategies — LLM-driven summarisation, hard token-budget composition, tool-result collapsing, oldest-first truncation); EvalItemResult+EvalScoreResult+EvalNotPassedError+ExpectedToolCall+RubricScore (evaluation result types — per-item status/scores, dimension rubric breakdown, tool-call assertion); FileSearchMatch+FileSearchResult+FileStoreEntry+InMemoryAgentFileStore+FileSystemAgentFileStore (file-access DTOs and store backends — line-level search matches, directory-listing entries, in-memory and disk-backed stores with symlink rejection); MemoryContextProvider+MemoryIndexEntry+MemoryTopicRecord+MemoryStore+MemoryFileStore (durable topic memory harness — MEMORY.md index, per-topic slugged records, file-backed multi-owner store); ContentVariableStore+InspectVariableInput+VariableReferenceContent (FIDES variable-indirection layer — store-and-reference pattern to prevent untrusted content entering LLM context); LabelTrackingFunctionMiddleware+PolicyEnforcementFunctionMiddleware (security label propagation — tiered label priority, source_integrity declaration, block/approval-on-violation); ChatTelemetryLayer+EmbeddingTelemetryLayer+OtelAttr+MessageListTimestampFilter (OTel telemetry mix-ins — GenAI semantic-convention span attributes, token usage/duration histograms, embedding tracing); WorkflowEvent+WorkflowErrorDetails+WorkflowEventSource+WorkflowRunState (workflow event system — typed generic events, factory methods, error details from exception, run-state enum); SubWorkflowRequestMessage+SubWorkflowResponseMessage+ExecutionContext (sub-workflow messaging protocol — parent/child HITL request/response handshake, collected-responses tracking); InMemorySkillsSource+SkillsSourceContext (in-memory skill serving — pre-built Skill instances, frozen context passed to every source in the pipeline) — source-verified at agent-framework 1.11.0."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 60
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 37

Verified against **agent-framework 1.11.0** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`.

Sub-packages introspected:
`agent_framework._compaction`,
`agent_framework._evaluation`,
`agent_framework._harness._file_access`,
`agent_framework._harness._memory`,
`agent_framework.security`,
`agent_framework.observability`,
`agent_framework._workflows._events`,
`agent_framework._workflows._workflow_executor`,
`agent_framework._skills`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) through [Vol. 36](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v36/) — 360+ classes covered.

This volume covers **ten class groups** across advanced compaction strategies, evaluation result types, file-access store DTOs, the durable memory harness, FIDES variable indirection, security label middleware, OTel telemetry layers, workflow event types, sub-workflow messaging, and in-memory skill serving.

| # | Class / group | Module |
|---|---|---|
| 1 | `SummarizationStrategy` · `TokenBudgetComposedStrategy` · `ToolResultCompactionStrategy` · `TruncationStrategy` | `agent_framework._compaction` |
| 2 | `EvalItemResult` · `EvalScoreResult` · `EvalNotPassedError` · `ExpectedToolCall` · `RubricScore` | `agent_framework._evaluation` |
| 3 | `FileSearchMatch` · `FileSearchResult` · `FileStoreEntry` · `InMemoryAgentFileStore` · `FileSystemAgentFileStore` | `agent_framework._harness._file_access` |
| 4 | `MemoryContextProvider` · `MemoryIndexEntry` · `MemoryTopicRecord` · `MemoryStore` · `MemoryFileStore` | `agent_framework._harness._memory` |
| 5 | `ContentVariableStore` · `InspectVariableInput` · `VariableReferenceContent` | `agent_framework.security` |
| 6 | `LabelTrackingFunctionMiddleware` · `PolicyEnforcementFunctionMiddleware` | `agent_framework.security` |
| 7 | `ChatTelemetryLayer` · `EmbeddingTelemetryLayer` · `OtelAttr` · `MessageListTimestampFilter` | `agent_framework.observability` |
| 8 | `WorkflowEvent` · `WorkflowErrorDetails` · `WorkflowEventSource` · `WorkflowRunState` | `agent_framework._workflows._events` |
| 9 | `SubWorkflowRequestMessage` · `SubWorkflowResponseMessage` · `ExecutionContext` | `agent_framework._workflows._workflow_executor` |
| 10 | `InMemorySkillsSource` · `SkillsSourceContext` | `agent_framework._skills` |

---

## 1 · Advanced Compaction Strategies

**Module:** `agent_framework._compaction`
**Install:** `pip install agent-framework`

The `_compaction` module provides four concrete strategies beyond the commonly documented `SlidingWindowStrategy` and `SelectiveToolCallCompactionStrategy`. Each implements the `CompactionStrategy` callable protocol (`async def __call__(self, messages: list[Message]) -> bool`).

### `TruncationStrategy`

Excludes oldest message groups (never partial tool-call groups) once an **included message count or token count** crosses `max_n`, trimming back to `compact_to`. It is the simplest deterministic strategy with no LLM calls.

```python
Constructor:
TruncationStrategy(
    *,
    max_n: int,              # trigger threshold (messages or tokens)
    compact_to: int,         # target after compaction (≤ max_n)
    tokenizer: TokenizerProtocol | None = None,  # token-based when provided
    preserve_system: bool = True,                # skip system groups
)
```

**Example 1 — message-count truncation**

```python
import asyncio
from agent_framework._compaction import TruncationStrategy
from agent_framework._types import Message

strategy = TruncationStrategy(max_n=20, compact_to=10)

async def demo():
    messages: list[Message] = [
        Message(role="user", contents=[f"turn {i}"]) for i in range(25)
    ]
    changed = await strategy(messages)
    included = [m for m in messages if not m.additional_properties.get("_excluded")]
    print(f"changed={changed}, included={len(included)}")  # changed=True, included=10

asyncio.run(demo())
```

**Example 2 — token-count truncation**

```python
import asyncio
from agent_framework._compaction import TruncationStrategy, CharacterEstimatorTokenizer
from agent_framework._types import Message

strategy = TruncationStrategy(
    max_n=200,          # tokens
    compact_to=100,
    tokenizer=CharacterEstimatorTokenizer(),
    preserve_system=True,
)

async def demo():
    messages: list[Message] = [
        Message(role="system", contents=["You are a helpful assistant."]),
    ] + [
        Message(role="user", contents=[f"This is message number {i} with some content."]) for i in range(20)
    ]
    changed = await strategy(messages)
    print(f"Compaction triggered: {changed}")

asyncio.run(demo())
```

**Example 3 — using `TruncationStrategy` inside `CompactionProvider`**

```python
import asyncio
from agent_framework import Agent
from agent_framework._compaction import CompactionProvider, TruncationStrategy, CharacterEstimatorTokenizer

async def demo(client):
    strategy = TruncationStrategy(max_n=50, compact_to=25, tokenizer=CharacterEstimatorTokenizer())
    agent = Agent(
        client=client,
        name="assistant",
        instructions="You are a concise assistant.",
        compaction_strategy=CompactionProvider(strategy=strategy),
    )
    response = await agent.run("Tell me something interesting.")
    print(response.text)

# asyncio.run(demo(your_client))
```

---

### `ToolResultCompactionStrategy`

**Collapses** older tool-call groups into compact summary messages of the form `[Tool results: get_weather: sunny; search: 10 results found]` rather than fully excluding them. This preserves a readable trace of tool calls without the full token cost of the original function-call/result structure.

```python
Constructor:
ToolResultCompactionStrategy(
    *,
    keep_last_tool_call_groups: int = 1,  # keep newest N groups verbatim; 0 = collapse all
)
```

**Example 4 — collapse all but the last tool-call group**

```python
import asyncio
from agent_framework._compaction import ToolResultCompactionStrategy
from agent_framework._types import Message
from agent_framework._compaction import annotate_message_groups

strategy = ToolResultCompactionStrategy(keep_last_tool_call_groups=1)

async def demo():
    # Build a minimal message list with tool call groups
    messages: list[Message] = [
        Message(role="user", contents=["What is the weather?"]),
        Message(role="assistant", contents=[{"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}]),
        Message(role="tool", contents=[{"type": "function_result", "call_id": "c1", "result": "Sunny, 18°C"}]),
        Message(role="user", contents=["And tomorrow?"]),
        Message(role="assistant", contents=[{"type": "function_call", "call_id": "c2", "name": "get_forecast", "arguments": "{}"}]),
        Message(role="tool", contents=[{"type": "function_result", "call_id": "c2", "result": "Cloudy, 15°C"}]),
    ]
    annotate_message_groups(messages)
    changed = await strategy(messages)
    print(f"changed={changed}")
    for m in messages:
        print(m.role, m.contents[:1])

asyncio.run(demo())
```

**Example 5 — collapse all tool groups (`keep_last=0`)**

```python
import asyncio
from agent_framework._compaction import ToolResultCompactionStrategy
from agent_framework._types import Message
from agent_framework._compaction import annotate_message_groups

strategy = ToolResultCompactionStrategy(keep_last_tool_call_groups=0)

async def demo():
    messages: list[Message] = [
        Message(role="assistant", contents=[{"type": "function_call", "call_id": "x1", "name": "search", "arguments": '{"q": "agent framework"}'}]),
        Message(role="tool", contents=[{"type": "function_result", "call_id": "x1", "result": "10 results found"}]),
    ]
    annotate_message_groups(messages)
    changed = await strategy(messages)
    # All tool groups collapsed; summary messages injected
    print(f"changed={changed}, messages={len(messages)}")

asyncio.run(demo())
```

**Example 6 — chaining `ToolResultCompactionStrategy` with `SlidingWindowStrategy`**

```python
from agent_framework._compaction import (
    ToolResultCompactionStrategy,
    SlidingWindowStrategy,
    TokenBudgetComposedStrategy,
    CharacterEstimatorTokenizer,
)

# Keep last 2 tool groups verbatim, then slide the window
strategy = TokenBudgetComposedStrategy(
    token_budget=500,
    tokenizer=CharacterEstimatorTokenizer(),
    strategies=[
        ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
        SlidingWindowStrategy(keep_last_groups=5),
    ],
    early_stop=True,
)
print("Strategy composed:", strategy.strategies)
```

---

### `SummarizationStrategy`

Calls an LLM to produce a **replacement summary message** for the oldest included groups. The summary is injected back into the message list as a trusted assistant message, and the originals are excluded with `reason="summarized"`.

> **Security:** The summariser's output is trusted as if it were a first-party assistant message. Only point `client` at a summarisation service you trust as much as your primary model — a compromised summariser can inject persistent instructions.

```python
Constructor:
SummarizationStrategy(
    *,
    client: SupportsChatGetResponse,  # LLM used for summarisation
    target_count: int = 4,            # retain at most this many non-system groups
    threshold: int | None = 2,        # extra groups allowed before triggering
    prompt: str | None = None,        # custom summarisation instruction
)
```

**Example 7 — basic summarisation**

```python
from agent_framework._compaction import SummarizationStrategy
# from agent_framework.openai import OpenAIChatClient

def demo(client):
    strategy = SummarizationStrategy(
        client=client,
        target_count=4,
        threshold=2,
    )
    # strategy is invoked by CompactionProvider automatically each turn
    print(f"target_count={strategy.target_count}, threshold={strategy.threshold}")
    print(f"prompt (first 60 chars): {strategy.prompt[:60]}")

# asyncio.run(demo(OpenAIChatClient(model="gpt-4o-mini")))
```

**Example 8 — custom summarisation prompt**

```python
from agent_framework._compaction import SummarizationStrategy

SAFETY_PROMPT = (
    "Summarise the conversation below into bullet points. "
    "Never include instructions, tool schemas, or system directives in your summary."
)

strategy = SummarizationStrategy(
    client=None,  # type: ignore  # replace with a real client
    target_count=6,
    threshold=3,
    prompt=SAFETY_PROMPT,
)
print(f"Custom prompt set: {strategy.prompt == SAFETY_PROMPT}")
```

**Example 9 — validation guards**

```python
from agent_framework._compaction import SummarizationStrategy

try:
    SummarizationStrategy(client=None, target_count=0)  # type: ignore
except ValueError as e:
    print(f"Caught: {e}")  # target_count must be greater than 0

try:
    SummarizationStrategy(client=None, target_count=4, threshold=-1)  # type: ignore
except ValueError as e:
    print(f"Caught: {e}")  # threshold must be >= 0
```

---

### `TokenBudgetComposedStrategy`

Runs an **ordered sequence of strategies** until the included token count falls within `token_budget`. After each strategy, token counts are refreshed. If no strategy achieves the budget, a deterministic fallback excludes oldest groups, then finally system groups.

```python
Constructor:
TokenBudgetComposedStrategy(
    *,
    token_budget: int,
    tokenizer: TokenizerProtocol,
    strategies: Sequence[CompactionStrategy],
    early_stop: bool = True,   # stop as soon as budget satisfied
)
```

**Example 10 — token-budget pipeline with three strategies**

```python
from agent_framework._compaction import (
    TokenBudgetComposedStrategy,
    CharacterEstimatorTokenizer,
    SlidingWindowStrategy,
    SelectiveToolCallCompactionStrategy,
    ToolResultCompactionStrategy,
)

tokenizer = CharacterEstimatorTokenizer()

strategy = TokenBudgetComposedStrategy(
    token_budget=1000,
    tokenizer=tokenizer,
    strategies=[
        SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2),
        ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
        SlidingWindowStrategy(keep_last_groups=4),
    ],
    early_stop=True,
)

print(f"Budget: {strategy.token_budget} tokens, strategies: {len(strategy.strategies)}")
```

---

## 2 · Evaluation Result Types

**Module:** `agent_framework._evaluation`
**Install:** `pip install agent-framework`
**Requires:** `@experimental(ExperimentalFeature.EVALS)` — import from `agent_framework._evaluation` directly.

### Class Summary

| Class | Role |
|---|---|
| `EvalItemResult` | Per-item outcome (pass/fail/error) with scores list |
| `EvalScoreResult` | Score from one evaluator on one item |
| `RubricScore` | Per-dimension score from a rubric-based evaluator |
| `ExpectedToolCall` | Data class asserting which tool an agent should call |
| `EvalNotPassedError` | Raised when any evaluation item fails |

```python
# Constructor signatures
# EvalItemResult (dataclass)
EvalItemResult(
    item_id: str,
    status: str,                          # "pass", "fail", or "error"
    scores: list[EvalScoreResult] = field(default_factory=list),
    error_code: str | None = None,
    error_message: str | None = None,
    response_id: str | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    token_usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
)

# EvalScoreResult (dataclass)
EvalScoreResult(
    name: str,                            # evaluator name
    score: float,
    passed: bool | None = None,
    sample: dict[str, Any] | None = None,
    dimensions: list[RubricScore] | None = None,
)

# RubricScore (frozen dataclass)
RubricScore(id: str, score: int | None, applicable: bool, weight: int, reason: str)

# ExpectedToolCall (dataclass)
ExpectedToolCall(name: str, arguments: dict[str, Any] | None = None)
```

**Example 11 — building and inspecting an `EvalItemResult`**

```python
from agent_framework._evaluation import EvalItemResult, EvalScoreResult

score = EvalScoreResult(name="relevance", score=0.85, passed=True)
item = EvalItemResult(
    item_id="item-001",
    status="pass",
    scores=[score],
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris.",
)

print(f"Passed: {item.is_passed}")       # True
print(f"Failed: {item.is_failed}")       # False
print(f"Error:  {item.is_error}")        # False
print(f"Scores: {item.scores[0].name}={item.scores[0].score}")
```

**Example 12 — rubric-based evaluator breakdown**

```python
from agent_framework._evaluation import EvalItemResult, EvalScoreResult, RubricScore

dim1 = RubricScore(id="accuracy", score=4, applicable=True, weight=3, reason="Factually correct")
dim2 = RubricScore(id="clarity",  score=3, applicable=True, weight=2, reason="Clear phrasing")
rubric = EvalScoreResult(
    name="overall_quality",
    score=3.5,
    passed=True,
    dimensions=[dim1, dim2],
)

item = EvalItemResult(item_id="item-002", status="pass", scores=[rubric])
for dim in item.scores[0].dimensions or []:
    print(f"  {dim.id}: {dim.score}/5 (weight={dim.weight}) — {dim.reason}")
```

**Example 13 — `ExpectedToolCall` and `EvalNotPassedError`**

```python
from agent_framework._evaluation import (
    EvalItemResult, EvalScoreResult, EvalNotPassedError, ExpectedToolCall
)

# Assert that the agent must call "get_weather" with specific args
expected = ExpectedToolCall(
    name="get_weather",
    arguments={"city": "London"},
)
print(f"Expected tool: {expected.name}, args: {expected.arguments}")

# Simulate a failed evaluation item
failed_score = EvalScoreResult(name="tool_use", score=0.0, passed=False)
failed_item = EvalItemResult(item_id="item-003", status="fail", scores=[failed_score])
print(f"Failed: {failed_item.is_failed}")  # True

# Raise EvalNotPassedError when results contain failures
results = [failed_item]
if any(r.is_failed for r in results):
    raise EvalNotPassedError("Evaluation did not pass: tool call not made")
```

---

## 3 · File-Access Store DTOs

**Module:** `agent_framework._harness._file_access`
**Install:** `pip install agent-framework`
**Requires:** `@experimental(ExperimentalFeature.HARNESS)`

### Class Summary

| Class | Role |
|---|---|
| `FileStoreEntry` | One directory-listing entry: `name` + `type` ("file"/"directory") |
| `FileSearchMatch` | One line match: `line_number` + `line` |
| `FileSearchResult` | Per-file search result: `file_name`, `snippet`, `matching_lines` |
| `InMemoryAgentFileStore` | Dict-backed `AgentFileStore` for tests |
| `FileSystemAgentFileStore` | Disk-backed `AgentFileStore` with symlink rejection |

```python
# Constructor signatures
FileStoreEntry(name: str, type: str)          # type in ("file", "directory")
FileSearchMatch(line_number: int, line: str)  # line_number >= 1
FileSearchResult(
    file_name: str,
    snippet: str = "",
    matching_lines: list[FileSearchMatch] | None = None,
)
InMemoryAgentFileStore()                      # empty in-memory store
FileSystemAgentFileStore(root_directory: str | os.PathLike)  # lazy-creates root
```

**Example 14 — `FileStoreEntry` type guards and serialisation**

```python
from agent_framework._harness._file_access import FileStoreEntry

file_entry = FileStoreEntry(name="report.md", type=FileStoreEntry.FILE)
dir_entry  = FileStoreEntry(name="archive",   type=FileStoreEntry.DIRECTORY)

print(file_entry.to_dict())  # {"name": "report.md", "type": "file"}
print(dir_entry.to_dict())   # {"name": "archive",   "type": "directory"}

# Round-trip
restored = FileStoreEntry.from_dict(file_entry.to_dict())
assert restored == file_entry

# Validation
try:
    FileStoreEntry(name="x", type="symlink")
except ValueError as e:
    print(f"Caught: {e}")
```

**Example 15 — `InMemoryAgentFileStore` CRUD + search**

```python
import asyncio
from agent_framework._harness._file_access import InMemoryAgentFileStore

store = InMemoryAgentFileStore()

async def demo():
    # Write files
    await store.write("notes/2026.md", "# 2026 Notes\nAgent framework rocks.")
    await store.write("notes/2025.md", "# 2025 Notes\nFirst experiments.")

    # Read
    content = await store.read("notes/2026.md")
    print(content[:30])  # "# 2026 Notes\nAgent framework"

    # List directory
    entries = await store.list_children("notes")
    for entry in entries:
        print(f"  {entry.type}: {entry.name}")

    # Search
    results = await store.search("notes", "Agent")
    for result in results:
        for match in result.matching_lines:
            print(f"  line {match.line_number}: {match.line}")

asyncio.run(demo())
```

**Example 16 — `FileSystemAgentFileStore` with overwrite guard**

```python
import asyncio, tempfile
from agent_framework._harness._file_access import FileSystemAgentFileStore

async def demo():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileSystemAgentFileStore(tmp)

        await store.write("config.json", '{"version": 1}')
        content = await store.read("config.json")
        print(content)  # {"version": 1}

        # Prevent accidental overwrite
        try:
            await store.write("config.json", '{"version": 2}', overwrite=False)
        except FileExistsError as e:
            print(f"Caught: {e}")

        # Exists check
        print(await store.file_exists("config.json"))  # True
        print(await store.file_exists("missing.txt"))   # False

asyncio.run(demo())
```

---

## 4 · Durable Memory Harness

**Module:** `agent_framework._harness._memory`
**Install:** `pip install agent-framework`
**Requires:** `@experimental(ExperimentalFeature.HARNESS)`

The memory harness provides **cross-session, per-user durable memory** backed by topic markdown files, a `MEMORY.md` index, and optional transcript history. The key classes are:

| Class | Role |
|---|---|
| `MemoryStore` | Abstract backing store |
| `MemoryFileStore` | Disk-backed store (one dir per owner/kind) |
| `MemoryTopicRecord` | One topic's memory bullets + metadata |
| `MemoryIndexEntry` | One pointer line in `MEMORY.md` |
| `MemoryContextProvider` | `HistoryProvider` that injects memory + exposes tools |

```python
# MemoryTopicRecord constructor
MemoryTopicRecord(
    *,
    topic: str,
    slug: str | None = None,       # stable filename stem; auto-derived from topic
    summary: str,
    memories: Sequence[str],       # bullet list; auto-deduped
    updated_at: str,               # ISO-format timestamp string
    session_ids: Sequence[str] | None = None,
)

# MemoryIndexEntry constructor
MemoryIndexEntry(topic: str, slug: str, summary: str, updated_at: str)

# MemoryFileStore constructor
MemoryFileStore(
    base_path: str | Path,
    *,
    kind: str = "memory",
    owner_prefix: str = "",
    owner_state_key: str,          # session.state key holding owner ID
    index_file_name: str = "MEMORY.md",
    topics_directory_name: str = "topics",
    transcripts_directory_name: str = "transcripts",
    state_file_name: str = "state.json",
    dumps: JsonDumps | None = None,
    loads: JsonLoads | None = None,
)
```

**Example 17 — building and round-tripping a `MemoryTopicRecord`**

```python
from agent_framework._harness._memory import MemoryTopicRecord

record = MemoryTopicRecord(
    topic="Python Tips",
    summary="Practical Python patterns from our sessions",
    memories=[
        "Prefer dataclasses for value objects",
        "Use asyncio.gather for concurrent I/O",
        "Prefer dataclasses for value objects",  # duplicate — auto-removed
    ],
    updated_at="2026-07-13T12:00:00Z",
    session_ids=["sess-001", "sess-002"],
)

print(f"slug: {record.slug}")               # "python-tips"
print(f"memories: {record.memories}")       # deduplicated list
print(f"sessions: {record.session_ids}")

d = record.to_dict()
restored = MemoryTopicRecord.from_dict(d)
assert restored.topic == record.topic
```

**Example 18 — `MemoryIndexEntry` lifecycle**

```python
from agent_framework._harness._memory import MemoryIndexEntry, MemoryTopicRecord

record = MemoryTopicRecord(
    topic="Design Patterns",
    summary="GoF patterns and when to use them",
    memories=["Prefer composition over inheritance", "Factory pattern for object creation"],
    updated_at="2026-07-13T09:30:00Z",
)

# Derive index entry from topic record
entry = MemoryIndexEntry.from_topic_record(record)
print(entry.to_dict())
# {"topic": "Design Patterns", "slug": "design-patterns", "summary": "...", "updated_at": "..."}

# Round-trip
restored_entry = MemoryIndexEntry.from_dict(entry.to_dict())
assert restored_entry.slug == entry.slug
```

**Example 19 — attaching `MemoryContextProvider` to an agent**

```python
import tempfile
from agent_framework._harness._memory import MemoryContextProvider, MemoryFileStore

async def demo(client):
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryFileStore(
            base_path=tmp,
            owner_state_key="user_id",   # set session.state["user_id"] before run
        )
        memory_provider = MemoryContextProvider(
            store=store,
            source_id="memory",
            recent_turns=2,             # inject last 2 turns alongside MEMORY.md
            index_line_limit=20,
            selection_limit=5,
            consolidation_client=client, # used for periodic consolidation
        )

        from agent_framework import Agent
        from agent_framework._sessions import AgentSession

        agent = Agent(
            client=client,
            name="assistant",
            instructions="You are a helpful assistant with persistent memory.",
            context_providers=[memory_provider],
        )
        session = AgentSession()
        session.state["user_id"] = "user-alice"

        response = await agent.run("Remember that I prefer dark mode.", session=session)
        print(response.text)

# asyncio.run(demo(your_client))
```

---

## 5 · FIDES Variable-Indirection Layer

**Module:** `agent_framework.security`
**Install:** `pip install agent-framework`
**Requires:** `@experimental(ExperimentalFeature.FIDES)`

Variable indirection prevents untrusted external content from entering the LLM context directly. Instead of passing raw external data as a message, you store it in a `ContentVariableStore` and give the LLM a `VariableReferenceContent` token. The LLM can then call the `inspect_variable` tool (schema: `InspectVariableInput`) to retrieve the value under explicit audit.

```python
# ContentVariableStore constructor
ContentVariableStore()  # no args; empty dict-backed store

# VariableReferenceContent constructor
VariableReferenceContent(
    variable_id: str,
    label: ContentLabel,
    description: str | None = None,
)
```

**Example 20 — store untrusted content and reference it**

```python
from agent_framework.security import (
    ContentVariableStore, VariableReferenceContent,
    ContentLabel, IntegrityLabel,
)

store = ContentVariableStore()
untrusted_label = ContentLabel(integrity=IntegrityLabel.UNTRUSTED)

# Store an external API response
var_id = store.store(
    {"temperature": "18°C", "forecast": "partly cloudy"},
    untrusted_label,
)
print(f"Stored as: {var_id}")  # "var_<16-hex-chars>"

# Build a reference for the LLM context
ref = VariableReferenceContent(
    variable_id=var_id,
    label=untrusted_label,
    description="Weather API response for London",
)
print(repr(ref))

# Retrieve the real content when authorised
content, label = store.retrieve(var_id)
print(content)  # {"temperature": "18°C", "forecast": "partly cloudy"}
```

**Example 21 — `InspectVariableInput` schema for tool routing**

```python
from agent_framework.security import InspectVariableInput

# Simulate LLM-supplied tool arguments
raw_args = {"variable_id": "var_abc123def456789", "reason": "User asked for weather details"}
inp = InspectVariableInput(**raw_args)
print(f"var_id={inp.variable_id}, reason={inp.reason}")

# Reason is optional
no_reason = InspectVariableInput(variable_id="var_xyz")
print(f"reason={no_reason.reason}")  # None
```

**Example 22 — `ContentVariableStore` exists / clear lifecycle**

```python
from agent_framework.security import ContentVariableStore, ContentLabel, IntegrityLabel

store = ContentVariableStore()
label = ContentLabel(integrity=IntegrityLabel.UNTRUSTED)

var1 = store.store("first payload", label)
var2 = store.store("second payload", label)

print(store.exists(var1))   # True
print(store.exists("var_missing"))  # False

store.clear()
print(store.exists(var1))   # False (all cleared)

try:
    store.retrieve(var1)
except KeyError as e:
    print(f"Caught: {e}")
```

---

## 6 · Security Label Middleware

**Module:** `agent_framework.security`
**Install:** `pip install agent-framework`
**Requires:** `@experimental(ExperimentalFeature.FIDES)`

Two `FunctionMiddleware` implementations enforce **information-flow control (IFC)** through tool calls.

### `LabelTrackingFunctionMiddleware`

Propagates `ContentLabel` through tool call chains using a three-tier priority:

| Priority | Source | When used |
|---|---|---|
| Tier 1 | Per-item embedded label in the result (`additional_properties.security_label`) | Always wins |
| Tier 2 | Tool's `source_integrity` declaration | No embedded label |
| Tier 3 | Join (combine) of all input argument labels | No embedded label AND no `source_integrity` |

```python
LabelTrackingFunctionMiddleware(
    default_integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED,
    default_confidentiality: ConfidentialityLabel | None = None,
    auto_hide_untrusted: bool = True,
    hide_threshold: IntegrityLabel = IntegrityLabel.UNTRUSTED,
)
```

**Example 23 — wiring label tracking into an agent**

```python
from agent_framework import Agent, tool
from agent_framework.security import (
    LabelTrackingFunctionMiddleware,
    ContentLabel, IntegrityLabel,
)

@tool
def fetch_external_data(query: str) -> str:
    """Fetch data from an external (untrusted) source."""
    return f"External result for {query}"

# Mark this tool as fetching untrusted data
fetch_external_data.additional_properties = {"source_integrity": "untrusted"}

label_tracker = LabelTrackingFunctionMiddleware(
    default_integrity=IntegrityLabel.UNTRUSTED,
    auto_hide_untrusted=True,
)

async def demo(client):
    agent = Agent(
        client=client,
        name="secure-agent",
        tools=[fetch_external_data],
        middleware=[label_tracker],
    )
    response = await agent.run("Search for agent framework news.")
    print(response.text)

# asyncio.run(demo(your_client))
```

### `PolicyEnforcementFunctionMiddleware`

Blocks or seeks approval for tools invoked in an **untrusted context**, unless the tool is in `allow_untrusted_tools`.

```python
PolicyEnforcementFunctionMiddleware(
    allow_untrusted_tools: set[str] | None = None,
    block_on_violation: bool = True,
    enable_audit_log: bool = True,
    approval_on_violation: bool = False,   # request user approval instead of blocking
)
```

**Example 24 — combining label tracking and policy enforcement**

```python
from agent_framework import Agent, tool
from agent_framework.security import (
    LabelTrackingFunctionMiddleware,
    PolicyEnforcementFunctionMiddleware,
)

@tool
def search_web(query: str) -> str:
    """Search the web (untrusted source)."""
    return f"Results for {query}"

@tool
def send_email(to: str, body: str) -> str:
    """Send an email (high-impact action)."""
    return f"Email sent to {to}"

label_tracker = LabelTrackingFunctionMiddleware(auto_hide_untrusted=True)
policy = PolicyEnforcementFunctionMiddleware(
    allow_untrusted_tools={"search_web"},  # search_web may run even in untrusted context
    block_on_violation=True,
    enable_audit_log=True,
)

async def demo(client):
    agent = Agent(
        client=client,
        name="policy-agent",
        tools=[search_web, send_email],
        middleware=[label_tracker, policy],  # label_tracker must run first
    )
    response = await agent.run("Search for news and then email me a summary.")
    print(response.text)

# asyncio.run(demo(your_client))
```

**Example 25 — approval-on-violation mode**

```python
from agent_framework.security import PolicyEnforcementFunctionMiddleware

policy = PolicyEnforcementFunctionMiddleware(
    allow_untrusted_tools=set(),
    approval_on_violation=True,   # request user approval rather than blocking outright
    enable_audit_log=True,
)
print(f"block_on_violation={policy.block_on_violation}")     # False (overridden)
print(f"approval_on_violation={policy.approval_on_violation}")  # True
print(f"audit_log length: {len(policy.audit_log)}")         # 0 initially
```

---

## 7 · OTel Telemetry Layers

**Module:** `agent_framework.observability`
**Install:** `pip install agent-framework`

### Class Summary

| Class | Role |
|---|---|
| `ChatTelemetryLayer` | Mix-in wrapping `get_response` with GenAI OTel spans |
| `EmbeddingTelemetryLayer` | Mix-in wrapping `get_embeddings` with OTel spans |
| `OtelAttr` | `str, Enum` of all GenAI semantic-convention attribute keys |
| `MessageListTimestampFilter` | Python `logging.Filter` that staggers log timestamps by index |

`ChatTelemetryLayer` and `EmbeddingTelemetryLayer` are generic **mix-in classes** applied via multiple inheritance by provider-specific clients (e.g. `OpenAIChatClient`). They record GenAI semantic-convention spans, token usage histograms, and duration histograms.

**Example 26 — enumerating `OtelAttr` span attribute keys**

```python
from agent_framework.observability import OtelAttr

# List all GenAI semantic-convention attribute keys
for attr in OtelAttr:
    print(f"  {attr.name}: {attr.value}")

# Common attributes used in spans
print(OtelAttr.OPERATION)       # "gen_ai.operation.name"
print(OtelAttr.INPUT_TOKENS)    # "gen_ai.usage.input_tokens"
print(OtelAttr.OUTPUT_TOKENS)   # "gen_ai.usage.output_tokens"
print(OtelAttr.AGENT_NAME)      # "gen_ai.agent.name"
```

**Example 27 — `ObservabilitySettings` + process-wide disable/re-enable**

```python
from agent_framework.observability import (
    OBSERVABILITY_SETTINGS,
    disable_instrumentation,
    enable_instrumentation,
)

# Inspect the process-wide singleton — this is what ChatTelemetryLayer checks.
print(f"ENABLED: {OBSERVABILITY_SETTINGS.ENABLED}")          # True by default
print(f"user_disabled: {OBSERVABILITY_SETTINGS.is_user_disabled}")  # False

# Sticky-disable all framework telemetry (e.g. during unit tests).
disable_instrumentation()
print(f"after disable: {OBSERVABILITY_SETTINGS.ENABLED}")    # False

# Re-enable (force=True clears the sticky flag).
enable_instrumentation(force=True)
print(f"after re-enable: {OBSERVABILITY_SETTINGS.ENABLED}")  # True
```

**Example 28 — `MessageListTimestampFilter` for sequenced log replay**

```python
import logging
from agent_framework.observability import MessageListTimestampFilter

# Create a logger that staggers message timestamps so they replay in order
logger = logging.getLogger("agent_framework.message_replay")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.addFilter(MessageListTimestampFilter())
logger.addHandler(handler)

# Emit records with chat_message_index so timestamps are staggered
for idx, text in enumerate(["User: hello", "Assistant: hi", "User: bye"]):
    record = logging.LogRecord(
        name="replay", level=logging.INFO, pathname="", lineno=0,
        msg=text, args=(), exc_info=None,
    )
    setattr(record, MessageListTimestampFilter.INDEX_KEY, idx)
    handler.emit(record)
```

---

## 8 · Workflow Event System

**Module:** `agent_framework._workflows._events`
**Install:** `pip install agent-framework`

### Class Summary

| Class | Role |
|---|---|
| `WorkflowRunState` | `str, Enum` — run-level lifecycle states |
| `WorkflowEventSource` | `str, Enum` — `FRAMEWORK` vs `EXECUTOR` origin |
| `WorkflowErrorDetails` | Structured exception info (type, message, traceback) |
| `WorkflowEvent[T]` | Generic event class with typed `data` payload; factory methods for each lifecycle event |

```python
# WorkflowRunState values
WorkflowRunState.STARTED
WorkflowRunState.IN_PROGRESS
WorkflowRunState.IN_PROGRESS_PENDING_REQUESTS
WorkflowRunState.IDLE
WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
WorkflowRunState.FAILED
WorkflowRunState.CANCELLED

# WorkflowEventSource values
WorkflowEventSource.FRAMEWORK
WorkflowEventSource.EXECUTOR
```

**Example 29 — building workflow events via factory methods**

```python
from agent_framework._workflows._events import (
    WorkflowEvent, WorkflowErrorDetails, WorkflowEventSource, WorkflowRunState
)

# Lifecycle events (data=None)
started = WorkflowEvent.started()
print(f"type={started.type}, source={started.source}")

status = WorkflowEvent.status(WorkflowRunState.IN_PROGRESS)
print(f"state={status.state}")  # WorkflowRunState.IN_PROGRESS

# Warning event
warning = WorkflowEvent.warning("Retrying after timeout")
print(f"type={warning.type}, data={warning.data}")

# Error from exception
try:
    raise ValueError("Unexpected input format")
except ValueError as exc:
    details = WorkflowErrorDetails.from_exception(exc, executor_id="step-1")
    error_event = WorkflowEvent.failed(details)
    print(f"error_type={error_event.details.error_type}")      # "ValueError"
    print(f"message={error_event.details.message}")            # "Unexpected input format"
    print(f"executor_id={error_event.details.executor_id}")    # "step-1"
    has_tb = error_event.details.traceback is not None
    print(f"traceback captured: {has_tb}")
```

**Example 30 — consuming workflow events from a running workflow**

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._events import WorkflowEvent, WorkflowRunState

async def demo(client):
    builder = WorkflowBuilder()
    # ... add executors ...

    workflow = builder.build()
    runner = workflow.create_runner()

    async for event in runner.run_stream("What is 2+2?"):
        if isinstance(event, WorkflowEvent):
            if event.type == "status":
                state: WorkflowRunState = event.state   # state lives on .state
                print(f"State changed: {state.value}")
            elif event.type == "failed":
                print(f"Workflow failed: {event.details.message}")  # error on .details
            elif event.type == "data":
                print(f"Output: {event.data}")

# asyncio.run(demo(your_client))
```

---

## 9 · Sub-Workflow Messaging Protocol

**Module:** `agent_framework._workflows._workflow_executor`
**Install:** `pip install agent-framework`

When a child workflow needs human-in-the-loop (HITL) input, it emits a `request_info` event. The parent `WorkflowExecutor` wraps this in a `SubWorkflowRequestMessage`, collects the response, and sends it back as a `SubWorkflowResponseMessage`. `ExecutionContext` tracks which responses have been collected for each ongoing sub-workflow execution.

```python
# SubWorkflowRequestMessage (dataclass)
SubWorkflowRequestMessage(
    source_event: WorkflowEvent,   # the original request_info event
    executor_id: str,              # parent executor responsible for this sub-workflow
)
# .create_response(data) → SubWorkflowResponseMessage (type-validates data)

# SubWorkflowResponseMessage (dataclass)
SubWorkflowResponseMessage(
    data: Any,
    source_event: WorkflowEvent,
)

# ExecutionContext (dataclass)
ExecutionContext(
    execution_id: str,
    collected_responses: dict[str, Any],       # request_id → response_data
    expected_response_count: int,
    pending_requests: dict[str, WorkflowEvent], # request_id → request_info event
)
```

**Example 31 — building a `SubWorkflowRequestMessage` and responding**

```python
from agent_framework._workflows._events import WorkflowEvent
from agent_framework._workflows._workflow_executor import (
    SubWorkflowRequestMessage,
)

# Simulate a request_info event emitted by a sub-workflow executor
request_event = WorkflowEvent.request_info(
    request_id="req-001",
    source_executor_id="email-confirmation-step",
    request_data="Please confirm the user's email address",
    response_type=str,
)

# Parent wraps it in a request message
req_msg = SubWorkflowRequestMessage(
    source_event=request_event,
    executor_id="email-confirmation-step",
)

# User provides the answer; create_response validates the type
resp_msg = req_msg.create_response("alice@example.com")
print(f"response data: {resp_msg.data}")
print(f"source request_id: {resp_msg.source_event.request_id}")

# Wrong type raises TypeError
try:
    req_msg.create_response(42)  # expected str
except TypeError as e:
    print(f"Caught: {e}")
```

**Example 32 — tracking collected responses in `ExecutionContext`**

```python
from agent_framework._workflows._events import WorkflowEvent
from agent_framework._workflows._workflow_executor import ExecutionContext

req1 = WorkflowEvent.request_info(
    request_id="req-001",
    source_executor_id="name-step",
    request_data="What is your name?",
    response_type=str,
)
req2 = WorkflowEvent.request_info(
    request_id="req-002",
    source_executor_id="age-step",
    request_data="What is your age?",
    response_type=int,
)

ctx = ExecutionContext(
    execution_id="run-001",
    collected_responses={},
    expected_response_count=2,
    pending_requests={req1.request_id: req1, req2.request_id: req2},
)

# Simulate receiving responses one at a time
ctx.collected_responses[req1.request_id] = "Alice"
print(f"received {len(ctx.collected_responses)}/{ctx.expected_response_count}")

ctx.collected_responses[req2.request_id] = 30
all_received = len(ctx.collected_responses) >= ctx.expected_response_count
print(f"all received: {all_received}")  # True
```

**Example 33 — full sub-workflow HITL pattern**

```python
from agent_framework._workflows._workflow_executor import SubWorkflowRequestMessage

async def demo(client, parent_runner):
    # When the parent runner yields a SubWorkflowRequestMessage, the orchestrator:
    # 1. Presents the question to the user
    # 2. Collects the answer
    # 3. Sends back SubWorkflowResponseMessage via the runner's respond() API

    async for event in parent_runner.run_stream("Run the sub-workflow"):
        if isinstance(event, SubWorkflowRequestMessage):
            print(f"Sub-workflow asks: {event.source_event.data}")
            # Collect user input (here simulated)
            user_answer = "confirmed"
            response = event.create_response(user_answer)
            await parent_runner.send_response(response)

# asyncio.run(demo(client, your_parent_runner))
```

---

## 10 · In-Memory Skill Serving

**Module:** `agent_framework._skills`
**Install:** `pip install agent-framework`

### `InMemorySkillsSource`

Serves pre-built `Skill` instances (e.g. `InlineSkill`, `FileSkill`) from a simple in-memory list. Unlike `FileSkillsSource` (reads from disk) or `MCPSkillsSource` (reads from MCP server), this source never touches I/O — ideal for tests and embedded skill definitions.

### `SkillsSourceContext`

A frozen dataclass that every `SkillsSource` receives when `get_skills(context)` is called. It carries the requesting agent and the current session, allowing sources and decorators to make per-agent or per-session decisions.

```python
# InMemorySkillsSource constructor
InMemorySkillsSource(skills: Sequence[Skill])

# SkillsSourceContext constructor (frozen dataclass)
SkillsSourceContext(
    agent: SupportsAgentRun,
    session: AgentSession | None = None,
)
```

**Example 34 — serving `InlineSkill` instances from memory**

```python
from agent_framework._skills import InlineSkill, InMemorySkillsSource, SkillsSourceContext, SkillFrontmatter

skill_a = InlineSkill(
    frontmatter=SkillFrontmatter(name="summariser", description="Summarise long text"),
    instructions="When summarising, extract key points as bullet points.",
)
skill_b = InlineSkill(
    frontmatter=SkillFrontmatter(name="code-reviewer", description="Review Python code"),
    instructions="Focus on readability, correctness, and security.",
)

source = InMemorySkillsSource([skill_a, skill_b])

async def demo(agent):
    context = SkillsSourceContext(agent=agent)
    skills = await source.get_skills(context)
    for skill in skills:
        print(f"  {skill.frontmatter.name}: {skill.frontmatter.description}")

# asyncio.run(demo(your_agent))
```

**Example 35 — using `InMemorySkillsSource` inside a `SkillsProvider`**

```python
from agent_framework import Agent
from agent_framework._skills import (
    InlineSkill, InMemorySkillsSource, SkillsProvider,
    FilteringSkillsSource, SkillFrontmatter,
)

skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="safety-check",
        description="Check output for safety concerns",
    ),
    instructions="Flag any content that could be harmful or misleading.",
)

# Wrap with a filter so only the "safety-agent" receives this skill
filtered = FilteringSkillsSource(
    source=InMemorySkillsSource([skill]),
    predicate=lambda s, ctx: ctx.agent.name == "safety-agent",
)

async def demo(client):
    agent = Agent(
        client=client,
        name="safety-agent",
        context_providers=[SkillsProvider(source=filtered)],
    )
    response = await agent.run("Is this output safe: 'Buy now at extreme discounts!'")
    print(response.text)

# asyncio.run(demo(your_client))
```

**Example 36 — `SkillsSourceContext` field access**

```python
from agent_framework._skills import SkillsSourceContext
from agent_framework._sessions import AgentSession

# Context is frozen — fields are set at construction and cannot be mutated
def demo(agent):
    session = AgentSession()
    ctx = SkillsSourceContext(agent=agent, session=session)

    print(f"agent name: {ctx.agent.name}")
    print(f"session is set: {ctx.session is not None}")

    # Frozen — mutation raises FrozenInstanceError
    try:
        ctx.session = None  # type: ignore
    except Exception as e:
        print(f"Caught: {type(e).__name__}")  # FrozenInstanceError

# demo(your_agent)
```

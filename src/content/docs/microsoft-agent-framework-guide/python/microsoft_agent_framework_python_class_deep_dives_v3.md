---
title: "Microsoft Agent Framework (Python) — 10-Class Deep Dives Vol. 3 (1.17.0)"
description: "Source-verified deep dives for WorkflowBuilder, SlidingWindowStrategy, TruncationStrategy, ContextWindowCompactionStrategy, LocalEvaluator, InlineSkill, FileAccessProvider, MemoryContextProvider, FileHistoryProvider, and MCPWebsocketTool — all verified against agent-framework 1.17.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-Class Deep Dives Vol. 3

**Verified against:** `agent-framework==1.17.0`
**Python requirement:** 3.10+

This volume covers 10 additional public classes spanning workflow construction, compaction strategies, local evaluation, skills, file access, memory, history persistence, and WebSocket MCP integration. Each section includes the full `__init__` signature, every meaningful method, and self-contained runnable examples verified against the 1.17.0 source.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

---

## 1. `WorkflowBuilder`

**Module:** `agent_framework._workflows._workflow_builder` (re-exported via `agent_framework`)

`WorkflowBuilder` is the primary entry point for constructing typed, directed-graph workflows. It wraps raw agents in `AgentExecutor` transparently, validates edge compatibility, and returns an immutable `Workflow` via `.build()`.

### Constructor

```python
WorkflowBuilder(
    max_iterations: int = 100,
    name: str | None = None,
    description: str | None = None,
    *,
    start_executor: Executor | SupportsAgentRun,
    checkpoint_storage: CheckpointStorage | None = None,
    output_from: list[Executor | SupportsAgentRun] | Literal["all"] | None = <sentinel>,
    intermediate_output_from: list[Executor | SupportsAgentRun]
                              | Literal["all", "all_other"] | None = <sentinel>,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `max_iterations` | `int` | Max supersteps before convergence timeout. Default 100. |
| `name` | `str \| None` | Human-readable stable identifier; auto-generated UUID name if omitted. |
| `description` | `str \| None` | Optional description embedded in the workflow. |
| `start_executor` | `Executor \| SupportsAgentRun` | Root node of the graph. |
| `checkpoint_storage` | `CheckpointStorage \| None` | Enables state persistence between runs. |
| `output_from` | `list \| "all" \| None` | Explicit output emitters. Omit to keep the default (all). |
| `intermediate_output_from` | `list \| "all" \| "all_other" \| None` | Executors that emit intermediate events. |

### Methods

| Method | Returns | Notes |
|---|---|---|
| `add_edge(source, target, condition=None)` | `Self` | Single directed edge with optional `(data) -> bool` condition guard. |
| `add_chain(executors)` | `Self` | Wire a list of executors sequentially. Raises `ValueError` if fewer than 2 executors. |
| `add_fan_out_edges(source, targets)` | `Self` | Broadcast one source to many targets concurrently. |
| `add_fan_in_edges(sources, target)` | `Self` | Merge many sources into one target (runs after all sources complete). |
| `add_switch_case_edge_group(source, cases)` | `Self` | Condition-routed edge; evaluates `Case` conditions in order, falls through to `Default`. |
| `add_multi_selection_edge_group(source, targets, selection_func)` | `Self` | Dynamic fan-out: `selection_func(message, target_ids) -> list[target_id]` at runtime. |
| `build()` | `Workflow` | Validate and freeze the graph into an immutable `Workflow`. |

### Example 1 — Linear chain with `.add_chain()`

```python
import asyncio
from typing_extensions import Never
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

class Cleaner(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text.strip().lower())

class Reverser(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text[::-1])

class Publisher(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"result: {text}")

cleaner   = Cleaner(id="cleaner")
reverser  = Reverser(id="reverser")
publisher = Publisher(id="publisher")

workflow = (
    WorkflowBuilder(start_executor=cleaner)
    .add_chain([cleaner, reverser, publisher])
    .build()
)

async def main() -> None:
    result = await workflow.run("  Hello World  ")
    print(result.get_outputs())  # ['result: dlrow olleh']

asyncio.run(main())
```

### Example 2 — Conditional routing with `.add_switch_case_edge_group()`

```python
import asyncio
from dataclasses import dataclass
from agent_framework import (
    Case, Default, Executor, WorkflowBuilder, WorkflowContext, handler
)

@dataclass
class Review:
    score: int
    text: str

class Scorer(Executor):
    @handler
    async def run(self, text: str, ctx: WorkflowContext[Review]) -> None:
        await ctx.send_message(Review(score=len(text), text=text))

class HighScoreHandler(Executor):
    @handler
    async def run(self, r: Review, ctx: WorkflowContext) -> None:
        print(f"High score ({r.score}): {r.text}")

class LowScoreHandler(Executor):
    @handler
    async def run(self, r: Review, ctx: WorkflowContext) -> None:
        print(f"Low score ({r.score}): {r.text}")

scorer  = Scorer(id="scorer")
high    = HighScoreHandler(id="high")
low     = LowScoreHandler(id="low")

workflow = (
    WorkflowBuilder(start_executor=scorer)
    .add_switch_case_edge_group(
        scorer,
        cases=[
            Case(target=high, condition=lambda r: r.score >= 10),
            Default(target=low),
        ],
    )
    .build()
)

async def main() -> None:
    await workflow.run("short")           # low score handler
    await workflow.run("a much longer sentence here")  # high score handler

asyncio.run(main())
```

### Example 3 — Dynamic multi-selection routing

```python
import asyncio
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

class Router(Executor):
    @handler
    async def route(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)

class UpperWorker(Executor):
    @handler
    async def work(self, text: str, ctx: WorkflowContext) -> None:
        print("Upper:", text.upper())

class LowerWorker(Executor):
    @handler
    async def work(self, text: str, ctx: WorkflowContext) -> None:
        print("Lower:", text.lower())

router = Router(id="router")
upper  = UpperWorker(id="upper")
lower  = LowerWorker(id="lower")

def select_targets(message: str, target_ids: list[str]) -> list[str]:
    if len(message) > 5:
        return target_ids  # send to all when message is long
    return [target_ids[0]]  # only upper worker for short messages

workflow = (
    WorkflowBuilder(start_executor=router)
    .add_multi_selection_edge_group(router, [upper, lower], select_targets)
    .build()
)

async def main() -> None:
    await workflow.run("hi")      # → only UpperWorker
    await workflow.run("hello world")  # → both workers

asyncio.run(main())
```

### Output selection modes

The snippets below are illustrative — `a`, `mid`, and `final` are placeholder names for `Executor` or `Agent` instances you have already constructed and registered with the builder.

```python
# All executors emit output (default)
# `a` is the start_executor Executor/Agent instance
WorkflowBuilder(start_executor=a)

# Only `final` emits output; other executor yields are hidden
# `final` is an Executor/Agent instance already added via add_edge / add_chain
WorkflowBuilder(start_executor=a, output_from=[final])

# `final` emits output; every other output-capable executor emits intermediate events
WorkflowBuilder(start_executor=a, output_from=[final],
                intermediate_output_from="all_other")
```

---

## 2. `SlidingWindowStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`SlidingWindowStrategy` implements a recency-based compaction strategy that keeps only the **most recent N non-system conversation groups**, discarding older ones. It is ideal for long-running conversations where only recent context is relevant but system instructions must be preserved.

### Constructor

```python
SlidingWindowStrategy(
    *,
    keep_last_groups: int,
    preserve_system: bool = True,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `keep_last_groups` | `int` | Number of most-recent non-system groups to retain. Must be ≥ 1. |
| `preserve_system` | `bool` | When `True` (default), all system groups remain in every compaction pass. |

`SlidingWindowStrategy` is a callable `CompactionStrategy`: it annotates messages in-place with `_excluded=True` for groups beyond the window and returns `True` if any messages were modified.

### Example — Keep last 3 conversation groups

```python
import asyncio
from agent_framework import Agent, SlidingWindowStrategy, CompactionProvider
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

strategy  = SlidingWindowStrategy(keep_last_groups=3, preserve_system=True)
compactor = CompactionProvider(before_strategy=strategy)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[compactor],
)

async def main() -> None:
    session = agent.create_session()
    for turn in ["Tell me about Paris.", "What about Rome?", "And Berlin?",
                 "Compare all three cities.", "Which has the best food?"]:
        response = await agent.run(turn, session=session)
        print(f"Q: {turn}")
        print(f"A: {response.text[:100]}...")

asyncio.run(main())
```

### Composing with other strategies

```python
from agent_framework import (
    SlidingWindowStrategy, SummarizationStrategy,
    TokenBudgetComposedStrategy, CharacterEstimatorTokenizer, CompactionProvider
)
from agent_framework.openai import OpenAIChatClient

client    = OpenAIChatClient()
window    = SlidingWindowStrategy(keep_last_groups=6)
summarize = SummarizationStrategy(client=client)

# Summarize old groups first, then slide the window
composed  = TokenBudgetComposedStrategy(
    strategies=[summarize, window],   # parameter is `strategies`, not `compaction_strategies`
    token_budget=8000,
    tokenizer=CharacterEstimatorTokenizer(),  # required
)
compactor = CompactionProvider(before_strategy=composed)
```

---

## 3. `TruncationStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`TruncationStrategy` triggers oldest-first message group removal when a threshold is exceeded. The threshold can be measured in **token count** (when `tokenizer` is provided) or **message count** (when `tokenizer` is omitted). Compaction runs when `max_n` is exceeded and removes groups until the metric drops to `compact_to`.

### Constructor

```python
TruncationStrategy(
    *,
    max_n: int,
    compact_to: int,
    tokenizer: TokenizerProtocol | None = None,
    preserve_system: bool = True,
    preserve_first_user_group: bool = False,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `max_n` | `int` | Trigger threshold (tokens or messages). |
| `compact_to` | `int` | Target metric after truncation. Must be `0 < compact_to ≤ max_n`. |
| `tokenizer` | `TokenizerProtocol \| None` | When provided, measures in tokens; otherwise in message count. |
| `preserve_system` | `bool` | System groups are never evicted. |
| `preserve_first_user_group` | `bool` | The earliest user group is always kept. |

`TokenizerProtocol` requires a single method: `count_tokens(text: str) -> int`. The built-in `CharacterEstimatorTokenizer` (4 chars/token) is a zero-dependency option.

### Example — Message-count-based truncation

```python
import asyncio
from agent_framework import Agent, TruncationStrategy, CompactionProvider
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Trigger when history exceeds 20 messages; trim to 10
strategy  = TruncationStrategy(max_n=20, compact_to=10, preserve_system=True)
compactor = CompactionProvider(before_strategy=strategy)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[compactor],
)

async def main() -> None:
    session = agent.create_session()
    for i in range(30):
        await agent.run(f"Question number {i}", session=session)
    print("Completed 30 turns with bounded history.")

asyncio.run(main())
```

### Example — Token-based truncation with custom tokenizer

```python
import asyncio
from agent_framework import (
    Agent, TruncationStrategy, CharacterEstimatorTokenizer, CompactionProvider
)
from agent_framework.openai import OpenAIChatClient

client    = OpenAIChatClient()
tokenizer = CharacterEstimatorTokenizer()

# Trigger at 4,000 estimated tokens; trim to 2,000
strategy  = TruncationStrategy(
    max_n=4000,
    compact_to=2000,
    tokenizer=tokenizer,
    preserve_system=True,
    preserve_first_user_group=True,   # always keep the original question
)
compactor = CompactionProvider(before_strategy=strategy)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a helpful research assistant.",
    context_providers=[compactor],
)

async def main() -> None:
    session = agent.create_session()
    response = await agent.run("Start a long research session.", session=session)
    print(response.text)

asyncio.run(main())
```

---

## 4. `ContextWindowCompactionStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`ContextWindowCompactionStrategy` is a sophisticated two-phase compaction pipeline derived directly from a model's context window size. It is the recommended strategy when you know your model's `max_context_window_tokens` and `max_output_tokens`.

**Phase 1 — Tool result eviction:** When included tokens exceed `tool_eviction_threshold × input_budget`, older tool-call groups are collapsed into summaries, retaining only the `keep_last_tool_call_groups` most recent groups.

**Phase 2 — Truncation:** After re-measuring tokens, if they still exceed `truncation_threshold × input_budget`, oldest non-system groups are removed.

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
    preserve_first_user_group: bool = False,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `max_context_window_tokens` | `int` | Model's maximum context window (e.g. 128,000). |
| `max_output_tokens` | `int` | Model's maximum output tokens (e.g. 16,384). |
| `tokenizer` | `TokenizerProtocol \| None` | Defaults to `CharacterEstimatorTokenizer` (4 chars/token). |
| `tool_eviction_threshold` | `float` | Fraction of input budget (0.0–1.0] at which tool eviction triggers. Default 0.5. |
| `truncation_threshold` | `float` | Fraction of input budget at which destructive truncation triggers. Must be ≥ `tool_eviction_threshold`. Default 0.8. |
| `keep_last_tool_call_groups` | `int` | Most recent tool-call groups retained verbatim during eviction. Default 4. |
| `preserve_first_user_group` | `bool` | Keep the first user group during truncation. Default `False`. |

The **input budget** is computed as: `max_context_window_tokens - max_output_tokens`.

### Example — GPT-4o context window setup

```python
import asyncio
from agent_framework import Agent, ContextWindowCompactionStrategy, CompactionProvider
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(model="gpt-4o")

# GPT-4o: 128k context window, 16k max output
strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=16_384,
    tool_eviction_threshold=0.5,
    truncation_threshold=0.8,
    keep_last_tool_call_groups=4,
    preserve_first_user_group=True,
)
compactor = CompactionProvider(before_strategy=strategy)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a research assistant with tool access.",
    context_providers=[compactor],
)

async def main() -> None:
    session = agent.create_session()
    response = await agent.run("Analyse the market trends.", session=session)
    print(response.text)

asyncio.run(main())
```

### Example — Custom tokenizer for precise token counting

```python
from agent_framework import (
    Agent, ContextWindowCompactionStrategy,
    CompactionProvider, TokenizerProtocol
)
from agent_framework.openai import OpenAIChatClient

# Plug in tiktoken for GPT-4o accurate counts
class TiktokenCounter:
    def __init__(self) -> None:
        import tiktoken
        self._enc = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

client    = OpenAIChatClient(model="gpt-4o")
tokenizer = TiktokenCounter()

strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=16_384,
    tokenizer=tokenizer,       # precise GPT-4o token counts
)
compactor = CompactionProvider(before_strategy=strategy)
agent = Agent(client=client, name="assistant",
              instructions="You are a helpful assistant.",
              context_providers=[compactor])
```

---

## 5. `LocalEvaluator`

**Module:** `agent_framework._evaluation` (re-exported via `agent_framework`)

`LocalEvaluator` runs agent evaluation checks locally — no cloud API calls or external services required. It implements the `Evaluator` protocol and accepts one or more `EvalCheck` functions. An `EvalItem` passes only when **all** checks pass; an item with no checks always fails.

> **Experimental:** `LocalEvaluator` is marked experimental under `ExperimentalFeature.EVALS`. Importing it from `agent_framework` works without additional configuration, but it may emit staged-API warnings and its interface can change between minor releases.

### Constructor

```python
LocalEvaluator(*checks: EvalCheck)
```

Each `EvalCheck` is a callable `(item: EvalItem) -> CheckResult | Awaitable[CheckResult]` that returns a `CheckResult(check_name, passed, reason)`.

### `.evaluate()` method

```python
async def evaluate(
    items: Sequence[EvalItem],
    *,
    eval_name: str = "Local Eval",
) -> EvalResults
```

Returns `EvalResults` with `.passed`, `.failed`, `.total`, `.per_evaluator`, and `.items` fields.

### Built-in check factory functions

The framework ships several built-in check factories (importable from `agent_framework`):

| Factory | Description |
|---|---|
| `keyword_check(keyword)` | Response text contains keyword (case-insensitive). |
| `tool_called_check(tool_name)` | Agent called the specified tool at least once. |
| `tool_call_args_match(tool_name, args)` | Agent called the tool with arguments matching the given dict. |

### Example — Keyword and tool call checks

```python
import asyncio
from agent_framework import (
    Agent, LocalEvaluator, EvalItem, evaluate_agent,
    keyword_check, tool_called_check, tool
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Sunny, 22°C in {location}"

weather_tool = get_weather  # @tool wraps the function into a FunctionTool

agent = Agent(
    client=client,
    name="weather-agent",
    instructions="Use the weather tool to answer questions.",
    tools=[weather_tool],
)

evaluator = LocalEvaluator(
    keyword_check("weather"),
    tool_called_check("get_weather"),
)

async def main() -> None:
    queries = ["What is the weather in London?", "How is the weather in Paris?"]
    results = await evaluate_agent(agent=agent, queries=queries, evaluators=evaluator)
    for r in results:
        print(f"{r.provider}: {r.passed}/{r.total} passed")
        for item in r.items:
            print(f"  Item: {item.status}")
            for score in item.scores:
                print(f"    {score.name}: {'✓' if score.passed else '✗'}")

asyncio.run(main())
```

### Example — Custom check function

```python
import asyncio
from agent_framework import LocalEvaluator, EvalItem, CheckResult, keyword_check

def length_check(min_chars: int):
    def check(item: EvalItem) -> CheckResult:
        response = item.response
        passed = len(response) >= min_chars
        return CheckResult(
            check_name=f"response_length_≥{min_chars}",
            passed=passed,
            reason=f"Got {len(response)} chars; expected ≥{min_chars}" if not passed else None,
        )
    return check

evaluator = LocalEvaluator(
    length_check(50),
    keyword_check("important"),
)
```

### Mixing with cloud evaluators

```python
from agent_framework import LocalEvaluator, keyword_check, evaluate_agent
# from agent_framework.foundry import FoundryEvals  # cloud evaluator

local = LocalEvaluator(keyword_check("answer"))
# foundry = FoundryEvals(project_client=..., model="gpt-4o")

results = await evaluate_agent(
    agent=agent,
    queries=["What is 2+2?"],
    evaluators=[local],   # add foundry here when available
)
```

---

## 6. `InlineSkill`

**Module:** `agent_framework._skills` (re-exported via `agent_framework`)

`InlineSkill` lets you define a reusable agent skill entirely in Python code, without requiring a filesystem-backed YAML file. It composes a `SkillFrontmatter` (metadata), instructions text, and optional resources and scripts into a single `Skill` object that can be registered with a `SkillsProvider`.

### Constructor

```python
InlineSkill(
    *,
    frontmatter: SkillFrontmatter,
    instructions: str,
    resources: Sequence[SkillResource] | None = None,
    scripts: Sequence[SkillScript] | None = None,
    argument_parser: SkillScriptArgumentParser | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `frontmatter` | `SkillFrontmatter` | Metadata: `name`, `description`, and optional `spec` fields. |
| `instructions` | `str` | The skill instructions injected into the agent's system prompt. |
| `resources` | `Sequence[SkillResource] \| None` | Pre-built resources (e.g. schema strings). |
| `scripts` | `Sequence[SkillScript] \| None` | Pre-built callable scripts. |
| `argument_parser` | `SkillScriptArgumentParser \| None` | Default argument transformer for `@skill.script` decorators. |

### Key methods

| Method | Returns | Notes |
|---|---|---|
| `get_content()` | `Coroutine[str]` | Returns synthesized XML content including name, instructions, resources, scripts. Result is cached after first call. |
| `get_resource(name)` | `SkillResource \| None` | Retrieve a registered resource by name. |
| `get_script(name)` | `SkillScript \| None` | Retrieve a registered script by name. |
| `@skill.resource` | decorator | Register a function as a resource callable. |
| `@skill.script` | decorator | Register a function as an invokable script. |

### Example — Database skill with an inline schema resource

```python
import asyncio
from agent_framework import (
    Agent, InlineSkill, SkillFrontmatter, InMemorySkillsSource, SkillsProvider
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Build the skill entirely in code
db_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="database-skill",
        description="Query and manage a relational database",
    ),
    instructions="""
Use this skill when the user asks about database records.
Always validate table names against the schema before querying.
Use parameterised queries to avoid SQL injection.
""".strip(),
)

@db_skill.resource
def schema() -> str:
    return """
CREATE TABLE users (id INT PRIMARY KEY, name TEXT, email TEXT);
CREATE TABLE orders (id INT, user_id INT, total DECIMAL);
""".strip()

@db_skill.script
def execute_query(query: str) -> str:
    # In production this would hit a real DB
    return f"Executed: {query} → 3 rows returned"

# Register with a skills provider
source   = InMemorySkillsSource(skills=[db_skill])
provider = SkillsProvider(source)  # `source` positional; accepts SkillsSource, Skill, or Sequence[Skill]

agent = Agent(
    client=client,
    name="db-agent",
    instructions="You have access to database skills.",
    context_providers=[provider],
)

async def main() -> None:
    response = await agent.run("List all users in the database.")
    print(response.text)

asyncio.run(main())
```

### Example — Skills with argument parsing

```python
from agent_framework import InlineSkill, SkillFrontmatter

def json_arg_parser(args: str) -> dict:
    import json
    return json.loads(args)

analysis_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="analysis-skill",
        description="Statistical analysis tools",
    ),
    instructions="Use these scripts for data analysis tasks.",
    argument_parser=json_arg_parser,  # scripts receive parsed dict, not raw string
)

@analysis_skill.script
def compute_mean(data: dict) -> float:
    values = data.get("values", [])
    return sum(values) / len(values) if values else 0.0

@analysis_skill.script
def compute_std(data: dict) -> float:
    import statistics
    values = data.get("values", [])
    return statistics.stdev(values) if len(values) > 1 else 0.0
```

---

## 7. `FileAccessProvider`

**Module:** `agent_framework._harness._file_access` (re-exported via `agent_framework`)

`FileAccessProvider` is a context provider that gives an agent CRUD, search, and line-replacement access to a shared `AgentFileStore`. It exposes 7 tools to the agent: `file_access_write`, `file_access_read`, `file_access_delete`, `file_access_ls`, `file_access_grep`, `file_access_replace`, and `file_access_replace_lines`.

> **Experimental:** `FileAccessProvider` is gated behind `ExperimentalFeature.HARNESS`.

### Constructor

```python
FileAccessProvider(
    store: AgentFileStore,
    *,
    source_id: str = "file_access",
    instructions: str | None = None,
    disable_write_tools: bool = False,
    disable_readonly_tool_approval: bool = False,
    disable_write_tool_approval: bool = False,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `store` | `AgentFileStore` | The backing file store (`InMemoryAgentFileStore` or `FileSystemAgentFileStore`). |
| `source_id` | `str` | Unique identifier for this provider. |
| `instructions` | `str \| None` | Custom instructions appended to the agent's system context. |
| `disable_write_tools` | `bool` | When `True`, only read-only tools are advertised (read, ls, grep). |
| `disable_readonly_tool_approval` | `bool` | Auto-approve read, ls, grep — no human confirmation required. |
| `disable_write_tool_approval` | `bool` | Auto-approve write, delete, replace, replace_lines. |

### Static approval rule methods

| Method | Returns | Notes |
|---|---|---|
| `read_only_tools_auto_approval_rule()` | `ToolApprovalRule` | Auto-approves only read-only tools; write tools still require approval. |
| `all_tools_auto_approval_rule()` | `ToolApprovalRule` | Auto-approves every file-access tool. |

### Example — Agent with in-memory file store

```python
import asyncio
from agent_framework import (
    Agent, FileAccessProvider, InMemoryAgentFileStore,
    ToolApprovalMiddleware,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

store    = InMemoryAgentFileStore()
provider = FileAccessProvider(
    store=store,
    disable_readonly_tool_approval=True,   # reads don't need approval
    disable_write_tool_approval=False,     # writes still require approval
)

agent = Agent(
    client=client,
    name="file-agent",
    instructions="You can read and write files.",
    context_providers=[provider],
    middleware=[ToolApprovalMiddleware(
        auto_approval_rules=[FileAccessProvider.read_only_tools_auto_approval_rule()],
    )],
)

async def main() -> None:
    # Pre-seed the store
    await store.write("notes.txt", "Important meeting notes.")

    response = await agent.run("Read my notes and summarise them.")
    print(response.text)

asyncio.run(main())
```

### Example — Filesystem-backed store with write restrictions

```python
import asyncio
from pathlib import Path
from agent_framework import (
    Agent, FileAccessProvider, FileSystemAgentFileStore,
    ToolApprovalMiddleware
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

store = FileSystemAgentFileStore(root_directory=Path("./workspace"))
provider = FileAccessProvider(
    store=store,
    source_id="workspace_files",
    disable_write_tools=True,           # read-only agent
    disable_readonly_tool_approval=True,
)

agent = Agent(
    client=client,
    name="read-only-agent",
    instructions="You can only read workspace files.",
    context_providers=[provider],
    middleware=[ToolApprovalMiddleware()],
)

async def main() -> None:
    response = await agent.run("List all files in the workspace.")
    print(response.text)

asyncio.run(main())
```

---

## 8. `MemoryContextProvider`

**Module:** `agent_framework._harness._memory` (re-exported via `agent_framework`)

`MemoryContextProvider` implements cross-session durable memory for agents. Each session it injects a `MEMORY.md` file with recent topic pointers into the agent's context, extracts new durable facts from the conversation transcript at session end, and periodically consolidates topic files to remove noise. It also exposes topic memory tools so the agent can read and write specific memory topics.

> **Experimental:** `MemoryContextProvider` is gated behind `ExperimentalFeature.HARNESS`.

### Constructor (key parameters)

```python
MemoryContextProvider(
    recent_turns: int = 0,
    load_tool_turns: bool = True,
    *,
    store: MemoryStore,
    source_id: str = "memory",
    context_prompt: str | None = None,
    index_line_limit: int = 200,
    index_line_length: int = 150,
    selection_limit: int = 3,
    max_extractions: int = 5,
    consolidation_interval: timedelta = timedelta(days=1),
    consolidation_min_sessions: int = 5,
    extraction_prompt: str = ...,      # see source for default
    consolidation_prompt: str = ...,   # see source for default
    consolidation_client: SupportsChatGetResponse | None = None,
    history_message_filter: HistoryMessageFilter | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `store` | `MemoryStore` | Backing store for the index, topics, and transcripts. |
| `recent_turns` | `int` | Inject this many recent transcript turns alongside durable memory. 0 = durable memory only. |
| `load_tool_turns` | `bool` | Include tool-call turns in the recent-turn window. |
| `selection_limit` | `int` | Max topic files loaded per turn (avoids bloating context). |
| `max_extractions` | `int` | Max new memory items extracted per turn. |
| `consolidation_interval` | `timedelta` | Minimum gap between consolidation runs. |
| `consolidation_min_sessions` | `int` | Consolidation waits until this many sessions have accumulated. |
| `consolidation_client` | `SupportsChatGetResponse \| None` | Separate (cheaper) model for consolidation passes. |

### Example — File-backed persistent memory

```python
import asyncio
from pathlib import Path
from agent_framework import (
    Agent, AgentSession, MemoryContextProvider, MemoryFileStore
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# MemoryFileStore is the concrete file-backed implementation of MemoryStore.
# owner_state_key="user_id" tells the store to look up session.state["user_id"]
# at runtime to scope memory per user — both sessions must supply the same value.
store = MemoryFileStore(
    base_path=Path("./agent_memory"),
    owner_state_key="user_id",
)
provider = MemoryContextProvider(
    store=store,
    recent_turns=3,          # inject last 3 turns alongside durable facts
    max_extractions=5,       # extract up to 5 facts per session
    selection_limit=4,       # load up to 4 topic files per turn
)

agent = Agent(
    client=client,
    name="memory-agent",
    instructions="You are a personal assistant with long-term memory.",
    context_providers=[provider],
)

async def main() -> None:
    # session_1 and session_2 share owner "alice" via state["user_id"]
    session_1 = AgentSession(session_id="session-1")
    session_1.state["user_id"] = "alice"
    await agent.run("My name is Alice and I prefer Python.", session=session_1)

    # New session with the same owner ID — agent recalls Alice's preference
    session_2 = AgentSession(session_id="session-2")
    session_2.state["user_id"] = "alice"
    response = await agent.run("What do you know about me?", session=session_2)
    print(response.text)

asyncio.run(main())
```

### Example — Separate consolidation model

```python
from datetime import timedelta
from agent_framework import Agent, MemoryContextProvider, MemoryFileStore
from agent_framework.openai import OpenAIChatClient
from pathlib import Path

main_client          = OpenAIChatClient(model="gpt-4o")
consolidation_client = OpenAIChatClient(model="gpt-4o-mini")  # cheaper model

store    = MemoryFileStore(base_path=Path("./memory"), owner_state_key="user_id")
provider = MemoryContextProvider(
    store=store,
    consolidation_client=consolidation_client,  # use mini for cleanup
    consolidation_interval=timedelta(hours=6),  # consolidate every 6 hours
    consolidation_min_sessions=3,               # after at least 3 sessions
)

agent = Agent(
    client=main_client,
    name="assistant",
    instructions="You are a helpful assistant with memory.",
    context_providers=[provider],
)
```

---

## 9. `FileHistoryProvider`

**Module:** `agent_framework._sessions` (re-exported via `agent_framework`)

`FileHistoryProvider` persists conversation history to disk, one append-only file per session. The default format is JSON Lines (one JSON object per message line); use `serialization_format="msgpack"` for binary MessagePack records. Both formats use `msgspec` for speed.

> **Experimental:** `FileHistoryProvider` is gated behind `ExperimentalFeature.FILE_HISTORY`.

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
    serialization_format: Literal["json", "msgpack"] = "json",
    dumps: JsonDumps | None = None,
    loads: JsonLoads | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `storage_path` | `str \| Path` | Directory where per-session files are stored. |
| `load_messages` | `bool` | Load stored messages before each invocation. |
| `store_inputs` | `bool` | Persist user input messages. |
| `store_context_messages` | `bool` | Persist injected context messages (e.g. from `CompactionProvider`). |
| `store_context_from` | `set[str] \| None` | When set, only persist context from these `source_id` values. |
| `store_outputs` | `bool` | Persist agent response messages. |
| `skip_excluded` | `bool` | When `True`, `get_messages()` omits messages with `_excluded=True`. |
| `serialization_format` | `"json" \| "msgpack"` | Storage format. `"msgpack"` is faster for large histories. |

### Key methods

| Method | Returns | Notes |
|---|---|---|
| `before_run(session_ctx)` | `Coroutine` | Loads stored messages into the session context before the agent runs. |
| `after_run(session_ctx)` | `Coroutine` | Appends new messages to the session file after the agent runs. |
| `get_messages(session_id)` | `Coroutine[list[Message]]` | Returns stored messages for a given session. |
| `save_messages(session_id, messages)` | `Coroutine` | Persists a message list for a session. |

### Example — Persistent conversation history

```python
import asyncio
from agent_framework import Agent, FileHistoryProvider, AgentSession
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

history = FileHistoryProvider(
    storage_path="./chat_history",
    store_inputs=True,
    store_outputs=True,
    skip_excluded=True,     # don't replay compaction-excluded messages
)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[history],
)

async def main() -> None:
    session = AgentSession(session_id="user-42")

    # First run — question
    r1 = await agent.run("What is the capital of France?", session=session)
    print("Turn 1:", r1.text)

    # Second run — agent recalls the session history automatically
    r2 = await agent.run("And what is its population?", session=session)
    print("Turn 2:", r2.text)   # agent knows we're talking about Paris

asyncio.run(main())
```

### Example — MessagePack format for high-throughput logging

```python
from agent_framework import Agent, FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

client  = OpenAIChatClient()
history = FileHistoryProvider(
    storage_path="./history_msgpack",
    serialization_format="msgpack",  # binary, faster I/O
    store_context_messages=True,     # also persist injected context
    store_context_from={"file_history", "memory"},  # only these providers
)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[history],
)
```

### Example — Selective context persistence

```python
from agent_framework import Agent, FileHistoryProvider, CompactionProvider
from agent_framework import SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Only store the original inputs and outputs, not compaction-injected messages
history = FileHistoryProvider(
    storage_path="./history",
    store_context_messages=False,  # skip compactor-injected messages
    skip_excluded=True,            # skip messages marked excluded by compaction
)

compactor = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=5)
)

agent = Agent(
    client=client,
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[history, compactor],
)
```

---

## 10. `MCPWebsocketTool`

**Module:** `agent_framework._mcp` (re-exported via `agent_framework`)

`MCPWebsocketTool` connects to a WebSocket-based MCP server, loading its tools and prompts so they are exposed to the agent. It is the WebSocket counterpart to `MCPStdioTool` (subprocess) and `MCPStreamableHTTPTool` (HTTP SSE). Use it for real-time services that maintain a persistent WebSocket connection.

### Constructor

```python
MCPWebsocketTool(
    name: str,
    url: str,
    *,
    tool_name_prefix: str | None = None,
    load_tools: bool = True,
    parse_tool_results: Callable[[types.CallToolResult], str | list[Content]] | None = None,
    load_prompts: bool = True,
    parse_prompt_results: Callable[[types.GetPromptResult], str] | None = None,
    request_timeout: int | None = None,
    session: ClientSession | None = None,
    description: str | None = None,
    approval_mode: Literal["always_require", "never_require"] | MCPSpecificApproval | None = None,
    allowed_tools: Collection[str] | None = None,
    use_progressive_disclosure: bool = False,
    always_load: Collection[str] | None = None,
    client: SupportsChatGetResponse | None = None,
    sampling_approval_callback: SamplingApprovalCallback | None = None,
    sampling_max_tokens: int | None = 4096,
    sampling_max_requests: int | None = 25,
    additional_properties: dict[str, Any] | None = None,
    task_options: MCPTaskOptions | None = None,
    **kwargs: Any,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `name` | `str` | Logical name for the MCP server. |
| `url` | `str` | WebSocket URL, e.g. `wss://service.example.com/mcp`. |
| `tool_name_prefix` | `str \| None` | Prepend a prefix to all exposed tool names to avoid collisions. |
| `load_tools` | `bool` | Fetch and expose MCP tools. |
| `load_prompts` | `bool` | Fetch and expose MCP prompts. |
| `allowed_tools` | `Collection[str] \| None` | Whitelist of MCP tool names to expose (others are hidden). |
| `approval_mode` | `str \| MCPSpecificApproval \| None` | Defaults to `None` (inherits server-side approval behavior). Set `"always_require"` to gate every tool call, or `"never_require"` to skip confirmation. |
| `use_progressive_disclosure` | `bool` | Load full tool specs lazily when called, keeping initial context small. |
| `request_timeout` | `int \| None` | Per-request timeout in seconds. |
| `sampling_max_tokens` | `int \| None` | Max tokens for MCP server-side sampling requests. Default 4096. |

### Key methods

| Method | Returns | Notes |
|---|---|---|
| `connect()` | `Coroutine` | Open the WebSocket connection to the MCP server. |
| `close()` | `Coroutine` | Close the connection. |
| `load_tools()` | `Coroutine` | Reload tools from the server (e.g. after reconnect). |
| `load_prompts()` | `Coroutine` | Reload prompts from the server. |
| `call_tool(name, arguments)` | `Coroutine` | Invoke a tool directly. |
| `get_prompt(name, arguments)` | `Coroutine` | Invoke a prompt directly. |
| `get_mcp_client()` | `ClientSession` | Access the raw MCP client session. |

`MCPWebsocketTool` is an async context manager — use it in `async with` to manage the connection lifecycle.

### Example — Real-time stock data via WebSocket MCP

```python
import asyncio
from agent_framework import Agent, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    mcp_tool = MCPWebsocketTool(
        name="market-data",
        url="wss://market.example.com/mcp",
        description="Real-time stock market data and analysis",
        approval_mode="never_require",   # auto-approve for unattended use
        request_timeout=30,
    )

    async with mcp_tool:
        agent = Agent(
            client=client,
            name="trader-agent",
            instructions="Use the market data tools to answer questions.",
            tools=[mcp_tool],
        )
        response = await agent.run("What is the current price of AAPL?")
        print(response.text)

asyncio.run(main())
```

### Example — Multiple WebSocket MCP servers with prefixes

```python
import asyncio
from agent_framework import Agent, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    stocks = MCPWebsocketTool(
        name="stocks",
        url="wss://stocks.example.com/mcp",
        tool_name_prefix="stocks",     # tools become: stocks_get_price, etc.
        approval_mode="never_require",
    )
    crypto = MCPWebsocketTool(
        name="crypto",
        url="wss://crypto.example.com/mcp",
        tool_name_prefix="crypto",     # tools become: crypto_get_price, etc.
        approval_mode="never_require",
    )

    async with stocks, crypto:
        agent = Agent(
            client=client,
            name="multi-market-agent",
            instructions="You have access to both stock and crypto market data.",
            tools=[stocks, crypto],
        )
        response = await agent.run("Compare the performance of AAPL and Bitcoin today.")
        print(response.text)

asyncio.run(main())
```

### Example — Selective tool loading with `allowed_tools`

```python
import asyncio
from agent_framework import Agent, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    mcp_tool = MCPWebsocketTool(
        name="data-service",
        url="wss://data.example.com/mcp",
        allowed_tools={"get_data", "list_tables"},  # hide admin tools
        approval_mode="never_require",
        use_progressive_disclosure=True,  # load specs lazily
    )

    async with mcp_tool:
        agent = Agent(
            client=client,
            name="data-agent",
            instructions="You can query data. Admin operations are restricted.",
            tools=[mcp_tool],
        )
        response = await agent.run("What tables are available?")
        print(response.text)

asyncio.run(main())
```

---

## Summary

| Class | Module | Primary use |
|---|---|---|
| `WorkflowBuilder` | `_workflows._workflow_builder` | Construct typed multi-agent workflows with any routing pattern |
| `SlidingWindowStrategy` | `_compaction` | Keep only the N most-recent conversation groups |
| `TruncationStrategy` | `_compaction` | Oldest-first eviction by token count or message count |
| `ContextWindowCompactionStrategy` | `_compaction` | Two-phase tool-eviction + truncation sized to model limits |
| `LocalEvaluator` | `_evaluation` | Run local check functions against agent outputs — no cloud required |
| `InlineSkill` | `_skills` | Define reusable skills in code with resources and scripts |
| `FileAccessProvider` | `_harness._file_access` | Give agents CRUD/grep access to a shared `AgentFileStore` |
| `MemoryContextProvider` | `_harness._memory` | Cross-session durable memory with automatic extraction and consolidation |
| `FileHistoryProvider` | `_sessions` | Persist conversation history to disk per session (JSONL or MessagePack) |
| `MCPWebsocketTool` | `_mcp` | Connect to WebSocket MCP servers for real-time data and tools |

Install the full package:

```bash
pip install agent-framework==1.17.0
```

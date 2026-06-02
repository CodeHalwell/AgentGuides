---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 3"
description: "Source-verified deep dives into 10 classes from agent-framework 1.7.0: BackgroundAgentsProvider, MemoryContextProvider + MemoryFileStore, TodoProvider + TodoFileStore, AgentModeProvider, SummarizationStrategy, ContextWindowCompactionStrategy, SlidingWindowStrategy, SelectiveToolCallCompactionStrategy, WorkflowViz, and MCPStreamableHTTPTool + MCPWebsocketTool."
framework: microsoft-agent-framework
language: python
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 3

Verified against **agent-framework 1.7.0** (installed, May 2026). Every constructor signature, parameter
description, and code example in this document was derived from the installed package source at
`/usr/local/lib/python3.11/dist-packages/agent_framework/`. No API name has been guessed or inferred
from documentation alone.

**Version note:** `agent-framework` 1.7.0 is the latest release. Docs in Vols. 1 and 2 reference 1.6.0;
the API surface is backward-compatible — only additive changes were made. Upgrade with:

```bash
pip install --upgrade agent-framework
```

Ten classes are covered in this volume, chosen to complement
[Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) and
[Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/). They
span four new capability areas new or significantly expanded in 1.7.0: **harness providers**
(concurrent sub-agents, persistent memory, todo planning, mode switching), **advanced compaction strategies**,
**workflow visualization**, and **additional MCP transports**.

> All harness classes (`BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`,
> `AgentModeProvider`) are decorated `@experimental` — they trigger an `ExperimentalWarning` on import.
> Suppress with `import warnings; warnings.filterwarnings("ignore", category=ExperimentalWarning)` if you
> want clean output during development.

---

## Table of Contents

1. [`BackgroundAgentsProvider` + `BackgroundTaskInfo` / `BackgroundTaskStatus`](#1-backgroundagentsprovider--backgroundtaskinfo--backgroundtaskstatus)
2. [`MemoryContextProvider` + `MemoryFileStore` + `MemoryStore`](#2-memorycontextprovider--memoryfilestore--memorystore)
3. [`TodoProvider` + `TodoItem` + `TodoFileStore`](#3-todoprovider--todoitem--todofilestore)
4. [`AgentModeProvider` + `get_agent_mode` / `set_agent_mode`](#4-agentmodeprovider--get_agent_mode--set_agent_mode)
5. [`SummarizationStrategy`](#5-summarizationstrategy)
6. [`ContextWindowCompactionStrategy`](#6-contextwindowcompactionstrategy)
7. [`SlidingWindowStrategy`](#7-slidingwindowstrategy)
8. [`SelectiveToolCallCompactionStrategy`](#8-selectivetoolcallcompactionstrategy)
9. [`WorkflowViz`](#9-workflowviz)
10. [`MCPStreamableHTTPTool` + `MCPWebsocketTool`](#10-mcpstreamablehttptool--mcpwebsockettool)

---

## 1. `BackgroundAgentsProvider` + `BackgroundTaskInfo` / `BackgroundTaskStatus`

**Source:** `agent_framework/_harness/_background_agents.py`

`BackgroundAgentsProvider` is a `ContextProvider` that gives an agent the ability to **spawn, monitor, and
collect results from concurrent sub-agents** without blocking the parent turn. Tasks run in their own
`AgentSession` via `asyncio.create_task`, so the parent agent can start multiple background tasks and only
`await` them when it needs results.

> **Experimental** — imports trigger `ExperimentalWarning`.

### How it works

The provider injects six tools into the agent's context at each `before_run`:

| Tool | What it does |
|---|---|
| `background_agents_start_task(agent_name, input, description)` | Launches a new background task; returns task ID immediately |
| `background_agents_wait_for_first_completion(task_ids)` | Awaits the first task in `task_ids` to finish |
| `background_agents_get_task_results(task_id)` | Returns text output of a completed task |
| `background_agents_get_all_tasks()` | Lists all tasks with status, agent name, and description |
| `background_agents_continue_task(task_id, text)` | Resumes a completed task on the same session |
| `background_agents_clear_completed_task(task_id)` | Frees memory for a done/failed task |

Task state (`BackgroundTaskInfo`) is serialized into `AgentSession.state` (keyed by `source_id`), so it
survives `CompactionProvider` rewrites. The live `asyncio.Task` handles are **not** serialized — if the
provider instance is lost (e.g. process restart), orphaned tasks are marked `LOST`.

### Constructor

```python
BackgroundAgentsProvider(
    agents: Sequence[SupportsAgentRun],
    *,
    source_id: str = "background_agents",
    instructions: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `agents` | Child agents available for delegation. Every agent **must** have a non-empty, unique (case-insensitive) `name`. |
| `source_id` | State key in `AgentSession.state`. Override when running multiple providers. |
| `instructions` | Custom instruction block; may include `{background_agents}` placeholder, which is replaced with a generated list of available agents + descriptions. |

### `BackgroundTaskStatus` enum

```python
class BackgroundTaskStatus(str, Enum):
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    LOST      = "lost"      # asyncio.Task handle was lost (e.g. process restart)
```

### Example 1: parallel research with two specialist sub-agents

```python
import asyncio
import warnings
from agent_framework import Agent, BackgroundAgentsProvider
from agent_framework.openai import OpenAIChatClient
from agent_framework.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


async def main() -> None:
    client = OpenAIChatClient()

    finance_agent = Agent(
        client=client,
        name="FinanceAgent",
        description="Analyses financial data and market trends",
        instructions="You are a finance expert. Answer concisely with data-driven insights.",
    )

    risk_agent = Agent(
        client=client,
        name="RiskAgent",
        description="Assesses business and technical risks",
        instructions="You are a risk analyst. Identify risks and suggest mitigations concisely.",
    )

    orchestrator = Agent(
        client=client,
        name="Orchestrator",
        instructions=(
            "You coordinate research by delegating to specialist agents. "
            "Always start tasks in parallel, wait for completion, then synthesize results."
        ),
        context_providers=[
            BackgroundAgentsProvider(agents=[finance_agent, risk_agent]),
        ],
    )

    session = orchestrator.create_session()
    response = await orchestrator.run(
        "Research the acquisition of Contoso Ltd. Start finance and risk analysis in parallel.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 2: waiting for multiple tasks, then synthesizing

```python
# The agent will autonomously call these tools in sequence:
#
#   1. background_agents_start_task("FinanceAgent", "Analyse Q4 revenue", "Finance analysis")
#      → "Background task 1 started on agent 'FinanceAgent'."
#
#   2. background_agents_start_task("RiskAgent", "Identify key risks", "Risk review")
#      → "Background task 2 started on agent 'RiskAgent'."
#
#   3. background_agents_wait_for_first_completion([1, 2])
#      → "Task 1 finished with status: completed."
#
#   4. background_agents_wait_for_first_completion([2])   # wait for second
#      → "Task 2 finished with status: completed."
#
#   5. background_agents_get_task_results(1)   # retrieve finance results
#   6. background_agents_get_task_results(2)   # retrieve risk results
#
#   7. background_agents_clear_completed_task(1)
#   8. background_agents_clear_completed_task(2)
#
#   9.  [synthesize and return final answer]
```

### Example 3: checking `BackgroundTaskInfo` from outside the agent

```python
from agent_framework import AgentSession, BackgroundAgentsProvider, BackgroundTaskStatus

async def check_tasks(session: AgentSession) -> None:
    state = session.state.get("background_agents", {})
    tasks_raw = state.get("tasks", [])
    for raw in tasks_raw:
        from agent_framework._harness._background_agents import BackgroundTaskInfo
        info = BackgroundTaskInfo.from_dict(raw)
        print(f"Task {info.id} [{info.status.value}] ({info.agent_name}): {info.description}")
        if info.status == BackgroundTaskStatus.COMPLETED:
            print(f"  Result: {info.result_text!r}")
```

### Key points

- Each background task gets its own `AgentSession` — history does not leak between tasks.
- `background_agents_wait_for_first_completion` uses `asyncio.wait(return_when=FIRST_COMPLETED)` — it truly awaits without blocking the event loop.
- If a task fails, `BackgroundTaskInfo.error_text` holds the exception string.
- `background_agents_continue_task` reuses the existing `AgentSession`, so the sub-agent retains its history.
- Tasks with status `LOST` indicate the provider instance was replaced (restart); start a fresh task instead.

---

## 2. `MemoryContextProvider` + `MemoryFileStore` + `MemoryStore`

**Source:** `agent_framework/_harness/_memory.py`

The memory harness gives agents **durable long-term semantic memory** across sessions. It works in two layers:

- **`MemoryFileStore`** — writes raw session transcripts to disk and extracts structured topic files.
- **`MemoryContextProvider`** — a `ContextProvider` that loads relevant topic files before each run and injects them into context.

Together they implement a **MEMORY.md** index + per-topic files pattern where the agent sees a compact summary
and can request individual topic files with the `memory_load_topic` tool.

> **Experimental** in 1.7.0 — imports trigger `ExperimentalWarning`.

### `MemoryFileStore` constructor

```python
MemoryFileStore(
    storage_path: str | Path,
    client: SupportsChatGetResponse,
    *,
    source_id: str = "memory",
    index_file_name: str = "MEMORY.md",
    topics_directory_name: str = "topics",
    transcripts_directory_name: str = "transcripts",
    state_file_name: str = "state.json",
    index_line_limit: int = 200,
    index_line_length: int = 150,
    selection_limit: int = 3,
    consolidation_min_sessions: int = 5,
    max_extractions: int = 5,
    consolidation_interval: timedelta = timedelta(hours=24),
)
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `storage_path` | required | Root directory for memory files (per-user or per-session sub-dirs recommended) |
| `client` | required | Chat client used to run extraction and consolidation LLM calls |
| `index_line_limit` | 200 | Max lines in MEMORY.md before triggering consolidation |
| `selection_limit` | 3 | Max topic files auto-loaded per turn |
| `consolidation_min_sessions` | 5 | Minimum sessions before overnight consolidation |

### `MemoryContextProvider` constructor

```python
MemoryContextProvider(
    store: MemoryStore,
    *,
    source_id: str = "memory",
    context_prompt: str = "## Memory\nUse MEMORY.md and the loaded topic files when they are relevant.",
)
```

### Example 1: persistent memory across sessions

```python
import asyncio
import warnings
from pathlib import Path
from agent_framework import Agent, MemoryContextProvider
from agent_framework._harness._memory import MemoryFileStore
from agent_framework.openai import OpenAIChatClient
from agent_framework.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


async def main() -> None:
    client = OpenAIChatClient()

    memory_store = MemoryFileStore(
        storage_path=Path("./agent_memory"),
        client=client,
    )
    memory_provider = MemoryContextProvider(store=memory_store)

    agent = Agent(
        client=client,
        name="PersonalAssistant",
        instructions=(
            "You are a personal assistant. Use your memory to recall facts about the user. "
            "After each conversation, important facts will be saved for later."
        ),
        context_providers=[memory_provider],
    )

    # First session — agent learns user preferences
    session1 = agent.create_session()
    await agent.run("My name is Alice and I prefer Python over TypeScript.", session=session1)
    await memory_provider.after_run(
        agent=agent, session=session1, context=None, state={}  # type: ignore
    )

    # Second session — agent recalls preferences from memory
    session2 = agent.create_session()
    response = await agent.run("What programming language should I use for this project?", session=session2)
    print(response.text)  # Should reference Alice's Python preference


asyncio.run(main())
```

### Example 2: scoping memory per user

```python
from pathlib import Path

def make_memory_store(user_id: str, client) -> MemoryFileStore:
    return MemoryFileStore(
        storage_path=Path(f"./memory/{user_id}"),
        client=client,
        consolidation_min_sessions=3,
        selection_limit=5,
    )
```

### Key points

- Memory extraction runs after sessions complete (call `after_run` on the store, or configure `CompactionProvider` to trigger it automatically).
- `MEMORY.md` holds a compact index (≤ `index_line_limit` lines); individual topic files hold details.
- The `memory_load_topic(topic)` tool is injected automatically — the agent requests specific topic files when it needs them.
- Consolidation merges transcript-derived facts across sessions every `consolidation_interval` (default 24 h) once `consolidation_min_sessions` sessions have accumulated.
- Storage path **should be scoped per user** to avoid cross-user memory bleed.

---

## 3. `TodoProvider` + `TodoItem` + `TodoFileStore`

**Source:** `agent_framework/_harness/_todo.py`

`TodoProvider` gives an agent a **structured task list** it can create, tick off, and query during long
multi-step work. State is persisted through `TodoStore` implementations:

- **`TodoSessionStore`** (default) — stores in `AgentSession.state`; lost when the session is garbage-collected.
- **`TodoFileStore`** — writes one JSON file per session; survives process restarts.

> **Experimental** in 1.7.0 — imports trigger `ExperimentalWarning`.

### `TodoItem` dataclass

```python
TodoItem(
    id: int,
    title: str,
    description: str | None = None,
    is_complete: bool = False,
)
```

### Injected tools

| Tool | Signature | Description |
|---|---|---|
| `todos_add` | `todos: list[{title, description?}]` | Add one or many todo items |
| `todos_complete` | `items: list[{id, reason}]` | Mark items done with a reason |
| `todos_remove` | `ids: list[int]` | Delete items no longer needed |
| `todos_get_remaining` | `()` | Returns incomplete items as JSON |
| `todos_get_all` | `()` | Returns all items (complete + incomplete) |

### `TodoProvider` constructor

```python
TodoProvider(
    source_id: str = "todo",
    *,
    instructions: str | None = None,
    store: TodoStore | None = None,   # defaults to TodoSessionStore()
)
```

### `TodoFileStore` constructor

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

Set `owner_state_key` to a session-state key that holds a user ID — the store then creates per-user
sub-directories automatically (using URL-safe base64 encoding for path safety).

### Example 1: agent with file-backed todo tracking

```python
import asyncio
import warnings
from pathlib import Path
from agent_framework import Agent, TodoProvider
from agent_framework._harness._todo import TodoFileStore
from agent_framework.openai import OpenAIChatClient
from agent_framework.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


async def main() -> None:
    client = OpenAIChatClient()

    agent = Agent(
        client=client,
        name="ProjectManager",
        instructions=(
            "You are a project manager. For any complex request, break it into todo items, "
            "execute them systematically, and mark each as complete when done."
        ),
        context_providers=[
            TodoProvider(
                store=TodoFileStore(base_path=Path("./todos")),
            ),
        ],
    )

    session = agent.create_session()
    response = await agent.run(
        "Plan and track the steps to set up a new Python microservice with FastAPI, "
        "Docker, and CI/CD.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 2: reading todo state from outside the agent

```python
from agent_framework import AgentSession
from agent_framework._harness._todo import TodoSessionStore, TodoItem

async def list_todos(session: AgentSession) -> list[TodoItem]:
    store = TodoSessionStore()
    return await store.load_items(session, source_id="todo")


async def main() -> None:
    # ... agent runs ...
    todos = await list_todos(session)
    for item in todos:
        status = "✓" if item.is_complete else "○"
        print(f"  {status} [{item.id}] {item.title}")
```

### Example 3: per-user file-backed todos

```python
from agent_framework._harness._todo import TodoFileStore

# Store a user_id in session state before running the agent:
#   session.state["user_id"] = "alice@example.com"

file_store = TodoFileStore(
    base_path="./todos",
    owner_state_key="user_id",   # reads session.state["user_id"]
)
# Files written to ./todos/<b64-encoded-user-id>/todos/<session-id>/todos.todo.json
```

### Key points

- Per-session `asyncio.Lock` prevents concurrent tool calls from corrupting the todo list.
- Writes are atomic: `TodoFileStore` writes to a `.tmp` file then `os.replace()` — crash-safe.
- `TodoCompleteInput.reason` is required — the model must explain how an item was finished.
- Provide `TodoProvider` alongside `AgentModeProvider` to combine plan/execute mode with task tracking.

---

## 4. `AgentModeProvider` + `get_agent_mode` / `set_agent_mode`

**Source:** `agent_framework/_harness/_mode.py`

`AgentModeProvider` adds an explicit **operating mode** to the agent. The default configuration ships two
modes (`"plan"` and `"execute"`) but any set of named modes with custom descriptions can be configured.
Mode state is persisted in `AgentSession.state`.

> **Experimental** in 1.7.0 — imports trigger `ExperimentalWarning`.

### Default modes

| Mode | Behavior |
|---|---|
| `plan` | Interactive — analyze requests, ask clarifying questions, create todos, write plan to memory, get user approval before switching to execute |
| `execute` | Autonomous — carry out the approved plan using best judgment, no questions, mark todos complete as work progresses |

### Constructor

```python
AgentModeProvider(
    source_id: str = "agent_mode",
    *,
    default_mode: str | None = None,           # first entry of mode_descriptions if None
    mode_descriptions: Mapping[str, str] | None = None,  # defaults to plan + execute
    instructions: str | None = None,           # custom instruction text; supports {available_modes} and {current_mode}
)
```

### Injected tools

| Tool | Description |
|---|---|
| `mode_get()` | Returns `{"mode": "<current>"}` |
| `mode_set(mode)` | Switches mode; returns `{"mode": "<new>", "message": "Mode changed to '<new>'"}` |

### Module-level helpers

```python
# Read mode from outside the agent (e.g. API handler, UI)
current = get_agent_mode(session, source_id="agent_mode", default_mode="plan")

# Write mode from outside the agent (injects a user-message notification on next turn)
set_agent_mode(session, "execute", source_id="agent_mode")
```

`set_agent_mode` stores the previous mode so the provider can inject a **user-role notification message**
on the next `before_run`. This is critical — system instructions alone are insufficient to redirect a model
that has already seen a `mode_set` tool call earlier in its chat history.

### Example 1: plan/execute mode with todo tracking

```python
import asyncio
import warnings
from agent_framework import Agent, AgentModeProvider, TodoProvider
from agent_framework.openai import OpenAIChatClient
from agent_framework.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


async def main() -> None:
    client = OpenAIChatClient()

    agent = Agent(
        client=client,
        name="ResearchAgent",
        instructions="You research topics thoroughly. Follow your mode instructions carefully.",
        context_providers=[
            AgentModeProvider(),   # starts in "plan" mode
            TodoProvider(),        # task tracking
        ],
    )

    session = agent.create_session()

    # Turn 1: agent enters plan mode — asks clarifications, creates a plan
    r1 = await agent.run(
        "Research the competitive landscape for LLM-based document processing.",
        session=session,
    )
    print(r1.text)

    # User approves the plan; switch to execute mode externally
    from agent_framework import set_agent_mode
    set_agent_mode(session, "execute")

    # Turn 2: agent sees mode-change notification, switches to autonomous execution
    r2 = await agent.run("Great plan! Please proceed.", session=session)
    print(r2.text)


asyncio.run(main())
```

### Example 2: custom modes (research → write → review)

```python
from agent_framework import AgentModeProvider

provider = AgentModeProvider(
    default_mode="research",
    mode_descriptions={
        "research": (
            "Gather information from available sources. Ask questions if scope is unclear. "
            "End by summarizing findings and asking to switch to 'write' mode."
        ),
        "write": (
            "Draft the document based on research findings. Work autonomously. "
            "End by asking user to review before switching to 'review' mode."
        ),
        "review": (
            "Check the draft for accuracy, completeness, and clarity. "
            "Annotate issues inline and suggest corrections."
        ),
    },
)
```

### Key points

- The mode value is stored in `session.state["agent_mode"]["current_mode"]` — you can read it directly.
- `set_agent_mode` bypasses the agent's own `mode_set` tool call — use it from external code (API handlers, UI).
- The provider always calls `mode_get` before injecting instructions so the current mode is fresh.
- Combine with `TodoProvider` for plan-track-execute workflows, and `MemoryContextProvider` for persistent planning state.

---

## 5. `SummarizationStrategy`

**Source:** `agent_framework/_compaction.py`

`SummarizationStrategy` reduces context window usage by **calling an LLM to produce rolling summaries** of
older message groups. Rather than discarding messages (like `TruncationStrategy`), it replaces them with
a compact linked summary that preserves semantic continuity.

### Constructor

```python
SummarizationStrategy(
    *,
    client: SupportsChatGetResponse,
    target_count: int = 4,
    threshold: int | None = 2,
    prompt: str | None = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `client` | required | Chat client used to call the summarization LLM |
| `target_count` | 4 | Target number of non-system messages to retain after summarization |
| `threshold` | 2 | Extra messages allowed above `target_count` before triggering (triggers at `target_count + threshold`) |
| `prompt` | built-in | Summarization instruction; default preserves goals, decisions, and open items |

**Trigger condition:** `included_non_system_message_count > target_count + threshold`

When triggered:
1. The strategy identifies the *oldest* groups to summarize (leaving `target_count` newest).
2. It calls the LLM with the messages to summarize.
3. It marks original messages as `excluded=True, reason="summarized"` and inserts the summary.
4. Bidirectional metadata links (summary → originals, originals → summary) are written to `message.additional_properties`.

### Example 1: rolling summary with `CompactionProvider`

```python
import asyncio
from agent_framework import Agent, CompactionProvider
from agent_framework._compaction import SummarizationStrategy
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    compaction_provider = CompactionProvider(
        before_strategy=SummarizationStrategy(
            client=client,
            target_count=6,   # keep 6 recent non-system messages
            threshold=3,      # trigger when 9+ messages accumulate
        ),
    )

    agent = Agent(
        client=client,
        instructions="You are a helpful assistant for long research conversations.",
        context_providers=[compaction_provider],
    )

    session = agent.create_session()
    for i in range(20):
        response = await agent.run(f"Question {i + 1}: What is the capital of France?", session=session)
        print(f"Turn {i + 1}: {response.text[:60]}…")


asyncio.run(main())
```

### Example 2: custom summarization prompt

```python
from agent_framework._compaction import SummarizationStrategy

strategy = SummarizationStrategy(
    client=client,
    target_count=4,
    prompt=(
        "Summarize this conversation in 3 sentences. Focus on: "
        "(1) decisions made, (2) open questions, (3) current state of work. "
        "Do not offer opinions or judgments."
    ),
)
```

### Key points

- If the LLM call fails (network error, rate limit), the strategy logs a warning and returns `False` — no messages are excluded. The agent continues normally.
- If the LLM returns empty text, the strategy also logs and returns `False`.
- Summary messages carry `SUMMARY_OF_MESSAGE_IDS_KEY` and `SUMMARY_OF_GROUP_IDS_KEY` annotations pointing to the originals.
- Original messages carry `SUMMARIZED_BY_SUMMARY_ID_KEY` pointing back to the summary.
- You can use `apply_compaction(messages, strategy=strategy)` to run summarization outside of an agent.

---

## 6. `ContextWindowCompactionStrategy`

**Source:** `agent_framework/_compaction.py`

`ContextWindowCompactionStrategy` is a **two-phase, token-budget–driven pipeline** that automatically
protects against context window overflow. It computes a safe input budget from the model's total context
window and maximum output tokens, then runs two sequential passes:

1. **Tool result eviction** (`ToolResultCompactionStrategy`) — collapses older tool-call groups into short
   summary lines when included tokens exceed `tool_eviction_threshold` × `input_budget`.
2. **Truncation** (`TruncationStrategy`) — removes oldest non-system groups when included tokens exceed
   `truncation_threshold` × `input_budget`.

This is the recommended out-of-the-box strategy for GPT-4o (128k), GPT-4o-mini (128k), and similar models.

### Constructor

```python
ContextWindowCompactionStrategy(
    *,
    max_context_window_tokens: int,
    max_output_tokens: int,
    tokenizer: TokenizerProtocol | None = None,   # defaults to CharacterEstimatorTokenizer
    tool_eviction_threshold: float = 0.5,          # triggers at 50% of input budget
    truncation_threshold: float = 0.8,             # triggers at 80% of input budget
    keep_last_tool_call_groups: int = 4,           # retain N most recent tool-call groups verbatim
)
```

Budget math:

```
input_budget = max_context_window_tokens - max_output_tokens
tool_eviction_tokens = int(input_budget * tool_eviction_threshold)
truncation_tokens    = int(input_budget * truncation_threshold)
```

### Example 1: GPT-4o with 128k context window

```python
import asyncio
from agent_framework import Agent, CompactionProvider, ContextWindowCompactionStrategy
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    strategy = ContextWindowCompactionStrategy(
        max_context_window_tokens=128_000,
        max_output_tokens=16_384,
        # tool_eviction triggers at 50% of (128k - 16k) = 55,808 tokens
        # truncation triggers at 80% = 89,292 tokens
    )

    agent = Agent(
        client=client,
        instructions="You are a long-context research assistant.",
        context_providers=[CompactionProvider(before_strategy=strategy)],
    )

    session = agent.create_session()
    response = await agent.run("Summarize the key developments in quantum computing over the last decade.", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 2: aggressive settings for small-context models

```python
from agent_framework import ContextWindowCompactionStrategy, CompactionProvider

strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=16_000,   # e.g. GPT-3.5-turbo-instruct
    max_output_tokens=2_000,
    tool_eviction_threshold=0.4,        # start evicting earlier
    truncation_threshold=0.7,
    keep_last_tool_call_groups=2,       # keep fewer tool groups to save space
)
provider = CompactionProvider(before_strategy=strategy)
```

### Example 3: bringing a custom tokenizer

```python
import tiktoken
from agent_framework import TokenizerProtocol, ContextWindowCompactionStrategy
from agent_framework._types import Message


class TiktokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    def count_tokens(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += 3  # role overhead per message
            for content in msg.contents:
                if hasattr(content, "text") and content.text:
                    total += len(self._enc.encode(content.text))
        return total


strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=16_384,
    tokenizer=TiktokenTokenizer(),
)
```

### Key points

- Phase 1 (tool eviction) and phase 2 (truncation) are **independent `TokenBudgetComposedStrategy` instances** — each fires only when its own threshold is exceeded.
- `CharacterEstimatorTokenizer` divides character count by 4 — accurate enough for most use cases without importing tiktoken.
- `keep_last_tool_call_groups=4` means the most recent 4 tool-call groups are kept verbatim; older ones are collapsed into `[Tool results: name: value; name: value]` lines.
- The strategy makes no LLM calls — it is purely rule-based.

---

## 7. `SlidingWindowStrategy`

**Source:** `agent_framework/_compaction.py`

`SlidingWindowStrategy` keeps only the **N most recent non-system message groups**, optionally preserving
system groups as stable anchors. It is the simplest compaction strategy — no LLM calls, no token counting.

### Constructor

```python
SlidingWindowStrategy(
    *,
    keep_last_groups: int,
    preserve_system: bool = True,
)
```

| Parameter | Description |
|---|---|
| `keep_last_groups` | Number of most-recent non-system included groups to retain (must be > 0) |
| `preserve_system` | If True, system groups are always retained regardless of position |

### Example 1: keep last 5 conversation turns

```python
import asyncio
from agent_framework import Agent, CompactionProvider
from agent_framework._compaction import SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    agent = Agent(
        client=client,
        instructions="You are a concise assistant.",
        context_providers=[
            CompactionProvider(
                before_strategy=SlidingWindowStrategy(keep_last_groups=10),
                # keeps last 10 non-system groups; earlier turns are excluded
            )
        ],
    )

    session = agent.create_session()
    for i in range(25):
        r = await agent.run(f"Respond briefly to message number {i + 1}.", session=session)
        print(f"[{i+1}] {r.text[:50]}…")


asyncio.run(main())
```

### Example 2: combine with `SelectiveToolCallCompactionStrategy`

```python
from agent_framework import CompactionProvider, TokenBudgetComposedStrategy
from agent_framework._compaction import (
    SlidingWindowStrategy,
    SelectiveToolCallCompactionStrategy,
    CharacterEstimatorTokenizer,
)

# First reduce tool chatter, then apply sliding window
composed = TokenBudgetComposedStrategy(
    token_budget=50_000,
    tokenizer=CharacterEstimatorTokenizer(),
    strategies=[
        SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=3),
        SlidingWindowStrategy(keep_last_groups=20),
    ],
)
provider = CompactionProvider(before_strategy=composed)
```

### Key points

- Excluded groups are marked with `reason="sliding_window"` in `message.additional_properties[EXCLUDE_REASON_KEY]`.
- `preserve_system=True` is the default — system instructions are never dropped.
- This strategy is **not token-aware** — combine it with `TokenBudgetComposedStrategy` if you need token-budget enforcement.
- `SlidingWindowStrategy` operates on **groups** (semantically related message clusters), not individual messages.

---

## 8. `SelectiveToolCallCompactionStrategy`

**Source:** `agent_framework/_compaction.py`

`SelectiveToolCallCompactionStrategy` reduces context window usage by **excluding older tool-call groups**,
keeping only the most recent N tool interactions verbatim. Unlike `ToolResultCompactionStrategy`, it
**fully removes** old tool groups (no summary is inserted).

### Constructor

```python
SelectiveToolCallCompactionStrategy(
    *,
    keep_last_tool_call_groups: int = 1,
)
```

| Parameter | Default | Description |
|---|---|---|
| `keep_last_tool_call_groups` | 1 | Number of most-recent tool-call groups to retain; set to 0 to remove all |

### Example 1: standalone tool-call compaction

```python
import asyncio
from agent_framework import Agent, CompactionProvider
from agent_framework._compaction import SelectiveToolCallCompactionStrategy
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    @tool
    def search(query: str) -> str:
        """Search the web for information."""
        return f"Result for: {query}"

    agent = Agent(
        client=client,
        tools=[search],
        instructions="You are a research assistant that uses search extensively.",
        context_providers=[
            CompactionProvider(
                before_strategy=SelectiveToolCallCompactionStrategy(
                    keep_last_tool_call_groups=2,  # keep last 2 tool interactions
                )
            )
        ],
    )

    session = agent.create_session()
    response = await agent.run("Research the history of Python programming language.", session=session)
    print(response.text)


asyncio.run(main())
```

### Comparison with `ToolResultCompactionStrategy`

| Strategy | What happens to old tool groups |
|---|---|
| `SelectiveToolCallCompactionStrategy` | Fully excluded (removed from context) |
| `ToolResultCompactionStrategy` | Replaced with `[Tool results: name: value; …]` summary line |

Use `SelectiveToolCallCompactionStrategy` when tool results are ephemeral (e.g. weather queries).
Use `ToolResultCompactionStrategy` when a compact record of what tools returned is useful for continuity.

### Key points

- Excluded groups get `reason="tool_call_compaction"`.
- Only groups annotated as `GROUP_KIND_KEY = "tool_call"` are targeted — user/assistant turns are untouched.
- `keep_last_tool_call_groups=0` removes **all** tool-call groups — useful for aggressive compression of tool-heavy agents.

---

## 9. `WorkflowViz`

**Source:** `agent_framework/_workflows/_viz.py`

`WorkflowViz` generates **DOT-format digraphs** (for graphviz) and **Mermaid flowcharts** from a compiled
`Workflow`. Sub-workflows hosted inside `WorkflowExecutor` instances appear as nested clusters/subgraphs.

### Constructor

```python
WorkflowViz(workflow: Workflow)
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `to_digraph(include_internal_executors=False)` | `str` | DOT format for graphviz |
| `to_mermaid(include_internal_executors=False)` | `str` | Mermaid `flowchart TD` format |
| `export(format, filename=None, include_internal_executors=False)` | `str` | Render to file (svg/png/pdf/dot); returns path |
| `save_svg(filename, …)` | `str` | Convenience: export as SVG |
| `save_png(filename, …)` | `str` | Convenience: export as PNG |
| `save_pdf(filename, …)` | `str` | Convenience: export as PDF |

`export()` requires `pip install graphviz>=0.20.0` and the system graphviz executables
(`sudo apt-get install graphviz` / `brew install graphviz`).

### Example 1: print a Mermaid flowchart

```python
from agent_framework import WorkflowBuilder, WorkflowViz, FunctionExecutor, executor


@executor
class Planner:
    async def run(self, context):
        return "Plan created."


@executor
class Researcher:
    async def run(self, context):
        return "Research done."


@executor
class Writer:
    async def run(self, context):
        return "Article drafted."


builder = WorkflowBuilder()
planner = builder.add_executor(Planner(), executor_id="planner")
researcher = builder.add_executor(Researcher(), executor_id="researcher")
writer = builder.add_executor(Writer(), executor_id="writer")

builder.add_edge(planner, researcher)
builder.add_edge(planner, writer)
builder.add_fan_in_edge(sources=[researcher, writer], target=planner)

workflow = builder.compile()
viz = WorkflowViz(workflow)

print(viz.to_mermaid())
# flowchart TD
#   planner["planner (Start)"];
#   researcher["researcher"];
#   writer["writer"];
#   ...
```

### Example 2: export as SVG for docs

```python
import asyncio
from pathlib import Path


def export_workflow_diagram(workflow, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    viz = WorkflowViz(workflow)
    path = viz.save_svg(str(output_dir / "workflow.svg"))
    print(f"Diagram saved to: {path}")
    return path
```

### Example 3: embed Mermaid in Markdown docs

```python
def workflow_to_markdown_mermaid(workflow) -> str:
    viz = WorkflowViz(workflow)
    mermaid = viz.to_mermaid()
    return f"```mermaid\n{mermaid}\n```"
```

### DOT output structure

- **Start node**: filled light green, labeled `"<id> (Start)"`.
- **Regular nodes**: light blue boxes.
- **Fan-in nodes**: light goldenrod ellipses, labeled `"fan-in"`.
- **Conditional edges**: dashed with `label="conditional"`.
- **Sub-workflow clusters**: dashed border with `label="sub-workflow: <executor_id>"`.

### Key points

- `include_internal_executors=True` reveals internal plumbing (e.g. `AgentExecutor` injected by `WorkflowAgent`).
- Node IDs in DOT are raw strings — special characters are escaped automatically.
- Sub-workflows are namespaced by executor ID to prevent node ID collisions in the DOT graph.
- Mermaid node IDs replace non-alphanumeric characters with `_` for Mermaid spec compliance.

---

## 10. `MCPStreamableHTTPTool` + `MCPWebsocketTool`

**Source:** `agent_framework/_mcp.py`

Both classes extend the abstract `MCPTool` base and connect to remote MCP servers using different transports:

| Class | Transport | Use when |
|---|---|---|
| `MCPStreamableHTTPTool` | HTTP/SSE (`streamable_http_client`) | Server-Sent Events stream; public APIs, Cloudflare Workers, Azure Functions |
| `MCPWebsocketTool` | WebSocket (`websocket_client`) | Full-duplex real-time; low-latency servers, persistent connections |

Both are async context managers — connect on `__aenter__`, disconnect on `__aexit__`.

### `MCPStreamableHTTPTool` constructor

```python
MCPStreamableHTTPTool(
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
    terminate_on_close: bool | None = None,
    client: SupportsChatGetResponse | None = None,
    additional_properties: dict[str, Any] | None = None,
    http_client: AsyncClient | None = None,
    header_provider: Callable[[dict[str, Any]], dict[str, str]] | None = None,
)
```

**New in 1.7.0:** `header_provider` — a callable that receives the tool call's runtime `kwargs` and returns
a `dict[str, str]` of HTTP headers to inject into every outbound request. Use this for per-request auth
token forwarding without constructing a new `httpx.AsyncClient` per call.

### `MCPWebsocketTool` constructor

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
    description: str | None = None,
    approval_mode: … | None = None,
    allowed_tools: Collection[str] | None = None,
    client: SupportsChatGetResponse | None = None,
    additional_properties: dict[str, Any] | None = None,
    **kwargs,   # forwarded to websocket_client()
)
```

`MCPWebsocketTool` requires `pip install mcp[ws]` (adds the `websockets` dependency).

### Example 1: HTTP/SSE MCP server

```python
import asyncio
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    mcp_tool = MCPStreamableHTTPTool(
        name="weather-api",
        url="https://mcp.weather-service.example.com/mcp",
        description="Real-time weather data and forecasts",
        tool_name_prefix="weather",     # tools appear as weather_get_forecast, etc.
        approval_mode="never_require",  # no user confirmation for weather queries
    )

    async with mcp_tool:
        agent = Agent(
            client=client,
            tools=[mcp_tool],
            instructions="You are a weather assistant. Always fetch current data before answering.",
        )
        response = await agent.run("What's the weather forecast for London this weekend?")
        print(response.text)


asyncio.run(main())
```

### Example 2: per-request auth with `header_provider`

```python
import asyncio
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework._middleware import FunctionInvocationContext


def make_auth_tool(get_token_for_user) -> MCPStreamableHTTPTool:
    """Create an MCP tool that injects a per-request Bearer token."""

    def header_provider(kwargs: dict) -> dict[str, str]:
        user_id = kwargs.get("_user_id", "")
        token = get_token_for_user(user_id)
        return {"Authorization": f"Bearer {token}"}

    return MCPStreamableHTTPTool(
        name="secure-api",
        url="https://api.internal.example.com/mcp",
        header_provider=header_provider,
    )


async def main() -> None:
    mcp_tool = make_auth_tool(get_token_for_user=lambda uid: f"token-{uid}")
    async with mcp_tool:
        agent = Agent(
            client=OpenAIChatClient(),
            tools=[mcp_tool],
        )
        response = await agent.run("Query the secure internal API.")
        print(response.text)
```

### Example 3: WebSocket MCP with real-time data

```python
import asyncio
from agent_framework import Agent, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    mcp_tool = MCPWebsocketTool(
        name="realtime-market",
        url="wss://market-data.example.com/mcp",
        description="Real-time market data and order management",
        tool_name_prefix="market",
        request_timeout=10,
    )

    async with mcp_tool:
        agent = Agent(
            client=client,
            tools=[mcp_tool],
            instructions="You are a market data analyst. Use real-time data for all queries.",
        )
        response = await agent.run("What is the current bid/ask spread for AAPL?")
        print(response.text)


asyncio.run(main())
```

### Example 4: selective tool approval

```python
from agent_framework import MCPStreamableHTTPTool

# Some MCP tools require approval; destructive ones always; read-only ones never
mcp_tool = MCPStreamableHTTPTool(
    name="database-api",
    url="https://db.example.com/mcp",
    approval_mode={
        "always_require_approval": ["db_delete_record", "db_truncate_table"],
        "never_require_approval": ["db_query", "db_list_tables"],
    },
)
```

### Example 5: custom result parser

```python
import json
from agent_framework import MCPStreamableHTTPTool
from mcp import types


def parse_result(result: types.CallToolResult) -> str:
    """Extract and pretty-print JSON from tool result."""
    for content in result.content:
        if hasattr(content, "text") and content.text:
            try:
                return json.dumps(json.loads(content.text), indent=2)
            except json.JSONDecodeError:
                return content.text
    return "[no result]"


mcp_tool = MCPStreamableHTTPTool(
    name="data-api",
    url="https://api.example.com/mcp",
    parse_tool_results=parse_result,
)
```

### Key points for `MCPStreamableHTTPTool`

- `http_client` (custom `httpx.AsyncClient`) and `header_provider` are mutually exclusive: if you pass `http_client`, headers from `header_provider` are still injected via an httpx event hook — the client is shared and must not be reused by other code after passing it to the tool.
- `terminate_on_close=True` (default) sends an HTTP `DELETE` to the MCP server's `/mcp` endpoint when the context manager exits.
- The `contextvars.ContextVar` (`_mcp_call_headers`) ensures header injection is scoped to the current async task — safe for concurrent tool calls.

### Key points for `MCPWebsocketTool`

- Requires `pip install mcp[ws]` — the `websockets` package is an optional dependency.
- Extra `**kwargs` are forwarded directly to `websocket_client()` — use them to pass TLS settings, ping intervals, etc.
- WebSocket connections are **persistent** — one connection is shared across all tool calls within the `async with` block.

### Transport comparison

| Feature | `MCPStdioTool` | `MCPStreamableHTTPTool` | `MCPWebsocketTool` |
|---|---|---|---|
| Protocol | stdin/stdout | HTTP + SSE | WebSocket |
| Connection | New process | HTTP keep-alive | Persistent WS |
| Best for | Local servers | Public APIs, serverless | Real-time, low-latency |
| Auth | Env vars | Headers, Bearer tokens | Headers (via kwargs) |
| `header_provider` | — | ✓ (new in 1.7.0) | — |
| Extra deps | None | `httpx` | `mcp[ws]` |

---

## Version history

| Version | Changes |
|---|---|
| 1.7.0 | Added `BackgroundAgentsProvider`, `MemoryContextProvider` / `MemoryFileStore`, `TodoProvider` / `TodoFileStore`, `AgentModeProvider`; `MCPStreamableHTTPTool.header_provider`; `ContextWindowCompactionStrategy` |
| 1.6.0 | `SummarizationStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`; `WorkflowViz`; `MCPStreamableHTTPTool` and `MCPWebsocketTool` GA |
| 1.5.0 | `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy` |
| 1.4.0 | `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `LocalEvaluator` |
| 1.3.0 | `FileHistoryProvider`, middleware layer |
| 1.2.0 | `WorkflowBuilder`, `FunctionalWorkflow`, `RunContext`, `InlineSkill` |
| 1.0.0 | `Agent`, `RawAgent`, `FunctionTool`, `MCPStdioTool`, `AgentSession` |

---

*Continue with [Class Deep Dives Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`.*

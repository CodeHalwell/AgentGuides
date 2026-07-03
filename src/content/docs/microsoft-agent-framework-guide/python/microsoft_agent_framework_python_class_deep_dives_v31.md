---
title: "Microsoft Agent Framework Python — Class Deep Dives Vol. 31"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.10.0: FileMemoryProvider, create_harness_agent, AgentLoopMiddleware advanced patterns, ToolApprovalMiddleware, FileSkillsSource depth and filter predicates, FoundryAgent, AgentExecutorResponse.with_text, todos_remaining and background_tasks_running loop helpers, custom ContextProvider with SessionContext.metadata, and InMemoryCheckpointStorage for workflow testing."
framework: microsoft-agent-framework
language: python
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 31

All examples are source-verified against **agent-framework 1.10.0**. This volume covers
ten class groups that are new in 1.10.0 or have been previously underdocumented.

---

## 1. `FileMemoryProvider`

**Module:** `agent_framework._harness._file_memory`  
**Import:** `from agent_framework import FileMemoryProvider` (via harness re-export)

`FileMemoryProvider` is a `ContextProvider` that gives an agent session-scoped,
file-based memory. Each memory is stored as an individual file. A
`memories.md` index is maintained automatically and injected into the agent's
context so the model knows what it has previously written.

The provider registers seven tools automatically:

| Tool | Description |
|---|---|
| `file_memory_write` | Write a file with optional description sidecar |
| `file_memory_read` | Read a named memory file |
| `file_memory_delete` | Delete a file and its description |
| `file_memory_ls` | List memory files (supports glob filter) |
| `file_memory_grep` | Regex search across memory file contents |
| `file_memory_replace` | Replace a substring in a memory file |
| `file_memory_replace_lines` | Replace specific line numbers in a memory file |

**Constructor:**

```python
FileMemoryProvider(
    store: AgentFileStore,
    *,
    source_id: str = "file_memory",
    scope: str | None = None,       # None → session_id used as working folder
    instructions: str | None = None, # None → built-in memory instructions used
)
```

Pass `scope` to group memories across sessions — for example by user ID — instead of
isolating them per session.

### Example 1 — Minimal session-scoped memory

The simplest setup uses `FileSystemAgentFileStore` (in-process filesystem) and lets the
provider derive the working folder from `session_id` automatically.

```python
import asyncio
from agent_framework import Agent, FileMemoryProvider
from agent_framework._harness._file_access import FileSystemAgentFileStore
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    store = FileSystemAgentFileStore("./agent-memory")
    memory = FileMemoryProvider(store=store)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a research assistant. Use file memory to track findings.",
        context_providers=[memory],
    )

    session = agent.create_session()
    # First turn — agent writes a memory file
    r1 = await agent.run(
        "Search for facts about agent-framework 1.10.0 and store them in memory.",
        session=session,
    )
    print(r1.text)

    # Second turn — agent reads back what it stored
    r2 = await agent.run(
        "Summarise everything you remember about agent-framework 1.10.0.",
        session=session,
    )
    print(r2.text)


asyncio.run(main())
```

### Example 2 — Shared memory across sessions with `scope`

Pass a fixed `scope` string (e.g. a user ID) so memories persist across separate
sessions for the same logical entity.

```python
import asyncio
from agent_framework import Agent, FileMemoryProvider
from agent_framework._harness._file_access import FileSystemAgentFileStore
from agent_framework.openai import OpenAIChatClient


async def run_for_user(user_id: str, prompt: str) -> str:
    store = FileSystemAgentFileStore("./shared-memory")
    memory = FileMemoryProvider(store=store, scope=user_id)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a personal assistant. Remember user preferences.",
        context_providers=[memory],
    )

    session = agent.create_session()
    response = await agent.run(prompt, session=session)
    return response.text


async def main() -> None:
    # Session 1: tell the agent a preference
    await run_for_user("alice", "Remember that I prefer concise bullet-point summaries.")

    # Session 2 (new session, same scope): preference is still visible
    result = await run_for_user("alice", "How should I format your responses?")
    print(result)  # Agent recalls bullet-point preference from shared memory


asyncio.run(main())
```

### Example 3 — Custom instructions and description sidecars

Override the default memory instructions and show how description sidecars help
the agent discover large files efficiently.

```python
import asyncio
from agent_framework import Agent, FileMemoryProvider
from agent_framework._harness._file_access import FileSystemAgentFileStore
from agent_framework.openai import OpenAIChatClient

CUSTOM_INSTRUCTIONS = """
## Memory Guidelines
You have file-based memory via file_memory_* tools.
- Always write a description when storing content longer than 200 words.
- Before starting any task, call file_memory_ls to check what you already know.
- Use file_memory_grep to search across memories before duplicating work.
"""


async def main() -> None:
    store = FileSystemAgentFileStore("./research-memory")
    memory = FileMemoryProvider(
        store=store,
        source_id="research_memory",
        instructions=CUSTOM_INSTRUCTIONS,
    )

    agent = Agent(
        client=OpenAIChatClient(),
        context_providers=[memory],
    )

    session = agent.create_session()

    # The agent will use file_memory_write with a description sidecar
    await agent.run(
        "Fetch the agent-framework changelog and store it as 'changelog.md' with "
        "a short description of what changed in 1.10.0.",
        session=session,
    )

    # Later: grep for a specific entry
    result = await agent.run(
        "What changed in agent-framework related to skills in 1.10.0?",
        session=session,
    )
    print(result.text)


asyncio.run(main())
```

---

## 2. `create_harness_agent`

**Module:** `agent_framework._harness._agent`  
**Import:** `from agent_framework import create_harness_agent`

`create_harness_agent` is a batteries-included factory that assembles a fully-wired
`Agent` from a chat client. It automatically configures:

- Function invocation loop
- Per-service-call history persistence (`InMemoryHistoryProvider`)
- Context-window compaction (when token budgets are given)
- `TodoProvider`, `AgentModeProvider`, `FileMemoryProvider`, `FileAccessProvider`
- `SkillsProvider` (via `skills_paths`)
- `BackgroundAgentsProvider`
- `ToolApprovalMiddleware`
- `AgentLoopMiddleware` (when `loop_should_continue` is provided)
- OpenTelemetry tracing (when `otel_provider_name` is given)

**Full signature (key parameters):**

```python
def create_harness_agent(
    client: SupportsChatGetResponse,
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    harness_instructions: str | None = None,  # None → DEFAULT_HARNESS_INSTRUCTIONS
    agent_instructions: str | None = None,    # appended after harness instructions
    tools: ToolTypes | Sequence[ToolTypes] | None = None,
    max_context_window_tokens: int | None = None,
    max_output_tokens: int | None = None,
    history_provider: HistoryProvider | None = None,
    disable_compaction: bool = False,
    disable_todo: bool = False,
    todo_provider: TodoProvider | None = None,
    disable_mode: bool = False,
    mode_provider: AgentModeProvider | None = None,
    disable_file_memory: bool = False,
    file_memory_store: AgentFileStore | None = None,
    disable_file_access: bool = False,
    file_access_store: AgentFileStore | None = None,
    file_access_disable_write_tools: bool = False,
    skills_provider: SkillsProvider | None = None,
    skills_paths: str | Path | Sequence[str | Path] | None = None,
    background_agents: Sequence[SupportsAgentRun] | None = None,
    background_agents_instructions: str | None = None,
    disable_web_search: bool = False,
    disable_tool_auto_approval: bool = False,
    auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None,
    loop_should_continue: ShouldContinueCallable | None = None,
    loop_next_message: NextMessageCallable | None = None,
    loop_max_iterations: int | None = 10,
    otel_provider_name: str | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: Sequence[MiddlewareTypes] | None = None,
    default_options: Mapping[str, Any] | None = None,
) -> Agent[Any]
```

### Example 1 — Minimal usage (zero config)

```python
import asyncio
from agent_framework import create_harness_agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Zero config: TodoProvider, FileMemoryProvider, FileAccessProvider all on by default.
    agent = create_harness_agent(OpenAIChatClient(model="gpt-4o"))

    session = agent.create_session()
    response = await agent.run("Plan a weekend trip to Seattle.", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 2 — Customised domain agent with compaction and skills

Enable context-window compaction (requires token params) and point the agent at a
local skills directory.

```python
import asyncio
from agent_framework import create_harness_agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = create_harness_agent(
        OpenAIChatClient(model="gpt-4o"),
        name="research-agent",
        agent_instructions=(
            "You are a research assistant. Focus on academic and technical sources. "
            "Always create a todo list before starting multi-step research tasks."
        ),
        max_context_window_tokens=128_000,
        max_output_tokens=16_000,
        skills_paths=["./skills", "./custom-skills"],
        disable_file_access=True,  # read-only environment
        disable_web_search=False,
    )

    session = agent.create_session()
    response = await agent.run(
        "Research the latest developments in multi-agent orchestration frameworks.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 — Loop until todo list is complete

Wire `todos_remaining` as the loop predicate so the agent keeps running until every
item on its self-created todo list is checked off.

```python
import asyncio
from agent_framework import create_harness_agent
from agent_framework._harness._loop import todos_remaining, todos_remaining_message
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = create_harness_agent(
        OpenAIChatClient(model="gpt-4o"),
        agent_instructions=(
            "You are a thorough task executor. Create a todo list for every multi-step task "
            "and mark each item complete as you finish it."
        ),
        loop_should_continue=todos_remaining(),       # loop while open todos exist
        loop_next_message=todos_remaining_message,    # reminds agent which todos remain
        loop_max_iterations=20,
    )

    session = agent.create_session()
    response = await agent.run(
        "Write a Python function to parse CSV, add type hints, write unit tests, "
        "and generate a short docstring. Use your todo list to track each step.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

---

## 3. `AgentLoopMiddleware` — Advanced Patterns

**Module:** `agent_framework._harness._loop`  
**Import:** `from agent_framework._harness._loop import AgentLoopMiddleware`

`AgentLoopMiddleware` re-runs an agent until a `should_continue` predicate returns
`False`. Beyond the basics (covered in Vol. 17), this section focuses on the lesser-used
parameters: `fresh_context`, `return_final_only`, `record_feedback`, and `inject_progress`.

**Constructor (full):**

```python
AgentLoopMiddleware(
    should_continue: ShouldContinueCallable,
    *,
    max_iterations: int | None = 10,
    next_message: NextMessageCallable | None = None,
    record_feedback: FeedbackCallable | None = None,
    inject_progress: bool = True,          # prepend accumulated feedback log to each iteration
    fresh_context: bool = False,           # restore session snapshot between iterations
    return_final_only: bool = False,       # return only last iteration's response
    additional_instructions: str | None = None,
)
```

### Example 1 — `return_final_only` for clean final responses

By default, the middleware aggregates all iteration messages into one `AgentResponse`.
Set `return_final_only=True` to return only the final pass (useful for streaming UIs).

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.openai import OpenAIChatClient


async def should_continue(*, iteration: int, last_result, **kwargs) -> bool:
    return iteration < 3 and "COMPLETE" not in last_result.text.upper()


async def main() -> None:
    loop_mw = AgentLoopMiddleware(
        should_continue,
        max_iterations=5,
        return_final_only=True,  # only the last iteration reaches the caller
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Iteratively improve your answer. End with 'COMPLETE' when done.",
        middleware=[loop_mw],
    )

    response = await agent.run("Write a haiku about software agents.")
    # response.text is only the final iteration's output
    print(response.text)


asyncio.run(main())
```

### Example 2 — `record_feedback` for a structured progress log

`record_feedback` is called once per iteration and its return value is appended to the
`progress` list. When `inject_progress=True` (the default) this log is prepended to
each subsequent iteration so the agent can learn from its own history.

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.openai import OpenAIChatClient


def record_quality(*, iteration: int, last_result, **kwargs) -> str:
    word_count = len(last_result.text.split())
    return f"Iteration {iteration}: {word_count} words"


async def should_continue(
    *, iteration: int, last_result, progress: list[str], **kwargs
) -> bool:
    word_count = len(last_result.text.split())
    print(f"[loop] iteration={iteration}, words={word_count}, log={progress}")
    return iteration < 4 and word_count < 150


async def main() -> None:
    loop_mw = AgentLoopMiddleware(
        should_continue,
        max_iterations=4,
        record_feedback=record_quality,    # logs per-iteration word counts
        inject_progress=True,              # progress log injected before each pass
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Expand the answer each iteration until it reaches 150 words.",
        middleware=[loop_mw],
    )

    response = await agent.run("Explain what a context provider does.")
    print(response.text)


asyncio.run(main())
```

### Example 3 — `fresh_context` for independent iterations

`fresh_context=True` snapshots the session before the first iteration and restores it
between passes. Each iteration starts from the original prompt, preventing earlier
intermediate answers from polluting later context. The accumulated `progress` log (from
`record_feedback`) is the only carry-over.

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.openai import OpenAIChatClient


approaches: list[str] = []


def capture_approach(*, iteration: int, last_result, **kwargs) -> str:
    approaches.append(last_result.text)
    return f"Approach {iteration} captured"


async def should_continue(*, iteration: int, **kwargs) -> bool:
    return iteration < 3  # always produce 3 independent approaches


async def main() -> None:
    loop_mw = AgentLoopMiddleware(
        should_continue,
        max_iterations=3,
        fresh_context=True,       # each pass sees the original question only
        record_feedback=capture_approach,
        inject_progress=False,    # don't leak approach N into approach N+1
        return_final_only=True,
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Generate a distinct, creative approach each time.",
        middleware=[loop_mw],
    )

    await agent.run("How would you design a multi-agent todo system?")

    for i, approach in enumerate(approaches, 1):
        print(f"\n--- Approach {i} ---\n{approach}")


asyncio.run(main())
```

---

## 4. `ToolApprovalMiddleware` + `ToolApprovalRule` + `ToolApprovalState`

**Module:** `agent_framework._harness._tool_approval`  
**Import:** `from agent_framework import ToolApprovalMiddleware, ToolApprovalRule`

`ToolApprovalMiddleware` intercepts tool calls and routes them through an approval
callback before execution. It supports standing rules (approved-once-applies-always)
serialised in `ToolApprovalState`.

**Constructor:**

```python
ToolApprovalMiddleware(
    *,
    source_id: str = "tool_approval",
    auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None,
)
```

**`ToolApprovalRule` constructor:**

```python
ToolApprovalRule(
    tool_name: str,
    arguments: Mapping[str, str] | None = None,
    *,
    server_label: str | None = None,
)
```

- `arguments=None` → rule matches every call to the tool regardless of arguments  
- `arguments={}` → rule matches only no-argument calls  
- `server_label` → scopes the rule to a specific hosted-tool server

### Example 1 — Standing approval rule: always approve a specific tool

Pre-seed the session with a `ToolApprovalRule` so the agent can call a tool without
prompting the user each time.

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._tool_approval import (
    ToolApprovalMiddleware,
    ToolApprovalRule,
    ToolApprovalState,
)
from agent_framework.openai import OpenAIChatClient


async def approval_callback(approval_requests, *, session, **kwargs):
    """Auto-approve if a standing rule covers the request; else ask the user."""
    state: ToolApprovalState = session.state.get("tool_approval") or ToolApprovalState()
    approved = []
    for req in approval_requests:
        rule_match = any(
            rule.tool_name == req.tool_name for rule in state.rules
        )
        if rule_match:
            approved.append(req.approve())
        else:
            print(f"Approval needed for: {req.tool_name}({req.arguments})")
            approved.append(req.approve())  # approve interactively in real code
    return approved


async def main() -> None:
    # Pre-seed a standing rule so get_weather is always approved
    initial_state = ToolApprovalState(
        rules=[ToolApprovalRule("get_weather")]
    )

    approval_mw = ToolApprovalMiddleware(source_id="tool_approval")

    from agent_framework import tool

    @tool
    def get_weather(location: str) -> str:
        """Return current weather for a location."""
        return f"Sunny, 22°C in {location}"

    agent = Agent(
        client=OpenAIChatClient(),
        tools=[get_weather],
        middleware=[approval_mw],
    )

    session = agent.create_session()
    # Store standing rule in session state
    session.state["tool_approval"] = initial_state

    response = await agent.run("What's the weather in Tokyo?", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 2 — Auto-approval callbacks for heuristic patterns

Supply `auto_approval_rules` — callables that inspect each pending request and
return `True` to auto-approve without surfacing it to the user.

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework._harness._tool_approval import ToolApprovalMiddleware
from agent_framework.openai import OpenAIChatClient

# Auto-approve any read-only tool (name starts with "get_" or "list_" or "search_")
def approve_readonly(tool_name: str, arguments: dict, **kwargs) -> bool:
    return tool_name.startswith(("get_", "list_", "search_"))


@tool
def get_stock_price(symbol: str) -> str:
    """Fetch current stock price for a ticker symbol."""
    return f"{symbol}: $142.00"


@tool
def list_portfolio() -> str:
    """List current portfolio holdings."""
    return "MSFT x10, AAPL x5"


@tool
def sell_stock(symbol: str, quantity: int) -> str:
    """Sell shares of a stock."""
    return f"Sold {quantity} shares of {symbol}"


async def main() -> None:
    mw = ToolApprovalMiddleware(auto_approval_rules=[approve_readonly])

    agent = Agent(
        client=OpenAIChatClient(),
        tools=[get_stock_price, list_portfolio, sell_stock],
        middleware=[mw],
    )

    session = agent.create_session()
    # get_stock_price and list_portfolio auto-approved; sell_stock surfaces for approval
    response = await agent.run(
        "Check the price of MSFT, list my portfolio, then sell 2 MSFT shares.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 — Serialising `ToolApprovalState` across sessions

Persist the accumulated `rules` from one session into the next so standing approvals
survive process restarts.

```python
import asyncio
import json
from agent_framework._harness._tool_approval import ToolApprovalRule, ToolApprovalState


async def save_approval_state(session_state: dict) -> str:
    """Serialise approval state to JSON."""
    state: ToolApprovalState | None = session_state.get("tool_approval")
    if state is None:
        return "{}"
    return json.dumps(state.to_dict())


async def load_approval_state(json_str: str) -> ToolApprovalState:
    """Restore approval state from JSON."""
    if not json_str or json_str == "{}":
        return ToolApprovalState()
    raw = json.loads(json_str)
    return ToolApprovalState.from_dict(raw)


async def main() -> None:
    # Build an initial state with two standing rules
    state = ToolApprovalState(
        rules=[
            ToolApprovalRule("read_file"),
            ToolApprovalRule("search_web", {"query": "agent-framework changelog"}),
        ]
    )

    # Serialize
    serialized = json.dumps(state.to_dict())
    print("Serialised:", serialized)

    # Restore in a new process
    restored = ToolApprovalState.from_dict(json.loads(serialized))
    print("Restored rules:", [r.tool_name for r in restored.rules])
    # Restored rules: ['read_file', 'search_web']


asyncio.run(main())
```

---

## 5. `FileSkillsSource` — Depth and Filter Predicates

**Module:** `agent_framework._skills`  
**Import:** `from agent_framework import FileSkillsSource`

`FileSkillsSource` discovers skills from `SKILL.md` files on the filesystem. New in
**1.10.0**: `search_depth` (default `2`) controls how deep into each skill directory
the scanner looks for scripts and resources; `script_filter` and `resource_filter`
predicates give fine-grained control over which files are included.

**Constructor:**

```python
FileSkillsSource(
    skill_paths: str | Path | Sequence[str | Path],
    *,
    script_runner: SkillScriptRunner | None = None,
    resource_extensions: tuple[str, ...] | None = None,  # default: .md .json .yaml .yml .csv .xml .txt
    script_extensions: tuple[str, ...] | None = None,    # default: .py
    search_depth: int = 2,                               # 1 = root only, 2 = root + one level
    script_filter: Callable[[str, str], bool] | None = None,  # (skill_name, rel_path) -> bool
    resource_filter: Callable[[str, str], bool] | None = None,
)
```

### Example 1 — Basic discovery with increased depth

Set `search_depth=3` for skills with nested subdirectories.

```python
import asyncio
from agent_framework import FileSkillsSource


async def main() -> None:
    source = FileSkillsSource(
        skill_paths=["./skills", "./plugins"],
        search_depth=3,  # scan root + 2 levels of subdirectories
    )

    skills = await source.get_skills()
    for skill in skills:
        print(f"{skill.name}: {len(skill.scripts)} scripts, {len(skill.resources)} resources")


asyncio.run(main())
```

### Example 2 — Exclude test files with `script_filter`

Prevent test scripts from being loaded as runnable skill scripts.

```python
import asyncio
from agent_framework import Agent, FileSkillsSource
from agent_framework._skills import SkillsProvider
from agent_framework.openai import OpenAIChatClient


def no_test_scripts(skill_name: str, rel_path: str) -> bool:
    """Exclude any file in a 'tests' directory or named test_*.py."""
    return not (rel_path.startswith("tests/") or rel_path.startswith("test_"))


async def main() -> None:
    source = FileSkillsSource(
        skill_paths="./skills",
        search_depth=3,
        script_filter=no_test_scripts,
    )

    provider = SkillsProvider(sources=[source])
    agent = Agent(
        client=OpenAIChatClient(),
        context_providers=[provider],
    )

    session = agent.create_session()
    response = await agent.run("List the available skills.", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 3 — `SkillsProvider.from_paths` with filter predicates

`SkillsProvider.from_paths` is the convenience wrapper for `FileSkillsSource` that
accepts the same `search_depth`, `script_filter`, and `resource_filter` parameters.

```python
import asyncio
from agent_framework import Agent
from agent_framework._skills import SkillsProvider
from agent_framework.openai import OpenAIChatClient


def only_production_resources(skill_name: str, rel_path: str) -> bool:
    """Only include resources from the 'data' subdirectory, not 'samples'."""
    return "samples/" not in rel_path and "fixtures/" not in rel_path


def exclude_generated_scripts(skill_name: str, rel_path: str) -> bool:
    """Exclude auto-generated scripts ending with _generated.py."""
    return not rel_path.endswith("_generated.py")


async def main() -> None:
    provider = SkillsProvider.from_paths(
        "./skills",
        search_depth=4,
        script_filter=exclude_generated_scripts,
        resource_filter=only_production_resources,
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You can use skills to complete complex tasks.",
        context_providers=[provider],
    )

    session = agent.create_session()
    response = await agent.run(
        "Use available skills to analyse the sales data.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

---

## 6. `FoundryAgent`

**Module:** `agent_framework.foundry` (requires `pip install agent-framework-foundry`)  
**Import:** `from agent_framework.foundry import FoundryAgent`

`FoundryAgent` is new in **1.10.0**. It wraps an existing Azure AI Foundry
PromptAgent or HostedAgent and exposes it as an `agent_framework.Agent`-compatible
object, so it can participate in `WorkflowBuilder` pipelines and multi-agent
orchestration patterns alongside locally-constructed agents.

> **Note:** `FoundryAgent` lives in the `agent-framework-foundry` sub-package.
> Install it with `pip install agent-framework-foundry`.

### Example 1 — Connect to an existing Foundry PromptAgent

```python
import asyncio
import os
from agent_framework.foundry import FoundryAgent
from azure.identity import DefaultAzureCredential


async def main() -> None:
    agent = FoundryAgent(
        project_endpoint=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        agent_name="my-prompt-agent",           # name of the agent in Foundry
        credential=DefaultAzureCredential(),
    )

    session = agent.create_session()
    response = await agent.run(
        "Summarise the quarterly sales report.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 2 — Pin to a specific agent version

```python
import asyncio
import os
from agent_framework.foundry import FoundryAgent
from azure.identity import DefaultAzureCredential


async def main() -> None:
    # Pin to version "v2" of the deployed agent for reproducibility
    agent = FoundryAgent(
        project_endpoint=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        agent_name="document-classifier",
        agent_version="v2",
        credential=DefaultAzureCredential(),
    )

    session = agent.create_session()
    response = await agent.run(
        "Classify the following document: [invoice content here]",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 — Use `FoundryAgent` inside a `WorkflowBuilder` pipeline

Chain a `FoundryAgent` with a local `AgentExecutor` in a sequential pipeline.

```python
import asyncio
import os
from agent_framework import Agent, WorkflowBuilder
from agent_framework.foundry import FoundryAgent
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential


async def main() -> None:
    # Step 1: extract key points with a Foundry-hosted agent
    extractor = FoundryAgent(
        project_endpoint=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        agent_name="key-point-extractor",
        credential=DefaultAzureCredential(),
    )

    # Step 2: format the output with a local agent
    formatter = Agent(
        client=OpenAIChatClient(),
        instructions="Convert the key points into a polished markdown report.",
    )

    workflow = (
        WorkflowBuilder()
        .add_agent(extractor, id="extract")
        .add_agent(formatter, id="format")
        .connect("extract", "format")
        .build()
    )

    result = await workflow.run("Analyse and format the Q3 earnings call transcript.")
    print(result.text)


asyncio.run(main())
```

---

## 7. `AgentExecutorResponse.with_text()`

**Module:** `agent_framework._workflows._agent_executor`  
**Import:** `from agent_framework import AgentExecutorResponse`

`with_text(text: str) -> AgentExecutorResponse` creates a new
`AgentExecutorResponse` with the text replaced but with `full_conversation` intact.
Without it, returning a plain `str` from a custom executor breaks context chaining
because downstream `AgentExecutor` instances only receive the one string.

**Signature:**

```python
def with_text(self, text: str) -> AgentExecutorResponse:
    ...
```

Returns a new `AgentExecutorResponse` whose `agent_response` contains a single
assistant message with `text`, and whose `full_conversation` is the prior conversation
followed by the new assistant message.

### Example 1 — Text transformation preserving conversation context

```python
import asyncio
from agent_framework import Agent, AgentExecutorResponse, WorkflowContext, WorkflowBuilder, executor
from agent_framework.openai import OpenAIChatClient


@executor(
    id="upper_case",
    input=AgentExecutorResponse,
    output=AgentExecutorResponse,
    workflow_output=str,
)
async def upper_case(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, str],
) -> None:
    upper_text = response.agent_response.text.upper()
    # with_text preserves full_conversation for downstream AgentExecutors
    await ctx.send_message(response.with_text(upper_text))
    await ctx.yield_output(upper_text)


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Answer concisely.")

    workflow = (
        WorkflowBuilder()
        .add_agent(agent, id="answerer")
        .add_executor(upper_case, id="upper_case")
        .connect("answerer", "upper_case")
        .build()
    )

    result = await workflow.run("What is the capital of France?")
    print(result)  # "PARIS"


asyncio.run(main())
```

### Example 2 — Summarisation executor maintaining context chain

```python
import asyncio
from agent_framework import Agent, AgentExecutorResponse, WorkflowContext, WorkflowBuilder, executor
from agent_framework.openai import OpenAIChatClient


@executor(
    id="summariser",
    input=AgentExecutorResponse,
    output=AgentExecutorResponse,
    workflow_output=str,
)
async def summarise(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, str],
) -> None:
    full_text = response.agent_response.text
    # Truncate to first 200 chars as a naive summary
    summary = full_text[:200].rsplit(" ", 1)[0] + "…"
    await ctx.send_message(response.with_text(summary))
    await ctx.yield_output(summary)


async def main() -> None:
    drafter = Agent(
        client=OpenAIChatClient(),
        instructions="Write a detailed 3-paragraph explanation.",
    )
    summariser_agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a downstream agent receiving prior context.",
    )

    workflow = (
        WorkflowBuilder()
        .add_agent(drafter, id="draft")
        .add_executor(summarise, id="summarise")
        .add_agent(summariser_agent, id="followup")
        .connect("draft", "summarise")
        .connect("summarise", "followup")
        .build()
    )

    result = await workflow.run("Explain context providers in agent-framework.")
    print(result.text)


asyncio.run(main())
```

### Example 3 — Chaining multiple `with_text` transformations

Show that `with_text` can be called repeatedly; each call returns a new independent
`AgentExecutorResponse` without mutating the prior one.

```python
import asyncio
from agent_framework import AgentExecutorResponse
from agent_framework._types import AgentResponse, Message


async def main() -> None:
    # Build a mock AgentExecutorResponse directly (no agent needed for this demo)
    original_message = Message("assistant", ["hello world"])
    base_response = AgentExecutorResponse(
        executor_id="demo",
        agent_response=AgentResponse(messages=[original_message]),
        full_conversation=[
            Message("user", ["Say hello"]),
            original_message,
        ],
    )

    # Apply a chain of text transforms — each produces a new independent object
    uppercased = base_response.with_text("HELLO WORLD")
    exclaimed = uppercased.with_text("HELLO WORLD!")

    print(base_response.agent_response.text)   # hello world   (unchanged)
    print(uppercased.agent_response.text)      # HELLO WORLD   (unchanged)
    print(exclaimed.agent_response.text)       # HELLO WORLD!

    # full_conversation only keeps messages up to (but not including) the original agent turn
    # plus the new assistant message — the chain stays clean
    print(len(exclaimed.full_conversation))    # 2 (user + new assistant)


asyncio.run(main())
```

---

## 8. `todos_remaining` and `background_tasks_running`

**Module:** `agent_framework._harness._loop`  
**Import:** `from agent_framework._harness._loop import todos_remaining, background_tasks_running`

These factory functions return `ShouldContinueCallable` predicates ready to pass to
`AgentLoopMiddleware` or `create_harness_agent(loop_should_continue=...)`. They
resolve the relevant provider (TodoProvider / BackgroundAgentsProvider) from
`agent.context_providers` at runtime — no need to pass the provider as an argument.

Companion message callables:
- `todos_remaining_message` — lists open todo items in the loop's `next_message`
- `background_tasks_running_message` — lists still-running background tasks

### Example 1 — Loop until all todos complete (with `looping_modes`)

```python
import asyncio
from agent_framework import create_harness_agent
from agent_framework._harness._loop import todos_remaining, todos_remaining_message
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # looping_modes restricts looping to the "execute" mode only
    # (agent won't loop endlessly during the "plan" mode)
    agent = create_harness_agent(
        OpenAIChatClient(model="gpt-4o"),
        agent_instructions=(
            "You operate in two modes: 'plan' and 'execute'. "
            "In plan mode, create a todo list. "
            "In execute mode, work through every todo and mark each complete."
        ),
        loop_should_continue=todos_remaining(looping_modes=["execute"]),
        loop_next_message=todos_remaining_message,
        loop_max_iterations=15,
    )

    session = agent.create_session()
    response = await agent.run(
        "Plan and execute a 3-step data pipeline: (1) fetch data, (2) clean it, (3) summarise it.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 2 — Loop until background tasks finish

```python
import asyncio
from agent_framework import create_harness_agent, Agent
from agent_framework._harness._loop import background_tasks_running, background_tasks_running_message
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # A sub-agent that the main agent can delegate work to
    sub_agent = Agent(
        client=OpenAIChatClient(),
        name="data-fetcher",
        instructions="Fetch and return the requested data.",
    )

    agent = create_harness_agent(
        OpenAIChatClient(model="gpt-4o"),
        agent_instructions=(
            "You can delegate data-fetching tasks to background agents. "
            "After dispatching background work, wait for all tasks to complete "
            "before producing your final answer."
        ),
        background_agents=[sub_agent],
        loop_should_continue=background_tasks_running(),
        loop_next_message=background_tasks_running_message,
        loop_max_iterations=10,
    )

    session = agent.create_session()
    response = await agent.run(
        "Fetch weather data for London, Paris, and Tokyo in parallel, "
        "then compare the results.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 — Combining both predicates with a custom `should_continue`

```python
import asyncio
from agent_framework import create_harness_agent, Agent
from agent_framework._harness._loop import (
    todos_remaining,
    background_tasks_running,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    todos_check = todos_remaining()
    bg_tasks_check = background_tasks_running()

    # Loop while either open todos OR running background tasks exist
    async def should_continue_combined(**kwargs) -> bool:
        return await todos_check(**kwargs) or bg_tasks_check(**kwargs)

    sub_agent = Agent(client=OpenAIChatClient(), name="helper")

    agent = create_harness_agent(
        OpenAIChatClient(model="gpt-4o"),
        background_agents=[sub_agent],
        loop_should_continue=should_continue_combined,
        loop_max_iterations=20,
    )

    session = agent.create_session()
    response = await agent.run(
        "Complete all tasks: gather data in background, analyse it, write a report.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

---

## 9. Custom `ContextProvider` with `SessionContext.metadata`

**Module:** `agent_framework._sessions`  
**Import:** `from agent_framework import ContextProvider, SessionContext`

`ContextProvider` is the base class for all providers that participate in the
context engineering pipeline. Override `before_run` to inject messages, instructions,
and tools, and `after_run` to process the response. Use `SessionContext.metadata`
to communicate between providers within the same invocation.

**`ContextProvider` interface:**

```python
class ContextProvider:
    def __init__(self, source_id: str): ...

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None: ...

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None: ...
```

**`SessionContext` key attributes:**

| Attribute | Type | Purpose |
|---|---|---|
| `session_id` | `str \| None` | Active session ID |
| `input_messages` | `list[Message]` | Messages from the caller |
| `context_messages` | `dict[str, list[Message]]` | Keyed by provider source_id |
| `instructions` | `list[str]` | System instructions from all providers |
| `tools` | `list[Any]` | Tools from all providers |
| `metadata` | `dict[str, Any]` | Shared dict for cross-provider communication |
| `response` | `AgentResponse \| None` | Set after invocation (in `after_run`) |

### Example 1 — Minimal custom context provider

Inject a dynamic greeting based on the time of day into every invocation.

```python
import asyncio
from datetime import datetime
from typing import Any
from agent_framework import Agent, ContextProvider
from agent_framework._sessions import SessionContext, AgentSession
from agent_framework._types import Message
from agent_framework.openai import OpenAIChatClient


class TimeAwareProvider(ContextProvider):
    def __init__(self) -> None:
        super().__init__(source_id="time_aware")

    async def before_run(
        self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        hour = datetime.now().hour
        period = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        context.extend_instructions(
            self.source_id,
            [f"The user is interacting in the {period}. Greet them appropriately."],
        )


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        context_providers=[TimeAwareProvider()],
    )

    session = agent.create_session()
    response = await agent.run("Hello!", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 2 — Cross-provider communication via `metadata`

Two providers coordinate: the first records timing in `metadata`, the second reads
it to add a latency annotation after the run.

```python
import asyncio
import time
from typing import Any
from agent_framework import Agent, ContextProvider
from agent_framework._sessions import SessionContext, AgentSession
from agent_framework.openai import OpenAIChatClient


class TimingProvider(ContextProvider):
    """Records invocation start time in metadata."""

    def __init__(self) -> None:
        super().__init__(source_id="timing")

    async def before_run(
        self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        context.metadata["invocation_start"] = time.monotonic()

    async def after_run(
        self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        start = context.metadata.get("invocation_start")
        if start is not None:
            elapsed = time.monotonic() - start
            context.metadata["invocation_elapsed_ms"] = round(elapsed * 1000)


class LatencyLoggingProvider(ContextProvider):
    """Logs elapsed time written by TimingProvider."""

    def __init__(self) -> None:
        super().__init__(source_id="latency_log")

    async def after_run(
        self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        elapsed = context.metadata.get("invocation_elapsed_ms")
        if elapsed is not None:
            print(f"[latency] invocation took {elapsed}ms")


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        # Providers run in list order; TimingProvider must come before LatencyLoggingProvider
        context_providers=[TimingProvider(), LatencyLoggingProvider()],
    )

    session = agent.create_session()
    response = await agent.run("What is 7 times 8?", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 3 — Per-provider `state` dict for session-scoped counters

Each provider receives a `state` dict scoped to that provider across invocations
within the same session. Use it to track per-session counters or cached data.

```python
import asyncio
from typing import Any
from agent_framework import Agent, ContextProvider
from agent_framework._sessions import SessionContext, AgentSession
from agent_framework.openai import OpenAIChatClient


class InvocationCounterProvider(ContextProvider):
    """Injects the current invocation number into every prompt."""

    def __init__(self) -> None:
        super().__init__(source_id="invocation_counter")

    async def before_run(
        self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        state["count"] = state.get("count", 0) + 1
        context.extend_instructions(
            self.source_id,
            [f"This is invocation #{state['count']} in this session."],
        )

    async def after_run(
        self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        print(f"Invocation #{state['count']} complete. Response length: {len(context.response.text)} chars")


async def main() -> None:
    counter = InvocationCounterProvider()
    agent = Agent(
        client=OpenAIChatClient(),
        context_providers=[counter],
    )

    session = agent.create_session()
    await agent.run("What is Python?", session=session)
    await agent.run("What is asyncio?", session=session)
    await agent.run("What is agent-framework?", session=session)
    # Invocation #1 complete. ...
    # Invocation #2 complete. ...
    # Invocation #3 complete. ...


asyncio.run(main())
```

---

## 10. `InMemoryCheckpointStorage` — Workflow Testing Patterns

**Module:** `agent_framework._workflows._checkpoint`  
**Import:** `from agent_framework import InMemoryCheckpointStorage`

`InMemoryCheckpointStorage` is a checkpoint backend that stores workflow state in a
Python dict. It is the recommended storage backend for **unit tests** and local
development because it requires no filesystem or database and resets automatically
when the process exits.

**Constructor:**

```python
InMemoryCheckpointStorage()  # no arguments
```

**API:**

| Method | Signature | Description |
|---|---|---|
| `save` | `(checkpoint) -> CheckpointID` | Save and return the ID |
| `load` | `(checkpoint_id) -> WorkflowCheckpoint` | Load by ID; raises on missing |
| `list_checkpoints` | `(*, workflow_name) -> list[WorkflowCheckpoint]` | All checkpoints for a workflow |
| `list_checkpoint_ids` | `(*, workflow_name) -> list[CheckpointID]` | IDs only |
| `get_latest` | `(*, workflow_name) -> WorkflowCheckpoint \| None` | Most recent by timestamp |
| `delete` | `(checkpoint_id) -> bool` | Delete; returns True if found |

### Example 1 — Run a checkpointed workflow with in-memory storage

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, InMemoryCheckpointStorage
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    storage = InMemoryCheckpointStorage()

    researcher = Agent(client=OpenAIChatClient(), instructions="Research the topic thoroughly.")
    writer = Agent(client=OpenAIChatClient(), instructions="Write a polished summary.")

    workflow = (
        WorkflowBuilder()
        .add_agent(researcher, id="research")
        .add_agent(writer, id="write")
        .connect("research", "write")
        .with_checkpointing(storage=storage, workflow_name="research-write")
        .build()
    )

    result = await workflow.run("Explain multi-agent orchestration.")
    print(result.text)

    # Inspect what was saved
    checkpoints = await storage.list_checkpoints(workflow_name="research-write")
    print(f"Saved {len(checkpoints)} checkpoint(s)")


asyncio.run(main())
```

### Example 2 — Resume a workflow from a checkpoint

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, InMemoryCheckpointStorage
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    storage = InMemoryCheckpointStorage()

    step1 = Agent(client=OpenAIChatClient(), instructions="Outline the topic.")
    step2 = Agent(client=OpenAIChatClient(), instructions="Expand each outline point.")
    step3 = Agent(client=OpenAIChatClient(), instructions="Write a conclusion.")

    workflow = (
        WorkflowBuilder()
        .add_agent(step1, id="outline")
        .add_agent(step2, id="expand")
        .add_agent(step3, id="conclude")
        .connect("outline", "expand")
        .connect("expand", "conclude")
        .with_checkpointing(storage=storage, workflow_name="essay-pipeline")
        .build()
    )

    # First run: checkpoint saved after each step
    result = await workflow.run("Write an essay about context providers.")

    # Get the latest checkpoint and resume (simulates process restart)
    checkpoint = await storage.get_latest(workflow_name="essay-pipeline")
    if checkpoint:
        resumed = await workflow.resume(checkpoint.checkpoint_id, storage=storage)
        print(resumed.text)


asyncio.run(main())
```

### Example 3 — Test a workflow's checkpoint behaviour

Use `InMemoryCheckpointStorage` in a unit test to verify that checkpoints are created
at the expected steps without touching disk.

```python
import asyncio
import pytest
from agent_framework import Agent, WorkflowBuilder, InMemoryCheckpointStorage
from agent_framework.openai import OpenAIChatClient


async def build_workflow(storage: InMemoryCheckpointStorage):
    step_a = Agent(client=OpenAIChatClient(), instructions="Step A: categorise the input.")
    step_b = Agent(client=OpenAIChatClient(), instructions="Step B: enrich the category.")

    return (
        WorkflowBuilder()
        .add_agent(step_a, id="categorise")
        .add_agent(step_b, id="enrich")
        .connect("categorise", "enrich")
        .with_checkpointing(storage=storage, workflow_name="test-workflow")
        .build()
    )


@pytest.mark.asyncio
async def test_checkpoints_are_saved():
    storage = InMemoryCheckpointStorage()
    workflow = await build_workflow(storage)

    await workflow.run("Test input: invoice document")

    checkpoints = await storage.list_checkpoints(workflow_name="test-workflow")
    # Expect at least one checkpoint (after step_a completes)
    assert len(checkpoints) >= 1

    latest = await storage.get_latest(workflow_name="test-workflow")
    assert latest is not None
    assert latest.workflow_name == "test-workflow"


@pytest.mark.asyncio
async def test_checkpoint_delete():
    storage = InMemoryCheckpointStorage()
    workflow = await build_workflow(storage)

    await workflow.run("Another test input")

    ids = await storage.list_checkpoint_ids(workflow_name="test-workflow")
    assert len(ids) > 0

    deleted = await storage.delete(ids[0])
    assert deleted is True

    remaining = await storage.list_checkpoint_ids(workflow_name="test-workflow")
    assert ids[0] not in remaining
```

---

## Quick-Reference: New in 1.10.0

| Class / Function | Module | What's new |
|---|---|---|
| `FileMemoryProvider` | `agent_framework._harness._file_memory` | Session-scoped file memory; 7 built-in tools |
| `create_harness_agent` | `agent_framework._harness._agent` | Batteries-included factory for fully-wired agents |
| `AgentLoopMiddleware` | `agent_framework._harness._loop` | `fresh_context`, `return_final_only`, `record_feedback` |
| `FileSkillsSource` | `agent_framework._skills` | `search_depth`, `script_filter`, `resource_filter` |
| `FoundryAgent` | `agent_framework.foundry` | Wraps existing Foundry PromptAgent / HostedAgent |
| `AgentExecutorResponse.with_text` | `agent_framework._workflows._agent_executor` | Preserves full conversation on text transform |
| `todos_remaining` | `agent_framework._harness._loop` | `looping_modes` parameter |
| `background_tasks_running` | `agent_framework._harness._loop` | Resolves `BackgroundAgentsProvider` from agent |

---

## See also

- [Vol. 17](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v17/) — `ToolApprovalMiddleware`, `AgentLoopMiddleware` (basics), `JudgeVerdict`
- [Vol. 24](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v24/) — `ToolApprovalScope`, `AgentLoopMiddleware` callable types, `ALWAYS_APPROVE` constants
- [Vol. 30](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v30/) — `BackgroundAgentsProvider`, `TodoItem`, `CompactionProvider`
- [Installation & Quickstart](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_installation_and_quickstart/) — getting started with `agent-framework` 1.10.0
- [Recipes](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/) — copy-paste patterns for common use cases

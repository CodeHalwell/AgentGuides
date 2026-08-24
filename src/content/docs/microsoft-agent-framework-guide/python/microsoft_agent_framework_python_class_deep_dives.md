---
title: "Microsoft Agent Framework (Python) — 10-Class Deep Dives (1.15.0)"
description: "Source-verified deep dives for WorkflowViz, FileMemoryProvider, AgentModeProvider, BackgroundAgentsProvider, ToolApprovalMiddleware, SwitchCaseEdgeGroup, MessageInjectionMiddleware, ToolResultCompactionStrategy, SummarizationStrategy, and TokenBudgetComposedStrategy — all verified against agent-framework 1.15.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-Class Deep Dives

**Verified against:** `agent-framework==1.15.0`
**Python requirement:** 3.10+

Each section below examines one public class from the `agent_framework` package in depth: what it does, its full `__init__` signature, every meaningful method, and one or more self-contained runnable examples verified against the 1.15.0 source.

---

## 1. `WorkflowViz`

**Module:** `agent_framework._workflows._viz` (re-exported via `agent_framework`)

`WorkflowViz` renders any `Workflow` object to DOT, Mermaid, SVG, PNG, or PDF. It walks the workflow's edge groups, identifies fan-in nodes, and emits recursively nested subgraphs for any `WorkflowExecutor` nodes.

### Constructor

```python
WorkflowViz(workflow: Workflow)
```

### Key methods

| Method | Returns | Notes |
|---|---|---|
| `to_digraph(include_internal_executors=False)` | `str` | DOT-format string. No extra dependencies. |
| `to_mermaid(include_internal_executors=False)` | `str` | Mermaid `flowchart TD` string. No extra dependencies. |
| `export(format, filename=None, include_internal_executors=False)` | `str` | `format` ∈ `"svg"`, `"png"`, `"pdf"`, `"dot"`. Writes a file; returns the path. SVG/PNG/PDF require `pip install graphviz>=0.20.0` **and** the system `graphviz` package. |
| `save_svg(filename, ...)` | `str` | Convenience wrapper for `export(format="svg", ...)`. |
| `save_png(filename, ...)` | `str` | Convenience wrapper for `export(format="png", ...)`. |
| `save_pdf(filename, ...)` | `str` | Convenience wrapper for `export(format="pdf", ...)`. |

### Installation

```bash
pip install agent-framework
pip install 'graphviz>=0.20.0'    # Python binding
apt-get install graphviz           # or: brew install graphviz
```

### Example — Mermaid and DOT in two lines

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, WorkflowViz
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

researcher = Agent(client=client, name="researcher",
                   instructions="Return facts about the topic as bullet points.")
writer     = Agent(client=client, name="writer",
                   instructions="Turn the bullet points into a polished paragraph.")

builder = WorkflowBuilder(start_executor=researcher)
builder.add_edge(researcher, writer)
workflow = builder.build()

viz = WorkflowViz(workflow)

mermaid = viz.to_mermaid()
print(mermaid)
# flowchart TD
#   researcher["researcher (Start)"];
#   writer["writer"];
#   researcher --> writer;

dot = viz.to_digraph()
print(dot)
# digraph Workflow { ... }
```

### Example — Export to SVG file

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, WorkflowViz
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
a = Agent(client=client, name="ingest")
b = Agent(client=client, name="enrich")
c = Agent(client=client, name="publish")

builder = WorkflowBuilder(start_executor=a)
builder.add_edge(a, b)
builder.add_edge(b, c)
workflow = builder.build()

path = WorkflowViz(workflow).save_svg("pipeline.svg")
print(f"Diagram saved to: {path}")
```

### Example — Conditional edges appear as dashed lines

```python
from agent_framework import Agent, WorkflowBuilder, WorkflowViz
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
classifier = Agent(client=client, name="classifier")
high_path  = Agent(client=client, name="high_priority_handler")
low_path   = Agent(client=client, name="low_priority_handler")

builder = WorkflowBuilder(start_executor=classifier)
builder.add_edge(classifier, high_path,
                 condition=lambda data: "high" in data.agent_response.text.lower())
builder.add_edge(classifier, low_path,
                 condition=lambda data: "high" not in data.agent_response.text.lower())
workflow = builder.build()

print(WorkflowViz(workflow).to_mermaid())
# classifier -. conditional .-> high_priority_handler;
# classifier -. conditional .-> low_priority_handler;
```

---

## 2. `FileMemoryProvider`

**Module:** `agent_framework._harness._file_memory` (re-exported via `agent_framework`)

`FileMemoryProvider` is a `ContextProvider` that gives an agent a **session-scoped, file-based memory** system. Each session gets its own working folder (derived from its `session_id`). The provider registers seven tools on the session context:

| Tool | Purpose |
|---|---|
| `file_memory_write` | Write (or overwrite) a memory file with optional description |
| `file_memory_read` | Read a file by name |
| `file_memory_delete` | Delete a file and its description sidecar |
| `file_memory_ls` | List files with optional glob filter |
| `file_memory_grep` | Case-insensitive regex search across files |
| `file_memory_replace` | Replace a substring in a file |
| `file_memory_replace_lines` | Replace specific lines by 1-based line number |

A `memories.md` index is maintained automatically and injected into the agent's context each turn so the model knows what it has stored.

### Constructor

```python
FileMemoryProvider(
    store: AgentFileStore,
    *,
    source_id: str = "file_memory",
    scope: str | None = None,        # None → isolate per session_id
    instructions: str | None = None, # None → use built-in instructions
)
```

### Key design points from source

- **Flat namespace**: Files may not contain path separators. `"notes/plan.md"` is rejected; `"plan.md"` is accepted.
- **Thread safety**: A per-instance `asyncio.Lock` serializes all write/delete operations and index rebuilds.
- **Index cap**: The `memories.md` index shows at most 50 entries.
- **Scope override**: Pass `scope="user-123"` to share memory across all sessions for a given user.

### Example — In-memory store for testing

```python
import asyncio
from agent_framework import Agent, FileMemoryProvider, InMemoryAgentFileStore
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    store = InMemoryAgentFileStore()
    memory = FileMemoryProvider(store=store)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a research assistant. Store important findings in memory.",
        context_providers=[memory],
    )

    session = agent.create_session()

    # First turn — agent discovers and stores a fact
    r1 = await agent.run(
        "Remember that the Eiffel Tower is 330 metres tall. Store it in memory.",
        session=session,
    )
    print(r1.text)

    # Second turn — agent reads from memory
    r2 = await agent.run("How tall is the Eiffel Tower?", session=session)
    print(r2.text)


asyncio.run(main())
```

### Example — Disk-backed store with cross-session scope

```python
import asyncio
from agent_framework import Agent, FileMemoryProvider, FileSystemAgentFileStore
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Persist to disk; all sessions for user "alice" share the same folder.
    store = FileSystemAgentFileStore(root_directory="/tmp/agent_memory")
    memory = FileMemoryProvider(store=store, scope="user-alice")

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a personal assistant. Keep notes about the user.",
        context_providers=[memory],
    )

    # Session A — write a preference
    session_a = agent.create_session()
    await agent.run(
        "The user prefers dark mode. Store this preference.",
        session=session_a,
    )

    # Session B — different session, same scope → same files
    session_b = agent.create_session()
    result = await agent.run(
        "Does the user have any stored preferences?",
        session=session_b,
    )
    print(result.text)  # Should mention dark mode preference


asyncio.run(main())
```

### Example — Description-aided discovery

```python
import asyncio
from agent_framework import Agent, FileMemoryProvider, InMemoryAgentFileStore
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    store = InMemoryAgentFileStore()
    memory = FileMemoryProvider(store=store)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions=(
            "You are a research agent. When you encounter large data (API responses, "
            "research results), write it to a memory file with a concise description. "
            "Use file_memory_grep to search for specific facts before asking me."
        ),
        context_providers=[memory],
    )
    session = agent.create_session()

    # The agent is instructed to write with a description so file_memory_ls can show context
    await agent.run(
        "Store the following: 'GPT-4 context window is 128k tokens'. "
        "Write it to a file called 'model_facts.md' with a helpful description.",
        session=session,
    )

    result = await agent.run(
        "Use grep to search your memory for facts about GPT-4.",
        session=session,
    )
    print(result.text)


asyncio.run(main())
```

---

## 3. `AgentModeProvider`

**Module:** `agent_framework._harness._mode` (re-exported via `agent_framework`)

`AgentModeProvider` is a `ContextProvider` that lets an agent switch between **operating modes** stored in session state. The default modes are `plan` (interactive, question-asking) and `execute` (autonomous). The provider exposes two tools:

| Tool | Purpose |
|---|---|
| `mode_get` | Return the current mode as JSON |
| `mode_set` | Switch to a different mode |

Two standalone helpers let orchestrators change the mode **externally** (outside of the agent's own tool call):

```python
get_agent_mode(session, ...) -> str
set_agent_mode(session, mode, ...) -> str
```

When the mode is changed externally, the next `before_run` injects a `user`-role message announcing the switch so the model re-orients without re-reading its system instructions.

### Constructor

```python
AgentModeProvider(
    source_id: str = "agent_mode",
    *,
    default_mode: str | None = None,            # None → first mode in mode_instructions
    mode_instructions: Mapping[str, str] | None = None,  # None → built-in plan/execute
    instructions: str | None = None,            # None → built-in instructions template
)
```

### Built-in mode workflow (from source)

**Plan mode** steps:
1. Analyse requirements and build a research plan
2. Create a todo list
3. Do exploratory checks if needed
4. Ask clarifying questions one by one
5. Write the plan to a memory file
6. Present the plan and ask for approval to switch to execute mode
7. When approved, call `mode_set("execute")`

**Execute mode** behaviour:
- If it's a simple question, answer directly
- Otherwise work autonomously, make decisions without asking the user, mark tasks as completed, and continue until done

### Example — Two-mode plan/execute agent

```python
import asyncio
from agent_framework import Agent, AgentModeProvider, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    mode_provider = AgentModeProvider()
    history = InMemoryHistoryProvider()

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a project planning assistant.",
        context_providers=[history, mode_provider],
    )
    session = agent.create_session()

    # Agent starts in plan mode — it will ask clarifying questions
    r1 = await agent.run(
        "I want to build a REST API for managing todos.",
        session=session,
    )
    print("PLAN MODE:", r1.text)

    # Simulate user approval — switch externally
    from agent_framework import set_agent_mode
    set_agent_mode(session, "execute")

    # Next turn — agent runs in execute mode
    r2 = await agent.run("Proceed with the plan.", session=session)
    print("EXECUTE MODE:", r2.text)


asyncio.run(main())
```

### Example — Custom modes

```python
import asyncio
from agent_framework import Agent, AgentModeProvider, get_agent_mode, set_agent_mode
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    mode_provider = AgentModeProvider(
        default_mode="research",
        mode_instructions={
            "research": (
                "Search for information, gather facts, and compile a brief report. "
                "Do not take any actions — only read and report."
            ),
            "write": (
                "Take the research report and produce a polished 500-word article. "
                "Work autonomously without asking questions."
            ),
            "review": (
                "Review the article for factual accuracy and grammar. "
                "Return a list of specific issues found."
            ),
        },
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a content creation assistant.",
        context_providers=[mode_provider],
    )

    session = agent.create_session()

    r1 = await agent.run("Research recent developments in quantum computing.", session=session)
    print("RESEARCH:", r1.text[:200])

    custom_modes = ["research", "write", "review"]
    set_agent_mode(session, "write", available_modes=custom_modes)
    r2 = await agent.run("Write the article now.", session=session)
    print("WRITE:", r2.text[:200])

    current = get_agent_mode(session, available_modes=custom_modes)
    print(f"Current mode: {current}")  # → write


asyncio.run(main())
```

### Example — Reading mode externally before routing

```python
from agent_framework import AgentSession, get_agent_mode, set_agent_mode

def route_next_step(session: AgentSession) -> str:
    """Return the next workflow step based on the agent's current mode."""
    mode = get_agent_mode(session)
    if mode == "plan":
        return "wait_for_approval"
    elif mode == "execute":
        return "run_tasks"
    else:
        return "unknown"
```

---

## 4. `BackgroundAgentsProvider`

**Module:** `agent_framework._harness._background_agents` (re-exported via `agent_framework`)

`BackgroundAgentsProvider` is a `ContextProvider` (marked experimental) that enables a **parent agent** to delegate work to named **background sub-agents** running concurrently in their own sessions. The parent doesn't block on their completion — it fires them off and checks later.

### Tools exposed to the agent

| Tool | Purpose |
|---|---|
| `background_agents_start_task` | Fire a task at a named agent; returns task ID |
| `background_agents_wait_for_first_completion` | Block until the first of N tasks finishes |
| `background_agents_get_task_results` | Read output of a completed task |
| `background_agents_get_all_tasks` | List all tasks with status |
| `background_agents_continue_task` | Send follow-up to a completed task's session |
| `background_agents_clear_completed_task` | Free a completed task's session |

### Task lifecycle

```
start_task → RUNNING → (wait_for_first_completion) → COMPLETED or FAILED
                                                   ↓
                                          get_task_results
                                                   ↓
                                      continue_task (optional)
                                                   ↓
                                      clear_completed_task
```

### Constructor

```python
BackgroundAgentsProvider(
    agents: Sequence[SupportsAgentRun],  # must have unique non-empty .name
    *,
    source_id: str = "background_agents",
    instructions: str | None = None,     # may contain {background_agents} placeholder
)
```

**Security note (from source):** Background agents receive text from the parent that may include untrusted content. Only supply agents you trust with the data the parent may pass to them.

### Example — Parallel research tasks

```python
import asyncio
from agent_framework import Agent, BackgroundAgentsProvider, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    # Sub-agents that will run in the background
    python_expert = Agent(
        client=client,
        name="python_expert",
        description="Answers Python-specific questions and explains Python concepts.",
        instructions="You are a Python expert. Answer concisely.",
    )
    rust_expert = Agent(
        client=client,
        name="rust_expert",
        description="Answers Rust-specific questions and explains Rust concepts.",
        instructions="You are a Rust expert. Answer concisely.",
    )

    # Orchestrator that delegates to background agents
    orchestrator = Agent(
        client=client,
        instructions=(
            "You are a research coordinator. When asked a multi-part question, "
            "delegate each part to the appropriate background agent, wait for all "
            "results, then synthesise a combined answer."
        ),
        context_providers=[
            InMemoryHistoryProvider(),
            BackgroundAgentsProvider(agents=[python_expert, rust_expert]),
        ],
    )

    session = orchestrator.create_session()
    result = await orchestrator.run(
        "Compare Python and Rust's approaches to memory management. "
        "Delegate each language to its expert and combine the results.",
        session=session,
    )
    print(result.text)


asyncio.run(main())
```

### Example — Managing task lifecycle explicitly

```python
import asyncio
from agent_framework import Agent, BackgroundAgentsProvider
from agent_framework.openai import OpenAIChatClient


async def show_task_management() -> None:
    """Illustrates the explicit task management pattern."""
    client = OpenAIChatClient()

    summarizer = Agent(
        client=client,
        name="summarizer",
        description="Summarizes text in 2-3 sentences.",
        instructions="You are a concise summarizer.",
    )

    orchestrator = Agent(
        client=client,
        instructions=(
            "You are an orchestrator. "
            "Use background_agents_start_task to start tasks, "
            "background_agents_wait_for_first_completion to wait, "
            "background_agents_get_task_results to retrieve, "
            "and background_agents_clear_completed_task to clean up."
        ),
        context_providers=[BackgroundAgentsProvider(agents=[summarizer])],
    )

    session = orchestrator.create_session()
    result = await orchestrator.run(
        "Start two summarization tasks: "
        "(1) Summarize 'Python is a high-level, interpreted language...' "
        "(2) Summarize 'Rust is a systems programming language...' "
        "Wait for both to finish, collect results, then clear the tasks.",
        session=session,
    )
    print(result.text)


asyncio.run(show_task_management())
```

### Example — Release session to clean up runtime state

```python
import asyncio
from agent_framework import Agent, BackgroundAgentsProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    worker_agent = Agent(client=client, instructions="Process documents.")

    # BackgroundAgentsProvider holds asyncio.Task and AgentSession references.
    # Call release_session() when the parent session is done to prevent leaks.
    provider = BackgroundAgentsProvider(agents=[worker_agent])
    agent = Agent(client=client, context_providers=[provider])
    session = agent.create_session()

    try:
        await agent.run("Process these documents in parallel.", session=session)
    finally:
        await provider.release_session(session, cancel_running=True, timeout=30.0)


asyncio.run(main())
```

---

## 5. `ToolApprovalMiddleware`

**Module:** `agent_framework._harness._tool_approval` (re-exported via `agent_framework`)

`ToolApprovalMiddleware` is an `AgentMiddleware` that **gates tool calls through a human-approval loop**. When the model requests a tool call, the middleware intercepts it, produces a `function_approval_request` content item, and only executes the tool after the caller confirms. Approvals are stored in session state so the user's decision survives restarts.

### Constructor

```python
ToolApprovalMiddleware(
    *,
    source_id: str = "tool_approval",
    auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None,
)
```

`auto_approval_rules` is a list of callables that receive a `function_call` content and return `True` to auto-approve without a human prompt.

> **Security (from source):** An auto-approval rule matched by name may approve *any* local tool with that name, not only the specific tool the rule was designed for. Name your tools carefully to avoid unintended matches.

### Example — Basic human-in-the-loop approval

```python
import asyncio
from agent_framework import Agent, AgentSession, ToolApprovalMiddleware
from agent_framework.openai import OpenAIChatClient
from agent_framework import tool


@tool
def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    import os
    os.remove(path)
    return f"Deleted {path}"


async def main() -> None:
    approval_middleware = ToolApprovalMiddleware()

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a file management assistant.",
        tools=[delete_file],
        middleware=[approval_middleware],
    )

    session = agent.create_session()

    # First call — agent requests delete_file
    response = await agent.run(
        "Please delete /tmp/test.txt",
        session=session,
    )

    # The response will contain an approval_request content item
    for msg in response.messages:
        for content in msg.contents:
            if content.type == "function_approval_request":
                print(f"Approval requested for: {content.name}({content.arguments})")
                # In a real app, show this to the user and get their decision


asyncio.run(main())
```

### Example — Auto-approve read-only tools

> **Security:** Auto-approving by tool name alone means *any* tool registered under that name is
> approved without a human prompt. For filesystem tools, always restrict the paths they can access
> (e.g. require a safe root prefix) before adding them to an auto-approval allowlist — otherwise
> the model can read `/etc/passwd` or other sensitive files without human review.

```python
import asyncio
import os
from agent_framework import Agent, ToolApprovalMiddleware
from agent_framework.openai import OpenAIChatClient
from agent_framework import tool

READ_ONLY_TOOLS = {"list_files", "read_file", "search_docs"}
SAFE_ROOT = "/tmp/workdir"  # restrict tools to this directory


def auto_approve_read_only(function_call) -> bool:
    """Auto-approve known read-only tools without human prompt."""
    return function_call.name in READ_ONLY_TOOLS


@tool
def list_files(directory: str) -> list[str]:
    """List files in the given directory (must be under /tmp/workdir)."""
    full = os.path.realpath(directory)
    if not full.startswith(os.path.realpath(SAFE_ROOT)):
        raise ValueError(f"Access denied: {directory}")
    return os.listdir(full)


@tool
def read_file(path: str) -> str:
    """Read the contents of a file (must be under /tmp/workdir)."""
    full = os.path.realpath(path)
    if not full.startswith(os.path.realpath(SAFE_ROOT)):
        raise ValueError(f"Access denied: {path}")
    with open(full) as f:
        return f.read()


@tool
def delete_file(path: str) -> str:
    """Delete a file — requires explicit approval."""
    import os
    os.remove(path)
    return f"Deleted {path}"


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a file assistant. List and read files freely; always confirm deletes.",
        tools=[list_files, read_file, delete_file],
        middleware=[
            ToolApprovalMiddleware(auto_approval_rules=[auto_approve_read_only])
        ],
    )
    session = agent.create_session()

    # list_files and read_file → auto-approved
    # delete_file → approval_request returned for human confirmation
    result = await agent.run(
        "List files in /tmp, then delete /tmp/old_log.txt",
        session=session,
    )
    print(result.text)


asyncio.run(main())
```

---

## 6. `SwitchCaseEdgeGroup`

**Module:** `agent_framework._workflows._edge` (re-exported via `agent_framework`)

`SwitchCaseEdgeGroup` models a **switch/case control flow** in a workflow graph. Exactly one branch fires per message — the runtime evaluates case predicates in order and routes to the first match; the mandatory `Default` catches everything else.

### Class hierarchy

```
EdgeGroup
└── FanOutEdgeGroup
    └── SwitchCaseEdgeGroup   ← wraps SwitchCaseEdgeGroupCase / SwitchCaseEdgeGroupDefault
```

### Supporting dataclasses

```python
SwitchCaseEdgeGroupCase(condition: Callable[[Any], bool], target_id: str)
SwitchCaseEdgeGroupDefault(target_id: str)
```

### Constructor

```python
SwitchCaseEdgeGroup(
    source_id: str,
    cases: Sequence[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault],
    *,
    id: str | None = None,
)
```

**Constraints (from source):**
- Must have at least two cases (including exactly one default)
- Default must be last (warning is emitted if it isn't)
- Predicates are evaluated in order; first match wins

### Example — Content-type routing

```python
import asyncio
from agent_framework import (
    Agent,
    WorkflowBuilder,
    Case,
    Default,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    classifier  = Agent(client=client, name="classifier",
                        instructions="Classify the input: respond with 'type: json', 'type: csv', or 'type: text'.")
    json_agent  = Agent(client=client, name="json_processor",
                        instructions="Parse and summarize the JSON data.")
    csv_agent   = Agent(client=client, name="csv_processor",
                        instructions="Parse and summarize the CSV data.")
    text_agent  = Agent(client=client, name="text_processor",
                        instructions="Summarize the text document.")

    # Route from classifier based on the output text.
    # Conditions receive AgentExecutorResponse — access text via .agent_response.text.
    builder = WorkflowBuilder(start_executor=classifier)
    builder.add_switch_case_edge_group(
        classifier,
        [
            Case(
                condition=lambda data: "type: json" in data.agent_response.text.lower(),
                target=json_agent,
            ),
            Case(
                condition=lambda data: "type: csv" in data.agent_response.text.lower(),
                target=csv_agent,
            ),
            Default(target=text_agent),
        ],
    )
    workflow = builder.build()

    result = await workflow.run('{"name": "Alice", "age": 30}')
    print(result.get_outputs()[-1].agent_response.text)


asyncio.run(main())
```

### Example — Priority routing with serialization

```python
from agent_framework import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)

# Build the group
group = SwitchCaseEdgeGroup(
    source_id="triage",
    cases=[
        SwitchCaseEdgeGroupCase(
            condition=lambda msg: msg.get("severity") == "critical",
            target_id="emergency_handler",
        ),
        SwitchCaseEdgeGroupCase(
            condition=lambda msg: msg.get("severity") == "high",
            target_id="priority_handler",
        ),
        SwitchCaseEdgeGroupDefault(target_id="standard_handler"),
    ],
)

# Serialize / inspect
snapshot = group.to_dict()
print(snapshot["cases"])
# [
#   {"type": "Case", "condition_name": "<lambda>", "target_id": "emergency_handler"},
#   {"type": "Case", "condition_name": "<lambda>", "target_id": "priority_handler"},
#   {"type": "Default", "target_id": "standard_handler"},
# ]
```

### Example — WorkflowViz shows switch branches correctly

```python
from agent_framework import Agent, WorkflowBuilder, Case, Default, WorkflowViz
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
triage          = Agent(client=client, name="triage")
emergency       = Agent(client=client, name="emergency_handler")
priority        = Agent(client=client, name="priority_handler")
standard        = Agent(client=client, name="standard_handler")

builder = WorkflowBuilder(start_executor=triage)
builder.add_switch_case_edge_group(
    triage,
    [
        Case(condition=lambda data: "critical" in data.agent_response.text.lower(), target=emergency),
        Case(condition=lambda data: "high" in data.agent_response.text.lower(), target=priority),
        Default(target=standard),
    ],
)
workflow = builder.build()

# WorkflowViz renders switch branches as dashed (conditional) lines
mermaid = WorkflowViz(workflow).to_mermaid()
# triage -. conditional .-> emergency_handler;
# triage -. conditional .-> priority_handler;
# triage -. conditional .-> standard_handler;
print(mermaid)
```

---

## 7. `MessageInjectionMiddleware`

**Module:** `agent_framework._sessions` (re-exported via `agent_framework`)

`MessageInjectionMiddleware` is a `ChatMiddleware` that lets **tool code or external orchestrators inject messages into a running agent session**. Injected messages are queued in session state and drained into the next model call automatically.

The free function `enqueue_messages(session, messages)` queues messages from any code path — inside a tool, in a `FunctionInvocationContext`, or from the calling application.

### Constructor

```python
MessageInjectionMiddleware()  # no parameters
```

### Key methods

| Method | Purpose |
|---|---|
| `enqueue_messages(session, messages)` | Queue messages for the next model call |
| `get_pending_messages(session)` | Snapshot the current queue without draining |

### Loop behaviour (from source)

After each model call, if there are newly queued messages **and** the response doesn't contain function calls that the function-layer must handle, the middleware loops internally — draining the queue and calling the model again. This avoids needing an explicit outer loop for simple injections.

### Example — Inject a message from inside a tool

```python
import asyncio
from agent_framework import (
    Agent, AgentSession, MessageInjectionMiddleware,
    FunctionInvocationContext, enqueue_messages
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import tool


injection_middleware = MessageInjectionMiddleware()


@tool
def fetch_user_profile(user_id: str, context: FunctionInvocationContext) -> str:
    """Fetch a user profile and inject enrichment messages into the session."""
    # Simulate a DB lookup
    profile = {"name": "Alice", "plan": "Pro", "credits": 150}

    # Inject extra context into the model's next turn
    if context.session:
        enqueue_messages(
            context.session,
            f"[System enrichment] User profile loaded: {profile}",
        )

    return f"Profile fetched for {user_id}"


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a billing support agent.",
        tools=[fetch_user_profile],
        middleware=[injection_middleware],
    )

    session = agent.create_session()
    result = await agent.run(
        "Look up profile for user user-42 and tell me about their plan.",
        session=session,
    )
    print(result.text)  # Will have access to the injected profile data


asyncio.run(main())
```

### Example — Inject from the calling application between turns

```python
import asyncio
from agent_framework import Agent, AgentSession, MessageInjectionMiddleware, enqueue_messages
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    injection_mw = MessageInjectionMiddleware()

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a monitoring agent.",
        middleware=[injection_mw],
    )

    session = agent.create_session()

    # First turn
    await agent.run("Start monitoring.", session=session)

    # Between turns — inject an alert from the application layer
    enqueue_messages(
        session,
        "⚠️ Alert: CPU usage exceeded 90% on node-3 at 14:32 UTC.",
    )

    # Next turn — the agent receives the injected alert in its context
    result = await agent.run(
        "Any new alerts to process?",
        session=session,
    )
    print(result.text)  # Agent will see and respond to the CPU alert


asyncio.run(main())
```

### Example — Inspect the queue before a run

```python
import asyncio
from agent_framework import Agent, MessageInjectionMiddleware, enqueue_messages, Message
from agent_framework.openai import OpenAIChatClient


async def inspect_queue() -> None:
    mw = MessageInjectionMiddleware()
    agent = Agent(client=OpenAIChatClient(), middleware=[mw])
    session = agent.create_session()

    enqueue_messages(session, "First injected message")
    enqueue_messages(session, "Second injected message")

    pending = mw.get_pending_messages(session)
    print(f"Queued: {len(pending)} messages")  # → 2

    for msg in pending:
        print(f"  [{msg.role}] {msg.text}")


asyncio.run(inspect_queue())
```

---

## 8. `ToolResultCompactionStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`ToolResultCompactionStrategy` **collapses older tool-call groups** into compact summary messages of the form `[Tool results: tool_name: result; ...]`. Unlike `SelectiveToolCallCompactionStrategy` (which fully excludes old groups), this strategy preserves a readable trace of what tools returned while reducing token overhead.

### Constructor

```python
ToolResultCompactionStrategy(
    *,
    keep_last_tool_call_groups: int = 1,  # must be >= 0
)
```

`keep_last_tool_call_groups` controls how many recent tool-call groups are left verbatim; all older ones are collapsed.

### Mechanics (from source)

1. Identifies all included tool-call groups
2. Builds a `call_id → tool_name` map from function-call contents
3. Collects tool results with their tool names
4. Inserts a summary message: `[Tool results: weather: sunny 18°C; translate: Bonjour]`
5. Marks original group messages as excluded
6. Adds bi-directional summary/source annotations

### Example — Compact old tool results after each turn

```python
import asyncio
from agent_framework import (
    Agent, CompactionProvider, InMemoryHistoryProvider,
    ToolResultCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 22°C in {city}"


@tool
def get_population(city: str) -> str:
    """Get population of a city."""
    return f"{city} has 3.5 million people"


async def main() -> None:
    history = InMemoryHistoryProvider()
    # Keep only the last 1 tool-call group verbatim; collapse all older ones
    compaction = CompactionProvider(
        after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
        history_source_id=history.source_id,
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a city information assistant.",
        tools=[get_weather, get_population],
        context_providers=[history, compaction],
    )

    session = agent.create_session()

    # Multiple turns — tool results accumulate
    await agent.run("What's the weather in Paris?", session=session)
    await agent.run("And the population?", session=session)

    # By now older tool results are collapsed into: [Tool results: get_weather: Sunny, 22°C in Paris]
    result = await agent.run("Summarize what you found about Paris.", session=session)
    print(result.text)


asyncio.run(main())
```

### Example — Combine with `TokenBudgetComposedStrategy`

```python
from agent_framework import (
    Agent, TokenBudgetComposedStrategy, ToolResultCompactionStrategy,
    SlidingWindowStrategy, CharacterEstimatorTokenizer,
    InMemoryHistoryProvider, CompactionProvider,
)
from agent_framework.openai import OpenAIChatClient


# Compose: first compact tool results, then slide the window, all within 8k tokens
tokenizer = CharacterEstimatorTokenizer()
strategy = TokenBudgetComposedStrategy(
    token_budget=8_000,
    tokenizer=tokenizer,
    strategies=[
        ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
        SlidingWindowStrategy(keep_last_groups=15),
    ],
)

history = InMemoryHistoryProvider()
compaction = CompactionProvider(
    before_strategy=strategy,
    history_source_id=history.source_id,
    tokenizer=tokenizer,
)

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[history, compaction],
)
```

---

## 9. `SummarizationStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`SummarizationStrategy` **replaces older message groups with LLM-generated summary text** rather than truncating or fully discarding them. It triggers when the count of included non-system messages exceeds `target_count + threshold`, summarizes the oldest groups, and keeps the newest.

### Constructor

```python
SummarizationStrategy(
    *,
    client: SupportsChatGetResponse,     # chat client used for summarization
    target_count: int = 4,               # target included non-system message count
    threshold: int | None = 2,           # buffer before triggering
    prompt: str | None = None,           # None → built-in 5-sentence prompt
    max_summary_input_tokens: int | None = 8_000,  # token budget for summarizer input
    tokenizer: TokenizerProtocol | None = None,
)
```

### Built-in prompt (from source)

> Generate a clear and complete summary of the entire conversation in no more than five sentences. The summary must preserve context, incorporate any earlier summary, and omit judgments or speculation.

### Security note (from source)

The summarizer's output **permanently replaces messages** in chat history and is trusted like any assistant message. A compromised summarization client could inject persistent instructions via indirect prompt injection. Only use a summarization service you trust as much as the primary model.

### Example — LLM-based compaction for long research sessions

```python
import asyncio
from agent_framework import (
    Agent, CompactionProvider, InMemoryHistoryProvider, SummarizationStrategy,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()

    # A dedicated summarization client (can use a cheaper/faster model)
    summary_client = OpenAIChatClient(model="gpt-4o-mini")

    history = InMemoryHistoryProvider()
    summarization = SummarizationStrategy(
        client=summary_client,
        target_count=6,    # keep 6 recent non-system messages verbatim
        threshold=3,       # triggers when count exceeds 9 (i.e., at 10 messages)
    )
    compaction = CompactionProvider(
        after_strategy=summarization,
        history_source_id=history.source_id,
    )

    agent = Agent(
        client=client,
        instructions="You are a long-running research agent.",
        context_providers=[history, compaction],
    )

    session = agent.create_session()

    # Simulate a long conversation — summarization will compress older turns
    topics = [
        "Explain quantum entanglement",
        "What are its applications in cryptography?",
        "Describe quantum key distribution",
        "What are the current limitations?",
        "What does the future look like?",
        "Summarize everything you've explained so far",
    ]
    for topic in topics:
        r = await agent.run(topic, session=session)
        print(f"Q: {topic}\nA: {r.text[:100]}...\n")


asyncio.run(main())
```

### Example — Custom summarization prompt

```python
from agent_framework import SummarizationStrategy
from agent_framework.openai import OpenAIChatClient

TECHNICAL_SUMMARY_PROMPT = """
Produce a technical summary of the conversation covering:
1. Problem statement and constraints
2. Approaches considered
3. Current solution and rationale
4. Open questions

Be concise but preserve all technical details. Incorporate any prior summary.
"""

strategy = SummarizationStrategy(
    client=OpenAIChatClient(model="gpt-4o-mini"),
    target_count=8,
    threshold=4,
    prompt=TECHNICAL_SUMMARY_PROMPT,
    max_summary_input_tokens=4_000,  # cap how much we send to the summarizer
)
```

### Example — Summarization inside `TokenBudgetComposedStrategy`

```python
from agent_framework import (
    TokenBudgetComposedStrategy, SummarizationStrategy,
    ToolResultCompactionStrategy, CharacterEstimatorTokenizer,
)
from agent_framework.openai import OpenAIChatClient

tokenizer = CharacterEstimatorTokenizer()

# Layer: first collapse old tool results, then summarize if still over budget
composed = TokenBudgetComposedStrategy(
    token_budget=12_000,
    tokenizer=tokenizer,
    strategies=[
        ToolResultCompactionStrategy(keep_last_tool_call_groups=3),
        SummarizationStrategy(
            client=OpenAIChatClient(model="gpt-4o-mini"),
            target_count=10,
            threshold=5,
            tokenizer=tokenizer,
        ),
    ],
    early_stop=True,  # stop as soon as budget is satisfied
)
```

---

## 10. `TokenBudgetComposedStrategy`

**Module:** `agent_framework._compaction` (re-exported via `agent_framework`)

`TokenBudgetComposedStrategy` **pipelines multiple compaction strategies under a shared token budget**. It runs each strategy in order, re-annotates token counts after each one, and stops early when the budget is met. If no strategy is sufficient, a deterministic fallback excludes the oldest groups (then anchors if needed) to enforce the limit.

### Constructor

```python
TokenBudgetComposedStrategy(
    *,
    token_budget: int,
    tokenizer: TokenizerProtocol,
    strategies: Sequence[CompactionStrategy],
    early_stop: bool = True,  # stop as soon as budget is satisfied
)
```

### Algorithm (from source)

1. Annotate groups and token counts
2. If already under budget → return unchanged (no-op)
3. For each strategy (in order): run it, re-annotate, check budget; stop early if `early_stop=True`
4. If still over budget: exclude oldest non-system groups one by one until satisfied
5. If still over budget (anchors alone exceed it): exclude system groups one by one

### Example — Three-stage compaction pipeline

```python
import asyncio
from agent_framework import (
    Agent, CompactionProvider, InMemoryHistoryProvider,
    TokenBudgetComposedStrategy, ToolResultCompactionStrategy,
    SlidingWindowStrategy, SummarizationStrategy,
    CharacterEstimatorTokenizer,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    tokenizer = CharacterEstimatorTokenizer()

    # Stage 1: collapse old tool results (cheap, no model call)
    # Stage 2: slide window (remove oldest groups)
    # Stage 3: summarize if still too long (model call, but only when needed)
    strategy = TokenBudgetComposedStrategy(
        token_budget=16_000,
        tokenizer=tokenizer,
        strategies=[
            ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
            SlidingWindowStrategy(keep_last_groups=20),
            SummarizationStrategy(
                client=OpenAIChatClient(model="gpt-4o-mini"),
                target_count=10,
                threshold=2,
                tokenizer=tokenizer,
            ),
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
    )

    session = agent.create_session()
    result = await agent.run("Begin a long research session.", session=session)
    print(result.text)


asyncio.run(main())
```

### Example — Tight budget with fallback

```python
from agent_framework import (
    TokenBudgetComposedStrategy, SlidingWindowStrategy,
    CharacterEstimatorTokenizer,
)

# Very tight budget → forces the deterministic fallback after SlidingWindowStrategy
tight_strategy = TokenBudgetComposedStrategy(
    token_budget=2_000,
    tokenizer=CharacterEstimatorTokenizer(),
    strategies=[
        SlidingWindowStrategy(keep_last_groups=5),
    ],
    early_stop=True,  # if SlidingWindowStrategy isn't enough, fallback excludes oldest groups
)
```

### Example — Disable early stop for analytics

```python
from agent_framework import (
    TokenBudgetComposedStrategy, ToolResultCompactionStrategy,
    SlidingWindowStrategy, CharacterEstimatorTokenizer,
)

# Run ALL strategies regardless of whether budget is met after an earlier stage.
# NOTE: strategies only run at all when the current history EXCEEDS token_budget.
# If the history is already under budget, the strategy returns immediately without
# calling any sub-strategy. Use a tight budget (or a very large history) so stages run.
full_compaction = TokenBudgetComposedStrategy(
    token_budget=500,         # tight — any real multi-turn history exceeds this
    tokenizer=CharacterEstimatorTokenizer(),
    strategies=[
        ToolResultCompactionStrategy(keep_last_tool_call_groups=0),  # collapse ALL tool results
        SlidingWindowStrategy(keep_last_groups=10),
    ],
    early_stop=False,         # keep running all stages even after budget is already met
)
```

---

## Quick-Reference: Which class to reach for

| Need | Class |
|---|---|
| Visualize a workflow as Mermaid / DOT / SVG | `WorkflowViz` |
| Give an agent persistent, searchable session memory | `FileMemoryProvider` |
| Structured plan → execute workflow within one agent | `AgentModeProvider` |
| Delegate subtasks to sub-agents concurrently | `BackgroundAgentsProvider` |
| Gate tool calls through human approval | `ToolApprovalMiddleware` |
| Route workflow messages with switch/case logic | `SwitchCaseEdgeGroup` |
| Inject messages into a running session from tools or app code | `MessageInjectionMiddleware` + `enqueue_messages` |
| Compact old tool results into short summaries | `ToolResultCompactionStrategy` |
| Replace old conversation with LLM-generated summaries | `SummarizationStrategy` |
| Layer multiple compaction strategies under a token budget | `TokenBudgetComposedStrategy` |

---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 8"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.0: AgentFileStore hierarchy, FileAccessProvider, MCPSkill+MCPSkillsSource, ToolMode, AgentEvalConverter+CheckResult+RubricScore, ChatContext, WorkflowAgent+WorkflowContext, TruncationStrategy, HistoryProvider+InMemoryHistoryProvider, DelegatingSkillsSource+InMemorySkillsSource+FunctionInvocationContext."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 31
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 8

Verified against **agent-framework 1.8.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source at `/usr/local/lib/python3.11/dist-packages/agent_framework/`. No API name has
been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings`

This volume focuses on **ten new class groups shipped in 1.8.0**:
the **file-system access harness** (`AgentFileStore` hierarchy + `FileAccessProvider`),
**MCP-native skill discovery** (`MCPSkill` + `MCPSkillsSource`),
**tool-choice control** (`ToolMode`), the **evaluation helper layer**
(`AgentEvalConverter` + `CheckResult` + `RubricScore`), the fully-documented
**`ChatContext`** for chat middleware, the **`WorkflowAgent` + `WorkflowContext`**
workflow-as-agent adapter, the new **`TruncationStrategy`** compaction primitive,
the **history provider base-class redesign** (`HistoryProvider` + `InMemoryHistoryProvider`),
and the **skills composition + function-invocation pipeline** additions
(`DelegatingSkillsSource`, `InMemorySkillsSource`, `FunctionInvocationContext`).

---

## Table of Contents

1. [`AgentFileStore` + `FileSystemAgentFileStore` + `InMemoryAgentFileStore`](#1-agentfilestore--filesystemagentfilestore--inmemoryagentfilestore)
2. [`FileAccessProvider` + `FileSearchResult` + `FileSearchMatch`](#2-fileaccessprovider--filesearchresult--filesearchmatch)
3. [`MCPSkill` + `MCPSkillResource` + `MCPSkillsSource`](#3-mcpskill--mcpskillresource--mcpskillssource)
4. [`ToolMode` + `validate_tool_mode`](#4-toolmode--validate_tool_mode)
5. [`AgentEvalConverter` + `ExpectedToolCall` + `CheckResult` + `RubricScore`](#5-agentévalconverter--expectedtoolcall--checkresult--rubricscore)
6. [`ChatContext`](#6-chatcontext)
7. [`WorkflowAgent` + `WorkflowContext`](#7-workflowagent--workflowcontext)
8. [`TruncationStrategy`](#8-truncationstrategy)
9. [`HistoryProvider` + `InMemoryHistoryProvider`](#9-historyprovider--inmemoryhistoryprovider)
10. [`DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`](#10-delegatingskillssource--inmemoryskillssource--functioninvocationcontext)

---

## 1. `AgentFileStore` + `FileSystemAgentFileStore` + `InMemoryAgentFileStore`

**Source:** `agent_framework._harness._file_access`  
**Status:** `@experimental(feature_id=ExperimentalFeature.HARNESS)`

The `AgentFileStore` ABC defines a portable, path-safe file I/O interface that
`FileAccessProvider` (see group 2) uses to expose CRUD + search tools to agents.
Two first-party implementations ship in 1.8.0: a disk-backed store and an
in-memory store. You can subclass `AgentFileStore` to plug in Azure Blob Storage,
AWS S3, or any other backend.

### Abstract interface (`AgentFileStore`)

```python
from abc import ABC, abstractmethod
from agent_framework import AgentFileStore, FileSearchResult

class AgentFileStore(ABC):
    @abstractmethod
    async def write_file(self, path: str, content: str, *, overwrite: bool = True) -> None: ...
    @abstractmethod
    async def read_file(self, path: str) -> str | None: ...
    @abstractmethod
    async def delete_file(self, path: str) -> bool: ...
    @abstractmethod
    async def list_files(self, directory: str = "") -> list[str]: ...
    @abstractmethod
    async def file_exists(self, path: str) -> bool: ...
    @abstractmethod
    async def search_files(
        self,
        directory: str,
        regex_pattern: str,
        file_pattern: str | None = None,
    ) -> list[FileSearchResult]: ...
    @abstractmethod
    async def create_directory(self, path: str) -> None: ...
```

All paths are relative to an implementation-defined root and must not escape it
(`..` traversal is rejected). The `file_pattern` in `search_files` is a glob
applied against file names (e.g. `"*.md"`), while `regex_pattern` is applied
case-insensitively against file contents.

### `FileSystemAgentFileStore` (disk-backed)

```python
from agent_framework import FileSystemAgentFileStore
import asyncio, warnings

warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

async def demo_disk_store():
    store = FileSystemAgentFileStore(root_directory="/tmp/agent_workspace")

    # Write and read
    await store.write_file("notes/plan.md", "# Plan\n\n- Step 1\n- Step 2")
    content = await store.read_file("notes/plan.md")
    print(content)  # # Plan\n\n- Step 1\n...

    # Atomic exclusive create (safe for race-free initialisation)
    try:
        await store.write_file("notes/plan.md", "other content", overwrite=False)
    except FileExistsError:
        print("File already exists — overwrite=False protected it")

    # List direct children of a directory
    files = await store.list_files("notes")
    print(files)  # ['plan.md']

    # Regex search — returns FileSearchResult objects
    results = await store.search_files("notes", r"Step \d+", "*.md")
    for r in results:
        print(r.file_name, r.snippet)
        for match in r.matching_lines:
            print(f"  line {match.line_number}: {match.line}")

    # Delete
    deleted = await store.delete_file("notes/plan.md")
    print("deleted:", deleted)  # True

asyncio.run(demo_disk_store())
```

**Security notes from source:**
- Symbolic links and reparse points anywhere along the resolved path are rejected.
- `O_NOFOLLOW` is passed on POSIX to block a late symlink swap on the leaf segment.
- Path escaping via `..` segments or absolute paths raises `ValueError` before the
  filesystem is touched.

### `InMemoryAgentFileStore` (in-memory, test-friendly)

```python
from agent_framework import InMemoryAgentFileStore
import asyncio, warnings

warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

async def demo_memory_store():
    store = InMemoryAgentFileStore()

    await store.write_file("reports/q1.txt", "Revenue: 120k")
    await store.write_file("reports/q2.txt", "Revenue: 145k")
    await store.write_file("notes.txt", "General notes")

    # List root
    root_files = await store.list_files()
    print(root_files)  # ['notes.txt']

    # List subdirectory
    reports = await store.list_files("reports")
    print(reports)  # ['q1.txt', 'q2.txt']

    # Search with glob filter
    results = await store.search_files("reports", r"Revenue:\s*\d+k", "q*.txt")
    for r in results:
        print(r.file_name)  # q1.txt, q2.txt

    # Concurrency — lock protects exclusive writes
    async def writer(n: int):
        await store.write_file(f"file_{n}.txt", f"content {n}", overwrite=False)

    await asyncio.gather(*(writer(i) for i in range(5)))
    print(await store.list_files())  # ['file_0.txt', ..., 'file_4.txt', ...]

asyncio.run(demo_memory_store())
```

### Custom backend example (Azure Blob Storage skeleton)

```python
from agent_framework import AgentFileStore, FileSearchResult
import re

class AzureBlobAgentFileStore(AgentFileStore):
    def __init__(self, container_client):
        self._container = container_client

    async def write_file(self, path: str, content: str, *, overwrite: bool = True) -> None:
        blob = self._container.get_blob_client(path)
        if not overwrite and await blob.exists():
            raise FileExistsError(f"Blob already exists: {path}")
        await blob.upload_blob(content.encode(), overwrite=overwrite)

    async def read_file(self, path: str) -> str | None:
        blob = self._container.get_blob_client(path)
        if not await blob.exists():
            return None
        stream = await blob.download_blob()
        return (await stream.readall()).decode()

    async def delete_file(self, path: str) -> bool:
        blob = self._container.get_blob_client(path)
        if not await blob.exists():
            return False
        await blob.delete_blob()
        return True

    async def list_files(self, directory: str = "") -> list[str]:
        prefix = f"{directory}/" if directory else ""
        names = []
        async for blob in self._container.list_blobs(name_starts_with=prefix):
            relative = blob.name[len(prefix):]
            if "/" not in relative:
                names.append(relative)
        return names

    async def file_exists(self, path: str) -> bool:
        return await self._container.get_blob_client(path).exists()

    async def search_files(
        self, directory: str, regex_pattern: str, file_pattern: str | None = None
    ) -> list[FileSearchResult]:
        # Enumerate and scan each blob in the directory
        results: list[FileSearchResult] = []
        pattern = re.compile(regex_pattern, re.IGNORECASE)
        for file_name in await self.list_files(directory):
            path = f"{directory}/{file_name}" if directory else file_name
            content = await self.read_file(path)
            if content and pattern.search(content):
                results.append(FileSearchResult(file_name=file_name, snippet=content[:200]))
        return results

    async def create_directory(self, path: str) -> None:
        pass  # Azure Blob Storage has no directory concept
```

---

## 2. `FileAccessProvider` + `FileSearchResult` + `FileSearchMatch`

**Source:** `agent_framework._harness._file_access`  
**Status:** `@experimental(feature_id=ExperimentalFeature.HARNESS)`

`FileAccessProvider` is a `ContextProvider` that injects **five tools** into the
agent's context before each model call, giving the agent read/write/delete/list/search
access to a shared `AgentFileStore`. It is the recommended way to give agents persistent
file workspace in 1.8.0.

### Constructor

```python
class FileAccessProvider(ContextProvider):
    def __init__(
        self,
        store: AgentFileStore,
        *,
        source_id: str = DEFAULT_FILE_ACCESS_SOURCE_ID,  # "file_access"
        instructions: str | None = None,   # None = use built-in default
        require_delete_approval: bool = True,  # human must approve deletes
    ) -> None: ...
```

### Five injected tools

| Tool name | `approval_mode` | Description |
|-----------|-----------------|-------------|
| `file_access_save_file(file_name, content, overwrite=False)` | `"never_require"` | Save a file; refuses overwrite by default |
| `file_access_read_file(file_name)` | `"never_require"` | Return file content or not-found message |
| `file_access_delete_file(file_name)` | `"always_require"` *(or `"never_require"`)* | Delete with optional human gate |
| `file_access_list_files(directory?)` | `"never_require"` | List direct children of a directory |
| `file_access_search_files(regex_pattern, file_pattern?, directory?)` | `"never_require"` | Grep-style regex search with glob filter |

### Basic usage

```python
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

from agent_framework import Agent, FileAccessProvider, FileSystemAgentFileStore
from agent_framework.openai import OpenAIChatClient

async def main():
    store = FileSystemAgentFileStore(root_directory="/tmp/agent_workspace")
    file_provider = FileAccessProvider(
        store,
        require_delete_approval=True,  # default — human must confirm deletes
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a file-management assistant. Use tools to read and write files.",
        context_providers=[file_provider],
    )

    response = await agent.run(
        "Create a file called 'meeting_notes.txt' with a summary of our Q3 goals."
    )
    print(response.text)
    # Verify the agent wrote the file
    content = await store.read_file("meeting_notes.txt")
    print("Written by agent:", content[:80])

asyncio.run(main())
```

### Autonomous (no approval) setup for CI/test environments

```python
from agent_framework import FileAccessProvider, InMemoryAgentFileStore

store = InMemoryAgentFileStore()
file_provider = FileAccessProvider(
    store,
    require_delete_approval=False,   # allow autonomous deletes
    instructions="You have full CRUD access to the workspace. Keep files organised.",
)
```

### `FileSearchResult` and `FileSearchMatch` data model

`AgentFileStore.search_files` returns `list[FileSearchResult]`. Each result has:

```python
from agent_framework import FileSearchResult, FileSearchMatch

# Construct a result manually (useful for tests)
result = FileSearchResult(
    file_name="report.md",
    snippet="Revenue: 145k on line 3 of report.md",
    matching_lines=[
        FileSearchMatch(line_number=3, line="Revenue: 145k"),
        FileSearchMatch(line_number=7, line="Revenue (YTD): 400k"),
    ],
)
print(result.to_dict())
# {
#   "file_name": "report.md",
#   "snippet": "Revenue: 145k on line 3 of report.md",
#   "matching_lines": [{"line_number": 3, "line": "Revenue: 145k"}, ...]
# }
```

Both types implement `SerializationMixin`: `to_dict()`, `to_json()`, and
`from_dict()` are available, making them easy to persist or pass through evaluation
pipelines.

### Multi-session shared workspace

```python
from agent_framework import FileAccessProvider, FileSystemAgentFileStore, Agent
from agent_framework.openai import OpenAIChatClient
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

# Same store — all agents share the same root
shared_store = FileSystemAgentFileStore("/tmp/shared_workspace")

async def make_agent(role: str) -> Agent:
    return Agent(
        client=OpenAIChatClient(),
        instructions=f"You are a {role}. Read and write files in the shared workspace.",
        context_providers=[FileAccessProvider(shared_store, require_delete_approval=False)],
    )

async def main():
    writer = await make_agent("document writer")
    reviewer = await make_agent("document reviewer")

    await writer.run("Write a draft proposal to 'proposal_draft.txt'.")
    review = await reviewer.run("Review the file 'proposal_draft.txt' and note any issues.")
    print(review.text)

asyncio.run(main())
```

---

## 3. `MCPSkill` + `MCPSkillResource` + `MCPSkillsSource`

**Source:** `agent_framework._skills`  
**Status:** `@experimental(feature_id=ExperimentalFeature.MCP_SKILLS)`

These three classes implement the **SEP-2640 Agent Skills over MCP** convention.
An MCP server that serves a `skill://index.json` resource is automatically
discovered as a set of `MCPSkill` instances by `MCPSkillsSource`. Skill content
(`SKILL.md`) is fetched lazily on demand; sibling resources (e.g. code examples,
checklists) are fetched via `MCPSkill.get_resource`.

### `MCPSkillsSource` — discovery

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent_framework import MCPSkillsSource, SkillsProvider, Agent
from agent_framework.openai import OpenAIChatClient
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

async def demo_mcp_skills():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_skills_server"],  # your SEP-2640 server
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover all skills from the MCP server
            source = MCPSkillsSource(session)
            skills = await source.get_skills()
            for skill in skills:
                print(skill.frontmatter.name, "—", skill.frontmatter.description)

asyncio.run(demo_mcp_skills())
```

### `MCPSkill` — lazy content fetch + sibling resources

```python
async def inspect_skill(skill: MCPSkill):
    # Fetch SKILL.md content on demand (cached after first call)
    content = await skill.get_content()
    print("SKILL.md:", content[:200])

    # Fetch a sibling resource (e.g. references/checklist.md)
    resource = await skill.get_resource("references/checklist.md")
    if resource is not None:
        data = await resource.read()  # returns str or bytes
        print("Checklist:", data)
```

### Wiring `MCPSkillsSource` into an agent via `SkillsProvider`

```python
from agent_framework import MCPSkillsSource, SkillsProvider, Agent
from agent_framework.openai import OpenAIChatClient
from mcp import ClientSession

async def build_agent_with_mcp_skills(session: ClientSession) -> Agent:
    source = MCPSkillsSource(session)
    provider = SkillsProvider(source)  # standard SkillsProvider wraps any SkillsSource

    return Agent(
        client=OpenAIChatClient(),
        instructions="You are a domain expert. Use skills to answer questions.",
        context_providers=[provider],
    )
```

### `MCPSkill` security validation

`MCPSkill.get_resource` rejects any resource name containing:
- Absolute paths (starts with `/`)
- URI schemes (`://`)
- Parent-traversal segments (`..`)

```python
# All of these return None silently (logged at DEBUG level)
await skill.get_resource("../../../etc/passwd")   # traversal
await skill.get_resource("/absolute/path")         # absolute
await skill.get_resource("http://evil.com/data")  # external URI
```

### `MCPSkillResource` — binary vs text

```python
from agent_framework import MCPSkillResource

async def read_resource(resource: MCPSkillResource):
    data = await resource.read()
    if isinstance(data, bytes):
        # Binary content (e.g. an image); base64-decoded from BlobResourceContents
        print("Binary:", len(data), "bytes")
    elif isinstance(data, str):
        print("Text:", data[:100])
    else:
        print("Empty resource")
```

---

## 4. `ToolMode` + `validate_tool_mode`

**Source:** `agent_framework._types`

`ToolMode` is a `TypedDict` that gives callers fine-grained control over which
model tools are available and how the model must use them on each chat turn.
Pass it as `tool_choice` in `ChatOptions` or via `agent.run(..., options=...)`.

### Class signature

```python
class ToolMode(TypedDict, total=False):
    mode: Literal["auto", "required", "none"]
    required_function_name: str       # only valid when mode == "required"
    allowed_tools: list[str]          # valid when mode == "auto" | "required"
```

| `mode` | Behaviour |
|--------|-----------|
| `"auto"` | Model may call tools or reply in text (default) |
| `"required"` | Model *must* call at least one tool |
| `"none"` | Model may not call any tools this turn |

### Usage patterns

```python
from agent_framework import Agent, ChatOptions, ToolMode, FunctionTool, tool
from agent_framework.openai import OpenAIChatClient

@tool
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

@tool
def get_news(topic: str) -> str:
    return f"Latest news on {topic}"

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    default_options={"tools": [get_weather, get_news]},
)

# Force the model to call get_weather on this turn
forced_response = await agent.run(
    "What is the weather in London?",
    options={"tool_choice": {"mode": "required", "required_function_name": "get_weather"}},
)

# Restrict available tools without removing them from default_options
auto_filtered = await agent.run(
    "Tell me about the weather.",
    options={"tool_choice": {"mode": "auto", "allowed_tools": ["get_weather"]}},
)

# Disable all tools for a single turn (e.g. for a pure-text answer)
text_only = await agent.run(
    "Summarise our conversation so far.",
    options={"tool_choice": {"mode": "none"}},
)
```

### `validate_tool_mode` — normalization helper

`validate_tool_mode` converts a bare string `"auto"` / `"required"` / `"none"` into
the canonical `ToolMode` dict, validates constraints, and raises `ContentError` on
invalid input.

```python
from agent_framework import validate_tool_mode

# String shorthand → ToolMode dict
mode = validate_tool_mode("required")
print(mode)  # {"mode": "required"}

# Full dict preserved
mode = validate_tool_mode({"mode": "auto", "allowed_tools": ["get_weather"]})
print(mode)  # {"mode": "auto", "allowed_tools": ["get_weather"]}

# Invalid combos raise ContentError
from agent_framework.exceptions import ContentError
try:
    validate_tool_mode({"mode": "none", "required_function_name": "get_weather"})
except ContentError as e:
    print(e)  # tool_choice with mode other than 'required' cannot have...

# None passthrough
print(validate_tool_mode(None))  # None
```

### Dynamic tool gating inside a workflow

```python
from agent_framework import WorkflowBuilder, Executor, WorkflowContext, handler
from agent_framework import Agent, ChatOptions, ToolMode

class PlanningExecutor(Executor):
    @handler
    async def plan(self, message: str, ctx: WorkflowContext[str]) -> None:
        # Force the orchestrator to produce a plan before calling any tools
        response = await self.agent.run(
            message,
            options={"tool_choice": {"mode": "none"}},
        )
        await ctx.send_message(response.text)
```

---

## 5. `AgentEvalConverter` + `ExpectedToolCall` + `CheckResult` + `RubricScore`

**Source:** `agent_framework._evaluation`  
**Status:** `@experimental(feature_id=ExperimentalFeature.EVALS)`

1.8.0 adds a cluster of evaluation helpers that bridge agent-framework's internal
types (`Message`, `FunctionTool`, `AgentResponse`) with the Foundry evaluator
schema and the generic `EvalItem` format.

### `AgentEvalConverter` — message and tool conversion

```python
from agent_framework import AgentEvalConverter, Message, Content, AgentResponse

# Convert agent messages to Foundry evaluator format
messages = [
    Message("user", [Content(type="text", text="What is 2+2?")]),
    Message("assistant", [Content(type="text", text="The answer is 4.")]),
]
foundry_messages = AgentEvalConverter.convert_messages(messages)
# [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}, ...]

# Function calls are converted to tool_call entries
tool_call_msg = Message("assistant", [
    Content(type="function_call", name="get_weather", call_id="call_123", arguments='{"city": "London"}')
])
converted = AgentEvalConverter.convert_message(tool_call_msg)
# [{"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "call_123", ...}]}]

# Function results → tool role messages (one per result)
tool_result_msg = Message("tool", [
    Content(type="function_result", call_id="call_123", result='{"temp": 18}')
])
converted_result = AgentEvalConverter.convert_message(tool_result_msg)
# [{"role": "tool", "tool_call_id": "call_123", "content": [...]}]
```

### `AgentEvalConverter.to_eval_item` — full interaction → `EvalItem`

```python
from agent_framework import AgentEvalConverter, Agent, EvalItem
from agent_framework.openai import OpenAIChatClient

@tool
def lookup_price(item: str) -> float:
    return 9.99

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a pricing assistant.",
    default_options={"tools": [lookup_price]},
)

response = await agent.run("How much does the widget cost?")

eval_item = AgentEvalConverter.to_eval_item(
    query="How much does the widget cost?",
    response=response,
    agent=agent,           # auto-extracts tool definitions
    context="Our catalogue lists widget at $9.99",
)
# eval_item is a standard EvalItem ready for LocalEvaluator or any Evaluator
```

### `ExpectedToolCall` — assert tool calls in automated evals

```python
from agent_framework import ExpectedToolCall, EvalItem
from agent_framework import LocalEvaluator, evaluator, EvalItemResult

expected = ExpectedToolCall(name="get_weather", arguments={"city": "London"})

@evaluator
async def tool_call_checker(item: EvalItem) -> EvalItemResult:
    messages = item.conversation or []
    for msg in messages:
        for c in (msg.contents or []):
            if c.type == "function_call" and c.name == expected.name:
                if expected.arguments is None:
                    return EvalItemResult(pass_=True, score=1.0, reason="Tool called")
                if c.arguments == expected.arguments:
                    return EvalItemResult(pass_=True, score=1.0, reason="Tool called with correct args")
    return EvalItemResult(pass_=False, score=0.0, reason=f"Tool '{expected.name}' not called")
```

### `CheckResult` — structured pass/fail from evaluator checks

```python
from agent_framework import CheckResult

result = CheckResult(
    passed=True,
    reason="The agent correctly identified the capital of France as Paris.",
    check_name="capital_city_check",
)
print(result.passed, result.check_name)  # True, "capital_city_check"
```

### `RubricScore` — per-dimension scores from rubric evaluators

```python
from agent_framework import RubricScore

score = RubricScore(
    id="coherence",
    score=4.0,
    max_score=5.0,
    reason="The response was mostly coherent but missed one connection.",
    label="Good",
)
print(f"{score.id}: {score.score}/{score.max_score} — {score.label}")

# Normalised value
normalized = score.score / score.max_score if score.max_score else 0.0
print(f"Normalised: {normalized:.2%}")  # 80.00%
```

### End-to-end eval pipeline using all four types

```python
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

from agent_framework import (
    Agent, AgentEvalConverter, ExpectedToolCall, LocalEvaluator,
    EvalItem, EvalItemResult, evaluator, tool
)
from agent_framework.openai import OpenAIChatClient

@tool
def get_temperature(city: str) -> str:
    return f"18°C in {city}"

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a weather assistant.",
    default_options={"tools": [get_temperature]},
)

@evaluator
async def must_call_get_temperature(item: EvalItem) -> EvalItemResult:
    for msg in (item.conversation or []):
        for c in (msg.contents or []):
            if c.type == "function_call" and c.name == "get_temperature":
                return EvalItemResult(pass_=True, score=1.0, reason="Tool called correctly")
    return EvalItemResult(pass_=False, score=0.0, reason="get_temperature not called")

async def run_eval():
    response = await agent.run("What is the temperature in Berlin?")
    item = AgentEvalConverter.to_eval_item(
        query="What is the temperature in Berlin?",
        response=response,
        agent=agent,
    )
    evaluators = [must_call_get_temperature]
    local = LocalEvaluator(evaluators=evaluators)
    results = await local.evaluate([item])
    print("Passed:", results.pass_rate)  # 1.0 if tool was called

asyncio.run(run_eval())
```

---

## 6. `ChatContext`

**Source:** `agent_framework._middleware`

`ChatContext` is the context object that flows through the **chat middleware pipeline**
(composed of `ChatMiddleware` instances). Added as a formally typed class in 1.8.0, it
replaces the implicit context dict pattern and exposes all chat-layer hooks.

### Constructor & attributes

```python
class ChatContext:
    client: SupportsChatGetResponse
    messages: Sequence[Message]
    options: Mapping[str, Any] | None
    stream: bool
    metadata: dict[str, Any]            # share data between middleware
    result: ChatResponse | ResponseStream | None   # set by or after call_next()
    kwargs: dict[str, Any]              # forwarded to the chat client
    function_invocation_kwargs: dict[str, Any]    # forwarded to tool invocations
    stream_transform_hooks: list[...]   # per-update transform callables
    stream_result_hooks: list[...]      # post-finalize hooks
    stream_cleanup_hooks: list[...]     # after stream consumed
```

### Introspecting a request

```python
from agent_framework import ChatMiddleware, ChatContext

class RequestLoggerMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next):
        print(f"Client: {context.client.__class__.__name__}")
        print(f"Messages: {len(context.messages)}, Streaming: {context.stream}")
        print(f"Model: {context.options.get('model') if context.options else 'default'}")
        await call_next()
        # After call_next(), context.result is populated
        if not context.stream and context.result:
            print(f"Usage: {context.result.usage}")
```

### Short-circuiting — cache-hit override

```python
import hashlib, json
from agent_framework import ChatMiddleware, ChatContext, ChatResponse, Content, Message

class SimpleCacheMiddleware(ChatMiddleware):
    def __init__(self):
        self._cache: dict[str, ChatResponse] = {}

    def _cache_key(self, context: ChatContext) -> str:
        payload = json.dumps([m.model_dump() for m in context.messages], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def process(self, context: ChatContext, call_next):
        key = self._cache_key(context)
        if key in self._cache:
            # Override result — downstream and the model are skipped entirely
            context.result = self._cache[key]
            return  # do NOT call call_next()
        await call_next()
        if context.result and not context.stream:
            self._cache[key] = context.result
```

### Streaming transform hooks

`stream_transform_hooks` receive each `ChatResponseUpdate` in order and can
modify (or filter) individual chunks before they reach the caller:

```python
from agent_framework import ChatMiddleware, ChatContext, ChatResponseUpdate

class UpperCaseTransformMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next):
        async def uppercase_transform(update: ChatResponseUpdate) -> ChatResponseUpdate:
            if update.content:
                update = update.model_copy(update={"content": update.content.upper()})
            return update

        context.stream_transform_hooks.append(uppercase_transform)
        await call_next()
```

### Metadata sharing between middleware layers

```python
from agent_framework import ChatMiddleware, ChatContext
import time

class TimingMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next):
        context.metadata["start_time"] = time.monotonic()
        await call_next()
        elapsed = time.monotonic() - context.metadata["start_time"]
        context.metadata["elapsed_ms"] = round(elapsed * 1000, 2)

class MetricsMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next):
        await call_next()
        elapsed = context.metadata.get("elapsed_ms", "?")
        print(f"[Metrics] Chat completed in {elapsed} ms")

# Wire both into the client
from agent_framework.openai import OpenAIChatClient
client = OpenAIChatClient(middleware=[TimingMiddleware(), MetricsMiddleware()])
```

---

## 7. `WorkflowAgent` + `WorkflowContext`

**Source:** `agent_framework._agents` / `agent_framework._workflows._workflow_context`

`WorkflowAgent` wraps any `Workflow` object as an `Agent`, making it usable
anywhere an `Agent` is expected — including as a participant in GroupChat,
Handoff, or Sequential orchestrations.

`WorkflowContext` is the typed context injected into every executor's `@handler`
method, enabling type-safe `send_message`, `yield_output`, state management, and
HITL requests.

### `WorkflowAgent` constructor

```python
class WorkflowAgent(BaseAgent):
    def __init__(
        self,
        workflow: Workflow,
        *,
        id: str | None = None,           # auto-generated if None
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        **kwargs,
    ) -> None: ...
```

**Constraints from source:**
- The workflow's start executor must accept `list[Message]` as input.
- Only `output` and `request_info` workflow events surface as agent responses.
- Use `output_from` in `WorkflowBuilder` to control which executor outputs are exposed.

### Basic WorkflowAgent usage

```python
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

from agent_framework import (
    WorkflowBuilder, Executor, WorkflowContext, WorkflowAgent, handler, tool, Message
)
from agent_framework.openai import OpenAIChatClient

@tool
def summarise(text: str) -> str:
    return f"Summary: {text[:50]}..."

class ProcessingExecutor(Executor):
    def __init__(self, client):
        from agent_framework import Agent
        self.agent = Agent(client=client, instructions="Summarise the input.", default_options={"tools": [summarise]})

    @handler
    async def process(self, messages: list[Message], ctx: WorkflowContext[str]) -> None:
        response = await self.agent.run(messages)
        await ctx.send_message(response.text)

async def main():
    client = OpenAIChatClient()
    executor = ProcessingExecutor(client)

    workflow = (
        WorkflowBuilder()
        .add_executor(executor)
        .set_start_executor(executor)
        .output_from(executor)
        .build()
    )

    workflow_agent = WorkflowAgent(
        workflow,
        name="SummarisationAgent",
        description="Summarises text input via a workflow.",
    )

    response = await workflow_agent.run("Please summarise this long document...")
    print(response.text)

asyncio.run(main())
```

### `WorkflowContext` — typed message passing

`WorkflowContext` is generic over two type parameters:
- `OutT` — the type of messages sent via `ctx.send_message()`
- `W_OutT` — the type of values yielded via `ctx.yield_output()`

```python
from agent_framework import Executor, WorkflowContext, handler, Message

# Single-output executor: sends str messages
class StringSender(Executor):
    @handler
    async def run(self, msg: str, ctx: WorkflowContext[str]) -> None:
        result = msg.upper()
        await ctx.send_message(result)

# Dual-output executor: sends ints and yields str workflow outputs
class DualOutputExecutor(Executor):
    @handler
    async def run(self, value: int, ctx: WorkflowContext[int, str]) -> None:
        await ctx.send_message(value * 2)           # downstream message
        await ctx.yield_output(f"processed {value}") # exposed at workflow level

# Union types for flexibility
class FlexibleExecutor(Executor):
    @handler
    async def run(self, data: str | dict, ctx: WorkflowContext[str | int, bool]) -> None:
        if isinstance(data, str):
            await ctx.send_message(len(data))
        else:
            await ctx.send_message(str(data))
        await ctx.yield_output(True)
```

### WorkflowAgent HITL (pending_requests)

When a workflow raises a HITL event (via `request_info`), the agent surfaces it
through `pending_requests`. Callers resume by passing the `request_id` back:

```python
async def hitl_loop(agent: WorkflowAgent, initial_message: str):
    response = await agent.run(initial_message)

    while agent.pending_requests:
        request_id, event = next(iter(agent.pending_requests.items()))
        print(f"Agent needs input: {event.data}")
        user_input = input("Your answer: ")
        # Resume by passing the request_id and user response
        response = await agent.run(
            user_input,
            options={"request_id": request_id},
        )

    print("Final response:", response.text)
```

### Embedding WorkflowAgent into a GroupChat

```python
from agent_framework import GroupChatBuilder, Agent
from agent_framework.orchestrations import GroupChatBuilder

research_workflow_agent = WorkflowAgent(research_workflow, name="Researcher")
writer_agent = Agent(client=OpenAIChatClient(), instructions="You are a writer.")

group = (
    GroupChatBuilder()
    .add_agents([research_workflow_agent, writer_agent])
    .build()
)
result = await group.run("Write a report on climate change.")
```

---

## 8. `TruncationStrategy`

**Source:** `agent_framework._compaction`

`TruncationStrategy` is a **new compaction strategy** added in 1.8.0. Unlike
`SlidingWindowStrategy` (which always keeps the most recent N messages) or
`SummarizationStrategy` (which condenses old turns), `TruncationStrategy` performs
**oldest-first hard removal** of whole message groups once a token or message-count
threshold is crossed.

### Constructor

```python
class TruncationStrategy:
    def __init__(
        self,
        *,
        max_n: int,                          # trigger threshold
        compact_to: int,                     # target after compaction
        tokenizer: TokenizerProtocol | None = None,  # token-based; else message count
        preserve_system: bool = True,        # protect system messages from removal
    ) -> None: ...
```

| Parameter | Purpose |
|-----------|---------|
| `max_n` | Trigger threshold — tokens (with tokenizer) or included message count (without) |
| `compact_to` | Target after compaction — must be ≤ `max_n` |
| `tokenizer` | Provide a `TokenizerProtocol` for token-based truncation |
| `preserve_system` | Keep system-role groups even when oldest-first removal is running |

**Invariants enforced at construction:**
- `max_n > 0`, `compact_to > 0`
- `compact_to ≤ max_n`

### Message-count based truncation

```python
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

from agent_framework import (
    Agent, CompactionProvider, TruncationStrategy
)
from agent_framework.openai import OpenAIChatClient

async def main():
    strategy = TruncationStrategy(
        max_n=20,      # compact when > 20 messages in context
        compact_to=10, # after compaction keep ≤ 10 messages
        preserve_system=True,
    )
    compaction = CompactionProvider(strategy=strategy)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a conversational assistant.",
        context_providers=[compaction],
    )

    # Simulate a long conversation — older turns are dropped automatically
    for i in range(30):
        response = await agent.run(f"Turn {i}: tell me something interesting.")
    print(response.text)

asyncio.run(main())
```

### Token-based truncation with tiktoken

```python
import tiktoken
from agent_framework import TruncationStrategy, TokenizerProtocol

class TiktokenTokenizer(TokenizerProtocol):
    def __init__(self, model: str = "gpt-4o"):
        self._enc = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

strategy = TruncationStrategy(
    max_n=4000,       # trigger at 4 000 tokens
    compact_to=2000,  # trim to 2 000 tokens
    tokenizer=TiktokenTokenizer(),
    preserve_system=True,
)
```

### Comparison table — compaction strategies

| Strategy | When to use | Side effects |
|----------|-------------|--------------|
| `TruncationStrategy` | Hard budget; no API calls; simple oldest-first drop | Loses old context permanently |
| `SlidingWindowStrategy` | Keep last N messages by count | Loses old context permanently |
| `SummarizationStrategy` | Preserve semantics; can call model | Extra latency + cost; needs a summary client |
| `ContextWindowCompactionStrategy` | Near-limit trigger; full context awareness | May call model; most complex |
| `TokenBudgetComposedStrategy` | Chained multi-strategy pipeline | Configurable; most flexible |

### Combining `TruncationStrategy` with summarisation fallback

```python
from agent_framework import (
    CompactionProvider, TokenBudgetComposedStrategy,
    TruncationStrategy, SummarizationStrategy
)
from agent_framework.openai import OpenAIChatClient

summary_client = OpenAIChatClient()

# Summarise first; only truncate if summary still too long
strategy = TokenBudgetComposedStrategy(
    strategies=[
        SummarizationStrategy(client=summary_client, max_n=8000, compact_to=4000),
        TruncationStrategy(max_n=4000, compact_to=2000),  # hard fallback
    ]
)
compaction = CompactionProvider(strategy=strategy)
```

---

## 9. `HistoryProvider` + `InMemoryHistoryProvider`

**Source:** `agent_framework._sessions`

1.8.0 introduces `HistoryProvider` as a formal ABC for conversation history storage,
with `InMemoryHistoryProvider` as the default built-in implementation. `FileHistoryProvider`
(Vol. 2) extends `HistoryProvider`; this volume documents the base class directly.

### `HistoryProvider` — abstract base class

```python
class HistoryProvider(ContextProvider):
    def __init__(
        self,
        source_id: str,
        *,
        load_messages: bool = True,
        store_inputs: bool = True,
        store_context_messages: bool = False,
        store_context_from: set[str] | None = None,
        store_outputs: bool = True,
    ): ...

    @abstractmethod
    async def get_messages(
        self, session_id: str | None, *, state: dict[str, Any] | None = None, **kwargs
    ) -> list[Message]: ...

    @abstractmethod
    async def save_messages(
        self, session_id: str | None, messages: Sequence[Message],
        *, state: dict[str, Any] | None = None, **kwargs
    ) -> None: ...
```

### Configuration flags

| Flag | Default | Effect |
|------|---------|--------|
| `load_messages` | `True` | Load history before each run; `False` = write-only (audit log pattern) |
| `store_inputs` | `True` | Persist the user's input messages |
| `store_context_messages` | `False` | Also persist context injected by other providers |
| `store_context_from` | `None` | If set, only store context from these `source_id`s |
| `store_outputs` | `True` | Persist the agent's response messages |

### Custom `HistoryProvider` — SQL backend

```python
import asyncio
from agent_framework import HistoryProvider, Message, Agent
from agent_framework.openai import OpenAIChatClient
import json

class SQLiteHistoryProvider(HistoryProvider):
    def __init__(self, db_path: str, *, source_id: str = "sqlite_history"):
        super().__init__(source_id)
        self._db_path = db_path
        self._conn = None

    async def _get_conn(self):
        if self._conn is None:
            import aiosqlite
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute(
                "CREATE TABLE IF NOT EXISTS history (session_id TEXT, messages TEXT)"
            )
            await self._conn.commit()
        return self._conn

    async def get_messages(self, session_id: str | None, *, state=None, **kwargs) -> list[Message]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT messages FROM history WHERE session_id = ? ORDER BY rowid",
            (session_id or "default",),
        )
        rows = await cursor.fetchall()
        messages = []
        for (raw,) in rows:
            messages.extend(Message.from_dict(m) for m in json.loads(raw))
        return messages

    async def save_messages(self, session_id: str | None, messages, *, state=None, **kwargs):
        conn = await self._get_conn()
        await conn.execute(
            "INSERT INTO history (session_id, messages) VALUES (?, ?)",
            (session_id or "default", json.dumps([m.model_dump() for m in messages])),
        )
        await conn.commit()
```

### `InMemoryHistoryProvider` — state-backed default

`InMemoryHistoryProvider` is the provider the framework auto-injects when no
`context_providers` are configured. It stores all messages in `session.state["messages"]`.

```python
from agent_framework import InMemoryHistoryProvider, Agent, AgentSession
from agent_framework.openai import OpenAIChatClient
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

async def main():
    # Explicitly wired — same as the framework default when no providers given
    history = InMemoryHistoryProvider(
        skip_excluded=True,   # omit compacted messages from loaded history
        store_context_messages=False,
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[history],
    )

    session = AgentSession()
    r1 = await agent.run("My name is Alice.", session=session)
    r2 = await agent.run("What is my name?", session=session)
    print(r2.text)   # "Your name is Alice."

asyncio.run(main())
```

### Audit-log provider (write-only)

```python
from agent_framework import HistoryProvider, Message
import json, datetime

class AuditLogProvider(HistoryProvider):
    def __init__(self, log_path: str):
        super().__init__(
            source_id="audit",
            load_messages=False,   # never inject into context
            store_outputs=True,
            store_inputs=True,
        )
        self._path = log_path

    async def get_messages(self, session_id, *, state=None, **kwargs) -> list[Message]:
        return []  # write-only

    async def save_messages(self, session_id, messages, *, state=None, **kwargs):
        with open(self._path, "a") as f:
            for msg in messages:
                f.write(json.dumps({
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "session": session_id,
                    "role": msg.role,
                    "text": msg.text,
                }) + "\n")
```

---

## 10. `DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`

**Source:** `agent_framework._skills`, `agent_framework._middleware`

This group covers three new building blocks: two skills-layer primitives and the
richly documented `FunctionInvocationContext` that enables **progressive tool exposure**.

### `InMemorySkillsSource` — pre-built skills in memory

The simplest `SkillsSource`: holds `Skill` instances that were constructed at
startup time rather than discovered at runtime.

```python
from agent_framework import InMemorySkillsSource, InlineSkill, SkillsProvider, Agent
from agent_framework.openai import OpenAIChatClient
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

async def main():
    python_skill = InlineSkill(
        name="python-style-guide",
        description="Python coding conventions and best practices",
        instructions="""
        When writing Python code:
        - Follow PEP 8 style guidelines
        - Use type hints for function signatures
        - Prefer list comprehensions over map/filter
        - Use f-strings for string formatting
        """,
    )

    security_skill = InlineSkill(
        name="security-review",
        description="Code security review guidelines",
        instructions="Check for SQL injection, XSS, insecure deserialization, and OWASP Top 10.",
    )

    source = InMemorySkillsSource([python_skill, security_skill])
    # Verify what's in the source
    skills = await source.get_skills()
    print([s.frontmatter.name for s in skills])  # ['python-style-guide', 'security-review']

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a code review assistant.",
        context_providers=[SkillsProvider(source)],
    )
    response = await agent.run("Review this code: def f(x): return eval(x)")
    print(response.text)

asyncio.run(main())
```

### `DelegatingSkillsSource` — composable decorator base

`DelegatingSkillsSource` is an ABC that wraps another `SkillsSource`.
Subclass it to add caching, filtering, logging, or any cross-cutting concern
without modifying the underlying source.

```python
from agent_framework import DelegatingSkillsSource, SkillsSource, Skill
import time

class CachingSkillsSource(DelegatingSkillsSource):
    """Cache skills for ``ttl`` seconds to avoid repeated discovery."""
    def __init__(self, inner_source: SkillsSource, ttl: float = 60.0):
        super().__init__(inner_source)
        self._cache: list[Skill] | None = None
        self._expires_at = 0.0
        self._ttl = ttl

    async def get_skills(self) -> list[Skill]:
        if self._cache is not None and time.monotonic() < self._expires_at:
            return self._cache
        skills = await self.inner_source.get_skills()
        self._cache = skills
        self._expires_at = time.monotonic() + self._ttl
        return skills

class TopicFilterSource(DelegatingSkillsSource):
    """Only expose skills whose name contains one of ``topics``."""
    def __init__(self, inner_source: SkillsSource, topics: list[str]):
        super().__init__(inner_source)
        self._topics = topics

    async def get_skills(self) -> list[Skill]:
        all_skills = await self.inner_source.get_skills()
        return [
            s for s in all_skills
            if any(t in s.frontmatter.name for t in self._topics)
        ]

# Compose: underlying source → topic filter → caching layer
from agent_framework import FileSkillsSource, SkillsProvider

base = FileSkillsSource("/skills")
filtered = TopicFilterSource(base, topics=["python", "security"])
cached = CachingSkillsSource(filtered, ttl=120.0)
provider = SkillsProvider(cached)
```

### `FunctionInvocationContext` — progressive tool exposure

`FunctionInvocationContext` is the context passed to `FunctionMiddleware.process` and
can also be received directly by a `@tool` function via type annotation.
Its `add_tools` / `remove_tools` methods (both `@experimental(PROGRESSIVE_TOOLS)`)
allow a tool to dynamically change which tools the model sees on the *next* iteration.

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext, tool, Agent
from agent_framework.openai import OpenAIChatClient
import asyncio, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="agent_framework")

@tool
def step_1_analyse(query: str) -> str:
    return f"Analysis of: {query}"

@tool
def step_2_report(analysis: str) -> str:
    return f"Report based on: {analysis}"

@tool
def load_next_step(ctx: FunctionInvocationContext) -> str:
    """Unlock step_2 after step_1 has been called."""
    ctx.add_tools([step_2_report])
    ctx.remove_tools([load_next_step])  # remove self
    return "Step 2 tools are now available."

async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Complete the pipeline: analyse, then report.",
        default_options={"tools": [step_1_analyse, load_next_step]},
    )
    response = await agent.run("Analyse the user retention data and produce a report.")
    print(response.text)

asyncio.run(main())
```

### `FunctionInvocationContext` inside middleware

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext, MiddlewareTermination

class CostGuardMiddleware(FunctionMiddleware):
    def __init__(self, budget_cents: float):
        self._spent = 0.0
        self._budget = budget_cents

    async def process(self, context: FunctionInvocationContext, call_next):
        # Inspect tool name and arguments before execution
        tool_name = context.function.name
        args = context.arguments

        estimated_cost = self._estimate_cost(tool_name, args)
        if self._spent + estimated_cost > self._budget:
            raise MiddlewareTermination(
                f"Budget exceeded: {self._spent:.2f}c spent, {estimated_cost:.2f}c requested"
            )

        await call_next()

        # After execution observe the result
        self._spent += estimated_cost
        print(f"[CostGuard] {tool_name} cost {estimated_cost:.2f}c; total {self._spent:.2f}c")

    def _estimate_cost(self, name: str, args) -> float:
        # Your cost model here
        return 0.5

# Wire into agent
from agent_framework.openai import OpenAIChatClient
client = OpenAIChatClient(middleware=[CostGuardMiddleware(budget_cents=5.0)])
```

### Accessing metadata across middleware

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext
import time

class ToolTimingMiddleware(FunctionMiddleware):
    async def process(self, context: FunctionInvocationContext, call_next):
        start = time.monotonic()
        await call_next()
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        context.metadata["elapsed_ms"] = elapsed_ms
        print(f"[Timing] {context.function.name} took {elapsed_ms} ms; result={context.result!r:.100}")
```

---

## Upgrade notes — 1.7.0 → 1.8.0

| Change | Details |
|--------|---------|
| **New** `AgentFileStore` + `FileAccessProvider` | File workspace harness (@experimental HARNESS). No breaking change. |
| **New** `MCPSkill` + `MCPSkillsSource` | MCP skills discovery via SEP-2640 (@experimental MCP_SKILLS). No breaking change. |
| **New** `ToolMode` TypedDict | Replaces bare string `"auto"` / `"required"` / `"none"` in `tool_choice`; old strings still accepted via `validate_tool_mode`. No breaking change. |
| **New** `AgentEvalConverter` + eval helper types | Eval framework additions (@experimental EVALS). No breaking change. |
| **New** `ChatContext` | Formally typed chat middleware context. Existing `ChatMiddleware` subclasses continue to work unchanged. |
| **New** `WorkflowAgent` | Wraps `Workflow` as `Agent`. No impact on existing workflows. |
| **New** `TruncationStrategy` | Additional compaction strategy. Existing strategies unchanged. |
| **Refactored** `HistoryProvider` | `FileHistoryProvider` now extends the new `HistoryProvider` ABC. The `load_messages` / `store_inputs` / `store_outputs` flags are inherited — no API break. |
| **New** `InMemoryHistoryProvider` | Made the default auto-injected provider explicit and publicly importable. The implicit behaviour is unchanged. |
| **New** `DelegatingSkillsSource` + `InMemorySkillsSource` | New skills composition primitives. Existing `FileSkillsSource` and `AggregatingSkillsSource` unchanged. |
| **New** `FunctionInvocationContext.add_tools` / `remove_tools` | Progressive tool exposure (@experimental PROGRESSIVE_TOOLS). Existing `FunctionMiddleware` unaffected. |
| **`agent-framework-core`** bumped to **1.8.0** | 261 public symbols (up from 242 in 1.7.0). Run `pip install agent-framework==1.8.0` to upgrade. |

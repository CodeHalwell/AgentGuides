---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 18"
description: "Source-verified deep dives into 10 class groups in agent-framework 1.9.0: Skill+SkillFrontmatter+SkillScriptRunner (Agent Skills spec), InlineSkill (code-defined skills with resource/script decorators), InMemorySkillsSource+AggregatingSkillsSource+FilteringSkillsSource+DeduplicatingSkillsSource (skill source composition), AgentFileStore+InMemoryAgentFileStore+FileSearchMatch+FileSearchResult (agent file storage), FileAccessProvider (agent CRUD/search over shared stores), BackgroundAgentsProvider+BackgroundTaskInfo+BackgroundTaskStatus (background sub-agent delegation), MemoryStore+MemoryTopicRecord+MemoryIndexEntry (durable memory backing store), WorkflowGraphValidator+EdgeDuplicationError+GraphConnectivityError+TypeCompatibilityError (workflow graph validation), MagenticBuilder+MagenticManagerBase+MagenticProgressLedger (Magentic One orchestration internals), LocalEvaluator+EvalItem+EvalScoreResult+ConversationSplit (local evaluation framework)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 41
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 18

Verified against **agent-framework 1.9.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework._skills`, `agent_framework._harness`,
`agent_framework._workflows._validation`, `agent_framework_orchestrations._magentic`,
`agent_framework._evaluation`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) · [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) · [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) · [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) · [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) · [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) · [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) · [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) · [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) · [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) · [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) · [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/) · [Vol. 13](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v13/) · [Vol. 14](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v14/) · [Vol. 15](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15/) · [Vol. 16](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v16/) · [Vol. 17](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v17/)

---

## 1. `Skill` · `SkillFrontmatter` · `SkillScriptRunner`

**Module:** `agent_framework._skills`  
**Status:** `@experimental(ExperimentalFeature.SKILLS)`

The Skills system implements the [Agent Skills specification](https://agentskills.io/specification): a portable, provider-agnostic format for packaging domain-specific capabilities (instructions, resources, runnable scripts) so agents can discover and consume them without hard-coded knowledge.

### `SkillFrontmatter`

L1 discovery metadata attached to every `Skill`. Validated at construction; post-construction mutations are **not** re-validated.

| Field | Type | Constraint |
|---|---|---|
| `name` | `str` | Lowercase letters, numbers, hyphens; max 64 chars; no leading/trailing/consecutive hyphens |
| `description` | `str` | ≤ 1024 characters |
| `license` | `str \| None` | Optional license name or reference |
| `compatibility` | `str \| None` | ≤ 500 characters |
| `allowed_tools` | `str \| None` | Space-delimited pre-approved tool names |
| `metadata` | `dict[str, str] \| None` | Shallow-copied at construction to prevent aliasing |

### `Skill` (ABC)

Abstract base. Subclasses must implement `frontmatter` (property) and `get_content()` (async). Optionally override `get_resource(name)` and `get_script(name)`.

### `SkillScriptRunner` (Protocol)

Runtime-checkable protocol for executing **file-based** skill scripts. Any sync or async callable matching `(skill, script, args) -> Any` satisfies it. Code-defined scripts (registered via `@skill.script`) always run in-process and bypass this.

```python
import asyncio
from agent_framework import Agent
from agent_framework._skills import (
    Skill, SkillFrontmatter, SkillScriptRunner,
    InlineSkill, FileSkill, FileSkillScript, SkillsProvider,
    InMemorySkillsSource,
)
from agent_framework.openai import OpenAIChatClient

# --- Build a SkillFrontmatter ---
fm = SkillFrontmatter(
    name="data-analysis",
    description="Statistical data analysis capability for tabular data.",
    license="MIT",
    compatibility="Python >= 3.11; pandas >= 2.0",
    allowed_tools="execute_code search_web",
    metadata={"version": "1.2", "author": "data-team"},
)
print(fm.name)          # "data-analysis"
print(fm.allowed_tools) # "execute_code search_web"
print(fm.metadata)      # {"version": "1.2", "author": "data-team"}
```

```python
# --- Custom SkillScriptRunner: run scripts via subprocess ---
import subprocess, json
from agent_framework._skills import FileSkill, FileSkillScript

class SubprocessRunner:
    def __call__(
        self,
        skill: FileSkill,
        script: FileSkillScript,
        args: dict | list | None = None,
    ):
        cmd = ["python", str(script.path)]
        if isinstance(args, dict):
            cmd += ["--args", json.dumps(args)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout.strip()

runner = SubprocessRunner()
# Pass to SkillsProvider(script_runner=runner) below
```

```python
# --- Inspect a Skill's metadata at runtime ---
async def describe_skill(skill: Skill) -> None:
    fm = skill.frontmatter
    content = await skill.get_content()
    print(f"Name        : {fm.name}")
    print(f"Description : {fm.description}")
    print(f"License     : {fm.license or '—'}")
    print(f"Allowed     : {fm.allowed_tools or 'all tools'}")
    print(f"Content size: {len(content)} chars")
    # Inspect a named resource if present
    schema_res = await skill.get_resource("schema")
    if schema_res:
        print(f"Resource 'schema': {await schema_res.get_content()}")
```

---

## 2. `InlineSkill`

**Module:** `agent_framework._skills`  
**Status:** `@experimental(ExperimentalFeature.SKILLS)`

A `Skill` defined entirely in Python code. Resources and scripts are registered via `@skill.resource` / `@skill.script` decorators. The synthesized XML content is cached after the first `await skill.get_content()` call, so add all resources/scripts **before** the first access.

```python
from agent_framework._skills import InlineSkill, SkillFrontmatter, SkillsProvider, InMemorySkillsSource

# Build the skill object
sql_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="sql-expert",
        description="Helps agents write and optimise SQL queries.",
    ),
    instructions=(
        "You are an expert SQL assistant. "
        "Use the 'schema' resource to understand table structure before writing queries. "
        "Always prefer CTEs over nested sub-queries for readability."
    ),
)

# Attach a resource (sync function returning string)
@sql_skill.resource
def schema() -> str:
    return """
    users(id INT PK, email TEXT, created_at TIMESTAMPTZ)
    orders(id INT PK, user_id INT FK->users.id, total NUMERIC, status TEXT)
    order_items(id INT PK, order_id INT FK->orders.id, sku TEXT, qty INT)
    """

# Attach another resource (async)
@sql_skill.resource
async def indexes() -> str:
    return "users(email), orders(user_id, status), order_items(order_id)"

# Attach a script (in-process, args as dict)
@sql_skill.script
def explain_query(sql: str) -> str:
    return f"EXPLAIN ANALYSE {sql}"
```

```python
import asyncio

async def use_inline_skill():
    content = await sql_skill.get_content()
    # XML envelope with <name>, <description>, <instructions>, <resources>, <scripts>
    print(content[:200])

    # Retrieve a resource by name
    schema_res = await sql_skill.get_resource("schema")
    if schema_res:
        print(await schema_res.get_content())

    # Retrieve a script by name
    explain_script = await sql_skill.get_script("explain_query")
    if explain_script:
        result = await explain_script.run({"sql": "SELECT * FROM orders WHERE status='pending'"})
        print(result)

asyncio.run(use_inline_skill())
```

```python
# Plug InlineSkill into an agent via SkillsProvider
from agent_framework import Agent
from agent_framework._skills import SkillsProvider, InMemorySkillsSource
from agent_framework.openai import OpenAIChatClient

async def agent_with_inline_skill():
    source = InMemorySkillsSource([sql_skill])
    provider = SkillsProvider(source)

    agent = Agent(
        name="sql-agent",
        instructions="You are a SQL assistant. Use your skills to help the user.",
        client=OpenAIChatClient(model="gpt-4o"),
        context_providers=[provider],
    )
    async for event in agent.run("Write a query to find all users with pending orders", stream=True):
        if event.type == "agent_response_update":
            print(event.update.text or "", end="", flush=True)
    print()
```

---

## 3. `InMemorySkillsSource` · `AggregatingSkillsSource` · `FilteringSkillsSource` · `DeduplicatingSkillsSource`

**Module:** `agent_framework._skills`  
**Status:** `@experimental(ExperimentalFeature.SKILLS)`

The skills source pipeline lets you compose, filter, and deduplicate skills from multiple origins without touching the consumer (`SkillsProvider`).

| Class | Role |
|---|---|
| `InMemorySkillsSource` | Holds pre-built `Skill` instances; identity source |
| `AggregatingSkillsSource` | Concatenates results from multiple sources (fan-in) |
| `FilteringSkillsSource` | Delegates to inner source, keeps only skills matching predicate |
| `DeduplicatingSkillsSource` | Delegates to inner source, first-one-wins by name (case-insensitive) |

```python
from agent_framework._skills import (
    InlineSkill, SkillFrontmatter, InMemorySkillsSource,
    AggregatingSkillsSource, FilteringSkillsSource, DeduplicatingSkillsSource,
)

# --- Build some skills ---
def make_skill(name: str, desc: str) -> InlineSkill:
    return InlineSkill(
        frontmatter=SkillFrontmatter(name=name, description=desc),
        instructions=f"Instructions for {name}.",
    )

sql_skill   = make_skill("sql-expert",   "SQL query generation")
web_skill   = make_skill("web-search",   "Real-time web search")
code_skill  = make_skill("code-helper",  "Code review and refactoring")
beta_skill  = make_skill("experimental", "Unstable beta capability")

# --- Identity source ---
prod_source = InMemorySkillsSource([sql_skill, web_skill, code_skill])
beta_source = InMemorySkillsSource([beta_skill, sql_skill])  # sql duplicated
```

```python
import asyncio

async def compose_sources():
    # Aggregate two sources (simple union — may include duplicates)
    all_source = AggregatingSkillsSource([prod_source, beta_source])
    skills = await all_source.get_skills()
    print([s.frontmatter.name for s in skills])
    # ['sql-expert', 'web-search', 'code-helper', 'experimental', 'sql-expert']

    # Deduplicate: first occurrence of 'sql-expert' wins
    deduped = DeduplicatingSkillsSource(all_source)
    skills = await deduped.get_skills()
    print([s.frontmatter.name for s in skills])
    # ['sql-expert', 'web-search', 'code-helper', 'experimental']

    # Filter: exclude beta skills by name pattern
    safe_only = FilteringSkillsSource(
        inner_source=deduped,
        predicate=lambda s: s.frontmatter.name != "experimental",
    )
    skills = await safe_only.get_skills()
    print([s.frontmatter.name for s in skills])
    # ['sql-expert', 'web-search', 'code-helper']

asyncio.run(compose_sources())
```

```python
# Full pipeline: aggregate → deduplicate → filter → serve via SkillsProvider
from agent_framework._skills import SkillsProvider

async def build_provider():
    pipeline = FilteringSkillsSource(
        inner_source=DeduplicatingSkillsSource(
            AggregatingSkillsSource([prod_source, beta_source])
        ),
        predicate=lambda s: (
            s.frontmatter.name != "experimental"
            and len(s.frontmatter.description) > 10
        ),
    )
    provider = SkillsProvider(pipeline)
    return provider

# The SkillsProvider lazy-loads skills on the first agent invocation.
```

---

## 4. `AgentFileStore` · `InMemoryAgentFileStore` · `FileSearchMatch` · `FileSearchResult`

**Module:** `agent_framework._harness._file_access`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

`AgentFileStore` is the ABC that all file-store backends must implement. `InMemoryAgentFileStore` is the reference in-process implementation (tests, lightweight demos). Both DTOs — `FileSearchMatch` (one matching line) and `FileSearchResult` (one matching file) — implement `SerializationMixin` with custom `to_dict` / `from_dict` that work correctly with `__slots__`.

### `AgentFileStore` public API

| Method | Signature | Description |
|---|---|---|
| `write_file` | `(path, content, *, overwrite=True)` | Write string content to path |
| `read_file` | `(path) → str \| None` | Read content or `None` if absent |
| `delete_file` | `(path) → bool` | Delete file; returns `True` if deleted |
| `file_exists` | `(path) → bool` | Check whether a file exists |
| `create_directory` | `(path)` | Create a directory (and any parents) |
| `list_files` | `(directory="") → list[str]` | Direct child file names |
| `list_directories` | `(directory="") → list[str]` | Direct child subdirectory names |
| `search_files` | `(directory, regex_pattern, file_pattern=None, *, recursive=False) → list[FileSearchResult]` | Regex search; optionally scoped to a sub-directory and filtered by filename glob |

```python
import asyncio
from agent_framework._harness._file_access import (
    InMemoryAgentFileStore, FileSearchMatch, FileSearchResult
)

async def demo_file_store():
    store = InMemoryAgentFileStore()

    # Write files
    await store.write_file("reports/q1.md", "# Q1 Report\nRevenue: $1.2M")
    await store.write_file("reports/q2.md", "# Q2 Report\nRevenue: $1.5M")
    await store.write_file("notes/todo.txt", "Buy milk\nFinish report")

    # Exclusive create — raises FileExistsError on collision
    try:
        await store.write_file("reports/q1.md", "NEW CONTENT", overwrite=False)
    except FileExistsError as exc:
        print(f"Caught: {exc}")

    # Read
    content = await store.read_file("reports/q1.md")
    print(content)  # "# Q1 Report\nRevenue: $1.2M"

    # List directory
    files = await store.list_files("reports")
    print(files)  # ["q1.md", "q2.md"]

    # Delete
    deleted = await store.delete_file("notes/todo.txt")
    print(deleted)  # True

asyncio.run(demo_file_store())
```

```python
# FileSearchMatch and FileSearchResult construction and round-trip
match = FileSearchMatch(line_number=3, line="Revenue: $1.2M")
print(match.line_number)  # 3
print(match.to_dict())    # {"line_number": 3, "line": "Revenue: $1.2M"}

restored = FileSearchMatch.from_dict({"line_number": 3, "line": "Revenue: $1.2M"})
assert restored == match

result = FileSearchResult(
    file_name="reports/q1.md",
    snippet="Revenue: $1.2M",
    matching_lines=[match],
)
print(result.to_dict())
# {"file_name": "reports/q1.md", "snippet": "Revenue: $1.2M",
#  "matching_lines": [{"line_number": 3, "line": "Revenue: $1.2M"}]}
```

```python
# Custom AgentFileStore backed by Azure Blob Storage
from agent_framework._harness._file_access import AgentFileStore
from azure.storage.blob.aio import BlobServiceClient

class AzureBlobFileStore(AgentFileStore):
    def __init__(self, connection_string: str, container: str):
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container = container

    async def write_file(self, path: str, content: str, *, overwrite: bool = True) -> None:
        blob = self._client.get_blob_client(container=self._container, blob=path)
        await blob.upload_blob(content.encode(), overwrite=overwrite)

    async def read_file(self, path: str) -> str | None:
        blob = self._client.get_blob_client(container=self._container, blob=path)
        try:
            stream = await blob.download_blob()
            return (await stream.readall()).decode()
        except Exception:
            return None

    async def delete_file(self, path: str) -> bool:
        blob = self._client.get_blob_client(container=self._container, blob=path)
        try:
            await blob.delete_blob()
            return True
        except Exception:
            return False

    async def list_files(self, directory: str = "") -> list[str]:
        prefix = f"{directory}/" if directory else ""
        container_client = self._client.get_container_client(self._container)
        names = [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
        return [n.removeprefix(prefix) for n in names if "/" not in n.removeprefix(prefix)]

    async def file_exists(self, path: str) -> bool:
        blob = self._client.get_blob_client(container=self._container, blob=path)
        return await blob.exists()

    async def create_directory(self, path: str) -> None:
        pass  # Azure Blob Storage uses virtual directories; no explicit creation needed

    async def list_directories(self, directory: str = "") -> list[str]:
        prefix = f"{directory}/" if directory else ""
        container_client = self._client.get_container_client(self._container)
        dirs: set[str] = set()
        async for blob in container_client.list_blobs(name_starts_with=prefix):
            rest = blob.name.removeprefix(prefix)
            if "/" in rest:
                dirs.add(rest.split("/")[0])
        return sorted(dirs)

    async def search_files(
        self,
        directory: str,
        regex_pattern: str,
        file_pattern: str | None = None,
        *,
        recursive: bool = False,
    ) -> list:
        raise NotImplementedError("Use Azure AI Search for full-text search over blob storage.")
```

---

## 5. `FileAccessProvider`

**Module:** `agent_framework._harness._file_access`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

`FileAccessProvider` is a `ContextProvider` that injects six file-management tools into the agent's context at every invocation:

| Tool name | Purpose |
|---|---|
| `file_access_save_file` | Save a file (refuses to overwrite by default) |
| `file_access_read_file` | Read the content of a file by name |
| `file_access_delete_file` | Delete a file (approval-gated by default) |
| `file_access_list_files` | List direct child files of a directory |
| `file_access_list_subdirectories` | List direct child subdirectories |
| `file_access_search_files` | Recursive regex search with optional glob filter |

Unlike `MemoryContextProvider`, the underlying store is **shared and persistent across sessions** — scoping is the caller's responsibility.

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._file_access import (
    FileAccessProvider, InMemoryAgentFileStore
)
from agent_framework.openai import OpenAIChatClient

async def agent_with_file_access():
    store = InMemoryAgentFileStore()
    # Pre-seed some data
    await store.write_file("context/project.md", "# Project Alpha\nDeadline: Q3 2026")
    await store.write_file("context/team.md", "Team lead: Alice\nEngineers: Bob, Carol")

    provider = FileAccessProvider(
        store=store,
        require_delete_approval=True,   # prompt user before any delete
    )

    agent = Agent(
        name="file-agent",
        instructions="You can read, write, search, and list files using your file tools.",
        client=OpenAIChatClient(model="gpt-4o"),
        context_providers=[provider],
    )

    async for event in agent.run("Who is the team lead on Project Alpha?", stream=True):
        if event.type == "agent_response_update":
            print(event.update.text or "", end="", flush=True)
    print()

asyncio.run(agent_with_file_access())
```

```python
# Custom instructions to restrict the agent's file access
provider = FileAccessProvider(
    store=store,
    source_id="report_files",
    instructions=(
        "You have access to the reports/ directory only. "
        "Always save new reports under reports/<YYYY-MM-DD>_<name>.md. "
        "Never delete files without explicit user confirmation."
    ),
    require_delete_approval=True,
)
```

```python
# Two agents sharing the same store — one writes, the other reads
async def two_agent_shared_store():
    store = InMemoryAgentFileStore()
    provider = FileAccessProvider(store=store)
    client = OpenAIChatClient(model="gpt-4o")

    writer = Agent(
        name="writer-agent",
        instructions="Save analysis results as Markdown files under results/.",
        client=client,
        context_providers=[provider],
    )
    reader = Agent(
        name="reader-agent",
        instructions="Read and summarize files from the results/ directory.",
        client=client,
        context_providers=[provider],
    )

    await writer.run("Analyse the market and save your findings.")
    await reader.run("Summarise everything in results/.")
```

---

## 6. `BackgroundAgentsProvider` · `BackgroundTaskInfo` · `BackgroundTaskStatus`

**Module:** `agent_framework._harness._background_agents`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

Enables a **parent agent to delegate work to named sub-agents that run concurrently in background asyncio tasks**, each in its own `AgentSession`. The provider injects six tools and maintains per-session runtime state (in-flight `asyncio.Task` objects) plus serializable `BackgroundTaskInfo` records in the session state.

### `BackgroundTaskStatus`

`str` enum: `RUNNING` · `COMPLETED` · `FAILED` · `LOST` (session reference unavailable after process restart).

### `BackgroundTaskInfo` fields

| Field | Type | Notes |
|---|---|---|
| `id` | `int` | Auto-incremented per session |
| `agent_name` | `str` | Name of the sub-agent |
| `description` | `str` | Human-readable task description |
| `status` | `BackgroundTaskStatus` | Default `RUNNING` |
| `result_text` | `str \| None` | Set on completion |
| `error_text` | `str \| None` | Set on failure |

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._background_agents import (
    BackgroundAgentsProvider, BackgroundTaskInfo, BackgroundTaskStatus
)
from agent_framework.openai import OpenAIChatClient

async def background_delegation_demo():
    client = OpenAIChatClient(model="gpt-4o")

    # Sub-agents available for delegation
    research_agent = Agent(
        name="research-agent",
        instructions="You are a research specialist. Produce concise summaries.",
        client=client,
    )
    code_agent = Agent(
        name="code-agent",
        instructions="You are a coding assistant. Return working Python code.",
        client=client,
    )

    # Coordinator agent with background delegation
    coordinator = Agent(
        name="coordinator",
        instructions=(
            "You coordinate tasks. Delegate research to 'research-agent' and "
            "coding tasks to 'code-agent' using background_agents_start_task. "
            "Wait for both to finish before synthesising the results."
        ),
        client=client,
        context_providers=[
            BackgroundAgentsProvider(agents=[research_agent, code_agent])
        ],
    )

    # Agent.run(stream=True) yields AgentResponseUpdate chunks directly
    async for chunk in coordinator.run(
        "Research async patterns in Python AND write a simple asyncio example. "
        "Run both tasks in parallel and then present the combined result.",
        stream=True,
    ):
        print(chunk.text or "", end="", flush=True)
    print()

asyncio.run(background_delegation_demo())
```

```python
# Inspect BackgroundTaskInfo serialization
info = BackgroundTaskInfo(
    id=1,
    agent_name="research-agent",
    description="Summarise async patterns",
    status=BackgroundTaskStatus.COMPLETED,
    result_text="Async patterns include: event loops, coroutines, Tasks, and gather.",
)
d = info.to_dict()
print(d)
# {"id": 1, "agent_name": "research-agent", "description": "...",
#  "status": "completed", "result_text": "Async patterns include..."}

restored = BackgroundTaskInfo.from_dict(d)
assert restored.status == BackgroundTaskStatus.COMPLETED
assert restored.result_text == info.result_text
```

```python
# Custom source_id when using multiple BackgroundAgentsProvider instances
from agent_framework._harness._background_agents import BackgroundAgentsProvider

provider_a = BackgroundAgentsProvider(
    agents=[research_agent],
    source_id="research_pool",
    instructions=(
        "You can delegate research tasks to the research pool: {background_agents}\n"
        "Use task IDs to wait for completion and retrieve results."
    ),
)
provider_b = BackgroundAgentsProvider(
    agents=[code_agent],
    source_id="code_pool",
)
# Both providers register distinct tool name-spaces, so an agent can use both.
```

---

## 7. `MemoryStore` · `MemoryTopicRecord` · `MemoryIndexEntry`

**Module:** `agent_framework._harness._memory`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

The memory harness persists knowledge across sessions as **topic files** (Markdown) plus a pointer index (`MEMORY.md`). `MemoryStore` is the ABC; concrete implementations (filesystem, Cosmos, Redis) inherit from it. `MemoryTopicRecord` and `MemoryIndexEntry` are the serializable data types.

### `MemoryTopicRecord`

Represents one `topics/<slug>.md` file.

| Field | Type | Notes |
|---|---|---|
| `topic` | `str` | Normalised topic name |
| `slug` | `str` | Stable filename stem (`_slugify_topic`) |
| `summary` | `str` | Short paragraph used in `MEMORY.md` |
| `memories` | `list[str]` | De-duplicated bullet list |
| `updated_at` | `str` | ISO-8601 timestamp |
| `session_ids` | `list[str]` | Contributing session IDs |

### `MemoryIndexEntry`

One pointer line in `MEMORY.md`.

| Field | Type | Notes |
|---|---|---|
| `topic` | `str` | Human-readable label |
| `slug` | `str` | `[topic](topics/<slug>.md)` link target |
| `summary` | `str` | Short summary for the pointer line |
| `updated_at` | `str` | Last update |

```python
from agent_framework._harness._memory import MemoryTopicRecord, MemoryIndexEntry

# Build a topic record
record = MemoryTopicRecord(
    topic="Python Async Patterns",
    summary="The user prefers asyncio.gather over explicit Task objects.",
    memories=[
        "User uses asyncio.gather for concurrent coroutines.",
        "User finds asyncio.TaskGroup cleaner for structured concurrency.",
        "User avoids loop.run_until_complete in async codebases.",
    ],
    updated_at="2026-06-20T10:30:00Z",
    session_ids=["sess-abc123", "sess-def456"],
)

print(record.topic)   # "Python Async Patterns"
print(record.slug)    # "python-async-patterns"
print(len(record.memories))  # 3
```

```python
# Round-trip: dict serialization
d = record.to_dict()
restored = MemoryTopicRecord.from_dict(d)
assert restored.topic == record.topic
assert restored.memories == record.memories
assert restored.session_ids == record.session_ids

# Markdown serialization — this is what gets written to disk
md = record.to_markdown()
print(md)
```

```python
# Parse Markdown back into a MemoryTopicRecord
raw_md = """# Python Async Patterns

Updated: 2026-06-20T10:30:00Z
Sessions: sess-abc123, sess-def456

## Summary
The user prefers asyncio.gather over explicit Task objects.

## Memories
- User uses asyncio.gather for concurrent coroutines.
- User finds asyncio.TaskGroup cleaner for structured concurrency.
- User avoids loop.run_until_complete in async codebases.
"""
parsed = MemoryTopicRecord.from_markdown(raw_md)
assert parsed.topic == "Python Async Patterns"
assert len(parsed.memories) == 3
```

```python
# MemoryIndexEntry: pointer line for MEMORY.md
entry = MemoryIndexEntry(
    topic="Python Async Patterns",
    slug="python-async-patterns",
    summary="The user prefers asyncio.gather over explicit Task objects.",
    updated_at="2026-06-20T10:30:00Z",
)
print(entry.to_pointer_line())
# "- [Python Async Patterns](topics/python-async-patterns.md): The user prefers asyncio.gather..."

# Derive from a topic record
entry2 = MemoryIndexEntry.from_topic_record(record)
assert entry2.slug == record.slug

# Custom max_length for pointer line
short = entry.to_pointer_line(max_length=60)
print(len(short))  # ≤ 60
```

```python
# Custom MemoryStore backed by a simple dict (for tests)
from agent_framework._harness._memory import MemoryStore, MemoryTopicRecord, MemoryIndexEntry
from pathlib import Path
from typing import Any, Sequence, Mapping

class DictMemoryStore(MemoryStore):
    """Toy in-memory store for unit tests."""

    def __init__(self):
        self._topics: dict[str, MemoryTopicRecord] = {}

    def list_topics(self, session, *, source_id: str) -> list[MemoryTopicRecord]:
        return list(self._topics.values())

    def get_topic(self, session, *, source_id: str, topic: str) -> MemoryTopicRecord:
        key = topic.lower()
        match = next((v for k, v in self._topics.items() if k == key), None)
        if match is None:
            raise KeyError(f"Topic not found: {topic}")
        return match

    def write_topic(self, session, record: MemoryTopicRecord, *, source_id: str) -> None:
        self._topics[record.topic.lower()] = record

    def delete_topic(self, session, *, source_id: str, topic: str) -> None:
        self._topics.pop(topic.lower(), None)

    def rebuild_index(self, session, *, source_id, line_limit, line_length) -> list[MemoryIndexEntry]:
        return [MemoryIndexEntry.from_topic_record(r) for r in self._topics.values()]

    def get_index_text(self, session, *, source_id, line_limit, line_length, index_entries=None) -> str:
        entries = index_entries or self.rebuild_index(session, source_id=source_id, line_limit=line_limit, line_length=line_length)
        return "\n".join(e.to_pointer_line() for e in entries[:line_limit])

    def read_state(self, session, *, source_id: str) -> dict[str, Any]:
        return {}

    def write_state(self, session, state: Mapping[str, Any], *, source_id: str) -> None:
        pass

    def get_transcripts_directory(self, session, *, source_id: str) -> Path:
        return Path("/tmp/transcripts")

    def search_transcripts(self, session, *, source_id, query, session_id=None, limit=20):
        return []
```

---

## 8. `WorkflowGraphValidator` · `EdgeDuplicationError` · `GraphConnectivityError` · `TypeCompatibilityError`

**Module:** `agent_framework._workflows._validation`

The `WorkflowGraphValidator` is the internal static-analysis pass that runs before a `Workflow` starts. It performs seven ordered checks:

| Check | Error type raised |
|---|---|
| Start executor present in graph | `GraphConnectivityError` |
| No duplicate edges | `EdgeDuplicationError` |
| Handler `WorkflowContext[T]` annotations valid | (build-time, skipped here) |
| Type compatibility source → target | `TypeCompatibilityError` |
| Graph connectivity (all reachable from start) | `GraphConnectivityError` |
| No self-loops (warning only) | — |
| No dead-ends (info only) | — |
| Output/intermediate executor designations valid | `WorkflowValidationError` |

```python
from agent_framework._workflows._validation import (
    WorkflowGraphValidator,
    EdgeDuplicationError,
    GraphConnectivityError,
    TypeCompatibilityError,
    WorkflowValidationError,
)
from agent_framework._workflows._edge import SingleEdgeGroup
from agent_framework._workflows._function_executor import FunctionExecutor
from agent_framework import WorkflowContext

# Build two simple executors to test the validator
async def step_a(ctx: WorkflowContext[str]) -> int:
    return len(ctx.message)

async def step_b(ctx: WorkflowContext[int]) -> str:
    return f"Length was {ctx.message}"

exec_a = FunctionExecutor(step_a, id="step_a")
exec_b = FunctionExecutor(step_b, id="step_b")

# Create an edge group A → B
edge_group = SingleEdgeGroup(source=exec_a, target=exec_b)

validator = WorkflowGraphValidator()
validator.validate_workflow(
    edge_groups=[edge_group],
    executors={"step_a": exec_a, "step_b": exec_b},
    start_executor=exec_a,
    output_executors=["step_b"],
)
print("Validation passed!")
```

```python
# Catch EdgeDuplicationError when edges are repeated
try:
    validator.validate_workflow(
        edge_groups=[edge_group, edge_group],   # same object twice
        executors={"step_a": exec_a, "step_b": exec_b},
        start_executor=exec_a,
        output_executors=["step_b"],
    )
except EdgeDuplicationError as exc:
    print(f"Duplicate edge: {exc.edge_id}")
    print(f"Validation type: {exc.validation_type}")
```

```python
# Catch GraphConnectivityError when a node is unreachable
async def orphan(ctx: WorkflowContext[str]) -> str:
    return "I am unreachable"

exec_orphan = FunctionExecutor(orphan, id="orphan")

try:
    validator.validate_workflow(
        edge_groups=[edge_group],
        executors={"step_a": exec_a, "step_b": exec_b, "orphan": exec_orphan},
        start_executor=exec_a,
        output_executors=["step_b"],
    )
except GraphConnectivityError as exc:
    print(f"Connectivity error: {exc}")
    # "'orphan' unreachable from start executor 'step_a'"
```

```python
# Catch TypeCompatibilityError on mismatched types
async def bad_target(ctx: WorkflowContext[list[float]]) -> str:
    return "expects a list, not int"

exec_bad = FunctionExecutor(bad_target, id="bad_target")
edge_bad = SingleEdgeGroup(source=exec_a, target=exec_bad)

try:
    validator.validate_workflow(
        edge_groups=[edge_bad],
        executors={"step_a": exec_a, "bad_target": exec_bad},
        start_executor=exec_a,
        output_executors=["bad_target"],
    )
except TypeCompatibilityError as exc:
    print(f"Type error: {exc}")
    # "step_a outputs [int] but bad_target expects [list[float]]"
```

---

## 9. `MagenticBuilder` · `MagenticManagerBase` · `MagenticProgressLedger` · `MagenticProgressLedgerItem`

**Module:** `agent_framework_orchestrations._magentic`

Magentic One is an LLM-powered orchestration pattern in which a **manager** coordinates multiple participant agents through iterative planning, progress tracking, and adaptive replanning. The progress ledger is the per-round diagnostic that determines whether to continue, replan, or finish.

### `MagenticProgressLedgerItem`

One boolean/string verdict in the ledger with a supporting `reason`.

### `MagenticProgressLedger`

Five-item ledger evaluated by the manager at each round:

| Field | `answer` type | Meaning |
|---|---|---|
| `is_request_satisfied` | `bool` | Goal met — ready to finish |
| `is_in_loop` | `bool` | Participants are cycling without progress |
| `is_progress_being_made` | `bool` | At least one meaningful step happened this round |
| `next_speaker` | `str` | Name of the participant to invoke next |
| `instruction_or_question` | `str` | Instruction or question for that participant |

### `MagenticManagerBase` (ABC)

| Abstract method | Role |
|---|---|
| `plan(context)` | Produce an initial task plan |
| `replan(context)` | Produce a revised plan after stall detection |
| `create_progress_ledger(context)` | Evaluate the current round's progress |
| `prepare_final_answer(context)` | Synthesise the final response |

```python
from agent_framework_orchestrations._magentic import (
    MagenticBuilder, MagenticOrchestrator, StandardMagenticManager,
    MagenticProgressLedger, MagenticProgressLedgerItem,
)
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# Inspect a progress ledger
ledger = MagenticProgressLedger(
    is_request_satisfied=MagenticProgressLedgerItem(reason="Both tasks complete.", answer=True),
    is_in_loop=MagenticProgressLedgerItem(reason="Different agents used each round.", answer=False),
    is_progress_being_made=MagenticProgressLedgerItem(reason="Code and research both advanced.", answer=True),
    next_speaker=MagenticProgressLedgerItem(reason="Need final synthesis.", answer="writer-agent"),
    instruction_or_question=MagenticProgressLedgerItem(
        reason="Writer should compile the final report.",
        answer="Compile the research and code into a final report.",
    ),
)
print(ledger.is_request_satisfied.answer)        # True
print(ledger.next_speaker.answer)                # "writer-agent"
print(ledger.instruction_or_question.answer)     # "Compile the research..."

# Round-trip
d = ledger.to_dict()
restored = MagenticProgressLedger.from_dict(d)
assert restored.is_request_satisfied.answer == True
```

```python
import asyncio
from agent_framework.openai import OpenAIChatClient

async def magentic_workflow():
    client = OpenAIChatClient(model="gpt-4o")

    # Participants
    researcher = Agent(
        name="researcher",
        instructions="Search for facts and produce concise research notes.",
        client=client,
    )
    coder = Agent(
        name="coder",
        instructions="Write clean, documented Python code based on given specifications.",
        client=client,
    )
    writer = Agent(
        name="writer",
        instructions="Compile research and code into a polished technical report.",
        client=client,
    )

    # Manager agent that drives the orchestration
    manager_agent = Agent(
        name="manager",
        instructions="You are the orchestration manager. Plan, delegate, monitor, and synthesise.",
        client=OpenAIChatClient(model="gpt-4o"),
    )

    # Build the Magentic workflow
    workflow = (
        MagenticBuilder(
            participants=[researcher, coder, writer],
            manager_agent=manager_agent,
            max_stall_count=2,          # replan after 2 stalled rounds
            max_round_count=20,         # hard cap on total rounds
            enable_plan_review=False,   # auto-approve plan
        )
        .build()
    )

    result = await workflow.run(
        "Build a Python async rate-limiter and document it with examples.",
    )
    print(result.output)

asyncio.run(magentic_workflow())
```

```python
# Custom MagenticManagerBase: minimal implementation
from agent_framework_orchestrations._magentic import (
    MagenticManagerBase, MagenticContext, MagenticProgressLedger, MagenticProgressLedgerItem
)
from agent_framework._types import Message

class DeterministicManager(MagenticManagerBase):
    """Always assigns work to the first participant, stops after 3 rounds."""

    def __init__(self, participant_name: str):
        super().__init__(max_stall_count=1, max_round_count=3)
        self._participant = participant_name
        self._round = 0

    async def plan(self, context: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Step 1: delegate all work to the specialist."])

    async def replan(self, context: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Continuing with the same plan."])

    async def create_progress_ledger(self, context: MagenticContext) -> MagenticProgressLedger:
        self._round += 1
        done = self._round >= 3
        return MagenticProgressLedger(
            is_request_satisfied=MagenticProgressLedgerItem(reason="Done?", answer=done),
            is_in_loop=MagenticProgressLedgerItem(reason="No loop.", answer=False),
            is_progress_being_made=MagenticProgressLedgerItem(reason="Always progressing.", answer=True),
            next_speaker=MagenticProgressLedgerItem(reason="One agent.", answer=self._participant),
            instruction_or_question=MagenticProgressLedgerItem(reason="Continue.", answer=context.task),
        )

    async def prepare_final_answer(self, context: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Task completed by the specialist."])
```

---

## 10. `LocalEvaluator` · `EvalItem` · `EvalScoreResult` · `ConversationSplit` · `ExpectedToolCall`

**Module:** `agent_framework._evaluation`  
**Status:** `@experimental(ExperimentalFeature.EVALS)`

The evaluation framework lets you assess agent responses without calling any external API. `LocalEvaluator` runs a battery of sync or async check functions over a list of `EvalItem` objects and aggregates pass/fail counts.

### Key types

| Type | Role |
|---|---|
| `EvalItem` | One query/response interaction; `query` and `response` properties derived from the conversation split |
| `ConversationSplit` | `LAST_TURN` or `FULL` — callable enum that splits a `list[Message]` into `(query_msgs, response_msgs)` |
| `ExpectedToolCall` | `name` + optional `arguments` — declares what tool calls the agent must make |
| `EvalScoreResult` | Per-evaluator score for one item: `name`, `score (0.0–1.0)`, `passed`, `sample`, `dimensions` |
| `CheckResult` | One check outcome: `passed`, `reason`, `check_name` |

```python
import asyncio
from agent_framework._evaluation import (
    EvalItem, ConversationSplit, ExpectedToolCall,
    LocalEvaluator, EvalScoreResult, CheckResult,
)
from agent_framework._types import Message

# Build a conversation
conversation = [
    Message(role="user",      contents=["What is the capital of France?"]),
    Message(role="assistant", contents=["The capital of France is Paris."]),
]

# EvalItem: LAST_TURN split (default)
item = EvalItem(
    conversation=conversation,
    expected_output="Paris",
    context="European geography facts",
)
print(item.query)    # "What is the capital of France?"
print(item.response) # "The capital of France is Paris."

# Explicit FULL split: first user message is query, everything else is response
item_full = EvalItem(
    conversation=conversation,
    split_strategy=ConversationSplit.FULL,
)
q_msgs, r_msgs = item_full.split_messages()
print(len(q_msgs), len(r_msgs))  # 1, 1
```

```python
# Per-turn items from a multi-turn conversation
multi_turn = [
    Message(role="user",      contents=["What is the capital of France?"]),
    Message(role="assistant", contents=["Paris."]),
    Message(role="user",      contents=["And Germany?"]),
    Message(role="assistant", contents=["Berlin."]),
]
items = EvalItem.per_turn_items(multi_turn, context="European capitals")
print(len(items))          # 2
print(items[0].query)      # "What is the capital of France?"
print(items[1].query)      # "What is the capital of France? And Germany?"
```

```python
# ExpectedToolCall — check the agent called the right tools
expected_calls = [
    ExpectedToolCall(name="get_weather", arguments={"location": "London"}),
    ExpectedToolCall(name="send_notification"),  # arguments=None → don't check args
]
item_with_tools = EvalItem(
    conversation=conversation,
    expected_tool_calls=expected_calls,
)
```

```python
# LocalEvaluator with built-in check functions
from agent_framework import keyword_check, tool_called_check, evaluate_agent

async def run_local_eval():
    # keyword_check: passes if the keyword appears in the response (case-insensitive)
    # tool_called_check: passes if the named tool was invoked in the conversation
    evaluator = LocalEvaluator(
        keyword_check("Paris"),
        keyword_check("capital"),
    )

    items = [
        EvalItem(
            conversation=[
                Message(role="user",      contents=["What is the capital of France?"]),
                Message(role="assistant", contents=["The capital of France is Paris."]),
            ],
            expected_output="Paris",
        ),
        EvalItem(
            conversation=[
                Message(role="user",      contents=["What is the capital of Germany?"]),
                Message(role="assistant", contents=["Berlin."]),   # missing "capital"
            ],
            expected_output="Berlin",
        ),
    ]

    results = await evaluator.evaluate(items, eval_name="Geography Eval")
    print(f"Passed : {results.passed_count}/{results.total_count}")
    print(f"Pass % : {results.pass_rate:.0%}")
    for item_result in results.items:
        print(f"  Item {item_result.item_id}: {item_result.status}")
        for score in item_result.scores:
            print(f"    [{score.name}] score={score.score:.1f} passed={score.passed}")

asyncio.run(run_local_eval())
```

```python
# Custom check function using the @evaluator decorator
from agent_framework import evaluator

@evaluator(name="length_check")
def response_length_check(item: EvalItem) -> CheckResult:
    """Pass if the agent's response is between 10 and 500 characters."""
    length = len(item.response)
    passed = 10 <= length <= 500
    return CheckResult(
        passed=passed,
        reason=f"Response length {length} {'in' if passed else 'out of'} range [10, 500]",
        check_name="length_check",
    )

@evaluator(name="no_apology_check")
async def no_apology_check(item: EvalItem) -> CheckResult:
    """Pass if the response does not start with an apology."""
    lower = item.response.lower().strip()
    apology_starters = ("i'm sorry", "i apologize", "apologies", "sorry")
    starts_with_apology = any(lower.startswith(s) for s in apology_starters)
    return CheckResult(
        passed=not starts_with_apology,
        reason="Starts with apology" if starts_with_apology else "No apology detected",
        check_name="no_apology_check",
    )

# Compose built-in + custom checks
strict_eval = LocalEvaluator(
    keyword_check("Paris"),
    response_length_check,
    no_apology_check,
)
```

```python
# End-to-end: evaluate a live agent with LocalEvaluator
from agent_framework import Agent, evaluate_agent
from agent_framework.openai import OpenAIChatClient

async def evaluate_geography_agent():
    agent = Agent(
        name="geography-bot",
        instructions="Answer geography questions concisely.",
        client=OpenAIChatClient(model="gpt-4o"),
    )

    queries = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the largest ocean?",
    ]

    evaluator = LocalEvaluator(
        response_length_check,
        no_apology_check,
    )

    results = await evaluate_agent(
        agent=agent,
        queries=queries,
        evaluators=evaluator,
    )
    # evaluate_agent returns a list of results, one per evaluator
    eval_result = results[0]
    print(f"Pass rate: {eval_result.pass_rate:.0%}")
    for item_result in eval_result.items:
        status_icon = "✓" if item_result.status == "pass" else "✗"
        print(f"  {status_icon} {item_result.item_id}")

asyncio.run(evaluate_geography_agent())
```

---

## Revision history

| Version | Date | Changes |
|---|---|---|
| Vol. 18 | 2026-06-20 | `Skill`+`SkillFrontmatter`+`SkillScriptRunner`; `InlineSkill`; skills source pipeline; `AgentFileStore`+`InMemoryAgentFileStore`+DTOs; `FileAccessProvider`; `BackgroundAgentsProvider`+`BackgroundTaskInfo`; `MemoryStore`+`MemoryTopicRecord`+`MemoryIndexEntry`; `WorkflowGraphValidator`+error types; Magentic progress ledger internals; `LocalEvaluator`+`EvalItem`+`ConversationSplit` — all source-verified at agent-framework 1.9.0 |

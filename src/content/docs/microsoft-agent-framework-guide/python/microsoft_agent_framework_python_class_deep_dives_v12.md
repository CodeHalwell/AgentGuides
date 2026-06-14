---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 12"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: Skill+SkillResource+SkillScript ABCs, FileSkill, InlineSkillResource+InlineSkillScript, FileSkillScript+SkillScriptRunner, SupportsAgentRun, RunnerContext, SwitchCaseEdgeGroupCase+SwitchCaseEdgeGroupDefault, ValidationTypeEnum+WorkflowValidationError hierarchy, A2AAgent+A2AAgentSession+A2AExecutor, WorkflowRunnerException+WorkflowCheckpointException+MiddlewareException."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 35
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 12

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source at `/tmp/af_install/agent_framework/`. No API name has been guessed or inferred
from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, `CompactionProvider`, strategy classes, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — harness providers, compaction strategies, `WorkflowViz`, MCP transports
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — message/chat types, `ResponseStream`, `AgentContext`, functional workflows, `SkillsSource` composition, eval data model, tokenizer, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — feature staging, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, embedding clients, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, orchestration builders, `AgentFactory`, `SecureAgentConfig`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — file store hierarchy, `FileAccessProvider`, `MCPSkill`, `ToolMode`, eval helpers, `ChatContext`, `WorkflowAgent`, compaction, history providers, skills composition
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `Mem0ContextProvider`, Redis providers, Magentic internals, `FileSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow`, `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, memory harness, todo harness, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `BaseAgent`
- [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) — telemetry layers, `Edge`+`EdgeGroup` primitives, `Case`+`Default`, `EdgeRunner` hierarchy, `ExecutionContext`, `WorkflowGraphValidator`, `MCPTool`, serialization mixin, `Evaluator`, `PerServiceCallHistoryPersistingMiddleware`

This volume covers **ten class groups** that were not documented in earlier volumes —
focusing on the Skills type-system ABCs, the code-defined and file-backed skill script
primitives, the structural agent protocol, the workflow runner-context protocol, the
serialisable edge-routing descriptors, the workflow validation error taxonomy, the A2A
protocol integration, and the remaining specialised exception classes:

| # | Class / group | Module |
|---|---|---|
| 1 | `Skill` + `SkillResource` + `SkillScript` | `agent_framework._skills` |
| 2 | `FileSkill` | `agent_framework._skills` |
| 3 | `InlineSkillResource` + `InlineSkillScript` | `agent_framework._skills` |
| 4 | `FileSkillScript` + `SkillScriptRunner` | `agent_framework._skills` |
| 5 | `SupportsAgentRun` | `agent_framework._agents` |
| 6 | `RunnerContext` | `agent_framework._workflows._runner_context` |
| 7 | `SwitchCaseEdgeGroupCase` + `SwitchCaseEdgeGroupDefault` | `agent_framework._workflows._edge` |
| 8 | `ValidationTypeEnum` + `WorkflowValidationError` + subclasses | `agent_framework._workflows._validation` |
| 9 | `A2AAgent` + `A2AAgentSession` + `A2AExecutor` | `agent_framework.a2a` |
| 10 | `WorkflowRunnerException` + `WorkflowCheckpointException` + `MiddlewareException` | `agent_framework.exceptions` |

> **Experimental notice:** All Skills classes carry the `@experimental(feature_id=ExperimentalFeature.SKILLS)` decorator. They emit `ExperimentalWarning` on import and **must not be used in production without accepting API-stability risk**. Suppress the warning with `warnings.filterwarnings("ignore", category=ExperimentalWarning)` in tests.

---

## 1 · `Skill` + `SkillResource` + `SkillScript`

**Module:** `agent_framework._skills`

These three abstract base classes form the **Skills type system**. Every concrete skill
class (`FileSkill`, `InlineSkill`, `ClassSkill`) extends `Skill`; every resource
implementation extends `SkillResource`; every script implementation extends `SkillScript`.
Understanding these ABCs is the prerequisite for writing custom skill backends.

### Class signatures

```python
class Skill(ABC):
    @property
    @abstractmethod
    def frontmatter(self) -> SkillFrontmatter: ...

    @abstractmethod
    async def get_content(self) -> str: ...

    async def get_resource(self, name: str) -> SkillResource | None:
        return None  # concrete classes override

    async def get_script(self, name: str) -> SkillScript | None:
        return None  # concrete classes override


class SkillResource(ABC):
    def __init__(
        self,
        *,
        name: str,             # required; raises ValueError if empty or whitespace-only
        description: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def read(self, **kwargs: Any) -> Any: ...


class SkillScript(ABC):
    def __init__(
        self,
        *,
        name: str,             # required; raises ValueError if empty or whitespace-only
        description: str | None = None,
    ) -> None: ...

    @property
    def parameters_schema(self) -> dict[str, Any] | None:
        return None  # concrete classes may override

    @abstractmethod
    async def run(
        self,
        skill: Skill,
        args: dict[str, Any] | list[str] | None = None,
        **kwargs: Any,
    ) -> Any: ...
```

### Key points

- `Skill.get_content()` is the **mandatory** method — it returns the full skill description (SKILL.md text for file skills, synthesised XML for code-defined skills). `SkillsProvider` calls this to inject the skill into the system prompt.
- `Skill.get_resource()` and `Skill.get_script()` return `None` by default. Override them only in concrete classes that support resources or scripts.
- `SkillResource.read()` receives `**kwargs` — these are the runtime keyword arguments forwarded from the agent invocation. A resource backed by a function that does not declare `**kwargs` will still work; the framework inspects the signature and omits unsupported kwargs.
- `SkillScript.parameters_schema` drives the JSON Schema that the LLM sees when deciding how to call the script. Return `None` (the default) to expose no parameter information; override to return a valid JSON Schema dict.
- `SkillScript.run()` receives `args` that may be a `dict` (for inline/function-backed scripts) or a `list[str]` (for file-backed, CLI-style scripts). Implementations must handle or reject the list form explicitly.

### Example 1 — Custom `Skill` backed by a database row

```python
import asyncio
import warnings
from agent_framework import SkillResource, SkillScript, SkillFrontmatter
from agent_framework._skills import Skill
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

class DBSkill(Skill):
    """A skill whose content is fetched from a database on demand."""

    def __init__(self, row: dict):
        self._row = row
        self._fm = SkillFrontmatter(
            name=row["name"],
            description=row.get("description"),
        )

    @property
    def frontmatter(self) -> SkillFrontmatter:
        return self._fm

    async def get_content(self) -> str:
        # In production this might hit an async DB driver
        return self._row["content"]


async def main():
    row = {
        "name": "sql-assistant",
        "description": "Knows your database schema",
        "content": "# SQL Assistant\nYou know the schema: users(id, email), orders(id, user_id, amount).",
    }
    skill = DBSkill(row)
    print(skill.frontmatter.name)          # sql-assistant
    print(await skill.get_content())       # # SQL Assistant\n...
    print(await skill.get_resource("x"))   # None — default


asyncio.run(main())
```

### Example 2 — Custom `SkillResource` with dynamic content

```python
import asyncio
import warnings
from agent_framework._skills import SkillResource
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

class LiveSchemaResource(SkillResource):
    """Returns the current database schema on every read."""

    async def read(self, **kwargs) -> str:
        # Simulate async DB introspection
        return "CREATE TABLE users (id INT, email TEXT);"


async def main():
    res = LiveSchemaResource(name="schema", description="Live DB schema")
    print(res.name)           # schema
    print(await res.read())   # CREATE TABLE users ...

asyncio.run(main())
```

### Example 3 — Custom `SkillScript` with typed args

```python
import asyncio
import warnings
from typing import Any
from agent_framework._skills import SkillScript, Skill
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

class SummaryScript(SkillScript):
    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_words": {"type": "integer", "description": "Maximum words in summary"}
            },
            "required": ["max_words"],
        }

    async def run(self, skill: Skill, args: dict | list | None = None, **kwargs) -> str:
        if isinstance(args, list):
            raise TypeError("SummaryScript requires keyword args, not a list.")
        max_words = (args or {}).get("max_words", 100)
        content = await skill.get_content()
        words = content.split()[:max_words]
        return " ".join(words)
```

---

## 2 · `FileSkill`

**Module:** `agent_framework._skills`

`FileSkill` is the concrete `Skill` implementation backed by a **SKILL.md file on disk**.
`FileSkillsSource` creates `FileSkill` instances by scanning directories; you can also
construct one manually for testing or custom discovery scenarios.

### Class signature

```python
class FileSkill(Skill):
    def __init__(
        self,
        *,
        frontmatter: SkillFrontmatter,
        content: str,            # raw SKILL.md text (including YAML frontmatter)
        path: str,               # absolute path to the skill directory
        resources: Sequence[SkillResource] | None = None,
        scripts: Sequence[SkillScript] | None = None,
    ) -> None: ...

    @property
    def frontmatter(self) -> SkillFrontmatter: ...

    async def get_content(self) -> str:
        # Returns content with an appended <scripts> XML block when scripts exist.
        # Result is cached after the first call.
        ...

    async def get_resource(self, name: str) -> SkillResource | None:
        # Case-insensitive lookup across self._resources
        ...

    async def get_script(self, name: str) -> SkillScript | None:
        # Case-insensitive lookup across self._scripts
        ...
```

### Key points

- `get_content()` **appends a `<scripts>` XML block** to the raw content when `scripts` is non-empty. This makes script names and parameter schemas discoverable by the LLM without manual documentation. The synthesised block looks like:
  ```xml
  <scripts>
    <script name="analyze">
      <description>Run analysis</description>
      <parameters_schema>{"type": "object", ...}</parameters_schema>
    </script>
  </scripts>
  ```
- The result of `get_content()` is **cached** after the first call. If you mutate `_scripts` after construction the cache will not reflect the change.
- `get_resource()` and `get_script()` do **case-insensitive** name matching, so `"Schema"` and `"schema"` are the same resource.
- `path` is an informational attribute only — `FileSkill` itself does not read from disk. `FileSkillsSource` reads the SKILL.md file before constructing the `FileSkill`.

### Example 1 — Constructing a FileSkill manually (testing)

```python
import asyncio
import warnings
from agent_framework import FileSkill, SkillFrontmatter
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def main():
    fm = SkillFrontmatter(name="test-skill", description="A test skill")
    skill = FileSkill(
        frontmatter=fm,
        content="# Test Skill\nDo things.",
        path="/skills/test-skill",
    )

    content = await skill.get_content()
    print(content)  # "# Test Skill\nDo things."

    # Second call returns cached value
    content2 = await skill.get_content()
    assert content is content2  # same object — cached

asyncio.run(main())
```

### Example 2 — FileSkill with scripts produces XML block

```python
import asyncio
import warnings
from agent_framework import FileSkill, SkillFrontmatter, InlineSkillScript
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def run_analysis(query: str) -> str:
    return f"Analysis result for: {query}"

async def main():
    fm = SkillFrontmatter(name="analytics-skill")
    script = InlineSkillScript(name="analyze", description="Analyse data", function=run_analysis)
    skill = FileSkill(
        frontmatter=fm,
        content="# Analytics\nYou help analyse data.",
        path="/skills/analytics",
        scripts=[script],
    )

    content = await skill.get_content()
    print(content)
    # Includes appended:
    # <scripts>
    #   <script name="analyze">
    #     <description>Analyse data</description>
    #     <parameters_schema>{"type":"object","properties":{"query":{...}},...}</parameters_schema>
    #   </script>
    # </scripts>

asyncio.run(main())
```

### Example 3 — Case-insensitive resource lookup

```python
import asyncio
import warnings
from agent_framework import FileSkill, SkillFrontmatter, InlineSkillResource
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def main():
    fm = SkillFrontmatter(name="my-skill")
    res = InlineSkillResource(name="Schema", content="id INT, name TEXT")
    skill = FileSkill(
        frontmatter=fm,
        content="# My Skill",
        path="/skills/my-skill",
        resources=[res],
    )

    found = await skill.get_resource("schema")   # lowercase — still found
    assert found is res
    missing = await skill.get_resource("other")
    assert missing is None

asyncio.run(main())
```

---

## 3 · `InlineSkillResource` + `InlineSkillScript`

**Module:** `agent_framework._skills`

These two classes are the **code-defined** counterparts to `FileSkillScript`. Use them when
you want a skill resource or script backed by a Python callable rather than a file on disk.

### Class signatures

```python
class InlineSkillResource(SkillResource):
    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        content: str | None = None,       # static string content
        function: Callable[..., Any] | None = None,  # dynamic callable
        # Exactly one of content or function must be provided
    ) -> None: ...

    async def read(self, **kwargs: Any) -> Any:
        # Returns content directly, or calls function (awaiting if async)
        ...


class InlineSkillScript(SkillScript):
    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        function: Callable[..., Any],   # required
    ) -> None: ...

    @property
    def parameters_schema(self) -> dict[str, Any] | None:
        # Lazily generated from function's signature via FunctionTool introspection
        # Returns None for functions with no introspectable parameters
        ...

    async def run(
        self,
        skill: Skill,
        args: dict[str, Any] | list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Raises TypeError if args is a list (list form only for file-based scripts)
        ...
```

### Key points

**`InlineSkillResource`:**
- **Mutual exclusivity** — exactly one of `content` or `function` must be provided. Providing neither or both raises `ValueError`.
- The `_accepts_kwargs` flag is precomputed at construction time by inspecting the function signature. This avoids repeated `inspect.signature()` calls on every `read()`.
- `function` may be sync or async — the framework awaits it automatically.

**`InlineSkillScript`:**
- `parameters_schema` is **lazily generated** on first access by creating a temporary `FunctionTool` and calling `.parameters()`. If the function has no parameters or none with introspectable types it returns `None`.
- `args` must be a `dict` or `None`. Passing a `list[str]` raises `TypeError` because keyword-style binding is the only supported mode for inline scripts.
- Like resources, `function` may be sync or async.

### Example 1 — Static vs callable `InlineSkillResource`

```python
import asyncio
import warnings
from agent_framework import InlineSkillResource
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def fetch_config() -> str:
    return '{"model": "gpt-4.1", "max_tokens": 4096}'

async def main():
    # Static resource
    static_res = InlineSkillResource(name="readme", content="This skill does X.")
    print(await static_res.read())   # "This skill does X."

    # Callable resource (async)
    dynamic_res = InlineSkillResource(name="config", function=fetch_config)
    print(await dynamic_res.read())  # '{"model": "gpt-4.1", "max_tokens": 4096}'

    # Callable with kwargs forwarding
    def schema_for_table(table: str = "users") -> str:
        schemas = {"users": "id INT, email TEXT", "orders": "id INT, amount DECIMAL"}
        return schemas.get(table, "unknown")

    schema_res = InlineSkillResource(name="table-schema", function=schema_for_table)
    # kwargs are only forwarded if the function declares **kwargs
    print(await schema_res.read())   # "id INT, email TEXT" (uses default)

asyncio.run(main())
```

### Example 2 — `InlineSkillScript` with auto-generated schema

```python
import asyncio
import warnings
from agent_framework import InlineSkillScript, SkillFrontmatter, InlineSkill
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def summarise(text: str, max_words: int = 50) -> str:
    words = text.split()[:max_words]
    return " ".join(words) + ("..." if len(text.split()) > max_words else "")

async def main():
    script = InlineSkillScript(name="summarise", description="Summarise text", function=summarise)

    # Schema is generated lazily from the function signature
    schema = script.parameters_schema
    print(schema)
    # {"type": "object", "properties": {"text": {...}, "max_words": {...}}, "required": ["text"]}

    # Run with keyword args
    skill_content = "Skill instructions here."
    fm = SkillFrontmatter(name="demo")
    skill = InlineSkill(frontmatter=fm, instructions=skill_content)
    result = await script.run(skill, args={"text": "Hello world this is a test sentence with many words."})
    print(result)  # "Hello world this is a test sentence with many words."

asyncio.run(main())
```

### Example 3 — Composing a skill with resources and scripts

```python
import asyncio
import warnings
from agent_framework import InlineSkill, InlineSkillResource, InlineSkillScript, SkillFrontmatter
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

DATABASE_SCHEMA = "users(id INT, email TEXT), orders(id INT, user_id INT, amount DECIMAL)"

async def query_db(sql: str) -> str:
    # Simulated query execution
    return f"Result of: {sql}"

async def main():
    fm = SkillFrontmatter(name="db-assistant", description="Database query skill")
    skill = InlineSkill(
        frontmatter=fm,
        instructions="You help users query the database. Use the schema resource before writing SQL.",
        resources=[
            InlineSkillResource(name="schema", content=DATABASE_SCHEMA),
        ],
        scripts=[
            InlineSkillScript(name="query", description="Execute a SQL query", function=query_db),
        ],
    )

    print(skill.frontmatter.name)                          # db-assistant
    resource = await skill.get_resource("schema")
    print(await resource.read())                           # users(id INT, ...
    script = await skill.get_script("query")
    result = await script.run(skill, args={"sql": "SELECT * FROM users LIMIT 10"})
    print(result)                                          # Result of: SELECT...

asyncio.run(main())
```

---

## 4 · `FileSkillScript` + `SkillScriptRunner`

**Module:** `agent_framework._skills`

`FileSkillScript` is the file-path-backed counterpart to `InlineSkillScript`. Where inline
scripts run a Python callable in-process, file scripts delegate execution to a
`SkillScriptRunner` — a `@runtime_checkable` protocol that can be any callable matching
the required signature.

### Class signatures

```python
class FileSkillScript(SkillScript):
    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        full_path: str,            # absolute path; raises ValueError if relative or empty
        runner: SkillScriptRunner | None = None,  # required for execution
    ) -> None: ...

    @property
    def parameters_schema(self) -> dict[str, Any] | None:
        # Always returns {"type": "array", "items": {"type": "string"}}
        # Signals to the LLM to pass CLI-style positional arguments as a JSON string array
        ...

    async def run(
        self,
        skill: Skill,
        args: dict[str, Any] | list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Raises TypeError if skill is not a FileSkill
        # Raises ValueError if runner is None
        # Delegates to runner(skill, self, args); awaits if runner returns a coroutine
        ...


@runtime_checkable
class SkillScriptRunner(Protocol):
    def __call__(
        self,
        skill: FileSkill,
        script: FileSkillScript,
        args: dict[str, Any] | list[str] | None = None,
    ) -> Any: ...
```

### Key points

- `full_path` must be an **absolute path**. Providing a relative path or empty string raises `ValueError` at construction time — this prevents silent path-resolution bugs.
- `parameters_schema` always returns the **fixed CLI array schema**: `{"type": "array", "items": {"type": "string"}}`. This tells the LLM that file scripts accept positional string arguments like command-line tools, not named keyword arguments.
- `run()` performs two type checks before delegating: it verifies the `skill` is a `FileSkill` and that a `runner` was provided. Both checks produce descriptive `TypeError`/`ValueError` messages.
- `SkillScriptRunner` is a `@runtime_checkable` `Protocol`. Any sync or async callable with the matching signature satisfies it — you can `isinstance(fn, SkillScriptRunner)` to verify.
- Since the runner can return a coroutine, `run()` awaits it when `inspect.isawaitable(result)` is `True`.

### Example 1 — Simple subprocess runner

```python
import asyncio
import subprocess
import warnings
from agent_framework import FileSkill, FileSkillScript, SkillFrontmatter
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

def subprocess_runner(skill: FileSkill, script: FileSkillScript, args=None) -> str:
    """Run the script file as a subprocess and return stdout."""
    cmd = ["python3", script.full_path] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {result.stderr}")
    return result.stdout.strip()

async def main():
    fm = SkillFrontmatter(name="data-pipeline")
    script = FileSkillScript(
        name="transform.py",
        description="Transform data file",
        full_path="/skills/data-pipeline/scripts/transform.py",
        runner=subprocess_runner,
    )

    # Check parameters schema
    print(script.parameters_schema)
    # {"type": "array", "items": {"type": "string"}}
    # The LLM will pass: ["--input", "data.csv", "--output", "out.json"]

asyncio.run(main())
```

### Example 2 — Async runner with sandboxed execution

```python
import asyncio
import warnings
from agent_framework import FileSkill, FileSkillScript, SkillFrontmatter
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def sandboxed_runner(skill: FileSkill, script: FileSkillScript, args=None) -> str:
    """Async runner that executes in a sandboxed environment."""
    # Simulate async sandboxed execution
    await asyncio.sleep(0)  # yield to event loop
    script_name = script.name
    skill_name = skill.frontmatter.name
    return f"[SANDBOX] Executed {script_name} from {skill_name} with args: {args}"

async def main():
    fm = SkillFrontmatter(name="secure-skill")
    script = FileSkillScript(
        name="run.py",
        full_path="/skills/secure-skill/scripts/run.py",
        runner=sandboxed_runner,   # async runner — awaited automatically
    )

    fm2 = SkillFrontmatter(name="secure-skill")
    skill = FileSkill(
        frontmatter=fm2,
        content="# Secure Skill",
        path="/skills/secure-skill",
        scripts=[script],
    )

    result = await script.run(skill, args=["--mode", "strict", "--input", "file.txt"])
    print(result)
    # [SANDBOX] Executed run.py from secure-skill with args: ['--mode', 'strict', '--input', 'file.txt']

asyncio.run(main())
```

### Example 3 — Verifying a callable satisfies `SkillScriptRunner`

```python
import warnings
from agent_framework import SkillScriptRunner
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

def my_runner(skill, script, args=None):
    return "done"

class CallableRunner:
    def __call__(self, skill, script, args=None):
        return "done"

print(isinstance(my_runner, SkillScriptRunner))         # True — plain function
print(isinstance(CallableRunner(), SkillScriptRunner))  # True — callable class
```

---

## 5 · `SupportsAgentRun`

**Module:** `agent_framework._agents`

`SupportsAgentRun` is a `@runtime_checkable` structural protocol that defines **the minimal
interface any agent must expose** to participate in workflows, orchestration, and A2A. You
never need to inherit from it — any class with the right attributes and methods satisfies
it automatically.

### Interface summary

```python
@runtime_checkable
class SupportsAgentRun(Protocol):
    id: str
    name: str | None
    description: str | None

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]]: ...

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def create_session(self, *, session_id: str | None = None) -> AgentSession: ...
    def get_session(self, service_session_id: str, *, session_id: str | None = None) -> AgentSession: ...
```

### Key points

- `WorkflowBuilder.add_edge()`, `add_chain()`, `add_fan_out_edges()` etc. all accept `SupportsAgentRun` alongside `Executor`. Any conforming agent can be dropped into a workflow graph.
- The protocol is `@runtime_checkable` — you can guard with `isinstance(obj, SupportsAgentRun)` before adding an agent to a workflow or passing it to `A2AExecutor`.
- All three built-in agent classes (`Agent`, `RawAgent`, `WorkflowAgent`) satisfy this protocol. Third-party agents from other frameworks can also participate if they expose these attributes and methods.
- `function_invocation_kwargs` allows per-call overrides of tool invocation configuration without modifying the agent constructor.
- `client_kwargs` are forwarded verbatim to the underlying chat client (e.g. `{"max_tokens": 1000}` for token capping).

### Example 1 — Minimal custom agent without inheriting framework classes

```python
import asyncio
from agent_framework import AgentResponse, AgentResponseUpdate, AgentSession, SupportsAgentRun

class EchoAgent:
    """A fully custom agent that echoes input — no framework inheritance needed."""

    def __init__(self, name: str):
        self.id = f"echo-{name}"
        self.name = name
        self.description = "Echoes input back"

    async def run(self, messages=None, *, stream=False, session=None, **kwargs):
        text = str(messages) if messages else "echo"
        if stream:
            async def _stream():
                yield AgentResponseUpdate(content=text)
            return _stream()
        return AgentResponse(messages=[], text=text)

    def create_session(self, *, session_id=None):
        return AgentSession(session_id=session_id)

    def get_session(self, service_session_id, *, session_id=None):
        return AgentSession(service_session_id=service_session_id, session_id=session_id)

async def main():
    agent = EchoAgent("my-echo")

    # Verify protocol compliance
    assert isinstance(agent, SupportsAgentRun)

    response = await agent.run("Hello!")
    print(response.text)   # "Hello!"

asyncio.run(main())
```

### Example 2 — Using a custom agent in a workflow

```python
import asyncio
from agent_framework import WorkflowBuilder, Agent, AgentResponse, AgentSession, SupportsAgentRun
from agent_framework.openai import OpenAIChatClient

class SentimentAgent:
    id = "sentiment"
    name = "Sentiment"
    description = "Classifies sentiment"

    async def run(self, messages=None, *, stream=False, session=None, **kwargs):
        text = str(messages or "")
        sentiment = "positive" if "good" in text.lower() else "negative"
        return AgentResponse(messages=[], text=sentiment)

    def create_session(self, *, session_id=None):
        return AgentSession(session_id=session_id)

    def get_session(self, sid, *, session_id=None):
        return AgentSession(service_session_id=sid, session_id=session_id)

async def main():
    llm_agent = Agent(client=OpenAIChatClient(), instructions="Summarize the text.")
    sentiment = SentimentAgent()

    assert isinstance(sentiment, SupportsAgentRun)

    # Both can be used in the same workflow graph
    workflow = (
        WorkflowBuilder()
        .add_chain([llm_agent, sentiment])
        .build()
    )
    print("Workflow built with custom agent")

asyncio.run(main())
```

### Example 3 — Protocol guard before workflow registration

```python
from agent_framework import SupportsAgentRun, WorkflowBuilder

def safe_add_to_workflow(builder: WorkflowBuilder, agents: list) -> WorkflowBuilder:
    """Only add agents that satisfy SupportsAgentRun; log others."""
    for agent in agents:
        if isinstance(agent, SupportsAgentRun):
            builder.add_chain([agent])
        else:
            print(f"Skipping {agent}: does not satisfy SupportsAgentRun")
    return builder
```

---

## 6 · `RunnerContext`

**Module:** `agent_framework._workflows._runner_context`

`RunnerContext` is the `@runtime_checkable` protocol that the **workflow execution engine**
(`Runner`) uses to communicate with its execution context. `InProcRunnerContext` (covered in
[Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/))
is the built-in implementation. Custom implementations can be used for testing, tracing, or
embedding workflows in non-standard environments.

### Interface summary (key methods)

```python
@runtime_checkable
class RunnerContext(Protocol):
    # Messaging
    async def send_message(self, message: WorkflowMessage) -> None: ...
    async def drain_messages(self) -> dict[str, list[WorkflowMessage]]: ...
    async def has_messages(self) -> bool: ...

    # Events
    async def add_event(self, event: WorkflowEvent) -> None: ...
    async def drain_events(self) -> list[WorkflowEvent]: ...
    async def has_events(self) -> bool: ...
    async def next_event(self) -> WorkflowEvent: ...

    # Streaming mode
    def set_streaming(self, streaming: bool) -> None: ...
    def is_streaming(self) -> bool: ...

    # Checkpointing (optional — has_checkpointing() gates access)
    def has_checkpointing(self) -> bool: ...
    def set_runtime_checkpoint_storage(self, storage: CheckpointStorage) -> None: ...
    def clear_runtime_checkpoint_storage(self) -> None: ...
    async def create_checkpoint(self, workflow_name, graph_signature_hash, state,
                                 previous_checkpoint_id, iteration_count, metadata=None) -> CheckpointID: ...
    async def load_checkpoint(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint | None: ...
    async def apply_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None: ...

    # HITL
    async def add_request_info_event(self, event: WorkflowEvent[Any]) -> None: ...
    async def send_request_info_response(self, request_id: str, response: Any) -> None: ...
    async def get_pending_request_info_events(self) -> dict[str, WorkflowEvent[Any]]: ...

    # Lifecycle
    def reset_for_new_run(self) -> None: ...

    # Yield output classification
    def set_yield_output_classifier(self, classifier: YieldOutputClassifier) -> None: ...
    def classify_yielded_output(self, executor_id: str) -> YieldOutputEventType | None: ...
```

### Key points

- **Messaging** — the runner routes `WorkflowMessage` objects between executors via `send_message()`/`drain_messages()`. The dict returned by `drain_messages()` maps executor IDs to message lists; the runner delivers each batch to the appropriate executor.
- **Events** — workflow events (including `request_info` HITL events and output events) flow through `add_event()`/`drain_events()`. The caller iterates `next_event()` to consume a workflow's output stream.
- **Checkpointing is guarded** — always call `has_checkpointing()` before `create_checkpoint()` or `load_checkpoint()`. Calling checkpoint methods without storage configured raises `WorkflowCheckpointException`.
- **HITL** — `add_request_info_event()` registers a pending human input request; `send_request_info_response()` resolves it; `get_pending_request_info_events()` returns all unresolved requests. The HITL mechanism is the same across all `RunnerContext` implementations.
- **Streaming mode** — `set_streaming(True)` switches agents to streaming mode before the workflow run starts. `Workflow.run()` calls this based on the `stream=True` argument you pass.

### Example 1 — Minimal test RunnerContext stub

```python
import asyncio
from collections import defaultdict
from agent_framework import WorkflowEvent, WorkflowMessage

class TestRunnerContext:
    """Minimal in-memory RunnerContext for unit testing workflow executors."""

    def __init__(self):
        self._messages: dict[str, list] = defaultdict(list)
        self._events: list = []
        self._streaming = False

    async def send_message(self, message: WorkflowMessage) -> None:
        self._messages[message.target_executor_id].append(message)

    async def drain_messages(self) -> dict:
        result = dict(self._messages)
        self._messages.clear()
        return result

    async def has_messages(self) -> bool:
        return bool(self._messages)

    async def add_event(self, event: WorkflowEvent) -> None:
        self._events.append(event)

    async def drain_events(self) -> list:
        events = list(self._events)
        self._events.clear()
        return events

    async def has_events(self) -> bool:
        return bool(self._events)

    async def next_event(self) -> WorkflowEvent:
        while not self._events:
            await asyncio.sleep(0.01)
        return self._events.pop(0)

    def has_checkpointing(self) -> bool:
        return False

    def set_runtime_checkpoint_storage(self, storage) -> None:
        pass

    def clear_runtime_checkpoint_storage(self) -> None:
        pass

    def reset_for_new_run(self) -> None:
        self._messages.clear()
        self._events.clear()

    def set_streaming(self, streaming: bool) -> None:
        self._streaming = streaming

    def is_streaming(self) -> bool:
        return self._streaming

    async def add_request_info_event(self, event) -> None:
        self._events.append(event)

    async def send_request_info_response(self, request_id: str, response) -> None:
        pass

    async def get_pending_request_info_events(self) -> dict:
        return {}

    def set_yield_output_classifier(self, classifier) -> None:
        pass

    def classify_yielded_output(self, executor_id: str):
        return None
```

### Example 2 — Checking checkpointing before creating a checkpoint

```python
import asyncio
from agent_framework import InMemoryCheckpointStorage, InProcRunnerContext

async def main():
    ctx = InProcRunnerContext()
    print(ctx.has_checkpointing())   # False — no storage configured

    # Set storage at runtime (overrides build-time config for this run only)
    storage = InMemoryCheckpointStorage()
    ctx.set_runtime_checkpoint_storage(storage)
    print(ctx.has_checkpointing())   # True

    # After this run, clear the override
    ctx.clear_runtime_checkpoint_storage()
    print(ctx.has_checkpointing())   # False again

asyncio.run(main())
```

---

## 7 · `SwitchCaseEdgeGroupCase` + `SwitchCaseEdgeGroupDefault`

**Module:** `agent_framework._workflows._edge`

These two classes are the **serialisable** counterparts to the runtime `Case` and `Default`
objects covered in [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/).
`Case`/`Default` carry live callables; `SwitchCaseEdgeGroupCase`/`SwitchCaseEdgeGroupDefault`
carry only metadata — a target executor ID and a human-readable condition name — and can be
safely persisted to disk or sent over the wire.

### Class signatures

```python
@dataclass(init=False)
class SwitchCaseEdgeGroupCase(DictConvertible):
    target_id: str
    condition_name: str | None
    type: str           # always "Case"

    def __init__(
        self,
        condition: Callable[[Any], bool] | None,  # None → missing-callable placeholder
        target_id: str,                            # required; raises ValueError if empty
        *,
        condition_name: str | None = None,
    ) -> None: ...

    @property
    def condition(self) -> Callable[[Any], bool]:
        # Returns the live callable, or a placeholder that raises RuntimeError when invoked
        ...

    def to_dict(self) -> dict[str, Any]:
        # {"target_id": ..., "type": "Case", "condition_name": ...}
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwitchCaseEdgeGroupCase:
        # Restores from dict; condition is set to a missing-callable placeholder
        ...


@dataclass(init=False)
class SwitchCaseEdgeGroupDefault(DictConvertible):
    target_id: str
    type: str           # always "Default"

    def __init__(self, target_id: str) -> None: ...  # raises ValueError if empty

    def to_dict(self) -> dict[str, Any]:
        # {"target_id": ..., "type": "Default"}
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwitchCaseEdgeGroupDefault: ...
```

### Key points

- **Vs `Case`/`Default`** — `Case` and `Default` (runtime objects) hold live Python callables that cannot be serialised. `SwitchCaseEdgeGroupCase` and `SwitchCaseEdgeGroupDefault` strip the callable and keep only the target ID and name. Use the serialisable variants for:
  - Storing workflow topology in a database or config file
  - Inspecting what routing branches exist without running the workflow
  - Checkpoint metadata (the `WorkflowCheckpoint` stores the serialised graph)
- **Missing callable placeholder** — `SwitchCaseEdgeGroupCase.from_dict()` reconstructs a case with `condition=None`. Accessing `.condition` returns a proxy that raises `RuntimeError` on invocation. This makes missing registrations immediately visible rather than silently routing to the wrong branch.
- `condition_name` is **auto-extracted** from the callable's `__name__` attribute when a real callable is provided. Lambda functions get `"<lambda>"`; named functions get their actual name.

### Example 1 — Creating and serialising routing cases

```python
from agent_framework import SwitchCaseEdgeGroupCase, SwitchCaseEdgeGroupDefault

def is_error(payload) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "error"

# Create case with live callable
case = SwitchCaseEdgeGroupCase(is_error, target_id="error_handler")
print(case.condition_name)    # "is_error"
print(case.target_id)         # "error_handler"
print(case.type)              # "Case"

# Serialise
data = case.to_dict()
print(data)
# {"target_id": "error_handler", "type": "Case", "condition_name": "is_error"}

# Deserialise — callable is replaced with a placeholder
restored = SwitchCaseEdgeGroupCase.from_dict(data)
print(restored.target_id)          # "error_handler"
print(restored.condition_name)     # "is_error"

# The condition property returns a placeholder — do not invoke without re-registering
try:
    restored.condition({"status": "error"})
except RuntimeError as e:
    print(f"Expected: {e}")

# Default branch
default = SwitchCaseEdgeGroupDefault(target_id="default_handler")
print(default.to_dict())
# {"target_id": "default_handler", "type": "Default"}
```

### Example 2 — Inspecting serialised workflow topology

```python
import json
from agent_framework import SwitchCaseEdgeGroupCase, SwitchCaseEdgeGroupDefault

# Imagine this dict was loaded from a database or YAML config
stored_topology = {
    "switch_cases": [
        {"target_id": "csv_handler",  "type": "Case",    "condition_name": "is_csv"},
        {"target_id": "json_handler", "type": "Case",    "condition_name": "is_json"},
        {"target_id": "fallback",     "type": "Default"},
    ]
}

def load_routing(topology: dict):
    for entry in topology["switch_cases"]:
        if entry["type"] == "Case":
            case = SwitchCaseEdgeGroupCase.from_dict(entry)
            print(f"Case: {case.condition_name!r} → {case.target_id!r}")
        elif entry["type"] == "Default":
            default = SwitchCaseEdgeGroupDefault.from_dict(entry)
            print(f"Default → {default.target_id!r}")

load_routing(stored_topology)
# Case: 'is_csv' → 'csv_handler'
# Case: 'is_json' → 'json_handler'
# Default → 'fallback'
```

### Example 3 — Re-registering conditions after deserialization

```python
from agent_framework import SwitchCaseEdgeGroupCase, Case, Default, WorkflowBuilder, Agent
from agent_framework.openai import OpenAIChatClient

def is_csv(payload) -> bool:
    return str(payload).strip().startswith("id,")

def is_json(payload) -> bool:
    return str(payload).strip().startswith("{")

def reload_condition(serialised_case: SwitchCaseEdgeGroupCase, registry: dict) -> Case:
    """Restore a serialised case with its live condition callable."""
    condition_fn = registry.get(serialised_case.condition_name)
    if condition_fn is None:
        raise RuntimeError(f"Condition {serialised_case.condition_name!r} not found in registry.")
    return Case(condition_fn, target_id=serialised_case.target_id)

registry = {"is_csv": is_csv, "is_json": is_json}
stored = {"target_id": "csv_handler", "type": "Case", "condition_name": "is_csv"}
case_meta = SwitchCaseEdgeGroupCase.from_dict(stored)
live_case = reload_condition(case_meta, registry)
print(live_case.condition({"type": "not-csv"}))   # False
```

---

## 8 · `ValidationTypeEnum` + `WorkflowValidationError` + subclasses

**Module:** `agent_framework._workflows._validation`

The workflow graph validator raises structured validation errors before `WorkflowBuilder.build()` completes. Each error carries a `ValidationTypeEnum` tag that identifies the type of constraint that was violated.

### Enum values

```python
class ValidationTypeEnum(Enum):
    EDGE_DUPLICATION           = "EDGE_DUPLICATION"
    EXECUTOR_DUPLICATION       = "EXECUTOR_DUPLICATION"
    TYPE_COMPATIBILITY         = "TYPE_COMPATIBILITY"
    GRAPH_CONNECTIVITY         = "GRAPH_CONNECTIVITY"
    HANDLER_OUTPUT_ANNOTATION  = "HANDLER_OUTPUT_ANNOTATION"
    OUTPUT_VALIDATION          = "OUTPUT_VALIDATION"
```

### Exception hierarchy

```python
class WorkflowValidationError(WorkflowException):
    def __init__(self, message: str, validation_type: ValidationTypeEnum) -> None: ...
    # __str__: "[VALIDATION_TYPE_VALUE] message"

class EdgeDuplicationError(WorkflowValidationError):
    # validation_type = ValidationTypeEnum.EDGE_DUPLICATION
    def __init__(self, edge_id: str) -> None: ...
    # message: "Duplicate edge detected: {edge_id}. Each edge in the workflow must be unique."

class TypeCompatibilityError(WorkflowValidationError):
    # validation_type = ValidationTypeEnum.TYPE_COMPATIBILITY
    def __init__(
        self,
        source_executor_id: str,
        target_executor_id: str,
        source_types: list[type],
        target_types: list[type],
    ) -> None: ...
    # message: "Type incompatibility between executors '...' -> '...'. Source outputs ... but target handles ..."

class GraphConnectivityError(WorkflowValidationError):
    # validation_type = ValidationTypeEnum.GRAPH_CONNECTIVITY
    def __init__(self, message: str) -> None: ...
```

### Key points

- `WorkflowValidationError.__str__()` prefixes the message with the enum value in brackets, e.g. `"[TYPE_COMPATIBILITY] Type incompatibility ..."`. This makes log output self-documenting.
- `TypeCompatibilityError` carries the full source and target type lists as attributes (`source_types`, `target_types`, `source_executor_id`, `target_executor_id`) — useful for programmatic error reporting.
- `EdgeDuplicationError` carries `edge_id` — the string key of the duplicate edge.
- Catching `WorkflowValidationError` catches all validation subtypes. Catching specific subclasses allows targeted recovery logic.
- **`HANDLER_OUTPUT_ANNOTATION`** and **`OUTPUT_VALIDATION`** currently map to `WorkflowValidationError` (no dedicated subclass) — they use the base class with the appropriate `validation_type`.

### Example 1 — Catching and inspecting validation errors

```python
import asyncio
from agent_framework import WorkflowBuilder, Agent, WorkflowValidationError, TypeCompatibilityError
from agent_framework._workflows._validation import ValidationTypeEnum
from agent_framework.openai import OpenAIChatClient

async def main():
    agent_a = Agent(client=OpenAIChatClient(), instructions="Step A")
    agent_b = Agent(client=OpenAIChatClient(), instructions="Step B")

    try:
        # This will fail validation if type constraints are violated
        workflow = (
            WorkflowBuilder()
            .add_edge(agent_a, agent_b)
            .build()
        )
    except TypeCompatibilityError as e:
        print(f"Type error: {e}")
        print(f"Source executor: {e.source_executor_id}")
        print(f"Target executor: {e.target_executor_id}")
        print(f"Validation type: {e.validation_type}")  # ValidationTypeEnum.TYPE_COMPATIBILITY
    except WorkflowValidationError as e:
        print(f"Validation failed [{e.validation_type.value}]: {e.message}")

asyncio.run(main())
```

### Example 2 — Matching validation type in error handling

```python
from agent_framework import WorkflowValidationError
from agent_framework._workflows._validation import ValidationTypeEnum

def handle_build_error(exc: Exception) -> None:
    if not isinstance(exc, WorkflowValidationError):
        raise exc

    match exc.validation_type:
        case ValidationTypeEnum.EDGE_DUPLICATION:
            print("Fix: remove the duplicate edge before building")
        case ValidationTypeEnum.EXECUTOR_DUPLICATION:
            print("Fix: each agent/executor must appear only once")
        case ValidationTypeEnum.TYPE_COMPATIBILITY:
            print("Fix: check output type of source matches input type of target")
        case ValidationTypeEnum.GRAPH_CONNECTIVITY:
            print("Fix: ensure all nodes are reachable from the start node")
        case ValidationTypeEnum.HANDLER_OUTPUT_ANNOTATION:
            print("Fix: add @handler return type annotation")
        case ValidationTypeEnum.OUTPUT_VALIDATION:
            print("Fix: verify output_from references a reachable executor")
        case _:
            print(f"Unknown validation type: {exc.validation_type}")
```

### Example 3 — Custom validation reporting

```python
from dataclasses import dataclass
from agent_framework import WorkflowValidationError
from agent_framework._workflows._validation import ValidationTypeEnum, TypeCompatibilityError, EdgeDuplicationError

@dataclass
class ValidationReport:
    errors: list[WorkflowValidationError]

    def summary(self) -> str:
        lines = [f"Workflow validation failed with {len(self.errors)} error(s):"]
        for err in self.errors:
            lines.append(f"  [{err.validation_type.value}] {err.message}")
        return "\n".join(lines)

# Collect errors from a build attempt
errors = []
try:
    from agent_framework import WorkflowBuilder
    wf = WorkflowBuilder().build()  # empty workflow — likely fails connectivity check
except WorkflowValidationError as e:
    errors.append(e)

if errors:
    report = ValidationReport(errors=errors)
    print(report.summary())
```

---

## 9 · `A2AAgent` + `A2AAgentSession` + `A2AExecutor`

**Module:** `agent_framework.a2a`

The A2A (Agent-to-Agent) protocol integration lets the framework **call remote agents** over
HTTP/JSON-RPC (`A2AAgent`) and **expose local agents** as A2A servers (`A2AExecutor`).
`A2AAgentSession` extends `AgentSession` with A2A-specific conversation state.

### Class signatures

```python
class A2AAgent(AgentTelemetryLayer, BaseAgent):
    AGENT_PROVIDER_NAME: Final[str] = "A2A"

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        description: str | None = None,
        agent_card: AgentCard | None = None,  # from a2a.types
        url: str | None = None,               # alternative to agent_card
        client: Client | None = None,         # pre-built A2A client
        http_client: httpx.AsyncClient | None = None,
        auth_interceptor: AuthInterceptor | None = None,
        timeout: float | httpx.Timeout | None = None,
        supported_protocol_bindings: list[str] | None = None,  # default: ["JSONRPC"]
        **kwargs: Any,
    ) -> None: ...
    # Raises ValueError if neither agent_card nor url is provided and client is None


class A2AAgentSession(AgentSession):
    _CONTEXT_ID_KEY = "a2a_context_id"
    _TASK_ID_KEY    = "a2a_task_id"
    _TASK_STATE_KEY = "a2a_task_state"

    def __init__(
        self,
        *,
        context_id: str | None = None,   # A2A conversation context ID
        task_id: str | None = None,       # most recent A2A task ID
        task_state: TaskState | None = None,  # from a2a.types (protobuf enum)
    ) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AAgentSession: ...


class A2AExecutor(AgentExecutor):
    def __init__(
        self,
        agent: SupportsAgentRun,
        stream: bool = False,
        run_kwargs: Mapping[str, Any] | None = None,
    ) -> None: ...
```

### Key points

**`A2AAgent`:**
- Accepts exactly one of: `url` (creates a minimal `AgentCard` internally), `agent_card` (full card for transport negotiation), or `client` (pre-built `a2a.client.Client`).
- Creates **two clients** internally: a streaming client (SSE transport, for `stream=True`) and a non-streaming client (single request/response, for `stream=False`).
- `auth_interceptor` is applied to both clients. Use it for API key, OAuth Bearer, or custom auth schemes.
- `timeout` may be a `float` (applied to all four httpx timeout components) or an `httpx.Timeout` object. Default is `10s` connect, `60s` read, `10s` write, `5s` pool.
- `supported_protocol_bindings` defaults to `["JSONRPC"]`. The A2A spec treats binding identifiers as open strings — you can add custom values for non-standard transports.

**`A2AAgentSession`:**
- Extends `AgentSession` with three extra state fields persisted via private class-level key constants.
- `task_state` is a protobuf `TaskState` enum. Distinguishing `input-required` from `completed` states lets callers decide whether to send a follow-up or start a fresh task.
- `to_dict()`/`from_dict()` are fully round-trip safe — the A2A state fields are added to (and restored from) the base `AgentSession.to_dict()` output.

**`A2AExecutor`:**
- Bridges a local `SupportsAgentRun` agent with the A2A server stack from the `a2a-sdk`.
- `run_kwargs` are forwarded verbatim to `agent.run(...)` — use them for `client_kwargs` (e.g. `{"max_tokens": 500}`) or `function_invocation_kwargs`.
- `stream=True` switches the agent to streaming mode; the executor wraps streaming updates as A2A SSE events.

### Example 1 — Calling a remote A2A agent by URL

```python
import asyncio
from agent_framework.a2a import A2AAgent

async def main():
    # Connect to any A2A-compliant agent by URL
    remote = A2AAgent(
        name="Remote Analyst",
        url="http://analytics-service:9000/",
        timeout=30.0,
    )

    response = await remote.run("Analyse Q1 sales data and summarise key trends.")
    print(response.text)

    # Streaming
    stream = remote.run("Describe the top 5 products.", stream=True)
    async for update in stream:
        print(update.content, end="", flush=True)
    final = await stream.get_final_response()
    print(f"\nTokens used: {final.usage}")

asyncio.run(main())
```

### Example 2 — Multi-turn conversation with `A2AAgentSession`

```python
import asyncio
from agent_framework.a2a import A2AAgent, A2AAgentSession

async def main():
    agent = A2AAgent(url="http://remote-agent:9000/")

    # Start a new session — context_id will be populated after the first response
    session = A2AAgentSession()

    response1 = await agent.run("What is the weather in London?", session=session)
    print(response1.text)
    print(f"Context ID: {session.context_id}")   # populated by the agent
    print(f"Task ID: {session.task_id}")

    # Continue the same conversation using the session
    response2 = await agent.run("And in Paris?", session=session)
    print(response2.text)

    # Persist and restore session
    stored = session.to_dict()
    # ... save stored somewhere ...
    restored = A2AAgentSession.from_dict(stored)
    print(restored.context_id)  # same as session.context_id

asyncio.run(main())
```

### Example 3 — Exposing a local agent as an A2A server

```python
import asyncio
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_jsonrpc_routes, create_agent_card_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from starlette.applications import Starlette
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.a2a import A2AExecutor

async def main():
    # Local agent to expose
    local_agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant that answers questions concisely.",
    )

    agent_card = AgentCard(
        name="Helpful Assistant",
        description="Answers questions via the A2A protocol.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(url="http://0.0.0.0:9000/", protocol_binding="JSONRPC"),
        ],
        skills=[],
    )

    executor = A2AExecutor(
        agent=local_agent,
        stream=True,
        run_kwargs={"client_kwargs": {"max_tokens": 2000}},
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card),
            *create_jsonrpc_routes(request_handler, "/"),
        ],
    )

    # Run with: uvicorn script:app --port 9000
    print("A2A server ready. Run: uvicorn <module>:app --port 9000")

asyncio.run(main())
```

### Example 4 — Authenticating a remote agent call

```python
import asyncio
import httpx
from agent_framework.a2a import A2AAgent

class BearerTokenInterceptor:
    """Simple Bearer token auth interceptor."""

    def __init__(self, token: str):
        self._token = token

    async def __call__(self, request: httpx.Request) -> httpx.Request:
        request.headers["Authorization"] = f"Bearer {self._token}"
        return request

async def main():
    interceptor = BearerTokenInterceptor(token="my-api-key-12345")
    agent = A2AAgent(
        url="https://secure-agent.example.com/",
        auth_interceptor=interceptor,
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    )

    response = await agent.run("What is 2 + 2?")
    print(response.text)

asyncio.run(main())
```

---

## 10 · `WorkflowRunnerException` + `WorkflowCheckpointException` + `MiddlewareException`

**Module:** `agent_framework.exceptions`

These three exception classes fill specific positions in the framework's exception hierarchy.
The main exception tree (covered in [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/)) provides the base classes; this volume covers the three leaf-level classes that production code is most likely to encounter in specialised error paths.

### Complete exception hierarchy

```
AgentFrameworkException
├── MiddlewareException          ← raised during middleware pipeline execution
├── UserInputRequiredException   ← HITL: agent paused waiting for user input
└── WorkflowException
    ├── WorkflowValidationError  ← graph validation (see § 8)
    ├── WorkflowConvergenceException  ← workflow did not converge in max iterations
    ├── WorkflowCheckpointException  ← NOT covered yet ← checkpoint read/write failure
    └── WorkflowRunnerException      ← NOT covered yet ← runner-level execution error
        └── WorkflowCheckpointException  ← ALSO a WorkflowRunnerException subtype
```

### Class signatures

```python
class MiddlewareException(AgentFrameworkException):
    """Raised when an error occurs during middleware pipeline execution."""
    # No additional attributes beyond AgentFrameworkException.
    # constructor: (message, inner_exception=None, log_level=logging.DEBUG)


class WorkflowRunnerException(WorkflowException):
    """Base exception for workflow runner errors."""
    # No additional attributes.
    # constructor: (message, inner_exception=None, log_level=logging.DEBUG)


class WorkflowCheckpointException(WorkflowRunnerException):
    """Raised for errors related to workflow checkpoints."""
    # Subclass of WorkflowRunnerException — catch either for checkpoint failures.
    # constructor: (message, inner_exception=None, log_level=logging.DEBUG)
```

### Key points

**`MiddlewareException`:**
- Raised when an exception propagates out of a middleware's `on_invoke()` (agent middleware) or corresponding chat/function middleware hook, **and** the framework wraps it in `MiddlewareException`.
- Distinct from `MiddlewareTermination` (a sentinel, not an exception) — `MiddlewareTermination` is raised intentionally to short-circuit the pipeline; `MiddlewareException` signals an unexpected error.
- Catch it to add logging or retry logic at the agent-call boundary without catching broader `AgentFrameworkException` types.

**`WorkflowRunnerException`:**
- The umbrella for errors that occur in the `Runner` (the Pregel superstep engine) rather than in the workflow graph validator.
- Typical causes: message delivery failure, executor crash during a superstep, or internal runner-state corruption.
- Catching `WorkflowRunnerException` also catches `WorkflowCheckpointException` since the latter subclasses it.

**`WorkflowCheckpointException`:**
- Raised when checkpoint I/O fails — a write cannot complete (disk full, permission denied), a read fails (storage unavailable), or a checkpoint with an incompatible topology hash is applied.
- Inherits from **both** `WorkflowRunnerException` **and** `WorkflowException` (via the chain), so it is caught by handlers for either parent class.
- The framework raises this from `RunnerContext.create_checkpoint()`, `load_checkpoint()`, and `apply_checkpoint()`. If `has_checkpointing()` returns `False`, accessing these methods without storage configured raises `WorkflowCheckpointException`.

### Example 1 — Catching checkpoint failures during a workflow run

```python
import asyncio
from agent_framework import Workflow, WorkflowBuilder, Agent
from agent_framework.exceptions import WorkflowCheckpointException, WorkflowRunnerException
from agent_framework.openai import OpenAIChatClient
from agent_framework import FileCheckpointStorage

async def run_workflow_with_checkpointing(workflow: Workflow, prompt: str):
    try:
        result = await workflow.run(
            prompt,
            checkpoint_storage=FileCheckpointStorage("/tmp/checkpoints"),
        )
        return result
    except WorkflowCheckpointException as e:
        # Checkpoint-specific failure — log and continue without checkpointing
        print(f"Checkpoint failed: {e}. Running without persistence.")
        return await workflow.run(prompt)
    except WorkflowRunnerException as e:
        # Broader runner failure — re-raise
        print(f"Runner error: {e}")
        raise

async def main():
    agent = Agent(client=OpenAIChatClient(), instructions="Help the user.")
    workflow = WorkflowBuilder().add_chain([agent]).build()
    result = await run_workflow_with_checkpointing(workflow, "Hello!")
    print(result.output)

asyncio.run(main())
```

### Example 2 — Middleware error isolation

```python
import asyncio
from agent_framework import Agent, AgentMiddleware, AgentContext
from agent_framework.exceptions import MiddlewareException
from agent_framework.openai import OpenAIChatClient

class LoggingMiddleware(AgentMiddleware):
    async def on_invoke(self, context: AgentContext, next_func):
        print(f"[LOG] Invoking agent: {context.agent_id}")
        try:
            return await next_func(context)
        except MiddlewareException as e:
            print(f"[LOG] Middleware error: {e}")
            raise
        except Exception as e:
            # Wrap unexpected errors so they can be distinguished
            raise MiddlewareException(
                f"Unexpected error in {type(self).__name__}: {e}",
                inner_exception=e,
            ) from e

async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        middleware=[LoggingMiddleware()],
    )
    response = await agent.run("What is the capital of France?")
    print(response.text)

asyncio.run(main())
```

### Example 3 — Distinguishing exception types in a workflow supervisor

```python
import asyncio
from agent_framework.exceptions import (
    AgentFrameworkException,
    MiddlewareException,
    WorkflowCheckpointException,
    WorkflowRunnerException,
    WorkflowValidationError,
    WorkflowConvergenceException,
    UserInputRequiredException,
)

async def supervised_run(coro):
    """Run a workflow coroutine and classify any framework exceptions."""
    try:
        return await coro
    except UserInputRequiredException as e:
        print(f"[HITL] Paused — awaiting user input: {e}")
    except WorkflowCheckpointException as e:
        print(f"[CHECKPOINT] Storage failure: {e}. Retry without checkpointing.")
    except WorkflowConvergenceException as e:
        print(f"[CONVERGENCE] Workflow did not converge: {e}. Increase max_iterations.")
    except WorkflowValidationError as e:
        print(f"[VALIDATION] Build-time error [{e.validation_type.value}]: {e.message}")
    except WorkflowRunnerException as e:
        print(f"[RUNNER] Execution error: {e}")
    except MiddlewareException as e:
        print(f"[MIDDLEWARE] Pipeline error: {e}")
    except AgentFrameworkException as e:
        print(f"[FRAMEWORK] Unclassified error: {e}")
    return None
```

---

## Cross-reference table

Classes NOT covered in this volume but closely related:

| Class | Covered in |
|-------|-----------|
| `InlineSkill` | [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) |
| `ClassSkill` + `SkillFrontmatter` | [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) |
| `FileSkillsSource` + `SkillsProvider` | [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) (also Vol. 9) |
| `MCPSkill` + `MCPSkillsSource` | [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) |
| `Case` + `Default` (runtime) | [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) |
| `SwitchCaseEdgeGroup` (builder) | [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) |
| `WorkflowGraphValidator` | [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) |
| `AgentExecutor` + `AgentExecutorRequest/Response` | [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) |
| `InProcRunnerContext` | [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) |
| `AgentSession` + `ContextProvider` | [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) |
| `BaseAgent` | [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) |
| Full exception hierarchy | [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) |
| `MiddlewareTermination` (sentinel) | [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) |

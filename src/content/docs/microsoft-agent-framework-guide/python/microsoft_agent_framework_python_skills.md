---
title: "Microsoft Agent Framework (Python) — Skills"
description: "Progressive-disclosure domain knowledge for agents. InlineSkill / ClassSkill / FileSkillsSource / SkillsProvider with code-defined and file-based skills, composable sources, script runners, and approval gates."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "Skills"
  order: 13
---

Skills are a **progressive-disclosure** knowledge pattern. Instead of stuffing every reference doc and procedure into the system prompt, you advertise skill *names and descriptions* (cheap), let the model decide which to load (`load_skill`), and then fetch resources (`read_skill_resource`) or run scripts (`run_skill_script`) on demand. The total context stays small until the agent actually needs deeper knowledge.

This follows the [Agent Skills specification](https://agentskills.io). Verified against `agent-framework` %%version=agent-framework%% (`agent_framework._skills`, re-checked %%versions-checked%%; first written for 1.6.0). The core skills API graduated from experimental in 1.11.0; MCP-backed skills (`MCPSkillsSource`) are still `ExperimentalFeature.MCP_SKILLS`.

## The primitives

| Class | Role |
|---|---|
| `InlineSkill` | Code-defined skill: name, description, instructions body, zero or more resources and scripts |
| `ClassSkill` | Abstract base class for reusable, distributable skill types — subclass it, override the `instructions` property, and decorate methods with `@ClassSkill.resource` / `@ClassSkill.script` |
| `InlineSkillResource` | Named static or dynamic content the model can fetch via `read_skill_resource` |
| `SkillScript` | Executable code (in-process callable or file on disk) the model can invoke via `run_skill_script` |
| `SkillsProvider` | A `ContextProvider` that advertises skills in the prompt and exposes the three tools |
| `FileSkillsSource` | Discovers skills from filesystem directories (SKILL.md files) |
| `MCPSkillsSource` | Discovers skills an MCP server publishes at `skill://index.json` (experimental) |
| `AggregatingSkillsSource` | Combines multiple `SkillsSource` instances into one |
| `FilteringSkillsSource` | Filters skills from a source by a predicate |
| `DeduplicatingSkillsSource` | Removes duplicate skills (by name) from a source |
| `CachingSkillsSource` | Caches another source's skills, optionally per caller and with a refresh interval |
| `SkillScriptRunner` | Strategy protocol for running file-based scripts (sandbox, subprocess, hosted) |

> **API change in 1.3.0:** `Skill` is now an abstract base class. Use `InlineSkill` for code-defined skills (with `instructions=` instead of `content=`). `SkillsProvider` now takes a positional `source` argument instead of `skills=`/`skill_paths=` keyword args. For file-based skills use `SkillsProvider.from_paths(...)`.
>
> **API change in 1.11.0:** the single `require_script_approval=` flag was replaced by three opt-out flags — `disable_load_skill_approval`, `disable_read_skill_resource_approval` and `disable_run_skill_script_approval` — and all three skill tools now **require approval by default**. The examples below leave the defaults in place, so the run returns approval requests (see [Script approval](#script-approval)) until you approve them or opt out for skills from a trusted source.

## Code-defined skill (`InlineSkill`)

The fastest path — define a skill in Python and register it with `SkillsProvider`:

```python
import asyncio
from agent_framework import Agent, InlineSkill, InlineSkillResource, SkillsProvider, SkillFrontmatter
from agent_framework.openai import OpenAIChatClient


db_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="db-ops",
        description="Query and describe the production PostgreSQL database.",
    ),
    instructions=(
        "When the user asks about data, first call `read_skill_resource` with "
        "'schema' to see the tables, then craft a SELECT query. Never run INSERT/UPDATE/DELETE."
    ),
    resources=[
        InlineSkillResource(
            name="schema",
            description="Current DB schema (compact).",
            content=(
                "users(id, email, created_at)\n"
                "orders(id, user_id, total, created_at)\n"
                "..."
            ),
        ),
    ],
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a data analyst.",
    context_providers=[SkillsProvider(db_skill)],
)


async def main() -> None:
    response = await agent.run("How many orders did customer u-42 place last month?")
    print(response.text)


asyncio.run(main())
```

The agent sees a system-prompt blurb listing `db-ops` plus its description. If it decides the question matches, it calls `load_skill("db-ops")` to pull `instructions`, then `read_skill_resource("db-ops", "schema")` to see the schema — neither is in the initial prompt.

Pass a list of skills as the source:

```python
provider = SkillsProvider([db_skill, stats_skill, tone_skill])
```

## Dynamic resources via `@skill.resource`

Resources can be callables — useful when content changes per request or is expensive to build upfront:

```python
from agent_framework import InlineSkill, InlineSkillResource, SkillFrontmatter

inventory_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="inventory",
        description="Check real-time stock levels.",
    ),
    instructions="Use `read_skill_resource('inventory', 'stock')` to see current stock.",
)


@inventory_skill.resource
async def stock() -> str:
    """Snapshot of current stock across warehouses."""
    rows = await fetch_stock_from_db()
    return "\n".join(f"{r.sku}: {r.qty}" for r in rows)
```

The decorator uses the function name and docstring as defaults. Both sync and async callables work.

### Constructing `InlineSkillResource` directly

Alternatively, pass a callable via the `function=` parameter:

```python
async def fetch_matrix() -> str:
    """Return the pricing matrix."""
    rows = await pricing_db.fetch_matrix()
    return "\n".join(f"{r.sku}: {r.price}" for r in rows)

resource = InlineSkillResource(
    name="matrix",
    description="Current pricing matrix.",
    function=fetch_matrix,
)

pricing_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="pricing",
        description="Pricing matrix lookup.",
    ),
    instructions="Use `read_skill_resource('pricing', 'matrix')` to get prices.",
    resources=[resource],
)
```

### Per-request context via `**kwargs`

Resource callables can declare `**kwargs` to receive runtime data forwarded by `agent.run(..., function_invocation_kwargs={...})`:

```python
from agent_framework import Agent, InlineSkill, SkillsProvider, SkillFrontmatter
from agent_framework.openai import OpenAIChatClient


tenant_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="tenant-pricing",
        description="Fetch pricing matrix for the current tenant.",
    ),
    instructions="Use `read_skill_resource('tenant-pricing', 'matrix')` to see the current matrix.",
)


@tenant_skill.resource
async def matrix(**kwargs) -> str:
    """Return the pricing matrix for the active tenant."""
    tenant_id = kwargs.get("tenant_id", "default")
    rows = await pricing_db.fetch_matrix(tenant_id)
    return "\n".join(f"{r.sku}: {r.price}" for r in rows)


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a pricing assistant.",
    context_providers=[SkillsProvider(tenant_skill)],
)

# tenant_id flows through function_invocation_kwargs into the resource's **kwargs.
await agent.run(
    "What's the SKU price for ACME's PRO plan?",
    function_invocation_kwargs={"tenant_id": "acme"},
)
```

Without `**kwargs` in the signature, the framework calls the resource as `function()` — runtime kwargs are silently dropped. The same rule applies to `SkillScript`.

## Scripts

Skills can bundle executable code. Code-defined scripts run in-process:

```python
from agent_framework import InlineSkill, SkillFrontmatter


stats_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="stats",
        description="Compute summary statistics on numeric data.",
    ),
    instructions="Use `run_skill_script` with name='summary'.",
)


@stats_skill.script
def summary(values: list[float]) -> dict[str, float]:
    """Return mean, min, and max for a list of numbers."""
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
```

Script arguments are inferred from the signature and advertised to the model as JSON schema — the model sends args as a JSON object (`{"values": [1, 2, 3]}`) and the framework routes them to your function.

### Parameterised `@skill.script`

Both `@skill.resource` and `@skill.script` support bare and parameterised forms:

```python
import json
from agent_framework import InlineSkill, SkillFrontmatter

skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="db-ops",
        description="PostgreSQL read-only operations.",
    ),
    instructions="Use load_skill to fetch the schema first, then craft queries.",
)


# Bare — name = "schema", description = the docstring
@skill.resource
def schema() -> str:
    """Compact PostgreSQL schema for the analytics warehouse."""
    with open("schema.sql", encoding="utf-8") as f:
        return f.read()


# Parameterised — override name and description
@skill.resource(name="recent-incidents", description="Last 7 days of pager-duty incidents.")
async def fetch_incidents() -> str:
    rows = await pager_duty.list_incidents(days=7)
    return "\n".join(f"- {r.title} ({r.severity})" for r in rows)


# Bare script — name = "list_tables"
@skill.script
def list_tables() -> str:
    """Return all table names as a JSON list."""
    return json.dumps(db_inspect.tables())


# Parameterised async script
@skill.script(name="run_query", description="Run a SELECT and return up to 100 rows as JSON.")
async def execute_query(sql: str) -> str:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are permitted.")
    rows = await db.fetch(sql, limit=100)
    return json.dumps([dict(r) for r in rows])
```

The decorators return the **original function unchanged**, so you can still call the function directly from tests.

## File-based skills (`FileSkillsSource`)

Store skills on disk and discover them with `SkillsProvider.from_paths()`:

```
skills/
├── contract-reviewer/
│   ├── SKILL.md          # frontmatter: name, description + body = instructions
│   ├── references/
│   │   └── clauses.md    # auto-discovered resource
│   └── scripts/
│       └── validate.py   # auto-discovered script (needs a runner)
└── tone-matcher/
    ├── SKILL.md
    └── voice-guide.md
```

```python
from agent_framework import Agent, SkillsProvider
from agent_framework.openai import OpenAIChatClient

# Recommended shortcut for file-based skills
provider = SkillsProvider.from_paths("./skills")
agent = Agent(client=OpenAIChatClient(), context_providers=[provider])
```

Or compose with `FileSkillsSource` directly for more control:

```python
from agent_framework import SkillsProvider, FileSkillsSource

source = FileSkillsSource(
    skill_paths=["./skills", "./domain-skills"],
    resource_extensions=(".md", ".json"),   # restrict discoverable resource types
    script_extensions=(".py",),
)
provider = SkillsProvider(source)
```

`SKILL.md` uses YAML frontmatter for metadata and Markdown for the body:

```markdown
---
name: contract-reviewer
description: Review SaaS contracts for non-standard clauses.
---
When the user pastes a contract, read `references/clauses.md` first. For each section, flag clauses that deviate from the reference set.
```

**Security.** File-based resource reads are protected against path traversal and symlink escape. Only load skills from trusted sources.

## `InMemorySkillsSource` — hold skills in memory

`InMemorySkillsSource` is the simplest `SkillsSource` implementation: it holds a fixed list of `Skill` objects in memory. Use it when you build skills programmatically and don't need file discovery.

```python
from agent_framework import (
    SkillFrontmatter,
    Agent,
    InlineSkill,
    InMemorySkillsSource,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatClient

skill_a = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="db-ops",
        description="Query and describe the production PostgreSQL database.",
    ),
    instructions="Use read_skill_resource('db-ops', 'schema') to see the tables, then craft SELECT queries only.",
)

skill_b = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="reporting",
        description="Generate structured business reports from query results.",
    ),
    instructions="Format data as Markdown tables with a short executive summary.",
)

# Build the source explicitly — useful when skill objects come from a factory or DI container
source = InMemorySkillsSource([skill_a, skill_b])

# Pass it directly to SkillsProvider
provider = SkillsProvider(source)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a data analyst.",
    context_providers=[provider],
)
```

`InMemorySkillsSource` is equivalent to passing a list directly to `SkillsProvider(source)` — the provider auto-wraps any `Sequence[Skill]` — but the explicit constructor is useful when you need to inspect or mutate the source before passing it downstream (e.g. wrap it in a `DeduplicatingSkillsSource` or `FilteringSkillsSource`).

## Skills served over MCP (`MCPSkillsSource`)

`MCPSkillsSource` (experimental, `ExperimentalFeature.MCP_SKILLS`) reads skills from an MCP server that follows the SEP-2640 convention: the server publishes a `skill://index.json` resource, and each entry is either a `skill-md` (a `SKILL.md` resource, fetched lazily with `resources/read` only when the skill is loaded) or an `archive` (a ZIP unpacked **in memory** into the skill's body plus resources). Scripts inside archives are exposed as read-only resources and never as runnable scripts. If the index is missing or malformed the source just returns no skills.

```python
import asyncio
from agent_framework import Agent, MCPSkillsSource, MCPStdioTool, SkillsProvider
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    async with MCPStdioTool(name="skills-server", command="python", args=["skills_server.py"]) as mcp:
        # session_provider (not client=) survives the tool reconnecting and swapping its session.
        source = MCPSkillsSource(session_provider=lambda: mcp.session)
        agent = Agent(
            client=OpenAIChatClient(),
            instructions="Use your skills to help the user.",
            tools=[mcp],
            context_providers=[SkillsProvider(source)],
        )
        print((await agent.run("Which skills can you use?")).text)


asyncio.run(main())
```

Pass exactly one of `client=` (a live `mcp.ClientSession`) or `session_provider=`. The `archive_max_file_count` (default 20), `archive_max_size_bytes` and `archive_max_uncompressed_size_bytes` (1 MB each), `archive_resource_extensions` and `archive_resource_search_depth` (default 2) limits protect against oversized or zip-bomb archives. A non-null `sha256:` digest on an archive entry is verified before extraction, and a mismatched archive is skipped with a warning. The source doesn't cache, so wrap it in `CachingSkillsSource` to avoid re-reading the index every run. Skills from a remote server are untrusted input, so keep the default approval flags on `SkillsProvider` unless you control the server.

## Sources, context and caching

Every source implements `async get_skills(context: SkillsSourceContext) -> list[Skill]`. The context carries the invoking `agent` and, when there is one, the `session`, so a source can return different skills for different agents, users or tenants.

`SkillsProvider` caches what its source returns: by default it discovers skills once and reuses them. `cache_refresh_interval=timedelta(minutes=5)` re-discovers them periodically, and `disable_caching=True` asks the source on every run (useful when the source itself is per-request, as in the filtering example below). For finer control, wrap a source in `CachingSkillsSource` yourself:

```python
from datetime import timedelta
from agent_framework import CachingSkillsSource, FileSkillsSource, SkillsProvider

cached = CachingSkillsSource(
    FileSkillsSource("./domain-skills"),
    refresh_interval=timedelta(minutes=10),
    # One cache entry per session id; return None to share one entry.
    cache_isolation_key_selector=lambda ctx: ctx.session.session_id if ctx.session else None,
)
provider = SkillsProvider(cached, disable_caching=True)   # the source does the caching
```

### `DelegatingSkillsSource` — your own decorators

`DelegatingSkillsSource` is the base class of the built-in decorators (`CachingSkillsSource`, `FilteringSkillsSource`, `DeduplicatingSkillsSource`). It stores the wrapped source as `self.inner_source`; override `get_skills(self, context)` to transform what it returns. For example, a decorator that hides skills marked `metadata: {"stage": "beta"}` from everyone except staff agents:

```python
from agent_framework import DelegatingSkillsSource, Skill, SkillsSourceContext


class HideBetaSkills(DelegatingSkillsSource):
    async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
        skills = await self.inner_source.get_skills(context)
        if context.agent is not None and getattr(context.agent, "name", "") == "staff-assistant":
            return skills
        return [s for s in skills if (s.frontmatter.metadata or {}).get("stage") != "beta"]
```

## Composable sources

### Aggregating multiple sources

Combine code-defined and file-based skills with `AggregatingSkillsSource`:

```python
from agent_framework import (
    SkillFrontmatter,
    Agent,
    AggregatingSkillsSource,
    DeduplicatingSkillsSource,
    FileSkillsSource,
    FilteringSkillsSource,
    InlineSkill,
    InMemorySkillsSource,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatClient

# Code-defined skill
analytics_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="analytics",
        description="Run data analytics queries.",
    ),
    instructions="Use the run_query script to run SQL SELECT queries.",
)

# File-based source
file_source = FileSkillsSource("./domain-skills")

# Aggregate both sources (each element must be a SkillsSource, so wrap the list)
combined = AggregatingSkillsSource([file_source, InMemorySkillsSource([analytics_skill])])

# Deduplicate in case both sources define a skill with the same name
unique = DeduplicatingSkillsSource(combined)

provider = SkillsProvider(unique)
agent = Agent(client=OpenAIChatClient(), context_providers=[provider])
```

### Filtering skills per request

Use `FilteringSkillsSource` to expose only skills relevant to the current user or context. The predicate receives the skill **and** the `SkillsSourceContext`; skill metadata lives on `skill.frontmatter`:

```python
from agent_framework import FilteringSkillsSource, FileSkillsSource, SkillsProvider

all_source = FileSkillsSource("./skills")

# Expose skills whose SKILL.md frontmatter has metadata: {domain: finance},
# plus anything the session has been granted explicitly.
finance_source = FilteringSkillsSource(
    inner_source=all_source,
    predicate=lambda skill, ctx: (
        (skill.frontmatter.metadata or {}).get("domain") == "finance"
        or (ctx.session is not None and skill.frontmatter.name in ctx.session.state.get("granted_skills", []))
    ),
)

provider = SkillsProvider(finance_source, disable_caching=True)   # re-filter on every run
```

### Deduplicating

When aggregating multiple sources that may overlap on names, wrap with `DeduplicatingSkillsSource` — it keeps the first occurrence of each name and silently drops duplicates:

```python
from agent_framework import DeduplicatingSkillsSource, AggregatingSkillsSource, FileSkillsSource

primary = FileSkillsSource("./primary-skills")
fallback = FileSkillsSource("./fallback-skills")

# Primary skills take precedence; fallback fills in anything not already defined
source = DeduplicatingSkillsSource(AggregatingSkillsSource([primary, fallback]))
```

## Running file-based scripts

File-based scripts need a `SkillScriptRunner` — pass it to `FileSkillsSource` or `SkillsProvider.from_paths`. A runner is any callable `(skill: FileSkill, script: FileSkillScript, args) -> result`; `skill.path` is the skill's directory, `script.full_path` the script file, and `args` a dict (or a list of strings, for command-line style arguments):

```python
import asyncio
import json
import sys
from agent_framework import FileSkill, FileSkillScript, SkillsProvider


async def subprocess_runner(skill: FileSkill, script: FileSkillScript, args: dict | list[str] | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, script.full_path, "--args", json.dumps(args or {}),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"Script {script.name} timed out after 30s")

    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8"))
    return stdout.decode("utf-8")


provider = SkillsProvider.from_paths("./skills", script_runner=subprocess_runner)
```

### Sandboxed runner (Docker)

For untrusted scripts, isolate execution with Docker:

```python
import asyncio
import json
from pathlib import Path
from agent_framework import FileSkill, FileSkillScript, SkillsProvider


class DockerSkillRunner:
    def __init__(self, image: str, *, network: str = "none", memory: str = "512m", timeout: float = 60) -> None:
        self.image = image
        self.network = network
        self.memory = memory
        self.timeout = timeout

    async def __call__(self, skill: FileSkill, script: FileSkillScript, args: dict | list[str] | None = None) -> str:
        relative = Path(script.full_path).relative_to(skill.path).as_posix()
        cmd = [
            "docker", "run", "--rm",
            f"--network={self.network}",
            f"--memory={self.memory}",
            "--read-only",
            "-v", f"{skill.path}:/skill:ro",
            self.image,
            "python", f"/skill/{relative}",
            "--args", json.dumps(args or {}),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Docker execution of {script.name} timed out after {self.timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8"))
        return stdout.decode("utf-8")


provider = SkillsProvider.from_paths(
    "./skills",
    script_runner=DockerSkillRunner(image="org/skill-sandbox:latest"),
)
```

## Script approval

Script execution is gated on human approval by default. A common set-up is to trust skill *discovery* and *reading* but keep a human in front of every script run:

```python
from agent_framework import Agent, Message, SkillsProvider
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[
        SkillsProvider.from_paths(
            "./skills",
            script_runner=my_runner,
            disable_load_skill_approval=True,           # trusted source: load freely
            disable_read_skill_resource_approval=True,  # trusted source: read freely
            # disable_run_skill_script_approval stays False -> scripts need approval
        )
    ],
)

session = agent.create_session()
response = await agent.run("Run the nightly report", session=session)

while response.user_input_requests:
    replies = []
    for request in response.user_input_requests:
        call = request.function_call
        print(f"Approve {call.name} with args {call.arguments}?")
        replies.append(request.to_function_approval_response(approved=True))
    response = await agent.run(Message("user", replies), session=session)

print(response.text)
```

When the agent tries to run a script, the run returns with a `function_approval_request` in `response.user_input_requests`. Reply on the same session with the approval responses to let the call proceed.

## `ClassSkill` — reusable skill types

Subclass `ClassSkill` to create a self-contained, distributable skill class. This pattern is ideal when different teams or deployments need the same skill with different configuration (e.g. different DB connection strings, API endpoints, or tenant IDs).

The correct API (verified against `agent-framework` %%version=agent-framework%%):

- `super().__init__(frontmatter=SkillFrontmatter(name=..., description=...))` — constructor takes only `frontmatter=`.
- `@property instructions(self) -> str` — **abstract**; must be overridden to return the skill body text.
- `@ClassSkill.resource` — decorator that marks an instance method as a resource auto-discovered at runtime.
- `@ClassSkill.script` — decorator that marks an instance method as an in-process script.

### Minimal example with `@ClassSkill.resource` and `@ClassSkill.script`

```python
import asyncio
import json
from agent_framework import (
    Agent,
    ClassSkill,
    SkillFrontmatter,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatClient


class DatabaseSkill(ClassSkill):
    """Read-only database access skill — one instance per connection string."""

    def __init__(self, connection_string: str) -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="database",
                description="Query the production PostgreSQL database.",
            )
        )
        self._conn = connection_string

    @property
    def instructions(self) -> str:
        return (
            "Use `read_skill_resource('database', 'schema')` to see the tables. "
            "Then craft a read-only SELECT query and run it with `run_skill_script('database', 'run-query', {\"sql\": \"...\"})`. "
            "Never run INSERT, UPDATE, DELETE, or DROP."
        )

    @ClassSkill.resource(description="Compact database schema (tables + columns).")
    async def schema(self) -> str:
        # In production, query information_schema using self._conn
        return "users(id, email, created_at)\norders(id, user_id, total, status)"

    @ClassSkill.script(name="run-query", description="Execute a SELECT and return up to 100 rows as JSON.")
    async def run_query(self, sql: str) -> str:
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")
        # In production: rows = await asyncpg.fetch(sql, limit=100)
        return json.dumps([{"id": 1, "example": True}])


# Different agents / environments get different connection strings — same class.
db_skill = DatabaseSkill(connection_string="postgresql://prod-host/mydb")

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a data analyst.",
    context_providers=[SkillsProvider(db_skill)],
)


async def main() -> None:
    session = agent.create_session()
    response = await agent.run("How many orders does user 42 have?", session=session)
    print(response.text)


asyncio.run(main())
```

Key points from the source:
- **Name defaults**: `@ClassSkill.resource` uses the method name (underscores → hyphens) as the resource name unless you pass `name=`.
- **Async/sync**: Both sync and async methods work for resources and scripts.
- **Caching**: `resources` and `scripts` properties are cached after first access. Instantiate a new `ClassSkill` object per agent that needs a fresh connection or config.
- **Distribution**: Ship a `ClassSkill` subclass in a shared library — callers import and instantiate it, framework handles introspection.

### Explicit property override — when the decorator pattern doesn't fit

Override `resources` directly for the most control (e.g. constructing resources from a dynamic config table):

```python
from agent_framework import (
    ClassSkill,
    InlineSkillResource,
    SkillFrontmatter,
)


class ConfigDrivenSkill(ClassSkill):
    def __init__(self, resource_configs: list[dict]) -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="config-skill",
                description="Dynamic skill built from a config table.",
            )
        )
        self._configs = resource_configs

    @property
    def instructions(self) -> str:
        names = ", ".join(c["name"] for c in self._configs)
        return f"Available resources: {names}. Use read_skill_resource to fetch each one."

    @property
    def resources(self) -> list[InlineSkillResource]:
        result = []
        for cfg in self._configs:
            content = cfg["content"]
            result.append(
                InlineSkillResource(
                    name=cfg["name"],
                    description=cfg.get("description", ""),
                    content=content,
                )
            )
        return result
```

### Packaging `ClassSkill` as a shared library

`ClassSkill` is ideal for distributing reusable skill types across teams or open-sourcing them. Because the class is self-contained — constructor, instructions, resources, and scripts all in one place — a shared library can export a `ClassSkill` subclass and callers instantiate it with their own configuration:

```python
# my_skills_library/github_skill.py
"""Distributable GitHub skill — install via pip install my-skills-library."""
import json
from agent_framework import ClassSkill, SkillFrontmatter


class GitHubSkill(ClassSkill):
    """Skill that gives an agent read access to a GitHub repository.

    Usage:
        from my_skills_library import GitHubSkill
        skill = GitHubSkill(token="ghp_…", owner="acme", repo="platform")
        agent = Agent(…, context_providers=[SkillsProvider(skill)])
    """

    def __init__(self, *, token: str, owner: str, repo: str) -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="github",
                description=f"Read-only access to {owner}/{repo} on GitHub.",
            )
        )
        self._token = token
        self._owner = owner
        self._repo = repo

    @property
    def instructions(self) -> str:
        return (
            f"You can read the GitHub repository {self._owner}/{self._repo}.\n"
            "Use read_skill_resource('github', 'readme') to see the README.\n"
            "Use run_skill_script('github', 'list-issues') to list open issues.\n"
            "Never write to the repository."
        )

    @ClassSkill.resource(description="README.md content.")
    async def readme(self) -> str:
        import httpx
        url = f"https://api.github.com/repos/{self._owner}/{self._repo}/readme"
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github.v3.raw"}
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r.text

    @ClassSkill.script(name="list-issues", description="List open issues as JSON.")
    async def list_issues(self, state: str = "open", limit: int = 20) -> str:
        import httpx
        url = f"https://api.github.com/repos/{self._owner}/{self._repo}/issues"
        params = {"state": state, "per_page": limit}
        headers = {"Authorization": f"Bearer {self._token}"}
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            issues = r.json()
        return json.dumps([{"id": i["number"], "title": i["title"]} for i in issues])
```

Callers in another service or team:

```python
import asyncio
from agent_framework import Agent, SkillsProvider
from agent_framework.openai import OpenAIChatClient
from my_skills_library import GitHubSkill   # import the packaged skill type


async def main() -> None:
    skill = GitHubSkill(
        token="ghp_…",
        owner="my-org",
        repo="backend-api",
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are an engineering assistant.",
        context_providers=[SkillsProvider(skill)],
    )

    r = await agent.run("Are there any open bugs about the auth module?")
    print(r.text)


asyncio.run(main())
```

Key distribution properties of `ClassSkill`:
- **Zero-dependency surface**: callers only need `agent-framework-core` and your library package.
- **Per-instance config**: each caller can connect to a different repo, endpoint, or secret without any shared global state.
- **Testability**: instantiate the skill in tests, call resource/script methods directly (they're just async methods), and assert on return values without running an agent.
- **IDE discoverability**: docstrings on the class and its constructor travel with the package — type checkers and IDEs surface them at the call site.

## `SkillFrontmatter` reference

`SkillFrontmatter` carries the metadata that identifies a skill and constrains how it is advertised. Validation runs at construction time and raises `ValueError` on the first violated rule.

| Field | Type | Required | Constraints |
|---|---|---|---|
| `name` | `str` | Yes | Lowercase letters, digits, and hyphens only. Max 64 characters. No leading, trailing, or consecutive hyphens. Must match `^[a-z0-9]+(-[a-z0-9]+)*$`. |
| `description` | `str` | Yes | Max 1024 characters. |
| `license` | `str \| None` | No | SPDX licence identifier (e.g. `"MIT"`, `"Apache-2.0"`). |
| `compatibility` | `str \| None` | No | Max 500 characters. Documents framework/version requirements. |
| `allowed_tools` | `str \| None` | No | Space-delimited list of pre-approved tool names the skill may call. Advisory metadata from the Agent Skills spec; the framework surfaces it but does not use it to bypass approval. |
| `metadata` | `dict[str, str] \| None` | No | Arbitrary string key-value pairs. Surfaced to `FilteringSkillsSource` predicates and custom runners. |

```python
from agent_framework import SkillFrontmatter

# Minimal — name and description only
frontmatter = SkillFrontmatter(
    name="db-ops",
    description="Query and describe the production PostgreSQL database.",
)

# Full — all optional fields
frontmatter = SkillFrontmatter(
    name="contract-reviewer",
    description="Review SaaS contracts for non-standard clauses (max 1024 chars).",
    license="MIT",
    compatibility="agent-framework>=1.3.0",
    allowed_tools="search_clauses fetch_precedents",  # space-separated
    metadata={"domain": "legal", "tier": "premium"},
)
```

**Name validation examples:**

```python
from agent_framework import SkillFrontmatter

SkillFrontmatter(name="db-ops", description="…")       # OK
SkillFrontmatter(name="my-cool-skill", description="…")  # OK — hyphens between words
# SkillFrontmatter(name="DB_Ops", description="…")      # ValueError — uppercase / underscore
# SkillFrontmatter(name="-db-ops", description="…")     # ValueError — leading hyphen
# SkillFrontmatter(name="db--ops", description="…")     # ValueError — consecutive hyphens
# SkillFrontmatter(name="a" * 65, description="…")      # ValueError — exceeds 64 chars
```

## `SkillsProvider` reference

```text
SkillsProvider(
    source,                          # SkillsSource | Sequence[Skill] | Skill
    *,
    instruction_template=None,       # custom system-prompt template; must contain {skills}
    disable_caching=False,           # rebuild tools/instructions on every invocation
    cache_refresh_interval=None,     # timedelta after which cached skills are re-discovered
    disable_load_skill_approval=False,           # False = load_skill needs approval
    disable_read_skill_resource_approval=False,  # False = read_skill_resource needs approval
    disable_run_skill_script_approval=False,     # False = run_skill_script needs approval
    source_id=None,                  # unique identifier for this provider instance
)

# Convenience factory for file-based skills
SkillsProvider.from_paths(
    skill_paths,                     # str | Path | Sequence[str | Path]
    *,
    script_runner=None,
    resource_extensions=(".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt"),
    script_extensions=(".py",),
    search_depth=2,
    script_filter=None,              # Callable[[skill_name, path], bool]
    resource_filter=None,            # Callable[[skill_name, path], bool]
    instruction_template=None,
    disable_caching=False,
    cache_refresh_interval=None,
    disable_load_skill_approval=False,
    disable_read_skill_resource_approval=False,
    disable_run_skill_script_approval=False,
    source_id=None,
)
```

### Custom instruction template

The default prompt advertises skills as XML under `<available_skills>`. Override for a different shape or tone:

```python
provider = SkillsProvider(
    [db_skill],
    instruction_template=(
        "You can use these domain skills:\n{skills}\n"
        "Call load_skill with the exact name when a task matches."
    ),
)
```

The `{skills}` placeholder is mandatory — the provider renders names and descriptions into it.

## What the agent sees

- `load_skill("db-ops")` → returns the skill `instructions` plus a list of resources/scripts with their descriptions.
- `read_skill_resource("db-ops", "schema")` → calls the resource callable (or returns static content) and returns the result.
- `run_skill_script("db-ops", "run_query", {"sql": "SELECT 1"})` → invokes the script callable.

## Compared to tools and MCP

| | Tools (`@tool`) | MCP | Skills |
|---|---|---|---|
| Surface | Flat list of callables | Remote server with dozens of tools | Progressive disclosure — names first, content on demand |
| Prompt cost | Full schema in every turn | Full schema in every turn | Name and description per skill until loaded |
| Best for | Small, always-available functions | Third-party integrations | Large knowledge domains + procedures |

Skills compose with tools and MCP. A skill's `instructions` often tells the agent to use specific tools — e.g. "Call `search_crm` with the customer ID then summarise."

## Patterns

**Rarely-used domain knowledge.** Accounting rules, tax brackets, safety protocols — huge content that matters 5% of the time. Skills keep it out of the default prompt.

**Multi-tenant agents.** Build a `FilteringSkillsSource` whose predicate reads the tenant from `ctx.session.state`, and set `disable_caching=True` on the provider (or isolate a `CachingSkillsSource` by tenant) so each request gets its tenant's skill set.

**Dynamic reference docs.** Back a resource with an async function that queries a live CMS / Notion / Confluence — the agent sees fresh content on every `read_skill_resource` call.

**Composable skill libraries.** Maintain skills in separate directories (shared core + team-specific). Aggregate with `AggregatingSkillsSource`, deduplicate with `DeduplicatingSkillsSource`, wrap in `SkillsProvider`.

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | 1.22.0 | Fixed examples that failed at run time: `FilteringSkillsSource` predicates take `(skill, context)`, skill metadata is on `skill.frontmatter`, `AggregatingSkillsSource` needs sources (not lists), script runners receive `FileSkill` / `FileSkillScript` (`script.full_path`, not `script.path`). Replaced a hand-written cache that shadowed the built-in `CachingSkillsSource` and used the old `get_skills()` signature; documented `SkillsSourceContext` and provider caching options. |

---
title: "Microsoft Agent Framework (Python) — Skills"
description: "Progressive-disclosure domain knowledge for agents. Skill / SkillResource / SkillScript / SkillsProvider with file-based and code-defined skills, script runners, and approval gates."
framework: microsoft-agent-framework
language: python
---

# Skills — Python

Skills are a **progressive-disclosure** knowledge pattern. Instead of stuffing every reference doc and procedure into the system prompt, you advertise skill *names and descriptions* (cheap), let the model decide which to load (`load_skill`), and then fetch resources (`read_skill_resource`) or run scripts (`run_skill_script`) on demand. The total context stays small until the agent actually needs deeper knowledge.

This follows the [Agent Skills specification](https://agentskills.io). Verified against `agent-framework-core==1.1.0` (`agent_framework._skills`). Marked `experimental` — API may evolve.

## The primitives

| Class | Role |
|---|---|
| `Skill` | A bundle: name, description, instructions body, zero or more resources and scripts |
| `SkillResource` | Named static or dynamic content the model can fetch via `read_skill_resource` |
| `SkillScript` | Executable code (in-process callable or file on disk) the model can invoke via `run_skill_script` |
| `SkillsProvider` | A `ContextProvider` that advertises skills in the prompt and exposes the three tools |
| `SkillScriptRunner` | Strategy protocol for running file-based scripts (sandbox, subprocess, hosted) |

## Code-defined skill

The fastest path — define a `Skill` in Python and register it:

```python
from agent_framework import Agent, Skill, SkillResource, SkillsProvider
from agent_framework.openai import OpenAIChatClient


db_skill = Skill(
    name="db-ops",
    description="Query and describe the production PostgreSQL database.",
    content=(
        "When the user asks about data, first call `read_skill_resource` with "
        "'schema' to see the tables, then craft a SELECT query. Never run INSERT/UPDATE/DELETE."
    ),
    resources=[
        SkillResource(
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
    context_providers=[SkillsProvider(skills=[db_skill])],
)

response = await agent.run("How many orders did customer u-42 place last month?")
```

The agent sees a system-prompt blurb listing `db-ops` plus its description. If it decides the question matches, it calls `load_skill("db-ops")` to pull `content`, then `read_skill_resource("db-ops", "schema")` to see the schema — neither is in the initial prompt.

## Dynamic resources

Resources can be callables — useful when content changes per request or is expensive to build upfront:

```python
skill = Skill(
    name="inventory",
    description="Check real-time stock levels.",
    content="Use `read_skill_resource('inventory', 'stock')` to see current stock.",
)


@skill.resource
async def stock() -> str:
    """Snapshot of current stock across warehouses."""
    rows = await fetch_stock_from_db()
    return "\n".join(f"{r.sku}: {r.qty}" for r in rows)
```

The decorator uses the function name and docstring as defaults. Both sync and async callables work.

### Per-request resource via `**kwargs`

Resource functions can opt into `**kwargs` to receive runtime arguments forwarded by `agent.run(..., function_invocation_kwargs={...})`. The framework only forwards `**kwargs` when the function declares it — adding the parameter is a deliberate signal that you want runtime data.

```python
from agent_framework import Agent, Skill, SkillsProvider
from agent_framework.openai import OpenAIChatClient


tenant_skill = Skill(
    name="tenant-pricing",
    description="Fetch pricing matrix for the current tenant.",
    content="Use `read_skill_resource('tenant-pricing', 'matrix')` to see the current matrix.",
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
    context_providers=[SkillsProvider(skills=[tenant_skill])],
)

# tenant_id flows through function_invocation_kwargs into the resource's **kwargs.
await agent.run(
    "What's the SKU price for ACME's PRO plan?",
    function_invocation_kwargs={"tenant_id": "acme"},
)
```

Without `**kwargs` in the signature, the framework calls the resource as `function()` — runtime kwargs are silently dropped. The detection happens at construction (`SkillResource.__init__` inspects the signature once), so adding or removing `**kwargs` does require a fresh resource registration.

The same `**kwargs` rule applies to `SkillScript`. Use it when you want the agent's per-call arguments validated against the schema (the regular path) **plus** ambient runtime data (tenant id, request id, user id) the model never sees.

## Scripts

Skills can bundle executable code too. Code-defined scripts run in-process (no runner needed):

```python
from agent_framework import Skill, SkillScript


skill = Skill(
    name="stats",
    description="Compute summary statistics on numeric data.",
    content="Use `run_skill_script` with name='summary'.",
)


@skill.script
def summary(values: list[float]) -> dict[str, float]:
    """Return mean, min, and max for a list of numbers."""
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
```

Script arguments are inferred from the signature and advertised to the model as JSON schema — the model sends args as a JSON object (`{"values": [1, 2, 3]}`) and the framework routes them to your function.

### Constructing `SkillScript` directly

Most of the time you'll add scripts via `@skill.script` (code-defined) or auto-discovery (file-based). Construct `SkillScript` directly when you're building a skill from runtime configuration — for instance a registry that maps skill ids to inline functions or to scripts living on disk:

```python
from agent_framework import Skill, SkillScript


def echo(value: str) -> str:
    """Return the value unchanged."""
    return value


# Code-defined: pass `function=`. The script runs in-process.
inline = SkillScript(name="echo", function=echo, description="Return the value unchanged.")
schema = inline.parameters_schema
# JSON Schema with Pydantic-generated `title` fields, e.g.:
# {"type": "object", "properties": {"value": {"type": "string", "title": "Value"}},
#  "required": ["value"], "title": "<lambda>_input"}
assert schema["type"] == "object"
assert "value" in schema["properties"]
assert schema["required"] == ["value"]

# File-based: pass `path=` (relative to the skill directory). Needs a SkillScriptRunner.
on_disk = SkillScript(name="validate.py", path="scripts/validate.py", description="Run validator.")
assert on_disk.function is None
assert on_disk.parameters_schema is None     # only code-defined scripts have a schema

skill = Skill(
    name="ops",
    description="Operations utilities.",
    content="Use `run_skill_script('ops', 'echo', {'value': '...'})` for echo.",
    scripts=[inline, on_disk],
)
```

A few constraints surfaced by `SkillScript.__init__` (each raises `ValueError` so configuration bugs surface at registration, not at runtime):

- `name` is required and non-empty.
- **Exactly one** of `function` or `path` must be supplied — passing both, or neither, raises.
- Adding `**kwargs` to a code-defined script's signature toggles a `_accepts_kwargs` fast-path: the framework forwards arbitrary arguments without validating them against the schema. Useful when the model passes auxiliary metadata you want to log but not type-check.

The lazy `parameters_schema` property reuses `FunctionTool.parameters()` — so the schema you see is identical to what you'd get from `@tool` on the same callable, which means client-side validators that already understand tool schemas accept skill scripts unchanged.

## File-based skills

Store skills on disk and discover them with `skill_paths=`:

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

provider = SkillsProvider(skill_paths="./skills")
agent = Agent(client=OpenAIChatClient(), context_providers=[provider])
```

`SKILL.md` uses YAML frontmatter for metadata and Markdown for the body:

```markdown
---
name: contract-reviewer
description: Review SaaS contracts for non-standard clauses.
---
When the user pastes a contract, read `references/clauses.md` first. For each section, flag clauses that deviate from the reference set.
```

**Security.** File-based resource reads are protected against path traversal and symlink escape. Resource extensions (`.md`, `.json`, `.yaml`, `.yml`, `.csv`, `.xml`, `.txt`) and script extensions (`.py`) are discovered automatically — restrict via `resource_extensions=` / `script_extensions=` if you want narrower surface. Only load skills from trusted sources.

### Running file-based scripts

File-based scripts need a `SkillScriptRunner` — the framework doesn't assume how you want to execute foreign code. The protocol is a single-method `runtime_checkable`, so any callable or class that matches is accepted:

```python
from typing import Protocol, runtime_checkable
from agent_framework import Skill, SkillScript


@runtime_checkable
class SkillScriptRunner(Protocol):
    async def __call__(
        self,
        skill: Skill,
        script: SkillScript,
        args: dict | None = None,
    ) -> str: ...
```

A minimal subprocess runner — good for trusted scripts that ship with your own app:

```python
import asyncio
import json
import sys
from pathlib import Path
from agent_framework import SkillsProvider, Skill, SkillScript


async def subprocess_runner(
    skill: Skill,
    script: SkillScript,
    args: dict | None = None,
) -> str:
    path = Path(skill.path) / script.path

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(path), "--args", json.dumps(args or {}),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()            # reap the child so we don't leak zombies
        raise RuntimeError(f"Script {script.name} timed out after 30s")

    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8"))
    return stdout.decode("utf-8")


provider = SkillsProvider(
    skill_paths="./skills",
    script_runner=subprocess_runner,
)
```

### Sandboxed runner (Docker / ACI / Firecracker)

For untrusted scripts, isolate execution. This runner shells out to `docker run` against a pre-built image that contains only the Python interpreter and the skill directory; tune it for your sandbox of choice:

```python
import asyncio
import json
from agent_framework import Skill, SkillScript


class DockerSkillRunner:
    def __init__(
        self,
        image: str,
        *,
        network: str = "none",
        memory: str = "512m",
        timeout: float = 60,
    ) -> None:
        self.image = image
        self.network = network
        self.memory = memory
        self.timeout = timeout

    async def __call__(
        self,
        skill: Skill,
        script: SkillScript,
        args: dict | None = None,
    ) -> str:
        cmd = [
            "docker", "run", "--rm",
            f"--network={self.network}",
            f"--memory={self.memory}",
            "--read-only",
            "-v", f"{skill.path}:/skill:ro",
            self.image,
            "python", f"/skill/{script.path}",
            "--args", json.dumps(args or {}),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"Docker execution of {script.path} timed out after {self.timeout}s"
            )
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8"))
        return stdout.decode("utf-8")


provider = SkillsProvider(
    skill_paths="./skills",
    script_runner=DockerSkillRunner(image="org/skill-sandbox:latest"),
)
```

Azure Container Instances, Firecracker microVMs, and AWS Lambda all slot into the same shape — build the container once, let the runner shell out per invocation. The runner is the only integration point that varies.

## Script approval

Gate every script execution on human approval:

```python
provider = SkillsProvider(
    skill_paths="./skills",
    script_runner=my_runner,
    require_script_approval=True,
)
```

When the agent tries to run a script, the run pauses and emits a `function_approval_request`. Your application presents it to the user and calls `request.to_function_approval_response(approved=True)` (or `False`) — see the [HITL page](./microsoft_agent_framework_python_hitl/#tool-approval) for the approval loop.

Full approval-loop skeleton — handle the pause, show the request to a human, and resume:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


agent = Agent(
    client=OpenAIChatClient(),
    context_providers=[SkillsProvider(
        skill_paths="./skills",
        script_runner=my_runner,
        require_script_approval=True,
    )],
)

session = agent.create_session()
stream = agent.run("Run the nightly report", session=session, stream=True)

async for update in stream:
    if update.type == "function_approval_request":
        proposal = update.data
        print(f"Approve script {proposal.function_call.name}"
              f" with args {proposal.function_call.arguments}?")
        # Present to a human, collect a decision, then resume THIS stream
        # so the paused tool call can continue in-place.
        approval = proposal.to_function_approval_response(approved=True)
        await stream.send_response(approval)
    elif update.type == "message":
        print(update.text)
```

Reject with `approved=False` and the model is told the call was declined — it can either stop, retry with different arguments, or pick a different approach. Always send the response back through the active `stream` (via `stream.send_response(...)`) rather than starting a fresh `agent.run(...)` — a new run would leave the original paused invocation unresolved.

## Decorator reference — `@skill.resource` and `@skill.script`

Both decorators accept two forms — bare (no parens) and parameterised. Bare uses the function name and docstring; parameterised lets you override either:

```python
import inspect
import json
from agent_framework import Skill

skill = Skill(
    name="db-ops",
    description="PostgreSQL read-only operations.",
    content="Use load_skill to fetch the schema first, then craft queries.",
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

The decorators only mutate the `Skill` instance — they return the **original function unchanged**, so you can still call the function directly from your own code (e.g. for unit tests).

### What the agent sees

- `load_skill("db-ops")` → returns the skill `content` plus a list of resources / scripts with their descriptions.
- `read_skill_resource("db-ops", "schema")` → calls `schema()` and returns the result.
- `run_skill_script("db-ops", "run_query", {"sql": "SELECT 1"})` → invokes `execute_query("SELECT 1")`.

The framework passes script arguments by name, so the function signature drives the JSON schema the model sees. Use `Annotated[T, "..."]` or Pydantic field metadata on parameters when you need richer descriptions.

### Schema introspection

`SkillScript` exposes the JSON schema for the script's parameters via `parameters_schema`:

```python
script = next(s for s in skill.scripts if s.name == "run_query")
print(script.parameters_schema)
# {'type': 'object', 'properties': {'sql': {...}}, 'required': ['sql']}
```

Useful when you want to validate inputs server-side before letting a runner execute them, or when you're building an admin UI that lists available scripts.

## Mixing code-defined and file-based skills

Both sources co-exist:

```python
provider = SkillsProvider(
    skill_paths=["./skills", "./additional-skills"],
    skills=[db_skill, stats_skill],   # plus code-defined ones
    script_runner=my_runner,
    require_script_approval=True,
)
```

## Custom instruction template

The default prompt advertises skills as XML under `<available_skills>`. Override for a different shape or tone:

```python
provider = SkillsProvider(
    skills=[db_skill],
    instruction_template=(
        "You can use these domain skills:\n{skills}\n"
        "Call load_skill with the exact name when a task matches."
    ),
)
```

The `{skills}` placeholder is mandatory — the provider renders names and descriptions into it.

## Compared to tools and MCP

| | Tools (`@tool`) | MCP | Skills |
|---|---|---|---|
| Surface | Flat list of callables | Remote server with dozens of tools | Progressive disclosure — names first, content on demand |
| Prompt cost | Full schema in every turn | Full schema in every turn | ~100 tokens per skill until loaded |
| Best for | Small, always-available functions | Third-party integrations | Large knowledge domains + procedures |

Skills compose with tools and MCP. A skill's `content` often tells the agent to use specific tools — e.g. "Call `search_crm` with the customer ID then summarise."

## Patterns

**Rarely-used domain knowledge.** Accounting rules, tax brackets, safety protocols — huge content that matters 5% of the time. Skills keep it out of the default prompt.

**Multi-tenant agents.** Load a tenant-specific skill set per request via a custom `ContextProvider` that wraps `SkillsProvider` and swaps the registered skills.

**Agent SDK parity.** If you're migrating from Claude's agent skills, this module is a direct port of the same progressive-disclosure pattern — specs line up.

**Dynamic reference docs.** Back a resource with a function that queries a live CMS / Notion / Confluence — the agent sees fresh content on every `read_skill_resource` call.

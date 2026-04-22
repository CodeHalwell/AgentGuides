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

File-based scripts need a `SkillScriptRunner` — the framework doesn't assume how you want to execute foreign code.

```python
import json
import subprocess
import sys
from pathlib import Path
from agent_framework import SkillsProvider, Skill, SkillScript


def subprocess_runner(skill: Skill, script: SkillScript, args: dict | None = None) -> str:
    path = Path(skill.path) / script.path
    result = subprocess.run(
        # sys.executable keeps the runner in the same venv/pyenv as the host.
        [sys.executable, str(path), "--args", json.dumps(args or {})],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


provider = SkillsProvider(
    skill_paths="./skills",
    script_runner=subprocess_runner,
)
```

For anything that handles untrusted code, run in a sandbox (Docker, Firecracker, Azure Container Instances). The runner is the integration point — plug in whatever sandbox you already have.

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

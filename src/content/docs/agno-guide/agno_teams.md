---
title: "Agno: teams"
description: "Multi-agent teams in Agno: the leader and its members, the four team modes (coordinate, route, broadcast, tasks), sharing context with members, nested teams, streaming, and human-in-the-loop inside a team."
framework: "agno"
language: "python"
sidebar:
  label: "Teams"
  order: 20
---

Checked against **agno %%version=agno%%** (checked %%versions-checked%%). Each mode below was run with scripted stand-in models for the leader and the members, so the delegation tool calls, member runs and returned output are Agno's real behaviour; the wording of answers is illustrative.

## What a team is

A `Team` is a leader model plus a list of members (agents or other teams). The leader receives the user's message, a description of each member, and delegation tools. How it may use them is set by `mode`. A team accepts almost everything an agent does (`instructions`, `tools`, `db`, `session_state`, `memory_manager`, `knowledge`, `output_schema`, `pre_hooks`/`post_hooks`, `stream`) and returns a `TeamRunOutput`.

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.websearch import WebSearchTools   # pip install ddgs

researcher = Agent(
    name="Researcher",
    role="Finds facts and figures on the web",
    model=OpenAIResponses(id="%%model=openai-small%%"),
    tools=[WebSearchTools()],            # without a tool, the Researcher could only recall what the model knows
)
writer = Agent(
    name="Writer",
    role="Writes short, clear copy",
    model=OpenAIResponses(id="%%model=openai-small%%"),
)

team = Team(
    name="Desk",
    model=OpenAIResponses(id="%%model=openai%%"),
    members=[researcher, writer],
    instructions="Check facts with the Researcher before the Writer drafts anything.",
    share_member_interactions=True,      # the Writer is shown what the Researcher was asked and answered
)

run = team.run("Write two sentences on where Leeds is.")
print(run.content)
for member_run in run.member_responses:
    print(member_run.agent_name, "->", member_run.content)
```

The leader's system message lists the members by id:

```text
<team_members>
<member id="researcher" name="Researcher">
  Role: Finds facts and figures on the web
</member>
<member id="writer" name="Writer">
  Role: Writes short, clear copy
</member>
</team_members>
```

A member's id is its `id` if you set one, otherwise its `name` made URL-safe (lower case, hyphens). Give members a `role` (and a `description` if needed): it is the only thing the leader knows about them besides the name, unless `add_member_tools_to_context=True` also lists their tools.

Members only see what the leader writes in the task, unless you share more. In the example above the Writer is a separate agent: without `share_member_interactions=True` it receives only the task text the leader chose to write for it, and the Researcher's findings reach it only if the leader copies them in. With the setting on, Agno prefixes the Writer's input with a `<member_interaction_context>` block listing each earlier member's task and response (checked by logging what the Writer's model received in both cases). See [Sharing context with members](#sharing-context-with-members).

## Modes

```python
from agno.team import Team, TeamMode
```

| Mode | Leader's tools | What happens | Use it for |
|------|---------------|--------------|-----------|
| `TeamMode.coordinate` (default) | `delegate_task_to_member(member_id, task)` | The leader writes a task for one member at a time (several in one turn when they are independent), reads the results and writes the final answer. | General supervisor pattern |
| `TeamMode.route` | `delegate_task_to_member` | The leader picks one member; that member's answer is returned as the team's answer without a second leader call. | Front desks, classifiers |
| `TeamMode.broadcast` | `delegate_task_to_members(task)` | The same task goes to every member; the leader combines the answers. `arun()` runs members concurrently, `run()` one after another. | Panels, second opinions |
| `TeamMode.tasks` | `create_task`, `execute_task`, `execute_tasks_parallel`, `update_task_status`, `add_task_note`, `list_tasks`, `mark_all_complete` | The leader breaks the goal into a task list with assignees and dependencies, runs tasks on members and loops until it calls `mark_all_complete` or reaches `max_iterations` (default 10). | Longer goals with several steps |

### Coordinate

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team, TeamMode
from agno.tools.websearch import WebSearchTools

team = Team(
    model=OpenAIResponses(id="%%model=openai%%"),
    members=[
        Agent(name="Researcher", role="Finds facts", model=OpenAIResponses(id="%%model=openai-small%%"),
              tools=[WebSearchTools()]),
        Agent(name="Writer", role="Writes copy", model=OpenAIResponses(id="%%model=openai-small%%")),
    ],
    mode=TeamMode.coordinate,
)
run = team.run("Where is Leeds?")
print([t.tool_name for t in run.tools])   # ['delegate_task_to_member']
```

By default the leader writes each member's task itself. `determine_input_for_members=False` passes the user's original input to members unchanged.

### Route

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team, TeamMode

front_desk = Team(
    name="Front desk",
    model=OpenAIResponses(id="%%model=openai-small%%"),
    members=[
        Agent(name="Billing", role="Invoices, refunds and payments", model=OpenAIResponses(id="%%model=openai%%")),
        Agent(name="Technical", role="Errors, outages and how-to questions", model=OpenAIResponses(id="%%model=openai%%")),
    ],
    mode=TeamMode.route,
)
run = front_desk.run("I was charged twice this month")
# run.content is Billing's answer; the leader model was called once, to choose
```

The members here have no tools, so they answer from the model's general knowledge. A real front desk gives each specialist the tools for its department (an invoice lookup for Billing, a status check for Technical), as in the [front-desk recipe](/agno-guide/agno_recipes/#front-desk-classify-then-route).

### Broadcast

```python
import asyncio

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.openai import OpenAIResponses
from agno.team import Team, TeamMode

panel = Team(
    model=OpenAIResponses(id="%%model=openai%%"),
    members=[
        Agent(name="Optimist", role="Argues for the proposal", model=Claude(id="%%model=anthropic%%")),
        Agent(name="Sceptic", role="Argues against the proposal", model=Gemini(id="%%model=gemini%%")),
    ],
    mode=TeamMode.broadcast,
    instructions="Summarise where the members agree and disagree.",
)

run = asyncio.run(panel.arun("Should we move our API to GraphQL?"))
print(len(run.member_responses))  # 2
```

### Tasks

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team, TeamMode
from agno.tools.websearch import WebSearchTools

project = Team(
    model=OpenAIResponses(id="%%model=openai%%"),
    members=[
        Agent(name="Researcher", role="Gathers sources", model=OpenAIResponses(id="%%model=openai-small%%"),
              tools=[WebSearchTools()]),
        Agent(name="Analyst", role="Draws conclusions from sources", model=OpenAIResponses(id="%%model=openai-small%%")),
    ],
    mode=TeamMode.tasks,
    max_iterations=8,
)
run = project.run("Compare the rail links from Leeds and York to London.")
tasks = run.session_state.get("_team_tasks", {}).get("tasks", [])  # absent if no task was created
print([(t["title"], t["assignee"], t["status"]) for t in tasks])
```

The task list lives in the team's session state under `_team_tasks` (each task has `id`, `title`, `assignee`, `status`, `dependencies`, `result`, `notes`), so it survives across runs when the team has a `db`. Streaming runs emit `TeamTaskCreated` and `TeamTaskUpdated` events.

### Mode from the older flags

`respond_directly=True` and `delegate_to_all_members=True` still exist. When `mode` is not given, Agno infers it from them (`respond_directly` → route, `delegate_to_all_members` → broadcast, neither → coordinate). When `mode` is given, it wins and the flags are reset to match.

## Sharing context with members

Members are full agents: they keep their own tools, knowledge and instructions. What they see of the team's conversation is controlled on the team:

| Setting | Effect |
|---------|--------|
| `add_team_history_to_members=True`, `num_team_history_runs=3` | Members receive the team's earlier runs in this session |
| `share_member_interactions=True` | Members delegated to later in a run see what earlier members were asked and answered |
| `add_member_tools_to_context=True` | The leader's member list includes each member's tools |
| `get_member_information_tool=True` | The leader gets a tool to look up member details instead |
| `add_history_to_context=True` | The leader sees its own earlier runs (as on an agent) |

Session state set on the team (`session_state=`) is shared with members during a run through `run_context.session_state`.

## Nested teams

A member can be a `Team`. It appears in the leader's member list with `type="team"` and its own members nested inside, and is delegated to like an agent:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.arxiv import ArxivTools   # pip install arxiv pypdf
from agno.tools.websearch import WebSearchTools

research = Team(
    name="Research",
    model=OpenAIResponses(id="%%model=openai%%"),
    members=[
        Agent(name="Web", role="Searches the web", model=OpenAIResponses(id="%%model=openai-small%%"),
              tools=[WebSearchTools()]),
        Agent(name="Papers", role="Reads papers on arXiv", model=OpenAIResponses(id="%%model=openai-small%%"),
              tools=[ArxivTools()]),
    ],
)
editorial = Team(
    name="Editorial",
    model=OpenAIResponses(id="%%model=openai%%"),
    members=[research, Agent(name="Editor", role="Edits drafts", model=OpenAIResponses(id="%%model=openai%%"))],
)
```

## Streaming

`team.run(..., stream=True, stream_events=True)` yields team events (`TeamRunStarted`, `TeamToolCallStarted`, `TeamRunContent`, `TeamRunCompleted`, …) with member events interleaved while a member works (`stream_member_events=True` is the default; set it to `False` to see only the leader). In a two-level run the inner team's `TeamRunStarted`…`TeamRunCompleted` sit between the outer leader's `TeamToolCallStarted` and `TeamToolCallCompleted`. With `Team(show_members_responses=True)`, `print_response()` also prints member answers in the terminal.

## Human-in-the-loop inside a team

If a member calls a tool that needs confirmation, user input or external execution, the whole team run pauses. The requirement says which member raised it:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools import tool


@tool(requires_confirmation=True)
def refund(order_id: str) -> str:
    """Refund an order."""
    return f"Refunded order {order_id}"


billing = Agent(name="Billing", role="Refunds", model=OpenAIResponses(id="%%model=openai%%"), tools=[refund])
team = Team(name="Support", model=OpenAIResponses(id="%%model=openai%%"), members=[billing], db=SqliteDb(db_file="agno.db"))

run = team.run("Please refund order 42", session_id="t-1")
for req in run.active_requirements:
    print(req.member_agent_name, req.tool_execution.tool_name, req.tool_execution.tool_args)
    req.confirm()
run = team.continue_run(run_id=run.run_id, requirements=run.requirements, session_id="t-1")
print(run.status, run.content)
```

## Choosing between a team and a workflow

Use a team when the model should decide who does what. Use a [workflow](/agno-guide/agno_workflows/) when you already know the order of steps and want it to run the same way every time; a workflow step can itself be a team. Route mode is often enough for "send this to the right specialist"; reach for tasks mode only when goals genuinely need several dependent steps, since every iteration is another leader call.

## Team reference

| Group | Parameters |
|-------|-----------|
| Members and mode | `members` (agents, teams, or a callable returning them), `mode`, `max_iterations`, `respond_directly`, `delegate_to_all_members`, `determine_input_for_members` |
| Member context | `add_team_history_to_members`, `num_team_history_runs`, `share_member_interactions`, `add_member_tools_to_context`, `get_member_information_tool`, `stream_member_events` |
| As on `Agent` | `model`, `fallback_models`, `name`, `id`, `role`, `description`, `instructions`, `expected_output`, `tools`, `tool_hooks`, `db`, `session_id`, `user_id`, `session_state`, `add_session_state_to_context`, `add_history_to_context`, `memory_manager`, `enable_agentic_memory`, `update_memory_on_run`, `knowledge`, `input_schema`, `output_schema`, `parser_model`, `output_model`, `pre_hooks`, `post_hooks`, `reasoning_model`, `stream`, `stream_events`, `markdown`, `debug_mode`, `telemetry` |

Methods match the agent's: `run`, `arun`, `continue_run`, `acontinue_run`, `print_response`, `get_session`, `get_session_state`, `get_chat_history`, `cancel_run`.
